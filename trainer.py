# trainer.py - Updated to use simplified registry

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

from config import Config
from data import get_data_loaders
from quantization.hooks import create_switch_hook
from utils.training import train_epoch, validate, save_checkpoint
from utils.quantization_metrics import log_quantization_stats_to_tensorboard

# Import the simplified registry
from models import create_model
from models.pretrained import load_pretrained_weights

class Trainer:
    """Simplified trainer using the new registry system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        self.work_dir = Path(config.system.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = self._create_model()
        self.train_loader, self.val_loader = get_data_loaders(self.config)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.config.training.lr, 
            momentum=0.9, weight_decay=self.config.training.weight_decay
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[100, 150], gamma=0.1
        )
        self.switch_hook = create_switch_hook(self.model, config.quantization.switch_iteration)
        self.writer = SummaryWriter(log_dir=str(self.work_dir / "tensorboard"))
        
        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Switch iteration: {config.quantization.switch_iteration}")
    
    def _create_model(self):
        """Create model using the simplified registry."""
        model_name = self.config.model.name.lower()
        
        # Use the simplified registry
        model = create_model(model_name, self.config)
        
        # Load pretrained weights if requested
        if self.config.model.pretrained:
            model = load_pretrained_weights(
                model, 
                model_name,
                num_classes=self.config.model.num_classes,
                img_size=getattr(self.config.model, 'img_size', 224)
            )
        
        # Move to device and store reference
        model = model.to(self.device)
        model.device = self.device
        return model
    
    def train(self) -> float:
        """Main training loop - unchanged from your original."""
        print("Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            
            # Training and validation
            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, 
                epoch, self.switch_hook, self.writer, 
                self.config.training.log_interval
            )
            
            val_accuracy = validate(self.model, self.val_loader, epoch, self.writer)
            self.scheduler.step()
            
            # Log metrics
            if self.config.quantization.method == "log":
                log_quantization_stats_to_tensorboard(self.writer, self.model, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
            
            # Handle best model and early stopping
            is_best = val_accuracy > self.best_metric
            if is_best:
                self.best_metric = val_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            save_checkpoint(
                self.model, self.optimizer, val_accuracy, epoch,
                str(self.work_dir), is_best
            )
            
            # Print summary
            epoch_time = time.time() - start_time
            hook_status = self.switch_hook.get_status()
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                  f"({epoch_time:.1f}s) - Val Acc: {val_accuracy:.2f}% "
                  f"(Best: {self.best_metric:.2f}%) - "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f} - "
                  f"Quantization: {'ON' if hook_status['switched'] else 'OFF'}")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print(f"Training completed! Best accuracy: {self.best_metric:.2f}%")
        self.writer.close()
        return self.best_metric