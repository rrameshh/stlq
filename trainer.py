# trainer.py - Missing from your refactored code
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

from config import Config
from models import create_model
from data import get_data_loaders
from quantization.hooks import create_switch_hook
from quantization.quant_config import QuantizationConfig
from utils.training import train_epoch, validate, save_checkpoint
from utils.quantization_metrics import log_quantization_stats_to_tensorboard

class Trainer:
    """Main trainer class that orchestrates the training process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        
        # Setup directories
        self.work_dir = Path(config.system.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_data_loaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.switch_hook = create_switch_hook(self.model, config.quantization.switch_iteration)
        self.writer = SummaryWriter(log_dir=str(self.work_dir / "tensorboard"))
        
        # Training state
        self.best_metric = 0.0
        self.patience_counter = 0
        
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🔧 Switch iteration: {config.quantization.switch_iteration}")
    
    def _create_model(self):
        """Create and configure the model."""
        # Create quantization config
        quant_config = QuantizationConfig(
            method=self.config.quantization.method,
            momentum=self.config.quantization.momentum,
            device=self.device,
            threshold=self.config.quantization.threshold,
            bits=self.config.quantization.bits,
            eps=self.config.quantization.eps
        )
        
        # Create model
        model = create_model(
            self.config.model.name,
            pretrained=self.config.model.pretrained,
            num_classes=self.config.model.num_classes,
            img_size=self.config.model.img_size,
            quantization_method=quant_config.method,
            **quant_config.__dict__
        )
        
        # Move to device and store reference
        model = model.to(self.device)
        model.device = self.device  # Store device reference
        
        return model
    
    def _create_data_loaders(self):
        """Create data loaders."""
        return get_data_loaders(self.config)
    
    def _create_optimizer(self):
        """Create optimizer."""
        return optim.SGD(
            self.model.parameters(),
            lr=self.config.training.lr,
            momentum=0.9,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
    
    def train(self) -> float:
        """Main training loop."""
        print("🚀 Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, 
                epoch, self.switch_hook, self.writer, 
                self.config.training.log_interval
            )
            
            # Validation  
            val_accuracy = validate(
                self.model, self.val_loader, epoch, self.writer
            )
            
            # Learning rate step
            self.scheduler.step()
            
            # Log quantization stats (for log quantization)
            if self.config.quantization.method == "log":
                log_quantization_stats_to_tensorboard(self.writer, self.model, epoch)
            
            # Log learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Check for improvement
            is_best = val_accuracy > self.best_metric
            if is_best:
                self.best_metric = val_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            save_checkpoint(
                self.model, self.optimizer, val_accuracy, epoch,
                str(self.work_dir), is_best
            )
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            hook_status = self.switch_hook.get_status()
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                  f"({epoch_time:.1f}s) - Val Acc: {val_accuracy:.2f}% "
                  f"(Best: {self.best_metric:.2f}%) - "
                  f"LR: {current_lr:.6f} - "
                  f"Quantization: {'ON' if hook_status['switched'] else 'OFF'}")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        print(f"🎉 Training completed! Best accuracy: {self.best_metric:.2f}%")
        self.writer.close()
        
        return self.best_metric