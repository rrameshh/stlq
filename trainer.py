# trainer.py - Fixed for QCA sparsification
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

from models import create_model
from models.pretrained import load_pretrained_weights

# Updated import for sparsification
try:
    from sparsify import create_sparsifier
except ImportError:
    print("Warning: sparsify module not found, sparsification will be disabled")
    def create_sparsifier(config):
        return None

class Trainer:
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.system.device)
        self.work_dir = Path(config.system.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
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

        # Create sparsifier (safe initialization)
        self.sparsifier = create_sparsifier(config)
        
        self.best_metric = 0.0
        self.patience_counter = 0
        

        effective_batch = config.data.batch_size * config.training.gradient_accumulation_steps
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {config.data.batch_size} → Effective: {effective_batch} "
              f"(accumulation: {config.training.gradient_accumulation_steps})")
        print(f"Switch iteration: {config.quantization.switch_iteration}")
        
        
        # Print sparsification info
        if self.sparsifier:
            print(f"Sparsification enabled: {config.sparsification.method}")
            print(f"Target sparsity: {config.sparsification.target_ratio:.1%}")
            print(f"Apply at epoch: {config.sparsification.apply_after_epoch}")
        else:
            print("Sparsification disabled")
    
    def _create_model(self):
        """Create model using the simplified registry."""
        model_name = self.config.model.name.lower()
       
        model = create_model(model_name, self.config)
        if self.config.model.pretrained:
            model = load_pretrained_weights(
                model, 
                model_name,
                num_classes=self.config.model.num_classes,
                img_size=getattr(self.config.model, 'img_size', 224)
            )
        model = model.to(self.device)
        model.device = self.device
        return model
    
    def train(self) -> float:
        print("Starting training...")
        
        for epoch in range(self.config.training.num_epochs):
            start_time = time.time()
            if (self.sparsifier and 
                epoch == self.config.sparsification.apply_after_epoch and 
                not hasattr(self.model, '_sparsified')):
                
                print(f"\n{'='*60}")
                print(f"APPLYING ACTIVATION-AWARE SPARSIFICATION AT EPOCH {epoch}")
                print(f"{'='*60}")

                try:
                    self.sparsifier.collect_activation_statistics(self.model, self.train_loader)
                    sparsification_results = self.sparsifier.apply_sparsification(
                        self.model, 
                        self.config.sparsification.target_ratio
                    )

                    actual_sparsity = sparsification_results.get('actual_sparsity', 0.0)
                    method = sparsification_results.get('method', 'unknown')
                    
                    print(f"Sparsification complete: {actual_sparsity:.1%} actual sparsity")
                    print(f"Method used: {method}")
                    
                    # Log to tensorboard
                    self.writer.add_scalar('Sparsification/ActualSparsity', actual_sparsity, epoch)
                    self.writer.add_scalar('Sparsification/TargetSparsity', self.config.sparsification.target_ratio, epoch)
                    
                    # Mark as sparsified
                    self.model._sparsified = True
                    self.model._sparsification_epoch = epoch
                    self.model._sparsification_results = sparsification_results
                    
                    print(f"{'='*60}\n")
                    
                except Exception as e:
                    print(f"❌ Sparsification failed: {e}")
                    print("Continuing training without sparsification...")
                    import traceback
                    traceback.print_exc()

            train_metrics = train_epoch(
                self.model, self.train_loader, self.optimizer, 
                epoch, self.switch_hook, self.writer, 
                self.config.training.log_interval, 
                accumulation_steps=self.config.training.gradient_accumulation_steps
            )
            
            val_accuracy = validate(self.model, self.val_loader, epoch, self.writer)
            self.scheduler.step()

            if (self.config.quantization.method in ["log", "adaptive_log"] and not hasattr(self.model, '_sparsified')):
                log_quantization_stats_to_tensorboard(self.writer, self.model, epoch)
            
            if self.config.quantization.method in ["log", "adaptive_log"]:
                log_quantization_stats_to_tensorboard(self.writer, self.model, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
            
            is_best = val_accuracy > self.best_metric
            if is_best:
                self.best_metric = val_accuracy
                self.patience_counter = 0
                
                if hasattr(self.model, '_sparsification_results'):
                    print(f"New best model with sparsification: {val_accuracy:.2f}%")
            else:
                self.patience_counter += 1
            
            save_checkpoint(
                self.model, self.optimizer, val_accuracy, epoch,
                str(self.work_dir), is_best
            )
            
            epoch_time = time.time() - start_time
            try:
                hook_status = self.switch_hook.get_status()
                quant_status = "ON" if hook_status.get('switched', False) else "OFF"
            except:
                quant_status = "UNKNOWN"
            
            # Enhanced summary with sparsification info
            summary = (f"Epoch {epoch+1}/{self.config.training.num_epochs} "
                      f"({epoch_time:.1f}s) - Val Acc: {val_accuracy:.2f}% "
                      f"(Best: {self.best_metric:.2f}%) - "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f} - "
                      f"Quant: {quant_status}")
            
            # Add sparsification status
            if hasattr(self.model, '_sparsified'):
                sparsity = self.model._sparsification_results.get('actual_sparsity', 0.0)
                summary += f" - Sparse: {sparsity:.1%}"
            
            print(summary)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Final summary
        print(f"\nTraining completed! Best accuracy: {self.best_metric:.2f}%")
        
        if hasattr(self.model, '_sparsification_results'):
            results = self.model._sparsification_results
            print(f"Final model sparsity: {results.get('actual_sparsity', 0.0):.1%}")
            print(f"Sparsification method: {results.get('method', 'unknown')}")
        
        self.writer.close()
        return self.best_metric
    
    def get_model_info(self):
        """Get detailed model information including sparsification status"""
        info = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'sparsified': hasattr(self.model, '_sparsified'),
        }
        
        if hasattr(self.model, '_sparsification_results'):
            info.update(self.model._sparsification_results)
        
        return info