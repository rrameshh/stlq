# trainer.py - Updated with storage integration
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
from models.io import save_quantized_model  # NEW IMPORT
from models.packaging import create_model_package  # NEW IMPORT

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
        

        effective_batch = config.training.batch_size * config.training.gradient_accumulation_steps
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {config.training.batch_size} → Effective: {effective_batch} "
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
    
    def save_enhanced_checkpoint(self, accuracy: float, epoch: int, is_best: bool = False):
        """Enhanced checkpoint saving with both old and new formats"""
        
        # Add training metadata to model
        self.model._best_accuracy = accuracy
        self.model._epochs_trained = epoch + 1
        
        if is_best:
            # 1. Save traditional checkpoint (for compatibility/resuming training)
            traditional_checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'config': self.config.__dict__  # Save full config
            }
            
            checkpoint_path = self.work_dir / "best_model.pth"
            torch.save(traditional_checkpoint, checkpoint_path)
            print(f" Traditional checkpoint saved: {checkpoint_path.name}")
            
            # Store model info for later packaging
            self._best_model_info = {
                'accuracy': accuracy,
                'epoch': epoch,
                'model_name': self._generate_model_name()
            }
            
            return checkpoint_path
        
        else:
            # Regular epoch checkpoint (traditional format)
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
            }
            checkpoint_path = self.work_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
            return checkpoint_path
    
    def _generate_model_name(self) -> str:
        """Generate descriptive model name"""
        name_parts = [
            self.config.model.name,
            self.config.quantization.method,
            self.config.data.dataset
        ]
        
        # Add sparsity info if applicable
        if hasattr(self.model, '_sparsified'):
            sparsity = self.model._sparsification_results.get('actual_sparsity', 0)
            name_parts.append(f"sparse{int(sparsity*100)}")
        
        # Add accuracy
        if hasattr(self.model, '_best_accuracy') and self.model._best_accuracy != 'unknown':
            acc_str = f"{float(self.model._best_accuracy):.1f}".replace('.', 'p')
            name_parts.append(f"acc{acc_str}")
        
        return "_".join(name_parts)
    
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
                    print(f" Sparsification failed: {e}")
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

            if self.config.quantization.method in ["log", "adaptive_log"]:
                log_quantization_stats_to_tensorboard(self.writer, self.model, epoch)
            
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[0], epoch)
            
            is_best = val_accuracy > self.best_metric
            if is_best:
                self.best_metric = val_accuracy
                self.patience_counter = 0
                
                # Use enhanced checkpoint saving
                paths = self.save_enhanced_checkpoint(val_accuracy, epoch, is_best=True)
                print(f"New best model saved with accuracy: {val_accuracy:.2f}%")
                if hasattr(self.model, '_sparsification_results'):
                    sparsity = self.model._sparsification_results.get('actual_sparsity', 0)
                    print(f"   Model sparsity: {sparsity:.1%}")
            else:
                self.patience_counter += 1
                # Save regular checkpoint
                self.save_enhanced_checkpoint(val_accuracy, epoch, is_best=False)
            
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
        
        # Create final model package
        self.finalize_training()
        
        # Final summary
        print(f"\nTraining completed! Best accuracy: {self.best_metric:.2f}%")
        if hasattr(self.model, '_sparsification_results'):
            results = self.model._sparsification_results
            print(f"Final model sparsity: {results.get('actual_sparsity', 0.0):.1%}")
            print(f"Sparsification method: {results.get('method', 'unknown')}")
        
        self.writer.close()
        return self.best_metric
    
    def finalize_training(self):
        """Create final model package for distribution (only at the end)"""
        print(f"\n Creating final model package...")
        
        try:
            # Load the best model from checkpoint
            best_checkpoint = self.work_dir / "best_model.pth"
            if not best_checkpoint.exists():
                print("⚠️  No best model checkpoint found")
                return None
            
            # Load best model state
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            
            # Restore metadata
            if hasattr(self, '_best_model_info'):
                self.model._best_accuracy = self._best_model_info['accuracy']
                self.model._epochs_trained = self._best_model_info['epoch'] + 1
            
            # Generate package directly (no intermediate files)
            model_name = self._generate_model_name()
            
            # Save directly to package format
            package_dir = self.work_dir / model_name
            package_dir.mkdir(exist_ok=True)
            
            # Save both formats directly in package
            quantized_path, fp32_path = save_quantized_model(
                self.model,
                self.config,
                package_dir / "model.pth",  # This creates model_quantized.pth and model_fp32.pth
                include_metadata=True
            )
            
            # Create metadata file
            from models.io import get_model_info
            import json
            info = get_model_info(quantized_path)
            with open(package_dir / "model_info.json", 'w') as f:
                json.dump(info, f, indent=2, default=str)
            
            print(f"   Final model package created: {package_dir.name}")
            print(f"   Location: {package_dir}")
            print(f"   Quantized: {info['file_size_mb']:.1f} MB")
            print(f"   FP32: ~{info['total_parameters'] * 4 / (1024**2):.1f} MB")
            
            return package_dir
                
        except Exception as e:
            print(f"⚠️  Could not create final package: {e}")
            print("   Training checkpoint is still available in best_model.pth")
            return None
    
    def get_final_structure(self) -> str:
        structure = f"""
Final Directory Structure:
{self.work_dir}/
├── tensorboard/                    # TensorBoard logs
├── config.yaml                     # Training configuration
├── best_model.pth                  # Training checkpoint (with optimizer state)
├── {self._generate_model_name()}/               # Final model package
│   ├── model_quantized.pth        # Quantized model (3-6x compressed)
│   ├── model_fp32.pth             # FP32 model (universal compatibility)
│   └── model_info.json            # Complete metadata
└── checkpoint_epoch_*.pth          # Training checkpoints
"""
        return structure