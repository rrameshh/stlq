# main.py - Universal QAT Training (Replaces all train_*.py files)
import argparse
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import data loaders
from data.cifar import get_cifar10_dataloaders
from data.imagenet100 import get_imagenet100_dataloaders

# Import QAT utilities
from ops import enable_quantization, disable_quantization, print_quantization_status
from utils.training import TrainingMetrics, validate, save_checkpoint


class UniversalQATTrainer:
    """
    Universal trainer that handles all model types
    Replaces train_cifar.py, train_vit.py, train_deit.py, train_mobilenet.py
    """
    
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_data_loaders()
        self.setup_model()
        self.setup_training_components()
        self.setup_logging()
        
    def setup_paths(self):
        """Setup output directories"""
        work_dir = f"{self.args.work_dir}_{self.args.model_name}_{self.args.quantization}_{self.args.dataset}"
        if getattr(self.args, 'use_teacher', False):
            work_dir += f"_{self.args.teacher_type}teacher"
        if getattr(self.args, 'quantize_classifier', False):
            work_dir += "_qhead"
            
        os.makedirs(work_dir, exist_ok=True)
        self.args.work_dir = work_dir
        
    def setup_data_loaders(self):
        """Setup data loaders based on dataset"""
        if self.args.dataset == "cifar10":
            self.train_loader, self.test_loader = get_cifar10_dataloaders(
                batch_size=self.args.batch_size, num_workers=self.args.num_workers
            )
            self.num_classes = 10
            self.img_size = 32
        elif self.args.dataset == "imagenet100":
            self.train_loader, self.test_loader = get_imagenet100_dataloaders(
                batch_size=self.args.batch_size, num_workers=self.args.num_workers
            )
            self.num_classes = 100
            self.img_size = 224
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
    def setup_model(self):
        """Setup model using existing factory functions"""
        model_kwargs = self._get_model_kwargs()
        
        if self.args.model_type == 'resnet':
            from networks.model_factory import resnet18, resnet50
            if self.args.model_variant == '18':
                self.model = resnet18(**model_kwargs)
            elif self.args.model_variant == '50':
                self.model = resnet50(**model_kwargs)
            else:
                raise ValueError(f"Unknown ResNet variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'mobilenet':
            from networks.model_factory import mobilenetv1, mobilenetv2, mobilenetv3_small, mobilenetv3_large
            if self.args.model_variant == 'v1':
                self.model = mobilenetv1(**model_kwargs)
            elif self.args.model_variant == 'v2':
                self.model = mobilenetv2(**model_kwargs)
            elif self.args.model_variant == 'v3_small':
                self.model = mobilenetv3_small(**model_kwargs)
            elif self.args.model_variant == 'v3_large':
                self.model = mobilenetv3_large(**model_kwargs)
            else:
                raise ValueError(f"Unknown MobileNet variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'vit':
            from networks.model_factory import vit_tiny, vit_small, vit_base, vit_large
            if self.args.model_variant == 'tiny':
                self.model = vit_tiny(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = vit_small(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = vit_base(**model_kwargs)
            elif self.args.model_variant == 'large':
                self.model = vit_large(**model_kwargs)
            else:
                raise ValueError(f"Unknown ViT variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'deit':
            from networks.model_factory import deit_tiny_model, deit_small_model, deit_base_model
            if self.args.model_variant == 'tiny':
                self.model = deit_tiny_model(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = deit_small_model(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = deit_base_model(**model_kwargs)
            else:
                raise ValueError(f"Unknown DeiT variant: {self.args.model_variant}")
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
            
        # Load pretrained weights if requested
        if self.args.pretrained:
            self._load_pretrained_weights()
            
        # Apply dataset-specific modifications
        self._modify_model_for_dataset()
        
        self.model.to(self.args.device)
        
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model-specific keyword arguments"""
        kwargs = {
            'quantization_method': self.args.quantization,
            'num_classes': self.num_classes,
            'device': self.args.device,
            'threshold': self.args.threshold,
            'bits': self.args.bits,
        }
        
        # Add model-specific parameters
        if self.args.model_type in ['vit', 'deit']:
            kwargs.update({
                'img_size': self.img_size,
                'patch_size': 4 if self.img_size == 32 else 16,
                'drop_rate': self.args.dropout,
                'attn_drop_rate': self.args.attn_dropout,
                'quantize_classifier': self.args.quantize_classifier
            })
            
        if self.args.model_type == 'deit':
            kwargs.update({
                'teacher_model': self._setup_teacher_model() if self.args.use_teacher else None
            })
            
        if self.args.model_type == 'mobilenet':
            kwargs.update({
                'width_multiplier': self.args.width_multiplier
            })
            
        return kwargs
    
    def _load_pretrained_weights(self):
        """Load pretrained weights for the model"""
        try:
            if self.args.model_type == 'resnet':
                from networks.load_pretrained import load_pretrained_resnet
                self.model = load_pretrained_resnet(self.model, num_classes=self.num_classes)
                
            elif self.args.model_type == 'mobilenet':
                from networks.load_pretrained import load_pretrained_mobilenet
                variant = self.args.model_variant.replace('_small', '').replace('_large', '')  # v3_small -> v3
                self.model = load_pretrained_mobilenet(
                    self.model, mobilenet_version=variant, num_classes=self.num_classes
                )
                
            elif self.args.model_type == 'vit':
                from networks.load_pretrained import load_pretrained_vit
                self.model = load_pretrained_vit(
                    self.model, variant=self.args.model_variant, num_classes=self.num_classes, img_size=self.img_size
                )
                
            elif self.args.model_type == 'deit':
                from networks.load_pretrained import load_pretrained_deit
                self.model = load_pretrained_deit(
                    self.model, variant=self.args.model_variant, num_classes=self.num_classes, img_size=self.img_size
                )
                
            print(f"✅ Loaded pretrained {self.args.model_type}-{self.args.model_variant} weights")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained weights: {e}")
            print("Falling back to random initialization")
    
    def _modify_model_for_dataset(self):
        """Apply dataset-specific model modifications"""
        if self.args.dataset == 'cifar10' and self.args.model_type == 'resnet':
            # Modify ResNet for CIFAR-10 (smaller images)
            if hasattr(self.model, 'conv1') and hasattr(self.model, 'config'):
                from ops.layers.all import UnifiedQuantizedConv2dBatchNorm2dReLU
                self.model.conv1 = UnifiedQuantizedConv2dBatchNorm2dReLU(
                    3, 64, kernel_size=3, stride=1, padding=1, 
                    bias=False, activation="relu", config=self.model.config
                )
                self.model.maxpool = nn.Identity()
                
        elif self.args.dataset == 'cifar10' and self.args.model_type == 'mobilenet':
            # Reduce stride for smaller CIFAR-10 images
            if hasattr(self.model, 'features') and len(self.model.features) > 0:
                first_conv = self.model.features[0]
                if hasattr(first_conv, 'conv2d'):
                    first_conv.conv2d.stride = (1, 1)
    
    def _setup_teacher_model(self):
        """Setup teacher model for distillation (DeiT)"""
        if not self.args.use_teacher:
            return None
            
        teacher_type = self.args.teacher_type
        
        if teacher_type == 'resnet':
            import torchvision.models as models
            teacher = models.resnet50(pretrained=True)
            if self.num_classes != 1000:
                teacher.fc = nn.Linear(teacher.fc.in_features, self.num_classes)
                
        elif teacher_type == 'vit':
            from networks.model_factory import vit_small, vit_base
            if self.args.model_variant == 'tiny':
                teacher = vit_small(quantization_method="linear", num_classes=self.num_classes)
            else:
                teacher = vit_base(quantization_method="linear", num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown teacher type: {teacher_type}")
            
        teacher.to(self.args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
            
        print(f"✅ Teacher model loaded: {teacher_type}")
        return teacher
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, loss function, etc."""
        # Optimizer - Auto-detect best for model type
        if self.args.model_type in ['vit', 'deit']:
            # Transformers prefer AdamW
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay
            )
        else:
            # CNNs prefer SGD
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
            
        # Scheduler
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs
            )
        elif self.args.scheduler == 'hstlq_cosine':
            # HSTLQ-style cosine schedule
            def hstlq_cosine_schedule(epoch):
                progress = epoch / self.args.num_epochs
                return math.cos(math.pi * progress / 2)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=hstlq_cosine_schedule)
        else:
            self.scheduler = None
            
        # Loss function
        if self.args.model_type == 'deit' and self.args.use_teacher:
            from networks.unified_deit import DeiTLoss
            self.criterion = DeiTLoss(
                teacher_model=self._setup_teacher_model(),
                distillation_alpha=self.args.distillation_alpha,
                distillation_tau=self.args.distillation_tau
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Quantization switch hook
        self.switch_hook = SwitchQuantizationModeHook(
            model=self.model, 
            switch_iter=self.args.switch_iter
        )
        
    def setup_logging(self):
        """Setup TensorBoard logging"""
        self.writer = SummaryWriter(self.args.work_dir)
        
    def train(self):
        """Main training loop"""
        print(f"🚀 Starting training: {self.args.model_type}-{self.args.model_variant}")
        print(f"📊 Dataset: {self.args.dataset} ({self.num_classes} classes)")
        print(f"🔧 Quantization: {self.args.quantization}")
        print(f"💾 Output: {self.args.work_dir}")
        
        # Show model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📈 Model parameters: {total_params:,}")
        
        if hasattr(self.model, 'config'):
            print(f"⚙️  Model config: {self.model.config}")
            
        # Show initial quantization status
        print("\n📋 Initial Quantization Status:")
        print_quantization_status(self.model)
        
        # Start with quantization disabled for calibration
        disable_quantization(self.model)
        
        # Training loop
        best_accuracy = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(self.args.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}, LR: {current_lr:.6f}")
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            accuracy = validate(self.model, self.test_loader, epoch, self.writer)
            
            # Check for best model
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                print(f"New best accuracy: {accuracy:.2f}%")
            else:
                epochs_without_improvement += 1
                
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, accuracy, epoch, self.args.work_dir, is_best=is_best)
            
            # Early stopping
            if epochs_without_improvement >= self.args.early_stop:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log learning rate
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            
        print(f"\n🏁 Training completed! Best accuracy: {best_accuracy:.2f}%")
        
        # Save final model
        torch.save(self.model.state_dict(), f'{self.args.work_dir}/final_model.pth')
        
        # Final status
        print("\n📋 Final Quantization Status:")
        print_quantization_status(self.model)
        
        self.writer.close()
        
        return best_accuracy
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch - handles all model types"""
        self.model.train()
        metrics = TrainingMetrics()
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            # Move to device
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.args.model_type == 'deit' and hasattr(self.criterion, 'teacher_model') and self.criterion.teacher_model is not None:
                # DeiT with distillation
                outputs = self.model(inputs, return_teacher_logits=True)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                logits = outputs[0]  # Use classification head for metrics
            else:
                # Standard training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                logits = outputs
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping for transformers
            if self.args.model_type in ['vit', 'deit']:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
            self.optimizer.step()
            
            # Update metrics
            metrics.update(loss.item(), logits, targets)
            
            # Check quantization switch
            iteration = epoch * len(self.train_loader) + i
            if self.switch_hook.after_train_iter(iteration):
                pass  # Already logged in hook
                
            # Log progress
            if (i + 1) % self.args.log_interval == 0:
                print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {metrics.avg_loss:.4f}, Acc: {metrics.accuracy:.2f}%")
                
                self.writer.add_scalar('Training/Loss', metrics.avg_loss, iteration)
                self.writer.add_scalar('Training/Accuracy', metrics.accuracy, iteration)
                metrics.reset()
                
        return {"loss": metrics.avg_loss, "accuracy": metrics.accuracy}


class SwitchQuantizationModeHook:
    """Quantization switching hook"""
    def __init__(self, model, switch_iter=5000):
        self.model = model
        self.switch_iter = switch_iter
        self.switched = False

    def after_train_iter(self, iteration):
        if iteration + 1 == self.switch_iter and not self.switched:
            print(f"🔄 Iteration {iteration+1}: Switching to activation quantization")
            print("\nBefore enabling quantization:")
            print_quantization_status(self.model)
                    
            enable_quantization(self.model)
            
            print("\nAfter enabling quantization:")
            print_quantization_status(self.model)
            self.switched = True
            return True
        return False


def create_parser() -> argparse.ArgumentParser:
    """Universal argument parser for all models"""
    parser = argparse.ArgumentParser("Universal QAT Training - Replaces all train_*.py files")
    
    # Model arguments
    parser.add_argument("--model-type", required=True, 
                       choices=["resnet", "mobilenet", "vit", "deit"], 
                       help="Model architecture type")
    parser.add_argument("--model-variant", required=True, 
                       help="Model variant: 18/50 (resnet), v1/v2/v3_small/v3_large (mobilenet), tiny/small/base/large (vit/deit)")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="cifar10", 
                       choices=["cifar10", "imagenet100"])
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    # Training arguments
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--scheduler", default="cosine", 
                       choices=["cosine", "hstlq_cosine", "none"])
    parser.add_argument("--early-stop", default=10, type=int)
    
    # Quantization arguments
    parser.add_argument("--quantization", default="linear", choices=["linear", "log"])
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int)
    parser.add_argument("--threshold", default=1e-5, type=float)
    parser.add_argument("--quantize-classifier", action="store_true")
    
    # Model-specific arguments
    parser.add_argument("--dropout", default=0.1, type=float, help="For ViT/DeiT")
    parser.add_argument("--attn-dropout", default=0.1, type=float, help="For ViT/DeiT")
    parser.add_argument("--width-multiplier", default=1.0, type=float, help="For MobileNet")
    
    # Distillation arguments (DeiT)
    parser.add_argument("--use-teacher", action="store_true")
    parser.add_argument("--teacher-type", default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=3.0, type=float)
    
    # System arguments
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", default="./output")
    parser.add_argument("--log-interval", default=100, type=int)
    
    return parser


def main():
    """Universal training main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set derived arguments
    args.model_name = f"{args.model_type}_{args.model_variant}"
    
    # Create trainer and run
    trainer = UniversalQATTrainer(args)
    best_accuracy = trainer.train()
    
    print(f"🎯 Final result: {best_accuracy:.2f}% accuracy")


if __name__ == "__main__":
    main()