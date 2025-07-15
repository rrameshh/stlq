# train_deit.py - Dedicated DeiT Training with Distillation
import argparse
import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Set cache directory
os.environ["HF_DATASETS_CACHE"] = "/scratch/roshnir-profile/qat/qat/hf_cache"
os.environ["TORCH_HOME"] = "/scratch/roshnir-profile/torch_cache"


# Import data loaders
from data.cifar import get_cifar10_dataloaders
from data.imagenet100 import get_imagenet100_dataloaders

# Import DeiT models
from networks.unified_deit import deit_tiny, deit_small, deit_base, DeiTLoss

# Import ViT for teacher models
from networks.model_factory import vit_small, vit_base

# Import QAT utilities
from ops import enable_quantization, disable_quantization, print_quantization_status
from ops.quant_config import QuantizationConfig
from utils.training import TrainingMetrics, validate, save_checkpoint


class SwitchQuantizationModeHook:
    """Hook to switch from calibration to quantization phase"""
    def __init__(self, model, switch_iter=50000):
        self.model = model
        self.switch_iter = switch_iter
        self.switched = False

    def after_train_iter(self, iteration):
        if iteration + 1 == self.switch_iter and not self.switched:
            print(f"Iteration {iteration+1}: Switching to activation quantization")
            print("\nBefore enabling quantization:")
            print_quantization_status(self.model)
                    
            enable_quantization(self.model)
            
            print("\nAfter enabling quantization:")
            print_quantization_status(self.model)
            print("Switched to activation quantization")
            self.switched = True
            return True
        return False


def train_epoch_deit(model, train_loader, optimizer, criterion, epoch, switch_hook, writer, log_interval=100):
    """Train DeiT for one epoch with distillation support"""
    model.train()
    metrics = TrainingMetrics()
    
    for i, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        device = model.config.device if hasattr(model, 'config') else next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Training step
        optimizer.zero_grad()
        
        # DeiT forward pass (returns multiple outputs during training)
        if model.training:
            outputs = model(inputs, return_teacher_logits=True)
            # outputs = (cls_logits, dist_logits, teacher_logits)
        else:
            outputs = model(inputs)  # Single output during validation
        
        # DeiT loss (handles distillation automatically)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            # Training mode: distillation loss
            loss_dict = criterion(outputs, targets)
            total_loss = loss_dict['total_loss']
            cls_logits = outputs[0]  # Use classification head for metrics
        else:
            # Validation mode or no teacher: standard cross-entropy
            cls_logits = outputs
            total_loss = F.cross_entropy(cls_logits, targets)
            loss_dict = {
                'total_loss': total_loss,
                'hard_loss': total_loss,
                'soft_loss': torch.tensor(0.0)
            }
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        metrics.update(total_loss.item(), cls_logits, targets)
        
        # Check if it's time to switch to activation quantization
        iteration = epoch * len(train_loader) + i
        if switch_hook.after_train_iter(iteration):
            pass  # Already printed in hook
        
        # Log progress periodically
        if (i + 1) % log_interval == 0:
            if isinstance(outputs, tuple) and len(outputs) == 3:
                print(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], '
                      f'Total Loss: {total_loss.item():.4f}, '
                      f'Hard Loss: {loss_dict["hard_loss"]:.4f}, '
                      f'Soft Loss: {loss_dict["soft_loss"]:.4f}, '
                      f'Accuracy: {metrics.accuracy:.2f}%')
                
                writer.add_scalar('Training/TotalLoss', total_loss.item(), iteration)
                writer.add_scalar('Training/HardLoss', loss_dict["hard_loss"], iteration)
                writer.add_scalar('Training/SoftLoss', loss_dict["soft_loss"], iteration)
            else:
                print(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], '
                      f'Loss: {total_loss.item():.4f}, Accuracy: {metrics.accuracy:.2f}%')
                writer.add_scalar('Training/Loss', total_loss.item(), iteration)
            
            writer.add_scalar('Training/Accuracy', metrics.accuracy, iteration)
            metrics.reset()
    
    return {"loss": metrics.avg_loss, "accuracy": metrics.accuracy}


def setup_deit_training(args):
    """Setup DeiT training components"""
    
    # Create data loaders based on dataset
    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 10
        img_size = 32
        patch_size = 4
    elif args.dataset == "imagenet100":
        train_loader, test_loader = get_imagenet100_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 100
        img_size = 224
        patch_size = 16
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Model selection
    deit_models = {
        "tiny": deit_tiny,
        "small": deit_small,
        "base": deit_base,
    }
    
    if args.deit_variant not in deit_models:
        raise ValueError(f"Unknown DeiT variant: {args.deit_variant}. Available: {list(deit_models.keys())}")
    
    # Teacher model setup (optional)
    teacher_model = None
    if args.use_teacher:
        print(f"Loading teacher model: {args.teacher_type}")
        if args.teacher_type == "resnet":
            import torchvision.models as models
            teacher_model = models.resnet50(pretrained=True)
            # Adapt for different number of classes
            if num_classes != 1000:
                teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
        elif args.teacher_type == "vit":
            # Use larger ViT as teacher
            if args.deit_variant == "tiny":
                teacher_model = vit_small(quantization_method="linear", num_classes=num_classes)
            else:
                teacher_model = vit_base(quantization_method="linear", num_classes=num_classes)
            
            # Load pretrained weights for teacher if available
            if args.pretrained:
                try:
                    from networks.load_pretrained import load_pretrained_vit
                    teacher_variant = "small" if args.deit_variant == "tiny" else "base"
                    teacher_model = load_pretrained_vit(
                        teacher_model, variant=teacher_variant, num_classes=num_classes
                    )
                    print(f"Loaded pretrained weights for teacher ViT-{teacher_variant}")
                except Exception as e:
                    print(f"Warning: Could not load pretrained teacher weights: {e}")
        
        # Move teacher to device and freeze
        if teacher_model:
            teacher_model.to(args.device)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print(f"Teacher model loaded and frozen: {sum(p.numel() for p in teacher_model.parameters()):,} parameters")
    
    # Create DeiT student model
    model = deit_models[args.deit_variant](
        quantization_method=args.quantization,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        device=args.device,
        threshold=args.threshold,
        bits=args.bits,
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        quantize_classifier=args.quantize_classifier,
        teacher_model=teacher_model
    )
    
    # Load pretrained weights for student
    if args.pretrained:
        try:
            from networks.load_pretrained import load_pretrained_deit
            model = load_pretrained_deit(
                model, 
                variant=args.deit_variant,
                num_classes=num_classes,
                img_size=img_size
            )
            print(f"Loaded pretrained DeiT-{args.deit_variant} weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained DeiT weights: {e}")
            print("Falling back to random initialization")
    
    model.to(args.device)
    
    # DeiT loss function (handles distillation)
    criterion = DeiTLoss(
        teacher_model=teacher_model,
        distillation_alpha=args.distillation_alpha,
        distillation_tau=args.distillation_tau
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - HSTLQ style cosine decay
    def hstlq_cosine_schedule(epoch):
        progress = epoch / args.num_epochs
        return math.cos(math.pi * progress / 2)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=hstlq_cosine_schedule)
    
    # Create quantization switch hook
    switch_hook = SwitchQuantizationModeHook(
        model=model, switch_iter=args.switch_iter
    )
    
    return train_loader, test_loader, model, optimizer, scheduler, switch_hook, criterion


def main():
    parser = argparse.ArgumentParser("Train DeiT with Quantization-Aware Training")
    
    # Data arguments
    parser.add_argument("--dataset", default="imagenet100", choices=["cifar10", "imagenet100"])
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--num-workers", default=4, type=int)
    
    # Model arguments
    parser.add_argument("--deit-variant", default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--attn-dropout", default=0.1, type=float, help="Attention dropout rate")
    
    # Training arguments
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate (HSTLQ paper setting)")
    parser.add_argument("--weight-decay", default=0.05, type=float, help="Weight decay")
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--early-stop", default=15, type=int)
    
    # Distillation arguments
    parser.add_argument("--use-teacher", action="store_true", help="Use teacher for distillation")
    parser.add_argument("--teacher-type", default="resnet", choices=["resnet", "vit"], 
                       help="Teacher model type")
    parser.add_argument("--distillation-alpha", default=0.5, type=float, 
                       help="Weight between hard and soft losses (0=only hard, 1=only soft)")
    parser.add_argument("--distillation-tau", default=3.0, type=float, 
                       help="Temperature for distillation softmax")
    
    # Quantization arguments
    parser.add_argument("--quantization", default="linear", choices=["linear", "log"])
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--switch-iter", default=50000, type=int, 
                       help="Iteration to switch to activation quantization")
    parser.add_argument("--threshold", default=1e-5, type=float)
    parser.add_argument("--quantize-classifier", action="store_true", 
                       help="Quantize classifier heads (default: keep FP32)")
    
    # System arguments
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", default="./output_deit")
    
    args = parser.parse_args()
    
    # Create work directory
    work_dir = f"{args.work_dir}_{args.deit_variant}_{args.quantization}_{args.dataset}"
    if args.use_teacher:
        work_dir += f"_{args.teacher_type}teacher"
    if args.quantize_classifier:
        work_dir += "_qhead"
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir
    
    writer = SummaryWriter(work_dir)
    
    # Setup training
    train_loader, test_loader, model, optimizer, scheduler, switch_hook, criterion = setup_deit_training(args)
    
    print(f"Training DeiT-{args.deit_variant} on {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Quantization method: {args.quantization}")
    print(f"Using teacher: {args.use_teacher} ({args.teacher_type if args.use_teacher else 'None'})")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show model configuration
    print(f"\nModel Config: {model.config}")
    
    # Show initial quantization status
    print("\nInitial Quantization Status:")
    print_quantization_status(model)
    
    # Start with quantization disabled for calibration
    disable_quantization(model)
    
    # Training loop
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Will switch to quantization after {args.switch_iter} iterations")
    if args.use_teacher:
        print(f"Distillation: α={args.distillation_alpha}, τ={args.distillation_tau}")
    
    for epoch in range(args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.num_epochs}, Learning Rate: {current_lr:.6f}")
        
        # Train one epoch
        train_metrics = train_epoch_deit(
            model, train_loader, optimizer, criterion, epoch, switch_hook, writer
        )
        
        # Validate
        accuracy = validate(model, test_loader, epoch, writer)
        
        # Check if this is the best model and save checkpoint
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            print(f"Best model saved at epoch {epoch+1} with accuracy {accuracy:.2f}%")
        else:
            epochs_without_improvement += 1
        
        save_checkpoint(model, optimizer, accuracy, epoch, args.work_dir, is_best=is_best)
        
        # Early stopping check
        if epochs_without_improvement >= args.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update the learning rate
        scheduler.step()
        
        # Log learning rate
        writer.add_scalar('Training/LearningRate', current_lr, epoch)
    
    print(f"\nTraining completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f'{args.work_dir}/final_model.pth')
    
    # Final evaluation and quantization status
    print("\nFinal Quantization Status:")
    print_quantization_status(model)
    
    # Print quantization summary
    from ops.layers.all import UnifiedQuantizedLinear
    quantized_layers = [m for m in model.modules() if isinstance(m, UnifiedQuantizedLinear)]
    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = sum(p.numel() for m in quantized_layers for p in m.parameters())
    
    print(f"\n📊 QUANTIZATION SUMMARY:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Quantized parameters: {quantized_params:,}")
    print(f"   Quantization coverage: {quantized_params/total_params*100:.1f}%")
    print(f"   Quantized layers: {len(quantized_layers)}")
    
    if args.use_teacher:
        print(f"\n🎓 DISTILLATION SUMMARY:")
        print(f"   Teacher model: {args.teacher_type}")
        print(f"   Distillation α: {args.distillation_alpha}")
        print(f"   Distillation τ: {args.distillation_tau}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()