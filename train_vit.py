# train_vit.py - UPDATED FOR INDUSTRY STANDARD SELECTIVE QUANTIZATION
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

# Import your existing data loaders
from data.cifar import get_cifar10_dataloaders
from data.imagenet100 import get_imagenet100_dataloaders

# Import ViT using your factory pattern
from networks.model_factory import vit_tiny, vit_small, vit_base, vit_large

# Import QAT utilities
from ops import enable_quantization, disable_quantization, print_quantization_status
from ops.quant_config import QuantizationConfig
from utils.training import TrainingMetrics, validate, save_checkpoint


class SwitchQuantizationModeHook:
    """
    Hook to switch from calibration to quantization phase
    UPDATED: Works with industry standard selective quantization
    """
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


def train_epoch_vit(model, train_loader, optimizer, epoch, switch_hook, writer, log_interval=100):
    """
    Train ViT for one epoch - SIMPLIFIED for industry standard approach
    No more complex tensor type handling needed!
    """
    model.train()
    metrics = TrainingMetrics()
    
    for i, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        device = model.config.device if hasattr(model, 'config') else next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # SIMPLIFIED: Industry standard models always output FP32!
        # No more complex tensor type checking needed
        loss = F.cross_entropy(outputs, targets)
        
        # Gradient clipping is often helpful for transformers
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics - outputs are always FP32 now
        metrics.update(loss.item(), outputs, targets)
        
        # Check if it's time to switch to activation quantization
        iteration = epoch * len(train_loader) + i
        if switch_hook.after_train_iter(iteration):
            pass  # Already printed in hook
        
        # Log progress periodically
        if (i + 1) % log_interval == 0:
            print(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], '
                  f'Loss: {metrics.avg_loss:.4f}, Accuracy: {metrics.accuracy:.2f}%')
            
            # Log to TensorBoard
            writer.add_scalar('Training/Loss', metrics.avg_loss, iteration)
            writer.add_scalar('Training/Accuracy', metrics.accuracy, iteration)
            metrics.reset()
    
    return {"loss": metrics.avg_loss, "accuracy": metrics.accuracy}


def setup_vit_training(args):
    """Setup ViT training components - UPDATED for industry standard"""
    
    # Create data loaders based on dataset
    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 10
        img_size = 32  # CIFAR-10 images are 32x32
        patch_size = 4  # Smaller patches for smaller images
    elif args.dataset == "imagenet100":
        train_loader, test_loader = get_imagenet100_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 100
        img_size = 224
        patch_size = 16
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create ViT model using your factory pattern
    vit_models = {
        "tiny": vit_tiny,
        "small": vit_small,
        # "base": vit_base,    # Uncomment when you add these to model_factory
        # "large": vit_large   # Uncomment when you add these to model_factory
    }
    
    if args.vit_variant not in vit_models:
        raise ValueError(f"Unknown ViT variant: {args.vit_variant}. Available: {list(vit_models.keys())}")
    
    # UPDATED: Pass quantize_classifier option for industry standard
    model = vit_models[args.vit_variant](
        quantization_method=args.quantization,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        device=args.device,
        threshold=args.threshold,
        bits=args.bits,
        drop_rate=args.dropout,
        attn_drop_rate=args.attn_dropout,
        quantize_classifier=args.quantize_classifier  # NEW: Industry standard option
    )
    
    model.to(args.device)
    
    # Create optimizer with ViT-specific settings
    # ViTs often benefit from different learning rates and weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - Cosine annealing with warmup
    def warmup_cosine_schedule(epoch, warmup_epochs=10):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Create quantization switch hook
    switch_hook = SwitchQuantizationModeHook(
        model=model, switch_iter=args.switch_iter
    )
    
    return train_loader, test_loader, model, optimizer, scheduler, switch_hook


def main():
    parser = argparse.ArgumentParser("Train ViT with Industry Standard QAT")
    
    # Data arguments
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "imagenet100"])
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size (ViTs often use smaller batches)")
    parser.add_argument("--num-workers", default=4, type=int)
    
    # Model arguments
    parser.add_argument("--vit-variant", default="small", choices=["tiny", "small"])  # UPDATED: Only available variants
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate")
    parser.add_argument("--attn-dropout", default=0.1, type=float, help="Attention dropout rate")
    
    # Training arguments
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate (ViTs often use lower LR)")
    parser.add_argument("--weight-decay", default=0.05, type=float, help="Weight decay")
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--early-stop", default=15, type=int)
    
    # Quantization arguments
    parser.add_argument("--quantization", default="linear", choices=["linear", "log"])
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int, 
                       help="Iteration to switch to activation quantization")
    parser.add_argument("--threshold", default=1e-5, type=float)
    
    # NEW: Industry standard quantization options
    parser.add_argument("--quantize-classifier", action="store_true", 
                       help="Quantize classifier head (default: keep FP32 per industry standard)")
    
    # System arguments
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", default="./output_vit")
    
    args = parser.parse_args()
    
    # Create work directory
    work_dir = f"{args.work_dir}_{args.vit_variant}_{args.quantization}_{args.dataset}"
    if args.quantize_classifier:
        work_dir += "_qhead"
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir
    
    writer = SummaryWriter(work_dir)
    
    # Setup training
    train_loader, test_loader, model, optimizer, scheduler, switch_hook = setup_vit_training(args)
    
    print(f"Training ViT-{args.vit_variant} on {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Quantization method: {args.quantization}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show model configuration
    print(f"\nModel Config: {model.config}")
    
    # UPDATED: Industry standard models print their own quantization strategy
    # (This is handled in the model's __init__ method)
    
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
    
    for epoch in range(args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.num_epochs}, Learning Rate: {current_lr:.6f}")
        
        # Train one epoch
        train_metrics = train_epoch_vit(model, train_loader, optimizer, epoch, switch_hook, writer)
        
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
    
    print(f"\n QUANTIZATION SUMMARY:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Quantized parameters: {quantized_params:,}")
    print(f"   Quantization coverage: {quantized_params/total_params*100:.1f}%")
    print(f"   Quantized layers: {len(quantized_layers)}")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()