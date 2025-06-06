# train_cifar.py
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.cifar import get_cifar10_dataloaders
from data.imagenet100 import get_imagenet100_dataloaders
from torch.utils.tensorboard import SummaryWriter

# Import QAT components
from ops import enable_quantization, disable_quantization, print_quantization_status
from networks.unified_resnet import resnet18  # Use unified ResNet
from ops.quant_config import QuantizationConfig

from utils.training import train_epoch, validate, save_checkpoint


class SwitchQuantizationModeHook:
    """
    Hook to switch from calibration phase to quantization-aware training phase.
    """
    def __init__(self, model, switch_iter=5000):
        self.model = model
        self.switch_iter = switch_iter
        self.switched = False

    def after_train_iter(self, iteration):
        """Check if it's time to switch to activation quantization."""
        if iteration + 1 == self.switch_iter and not self.switched:
            print(f"Iteration {iteration+1}: Switching to activation quantization")
            print("\nBefore enabling quantization:")
            print_quantization_status(self.model)
                    
            enable_quantization(self.model)
            
            print("\nAfter enabling quantization:")
            print_quantization_status(self.model)
            self.switched = True
            return True
        return False


def setup_training_components(args):
    """Setup all training components"""
    # Create data loaders
    train_loader, test_loader = get_imagenet100_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    

    model = resnet18(
        quantization_method=args.quantization,
        num_classes=100,
        device=args.device,
        threshold=args.threshold,
        bits=args.bits
    )
    
    # Modify for CIFAR-10 (32x32 images)
    # Replace first conv and remove maxpool for small images
    # if hasattr(model, 'conv1') and hasattr(model, 'config'):
    #     from ops.layers.all import UnifiedQuantizedConv2dBatchNorm2dReLU
        
    #     # Use the model's existing config
    #     model.conv1 = UnifiedQuantizedConv2dBatchNorm2dReLU(
    #         3, 64, kernel_size=3, stride=1, padding=1, 
    #         bias=False, activation="relu", config=model.config
    #     )
    #     model.maxpool = nn.Identity()
    
    model.to(args.device)

    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=0.9, weight_decay=1e-4
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    
    # Create quantization switch hook
    switch_hook = SwitchQuantizationModeHook(
        model=model, switch_iter=args.switch_iter
    )
    
    return train_loader, test_loader, model, optimizer, scheduler, switch_hook


def run_training_loop(args, train_loader, test_loader, model, optimizer, scheduler, switch_hook, writer):
    """Main training loop"""
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Will switch to quantization after {args.switch_iter} iterations")
    
    for epoch in range(args.num_epochs):
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.num_epochs}, Learning Rate: {current_lr:.6f}")
        
        # Train one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, switch_hook, writer)
        
        # Validate
        accuracy = validate(model, test_loader, epoch, writer)
        
        # Check if this is the best model and save checkpoint
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        save_checkpoint(model, optimizer, accuracy, epoch, args.work_dir, is_best=is_best)
        
        # Early stopping check
        if epochs_without_improvement >= args.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Update the learning rate
        scheduler.step()
    
    return best_accuracy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Train QAT ResNet on CIFAR-10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--work-dir", default="./output", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int, 
                       help="Iteration to switch to activation quantization")
    parser.add_argument("--quantization", default="linear", type=str,
                       choices=["linear", "log"], help="Quantization method to use")
    parser.add_argument("--threshold", default=1e-5, type=float,
                       help="Threshold for log quantization")
    parser.add_argument("--early-stop", default=10, type=int,
                       help="Early stopping patience (epochs without improvement)")
    
    args = parser.parse_args()
    
    # Create work directory based on quantization method
    work_dir = f"{args.work_dir}_{args.quantization}"
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir
    
    writer = SummaryWriter(work_dir)

    train_loader, test_loader, model, optimizer, scheduler, switch_hook = setup_training_components(args)

    print(f"Device used: {args.device}")
    print(f"Model weights on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using {args.quantization} quantization method")
    
    if args.quantization == "log":
        print(f"Using threshold value: {args.threshold}")
    
    # Show model configuration
    if hasattr(model, 'config'):
        print(f"\nModel Config: {model.config}")
    
    # Show initial quantization status
    print("\nInitial Quantization Status:")
    print_quantization_status(model)
    
    # Disable quantization for calibration phase
    disable_quantization(model)
    
    best_accuracy = run_training_loop(args, train_loader, test_loader, model, optimizer, scheduler, switch_hook, writer)
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f'{work_dir}/final_model.pth')
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()