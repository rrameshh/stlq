# train_mobilenet.py
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
from networks.load_pretrained import load_pretrained_mobilenet


# Import QAT components
from ops import enable_quantization, disable_quantization, print_quantization_status
from networks.model_factory import mobilenetv1, mobilenetv2


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
    
    # Create data loaders based on dataset
    if args.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 10
    elif args.dataset == "imagenet100":
        train_loader, test_loader = get_imagenet100_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create MobileNet model
    if args.mobilenet_version == "v1":
 
            model = mobilenetv1(
                quantization_method=args.quantization,
                num_classes=num_classes,
                device=args.device,
                threshold=args.threshold,
                bits=args.bits,
                width_multiplier=args.width_multiplier
        )
    elif args.mobilenet_version == "v2":
        model = mobilenetv2(
            quantization_method=args.quantization,
            num_classes=num_classes,
            device=args.device,
            threshold=args.threshold,
            bits=args.bits,
            width_multiplier=args.width_multiplier
        )
    elif args.mobilenet_version == "v3_large":
        from networks.unified_mobilenetv3 import mobilenetv3_large
        model = mobilenetv3_large(
            quantization_method=args.quantization,
            num_classes=num_classes,
            device=args.device,
            threshold=args.threshold,
            bits=args.bits,
            width_multiplier=args.width_multiplier
        )
    elif args.mobilenet_version == "v3_small":
        from networks.unified_mobilenetv3 import mobilenetv3_small
        model = mobilenetv3_small(
            quantization_method=args.quantization,
            num_classes=num_classes,
            device=args.device,
            threshold=args.threshold,
            bits=args.bits,
            width_multiplier=args.width_multiplier
        )
    else:
        raise ValueError(f"Unknown MobileNet version: {args.mobilenet_version}")
    
    if args.pretrained:
        model = load_pretrained_mobilenet(
            model, 
            mobilenet_version=args.mobilenet_version,
            num_classes=num_classes
        )
    
    # Modify model for CIFAR-10 if needed (smaller input resolution)
    if args.dataset == "cifar10":
        # For CIFAR-10, we might want to adjust the first conv layer stride
        # since CIFAR-10 images are 32x32 instead of 224x224
        first_conv = model.features[0]
        if hasattr(first_conv, 'conv2d'):
            first_conv.conv2d.stride = (1, 1)  # Reduce stride for smaller images
    
    model.to(args.device)

    # Create optimizer with different LR for different parts
    if args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, 
            momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.scheduler == "multistep":
        milestones = [int(args.num_epochs * 0.3), int(args.num_epochs * 0.6), int(args.num_epochs * 0.8)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )
    else:
        scheduler = None
    
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
        if scheduler is not None:
            scheduler.step()
    
    return best_accuracy


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Train QAT MobileNet")
    
    # Model arguments
    parser.add_argument("--mobilenet-version", default="v2", type=str, choices=["v1", "v2", "v3_large", "v3_small"],
                       help="MobileNet version to use")
    parser.add_argument("--width-multiplier", default=1.0, type=float,
                       help="Width multiplier for MobileNet")
    parser.add_argument("--quantization", default="linear", type=str,
                       choices=["linear", "log"], help="Quantization method to use")
    parser.add_argument("--bits", default=8, type=int, help="Number of quantization bits")
    parser.add_argument("--threshold", default=1e-5, type=float,
                       help="Threshold for log quantization")
    
    # Dataset arguments
    parser.add_argument("--dataset", default="cifar10", type=str, 
                       choices=["cifar10", "imagenet100"], help="Dataset to use")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    # Training arguments
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "adam"])
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--scheduler", default="cosine", type=str, 
                       choices=["cosine", "step", "multistep", "none"])
    parser.add_argument("--step-size", default=30, type=int, help="Step size for StepLR")
    parser.add_argument("--gamma", default=0.1, type=float, help="Gamma for StepLR")
    parser.add_argument("--pretrained", nargs='?', const=True, default=False, help="use pretrained model")

    # QAT specific arguments
    parser.add_argument("--switch-iter", default=5000, type=int, 
                       help="Iteration to switch to activation quantization")
    
    # System arguments
    parser.add_argument("--work-dir", default="./output_mobilenet", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--early-stop", default=10, type=int,
                       help="Early stopping patience (epochs without improvement)")
    
    args = parser.parse_args()
    
    # Create work directory based on configuration
    work_dir = f"{args.work_dir}_{args.mobilenet_version}_{args.quantization}_{args.dataset}"
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir
    
    writer = SummaryWriter(work_dir)

    train_loader, test_loader, model, optimizer, scheduler, switch_hook = setup_training_components(args)

    print(f"Device used: {args.device}")
    print(f"Model weights on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using MobileNet {args.mobilenet_version} with {args.quantization} quantization")
    print(f"Width multiplier: {args.width_multiplier}")
    print(f"Dataset: {args.dataset}")
    
    if args.quantization == "log":
        print(f"Using threshold value: {args.threshold}")
    
    # Show model configuration
    if hasattr(model, 'config'):
        print(f"\nModel Config: {model.config}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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