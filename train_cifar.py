
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
from ops import enable_quantization, disable_quantization
from networks.resnet_factory import resnet18

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
        """
        Check if it's time to switch to activation quantization.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            True if just switched, False otherwise
        """
        if iteration + 1 == self.switch_iter and not self.switched:
            print(f"Iteration {iteration+1}: Switching to activation quantization")
            print("\nBefore enabling quantization:")
            
            # For mixed precision, show both linear and log stats
            for name, module in self.model.named_modules():
                # Linear quantization stats (running_min/max for activations)
                if hasattr(module, 'running_min') and hasattr(module, 'running_max'):
                    print(f"{name:30} | running_min={module.running_min.item():.6f} | running_max={module.running_max.item():.6f}")
                # Log quantization stats (running_max_abs)
                elif hasattr(module, 'running_max_abs'):
                    print(f"{name:30} | running_max_abs={module.running_max_abs.item():.6f}")
                    
            enable_quantization(self.model)
            
            print("\nAfter enabling quantization:")
            for name, module in self.model.named_modules():
                if hasattr(module, 'activation_quantization'):
                    print(f"{name:30} | activation_quant={module.activation_quantization}")
            self.switched = True
            return True
        return False

class ImageNet100ResNet(nn.Module):
    """
    ResNet model adapted for ImageNet-100 with quantization support.
    """
    def __init__(self, quantization_method="linear", device=None, **kwargs):
        super().__init__()

        # Store parameters
        self.device = device
        self.quantization_method = quantization_method

        # Create model with 100 classes instead of 10
        if quantization_method.lower() == "linear":
            model_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['device', 'threshold', 'weight_threshold']}
            
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=100, **model_kwargs)  # Changed to 100
            
            from ops.linear import QuantizedConv2dBatchNorm2dReLU
            conv_layer = lambda *args, **kw: QuantizedConv2dBatchNorm2dReLU(
                *args, **kw, device=device)
        elif quantization_method.lower() == "log":
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=100, device=device,  # Changed to 100
                                  **kwargs)
            
            from ops.log import LogQuantizedConv2dBatchNorm2dReLU, LogQuantConfig
            config = LogQuantConfig(device=device, 
                              threshold=kwargs.get('threshold', 1e-5))
            conv_layer = lambda *args, **kw: LogQuantizedConv2dBatchNorm2dReLU(
                *args, **kw, config=config, device=device)
        elif quantization_method.lower() == "mixed":
            # Mixed precision quantization
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=100, device=device,
                                 weight_threshold=kwargs.get('weight_threshold', 1e-5))
            
            from ops.mixed import MixedQuantizedConv2dBathNorm2dReLU, MixedQuantConfig
            # Use the existing model config instead of creating a new one
            if hasattr(self.model, 'config'):
                config = self.model.config
            else:
                config = MixedQuantConfig(device=device, 
                                  weight_threshold=kwargs.get('weight_threshold', 1e-5))
            conv_layer = lambda *args, **kw: MixedQuantizedConv2dBathNorm2dReLU(
                *args, **kw, config=config, device=device)
        else:
            raise ValueError(f"Unsupported quantization method: {quantization_method}")
        
        # For ImageNet-100, we keep the standard ImageNet architecture
        # Don't modify conv1 or remove maxpool like we did for CIFAR-10
        # The model should work with 224x224 images as-is
        
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

class CIFAR10ResNet(nn.Module):
    """
    ResNet model adapted for CIFAR-10 with quantization support.
    """
    def __init__(self, quantization_method="linear", device=None, **kwargs):
        super().__init__()

        # Store parameters that we'll need for layer creation
        self.device = device
        self.quantization_method = quantization_method

        # Create model with the specified quantization method
        # Handle parameters differently based on the quantization method
        if quantization_method.lower() == "linear":
            # For linear quantization, we only pass compatible parameters
            # and handle device separately
            model_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['device', 'threshold', 'weight_threshold']}
            
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=10, **model_kwargs)
            
            from ops.linear import QuantizedConv2dBatchNorm2dReLU
            # Pass device explicitly to the convolution layer
            conv_layer = lambda *args, **kw: QuantizedConv2dBatchNorm2dReLU(
                *args, **kw, device=device)
        elif quantization_method.lower() == "log":
            # For log quantization, we pass all parameters
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=10, device=device, 
                                  **kwargs)
            
            from ops.log import LogQuantizedConv2dBatchNorm2dReLU, LogQuantConfig
            # Create config to pass to the conv layer
            config = LogQuantConfig(device=device, 
                              threshold=kwargs.get('threshold', 1e-5))
            # Partial function to pass config
            conv_layer = lambda *args, **kw: LogQuantizedConv2dBatchNorm2dReLU(
                *args, **kw, config=config, device=device)
        elif quantization_method.lower() == "mixed":
            # Mixed precision quantization
            self.model = resnet18(quantization_method=quantization_method, 
                                 num_classes=10, device=device,
                                 weight_threshold=kwargs.get('weight_threshold', 1e-5))
            
            from ops.mixed import MixedQuantizedConv2dBathNorm2dReLU, MixedQuantConfig
            # Use the existing model config instead of creating a new one
            if hasattr(self.model, 'config'):
                config = self.model.config
            else:
                config = MixedQuantConfig(device=device, 
                                  weight_threshold=kwargs.get('weight_threshold', 1e-5))
            # Partial function to pass config
            conv_layer = lambda *args, **kw: MixedQuantizedConv2dBathNorm2dReLU(
                *args, **kw, config=config, device=device)
        else:
            raise ValueError(f"Unsupported quantization method: {quantization_method}")
        
        # Modify the first layer to work with CIFAR-10 images (32x32)
        self.model.conv1 = conv_layer(
            3, 64, kernel_size=3, stride=1, padding=1, 
            bias=False, activation="relu"
        )
        # Remove maxpool as it's too aggressive for small CIFAR images
        self.model.maxpool = nn.Identity()
        
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)


def setup_training_components(args):
    """Setup all training components"""
    # Create data loaders
    # train_loader, test_loader = get_cifar10_dataloaders(
    #     batch_size=args.batch_size, num_workers=args.num_workers
    # )
    train_loader, test_loader = get_imagenet100_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    
    # Create model
    model_kwargs = {}
    if args.quantization == "log":
        model_kwargs['threshold'] = args.threshold
    elif args.quantization == "mixed":
        model_kwargs['weight_threshold'] = args.weight_threshold
    
    # model = CIFAR10ResNet(
    #     quantization_method=args.quantization,
    #     device=args.device, 
    #     **model_kwargs
    # )
    
    model = ImageNet100ResNet(  # Updated class name
        quantization_method=args.quantization,
        device=args.device, 
        **model_kwargs
    )

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
    parser.add_argument("--switch-iter", default=5000, type=int, 
                       help="Iteration to switch to activation quantization")
    parser.add_argument("--quantization", default="linear", type=str,
                       choices=["linear", "log", "mixed"], help="Quantization method to use")
    parser.add_argument("--threshold", default=1e-5, type=float,
                       help="Threshold for second-order quantization in log method")
    parser.add_argument("--weight-threshold", default=1e-5, type=float,
                       help="Weight threshold for mixed precision quantization")
    parser.add_argument("--early-stop", default=10, type=int,
                       help="Early stopping patience (epochs without improvement)")
    
    args = parser.parse_args()
    
    # Create work directory based on quantization method
    work_dir = f"{args.work_dir}_{args.quantization}"
    os.makedirs(work_dir, exist_ok=True)
    args.work_dir = work_dir  # Update args to pass to other functions
    
    writer = SummaryWriter(work_dir)

    train_loader, test_loader, model, optimizer, scheduler, switch_hook = setup_training_components(args)

    print(f"Device used: {model.device}")
    print(f"Model weights on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using {args.quantization} quantization method")
    
    if args.quantization == "log":
        print(f"Using threshold value: {args.threshold}")
    elif args.quantization == "mixed":
        print(f"Using weight threshold value: {args.weight_threshold}")
    
    print("\nConfig Object Tracking:")
    for name, module in model.named_modules():
        if hasattr(module, 'config'):
            config_type = type(module.config).__name__
            print(f"{name:30} | config_id={id(module.config)} | type={config_type}")
            
            # Print different config details based on type
            if hasattr(module.config, 'threshold'):
                print(f"{' '*32} | threshold={module.config.threshold}")
            if hasattr(module.config, 'weight_threshold'):
                print(f"{' '*32} | weight_threshold={module.config.weight_threshold}")
    
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