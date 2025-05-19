# train_cifar.py
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import QAT components
from ops import enable_quantization, disable_quantization
from networks.resnet_factory import resnet18


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
            enable_quantization(self.model)
            self.switched = True
            return True
        return False


class CIFAR10ResNet(nn.Module):
    """
    ResNet model adapted for CIFAR-10 with quantization support.
    """
    def __init__(self, quantization_method="linear", device=None, **kwargs):
        super().__init__()

        # Store parameters that we'll need for layer creation
        self.device = device
        # self.threshold = kwargs.get('threshold', 1e-5)
        self.quantization_method = quantization_method

        # Create model with the specified quantization method
        # Handle parameters differently based on the quantization method
        if quantization_method.lower() == "linear":
            # For linear quantization, we only pass compatible parameters
            # and handle device separately
            model_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['device', 'threshold']}
            
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


def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """Create data loaders for CIFAR-10."""
    # Data transformations
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, epoch, switch_hook, writer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(train_loader):
        # Move data to device
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Check if it's time to switch to activation quantization
        if switch_hook.after_train_iter(epoch * len(train_loader) + i):
            print("Switched to activation quantization")
        
        # Update statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log every 100 batches
        if (i + 1) % 100 == 0:
            accuracy = 100.0 * correct / total
            print(f'Epoch [{epoch+1}], Batch [{i+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/100:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Log to TensorBoard
            iteration = epoch * len(train_loader) + i
            writer.add_scalar('Training/Loss', running_loss/100, iteration)
            writer.add_scalar('Training/Accuracy', accuracy, iteration)
            
            running_loss = 0.0
            correct = 0
            total = 0


def validate(model, test_loader, epoch, writer):
    """Validate the model."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return accuracy


def save_checkpoint(model, optimizer, accuracy, epoch, save_dir):
    """Save model checkpoint."""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': accuracy,
        'epoch': epoch,
    }
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, f'{save_dir}/checkpoint_epoch{epoch+1}.pth')
    print(f'Checkpoint saved at epoch {epoch+1}')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Train QAT ResNet on CIFAR-10")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-epochs", default=200, type=int)
    parser.add_argument("--work-dir", default="./output", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int, 
                       help="Iteration to switch to activation quantization")
    parser.add_argument("--quantization", default="linear", type=str,
                       choices=["linear", "log"], help="Quantization method to use")
    parser.add_argument("--threshold", default=1e-5, type=float,
                       help="Threshold for second-order quantization in log method")
    parser.add_argument("--early-stop", default=10, type=int,
                       help="Early stopping patience (epochs without improvement)")
    
    args = parser.parse_args()
    
    # Create work directory based on quantization method
    work_dir = f"{args.work_dir}_{args.quantization}"
    os.makedirs(work_dir, exist_ok=True)
    
    writer = SummaryWriter(work_dir)
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Create model with specified quantization method
    # Only pass the threshold parameter for log quantization
    model_kwargs = {}
    if args.quantization == "log":
        model_kwargs['threshold'] = args.threshold
    
    model = CIFAR10ResNet(
        quantization_method=args.quantization,
        device=args.device, 
        **model_kwargs
    )
    
    print(f"Device used: {model.device}")
    print(f"Model weights on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using {args.quantization} quantization method")
    if args.quantization == "log":
        print(f"Using threshold value: {args.threshold}")
    
    # Disable quantization for calibration phase
    disable_quantization(model)
    
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
    
    # Training loop
    best_accuracy = 0.0
    epochs_without_improvement = 0
    print(f"Starting training for {args.num_epochs} epochs...")
    print(f"Will switch to quantization after {args.switch_iter} iterations")
    
    for epoch in range(args.num_epochs):
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.num_epochs}, Learning Rate: {current_lr:.6f}")
        
        # Train one epoch
        train_epoch(model, train_loader, optimizer, epoch, switch_hook, writer)
        
        # Validate
        accuracy = validate(model, test_loader, epoch, writer)
        
        # Early stopping and checkpoint saving
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, optimizer, accuracy, epoch, work_dir)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update the learning rate
        scheduler.step()
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f'{work_dir}/final_model.pth')
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()