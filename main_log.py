# Modified training script with threshold configuration
# train.py

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
from networks.log_resnet import resnet18, BasicBlock
from ops import LogQuantizedOperator, enable_quantization, disable_quantization

from ops.log import (
    LogQuantConfig, LogQuantize, LogQuantizedAdaptiveAvgPool2d, 
    LogQuantizedConv2dBatchNorm2dReLU, LogQuantizedFlatten, 
    LogQuantizedLinear, LogQuantizedMaxPool2d, LogQuantizedReLU, 
    LogQuantizedAdd, LogQuantizedTensor
)



class SwitchQuantizationModeHook:
    """
    Hook to switch from calibration phase to quantization-aware training phase.
    
    During the first switch_iter iterations, the model calibrates statistics
    without applying quantization to activations. After switch_iter iterations,
    it enables activation quantization.
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
            # Use the helper function to enable quantization on all operators
            enable_quantization(self.model)
            self.switched = True
            return True
        return False


class CIFAR10ResNet(nn.Module):
    """
    ResNet model adapted for CIFAR-10 with log quantization support.
    """
    # def __init__(self, device, threshold=1e-5):
    #     super().__init__()

    #     # Pass threshold to the model explicitly
    #     model = resnet18(num_classes=10, device=device, threshold=threshold)

        
    #     # Modify the first layer to work with CIFAR-10 images (32x32)
    #     model.conv1 = LogQuantizedConv2dBatchNorm2dReLU(
    #         3, 64, kernel_size=3, stride=1, padding=1, 
    #         bias=False, activation="relu", 
    #         config=model.config,  # Share the configuration
    #         device=device
    #     )

    #     # Remove maxpool as it's too aggressive for small CIFAR images
    #     model.maxpool = nn.Identity()
        
        
    #     self.model = model
    #     self.device = device
    #     self.model.to(self.device)
    def __init__(self, device, threshold=1e-5):
        super().__init__()
        
        # Create the model with shared config for internal layers
        model = resnet18(num_classes=10, device=device, threshold=threshold)
        
        # Create a separate config for the first layer
        first_layer_config = LogQuantConfig(
            momentum=0.1,
            threshold=threshold,
            eps=1e-8,
            bits=8,
            device=device
        )
        
        # Replace first layer with one using the separate conf ig
        model.conv1 = LogQuantizedConv2dBatchNorm2dReLU(
            3, 64, kernel_size=3, stride=1, padding=1, 
            bias=False, activation="relu", 
            config=first_layer_config,  # Separate config
            device=device
        )
        
        # Rest of initialization
        model.maxpool = nn.Identity()
        self.model = model
        self.device = device
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
    parser.add_argument("--work-dir", default="./output_log", type=str)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int, 
                      help="Iteration to switch to activation quantization")
    parser.add_argument("--threshold", default=1000000, type=float,
                      help="Threshold for second-order quantization")
    
    args = parser.parse_args()
    
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(args.work_dir)
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    model = CIFAR10ResNet(device=args.device, threshold=args.threshold)
    print(f"Device used: {model.device}")
    print(f"Model weights on CUDA: {next(model.parameters()).is_cuda}")
    print(f"Using threshold value: {args.threshold}")
    
    # quantization is disabled for calibration phase
    disable_quantization(model)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, 
        momentum=0.9, weight_decay=1e-4
    )
    

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )
    
    switch_hook = SwitchQuantizationModeHook(
        model=model, switch_iter=args.switch_iter
    )
    
    # Training loop
    best_accuracy = 0.0
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
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(model, optimizer, accuracy, epoch, args.work_dir)
        
        # Update the learning rate
        scheduler.step()
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f'{args.work_dir}/final_model.pth')
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()