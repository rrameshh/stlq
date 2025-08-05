


import torch
import torchvision.transforms as transforms
import torchvision
import os

from torch.utils.data import DistributedSampler, DataLoader, Dataset

from datasets import load_dataset

from PIL import Image
import os

os.environ["HF_DATASETS_CACHE"] = "/scratch/roshnir-profile/qat/qat/hf_cache"

class ImageNet100Dataset(Dataset):
    """Custom Dataset wrapper for ImageNet-100 from Hugging Face datasets"""
    
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        # Ensure image is in RGB format (some might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_imagenet100_dataloaders(batch_size=128, num_workers=4, image_size=224):
    """
    Create data loaders for ImageNet-100.
    
    Args:
        batch_size: Batch size for training and validation
        num_workers: Number of worker processes for data loading
        image_size: Size to resize images to (default 224 for ImageNet)
    
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    
    # Load the dataset from Hugging Face
    print("Loading ImageNet-100 dataset from Hugging Face...")
    ds = load_dataset("clane9/imagenet-100")
    
    # Create custom datasets
    train_dataset = ImageNet100Dataset(ds["train"], transform=train_transform)
    val_dataset = ImageNet100Dataset(ds["validation"], transform=val_transform)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # Helps with batch norm stability
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader


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