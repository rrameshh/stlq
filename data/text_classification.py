# data/text_classification.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Tuple, List, Dict, Optional
import os

# Set cache directory
os.environ["HF_DATASETS_CACHE"] = "/scratch/roshnir-profile/qat/qat/hf_cache"

class SimpleTextClassificationDataset(Dataset):
    """Simple dataset for text classification tasks"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Simple tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return (
            encoding['input_ids'].flatten(),
            encoding['attention_mask'].flatten(), 
            torch.tensor(label, dtype=torch.long)
        )


def get_imdb_dataloaders(batch_size: int = 32, max_length: int = 128,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    """Load IMDB sentiment analysis dataset"""
    print("Loading IMDB dataset...")
    
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Use BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    train_dataset = SimpleTextClassificationDataset(
        dataset['train']['text'], 
        dataset['train']['label'],
        tokenizer,
        max_length
    )
    
    test_dataset = SimpleTextClassificationDataset(
        dataset['test']['text'],
        dataset['test']['label'], 
        tokenizer,
        max_length
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, 2  # 2 classes (positive/negative)


def get_sst2_dataloaders(batch_size: int = 32, max_length: int = 128,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    """Load SST-2 sentiment analysis dataset (smaller than IMDB)"""
    print("Loading SST-2 dataset...")
    
    # Load dataset
    dataset = load_dataset("glue", "sst2")
    
    # Use BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets
    train_dataset = SimpleTextClassificationDataset(
        dataset['train']['sentence'],
        dataset['train']['label'], 
        tokenizer,
        max_length
    )
    
    test_dataset = SimpleTextClassificationDataset(
        dataset['validation']['sentence'],  # SST-2 uses 'validation' as test
        dataset['validation']['label'],
        tokenizer,
        max_length
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, 2


# Helper function for TinyBERT validation
def validate_tinybert(model, test_loader, epoch: int, writer, device):
    """Validation function specifically for TinyBERT"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) 
            labels = labels.to(device)
            
            # Forward pass
            logits, loss = model(input_ids, attention_mask, labels=labels)
            
            # Accumulate loss and accuracy
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    print(f'TinyBERT Validation - Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Log to TensorBoard
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return accuracy