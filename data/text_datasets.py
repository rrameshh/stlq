import torch
from torch.utils.data import Dataset, DataLoader
import os
import requests
import numpy as np
from typing import List, Tuple
from pathlib import Path
import math
import torch.nn.functional as F

class CharLevelDataset(Dataset):
    """Character-level dataset for language modeling"""
    
    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len
        
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
        print(f"Dataset loaded: {len(text)} characters, {self.vocab_size} unique")
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Get sequence and target (next character prediction)
        seq = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return seq, target


class SimpleTokenDataset(Dataset):
    """Simple word-level tokenization dataset"""
    
    def __init__(self, text: str, seq_len: int = 64, vocab_size: int = 10000):
        self.seq_len = seq_len
        
        # Simple word tokenization
        words = text.lower().split()
        
        # Get most common words
        from collections import Counter
        word_counts = Counter(words)
        most_common = word_counts.most_common(vocab_size - 4)  # Reserve special tokens
        
        # Create vocabulary with special tokens
        self.vocab = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3
        }
        for i, (word, _) in enumerate(most_common):
            self.vocab[word] = i + 4
        
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Convert text to indices
        self.data = []
        for word in words:
            idx = self.vocab.get(word, self.vocab['<unk>'])
            self.data.append(idx)
        
        print(f"Dataset loaded: {len(words)} words, {self.vocab_size} vocab size")
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return seq, target


def download_text_file(url: str, filename: str) -> str:
    """Download text file if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


# def get_shakespeare_dataloaders(batch_size=32, seq_len=128, val_split=0.1, 
#                                num_workers=2, char_level=True):
#     """
#     Load Shakespeare dataset for character or word-level modeling
    
#     Args:
#         batch_size: Batch size for training and validation
#         seq_len: Sequence length for language modeling
#         val_split: Fraction of data to use for validation
#         num_workers: Number of data loading workers
#         char_level: If True, use character-level; if False, use word-level
    
#     Returns:
#         train_loader, val_loader, vocab_size
#     """
    
#     # Download Shakespeare text
#     url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
#     filename = "./data/shakespeare.txt"
#     os.makedirs("./data", exist_ok=True)
    
#     text = download_text_file(url, filename)
    
#     # Split into train/val
#     split_idx = int(len(text) * (1 - val_split))
#     train_text = text[:split_idx]
#     val_text = text[split_idx:]
    
#     # Create datasets
#     if char_level:
#         train_dataset = CharLevelDataset(train_text, seq_len)
#         val_dataset = CharLevelDataset(val_text, seq_len)
#         # Use same vocab as training set
#         val_dataset.char_to_idx = train_dataset.char_to_idx
#         val_dataset.idx_to_char = train_dataset.idx_to_char
#         val_dataset.vocab_size = train_dataset.vocab_size
#         vocab_size = train_dataset.vocab_size
#     else:
#         train_dataset = SimpleTokenDataset(train_text, seq_len)
#         val_dataset = SimpleTokenDataset(val_text, seq_len)

#         print(f"🔍 VOCAB DEBUG:")
#         print(f"  Train vocab size: {len(train_dataset.vocab)}")
#         print(f"  Val vocab size: {len(val_dataset.vocab)}")
        
#         train_words = set(train_dataset.vocab.keys())
#         val_words = set(val_dataset.vocab.keys())
#         overlap = train_words & val_words
        
#         print(f"  Vocabulary overlap: {len(overlap)}")
#         print(f"  Train-only words: {len(train_words - val_words)}")
#         print(f"  Val-only words: {len(val_words - train_words)}")

#         if len(val_words - train_words) > 50:
#             print(f"  🚨 VAL-ONLY WORDS: {list(val_words - train_words)[:20]}")
#         # Use same vocab as training set  
#         val_dataset.vocab = train_dataset.vocab
#         val_dataset.idx_to_word = train_dataset.idx_to_word
#         val_dataset.vocab_size = train_dataset.vocab_size
#         vocab_size = train_dataset.vocab_size

#     print(f"vocab size in text datasets {vocab_size}")
    
#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False, 
#         num_workers=num_workers,
#         pin_memory=True
#     )
    
#     return train_loader, val_loader, vocab_size
    
    # Fixed version for data/text_datasets.py

def get_shakespeare_dataloaders(batch_size=32, seq_len=128, val_split=0.1, 
                               num_workers=2, char_level=True):
    """
    FIXED: Build vocabulary from full dataset before splitting
    """
    
    # Download Shakespeare text
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "./data/shakespeare.txt"
    os.makedirs("./data", exist_ok=True)
    
    text = download_text_file(url, filename)
    
    # Split into train/val
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    if char_level:
        train_dataset = CharLevelDataset(train_text, seq_len)
        val_dataset = CharLevelDataset(val_text, seq_len)
        # Use same vocab as training set
        val_dataset.char_to_idx = train_dataset.char_to_idx
        val_dataset.idx_to_char = train_dataset.idx_to_char
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
    else:
        # FIXED: Build vocabulary from FULL text, then apply to both splits
        print("Building vocabulary from full dataset...")
        
        # Tokenize full text to get complete vocabulary
        full_words = text.lower().split()
        
        # Get most common words from FULL dataset
        from collections import Counter
        word_counts = Counter(full_words)
        vocab_size = 10000  # Keep your desired vocab size
        most_common = word_counts.most_common(vocab_size - 4)  # Reserve special tokens
        
        # Create shared vocabulary
        shared_vocab = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3
        }
        for i, (word, _) in enumerate(most_common):
            shared_vocab[word] = i + 4
        
        shared_vocab_size = len(shared_vocab)
        shared_idx_to_word = {v: k for k, v in shared_vocab.items()}
        
        print(f"Built shared vocabulary: {shared_vocab_size} words")
        
        # Create train dataset with shared vocab
        train_dataset = SimpleTokenDataset(train_text, seq_len, vocab_size)
        # OVERRIDE with shared vocabulary
        train_dataset.vocab = shared_vocab
        train_dataset.idx_to_word = shared_idx_to_word
        train_dataset.vocab_size = shared_vocab_size
        
        # Reprocess training data with shared vocab
        train_words = train_text.lower().split()
        train_dataset.data = []
        for word in train_words:
            idx = shared_vocab.get(word, shared_vocab['<unk>'])
            train_dataset.data.append(idx)
        
        # Create val dataset with shared vocab
        val_dataset = SimpleTokenDataset(val_text, seq_len, vocab_size)
        # OVERRIDE with shared vocabulary
        val_dataset.vocab = shared_vocab
        val_dataset.idx_to_word = shared_idx_to_word
        val_dataset.vocab_size = shared_vocab_size
        
        # Reprocess validation data with shared vocab
        val_words = val_text.lower().split()
        val_dataset.data = []
        for word in val_words:
            idx = shared_vocab.get(word, shared_vocab['<unk>'])
            val_dataset.data.append(idx)
        
        vocab_size = shared_vocab_size
        
        # VERIFICATION DEBUG:
        print(f"🔍 FIXED VOCAB DEBUG:")
        print(f"  Train vocab size: {len(train_dataset.vocab)}")
        print(f"  Val vocab size: {len(val_dataset.vocab)}")
        print(f"  Vocabularies identical: {train_dataset.vocab == val_dataset.vocab}")
        
        # Check for unknown tokens
        train_unks = sum(1 for idx in train_dataset.data if idx == shared_vocab['<unk>'])
        val_unks = sum(1 for idx in val_dataset.data if idx == shared_vocab['<unk>'])
        print(f"  Train <unk> tokens: {train_unks}/{len(train_dataset.data)} ({100*train_unks/len(train_dataset.data):.1f}%)")  
        print(f"  Val <unk> tokens: {val_unks}/{len(val_dataset.data)} ({100*val_unks/len(val_dataset.data):.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab_size


def get_simple_text_dataloaders(text_file: str, batch_size=32, seq_len=128, 
                               val_split=0.1, num_workers=2, char_level=True):
    """
    Load any text file for language modeling
    
    Args:
        text_file: Path to text file
        batch_size: Batch size for training and validation
        seq_len: Sequence length for language modeling
        val_split: Fraction of data to use for validation
        num_workers: Number of data loading workers
        char_level: If True, use character-level; if False, use word-level
    
    Returns:
        train_loader, val_loader, vocab_size
    """
    
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into train/val
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    if char_level:
        train_dataset = CharLevelDataset(train_text, seq_len)
        val_dataset = CharLevelDataset(val_text, seq_len)
        # Use same vocab as training set
        val_dataset.char_to_idx = train_dataset.char_to_idx
        val_dataset.idx_to_char = train_dataset.idx_to_char
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
    else:
        train_dataset = SimpleTokenDataset(train_text, seq_len)
        val_dataset = SimpleTokenDataset(val_text, seq_len)
        # Use same vocab as training set
        val_dataset.vocab = train_dataset.vocab
        val_dataset.idx_to_word = train_dataset.idx_to_word
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, vocab_size


def get_wikitext_dataloaders(batch_size=32, seq_len=128, num_workers=2, char_level=False):
    """
    Load WikiText-2 dataset (requires datasets library)
    
    Args:
        batch_size: Batch size for training and validation
        seq_len: Sequence length for language modeling  
        num_workers: Number of data loading workers
        char_level: If True, use character-level; if False, use word-level
    
    Returns:
        train_loader, val_loader, vocab_size
    """
    try:
        from datasets import load_dataset
        
        # Load WikiText-2
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Combine train and validation text
        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])
        
        # Create datasets
        if char_level:
            train_dataset = CharLevelDataset(train_text, seq_len)
            val_dataset = CharLevelDataset(val_text, seq_len)
            # Use same vocab as training set
            val_dataset.char_to_idx = train_dataset.char_to_idx
            val_dataset.idx_to_char = train_dataset.idx_to_char
            val_dataset.vocab_size = train_dataset.vocab_size
            vocab_size = train_dataset.vocab_size
        else:
            train_dataset = SimpleTokenDataset(train_text, seq_len)
            val_dataset = SimpleTokenDataset(val_text, seq_len)
            # Use same vocab as training set
            val_dataset.vocab = train_dataset.vocab
            val_dataset.idx_to_word = train_dataset.idx_to_word
            val_dataset.vocab_size = train_dataset.vocab_size
            vocab_size = train_dataset.vocab_size
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, vocab_size
        
    except ImportError:
        print("datasets library not found. Install with: pip install datasets")
        print("Falling back to Shakespeare dataset...")
        return get_shakespeare_dataloaders(batch_size, seq_len, 0.1, num_workers, char_level)


# Utility functions for text generation and evaluation
def generate_text(model, dataset, device, prompt="", max_length=100, temperature=1.0):
    """Generate text from trained model"""
    model.eval()
    
    if hasattr(dataset, 'char_to_idx'):  # Character-level
        if prompt:
            input_ids = torch.tensor([dataset.char_to_idx.get(c, 0) for c in prompt[-model.max_seq_len:]], 
                                   dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        generated = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature)
        text = ''.join([dataset.idx_to_char[idx.item()] for idx in generated[0]])
        
    else:  # Word-level
        if prompt:
            words = prompt.lower().split()[-model.max_seq_len:]
            input_ids = torch.tensor([dataset.vocab.get(w, dataset.vocab['<unk>']) for w in words],
                                   dtype=torch.long, device=device).unsqueeze(0)
        else:
            input_ids = torch.tensor([[dataset.vocab['<bos>']]], dtype=torch.long, device=device)
        
        generated = model.generate(input_ids, max_new_tokens=max_length, temperature=temperature)
        words = [dataset.idx_to_word[idx.item()] for idx in generated[0]]
        text = ' '.join(words)
    
    return text


# def compute_perplexity(model, data_loader, device):
#     """Compute perplexity on validation set"""
#     model.eval()
#     total_loss = 0
#     total_tokens = 0
    
#     with torch.no_grad():
#         for inputs, targets in data_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             logits, loss = model(inputs, targets)
            
#             total_loss += loss.item() * targets.numel()
#             total_tokens += targets.numel()
    
#     avg_loss = total_loss / total_tokens
#     perplexity = torch.exp(torch.tensor(avg_loss))
    
#     return perplexity.item()

def compute_perplexity(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get raw logits, compute loss manually
            logits = model(inputs)  # Don't pass targets
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            # Compute per-token loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_targets.view(-1), 
                reduction='sum'  # Sum, don't average
            )
            
            total_loss += loss.item()
            total_tokens += shift_targets.numel()
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

