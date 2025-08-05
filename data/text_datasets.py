# data/language.py - Clean text dataset handling
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import os
import requests
import math
import torch.nn.functional as F
from collections import Counter
from typing import Tuple, List

# Set cache directory

os.environ["HF_DATASETS_CACHE"] = "/scratch/roshnir-profile/qat/qat/hf_cache"


# ============================================================================
# Language Modeling Datasets (TinyGPT)
# ============================================================================

class CharLevelDataset(Dataset):
    """Character-level dataset for language modeling."""
    
    def __init__(self, text: str, seq_len: int = 128):
        self.seq_len = seq_len
        
        # Create character mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert to indices
        self.data = [self.char_to_idx[ch] for ch in text]
        
        print(f"Char dataset: {len(text)} chars, vocab={self.vocab_size}")
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return seq, target

class WordLevelDataset(Dataset):
    """Word-level dataset for language modeling."""
    
    def __init__(self, text: str, seq_len: int = 64, vocab_size: int = 10000):
        self.seq_len = seq_len
        
        # Tokenize
        words = text.lower().split()
        
        # Build vocabulary from most common words
        word_counts = Counter(words)
        most_common = word_counts.most_common(vocab_size - 4)
        
        # Create vocab with special tokens
        self.vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        for i, (word, _) in enumerate(most_common):
            self.vocab[word] = i + 4
        
        self.vocab_size = len(self.vocab)
        self.idx_to_word = {v: k for k, v in self.vocab.items()}
        
        # Convert to indices
        self.data = [self.vocab.get(word, self.vocab['<unk>']) for word in words]
        
        print(f"Word dataset: {len(words)} words, vocab={self.vocab_size}")
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return seq, target

# ============================================================================
# Text Classification Datasets (TinyBERT)
# ============================================================================

class TextClassificationDataset(Dataset):
    """Dataset for text classification tasks."""
    
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

# ============================================================================
# Data Loader Functions
# ============================================================================

def _download_text(url: str, filename: str) -> str:
    """Download text file if needed."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def _create_shared_vocab(text: str, vocab_size: int = 10000):
    """Create shared vocabulary from full text (fixes vocab mismatch bug)."""
    words = text.lower().split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(vocab_size - 4)
    
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    for i, (word, _) in enumerate(most_common):
        vocab[word] = i + 4
    
    return vocab, {v: k for k, v in vocab.items()}

def _apply_vocab_to_text(text: str, vocab: dict) -> List[int]:
    """Convert text to indices using shared vocabulary."""
    words = text.lower().split()
    return [vocab.get(word, vocab['<unk>']) for word in words]

def get_shakespeare_dataloaders(batch_size=32, seq_len=128, val_split=0.1, 
                               num_workers=2, char_level=True):
    """Load Shakespeare dataset."""
    # Download
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = "./data/shakespeare.txt"
    os.makedirs("./data", exist_ok=True)
    text = _download_text(url, filename)
    
    # Split
    split_idx = int(len(text) * (1 - val_split))
    train_text, val_text = text[:split_idx], text[split_idx:]
    
    if char_level:
        # Character-level
        train_dataset = CharLevelDataset(train_text, seq_len)
        val_dataset = CharLevelDataset(val_text, seq_len)
        
        # Share vocabulary
        val_dataset.char_to_idx = train_dataset.char_to_idx
        val_dataset.idx_to_char = train_dataset.idx_to_char
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
        
    else:
        # Word-level with shared vocabulary (FIXED)
        shared_vocab, shared_idx_to_word = _create_shared_vocab(text)
        
        # Create datasets with shared vocab
        train_dataset = WordLevelDataset(train_text, seq_len)
        val_dataset = WordLevelDataset(val_text, seq_len)
        
        # Override with shared vocabulary
        for dataset in [train_dataset, val_dataset]:
            dataset.vocab = shared_vocab
            dataset.idx_to_word = shared_idx_to_word
            dataset.vocab_size = len(shared_vocab)
        
        # Reprocess data with shared vocab
        train_dataset.data = _apply_vocab_to_text(train_text, shared_vocab)
        val_dataset.data = _apply_vocab_to_text(val_text, shared_vocab)
        vocab_size = len(shared_vocab)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, vocab_size

def get_wikitext_dataloaders(batch_size=32, seq_len=128, num_workers=2, char_level=False):
    """Load WikiText-2 dataset."""
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])
        
        if char_level:
            train_dataset = CharLevelDataset(train_text, seq_len)
            val_dataset = CharLevelDataset(val_text, seq_len)
            val_dataset.char_to_idx = train_dataset.char_to_idx
            val_dataset.idx_to_char = train_dataset.idx_to_char
            val_dataset.vocab_size = train_dataset.vocab_size
            vocab_size = train_dataset.vocab_size
        else:
            train_dataset = WordLevelDataset(train_text, seq_len)
            val_dataset = WordLevelDataset(val_text, seq_len)
            val_dataset.vocab = train_dataset.vocab
            val_dataset.idx_to_word = train_dataset.idx_to_word
            val_dataset.vocab_size = train_dataset.vocab_size
            vocab_size = train_dataset.vocab_size
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
        
        return train_loader, val_loader, vocab_size
        
    except ImportError:
        print("datasets library not found. Falling back to Shakespeare...")
        return get_shakespeare_dataloaders(batch_size, seq_len, 0.1, num_workers, char_level)

def get_custom_text_dataloaders(text_file: str, batch_size=32, seq_len=128, 
                               val_split=0.1, num_workers=2, char_level=True):
    """Load custom text file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    split_idx = int(len(text) * (1 - val_split))
    train_text, val_text = text[:split_idx], text[split_idx:]
    
    if char_level:
        train_dataset = CharLevelDataset(train_text, seq_len)
        val_dataset = CharLevelDataset(val_text, seq_len)
        val_dataset.char_to_idx = train_dataset.char_to_idx
        val_dataset.idx_to_char = train_dataset.idx_to_char
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
    else:
        train_dataset = WordLevelDataset(train_text, seq_len)
        val_dataset = WordLevelDataset(val_text, seq_len)
        val_dataset.vocab = train_dataset.vocab
        val_dataset.idx_to_word = train_dataset.idx_to_word
        val_dataset.vocab_size = train_dataset.vocab_size
        vocab_size = train_dataset.vocab_size
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, vocab_size

def get_imdb_dataloaders(batch_size=32, max_length=128, num_workers=4):
    """Load IMDB sentiment dataset."""
    print("Loading IMDB dataset...")
    
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = TextClassificationDataset(
        dataset['train']['text'], dataset['train']['label'], tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        dataset['test']['text'], dataset['test']['label'], tokenizer, max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, 2

def get_sst2_dataloaders(batch_size=32, max_length=128, num_workers=4):
    """Load SST-2 sentiment dataset."""
    print("Loading SST-2 dataset...")
    
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = TextClassificationDataset(
        dataset['train']['sentence'], dataset['train']['label'], tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        dataset['validation']['sentence'], dataset['validation']['label'], tokenizer, max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, 2

# ============================================================================
# Utility Functions
# ============================================================================

def compute_perplexity(model, data_loader, device):
    """Compute perplexity for language models."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get logits without targets to avoid automatic loss computation
            logits = model(inputs)
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            
            # Compute loss manually
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_targets.numel()
    
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def generate_text(model, dataset, device, prompt="", max_length=100, temperature=1.0):
    """Generate text from trained model."""
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