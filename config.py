# config.py - Missing from your refactored code
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml
from pathlib import Path
import torch

@dataclass
class DataConfig:
    dataset: str = "cifar10"  # cifar10, imagenet100, shakespeare, imdb, sst2
    batch_size: int = 128
    num_workers: int = 4
    seq_len: int = 128  # For language models
    
@dataclass 
class ModelConfig:
    name: str = "resnet18"  # resnet18, vit_small, etc.
    num_classes: int = 10
    img_size: int = 224
    vocab_size: Optional[int] = None  # Set by dataset for language models
    pretrained: bool = False
    
@dataclass
class QuantizationConfig:
    method: str = "linear"  # linear, log
    momentum: float = 0.1
    bits: int = 8
    threshold: float = 1e-5  # For log quantization
    eps: float = 1e-8
    switch_iteration: int = 5000  # When to enable activation quantization
    
@dataclass
class TrainingConfig:
    num_epochs: int = 200
    lr: float = 0.1
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    log_interval: int = 100
    
@dataclass
class SystemConfig:
    device: str = "cuda:0"
    work_dir: str = "./output"
    seed: int = 42
    
@dataclass 
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create config objects
        config = cls()
        
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'quantization' in data:
            config.quantization = QuantizationConfig(**data['quantization'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'system' in data:
            config.system = SystemConfig(**data['system'])
            
        return config
    
    def save_yaml(self, path: Path):
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'quantization': self.quantization.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def apply_overrides(self, overrides: List[str]):
        """Apply command line overrides (key=value format)."""
        for override in overrides:
            if '=' not in override:
                continue
                
            key, value = override.split('=', 1)
            keys = key.split('.')
            
            # Parse value
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            # Apply override
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)
    
    def print_summary(self):
        """Print config summary."""
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Dataset: {self.data.dataset}")
        print(f"Model: {self.model.name}")
        print(f"Quantization: {self.quantization.method}")
        print(f"Batch size: {self.data.batch_size}")
        print(f"Learning rate: {self.training.lr}")
        print(f"Device: {self.system.device}")
        print(f"Work dir: {self.system.work_dir}")
        print("=" * 60)


        