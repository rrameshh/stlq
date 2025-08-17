# config.py - Enhanced with validation and defaults
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path
import torch

@dataclass
class DataConfig:
    dataset: str = "cifar10"
    batch_size: int = 128
    num_workers: int = 4
    seq_len: int = 128  # For language models
    
    def __post_init__(self):
        valid_datasets = ["cifar10", "imagenet100", "shakespeare", "imdb", "sst2"]
        if self.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset '{self.dataset}'. Valid: {valid_datasets}")

@dataclass 
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 10
    img_size: int = 224
    vocab_size: Optional[int] = None
    pretrained: bool = False
    
    def __post_init__(self):
        # Auto-set num_classes based on dataset
        if hasattr(self, '_dataset'):
            dataset_classes = {
                "cifar10": 10,
                "imagenet100": 100,
                "imdb": 2,
                "sst2": 2
            }
            if self._dataset in dataset_classes:
                self.num_classes = dataset_classes[self._dataset]

@dataclass
class QuantizationConfig:
    method: str = "linear"
    momentum: float = 0.1
    bits: int = 8
    threshold: float = 1e-5
    eps: float = 1e-8
    switch_iteration: int = 5000
    adaptive_threshold: bool = False
    target_second_word_ratio: float = 0.25
    
    def __post_init__(self):
        if self.method not in ["linear", "log", "adaptive_log"]:
            raise ValueError(f"Invalid quantization method '{self.method}'. Valid: ['linear', 'log']")
        if self.bits not in [4, 8, 16]:
            print(f"Warning: Unusual bit width {self.bits}. Common values: [4, 8, 16]")

@dataclass
class SparsificationConfig:
    enabled: bool = False
    method: str = "quantization_informed_progressive"
    target_ratio: float = 0.5  # Target sparsity ratio
    
    # Progressive joint optimization settings
    cost_penalty: int = 1.0

    adaptation_interval: int = 5      # Adapt every N epochs
    initial_sw_target: float = 0.25   # Starting point (will be diversified)
    sw_learning_rate: float = 0.1     # How fast to adapt
    efficiency_threshold: float = 200.0  # Efficiency target
    
    
    # Training schedule
    apply_after_epoch: int = 50  # Apply sparsification after this epoch
    
    # Baseline methods for comparison
    baseline_method: str = "magnitude"  # "magnitude", "snip" for comparison
    
    def __post_init__(self):
        if self.enabled:
            if not 0.0 < self.target_ratio < 1.0:
                raise ValueError(f"target_ratio must be between 0 and 1, got {self.target_ratio}")
            # if abs(self.complexity_weight + self.magnitude_weight - 1.0) > 0.1:
            #     print(f"Warning: α + β = {self.complexity_weight + self.magnitude_weight:.2f}, should sum to ~1.0")


@dataclass
class TrainingConfig:
    num_epochs: int = 200
    lr: float = 0.1
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    log_interval: int = 100
    
@dataclass
class SystemConfig:
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    work_dir: str = "./output"
    seed: int = 42
    
@dataclass 
class Config:
    """Main configuration with auto-validation and smart defaults."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    sparsification: SparsificationConfig = field(default_factory=SparsificationConfig)  # Add this line 
    
    def __post_init__(self):
        """Auto-adjust related settings."""
        # Link dataset to model for auto num_classes
        self.model._dataset = self.data.dataset
        self.model.__post_init__()
        
        # Auto-adjust work_dir based on quantization method
        if self.system.work_dir == "./output":
            suffix = f"_{self.quantization.method}"  # Initialize suffix here
            self.system.work_dir = f"./output_{self.quantization.method}"
            if self.sparsification.enabled:  # Add this block
                suffix += f"_sparse{int(self.sparsification.target_ratio*100)}"


        
        # Validate device
        if "cuda" in self.system.device and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, switching to CPU")
            self.system.device = "cpu"
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load and validate config from YAML."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        

        for section_name in ['data', 'model', 'quantization', 'training', 'system', 'sparsification']:
            if section_name in data:
                section = getattr(config, section_name)
                for key, value in data[section_name].items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        print(f"Warning: Unknown config key '{section_name}.{key}' ignored")
        

        config.__post_init__()
        return config
    
    def save_yaml(self, path: Path):
        """Save config to YAML."""

        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'data': self.data.__dict__,
            'model': {k: v for k, v in self.model.__dict__.items() if not k.startswith('_')},
            'quantization': self.quantization.__dict__,
            'training': self.training.__dict__,
            'system': self.system.__dict__,
            'sparsification': self.sparsification.__dict__,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    
    def print_summary(self):
        """Print clean config summary."""
        print("=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Dataset: {self.data.dataset} (classes: {self.model.num_classes})")
        print(f"Model: {self.model.name} {'(pretrained)' if self.model.pretrained else ''}")
        print(f"Quantization: {self.quantization.method} ({self.quantization.bits}-bit)")
        print(self.sparsification)
        if self.sparsification.enabled:
            print(f"Sparsification: {self.sparsification.method} ({self.sparsification.target_ratio:.1%} target)")
            print(f"  - Apply after epoch: {self.sparsification.apply_after_epoch}")
  
        else:
            print("Sparsification: Disabled")
        print(f"Training: {self.training.num_epochs} epochs @ LR {self.training.lr}")
        print(f"Device: {self.system.device}")
        print(f"Output: {self.system.work_dir}")

        print("=" * 60)