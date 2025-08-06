from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class QuantizationConfig:
    method: str = "linear"  # "linear", "log"
    momentum: float = 0.1
    device: Optional[torch.device] = None
    
    bits: int = 8
    
    threshold: float = 1e-5
    eps: float = 1e-8
    
    def __post_init__(self):
        """Validate and set defaults after initialization"""
        if self.method not in ["linear", "log"]:
            raise ValueError(f"Unknown method: {self.method}")
        
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"