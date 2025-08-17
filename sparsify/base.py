# sparsification/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

class SparsificationMethod(ABC):
    """Base class for all sparsification methods"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return method name for logging"""
        pass
    
    @abstractmethod
    def apply(self, model: nn.Module, target_sparsity: float) -> Dict[str, Any]:
        """
        Apply sparsification to model
        
        Args:
            model: Model to sparsify
            target_sparsity: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Dict with results including actual_sparsity, method info, etc.
        """
        pass

def get_quantizable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all layers that can be quantized (and thus sparsified)"""
    quantizable = []
    
    for name, module in model.named_modules():
        # Look for quantized layers in your framework
        if hasattr(module, 'linear') and hasattr(module, 'strategy'):
            quantizable.append((name, module))
        elif hasattr(module, 'conv2d') and hasattr(module, 'strategy'):
            quantizable.append((name, module))
    
    return quantizable

def get_weight_tensor(module: nn.Module) -> torch.Tensor:
    """Extract weight tensor from quantized module"""
    if hasattr(module, 'linear'):
        return module.linear.weight
    elif hasattr(module, 'conv2d'):
        return module.conv2d.weight
    else:
        raise ValueError(f"Cannot extract weight from {type(module)}")

def get_quantization_strategy(module: nn.Module):
    """Get quantization strategy from module"""
    if hasattr(module, 'strategy'):
        return module.strategy
    return None

def is_critical_layer(name: str) -> bool:
    """Check if layer should be preserved from sparsification"""
    critical_patterns = [
        'patch_embed',  # ViT patch embedding
        'head',         # Classification head
        'norm',         # Layer norms
        'cls_token',    # Class token
        'pos_embed',    # Position embedding
        'dist_token'    # Distillation token (DeiT)
    ]
    return any(pattern in name for pattern in critical_patterns)

def compute_actual_sparsity(model: nn.Module) -> float:
    """Compute actual sparsity of model"""
    total_params = 0
    zero_params = 0
    
    for name, module in get_quantizable_layers(model):
        weight = get_weight_tensor(module)
        total_params += weight.numel()
        zero_params += (weight == 0).sum().item()
    
    return zero_params / total_params if total_params > 0 else 0.0