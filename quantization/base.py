# ops/base.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Any

class QuantizedTensorBase(ABC):
    """Base class for all quantized tensor implementations."""
    
    @abstractmethod
    def dequantize(self) -> torch.Tensor:
        """Convert quantized tensor back to floating point."""
        pass
    
    @abstractmethod
    def map(self, func: Callable):
        """Apply a function to all tensor components."""
        pass
    
    def reshape(self, *shape):
        """Reshape all tensor components."""
        return self.map(lambda x: x.reshape(*shape))
    
    def permute(self, *permutation):
        """Permute dimensions of all tensor components."""
        return self.map(lambda x: x.permute(*permutation))
    
    @property
    @abstractmethod
    def shape(self):
        """Return the shape of the quantized tensor."""
        pass
    
    @property
    @abstractmethod
    def device(self):
        """Return the device of the quantized tensor."""
        pass
    
    @property
    @abstractmethod
    def dtype(self):
        """Return the data type of the quantized tensor."""
        pass
    
    def to(self, device):
        """Move tensor to device."""
        return self.map(lambda x: x.to(device))
    
    def cuda(self):
        """Move tensor to CUDA."""
        return self.to('cuda')
    
    def cpu(self):
        """Move tensor to CPU."""
        return self.to('cpu')


class QuantizedOperatorBase(nn.Module):
    """
    Base class for quantized operators, containing common functionality.
    
    This class provides the foundation for all quantization methods (linear, log, mixed).
    Subclasses should implement method-specific functionality.
    """
    
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__()
        self.activation_quantization = False
        self.momentum = momentum
        self.device = device
        
        # All quantized operators need to track batches
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long, device=device))
    
    def update_stats(self, output: torch.Tensor):
        """
        Update statistics needed for quantization.
        
        This is a template method - subclasses should override _update_stats_impl
        to provide method-specific statistics updates.
        """
        if not self.training:
            return
        
        self._update_stats_impl(output)
        self.num_batches_tracked.data.copy_(self.num_batches_tracked + 1)
    
    def _update_stats_impl(self, output: torch.Tensor):
        """
        Implementation of statistics update, to be overridden by subclasses.
        
        For linear quantization: update running_min/max
        For log quantization: update running_max_abs  
        For unified operators: route based on config.method
        """
        # Default implementation does nothing - subclasses override
        pass
    
    def quantize_output(self, output: torch.Tensor):
        """
        Quantize an output tensor.
        
        Subclasses should implement this based on their quantization method.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def quantize_weight(self, weight: torch.Tensor, per_channel=True):
        """
        Quantize a weight tensor.
        
        This is optional for operators that don't have weights (like Add, ReLU).
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def quantize_bias(self, quantized_input, quantized_weight, bias):
        """
        Quantize a bias tensor.
        
        This is optional for operators that don't have bias.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_quantization_info(self):
        """Get information about this layer's quantization state."""
        info = {
            'type': type(self).__name__,
            'activation_quantization': self.activation_quantization,
            'momentum': self.momentum,
            'num_batches_tracked': self.num_batches_tracked.item() if self.num_batches_tracked is not None else None,
            'device': str(self.device) if self.device else None
        }
        
        # Add method-specific stats if available
        if hasattr(self, 'running_min') and hasattr(self, 'running_max'):
            info['quantization_method'] = 'linear'
            info['running_min'] = self.running_min.item()
            info['running_max'] = self.running_max.item()
        elif hasattr(self, 'running_max_abs'):
            info['quantization_method'] = 'log'
            info['running_max_abs'] = self.running_max_abs.item()
        elif hasattr(self, 'config'):
            info['quantization_method'] = getattr(self.config, 'method', 'unknown')
            
        return info
    
    def reset_stats(self):
        """Reset quantization statistics to initial state."""
        if hasattr(self, 'num_batches_tracked'):
            self.num_batches_tracked.data.fill_(0)
            
        # Reset method-specific stats
        if hasattr(self, 'running_min') and hasattr(self, 'running_max'):
            self.running_min.data.fill_(0)
            self.running_max.data.fill_(0)
        elif hasattr(self, 'running_max_abs'):
            # Use eps from config if available
            eps = getattr(self.config, 'eps', 1e-8) if hasattr(self, 'config') else 1e-8
            self.running_max_abs.data.fill_(eps)
    
    def __repr__(self):
        info = self.get_quantization_info()
        return f"{info['type']}(activation_quantization={info['activation_quantization']}, method={info.get('quantization_method', 'unknown')})"