# qat/ops/base.py
import torch
import torch.nn as nn

class QuantizedTensorBase:
    """Base class for quantized tensors, to be inherited by specific implementations."""
    
    def dequantize(self) -> torch.Tensor:
        """Convert quantized tensor back to floating point."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def map(self, func):
        """Apply a function to all tensor components."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def reshape(self, *shape):
        """Reshape all tensor components."""
        return self.map(lambda x: x.reshape(*shape))
    
    def permute(self, *permutation):
        """Permute dimensions of all tensor components."""
        return self.map(lambda x: x.permute(*permutation))
    
    @property
    def shape(self):
        """Return the shape of the quantized tensor."""
        raise NotImplementedError("Subclasses must implement this method")


class QuantizedOperatorBase(nn.Module):
    """Base class for quantized operators, containing common functionality."""
    
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__()
        self.activation_quantization = False
        self.momentum = momentum
        self.device = device
        
        # Register buffers that all quantization methods will need
        self.register_buffer('num_batches_tracked',
                         torch.tensor(0, dtype=torch.long, device=device))
    
    def update_stats(self, output: torch.Tensor):
        """Update statistics needed for quantization."""
        if not self.training:
            return
        
        self._update_stats_impl(output)
        self.num_batches_tracked.data.copy_(self.num_batches_tracked + 1)
    
    def _update_stats_impl(self, output: torch.Tensor):
        """Implementation of statistics update, to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def quantize_output(self, output: torch.Tensor):
        """Quantize an output tensor."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def quantize_weight(self, weight: torch.Tensor, per_channel=True):
        """Quantize a weight tensor."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def quantize_bias(self, quantized_input, quantized_weight, bias):
        """Quantize a bias tensor."""
        raise NotImplementedError("Subclasses must implement this method")