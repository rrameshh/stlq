# ops/tensors/linear.py
import torch
from typing import Optional
from .base import QuantizedTensorBase

class LinearQuantizedTensor(QuantizedTensorBase):
    """Linear/Uniform quantized tensor implementation."""
    
    def __init__(self, q: torch.Tensor, s: torch.Tensor, z: torch.Tensor, 
                 r: Optional[torch.Tensor] = None):
        """
        Args:
            q: Quantized values (int8/int32)
            s: Scale factor(s)
            z: Zero point(s)
            r: Real values (for gradient computation, optional)
        """
        self.q = q  # quantized values
        self.s = s  # scale  
        self.z = z  # zero point
        self.r = r  # real values (for STE - Straight Through Estimator)
    
    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        if self.r is not None:
            return self.r
        else:
            # Prevent overflow by casting to int32 first
            return self.s * (self.q.to(torch.int32) - self.z)
    
    def map(self, func):
        """Apply function to all tensor components."""
        return LinearQuantizedTensor(
            func(self.q),
            self.s, 
            self.z,
            None if self.r is None else func(self.r)
        )
    
    @property
    def shape(self):
        return self.q.shape
    
    @property
    def device(self):
        return self.q.device
    
    @property
    def dtype(self):
        return self.q.dtype
    
    def __repr__(self):
        return (f"LinearQuantizedTensor(shape={self.shape}, "
                f"dtype={self.dtype}, device={self.device})")