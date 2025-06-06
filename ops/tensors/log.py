# ops/tensors/log.py
import torch
from typing import Optional
from .base import QuantizedTensorBase

class LogQuantizedTensor(QuantizedTensorBase):
    """Log quantized tensor implementation."""
    
    def __init__(self, q1, a, s, q2, s_err, r=None) -> None:
        self.q1 = q1  # Primary quantization exponents
        self.a = a    # Scale factors (row/channel-wise max)
        self.s = s    # Sign of original values
        self.r = r    # Real values (for STE)
        
        self.q2 = q2       # Secondary quantization exponents (optional)
        self.s_err = s_err # Sign of error values (optional)

    def dequantize(self) -> torch.Tensor:
        if self.r is not None:
            return self.r
        
        # Ensure a is broadcastable to weight shape
        if self.a.dim() == 1:  # Per-channel case
            a_expanded = self.a.view(-1, *([1] * (self.q1.dim() - 1)))
        else:  # Per-tensor case
            a_expanded = self.a
        
        # Primary reconstruction
        prim = self.s * a_expanded * torch.pow(2.0, -self.q1.float())
        
        # Secondary reconstruction if available
        if self.q2 is not None and self.s_err is not None:
            second = self.s_err * a_expanded * torch.pow(2.0, -self.q2.float())
            return prim + second
        
        return prim

 

    # def dequantize(self) -> torch.Tensor:
    #     if self.r is not None:
    #         return self.r
    #     else:
    #         eps = 1e-8
    #         # Primary reconstruction
    #         prim = (self.s * 
    #                self.a.view(*self.a.shape, *([1] * (self.q1.dim() - self.a.dim()))) * 
    #                torch.maximum((2.0 ** (-self.q1)), torch.tensor(eps, device=self.q1.device)))
            
    #         # Secondary reconstruction if available
    #         if self.q2 is not None and self.s_err is not None:
    #             second = (self.s_err * 
    #                      self.a.view(*self.a.shape, *([1] * (self.q2.dim() - self.a.dim()))) * 
    #                      torch.maximum((2.0 ** (-self.q2)), torch.tensor(eps, device=self.q2.device)))
    #             return prim + second
    
    #         return prim

    def map(self, func):
        return LogQuantizedTensor(
            func(self.q1),
            self.a, 
            self.s,
            None if self.q2 is None else func(self.q2),
            None if self.s_err is None else func(self.s_err),
            None if self.r is None else func(self.r)
        )

    @property
    def shape(self):
        return self.q1.shape
    
    @property
    def device(self):
        return self.q1.device
    
    @property
    def dtype(self):
        return self.q1.dtype
    
    def __repr__(self):
        return (f"LogQuantizedTensor(shape={self.shape}, "
                f"device={self.device}, dtype={self.dtype})")