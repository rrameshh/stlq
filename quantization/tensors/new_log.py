import torch
from typing import Optional
from .base import QuantizedTensorBase

class LogQuantizedTensor(QuantizedTensorBase):
    """Log quantized tensor implementation following HSTLQ approach."""
    
    def __init__(self, q1, a, s, q2, s_err, second_word_mask, r=None) -> None:
        self.q1 = q1  # Primary quantization exponents  
        self.a = a    # Scale factors (row/channel-wise max)
        self.s = s    # Sign of original values
        self.r = r    # Real values (for STE)
        
        self.q2 = q2       # Secondary quantization exponents (optional)
        self.s_err = s_err # Sign of error values (optional) 
        self.second_word_mask = second_word_mask

    def dequantize(self) -> torch.Tensor:
        if self.r is not None:
            return self.r
        
        # Ensure a is broadcastable to weight shape
        if self.a.dim() == 1:  # Per-channel case
            a_expanded = self.a.view(-1, *([1] * (self.q1.dim() - 1)))
        else:  # Per-tensor case
            a_expanded = self.a
        
        # Primary reconstruction: s(X)·2^(-XQ1)·α
        zero_mask = (self.q1 == 0)
        prim = torch.zeros_like(self.q1, dtype=torch.float32)
        
        # Only compute for non-zero quantization codes (q1 > 0)
        if torch.any(~zero_mask):
            non_zero_mask = ~zero_mask
            prim[non_zero_mask] = (self.s[non_zero_mask] * 
                                   torch.pow(2.0, -self.q1[non_zero_mask].float()) * 
                                   a_expanded.expand_as(self.q1)[non_zero_mask])
        
        # Secondary reconstruction: s(Err)·2^(-XQ2)·α
        if self.q2 is not None and self.s_err is not None:
            # Second words also follow the same zero handling
            second_zero_mask = (self.q2 == 0)
            second = torch.zeros_like(self.q2, dtype=torch.float32)
            
            # Only compute for non-zero second-word quantization codes
            if torch.any(~second_zero_mask):
                second_non_zero_mask = ~second_zero_mask
                second[second_non_zero_mask] = (self.s_err[second_non_zero_mask] * 
                                                torch.pow(2.0, -(self.q2[second_non_zero_mask].float())) * 
                                                a_expanded.expand_as(self.q2)[second_non_zero_mask])
            
            return prim + second
        
        return prim

    def map(self, func):
        return LogQuantizedTensor(
            func(self.q1),
            self.a, 
            func(self.s),
            None if self.q2 is None else func(self.q2),
            None if self.s_err is None else func(self.s_err),
            None if self.second_word_mask is None else func(self.second_word_mask),
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