# ops/strategies/linear.py
from .base import QuantizationStrategy
from ..tensors.linear import LinearQuantizedTensor
import torch

class LinearStrategy(QuantizationStrategy):
    """Linear quantization strategy - matches your original QuantizedConv2dBatchNorm2dReLU._quantize_weight"""
    
    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True):        
        if per_channel and weight.ndim >= 2:
            return self._quantize_weight_per_channel(weight)
        else:
            return self._quantize_weight_per_tensor(weight)
    
    def _quantize_weight_per_channel(self, weight: torch.Tensor):
        # Quantize weight to -127 ~ 127. Note that -128 is excluded.
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        a = weight_reshaped.min(dim=1).values
        b = weight_reshaped.max(dim=1).values
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))
        z = z.reshape(z.shape[0], 1, 1, 1)
        s = s.reshape(s.shape[0], 1, 1, 1)

        q = torch.maximum(torch.minimum(
            weight / s + z, torch.tensor(127)), torch.tensor(-127)).round().to(torch.int8)
        
        return LinearQuantizedTensor(q, s, z)
    
    def _quantize_weight_per_tensor(self, weight: torch.Tensor):
        # Quantize weight to -127 ~ 127. Note that -128 is excluded.
        a = weight.min()
        b = weight.max()
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))

        q = torch.maximum(torch.minimum(
            weight / s + z, torch.tensor(127)), torch.tensor(-127)).round().to(torch.int8)
        
        return LinearQuantizedTensor(q, s, z)
    
    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """
        Unified bias quantization handling both Conv2d and Linear cases.
        
        Conv2d: weight_scale is [out_ch, 1, 1, 1] -> reshape to [out_ch]
        Linear: weight_scale is scalar -> broadcast to [out_features]
        """
        bias_scale_raw = quantized_weight.s * quantized_input.s
        
        if bias_scale_raw.numel() > 1:
            # Per-channel case (Conv2d): flatten multi-dimensional scale
            s = bias_scale_raw.reshape(-1)
        else:
            # Per-tensor case (Linear): broadcast scalar to match bias shape
            s = bias_scale_raw.expand_as(bias)
            
        z = torch.zeros_like(s).to(torch.int32)
        q = (bias / s).round().to(torch.int32)
        
        return LinearQuantizedTensor(q, s, z)
