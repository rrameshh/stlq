#ops/strategies/log.py
from .base import QuantizationStrategy
from ..tensors.log import LogQuantizedTensor
import torch

class LogStrategy(QuantizationStrategy):
    
    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True):
        """
        1. Calculate per-channel or per-tensor max absolute values (a)
        2. Get sign tensor (s)
        3. Compute primary quantization: q1 = -log2(|weight|/a + eps)
        4. Calculate residual error
        5. Optionally compute secondary quantization (q2) if error > threshold
        """
        config = self.config  # Assuming config has threshold, eps, max_value, etc.
        
        if per_channel:
            return self._quantize_weight_per_channel(weight, config)
        else:
            return self._quantize_weight_per_tensor(weight, config)
    
    def _quantize_weight_per_channel(self, weight: torch.Tensor, config):
        """Per-channel log quantization - your exact logic."""
        # Reshape for per-channel scaling
        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, rest]
        
        # Get max absolute value per output channel
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        
        # Reshape a for broadcasting
        a_view = a.view(a.shape[0], *([1] * (weight.dim() - 1)))
        
        # Get sign tensor
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # Compute exponents (primary quantization)
        normalized = torch.abs(weight) / a_view + config.eps
        q1 = -torch.log2(normalized)
        
        # Clamp to int8 range and round
        max_value = (2 ** config.bits) - 1
        q1 = torch.clamp(q1.round(), 0, max_value).to(torch.int8)

        # Calculate residual error
        err = (weight / a_view + config.eps) - (2 ** -q1)

        # Conditionally create second-order quantization based on threshold
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > config.threshold):
            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            
            max_value = (2 ** config.bits) - 1
            normalized_err = torch.abs(err) / a_view + config.eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, max_value).to(torch.int8)
        
        # Return with a.squeeze() to match your original logic
        return LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err)
    
    def _quantize_weight_per_tensor(self, weight: torch.Tensor, config):
        """Per-tensor log quantization - your exact logic."""
        # Per-tensor scaling
        a = weight.abs().max()
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        
        # Get sign tensor
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # Compute exponents
        normalized = torch.abs(weight) / a + config.eps
        q1 = -torch.log2(normalized)
        
        # Clamp to int8 range and round
        max_value = (2 ** config.bits) - 1
        q1 = torch.clamp(q1.round(), 0, max_value).to(torch.int8)

        # Calculate residual error
        err = (weight / a + config.eps) - (2 ** -q1)

        # Conditionally create second-order quantization based on threshold
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > config.threshold):
            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            
            max_value = (2 ** config.bits) - 1
            normalized_err = torch.abs(err) / a + config.eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, max_value).to(torch.int8)
        
        return LogQuantizedTensor(q1, a, s, q2, s_err)
    
    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """For log quantization, return bias as-is (matches your implementation)."""
        return bias