#ops/strategies/log.py
from .base import QuantizationStrategy
from ..tensors.log import LogQuantizedTensor
import torch
from ..quant_config import QuantizationConfig
import numpy as np

class LogStrategy(QuantizationStrategy):
    
    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True):
        """
        1. Calculate per-channel or per-tensor max absolute values (a)
        2. Get sign tensor (s)
        3. Handle zero weights explicitly (q=0, reconstructs to 0)
        4. Compute primary quantization: q1 = -log2(|weight|/a) for non-zero weights
        5. Calculate residual error
        6. Optionally compute secondary quantization (q2) if error > threshold
        """
        config = self.config  # Assuming config has threshold, eps, max_value, etc.
        
        if per_channel:
            return self._quantize_weight_per_channel(weight, config)
        else:
            return self._quantize_weight_per_tensor(weight, config)
    
    def _quantize_weight_per_channel(self, weight: torch.Tensor, config):
        """Per-channel log quantization with proper zero handling following STLQ approach."""
        # Reshape for per-channel scaling
        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, rest]
       
        # Get max absolute value per output channel
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        a_view = a.view(a.shape[0], *([1] * (weight.dim() - 1)))
        
        # Handle signs
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)

        # STLQ approach: zeros get special handling
        zero_mask = (weight == 0.0)
        non_zero_mask = ~zero_mask
        
        max_value = (2 ** config.bits) - 1
        
        # Initialize q1 with zeros (quantization code 0 represents zero weights)
        q1 = torch.zeros_like(weight, dtype=torch.uint8)
        
        # Only compute log quantization for non-zero weights
        if torch.any(non_zero_mask):
            normalized = torch.abs(weight[non_zero_mask]) / a_view.expand_as(weight)[non_zero_mask]
            q1_non_zero = -torch.log2(normalized)
            # For non-zero weights, quantization codes start from 1 (following STLQ paper)
            q1_non_zero = torch.clamp(q1_non_zero.round(), 1, max_value)
            q1[non_zero_mask] = q1_non_zero.to(torch.uint8)
        
        # Calculate residual error
        # For zero weights: err = 0 - 0 = 0 (perfect reconstruction)
        # For non-zero weights: err = original - reconstructed
        reconstructed = torch.zeros_like(weight)
        if torch.any(non_zero_mask):
            reconstructed[non_zero_mask] = s[non_zero_mask] * torch.pow(2.0, -q1[non_zero_mask].float()) * a_view.expand_as(weight)[non_zero_mask]
        
        err = weight - reconstructed
        
        # Zero weights will have zero error, so they won't participate in second-word quantization
        err_magnitude = torch.abs(err)
        second_word_mask = (err_magnitude > config.threshold) & non_zero_mask  # Exclude zeros

        q2 = None
        s_err = None
        
        if torch.any(second_word_mask):
            q2 = torch.zeros_like(q1)
            s_err = torch.zeros_like(s)

            sel_err = err[second_word_mask]  # [num_selected]
            a_expanded = a_view.expand_as(weight)
            sel_a = a_expanded[second_word_mask]

            s_err_selected = torch.sign(sel_err)
            s_err_selected = torch.where(s_err_selected == 0, torch.ones_like(s_err_selected), s_err_selected)

            normalized_err = torch.abs(sel_err) / sel_a
            q2_selected = -torch.log2(normalized_err)
            # Apply offset of 2 as per HSTLQ paper (Eq. 9): clamp(..., 2, 2^bit-1+2)
            # But store the offset-adjusted value for hardware efficiency
            q2_selected = torch.clamp(q2_selected.round() - 2, 0, max_value).to(torch.uint8)
            
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        return LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err, second_word_mask)
    
    def _quantize_weight_per_tensor(self, weight: torch.Tensor, config):
        """Per-tensor log quantization with proper zero handling following STLQ approach."""
        # Per-tensor scaling
        a = weight.abs().max()
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        
        # Handle signs
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # STLQ approach: zeros get special handling
        zero_mask = (weight == 0.0)
        non_zero_mask = ~zero_mask
        
        max_value = (2 ** config.bits) - 1
        
        # Initialize q1 with zeros (quantization code 0 represents zero weights)
        q1 = torch.zeros_like(weight, dtype=torch.uint8)
        
        # Only compute log quantization for non-zero weights
        if torch.any(non_zero_mask):
            normalized = torch.abs(weight[non_zero_mask]) / a
            q1_non_zero = -torch.log2(normalized)
            # For non-zero weights, quantization codes start from 1 (following STLQ paper)
            q1_non_zero = torch.clamp(q1_non_zero.round(), 1, max_value)
            q1[non_zero_mask] = q1_non_zero.to(torch.uint8)

        # Calculate residual error
        # For zero weights: err = 0 - 0 = 0 (perfect reconstruction)
        # For non-zero weights: err = original - reconstructed
        reconstructed = torch.zeros_like(weight)
        if torch.any(non_zero_mask):
            reconstructed[non_zero_mask] = s[non_zero_mask] * torch.pow(2.0, -q1[non_zero_mask].float()) * a
        
        err = weight - reconstructed

        # Zero weights will have zero error, so they won't participate in second-word quantization
        err_mag = torch.abs(err)
        second_word_mask = (err_mag > config.threshold) & non_zero_mask  # Exclude zeros

        q2 = None
        s_err = None
  
        if torch.any(second_word_mask):
            q2 = torch.zeros_like(q1)
            s_err = torch.zeros_like(s)
            sel_err = err[second_word_mask]
            sel_a = a

            s_err_selected = torch.sign(sel_err)
            s_err_selected = torch.where(s_err_selected == 0, torch.ones_like(s_err_selected), s_err_selected)

            normalized_err = torch.abs(sel_err) / sel_a
            q2_selected = -torch.log2(normalized_err)
            # Apply offset of 2 as per HSTLQ paper (Eq. 9): clamp(..., 2, 2^bit-1+2)
            # But store the offset-adjusted value for hardware efficiency
            q2_selected = torch.clamp(q2_selected.round() - 2, 0, max_value).to(torch.uint8)
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        return LogQuantizedTensor(q1, a, s, q2, s_err, second_word_mask)
    
    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """For log quantization, return bias as-is (matches your implementation)."""
        return bias