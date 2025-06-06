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
        """Per-channel log quantization with proper selective second words."""
        # Reshape for per-channel scaling
        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, rest]
       
        # Get max absolute value per output channel
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        a_view = a.view(a.shape[0], *([1] * (weight.dim() - 1)))
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # Compute exponents (primary quantization)
        normalized = torch.abs(weight) / a_view 
        # q1_float = -torch.log2(normalized)
        
 
        max_value = (2 ** config.bits) - 1
        # q1_float = torch.clamp(q1_float, 0, max_value)
        q1 = -torch.log2(normalized)
        q1 = torch.clamp(q1.round(), 0, max_value)


  
        # Calculate residual error
        err = (weight / a_view ) - torch.pow(2.0, -q1)
        # q1 = q1_float.round().to(torch.int8)

        q1 = q1.to(torch.int8)



        # Conditionally create second-order quantization based on threshold
        err_magnitude = torch.abs(err)
        second_word_mask = err_magnitude > config.threshold

        # q2 = None
        # s_err = None

        q2 = torch.zeros_like(q1)
        s_err = torch.zeros_like(s)
        
        if torch.any(second_word_mask):

            # total_elements = second_word_mask.numel()
            # second_order_elements = second_word_mask.sum().item()
            # percentage = 100.0 * second_order_elements / total_elements
            # print(f"DEBUG: Layer using {second_order_elements}/{total_elements} ({percentage:.1f}%) second-order elements, threshold={config.threshold}")

            sel_err = err[second_word_mask]  # [num_selected]
            a_expanded = a_view.expand_as(weight)
            sel_a = a_expanded[second_word_mask]

            s_err_selected = torch.sign(sel_err)
            s_err_selected = torch.where(s_err_selected == 0, torch.ones_like(s_err_selected), s_err_selected)

            normalized_err = torch.abs(sel_err) / sel_a
            q2_selected = -torch.log2(normalized_err)
            q2_selected = torch.clamp(q2_selected.round(), 0, max_value).to(torch.int8)
            
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        return LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err)
    
    def _quantize_weight_per_tensor(self, weight: torch.Tensor, config):
        """Per-tensor log quantization - your exact logic."""
        # Per-tensor scaling
        a = weight.abs().max()
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        normalized = torch.abs(weight) / a
        # q1_float = -torch.log2(normalized)
        q1 = -torch.log2(normalized)

        max_value = (2 ** config.bits) - 1
        # q1_float = torch.clamp(q1_float, 0, max_value)
        q1 = -torch.clamp(q1.round(), 0, max_value)

        err = (weight / a) - torch.pow(2.0, -q1)
        # q1 = q1_float.round().to(torch.int8)
        q1 = q1.to(torch.int8)


        err_mag = torch.abs(err)

        second_word_mask = err_mag > config.threshold

        q2 = None
        s_err = None

        q2 = torch.zeros_like(q1)
        s_err =  torch.zeros_like(s)
  
        if torch.any(second_word_mask):
           sel_err = err[second_word_mask]
           sel_a =  a[second_word_mask] if a.numel() > 1 else a

           s_err_selected = torch.sign(sel_err)
           s_err_selected = torch.where(s_err_selected == 0, torch.ones_like(s_err_selected), s_err_selected)

           normalized_err = torch.abs(sel_err) / sel_a
           q2_selected = -torch.log2(normalized_err)
           q2_selected = torch.clamp(q2_selected.round(), 0, max_value).to(torch.int8)
           q2[second_word_mask] = q2_selected
           s_err[second_word_mask] = s_err_selected
           s_err = s_err.view_as(s)


        return LogQuantizedTensor(q1, a, s, q2, s_err)
    
    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """For log quantization, return bias as-is (matches your implementation)."""
        return bias





