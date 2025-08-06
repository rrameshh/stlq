#ops/strategies/new_log.py
from .base import QuantizationStrategy
from ..tensors.new_log import LogQuantizedTensor
import torch
from ..quant_config import QuantizationConfig
import numpy as np

class LogStrategy(QuantizationStrategy):

    def __init__(self, config):
        super().__init__(config)
       
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

        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, rest]
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]

        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        a_view = a.view(a.shape[0], *([1] * (weight.dim() - 1)))
        s = torch.sign(weight)

        zero_mask = (weight == 0.0)
        non_zero_mask = ~zero_mask

        max_value = (2 ** config.bits) - 1
        q1 = torch.zeros_like(weight, dtype=torch.uint8)

        if torch.any(non_zero_mask):
            normalized = torch.abs(weight[non_zero_mask]) / a_view.expand_as(weight)[non_zero_mask]
           
            normalized = normalized.float()
            q1_non_zero = -torch.log2(normalized)
            q1_non_zero = torch.clamp((q1_non_zero.round()), 0, max_value)
            q1[non_zero_mask] = q1_non_zero.to(torch.uint8) # check dtype


        normalized_weight = (weight / a_view).float()
        reconstructed_normalized = torch.zeros_like(normalized_weight, dtype = torch.float32)
        reconstructed_normalized = reconstructed_normalized.float()


        if torch.any(non_zero_mask):
            reconstructed_normalized[non_zero_mask] = torch.pow(2.0, -q1[non_zero_mask].float())
        
        err_normalized = normalized_weight - reconstructed_normalized
        err_normalized = err_normalized.float()
        err_magnitude = torch.abs(err_normalized)
        second_word_mask = (err_magnitude > config.threshold) & non_zero_mask  # Exclude zeros

        q2 = None
        s_err = None

        if torch.any(second_word_mask):
            q2 = torch.zeros_like(q1)
            s_err = torch.zeros_like(s)
            selected_err_normalized = err_normalized[second_word_mask]
            s_err_selected = torch.sign(selected_err_normalized)

            selected_err_mag = torch.abs(selected_err_normalized).float()
            if torch.any(selected_err_mag <= 0):
                print(f"WARNING: Zero or negative error magnitudes found: {(selected_err_mag <= 0).sum().item()}")
                selected_err_mag = torch.clamp(selected_err_mag, min=1e-8)  # Prevent log(0)

            q2_selected = -torch.log2(selected_err_mag)
        
            q2_selected = torch.clamp((q2_selected.round()), 2, max_value + 2).to(torch.uint8)
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        result =  LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err, second_word_mask)
        result = LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err, second_word_mask)
    
        return result

    
    def _quantize_weight_per_tensor(self, weight: torch.Tensor, config):

        a = weight.abs().max()
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        s = torch.sign(weight)

        zero_mask = (weight == 0.0)
        non_zero_mask = ~zero_mask
        
        max_value = (2 ** config.bits) - 1
        
        q1 = torch.zeros_like(weight, dtype=torch.uint8)
        if torch.any(non_zero_mask):
            normalized = torch.abs(weight[non_zero_mask]) / a
            q1_non_zero = -torch.log2(normalized)
            q1_non_zero = torch.clamp((q1_non_zero.round()), 0, max_value)
            q1[non_zero_mask] = q1_non_zero.to(torch.uint8)
        

        normalized_weight = weight / a
        reconstructed_normalized = torch.zeros_like(normalized_weight)
        if torch.any(non_zero_mask):
            reconstructed_normalized[non_zero_mask] = torch.pow(2.0, -q1[non_zero_mask])
        
        err = normalized_weight - reconstructed_normalized
        err_mag = torch.abs(err)
        second_word_mask = (err_mag > config.threshold) & non_zero_mask  # Exclude zeros


        q2 = None
        s_err = None
  
        if torch.any(second_word_mask):
            q2 = torch.zeros_like(q1)
            s_err = torch.zeros_like(s)
            sel_err = err[second_word_mask]
            s_err_selected = torch.sign(sel_err)

            selected_err_mag = torch.abs(sel_err)
            q2_selected = -torch.log2(selected_err_mag)
            q2_selected = torch.clamp((q2_selected.round()), 2, max_value + 2).to(torch.uint8)
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        return LogQuantizedTensor(q1, a, s, q2, s_err, second_word_mask)
    
    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """For log quantization, return bias as-is (matches your implementation)."""
        return bias
