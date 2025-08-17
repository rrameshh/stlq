
from .base import QuantizationStrategy
from ..tensors.new_log import LogQuantizedTensor
import torch
from ..quant_config import QuantizationConfig
import numpy as np

class AdaptiveLogStrategy(QuantizationStrategy):
    """Log quantization strategy with adaptive thresholds - NO ZERO SPECIAL CASING"""

    def __init__(self, config):
        super().__init__(config)
        self.target_second_word_ratio = getattr(config, 'target_second_word_ratio', 0.25)  # 25% default
        self.adaptive_threshold = getattr(config, 'adaptive_threshold', True)
       
    def quantize_weight(self, weight: torch.Tensor, per_channel: bool = True):
        """Quantize weights with adaptive thresholds - treats ALL weights normally"""
        config = self.config
        if per_channel:
            return self._quantize_weight_per_channel_adaptive(weight, config)
        else:
            return self._quantize_weight_per_tensor_adaptive(weight, config)
    
    def _quantize_weight_per_channel_adaptive(self, weight: torch.Tensor, config):
        """Per-channel quantization - NO SPECIAL ZERO HANDLING"""
        
        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, rest]
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]

        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        a_view = a.view(a.shape[0], *([1] * (weight.dim() - 1)))
        s = torch.sign(weight)

        # REMOVED: No special zero_mask or non_zero_mask
        # ALL weights get quantized normally
        max_value = (2 ** config.bits) - 1
        q1 = torch.zeros_like(weight, dtype=torch.uint8)

        # Quantize ALL weights (including zeros)
        normalized = torch.abs(weight) / a_view.expand_as(weight)
        normalized = normalized.float()
        
        # Handle zeros by adding small epsilon to avoid log(0)
        normalized = torch.clamp(normalized, min=config.eps)
        
        q1_all = -torch.log2(normalized)
        q1_all = torch.clamp((q1_all.round()), 0, max_value)
        q1 = q1_all.to(torch.uint8)

        # Reconstruction - treat ALL codes normally
        normalized_weight = (weight / a_view).float()
        reconstructed_normalized = torch.sign(weight) * torch.pow(2.0, -q1.float())
        
        err_normalized = normalized_weight - reconstructed_normalized
        err_normalized = err_normalized.float()
        err_magnitude = torch.abs(err_normalized)
        
        # ADAPTIVE THRESHOLD COMPUTATION - consider ALL weights
        if self.adaptive_threshold:
            # ALL errors are valid now (no zero exclusion)
            valid_errors = err_magnitude  # Consider all weights
            
            if valid_errors.numel() > 0:
                # Use quantile to set threshold for target ratio
                quantile_level = 1.0 - self.target_second_word_ratio
                adaptive_threshold = torch.quantile(valid_errors, quantile_level)
                
                # Ensure threshold is reasonable (not too small)
                adaptive_threshold = torch.maximum(
                    adaptive_threshold, 
                    torch.tensor(config.eps, device=weight.device)
                )
                
                # ALL weights can potentially get second words
                second_word_mask = (err_magnitude > adaptive_threshold)
            else:
                second_word_mask = torch.zeros_like(weight, dtype=torch.bool)
        else:
            # Use fixed threshold - ALL weights considered
            second_word_mask = (err_magnitude > config.threshold)

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
                selected_err_mag = torch.clamp(selected_err_mag, min=1e-8)

            q2_selected = -torch.log2(selected_err_mag)
            q2_selected = torch.clamp((q2_selected.round()), 0, max_value).to(torch.uint8)  # CHANGED: Allow q2=0
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        result = LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err, second_word_mask)
        
        # Log statistics for debugging
        if hasattr(self, '_debug_mode') and self._debug_mode:
            actual_ratio = second_word_mask.sum().item() / weight.numel()
            threshold_used = adaptive_threshold.item() if self.adaptive_threshold else config.threshold
            print(f"Target SW ratio: {self.target_second_word_ratio:.3f}, "
                  f"Actual SW ratio: {actual_ratio:.3f}, "
                  f"Threshold used: {threshold_used:.6f}")
            
            # Debug q=0 statistics
            q1_zeros = (q1 == 0).sum().item()
            q2_zeros = (q2 == 0).sum().item() if q2 is not None else 0
            print(f"q1=0 count: {q1_zeros}/{q1.numel()}, q2=0 count: {q2_zeros}")
    
        return result

    def _quantize_weight_per_tensor_adaptive(self, weight: torch.Tensor, config):
        """Per-tensor quantization - NO SPECIAL ZERO HANDLING"""
        
        a = weight.abs().max()
        a = torch.maximum(a, torch.tensor(config.eps, device=weight.device))
        s = torch.sign(weight)

        # REMOVED: No special zero_mask or non_zero_mask
        max_value = (2 ** config.bits) - 1
        
        q1 = torch.zeros_like(weight, dtype=torch.uint16)
        
        # Quantize ALL weights (including zeros)
        normalized = torch.abs(weight) / a
        normalized = torch.clamp(normalized, min=config.eps)  # Avoid log(0)
        
        q1_all = -torch.log2(normalized)
        q1_all = torch.clamp((q1_all.round()), 0, max_value)
        q1 = q1_all.to(torch.uint16)

        # Compute errors for ALL weights
        normalized_weight = weight / a
        reconstructed_normalized = torch.sign(weight) * torch.pow(2.0, -q1.float())
        
        err = normalized_weight - reconstructed_normalized
        err_mag = torch.abs(err)
        
        # ADAPTIVE THRESHOLD COMPUTATION - consider ALL weights
        if self.adaptive_threshold:
            valid_errors = err_mag  # All errors are valid
            
            if valid_errors.numel() > 0:
                quantile_level = 1.0 - self.target_second_word_ratio
                adaptive_threshold = torch.quantile(valid_errors, quantile_level)
                adaptive_threshold = torch.maximum(
                    adaptive_threshold, 
                    torch.tensor(config.eps, device=weight.device)
                )
                second_word_mask = (err_mag > adaptive_threshold)
            else:
                second_word_mask = torch.zeros_like(weight, dtype=torch.bool)
        else:
            second_word_mask = (err_mag > config.threshold)

        q2 = None
        s_err = None
  
        if torch.any(second_word_mask):
            q2 = torch.zeros_like(q1)
            s_err = torch.zeros_like(s)
            sel_err = err[second_word_mask]
            s_err_selected = torch.sign(sel_err)

            selected_err_mag = torch.abs(sel_err)
            q2_selected = -torch.log2(selected_err_mag)
            q2_selected = torch.clamp((q2_selected.round()), 0, max_value)  # CHANGED: Allow q2=0
            q2[second_word_mask] = q2_selected
            s_err[second_word_mask] = s_err_selected

        return LogQuantizedTensor(q1, a, s, q2, s_err, second_word_mask)

    def set_debug_mode(self, debug=True):
        """Enable debug logging"""
        self._debug_mode = debug

    def quantize_bias(self, bias: torch.Tensor, quantized_input, quantized_weight):
        """For log quantization, return bias as-is"""
        return bias