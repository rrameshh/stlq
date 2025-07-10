
from typing import Optional, Union
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..base import QuantizedOperatorBase
from ..tensors.linear import LinearQuantizedTensor
from ..tensors.new_log import LogQuantizedTensor
from ..strategies.factory import create_strategy
from ..quant_config import QuantizationConfig

from .all import UnifiedQuantizedOperator

# Type alias for any quantized tensor
QuantizedTensorType = Union[LinearQuantizedTensor, LogQuantizedTensor]

class UnifiedQuantizedConvBatchNormUnfused(UnifiedQuantizedOperator):
    """Unfused Conv2d+BN+ReLU layer - applies conv and bn separately (no weight fusion)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', activation=None,
                 config: QuantizationConfig = None):
        super().__init__(config)
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode, device=config.device)
        self.bn2d = nn.BatchNorm2d(out_channels, device=config.device)
        assert self.conv2d.padding_mode == "zeros"
        self.activation = activation
        assert self.activation in ["relu", None]

    def _apply_activation(self, output):
        if self.activation == "relu":
            return F.relu(output)
        elif self.activation is None:
            return output

    def _quantize_weight(self, weight: torch.Tensor):
        """Use strategy for weight quantization"""
        return self.strategy.quantize_weight(weight, per_channel=True)

    def _quantize_bias(self, quantized_input: QuantizedTensorType, quantized_weight: QuantizedTensorType, bias: torch.Tensor):
        """Use strategy for bias quantization"""
        return self.strategy.quantize_bias(bias, quantized_input, quantized_weight)

    def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Proper unfused forward pass - quantize only conv weights, keep BN separate"""
        
        with torch.no_grad():
            # Step 1: Quantize ONLY the conv weights (no fusion with BN)
            quantized_conv_weight = self._quantize_weight(self.conv2d.weight)
            
            # Step 2: Simulate the quantized conv operation
            if self.conv2d.bias is not None:
                # For unfused, we can quantize conv bias independently since BN will handle scaling
                quantized_conv_bias = self._quantize_conv_bias_only(self.conv2d.bias)
                conv_bias_val = quantized_conv_bias.dequantize() if hasattr(quantized_conv_bias, 'dequantize') else quantized_conv_bias
                
                simulated_conv_output = F.conv2d(
                    input, quantized_conv_weight.dequantize(), conv_bias_val,
                    self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
                )
            else:
                simulated_conv_output = F.conv2d(
                    input, quantized_conv_weight.dequantize(), None,
                    self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
                )
            
            # Step 3: Apply BatchNorm to simulated conv output (BN params are NOT quantized)
            simulated_bn_output = self.bn2d(simulated_conv_output)
            
            # Step 4: Apply activation
            simulated_output = self._apply_activation(simulated_bn_output)
            
            # Step 5: Update stats for this layer's output
            self.update_stats(simulated_output)

        # Real forward pass: conv -> bn -> activation (all with original weights)
        real_conv_output = self.conv2d(input)
        real_bn_output = self.bn2d(real_conv_output)
        real_output = self._apply_activation(real_bn_output)
        
        # Straight-through estimator
        return real_output - (real_output - simulated_output).detach()

    def _quantize_conv_bias_only(self, bias: torch.Tensor):
        """Quantize conv bias independently (not accounting for BN scaling)"""
        # For unfused operations, we quantize conv bias on its own scale
        # BN will handle the scaling afterward
        # return self.strategy.quantize_bias_independent(bias)
        return bias

    # You'll also need to update your quantized forward pass to be consistent:
    def _activation_quantized_forward(self, input: QuantizedTensorType) -> QuantizedTensorType:
        """Forward pass when input activations are already quantized - proper unfused"""
        
        with torch.no_grad():
            # Step 1: Quantize conv weights only
            quantized_conv_weight = self._quantize_weight(self.conv2d.weight)
            
            # Step 2: Handle conv bias quantization
            if self.conv2d.bias is not None:
                quantized_conv_bias = self._quantize_conv_bias_only(self.conv2d.bias)
                conv_bias_val = quantized_conv_bias.dequantize() if hasattr(quantized_conv_bias, 'dequantize') else quantized_conv_bias
                
                simulated_conv_output = F.conv2d(
                    input.dequantize(), quantized_conv_weight.dequantize(), conv_bias_val,
                    self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
                )
            else:
                simulated_conv_output = F.conv2d(
                    input.dequantize(), quantized_conv_weight.dequantize(), None,
                    self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
                )
            
            # Step 3: Apply BatchNorm (with float32 BN parameters)
            simulated_bn_output = self.bn2d(simulated_conv_output)
            
            # Step 4: Apply activation
            simulated_output = self._apply_activation(simulated_bn_output)
            
            # Step 5: Update stats and quantize output
            self.update_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)

        # Real forward pass (unfused): conv -> bn -> activation
        real_conv_output = self.conv2d(input.dequantize())
        real_bn_output = self.bn2d(real_conv_output)
        real_output = self._apply_activation(real_bn_output)
        
        # Straight-through estimator
        quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output
        

    # def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
    #     """Debug the normal forward pass"""
        
    #     # Check input
    #     if torch.isnan(input).any():
    #         print(f"🚨 INPUT has NaN!")
    #         return input
        
    #     # Step 1: Conv2d
    #     conv_output = self.conv2d(input)
    #     if torch.isnan(conv_output).any():
    #         print(f"🚨 CONV OUTPUT has NaN!")
    #         print(f"   Conv weight range: [{self.conv2d.weight.min():.6f}, {self.conv2d.weight.max():.6f}]")
    #         print(f"   Conv weight has NaN: {torch.isnan(self.conv2d.weight).any()}")
    #         if self.conv2d.bias is not None:
    #             print(f"   Conv bias has NaN: {torch.isnan(self.conv2d.bias).any()}")
    #         return torch.zeros_like(conv_output)
        
    #     # Step 2: BatchNorm
    #     bn_output = self.bn2d(conv_output)
    #     if torch.isnan(bn_output).any():
    #         print(f"🚨 BATCHNORM OUTPUT has NaN!")
    #         print(f"   BN running_mean has NaN: {torch.isnan(self.bn2d.running_mean).any()}")
    #         print(f"   BN running_var has NaN: {torch.isnan(self.bn2d.running_var).any()}")
    #         print(f"   BN weight has NaN: {torch.isnan(self.bn2d.weight).any()}")
    #         print(f"   BN bias has NaN: {torch.isnan(self.bn2d.bias).any()}")
    #         return torch.zeros_like(bn_output)
        
    #     # Step 3: Activation
    #     real_output = self._apply_activation(bn_output)
    #     if torch.isnan(real_output).any():
    #         print(f"🚨 ACTIVATION OUTPUT has NaN!")
    #         return torch.zeros_like(real_output)
        
    #     # print(f"✅ Normal forward pass successful")
    #     return real_output






    def forward(self, input: Union[torch.Tensor, QuantizedTensorType]) -> Union[torch.Tensor, QuantizedTensorType]:
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return self._activation_not_quantized_forward(input)

