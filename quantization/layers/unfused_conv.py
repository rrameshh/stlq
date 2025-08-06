
from typing import Optional, Union
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..base import QuantizedOperatorBase
from ..tensors.linear import LinearQuantizedTensor
from ..tensors.new_log import LogQuantizedTensor
from ..strategies.factory import create_strategy
from ..quant_config import QuantizationConfig

from .quantized import Quantizer
r
QuantizedTensorType = Union[LinearQuantizedTensor, LogQuantizedTensor]

class QConvBNUnfused(Quantizer):
    """applies conv and bn separately (no weight fusion)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', activation=None,
                 config: QuantizationConfig = None):
        super().__init__(config)
        
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode, device=config.device)
        self.bn2d = nn.BatchNorm2d(out_channels, device=config.device)

        assert self.conv2d.padding_mode == "zeros"
        self.activation = activation
        assert self.activation in ["relu", "hardswish", None], f"Unsupported activation: {self.activation}"

    def _apply_activation(self, output):
        if self.activation == "relu":
            return F.relu(output)
        elif self.activation == "hardswish":
            return F.hardswish(output)
        elif self.activation is None:
            return output
        

    def _quantize_weight(self, weight: torch.Tensor):
        return self.strategy.quantize_weight(weight, per_channel=True)

    def _quantize_bias(self, quantized_input: QuantizedTensorType, quantized_weight: QuantizedTensorType, bias: torch.Tensor):
        return self.strategy.quantize_bias(bias, quantized_input, quantized_weight)

    def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
            quantized_conv_weight = self._quantize_weight(self.conv2d.weight)
            
            # Simulated quantized conv output
            simulated_conv_output = F.conv2d(
                input, quantized_conv_weight.dequantize(), self.conv2d.bias,
                self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            )
        
        # Real conv output (original weights)
        real_conv_output = self.conv2d(input)
        
        # STE on convolution
        ste_conv_output = real_conv_output - (real_conv_output - simulated_conv_output).detach()
        
        # BN on convolved output
        bn_output = self.bn2d(ste_conv_output)
        final_output = self._apply_activation(bn_output)
        
        self.update_stats(final_output)
        return final_output

    def _quantize_conv_bias_only(self, bias: torch.Tensor):
        """Quantize conv bias independently (not accounting for BN scaling)"""
        # For unfused operations, quantize conv bias on its own scale
        # BN will handle the scaling afterward
        # return self.strategy.quantize_bias_independent(bias)
        return bias
    

    def _activation_quantized_forward(self, input: QuantizedTensorType) -> QuantizedTensorType:
        
        with torch.no_grad():
            # Quantize conv weights only
            quantized_conv_weight = self._quantize_weight(self.conv2d.weight)
            quantized_conv_bias = None
            
            # Handle conv bias quantization
            if self.conv2d.bias is not None:
                quantized_conv_bias = self._quantize_conv_bias_only(self.conv2d.bias)
                conv_bias_val = quantized_conv_bias.dequantize() if hasattr(quantized_conv_bias, 'dequantize') else quantized_conv_bias
            else:
                conv_bias_val = None
            
            # simulated quantized conv output (for STE)
            simulated_conv_output = F.conv2d(
                input.dequantize(), quantized_conv_weight.dequantize(), conv_bias_val,
                self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            )
        
        # real conv output (original weights)
        real_conv_output = self.conv2d(input.dequantize())
        ste_conv_output = real_conv_output - (real_conv_output - simulated_conv_output).detach()
        
         # BN on convolved output
        bn_output = self.bn2d(ste_conv_output)
        final_output = self._apply_activation(bn_output)
        
        # update stats and quantize output
        self.update_stats(final_output)
        quantized_output = self.quantize_output(final_output)
        
        return quantized_output



    def forward(self, input: Union[torch.Tensor, QuantizedTensorType]) -> Union[torch.Tensor, QuantizedTensorType]:
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return self._activation_not_quantized_forward(input)

