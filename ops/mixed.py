from typing import Optional, Union
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base import QuantizedOperatorBase, QuantizedTensorBase
from .linear import QuantizedTensor
from .log import LogQuantUtils, LogQuantConfig


class MixedQuantConfig:
    """Configuration for mixed precision quantization."""
    
    def __init__(
        self,
        # Linear quantization config for activations
        activation_momentum=0.1,
        # Log quantization config for weights
        weight_threshold=1e-5,
        weight_eps=1e-8,
        weight_bits=8,
        device=None
    ):
        self.activation_momentum = activation_momentum
        self.weight_threshold = weight_threshold
        self.weight_eps = weight_eps
        self.weight_bits = weight_bits
        self.device = device
        
        self.log_config = LogQuantConfig(
            momentum= activation_momentum,
            threshold=weight_threshold,
            eps=weight_eps,
            bits=weight_bits,
            device=device
        )

class MixedQuantOperator(QuantizedOperatorBase):
    def __init__(self, config=None, device=None) -> None:
       
        
        if config is None:
            self.config = MixedQuantConfig(device=device)
        else:
            self.config = config

        super().__init__(momentum=config.activation_momentum, device=device)
            
        self.activation_quantization = False
        
        # Linear quantization buffers for activations (like your linear.py)
        self.register_buffer('running_min', torch.zeros(1, device=device))
        self.register_buffer('running_max', torch.zeros(1, device=device))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long, device=device))
        
    def update_min_max_stats(self, output: torch.Tensor):
        if not self.training:
            return
        min = output.min()
        max = output.max()
        eps = torch.ones_like(min) * 1e-6
        min = torch.minimum(-eps, min)
        max = torch.maximum(eps, max)
        if self.num_batches_tracked == 0:
            self.running_min.data.copy_(min)
            self.running_max.data.copy_(max)
        else:
            self.running_min.data.copy_(
                min * self.momentum + self.running_min * (1 - self.momentum))
            self.running_max.data.copy_(
                max * self.momentum + self.running_max * (1 - self.momentum))
        self.num_batches_tracked.data.copy_(self.num_batches_tracked + 1)


    def calc_output_scale_and_zero_point(self):
        z = (127 - self.running_max * 255 /
             (self.running_max - self.running_min)).round().to(torch.int8)
        s = self.running_max / (127 - z.to(torch.float32))
        return s, z

    def quantize_output(self, output: torch.Tensor):
        assert self.num_batches_tracked >= 1
        s, z = self.calc_output_scale_and_zero_point()
        q = torch.maximum(torch.minimum(
            output / s + z, torch.tensor(127)), torch.tensor(-128)).round().to(torch.int8)
        return QuantizedTensor(q, s, z)

    
    def quantize_weight_log(self, weight: torch.Tensor):
        # Use the utility function
        return LogQuantUtils.quantize_weight(weight, self.config.log_config, per_channel=True)
    

class MixedQuantize(MixedQuantOperator):
    def __init__(self, config=None, device=None) -> None:  #
        super().__init__(config, device) 

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, QuantizedTensor]:
        if self.activation_quantization:
            self.update_min_max_stats(input)
            output = self.quantize_output(input)
            return output
        else:
            self.update_min_max_stats(input)
            return input


class MixedQuantizedConv2dBathNorm2dReLU(MixedQuantOperator):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        activation=None,
        config=None,
        device=None
        
    ):
        super().__init__(config, device)
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation,
            groups, bias, padding_mode, device
        )
        self.bn2d = nn.BatchNorm2d(out_channels, device=device)

        assert self.conv2d.padding_mode == "zeros"

        self.activation = activation
        assert self.activation in ["relu", None]

    
    def _apply_activation(self, output):
        if self.activation == "relu":
            return F.relu(output)
        elif self.activation is None:
            return output
        

    def _get_bn2d_mean_and_var(self, input):
        if self.training:
            conv2d_output = self.conv2d(input)
            conv2d_output_reshaped = conv2d_output \
                .transpose(0, 1).reshape(self.conv2d.out_channels, -1)
            mean = conv2d_output_reshaped.mean(1)
            var = conv2d_output_reshaped.var(1)
        else:
            mean = self.bn2d.running_mean
            var = self.bn2d.running_var
        return mean, var
    
    def _get_fused_weight_and_bias(self, input):
        mean, var = self._get_bn2d_mean_and_var(input)
        sqrt_var = torch.sqrt(var + self.bn2d.eps)
        fused_weight = self.conv2d.weight * self.bn2d.weight.reshape(self.conv2d.out_channels, 1, 1, 1) / \
            sqrt_var.reshape(self.conv2d.out_channels, 1, 1, 1)
        bias = torch.zeros_like(mean) \
            if self.conv2d.bias is None else self.conv2d.bias
        fused_bias = (bias - mean) / sqrt_var * \
            self.bn2d.weight + self.bn2d.bias
        return fused_weight, fused_bias
    
    
    def _quantize_bias(self, quantized_input: QuantizedTensor, quantized_weight_log, bias: torch.Tensor):
        """For mixed precision, we need to handle the log-quantized weight properly."""
        # Since weight is log-quantized, we need to get its effective scale
        # For simplicity, we'll use the bias as-is (could be improved)
        return bias
    

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        """Forward with activation quantization enabled."""
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(
                input.dequantize()
            )
            # Use LOG quantization for weights
            quantized_fused_weight_log = self.quantize_weight_log(fused_weight)
            quantized_fused_bias = self._quantize_bias(
                input, quantized_fused_weight_log, fused_bias)
            
            simulated_output = self._apply_activation(F.conv2d(
                input.dequantize(), quantized_fused_weight_log.dequantize(),
                quantized_fused_bias, self.conv2d.stride,
                self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            
            # Use LINEAR quantization for activations
            self.update_min_max_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)

        real_output = self._apply_activation(
            self.bn2d(self.conv2d(input.dequantize())))
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output
    

    def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward with activation quantization disabled."""
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(input)
            # Use LOG quantization for weights even during calibration
            quantized_fused_weight_log = self.quantize_weight_log(fused_weight)
            simulated_output = self._apply_activation(F.conv2d(
                input, quantized_fused_weight_log.dequantize(),
                fused_bias, self.conv2d.stride,
                self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            # Use LINEAR quantization stats for activations
            self.update_min_max_stats(simulated_output)

        real_output = self._apply_activation(self.bn2d(self.conv2d(input)))
        return real_output - (real_output - simulated_output).detach()
    
    def forward(self, input: Union[torch.Tensor, QuantizedTensor]) -> Union[torch.Tensor, QuantizedTensor]:
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return self._activation_not_quantized_forward(input)


class MixedQuantLinear(MixedQuantOperator):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config=None, device=None) -> None:
        super().__init__(config, device)
        self.linear = nn.Linear(in_features, out_features, bias, device)

    def _quantize_bias(self, quantized_input: QuantizedTensor, quantized_weight_log, bias: torch.Tensor):
        """Handle bias quantization for mixed precision."""
        return bias
    

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        with torch.no_grad():
            # Use LOG quantization for weights
            quantized_weight_log = self.quantize_weight_log(self.linear.weight)

            if self.linear.bias is not None:
                quantized_bias = self._quantize_bias(input, quantized_weight_log, self.linear.bias)
                simulated_output = F.linear(input.dequantize(), quantized_weight_log.dequantize(),
                                            quantized_bias)
            else:
                simulated_output = F.linear(
                    input.dequantize(), quantized_weight_log.dequantize(), None)

            # Use LINEAR quantization for activations
            self.update_min_max_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)

        real_output = self.linear(input.dequantize())
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output
    


    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            output = self.linear(input)
            self.update_min_max_stats(output)
            return output
        


