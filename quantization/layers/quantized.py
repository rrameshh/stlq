from typing import Optional, Union
import torch
import torch.nn.functional as F
import torch.nn as nn

from ..base import QuantizedOperatorBase
from ..tensors.linear import LinearQuantizedTensor
from ..tensors.new_log import LogQuantizedTensor
from ..strategies.factory import create_strategy
from ..quant_config import QuantizationConfig

QuantizedTensorType = LinearQuantizedTensor

class Quantizer(QuantizedOperatorBase):
    def __init__(self, config: QuantizationConfig):
        super().__init__(config.momentum, config.device)
        self.config = config
        self.strategy = create_strategy(config)
        self.activation_quantization = False
        
        self.register_buffer('running_min', torch.zeros(1, device=config.device))
        self.register_buffer('running_max', torch.zeros(1, device=config.device))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long, device=config.device))

    def update_stats(self, output: torch.Tensor):
        if not self.training:
            return
        min_val = output.min()
        max_val = output.max()
        eps = torch.ones_like(min_val) * 1e-6
        min_val = torch.minimum(-eps, min_val)
        max_val = torch.maximum(eps, max_val)
        
        if self.num_batches_tracked == 0:
            self.running_min.data.copy_(min_val)
            self.running_max.data.copy_(max_val)
        else:
            self.running_min.data.copy_(
                min_val * self.momentum + self.running_min * (1 - self.momentum))
            self.running_max.data.copy_(
                max_val * self.momentum + self.running_max * (1 - self.momentum))
                
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
        return LinearQuantizedTensor(q, s, z)
        


class Quantize(Quantizer):
    def forward(self, input: torch.Tensor) -> QuantizedTensorType:
        if self.activation_quantization:
            self.update_stats(input)
            return self.quantize_output(input)
        else:
            self.update_stats(input)
            return input


class QConv2dBNRelu(Quantizer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', activation=None,
                 config: QuantizationConfig = None):
        super().__init__(config)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                               dilation, groups, bias, padding_mode, config.device)
        self.bn2d = nn.BatchNorm2d(out_channels, device=config.device)
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
            # get convolved output
            conv2d_output = self.conv2d(input) 
            conv2d_output_reshaped = conv2d_output \
                .transpose(0, 1).reshape(self.conv2d.out_channels, -1)
             # switch th 0th and 1st dim and reshape it so that every channel is a list
            mean = conv2d_output_reshaped.mean(1) # get mean across the 1 dim (columns)
            var = conv2d_output_reshaped.var(1) # get means across the 1 dim (rows)
        else:
            mean = self.bn2d.running_mean
            var = self.bn2d.running_var
        return mean, var

    def _get_fused_weight_and_bias(self, input):
         # get mean and variance per channel
        mean, var = self._get_bn2d_mean_and_var(input)
         # sqrt(var + epsilon) to avoid weird division
        sqrt_var = torch.sqrt(var + self.bn2d.eps)
        fused_weight = (self.conv2d.weight * 
                       self.bn2d.weight.reshape(self.conv2d.out_channels, 1, 1, 1) / 
                       sqrt_var.reshape(self.conv2d.out_channels, 1, 1, 1))
        
                        # fused_weight = convolved weight * batchnormalized weight 
                        # the batch normalized weight is reshaped to be in per channel basically
                        # divide by the per channel sqrt(var + eps)
        
        bias = torch.zeros_like(mean) if self.conv2d.bias is None else self.conv2d.bias
        fused_bias = (bias - mean) / sqrt_var * self.bn2d.weight + self.bn2d.bias
        return fused_weight, fused_bias

    def _quantize_weight(self, weight: torch.Tensor):
        return self.strategy.quantize_weight(weight, per_channel=True)

    def _quantize_bias(self, quantized_input: QuantizedTensorType, quantized_weight: QuantizedTensorType, bias: torch.Tensor):
        return self.strategy.quantize_bias(bias, quantized_input, quantized_weight)

    def _activation_quantized_forward(self, input: QuantizedTensorType) -> QuantizedTensorType:
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(input.dequantize())
            quantized_weight = self._quantize_weight(fused_weight)
            quantized_bias = self._quantize_bias(input, quantized_weight, fused_bias)
            
            if hasattr(quantized_bias, 'dequantize'):
                bias_val = quantized_bias.dequantize()
            else:
                bias_val = quantized_bias
                
            simulated_output = self._apply_activation(F.conv2d(
                input.dequantize(), quantized_weight.dequantize(), bias_val,
                self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            
            self.update_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)

        real_output = self._apply_activation(self.bn2d(self.conv2d(input.dequantize())))
        quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output

    def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(input)
            quantized_weight = self._quantize_weight(fused_weight)
            simulated_output = self._apply_activation(F.conv2d(
                input, quantized_weight.dequantize(), fused_bias,
                self.conv2d.stride, self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            self.update_stats(simulated_output)

        real_output = self._apply_activation(self.bn2d(self.conv2d(input)))
        return real_output - (real_output - simulated_output).detach()
    
    def forward(self, input: Union[torch.Tensor, QuantizedTensorType]) -> Union[torch.Tensor, QuantizedTensorType]:
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor))
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return self._activation_not_quantized_forward(input)


class QLinear(Quantizer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: QuantizationConfig = None):
        super().__init__(config)
        self.linear = nn.Linear(in_features, out_features, bias, config.device)

    def _quantize_weight(self, weight: torch.Tensor):
        return self.strategy.quantize_weight(weight, per_channel=False)

    def _quantize_bias(self, quantized_input: QuantizedTensorType, quantized_weight: QuantizedTensorType, bias: torch.Tensor):
        return self.strategy.quantize_bias(bias, quantized_input, quantized_weight)

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor))
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            output = self.linear(input)
            self.update_stats(output)
            return output

    def _activation_quantized_forward(self, input: QuantizedTensorType) -> QuantizedTensorType:
        with torch.no_grad():
            quantized_weight = self._quantize_weight(self.linear.weight)
            if self.linear.bias is not None:
                quantized_bias = self._quantize_bias(input, quantized_weight, self.linear.bias)
                if hasattr(quantized_bias, 'dequantize'):
                    bias_val = quantized_bias.dequantize()
                else:
                    bias_val = quantized_bias
                simulated_output = F.linear(input.dequantize(), quantized_weight.dequantize(), bias_val)
            else:
                simulated_output = F.linear(input.dequantize(), quantized_weight.dequantize(), None)

            self.update_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)

        real_output = self.linear(input.dequantize())
        quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output
    


class QAdd(Quantizer):
    def _rescale_to_common_scale(self, x: QuantizedTensorType, y: QuantizedTensorType):
        if isinstance(x, LinearQuantizedTensor) and isinstance(y, LinearQuantizedTensor):
            s = y.s
            q = ((x.s / y.s) * (x.q.to(torch.int16) - x.z)).round().to(torch.int16)
            z = torch.zeros_like(q)
            return LinearQuantizedTensor(q, s, z)
  
        return x

    def forward(self, x, y):
        if self.activation_quantization:
            assert isinstance(x, (LinearQuantizedTensor, LogQuantizedTensor))
            assert isinstance(y, (LinearQuantizedTensor, LogQuantizedTensor))

            with torch.no_grad():

                rescaled_x = self._rescale_to_common_scale(x, y)
                simulated_output = rescaled_x.dequantize() + y.dequantize()
                
                self.update_stats(simulated_output)
                quantized_simulated_output = self.quantize_output(simulated_output)

            real_output = x.dequantize() + y.dequantize()
            quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output
        else:
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            output = x + y
            self.update_stats(output)
            return output


class QRelu(Quantizer):
    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))
            simulated_output = F.relu(input.dequantize())
            
            with torch.no_grad():
                self.update_stats(simulated_output)
                quantized_simulated_output = self.quantize_output(simulated_output)
            
            quantized_simulated_output.r = simulated_output - (simulated_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output
        else:
            assert isinstance(input, torch.Tensor)
            output = F.relu(input)
            self.update_stats(output)
            return output


class QAdaptiveAvgPool2d(Quantizer):
    def __init__(self, output_size, config: QuantizationConfig):
        super().__init__(config)
        self.output_size = output_size

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))

            # pooling in quantized domain
            with torch.no_grad():
                q = F.adaptive_avg_pool2d(input.q.to(torch.float32), self.output_size).round().to(torch.int8)
                quantized_simulated_output = LinearQuantizedTensor(q, input.s, input.z)
      
            real_output = F.adaptive_avg_pool2d(input.dequantize(), self.output_size)
            quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output
        else:
            assert isinstance(input, torch.Tensor)
            return F.adaptive_avg_pool2d(input, self.output_size)
  


class QMaxPool2d(Quantizer):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, config: QuantizationConfig = None):
        super().__init__(config)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))

            # pooling in quantized domain
            with torch.no_grad():
                q = F.max_pool2d(input.q.to(torch.float32), self.kernel_size, self.stride,
                                self.padding, self.dilation).round().to(torch.int8)
                quantized_simulated_output = LinearQuantizedTensor(q, input.s, input.z)

            
            real_output = F.max_pool2d(input.dequantize(), self.kernel_size, self.stride, self.padding, self.dilation)
            quantized_simulated_output.r = real_output - (real_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output
        else:
            assert isinstance(input, torch.Tensor)
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation)


class QFlatten(Quantizer):
    def __init__(self, start_dim, end_dim=-1, config: QuantizationConfig = None):
        super().__init__(config)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, (LinearQuantizedTensor, LogQuantizedTensor))
            # reshape all components of the quantized tensor
            q = torch.flatten(input.q, self.start_dim, self.end_dim)
            r = torch.flatten(input.r, self.start_dim, self.end_dim)
            return LinearQuantizedTensor(q, input.s, input.z, r)

        else:
            assert isinstance(input, torch.Tensor)
       
            return torch.flatten(input, self.start_dim, self.end_dim)