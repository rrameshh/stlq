# log_ops.py

from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn


class LogQuantConfig:
    """Configuration class for log quantization parameters."""
    
    def __init__(
        self,
        momentum: float = 0.1,
        threshold: float = 1e-5, 
        eps: float = 1e-8,
        bits: int = 8,
        device = None
    ):
        self.momentum = momentum
        self.threshold = threshold
        self.eps = eps
        self.bits = bits
        self.max_value = (2 ** bits) - 1  # For int8, this is 255
        self.device = device


class LogQuantizedTensor:
    q1: torch.Tensor
    a: torch.Tensor  # row wise max
    s: torch.Tensor  # sign of the original?
    r: Optional[torch.tensor]

    q2: torch.Tensor
    # b: torch.Tensor # row wise max
    s_err: torch.Tensor  # sign of the original?
    # r: Optional[torch.tensor]

    def __init__(self, q1, a, s, q2, s_err, r=None) -> None:
        self.q1 = q1
        self.a = a
        self.s = s
        self.r = r

        self.q2 = q2
        self.s_err = s_err

    def dequantize(self) -> torch.Tensor:
        if self.r is not None:
            return self.r
        else:
            eps = 1e-8
            prim = self.s * self.a.view(*self.a.shape, *([1] * (self.q1.dim() - self.a.dim()))) * torch.maximum((2.0 ** (-self.q1)), torch.tensor(eps))
            if self.q2 is not None and self.s_err is not None:
                second = self.s_err * self.a.view(*self.a.shape, *([1] * (self.q2.dim() - self.a.dim()))) * torch.maximum((2.0 ** (-self.q2)), torch.tensor(eps))
                return prim + second
            return prim

        
    def map(self, func):
        return LogQuantizedTensor(
            func(self.q1),
            self.a, self.s,
            None if self.q2 is None else func(self.q2),
            None if self.s_err is None else func(self.s_err),
            None if self.r is None else func(self.r)
        )

    def reshape(self, *shape):
        return self.map(lambda x: x.reshape(*shape))
    
    def permute(self, *permutation):
        return self.map(lambda x: x.permute(*permutation))
    
    @property
    def shape(self):
        return self.q1.shape


# Utility functions for quantization
class LogQuantUtils:
    """Utility functions for log quantization operations."""
    
    @staticmethod
    def quantize_tensor(
        tensor: torch.Tensor,
        scale: torch.Tensor,
        config: LogQuantConfig
    ) -> LogQuantizedTensor:
        """Quantize a tensor using log quantization."""
        # Get signs
        s = torch.sign(tensor)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # Compute exponents with small epsilon to avoid log(0)
        normalized = torch.abs(tensor) / scale + config.eps
        q1 = -torch.log2(normalized)
        
        # Clamp to int range and round
        q1 = torch.clamp(q1.round(), 0, config.max_value).to(torch.int8)

        # Calculate residual error
        err = (tensor / scale + config.eps) - (2 ** -q1)

        # Conditionally create second-order quantization based on threshold
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > config.threshold):
            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            
            normalized_err = torch.abs(err) / scale + config.eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, config.max_value).to(torch.int8)
        
        return LogQuantizedTensor(q1, scale, s, q2, s_err)
    
    @staticmethod
    def quantize_weight(
        weight: torch.Tensor,
        config: LogQuantConfig,
        per_channel: bool = True
    ) -> LogQuantizedTensor:
        """Quantize weights using per-channel or per-tensor scaling."""
        if per_channel:
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
        
            # Compute exponents
            normalized = torch.abs(weight) / a_view + config.eps
            q1 = -torch.log2(normalized)
            
            # Clamp to int8 range and round
            q1 = torch.clamp(q1.round(), 0, config.max_value).to(torch.int8)

            # Calculate residual error
            err = (weight / a_view + config.eps) - (2 ** -q1)
        else:
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
            q1 = torch.clamp(q1.round(), 0, config.max_value).to(torch.int8)

            # Calculate residual error
            err = (weight / a + config.eps) - (2 ** -q1)

        # Conditionally create second-order quantization based on threshold
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > config.threshold):
            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            
            if per_channel:
                normalized_err = torch.abs(err) / a_view + config.eps
            else:
                normalized_err = torch.abs(err) / a + config.eps
                
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, config.max_value).to(torch.int8)
        
        return LogQuantizedTensor(q1, a if not per_channel else a.squeeze(), s, q2, s_err)


class LogQuantizedOperator(nn.Module):
    def __init__(self, config=None, device=None) -> None:
        super().__init__()
        
        # Create config if not provided
        if config is None:
            self.config = LogQuantConfig(device=device)
        else:
            self.config = config
            
        self.activation_quantization = False
        
        # Register buffers for tracking statistics
        self.register_buffer(
            'running_max_abs', 
            torch.ones(1, device=self.config.device) * self.config.eps
        )
        self.register_buffer(
            'num_batches_tracked',
            torch.tensor(0, dtype=torch.long, device=self.config.device)
        )
                            
    def update_max_abs_stats(self, output: torch.Tensor):
        if not self.training:
            return
            
        # For log quantization, we only need maximum absolute value
        max_abs = output.abs().max()
        max_abs = torch.maximum(max_abs, torch.tensor(self.config.eps, device=output.device))
        
        # Update running stats with momentum
        if self.num_batches_tracked == 0:
            self.running_max_abs.data.copy_(max_abs)
        else:
            self.running_max_abs.data.copy_(
                max_abs * self.config.momentum + 
                self.running_max_abs * (1 - self.config.momentum)
            )
                
        self.num_batches_tracked.data.copy_(self.num_batches_tracked + 1)
        
    def quantize_log_output(self, output: torch.Tensor):
        assert self.num_batches_tracked >= 1
        
        return LogQuantUtils.quantize_tensor(output, self.running_max_abs, self.config)


class LogQuantize(LogQuantizedOperator):
    def __init__(self, config=None, device=None) -> None:
        super().__init__(config, device)

    def forward(self, input: torch.Tensor) -> Union[torch.Tensor, LogQuantizedTensor]:
        if self.activation_quantization:
            self.update_max_abs_stats(input)
            output = self.quantize_log_output(input)
            return output
        else:
            self.update_max_abs_stats(input)
            return input


class LogQuantizedConv2dBatchNorm2dReLU(LogQuantizedOperator):
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
            groups, bias, padding_mode, device=self.config.device
        )
        self.bn2d = nn.BatchNorm2d(out_channels, device=self.config.device)

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
    

    def _quantize_weight(self, weight: torch.Tensor):
        # Use the utility function for weight quantization
        return LogQuantUtils.quantize_weight(weight, self.config, per_channel=True)

    def _quantize_bias(self, quantized_input, quantized_weight, bias):
        return bias
       
    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(
                input.dequantize()
            )
            quantized_fused_weight = self._quantize_weight(fused_weight)
            quantized_fused_bias = self._quantize_bias(
                input, quantized_fused_weight, fused_bias)
            simulated_output = self._apply_activation(F.conv2d(
                input.dequantize(), quantized_fused_weight.dequantize(),
                quantized_fused_bias, self.conv2d.stride,
                self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            self.update_max_abs_stats(simulated_output)
            quantized_simulated_output = self.quantize_log_output(
                simulated_output)

        real_output = self._apply_activation(
            self.bn2d(self.conv2d(input.dequantize())))
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output

    def _activation_not_quantized_forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(
                input
            )
            # quantize weight but not bias since bias quantization relies on a quantized input tensor
            quantized_fused_weight = self._quantize_weight(fused_weight)
            simulated_output = self._apply_activation(F.conv2d(
                input, quantized_fused_weight.dequantize(),
                fused_bias, self.conv2d.stride,
                self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            self.update_max_abs_stats(simulated_output)

        real_output = self._apply_activation(self.bn2d(self.conv2d(input)))
        return real_output - (real_output - simulated_output).detach()

    def forward(self, input: Union[torch.Tensor, LogQuantizedTensor]) -> Union[torch.Tensor, LogQuantizedTensor]:
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return self._activation_not_quantized_forward(input)


class LogQuantizedAdd(LogQuantizedOperator):
    def __init__(self, config=None, device=None) -> None:
        super().__init__(config, device)

    def forward(self, x, y):
        if self.activation_quantization:
            assert isinstance(x, LogQuantizedTensor) and \
                isinstance(y, LogQuantizedTensor)
            
            with torch.no_grad():
                simulated_output = x.dequantize() + y.dequantize()
                self.update_max_abs_stats(simulated_output)
                quantized_simulated_output = self.quantize_log_output(
                    simulated_output
                )
            
            real_output = x.dequantize() + y.dequantize()
            quantized_simulated_output.r = real_output - \
                (real_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output
        
        else:
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            output = x + y
            self.update_max_abs_stats(output)
            return output
        

class LogQuantizedAdaptiveAvgPool2d(LogQuantizedOperator):
    def __init__(self, output_size, config=None, device=None) -> None:
        super().__init__(config, device)
        self.output_size = output_size

    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            input_dequant = input.dequantize()
            pooled = F.adaptive_avg_pool2d(input_dequant, self.output_size)
            
            # Use the config threshold instead of hardcoded value
            scale = torch.maximum(pooled.abs().max(), torch.tensor(self.config.eps, device=pooled.device))
            s = torch.sign(pooled)
            s = torch.where(s == 0, torch.ones_like(s), s)

            eps = self.config.eps
            normalized = torch.abs(pooled) / scale + eps
            q1 = -torch.log2(normalized)
            q1 = torch.clamp(q1.round(), 0, self.config.max_value).to(torch.int8)

            err = (pooled / scale + eps) - (2 ** -q1)
            q2 = None
            s_err = None
            if torch.any(torch.abs(err) > self.config.threshold):
                s_err = torch.sign(err)
                s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
                normalized_err = torch.abs(err) / scale + eps
                q2 = -torch.log2(normalized_err)
                q2 = torch.clamp(q2.round(), 0, self.config.max_value).to(torch.int8)
            
            quantized_simulated_output = LogQuantizedTensor(q1, scale, s, q2, s_err)

        real_output = F.adaptive_avg_pool2d(input.dequantize(), self.output_size)
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
            
        return quantized_simulated_output
    
    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return F.adaptive_avg_pool2d(input, self.output_size)


class LogQuantizedMaxPool2d(LogQuantizedOperator):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, config=None, device=None):
        super().__init__(config, device)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation    


    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            input_dequant = input.dequantize()
            max_pooled = F.max_pool2d(input_dequant, self.kernel_size, self.stride, 
                                      self.padding, self.dilation)

            # Use the config threshold instead of hardcoded value
            scale = torch.maximum(max_pooled.abs().max(), torch.tensor(self.config.eps, device=max_pooled.device))
            s = torch.sign(max_pooled)
            s = torch.where(s == 0, torch.ones_like(s), s)

            eps = self.config.eps
            normalized = torch.abs(max_pooled) / scale + eps
            q1 = -torch.log2(normalized)
            q1 = torch.clamp(q1.round(), 0, self.config.max_value).to(torch.int8)

            err = (max_pooled / scale + eps) - (2 ** -q1)
            q2 = None
            s_err = None
            if torch.any(torch.abs(err) > self.config.threshold):
                s_err = torch.sign(err)
                s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
                normalized_err = torch.abs(err) / scale + eps
                q2 = -torch.log2(normalized_err)
                q2 = torch.clamp(q2.round(), 0, self.config.max_value).to(torch.int8)
            
            quantized_simulated_output = LogQuantizedTensor(q1, scale, s, q2, s_err)
            
        real_output = F.max_pool2d(input.dequantize(), self.kernel_size, self.stride, 
                                  self.padding, self.dilation)
        
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        
        return quantized_simulated_output
    
    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation)
        
        
class LogQuantizedReLU(LogQuantizedOperator):
    def __init__(self, config=None, device=None) -> None:
        super().__init__(config, device)

    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        simulated_output = F.relu(input.dequantize())
        with torch.no_grad():
            self.update_max_abs_stats(simulated_output)
            quantized_simulated_output = self.quantize_log_output(simulated_output)
        r = simulated_output - (simulated_output -
                                quantized_simulated_output.dequantize()).detach()
        quantized_simulated_output.r = r
        return quantized_simulated_output
    
    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            output = F.relu(input)
            self.update_max_abs_stats(output)
            return output
    

class LogQuantizedLinear(LogQuantizedOperator):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config=None, device=None, per_channel=False) -> None:
        super().__init__(config, device)
        self.linear = nn.Linear(in_features, out_features, bias, device=self.config.device)
        self.per_channel = per_channel

    def _quantize_weight(self, weight: torch.Tensor):
        # Use the utility function
        return LogQuantUtils.quantize_weight(weight, self.config, per_channel=self.per_channel)

    def _quantize_bias(self, quantized_input, quantized_weight, bias):
        return bias
    
    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            quantized_weight = self._quantize_weight(self.linear.weight)

            if self.linear.bias is not None:
                quantized_bias = self._quantize_bias(input, quantized_weight, self.linear.bias)
                simulated_output = F.linear(input.dequantize(), quantized_weight.dequantize(),
                                            quantized_bias)
            else:
                simulated_output = F.linear(
                    input.dequantize(), quantized_weight.dequantize(), None)
            self.update_max_abs_stats(simulated_output)
            quantized_simulated_output = self.quantize_log_output(simulated_output)
        real_output = self.linear(input.dequantize())
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output
    
    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            output = self.linear(input)
            self.update_max_abs_stats(output)
            return output


class LogQuantizedFlatten(LogQuantizedOperator):
    def __init__(self, start_dim, end_dim=-1, config=None, device=None) -> None:
        super().__init__(config, device)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            q1 = torch.flatten(input.q1, self.start_dim, self.end_dim)
            r = None if input.r is None else torch.flatten(input.r, self.start_dim, self.end_dim)
            
            q2 = None
            if input.q2 is not None:
                q2 = torch.flatten(input.q2, self.start_dim, self.end_dim)
                
            return LogQuantizedTensor(q1, input.a, input.s, q2, input.s_err, r)
        else:
            assert isinstance(input, torch.Tensor)
            return torch.flatten(input, self.start_dim, self.end_dim)


# Helper functions to enable/disable quantization across the model
def enable_quantization(model):
    """Enable quantization for all quantized layers in a model."""
    for module in model.modules():
        if isinstance(module, LogQuantizedOperator):
            module.activation_quantization = True

def disable_quantization(model):
    """Disable quantization for all quantized layers in a model."""
    for module in model.modules():
        if isinstance(module, LogQuantizedOperator):
            module.activation_quantization = False