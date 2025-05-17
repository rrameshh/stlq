from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn


class LogQuantizedTensor:
    q1: torch.Tensor
    a: torch.Tensor # row wise max
    s: torch.Tensor # sign of the original?
    r: Optional[torch.tensor]

    q2: torch.Tensor
    # b: torch.Tensor # row wise max
    s_err: torch.Tensor # sign of the original?
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
    


class LogQuantizedOperator(nn.Module):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__()
        self.activation_quantization = False
        self.momentum = momentum
        
        # only need to track max absolute values
        self.register_buffer('running_max_abs', torch.ones(1, device=device) * 1e-8)
        self.register_buffer('num_batches_tracked',
                            torch.tensor(0, dtype=torch.long, device=device))
                            
    def update_max_abs_stats(self, output: torch.Tensor):
        if not self.training:
            return
            
        # For log quantization, we only need maximum absolute value
        max_abs = output.abs().max()
        max_abs = torch.maximum(max_abs, torch.tensor(1e-8))
        
        # Update running stats with momentum
        if self.num_batches_tracked == 0:
            self.running_max_abs.data.copy_(max_abs)
        else:
            self.running_max_abs.data.copy_(
                max_abs * self.momentum + self.running_max_abs * (1 - self.momentum))
                
        self.num_batches_tracked.data.copy_(self.num_batches_tracked + 1)
        
    def quantize_log_output(self, output: torch.Tensor):
        assert self.num_batches_tracked >= 1

        threshold = 10000000
        
        # Get scale (maximum absolute value)
        a = self.running_max_abs
        
        
        # Get signs
        s = torch.sign(output)
        s = torch.where(s == 0, torch.ones_like(s), s)
        
        # Compute exponents (with small epsilon to avoid log(0))
        eps = 1e-8
        normalized = torch.abs(output) / a + eps
        q1 = -torch.log2(normalized) 
        
        # Clamp to int8 range and round
        q1 = torch.clamp(q1.round(), 0, 127).to(torch.int8)

        err = (output / a + eps) - (2 ** -q1)

        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > threshold):

            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            normalized_err = torch.abs(err) / a + eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, 127).to(torch.int8)
        
        return LogQuantizedTensor(q1, a, s, q2, s_err)



class LogQuantize(LogQuantizedOperator):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

    def forward(self, input: torch.Tensor) -> LogQuantizedTensor:
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
        momentum=0.1,
        device=None
    ):
        super().__init__(momentum, device)
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
    

    #IMPORTANT FUNCTION
    def _quantize_weight(self, weight: torch.Tensor):
    
        # For convolutional weights, shape is [out_channels, in_channels, kernel_height, kernel_width]
        threshold = 10000000
        
        weight_reshaped = weight.reshape(weight.shape[0], -1)  # [out_channels, in_channels*kh*kw]

        # Get sign tensor - this should match the shape of q1 tensor
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)
    
        # Get max absolute value per output channel
        a = weight_reshaped.abs().max(dim=1).values  # [out_channels]
        
        # Reshape a to make broadcasting work later
        a = a.view(a.shape[0], *([1] * (weight.dim() - 1)))  # [out_channels, 1, 1, 1] for 2D conv
        
        eps = 1e-8
        normalized = torch.abs(weight) / a + eps
        q1 = -torch.log2(normalized)
        
        # Clamp to int8 range and round
        q1 = torch.clamp(q1.round(), 0, 127).to(torch.int8)

        err = (weight / a + eps) - (2 ** -q1)
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > threshold):
            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            normalized_err = torch.abs(err) / a + eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, 127).to(torch.int8)
        
        return LogQuantizedTensor(q1, a.squeeze(), s, q2, s_err)

    


    def _quantize_bias(self, quantized_input: LogQuantizedTensor, quantized_weight: LogQuantizedTensor, bias: torch.Tensor):
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
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

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
    def __init__(self, output_size) -> None:
        super().__init__(0.1, None)
        self.output_size = output_size

    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            input_dequant = input.dequantize()
            pooled = F.adaptive_avg_pool2d(input_dequant, self.output_size)
            s = torch.sign(pooled)
            max_abs = pooled.abs().max()
            a = torch.maximum(max_abs, torch.tensor(8))

            eps = 1e-8
            normalized = torch.abs(pooled) / a + eps
            q1 = -torch.log2(normalized)
            q1 = torch.clamp(q1.round(), 0, 127).to(torch.int8)

            threshold = 10000000
            err = (pooled / a + eps) - (2 ** -q1)
            q2 = None
            s_err = None
            if torch.any(torch.abs(err) > threshold):

                s_err = torch.sign(err)
                s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
                normalized_err = torch.abs(err) / a + eps
                q2 = -torch.log2(normalized_err)
                q2 = torch.clamp(q2.round(), 0, 127).to(torch.int8)
            
            quantized_simulated_output = LogQuantizedTensor(q1, a, s, q2, s_err)

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
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(0.1, None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation    


    def _activation_quantized_forward(self, input: LogQuantizedTensor) -> LogQuantizedTensor:
        with torch.no_grad():
            input_dequant = input.dequantize()
            max_pooled = F.max_pool2d(input_dequant, self.kernel_size, self.stride, 
                                      self.padding, self.dilation)

            s = torch.sign(max_pooled)
            max_abs = max_pooled.abs().max()
            a = torch.maximum(max_abs, torch.tensor(1e-8))

            eps = 1e-8
            normalized = torch.abs(max_pooled) / a + eps
            q1 = -torch.log2(normalized)
            q1 = torch.clamp(q1.round(), 0, 127).to(torch.int8)

            threshold = 10000000
            err = (max_pooled / a + eps) - (2 ** -q1)
            q2 = None
            s_err = None
            if torch.any(torch.abs(err) > threshold):

                s_err = torch.sign(err)
                s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
                normalized_err = torch.abs(err) / a + eps
                q2 = -torch.log2(normalized_err)
                q2 = torch.clamp(q2.round(), 0, 127).to(torch.int8)

            
            quantized_simulated_output = LogQuantizedTensor(q1, a, s, q2, s_err)
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
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

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
    
# difference between scale per tensor and scale per weight

class LogQuantizedLinear(LogQuantizedOperator):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)
        self.linear = nn.Linear(in_features, out_features, bias, device)

    def _quantize_weight(self, weight: torch.Tensor):

        a = weight.abs().max()
        # a = weight.abs().max(dim=1)[0]  
        s = torch.sign(weight)
        s = torch.where(s == 0, torch.ones_like(s), s)

        eps = 1e-8
        normalized = torch.abs(weight) / a + eps
        q1 = -torch.log2(normalized)
    
        q1 = torch.clamp(q1.round(), 0, 127).to(torch.int8)

        threshold = 10000000
        err = (weight / a + eps) - (2 ** -q1)
        q2 = None
        s_err = None
        if torch.any(torch.abs(err) > threshold):

            s_err = torch.sign(err)
            s_err = torch.where(s_err == 0, torch.ones_like(s_err), s_err)
            normalized_err = torch.abs(err) / a + eps
            q2 = -torch.log2(normalized_err)
            q2 = torch.clamp(q2.round(), 0, 127).to(torch.int8)
    
        return LogQuantizedTensor(q1, a, s, q2, s_err)

        
    def _quantize_bias(self, quantized_input: LogQuantizedTensor, quantized_weight: LogQuantizedTensor, bias: torch.Tensor):
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
    def __init__(self, start_dim, end_dim=-1) -> None:
        super().__init__(0.1, None)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, LogQuantizedTensor)
            q1 = torch.flatten(input.q1, self.start_dim, self.end_dim)
            r = torch.flatten(input.r, self.start_dim, self.end_dim)
            if input.q2 is not None:
                q2 =  torch.flatten(input.q2, self.start_dim, self.end_dim)
            else:
                q2 = None
            return LogQuantizedTensor(q1, input.a, input.s, q2, input.s_err, r)
        else:
            assert isinstance(input, torch.Tensor)
            return torch.flatten(input, self.start_dim, self.end_dim)



