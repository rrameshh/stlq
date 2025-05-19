from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch.nn as nn


class QuantizedTensor:
    q: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor
    r: Optional[torch.Tensor]

    def __init__(self, q, s, z, r=None) -> None:
        self.q = q
        self.s = s
        self.z = z
        self.r = r

    def dequantize(self) -> torch.Tensor:
        if self.r is not None:
            return self.r
        else:
            # prevent (q - z) from overflow
            return self.s * (self.q.to(torch.int32) - self.z)

    def map(self, func):
        return QuantizedTensor(
            func(self.q),
            self.s, self.z,
            None if self.r is None else func(self.r)
        )

    def reshape(self, *shape):
        return self.map(lambda x: x.reshape(*shape))

    def permute(self, *permutation):
        return self.map(lambda x: x.permute(*permutation))

    @property
    def shape(self):
        return self.q.shape


class QuantizedOperator(nn.Module):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__()
        self.activation_quantization = False
        self.momentum = momentum

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


class Quantize(QuantizedOperator):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

    def forward(self, input: torch.Tensor) -> QuantizedTensor:
        if self.activation_quantization:
            self.update_min_max_stats(input)
            output = self.quantize_output(input)
            return output
        else:
            self.update_min_max_stats(input)
            return input


class QuantizedConv2dBatchNorm2dReLU(QuantizedOperator):
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

    def _quantize_weight(self, weight: torch.Tensor):
        # Quantize weight to -127 ~ 127. Note that -128 is excluded.
        weight_reshaped = weight.reshape(weight.shape[0], -1)
        a = weight_reshaped.min(dim=1).values
        b = weight_reshaped.max(dim=1).values
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))
        z = z.reshape(z.shape[0], 1, 1, 1)
        s = s.reshape(s.shape[0], 1, 1, 1)

        q = torch.maximum(torch.minimum(
            weight / s + z, torch.tensor(127)), torch.tensor(-127)).round().to(torch.int8)
        return QuantizedTensor(q, s, z)

    def _quantize_bias(self, quantized_input: QuantizedTensor, quantized_weight: QuantizedTensor, bias: torch.Tensor):
        s = (quantized_weight.s * quantized_input.s).reshape(-1)
        z = torch.zeros_like(s).to(torch.int32)
        q = (bias / s).round().to(torch.int32)
        return QuantizedTensor(q, s, z)

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        with torch.no_grad():
            fused_weight, fused_bias = self._get_fused_weight_and_bias(
                input.dequantize()
            )
            quantized_fused_weight = self._quantize_weight(fused_weight)
            quantized_fused_bias = self._quantize_bias(
                input, quantized_fused_weight, fused_bias)
            simulated_output = self._apply_activation(F.conv2d(
                input.dequantize(), quantized_fused_weight.dequantize(),
                quantized_fused_bias.dequantize(), self.conv2d.stride,
                self.conv2d.padding, self.conv2d.dilation, self.conv2d.groups
            ))
            self.update_min_max_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(
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


class QuantizedAdd(QuantizedOperator):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

    def _rescale_x(self, x: QuantizedTensor, y: QuantizedTensor):
        s = y.s
        q = ((x.s / y.s) * (x.q.to(torch.int16) - x.z)).round().to(torch.int16)
        z = torch.zeros_like(q)
        return QuantizedTensor(q, s, z)

    def forward(self, x, y):
        if self.activation_quantization:
            assert isinstance(x, QuantizedTensor) and \
                isinstance(y, QuantizedTensor)

            with torch.no_grad():
                simulated_output = self._rescale_x(x, y).dequantize() \
                    + y.dequantize()
                self.update_min_max_stats(simulated_output)
                quantized_simulated_output = self.quantize_output(
                    simulated_output)

            real_output = x.dequantize() + y.dequantize()
            quantized_simulated_output.r = real_output - \
                (real_output - quantized_simulated_output.dequantize()).detach()
            return quantized_simulated_output

        else:
            assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            output = x + y
            self.update_min_max_stats(output)
            return output


class QuantizedAdaptiveAvgPool2d(QuantizedOperator):
    def __init__(self, output_size) -> None:
        super().__init__(0.1, None)
        self.output_size = output_size

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        with torch.no_grad():
            q = F.adaptive_avg_pool2d(input.q.to(torch.float32), self.output_size) \
                .round().to(torch.int8)
            quantized_simulated_output = QuantizedTensor(q, input.s, input.z)
        real_output = F.adaptive_avg_pool2d(
            input.dequantize(), self.output_size)
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return F.adaptive_avg_pool2d(input, self.output_size)


class QuantizedMaxPool2d(QuantizedOperator):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__(0.1, None)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        with torch.no_grad():
            q = F.max_pool2d(input.q.to(torch.float32), self.kernel_size, self.stride,
                             self.padding, self.dilation).round().to(torch.int8)
            quantized_simulated_output = QuantizedTensor(q, input.s, input.z)
        real_output = F.max_pool2d(
            input.dequantize(), self.kernel_size, self.stride, self.padding, self.dilation)
        quantized_simulated_output.r = real_output - \
            (real_output - quantized_simulated_output.dequantize()).detach()
        return quantized_simulated_output

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation)


class QuantizedReLU(QuantizedOperator):
    def __init__(self, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        simulated_output = F.relu(input.dequantize())
        with torch.no_grad():
            self.update_min_max_stats(simulated_output)
            quantized_simulated_output = self.quantize_output(simulated_output)
        r = simulated_output - (simulated_output -
                                quantized_simulated_output.dequantize()).detach()
        quantized_simulated_output.r = r
        return quantized_simulated_output

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            return self._activation_quantized_forward(input)
        else:
            assert isinstance(input, torch.Tensor)
            output = F.relu(input)
            self.update_min_max_stats(output)
            return output


class QuantizedLinear(QuantizedOperator):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, momentum=0.1, device=None) -> None:
        super().__init__(momentum, device)
        self.linear = nn.Linear(in_features, out_features, bias, device)

    def _quantize_weight(self, weight: torch.Tensor):
        # Quantize weight to -127 ~ 127. Note that -128 is excluded.
        a = weight.min()
        b = weight.max()
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))

        q = torch.maximum(torch.minimum(
            weight / s + z, torch.tensor(127)), torch.tensor(-127)).round().to(torch.int8)
        return QuantizedTensor(q, s, z)

    def _quantize_bias(self, quantized_input: QuantizedTensor, quantized_weight: QuantizedTensor, bias: torch.Tensor):
        s = quantized_weight.s * quantized_input.s
        z = torch.zeros_like(s).to(torch.int32)
        q = (bias / s).round().to(torch.int32)
        return QuantizedTensor(q, s, z)

    def _activation_quantized_forward(self, input: QuantizedTensor) -> QuantizedTensor:
        with torch.no_grad():
            quantized_weight = self._quantize_weight(self.linear.weight)

            if self.linear.bias is not None:
                quantized_bias = self._quantize_bias(
                    input, quantized_weight, self.linear.bias)
                simulated_output = F.linear(input.dequantize(), quantized_weight.dequantize(),
                                            quantized_bias.dequantize())
            else:
                simulated_output = F.linear(
                    input.dequantize(), quantized_weight.dequantize(), None)

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


class QuantizedFlatten(QuantizedOperator):
    def __init__(self, start_dim, end_dim=-1) -> None:
        super().__init__(0.1, None)
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        if self.activation_quantization:
            assert isinstance(input, QuantizedTensor)
            q = torch.flatten(input.q, self.start_dim, self.end_dim)
            r = torch.flatten(input.r, self.start_dim, self.end_dim)
            return QuantizedTensor(q, input.s, input.z, r)
        else:
            assert isinstance(input, torch.Tensor)
            return torch.flatten(input, self.start_dim, self.end_dim)
        
