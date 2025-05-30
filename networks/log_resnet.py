# Modified from torchvision.models.resnet
# networks/log_resnet.py - with shared config support

from typing import Optional, Callable, Type, List, Union, Any

import torch
import torch.nn as nn

from ops.log import (
    LogQuantConfig, LogQuantize, LogQuantizedAdaptiveAvgPool2d, 
    LogQuantizedConv2dBatchNorm2dReLU, LogQuantizedFlatten, 
    LogQuantizedLinear, LogQuantizedMaxPool2d, LogQuantizedReLU, 
    LogQuantizedAdd, LogQuantizedTensor
)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, 
            dilation: int = 1, activation=None, config=None, device=None):
    """3x3 convolution with padding"""
    return LogQuantizedConv2dBatchNorm2dReLU(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        activation=activation,
        config=config,
        device=device
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, activation=None, 
            config=None, device=None):
    """1x1 convolution"""
    return LogQuantizedConv2dBatchNorm2dReLU(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
        activation=activation, config=config, device=device
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        config: Optional[LogQuantConfig] = None,
        device = None
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride, 
                            activation="relu", config=config, device=device)
        self.conv2 = conv3x3(planes, planes, 
                            config=config, device=device)
        self.relu = LogQuantizedReLU(config=config, device=device)
        self.add = LogQuantizedAdd(config=config, device=device)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        config: Optional[LogQuantConfig] = None,
        device = None
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, activation="relu", 
                           config=config, device=device)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, 
                           activation="relu", config=config, device=device)
        self.conv3 = conv1x1(width, planes * self.expansion, 
                           config=config, device=device)
        self.relu = LogQuantizedReLU(config=config, device=device)
        self.add = LogQuantizedAdd(config=config, device=device)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        threshold: float = 1e-5,  # Use recommended threshold instead of 10000000
        device = None
    ) -> None:
        super().__init__()

        # Create a shared configuration for all layers
        self.config = LogQuantConfig(
            momentum=0.1,
            threshold=threshold,
            eps=1e-8,
            bits=8,
            device=device
        )
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.quantize = LogQuantize(config=self.config, device=device)

        self.conv1 = LogQuantizedConv2dBatchNorm2dReLU(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False,
            activation="relu", config=self.config, device=device
        )
        self.maxpool = LogQuantizedMaxPool2d(
            kernel_size=3, stride=2, padding=1, 
            config=self.config, device=device
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = LogQuantizedAdaptiveAvgPool2d(
            (1, 1), config=self.config, device=device
        )
        self.flatten = LogQuantizedFlatten(1, config=self.config, device=device)
        self.fc = LogQuantizedLinear(
            512 * block.expansion, 
            num_classes, 
            config=self.config, 
            device=device
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.conv3.bn2d.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.conv2.bn2d.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes, 
                    planes * block.expansion, 
                    stride, 
                    config=self.config, 
                    device=self.config.device
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, 
                self.groups, self.base_width, previous_dilation,
                config=self.config, device=self.config.device
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    config=self.config,
                    device=self.config.device
                )
            )

        return nn.Sequential(*layers)

    def _dequantize(self, x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, LogQuantizedTensor):
            return x.dequantize()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.quantize(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self._dequantize(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)