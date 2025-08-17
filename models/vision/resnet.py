from typing import Optional, Type, List, Union, Any
import torch
import torch.nn as nn

from quantization.layers.quantized import (
    Quantize,
    QConv2dBNRelu, 
    QLinear,
    QAdd,
    QRelu,
    QMaxPool2d,
    QAdaptiveAvgPool2d,
    QFlatten
)
from quantization.layers.unfused_conv import QConvBNUnfused
from quantization.quant_config import QuantizationConfig


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, 
           dilation: int = 1, activation=None, config: QuantizationConfig = None):
    """3x3 convolution with padding"""
    return QConv2dBNRelu(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
        activation=activation, config=config
    )
    # return QConvBNUnfused(
    #     in_planes, out_planes, kernel_size=3, stride=stride,
    #     padding=dilation, groups=groups, bias=False, dilation=dilation,
    #     activation=activation, config=config
    # )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, activation=None,
           config: QuantizationConfig = None):
    """1x1 convolution"""
    return QConv2dBNRelu(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
        activation=activation, config=config
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
        config: QuantizationConfig = None
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride, activation="relu", config=config)
        self.conv2 = conv3x3(planes, planes, config=config)
        self.relu = QRelu(config=config)
        self.add = QAdd(config=config)
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
        config: QuantizationConfig = None
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(inplanes, width, activation="relu", config=config)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, activation="relu", config=config)
        self.conv3 = conv1x1(width, planes * self.expansion, config=config)
        self.relu = QRelu(config=config)
        self.add = QAdd(config=config)
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
        config: QuantizationConfig,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.device = config.device
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.quantize = Quantize(config=config)
        self.conv1 = QConv2dBNRelu(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, 
            bias=False, activation="relu", config=config
        )
        self.maxpool = QMaxPool2d(kernel_size=3, stride=2, padding=1, config=config)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = QAdaptiveAvgPool2d((1, 1), config=config)
        self.flatten = QFlatten(1, config=config)
        self.fc = QLinear(512 * block.expansion, num_classes, config=config)

      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and hasattr(m.conv3, 'bn2d'):
                    nn.init.constant_(m.conv3.bn2d.weight, 0)
                elif isinstance(m, BasicBlock) and hasattr(m.conv2, 'bn2d'):
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
                conv1x1(self.inplanes, planes * block.expansion, stride, config=self.config),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, config=self.config
        ))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation, config=self.config
            ))

        return nn.Sequential(*layers)

    def _dequantize(self, x):
        """Handle dequantization for any tensor type"""
        if isinstance(x, torch.Tensor):
            return x
        elif hasattr(x, 'dequantize'):
            return x.dequantize()
        else:
            return x

    def forward(self, x):
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


def resnet18(main_config, **kwargs):
    config = QuantizationConfig(
        method=main_config.quantization.method,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    return ResNet(BasicBlock, [2, 2, 2, 2], config=config, 
                  num_classes=main_config.model.num_classes, **kwargs)

def resnet50(main_config, **kwargs):
    config = QuantizationConfig(
        method=main_config.quantization.method,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    return ResNet(Bottleneck, [3, 4, 6, 3], config=config,
                  num_classes=main_config.model.num_classes, **kwargs)