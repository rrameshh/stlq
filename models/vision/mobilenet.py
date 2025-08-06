import torch
import torch.nn as nn
from typing import Optional, List, Union

from quantization.layers.quantized import (
    Quantize,
    QConv2dBNRelu, 
    QLinear,
    QAdd,
    QAdaptiveAvgPool2d,
    QFlatten
)
from quantization.quant_config import QuantizationConfig
from quantization.layers.unfused_conv import QConvBNUnfused


class MobileNetV1Block(nn.Module):
    """MobileNetV1 block with depthwise separable convolution"""
    
    def __init__(self, in_channels, out_channels, stride=1, config: QuantizationConfig = None):
        super().__init__()
        
        self.depthwise_separable = QConvBNUnfused(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
            activation="relu", config=config
        )

    def forward(self, x):
        return self.depthwise_separable(x)


class MobileNetV2Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=1, 
                 config: QuantizationConfig = None):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        
        # Expansion phase (1x1 pointwise)
        if expand_ratio != 1:
            layers.append(QConvBNUnfused(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
                bias=False, activation="relu", config=config
            ))
        
        # Depthwise convolution
        layers.append(QConvBNUnfused(
            hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
            groups=hidden_dim, bias=False, activation="relu", config=config
        ))
        
        # Projection phase (1x1 pointwise, no activation)
        layers.append(QConvBNUnfused(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0,
            bias=False, activation=None, config=config
        ))
        
        self.conv = nn.Sequential(*layers)
        
        if self.use_residual:
            self.add = QAdd(config=config)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = self.add(out, x)
        return out


class MobileNetV1(nn.Module):
    
    def __init__(self, config: QuantizationConfig, num_classes: int = 1000, 
                 width_multiplier: float = 1.0):
        super().__init__()
        
        self.config = config
        self.device = config.device
        
        # Calculate channel dimensions with width multiplier
        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = _make_divisible(32 * width_multiplier)
        
        self.quantize = Quantize(config=config)
        
        # First conv layer
        self.features = nn.ModuleList([
            QConv2dBNRelu(
                3, input_channel, kernel_size=3, stride=2, padding=1,
                bias=False, activation="relu", config=config
            )
        ])
        
        # MobileNet configuration: [channels, stride]
        mobilenet_config = [
            [64, 1], [128, 2], [128, 1], [256, 2], [256, 1],
            [512, 2], [512, 1], [512, 1], [512, 1], [512, 1], [512, 1],
            [1024, 2], [1024, 1]
        ]
        
        # Build depthwise separable layers
        for channels, stride in mobilenet_config:
            output_channel = _make_divisible(channels * width_multiplier)
            self.features.append(MobileNetV1Block(
                input_channel, output_channel, stride, config
            ))
            input_channel = output_channel
        
        self.avgpool = QAdaptiveAvgPool2d((1, 1), config=config)
        self.flatten = QFlatten(1, config=config)
        self.classifier = QLinear(input_channel, num_classes, config=config)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
        
        for layer in self.features:
            x = layer(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = self._dequantize(x)
        
        return x


class MobileNetV2(nn.Module):
    
    def __init__(self, config: QuantizationConfig, num_classes: int = 1000, 
                 width_multiplier: float = 1.0):
        super().__init__()
        
        self.config = config
        self.device = config.device
        
        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = _make_divisible(32 * width_multiplier)
        last_channel = _make_divisible(1280 * max(1.0, width_multiplier))
        
        self.quantize = Quantize(config=config)
        
        # First conv layer
        self.features = nn.ModuleList([
            QConvBNUnfused(
                3, input_channel, kernel_size=3, stride=2, padding=1,
                bias=False, activation="relu", config=config
            )
        ])
        
        # MobileNetV2 configuration: [expand_ratio, channels, num_blocks, stride]
        mobilenetv2_config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Build inverted residual blocks
        for expand_ratio, channels, num_blocks, stride in mobilenetv2_config:
            output_channel = _make_divisible(channels * width_multiplier)
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                self.features.append(MobileNetV2Block(
                    input_channel, output_channel, block_stride, expand_ratio, config
                ))
                input_channel = output_channel
        
        # Last conv layer
        self.features.append(QConvBNUnfused(
            input_channel, last_channel, kernel_size=1, stride=1, padding=0,
            bias=False, activation="relu", config=config
        ))
        
        self.avgpool = QAdaptiveAvgPool2d((1, 1), config=config)
        self.flatten = QFlatten(1, config=config)
        self.classifier = QLinear(last_channel, num_classes, config=config)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
        
        for layer in self.features:
            x = layer(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        x = self._dequantize(x)
        
        return x
    
def mobilenetv1(main_config, **kwargs):

    config = QuantizationConfig(
        method=main_config.quantization.method,
        momentum=main_config.quantization.momentum,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    return MobileNetV1(config=config, num_classes=main_config.model.num_classes, **kwargs)

def mobilenetv2(main_config, **kwargs):

    config = QuantizationConfig(
        method=main_config.quantization.method,
        momentum=main_config.quantization.momentum,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    return MobileNetV2(config=config, num_classes=main_config.model.num_classes, **kwargs)