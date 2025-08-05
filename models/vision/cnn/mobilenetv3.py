# models/vision/cnn/mobilenetv3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
import math

from quantization.layers.all import (
    UnifiedQuantize,
    UnifiedQuantizedConv2dBatchNorm2dReLU, 
    UnifiedQuantizedLinear,
    UnifiedQuantizedAdd,
    UnifiedQuantizedReLU,
    UnifiedQuantizedAdaptiveAvgPool2d,
    UnifiedQuantizedFlatten
)
from quantization.quant_config import QuantizationConfig
from quantization.layers.unfused_conv import UnifiedQuantizedConvBatchNormUnfused
from quantization.tensors.linear import LinearQuantizedTensor


def _make_divisible(v, divisor=8, min_value=None):
    """Make channels divisible by divisor"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    
    def __init__(self, channels, reduction=4, config: QuantizationConfig = None):
        super().__init__()
        reduced_channels = _make_divisible(channels // reduction)
        
        # QUANTIZED: Heavy compute operations
        self.avgpool = UnifiedQuantizedAdaptiveAvgPool2d((1, 1), config=config)
        self.fc1 = UnifiedQuantizedLinear(channels, reduced_channels, config=config)
        self.fc2 = UnifiedQuantizedLinear(reduced_channels, channels, config=config)
        
        # FP32: Lightweight, numerically sensitive operations
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        
        # Input quantizer for explicit transition management
        self.input_quantizer = UnifiedQuantize(config=config)
        
    def forward(self, x):
        """
        EXPLICIT FLOW: Clear transitions with explicit case handling
        
        Flow:
        1. Quantized/FP32 input → Ensure quantized → INT8
        2. INT8 avgpool + reshape → INT8
        3. INT8 fc1 → Dequantize → FP32
        4. FP32 ReLU → Keep FP32  
        5. FP32 → Quantize → INT8 fc2 → Dequantize → FP32
        6. FP32 HardSigmoid → Keep FP32
        7. Apply attention in FP32
        """
        original_x = x
        b, c, h, w = x.shape
        
        # ============ EXPLICIT TRANSITION 1: ENSURE QUANTIZED INPUT ============
        # Handle both quantized and FP32 inputs explicitly
        if isinstance(x, LinearQuantizedTensor):
            x_quantized = x
        else:
            # Input is FP32 (from LayerNorm or previous layer), quantize it
            x_quantized = self.input_quantizer(x)
        
        # ============ QUANTIZED COMPUTE: POOLING + FC1 ============
        se_quantized = self.avgpool(x_quantized)                    # INT8 pooling
        se_quantized = se_quantized.view(b, c)                      # INT8 reshape
        se_quantized = self.fc1(se_quantized)                       # INT8 linear
        
        # ============ EXPLICIT TRANSITION 2: INT8 → FP32 ============
        # Dequantize for ReLU (activation functions need FP32 precision)
        if isinstance(se_quantized, LinearQuantizedTensor):
            se_fp32 = se_quantized.dequantize()
        else:
            se_fp32 = se_quantized
            
        # ============ FP32 COMPUTE: RELU ACTIVATION ============
        se_fp32 = F.relu(se_fp32)                                   # FP32 activation
        
        # ============ EXPLICIT TRANSITION 3: FP32 → INT8 → FP32 ============  
        # Quantize for heavy fc2 computation
        se_quantized = self.input_quantizer(se_fp32)                # FP32 → INT8
        se_quantized = self.fc2(se_quantized)                       # INT8 linear
        
        # Dequantize for sigmoid (numerically sensitive)
        if isinstance(se_quantized, LinearQuantizedTensor):
            se_fp32 = se_quantized.dequantize()
        else:
            se_fp32 = se_quantized
            
        # ============ FP32 COMPUTE: SIGMOID + ATTENTION ============
        se_fp32 = self.hardsigmoid(se_fp32)                        # FP32 sigmoid
        se_fp32 = se_fp32.view(b, c, 1, 1)                         # FP32 reshape
        
        # Extract FP32 values from original input for attention
        if isinstance(original_x, LinearQuantizedTensor):
            x_fp32 = original_x.dequantize()
        else:
            x_fp32 = original_x
            
        # Apply attention in FP32
        result_fp32 = x_fp32 * se_fp32                              # FP32 attention
        
        # ============ RETURN TYPE MATCHING ============
        # Return same type as input for consistent flow
        if isinstance(original_x, LinearQuantizedTensor):
            # Convert result back to quantized format
            result_quantized = self.input_quantizer(result_fp32)
            return result_quantized
        else:
            # Input was FP32, return FP32
            return result_fp32

class HardSwish(nn.Module):
    """Hard Swish activation function"""
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class MobileNetV3Block(nn.Module):
    """MobileNetV3 Inverted Residual Block with SE and various activations"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=1, 
                 use_se=False, activation='relu', config: QuantizationConfig = None):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = _make_divisible(in_channels * expand_ratio)
        
        layers = []
        
        # Expansion phase (1x1 pointwise) - only if expand_ratio != 1
        if expand_ratio != 1:
            layers.append(UnifiedQuantizedConvBatchNormUnfused(
                in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
                bias=False, activation=activation, config=config
            ))
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.append(UnifiedQuantizedConvBatchNormUnfused(
            hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=hidden_dim, bias=False, activation=activation, config=config
        ))
        
        # Squeeze-and-Excitation
        if use_se:
            layers.append(SEModule(hidden_dim, config=config))
        
        # Projection phase (1x1 pointwise, no activation)
        layers.append(UnifiedQuantizedConvBatchNormUnfused(
            hidden_dim, out_channels, kernel_size=1, stride=1, padding=0,
            bias=False, activation=None, config=config
        ))
        
        self.conv = nn.Sequential(*layers)
        
        if self.use_residual:
            self.add = UnifiedQuantizedAdd(config=config)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = self.add(out, x)
        return out


class UnifiedMobileNetV3(nn.Module):
    """MobileNetV3 with unified quantization support"""
    
    def __init__(self, config: QuantizationConfig, variant="large", num_classes: int = 1000, 
                 width_multiplier: float = 1.0):
        super().__init__()
        
        self.config = config
        self.device = config.device
        self.variant = variant
        
        # Get configuration for the variant
        if variant == "large":
            block_configs = self._get_large_config()
            last_channel = _make_divisible(960 * width_multiplier)
        elif variant == "small":
            block_configs = self._get_small_config()
            last_channel = _make_divisible(576 * width_multiplier)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # First conv layer
        input_channel = _make_divisible(16 * width_multiplier)
        self.quantize = UnifiedQuantize(config=config)
        
        self.features = nn.ModuleList([
            UnifiedQuantizedConvBatchNormUnfused(
                3, input_channel, kernel_size=3, stride=2, padding=1,
                bias=False, activation="hardswish" if variant == "large" else "relu", 
                config=config
            )
        ])
        
        # Build inverted residual blocks
        for exp_ratio, channels, kernel_size, stride, use_se, activation in block_configs:
            output_channel = _make_divisible(channels * width_multiplier)
            self.features.append(MobileNetV3Block(
                input_channel, output_channel, kernel_size, stride, exp_ratio,
                use_se, activation, config
            ))
            input_channel = output_channel
        
        # Last conv layer
        self.features.append(UnifiedQuantizedConvBatchNormUnfused(
            input_channel, last_channel, kernel_size=1, stride=1, padding=0,
            bias=False, activation="hardswish" if variant == "large" else "relu", 
            config=config
        ))
        
        # Classifier
        self.avgpool = UnifiedQuantizedAdaptiveAvgPool2d((1, 1), config=config)
        self.flatten = UnifiedQuantizedFlatten(1, config=config)
        
        # Classifier head with optional intermediate layer for large variant
        if variant == "large":
            classifier_input = last_channel
            self.pre_classifier = UnifiedQuantizedLinear(
                classifier_input, 1280, bias=True, config=config
            )
            self.classifier = UnifiedQuantizedLinear(1280, num_classes, config=config)
        else:
            classifier_input = last_channel
            self.classifier = UnifiedQuantizedLinear(classifier_input, num_classes, config=config)
        
        self._initialize_weights()
    
    def _get_large_config(self):
        """Configuration for MobileNetV3-Large"""
        # Format: [expand_ratio, channels, kernel_size, stride, use_se, activation]
        return [
            [1, 16, 3, 1, False, 'relu'],
            [4, 24, 3, 2, False, 'relu'],
            [3, 24, 3, 1, False, 'relu'],
            [3, 40, 5, 2, True, 'relu'],
            [3, 40, 5, 1, True, 'relu'],
            [3, 40, 5, 1, True, 'relu'],
            [6, 80, 3, 2, False, 'hardswish'],
            [2.5, 80, 3, 1, False, 'hardswish'],
            [2.3, 80, 3, 1, False, 'hardswish'],
            [2.3, 80, 3, 1, False, 'hardswish'],
            [6, 112, 3, 1, True, 'hardswish'],
            [6, 112, 3, 1, True, 'hardswish'],
            [6, 160, 5, 2, True, 'hardswish'],
            [6, 160, 5, 1, True, 'hardswish'],
            [6, 160, 5, 1, True, 'hardswish'],
        ]
    
    def _get_small_config(self):
        """Configuration for MobileNetV3-Small"""
        # Format: [expand_ratio, channels, kernel_size, stride, use_se, activation]
        return [
            [1, 16, 3, 2, True, 'relu'],
            [4.5, 24, 3, 2, False, 'relu'],
            [3.67, 24, 3, 1, False, 'relu'],
            [4, 40, 5, 2, True, 'hardswish'],
            [6, 40, 5, 1, True, 'hardswish'],
            [6, 40, 5, 1, True, 'hardswish'],
            [3, 48, 5, 1, True, 'hardswish'],
            [3, 48, 5, 1, True, 'hardswish'],
            [6, 96, 5, 2, True, 'hardswish'],
            [6, 96, 5, 1, True, 'hardswish'],
            [6, 96, 5, 1, True, 'hardswish'],
        ]

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
                if m.bias is not None:
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
        
        # Classifier path depends on variant
        if self.variant == "large":
            x = self.pre_classifier(x)
            # Add hardswish activation here if quantizing pre_classifier
            x = self.classifier(x)
        else:
            x = self.classifier(x)
        
        x = self._dequantize(x)
        return x


# Factory functions
def create_mobilenetv3(
    variant: str,
    quantization_method: str = "linear",
    **kwargs
) -> UnifiedMobileNetV3:
    """
    Factory function to create a unified MobileNetV3 with specified quantization method.
    
    Args:
        variant: 'large' or 'small'
        quantization_method: 'linear' or 'log'
        **kwargs: Additional arguments (num_classes, device, etc.)
    """
    # Extract config-specific parameters
    device = kwargs.pop('device', None)
    threshold = kwargs.pop('threshold', 1e-5)
    momentum = kwargs.pop('momentum', 0.1)
    bits = kwargs.pop('bits', 8)
    
    # Create config based on method
    config = QuantizationConfig(
        method=quantization_method,
        momentum=momentum,
        device=device,
        threshold=threshold,
        bits=bits
    )
    
    return UnifiedMobileNetV3(config=config, variant=variant, **kwargs)


def mobilenetv3_large(quantization_method="linear", **kwargs):
    """MobileNetV3-Large with unified quantization"""
    return create_mobilenetv3('large', quantization_method, **kwargs)


def mobilenetv3_small(quantization_method="linear", **kwargs):
    """MobileNetV3-Small with unified quantization"""
    return create_mobilenetv3('small', quantization_method, **kwargs)
