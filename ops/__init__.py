from .base import QuantizedTensorBase, QuantizedOperatorBase
from .quant_config import QuantConfigBase, LinearQuantConfig, LogQuantConfig
import torch
import torch.nn as nn

# Import specific quantization implementations
try:
    from .linear import (
        QuantizedTensor, QuantizedOperator, Quantize, QuantizedAdd,
        QuantizedConv2dBatchNorm2dReLU, QuantizedReLU, QuantizedAdaptiveAvgPool2d,
        QuantizedMaxPool2d, QuantizedLinear, QuantizedFlatten
    )
except ImportError:
    pass

try:
    from .log import (
        LogQuantizedTensor, LogQuantizedOperator, LogQuantize, LogQuantizedAdd,
        LogQuantizedConv2dBatchNorm2dReLU, LogQuantizedReLU, LogQuantizedAdaptiveAvgPool2d,
        LogQuantizedMaxPool2d, LogQuantizedLinear, LogQuantizedFlatten
    )
except ImportError:
    pass

# Helper functions for all quantization methods
def enable_quantization(model):
    """Enable quantization for all quantized layers in a model."""
    for module in model.modules():
        if isinstance(module, QuantizedOperatorBase):
            module.activation_quantization = True

def disable_quantization(model):
    """Disable quantization for all quantized layers in a model."""
    for module in model.modules():
        if isinstance(module, QuantizedOperatorBase):
            module.activation_quantization = False
