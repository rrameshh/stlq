# networks/resnet_base.py
from typing import Optional, Type, List, Union, Any
import torch
import torch.nn as nn
from ops.base import QuantizedTensorBase, QuantizedOperatorBase


class ResNetBlockBase(nn.Module):
    """Base class for ResNet blocks (BasicBlock or Bottleneck)."""
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
        **kwargs
    ) -> None:
        super().__init__()
        self.stride = stride
        self.downsample = downsample


class ResNetBase(nn.Module):
    """Base class for ResNet models with any quantization strategy."""
    
    def __init__(
        self,
        block: Type[ResNetBlockBase],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        
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
        
        # These will be implemented by subclasses
        self.quantize = None
        self.conv1 = None
        self.maxpool = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.avgpool = None
        self.flatten = None
        self.fc = None
        
    def _make_layer(
        self,
        block: Type[ResNetBlockBase],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        **kwargs
    ) -> nn.Sequential:
        """Implement in subclass"""
        raise NotImplementedError
    
    def _dequantize(self, x):
        """Dequantize if x is a quantized tensor, otherwise return x."""
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, QuantizedTensorBase):
            return x.dequantize()
    
    def _forward_impl(self, x):
        """Forward pass implementation."""
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