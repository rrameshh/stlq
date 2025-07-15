from .unified_resnet import resnet18 as unified_resnet18, resnet50 as unified_resnet50
from .unified_mobilenet import mobilenetv1 as unified_mobilenetv1, mobilenetv2 as unified_mobilenetv2  # Fixed typo: unfified -> unified
from .unified_vit import (
    industry_vit_tiny as unified_vit_tiny, 
    industry_vit_small as unified_vit_small,
    industry_vit_base as unified_vit_base,
    industry_vit_large as unified_vit_large
)
from .unified_deit import deit_tiny, deit_small, deit_base


def resnet18(quantization_method="linear", **kwargs):
    """
    Create ResNet-18 with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_resnet18(quantization_method=quantization_method, **kwargs)

def resnet50(quantization_method="linear", **kwargs):
    """
    Create ResNet-50 with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"  
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_resnet50(quantization_method=quantization_method, **kwargs)


def mobilenetv1(quantization_method="linear", **kwargs):
    """
    Create MobileNetV1 with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_mobilenetv1(quantization_method=quantization_method, **kwargs)

def mobilenetv2(quantization_method="linear", **kwargs):
    """
    Create MobileNetV2 with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"  
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_mobilenetv2(quantization_method=quantization_method, **kwargs)


# ViT models  
def vit_tiny(quantization_method="linear", **kwargs):
    """
    Create ViT-Tiny with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_vit_tiny(quantization_method=quantization_method, **kwargs)

def vit_small(quantization_method="linear", **kwargs):
    """
    Create ViT-Small with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_vit_small(quantization_method=quantization_method, **kwargs)

def vit_base(quantization_method="linear", **kwargs):
    """
    Create ViT-Base with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_vit_base(quantization_method=quantization_method, **kwargs)

def vit_large(quantization_method="linear", **kwargs):
    """
    Create ViT-Large with any quantization method.
    
    Args:
        quantization_method: "linear" or "log"
        **kwargs: device, threshold, num_classes, etc.
    """
    return unified_vit_large(quantization_method=quantization_method, **kwargs)


# DeiT models
def deit_tiny_model(quantization_method="linear", **kwargs):
    return deit_tiny(quantization_method=quantization_method, **kwargs)

def deit_small_model(quantization_method="linear", **kwargs):
    return deit_small(quantization_method=quantization_method, **kwargs)

def deit_base_model(quantization_method="linear", **kwargs):
    return deit_base(quantization_method=quantization_method, **kwargs)