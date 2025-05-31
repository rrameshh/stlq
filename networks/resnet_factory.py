
from .unified_resnet import resnet18 as unified_resnet18, resnet50 as unified_resnet50

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