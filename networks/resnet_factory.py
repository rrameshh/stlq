# networks/resnet_factory.py
from typing import Type, Any
from .resnet_base import ResNetBase, ResNetBlockBase

def create_resnet(
    block_type: Type[ResNetBlockBase],
    layers: list,
    quantization_method: str = "linear",
    **kwargs
) -> ResNetBase:
    """
    Factory function to create a ResNet model with specified quantization method.
    
    Args:
        block_type: The block class (BasicBlock or Bottleneck)
        layers: Number of blocks in each layer
        quantization_method: 'linear' or 'log'
        **kwargs: Additional arguments for the specific ResNet implementation
        
    Returns:
        A ResNet model with the specified quantization
    """
    # Create a clean kwargs dict for each method
    filtered_kwargs = kwargs.copy()
    
    if quantization_method.lower() == "linear":
        from .resnet import _resnet
        # Remove parameters that linear quantization doesn't support
        params_to_remove = ['device', 'threshold']
        for param in params_to_remove:
            if param in filtered_kwargs:
                filtered_kwargs.pop(param)
    elif quantization_method.lower() == "log":
        from .log_resnet import _resnet
        # Log quantization supports all parameters
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    return _resnet(block_type, layers, **filtered_kwargs)


def resnet18(quantization_method="linear", **kwargs):
    """ResNet-18 factory function."""
    # Import the right block type based on quantization method
    if quantization_method.lower() == "linear":
        from .resnet import BasicBlock
    elif quantization_method.lower() == "log":
        from .log_resnet import BasicBlock 
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    return create_resnet(BasicBlock, [2, 2, 2, 2], 
                        quantization_method=quantization_method, **kwargs)


def resnet50(quantization_method="linear", **kwargs):
    """ResNet-50 factory function."""
    # Import the right block type based on quantization method
    if quantization_method.lower() == "linear":
        from .resnet import Bottleneck
    elif quantization_method.lower() == "log":
        from .log_resnet import Bottleneck
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    return create_resnet(Bottleneck, [3, 4, 6, 3], 
                        quantization_method=quantization_method, **kwargs)