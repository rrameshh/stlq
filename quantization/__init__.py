# ops/__init__.py
from .base import QuantizedTensorBase, QuantizedOperatorBase
from .quant_config import QuantizationConfig

# Import tensor implementations
try:
    from .tensors.linear import LinearQuantizedTensor
    from .tensors.new_log import LogQuantizedTensor
except ImportError:
    pass

# Import strategies
try:
    from .strategies.factory import create_strategy
    from .strategies.linear import LinearStrategy
    from .strategies.new_log import LogStrategy
except ImportError:
    pass

# Import unified layers
try:
    from .layers.all import (
        UnifiedQuantize, QConv2dBNRelu, 
        QLinear, QAdd, QRelu,
        QMaxPool2d, UnifiedQuantizedAdaptiveAvgPool2d,
        QFlatten
    )
except ImportError:
    pass


def enable_quantization(model):
    """Enable quantization for all quantized layers in a model."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'activation_quantization'):
            module.activation_quantization = True
            count += 1
    print(f"Enabled quantization for {count} layers")


def disable_quantization(model):
    """Disable quantization for all quantized layers in a model."""
    count = 0
    for module in model.modules():
        if hasattr(module, 'activation_quantization'):
            module.activation_quantization = False
            count += 1
    print(f"Disabled quantization for {count} layers")


def get_quantization_status(model):
    """Get detailed quantization status of all layers in the model."""
    status = {}
    for name, module in model.named_modules():
        if hasattr(module, 'activation_quantization'):
            status[name] = {
                'type': type(module).__name__,
                'quantization_enabled': module.activation_quantization,
                'num_batches_tracked': getattr(module, 'num_batches_tracked', None)
            }
            
            # Add method-specific stats
            if hasattr(module, 'running_min') and hasattr(module, 'running_max'):
                status[name]['stats_type'] = 'linear'
                if hasattr(module.num_batches_tracked, 'item') and module.num_batches_tracked > 0:
                    status[name]['running_min'] = module.running_min.item()
                    status[name]['running_max'] = module.running_max.item()
            elif hasattr(module, 'running_max_abs'):
                status[name]['stats_type'] = 'log'
                if hasattr(module.num_batches_tracked, 'item') and module.num_batches_tracked > 0:
                    status[name]['running_max_abs'] = module.running_max_abs.item()
    
    return status


def print_quantization_status(model):
    """Print a human-readable quantization status."""
    status = get_quantization_status(model)
    
    if not status:
        print("No quantized layers found in model")
        return
    
    print(f"\nQuantization Status for {len(status)} layers:")
    print("-" * 80)
    
    for name, info in status.items():
        enabled_str = "✓" if info['quantization_enabled'] else "✗"
        batches = info.get('num_batches_tracked', 'N/A')
        if batches is not None and hasattr(batches, 'item'):
            batches = batches.item()
        
        print(f"{enabled_str} {name:<35} | {info['type']:<30} | Batches: {batches}")
        
        # Print stats if available and batches > 0
        if info.get('stats_type') == 'linear' and 'running_min' in info:
            print(f"    Running range: [{info['running_min']:.6f}, {info['running_max']:.6f}]")
        elif info.get('stats_type') == 'log' and 'running_max_abs' in info:
            print(f"    Running max_abs: {info['running_max_abs']:.6f}")


# Export the key functions
__all__ = [
    'QuantizedTensorBase', 'QuantizedOperatorBase', 'QuantizationConfig',
    'enable_quantization', 'disable_quantization', 
    'get_quantization_status', 'print_quantization_status',
    'create_strategy', 'LinearStrategy', 'LogStrategy',
    'LinearQuantizedTensor', 'LogQuantizedTensor'
]