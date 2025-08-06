# models/__init__.py - Replace the entire registry.py with this:
from .vision.resnet import resnet18, resnet50
from .vision.vit import vit_small  # Just keep the ones you use
from .vision.swin import swin_tiny

MODELS = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'vit_small': vit_small,
    'swin_tiny': swin_tiny,
}

def create_model(name, config):
    """Simple model creation - no complex parameter validation"""
    if name not in MODELS:
        raise ValueError(f"Model {name} not found")
    
    # Convert main config to quantization config  
    from quantization.quant_config import QuantizationConfig
    quant_config = QuantizationConfig(
        method=config.quantization.method,
        threshold=config.quantization.threshold,
        device=config.system.device
    )
    
    return MODELS[name](
        quantization_method=config.quantization.method,
        num_classes=config.model.num_classes,
        img_size=getattr(config.model, 'img_size', 224),
        device=config.system.device,
        threshold=config.quantization.threshold,
        bits=config.quantization.bits
    )