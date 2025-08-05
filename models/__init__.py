# models/__init__.py - Clean modular model access

# Direct imports from organized structure
from .vision.cnn.resnet import resnet18, resnet50
from .vision.cnn.mobilenet import mobilenetv1, mobilenetv2
from .vision.cnn.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from .vision.transformer.vit import industry_vit_tiny as vit_tiny, industry_vit_small as vit_small, industry_vit_base as vit_base
from .vision.transformer.deit import deit_tiny, deit_small, deit_base
from .vision.transformer.swin import swin_tiny, swin_small, swin_base
from .language.tinybert import tiny_bert_tiny, tiny_bert_mini, tiny_bert_small, tiny_bert_base

# Import consolidated pretrained loading
from .pretrained import load_pretrained_weights

# Simple model registry - no bloated metadata
MODELS = {
    # CNN models
    'resnet18': resnet18,
    'resnet50': resnet50,
    'mobilenetv1': mobilenetv1,
    'mobilenetv2': mobilenetv2,
    'mobilenetv3_small': mobilenetv3_small,
    'mobilenetv3_large': mobilenetv3_large,
    
    # Vision transformers
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'deit_tiny': deit_tiny,
    'deit_small': deit_small,
    'deit_base': deit_base,
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
    'swin_base': swin_base,
    
    # Language models
    'tinybert_tiny': tiny_bert_tiny,
    'tinybert_mini': tiny_bert_mini,
    'tinybert_small': tiny_bert_small,
    'tinybert_base': tiny_bert_base,
}

def create_model(model_name: str, pretrained: bool = False, **kwargs):
    """
    Create model with optional pretrained weights.
    
    Args:
        model_name: Name like 'resnet18', 'vit_small', etc.
        pretrained: Whether to load pretrained weights
        **kwargs: Model constructor arguments
        
    Returns:
        Model with optional pretrained weights
    """
    if model_name not in MODELS:
        available = ', '.join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    # Create model
    constructor = MODELS[model_name]
    model = constructor(**kwargs)
    
    # Load pretrained weights if requested
    if pretrained:
        try:
            model = load_pretrained_weights(model, model_name, **kwargs)
            print(f"✅ Loaded pretrained weights for {model_name}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load pretrained weights for {model_name}: {e}")
            print("Continuing with random initialization...")
    
    return model

def list_models():
    """List all available models."""
    return sorted(MODELS.keys())

# Export everything
__all__ = ['create_model', 'list_models'] + list(MODELS.keys())