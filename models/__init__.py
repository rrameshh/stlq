# models/__init__.py - Clean interface using registry
from .registry import create_model, list_models, get_model_type
from .pretrained import load_pretrained_weights

# Export the key functions
__all__ = ['create_model', 'list_models', 'get_model_type', 'load_pretrained_weights']