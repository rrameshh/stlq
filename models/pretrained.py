# models/pretrained.py - Single pretrained loading module

import torch
import torchvision.models as models
import timm
from pathlib import Path

# Import your existing loading functions
from .load_pretrained import (
    load_pretrained_resnet,
    load_pretrained_mobilenet,
    load_pretrained_vit, 
    load_pretrained_deit,
    load_pretrained_swin
)

# Model type mapping - auto-detect from name
MODEL_TYPES = {
    # CNN models
    'resnet': ['resnet18', 'resnet50'],
    'mobilenet': ['mobilenetv1', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large'],
    
    # Transformer models  
    'vit': ['vit_tiny', 'vit_small', 'vit_base'],
    'deit': ['deit_tiny', 'deit_small', 'deit_base'],
    'swin': ['swin_tiny', 'swin_small', 'swin_base'],
    
    # Language models (no pretrained)
    'language': ['tinybert_tiny', 'tinybert_mini', 'tinybert_small', 'tinybert_base'],
}

# Pretrained loader mapping
PRETRAINED_LOADERS = {
    'resnet': load_pretrained_resnet,
    'mobilenet': load_pretrained_mobilenet,
    'vit': load_pretrained_vit,
    'deit': load_pretrained_deit,
    'swin': load_pretrained_swin,
}

def _get_model_type(model_name: str) -> str:
    """Auto-detect model type from name."""
    for model_type, model_list in MODEL_TYPES.items():
        if model_name in model_list:
            return model_type
    
    # Fallback: detect from name prefix
    if model_name.startswith(('resnet', 'mobilenet', 'vit', 'deit', 'swin')):
        return model_name.split('_')[0].replace('net', '').replace('v3', '')
    
    return 'unknown'

def _extract_variant(model_name: str) -> str:
    """Extract variant from model name."""
    # Handle special cases
    variant_map = {
        'resnet18': '18',
        'resnet50': '50',
        'mobilenetv1': 'v1',
        'mobilenetv2': 'v2', 
        'mobilenetv3_small': 'v3_small',
        'mobilenetv3_large': 'v3_large',
        'vit_tiny': 'tiny',
        'vit_small': 'small',
        'vit_base': 'base',
        'deit_tiny': 'tiny',
        'deit_small': 'small',
        'deit_base': 'base',
        'swin_tiny': 'tiny',
        'swin_small': 'small',
        'swin_base': 'base',
    }
    
    return variant_map.get(model_name, model_name.split('_')[-1])

def load_pretrained_weights(model, model_name: str, **kwargs):
    """
    Universal pretrained weight loader.
    
    Args:
        model: Your quantized model instance
        model_name: Model name (e.g., 'resnet18', 'vit_small')
        **kwargs: Model arguments (num_classes, img_size, etc.)
        
    Returns:
        Model with pretrained weights loaded
    """
    model_type = _get_model_type(model_name)
    
    # Skip loading for language models (no pretrained available)
    if model_type == 'language':
        print(f"No pretrained weights available for {model_name} (language model)")
        return model
    
    # Get the appropriate loader
    if model_type not in PRETRAINED_LOADERS:
        print(f"No pretrained loader available for model type: {model_type}")
        return model
    
    loader_func = PRETRAINED_LOADERS[model_type]
    variant = _extract_variant(model_name)
    
    # Extract relevant kwargs for pretrained loading
    loading_kwargs = {
        'num_classes': kwargs.get('num_classes', 1000),
        'img_size': kwargs.get('img_size', 224),
    }
    
    # Load pretrained weights
    return loader_func(model, variant=variant, **loading_kwargs)