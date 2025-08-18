
from .vision.resnet import resnet18, resnet50
from .vision.vit import vit_small, vit_tiny, vit_base, vit_large
from .vision.swin import swin_tiny, swin_small, swin_base
from .vision.deit import deit_tiny, deit_small, deit_base
from .vision.mobilenet import mobilenetv1, mobilenetv2
from .vision.mobilenetv3 import mobilenetv3_large, mobilenetv3_small

from .language.tinygpt import tinygpt_nano, tinygpt_micro, tinygpt_mini, tinygpt_small
from .language.tinybert import tinybert_tiny, tinybert_mini, tinybert_small, tinybert_base

MODELS = {
    # CNN models
    'resnet18': resnet18,
    'resnet50': resnet50,
    'mobilenetv1': mobilenetv1,
    'mobilenetv2': mobilenetv2,
    'mobilenetv3_small': mobilenetv3_small,
    'mobilenetv3_large': mobilenetv3_large,
    
    # Vision Transformers
    'vit_tiny': vit_tiny,
    'vit_small': vit_small,
    'vit_base': vit_base,
    'vit_large': vit_large,
    'deit_tiny': deit_tiny,
    'deit_small': deit_small,
    'deit_base': deit_base,
    'swin_tiny': swin_tiny,
    'swin_small': swin_small,
    'swin_base': swin_base,

    'tinygpt_nano': tinygpt_nano,
    'tinygpt_micro': tinygpt_micro,
    'tinygpt_mini': tinygpt_mini,
    'tinygpt_small': tinygpt_small,
    'tinybert_tiny': tinybert_tiny,
    'tinybert_mini': tinybert_mini,
    'tinybert_small': tinybert_small,
    'tinybert_base': tinybert_base,
}

MODEL_TYPES = {
    'resnet18': 'cnn',
    'resnet50': 'cnn',
    'mobilenetv1': 'cnn',
    'mobilenetv2': 'cnn',
    'mobilenetv3_small': 'cnn',
    'mobilenetv3_large': 'cnn',
    'vit_tiny': 'transformer',
    'vit_small': 'transformer',
    'vit_base': 'transformer',
    'vit_large': 'transformer',
    'deit_tiny': 'transformer',
    'deit_small': 'transformer',
    'deit_base': 'transformer',
    'swin_tiny': 'transformer',
    'swin_small': 'transformer',
    'swin_base': 'transformer',

    'tinygpt_nano': 'language_model',
    'tinygpt_micro': 'language_model',
    'tinygpt_mini': 'language_model',
    'tinygpt_small': 'language_model',
    'tinybert_tiny': 'language_model',
    'tinybert_mini': 'language_model',
    'tinybert_small': 'language_model',
    'tinybert_base': 'language_model',
}

def create_model(name, config):
    if name not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    
    model_func = MODELS[name]
    return model_func(config)

def list_models():
    return list(MODELS.keys())

def get_model_type(name):
    if name not in MODEL_TYPES:
        return 'unknown'
    return MODEL_TYPES[name]