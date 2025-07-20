from .unified_resnet import resnet18 as unified_resnet18, resnet50 as unified_resnet50
from .unified_mobilenet import mobilenetv1 as unified_mobilenetv1, mobilenetv2 as unified_mobilenetv2  # Fixed typo: unfified -> unified
from .unified_vit import (
    industry_vit_tiny as unified_vit_tiny, 
    industry_vit_small as unified_vit_small,
    industry_vit_base as unified_vit_base,
    industry_vit_large as unified_vit_large
)
from .unified_mobilenetv3 import mobilenetv3_large_factory, mobilenetv3_small_factory

from .unified_deit import deit_tiny, deit_small, deit_base
from .unified_tiny_gpt import tiny_gpt_micro, tiny_gpt_mini, tiny_gpt_nano, tiny_gpt_small


def resnet18(quantization_method="linear", **kwargs):
    return unified_resnet18(quantization_method=quantization_method, **kwargs)

def resnet50(quantization_method="linear", **kwargs):
    return unified_resnet50(quantization_method=quantization_method, **kwargs)


def mobilenetv1(quantization_method="linear", **kwargs):
    return unified_mobilenetv1(quantization_method=quantization_method, **kwargs)

def mobilenetv2(quantization_method="linear", **kwargs):
    return unified_mobilenetv2(quantization_method=quantization_method, **kwargs)


def mobilenetv3_small(quantization_method="linear", **kwargs):
    return mobilenetv3_small_factory(quantization_method=quantization_method, **kwargs)

def mobilenetv3_large(quantization_method="linear", **kwargs):
    return mobilenetv3_large_factory(quantization_method=quantization_method, **kwargs)

# ViT models  
def vit_tiny(quantization_method="linear", **kwargs):
    return unified_vit_tiny(quantization_method=quantization_method, **kwargs)

def vit_small(quantization_method="linear", **kwargs):
    return unified_vit_small(quantization_method=quantization_method, **kwargs)

def vit_base(quantization_method="linear", **kwargs):
    return unified_vit_base(quantization_method=quantization_method, **kwargs)

def vit_large(quantization_method="linear", **kwargs):
    return unified_vit_large(quantization_method=quantization_method, **kwargs)

# DeiT models
def deit_tiny_model(quantization_method="linear", **kwargs):
    return deit_tiny(quantization_method=quantization_method, **kwargs)

def deit_small_model(quantization_method="linear", **kwargs):
    return deit_small(quantization_method=quantization_method, **kwargs)

def deit_base_model(quantization_method="linear", **kwargs):
    return deit_base(quantization_method=quantization_method, **kwargs)


def tinygpt_nano(quantization_method="linear", **kwargs):
    return tiny_gpt_nano(quantization_method=quantization_method, **kwargs)

def tinygpt_micro(quantization_method="linear", **kwargs):
    return tiny_gpt_micro(quantization_method=quantization_method, **kwargs)

def tinygpt_mini(quantization_method="linear", **kwargs):
    return tiny_gpt_mini(quantization_method=quantization_method, **kwargs)

def tinygpt_small(quantization_method="linear", **kwargs):
    return tiny_gpt_small(quantization_method=quantization_method, **kwargs)