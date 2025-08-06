# models/registry.py - Centralized model registration
from dataclasses import dataclass
from typing import Dict, Any, Callable, Set
from quantization.quant_config import QuantizationConfig

@dataclass
class ModelSpec:
    """Specification for a model including supported parameters."""
    constructor: Callable
    supports_img_size: bool = False
    supports_vocab_size: bool = False
    default_img_size: int = 224
    model_type: str = "vision"  # "vision", "language"

class ModelRegistry:
    """Centralized model registry with parameter validation."""
    
    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._register_all_models()
    
    def register(self, name: str, spec: ModelSpec):
        """Register a model with its specification."""
        self._models[name] = spec
    
    def create_model(self, name: str, quant_config: QuantizationConfig, **kwargs) -> Any:
        """Create a model with proper parameter filtering."""
        if name not in self._models:
            available = ', '.join(sorted(self._models.keys()))
            raise ValueError(f"Unknown model '{name}'. Available: {available}")
        
        spec = self._models[name]
        
        # Build filtered kwargs
        model_kwargs = {
            'quantization_method': quant_config.method,
            **quant_config.__dict__
        }
        
        # Add model-specific parameters only if supported
        if spec.supports_img_size and 'img_size' in kwargs:
            model_kwargs['img_size'] = kwargs['img_size']
        elif spec.supports_img_size:
            model_kwargs['img_size'] = spec.default_img_size
            
        if spec.supports_vocab_size and 'vocab_size' in kwargs:
            model_kwargs['vocab_size'] = kwargs['vocab_size']
        
        # Always add num_classes for classification models
        if 'num_classes' in kwargs:
            model_kwargs['num_classes'] = kwargs['num_classes']
        
        # Add pretrained flag
        if 'pretrained' in kwargs:
            model_kwargs['pretrained'] = kwargs['pretrained']
        
        return spec.constructor(**model_kwargs)
    
    def get_model_type(self, name: str) -> str:
        """Get the type of a model (vision/language)."""
        if name not in self._models:
            return "unknown"
        return self._models[name].model_type
    
    def list_models(self) -> Dict[str, str]:
        """List all available models with their types."""
        return {name: spec.model_type for name, spec in self._models.items()}
    
    def _register_all_models(self):
        """Register all available models."""
        # Import here to avoid circular imports
        from .vision.cnn.resnet import resnet18, resnet50
        from .vision.cnn.mobilenet import mobilenetv1, mobilenetv2
        from .vision.cnn.mobilenetv3 import mobilenetv3_small, mobilenetv3_large
        from .vision.transformer.vit import industry_vit_tiny, industry_vit_small, industry_vit_base
        from .vision.transformer.deit import deit_tiny, deit_small, deit_base
        from .vision.transformer.swin import swin_tiny, swin_small, swin_base
        from .language.tinybert import tiny_bert_tiny, tiny_bert_mini, tiny_bert_small, tiny_bert_base
        from .language.tinygpt import tiny_gpt_nano, tiny_gpt_micro, tiny_gpt_mini, tiny_gpt_small
        
        # CNN Models - no img_size needed
        self.register('resnet18', ModelSpec(resnet18, model_type="vision"))
        self.register('resnet50', ModelSpec(resnet50, model_type="vision"))
        self.register('mobilenetv1', ModelSpec(mobilenetv1, model_type="vision"))
        self.register('mobilenetv2', ModelSpec(mobilenetv2, model_type="vision"))
        self.register('mobilenetv3_small', ModelSpec(mobilenetv3_small, model_type="vision"))
        self.register('mobilenetv3_large', ModelSpec(mobilenetv3_large, model_type="vision"))
        
        # Vision Transformers - need img_size
        self.register('vit_tiny', ModelSpec(industry_vit_tiny, supports_img_size=True, model_type="vision"))
        self.register('vit_small', ModelSpec(industry_vit_small, supports_img_size=True, model_type="vision"))
        self.register('vit_base', ModelSpec(industry_vit_base, supports_img_size=True, model_type="vision"))
        self.register('deit_tiny', ModelSpec(deit_tiny, supports_img_size=True, model_type="vision"))
        self.register('deit_small', ModelSpec(deit_small, supports_img_size=True, model_type="vision"))
        self.register('deit_base', ModelSpec(deit_base, supports_img_size=True, model_type="vision"))
        self.register('swin_tiny', ModelSpec(swin_tiny, supports_img_size=True, model_type="vision"))
        self.register('swin_small', ModelSpec(swin_small, supports_img_size=True, model_type="vision"))
        self.register('swin_base', ModelSpec(swin_base, supports_img_size=True, model_type="vision"))
        
        # Language Models - need vocab_size
        self.register('tinybert_tiny', ModelSpec(tiny_bert_tiny, supports_vocab_size=True, model_type="language"))
        self.register('tinybert_mini', ModelSpec(tiny_bert_mini, supports_vocab_size=True, model_type="language"))
        self.register('tinybert_small', ModelSpec(tiny_bert_small, supports_vocab_size=True, model_type="language"))
        self.register('tinybert_base', ModelSpec(tiny_bert_base, supports_vocab_size=True, model_type="language"))
        self.register('tinygpt_nano', ModelSpec(tiny_gpt_nano, supports_vocab_size=True, model_type="language"))
        self.register('tinygpt_micro', ModelSpec(tiny_gpt_micro, supports_vocab_size=True, model_type="language"))
        self.register('tinygpt_mini', ModelSpec(tiny_gpt_mini, supports_vocab_size=True, model_type="language"))
        self.register('tinygpt_small', ModelSpec(tiny_gpt_small, supports_vocab_size=True, model_type="language"))

# Global registry instance
_registry = ModelRegistry()

def create_model(name: str, quant_config: QuantizationConfig, **kwargs):
    """Factory function using the registry."""
    return _registry.create_model(name, quant_config, **kwargs)

def list_models():
    """List all available models."""
    return _registry.list_models()

def get_model_type(name: str):
    """Get model type."""
    return _registry.get_model_type(name)