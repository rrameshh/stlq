import torch
import torchvision.models as models
import timm
from collections import OrderedDict


# ============================================================================
# UNIFIED CORE FUNCTIONS (handles all models)
# ============================================================================

def _detect_quantized_layers(custom_dict):
    """Detect if model uses quantized layers (.linear submodule)"""
    for key in custom_dict.keys():
        if '.linear.weight' in key:
            return True
    return False


def _transfer_weights_universal(pretrained_dict, custom_dict, weight_mapping, num_classes=100):

    transferred, skipped, errors = 0, 0, 0
    
    for pretrained_key, custom_key in weight_mapping.items():
        if pretrained_key in pretrained_dict and custom_key in custom_dict:
            try:
                pretrained_param = pretrained_dict[pretrained_key]
                custom_param = custom_dict[custom_key]
                
                if pretrained_param.shape == custom_param.shape:
                    custom_dict[custom_key].copy_(pretrained_param)
                    transferred += 1
                elif 'head' in custom_key and num_classes != 1000:
                    print(f"Skipping classifier due to class mismatch: {pretrained_param.shape} vs {custom_param.shape}")
                    skipped += 1
                else:
                    print(f"Shape mismatch {pretrained_key}: {pretrained_param.shape} vs {custom_key}: {custom_param.shape}")
                    errors += 1
            except Exception as e:
                print(f"Error transferring {pretrained_key} -> {custom_key}: {e}")
                errors += 1
        else:
            skipped += 1
    
    print(f"Weight transfer: {transferred} transferred, {skipped} skipped, {errors} errors")
    return transferred, skipped, errors


def _reset_quantization_params(model):
    """Reset quantization parameters for all models"""
    for module in model.modules():
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked.data.fill_(0)
        if hasattr(module, 'running_min'):
            module.running_min.data.fill_(0.0)
        if hasattr(module, 'running_max'):
            module.running_max.data.fill_(0.0)
        if hasattr(module, 'bn2d'):
            module.bn2d.reset_running_stats()
        if hasattr(module, 'reset_stats'):
            module.reset_stats()


def _finalize_loading(model, custom_dict, model_name):
    """Finalize weight loading for all models"""
    model.load_state_dict(custom_dict)
    _reset_quantization_params(model)
    print(f"Loaded pretrained {model_name} weights")
    return model


# ============================================================================
# TRANSFORMER MODELS (ViT, DeiT, Swin) - Unified Approach
# ============================================================================

def _create_transformer_block_mapping(pretrained_dict, custom_dict, block_prefix, is_quantized):
    """Create mapping for transformer blocks (works for ViT, DeiT, Swin)"""
    mapping = {}
    
    # Layer norms (always FP32)
    mapping.update({
        f'{block_prefix}.norm1.weight': f'{block_prefix}.norm1.weight',
        f'{block_prefix}.norm1.bias': f'{block_prefix}.norm1.bias',
        f'{block_prefix}.norm2.weight': f'{block_prefix}.norm2.weight',
        f'{block_prefix}.norm2.bias': f'{block_prefix}.norm2.bias',
    })
    
    # Attention layers
    if is_quantized:
        mapping.update({
            f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.linear.weight',
            f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.linear.bias',
            f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.linear.weight',
            f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.linear.bias',
        })
        # MLP layers  
        mapping.update({
            f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.linear.weight',
            f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.linear.bias',
            f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.linear.weight',
            f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.linear.bias',
        })
    else:
        mapping.update({
            f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.weight',
            f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.bias',
            f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.weight',
            f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.bias',
            f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.weight',
            f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.bias',
            f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.weight',
            f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.bias',
        })
    
    return mapping


def _load_transformer_weights(model, timm_model_name, num_classes, img_size, model_type):
    """Unified transformer weight loading for ViT, DeiT, Swin"""
    print(f"Loading pretrained {model_type} weights from {timm_model_name}...")
    
    # Load pretrained model
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True, img_size=img_size)
    except Exception as e:
        print(f"Failed to load from timm: {e}")
        return model
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    is_quantized = _detect_quantized_layers(custom_dict)
    
    # Create weight mapping based on model type
    if model_type == "swin":
        weight_mapping = _create_swin_mapping(pretrained_dict, custom_dict, is_quantized)
    else:  # ViT or DeiT
        weight_mapping = _create_vit_deit_mapping(pretrained_dict, custom_dict, is_quantized, model_type)
    
    # Transfer weights
    _transfer_weights_universal(pretrained_dict, custom_dict, weight_mapping, num_classes)
    
    return _finalize_loading(model, custom_dict, model_type)


def _create_vit_deit_mapping(pretrained_dict, custom_dict, is_quantized, model_type):
    """Create ViT/DeiT weight mapping"""
    mapping = {}
    
    # Patch embedding
    if 'patch_embed.projection.0.weight' in custom_dict:
        mapping.update({
            'patch_embed.proj.weight': 'patch_embed.projection.0.weight',
            'patch_embed.proj.bias': 'patch_embed.projection.0.bias',
        })
    
    # Position/class embeddings
    mapping.update({
        'pos_embed': 'pos_embed',
        'cls_token': 'cls_token',
    })
    
    # DeiT-specific distillation token
    if model_type == "deit" and 'dist_token' in custom_dict:
        mapping['dist_token'] = 'dist_token'
    
    # Transformer blocks
    num_blocks = len([k for k in custom_dict.keys() if k.startswith('blocks.') and '.norm1.weight' in k])
    for i in range(num_blocks):
        block_mapping = _create_transformer_block_mapping(
            pretrained_dict, custom_dict, f'blocks.{i}', is_quantized
        )
        mapping.update(block_mapping)
    
    # Final norm
    mapping.update({
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    })
    
    # Classification head(s)
    if model_type == "deit" and 'head_cls.weight' in custom_dict:
        # DeiT dual heads
        head_suffix = '.linear' if is_quantized else ''
        mapping.update({
            'head.weight': f'head_cls{head_suffix}.weight',
            'head.bias': f'head_cls{head_suffix}.bias',
            'head_dist.weight': f'head_dist{head_suffix}.weight',
            'head_dist.bias': f'head_dist{head_suffix}.bias',
        })
    else:
        # Single head
        head_suffix = '.linear' if is_quantized else ''
        mapping.update({
            'head.weight': f'head{head_suffix}.weight',
            'head.bias': f'head{head_suffix}.bias',
        })
    
    return mapping


def _create_swin_mapping(pretrained_dict, custom_dict, is_quantized):
    """Create Swin weight mapping with adaptive depths"""
    mapping = {}
    
    # Patch embedding
    if 'patch_embed.projection.weight' in custom_dict:
        mapping.update({
            'patch_embed.proj.weight': 'patch_embed.projection.weight',
            'patch_embed.proj.bias': 'patch_embed.projection.bias',
        })
    
    # Detect depths dynamically
    def get_depths(state_dict):
        depths = {}
        for key in state_dict.keys():
            if 'layers.' in key and '.blocks.' in key and '.norm1.weight' in key:
                parts = key.split('.')
                layer_idx, block_idx = int(parts[1]), int(parts[3])
                depths[layer_idx] = max(depths.get(layer_idx, 0), block_idx + 1)
        return [depths.get(i, 0) for i in range(max(depths.keys()) + 1)] if depths else []
    
    pretrained_depths = get_depths(pretrained_dict)
    custom_depths = get_depths(custom_dict)
    print(f"Depths - Pretrained: {pretrained_depths}, Custom: {custom_depths}")
    
    # Map transformer blocks
    for layer_idx in range(max(len(pretrained_depths), len(custom_depths))):
        pretrained_blocks = pretrained_depths[layer_idx] if layer_idx < len(pretrained_depths) else 0
        custom_blocks = custom_depths[layer_idx] if layer_idx < len(custom_depths) else 0
        blocks_to_map = min(pretrained_blocks, custom_blocks)
        
        for block_idx in range(blocks_to_map):
            block_mapping = _create_transformer_block_mapping(
                pretrained_dict, custom_dict, f'layers.{layer_idx}.blocks.{block_idx}', is_quantized
            )
            mapping.update(block_mapping)
    
    # Downsample layers (adaptive mapping)
    pretrained_downsamples = [int(k.split('.')[1]) for k in pretrained_dict.keys() if 'downsample.norm.weight' in k]
    custom_downsamples = [int(k.split('.')[1]) for k in custom_dict.keys() if 'downsample.norm.weight' in k]
    
    for i, (custom_idx, pretrained_idx) in enumerate(zip(custom_downsamples, pretrained_downsamples)):
        reduction_suffix = '.linear' if is_quantized else ''
        mapping.update({
            f'layers.{pretrained_idx}.downsample.norm.weight': f'layers.{custom_idx}.downsample.norm.weight',
            f'layers.{pretrained_idx}.downsample.norm.bias': f'layers.{custom_idx}.downsample.norm.bias',
            f'layers.{pretrained_idx}.downsample.reduction.weight': f'layers.{custom_idx}.downsample.reduction{reduction_suffix}.weight',
            f'layers.{pretrained_idx}.downsample.reduction.bias': f'layers.{custom_idx}.downsample.reduction{reduction_suffix}.bias',
        })
    
    # Final norm and head
    head_suffix = '.linear' if is_quantized else ''
    mapping.update({
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
        'head.weight': f'head{head_suffix}.weight',
        'head.bias': f'head{head_suffix}.bias',
    })
    
    return mapping


# ============================================================================
# CNN MODELS (ResNet, MobileNet)
# ============================================================================

def _load_cnn_weights(model, pretrained_model, weight_mapping, model_name, num_classes):
    """Unified CNN weight loading"""
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    
    _transfer_weights_universal(pretrained_dict, custom_dict, weight_mapping, num_classes)
    return _finalize_loading(model, custom_dict, model_name)


def _create_resnet_mapping(custom_dict, num_classes):
    """Create ResNet weight mapping"""
    mapping = {
        'conv1.weight': 'conv1.conv2d.weight',
        'bn1.weight': 'conv1.bn2d.weight',
        'bn1.bias': 'conv1.bn2d.bias',
        'bn1.running_mean': 'conv1.bn2d.running_mean',
        'bn1.running_var': 'conv1.bn2d.running_var',
        'bn1.num_batches_tracked': 'conv1.bn2d.num_batches_tracked',
    }
    
    # Layer blocks
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            for conv_idx in range(1, 3):
                mapping.update({
                    f'layer{layer_idx}.{block_idx}.conv{conv_idx}.weight': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.conv2d.weight',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.weight': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.weight',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.bias': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.bias',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.running_mean': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.running_mean',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.running_var': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.running_var',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.num_batches_tracked': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.num_batches_tracked',
                })
    
    # Downsample layers
    for layer_idx in [2, 3, 4]:
        mapping.update({
            f'layer{layer_idx}.0.downsample.0.weight': f'layer{layer_idx}.0.downsample.0.conv2d.weight',
            f'layer{layer_idx}.0.downsample.1.weight': f'layer{layer_idx}.0.downsample.0.bn2d.weight',
            f'layer{layer_idx}.0.downsample.1.bias': f'layer{layer_idx}.0.downsample.0.bn2d.bias',
            f'layer{layer_idx}.0.downsample.1.running_mean': f'layer{layer_idx}.0.downsample.0.bn2d.running_mean',
            f'layer{layer_idx}.0.downsample.1.running_var': f'layer{layer_idx}.0.downsample.0.bn2d.running_var',
            f'layer{layer_idx}.0.downsample.1.num_batches_tracked': f'layer{layer_idx}.0.downsample.0.bn2d.num_batches_tracked',
        })
    
    # Classifier
    if num_classes == 1000:
        mapping.update({
            'fc.weight': 'fc.linear.weight',
            'fc.bias': 'fc.linear.bias',
        })
    
    return mapping


# ============================================================================
# PUBLIC API - Main Functions
# ============================================================================

def load_pretrained_vit(model, variant="small", num_classes=100, img_size=224):
    """Load pretrained ViT weights"""
    timm_models = {
        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224", 
        "base": "vit_base_patch16_224",
        "large": "vit_large_patch16_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "vit")


def load_pretrained_deit(model, variant="small", num_classes=100, img_size=224):
    """Load pretrained DeiT weights"""
    timm_models = {
        "tiny": "deit_tiny_patch16_224",
        "small": "deit_small_patch16_224", 
        "base": "deit_base_patch16_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "deit")


def load_pretrained_swin(model, variant="tiny", num_classes=100, img_size=224):
    """Load pretrained Swin weights (fully adaptive)"""
    timm_models = {
        "tiny": "swin_tiny_patch4_window7_224",
        "small": "swin_small_patch4_window7_224", 
        "base": "swin_base_patch4_window7_224",
        "large": "swin_large_patch4_window7_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "swin")


def load_pretrained_resnet(model, num_classes=100):
    """Load pretrained ResNet weights"""
    pretrained_model = models.resnet18(pretrained=True)
    weight_mapping = _create_resnet_mapping(model.state_dict(), num_classes)
    return _load_cnn_weights(model, pretrained_model, weight_mapping, "ResNet", num_classes)


def load_pretrained_mobilenet(model, version="v2", num_classes=100):
    """Load pretrained MobileNet weights (simplified - v2 only for brevity)"""
    if version == "v2":
        pretrained_model = models.mobilenet_v2(pretrained=True)
        # Simplified mapping - you can expand this
        weight_mapping = {
            'features.0.0.weight': 'features.0.conv2d.weight',
            'features.0.1.weight': 'features.0.bn2d.weight',
            'features.0.1.bias': 'features.0.bn2d.bias',
            # Add more mappings as needed...
        }
        return _load_cnn_weights(model, pretrained_model, weight_mapping, f"MobileNet{version}", num_classes)
    else:
        print(f"MobileNet {version} not implemented in compact version")
        return model


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# ViT variants
def load_pretrained_vit_tiny(model, **kwargs):
    return load_pretrained_vit(model, variant="tiny", **kwargs)

def load_pretrained_vit_small(model, **kwargs):
    return load_pretrained_vit(model, variant="small", **kwargs)

def load_pretrained_vit_base(model, **kwargs):
    return load_pretrained_vit(model, variant="base", **kwargs)

# DeiT variants
def load_pretrained_deit_tiny(model, **kwargs):
    return load_pretrained_deit(model, variant="tiny", **kwargs)

def load_pretrained_deit_small(model, **kwargs):
    return load_pretrained_deit(model, variant="small", **kwargs)

# Swin variants  
def load_pretrained_swin_tiny(model, **kwargs):
    return load_pretrained_swin(model, variant="tiny", **kwargs)

def load_pretrained_swin_small(model, **kwargs):
    return load_pretrained_swin(model, variant="small", **kwargs)

def load_pretrained_swin_base(model, **kwargs):
    return load_pretrained_swin(model, variant="base", **kwargs)