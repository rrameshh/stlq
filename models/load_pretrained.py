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
    
    model.load_state_dict(custom_dict)
    _reset_quantization_params(model)
    print(f"Loaded pretrained {model_name} weights")
    return model


# ============================================================================
# TRANSFORMER MODELS (ViT, DeiT, Swin) 
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

    print(f"Loading pretrained {model_type} weights from {timm_model_name}...")
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True, img_size=img_size)
    except Exception as e:
        print(f"Failed to load from timm: {e}")
        return model
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    is_quantized = _detect_quantized_layers(custom_dict)
    
    if model_type == "swin":
        weight_mapping = _create_swin_mapping(pretrained_dict, custom_dict, is_quantized)
    else: 
        weight_mapping = _create_vit_deit_mapping(pretrained_dict, custom_dict, is_quantized, model_type)
    
    _transfer_weights_universal(pretrained_dict, custom_dict, weight_mapping, num_classes)
    return _finalize_loading(model, custom_dict, model_type)


def _create_vit_deit_mapping(pretrained_dict, custom_dict, is_quantized, model_type):

    mapping = {}
    
    if 'patch_embed.projection.0.weight' in custom_dict:
        mapping.update({
            'patch_embed.proj.weight': 'patch_embed.projection.0.weight',
            'patch_embed.proj.bias': 'patch_embed.projection.0.bias',
        })
    
    mapping.update({
        'pos_embed': 'pos_embed',
        'cls_token': 'cls_token',
    })
    
    if model_type == "deit" and 'dist_token' in custom_dict:
        mapping['dist_token'] = 'dist_token'
    

    num_blocks = len([k for k in custom_dict.keys() if k.startswith('blocks.') and '.norm1.weight' in k])
    for i in range(num_blocks):
        block_mapping = _create_transformer_block_mapping(
            pretrained_dict, custom_dict, f'blocks.{i}', is_quantized
        )
        mapping.update(block_mapping)
    
    mapping.update({
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    })
    
    if model_type == "deit" and 'head_cls.weight' in custom_dict:

        head_suffix = '.linear' if is_quantized else ''
        mapping.update({
            'head.weight': f'head_cls{head_suffix}.weight',
            'head.bias': f'head_cls{head_suffix}.bias',
            'head_dist.weight': f'head_dist{head_suffix}.weight',
            'head_dist.bias': f'head_dist{head_suffix}.bias',
        })
    else:

        head_suffix = '.linear' if is_quantized else ''
        mapping.update({
            'head.weight': f'head{head_suffix}.weight',
            'head.bias': f'head{head_suffix}.bias',
        })
    
    return mapping


def _create_swin_mapping(pretrained_dict, custom_dict, is_quantized):

    mapping = {}
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

    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    
    _transfer_weights_universal(pretrained_dict, custom_dict, weight_mapping, num_classes)
    return _finalize_loading(model, custom_dict, model_name)


# ============================================================================
# Main Functions
# ============================================================================

def load_pretrained_vit(model, variant="small", num_classes=100, img_size=224):
   
    timm_models = {
        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224", 
        "base": "vit_base_patch16_224",
        "large": "vit_large_patch16_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "vit")


def load_pretrained_deit(model, variant="small", num_classes=100, img_size=224):
    
    timm_models = {
        "tiny": "deit_tiny_patch16_224",
        "small": "deit_small_patch16_224", 
        "base": "deit_base_patch16_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "deit")


def load_pretrained_swin(model, variant="tiny", num_classes=100, img_size=224):
    
    timm_models = {
        "tiny": "swin_tiny_patch4_window7_224",
        "small": "swin_small_patch4_window7_224", 
        "base": "swin_base_patch4_window7_224",
        "large": "swin_large_patch4_window7_224"
    }
    return _load_transformer_weights(model, timm_models[variant], num_classes, img_size, "swin")


def load_pretrained_resnet(model, variant="18", num_classes=100):
    
    if variant == "18":
        pretrained_model = models.resnet18(pretrained=True)
        print("Loading ResNet-18 pretrained weights...")
    elif variant == "50":
        pretrained_model = models.resnet50(pretrained=True)
        print("Loading ResNet-50 pretrained weights...")
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")
    
    weight_mapping = _create_resnet_mapping(model.state_dict(), num_classes, variant)
    return _load_cnn_weights(model, pretrained_model, weight_mapping, f"ResNet-{variant}", num_classes)


def _create_resnet_mapping(custom_dict, num_classes, variant="18"):

    mapping = {
        'conv1.weight': 'conv1.conv2d.weight',
        'bn1.weight': 'conv1.bn2d.weight',
        'bn1.bias': 'conv1.bn2d.bias',
        'bn1.running_mean': 'conv1.bn2d.running_mean',
        'bn1.running_var': 'conv1.bn2d.running_var',
        'bn1.num_batches_tracked': 'conv1.bn2d.num_batches_tracked',
    }
    
    if variant == '18':
        layer_blocks = [2, 2, 2, 2] 
        num_convs = 2 
    elif variant in '50':
        layer_blocks = [3, 4, 6, 3]  # ResNet-50
        num_convs = 3 
    else:
        raise ValueError(f"Unknown ResNet variant: {variant}")
    
    # Layer blocks - dynamically handle different numbers of blocks
    for layer_idx in range(1, 5):
        num_blocks = layer_blocks[layer_idx - 1]
        for block_idx in range(num_blocks):
            for conv_idx in range(1, num_convs + 1):
                mapping.update({
                    f'layer{layer_idx}.{block_idx}.conv{conv_idx}.weight': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.conv2d.weight',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.weight': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.weight',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.bias': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.bias',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.running_mean': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.running_mean',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.running_var': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.running_var',
                    f'layer{layer_idx}.{block_idx}.bn{conv_idx}.num_batches_tracked': f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.num_batches_tracked',
                })
    
    # Downsample layers (only for layers 2, 3, 4)
    for layer_idx in [2, 3, 4]:
        mapping.update({
            f'layer{layer_idx}.0.downsample.0.weight': f'layer{layer_idx}.0.downsample.0.conv2d.weight',
            f'layer{layer_idx}.0.downsample.1.weight': f'layer{layer_idx}.0.downsample.0.bn2d.weight',
            f'layer{layer_idx}.0.downsample.1.bias': f'layer{layer_idx}.0.downsample.0.bn2d.bias',
            f'layer{layer_idx}.0.downsample.1.running_mean': f'layer{layer_idx}.0.downsample.0.bn2d.running_mean',
            f'layer{layer_idx}.0.downsample.1.running_var': f'layer{layer_idx}.0.downsample.0.bn2d.running_var',
            f'layer{layer_idx}.0.downsample.1.num_batches_tracked': f'layer{layer_idx}.0.downsample.0.bn2d.num_batches_tracked',
        })
    
    if 'fc.linear.weight' in custom_dict:
        mapping.update({
            'fc.weight': 'fc.linear.weight',
            'fc.bias': 'fc.linear.bias',
        })
    else:
        mapping.update({
            'fc.weight': 'fc.weight',
            'fc.bias': 'fc.bias',
        })
    
    return mapping

def _create_mobilenetv1_timm_mapping(custom_dict, num_classes):

    mapping = {}
    
    print("Creating MobileNet-v1 timm mapping...")
    
    # First conv layer (conv_stem in timm -> features.0 in custom)
    mapping.update({
        'conv_stem.weight': 'features.0.conv2d.weight',
        'bn1.weight': 'features.0.bn2d.weight',
        'bn1.bias': 'features.0.bn2d.bias',
        'bn1.running_mean': 'features.0.bn2d.running_mean',
        'bn1.running_var': 'features.0.bn2d.running_var',
        'bn1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    # MobileNet-v1 depthwise separable blocks
    # timm uses 'blocks.X' numbering, custom uses 'features.X' numbering
    block_configs = [
        # (timm_block_idx, custom_features_idx)
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)
    ]
    
    for timm_idx, custom_idx in block_configs:
        # Depthwise convolution
        mapping.update({
            f'blocks.{timm_idx}.conv_dw.weight': f'features.{custom_idx}.depthwise.conv2d.weight',
            f'blocks.{timm_idx}.bn_dw.weight': f'features.{custom_idx}.depthwise.bn2d.weight',
            f'blocks.{timm_idx}.bn_dw.bias': f'features.{custom_idx}.depthwise.bn2d.bias',
            f'blocks.{timm_idx}.bn_dw.running_mean': f'features.{custom_idx}.depthwise.bn2d.running_mean',
            f'blocks.{timm_idx}.bn_dw.running_var': f'features.{custom_idx}.depthwise.bn2d.running_var',
            f'blocks.{timm_idx}.bn_dw.num_batches_tracked': f'features.{custom_idx}.depthwise.bn2d.num_batches_tracked',
        })
        
        # Pointwise convolution  
        mapping.update({
            f'blocks.{timm_idx}.conv_pw.weight': f'features.{custom_idx}.pointwise.conv2d.weight',
            f'blocks.{timm_idx}.bn_pw.weight': f'features.{custom_idx}.pointwise.bn2d.weight',
            f'blocks.{timm_idx}.bn_pw.bias': f'features.{custom_idx}.pointwise.bn2d.bias',
            f'blocks.{timm_idx}.bn_pw.running_mean': f'features.{custom_idx}.pointwise.bn2d.running_mean',
            f'blocks.{timm_idx}.bn_pw.running_var': f'features.{custom_idx}.pointwise.bn2d.running_var',
            f'blocks.{timm_idx}.bn_pw.num_batches_tracked': f'features.{custom_idx}.pointwise.bn2d.num_batches_tracked',
        })
    
    # Global average pooling (if exists in custom model)
    if 'avgpool.weight' in custom_dict:
        mapping['global_pool.weight'] = 'avgpool.weight'
    
    # Classifier layer
    if 'classifier.linear.weight' in custom_dict:
        # Quantized classifier
        mapping.update({
            'classifier.weight': 'classifier.linear.weight',
            'classifier.bias': 'classifier.linear.bias',
        })
    elif 'classifier.weight' in custom_dict:
        # Regular classifier
        mapping.update({
            'classifier.weight': 'classifier.weight',
            'classifier.bias': 'classifier.bias',
        })
    
    print(f"Created {len(mapping)} weight mappings for MobileNet-v1")
    
    # Debug: Print first few mappings
    print("Sample mappings:")
    for i, (pretrained_key, custom_key) in enumerate(mapping.items()):
        if i < 5:
            print(f"  {pretrained_key} -> {custom_key}")
    
    return mapping

def load_pretrained_mobilenet(model, variant="v2", num_classes=100):

    print(f"Loading pretrained MobileNet-{variant} weights...")
    
    try:
        if variant == "v1":
            # Use timm for MobileNet v1
            import timm
            try:
                pretrained_model = timm.create_model('mobilenetv1_100', pretrained=True)
                weight_mapping = _create_mobilenetv1_timm_mapping(model.state_dict(), num_classes)
                return _load_cnn_weights(model, pretrained_model, weight_mapping, f"MobileNet-{variant}", num_classes)
            except Exception as timm_error:
                print(f"⚠️  Failed to load from timm: {timm_error}")
                print("Trying alternative timm model names...")
                
                # Try alternative model names
                alternative_names = ['mobilenet_v1', 'mobilenetv1_075', 'mobilenetv1_050']
                for alt_name in alternative_names:
                    try:
                        pretrained_model = timm.create_model(alt_name, pretrained=True)
                        weight_mapping = _create_mobilenetv1_timm_mapping(model.state_dict(), num_classes)
                        print(f"Successfully loaded {alt_name} from timm")
                        return _load_cnn_weights(model, pretrained_model, weight_mapping, f"MobileNet-{variant}", num_classes)
                    except:
                        continue
                
                raise ValueError("No working MobileNet-v1 model found in timm")
            
        elif variant == "v2":
            import torchvision.models as models
            pretrained_model = models.mobilenet_v2(pretrained=True)
            weight_mapping = _create_mobilenetv2_mapping(model.state_dict(), num_classes)
            return _load_cnn_weights(model, pretrained_model, weight_mapping, f"MobileNet-{variant}", num_classes)
            
        elif variant in ["v3_small", "v3_large"]:
            import torchvision.models as models
            if variant == "v3_small":
                pretrained_model = models.mobilenet_v3_small(pretrained=True)
            else:
                pretrained_model = models.mobilenet_v3_large(pretrained=True)
            weight_mapping = _create_mobilenetv3_mapping(model.state_dict(), num_classes)
            return _load_cnn_weights(model, pretrained_model, weight_mapping, f"MobileNet-{variant}", num_classes)
            
        else:
            raise ValueError(f"Unknown MobileNet version: {variant}")
        
    except Exception as e:
        print(f"⚠️  Failed to load MobileNet-{variant} pretrained weights: {e}")
        print("Continuing with random initialization...")
        return model


def _create_mobilenetv2_mapping(custom_dict, num_classes):

    mapping = {}
    
    # First conv layer (features.0)
    mapping.update({
        'features.0.0.weight': 'features.0.conv2d.weight',
        'features.0.1.weight': 'features.0.bn2d.weight',
        'features.0.1.bias': 'features.0.bn2d.bias',
        'features.0.1.running_mean': 'features.0.bn2d.running_mean',
        'features.0.1.running_var': 'features.0.bn2d.running_var',
        'features.0.1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    # MobileNet v2 inverted residual block structure:
    # Block 1: no expansion (t=1)
    # Blocks 2-7: t=6 
    # Blocks 8-14: t=6
    # Blocks 15-17: t=6
    
    # Inverted residual blocks (features.1 to features.17)
    mobilenetv2_config = [
        # (block_idx, has_expansion, stride)
        (1, False, 1),   # First inverted residual (no expansion, t=1)
        (2, True, 2), (3, True, 1),  # Stage 2
        (4, True, 2), (5, True, 1), (6, True, 1),  # Stage 3  
        (7, True, 2), (8, True, 1), (9, True, 1), (10, True, 1),  # Stage 4
        (11, True, 2), (12, True, 1), (13, True, 1),  # Stage 5
        (14, True, 1), (15, True, 1), (16, True, 1),  # Stage 6
        (17, True, 1),  # Stage 7
    ]
    
    for block_idx, has_expansion, stride in mobilenetv2_config:
        if has_expansion:
            # Expansion layer (1x1 conv)
            if f'features.{block_idx}.conv.0.0.weight' in custom_dict:
                mapping.update({
                    f'features.{block_idx}.conv.0.0.weight': f'features.{block_idx}.conv.0.conv2d.weight',
                    f'features.{block_idx}.conv.0.1.weight': f'features.{block_idx}.conv.0.bn2d.weight',
                    f'features.{block_idx}.conv.0.1.bias': f'features.{block_idx}.conv.0.bn2d.bias',
                    f'features.{block_idx}.conv.0.1.running_mean': f'features.{block_idx}.conv.0.bn2d.running_mean',
                    f'features.{block_idx}.conv.0.1.running_var': f'features.{block_idx}.conv.0.bn2d.running_var',
                    f'features.{block_idx}.conv.0.1.num_batches_tracked': f'features.{block_idx}.conv.0.bn2d.num_batches_tracked',
                })
            
            # Depthwise layer (3x3 depthwise conv)
            if f'features.{block_idx}.conv.1.0.weight' in custom_dict:
                mapping.update({
                    f'features.{block_idx}.conv.1.0.weight': f'features.{block_idx}.conv.1.conv2d.weight',
                    f'features.{block_idx}.conv.1.1.weight': f'features.{block_idx}.conv.1.bn2d.weight',
                    f'features.{block_idx}.conv.1.1.bias': f'features.{block_idx}.conv.1.bn2d.bias',
                    f'features.{block_idx}.conv.1.1.running_mean': f'features.{block_idx}.conv.1.bn2d.running_mean',
                    f'features.{block_idx}.conv.1.1.running_var': f'features.{block_idx}.conv.1.bn2d.running_var',
                    f'features.{block_idx}.conv.1.1.num_batches_tracked': f'features.{block_idx}.conv.1.bn2d.num_batches_tracked',
                })
            
            # Pointwise layer (1x1 conv, no activation)
            if f'features.{block_idx}.conv.2.0.weight' in custom_dict:
                mapping.update({
                    f'features.{block_idx}.conv.2.0.weight': f'features.{block_idx}.conv.2.conv2d.weight',
                    f'features.{block_idx}.conv.2.1.weight': f'features.{block_idx}.conv.2.bn2d.weight',
                    f'features.{block_idx}.conv.2.1.bias': f'features.{block_idx}.conv.2.bn2d.bias',
                    f'features.{block_idx}.conv.2.1.running_mean': f'features.{block_idx}.conv.2.bn2d.running_mean',
                    f'features.{block_idx}.conv.2.1.running_var': f'features.{block_idx}.conv.2.bn2d.running_var',
                    f'features.{block_idx}.conv.2.1.num_batches_tracked': f'features.{block_idx}.conv.2.bn2d.num_batches_tracked',
                })
        
        else:
            # Block 1: no expansion, starts with depthwise
            # Depthwise layer (3x3 depthwise conv)
            if f'features.{block_idx}.conv.0.0.weight' in custom_dict:
                mapping.update({
                    f'features.{block_idx}.conv.0.0.weight': f'features.{block_idx}.conv.0.conv2d.weight',
                    f'features.{block_idx}.conv.0.1.weight': f'features.{block_idx}.conv.0.bn2d.weight',
                    f'features.{block_idx}.conv.0.1.bias': f'features.{block_idx}.conv.0.bn2d.bias',
                    f'features.{block_idx}.conv.0.1.running_mean': f'features.{block_idx}.conv.0.bn2d.running_mean',
                    f'features.{block_idx}.conv.0.1.running_var': f'features.{block_idx}.conv.0.bn2d.running_var',
                    f'features.{block_idx}.conv.0.1.num_batches_tracked': f'features.{block_idx}.conv.0.bn2d.num_batches_tracked',
                })
            
            # Pointwise layer (1x1 conv)
            if f'features.{block_idx}.conv.1.0.weight' in custom_dict:
                mapping.update({
                    f'features.{block_idx}.conv.1.0.weight': f'features.{block_idx}.conv.1.conv2d.weight',
                    f'features.{block_idx}.conv.1.1.weight': f'features.{block_idx}.conv.1.bn2d.weight',
                    f'features.{block_idx}.conv.1.1.bias': f'features.{block_idx}.conv.1.bn2d.bias',
                    f'features.{block_idx}.conv.1.1.running_mean': f'features.{block_idx}.conv.1.bn2d.running_mean',
                    f'features.{block_idx}.conv.1.1.running_var': f'features.{block_idx}.conv.1.bn2d.running_var',
                    f'features.{block_idx}.conv.1.1.num_batches_tracked': f'features.{block_idx}.conv.1.bn2d.num_batches_tracked',
                })
    
    # Last conv layer (features.18) - 1x1 conv to final channels
    if 'features.18.0.weight' in custom_dict:
        mapping.update({
            'features.18.0.weight': 'features.18.conv2d.weight',
            'features.18.1.weight': 'features.18.bn2d.weight',
            'features.18.1.bias': 'features.18.bn2d.bias',
            'features.18.1.running_mean': 'features.18.bn2d.running_mean',
            'features.18.1.running_var': 'features.18.bn2d.running_var',
            'features.18.1.num_batches_tracked': 'features.18.bn2d.num_batches_tracked',
        })
    
    if 'classifier.linear.weight' in custom_dict:
        mapping.update({
            'classifier.weight': 'classifier.linear.weight',
            'classifier.bias': 'classifier.linear.bias',
        })
    elif 'classifier.weight' in custom_dict:
        mapping.update({
            'classifier.weight': 'classifier.weight',
            'classifier.bias': 'classifier.bias',
        })
    
    print(f"Created {len(mapping)} weight mappings for MobileNet v2")
    return mapping

def _map_mobilenetv3_conv_bn(mapping, block_prefix, pretrained_conv_idx, custom_conv_idx):
    """Helper function to map conv+bn layers for MobileNetV3"""
    mapping.update({
        f'{block_prefix}.conv.{pretrained_conv_idx}.0.weight': f'{block_prefix}.conv.{custom_conv_idx}.conv2d.weight',
        f'{block_prefix}.conv.{pretrained_conv_idx}.1.weight': f'{block_prefix}.conv.{custom_conv_idx}.bn2d.weight',
        f'{block_prefix}.conv.{pretrained_conv_idx}.1.bias': f'{block_prefix}.conv.{custom_conv_idx}.bn2d.bias',
        f'{block_prefix}.conv.{pretrained_conv_idx}.1.running_mean': f'{block_prefix}.conv.{custom_conv_idx}.bn2d.running_mean',
        f'{block_prefix}.conv.{pretrained_conv_idx}.1.running_var': f'{block_prefix}.conv.{custom_conv_idx}.bn2d.running_var',
        f'{block_prefix}.conv.{pretrained_conv_idx}.1.num_batches_tracked': f'{block_prefix}.conv.{custom_conv_idx}.bn2d.num_batches_tracked',
    })

def _create_mobilenetv3_mapping(custom_dict, num_classes):
    mapping = {}
    
    print("Creating MobileNet-v3 weight mapping...")
    
    # First conv layer (features.0)
    mapping.update({
        'features.0.0.weight': 'features.0.conv2d.weight',
        'features.0.1.weight': 'features.0.bn2d.weight',
        'features.0.1.bias': 'features.0.bn2d.bias',
        'features.0.1.running_mean': 'features.0.bn2d.running_mean',
        'features.0.1.running_var': 'features.0.bn2d.running_var',
        'features.0.1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    
    # Find all feature layers that are MobileNetV3 blocks
    feature_blocks = []
    for key in custom_dict.keys():
        if key.startswith('features.') and '.conv.' in key and '.conv2d.weight' in key:
            # Extract block index and conv index: features.X.conv.Y.conv2d.weight
            parts = key.split('.')
            if len(parts) >= 5 and parts[0] == 'features':
                try:
                    block_idx = int(parts[1])
                    conv_idx = int(parts[3])
                    feature_blocks.append((block_idx, conv_idx))
                except ValueError:
                    continue
    
    # Group by block index
    blocks_info = {}
    for block_idx, conv_idx in feature_blocks:
        if block_idx not in blocks_info:
            blocks_info[block_idx] = []
        blocks_info[block_idx].append(conv_idx)
    
    # Sort and process each block
    for block_idx in sorted(blocks_info.keys()):
        if block_idx == 0:  # Skip first conv (already handled)
            continue
            
        conv_indices = sorted(blocks_info[block_idx])
        num_convs = len(conv_indices)
        
        print(f"  Block {block_idx}: {num_convs} convolutions")
        
        # Map based on number of convolutions in the block
        if num_convs == 2:
            # Block without expansion (depthwise + pointwise)
            # Conv 0: Depthwise (3x3)
            # Conv 1: Pointwise (1x1, no activation)
            _map_mobilenetv3_conv_bn(mapping, f'features.{block_idx}', 0, 0)  # Depthwise
            _map_mobilenetv3_conv_bn(mapping, f'features.{block_idx}', 1, 1)  # Pointwise
            
        elif num_convs == 3:
            # Standard inverted residual block (expansion + depthwise + pointwise)
            # Conv 0: Expansion (1x1)
            # Conv 1: Depthwise (3x3) 
            # Conv 2: Pointwise (1x1, no activation)
            _map_mobilenetv3_conv_bn(mapping, f'features.{block_idx}', 0, 0)  # Expansion
            _map_mobilenetv3_conv_bn(mapping, f'features.{block_idx}', 1, 1)  # Depthwise
            _map_mobilenetv3_conv_bn(mapping, f'features.{block_idx}', 2, 2)  # Pointwise
            
        # Handle SE (Squeeze-and-Excitation) modules if present
        se_fc1_key = f'features.{block_idx}.conv.se.fc1.linear.weight'
        se_fc2_key = f'features.{block_idx}.conv.se.fc2.linear.weight'
        
        if se_fc1_key in custom_dict:
            # Map SE module weights
            mapping.update({
                f'features.{block_idx}.squeeze_excite.fc1.weight': f'features.{block_idx}.conv.se.fc1.linear.weight',
                f'features.{block_idx}.squeeze_excite.fc1.bias': f'features.{block_idx}.conv.se.fc1.linear.bias',
                f'features.{block_idx}.squeeze_excite.fc2.weight': f'features.{block_idx}.conv.se.fc2.linear.weight',
                f'features.{block_idx}.squeeze_excite.fc2.bias': f'features.{block_idx}.conv.se.fc2.linear.bias',
            })
    
    # Last conv layer (usually the highest numbered feature)
    max_feature_idx = max(int(k.split('.')[1]) for k in custom_dict.keys() 
                         if k.startswith('features.') and '.conv2d.weight' in k and 'conv.' not in k)
    
    if f'features.{max_feature_idx}.conv2d.weight' in custom_dict:
        mapping.update({
            f'features.{max_feature_idx}.0.weight': f'features.{max_feature_idx}.conv2d.weight',
            f'features.{max_feature_idx}.1.weight': f'features.{max_feature_idx}.bn2d.weight',
            f'features.{max_feature_idx}.1.bias': f'features.{max_feature_idx}.bn2d.bias',
            f'features.{max_feature_idx}.1.running_mean': f'features.{max_feature_idx}.bn2d.running_mean',
            f'features.{max_feature_idx}.1.running_var': f'features.{max_feature_idx}.bn2d.running_var',
            f'features.{max_feature_idx}.1.num_batches_tracked': f'features.{max_feature_idx}.bn2d.num_batches_tracked',
        })
    
    # Classifier layers
    # MobileNetV3-Large has a pre-classifier layer, V3-Small doesn't
    if 'pre_classifier.linear.weight' in custom_dict:
        # V3-Large with pre-classifier
        mapping.update({
            'classifier.0.weight': 'pre_classifier.linear.weight',
            'classifier.0.bias': 'pre_classifier.linear.bias',
            'classifier.3.weight': 'classifier.linear.weight',
            'classifier.3.bias': 'classifier.linear.bias',
        })
    elif 'classifier.linear.weight' in custom_dict:
        # Quantized classifier
        mapping.update({
            'classifier.weight': 'classifier.linear.weight',
            'classifier.bias': 'classifier.linear.bias',
        })
    elif 'classifier.weight' in custom_dict:
        # Regular classifier
        mapping.update({
            'classifier.weight': 'classifier.weight',
            'classifier.bias': 'classifier.bias',
        })
    
    print(f"Created {len(mapping)} weight mappings for MobileNet v3")
    
    # Debug: Print some sample mappings
    print("Sample mappings:")
    for i, (pretrained_key, custom_key) in enumerate(mapping.items()):
        if i < 5:
            print(f"  {pretrained_key} -> {custom_key}")
    
    return mapping


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