import torch
import torchvision.models as models
from networks.unified_resnet import resnet18


import timm
import torch
import torch.nn as nn
from collections import OrderedDict



def load_pretrained_resnet(model, num_classes=100):
    
    # Load pretrained model
    pretrained_model = models.resnet18(pretrained=True)
    

    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()

    weight_mapping = {}
    weight_mapping.update({
        'conv1.weight': 'conv1.conv2d.weight',
        'bn1.weight': 'conv1.bn2d.weight',
        'bn1.bias': 'conv1.bn2d.bias',
        'bn1.running_mean': 'conv1.bn2d.running_mean',
        'bn1.running_var': 'conv1.bn2d.running_var',
        'bn1.num_batches_tracked': 'conv1.bn2d.num_batches_tracked',
    })
    
    for layer_idx in range(1, 5):
        for block_idx in range(2):
            for conv_idx in range(1, 3):
                weight_mapping[f'layer{layer_idx}.{block_idx}.conv{conv_idx}.weight'] = \
                    f'layer{layer_idx}.{block_idx}.conv{conv_idx}.conv2d.weight'
                
                for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                    weight_mapping[f'layer{layer_idx}.{block_idx}.bn{conv_idx}.{param}'] = \
                        f'layer{layer_idx}.{block_idx}.conv{conv_idx}.bn2d.{param}'
    
    for layer_idx in [2, 3, 4]:
        weight_mapping[f'layer{layer_idx}.0.downsample.0.weight'] = \
            f'layer{layer_idx}.0.downsample.0.conv2d.weight'
        
        for param in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
            weight_mapping[f'layer{layer_idx}.0.downsample.1.{param}'] = \
                f'layer{layer_idx}.0.downsample.0.bn2d.{param}'
    
    if num_classes == 1000:
        weight_mapping.update({
            'fc.weight': 'fc.linear.weight',
            'fc.bias': 'fc.linear.bias',
        })
    
    transferred = 0
    skipped = 0
    errors = 0
    
    for pretrained_key, custom_key in weight_mapping.items():
        if pretrained_key in pretrained_dict and custom_key in custom_dict:
            try:
                pretrained_param = pretrained_dict[pretrained_key]
                custom_param = custom_dict[custom_key]
                
                if pretrained_param.shape == custom_param.shape:
                    custom_dict[custom_key].copy_(pretrained_param)
                    transferred += 1
                else:
                    print(f"Shape mismatch {pretrained_key}: {pretrained_param.shape} vs {custom_param.shape}")
                    errors += 1
            except Exception as e:
                print(f"Error transferring {pretrained_key}: {e}")
                errors += 1
        else:
            print(f"Missing key: {pretrained_key} or {custom_key}")
            skipped += 1

    

   
    # Load the updated state dict
    model.load_state_dict(custom_dict)


    for name, module in model.named_modules():
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked.data.fill_(0)
        if hasattr(module, 'running_min'):
            module.running_min.data.fill_(0.0)
        if hasattr(module, 'running_max'):
            module.running_max.data.fill_(0.0)
        # Reset BatchNorm stats too
        if hasattr(module, 'bn2d'):
            module.bn2d.reset_running_stats()
    

    
    # Summary
    # # Save the model
    # output_file = f'resnet18_{quantization}_pretrained.pth'
    # torch.save(model.state_dict(), output_file)
    # print(f"    Saved to: {output_file}")
    
    return model


def load_pretrained_mobilenetv1(model, num_classes=100):
    """
    Load pretrained MobileNetV1 weights from torchvision into custom quantized model
    """
    print("Loading pretrained MobileNetV1 weights...")
    
    pretrained_model = models.mobilenet_v2(pretrained=True)  # Note: torchvision doesn't have v1, using v2
    print("Warning: Using MobileNetV2 pretrained weights for MobileNetV1 (limited compatibility)")
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    weight_mapping = {}
    
    weight_mapping.update({
        'features.0.0.weight': 'features.0.conv2d.weight',
        'features.0.1.weight': 'features.0.bn2d.weight',
        'features.0.1.bias': 'features.0.bn2d.bias',
        'features.0.1.running_mean': 'features.0.bn2d.running_mean',
        'features.0.1.running_var': 'features.0.bn2d.running_var',
        'features.0.1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    separable_layers = [
        (1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18), (7, 21), (8, 24), (9, 27), (10, 30), (11, 33), (12, 36)
    ]
    
    for custom_idx, pretrained_idx in separable_layers:
        if custom_idx <= 13:  # Limit to available layers
            # Depthwise conv
            weight_mapping[f'features.{pretrained_idx}.conv.0.0.weight'] = f'features.{custom_idx}.depthwise.conv2d.weight'
            weight_mapping[f'features.{pretrained_idx}.conv.0.1.weight'] = f'features.{custom_idx}.depthwise.bn2d.weight'
            weight_mapping[f'features.{pretrained_idx}.conv.0.1.bias'] = f'features.{custom_idx}.depthwise.bn2d.bias'
            weight_mapping[f'features.{pretrained_idx}.conv.0.1.running_mean'] = f'features.{custom_idx}.depthwise.bn2d.running_mean'
            weight_mapping[f'features.{pretrained_idx}.conv.0.1.running_var'] = f'features.{custom_idx}.depthwise.bn2d.running_var'
            weight_mapping[f'features.{pretrained_idx}.conv.0.1.num_batches_tracked'] = f'features.{custom_idx}.depthwise.bn2d.num_batches_tracked'
            
            # Pointwise conv
            weight_mapping[f'features.{pretrained_idx}.conv.1.weight'] = f'features.{custom_idx}.pointwise.conv2d.weight'
            weight_mapping[f'features.{pretrained_idx}.conv.2.weight'] = f'features.{custom_idx}.pointwise.bn2d.weight'
            weight_mapping[f'features.{pretrained_idx}.conv.2.bias'] = f'features.{custom_idx}.pointwise.bn2d.bias'
            weight_mapping[f'features.{pretrained_idx}.conv.2.running_mean'] = f'features.{custom_idx}.pointwise.bn2d.running_mean'
            weight_mapping[f'features.{pretrained_idx}.conv.2.running_var'] = f'features.{custom_idx}.pointwise.bn2d.running_var'
            weight_mapping[f'features.{pretrained_idx}.conv.2.num_batches_tracked'] = f'features.{custom_idx}.pointwise.bn2d.num_batches_tracked'
    
    # Skip classifier for now (will be handled separately)
    
    return _transfer_weights(model, pretrained_dict, custom_dict, weight_mapping, num_classes)


def load_pretrained_mobilenetv2(model, num_classes=100):
    """
    Load pretrained MobileNetV2 weights from torchvision into custom quantized model
    """
    print("Loading pretrained MobileNetV2 weights...")
    
    pretrained_model = models.mobilenet_v2(pretrained=True)
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    weight_mapping = {}
    
    # First conv layer
    weight_mapping.update({
        'features.0.0.weight': 'features.0.conv2d.weight',
        'features.0.1.weight': 'features.0.bn2d.weight',
        'features.0.1.bias': 'features.0.bn2d.bias',
        'features.0.1.running_mean': 'features.0.bn2d.running_mean',
        'features.0.1.running_var': 'features.0.bn2d.running_var',
        'features.0.1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    # Inverted residual blocks
    # MobileNetV2 has a complex structure, so we'll map what we can
    block_mapping = [
        (1, 1),   # First inverted residual
        (2, 2), (3, 3),   # Second stage
        (4, 4), (5, 5), (6, 6),   # Third stage  
        (7, 7), (8, 8), (9, 9), (10, 10),   # Fourth stage
        (11, 11), (12, 12), (13, 13),   # Fifth stage
        (14, 14), (15, 15), (16, 16),   # Sixth stage
        (17, 17),   # Seventh stage
    ]
    
    for pretrained_idx, custom_idx in block_mapping:
        if pretrained_idx < 18 and custom_idx < len(model.features):
            # Expansion conv (1x1)
            if hasattr(model.features[custom_idx], 'expansion'):
                weight_mapping[f'features.{pretrained_idx}.conv.0.weight'] = f'features.{custom_idx}.expansion.conv2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.1.weight'] = f'features.{custom_idx}.expansion.bn2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.1.bias'] = f'features.{custom_idx}.expansion.bn2d.bias'
                weight_mapping[f'features.{pretrained_idx}.conv.1.running_mean'] = f'features.{custom_idx}.expansion.bn2d.running_mean'
                weight_mapping[f'features.{pretrained_idx}.conv.1.running_var'] = f'features.{custom_idx}.expansion.bn2d.running_var'
                weight_mapping[f'features.{pretrained_idx}.conv.1.num_batches_tracked'] = f'features.{custom_idx}.expansion.bn2d.num_batches_tracked'
            
            # Depthwise conv (3x3)
            if hasattr(model.features[custom_idx], 'depthwise'):
                weight_mapping[f'features.{pretrained_idx}.conv.3.weight'] = f'features.{custom_idx}.depthwise.conv2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.4.weight'] = f'features.{custom_idx}.depthwise.bn2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.4.bias'] = f'features.{custom_idx}.depthwise.bn2d.bias'
                weight_mapping[f'features.{pretrained_idx}.conv.4.running_mean'] = f'features.{custom_idx}.depthwise.bn2d.running_mean'
                weight_mapping[f'features.{pretrained_idx}.conv.4.running_var'] = f'features.{custom_idx}.depthwise.bn2d.running_var'
                weight_mapping[f'features.{pretrained_idx}.conv.4.num_batches_tracked'] = f'features.{custom_idx}.depthwise.bn2d.num_batches_tracked'
            
            # Projection conv (1x1)
            if hasattr(model.features[custom_idx], 'projection'):
                weight_mapping[f'features.{pretrained_idx}.conv.6.weight'] = f'features.{custom_idx}.projection.conv2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.7.weight'] = f'features.{custom_idx}.projection.bn2d.weight'
                weight_mapping[f'features.{pretrained_idx}.conv.7.bias'] = f'features.{custom_idx}.projection.bn2d.bias'
                weight_mapping[f'features.{pretrained_idx}.conv.7.running_mean'] = f'features.{custom_idx}.projection.bn2d.running_mean'
                weight_mapping[f'features.{pretrained_idx}.conv.7.running_var'] = f'features.{custom_idx}.projection.bn2d.running_var'
                weight_mapping[f'features.{pretrained_idx}.conv.7.num_batches_tracked'] = f'features.{custom_idx}.projection.bn2d.num_batches_tracked'
    
    if len(model.features) > 18:
        weight_mapping.update({
            'features.18.0.weight': f'features.{len(model.features)-1}.conv2d.weight',
            'features.18.1.weight': f'features.{len(model.features)-1}.bn2d.weight',
            'features.18.1.bias': f'features.{len(model.features)-1}.bn2d.bias',
            'features.18.1.running_mean': f'features.{len(model.features)-1}.bn2d.running_mean',
            'features.18.1.running_var': f'features.{len(model.features)-1}.bn2d.running_var',
            'features.18.1.num_batches_tracked': f'features.{len(model.features)-1}.bn2d.num_batches_tracked',
        })
    
    # Classifier (if num_classes matches)
    if num_classes == 1000:
        weight_mapping.update({
            'classifier.1.weight': 'classifier.linear.weight',
            'classifier.1.bias': 'classifier.linear.bias',
        })
    
    return _transfer_weights(model, pretrained_dict, custom_dict, weight_mapping, num_classes)


def _transfer_weights(model, pretrained_dict, custom_dict, weight_mapping, num_classes):
    transferred = 0
    skipped = 0
    errors = 0
    for pretrained_key, custom_key in weight_mapping.items():
        if pretrained_key in pretrained_dict and custom_key in custom_dict:
            try:
                pretrained_param = pretrained_dict[pretrained_key]
                custom_param = custom_dict[custom_key]
                
                if pretrained_param.shape == custom_param.shape:
                    custom_dict[custom_key].copy_(pretrained_param)
                    transferred += 1
                else:
                    print(f"Shape mismatch {pretrained_key}: {pretrained_param.shape} vs {custom_key}: {custom_param.shape}")
                    errors += 1
            except Exception as e:
                print(f"Error transferring {pretrained_key} -> {custom_key}: {e}")
                errors += 1
        else:
            missing_key = pretrained_key if pretrained_key not in pretrained_dict else custom_key
            skipped += 1
    
    print(f"Weight transfer summary:")
    print(f"Transferred: {transferred}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    
    # Load the updated state dict
    model.load_state_dict(custom_dict)
    
    # Reset quantization-specific parameters
    print("Resetting quantization parameters...")
    for name, module in model.named_modules():
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked.data.fill_(0)
        if hasattr(module, 'running_min'):
            module.running_min.data.fill_(0.0)
        if hasattr(module, 'running_max'):
            module.running_max.data.fill_(0.0)
        # Reset BatchNorm stats
        if hasattr(module, 'bn2d'):
            module.bn2d.reset_running_stats()
    
    return model


def load_pretrained_mobilenetv3(model, variant="large", num_classes=100):
    """
    Load pretrained MobileNetV3 weights from torchvision into custom quantized model
    """
    print(f"Loading pretrained MobileNetV3-{variant} weights...")
    
    import torchvision.models as models
    
    # Load pretrained model from torchvision
    if variant == "large":
        pretrained_model = models.mobilenet_v3_large(pretrained=True)
    elif variant == "small":
        pretrained_model = models.mobilenet_v3_small(pretrained=True)
    else:
        raise ValueError(f"Unknown MobileNetV3 variant: {variant}")
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    weight_mapping = {}
    
    # First conv layer
    weight_mapping.update({
        'features.0.0.weight': 'features.0.conv2d.weight',
        'features.0.1.weight': 'features.0.bn2d.weight',
        'features.0.1.bias': 'features.0.bn2d.bias',
        'features.0.1.running_mean': 'features.0.bn2d.running_mean',
        'features.0.1.running_var': 'features.0.bn2d.running_var',
        'features.0.1.num_batches_tracked': 'features.0.bn2d.num_batches_tracked',
    })
    
    # Get the number of inverted residual blocks
    num_blocks = len([k for k in custom_dict.keys() if 'features.' in k and '.conv.' in k]) // 3
    
    # Map inverted residual blocks
    pretrained_block_idx = 1  # Start from features.1 in pretrained model
    custom_block_idx = 1      # Start from features.1 in custom model
    
    while pretrained_block_idx < 16 and custom_block_idx < len(model.features) - 1:  # -1 for last conv
        pretrained_prefix = f'features.{pretrained_block_idx}'
        custom_prefix = f'features.{custom_block_idx}'
        
        # Check if this is an inverted residual block in pretrained model
        if f'{pretrained_prefix}.block.0.0.weight' in pretrained_dict:
            # Expansion conv (if exists)
            if f'{pretrained_prefix}.block.0.0.weight' in pretrained_dict:
                weight_mapping.update({
                    f'{pretrained_prefix}.block.0.0.weight': f'{custom_prefix}.conv.0.conv2d.weight',
                    f'{pretrained_prefix}.block.0.1.weight': f'{custom_prefix}.conv.0.bn2d.weight',
                    f'{pretrained_prefix}.block.0.1.bias': f'{custom_prefix}.conv.0.bn2d.bias',
                    f'{pretrained_prefix}.block.0.1.running_mean': f'{custom_prefix}.conv.0.bn2d.running_mean',
                    f'{pretrained_prefix}.block.0.1.running_var': f'{custom_prefix}.conv.0.bn2d.running_var',
                    f'{pretrained_prefix}.block.0.1.num_batches_tracked': f'{custom_prefix}.conv.0.bn2d.num_batches_tracked',
                })
                conv_offset = 1
            else:
                conv_offset = 0
            
            # Depthwise conv
            weight_mapping.update({
                f'{pretrained_prefix}.block.{conv_offset}.0.weight': f'{custom_prefix}.conv.{conv_offset}.conv2d.weight',
                f'{pretrained_prefix}.block.{conv_offset}.1.weight': f'{custom_prefix}.conv.{conv_offset}.bn2d.weight',
                f'{pretrained_prefix}.block.{conv_offset}.1.bias': f'{custom_prefix}.conv.{conv_offset}.bn2d.bias',
                f'{pretrained_prefix}.block.{conv_offset}.1.running_mean': f'{custom_prefix}.conv.{conv_offset}.bn2d.running_mean',
                f'{pretrained_prefix}.block.{conv_offset}.1.running_var': f'{custom_prefix}.conv.{conv_offset}.bn2d.running_var',
                f'{pretrained_prefix}.block.{conv_offset}.1.num_batches_tracked': f'{custom_prefix}.conv.{conv_offset}.bn2d.num_batches_tracked',
            })
            
            # SE module (if exists) - skip for now as it's complex
            se_offset = 0
            if f'{pretrained_prefix}.block.{conv_offset+1}.fc1.weight' in pretrained_dict:
                se_offset = 1
                # Could map SE weights here if needed
            
            # Projection conv
            final_conv_idx = conv_offset + 1 + se_offset
            weight_mapping.update({
                f'{pretrained_prefix}.block.{final_conv_idx}.0.weight': f'{custom_prefix}.conv.{-1}.conv2d.weight',
                f'{pretrained_prefix}.block.{final_conv_idx}.1.weight': f'{custom_prefix}.conv.{-1}.bn2d.weight',
                f'{pretrained_prefix}.block.{final_conv_idx}.1.bias': f'{custom_prefix}.conv.{-1}.bn2d.bias',
                f'{pretrained_prefix}.block.{final_conv_idx}.1.running_mean': f'{custom_prefix}.conv.{-1}.bn2d.running_mean',
                f'{pretrained_prefix}.block.{final_conv_idx}.1.running_var': f'{custom_prefix}.conv.{-1}.bn2d.running_var',
                f'{pretrained_prefix}.block.{final_conv_idx}.1.num_batches_tracked': f'{custom_prefix}.conv.{-1}.bn2d.num_batches_tracked',
            })
        
        pretrained_block_idx += 1
        custom_block_idx += 1
    
    # Last conv layer
    last_features_idx = len(model.features) - 1
    if variant == "large":
        weight_mapping.update({
            'features.16.0.weight': f'features.{last_features_idx}.conv2d.weight',
            'features.16.1.weight': f'features.{last_features_idx}.bn2d.weight',
            'features.16.1.bias': f'features.{last_features_idx}.bn2d.bias',
            'features.16.1.running_mean': f'features.{last_features_idx}.bn2d.running_mean',
            'features.16.1.running_var': f'features.{last_features_idx}.bn2d.running_var',
            'features.16.1.num_batches_tracked': f'features.{last_features_idx}.bn2d.num_batches_tracked',
        })
    
    # Classifier
    if num_classes == 1000:
        if variant == "large":
            # MobileNetV3-Large has pre_classifier
            weight_mapping.update({
                'classifier.0.weight': 'pre_classifier.linear.weight',
                'classifier.0.bias': 'pre_classifier.linear.bias',
                'classifier.3.weight': 'classifier.linear.weight',
                'classifier.3.bias': 'classifier.linear.bias',
            })
        else:
            # MobileNetV3-Small direct classifier
            weight_mapping.update({
                'classifier.0.weight': 'classifier.linear.weight',
                'classifier.0.bias': 'classifier.linear.bias',
            })
    
    return _transfer_weights(model, pretrained_dict, custom_dict, weight_mapping, num_classes)


# Update the main load_pretrained_mobilenet function
def load_pretrained_mobilenet(model, mobilenet_version="v2", num_classes=100):
    if mobilenet_version == "v1":
        return load_pretrained_mobilenetv1(model, num_classes)
    elif mobilenet_version == "v2":
        return load_pretrained_mobilenetv2(model, num_classes)
    elif mobilenet_version == "v3_large":
        return load_pretrained_mobilenetv3(model, variant="large", num_classes=num_classes)
    elif mobilenet_version == "v3_small":
        return load_pretrained_mobilenetv3(model, variant="small", num_classes=num_classes)
    else:
        raise ValueError(f"Unknown MobileNet version: {mobilenet_version}")


def load_pretrained_vit(model, variant="small", num_classes=100, img_size=224):
    """
    Load pretrained ViT weights from timm into your quantized ViT model.
    
    Args:
        model: Your quantized ViT model
        variant: ViT variant ("tiny", "small", "base", "large")
        num_classes: Number of output classes
        img_size: Input image size
    
    Returns:
        model: Model with loaded pretrained weights
    """
    print(f"Loading pretrained ViT-{variant} weights...")
    
    # Map your variants to timm model names
    timm_model_names = {
        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224", 
        "base": "vit_base_patch16_224",
        "large": "vit_large_patch16_224"
    }
    
    if variant not in timm_model_names:
        raise ValueError(f"Unknown ViT variant: {variant}. Available: {list(timm_model_names.keys())}")
    
    # Load pretrained model from timm
    timm_model_name = timm_model_names[variant]
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True, img_size=img_size)
        print(f"Successfully loaded {timm_model_name} from timm")
    except Exception as e:
        print(f"Failed to load from timm: {e}")
        print("Falling back to random initialization")
        return model
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    
    # Create weight mapping between timm model and your quantized model
    weight_mapping = _create_vit_weight_mapping(pretrained_dict, custom_dict, variant)
    
    # Transfer weights
    transferred, skipped, errors = _transfer_vit_weights(
        pretrained_dict, custom_dict, weight_mapping, num_classes
    )
    
    print(f"Weight transfer summary:")
    print(f"  Transferred: {transferred}")
    print(f"  Skipped: {skipped}") 
    print(f"  Errors: {errors}")
    
    # Load the updated state dict
    model.load_state_dict(custom_dict)
    
    # Reset quantization-specific parameters
    _reset_vit_quantization_params(model)
    
    return model

def _create_vit_weight_mapping(pretrained_dict, custom_dict, variant):
    """
    FIXED: Create proper mapping between timm ViT and your quantized ViT
    
    The issue was: Your ViT uses UnifiedQuantizedLinear which creates:
    - blocks.0.attn.qkv.linear.weight (not blocks.0.attn.qkv.weight)
    """
    weight_mapping = {}
    
    # Debug: Print structure to understand the mismatch
    print("🔍 DEBUG: Sample timm keys:")
    timm_keys = [k for k in pretrained_dict.keys() if 'blocks.0.attn' in k][:5]
    for key in timm_keys:
        print(f"  timm: {key}")
    
    print("🔍 DEBUG: Sample custom keys:")
    custom_keys = [k for k in custom_dict.keys() if 'blocks.0.attn' in k][:5]
    for key in custom_keys:
        print(f"  custom: {key}")
    
    # 1. Patch embedding weights (FP32 - should work)
    weight_mapping.update({
        'patch_embed.proj.weight': 'patch_embed.projection.0.weight',
        'patch_embed.proj.bias': 'patch_embed.projection.0.bias',
    })
    
    # 2. Position and class token embeddings (FP32 - should work)
    weight_mapping.update({
        'pos_embed': 'pos_embed',
        'cls_token': 'cls_token',
    })
    
    # 3. Find number of transformer blocks in YOUR model
    num_blocks = len([k for k in custom_dict.keys() if k.startswith('blocks.') and '.norm1.weight' in k])
    print(f"🔍 Found {num_blocks} transformer blocks in custom model")
    
    for i in range(num_blocks):
        block_prefix = f'blocks.{i}'
        
        # Layer norms (FP32 - should work)
        weight_mapping.update({
            f'{block_prefix}.norm1.weight': f'{block_prefix}.norm1.weight',
            f'{block_prefix}.norm1.bias': f'{block_prefix}.norm1.bias',
            f'{block_prefix}.norm2.weight': f'{block_prefix}.norm2.weight', 
            f'{block_prefix}.norm2.bias': f'{block_prefix}.norm2.bias',
        })
        
        # FIXED: Check if your model uses UnifiedQuantizedLinear (has .linear submodule)
        qkv_linear_key = f'{block_prefix}.attn.qkv.linear.weight'
        qkv_direct_key = f'{block_prefix}.attn.qkv.weight'
        
        if qkv_linear_key in custom_dict:
            # Your model uses UnifiedQuantizedLinear - map to .linear submodule
            print(f"  ✅ Block {i}: Using UnifiedQuantizedLinear structure")
            weight_mapping.update({
                f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.linear.weight',
                f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.linear.bias',
                f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.linear.weight',
                f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.linear.bias',
            })
            
            # MLP layers with .linear submodule
            weight_mapping.update({
                f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.linear.weight',
                f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.linear.bias',
                f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.linear.weight',
                f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.linear.bias',
            })
            
        elif qkv_direct_key in custom_dict:
            # Your model uses regular nn.Linear - direct mapping
            print(f"  ⚠️  Block {i}: Using regular nn.Linear structure")
            weight_mapping.update({
                f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.weight',
                f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.bias',
                f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.weight',
                f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.bias',
            })
            
            # MLP layers direct
            weight_mapping.update({
                f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.weight',
                f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.bias',
                f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.weight',
                f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.bias',
            })
        else:
            print(f"  ❌ Block {i}: Cannot find QKV layer structure!")
            print(f"     Looking for: {qkv_linear_key} or {qkv_direct_key}")
    
    # 4. Final layer norm (FP32)
    weight_mapping.update({
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    })
    
    # 5. Classification head - check for .linear submodule
    if 'head.weight' in pretrained_dict:
        head_linear_key = 'head.linear.weight'
        head_direct_key = 'head.weight'
        
        if head_linear_key in custom_dict:
            # Quantized head
            weight_mapping.update({
                'head.weight': 'head.linear.weight',
                'head.bias': 'head.linear.bias',
            })
            print("  ✅ Head: Using UnifiedQuantizedLinear structure")
        elif head_direct_key in custom_dict:
            # FP32 head
            weight_mapping.update({
                'head.weight': 'head.weight', 
                'head.bias': 'head.bias',
            })
            print("  ✅ Head: Using regular nn.Linear structure")
    
    print(f"🔍 Created {len(weight_mapping)} weight mappings")
    return weight_mapping


def _transfer_vit_weights(pretrained_dict, custom_dict, weight_mapping, num_classes):
    """Transfer weights using the mapping."""
    transferred = 0
    skipped = 0
    errors = 0
    
    for pretrained_key, custom_key in weight_mapping.items():
        if pretrained_key in pretrained_dict and custom_key in custom_dict:
            try:
                pretrained_param = pretrained_dict[pretrained_key]
                custom_param = custom_dict[custom_key]
                
                # Handle shape mismatches (e.g., different number of classes)
                if pretrained_param.shape == custom_param.shape:
                    custom_dict[custom_key].copy_(pretrained_param)
                    transferred += 1
                elif 'head' in custom_key and num_classes != 1000:
                    # Skip classifier if different number of classes
                    print(f"Skipping classifier layer due to class mismatch: {pretrained_param.shape} vs {custom_param.shape}")
                    skipped += 1
                else:
                    print(f"Shape mismatch {pretrained_key}: {pretrained_param.shape} vs {custom_key}: {custom_param.shape}")
                    errors += 1
                    
            except Exception as e:
                print(f"Error transferring {pretrained_key} -> {custom_key}: {e}")
                errors += 1
        else:
            missing_key = pretrained_key if pretrained_key not in pretrained_dict else custom_key
            print(f"Missing key during transfer: {missing_key}")
            skipped += 1
    
    return transferred, skipped, errors


def _reset_vit_quantization_params(model):
    """Reset quantization-specific parameters for ViT."""
    print("Resetting quantization parameters for ViT...")
    
    for name, module in model.named_modules():
        # Reset quantization statistics
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked.data.fill_(0)
        if hasattr(module, 'running_min'):
            module.running_min.data.fill_(0.0)
        if hasattr(module, 'running_max'):
            module.running_max.data.fill_(0.0)
            
        # Reset any other quantization-specific buffers
        if hasattr(module, 'reset_stats'):
            module.reset_stats()


# Convenience function for different ViT variants
def load_pretrained_vit_tiny(model, **kwargs):
    return load_pretrained_vit(model, variant="tiny", **kwargs)

def load_pretrained_vit_small(model, **kwargs):
    return load_pretrained_vit(model, variant="small", **kwargs)

def load_pretrained_vit_base(model, **kwargs):
    return load_pretrained_vit(model, variant="base", **kwargs)

def load_pretrained_vit_large(model, **kwargs):
    return load_pretrained_vit(model, variant="large", **kwargs)


# Add to networks/load_pretrained.py

def load_pretrained_deit(model, variant="small", num_classes=100, img_size=224):
    """
    Load pretrained DeiT weights - EFFICIENT: Reuses ViT weight loading!
    
    DeiT weights from timm include the distillation token, so we can load them directly.
    """
    print(f"Loading pretrained DeiT-{variant} weights...")
    
    # Map DeiT variants to timm model names
    timm_model_names = {
        "tiny": "deit_tiny_patch16_224",
        "small": "deit_small_patch16_224", 
        "base": "deit_base_patch16_224",
    }
    
    if variant not in timm_model_names:
        raise ValueError(f"Unknown DeiT variant: {variant}. Available: {list(timm_model_names.keys())}")
    
    # Load pretrained DeiT model from timm
    timm_model_name = timm_model_names[variant]
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True, img_size=img_size)
        print(f"Successfully loaded {timm_model_name} from timm")
    except Exception as e:
        print(f"Failed to load DeiT from timm: {e}")
        print("Falling back to ViT weights (will miss distillation token)")
        # Fallback to ViT weights
        return load_pretrained_vit(model, variant, num_classes, img_size)
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()
    
    # Create weight mapping (reuses ViT mapping + distillation token)
    weight_mapping = _create_deit_weight_mapping(pretrained_dict, custom_dict, variant)
    
    # Transfer weights (reuses ViT transfer function)
    transferred, skipped, errors = _transfer_vit_weights(
        pretrained_dict, custom_dict, weight_mapping, num_classes
    )
    
    print(f"DeiT weight transfer summary:")
    print(f"  Transferred: {transferred}")
    print(f"  Skipped: {skipped}") 
    print(f"  Errors: {errors}")
    
    # Load the updated state dict
    model.load_state_dict(custom_dict)
    
    # Reset quantization parameters (reuses ViT function)
    _reset_vit_quantization_params(model)
    
    return model


def _create_deit_weight_mapping(pretrained_dict, custom_dict, variant):
    """
    Create DeiT weight mapping - EFFICIENT: Extends ViT mapping
    """
    # Start with ViT mapping (reuses existing function!)
    weight_mapping = _create_vit_weight_mapping(pretrained_dict, custom_dict, variant)
    
    # Add DeiT-specific mappings
    
    # 1. Distillation token
    weight_mapping.update({
        'dist_token': 'dist_token',
    })
    
    # 2. Position embeddings (already handled by ViT mapping, but DeiT has 2 tokens)
    # The pos_embed will be handled automatically since both have same key name
    
    # 3. Dual classification heads
    if 'head_dist.weight' in pretrained_dict:
        # Check if custom model has quantized heads
        if 'head_cls.linear.weight' in custom_dict:
            # Quantized heads
            weight_mapping.update({
                'head.weight': 'head_cls.linear.weight',      # Main classifier
                'head.bias': 'head_cls.linear.bias',
                'head_dist.weight': 'head_dist.linear.weight', # Distillation classifier  
                'head_dist.bias': 'head_dist.linear.bias',
            })
        else:
            # FP32 heads
            weight_mapping.update({
                'head.weight': 'head_cls.weight',       # Main classifier
                'head.bias': 'head_cls.bias', 
                'head_dist.weight': 'head_dist.weight', # Distillation classifier
                'head_dist.bias': 'head_dist.bias',
            })
    
    # Remove the old single head mapping if it exists
    if 'head.weight' in weight_mapping and 'head_cls.weight' in custom_dict:
        # We now have dual heads, so remove old single head mapping
        pass
    
    print(f"🔍 Created {len(weight_mapping)} DeiT weight mappings (including distillation token)")
    return weight_mapping


def _debug_actual_swin_structure(model):
    """Debug what blocks actually exist in the built model"""
    print("🔍 Actual model structure:")
    
    for layer_idx in range(4):
        layer = getattr(model, 'layers', None)
        if layer and len(layer) > layer_idx:
            actual_layer = layer[layer_idx]
            num_blocks = len(actual_layer.blocks) if hasattr(actual_layer, 'blocks') else 0
            print(f"  Layer {layer_idx}: {num_blocks} blocks")
        else:
            print(f"  Layer {layer_idx}: NOT FOUND")
    
    # Also check state dict keys
    state_dict_blocks = {}
    for key in model.state_dict().keys():
        if 'layers.' in key and '.blocks.' in key:
            parts = key.split('.')
            if len(parts) >= 4:
                layer_idx = int(parts[1])
                block_idx = int(parts[3])
                if layer_idx not in state_dict_blocks:
                    state_dict_blocks[layer_idx] = set()
                state_dict_blocks[layer_idx].add(block_idx)
    
    print("🔍 State dict structure:")
    for layer_idx in sorted(state_dict_blocks.keys()):
        max_block = max(state_dict_blocks[layer_idx])
        print(f"  Layer {layer_idx}: Blocks 0-{max_block} ({max_block + 1} total)")


def load_pretrained_swin(model, variant="tiny", num_classes=100, img_size=224):
    """
    Load pretrained Swin weights from timm into your quantized Swin model
    """
    print(f"Loading pretrained Swin-{variant} weights...")

    _debug_actual_swin_structure(model)

    
    # Map variants to timm model names
    timm_model_names = {
        "tiny": "swin_tiny_patch4_window7_224",
        "small": "swin_small_patch4_window7_224", 
        "base": "swin_base_patch4_window7_224"
    }
    
    if variant not in timm_model_names:
        raise ValueError(f"Unknown Swin variant: {variant}. Available: {list(timm_model_names.keys())}")
    
    # Load pretrained model from timm
    timm_model_name = timm_model_names[variant]
    try:
        pretrained_model = timm.create_model(timm_model_name, pretrained=True, img_size=img_size)
        print(f"Successfully loaded {timm_model_name} from timm")
    except Exception as e:
        print(f"Failed to load Swin from timm: {e}")
        print("Falling back to random initialization")
        return model
    
    pretrained_dict = pretrained_model.state_dict()
    custom_dict = model.state_dict()

    print("🔍 Timm Swin downsample dimensions:")
    for key, tensor in pretrained_dict.items():
        if 'downsample.reduction.weight' in key:
            print(f"  {key}: {tensor.shape}")
        if 'downsample.norm.weight' in key:
            print(f"  {key}: {tensor.shape}")
    
    
    # Create weight mapping
    weight_mapping = _create_swin_weight_mapping(pretrained_dict, custom_dict, variant)

    transferred, skipped, errors = _transfer_swin_weights(
        pretrained_dict, custom_dict, weight_mapping, num_classes
    )
    
    print(f"Swin weight transfer summary:")
    print(f"  Transferred: {transferred}")
    print(f"  Skipped: {skipped}") 
    print(f"  Errors: {errors}")
    
    # Load the updated state dict
    model.load_state_dict(custom_dict)
    
    # Reset quantization parameters
    _reset_swin_quantization_params(model)
    
    return model


def _create_swin_weight_mapping(pretrained_dict, custom_dict, variant):
    weight_mapping = {}
    
    # 1. Patch embedding - FIX THE NAMING
    if 'patch_embed.proj.weight' in pretrained_dict:
        if 'patch_embed.projection.weight' in custom_dict:
            weight_mapping.update({
                'patch_embed.proj.weight': 'patch_embed.projection.weight',
                'patch_embed.proj.bias': 'patch_embed.projection.bias',
            })
        elif 'patch_embed.projection.0.weight' in custom_dict:
            # Alternative naming structure
            weight_mapping.update({
                'patch_embed.proj.weight': 'patch_embed.projection.0.weight',
                'patch_embed.proj.bias': 'patch_embed.projection.0.bias',
            })
    
    # 2. Use CORRECT Swin-Tiny depths
    correct_depths = [2, 2, 6, 2]  # Standard Swin-Tiny
    
    print(f"🔧 Using correct Swin-Tiny depths: {correct_depths}")
    
    # 3. Map each layer with EXACT block counts
    for layer_idx in range(4):
        layer_prefix = f'layers.{layer_idx}'
        num_blocks = correct_depths[layer_idx]  # Use known architecture
        
        print(f"🔧 Layer {layer_idx}: Mapping {num_blocks} blocks (not detecting)")
        
        for block_idx in range(num_blocks):
            block_prefix = f'{layer_prefix}.blocks.{block_idx}'
            
            # Check if this block actually exists in custom model
            test_key = f'{block_prefix}.norm1.weight'
            if test_key not in custom_dict:
                print(f"⚠️  Skipping {block_prefix} - doesn't exist in custom model")
                continue
            
            # Layer norms (always FP32)
            weight_mapping.update({
                f'{block_prefix}.norm1.weight': f'{block_prefix}.norm1.weight',
                f'{block_prefix}.norm1.bias': f'{block_prefix}.norm1.bias',
                f'{block_prefix}.norm2.weight': f'{block_prefix}.norm2.weight',
                f'{block_prefix}.norm2.bias': f'{block_prefix}.norm2.bias',
            })
            
            # FIXED: Check for quantized vs regular attention
            if f'{block_prefix}.attn.qkv.linear.weight' in custom_dict:
                # Quantized attention layers
                weight_mapping.update({
                    f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.linear.weight',
                    f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.linear.bias',
                    f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.linear.weight',
                    f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.linear.bias',
                })
                
                # Quantized MLP layers
                weight_mapping.update({
                    f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.linear.weight',
                    f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.linear.bias',
                    f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.linear.weight',
                    f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.linear.bias',
                })
            elif f'{block_prefix}.attn.qkv.weight' in custom_dict:
                # Regular attention layers
                weight_mapping.update({
                    f'{block_prefix}.attn.qkv.weight': f'{block_prefix}.attn.qkv.weight',
                    f'{block_prefix}.attn.qkv.bias': f'{block_prefix}.attn.qkv.bias',
                    f'{block_prefix}.attn.proj.weight': f'{block_prefix}.attn.proj.weight',
                    f'{block_prefix}.attn.proj.bias': f'{block_prefix}.attn.proj.bias',
                    f'{block_prefix}.mlp.fc1.weight': f'{block_prefix}.mlp.fc1.weight',
                    f'{block_prefix}.mlp.fc1.bias': f'{block_prefix}.mlp.fc1.bias',
                    f'{block_prefix}.mlp.fc2.weight': f'{block_prefix}.mlp.fc2.weight',
                    f'{block_prefix}.mlp.fc2.bias': f'{block_prefix}.mlp.fc2.bias',
                })
    
    # 4. FIXED: Downsample mapping with correct indexing
    # Timm has downsample layers at indices [1, 2, 3] (no layer 0 downsample)
    # Your model has downsample layers at indices [0, 1, 2]
    downsample_mapping = {
        0: 1,  # Your Layer 0 downsample → timm Layer 1 downsample  
        1: 2,  # Your Layer 1 downsample → timm Layer 2 downsample
        2: 3   # Your Layer 2 downsample → timm Layer 3 downsample
    }
    
    for custom_layer_idx in range(3):  # Your layers 0,1,2 have downsample
        timm_layer_idx = downsample_mapping[custom_layer_idx]
        
        custom_prefix = f'layers.{custom_layer_idx}.downsample'
        timm_prefix = f'layers.{timm_layer_idx}.downsample'
        
        # Check if downsample exists in both
        if f'{timm_prefix}.norm.weight' in pretrained_dict and f'{custom_prefix}.norm.weight' in custom_dict:
            
            print(f"🔧 Mapping downsample: Your Layer {custom_layer_idx} ← Timm Layer {timm_layer_idx}")
            
            # Norm layer
            weight_mapping.update({
                f'{timm_prefix}.norm.weight': f'{custom_prefix}.norm.weight',
                f'{timm_prefix}.norm.bias': f'{custom_prefix}.norm.bias',
            })
            
            # Reduction layer
            if f'{custom_prefix}.reduction.linear.weight' in custom_dict:
                # Quantized reduction
                weight_mapping.update({
                    f'{timm_prefix}.reduction.weight': f'{custom_prefix}.reduction.linear.weight',
                    f'{timm_prefix}.reduction.bias': f'{custom_prefix}.reduction.linear.bias',
                })
            elif f'{custom_prefix}.reduction.weight' in custom_dict:
                # Regular reduction
                weight_mapping.update({
                    f'{timm_prefix}.reduction.weight': f'{custom_prefix}.reduction.weight',
                    f'{timm_prefix}.reduction.bias': f'{custom_prefix}.reduction.bias',
                })
        else:
            print(f"⚠️  Skipping downsample mapping for layer {custom_layer_idx} - keys not found")
    
    # 5. Final norm
    weight_mapping.update({
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    })
    
    # 6. Classification head
    if 'head.weight' in pretrained_dict:
        if 'head.linear.weight' in custom_dict:
            weight_mapping.update({
                'head.weight': 'head.linear.weight',
                'head.bias': 'head.linear.bias',
            })
        elif 'head.weight' in custom_dict:
            weight_mapping.update({
                'head.weight': 'head.weight',
                'head.bias': 'head.bias',
            })
    
    print(f"🔍 Created {len(weight_mapping)} Swin weight mappings")
    return weight_mapping


def _transfer_swin_weights(pretrained_dict, custom_dict, weight_mapping, num_classes):
    """Transfer Swin weights (reuses your existing logic)"""
    transferred = 0
    skipped = 0
    errors = 0
    
    for pretrained_key, custom_key in weight_mapping.items():
        if pretrained_key in pretrained_dict and custom_key in custom_dict:
            try:
                pretrained_param = pretrained_dict[pretrained_key]
                custom_param = custom_dict[custom_key]
                
                if pretrained_param.shape == custom_param.shape:
                    custom_dict[custom_key].copy_(pretrained_param)
                    transferred += 1
                elif 'head' in custom_key and num_classes != 1000:
                    # Skip classifier if different number of classes
                    print(f"Skipping classifier layer due to class mismatch: {pretrained_param.shape} vs {custom_param.shape}")
                    skipped += 1
                else:
                    print(f"Shape mismatch {pretrained_key}: {pretrained_param.shape} vs {custom_key}: {custom_param.shape}")
                    errors += 1
                    
            except Exception as e:
                print(f"Error transferring {pretrained_key} -> {custom_key}: {e}")
                errors += 1
        else:
            missing_key = pretrained_key if pretrained_key not in pretrained_dict else custom_key
            print(f"Missing key during transfer: {missing_key}")
            skipped += 1
    
    return transferred, skipped, errors


def _reset_swin_quantization_params(model):
    """Reset quantization parameters for Swin (reuses your existing logic)"""
    print("Resetting quantization parameters for Swin...")
    
    for name, module in model.named_modules():
        # Reset quantization statistics
        if hasattr(module, 'num_batches_tracked'):
            module.num_batches_tracked.data.fill_(0)
        if hasattr(module, 'running_min'):
            module.running_min.data.fill_(0.0)
        if hasattr(module, 'running_max'):
            module.running_max.data.fill_(0.0)
            
        # Reset any other quantization-specific buffers
        if hasattr(module, 'reset_stats'):
            module.reset_stats()