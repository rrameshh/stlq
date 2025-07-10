import torch
import torchvision.models as models
from networks.unified_resnet import resnet18



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


def load_pretrained_mobilenet(model, mobilenet_version="v2", num_classes=100):
    if mobilenet_version == "v1":
        return load_pretrained_mobilenetv1(model, num_classes)
    elif mobilenet_version == "v2":
        return load_pretrained_mobilenetv2(model, num_classes)
    else:
        raise ValueError(f"Unknown MobileNet version: {mobilenet_version}")


