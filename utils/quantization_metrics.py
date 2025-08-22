import torch
from quantization.tensors.new_log import LogQuantizedTensor

def compute_second_word_ratio(model):

    stats = {
        'total_weights': 0,
        'second_word_weights': 0,
        'layer_stats': {},
        'overall_ratio': 0.0
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'strategy') and hasattr(module.strategy, 'quantize_weight'):
            if hasattr(module, 'linear'):  # UnifiedQuantizedLinear
                weight = module.linear.weight
            elif hasattr(module, 'conv2d'):  # UnifiedQuantizedConv
                weight = module.conv2d.weight
            else:
                continue
                
            try:
                quantized_weight = module.strategy.quantize_weight(weight, per_channel=True)
                if isinstance(quantized_weight, LogQuantizedTensor):
                    layer_stats = analyze_log_quantized_tensor(quantized_weight, name)
                    stats['layer_stats'][name] = layer_stats
                    stats['total_weights'] += layer_stats['total_weights']
                    stats['second_word_weights'] += layer_stats['second_word_weights']
                    
            except Exception as e:
                print(f"Warning: Could not analyze layer {name}: {e}")
                continue
    if stats['total_weights'] > 0:
        stats['overall_ratio'] = stats['second_word_weights'] / stats['total_weights']
    
    return stats


def analyze_log_quantized_tensor(log_tensor, layer_name):
    stats = {
        'layer_name': layer_name,
        'total_weights': log_tensor.q1.numel(),
        'second_word_weights': 0,
        'zero_weights': (log_tensor.q1 == 0).sum().item(),
        'non_zero_weights': (log_tensor.q1 > 0).sum().item(),
        'ratio': 0.0
    }
    
    if log_tensor.second_word_mask is not None:
        stats['second_word_weights'] = log_tensor.second_word_mask.sum().item()
        if stats['total_weights'] > 0:
            stats['ratio'] = stats['second_word_weights'] / stats['total_weights']
    
    return stats

def log_quantization_stats_to_tensorboard(writer, model, epoch):

    stats = compute_second_word_ratio(model)
    writer.add_scalar('Quantization/SecondWordRatio_Overall', stats['overall_ratio'], epoch)
    writer.add_scalar('Quantization/TotalWeights', stats['total_weights'], epoch)
    writer.add_scalar('Quantization/SecondWordWeights', stats['second_word_weights'], epoch)
    
    # Log per-layer statistics (for detailed analysis)
    layer_ratios = []
    for layer_name, layer_stats in stats['layer_stats'].items():
        clean_name = layer_name.replace('.', '/')
        writer.add_scalar(f'Quantization/LayerRatio/{clean_name}', 
                         layer_stats['ratio'], epoch)
        layer_ratios.append(layer_stats['ratio'])
    
    if layer_ratios:
        layer_ratios_tensor = torch.tensor(layer_ratios)
        writer.add_scalar('Quantization/SecondWordRatio_Mean', layer_ratios_tensor.mean().item(), epoch)
        writer.add_scalar('Quantization/SecondWordRatio_Std', layer_ratios_tensor.std().item(), epoch)
        writer.add_scalar('Quantization/SecondWordRatio_Max', layer_ratios_tensor.max().item(), epoch)
        writer.add_scalar('Quantization/SecondWordRatio_Min', layer_ratios_tensor.min().item(), epoch)
    
    print(f"\nQuantization Stats (Epoch {epoch+1}):")
    print(f"   Overall second-word ratio: {stats['overall_ratio']:.4f}")
    print(f"   Total weights: {stats['total_weights']:,}")
    print(f"   Second-word weights: {stats['second_word_weights']:,}")
    
    if layer_ratios:
        print(f"   Layer ratios - Mean: {layer_ratios_tensor.mean():.4f}, "
              f"Std: {layer_ratios_tensor.std():.4f}, "
              f"Range: [{layer_ratios_tensor.min():.4f}, {layer_ratios_tensor.max():.4f}]")

def print_quick_second_word_stats(model, epoch):
    try:
        stats = compute_second_word_ratio(model)
        ratio = stats['overall_ratio']
        total = stats['total_weights']
        second = stats['second_word_weights']
        
        print(f"   Second-word ratio: {ratio:.4f} ({second:,}/{total:,})")
    except Exception as e:
        print(f"   Could not compute second-word ratio: {e}")

def get_quantization_summary(model): # activation only
    summary = {
        'total_layers': 0,
        'quantized_layers': 0,
        'enabled_layers': 0,
        'method': 'unknown'
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'activation_quantization'):
            summary['total_layers'] += 1
            summary['quantized_layers'] += 1
            
            if module.activation_quantization:
                summary['enabled_layers'] += 1
                
            # Get quantization method from config
            if hasattr(module, 'config') and hasattr(module.config, 'method'):
                summary['method'] = module.config.method
    
    return summary