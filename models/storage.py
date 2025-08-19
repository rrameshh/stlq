# models/storage.py - Custom format for storing quantized models
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
from datetime import datetime

from quantization.tensors.linear import LinearQuantizedTensor
from quantization.tensors.new_log import LogQuantizedTensor
from quantization.quant_config import QuantizationConfig


class QuantizedModelStorage:
    """Custom storage format for quantized models with maximum compression"""
    
    @staticmethod
    def extract_quantized_state(model) -> Dict[str, Any]:
        """Extract quantized representation from all layers"""
        quantized_state = {}
        total_params = 0
        quantized_params = 0
        
        for name, module in model.named_modules():
            layer_data = QuantizedModelStorage._extract_layer_data(module)
            if layer_data:
                quantized_state[name] = layer_data
                total_params += layer_data.get('param_count', 0)
                quantized_params += 1
        
        return {
            'layers': quantized_state,
            'stats': {
                'total_quantized_layers': quantized_params,
                'total_parameters': total_params,
                'extraction_timestamp': datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def _extract_layer_data(module) -> Optional[Dict[str, Any]]:
        """Extract quantized data from a single layer"""
        
        # Linear layers (QLinear)
        if hasattr(module, 'linear') and hasattr(module, 'strategy'):
            weight = module.linear.weight
            bias = module.linear.bias
            
            # Get quantized representation
            try:
                quantized_weight = module.strategy.quantize_weight(weight, per_channel=False)
                layer_data = {
                    'type': 'linear',
                    'shape': list(weight.shape),
                    'param_count': weight.numel(),
                    'has_bias': bias is not None,
                    'quantization_method': module.config.method,
                    'weight_data': QuantizedModelStorage._serialize_quantized_tensor(quantized_weight)
                }
                
                if bias is not None:
                    layer_data['bias_data'] = bias.detach().cpu().numpy().astype(np.float32)
                
                return layer_data
                
            except Exception as e:
                print(f"Warning: Could not quantize layer {type(module).__name__}: {e}")
                return None
        
        # Convolutional layers (QConv2dBNRelu, QConvBNUnfused)
        elif hasattr(module, 'conv2d') and hasattr(module, 'strategy'):
            weight = module.conv2d.weight
            bias = module.conv2d.bias
            
            try:
                quantized_weight = module.strategy.quantize_weight(weight, per_channel=True)
                layer_data = {
                    'type': 'conv2d',
                    'shape': list(weight.shape),
                    'param_count': weight.numel(),
                    'has_bias': bias is not None,
                    'quantization_method': module.config.method,
                    'weight_data': QuantizedModelStorage._serialize_quantized_tensor(quantized_weight)
                }
                
                if bias is not None:
                    layer_data['bias_data'] = bias.detach().cpu().numpy().astype(np.float32)
                
                # Store BN parameters if they exist
                if hasattr(module, 'bn2d'):
                    layer_data['bn_data'] = {
                        'weight': module.bn2d.weight.detach().cpu().numpy().astype(np.float32),
                        'bias': module.bn2d.bias.detach().cpu().numpy().astype(np.float32),
                        'running_mean': module.bn2d.running_mean.detach().cpu().numpy().astype(np.float32),
                        'running_var': module.bn2d.running_var.detach().cpu().numpy().astype(np.float32),
                        'eps': module.bn2d.eps,
                        'momentum': module.bn2d.momentum
                    }
                
                return layer_data
                
            except Exception as e:
                print(f"Warning: Could not quantize conv layer {type(module).__name__}: {e}")
                return None
        
        return None
    
    @staticmethod
    def _serialize_quantized_tensor(qtensor) -> Dict[str, Any]:
        """Serialize quantized tensor to storage format"""
        
        if isinstance(qtensor, LinearQuantizedTensor):
            return {
                'tensor_type': 'linear',
                'q': qtensor.q.detach().cpu().numpy(),
                's': qtensor.s.detach().cpu().numpy().astype(np.float32),
                'z': qtensor.z.detach().cpu().numpy(),
            }
        
        elif isinstance(qtensor, LogQuantizedTensor):
            data = {
                'tensor_type': 'log',
                'q1': qtensor.q1.detach().cpu().numpy(),
                'a': qtensor.a.detach().cpu().numpy().astype(np.float32),
                's': qtensor.s.detach().cpu().numpy(),
            }
            
            # Optional second word data
            if qtensor.q2 is not None:
                data['q2'] = qtensor.q2.detach().cpu().numpy()
                data['s_err'] = qtensor.s_err.detach().cpu().numpy()
                data['second_word_mask'] = qtensor.second_word_mask.detach().cpu().numpy()
            
            return data
        
        else:
            raise ValueError(f"Unknown quantized tensor type: {type(qtensor)}")
    
    @staticmethod
    def _deserialize_quantized_tensor(data: Dict[str, Any], device: str = 'cpu'):
        """Recreate quantized tensor from storage format"""
        
        if data['tensor_type'] == 'linear':
            q = torch.from_numpy(data['q']).to(device)
            s = torch.from_numpy(data['s']).to(device)
            z = torch.from_numpy(data['z']).to(device)
            return LinearQuantizedTensor(q, s, z)
        
        elif data['tensor_type'] == 'log':
            q1 = torch.from_numpy(data['q1']).to(device)
            a = torch.from_numpy(data['a']).to(device)
            s = torch.from_numpy(data['s']).to(device)
            
            q2 = None
            s_err = None
            second_word_mask = None
            
            if 'q2' in data:
                q2 = torch.from_numpy(data['q2']).to(device)
                s_err = torch.from_numpy(data['s_err']).to(device)
                second_word_mask = torch.from_numpy(data['second_word_mask']).to(device)
            
            return LogQuantizedTensor(q1, a, s, q2, s_err, second_word_mask)
        
        else:
            raise ValueError(f"Unknown tensor type: {data['tensor_type']}")


def _create_model_from_metadata(save_data, device):
    """Create model from saved metadata"""
    model_info = save_data['model_info']
    quant_config = save_data['quantization_config']
    
    # Reconstruct config
    from config import Config
    config = Config()
    config.model.name = model_info['name']
    config.model.num_classes = model_info['num_classes']
    if model_info['img_size']:
        config.model.img_size = model_info['img_size']
    
    config.quantization.method = quant_config['method']
    config.quantization.bits = quant_config['bits']
    config.quantization.momentum = quant_config['momentum']
    config.quantization.threshold = quant_config['threshold']
    config.system.device = device
    
    # Create model
    from models import create_model
    model = create_model(model_info['name'], config)
    return model.to(device)


def _restore_model_metadata(model, save_data):
    """Restore training metadata to model"""
    if 'training_metadata' in save_data:
        metadata = save_data['training_metadata']
        model._best_accuracy = metadata.get('best_accuracy')
        model._epochs_trained = metadata.get('epochs_trained')
    
    if 'sparsification_info' in save_data:
        model._sparsification_results = save_data['sparsification_info']
        model._sparsified = True


def _load_quantized_weights(model, quantized_state, device):
    """Load quantized weights into model"""
    layers_data = quantized_state['layers']
    
    for name, module in model.named_modules():
        if name in layers_data:
            layer_data = layers_data[name]
            
            try:
                # Reconstruct quantized tensor
                qtensor = QuantizedModelStorage._deserialize_quantized_tensor(
                    layer_data['weight_data'], device
                )
                
                # Get dequantized weights
                dequantized_weight = qtensor.dequantize()
                
                # Load into appropriate weight tensor
                if hasattr(module, 'linear'):
                    module.linear.weight.data = dequantized_weight.reshape(module.linear.weight.shape)
                    if layer_data['has_bias'] and 'bias_data' in layer_data:
                        bias_data = torch.from_numpy(layer_data['bias_data']).to(device)
                        module.linear.bias.data = bias_data
                        
                elif hasattr(module, 'conv2d'):
                    module.conv2d.weight.data = dequantized_weight.reshape(module.conv2d.weight.shape)
                    if layer_data['has_bias'] and 'bias_data' in layer_data:
                        bias_data = torch.from_numpy(layer_data['bias_data']).to(device)
                        module.conv2d.bias.data = bias_data
                    
                    # Load BN parameters if they exist
                    if 'bn_data' in layer_data and hasattr(module, 'bn2d'):
                        bn_data = layer_data['bn_data']
                        module.bn2d.weight.data = torch.from_numpy(bn_data['weight']).to(device)
                        module.bn2d.bias.data = torch.from_numpy(bn_data['bias']).to(device)
                        module.bn2d.running_mean.data = torch.from_numpy(bn_data['running_mean']).to(device)
                        module.bn2d.running_var.data = torch.from_numpy(bn_data['running_var']).to(device)
                        module.bn2d.eps = bn_data['eps']
                        module.bn2d.momentum = bn_data['momentum']
                
            except Exception as e:
                print(f"Warning: Failed to load layer {name}: {e}")