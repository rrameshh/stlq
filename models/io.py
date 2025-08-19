# models/io.py - Save/Load interface for quantized models
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .storage import (
    QuantizedModelStorage, 
    _create_model_from_metadata, 
    _restore_model_metadata, 
    _load_quantized_weights
)


def save_quantized_model(model, config, save_path: str, include_metadata: bool = True):
    """
    Save model as .pth file with both quantized and FP32 representations
    
    Args:
        model: Trained quantized model
        config: Training configuration  
        save_path: Path to save (will create both _quantized.pth and _fp32.pth)
        include_metadata: Whether to include training metadata
    
    Returns:
        Tuple of (quantized_path, fp32_path)
    """
    save_path = Path(save_path)
    base_name = save_path.stem
    save_dir = save_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving quantized model to {save_dir}/{base_name}_*.pth...")
    
    # Extract quantized state
    quantized_state = QuantizedModelStorage.extract_quantized_state(model)
    
    # Prepare common metadata
    common_metadata = {
        'format_version': '1.0',
        'model_info': {
            'name': config.model.name,
            'num_classes': config.model.num_classes,
            'img_size': getattr(config.model, 'img_size', None),
            'pretrained_source': 'imagenet' if config.model.pretrained else None
        },
        'quantization_config': {
            'method': config.quantization.method,
            'bits': config.quantization.bits,
            'momentum': config.quantization.momentum,
            'threshold': config.quantization.threshold,
            'adaptive_threshold': getattr(config.quantization, 'adaptive_threshold', False),
            'target_second_word_ratio': getattr(config.quantization, 'target_second_word_ratio', None)
        }
    }
    
    # Add training metadata if requested
    if include_metadata:
        common_metadata['training_metadata'] = {
            'dataset': config.data.dataset,
            'epochs_trained': getattr(model, '_epochs_trained', 'unknown'),
            'best_accuracy': getattr(model, '_best_accuracy', 'unknown'),
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Add sparsification info if available
        if hasattr(model, '_sparsification_results'):
            common_metadata['sparsification_info'] = model._sparsification_results
    
    # Save quantized version
    quantized_path = save_dir / f"{base_name}_quantized.pth"
    quantized_data = {
        **common_metadata,
        'model_type': 'quantized',
        'quantized_state': quantized_state,
    }
    torch.save(quantized_data, quantized_path)
    
    # Save FP32 version for compatibility
    fp32_path = save_dir / f"{base_name}_fp32.pth"
    fp32_data = {
        **common_metadata,
        'model_type': 'fp32',
        'state_dict': model.state_dict(),
    }
    torch.save(fp32_data, fp32_path)
    
    # Calculate compression stats
    quantized_size_mb = quantized_path.stat().st_size / (1024 * 1024)
    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    compression_ratio = fp32_size_mb / quantized_size_mb
    total_params = quantized_state['stats']['total_parameters']
    
    print(f"✅ Saved quantized model:")
    print(f"   Quantized: {quantized_path.name} ({quantized_size_mb:.2f} MB)")
    print(f"   FP32: {fp32_path.name} ({fp32_size_mb:.2f} MB)")
    print(f"   Compression: {compression_ratio:.1f}x")
    print(f"   Parameters: {total_params:,}")
    print(f"   Quantized layers: {quantized_state['stats']['total_quantized_layers']}")
    
    return quantized_path, fp32_path


def load_model(load_path: str, device: str = 'auto', prefer_quantized: bool = True):
    """
    Universal model loader - automatically detects quantized vs FP32
    
    Args:
        load_path: Path to model file (.pth)
        device: Device to load to ('auto', 'cpu', 'cuda', etc.)
        prefer_quantized: If both formats exist, prefer quantized version
        
    Returns:
        Loaded model
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    load_path = Path(load_path)
    
    # Handle different path scenarios
    if '_quantized.pth' in str(load_path):
        # Direct quantized path
        return load_quantized_model(load_path, device)
    elif '_fp32.pth' in str(load_path):
        # Direct FP32 path
        return load_fp32_model(load_path, device)
    else:
        # Base path - try to find best version
        base_path = load_path.parent / load_path.stem
        quantized_path = Path(str(base_path) + '_quantized.pth')
        fp32_path = Path(str(base_path) + '_fp32.pth')
        
        if prefer_quantized and quantized_path.exists():
            print(f"Found quantized version, loading: {quantized_path.name}")
            return load_quantized_model(quantized_path, device)
        elif fp32_path.exists():
            print(f"Loading FP32 version: {fp32_path.name}")
            return load_fp32_model(fp32_path, device)
        elif load_path.exists():
            # Try to load as regular PyTorch model
            print(f"Loading as regular PyTorch model: {load_path.name}")
            return torch.load(load_path, map_location=device)
        else:
            raise FileNotFoundError(f"No model found at {load_path}")


def load_quantized_model(load_path: str, device: str = 'auto'):
    """Load model from quantized .pth format"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading quantized model from {load_path}...")
    
    # Load data
    save_data = torch.load(load_path, map_location=device)
    
    # Validate format
    if save_data.get('model_type') != 'quantized':
        raise ValueError(f"File is not a quantized model: {load_path}")
    
    # Extract info and create model
    model = _create_model_from_metadata(save_data, device)
    
    # Load quantized weights
    _load_quantized_weights(model, save_data['quantized_state'], device)
    
    # Restore metadata
    _restore_model_metadata(model, save_data)
    
    print(f"✅ Loaded quantized model successfully")
    return model


def load_fp32_model(load_path: str, device: str = 'auto'):
    """Load model from FP32 .pth format"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading FP32 model from {load_path}...")
    
    # Load data
    save_data = torch.load(load_path, map_location=device)
    
    # Check if it's our format or regular PyTorch
    if save_data.get('model_type') == 'fp32':
        # Our format
        model = _create_model_from_metadata(save_data, device)
        model.load_state_dict(save_data['state_dict'])
        _restore_model_metadata(model, save_data)
    else:
        # Regular PyTorch model
        if isinstance(save_data, dict) and 'state_dict' in save_data:
            # Checkpoint format
            raise NotImplementedError("Please provide model architecture info for checkpoint loading")
        else:
            # Direct state dict
            model = save_data
    
    print(f"✅ Loaded FP32 model successfully")
    return model


def get_model_info(load_path: str) -> Dict[str, Any]:
    """Get model information without loading the full model"""
    save_data = torch.load(load_path, map_location='cpu')
    
    if save_data.get('model_type') == 'quantized':
        info = {
            'model_type': 'quantized',
            'model_name': save_data['model_info']['name'],
            'num_classes': save_data['model_info']['num_classes'],
            'quantization_method': save_data['quantization_config']['method'],
            'bits': save_data['quantization_config']['bits'],
            'total_parameters': save_data['quantized_state']['stats']['total_parameters'],
            'quantized_layers': save_data['quantized_state']['stats']['total_quantized_layers'],
            'file_size_mb': Path(load_path).stat().st_size / (1024 * 1024)
        }
    elif save_data.get('model_type') == 'fp32':
        # Count parameters from state dict
        total_params = sum(p.numel() for p in save_data['state_dict'].values() if p.dim() > 0)
        info = {
            'model_type': 'fp32',
            'model_name': save_data['model_info']['name'],
            'num_classes': save_data['model_info']['num_classes'],
            'total_parameters': total_params,
            'file_size_mb': Path(load_path).stat().st_size / (1024 * 1024)
        }
    else:
        # Regular PyTorch model - limited info
        if isinstance(save_data, dict) and 'model' in save_data:
            # Checkpoint format
            total_params = sum(p.numel() for p in save_data['model'].values() if hasattr(p, 'numel'))
        else:
            total_params = "unknown"
        
        info = {
            'model_type': 'pytorch_native',
            'total_parameters': total_params,
            'file_size_mb': Path(load_path).stat().st_size / (1024 * 1024)
        }
    
    if 'training_metadata' in save_data:
        info.update(save_data['training_metadata'])
    
    if 'sparsification_info' in save_data:
        info['sparsity'] = save_data['sparsification_info'].get('actual_sparsity', 0.0)
    
    return info