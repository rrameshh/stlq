# models/packaging.py - Model packaging and distribution utilities
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from .io import get_model_info


def create_model_package(model_dir: str, package_name: str = None):
    """
    Package model files into a organized directory with documentation
    
    Args:
        model_dir: Directory containing *_quantized.pth and *_fp32.pth files
        package_name: Name for the package (auto-generated if None)
    
    Returns:
        Path to created package directory
    """
    model_dir = Path(model_dir)
    
    # Find model files
    quantized_files = list(model_dir.glob("*_quantized.pth"))
    fp32_files = list(model_dir.glob("*_fp32.pth"))
    
    if not quantized_files:
        raise ValueError(f"No quantized model files found in {model_dir}")
    
    # Use first quantized file for metadata
    info = get_model_info(quantized_files[0])
    
    if package_name is None:
        package_name = f"{info['model_name']}_{info.get('quantization_method', 'quantized')}"
        if 'sparsity' in info and info['sparsity'] > 0:
            package_name += f"_sparse{int(info['sparsity']*100)}"
    
    # Create package directory
    package_dir = model_dir / package_name
    package_dir.mkdir(exist_ok=True)
    
    # Copy files with standard names
    if quantized_files:
        shutil.copy2(quantized_files[0], package_dir / "model_quantized.pth")
    if fp32_files:
        shutil.copy2(fp32_files[0], package_dir / "model_fp32.pth")
    
    # Create metadata file
    with open(package_dir / "model_info.json", 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    print(f"✅ Created model package: {package_dir}")
    return package_dir
