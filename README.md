# QAT Framework

A flexible and extensible Quantization-Aware Training (QAT) framework for PyTorch, supporting multiple quantization methods and model architectures for efficient deep learning research and deployment.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

This framework allows for the training of quantized neural networks with minimal code changes. It supports both traditional linear quantization and selective-two word logarithmic quantization methods, with built-in support for popular model architectures.

### Key Features

- **Multiple Quantization Methods**: Linear (uniform) and logarithmic quantization with configurable bit-widths
- **Model Support**: 20+ pre-built models including ResNet, MobileNet, Vision Transformers, and language models
- **Flexible Configuration**: YAML-based configuration system with validation and override support
- **Research Tools**: Comprehensive logging, metrics tracking, and checkpoint management
- **Easy Extension**: Modular design allows easy addition of new models and quantization methods

## Supported Models and Datasets

### Vision Models
| Model Family | Variants | Pretrained | Datasets |
|-------------|----------|------------|----------|
| ResNet | ResNet-18, ResNet-50 | ImageNet | CIFAR-10, ImageNet-100 |
| MobileNet | v1, v2, v3 (small/large) | ImageNet | CIFAR-10, ImageNet-100 |
| Vision Transformer | ViT-Tiny/Small/Base | ImageNet-21k | CIFAR-10, ImageNet-100 |
| DeiT | DeiT-Tiny/Small/Base | ImageNet | CIFAR-10, ImageNet-100 |
| Swin Transformer | Swin-Tiny/Small/Base | ImageNet | CIFAR-10, ImageNet-100 |

### Language Models
| Model Family | Variants | Datasets |
|-------------|----------|----------|
| TinyBERT | Tiny/Mini/Small/Base | IMDB, SST-2 |
| TinyGPT | Nano/Micro/Mini/Small | Shakespeare, WikiText |

## Installation

### Requirements
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA 11.0+ (optional, for GPU acceleration)

### Install from source
```bash
git clone https://github.com/rrameshh/stlq.git
cd stlq
pip install -r requirements.txt
```

### Dependencies
```
torch>=2.0.0,<2.5.0
torchvision>=0.15.0,<0.20.0
transformers>=4.20.0,<5.0.0
datasets>=2.0.0,<3.0.0
timm>=0.9.0,<1.0.0
tensorboard>=2.10.0
PyYAML>=6.0
numpy>=1.21.0,<2.0.0
```

## Quick Start

### Basic Training Example

1. **Prepare a configuration file** (see `configs/cifar10_resnet18_linear.yaml`):

```yaml
data:
  dataset: "cifar10"
  batch_size: 128
  num_workers: 4

model:
  name: "resnet18"
  num_classes: 10
  pretrained: false

quantization:
  method: "linear"
  bits: 8
  switch_iteration: 5000

training:
  num_epochs: 200
  lr: 0.1
  weight_decay: 0.0001

system:
  device: "cuda:0"
  work_dir: "./output_linear"
```

2. **Start training**:

```bash
python main.py --config configs/cifar10_resnet18_linear.yaml
```

### Advanced Usage

**Override configuration parameters**:
```bash
python main.py --config configs/base_config.yaml \
  --override training.lr=0.01 quantization.bits=4 training.num_epochs=100
```

**Logarithmic quantization**:
```bash
python main.py --config configs/cifar10_resnet18_log.yaml
```

**Training with pretrained weights**:
```bash
python main.py --config configs/imagenet_mobilenet.yaml
```

## Configuration System

The framework uses a hierarchical YAML configuration system with five main sections:

- **data**: Dataset selection, batch size, data loading parameters
- **model**: Model architecture, number of classes, pretrained weights
- **quantization**: Quantization method, bit-width, switching parameters
- **training**: Learning rate, epochs, optimization settings
- **system**: Device selection, output directories, random seeds

### Configuration Validation

All configurations are automatically validated with helpful error messages:

```python
# Automatic type conversion and validation
config = Config.from_yaml("my_config.yaml")
config.print_summary()  # Display configuration overview
```

## Monitoring and Logging

### TensorBoard Integration
```bash
tensorboard --logdir=./output_linear/tensorboard
```

Logged metrics include:
- Training and validation loss/accuracy
- Learning rate schedules
- Quantization statistics (for logarithmic method)
- Model parameter distributions

### Checkpoint Management
- Automatic saving of best models
- Resume training from checkpoints
- Model state and optimizer state preservation

## Extending the Framework

### Adding New Models

1. **Create model definition**:
```python
# models/vision/my_model.py
def my_custom_model(main_config, **kwargs):
    config = QuantizationConfig(
        method=main_config.quantization.method,
        # ... other parameters
    )
    return MyModel(config=config, **kwargs)
```

2. **Register in model registry**:
```python
# models/registry.py
MODELS = {
    'my_custom_model': my_custom_model,
    # ... existing models
}
```

### Adding New Quantization Methods

1. **Implement quantization strategy**:
```python
# quantization/strategies/my_method.py
class MyQuantizationStrategy(QuantizationStrategy):
    def quantize_weight(self, weight, per_channel=True):
        # Implementation here
        pass
```

2. **Register in factory**:
```python
# quantization/strategies/factory.py
def create_strategy(config):
    if config.method == "my_method":
        return MyQuantizationStrategy(config)
    # ... existing methods
```
