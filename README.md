# QAT Framework

This repository provides a flexible Quantization-Aware Training (QAT) implementation for PyTorch models, supporting both linear and logarithmic quantization methods for CIFAR-10 classification.

## Setup

```bash
git clone https://github.com/rrameshh/stlq.git
cd stlq
pip install -r requirements.txt
```

## Usage

### Training a QAT Model on CIFAR-10

For linear quantization:

```bash
python train_cifar.py --quantization linear --batch-size 128 --num-epochs 200 --lr 0.1 --switch-iter 5000
```

For logarithmic quantization:

```bash
python train_cifar.py --quantization log --batch-size 128 --num-epochs 200 --lr 0.1 --switch-iter 5000 --threshold 1e-5
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch-size` | Batch size for training | 128 |
| `--lr` | Initial learning rate | 0.1 |
| `--num-epochs` | Number of training epochs | 200 |
| `--work-dir` | Directory to save checkpoints and logs | "./output" |
| `--device` | Device to use for training | "cuda:0" if available, else "cpu" |
| `--num-workers` | Number of workers for data loading | 4 |
| `--switch-iter` | Iteration to switch from calibration to activation quantization | 5000 |
| `--quantization` | Quantization method ("linear" or "log") | "linear" |
| `--threshold` | Threshold for second-order quantization in log method | 1e-5 |
| `--early-stop` | Early stopping patience (epochs without improvement) | 10 |

## Training Process

The training process consists of two phases:

1. **Calibration Phase:** The model collects statistics about activation ranges without actually quantizing activations. This phase allows the model to learn the typical distribution of activation values, which will be used to determine appropriate quantization parameters.

2. **Quantization-Aware Training Phase:** After statistics collection, the model simulates quantization effects during both forward and backward passes. This allows the model to adapt its weights to minimize the impact of quantization.

The transition between these phases is controlled by the `--switch-iter` parameter. During training, a SwitchQuantizationModeHook checks the current iteration and enables activation quantization when reaching the specified iteration.

### Monitoring Training

Training progress is logged to TensorBoard, which allows you to monitor:

- Training and validation loss
- Accuracy metrics
- Learning rate changes

View the logs using:

```bash
tensorboard --logdir=./output_linear  # For linear quantization results
tensorboard --logdir=./output_log     # For logarithmic quantization results
```

