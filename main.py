# main.py - Universal QAT Training with TinyGPT Support
import argparse
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import data loaders
from data.cifar import get_cifar10_dataloaders
from data.imagenet100 import get_imagenet100_dataloaders
from data.text_datasets import (  # NEW: Text dataset support
    get_shakespeare_dataloaders,
    get_wikitext_dataloaders,
    get_simple_text_dataloaders,
    generate_text,
    compute_perplexity
)
from data.text_classification import get_imdb_dataloaders, get_sst2_dataloaders, validate_tinybert


# Import QAT utilities
from ops import enable_quantization, disable_quantization, print_quantization_status
from utils.training import TrainingMetrics, validate, save_checkpoint


class UniversalQATTrainer:
    """
    Universal trainer that handles all model types including TinyGPT
    """
    
    def __init__(self, args):
        self.args = args
        self.setup_paths()
        self.setup_data_loaders()
        self.setup_model()
        self.setup_training_components()
        self.setup_logging()
        
    def setup_paths(self):
        """Setup output directories"""
        work_dir = f"{self.args.work_dir}_{self.args.model_name}_{self.args.quantization}"
        
        # Add dataset info to work dir
        if self.args.model_type == 'tinygpt':
            work_dir += f"_{self.args.text_dataset}"
        elif self.args.model_type == 'tinybert':
            work_dir += f"_{self.args.text_classification_dataset}"
        else:
            # Vision models
            work_dir += f"_{self.args.dataset}"
                
        # Special flags
        if getattr(self.args, 'use_teacher', False):
            work_dir += f"_{self.args.teacher_type}teacher"
        if getattr(self.args, 'quantize_classifier', False):
            work_dir += "_qhead"
        if getattr(self.args, 'char_level', False):
            work_dir += "_char"
            
        os.makedirs(work_dir, exist_ok=True)
        self.args.work_dir = work_dir
        
    def setup_data_loaders(self):
        """Setup data loaders based on model type and dataset"""
        if self.args.model_type == 'tinygpt':
            # Text datasets for language modeling
            self._setup_text_data_loaders()
        elif self.args.model_type == 'tinybert':
            self._setup_text_classification_data_loaders()
        else:
            # Vision datasets for image classification
            self._setup_vision_data_loaders()
    
    def _setup_text_data_loaders(self):
        """Setup text data loaders for TinyGPT"""
        if self.args.text_dataset == "shakespeare":
            self.train_loader, self.test_loader, self.vocab_size = get_shakespeare_dataloaders(
                batch_size=self.args.batch_size,
                seq_len=self.args.seq_len,
                val_split=0.1,
                num_workers=self.args.num_workers,
                char_level=self.args.char_level
            )
        elif self.args.text_dataset == "wikitext":
            self.train_loader, self.test_loader, self.vocab_size = get_wikitext_dataloaders(
                batch_size=self.args.batch_size,
                seq_len=self.args.seq_len,
                num_workers=self.args.num_workers,
                char_level=self.args.char_level
            )
        elif self.args.text_dataset == "custom" and self.args.text_file:
            self.train_loader, self.test_loader, self.vocab_size = get_simple_text_dataloaders(
                text_file=self.args.text_file,
                batch_size=self.args.batch_size,
                seq_len=self.args.seq_len,
                val_split=0.1,
                num_workers=self.args.num_workers,
                char_level=self.args.char_level
            )
        else:
            raise ValueError(f"Unknown text dataset: {self.args.text_dataset}")
        
        # For language modeling, we don't have traditional "classes"
        self.num_classes = self.vocab_size
        self.is_language_model = True
        
        print(f" Text dataset: {self.args.text_dataset}")
        print(f" Vocab size: {self.vocab_size}")
        print(f" Sequence length: {self.args.seq_len}")
        print(f" Character-level: {self.args.char_level}")

    def _setup_text_classification_data_loaders(self):
        """Setup text classification data loaders for TinyBERT"""
        if self.args.text_classification_dataset == "imdb":
            self.train_loader, self.test_loader, self.num_classes = get_imdb_dataloaders(
                batch_size=self.args.batch_size,
                max_length=self.args.seq_len,  # reuse seq_len arg
                num_workers=self.args.num_workers
            )
        elif self.args.text_classification_dataset == "sst2":
            self.train_loader, self.test_loader, self.num_classes = get_sst2_dataloaders(
                batch_size=self.args.batch_size,
                max_length=self.args.seq_len,
                num_workers=self.args.num_workers
            )
        else:
            raise ValueError(f"Unknown text classification dataset: {self.args.text_classification_dataset}")
        
        self.is_language_model = False  # TinyBERT is classification, not language modeling
        
        print(f" Text classification dataset: {self.args.text_classification_dataset}")
        print(f" Number of classes: {self.num_classes}")
        print(f" Max sequence length: {self.args.seq_len}")
    
    def _setup_vision_data_loaders(self):
        """Setup vision data loaders for CNN/ViT models"""
        if self.args.dataset == "cifar10":
            self.train_loader, self.test_loader = get_cifar10_dataloaders(
                batch_size=self.args.batch_size, num_workers=self.args.num_workers
            )
            self.num_classes = 10
            self.img_size = 32
        elif self.args.dataset == "imagenet100":
            self.train_loader, self.test_loader = get_imagenet100_dataloaders(
                batch_size=self.args.batch_size, num_workers=self.args.num_workers
            )
            self.num_classes = 100
            self.img_size = 224
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        self.is_language_model = False
        
    def setup_model(self):
        """Setup model using existing factory functions"""
        model_kwargs = self._get_model_kwargs()
        
        if self.args.model_type == 'tinygpt':
            from networks.model_factory import tinygpt_nano, tinygpt_micro, tinygpt_mini, tinygpt_small
            
            if self.args.model_variant == 'nano':
                self.model = tinygpt_nano(**model_kwargs)
            elif self.args.model_variant == 'micro':
                self.model = tinygpt_micro(**model_kwargs)
            elif self.args.model_variant == 'mini':
                self.model = tinygpt_mini(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = tinygpt_small(**model_kwargs)
            else:
                raise ValueError(f"Unknown TinyGPT variant: {self.args.model_variant}")
            
        elif self.args.model_type == 'tinybert':
            # TinyBERT model creation
            from networks.model_factory import tinybert_base, tinybert_mini, tinybert_small, tinybert_tiny

            if self.args.model_variant == 'tiny':
                self.model = tinybert_tiny(**model_kwargs)
            elif self.args.model_variant == 'mini':
                self.model = tinybert_mini(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = tinybert_small(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = tinybert_base(**model_kwargs)
            else:
                raise ValueError(f"Unknown TinyBERT variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'resnet':
            from networks.model_factory import resnet18, resnet50
            if self.args.model_variant == '18':
                self.model = resnet18(**model_kwargs)
            elif self.args.model_variant == '50':
                self.model = resnet50(**model_kwargs)
            else:
                raise ValueError(f"Unknown ResNet variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'mobilenet':
            from networks.model_factory import mobilenetv1, mobilenetv2, mobilenetv3_small, mobilenetv3_large
            if self.args.model_variant == 'v1':
                self.model = mobilenetv1(**model_kwargs)
            elif self.args.model_variant == 'v2':
                self.model = mobilenetv2(**model_kwargs)
            elif self.args.model_variant == 'v3_small':
                self.model = mobilenetv3_small(**model_kwargs)
            elif self.args.model_variant == 'v3_large':
                self.model = mobilenetv3_large(**model_kwargs)
            else:
                raise ValueError(f"Unknown MobileNet variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'vit':
            from networks.model_factory import vit_tiny, vit_small, vit_base, vit_large
            if self.args.model_variant == 'tiny':
                self.model = vit_tiny(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = vit_small(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = vit_base(**model_kwargs)
            elif self.args.model_variant == 'large':
                self.model = vit_large(**model_kwargs)
            else:
                raise ValueError(f"Unknown ViT variant: {self.args.model_variant}")
                
        elif self.args.model_type == 'deit':
            from networks.model_factory import deit_tiny_model, deit_small_model, deit_base_model
            if self.args.model_variant == 'tiny':
                self.model = deit_tiny_model(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = deit_small_model(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = deit_base_model(**model_kwargs)
            else:
                raise ValueError(f"Unknown DeiT variant: {self.args.model_variant}")
            

        elif self.args.model_type == 'swin':
            from networks.model_factory import swin_tiny_model, swin_small_model, swin_base_model
            if self.args.model_variant == 'tiny':
                self.model = swin_tiny_model(**model_kwargs)
            elif self.args.model_variant == 'small':
                self.model = swin_small_model(**model_kwargs)
            elif self.args.model_variant == 'base':
                self.model = swin_base_model(**model_kwargs)
            else:
                raise ValueError(f"Unknown Swin variant: {self.args.model_variant}")
        else:
            raise ValueError(f"Unknown model type: {self.args.model_type}")
            
        # Load pretrained weights if requested (vision models only)
        if self.args.pretrained and not self.is_language_model:
            self._load_pretrained_weights()
            
        # Apply dataset-specific modifications (vision models only)
        if not self.is_language_model:
            self._modify_model_for_dataset()
        
        self.model.to(self.args.device)
        
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model-specific keyword arguments"""
        kwargs = {
            'quantization_method': self.args.quantization,
            'device': self.args.device,
            'threshold': self.args.threshold,
            'bits': self.args.bits,
        }
        
        if self.args.model_type == 'tinygpt':
            # TinyGPT-specific parameters
            kwargs.update({
                'vocab_size': self.vocab_size,
                'max_seq_len': self.args.seq_len,
                'dropout': self.args.dropout,
                'quantize_classifier': self.args.quantize_classifier
            })
        elif self.args.model_type == 'tinybert':
            # TinyBERT-specific parameters  
            kwargs.update({
                'vocab_size': 30522,  # BERT vocab size
                'max_position_embeddings': self.args.seq_len,
                'num_classes': self.num_classes,
                'dropout': self.args.dropout,
                'quantize_classifier': self.args.quantize_classifier
            })
        else:
            # Vision model parameters
            kwargs.update({
                'num_classes': self.num_classes,
            })
            
            # Model-specific parameters for vision models
            if self.args.model_type in ['vit', 'deit', 'swin']:
                kwargs.update({
                    'img_size': self.img_size,
                    'patch_size': 4 if self.img_size == 32 else 16,
                    'drop_rate': self.args.dropout,
                    'attn_drop_rate': self.args.attn_dropout,
                    'quantize_classifier': self.args.quantize_classifier
                })
                
            if self.args.model_type == 'deit':
                kwargs.update({
                    'teacher_model': self._setup_teacher_model() if self.args.use_teacher else None
                })
                
            if self.args.model_type == 'mobilenet':
                kwargs.update({
                    'width_multiplier': self.args.width_multiplier
                })
            
        return kwargs
    
    def _load_pretrained_weights(self):
        """Load pretrained weights for vision models"""
        try:
            if self.args.model_type == 'resnet':
                from networks.load_pretrained import load_pretrained_resnet
                self.model = load_pretrained_resnet(self.model, num_classes=self.num_classes)
                
            elif self.args.model_type == 'mobilenet':
                from networks.load_pretrained import load_pretrained_mobilenet
                variant = self.args.model_variant.replace('_small', '').replace('_large', '')
                self.model = load_pretrained_mobilenet(
                    self.model, mobilenet_version=variant, num_classes=self.num_classes
                )
                
            elif self.args.model_type == 'vit':
                from networks.load_pretrained import load_pretrained_vit
                self.model = load_pretrained_vit(
                    self.model, variant=self.args.model_variant, num_classes=self.num_classes, img_size=self.img_size
                )
                
            elif self.args.model_type == 'deit':
                from networks.load_pretrained import load_pretrained_deit
                self.model = load_pretrained_deit(
                    self.model, variant=self.args.model_variant, num_classes=self.num_classes, img_size=self.img_size
                )
                
            print(f" Loaded pretrained {self.args.model_type}-{self.args.model_variant} weights")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained weights: {e}")
            print("Falling back to random initialization")
    
    def _modify_model_for_dataset(self):
        """Apply dataset-specific model modifications for vision models"""
        if self.args.dataset == 'cifar10' and self.args.model_type == 'resnet':
            # Modify ResNet for CIFAR-10 (smaller images)
            if hasattr(self.model, 'conv1') and hasattr(self.model, 'config'):
                from ops.layers.all import UnifiedQuantizedConv2dBatchNorm2dReLU
                self.model.conv1 = UnifiedQuantizedConv2dBatchNorm2dReLU(
                    3, 64, kernel_size=3, stride=1, padding=1, 
                    bias=False, activation="relu", config=self.model.config
                )
                self.model.maxpool = nn.Identity()
                
        elif self.args.dataset == 'cifar10' and self.args.model_type == 'mobilenet':
            # Reduce stride for smaller CIFAR-10 images
            if hasattr(self.model, 'features') and len(self.model.features) > 0:
                first_conv = self.model.features[0]
                if hasattr(first_conv, 'conv2d'):
                    first_conv.conv2d.stride = (1, 1)
    
    def _setup_teacher_model(self):
        """Setup teacher model for distillation (DeiT)"""
        if not self.args.use_teacher:
            return None
            
        teacher_type = self.args.teacher_type
        
        if teacher_type == 'resnet':
            import torchvision.models as models
            teacher = models.resnet50(pretrained=True)
            if self.num_classes != 1000:
                teacher.fc = nn.Linear(teacher.fc.in_features, self.num_classes)
                
        elif teacher_type == 'vit':
            from networks.model_factory import vit_small, vit_base
            if self.args.model_variant == 'tiny':
                teacher = vit_small(quantization_method="linear", num_classes=self.num_classes)
            else:
                teacher = vit_base(quantization_method="linear", num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown teacher type: {teacher_type}")
            
        teacher.to(self.args.device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
            
        print(f" Teacher model loaded: {teacher_type}")
        return teacher
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, loss function, etc."""
        # Optimizer - Auto-detect best for model type
        if self.args.model_type in ['vit', 'deit', 'tinygpt', 'tinybert']:
            # Transformers prefer AdamW
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay
            )
        else:
            # CNNs prefer SGD
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.lr,
                momentum=0.9,
                weight_decay=self.args.weight_decay
            )
            
        # Scheduler
        if self.args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs
            )
        elif self.args.scheduler == 'hstlq_cosine':
            # HSTLQ-style cosine schedule
            def hstlq_cosine_schedule(epoch):
                progress = epoch / self.args.num_epochs
                return math.cos(math.pi * progress / 2)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=hstlq_cosine_schedule)
        else:
            self.scheduler = None
            
        # Loss function
        if self.args.model_type == 'deit' and self.args.use_teacher:
            from networks.unified_deit import DeiTLoss
            self.criterion = DeiTLoss(
                teacher_model=self._setup_teacher_model(),
                distillation_alpha=self.args.distillation_alpha,
                distillation_tau=self.args.distillation_tau
            )
        else:
            # Standard cross-entropy for both vision and language models
            self.criterion = nn.CrossEntropyLoss()
            
        # Quantization switch hook
        self.switch_hook = SwitchQuantizationModeHook(
            model=self.model, 
            switch_iter=self.args.switch_iter
        )
        
    def setup_logging(self):
        """Setup TensorBoard logging"""
        self.writer = SummaryWriter(self.args.work_dir)
        
    def train(self):
        """Main training loop"""
        model_desc = f"{self.args.model_type}-{self.args.model_variant}"
        # if self.is_language_model:
        #     dataset_desc = f"{self.args.text_dataset} ({'char' if self.args.char_level else 'word'}-level)"
        # else:
        #     dataset_desc = f"{self.args.dataset} ({self.num_classes} classes)"

        if self.is_language_model:
            dataset_desc = f"{self.args.text_dataset} ({'char' if self.args.char_level else 'word'}-level)"
        elif self.args.model_type == 'tinybert':
            dataset_desc = f"{self.args.text_classification_dataset} ({self.num_classes} classes)"
        else:
            dataset_desc = f"{self.args.dataset} ({self.num_classes} classes)"

            
        print(f" Starting training: {model_desc}")
        print(f" Dataset: {dataset_desc}")
        print(f" Quantization: {self.args.quantization}")
        print(f" Output: {self.args.work_dir}")
        
        # Show model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f" Model parameters: {total_params:,}")
        
        if hasattr(self.model, 'config'):
            print(f"  Model config: {self.model.config}")
            
        # Show initial quantization status
        print("\n Initial Quantization Status:")
        print_quantization_status(self.model)
        
        # Start with quantization disabled for calibration
        disable_quantization(self.model)
        
        # Training loop
        best_metric = float('inf') if self.is_language_model else 0.0
        epochs_without_improvement = 0
        
        for epoch in range(self.args.num_epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.args.num_epochs}, LR: {current_lr:.6f}")
            
            # Train one epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate
            if self.is_language_model:
                val_metric = self._validate_language_model(epoch)
                is_best = val_metric < best_metric  # Lower perplexity is better
                if is_best:
                    best_metric = val_metric
                    print(f"New best perplexity: {val_metric:.2f}")

            elif self.args.model_type == 'tinybert':
                val_metric = validate_tinybert(self.model, self.test_loader, epoch, self.writer, self.args.device)
                is_best = val_metric > best_metric  # Higher accuracy is better
                if is_best:
                    best_metric = val_metric
                    print(f"New best accuracy: {val_metric:.2f}%")
            
            else:
                val_metric = validate(self.model, self.test_loader, epoch, self.writer)
                is_best = val_metric > best_metric  # Higher accuracy is better
                if is_best:
                    best_metric = val_metric
                    print(f"New best accuracy: {val_metric:.2f}%")
            
            if is_best:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, val_metric, epoch, self.args.work_dir, is_best=is_best)
            
            # Early stopping
            if epochs_without_improvement >= self.args.early_stop:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
            # Log learning rate
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            
            # Generate sample text for language models
            if self.is_language_model and epoch % 5 == 0:
                self._generate_sample_text(epoch)
        
        metric_name = "perplexity" if self.is_language_model else "accuracy"
        print(f"\n Training completed! Best {metric_name}: {best_metric:.2f}")
        
        # Save final model
        torch.save(self.model.state_dict(), f'{self.args.work_dir}/final_model.pth')
        
        # Final status
        print("\n Final Quantization Status:")
        print_quantization_status(self.model)
        
        self.writer.close()
        
        return best_metric
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch - handles all model types"""
        self.model.train()
        metrics = TrainingMetrics()
        
        for i, data in enumerate(self.train_loader):
            if self.is_language_model:
                # Language modeling: inputs and targets from same batch
                inputs, targets = data
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # TinyGPT returns (logits, loss) when targets provided
                logits, loss = self.model(inputs, targets)
                
                # For metrics, we need to reshape for accuracy calculation
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for language models
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                metrics.update(loss.item(), logits_flat, targets_flat)
            elif self.args.model_type == 'tinybert':
                # Text classification: TinyBERT
                input_ids, attention_mask, labels = data
                input_ids = input_ids.to(self.args.device)
                attention_mask = attention_mask.to(self.args.device)
                labels = labels.to(self.args.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, loss = self.model(input_ids, attention_mask, labels=labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update metrics
                metrics.update(loss.item(), logits, labels) 
            else:
                # Vision models: standard classification
                inputs, targets = data
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.args.model_type == 'deit' and hasattr(self.criterion, 'teacher_model') and self.criterion.teacher_model is not None:
                    # DeiT with distillation
                    outputs = self.model(inputs, return_teacher_logits=True)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['total_loss']
                    logits = outputs[0]  # Use classification head for metrics
                else:
                    # Standard training
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    logits = outputs
                    
                # Backward pass
                loss.backward()
                
                # Gradient clipping for transformers
                if self.args.model_type in ['vit', 'deit']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                self.optimizer.step()
                
                # Update metrics
                metrics.update(loss.item(), logits, targets)
            
            # Check quantization switch
            iteration = epoch * len(self.train_loader) + i
            if self.switch_hook.after_train_iter(iteration):
                pass  # Already logged in hook
                
            # Log progress
            if (i + 1) % self.args.log_interval == 0:
                # metric_name = "Loss" if self.is_language_model else "Acc"
                # print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {metrics.avg_loss:.4f}, {metric_name}: {metrics.accuracy:.2f}")
                if self.is_language_model:
                    metric_name = "Token-Acc"
                    print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {metrics.avg_loss:.4f}, {metric_name}: {metrics.accuracy:.1f}%")
                else:
                    metric_name = "Acc"
                    print(f"  Batch [{i+1}/{len(self.train_loader)}], Loss: {metrics.avg_loss:.4f}, {metric_name}: {metrics.accuracy:.2f}%")
                
                self.writer.add_scalar('Training/Loss', metrics.avg_loss, iteration)
                self.writer.add_scalar('Training/Accuracy', metrics.accuracy, iteration)
                metrics.reset()
               
                
        return {"loss": metrics.avg_loss, "accuracy": metrics.accuracy}
    
    def _validate_language_model(self, epoch: int) -> float:
        """Validate language model and return perplexity"""
        from data.text_datasets import compute_perplexity
        
        perplexity = compute_perplexity(self.model, self.test_loader, self.args.device)
        
        print(f'Validation - Epoch: {epoch+1}, Perplexity: {perplexity:.2f}')
        
        # Log to TensorBoard
        self.writer.add_scalar('Validation/Perplexity', perplexity, epoch)
        
        return perplexity
    
    def _generate_sample_text(self, epoch: int):
        """Generate sample text for language models"""
        try:
            # Get dataset from train_loader for vocab mappings
            dataset = self.train_loader.dataset
            
            from data.text_datasets import generate_text
            
            # Generate with different prompts and temperatures
            prompts = ["", "The", "Once upon a time"]
            temps = [0.8, 1.0, 1.2]
            
            print(f"\nSample generations (Epoch {epoch+1}):")
            for i, (prompt, temp) in enumerate(zip(prompts, temps)):
                text = generate_text(
                    self.model, dataset, self.args.device, 
                    prompt=prompt, max_length=50, temperature=temp
                )
                print(f"  {i+1}. T={temp}, Prompt='{prompt}': {text[:100]}...")
        except Exception as e:
            print(f"Could not generate sample text: {e}")


class SwitchQuantizationModeHook:
    """Quantization switching hook"""
    def __init__(self, model, switch_iter=5000):
        self.model = model
        self.switch_iter = switch_iter
        self.switched = False

    def after_train_iter(self, iteration):
        if iteration + 1 == self.switch_iter and not self.switched:
            print(f"Iteration {iteration+1}: Switching to activation quantization")
            # enable_quantization(self.model)  # Uncomment when ready
            self.switched = True
            return True
        return False


def create_parser() -> argparse.ArgumentParser:
    """Universal argument parser for all models including TinyGPT"""
    parser = argparse.ArgumentParser("Universal QAT Training - Now with TinyGPT!")
    
    # Model arguments
    parser.add_argument("--model-type", required=True, 
                       choices=["resnet", "mobilenet", "vit", "deit", "swin", "tinygpt", "tinybert"], 
                       help="Model architecture type")
    parser.add_argument("--model-variant", required=True, 
                        help=(
                            "Model variant: 18/50 (resnet), "
                            "v1/v2/v3_small/v3_large (mobilenet), "
                            "tiny/small/base/large (vit/deit), "
                            "tiny/small/base (swin), "
                            "nano/micro/mini/small (tinygpt), "
                            "tiny/mini/small/base (tinybert)"
                        ))
        
    # Dataset arguments
    parser.add_argument("--dataset", default="cifar10", 
                       choices=["cifar10", "imagenet100"],
                       help="Vision dataset (for non-language models)")
    parser.add_argument("--text-dataset", default="shakespeare",
                       choices=["shakespeare", "wikitext", "custom"],
                       help="Text dataset (for TinyGPT)")
    parser.add_argument("--text-classification-dataset", default="sst2",
                       choices=["imdb", "sst2"],
                       help="Text classification dataset (for TinyBERT)")
    
    parser.add_argument("--text-file", type=str,
                       help="Path to custom text file (when text-dataset=custom)")
    parser.add_argument("--char-level", action="store_true",
                       help="Use character-level tokenization (default: word-level)")
    parser.add_argument("--seq-len", default=128, type=int,
                       help="Sequence length for language modeling")
    
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    
    # Training arguments
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--scheduler", default="cosine", 
                       choices=["cosine", "hstlq_cosine", "none"])
    parser.add_argument("--early-stop", default=10, type=int)
    
    # Quantization arguments
    parser.add_argument("--quantization", default="linear", choices=["linear", "log"])
    parser.add_argument("--bits", default=8, type=int)
    parser.add_argument("--switch-iter", default=5000, type=int)
    parser.add_argument("--threshold", default=1e-5, type=float)
    parser.add_argument("--quantize-classifier", action="store_true")
    
    # Model-specific arguments
    parser.add_argument("--dropout", default=0.1, type=float, help="For ViT/DeiT/TinyGPT")
    parser.add_argument("--attn-dropout", default=0.1, type=float, help="For ViT/DeiT")
    parser.add_argument("--width-multiplier", default=1.0, type=float, help="For MobileNet")
    
    # Distillation arguments (DeiT)
    parser.add_argument("--use-teacher", action="store_true")
    parser.add_argument("--teacher-type", default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--distillation-alpha", default=0.5, type=float)
    parser.add_argument("--distillation-tau", default=3.0, type=float)
    
    # System arguments
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained weights (vision models only)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", default="./output")
    parser.add_argument("--log-interval", default=100, type=int)
    
    return parser


def main():
    """Universal training main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set derived arguments
    args.model_name = f"{args.model_type}_{args.model_variant}"
    
    # Validation
    if args.model_type == 'tinygpt' and not hasattr(args, 'text_dataset'):
        args.text_dataset = 'shakespeare'

    if args.model_type == 'tinybert' and not hasattr(args, 'text_classification_dataset'):
        args.text_classification_dataset = 'sst2' 
    
    if args.text_dataset == 'custom' and not args.text_file:
        raise ValueError("--text-file required when --text-dataset=custom")
    
    # Auto-adjust learning rate for different model types
    if args.model_type == 'tinygpt' and args.lr == 0.01:
        args.lr = 3e-4  # Better default for language models
        print(f"Auto-adjusted LR to {args.lr} for TinyGPT")
    if args.model_type == 'tinybert' and args.lr == 0.01:
        args.lr = 1e-4  # Better default for language models
        print(f"Auto-adjusted LR to {args.lr} for TinyBERT")
    
    # Create trainer and run
    trainer = UniversalQATTrainer(args)
    best_metric = trainer.train()
    
    metric_name = "perplexity" if trainer.is_language_model else "accuracy"
    print(f"🎯 Final result: {best_metric:.2f} {metric_name}")


if __name__ == "__main__":
    main()