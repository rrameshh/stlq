# networks/unified_deit.py - Corrected DeiT Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .unified_vit import (
    IndustryStandardViT, 
    SelectiveQuantizedTransformerBlock,
    PatchEmbedding
)
from ops.layers.all import UnifiedQuantizedLinear, UnifiedQuantize
from ops.quant_config import QuantizationConfig


class DeiTModel(IndustryStandardViT):
    """
    Data-efficient Image Transformer (DeiT)
    
    Key differences from ViT:
    1. Distillation token (in addition to class token)
    2. Dual classification heads (class + distillation)
    3. Teacher-student training support
    4. Same transformer architecture (including MLP layers)
    
    Inherits from IndustryStandardViT to reuse all the transformer infrastructure.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, 
                 act_layer=None, config=None, teacher_model=None):
        
        # Initialize base ViT (gets all transformer blocks with MLP)
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer, act_layer=act_layer, config=config
        )
        
        # DeiT-specific additions
        self.teacher_model = teacher_model
        self.num_tokens = 2  # cls + distillation tokens
        
        # Add distillation token (similar to cls_token)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Update position embeddings for extra token
        num_patches = self.patch_embed.num_patches
        # Delete the old position embedding
        del self.pos_embed
        # Create new one with space for both tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        
        # Replace single head with dual classification heads
        # Remove the inherited single head
        if hasattr(self, 'head'):
            del self.head
        if hasattr(self, 'head_quantizer'):
            del self.head_quantizer
            
        # Create dual heads
        quantize_classifier = getattr(config, 'quantize_classifier', False)
        if quantize_classifier:
            # Both heads quantized
            self.head_cls = UnifiedQuantizedLinear(embed_dim, num_classes, config=config)
            self.head_dist = UnifiedQuantizedLinear(embed_dim, num_classes, config=config)
            self.head_quantizer = UnifiedQuantize(config=config)
        else:
            # Industry standard: Keep heads in FP32
            self.head_cls = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head_dist = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize new parameters
        self._init_deit_weights()
        
    def _init_deit_weights(self):
        """Initialize DeiT-specific parameters"""
        torch.nn.init.trunc_normal_(self.dist_token, std=0.02)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize dual heads
        if hasattr(self.head_cls, 'linear'):  # Quantized head
            torch.nn.init.trunc_normal_(self.head_cls.linear.weight, std=0.02)
            torch.nn.init.trunc_normal_(self.head_dist.linear.weight, std=0.02)
            if self.head_cls.linear.bias is not None:
                nn.init.constant_(self.head_cls.linear.bias, 0)
            if self.head_dist.linear.bias is not None:
                nn.init.constant_(self.head_dist.linear.bias, 0)
        else:  # FP32 head
            torch.nn.init.trunc_normal_(self.head_cls.weight, std=0.02)
            torch.nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
            if self.head_cls.bias is not None:
                nn.init.constant_(self.head_cls.bias, 0)
            if self.head_dist.bias is not None:
                nn.init.constant_(self.head_dist.bias, 0)
    
    def forward_features(self, x):
        """
        Forward through transformer blocks with dual tokens
        """
        B = x.shape[0]
        
        # Patch embedding (reuses parent implementation)
        x = self.patch_embed(x)
        
        # Add BOTH cls and distillation tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        
        # Position embeddings (now includes both tokens)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks (reuses ALL parent implementation including MLP!)
        for blk in self.blocks:
            x = blk(x)
        
        # Final norm (reuses parent implementation)
        x = self.norm(x)
        
        return x

    def forward(self, x, return_teacher_logits=False):
        """
        Forward pass with dual outputs for distillation training
        
        Returns:
            - During training: (cls_logits, dist_logits, teacher_logits)
            - During inference: averaged logits or cls_logits only
        """
        # Store original input for teacher model
        original_input = x
        
        # Forward through DeiT
        features = self.forward_features(x)
        
        # Extract tokens
        cls_token = features[:, 0]  # Classification token
        dist_token = features[:, 1]  # Distillation token
        
        # Dual classification heads
        if hasattr(self, 'head_quantizer'):
            # Quantized heads
            cls_quantized = self.head_quantizer(cls_token)
            dist_quantized = self.head_quantizer(dist_token)
            
            cls_logits = self.head_cls(cls_quantized)
            dist_logits = self.head_dist(dist_quantized)
            
            # Dequantize outputs if needed
            if hasattr(cls_logits, 'dequantize'):
                cls_logits = cls_logits.dequantize()
            if hasattr(dist_logits, 'dequantize'):
                dist_logits = dist_logits.dequantize()
        else:
            # FP32 heads
            cls_logits = self.head_cls(cls_token)
            dist_logits = self.head_dist(dist_token)
        
        # Get teacher logits if available and requested
        teacher_logits = None
        if self.training and self.teacher_model is not None and return_teacher_logits:
            with torch.no_grad():
                # FIXED: Pass original input (raw images) to teacher model
                teacher_logits = self.teacher_model(original_input)
        
        if self.training:
            # Training: return all logits for distillation loss
            return cls_logits, dist_logits, teacher_logits
        else:
            # Inference: return averaged prediction (DeiT paper approach)
            return (cls_logits + dist_logits) / 2


class DeiTLoss(nn.Module):
    """
    DeiT loss combining classification + distillation
    """
    
    def __init__(self, teacher_model=None, distillation_alpha=0.5, distillation_tau=3.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = distillation_alpha  # Weight between hard and soft losses
        self.tau = distillation_tau      # Temperature for distillation
        
        self.hard_loss = nn.CrossEntropyLoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_outputs, targets):
        """
        Compute DeiT loss: hard loss + distillation loss
        
        Args:
            student_outputs: (cls_logits, dist_logits, teacher_logits)
            targets: Ground truth labels
        """
        cls_logits, dist_logits, teacher_logits = student_outputs
        
        # Hard loss: student vs ground truth
        hard_loss_cls = self.hard_loss(cls_logits, targets)
        hard_loss_dist = self.hard_loss(dist_logits, targets)
        hard_loss = (hard_loss_cls + hard_loss_dist) / 2
        
        # Soft loss: student vs teacher (if teacher available)
        soft_loss = torch.tensor(0.0, device=cls_logits.device)
        if teacher_logits is not None:
            # Distillation loss for both heads
            soft_targets = F.softmax(teacher_logits / self.tau, dim=1)
            soft_pred_cls = F.log_softmax(cls_logits / self.tau, dim=1)
            soft_pred_dist = F.log_softmax(dist_logits / self.tau, dim=1)
            
            soft_loss_cls = self.soft_loss(soft_pred_cls, soft_targets) * (self.tau ** 2)
            soft_loss_dist = self.soft_loss(soft_pred_dist, soft_targets) * (self.tau ** 2)
            soft_loss = (soft_loss_cls + soft_loss_dist) / 2
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }


# Factory functions
def create_deit(
    variant="small", 
    quantization_method="linear", 
    teacher_model=None,
    **kwargs
) -> DeiTModel:
    """
    Create DeiT model using the corrected implementation
    """
    
    configs = {
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6}, 
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    # Extract config parameters
    device = kwargs.pop('device', 'cuda:0')
    threshold = kwargs.pop('threshold', 1e-5)
    momentum = kwargs.pop('momentum', 0.1)
    bits = kwargs.pop('bits', 8)
    quantize_classifier = kwargs.pop('quantize_classifier', False)
    
    # Create quantization config
    config = QuantizationConfig(
        method=quantization_method,
        momentum=momentum,
        device=device,
        threshold=threshold,
        bits=bits
    )
    config.quantize_classifier = quantize_classifier
    
    # Merge configs
    model_config = configs[variant]
    model_config.update(kwargs)
    
    return DeiTModel(config=config, teacher_model=teacher_model, **model_config)


# Convenient factory functions
def deit_tiny(**kwargs):
    return create_deit("tiny", **kwargs)

def deit_small(**kwargs):
    return create_deit("small", **kwargs)

def deit_base(**kwargs):
    return create_deit("base", **kwargs)