import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from quantization.layers.quantized import (
    Quantize,
    QLinear,
)
from quantization.quant_config import QuantizationConfig
from quantization.tensors.linear import LinearQuantizedTensor


# ====================  FP32 PATCH EMBEDDING ====================
class PatchEmbedding(nn.Module):
    """
    INDUSTRY STANDARD: Patch embedding stays in FP32
    Lightweight operation, precision-critical for feature extraction
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Regular FP32 operations (industry standard)
        # can be quantized if needed
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
        )
        
    def forward(self, x):
        """Standard FP32 forward pass"""
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match expected size ({self.img_size}x{self.img_size})"
        
        x = self.projection(x)  # [B, embed_dim, P_col, P_row]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


# ==================== EXPLICIT FLOW: ATTENTION WITH CLEAR TRANSITIONS ====================
class SelectiveQuantizedMultiHeadAttention(nn.Module):
    """
    EXPLICIT FLOW: Clear transitions between FP32 and quantized operations
    No more flexible handling - explicit case management
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., config=None):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers
        self.qkv = QLinear(dim, dim * 3, bias=qkv_bias, config=config)
        self.proj = QLinear(dim, dim, bias=qkv_bias, config=config)
        
        # FP32: Lightweight operations
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Input quantizer for explicit transition management
        self.input_quantizer = Quantize(config=config)
    
    def forward(self, x):
        """
        EXPLICIT FLOW: Clear transitions with explicit case handling
        
        Flow:
        1. FP32 input → Quantize → INT8
        2. INT8 QKV projection → Dequantize → FP32
        3. FP32 attention computation → Keep FP32
        4. FP32 → Quantize → INT8 projection → Dequantize → FP32
        """
        B, N, C = x.shape
        
        # ============ EXPLICIT TRANSITION 1: FP32 → INT8 ============
        # Input should be FP32 from LayerNorm
        assert isinstance(x, torch.Tensor) and not isinstance(x, LinearQuantizedTensor), \
            "Expected FP32 input from LayerNorm"
        
        # Quantize input for heavy compute
        x_quantized = self.input_quantizer(x)
        
        # ============ QUANTIZED COMPUTE: QKV PROJECTION ============
        qkv_quantized = self.qkv(x_quantized)
        
        # ============ EXPLICIT TRANSITION 2: INT8 → FP32 ============
        # Dequantize for attention computation (numerically sensitive)
        if isinstance(qkv_quantized, LinearQuantizedTensor):
            qkv_fp32 = qkv_quantized.dequantize()
        else:
            qkv_fp32 = qkv_quantized
        
        # ============ FP32 COMPUTE: ATTENTION ============
        qkv_reshaped = qkv_fp32.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        # Attention computation in FP32 (industry standard)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # ============ EXPLICIT TRANSITION 3: FP32 → INT8 → FP32 ============
        # Quantize for heavy output projection
        out_quantized = self.input_quantizer(out)
        
        # Quantized projection
        out_proj_quantized = self.proj(out_quantized)
        
        # Dequantize for next layer (residual connection needs FP32)
        if isinstance(out_proj_quantized, LinearQuantizedTensor):
            out_fp32 = out_proj_quantized.dequantize()
        else:
            out_fp32 = out_proj_quantized
            
        out_fp32 = self.proj_drop(out_fp32)
        
        # Return FP32 for residual connection
        return out_fp32


# ==================== EXPLICIT FLOW: MLP WITH CLEAR TRANSITIONS ====================
class SelectiveQuantizedMLP(nn.Module):
    """
    EXPLICIT FLOW: Clear transitions between FP32 and quantized operations
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=None, drop=0., config=None):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers
        self.fc1 = QLinear(in_features, hidden_features, config=config)
        self.fc2 = QLinear(hidden_features, out_features, config=config)
        
        # FP32: Activation and dropout
        self.act = act_layer() if act_layer else nn.GELU()
        self.drop = nn.Dropout(drop)
        
        # Input quantizer
        self.input_quantizer = Quantize(config=config)
    
    def forward(self, x):
        """
        EXPLICIT FLOW: Clear transitions with explicit case handling
        
        Flow:
        1. FP32 input → Quantize → INT8
        2. INT8 FC1 → Dequantize → FP32
        3. FP32 activation → Keep FP32
        4. FP32 → Quantize → INT8 FC2 → Dequantize → FP32
        """
        
        # ============ EXPLICIT TRANSITION 1: FP32 → INT8 ============
        assert isinstance(x, torch.Tensor) and not isinstance(x, LinearQuantizedTensor), \
            "Expected FP32 input from LayerNorm"
        
        # Quantize input for heavy compute
        x_quantized = self.input_quantizer(x)
        
        # ============ QUANTIZED COMPUTE: FC1 ============
        x_quantized = self.fc1(x_quantized)
        
        # ============ EXPLICIT TRANSITION 2: INT8 → FP32 ============
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
        
        # ============ FP32 COMPUTE: ACTIVATION ============
        x_fp32 = self.act(x_fp32)
        x_fp32 = self.drop(x_fp32)
        
        # ============ EXPLICIT TRANSITION 3: FP32 → INT8 → FP32 ============
        # Quantize for heavy compute
        x_quantized = self.input_quantizer(x_fp32)
        
        # Quantized FC2
        x_quantized = self.fc2(x_quantized)
        
        # Dequantize for next layer
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
            
        x_fp32 = self.drop(x_fp32)
        
        # Return FP32 for residual connection
        return x_fp32


# ==================== EXPLICIT FLOW: TRANSFORMER BLOCK ====================
class SelectiveQuantizedTransformerBlock(nn.Module):
    """
    EXPLICIT FLOW: Mixed precision transformer block with clear transitions
    - FP32: LayerNorm, residual connections
    - Quantized: Attention and MLP linear layers
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., norm_layer=nn.LayerNorm, act_layer=None, config=None):
        super().__init__()
        
        self.config = config
        
        # FP32: LayerNorm (industry standard)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # EXPLICIT: Attention and MLP with clear transitions
        self.attn = SelectiveQuantizedMultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop, config=config
        )
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SelectiveQuantizedMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, drop=drop, config=config
        )
    
    def forward(self, x):
        """
        EXPLICIT FLOW: Clear mixed precision flow
        
        Flow:
        1. FP32 input → FP32 LayerNorm → FP32
        2. FP32 → Attention (internal quantization) → FP32
        3. FP32 residual connection → FP32
        4. FP32 → FP32 LayerNorm → FP32
        5. FP32 → MLP (internal quantization) → FP32
        6. FP32 residual connection → FP32
        """
        
        # Ensure input is FP32
        assert isinstance(x, torch.Tensor) and not isinstance(x, LinearQuantizedTensor), \
            "Transformer block expects FP32 input"
        
        # ============ ATTENTION BLOCK ============
        # FP32 LayerNorm → FP32
        normed_x1 = self.norm1(x)
        assert isinstance(normed_x1, torch.Tensor) and not isinstance(normed_x1, LinearQuantizedTensor)
        
        # FP32 → Attention (handles quantization internally) → FP32
        attn_out = self.attn(normed_x1)
        assert isinstance(attn_out, torch.Tensor) and not isinstance(attn_out, LinearQuantizedTensor)
        
        # FP32 residual connection
        x = x + attn_out
        
        # ============ MLP BLOCK ============
        # FP32 LayerNorm → FP32
        normed_x2 = self.norm2(x)
        assert isinstance(normed_x2, torch.Tensor) and not isinstance(normed_x2, LinearQuantizedTensor)
        
        # FP32 → MLP (handles quantization internally) → FP32
        mlp_out = self.mlp(normed_x2)
        assert isinstance(mlp_out, torch.Tensor) and not isinstance(mlp_out, LinearQuantizedTensor)
        
        # FP32 residual connection
        x = x + mlp_out
        
        return x


# ==================== EXPLICIT FLOW: VIT MODEL ====================
class ViT(nn.Module):

    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm, 
                 act_layer=None, config=None):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        
        # ============ FP32 SECTION ============
        
        # Patch embedding (FP32)
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Learnable embeddings (FP32)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Final norm (FP32)
        self.norm = norm_layer(embed_dim)
        
        # ============ EXPLICIT MIXED PRECISION SECTION ============
        
        # Transformer blocks with explicit transitions
        self.blocks = nn.ModuleList([
            SelectiveQuantizedTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, act_layer=act_layer, config=config
            )
            for _ in range(depth)
        ])
        
        # Classification head
        quantize_classifier = getattr(config, 'quantize_classifier', False)
        if quantize_classifier:
            self.head = QLinear(embed_dim, num_classes, config=config)
            self.head_quantizer = Quantize(config=config)
        else:
            # Industry standard: Keep classifier in FP32
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ViT paper"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    

    
    def forward_features(self, x):
        """
        EXPLICIT FLOW: Clear transitions documented at each step
        """
        B = x.shape[0]
        
        # ============ FP32 EMBEDDING PHASE ============
        
        # Patch embedding (FP32)
        x = self.patch_embed(x)
        
        # Add class token and position embeddings (FP32)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # ============ EXPLICIT MIXED PRECISION COMPUTATION ============
        
        # Transformer blocks (explicit transitions)
        for i, blk in enumerate(self.blocks):
            # print(f"\n--- Block {i} ---")
            x = blk(x)  # Handles explicit transitions internally
        
        # ============ FP32 OUTPUT PHASE ============
        
        # Final processing (FP32)
        x = self.norm(x)
        
        return x

    def forward(self, x):
        """Explicit flow forward pass with transition logging"""
        x = self.forward_features(x)
        
        # Classification head
        cls_token = x[:, 0]
        
        if hasattr(self, 'head_quantizer'):
            # Quantized classifier path
            cls_quantized = self.head_quantize(cls_token)
            
            x = self.head(cls_quantized)
            
            # Dequantize output
            if isinstance(x, LinearQuantizedTensor):
                x = x.dequantize()
        else:
            # FP32 classifier path
            x = self.head(cls_token)
        
        return x


# # ==================== FACTORY FUNCTIONS ====================
# def create_industry_standard_vit(
#     variant="small", 
#     quantization_method="linear", 
#     **kwargs
# ) -> ViT:
#     """
#     Create Vision Transformer with Explicit Flow Management
#     """
    
#     configs = {
#         "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
#         "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
#         "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
#         "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
#         "huge": {"embed_dim": 1280, "depth": 32, "num_heads": 16}
#     }
    
#     if variant not in configs:
#         raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
#     # Extract config parameters
#     device = kwargs.pop('device', 'cuda:0')
#     threshold = kwargs.pop('threshold', 1e-5)
#     momentum = kwargs.pop('momentum', 0.1)
#     bits = kwargs.pop('bits', 8)
#     quantize_classifier = kwargs.pop('quantize_classifier', False)
    
#     # Create quantization config
#     config = QuantizationConfig(
#         method=quantization_method,
#         momentum=momentum,
#         device=device,
#         threshold=threshold,
#         bits=bits
#     )
#     config.quantize_classifier = quantize_classifier
    
#     # Merge configs
#     model_config = configs[variant]
#     model_config.update(kwargs)
    
#     return ViT(config=config, **model_config)


# # Convenient factory functions
# def vit_tiny(**kwargs):
#     return create_industry_standard_vit("tiny", **kwargs)

# def vit_small(**kwargs):
#     return create_industry_standard_vit("small", **kwargs)

# def vit_base(**kwargs):
#     return create_industry_standard_vit("base", **kwargs)

# def vit_large(**kwargs):
#     return create_industry_standard_vit("large", **kwargs)
    

def vit_tiny(main_config, **kwargs):
    """ViT-Tiny - takes main config"""
    config = QuantizationConfig(
        method=main_config.quantization.method,
        momentum=main_config.quantization.momentum,
        adaptive_threshold=main_config.quantization.adaptive_threshold,
        target_second_word_ratio=main_config.quantization.target_second_word_ratio,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    config.quantize_classifier = False
    
    return ViT(
        embed_dim=192, depth=12, num_heads=3,  # tiny config
        config=config,
        num_classes=main_config.model.num_classes,
        img_size=main_config.model.img_size,
        **kwargs
    )

def vit_small(main_config, **kwargs):
    """ViT-Small - takes main config"""
    config = QuantizationConfig(
        method=main_config.quantization.method,
        momentum=main_config.quantization.momentum,
        adaptive_threshold=main_config.quantization.adaptive_threshold,
        target_second_word_ratio=main_config.quantization.target_second_word_ratio,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    config.quantize_classifier = False
    
    return ViT(
        embed_dim=384, depth=12, num_heads=6,  # small config
        config=config,
        num_classes=main_config.model.num_classes,
        img_size=main_config.model.img_size,
        **kwargs
    )

def vit_base(main_config, **kwargs):
    """ViT-Base - takes main config"""
    config = QuantizationConfig(
        method=main_config.quantization.method,
        adaptive_threshold=main_config.quantization.adaptive_threshold,
        target_second_word_ratio=main_config.quantization.target_second_word_ratio,
        momentum=main_config.quantization.momentum,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    config.quantize_classifier = False
    
    return ViT(
        embed_dim=768, depth=12, num_heads=12,  # base config
        config=config,
        num_classes=main_config.model.num_classes,
        img_size=main_config.model.img_size,
        **kwargs
    )

def vit_large(main_config, **kwargs):
    """ViT-Large - takes main config"""  
    config = QuantizationConfig(
        method=main_config.quantization.method,
        adaptive_threshold=main_config.quantization.adaptive_threshold,
        target_second_word_ratio=main_config.quantization.target_second_word_ratio,
        momentum=main_config.quantization.momentum,
        device=main_config.system.device,
        threshold=main_config.quantization.threshold,
        bits=main_config.quantization.bits
    )
    config.quantize_classifier = False
    
    return ViT(
        embed_dim=1024, depth=24, num_heads=16,  # large config
        config=config,
        num_classes=main_config.model.num_classes,
        img_size=main_config.model.img_size,
        **kwargs
    )