# networks/unified_swin.py - FIXED Swin Transformer Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

from ops.layers.all import (
    UnifiedQuantize,
    UnifiedQuantizedLinear,
)
from ops.quant_config import QuantizationConfig
from ops.tensors.linear import LinearQuantizedTensor


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows of given size
    FIXED: Better handling of padding and dimensions
    """
    B, H, W, C = x.shape
    
    # Calculate padding needed
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    
    # Reshape to windows
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, H, W

def window_reverse(windows, window_size, H, W, original_H, original_W):
    """
    Reverse window partition and remove padding
    FIXED: Use padded dimensions for calculation, original dimensions for cropping
    """
    # Calculate padded dimensions (what was actually used in window_partition)
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    padded_H = H + pad_h
    padded_W = W + pad_w
    
    # Use padded dimensions for the reverse operation
    num_windows_h = padded_H // window_size
    num_windows_w = padded_W // window_size
    
    # Calculate batch size using padded dimensions
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    
    # Debug info for troubleshooting
    if windows.shape[0] % (num_windows_h * num_windows_w) != 0:
        print(f"🚨 window_reverse error:")
        print(f"   windows.shape: {windows.shape}")
        print(f"   window_size: {window_size}")
        print(f"   Original H, W: {H}, {W}")
        print(f"   Padded H, W: {padded_H}, {padded_W}")
        print(f"   num_windows_h, num_windows_w: {num_windows_h}, {num_windows_w}")
        print(f"   Expected total windows: {num_windows_h * num_windows_w}")
        print(f"   Actual num windows: {windows.shape[0]}")
        print(f"   Calculated B: {B}")
        raise RuntimeError("Window count mismatch in window_reverse")
    
    # Reshape using padded dimensions
    C = windows.shape[-1]
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, padded_H, padded_W, C)
    
    # Crop to original dimensions (remove padding)
    x = x[:, :original_H, :original_W, :]
    
    return x



class WindowAttention(nn.Module):

    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, config=None):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size  # Can be tuple (H, W) or int
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Handle window_size as tuple or int
        if isinstance(window_size, int):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size
            
        self.qkv = UnifiedQuantizedLinear(dim, dim * 3, bias=qkv_bias, config=config)
        self.proj = UnifiedQuantizedLinear(dim, dim, config=config)
        self.input_quantizer = UnifiedQuantize(config=config)
        
        # Relative position bias with proper window size handling
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        
        # Initialize relative position index properly
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x, mask=None):

        B_, N, C = x.shape
        
        # ============ SAME AS VIT - QUANTIZED ATTENTION ============
        x_quantized = self.input_quantizer(x)
        qkv_quantized = self.qkv(x_quantized)
        
        if isinstance(qkv_quantized, LinearQuantizedTensor):
            qkv = qkv_quantized.dequantize()
        else:
            qkv = qkv_quantized
        
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # ============ SWIN-SPECIFIC: ADD RELATIVE POSITION BIAS ============
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided (for shifted windows)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # ============ SAME AS VIT - QUANTIZED PROJECTION ============
        x_quantized = self.input_quantizer(x)
        x = self.proj(x_quantized)
        
        if isinstance(x, LinearQuantizedTensor):
            x = x.dequantize()
        
        return x


class SwinTransformerBlock(nn.Module):
    """FIXED: Swin block with proper dimension handling"""
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, 
                 qkv_bias=True, drop=0.0, config=None):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
            print(f"🔧 Block adjusted window_size to {self.window_size} for resolution {self.input_resolution}")
            
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        # REUSE YOUR VIT COMPONENTS!
        self.norm1 = nn.LayerNorm(dim)
        # CRITICAL FIX: Use the adjusted window_size for attention
        self.attn = WindowAttention(dim, self.window_size, num_heads, qkv_bias, config)
        
        self.norm2 = nn.LayerNorm(dim)
        # REUSE YOUR VIT MLP DIRECTLY!
        from .unified_vit import SelectiveQuantizedMLP
        self.mlp = SelectiveQuantizedMLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            drop=drop, 
            config=config
        )
        
        # FIXED: Create attention mask for shifted windows with correct dimensions
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                       slice(-self.window_size, -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            # mask_windows = window_partition(img_mask, self.window_size)
            mask_windows, _, _ = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        """FIXED: Forward pass with proper dimension handling"""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: expected {H * W}, got {L}"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows - get padded dimensions
        x_windows, padded_H, padded_W = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, window_size*window_size, C]
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [nW*B, window_size*window_size, C]
        
        # Merge windows - use padded dimensions for reverse, original for crop
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, padded_H, padded_W, H, W)  # [B, H', W', C]
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)
        x = x.reshape(B, H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x
    

class PatchMerging(nn.Module):
    
    def __init__(self, input_resolution, dim, config=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = UnifiedQuantizedLinear(4 * dim, 2 * dim, bias=False, config=config)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x):
        """
        x: B, H*W, C
        FIXED: Handle odd dimensions by padding
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: expected {H * W}, got {L}"
        
        x = x.view(B, H, W, C)
        
        # FIXED: Pad if dimensions are odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # Pad the last row/column if needed
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            # Update H, W to padded dimensions
            H = H + pad_h
            W = W + pad_w
        
        # FIXED: Merge 2x2 patches properly with guaranteed even dimensions
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        if isinstance(x, LinearQuantizedTensor):
            x = x.dequantize()
        
        return x
class BasicLayer(nn.Module):
    """FIXED: Basic layer with proper resolution tracking and odd dimension handling"""
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4., qkv_bias=True, drop=0., downsample=None, config=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                config=config
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, config=config)
            # FIXED: Calculate output resolution after downsampling with padding consideration
            H, W = input_resolution
            # Account for potential padding in PatchMerging
            padded_H = H + (H % 2)  # Add 1 if odd
            padded_W = W + (W % 2)  # Add 1 if odd
            self.output_resolution = [padded_H // 2, padded_W // 2]
        else:
            self.downsample = None
            self.output_resolution = input_resolution
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x

class SwinTransformer(nn.Module):
    """FIXED: Swin Transformer with proper initialization and dimension tracking"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                    embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                    window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0., 
                    attn_drop_rate=0., drop_path_rate=0.1, config=None):
            super().__init__()

            print(f"🔍 Swin Init Debug:")
            print(f"  img_size: {img_size}")
            print(f"  patch_size: {patch_size}")
        
            self.num_classes = num_classes
            self.num_layers = len(depths)
            self.embed_dim = embed_dim
            self.mlp_ratio = mlp_ratio
            self.window_size = window_size
            self.img_size = img_size
            self.patch_size = patch_size
            
            # CRITICAL FIX: Calculate actual patch resolution
            self.patches_resolution = [img_size // patch_size, img_size // patch_size]
            print(f"  calculated patches_resolution: {self.patches_resolution}")
            
            # Auto-adjust window size for small images
            if window_size > min(self.patches_resolution):
                window_size = min(self.patches_resolution)
                self.window_size = window_size
                print(f"Swin Init: img_size={img_size}, patch_size={patch_size}, patches_resolution={self.patches_resolution}, window_size={window_size}")
            
            # Patch embedding
            self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_drop = nn.Dropout(p=drop_rate)
            
            # Build layers with CORRECT resolution tracking
            self.layers = nn.ModuleList()
            current_resolution = self.patches_resolution.copy()  # Start with actual patch resolution
            
            for i_layer in range(self.num_layers):
                layer_dim = int(embed_dim * 2 ** i_layer)
                
                # Adjust window size for each layer based on its resolution
                layer_window_size = min(window_size, min(current_resolution))
                print(f"Layer {i_layer}: resolution={current_resolution}, window_size={layer_window_size}")
                
                # ADDITIONAL FIX: For CIFAR-10, be more conservative with window sizes
                if max(current_resolution) <= 8:  # Small resolutions like CIFAR-10
                    layer_window_size = min(layer_window_size, 4)  # Cap at 4 for small images
                    print(f"Small image detected, capped window_size to {layer_window_size}")
                
                layer = BasicLayer(
                    dim=layer_dim,
                    input_resolution=current_resolution.copy(),  # Use current resolution
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=layer_window_size,  # Use adjusted window size for this layer
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    config=config
                )
                self.layers.append(layer)
                
                # Update resolution for next layer (after potential downsampling)
                if i_layer < self.num_layers - 1:  # Has downsampling
                    current_resolution = layer.output_resolution.copy()

            print("Layer dimensions:")
            for i, layer in enumerate(self.layers):
                layer_dim = int(embed_dim * 2 ** i)
                print(f"  Layer {i}: dim={layer_dim}")
                
                if hasattr(layer, 'downsample') and layer.downsample:
                    if hasattr(layer.downsample, 'reduction'):
                        if hasattr(layer.downsample.reduction, 'linear'):
                            print(f"    Downsample: {layer.downsample.reduction.linear.in_features} → {layer.downsample.reduction.linear.out_features}")
                        else:
                            print(f"    Downsample: {layer.downsample.reduction.in_features} → {layer.downsample.reduction.out_features}")
            
            self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
            
            # Classification head
            final_dim = int(embed_dim * 2 ** (self.num_layers - 1))
            quantize_classifier = getattr(config, 'quantize_classifier', False)
            if quantize_classifier:
                self.head = UnifiedQuantizedLinear(final_dim, num_classes, config=config)
                self.head_quantizer = UnifiedQuantize(config=config)
            else:
                self.head = nn.Linear(final_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.pos_drop(x)
        
        # Apply Swin Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # [B, H*W, C]
        x = x.mean(dim=1)  # Global average pooling [B, C]
        
        # Classification head
        if hasattr(self, 'head_quantizer'):
            # Quantized classifier
            x_quantized = self.head_quantizer(x)
            x = self.head(x_quantized)
            if isinstance(x, LinearQuantizedTensor):
                x = x.dequantize()
        else:
            # FP32 classifier
            x = self.head(x)
        
        return x
    
def create_swin_transformer(variant="tiny", quantization_method="linear", **kwargs):
    """Create Swin Transformer variants with PROTECTED configurations"""
    
    configs = {
        "tiny": {
            "embed_dim": 96, 
            "depths": [2, 2, 6, 2], 
            "num_heads": [3, 6, 12, 24],
            "window_size": 7
        },
        "small": {
            "embed_dim": 96, 
            "depths": [2, 2, 18, 2], 
            "num_heads": [3, 6, 12, 24],
            "window_size": 7
        },
        "base": {
            "embed_dim": 128, 
            "depths": [2, 2, 18, 2], 
            "num_heads": [4, 8, 16, 32],
            "window_size": 7
        },
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    # Extract config parameters
    device = kwargs.pop('device', 'cuda:0')
    threshold = kwargs.pop('threshold', 1e-5)
    momentum = kwargs.pop('momentum', 0.1)
    bits = kwargs.pop('bits', 8)
    quantize_classifier = kwargs.pop('quantize_classifier', False)
    
    # Handle img_size properly
    img_size = kwargs.get('img_size', 224)
    print(f"Swin factory: img_size={img_size} from kwargs={kwargs.get('img_size')}")
    
    from ops.quant_config import QuantizationConfig
    config = QuantizationConfig(
        method=quantization_method,
        momentum=momentum,
        device=device,
        threshold=threshold,
        bits=bits
    )
    config.quantize_classifier = quantize_classifier
    
    model_config = configs[variant].copy()  # Make a copy first
    
    # Only allow safe parameters to be overridden
    safe_kwargs = {
        'img_size': kwargs.get('img_size', 224),
        'patch_size': kwargs.get('patch_size', 4),
        'in_chans': kwargs.get('in_chans', 3),
        'num_classes': kwargs.get('num_classes', 1000),
        'drop_rate': kwargs.get('drop_rate', 0.0),
        'attn_drop_rate': kwargs.get('attn_drop_rate', 0.0),
        'mlp_ratio': kwargs.get('mlp_ratio', 4.0),
        'qkv_bias': kwargs.get('qkv_bias', True)
    }
    
    # Update with only safe parameters
    model_config.update(safe_kwargs)
    
    print(f"Final Swin config: depths={model_config['depths']}, num_heads={model_config['num_heads']}")
    
    return SwinTransformer(config=config, **model_config)


def swin_tiny(quantization_method="linear", **kwargs):
    return create_swin_transformer("tiny", quantization_method, **kwargs)

def swin_small(quantization_method="linear", **kwargs):
    return create_swin_transformer("small", quantization_method, **kwargs)

def swin_base(quantization_method="linear", **kwargs):
    return create_swin_transformer("base", quantization_method, **kwargs)