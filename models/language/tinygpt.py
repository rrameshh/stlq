# networks/language/tinygpt.py - Tiny GPT with Unified Quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from quantization.layers.all import (
    UnifiedQuantize,
    UnifiedQuantizedLinear,
)
from quantization.quant_config import QuantizationConfig
from quantization.tensors.linear import LinearQuantizedTensor


class GPTAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, dropout=0.1, config=None):
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers (like your ViT)
        self.qkv = UnifiedQuantizedLinear(dim, dim * 3, bias=True, config=config)
        self.proj = UnifiedQuantizedLinear(dim, dim, bias=True, config=config)
        
        # FP32: Lightweight operations
        self.dropout = nn.Dropout(dropout)
        
        # Input quantizer for explicit transition management
        self.input_quantizer = UnifiedQuantize(config=config)
        
        # Causal mask (will be registered as buffer)
        self.register_buffer("causal_mask", None)
    
    def _get_causal_mask(self, seq_len, device):
        """Get or create causal mask"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create causal mask: upper triangular matrix of -inf
            mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
            self.register_buffer("causal_mask", mask.to(device))
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x):
        """
        Causal self-attention forward pass
        Similar flow to your ViT attention but with causal masking
        """
        B, T, C = x.shape  # batch, sequence length, channels
        
        # ============ EXPLICIT TRANSITION 1: FP32 → INT8 ============
        assert isinstance(x, torch.Tensor) and not isinstance(x, LinearQuantizedTensor), \
            "Expected FP32 input from LayerNorm"
        
        # Quantize input for heavy compute
        x_quantized = self.input_quantizer(x)
        
        # ============ QUANTIZED COMPUTE: QKV PROJECTION ============
        qkv_quantized = self.qkv(x_quantized)
        
        # ============ EXPLICIT TRANSITION 2: INT8 → FP32 ============
        if isinstance(qkv_quantized, LinearQuantizedTensor):
            qkv_fp32 = qkv_quantized.dequantize()
        else:
            qkv_fp32 = qkv_quantized
        
        # ============ FP32 COMPUTE: CAUSAL ATTENTION ============
        qkv_reshaped = qkv_fp32.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        # Attention computation with causal masking
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(T, x.device)
        attn_scores = attn_scores + causal_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        
        # ============ EXPLICIT TRANSITION 3: FP32 → INT8 → FP32 ============
        out_quantized = self.input_quantizer(out)
        out_proj_quantized = self.proj(out_quantized)
        
        if isinstance(out_proj_quantized, LinearQuantizedTensor):
            out_fp32 = out_proj_quantized.dequantize()
        else:
            out_fp32 = out_proj_quantized
            
        out_fp32 = self.dropout(out_fp32)
        return out_fp32


class GPTMLP(nn.Module):
    """
    GPT MLP block - same as your ViT MLP but typically larger expansion
    """
    
    def __init__(self, dim, expansion_factor=4, dropout=0.1, config=None):
        super().__init__()
        
        hidden_dim = dim * expansion_factor
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers
        self.fc1 = UnifiedQuantizedLinear(dim, hidden_dim, config=config)
        self.fc2 = UnifiedQuantizedLinear(hidden_dim, dim, config=config)
        
        # FP32: Activation and dropout
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Input quantizer
        self.input_quantizer = UnifiedQuantize(config=config)
    
    def forward(self, x):
        """Same flow as your ViT MLP"""
        # FP32 → INT8
        x_quantized = self.input_quantizer(x)
        
        # INT8 FC1 → FP32
        x_quantized = self.fc1(x_quantized)
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
        
        # FP32 activation
        x_fp32 = self.act(x_fp32)
        x_fp32 = self.dropout(x_fp32)
        
        # FP32 → INT8 → FP32
        x_quantized = self.input_quantizer(x_fp32)
        x_quantized = self.fc2(x_quantized)
        
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
            
        x_fp32 = self.dropout(x_fp32)
        return x_fp32


class GPTBlock(nn.Module):
    """
    GPT Transformer Block - similar to your ViT block
    Uses pre-norm (like GPT-2) instead of post-norm
    """
    
    def __init__(self, dim, num_heads, expansion_factor=4, dropout=0.1, config=None):
        super().__init__()
        
        self.config = config
        
        # FP32: LayerNorm (same as ViT)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Quantized attention and MLP (same flow as ViT)
        self.attn = GPTAttention(dim, num_heads=num_heads, dropout=dropout, config=config)
        self.mlp = GPTMLP(dim, expansion_factor=expansion_factor, dropout=dropout, config=config)
    
    def forward(self, x):
        """
        Pre-norm transformer block (GPT-2 style):
        x = x + attn(ln(x))
        x = x + mlp(ln(x))
        """
        # Attention block with pre-norm
        normed_x1 = self.ln1(x)
        attn_out = self.attn(normed_x1)
        x = x + attn_out
        
        # MLP block with pre-norm
        normed_x2 = self.ln2(x)
        mlp_out = self.mlp(normed_x2)
        x = x + mlp_out
        
        return x


class TinyGPT(nn.Module):
    """
    Tiny GPT model with unified quantization
    Decoder-only transformer for language modeling
    """
    
    def __init__(self, vocab_size=50257, max_seq_len=1024, dim=384, depth=6,
                 num_heads=6, expansion_factor=4, dropout=0.1, config=None):
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim

        print(f"🔍 TinyGPT DEBUG: Creating model with vocab_size={vocab_size}")
        
        # ============ FP32 EMBEDDINGS (like ViT patch embedding) ============
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # ============ QUANTIZED TRANSFORMER BLOCKS ============
        self.blocks = nn.ModuleList([
            GPTBlock(dim, num_heads, expansion_factor, dropout, config)
            for _ in range(depth)
        ])
        
        # ============ FP32 OUTPUT LAYERS ============
        self.ln_final = nn.LayerNorm(dim)
        
        # Language modeling head
        quantize_classifier = getattr(config, 'quantize_classifier', False)
        if quantize_classifier:
            self.lm_head = UnifiedQuantizedLinear(dim, vocab_size, config=config)
            self.head_quantizer = UnifiedQuantize(config=config)
        else:
            # Tie weights with token embedding (standard practice)
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following GPT-2 initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass for language modeling
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            targets: [batch_size, seq_len] target tokens for loss computation
            
        Returns:
            logits or (logits, loss)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        # ============ FP32 EMBEDDING PHASE ============
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)  # [B, T, dim]
        pos_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.position_embedding(pos_ids)   # [T, dim]
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # ============ QUANTIZED TRANSFORMER BLOCKS ============
        for block in self.blocks:
            x = block(x)
        
        # ============ FP32 OUTPUT PHASE ============
        x = self.ln_final(x)
        
        # Language modeling head
        if hasattr(self, 'head_quantizer'):
            # Quantized head
            x_quantized = self.head_quantizer(x)
            logits = self.lm_head(x_quantized)
            if isinstance(logits, LinearQuantizedTensor):
                logits = logits.dequantize()
        else:
            # FP32 head (weight-tied)
            logits = self.lm_head(x)
        
        if targets is not None:
            # Compute loss (shift targets for next-token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_targets.view(-1), ignore_index=-1)
            return logits, loss
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Simple autoregressive generation
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                logits = self(input_ids)
                
                # Focus on last token's logits
                logits = logits[:, -1, :] / temperature
                
                # Optional top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check if we need to truncate (sliding window)
                if input_ids.size(1) > self.max_seq_len:
                    input_ids = input_ids[:, 1:]  # Remove first token
        
        return input_ids


# Factory functions
def create_tiny_gpt(variant="nano", quantization_method="linear", **kwargs):
    """
    Create Tiny GPT variants
    """
    
    configs = {
        "nano": {"dim": 192, "depth": 4, "num_heads": 3, "max_seq_len": 256},    # ~2M params
        "micro": {"dim": 256, "depth": 6, "num_heads": 4, "max_seq_len": 512},   # ~8M params  
        "mini": {"dim": 384, "depth": 6, "num_heads": 6, "max_seq_len": 1024},   # ~24M params
        "small": {"dim": 512, "depth": 8, "num_heads": 8, "max_seq_len": 1024},  # ~52M params
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
    
    return TinyGPT(config=config, **model_config)


# Convenient factory functions
def tiny_gpt_nano(**kwargs):
    return create_tiny_gpt("nano", **kwargs)

def tiny_gpt_micro(**kwargs):
    return create_tiny_gpt("micro", **kwargs)

def tiny_gpt_mini(**kwargs):
    return create_tiny_gpt("mini", **kwargs)

def tiny_gpt_small(**kwargs):
    return create_tiny_gpt("small", **kwargs)