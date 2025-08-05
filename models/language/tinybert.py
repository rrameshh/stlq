# models/language/tinybert.py - TinyBERT with Unified Quantization
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from quantization.layers.all import (
    Quantizer,
    QLinear,
)
from quantization.quant_config import QuantizationConfig
from quantization.tensors.linear import LinearQuantizedTensor


class BERTEmbeddings(nn.Module):
    """
    BERT embeddings: token + position + token_type embeddings
    Kept in FP32 for precision (industry standard)
    """
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, 
                 type_vocab_size=2, dropout=0.1):
        super().__init__()
        
        # All embeddings in FP32
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        # Layer norm and dropout (FP32)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Pre-compute position ids
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        token_emb = self.token_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = token_emb + position_emb + token_type_emb
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTAttention(nn.Module):
    """
    BERT Multi-head self-attention - similar to your ViT/TinyGPT attention
    but without causal masking (bidirectional)
    """
    
    def __init__(self, hidden_size, num_heads=8, dropout=0.1, config=None):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers (like your ViT)
        self.qkv = QLinear(hidden_size, hidden_size * 3, bias=True, config=config)
        self.out_proj = QLinear(hidden_size, hidden_size, bias=True, config=config)
        
        # FP32: Lightweight operations
        self.dropout = nn.Dropout(dropout)
        
        # Input quantizer for explicit transition management
        self.input_quantizer = Quantizer(config=config)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        BERT self-attention (bidirectional, unlike TinyGPT's causal attention)
        """
        B, T, C = hidden_states.shape
        
        # ============ EXPLICIT TRANSITION 1: FP32 → INT8 ============
        assert isinstance(hidden_states, torch.Tensor) and not isinstance(hidden_states, LinearQuantizedTensor), \
            "Expected FP32 input from LayerNorm"
        
        # Quantize input for heavy compute
        x_quantized = self.input_quantizer(hidden_states)
        
        # ============ QUANTIZED COMPUTE: QKV PROJECTION ============
        qkv_quantized = self.qkv(x_quantized)
        
        # ============ EXPLICIT TRANSITION 2: INT8 → FP32 ============
        if isinstance(qkv_quantized, LinearQuantizedTensor):
            qkv_fp32 = qkv_quantized.dequantize()
        else:
            qkv_fp32 = qkv_quantized
        
        # ============ FP32 COMPUTE: SELF-ATTENTION ============
        qkv_reshaped = qkv_fp32.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_reshaped[0], qkv_reshaped[1], qkv_reshaped[2]
        
        # Attention computation (bidirectional - no causal mask)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        
        # Apply attention mask if provided (for padding tokens)
        if attention_mask is not None:
            # Convert attention mask to attention scores mask
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attn_scores = attn_scores + extended_attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = attn_weights @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        
        # ============ EXPLICIT TRANSITION 3: FP32 → INT8 → FP32 ============
        out_quantized = self.input_quantizer(out)
        out_proj_quantized = self.out_proj(out_quantized)
        
        if isinstance(out_proj_quantized, LinearQuantizedTensor):
            out_fp32 = out_proj_quantized.dequantize()
        else:
            out_fp32 = out_proj_quantized
            
        out_fp32 = self.dropout(out_fp32)
        return out_fp32


class BERTMLP(nn.Module):
    """
    BERT MLP block - same flow as your ViT/TinyGPT MLP
    """
    
    def __init__(self, hidden_size, intermediate_size, dropout=0.1, config=None):
        super().__init__()
        
        self.config = config
        
        # QUANTIZED: Heavy compute linear layers
        self.dense1 = QLinear(hidden_size, intermediate_size, config=config)
        self.dense2 = QLinear(intermediate_size, hidden_size, config=config)
        
        # FP32: Activation and dropout
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Input quantizer
        self.input_quantizer = Quantizer(config=config)
    
    def forward(self, hidden_states):
        """Same flow as your ViT/TinyGPT MLP"""
        # FP32 → INT8
        x_quantized = self.input_quantizer(hidden_states)
        
        # INT8 dense1 → FP32
        x_quantized = self.dense1(x_quantized)
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
        
        # FP32 activation
        x_fp32 = self.act(x_fp32)
        x_fp32 = self.dropout(x_fp32)
        
        # FP32 → INT8 → FP32
        x_quantized = self.input_quantizer(x_fp32)
        x_quantized = self.dense2(x_quantized)
        
        if isinstance(x_quantized, LinearQuantizedTensor):
            x_fp32 = x_quantized.dequantize()
        else:
            x_fp32 = x_quantized
            
        x_fp32 = self.dropout(x_fp32)
        return x_fp32


class BERTLayer(nn.Module):
    """
    BERT Transformer Layer - similar to your ViT transformer block
    """
    
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1, config=None):
        super().__init__()
        
        self.config = config
        
        # FP32: LayerNorm (industry standard)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
        
        # Quantized attention and MLP
        self.attention = BERTAttention(hidden_size, num_heads, dropout, config)
        self.mlp = BERTMLP(hidden_size, intermediate_size, dropout, config)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        BERT layer forward pass with explicit quantization flow
        """
        # Ensure input is FP32
        assert isinstance(hidden_states, torch.Tensor) and not isinstance(hidden_states, LinearQuantizedTensor), \
            "BERT layer expects FP32 input"
        
        # ============ ATTENTION BLOCK ============
        # FP32 LayerNorm → FP32
        normed_states = self.attention_norm(hidden_states)
        
        # FP32 → Attention (handles quantization internally) → FP32
        attn_out = self.attention(normed_states, attention_mask)
        
        # FP32 residual connection
        hidden_states = hidden_states + attn_out
        
        # ============ MLP BLOCK ============
        # FP32 LayerNorm → FP32
        normed_states = self.mlp_norm(hidden_states)
        
        # FP32 → MLP (handles quantization internally) → FP32
        mlp_out = self.mlp(normed_states)
        
        # FP32 residual connection
        hidden_states = hidden_states + mlp_out
        
        return hidden_states


class BERTPooler(nn.Module):
    """
    BERT pooler - extracts [CLS] token representation for classification
    """
    
    def __init__(self, hidden_size, config=None):
        super().__init__()
        
        self.config = config
        
        # Classification head choice
        quantize_classifier = getattr(config, 'quantize_classifier', False)
        if quantize_classifier:
            self.dense = QLinear(hidden_size, hidden_size, config=config)
            self.pooler_quantizer = Quantizer(config=config)
        else:
            # Industry standard: Keep pooler in FP32
            self.dense = nn.Linear(hidden_size, hidden_size)
        
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # Extract [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        
        if hasattr(self, 'pooler_quantizer'):
            # Quantized pooler
            first_token_quantized = self.pooler_quantizer(first_token_tensor)
            pooled_output = self.dense(first_token_quantized)
            
            if isinstance(pooled_output, LinearQuantizedTensor):
                pooled_output = pooled_output.dequantize()
        else:
            # FP32 pooler
            pooled_output = self.dense(first_token_tensor)
        
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TinyBERT(nn.Module):
    """
    TinyBERT model with unified quantization
    Encoder-only transformer for classification and understanding tasks
    """
    
    def __init__(self, vocab_size=30522, hidden_size=384, num_layers=4, num_heads=6, 
                 intermediate_size=1536, max_position_embeddings=512, type_vocab_size=2,
                 dropout=0.1, num_classes=2, config=None):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # ============ FP32 EMBEDDINGS ============
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout
        )
        
        # ============ QUANTIZED TRANSFORMER LAYERS ============
        self.encoder_layers = nn.ModuleList([
            BERTLayer(hidden_size, num_heads, intermediate_size, dropout, config)
            for _ in range(num_layers)
        ])
        
        # ============ POOLING AND CLASSIFICATION ============
        self.pooler = BERTPooler(hidden_size, config)
        
        # Classification head
        quantize_classifier = getattr(config, 'quantize_classifier', False)
        if quantize_classifier:
            self.classifier = QLinear(hidden_size, num_classes, config=config)
            self.classifier_quantizer = Quantizer(config=config)
        else:
            # Industry standard: Keep classifier in FP32
            self.classifier = nn.Linear(hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        
        # Dropout for classifier
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following BERT initialization"""
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
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass for classification tasks
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            attention_mask: [batch_size, seq_len] attention mask (1 for real tokens, 0 for padding)
            token_type_ids: [batch_size, seq_len] token type ids (for sentence pairs)
            labels: [batch_size] labels for classification (optional)
            
        Returns:
            If labels provided: (logits, loss)
            Else: logits
        """
        
        # ============ FP32 EMBEDDING PHASE ============
        embedded = self.embeddings(input_ids, token_type_ids)
        
        # ============ QUANTIZED TRANSFORMER LAYERS ============
        hidden_states = embedded
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # ============ POOLING AND CLASSIFICATION ============
        # Extract [CLS] representation
        pooled_output = self.pooler(hidden_states)
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        if hasattr(self, 'classifier_quantizer'):
            # Quantized classifier
            pooled_quantized = self.classifier_quantizer(pooled_output)
            logits = self.classifier(pooled_quantized)
            if isinstance(logits, LinearQuantizedTensor):
                logits = logits.dequantize()
        else:
            # FP32 classifier
            logits = self.classifier(pooled_output)
        
        if labels is not None:
            # Compute loss
            if self.num_classes == 1:
                # Regression
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                # Classification
                loss = F.cross_entropy(logits, labels)
            return logits, loss
        
        return logits


# Factory functions for different TinyBERT sizes
def create_tiny_bert(variant="tiny", quantization_method="linear", **kwargs):
    """
    Create TinyBERT variants
    """
    
    configs = {
        # Based on TinyBERT paper configurations
        "tiny": {"hidden_size": 128, "num_layers": 2, "num_heads": 2, "intermediate_size": 512},      # ~1M params
        "mini": {"hidden_size": 256, "num_layers": 4, "num_heads": 4, "intermediate_size": 1024},     # ~4M params  
        "small": {"hidden_size": 384, "num_layers": 4, "num_heads": 6, "intermediate_size": 1536},    # ~8M params
        "base": {"hidden_size": 512, "num_layers": 6, "num_heads": 8, "intermediate_size": 2048},     # ~20M params
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
    
    return TinyBERT(config=config, **model_config)


# Convenient factory functions
def tiny_bert_tiny(**kwargs):
    return create_tiny_bert("tiny", **kwargs)

def tiny_bert_mini(**kwargs):
    return create_tiny_bert("mini", **kwargs)

def tiny_bert_small(**kwargs):
    return create_tiny_bert("small", **kwargs)

def tiny_bert_base(**kwargs):
    return create_tiny_bert("base", **kwargs)

