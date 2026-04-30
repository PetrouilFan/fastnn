"""Attention mechanisms for LLM models.

Provides implementations for:
- GQAAttention: Grouped Query Attention (Llama, LFM2.5)
- FullAttention: Standard full attention
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np
import fastnn._core as _core
from fastnn import Linear, matmul, softmax
from fastnn.models.llm.base import LLMConfig, LLMBlock


class BaseAttention(ABC):
    """Base class for attention mechanisms."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (1.0 / (head_dim ** 0.5))
        self.repeat_factor = num_heads // num_kv_heads


class GQAAttention(LLMBlock):
    """Grouped Query Attention (GQA) implementation.
    
    Used by Llama, LFM2.5, and other modern LLMs.
    GQA reduces KV cache by sharing keys/values across query heads.
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv = config.num_key_value_heads
        head_dim = config.head_dim
        
        # Q projection: hidden -> hidden (all heads)
        self.q_proj = Linear(hidden, hidden, bias=False)
        # K projection: hidden -> num_kv * head_dim
        self.k_proj = Linear(hidden, num_kv * head_dim, bias=False)
        # V projection: hidden -> num_kv * head_dim
        self.v_proj = Linear(hidden, num_kv * head_dim, bias=False)
        # O projection: hidden -> hidden
        self.o_proj = Linear(hidden, hidden, bias=False)
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        self.repeat_factor = num_heads // num_kv
        
        # Causal mask
        self._causal_mask = None
    
    def forward(self, x, position_ids=None, cos_cache=None, sin_cache=None, **kwargs):
        """Forward pass with GQA.
        
        Args:
            x: [batch, seq_len, hidden_size]
            position_ids: Optional position IDs
            cos_cache, sin_cache: Optional RoPE caches
            
        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V - these return 3D tensors
        q = self.q_proj(x)  # [batch, seq_len, hidden]
        k = self.k_proj(x)  # [batch, seq_len, num_kv * head_dim]
        v = self.v_proj(x)  # [batch, seq_len, num_kv * head_dim]
        
        # Convert to numpy for reshape/permute operations, then back
        # This avoids the 4D tensor issue with fastnn Linear
        q_raw = q.numpy()
        k_raw = k.numpy()
        v_raw = v.numpy()
        
        q_np = q_raw.reshape([batch, seq_len, self.num_heads, self.head_dim])
        q_np = q_np.transpose(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]
        
        k_np = k_raw.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        k_np = k_np.transpose(0, 2, 1, 3)  # [batch, num_kv, seq_len, head_dim]
        
        v_np = v_raw.reshape([batch, seq_len, self.num_kv_heads, self.head_dim])
        v_np = v_np.transpose(0, 2, 1, 3)  # [batch, num_kv, seq_len, head_dim]
        
        # Apply RoPE if caches provided
        if cos_cache is not None and sin_cache is not None:
            q_np, k_np = self._apply_rope_np(q_np, k_np, cos_cache, sin_cache)
        
        # Repeat K, V for GQA
        if self.repeat_factor > 1:
            k_np = np.repeat(k_np, self.repeat_factor, axis=1)
            v_np = np.repeat(v_np, self.repeat_factor, axis=1)
        
        # Compute attention scores using numpy
        attn_scores = np.einsum("bhqd,bhkd->bhqk", q_np, k_np) * self.scale
        
        # Causal mask
        mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        attn_scores = attn_scores + mask
        
        # Stable softmax: subtract max before exp
        attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
        attn_scores_shifted = attn_scores - attn_scores_max
        exp_scores = np.exp(attn_scores_shifted)
        attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
        
        # Apply to V
        context = np.einsum("bhqk,bhkd->bhqd", attn_weights, v_np)
        
        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        context = context.transpose(0, 2, 1, 3).reshape([batch, seq_len, self.config.hidden_size])
        
        # Convert back to fastnn tensor for output projection
        context = _core.tensor_from_list(context.flatten().tolist(), list(context.shape))
        
        # Output projection
        return self.o_proj(context)
    
    def __call__(self, x, position_ids=None, cos_cache=None, sin_cache=None, **kwargs):
        return self.forward(x, position_ids, cos_cache, sin_cache, **kwargs)
    
    def _apply_rope_np(self, q, k, cos, sin):
        """Apply rotary position embeddings using numpy.
        
        Args:
            q, k: [batch, heads, seq_len, head_dim]
            cos, sin: [batch, 1, head_dim] (after compute)
        """
        # q, k are numpy arrays (converted in forward method)
        # cos, sin may be tensors or arrays
        cos_np = cos.numpy() if hasattr(cos, 'numpy') else cos
        sin_np = sin.numpy() if hasattr(sin, 'numpy') else sin
        
        # cos_np/sin_np have shape (batch, 1, head_dim) or (batch, heads, head_dim)
        # Need to reshape for broadcasting: (batch, 1, 1, head_dim) to match (batch, heads, seq_len, head_dim)
        if cos_np.ndim == 3:
            # (batch, 1, head_dim) -> (batch, 1, 1, head_dim)
            cos_expanded = cos_np[:, np.newaxis, :, :]
            sin_expanded = sin_np[:, np.newaxis, :, :]
        else:
            # Already has right shape
            cos_expanded = cos_np
            sin_expanded = sin_np
        
        # Split head dim in half
        head_dim = self.head_dim
        half_dim = head_dim // 2
        
        # Apply to Q
        q1 = q[..., :half_dim]
        q2 = q[..., half_dim:]
        q_rotated = np.concatenate([q2 * -1, q1], axis=-1)
        q_out = q * cos_expanded + q_rotated * sin_expanded
        
        # Apply to K
        k1 = k[..., :half_dim]
        k2 = k[..., half_dim:]
        k_rotated = np.concatenate([k2 * -1, k1], axis=-1)
        k_out = k * cos_expanded + k_rotated * sin_expanded
        
        # Return numpy arrays (NOT tensors)
        return q_out, k_out
    
    def _repeat_kv(self, x):
        """Repeat KV heads for GQA.
        
        Args:
            x: [batch, num_kv_heads, seq_len, head_dim]
            
        Returns:
            x: [batch, num_heads, seq_len, head_dim]
        """
        if self.repeat_factor == 1:
            return x
        return np.repeat(x, self.repeat_factor, axis=1)
    
    def parameters(self):
        return (
            self.q_proj.parameters() +
            self.k_proj.parameters() +
            self.v_proj.parameters() +
            self.o_proj.parameters()
        )
    
    def named_parameters(self):
        params = []
        prefix = f"layers.{self.layer_idx}.self_attn."
        
        for name, p in self.q_proj.named_parameters():
            params.append((f"{prefix}q_proj.{name}", p))
        for name, p in self.k_proj.named_parameters():
            params.append((f"{prefix}k_proj.{name}", p))
        for name, p in self.v_proj.named_parameters():
            params.append((f"{prefix}v_proj.{name}", p))
        for name, p in self.o_proj.named_parameters():
            params.append((f"{prefix}o_proj.{name}", p))
        
        return params
