"""RoPE (Rotary Position Embedding) utilities."""

import numpy as np
import fastnn._core as _core
from fastnn import tensor_from_array
from typing import Optional


class RoPECache:
    """Precomputed RoPE (Rotary Position Embedding) cache.
    
    Generates cos/sin arrays for rotary position embeddings.
    Used by Llama, LFM, and other modern LLMs.
    """
    
    def __init__(
        self, 
        config, 
        max_seq_len: int = None
    ):
        self.config = config
        self.max_seq_len = max_seq_len or config.max_position_embeddings
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        
        # Cache for computed values
        self._cos_cache: Optional[_core.Tensor] = None
        self._sin_cache: Optional[_core.Tensor] = None
        self._cached_seq_len = 0
    
    def compute(self, seq_len: int):
        """Compute RoPE cache for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            (cos, sin) numpy arrays
        """
        if seq_len > self._cached_seq_len:
            self._compute_cache(seq_len)
        
        # Return numpy arrays - full cache
        cos_np = self._cos_cache.numpy()[:1, :seq_len, :]
        sin_np = self._sin_cache.numpy()[:1, :seq_len, :]
        
        return cos_np, sin_np
    
    def _compute_cache(self, seq_len: int):
        """Compute and cache cos/sin values."""
        seq_len = min(seq_len, self.max_seq_len)
        
        # Build frequency positions
        positions = np.arange(seq_len, dtype=np.float32)
        
        # Compute inv_freq: 1 / (rope_theta ^ (2i/d))
        inv_freq = 1.0 / (
            self.rope_theta ** (
                np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim
            )
        )
        
        # Compute angles: positions * inv_freq
        angles = positions[:, None] * inv_freq[None, :]
        
        # Compute cos/sin
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)
        
        # Expand to full head_dim by interleaving
        head_dim = self.head_dim
        cos_full = np.zeros((seq_len, head_dim), dtype=np.float32)
        sin_full = np.zeros((seq_len, head_dim), dtype=np.float32)
        
        # Interleave: [cos0, cos0, cos1, cos1, ...]
        cos_full[:, ::2] = cos
        cos_full[:, 1::2] = cos
        sin_full[:, ::2] = sin
        sin_full[:, 1::2] = sin
        
        # Add batch dimension for broadcasting
        cos_full = cos_full[np.newaxis, :, :]  # [1, seq_len, head_dim]
        sin_full = sin_full[np.newaxis, :, :]
        
        # Convert to fastnn tensors
        self._cos_cache = tensor_from_array(cos_full)
        self._sin_cache = tensor_from_array(sin_full)
        self._cached_seq_len = seq_len
    
    def apply(self, x, position_ids=None):
        """Apply RoPE to input tensor.
        
        Args:
            x: Query or Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position IDs
            
        Returns:
            RoPE-applied tensor
        """
        seq_len = x.shape[2]
        cos, sin = self.compute(seq_len)
        
        # Apply using numpy (for now)
        x_np = x.numpy()
        cos_np = cos.numpy()
        sin_np = sin.numpy()
        
        # Split head dim in half
        half_dim = self.head_dim // 2
        
        x1 = x_np[..., :half_dim]
        x2 = x_np[..., half_dim:]
        
        # Rotate: [-x2, x1]
        x_rotated = np.concatenate([x2 * -1, x1], axis=-1)
        
        # Apply: x * cos + x_rotated * sin
        x_out = x_np * cos_np + x_rotated * sin_np
        
        return tensor_from_array(x_out)


def create_rope_cache(config, max_seq_len: int = None) -> RoPECache:
    """Factory function to create RoPE cache.
    
    Args:
        config: Model config
        max_seq_len: Optional max sequence length override
        
    Returns:
        RoPECache instance
    """
    return RoPECache(config, max_seq_len)