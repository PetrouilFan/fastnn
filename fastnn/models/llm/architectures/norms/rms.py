"""Normalization layers for LLM models.

Provides:
- RMSNorm: Root Mean Square Normalization (Llama, LFM)
- LayerNorm: Standard Layer Normalization
"""

import numpy as np
import fastnn._core as _core
from fastnn.models.llm.base import LLMConfig, LLMBlock


class RMSNorm(LLMBlock):
    """RMSNorm (Root Mean Square Normalization).
    
    Used by Llama, LFM, and other modern LLMs.
    Formula: x / sqrt(mean(x^2)) * weight
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int, normalized_dim: int = None):
        super().__init__(config, layer_idx)
        
        self.normalized_dim = normalized_dim or config.hidden_size
        self.eps = config.rms_norm_eps
        
        # Weight for RMSNorm
        self.weight = _core.ones([self.normalized_dim])
    
    def forward(self, x, **kwargs):
        """RMSNorm forward pass."""
        # Use explicit dim based on tensor rank (avoid dim=-1 bug in fastnn)
        rank = len(x.shape)
        dim = rank - 1  # Last dimension
        
        x_sq = x * x
        mean_sq = _core.mean(x_sq, dim=dim, keepdim=True)
        eps_tensor = _core.full_like(mean_sq, float(self.eps))
        rms = _core.sqrt(mean_sq + eps_tensor)
        
        return x / rms * self.weight
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def parameters(self):
        return [self.weight]
    
    def named_parameters(self):
        return [("weight", self.weight)]


class LayerNorm(LLMBlock):
    """Standard Layer Normalization.
    
    Formula: (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int, normalized_dim: int = None):
        super().__init__(config, layer_idx)
        
        self.normalized_dim = normalized_dim or config.hidden_size
        self.eps = config.rms_norm_eps
        
        self.weight = _core.ones([self.normalized_dim])
        self.bias = _core.zeros([self.normalized_dim])
    
    def forward(self, x, **kwargs):
        """LayerNorm forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            Normalized: [batch, seq_len, hidden_size]
        """
        mean = _core.mean(x, dim=-1, keepdim=True)
        var = _core.mean((x - mean) ** 2, dim=-1, keepdim=True)
        eps_tensor = _core.full_like(var, float(self.eps))
        x_norm = (x - mean) / _core.sqrt(var + eps_tensor)
        
        return x_norm * self.weight + self.bias
    
    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def named_parameters(self):
        return [("weight", self.weight), ("bias", self.bias)]


# Register normalization types
from fastnn.models.llm.base import BlockRegistry
BlockRegistry.register("rms_norm", RMSNorm)
BlockRegistry.register("layer_norm", LayerNorm)