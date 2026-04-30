"""Configuration classes for LLM models.

Provides concrete config implementations for:
- LlamaConfig: Meta's Llama models
- LFM2.5Config: Liquid AI's LFM2.5 models
"""

from typing import List, Optional, Dict, Any
from fastnn.models.llm.base import LLMConfig


class Gemma4Config(LLMConfig):
    """Configuration for Google's Gemma 4 models."""
    
    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 1536,
        intermediate_size: int = 6144,
        intermediate_sizes: Optional[List[int]] = None,
        num_hidden_layers: int = 35,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        head_dim_swa: int = 256,
        rope_theta: float = 1000000.0,
        rope_theta_swa: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 131072,
        sliding_window: int = 512,
        sliding_window_pattern: Optional[List[bool]] = None,
        shared_kv_layers: int = 20,
        final_logit_softcapping: float = 30.0,
        bos_token_id: int = 2,
        eos_token_id: int = 106,
        tie_embedding: bool = True,
        model_type: str = "gemma4",
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_sizes = intermediate_sizes
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.head_dim_swa = head_dim_swa
        self.rope_theta = rope_theta
        self.rope_theta_swa = rope_theta_swa
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.shared_kv_layers = shared_kv_layers
        self.final_logit_softcapping = final_logit_softcapping
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_embedding = tie_embedding
        self.model_type = "gemma4"
    
    @property
    def layer_types(self) -> List[str]:
        """Return attention type for each layer based on sliding_window_pattern."""
        if self.sliding_window_pattern:
            return [
                "sliding_attention" if sw else "full_attention"
                for sw in self.sliding_window_pattern
            ]
        # Default pattern: every 5th layer is full attention
        return [
            "sliding_attention" if (i % 5 != 4) else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
    
    def get_intermediate_size(self, layer_idx: int) -> int:
        """Get FFN intermediate size for a specific layer."""
        if self.intermediate_sizes and layer_idx < len(self.intermediate_sizes):
            return self.intermediate_sizes[layer_idx]
        return self.intermediate_size
    
    def get_head_dim(self, layer_idx: int) -> int:
        """Get head dimension for a specific layer (SWA vs full)."""
        if self.sliding_window_pattern and layer_idx < len(self.sliding_window_pattern):
            return self.head_dim_swa if self.sliding_window_pattern[layer_idx] else self.head_dim
        return self.head_dim
    
    def is_sliding_window(self, layer_idx: int) -> bool:
        """Check if layer uses sliding window attention."""
        pattern = self.sliding_window_pattern or []
        if layer_idx < len(pattern):
            return pattern[layer_idx]
        return layer_idx % 5 != 4
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Gemma4Config':
        known = {
            "vocab_size", "hidden_size", "intermediate_size",
            "intermediate_sizes", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "head_dim_swa", "rope_theta",
            "rope_theta_swa", "rms_norm_eps", "max_position_embeddings",
            "sliding_window", "sliding_window_pattern", "shared_kv_layers",
            "final_logit_softcapping", "bos_token_id", "eos_token_id",
            "tie_embedding"
        }
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class LlamaConfig(LLMConfig):
    """Configuration for Llama models.
    
    Supports Llama 1, 2, and 3 variants with different sizes.
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 131072,
        hidden_act: str = "silu",
        bos_token_id: int = 128000,
        eos_token_id: int = 128009,
        rope_scaling: Optional[Dict] = None,
        tie_embedding: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rope_scaling = rope_scaling
        self.tie_embedding = tie_embedding
        self.model_type = "llama"
        
        # All Llama layers are attention type
        self._layer_types = ["attention"] * num_hidden_layers
    
    @property
    def layer_types(self) -> List[str]:
        return self._layer_types
    
    @layer_types.setter
    def layer_types(self, value: List[str]):
        self._layer_types = value
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LlamaConfig':
        known = {
            "vocab_size", "hidden_size", "intermediate_size", 
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "rope_theta", "rms_norm_eps", "max_position_embeddings",
            "hidden_act", "bos_token_id", "eos_token_id", "rope_scaling",
            "tie_embedding"
        }
        filtered = {k: v for k, v in d.items() if k in known}
        # Handle rope_scaling
        if "rope_scaling" in d and d["rope_scaling"]:
            filtered["rope_scaling"] = d["rope_scaling"]
        return cls(**filtered)


class LFM2_5Config(LLMConfig):
    """Configuration for Liquid AI's LFM2.5 models.
    
    LFM2.5 has a unique hybrid architecture with:
    - LIV Conv blocks: w1, w2, w3 feed-forward with SwiGLU
    - Full attention blocks: GQA attention
    
    The layer_types specifies which block type each layer uses.
    """
    
    # Default LFM2.5-350M layer pattern
    DEFAULT_LAYER_TYPES = [
        "conv", "conv", "attention",
        "conv", "conv", "attention",
        "conv", "conv", "attention",
        "conv", "attention",
        "conv", "attention",
        "conv", "attention",
        "conv"
    ]
    
    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 1024,
        intermediate_size: int = 4608,  # Default to 4608 for LFM2.5-350M
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 128000,
        layer_types: Optional[List[str]] = None,
        # LFM-specific params
        conv_dim: int = 1024,
        conv_L_cache: int = 3,
        conv_bias: bool = False,
        use_pos_enc: bool = True,
        tie_embedding: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 7,
        pad_token_id: int = 0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.model_type = "lfm2"
        
        # LFM-specific
        self.conv_dim = conv_dim
        self.conv_L_cache = conv_L_cache
        self.conv_bias = conv_bias
        self.use_pos_enc = use_pos_enc
        self.tie_embedding = tie_embedding
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # Layer types - default or custom
        if layer_types is None:
            self._layer_types = self.DEFAULT_LAYER_TYPES[:num_hidden_layers]
        else:
            self._layer_types = layer_types
        
        # Validate layer count matches
        if len(self._layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self._layer_types)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )
    
    @property
    def layer_types(self) -> List[str]:
        return self._layer_types
    
    @layer_types.setter
    def layer_types(self, value: List[str]):
        self._layer_types = value
    
    def get_layer_type(self, idx: int) -> str:
        """Get layer type for a specific layer index."""
        return self._layer_types[idx]
    
    @property
    def num_conv_layers(self) -> int:
        """Count of convolution layers."""
        return sum(1 for t in self._layer_types if t == "conv")
    
    @property
    def num_attention_layers(self) -> int:
        """Count of attention layers."""
        return sum(1 for t in self._layer_types if t == "attention")
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LFM2_5Config':
        known = {
            "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "rope_theta", "rms_norm_eps", "max_position_embeddings",
            "layer_types", "conv_dim", "conv_L_cache", "conv_bias",
            "use_pos_enc", "tie_embedding", "bos_token_id", "eos_token_id",
            "pad_token_id"
        }
        filtered = {k: v for k, v in d.items() if k in known}
        
        # Compute head_dim if not provided or None
        if filtered.get('head_dim') is None:
            num_heads = filtered.get('num_attention_heads', 16)
            hidden = filtered.get('hidden_size', 1024)
            filtered['head_dim'] = hidden // num_heads if num_heads > 0 else 64
        
        # NOTE: intermediate_size in config.json (6656) doesn't match actual 
        # weight shapes (4608). We'll compute the actual from weights if available.
        # For now, use the value from config but note this discrepancy.
        # The loader should override this if needed.
        
        return cls(**filtered)


def create_config(model_type: str, **kwargs) -> LLMConfig:
    """Factory function to create config by model type.
    
    Args:
        model_type: Model type identifier (e.g., "llama", "lfm2")
        **kwargs: Config parameters
        
    Returns:
        LLMConfig instance
    """
    config_classes = {
        "llama": LlamaConfig,
        "llama2": LlamaConfig,
        "llama3": LlamaConfig,
        "lfm2": LFM2_5Config,
        "lfm2.5": LFM2_5Config,
        "gemma4": Gemma4Config,
    }
    
    if model_type not in config_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(config_classes.keys())}")
    
    return config_classes[model_type](**kwargs)