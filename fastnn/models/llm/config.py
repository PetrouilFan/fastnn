"""Configuration classes for LLM models.

Provides concrete config implementations for:
- LlamaConfig: Meta's Llama models
- LFM2.5Config: Liquid AI's LFM2.5 models
"""

from typing import List, Optional, Dict, Any
from fastnn.models.llm.base import LLMConfig


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
    }
    
    if model_type not in config_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(config_classes.keys())}")
    
    return config_classes[model_type](**kwargs)