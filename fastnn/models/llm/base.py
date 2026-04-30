"""Base classes for LLM module.

This module provides abstract base classes for:
- LLMConfig: Configuration for any LLM
- LLMBlock: Base class for any layer/block type
- LLMModel: Base class for complete models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, TypeVar, Generic
import json
import os

import numpy as np

import fastnn._core as _core
from fastnn import Linear, Embedding


T = TypeVar('T', bound='LLMBlock')


class LLMConfig(ABC):
    """Abstract base configuration for any LLM model.
    
    All model configurations should inherit from this class
    and implement the required abstract properties.
    """
    
    # Core dimensions
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    
    # Attention parameters
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    
    # Normalization
    rms_norm_eps: float
    
    # Position encoding
    rope_theta: float
    max_position_embeddings: int
    
    # Special tokens
    bos_token_id: int
    eos_token_id: int
    
    # Model type identifier
    model_type: str
    
    @property
    @abstractmethod
    def layer_types(self) -> List[str]:
        """Return list of layer types for each layer.
        
        For example: ["conv", "conv", "attention", "conv", ...]
        """
        pass
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LLMConfig':
        """Create config from dictionary (e.g., from config.json)."""
        known = {k for k in dir(cls) if not k.startswith('_')}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)
    
    @classmethod
    def from_json(cls, path: str) -> 'LLMConfig':
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k in dir(self) 
                if not k.startswith('_') and not callable(getattr(self, k))}


class LLMParameter:
    """Wrapper for a model parameter with metadata."""
    
    def __init__(self, tensor, name: str = ""):
        self.tensor = tensor
        self.name = name
    
    @property
    def shape(self):
        return self.tensor.shape
    
    def numpy(self):
        return self.tensor.numpy()


class LLMBlock(ABC):
    """Abstract base class for any LLM layer/block.
    
    Blocks can be:
    - Attention blocks (GQAAttention, FullAttention)
    - Feed-forward blocks (SwiGLU, MLP)
    - Convolution blocks (LIVConv)
    - Normalization blocks
    
    Each block should implement forward() method.
    """
    
    def __init__(self, config: LLMConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
    
    @abstractmethod
    def forward(self, x, **kwargs):
        """Forward pass through the block.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            **kwargs: Additional arguments (position_ids, attention_mask, etc.)
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        pass
    
    def parameters(self) -> List[LLMParameter]:
        """Return list of parameters in this block."""
        return []
    
    def named_parameters(self) -> List[tuple]:
        """Return list of (name, parameter) tuples."""
        return []
    
    def eval(self):
        """Set to evaluation mode (no dropout, etc.)."""
        pass
    
    def train(self):
        """Set to training mode."""
        pass


class LLMModel(ABC):
    """Abstract base class for complete LLM models.
    
    All model implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._built = False
    
    @abstractmethod
    def forward(self, input_ids, **kwargs):
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        pass
    
    @abstractmethod
    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Load weights from state dictionary.
        
        Args:
            state_dict: Dictionary mapping parameter names to tensors
        """
        pass
    
    @abstractmethod
    def parameters(self) -> List[LLMParameter]:
        """Return all parameters in the model."""
        pass
    
    def named_parameters(self) -> List[tuple]:
        """Return all named parameters."""
        return []
    
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'LLMModel':
        """Load model from pretrained files.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional loading arguments
            
        Returns:
            Loaded model instance
        """
        pass
    
    def eval(self):
        """Set to evaluation mode."""
        self._eval_mode = True
    
    def train(self):
        """Set to training mode."""
        self._eval_mode = False
    
    @property
    def device(self) -> str:
        """Return device (cpu/cuda)."""
        return "cpu"
    
    @property
    def dtype(self) -> str:
        """Return dtype (float32/float16)."""
        return "float32"
    
    def generate(
        self, 
        tokenizer, 
        prompt: str, 
        max_tokens: int = 100, 
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text string
        """
        raise NotImplementedError("Generate not implemented")


class BlockRegistry:
    """Registry for layer block types.
    
    Allows registering new block types and creating them by name.
    """
    
    _blocks: Dict[str, Type[LLMBlock]] = {}
    
    @classmethod
    def register(cls, name: str, block_class: Type[LLMBlock]) -> None:
        """Register a block type."""
        cls._blocks[name] = block_class
    
    @classmethod
    def create(cls, name: str, config: LLMConfig, layer_idx: int) -> LLMBlock:
        """Create a block by name."""
        if name not in cls._blocks:
            raise ValueError(f"Unknown block type: {name}. Available: {list(cls._blocks.keys())}")
        return cls._blocks[name](config, layer_idx)
    
    @classmethod
    def available(cls) -> List[str]:
        """List available block types."""
        return list(cls._blocks.keys())


class ModelRegistry:
    """Registry for model types.
    
    Allows registering new models and loading them by name.
    """
    
    _models: Dict[str, Type[LLMModel]] = {}
    _configs: Dict[str, Type[LLMConfig]] = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[LLMModel]) -> None:
        """Register a model class."""
        cls._models[name] = model_class
    
    @classmethod
    def register_config(cls, name: str, config_class: Type[LLMConfig]) -> None:
        """Register a config class."""
        cls._configs[name] = config_class
    
    @classmethod
    def create_model(cls, name: str, config: LLMConfig) -> LLMModel:
        """Create a model by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name](config)
    
    @classmethod
    def get_config(cls, model_type: str) -> Optional[Type[LLMConfig]]:
        """Get config class by model type."""
        return cls._configs.get(model_type)
    
    @classmethod
    def available_models(cls) -> List[str]:
        """List available models."""
        return list(cls._models.keys())


def auto_register(cls: Type) -> Type:
    """Decorator to auto-register a model or block.
    
    Usage:
        @auto_register
        class MyModel(LLMModel):
            ...
    """
    # Auto-register based on class name
    name = cls.__name__.replace('Model', '').lower()
    
    if issubclass(cls, LLMModel):
        ModelRegistry.register_model(name, cls)
    elif issubclass(cls, LLMBlock):
        BlockRegistry.register(name, cls)
    
    return cls