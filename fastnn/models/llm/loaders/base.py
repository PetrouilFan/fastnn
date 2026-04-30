"""Base loader interface for LLM models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from fastnn.models.llm.base import LLMConfig, LLMModel


class BaseLoader(ABC):
    """Abstract base class for model loaders.
    
    All loaders must implement the load() method.
    """
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> LLMModel:
        """Load model from path.
        
        Args:
            model_path: Path to model directory or file
            **kwargs: Additional loading options
            
        Returns:
            Loaded model instance
        """
        pass
    
    def get_config_class(self, model_type: str) -> type:
        """Get the config class for this loader.
        
        Args:
            model_type: Model type identifier
            
        Returns:
            Config class
        """
        from fastnn.models.llm.config import LlamaConfig, LFM2_5Config
        
        config_map = {
            "llama": LlamaConfig,
            "llama2": LlamaConfig,
            "llama3": LlamaConfig,
            "lfm2": LFM2_5Config,
            "lfm2.5": LFM2_5Config,
        }
        
        return config_map.get(model_type, LlamaConfig)


class ModelInfo:
    """Information about a loaded model."""
    
    def __init__(
        self,
        model: LLMModel,
        config: LLMConfig,
        load_time: float,
        model_path: str,
        backend: str
    ):
        self.model = model
        self.config = config
        self.load_time = load_time
        self.model_path = model_path
        self.backend = backend
    
    def __repr__(self):
        return (
            f"ModelInfo(backend={self.backend}, "
            f"layers={self.config.num_hidden_layers}, "
            f"hidden={self.config.hidden_size}, "
            f"load_time={self.load_time:.2f}s)"
        )