"""FastNN model loader.

Loads models using the fastnn engine (custom Rust-based neural network library).
"""

import os
import json
import time
from typing import Optional, Any, Dict
import fastnn
from fastnn.models.llm.loaders.base import BaseLoader, ModelInfo
from fastnn.models.llm.base import LLMConfig
from fastnn.models.llm.config import LlamaConfig, LFM2_5Config


class FastNNLoader(BaseLoader):
    """Loader for fastnn models.
    
    Loads LLM models using the fastnn custom neural network library.
    """
    
    def __init__(self):
        self.name = "fastnn"
    
    def load(
        self, 
        model_path: str, 
        model_type: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """Load model using fastnn backend.
        
        Args:
            model_path: Path to model directory
            model_type: Model type (llama, lfm2.5, etc.)
            **kwargs: Additional options
            
        Returns:
            ModelInfo with loaded model
        """
        start_time = time.time()
        
        # Auto-detect model type from config if not provided
        if model_type is None:
            model_type = self._detect_model_type(model_path)
        
        # Load config first
        config = self._load_config(model_path, model_type)
        
        # Load weights to get actual dimensions
        from fastnn.io import load_safetensors
        weights = load_safetensors(model_path)
        
        # Find actual intermediate size from weights BEFORE creating model
        actual_intermediate = None
        for k in weights.keys():
            if 'feed_forward.w1.weight' in k:
                actual_intermediate = weights[k].numpy().shape[0]
                break
        
        # Override intermediate_size in config if different from config.json
        if actual_intermediate and model_type in ('lfm2', 'lfm2.5'):
            if hasattr(config, 'intermediate_size'):
                if actual_intermediate != config.intermediate_size:
                    print(f"  Note: Fixing intermediate_size: {config.intermediate_size} -> {actual_intermediate}")
                    config.intermediate_size = actual_intermediate
        
        # Create model based on type (now with correct intermediate_size)
        if model_type in ("lfm2", "lfm2.5"):
            from fastnn.models.llm.models.lfm import LFM2_5Model
            model = LFM2_5Model(config)
        elif model_type in ("llama", "llama2", "llama3"):
            from fastnn.models.llm.models.llama import LlamaModel
            model = LlamaModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_weights(weights)
        
        model.eval()
        
        load_time = time.time() - start_time
        
        return ModelInfo(
            model=model,
            config=config,
            load_time=load_time,
            model_path=model_path,
            backend="fastnn"
        )
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type from config.json."""
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            return "llama"  # Default
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get("model_type", "")
        
        if model_type in ("llama", "llama2", "llama3"):
            return "llama"
        elif model_type in ("lfm", "lfm2", "lfm2.5"):
            return "lfm2.5"
        
        # Check architectures
        architectures = config.get("architectures", [])
        if "Lfm2ForCausalLM" in architectures:
            return "lfm2.5"
        elif "LlamaForCausalLM" in architectures:
            return "llama"
        
        return "llama"  # Default
    
    def _load_config(self, model_path: str, model_type: str) -> LLMConfig:
        """Load model configuration."""
        config_path = os.path.join(model_path, "config.json")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        if model_type in ("lfm2", "lfm2.5"):
            return LFM2_5Config.from_dict(config_dict)
        else:
            return LlamaConfig.from_dict(config_dict)


def load_model(
    model_path: str,
    model_type: Optional[str] = None,
    **kwargs
) -> ModelInfo:
    """Convenience function to load model with fastnn.
    
    Args:
        model_path: Path to model directory
        model_type: Optional model type override
        **kwargs: Additional options
        
    Returns:
        ModelInfo with loaded model
    """
    loader = FastNNLoader()
    return loader.load(model_path, model_type, **kwargs)