"""Transformers model loader.

Loads models using HuggingFace Transformers as reference implementation.
"""

import os
import json
import time
from typing import Optional, Any
from fastnn.models.llm.loaders.base import BaseLoader, ModelInfo
from fastnn.models.llm.base import LLMConfig


class TransformersLoader(BaseLoader):
    """Loader for HuggingFace Transformers models.
    
    Loads LLM models using the HuggingFace Transformers library.
    Used as a reference for comparison with fastnn.
    """
    
    def __init__(self):
        self.name = "transformers"
    
    def load(
        self, 
        model_path: str, 
        model_type: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs
    ) -> ModelInfo:
        """Load model using Transformers backend.
        
        Args:
            model_path: Path to model directory
            model_type: Model type (llama, lfm2.5, etc.)
            device: Device to load on (cpu/cuda)
            dtype: Data type (float32/float16/bfloat16)
            **kwargs: Additional options
            
        Returns:
            ModelInfo with loaded model
        """
        start_time = time.time()
        
        # Import transformers
        try:
            from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")
        
        # Auto-detect model type from config if not provided
        if model_type is None:
            model_type = self._detect_model_type(model_path)
        
        # Load config
        config = self._load_config(model_path, model_type)
        
        # Map dtype
        dtype_map = {
            "float32": "float32",
            "float16": "float16", 
            "bfloat16": "bfloat16",
        }
        torch_dtype = dtype_map.get(dtype, "float32")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
            **kwargs
        )
        
        model.eval()
        
        load_time = time.time() - start_time
        
        return ModelInfo(
            model=model,
            config=config,
            load_time=load_time,
            model_path=model_path,
            backend="transformers"
        )
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect model type from config.json."""
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            return "llama"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_type = config.get("model_type", "")
        
        if model_type in ("llama", "llama2", "llama3"):
            return "llama"
        elif model_type in ("lfm", "lfm2", "lfm2.5"):
            return "lfm2.5"
        
        architectures = config.get("architectures", [])
        if "Lfm2ForCausalLM" in architectures:
            return "lfm2.5"
        elif "LlamaForCausalLM" in architectures:
            return "llama"
        
        return "llama"
    
    def _load_config(self, model_path: str, model_type: str) -> LLMConfig:
        """Load model configuration."""
        config_path = os.path.join(model_path, "config.json")
        return AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    
    def load_tokenizer(self, model_path: str):
        """Load tokenizer for a model.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Tokenizer instance
        """
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_model(
    model_path: str,
    model_type: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    **kwargs
) -> ModelInfo:
    """Convenience function to load model with Transformers.
    
    Args:
        model_path: Path to model directory
        model_type: Optional model type override
        device: Device to load on
        dtype: Data type
        **kwargs: Additional options
        
    Returns:
        ModelInfo with loaded model
    """
    loader = TransformersLoader()
    return loader.load(model_path, model_type, device, dtype, **kwargs)