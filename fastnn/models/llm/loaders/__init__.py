"""Model loaders package."""

from typing import Optional

from fastnn.models.llm.loaders.base import BaseLoader, ModelInfo
from fastnn.models.llm.loaders.fastnn_loader import FastNNLoader, load_model as fastnn_load_model
from fastnn.models.llm.loaders.transformers_loader import TransformersLoader, load_model as transformers_load_model


def load_model(
    model_path: str,
    backend: str = "fastnn",
    model_type: Optional[str] = None,
    **kwargs
) -> ModelInfo:
    """Load model with specified backend.
    
    Args:
        model_path: Path to model directory
        backend: Backend to use ("fastnn" or "transformers")
        model_type: Optional model type
        **kwargs: Backend-specific options
        
    Returns:
        ModelInfo with loaded model
    """
    if backend == "fastnn":
        return fastnn_load_model(model_path, model_type, **kwargs)
    elif backend == "transformers":
        return transformers_load_model(model_path, model_type, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'fastnn' or 'transformers'.")


__all__ = [
    "BaseLoader",
    "ModelInfo",
    "FastNNLoader",
    "TransformersLoader", 
    "load_model",
]