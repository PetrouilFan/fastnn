"""LLM models package for fastnn.

This package provides a modular system for loading and running large language models.

Architecture:
- base.py: Abstract base classes (LLMConfig, LLMBlock, LLMModel)
- config.py: Configuration classes
- loaders/: Model loading (transformers, fastnn, onnx)
- architectures/: Reusable components (attention, ffn, norms)
- models/: Concrete model implementations (LFM2.5, Llama, etc.)
- utils/: Weight mapping, RoPE, caching utilities

Quick start:
    from fastnn.models.llm import ModelRegistry, load_model
    
    # Load a model using fastnn
    model = load_model("LFM2.5-350M", backend="fastnn")
    
    # Or using transformers as reference
    model_ref = load_model("LFM2.5-350M", backend="transformers")
    
    # Compare outputs
    from fastnn.models.llm.compare import ModelComparator
    comparator = ModelComparator({"fastnn": model, "transformers": model_ref})
    results = comparator.compare("Hello world")
    results.print_report()
"""

from fastnn.models.llm.base import (
    LLMConfig,
    LLMBlock, 
    LLMModel,
    LLMParameter,
    BlockRegistry,
    ModelRegistry,
    auto_register,
)
from fastnn.models.llm.config import LlamaConfig, LFM2_5Config
from fastnn.models.llm.loaders import FastNNLoader, TransformersLoader, load_model
from fastnn.models.llm.compare import ModelComparator

__all__ = [
    # Base classes
    "LLMConfig",
    "LLMBlock", 
    "LLMModel",
    "LLMParameter",
    "BlockRegistry",
    "ModelRegistry",
    "auto_register",
    # Configs
    "LlamaConfig", 
    "LFM2_5Config",
    # Loaders
    "FastNNLoader",
    "TransformersLoader", 
    "load_model",
    # Comparison
    "ModelComparator",
]