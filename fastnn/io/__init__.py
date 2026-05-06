"""Unified serialization API for fastnn.

This package provides a unified interface for saving/loading models,
converting from other formats (PyTorch, ONNX), and managing model I/O.
"""

from fastnn.io.serialization import *  # noqa: F401, F403
from fastnn.io.export import *  # noqa: F401, F403
from fastnn.io.onnx import *  # noqa: F401, F403

# Re-export key functions with simplified names
from fastnn.io.serialization import save_model as _save_model
from fastnn.io.serialization import load_model as _load_model
from fastnn.io.export import save_fnn_model as _save_pytorch
from fastnn.io.onnx import import_onnx as _import_onnx


def save(model, path: str, format: str = "fnn-v2") -> None:
    """Save a model to file.
    
    Args:
        model: The model to save.
        path: Path to save to.
        format: Format to save as ("fnn-v2", "pytorch", etc.)
    """
    if format == "fnn-v2":
        _save_model(model, path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load(path: str) -> object:
    """Load a model from file.
    
    Args:
        path: Path to load from.
        
    Returns:
        The loaded model.
    """
    return _load_model(path)


def convert_from_pytorch(torch_model, path: str) -> None:
    """Convert a PyTorch model to fastnn format.
    
    Args:
        torch_model: The PyTorch model to convert.
        path: Path to save the converted model.
    """
    _save_pytorch(torch_model, path)


def convert_from_onnx(onnx_path: str, fnn_path: str) -> dict:
    """Convert an ONNX model to fastnn format.
    
    Args:
        onnx_path: Path to the ONNX model.
        fnn_path: Path to save the fastnn model.
        
    Returns:
        Dictionary with model info.
    """
    return _import_onnx(onnx_path, fnn_path)


__all__ = [
    "save",
    "load",
    "convert_from_pytorch",
    "convert_from_onnx",
    "save_model",
    "load_model",
    "save_fnn_model",
    "import_onnx",
    "MODEL_MAGIC",
    "OPTIMIZER_MAGIC",
    "MODEL_VERSION",
    "OPTIMIZER_VERSION",
    "write_tensor",
    "read_tensor",
]
