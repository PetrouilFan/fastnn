"""Unified serialization API for fastnn.

This package provides a unified interface for saving/loading models,
converting from other formats (PyTorch, ONNX), and managing model I/O.
"""

# Canonical constants for fastnn serialization formats
MODEL_MAGIC = b"FNN\x00"
OPTIMIZER_MAGIC = b"FNO\x00"
MODEL_VERSION = 2
OPTIMIZER_VERSION = 1


def _pack_u64(value: int) -> bytes:
    """Pack a 64-bit unsigned integer (little-endian)."""
    import struct
    return struct.pack("<Q", value)


def _pack_i64(value: int) -> bytes:
    """Pack a 64-bit signed integer (little-endian)."""
    import struct
    return struct.pack("<q", value)


def _pack_u32(value: int) -> bytes:
    """Pack a 32-bit unsigned integer (little-endian)."""
    import struct
    return struct.pack("<I", value)


def _pack_u8(value: int) -> bytes:
    """Pack an 8-bit unsigned integer."""
    import struct
    return struct.pack("<B", value)


def _pack_f64(value: float) -> bytes:
    """Pack a 64-bit float (double, little-endian)."""
    import struct
    return struct.pack("<d", value)


def _unpack_u64(data: bytes) -> int:
    """Unpack a 64-bit unsigned integer (little-endian)."""
    import struct
    return struct.unpack("<Q", data)[0]


def _unpack_i64(data: bytes) -> int:
    """Unpack a 64-bit signed integer (little-endian)."""
    import struct
    return struct.unpack("<q", data)[0]


def _unpack_u32(data: bytes) -> int:
    """Unpack a 32-bit unsigned integer (little-endian)."""
    import struct
    return struct.unpack("<I", data)[0]


def _unpack_u8(data: bytes) -> int:
    """Unpack an 8-bit unsigned integer."""
    import struct
    return struct.unpack("<B", data)[0]


def _unpack_f64(data: bytes) -> float:
    """Unpack a 64-bit float (double, little-endian)."""
    import struct
    return struct.unpack("<d", data)[0]


def write_tensor(f, name: str, data) -> None:
    """Write a tensor to the file.

    Args:
        f: File object opened in binary write mode.
        name: Parameter name.
        data: Tensor data as numpy array (float32).
    """
    import numpy as np
    name_bytes = name.encode("utf-8")
    shape = list(data.shape)
    data_f32 = data.astype(np.float32, copy=False).ravel()
    f.write(_pack_u64(len(name_bytes)))
    f.write(name_bytes)
    f.write(_pack_u64(len(shape)))
    for d in shape:
        f.write(_pack_i64(d))
    f.write(_pack_u64(len(data_f32)))
    f.write(data_f32.tobytes())


def read_tensor(f) -> tuple:
    """Read a tensor from the file.

    Returns:
        Tuple of (name, data) where data is numpy array.
    """
    import numpy as np
    name_len = _unpack_u64(f.read(8))
    name = f.read(name_len).decode("utf-8")
    shape_len = _unpack_u64(f.read(8))
    shape = [_unpack_i64(f.read(8)) for _ in range(shape_len)]
    data_len = _unpack_u64(f.read(8))
    data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
    return name, data.reshape(shape)


from fastnn.io.serialization import *  # noqa: F401, F403
from fastnn.io.export import *  # noqa: F401, F403
from fastnn.io.onnx import *  # noqa: F401, F403

# Re-export key functions with simplified names
from fastnn.io.serialization import save_model
from fastnn.io.serialization import load_model
from fastnn.io.serialization import save_optimizer
from fastnn.io.serialization import load_optimizer
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
    "save_optimizer",
    "load_optimizer",
    "save_fnn_model",
    "import_onnx",
    "MODEL_MAGIC",
    "OPTIMIZER_MAGIC",
    "MODEL_VERSION",
    "OPTIMIZER_VERSION",
    "write_tensor",
    "read_tensor",
    "_pack_u64",
    "_pack_i64",
    "_pack_u32",
    "_pack_u8",
    "_pack_f64",
    "_unpack_u64",
    "_unpack_i64",
    "_unpack_u32",
    "_unpack_u8",
    "_unpack_f64",
]
