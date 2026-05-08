"""Unified serialization API for fastnn.

This package provides a unified interface for saving/loading models,
converting from other formats (PyTorch, ONNX), and managing model I/O.
"""

import struct
import json
from contextlib import contextmanager
from pathlib import Path

# Canonical constants for fastnn serialization formats
MODEL_MAGIC = b"FNN\x00"
OPTIMIZER_MAGIC = b"FNO\x00"
MODEL_VERSION = 2
OPTIMIZER_VERSION = 1


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""
    pass


@contextmanager
def serialization_error(action):
    try:
        yield
    except (OSError, ValueError) as e:
        raise SerializationError(f"Failed to {action}: {e}") from e


def _pack_u64(value: int) -> bytes:
    """Pack a 64-bit unsigned integer (little-endian)."""
    return struct.pack("<Q", value)


def _pack_i64(value: int) -> bytes:
    """Pack a 64-bit signed integer (little-endian)."""
    return struct.pack("<q", value)


def _pack_u32(value: int) -> bytes:
    """Pack a 32-bit unsigned integer (little-endian)."""
    return struct.pack("<I", value)


def _pack_u8(value: int) -> bytes:
    """Pack an 8-bit unsigned integer."""
    return struct.pack("<B", value)


def _pack_f64(value: float) -> bytes:
    """Pack a 64-bit float (double, little-endian)."""
    return struct.pack("<d", value)


def _unpack_u64(data: bytes) -> int:
    """Unpack a 64-bit unsigned integer (little-endian)."""
    return struct.unpack("<Q", data)[0]


def _unpack_i64(data: bytes) -> int:
    """Unpack a 64-bit signed integer (little-endian)."""
    return struct.unpack("<q", data)[0]


def _unpack_u32(data: bytes) -> int:
    """Unpack a 32-bit unsigned integer (little-endian)."""
    return struct.unpack("<I", data)[0]


def _unpack_u8(data: bytes) -> int:
    """Unpack an 8-bit unsigned integer."""
    return struct.unpack("<B", data)[0]


def _unpack_f64(data: bytes) -> float:
    """Unpack a 64-bit float (double, little-endian)."""
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


def write_fnn_file(f, header, params, magic=MODEL_MAGIC, version=MODEL_VERSION):
    """Write standard .fnn file format (magic + version + JSON header + parameters)."""
    f.write(magic)
    f.write(_pack_u32(version))
    header_json = json.dumps(header, indent=2)
    header_bytes = header_json.encode("utf-8")
    f.write(_pack_u64(len(header_bytes)))
    f.write(header_bytes)
    f.write(_pack_u64(len(params)))
    for name, data in params:
        write_tensor(f, name, data)


def read_fnn_header(f):
    """Read and return (magic, version, header_dict, num_params) from .fnn file."""
    magic = f.read(4)
    if len(magic) != 4:
        raise ValueError("Incomplete file: failed to read magic bytes")
    version = _unpack_u32(f.read(4))
    header_len = _unpack_u64(f.read(8))
    header_bytes = f.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError("Incomplete file: failed to read header")
    header_dict = json.loads(header_bytes.decode("utf-8"))
    num_params = _unpack_u64(f.read(8))
    return magic, version, header_dict, num_params


def read_fnn_parameters(f, num_params):
    """Read and return dict of {name: data} for num_params tensors."""
    params = {}
    for _ in range(num_params):
        name, data = read_tensor(f)
        params[name] = data
    return params


from fastnn.io.serialization import (
    save_model,
    load_model,
    save_optimizer,
    load_optimizer,
    save_state_dict,
    load_state_dict,
)
from fastnn.io.export import (
    export_pytorch_model,
    save_fnn_model,
    load_fnn_model,
)
from fastnn.io.onnx import (
    import_onnx,
)


def save(model, path: str, format: str = "fnn-v2") -> None:
    """Save a model to file.
    
    Args:
        model: The model to save.
        path: Path to save to.
        format: Format to save as ("fnn-v2", "pytorch", etc.)
    """
    path = Path(path)
    if format == "fnn-v2":
        save_model(model, str(path))
    else:
        raise ValueError(f"Unsupported format: {format}")


def load(path: str) -> object:
    """Load a model from file.
    
    Args:
        path: Path to load from.
        
    Returns:
        The loaded model.
    """
    return load_model(str(Path(path)))


def convert_from_pytorch(torch_model, path: str) -> None:
    """Convert a PyTorch model to fastnn format.
    
    Args:
        torch_model: The PyTorch model to convert.
        path: Path to save the converted model.
    """
    save_fnn_model(torch_model, str(Path(path)))


def convert_from_onnx(onnx_path: str, fnn_path: str) -> dict:
    """Convert an ONNX model to fastnn format.
    
    Args:
        onnx_path: Path to the ONNX model.
        fnn_path: Path to save the fastnn model.
        
    Returns:
        Dictionary with model info.
    """
    return import_onnx(str(Path(onnx_path)), str(Path(fnn_path)))


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
    "write_fnn_file",
    "read_fnn_header",
    "read_fnn_parameters",
    "serialization_error",
]
