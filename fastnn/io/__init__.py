"""Unified serialization API for fastnn.

This package provides a unified interface for saving/loading models,
converting from other formats (PyTorch, ONNX), and managing model I/O.
"""

import struct
import json
import numpy as np
from contextlib import contextmanager
from pathlib import Path

# Canonical constants for fastnn serialization formats
MODEL_MAGIC = b"FNN\x00"
OPTIMIZER_MAGIC = b"FNO\x00"
MODEL_VERSION = 3  # v3: dtype-tagged packed tensors
OPTIMIZER_VERSION = 1

# Dtype tags for v3 serialization (matches Precision enum values)
DTYPE_F32 = 0  # full float32
DTYPE_F16 = 1  # PackedTensor<F16x2>
DTYPE_U8  = 2  # PackedTensor<U8x4>
DTYPE_U4  = 3  # PackedTensor<U4x8>


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
    name_len = _unpack_u64(f.read(8))
    name = f.read(name_len).decode("utf-8")
    shape_len = _unpack_u64(f.read(8))
    shape = [_unpack_i64(f.read(8)) for _ in range(shape_len)]
    data_len = _unpack_u64(f.read(8))
    data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
    return name, data.reshape(shape)


def write_fnn_file(f, header, params, magic=MODEL_MAGIC, version=MODEL_VERSION):
    """Write standard .fnn file format (magic + version + JSON header + parameters).

    For v2 (default): writes all parameters as f32 tensors.
    For v3+: uses dtype-tagged tensor format (see write_fnn_file_v3).
    """
    if version >= 3:
        return write_fnn_file_v3(f, header, params, magic=magic, version=version)

    f.write(magic)
    f.write(_pack_u32(version))
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode("utf-8")
    f.write(_pack_u64(len(header_bytes)))
    f.write(header_bytes)
    f.write(_pack_u64(len(params)))
    for name, data in params:
        write_tensor(f, name, data)


def write_fnn_file_v3(f, header, params_v3, magic=MODEL_MAGIC, version=3):
    """Write v3 .fnn file with dtype-tagged tensors.

    Args:
        f: File object opened in binary write mode.
        header: JSON-serializable header dict.
        params_v3: List of (name, data, dtype, scales, zeros) tuples.
            - data: bytes for packed, numpy array for F32
            - dtype: DTYPE_F32/DTYPE_U4/DTYPE_U8/DTYPE_F16
            - scales: list of float (per-channel)
            - zeros: list of float (per-channel)
        magic: File magic bytes.
        version: File format version (default 3).
    """
    f.write(magic)
    f.write(_pack_u32(version))
    header_json = json.dumps(header, separators=(',', ':'))
    header_bytes = header_json.encode("utf-8")
    f.write(_pack_u64(len(header_bytes)))
    f.write(header_bytes)
    f.write(_pack_u64(len(params_v3)))
    for item in params_v3:
        if len(item) == 6:
            name, data, dtype, scales, zeros, shape = item
        elif len(item) == 5:
            name, data, dtype, scales, zeros = item
            shape = None
        elif len(item) == 2:
            name, data = item
            dtype, scales, zeros, shape = DTYPE_F32, [], [], None
        else:
            raise ValueError(f"Invalid param tuple length: {len(item)}")
        write_tensor_v3(f, name, data, dtype, scales, zeros, shape)


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


def read_fnn_parameters(f, num_params, version=MODEL_VERSION):
    """Read and return dict of {name: data} for num_params tensors.

    Args:
        f: File object.
        num_params: Number of parameters to read.
        version: File format version. v3+ uses dtype-tagged format,
                 v2 reads all tensors as F32.

    Returns:
        For v2: dict of {name: ndarray}
        For v3+: dict of {name: (ndarray|bytes, dtype, scales, zeros)}
    """
    if version >= 3:
        return read_fnn_parameters_v3(f, num_params)
    params = {}
    for _ in range(num_params):
        name, data = read_tensor(f)
        params[name] = data
    return params


def read_fnn_parameters_v3(f, num_params):
    """Read v3 parameters, returning with dtype/scales/zeros metadata.

    Returns:
        dict of {name: (data, dtype, scales, zeros, shape)}
        - For F32: data is ndarray, dtype=DTYPE_F32, scales=[], zeros=[], shape=list
        - For packed: data is bytes, dtype is tag, scales/zeros are lists, shape=list
    """
    params = {}
    for _ in range(num_params):
        name, data, dtype, scales, zeros, shape = read_tensor_v3(f)
        params[name] = (data, dtype, scales, zeros, shape)
    return params


# ---- v3 serialization (dtype-tagged) ----

def write_tensor_v3(f, name: str, data, dtype: int = DTYPE_F32, scales=None, zeros=None, shape=None) -> None:
    """Write a dtype-tagged tensor to file (v3 format).

    Args:
        f: File object opened in binary write mode.
        name: Parameter name.
        data: Packed bytes (for quantized) or numpy f32 array (for F32).
        dtype: Dtype tag (DTYPE_F32/DTYPE_U4/DTYPE_U8/DTYPE_F16).
        scales: Optional list of per-channel scales (float).
        zeros: Optional list of per-channel zeros (float).
        shape: Optional explicit shape list (required for packed bytes data).
    """
    name_bytes = name.encode("utf-8")
    f.write(_pack_u64(len(name_bytes)))
    f.write(name_bytes)

    if isinstance(data, np.ndarray):
        shape = list(data.shape) if shape is None else shape
        data_bytes = data.astype(np.float32, copy=False).ravel().tobytes()
    elif isinstance(data, bytes):
        if shape is None:
            shape = []
        data_bytes = data
    else:
        raise TypeError(f"Expected ndarray or bytes, got {type(data)}")

    f.write(_pack_u64(len(shape)))
    for d in shape:
        f.write(_pack_i64(d))
    f.write(_pack_u8(dtype))

    # Write scales
    if scales is not None:
        scales_arr = list(scales)
        f.write(_pack_u64(len(scales_arr)))
        for s in scales_arr:
            f.write(_pack_f64(float(s)))
    else:
        f.write(_pack_u64(0))

    # Write zeros
    if zeros is not None:
        zeros_arr = list(zeros)
        f.write(_pack_u64(len(zeros_arr)))
        for z in zeros_arr:
            f.write(_pack_f64(float(z)))
    else:
        f.write(_pack_u64(0))

    # Write data bytes
    f.write(_pack_u64(len(data_bytes)))
    f.write(data_bytes)


def read_tensor_v3(f) -> tuple:
    """Read a dtype-tagged tensor from file (v3 format).

    Returns:
        Tuple of (name, data, dtype, scales, zeros, shape) where:
        - data is bytes for quantized, numpy array for F32
        - dtype is int tag
        - scales is list of float
        - zeros is list of float
        - shape is list of int (the logical tensor shape)
    """
    name_len = _unpack_u64(f.read(8))
    name = f.read(name_len).decode("utf-8")
    shape_len = _unpack_u64(f.read(8))
    shape = [_unpack_i64(f.read(8)) for _ in range(shape_len)]
    dtype = _unpack_u8(f.read(1))

    n_scales = _unpack_u64(f.read(8))
    scales = [_unpack_f64(f.read(8)) for _ in range(n_scales)]

    n_zeros = _unpack_u64(f.read(8))
    zeros = [_unpack_f64(f.read(8)) for _ in range(n_zeros)]

    data_len = _unpack_u64(f.read(8))

    if dtype == DTYPE_F32:
        # F32: data is stored as f32 values
        n_floats = data_len // 4
        data = np.frombuffer(f.read(n_floats * 4), dtype=np.float32).reshape(shape)
    else:
        # Packed: data is raw bytes
        data = f.read(data_len)

    return name, data, dtype, scales, zeros, shape


def read_tensor_auto(f, version: int) -> tuple:
    """Read a tensor with automatic version detection.

    Args:
        f: File object.
        version: Model file version (2 or 3).

    Returns:
        Tuple of (name, data, dtype, scales, zeros).
        For v2: dtype=DTYPE_F32, scales=[], zeros=[].
        For v3: uses dtype tag from file.
    """
    if version >= 3:
        return read_tensor_v3(f)
    else:
        # v2 format: all params are F32
        name, arr = read_tensor(f)
        return name, arr, DTYPE_F32, [], []


from fastnn.io.serialization import _save_model as save_model, _load_model as load_model
from fastnn.io.export import save_fnn_model
from fastnn.io.onnx import import_onnx
from fastnn.io.graph_builder import build_model_from_fnn
from fastnn.io.dag_model import DAGModel
import fastnn._core as _core
AotExecutor = getattr(_core, 'AotExecutor', None)


def save(model, path: str, format: str = "fnn-v2") -> None:
    """Save a model to file.

    Args:
        model: The model to save.
        path: Path to save to.
        format: Format to save as ("fnn-v2", "pytorch", etc.)
    """
    path = Path(path)
    if format == "fnn-v2":
        save_model(model, str(path), version=2)
    elif format == "fnn-v3":
        save_model(model, str(path), version=3)
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


def load_onnx_model(onnx_path: str, fnn_path: str) -> dict:
    """
    Import an ONNX model and save it in fastnn format.

    This is an alias for convert_from_onnx that follows the
    load_fnn_model naming convention.

    Args:
        onnx_path: Path to the .onnx file.
        fnn_path: Path where the .fnn output will be saved.

    Returns:
        Dictionary with model info including layers, parameter count,
        and input/output shapes.
    """
    return import_onnx(str(Path(onnx_path)), str(Path(fnn_path)))


__all__ = [
    "save",
    "load",
    "convert_from_pytorch",
    "convert_from_onnx",
    "load_onnx_model",
    "build_model_from_fnn",
    "DAGModel",
    "AotExecutor",
]
