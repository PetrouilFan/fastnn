"""Shared serialization utilities for fastnn.

Provides common functions for reading and writing tensor data
in the fastnn serialization format.
"""

import struct
from typing import Tuple
import numpy as np

# Canonical constants for fastnn serialization formats
MODEL_MAGIC = b"FNN\x00"
OPTIMIZER_MAGIC = b"FNO\x00"
MODEL_VERSION = 2
OPTIMIZER_VERSION = 1


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


def write_tensor(f, name: str, data: np.ndarray) -> None:
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


def read_tensor(f) -> Tuple[str, np.ndarray]:
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