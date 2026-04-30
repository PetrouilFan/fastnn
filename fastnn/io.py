import struct
import json
from typing import Dict, Any

import numpy as np

import fastnn._core as _core


MAGIC = b"FNN\x00"
VERSION = 1

HAS_SAFETENSORS = True


def _safetensors_dtype_to_numpy(dtype_str: str) -> np.dtype:
    """Convert safetensors dtype string to numpy dtype."""
    dtype_map = {
        "F64": np.float64,
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.uint16,  # Will convert to float32 later
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U64": np.uint64,
        "U32": np.uint32,
        "U16": np.uint16,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }
    return np.dtype(dtype_map.get(dtype_str, np.float32))


def load_safetensors(path: str) -> Dict[str, Any]:
    """Load weights from a HuggingFace safetensors file.

    Args:
        path: Path to .safetensors file (directory or file path)

    Returns:
        Dictionary mapping parameter names to fastnn tensors
    """
    import os

    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")

    return _load_safetensors_numpy(path)


def _load_safetensors_numpy(path: str) -> Dict[str, Any]:
    """Load safetensors without torch - uses raw parsing.

    This reads the safetensors format directly.

    Args:
        path: Path to .safetensors file

    Returns:
        Dictionary mapping parameter names to fastnn tensors
    """
    with open(path, "rb") as f:
        header_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_bytes)[0]

        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)

        result = {}
        for tensor_name, tensor_meta in header.items():
            data_offsets = tensor_meta.get("data_offsets", [])
            if not data_offsets:
                continue

            offset, size = data_offsets[0], data_offsets[1] - data_offsets[0]
            dtype_str = tensor_meta.get("dtype", "F32")
            shape = tensor_meta.get("shape", [])

        raw_dtype = _safetensors_dtype_to_numpy(dtype_str)

        tensor_start = 8 + header_size + offset
        f.seek(tensor_start)
        data_bytes = f.read(size)

        np_data = np.frombuffer(data_bytes, dtype=raw_dtype)
        
        # For BF16: reshape first (as uint16), then convert to float32
        # For other types: reshape directly
        if dtype_str == "BF16":
            np_data = np_data.reshape(shape)
            # BF16 is stored as uint16 but represents bfloat16
            # Need to convert properly: shift left by 16 to align with FP32 mantissa
            np_data = (np_data.astype(np.uint32) << 16).view(np.float32)
        else:
            np_data = np_data.reshape(shape)
            if dtype_str == "F16":
                np_data = np_data.astype(np.float32)

        data = np_data.flatten().tolist()
        shape_list = list(np_data.shape)
        ftensor = _core.tensor_from_list(data, shape_list)
        result[tensor_name] = ftensor

        return result
