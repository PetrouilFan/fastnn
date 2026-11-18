"""Serialization module for fastnn.

Provides functions to save and load models and state dicts.
Supports a versioned serialization format.
"""
__all__ = ["SerializationError"]

from typing import Dict, Any, Optional
import json
import numpy as np
import struct

from fastnn.tensor import tensor
from fastnn.io import (
    SerializationError,
    write_tensor,
    read_tensor,
    _pack_u64,
    _pack_u32,
    _pack_u8,
    _pack_f64,
    _unpack_u64,
    _unpack_u32,
    _unpack_u8,
    _unpack_f64,
    MODEL_MAGIC,
    MODEL_VERSION,
    serialization_error,
)
from fastnn.utils.tensor_utils import to_numpy


def _save_model(model: Any, path: str, version: int = MODEL_VERSION) -> None:
    """Save a model's parameters to a file.

    The serialization format is versioned to allow future extensions while
    maintaining backward compatibility.

    Args:
        model: Model object with `parameters()` or `named_parameters()` method.
        path: Path to output file (should end with .fnn).
        version: Serialization format version to use (default: latest).

    Raises:
        SerializationError: If writing fails or version is unsupported.
        ValueError: If model does not have required methods.

    Examples:
        >>> import fastnn as fnn
        >>> model = fnn.models.MLP(input_dim=10, hidden_dims=[20], output_dim=1)
        >>> fnn.io.save(model, "model.fnn")
    """
    with serialization_error("save model"):
        params = model.parameters() if hasattr(model, "parameters") else []
        named = model.named_parameters() if hasattr(model, "named_parameters") else []
        if named:
            param_list = named
        else:
            param_list = [(f"param_{i}", p) for i, p in enumerate(params)]

        with open(path, "wb") as f:
            f.write(MODEL_MAGIC)
            f.write(_pack_u32(version))

            if version == 1:
                # Version 1 format: uses write_tensor for consistency
                f.write(_pack_u64(len(param_list)))
                for name, tensor in param_list:
                    write_tensor(f, name, to_numpy(tensor))

            elif version == 2:
                # Version 2 has a per-tensor gradient flag. Inference-only tensors
                # have no gradient, so the flag is written unset; the reader still
                # accepts a set flag so archived v2 files stay readable.
                f.write(_pack_u64(len(param_list)))
                for name, tensor in param_list:
                    write_tensor(f, name, to_numpy(tensor))
                    f.write(_pack_u8(0))

            elif version == 3:
                # Version 3: dtype-tagged packed tensors (U4, U8, F16, F32)
                from fastnn.io import write_fnn_file
                header = {
                    "version": 3,
                    "num_params": len(param_list),
                }
                # Convert parameters to v3 format: (name, data_f32, DTYPE_F32, [], [])
                params_v3 = []
                for name, tensor in param_list:
                    data_f32 = to_numpy(tensor).astype(np.float32, copy=False)
                    params_v3.append((name, data_f32))
                write_fnn_file(f, header, params_v3, version=3)

            else:
                raise SerializationError(f"Unsupported serialization version: {version}")


def _load_model(path: str, version: Optional[int] = None) -> Dict[str, Any]:
    """Load model parameters from a file.

    Automatically detects the file version and loads accordingly.

    Args:
        path: Path to input file.
        version: Expected version (optional). If provided, mismatch raises error.

    Returns:
        Dictionary mapping parameter names to tensors.

    Raises:
        SerializationError: If reading fails or file format is invalid.
        ValueError: If version mismatch.

    Examples:
        >>> params = fnn.io.load("model.fnn")
    """
    with serialization_error("load model"):
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MODEL_MAGIC:
                raise SerializationError("Invalid file format: expected FNN magic bytes")

            file_version = _unpack_u32(f.read(4))
            if version is not None and file_version != version:
                raise SerializationError(f"File version {file_version} does not match expected {version}")

            if file_version == 1:
                num_params = _unpack_u64(f.read(8))
                result = {}
                for _ in range(num_params):
                    name, data = read_tensor(f)
                    result[name] = tensor(data, list(data.shape))
                return result

            elif file_version == 2:
                num_params = _unpack_u64(f.read(8))
                result = {}
                for _ in range(num_params):
                    name, data = read_tensor(f)
                    result[name] = tensor(data, list(data.shape))
                    # Read the gradient flag. Gradients are no longer attached to
                    # tensors, but the archived payload must still be consumed so
                    # the following tensors decode correctly.
                    has_grad = _unpack_u8(f.read(1))
                    if has_grad:
                        _, grad_data = read_tensor(f)
                        result[name + ".grad"] = tensor(grad_data, list(grad_data.shape))
                return result

            elif file_version >= 3:
                # v3+: dtype-tagged packed tensors — use unified reader
                from fastnn.io import read_fnn_parameters
                # Already read magic + version (8 bytes), continue from current position
                header_len = _unpack_u64(f.read(8))
                header_bytes = f.read(header_len)
                _header = json.loads(header_bytes.decode("utf-8"))
                num_params = _unpack_u64(f.read(8))
                raw_params = read_fnn_parameters(f, num_params, version=file_version)
                result = {}
                for name, val in raw_params.items():
                    if isinstance(val, tuple):
                        data, dtype, scales, zeros, shape = val
                        if isinstance(data, np.ndarray):
                            result[name] = tensor(data, list(data.shape))
                        else:
                            # Packed quantized data — dequantize with scales/zeros
                            raw = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
                            s = np.asarray(scales, dtype=np.float32)
                            z = np.asarray(zeros, dtype=np.float32)
                            if s.size > 0:
                                dequantized = s * (raw - z)
                            else:
                                dequantized = raw
                            if shape:
                                dequantized = dequantized.reshape(shape)
                            result[name] = tensor(dequantized, list(dequantized.shape))
                    else:
                        result[name] = tensor(val, list(val.shape))
                return result

            else:
                raise SerializationError(f"Unsupported file version: {file_version}")


def _save_state_dict(model: Any, path: str, version: int = MODEL_VERSION) -> None:
    """Save a model state dictionary (parameters) to a file.

    Args:
        model: Model object with `named_parameters()` method.
        path: Path to output file.
        version: Serialization format version.
    """
    _save_model(model, path, version)


def _load_state_dict(path: str) -> Dict[str, Any]:
    """Load state dictionary from file."""
    return _load_model(path)
