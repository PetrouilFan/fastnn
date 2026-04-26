"""Serialization module for fastnn.

Provides functions to save and load models, state dicts, and optimizer states.
Supports versioned serialization format for backward and forward compatibility.
"""

import struct
import warnings
from typing import Dict, Any, Optional, Union
import numpy as np

from fastnn.core import tensor
from fastnn._core import FastnnError, IoError

# Current serialization format version
CURRENT_VERSION = 2

# Magic bytes for fastnn model files
MODEL_MAGIC = b"FNN\x00"
# Magic bytes for optimizer state files
OPTIMIZER_MAGIC = b"FNO\x00"


class SerializationError(IoError):
    """Raised when serialization or deserialization fails."""
    pass


def _write_tensor(f, name: str, data: np.ndarray) -> None:
    """Write a tensor to the file.
    
    Args:
        f: File object opened in binary write mode.
        name: Parameter name.
        data: Tensor data as numpy array (float32).
    """
    name_bytes = name.encode("utf-8")
    shape = list(data.shape)
    data_f32 = data.astype(np.float32).flatten()
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<Q", len(shape)))
    for d in shape:
        f.write(struct.pack("<q", d))
    f.write(struct.pack("<Q", len(data_f32)))
    f.write(data_f32.tobytes())


def _read_tensor(f) -> tuple[str, np.ndarray]:
    """Read a tensor from the file.
    
    Returns:
        Tuple of (name, data) where data is numpy array.
    """
    name_len = struct.unpack("<Q", f.read(8))[0]
    name = f.read(name_len).decode("utf-8")
    shape_len = struct.unpack("<Q", f.read(8))[0]
    shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_len)]
    data_len = struct.unpack("<Q", f.read(8))[0]
    data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
    return name, data.reshape(shape)


def save_model(model: Any, path: str, version: int = CURRENT_VERSION) -> None:
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
        >>> fnn.save_model(model, "model.fnn")
    """
    try:
        params = model.parameters() if hasattr(model, "parameters") else []
        named = model.named_parameters() if hasattr(model, "named_parameters") else []
        if named:
            param_list = named
        else:
            param_list = [(f"param_{i}", p) for i, p in enumerate(params)]
        
        with open(path, "wb") as f:
            f.write(MODEL_MAGIC)
            f.write(struct.pack("<I", version))
            
            if version == 1:
                # Original format: no version field per tensor, no grad storage
                f.write(struct.pack("<Q", len(param_list)))
                for name, tensor in param_list:
                    name_bytes = name.encode("utf-8")
                    f.write(struct.pack("<Q", len(name_bytes)))
                    f.write(name_bytes)
                    shape = tensor.shape
                    f.write(struct.pack("<Q", len(shape)))
                    for d in shape:
                        f.write(struct.pack("<q", d))
                    data = tensor.numpy().astype(np.float32).flatten()
                    f.write(struct.pack("<Q", len(data)))
                    f.write(data.tobytes())
            
            elif version == 2:
                # Version 2 adds per-tensor version and optional gradient storage
                f.write(struct.pack("<Q", len(param_list)))
                for name, tensor in param_list:
                    # Write tensor header with version
                    f.write(struct.pack("<I", 2))  # tensor format version
                    _write_tensor(f, name, tensor.numpy().astype(np.float32))
                    # Write gradient if present
                    grad = tensor.grad
                    if grad is not None:
                        f.write(struct.pack("<B", 1))
                        _write_tensor(f, name + ".grad", grad.numpy().astype(np.float32))
                    else:
                        f.write(struct.pack("<B", 0))
            
            else:
                raise SerializationError(f"Unsupported serialization version: {version}")
    
    except (OSError, struct.error, ValueError) as e:
        raise SerializationError(f"Failed to save model: {e}") from e


def load_model(path: str, version: Optional[int] = None) -> Dict[str, Any]:
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
        >>> params = fnn.load_model("model.fnn")
        >>> model.load_state_dict(params)
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != MODEL_MAGIC:
                raise SerializationError("Invalid file format: expected FNN magic bytes")
            
            file_version = struct.unpack("<I", f.read(4))[0]
            if version is not None and file_version != version:
                raise ValueError(f"File version {file_version} does not match expected {version}")
            
            if file_version == 1:
                num_params = struct.unpack("<Q", f.read(8))[0]
                result = {}
                for _ in range(num_params):
                    name_len = struct.unpack("<Q", f.read(8))[0]
                    name = f.read(name_len).decode("utf-8")
                    shape_len = struct.unpack("<Q", f.read(8))[0]
                    shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_len)]
                    data_len = struct.unpack("<Q", f.read(8))[0]
                    data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
                    result[name] = tensor(data.copy(), list(shape))
                return result
            
            elif file_version == 2:
                num_params = struct.unpack("<Q", f.read(8))[0]
                result = {}
                for _ in range(num_params):
                    tensor_version = struct.unpack("<I", f.read(4))[0]
                    if tensor_version != 2:
                        raise SerializationError(f"Unsupported tensor version: {tensor_version}")
                    name, data = _read_tensor(f)
                    result[name] = tensor(data.copy(), list(data.shape))
                    # Read gradient flag
                    has_grad = struct.unpack("<B", f.read(1))[0]
                    if has_grad:
                        grad_name = name + ".grad"
                        _, grad_data = _read_tensor(f)
                        # Gradient storage not automatically attached; ignore for now
                        # Could store in separate dict
                return result
            
            else:
                raise SerializationError(f"Unsupported file version: {file_version}")
    
    except (OSError, struct.error, ValueError) as e:
        raise SerializationError(f"Failed to load model: {e}") from e


def save_state_dict(model: Any, path: str, version: int = CURRENT_VERSION) -> None:
    """Save model state dictionary (parameters and gradients) to a file.
    
    Args:
        model: Model object with `named_parameters()` method.
        path: Path to output file.
        version: Serialization format version.
    
    Examples:
        >>> fnn.save_state_dict(model, "state.fnn")
    """
    # For now, same as save_model but only named parameters
    save_model(model, path, version)


def load_state_dict(path: str) -> Dict[str, Any]:
    """Load state dictionary from file.
    
    Args:
        path: Path to input file.
    
    Returns:
        State dictionary.
    """
    return load_model(path)


def save_optimizer(opt: Any, path: str, version: int = CURRENT_VERSION) -> None:
    """Save optimizer state to a file.
    
    Args:
        opt: Optimizer object with `params`, `lr`, `betas`, `eps`, `weight_decay` attributes.
        path: Path to output file.
        version: Serialization format version.
    
    Raises:
        SerializationError: If saving fails.
    
    Examples:
        >>> optimizer = fnn.SGD(model.parameters(), lr=0.01)
        >>> fnn.save_optimizer(optimizer, "opt.fno")
    """
    try:
        with open(path, "wb") as f:
            f.write(OPTIMIZER_MAGIC)
            f.write(struct.pack("<I", version))
            lr = opt.lr if hasattr(opt, "lr") else 0.0
            f.write(struct.pack("<d", lr))
            betas = opt.betas if hasattr(opt, "betas") else (0.9, 0.999)
            f.write(struct.pack("<d", betas[0]))
            f.write(struct.pack("<d", betas[1]))
            eps = opt.eps if hasattr(opt, "eps") else 1e-8
            f.write(struct.pack("<d", eps))
            wd = opt.weight_decay if hasattr(opt, "weight_decay") else 0.0
            f.write(struct.pack("<d", wd))
            n = len(opt.params) if hasattr(opt, "params") else 0
            f.write(struct.pack("<Q", n))
            for i in range(n):
                for state_tensor_name in ["m", "v", "v_hat"]:
                    state_list = getattr(opt, state_tensor_name, None)
                    if state_list and i < len(state_list):
                        data = state_list[i].numpy().astype(np.float32).flatten()
                        f.write(struct.pack("<B", 1))
                        f.write(struct.pack("<Q", len(data)))
                        f.write(data.tobytes())
                    else:
                        f.write(struct.pack("<B", 0))
                        f.write(struct.pack("<Q", 0))
            step_list = getattr(opt, "step", None)
            if step_list:
                for s in step_list:
                    f.write(struct.pack("<Q", s))
    
    except (OSError, struct.error, ValueError) as e:
        raise SerializationError(f"Failed to save optimizer: {e}") from e


def load_optimizer(opt: Any, path: str) -> None:
    """Load optimizer state from a file.
    
    Args:
        opt: Optimizer object to load state into.
        path: Path to input file.
    
    Raises:
        SerializationError: If loading fails.
    
    Examples:
        >>> optimizer = fnn.SGD(model.parameters(), lr=0.01)
        >>> fnn.load_optimizer(optimizer, "opt.fno")
    """
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != OPTIMIZER_MAGIC:
                raise SerializationError("Invalid optimizer state file")
            
            file_version = struct.unpack("<I", f.read(4))[0]
            # Currently only version 1 supported
            if file_version != 1:
                raise SerializationError(f"Unsupported optimizer version: {file_version}")
            
            opt.lr = struct.unpack("<d", f.read(8))[0]
            b1 = struct.unpack("<d", f.read(8))[0]
            b2 = struct.unpack("<d", f.read(8))[0]
            if hasattr(opt, "betas"):
                opt.betas = (b1, b2)
            opt.eps = struct.unpack("<d", f.read(8))[0]
            if hasattr(opt, "weight_decay"):
                opt.weight_decay = struct.unpack("<d", f.read(8))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            for i in range(n):
                for state_tensor_name in ["m", "v", "v_hat"]:
                    state_list = getattr(opt, state_tensor_name, None)
                    has_data = struct.unpack("<B", f.read(1))[0]
                    data_len = struct.unpack("<Q", f.read(8))[0]
                    if has_data and state_list and i < len(state_list):
                        data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
                        state_list[i].copy_(tensor(data.copy(), list(state_list[i].shape)))
                    elif has_data:
                        f.read(data_len * 4)
            step_list = getattr(opt, "step", None)
            if step_list:
                for i in range(len(step_list)):
                    step_list[i] = struct.unpack("<Q", f.read(8))[0]
    
    except (OSError, struct.error, ValueError) as e:
        raise SerializationError(f"Failed to load optimizer: {e}") from e
