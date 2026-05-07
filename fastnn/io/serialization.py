"""Serialization module for fastnn.

Provides functions to save and load models, state dicts, and optimizer states.
Supports versioned serialization format for backward and forward compatibility.
"""

from typing import Dict, Any, Optional
import numpy as np
import struct

from fastnn.tensor import tensor
from fastnn._core import IoError
from fastnn.io import (
    write_tensor,
    read_tensor,
    _pack_u64,
    _pack_i64,
    _pack_u32,
    _pack_u8,
    _pack_f64,
    _unpack_u64,
    _unpack_i64,
    _unpack_u32,
    _unpack_u8,
    _unpack_f64,
    MODEL_MAGIC,
    OPTIMIZER_MAGIC,
    MODEL_VERSION,
    OPTIMIZER_VERSION,
)
from fastnn.utils.tensor_utils import to_numpy

# Current serialization format version
CURRENT_VERSION = MODEL_VERSION


class SerializationError(IoError):
    """Raised when serialization or deserialization fails."""
    pass


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
            f.write(_pack_u32(version))
            
            if version == 1:
                # Version 1 format: uses write_tensor for consistency
                f.write(_pack_u64(len(param_list)))
                for name, tensor in param_list:
                    write_tensor(f, name, to_numpy(tensor))
            
            elif version == 2:
                # Version 2 adds per-tensor version and optional gradient storage
                f.write(_pack_u64(len(param_list)))
                for name, tensor in param_list:
                    # Write tensor using unified format
                    write_tensor(f, name, to_numpy(tensor))
                    # Write gradient if present
                    grad = tensor.grad
                    if grad is not None:
                        f.write(_pack_u8(1))
                        write_tensor(f, name + ".grad", to_numpy(grad))
                    else:
                        f.write(_pack_u8(0))
            
            else:
                raise SerializationError(f"Unsupported serialization version: {version}")
    
    except (OSError, ValueError) as e:
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
            
            file_version = _unpack_u32(f.read(4))
            if version is not None and file_version != version:
                raise ValueError(f"File version {file_version} does not match expected {version}")
            
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
                    # Read gradient flag
                    has_grad = _unpack_u8(f.read(1))
                    if has_grad:
                        grad_name = name + ".grad"
                        _, grad_data = read_tensor(f)
                        # Attach gradient to the tensor
                        grad_tensor = tensor(grad_data, list(grad_data.shape))
                        # Store gradient in autograd metadata if available
                        if hasattr(result[name], 'inner') and hasattr(result[name].inner, 'autograd_meta'):
                            meta = result[name].inner.autograd_meta
                            if meta is not None:
                                meta.grad = grad_tensor
                        # Also store in result dict for access
                        result[grad_name] = grad_tensor
                return result
            
            else:
                raise SerializationError(f"Unsupported file version: {file_version}")
    
    except (OSError, ValueError) as e:
        raise SerializationError(f"Failed to load model: {e}") from e


def save_state_dict(model: Any, path: str, version: int = CURRENT_VERSION) -> None:
    """Save model state dictionary (parameters and gradients) to a file.
    
    Deprecated: Use save_model instead.
    """
    import warnings
    warnings.warn("save_state_dict is deprecated, use save_model instead", DeprecationWarning, stacklevel=2)
    save_model(model, path, version)


def load_state_dict(path: str) -> Dict[str, Any]:
    """Load state dictionary from file.
    
    Deprecated: Use load_model instead.
    """
    import warnings
    warnings.warn("load_state_dict is deprecated, use load_model instead", DeprecationWarning, stacklevel=2)
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
            f.write(_pack_u32(version))
            lr = opt.lr if hasattr(opt, "lr") else 0.0
            f.write(_pack_f64(lr))
            betas = opt.betas if hasattr(opt, "betas") else (0.9, 0.999)
            f.write(_pack_f64(betas[0]))
            f.write(_pack_f64(betas[1]))
            eps = opt.eps if hasattr(opt, "eps") else 1e-8
            f.write(_pack_f64(eps))
            wd = opt.weight_decay if hasattr(opt, "weight_decay") else 0.0
            f.write(_pack_f64(wd))
            n = len(opt.params) if hasattr(opt, "params") else 0
            f.write(_pack_u64(n))
            for i in range(n):
                for state_tensor_name in ["m", "v", "v_hat"]:
                    state_list = getattr(opt, state_tensor_name, None)
                    if state_list and i < len(state_list):
                        data = to_numpy(state_list[i]).astype(np.float32, copy=False).ravel()
                        f.write(_pack_u8(1))
                        f.write(_pack_u64(len(data)))
                        f.write(data.tobytes())
                    else:
                        f.write(_pack_u8(0))
                        f.write(_pack_u64(0))
            step_list = getattr(opt, "step", None)
            if step_list and not callable(step_list):
                for s in step_list:
                    f.write(_pack_u64(s))
    
    except (OSError, ValueError) as e:
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
            
            file_version = _unpack_u32(f.read(4))
            # Currently only version 1 and 2 supported
            if file_version not in (1, 2):
                raise SerializationError(f"Unsupported optimizer version: {file_version}")
            
            opt.lr = _unpack_f64(f.read(8))
            b1 = _unpack_f64(f.read(8))
            b2 = _unpack_f64(f.read(8))
            if hasattr(opt, "betas"):
                opt.betas = (b1, b2)
            opt.eps = _unpack_f64(f.read(8))
            if hasattr(opt, "weight_decay"):
                opt.weight_decay = _unpack_f64(f.read(8))
            n = _unpack_u64(f.read(8))
            for i in range(n):
                for state_tensor_name in ["m", "v", "v_hat"]:
                    state_list = getattr(opt, state_tensor_name, None)
                    has_data = _unpack_u8(f.read(1))
                    data_len = _unpack_u64(f.read(8))
                    if has_data and state_list and i < len(state_list):
                        data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
                        state_list[i].copy_(tensor(data.copy(), list(state_list[i].shape)))
                    elif has_data:
                        f.read(data_len * 4)
            step_list = getattr(opt, "step", None)
            if step_list:
                for i in range(len(step_list)):
                    step_list[i] = _unpack_u64(f.read(8))
    
    except (OSError, ValueError) as e:
        raise SerializationError(f"Failed to load optimizer: {e}") from e
