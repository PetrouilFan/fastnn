"""Tensor constructors and Tensor type exports."""

import math

import numpy as np

import fastnn._core as _core

Tensor = _core.PyTensor


def _ensure_tensor_ready(arr):
    """Ensure numpy array is C-contiguous and float32 dtype."""
    if arr.flags['C_CONTIGUOUS'] and arr.dtype == np.float32:
        return arr
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def _flatten(nested):
    """Flatten a nested list iteratively."""
    result = []
    stack = [iter(nested)]
    while stack:
        try:
            item = next(stack[-1])
            if isinstance(item, (list, tuple)):
                stack.append(iter(item))
            else:
                result.append(item)
        except StopIteration:
            stack.pop()
    return result

def tensor(data, shape, device=None, dtype=None):
    if dtype is not None:
        raise ValueError("dtype not yet supported")
    if isinstance(data, np.ndarray):
        data = _ensure_tensor_ready(data)
        shape = shape if shape is not None else list(data.shape)
        return _core.tensor_from_data(data.flatten().tolist(), shape, device)
    flat_data = _flatten(data)
    expected_size = math.prod(shape) if shape else len(flat_data)
    if len(flat_data) != expected_size:
        raise ValueError(f"Shape {shape} has {expected_size} elements but got {len(flat_data)} values")
    return _core.tensor_from_data(flat_data, shape, device)


def zeros(shape, dtype=None, device=None):
    return _core.zeros(shape, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None):
    return _core.ones(shape, dtype=dtype, device=device)


def full(shape, value: float, dtype=None, device=None):
    return _core.full(shape, value, dtype=dtype, device=device)


def eye(n: int, device=None):
    return _core.eye(n, device=device)


def arange(start: float, end: float, step: float = 1.0, device=None):
    return _core.arange(start, end, step, device=device)


def linspace(start: float, end: float, steps: int, device=None):
    return _core.linspace(start, end, steps, device=device)


def rand(shape, device=None):
    return _core.rand_uniform(shape, device=device)


def randn(shape, device=None):
    return _core.randn(shape, device=device)


def randint(low: int, high: int, shape, device=None):
    return _core.randint(shape, low, high, device=device)


def zeros_like(tensor, device=None):
    return _core.zeros_like(tensor, device=device)


def ones_like(tensor, device=None):
    return _core.ones_like(tensor, device=device)


def full_like(tensor, value: float, device=None):
    return _core.full_like(tensor, value, device=device)


def from_numpy(ndarray, device=None):
    """Create a tensor from a numpy array using zero-copy buffer protocol."""
    return tensor(ndarray, None, device)


__all__ = [
    "Tensor",
    "tensor",
    "from_numpy",
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "rand",
    "randn",
    "randint",
    "zeros_like",
    "ones_like",
    "full_like",
]

