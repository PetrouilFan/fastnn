"""Tensor constructors and Tensor type exports."""

import numpy as np

import fastnn._core as _core

Tensor = _core.PyTensor


def _flatten(nested):
    """Flatten a nested list iteratively."""
    result = []
    stack = [iter(nested)]
    while stack:
        try:
            item = next(stack[-1])
            if isinstance(item, list):
                stack.append(iter(item))
            else:
                result.append(item)
        except StopIteration:
            stack.pop()
    return result

def tensor(data, shape, device=None):
    flat_data = _flatten(data)
    return _core.tensor_from_data(flat_data, shape, device)


def zeros(shape, device=None):
    return _core.zeros(shape, device=device)


def ones(shape, device=None):
    return _core.ones(shape, device=device)


def full(shape, value: float, device=None):
    return _core.full(shape, value, device=device)


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
    return _core.randint(low, high, shape, device=device)


def zeros_like(tensor, device=None):
    return _core.zeros_like(tensor, device=device)


def ones_like(tensor, device=None):
    return _core.ones_like(tensor, device=device)


def full_like(tensor, value: float, device=None):
    return _core.full_like(tensor, value, device=device)


def from_numpy(ndarray):
    """Create a tensor from a numpy array using zero-copy buffer protocol."""
    # Ensure the array is contiguous and of dtype float32
    if not ndarray.flags['C_CONTIGUOUS']:
        ndarray = np.ascontiguousarray(ndarray)
    if ndarray.dtype != np.float32:
        ndarray = ndarray.astype(np.float32)
    return _core.tensor_from_buffer(ndarray)


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

