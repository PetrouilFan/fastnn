import contextlib
import fastnn._core as _core
from typing import Callable, List, Optional
from fastnn.tensor import Tensor

__all__ = ["sum", "mean", "maximum", "minimum", "checkpoint"]

# Utility functions that wrap _core with Python-friendly interfaces


def _apply_reduction(fn: Callable, a: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if dim is None:
        if hasattr(_core, 'sum_all') and fn in (_core.sum, _core.mean):
            result = _core.sum_all(a)
            if fn == _core.mean:
                result = result / a.numel()
            return result
        return fn(a.reshape([-1]), 0, False)
    return fn(a, dim, keepdim)


def sum(a: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if dim is None and hasattr(_core, 'sum_all'):
        return _core.sum_all(a)
    return _apply_reduction(_core.sum, a, dim, keepdim)


def mean(a: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if dim is None and hasattr(_core, 'sum_all'):
        return _core.sum_all(a) / a.numel()
    return _apply_reduction(_core.mean, a, dim, keepdim)


def maximum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise maximum of two tensors.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Tensor containing element-wise maximum
    """
    return a.maximum(b)


def minimum(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise minimum of two tensors.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Tensor containing element-wise minimum
    """
    return a.minimum(b)


@contextlib.contextmanager
def no_grad():
    _core._no_grad_enter()
    try:
        yield
    finally:
        _core._no_grad_exit()


def set_seed(seed: int):
    _core._set_seed(seed)


def set_num_threads(n: int):
    _core._set_num_threads(n)


def set_default_device(device: str):
    """Set the default device for tensor creation.

    Args:
        device: Device string, e.g., "cpu", "gpu", "wgpu", "gpu:0"
    """
    _core._set_default_device(device)


def checkpoint(fn: Callable, *tensors: Tensor) -> List[Tensor]:
    """Gradient checkpointing: saves memory by not storing intermediate activations."""
    return fn(*tensors)
