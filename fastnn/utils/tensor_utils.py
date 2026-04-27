"""Tensor conversion utilities for fastnn.

Provides centralized functions for converting between fastnn tensors and numpy arrays.
"""

from typing import Any
import numpy as np


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert a fastnn tensor to a numpy array.
    
    If the input is already a numpy array or other type, it is returned as-is.
    
    Args:
        tensor: A fastnn tensor (with .numpy() method) or any other object.
    
    Returns:
        Numpy array representation of the input.
    
    Examples:
        >>> import fastnn as fnn
        >>> t = fnn.randn([3, 4])
        >>> arr = to_numpy(t)
        >>> isinstance(arr, np.ndarray)
        True
    """
    if hasattr(tensor, "numpy"):
        result = tensor.numpy()
        if not isinstance(result, np.ndarray):
            return np.array(result, dtype=np.float32)
        return result
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor, dtype=np.float32)


def to_tensor(array: Any, device: Any = None) -> Any:
    """Convert a numpy array (or list) to a fastnn tensor.
    
    If the input is already a fastnn tensor, it is returned as-is.
    
    Args:
        array: A numpy array, list, or other array-like object.
        device: Optional device specification for the tensor.
    
    Returns:
        fastnn tensor representation of the input.
    
    Examples:
        >>> import numpy as np
        >>> import fastnn as fnn
        >>> arr = np.random.randn(3, 4).astype(np.float32)
        >>> t = to_tensor(arr)
        >>> hasattr(t, "numpy")
        True
    """
    if hasattr(array, "numpy"):
        return array
    import fastnn as fnn
    if isinstance(array, np.ndarray):
        return fnn.tensor(array.flatten().tolist(), list(array.shape), device=device)
    # Handle list by converting to numpy first
    arr = np.array(array)
    return fnn.tensor(arr.flatten().tolist(), list(arr.shape), device=device)