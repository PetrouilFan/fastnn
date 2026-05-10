"""Tensor conversion utilities for fastnn.

Provides centralized functions for converting between fastnn tensors and numpy arrays.
"""

from typing import Any
import numpy as np
import fastnn as fnn


def _infer_shape(lst):
    """Infer the shape of a regular nested list.
    
    Raises:
        ValueError: If the input is a ragged/irregular list (elements have different lengths).
    """
    shape = []
    current = lst
    while isinstance(current, list):
        if not current:
            shape.append(0)
            break
        shape.append(len(current))
        first_len = len(current[0])
        for item in current:
            if not isinstance(item, list):
                break
            if len(item) != first_len:
                raise ValueError(
                    f"Irregular/ragged list detected: elements have inconsistent lengths "
                    f"at depth {len(shape)}. First element has length {first_len}, "
                    f"but found element with length {len(item)}. "
                    f"Only regular (rectangular) nested lists are supported."
                )
        current = current[0]
    return tuple(shape)


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
            return np.array(result)
        return result
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor)


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
    if isinstance(array, np.ndarray):
        return fnn.from_numpy(array, device)
    # Infer shape from nested list and pass directly to fnn.tensor
    shape = _infer_shape(array)
    return fnn.tensor(array, shape, device)

__all__ = ["to_numpy", "to_tensor"]