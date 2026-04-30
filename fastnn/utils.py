"""Helper utilities for converting numpy arrays to fastnn tensors."""

import numpy as np
import fastnn._core as _core


def tensor_from_array(np_array):
    """Convert a numpy array to a fastnn tensor.
    
    Args:
        np_array: numpy.ndarray
        
    Returns:
        fastnn._core.PyTensor
    """
    data = np_array.flatten().tolist()
    shape = list(np_array.shape)
    return _core.tensor_from_list(data, shape)
