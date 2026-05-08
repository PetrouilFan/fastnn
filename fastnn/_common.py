"""Common imports and utilities shared across fastnn modules."""

import numpy as np
import struct

# Re-export for convenience
__all__ = ['np', 'struct', 'first_or_self']


def first_or_self(value):
    """Extract first element if value is a tuple/list, otherwise return value.
    
    Useful for converting PyTorch-style tuple parameters (kernel_size, stride, etc.)
    to scalar values when they are symmetric.
    """
    if isinstance(value, (tuple, list)):
        return value[0]
    return value
