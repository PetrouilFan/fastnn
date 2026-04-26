"""Optimization algorithms for fastnn.

This module provides various optimization algorithms, including SGD, Adam, AdamW,
Muon, Lion, and RMSprop. All optimizers are implemented in Rust for performance.

Examples:
    >>> import fastnn as fnn
    >>> model = fnn.models.MLP(input_dim=784, hidden_dims=[256], output_dim=10)
    >>> optimizer = fnn.optimizers.SGD(model.parameters(), lr=0.01)
    >>> loss = ...
    >>> loss.backward()
    >>> optimizer.step()
"""

import warnings
from typing import Any, List, Optional, Union

import fastnn._core as _core

# Re-export optimizer classes, removing 'Py' prefix for cleaner API.
# The original Rust classes are named PySGD, PyAdam, etc. We provide aliases
# without the prefix for user convenience.

SGD = _core.PySGD
Adam = _core.PyAdam
AdamW = _core.PyAdamW
Muon = _core.PyMuon
Lion = _core.PyLion
RMSprop = _core.PyRMSprop

# Additional utility functions for optimizers (if any)

def clip_grad_norm_(parameters: Union[Any, List[Any]], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradient norm of parameters.
    
    Args:
        parameters: Iterable of parameters or a single parameter.
        max_norm: Maximum allowed gradient norm.
        norm_type: Type of norm (default: 2.0).
    
    Returns:
        The total norm of the parameters (before clipping).
    
    Examples:
        >>> optimizer = fnn.optimizers.SGD(model.parameters(), lr=0.01)
        >>> loss.backward()
        >>> fnn.optimizers.clip_grad_norm_(model.parameters(), 1.0)
        >>> optimizer.step()
    """
    return _core.clip_grad_norm_(parameters, max_norm, norm_type)


def clip_grad_value_(parameters: Union[Any, List[Any]], clip_value: float) -> None:
    """Clip gradient values of parameters.
    
    Args:
        parameters: Iterable of parameters or a single parameter.
        clip_value: Maximum allowed absolute value for gradients.
    
    Examples:
        >>> fnn.optimizers.clip_grad_value_(model.parameters(), 0.5)
    """
    _core.clip_grad_value_(parameters, clip_value)
