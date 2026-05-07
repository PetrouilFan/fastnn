"""Common type definitions for fastnn."""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import numpy as np
    import fastnn._core as _core

# Common tensor-like types
ArrayLike = Union["np.ndarray", List, Tuple]
Shape = Tuple[int, ...]
DType = str
Tensor = "_core.PyTensor"

# Common collections
TensorList = List["Tensor"]
ParamList = List[Tuple[str, "Tensor"]]

# Optional types
OptTensor = Optional["Tensor"]
OptShape = Optional[Shape]
OptStr = Optional[str]
OptInt = Optional[int]
OptFloat = Optional[float]
OptBool = Optional[bool]

__all__ = [
    'Any', 'Callable', 'Dict', 'List', 'Optional', 'Sequence', 'Tuple', 'Union',
    'ArrayLike', 'Shape', 'DType', 'Tensor', 'TensorList', 'ParamList',
    'OptTensor', 'OptShape', 'OptStr', 'OptInt', 'OptFloat', 'OptBool',
]
