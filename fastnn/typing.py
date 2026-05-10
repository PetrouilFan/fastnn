from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from fastnn._core import PyTensor

# Common tensor-like types
ArrayLike = Union[np.ndarray, List, Tuple]
Shape = Tuple[int, ...]
DType = Literal["f32", "f64", "i32", "i64", "bool", "f16", "bf16"]
Tensor = "PyTensor"  # Fixed recursive definition, forward reference to PyTensor

# Common collections
TensorList = List[Tensor]
ParamList = List[Tuple[str, Tensor]]

# Optional types
OptTensor = Optional[Tensor]
OptShape = Optional[Shape]
OptStr = Optional[str]
OptInt = Optional[int]
OptFloat = Optional[float]
OptBool = Optional[bool]

# Device type
Device = Literal["cpu", "wgpu", "gpu", "gpu:0", "gpu:1", "gpu:2", "gpu:3"]

__all__ = [
    "Any", "Callable", "Dict", "List", "Optional", "Sequence", "Tuple", "Union",
    "ArrayLike", "Shape", "DType", "Tensor", "TensorList", "ParamList",
    "OptTensor", "OptShape", "OptStr", "OptInt", "OptFloat", "OptBool",
    "Device",
]
