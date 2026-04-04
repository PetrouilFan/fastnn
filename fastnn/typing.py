from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from fastnn._core import PyTensor

Tensor = Union["PyTensor", "Tensor"]
Device = Literal["cpu", "wgpu"]
DType = Literal["f32", "f64", "i32", "i64", "bool", "f16", "bf16"]
