"""Tensor constructors and Tensor type exports."""

import numpy as np

import fastnn._core as _core

Tensor = _core.PyTensor


def _flatten(nested):
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(_flatten(item))
        else:
            result.append(item)
    return result


def tensor(data, shape, device=None):
    flat_data = _flatten(data)
    return _core.tensor_from_data(flat_data, shape, device)


def rand(shape, device=None):
    return _core.rand_uniform(shape, device=device)


def randn(shape, device=None):
    return _core.randn(shape, device=device)


zeros = _core.zeros
ones = _core.ones
full = _core.full
eye = _core.eye
arange = _core.arange
linspace = _core.linspace
randint = _core.randint
zeros_like = _core.zeros_like
ones_like = _core.ones_like
full_like = _core.full_like


def patch_tensor_methods():
    original_numpy = Tensor.numpy

    def numpy_array(self):
        data = original_numpy(self)
        dtype_map = {
            "f32": np.float32,
            "f64": np.float64,
            "i32": np.int32,
            "i64": np.int64,
            "bool": np.bool_,
            "f16": np.float16,
            "bf16": np.float32,
        }
        np_dtype = dtype_map.get(self.dtype, np.float32)
        return np.array(data, dtype=np_dtype).reshape(self.shape)

    Tensor.numpy = numpy_array


__all__ = [
    "Tensor",
    "tensor",
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
    "patch_tensor_methods",
]

