"""Neural network module aliases exported by the Rust core.

.. deprecated::
    Use `fastnn.layers` or direct imports from `fastnn` instead.
    This module is kept for backward compatibility.
"""

import warnings
warnings.warn(
    "fastnn.nn is deprecated, use fastnn.layers or direct fastnn imports instead",
    DeprecationWarning,
    stacklevel=2
)

import fastnn._core as _core

Linear = _core.Linear
Conv1d = _core.Conv1d
Conv2d = _core.Conv2d
Conv3d = _core.Conv3d
ConvTranspose2d = _core.ConvTranspose2d
MaxPool2d = _core.MaxPool2d
LayerNorm = _core.LayerNorm
BatchNorm1d = _core.BatchNorm1d
BatchNorm2d = _core.BatchNorm2d
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
Dropout = _core.Dropout
Dropout2d = _core.Dropout2d
Embedding = _core.Embedding
Upsample = _core.Upsample
ReLU = _core.ReLU
GELU = _core.Gelu
Gelu = _core.Gelu
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU
LeakyReLU = _core.LeakyReLU
Softplus = _core.Softplus
Hardswish = _core.Hardswish
Elu = _core.Elu
Mish = _core.Mish
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d
Sequential = _core.Sequential_
ModuleList = _core.ModuleList
TransformerEncoder = _core.PyTransformerEncoder

__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose2d",
    "MaxPool2d",
    "LayerNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "RMSNorm",
    "GroupNorm",
    "Dropout",
    "Dropout2d",
    "Embedding",
    "Upsample",
    "ReLU",
    "GELU",
    "Gelu",
    "Sigmoid",
    "Tanh",
    "SiLU",
    "LeakyReLU",
    "Softplus",
    "Hardswish",
    "Elu",
    "Mish",
    "AdaptiveAvgPool2d",
    "Sequential",
    "ModuleList",
    "TransformerEncoder",
]

