"""Neural network modules (re-exports from core)."""

import fastnn._core as _core

Linear = _core.Linear
Conv2d = _core.Conv2d
Conv1d = _core.Conv1d
Conv3d = _core.Conv3d
ConvTranspose2d = _core.ConvTranspose2d
LayerNorm = _core.LayerNorm
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
BatchNorm1d = _core.BatchNorm1d
BatchNorm2d = _core.BatchNorm2d
Dropout = _core.Dropout
Dropout2d = _core.Dropout2d
Embedding = _core.Embedding
Upsample = _core.Upsample
MaxPool2d = _core.MaxPool2d
AvgPool2d = _core.AvgPool2d
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d
ReLU = _core.ReLU
GELU = _core.Gelu  # Alias for Rust's Gelu (case matches Rust convention)
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU
LeakyReLU = _core.LeakyReLU
Softplus = _core.Softplus
Hardswish = _core.Hardswish
Elu = _core.Elu
Softmax = _core.Softmax
PReLU = _core.PReLU
Mish = _core.Mish
Sequential = _core.Sequential_
ModuleList = _core.ModuleList
ResidualBlock = _core.ResidualBlock
FusedConvBn = _core.FusedConvBn
FusedConvBnRelu = _core.FusedConvBnRelu
FusedConvBnGelu = _core.FusedConvBnGelu

# Python layers
from fastnn.layers import Flatten, PySequential, BasicBlock

# Weight initialization (use submodule import to avoid __getattr__ fallback)
import fastnn.init as init


# Model classes
def MLP(*args, **kwargs):
    from fastnn.models.mlp import MLP
    return MLP(*args, **kwargs)


__all__ = [
    "Linear",
    "Conv2d",
    "Conv1d",
    "Conv3d",
    "ConvTranspose2d",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm1d",
    "BatchNorm2d",
    "Dropout",
    "Dropout2d",
    "Embedding",
    "Upsample",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "SiLU",
    "LeakyReLU",
    "Softplus",
    "Hardswish",
    "Elu",
    "Softmax",
    "PReLU",
    "Mish",
    "Sequential",
    "ModuleList",
    "ResidualBlock",
    "FusedConvBn",
    "FusedConvBnRelu",
    "FusedConvBnGelu",
    "Flatten",
    "PySequential",
    "BasicBlock",
    "MLP",
    "init",
]
