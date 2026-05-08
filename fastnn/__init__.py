import numpy as np
import fastnn._core as _core

__version__ = "1.1.0"

# Exception hierarchy - imported from _core (Rust side)
FastnnError = _core.FastnnError
ShapeError = _core.ShapeError
DtypeError = _core.DtypeError
DeviceError = _core.DeviceError
AutogradError = _core.AutogradError
OptimizerError = _core.OptimizerError
IoError = _core.IoError
CudaError = _core.CudaError

from fastnn.core import (  # noqa: E402
    no_grad,
    set_seed,
    set_num_threads,
    set_default_device,
    checkpoint,
)
from fastnn.data import DataLoader, Dataset, TensorDataset  # noqa: E402
from fastnn.callbacks import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    CSVLogger,
)
from fastnn.parallel import DataParallel  # noqa: E402
from fastnn.tensor import _flatten  # noqa: F401
from fastnn.tensor import from_numpy as tensor_from_numpy  # noqa: F401
from fastnn.layers import Flatten, PySequential, BasicBlock, MaxPool2d  # noqa: F401, E402
from fastnn.io import (  # noqa: E402
    save as io_save,
    load as io_load,
    convert_from_pytorch,
    convert_from_onnx,
)

__all__ = [
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "checkpoint",
    "Tensor",
    "DataLoader",
    "Dataset",
    "TensorDataset",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "CSVLogger",
    "DataParallel",
    "models",
    # Exception hierarchy
    "FastnnError",
    "ShapeError",
    "DtypeError",
    "DeviceError",
    "AutogradError",
    "OptimizerError",
    "IoError",
    "CudaError",
]


def __getattr__(name):
    if name == "models":
        import fastnn.models

        return fastnn.models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Ensure 'models' is resolvable for 'from fastnn import *' across Python versions
try:
    import fastnn.models as models
except ImportError:
    pass





def _patch_numpy(tensor_cls):
    _original_numpy = tensor_cls.numpy

    def _new_numpy(self):
        data = _original_numpy(self)
        shape = self.shape
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
        return np.array(data, dtype=np_dtype).reshape(shape)

    tensor_cls.numpy = _new_numpy


def _patch_backward(tensor_cls):
    _original_backward = tensor_cls.backward

    def _new_backward(self, grad=None):
        return _original_backward(self, grad)

    tensor_cls.backward = _new_backward


_patch_numpy(_core.PyTensor)
_patch_backward(_core.PyTensor)

Tensor = _core.PyTensor
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


# Re-export with proper device handling
def rand(shape, device=None):
    """Generate random tensor with uniform distribution."""
    return _core.rand_uniform(shape, device=device)


def randn(shape, device=None):
    """Generate random tensor with normal distribution."""
    return _core.randn(shape, device=device)


add = _core.add
sub = _core.sub
mul = _core.mul
div = _core.div
matmul = _core.matmul
im2col = _core.im2col
neg = _core.neg
abs = _core.abs
exp = _core.exp
log = _core.log
sqrt = _core.sqrt
pow = _core.pow
clamp = _core.clamp
relu = _core.relu
fused_add_relu = _core.fused_add_relu
gelu = _core.gelu
sigmoid = _core.sigmoid
tanh = _core.tanh
silu = _core.silu
softmax = _core.softmax
log_softmax = _core.log_softmax
fused_conv_bn_silu = _core.fused_conv_bn_silu
FusedConvBn = _core.FusedConvBn
FusedConvBnRelu = _core.FusedConvBnRelu
FusedConvBnGelu = _core.FusedConvBnGelu
argmax = _core.argmax
argmin = _core.argmin
cat = _core.cat
stack = _core.stack
sum = _core.sum
mean = _core.mean

max = _core.max
min = _core.min
maximum = _core.maximum
minimum = _core.minimum
mse_loss = _core.mse_loss
cross_entropy_loss = _core.cross_entropy_loss
Linear = _core.Linear
Conv2d = _core.Conv2d
LayerNorm = _core.LayerNorm
BatchNorm1d = _core.BatchNorm1d
Dropout = _core.Dropout
Embedding = _core.Embedding
ReLU = _core.ReLU
GELU = _core.Gelu
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU
Sequential = _core.Sequential_

ModuleList = _core.ModuleList
SGD = _core.PySGD
Adam = _core.PyAdam
AdamW = _core.PyAdamW
Muon = _core.PyMuon
LeakyReLU = _core.LeakyReLU
Softplus = _core.Softplus
Hardswish = _core.Hardswish
cat = _core.cat
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
BatchNorm2d = _core.BatchNorm2d
Lion = _core.PyLion
bce_with_logits = _core.bce_with_logits
huber_loss = _core.huber_loss
ConvTranspose2d = _core.ConvTranspose2d
Conv1d = _core.Conv1d
Conv3d = _core.Conv3d
einsum = _core.einsum
flash_attention = _core.flash_attention
ResidualBlock = _core.ResidualBlock
clip_grad_norm_ = _core.clip_grad_norm_
clip_grad_value_ = _core.clip_grad_value_
Dropout2d = _core.Dropout2d
Upsample = _core.Upsample
RMSprop = _core.PyRMSprop
Elu = _core.Elu
Mish = _core.Mish
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d


from fastnn.schedulers import LRScheduler, StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau  # noqa: F401

allocator_stats = _core.allocator_stats
list_registered_ops = _core.list_registered_ops
batched_mlp_forward = _core.batched_mlp_forward


# Make fastnn.tensor module callable (delegates to tensor.tensor function)
class _TensorModuleWrapper:
    """Wrapper that makes the tensor module callable."""

    def __init__(self, module):
        object.__setattr__(self, "_module", module)

    def __call__(self, *args, **kwargs):
        return self._module.tensor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __setattr__(self, name, value):
        setattr(self._module, name, value)

    def __delattr__(self, name):
        delattr(self._module, name)


import sys

_tensor_module = sys.modules.get("fastnn.tensor")
if _tensor_module is not None:
    _tensor_wrapper = _TensorModuleWrapper(_tensor_module)
    sys.modules["fastnn.tensor"] = _tensor_wrapper
    import fastnn

    fastnn.__dict__["tensor"] = _tensor_wrapper
