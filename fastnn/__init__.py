import numpy as np
import fastnn._core as _core

__version__ = "1.0.0"

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
from fastnn.layers import (
    MaxPool2dPy,
    Flatten,
    PySequential,
    BasicBlock,
)
from fastnn.models import MLP, Transformer
from fastnn.serialization import (
    save_model,
    load_model,
    save_state_dict,
    load_state_dict,
    save_optimizer,
    load_optimizer,
)
from fastnn.onnx_import import import_onnx
from fastnn.optimizers import (
    SGD,
    Adam,
    AdamW,
    Muon,
    Lion,
)
from fastnn.schedulers import (
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
)

__all__ = [
    # Core utilities
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "checkpoint",
    "tensor",
    "zeros",
    "ones",
    "full",
    "eye",
    "arange",
    "linspace",
    "randint",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand",
    "randn",
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "pow",
    "clamp",
    "relu",
    "fused_add_relu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    "softmax",
    "log_softmax",
    "embedding",
    "argmax",
    "argmin",
    "sum",
    "mean",
    "max",
    "min",
    "mse_loss",
    "cross_entropy_loss",
    # Layers (Rust)
    "Linear",
    "Conv2d",
    "LayerNorm",
    "BatchNorm1d",
    "Dropout",
    "Embedding",
    "ReLU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "SiLU",
    "Sequential",
    "ModuleList",
    "MaxPool2d",
    "ConvTranspose2d",
    "Conv1d",
    "Conv3d",
    "einsum",
    "flash_attention",
    "ResidualBlock",
    "clip_grad_norm_",
    "clip_grad_value_",
    "Dropout2d",
    "Upsample",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm2d",
    "LeakyReLU",
    "Softplus",
    "Hardswish",
    "cat",
    "RMSprop",
    "Elu",
    "Mish",
    "AdaptiveAvgPool2d",
    # Layers (Python)
    "MaxPool2dPy",
    "Flatten",
    "PySequential",
    "BasicBlock",
    # Models
    "MLP",
    "Transformer",
    # Serialization
    "save_model",
    "load_model",
    "save_state_dict",
    "load_state_dict",
    "save_optimizer",
    "load_optimizer",
    "import_onnx",
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    "Muon",
    "Lion",
    # Schedulers
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "ExponentialLR",
    "ReduceLROnPlateau",
    # Data and callbacks
    "DataLoader",
    "Dataset",
    "TensorDataset",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "CSVLogger",
    "DataParallel",
    # Fused layers
    "FusedConvBnSilu",
    # Internal
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
    # Utilities
    "allocator_stats",
    "list_registered_ops",
    "batched_mlp_forward",
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
embedding = _core.embedding
argmax = _core.argmax
argmin = _core.argmin
sum = _core.sum
mean = _core.mean

max = _core.max
min = _core.min
mse_loss = _core.mse_loss
cross_entropy_loss = _core.cross_entropy_loss


# Import fused layers if available
try:
    FusedConvBnSilu = _core.PyFusedConvBnSilu
except AttributeError:
    FusedConvBnSilu = None


def import_onnx(onnx_path: str, fnn_path: str):
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file

    Returns:
        Dictionary with model info (layers, input_shape, output_shape)
    """
    from fastnn.onnx_import import import_onnx as _import

    return _import(onnx_path, fnn_path)




allocator_stats = _core.allocator_stats
list_registered_ops = _core.list_registered_ops
batched_mlp_forward = _core.batched_mlp_forward
