import numpy as np
import fastnn._core as _core

__version__ = "1.2.0"

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
from fastnn.losses import *  # noqa: F403
from fastnn.parallel import DataParallel  # noqa: E402
from fastnn.optimizers import (  # noqa: E402
    SGD,
    Adam,
    AdamW,
    Muon,
    Lion,
    RMSprop,
    clip_grad_norm_,
    clip_grad_value_,
)
from fastnn.tensor import (  # noqa: E402
    Tensor,
    zeros,
    ones,
    full,
    eye,
    arange,
    linspace,
    randint,
    zeros_like,
    ones_like,
    full_like,
    rand,
    randn,
    tensor,
    from_numpy as tensor_from_numpy,
    _flatten,
)
from fastnn.layers import Flatten, PySequential, BasicBlock  # noqa: F401, E402
MaxPool2d = _core.MaxPool2d
from fastnn.io import (  # noqa: E402
    save as io_save,
    load as io_load,
    convert_from_pytorch,
    convert_from_onnx,
    DAGModel,
)
from fastnn.io.graph_builder import build_model_from_fnn  # noqa: E402

__all__ = [
    # Context managers and utilities
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "checkpoint",
    "load_state_dict",
    "import_onnx",
    "allocator_stats",
    "list_registered_ops",
    "batched_mlp_forward",
    # Tensor and factories
    "Tensor",
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
    "tensor",
    "tensor_from_numpy",
    "_flatten",
    # Data loading
    "DataLoader",
    "Dataset",
    "TensorDataset",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "CSVLogger",
    # Parallel
    "DataParallel",
    # Models
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
    # Layers and modules (Rust implementations)
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
    "Mish",
    "Sequential",
    "ModuleList",
    "FusedConvBn",
    "FusedConvBnRelu",
    "FusedConvBnGelu",
    "ResidualBlock",
    # Python layers
    "Flatten",
    "PySequential",
    "BasicBlock",
    # Activation functions (functional)
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    "softmax",
    "log_softmax",
    "fused_add_relu",
    "fused_conv_bn_silu",
    # Packed tensor constructors (numpy interop)
    "packed_tensor",
    "packed_tensor_from_f32",
    # Tensor operations
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "im2col",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "pow",
    "clamp",
    "argmax",
    "argmin",
    "cat",
    "stack",
    "sum",
    "mean",
    "max",
    "min",
    "maximum",
    "minimum",
    "einsum",
    "flash_attention",
    # Loss functions
    "mse_loss",
    "cross_entropy_loss",
    "bce_with_logits",
    "huber_loss",
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    "Muon",
    "Lion",
    "RMSprop",
    "clip_grad_norm_",
    "clip_grad_value_",
    # Schedulers
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "ExponentialLR",
    "ReduceLROnPlateau",
    # Activations module
    "activations",
    # IO functions
    "io_save",
    "io_load",
    "convert_from_pytorch",
    "convert_from_onnx",
    "DAGModel",
    "load_onnx_model",
    # Phase 0 additions
    "PReLU",
    "AvgPool2d",
    # Phase 1 additions
    "DAGExecutor",
    # Phase 3 additions
    "where",
    "gather",
    "repeat",
    "expand",
    "fnn_slice",
    "topk",
    "leaky_relu",
    "elu",
    "softplus",
    "hardswish",
    # YOLO model wrapper
    "YOLO",
    "load_yolo",
    "build_model_from_fnn",
    # NMS utilities
    "nms",
    "yolo_decode",
    "yolo_dfl_decode",
    "xywh2xyxy",
    "scale_boxes",
]


def __getattr__(name):
    if name == "models":
        import fastnn.models
        return fastnn.models
    if name == "activations":
        import fastnn.activations
        return fastnn.activations
    if name == "YOLO":
        from fastnn.models.yolo import YOLO
        return YOLO
    if name == "load_yolo":
        from fastnn.models.yolo import load_yolo
        return load_yolo
    if name in ("nms", "yolo_decode", "yolo_dfl_decode", "xywh2xyxy", "scale_boxes"):
        from fastnn.utils.nms import nms, yolo_decode, yolo_dfl_decode, xywh2xyxy, scale_boxes
        return {
            "nms": nms,
            "yolo_decode": yolo_decode,
            "yolo_dfl_decode": yolo_dfl_decode,
            "xywh2xyxy": xywh2xyxy,
            "scale_boxes": scale_boxes,
        }[name]
    if name == "load_onnx_model":
        from fastnn.io import load_onnx_model
        return load_onnx_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Ensure 'models' is resolvable for 'from fastnn import *' across Python versions
try:
    import fastnn.models as models
except ImportError:
    pass





_NUMPY_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
    "f16": np.float16,
    "bf16": np.float32,
}


def _patch_numpy(tensor_cls):
    _original_numpy = tensor_cls.numpy

    def _new_numpy(self):
        data = _original_numpy(self)
        shape = self.shape
        np_dtype = _NUMPY_DTYPE_MAP.get(self.dtype, np.float32)
        return np.array(data, dtype=np_dtype).reshape(shape)

    tensor_cls.numpy = _new_numpy


def _patch_backward(tensor_cls):
    _original_backward = tensor_cls.backward

    def _new_backward(self, grad=None):
        return _original_backward(self, grad)

    tensor_cls.backward = _new_backward


_patch_numpy(_core.PyTensor)
_patch_backward(_core.PyTensor)


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

LeakyReLU = _core.LeakyReLU
Softmax = _core.Softmax
Softplus = _core.Softplus
Hardswish = _core.Hardswish
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
BatchNorm2d = _core.BatchNorm2d
Lion = _core.PyLion
bce_with_logits = _core.bce_with_logits
ConvTranspose2d = _core.ConvTranspose2d
Conv1d = _core.Conv1d
Conv3d = _core.Conv3d
einsum = _core.einsum
flash_attention = _core.flash_attention
ResidualBlock = _core.ResidualBlock
Dropout2d = _core.Dropout2d
Upsample = _core.Upsample
RMSprop = _core.PyRMSprop
Elu = _core.Elu
Mish = _core.Mish
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d

# Packed linear
Linear4 = _core.PyLinear4
Linear8 = _core.PyLinear8
Linear16 = _core.PyLinear16
Linear32 = _core.PyLinear32

# Packed tensors
PackedTensor4 = _core.PyPackedTensor4
PackedTensor8 = _core.PyPackedTensor8
PackedTensor16 = _core.PyPackedTensor16
PackedTensor32 = _core.PyPackedTensor32

def packed_tensor(data, shape=None, scale=1.0, zero=0.0, dtype="u4"):
    """Create a packed tensor from data (list or numpy array).

    Args:
        data: numpy array or list
        shape: optional shape (auto-detected from numpy array)
        scale: quantization scale
        zero: quantization zero point
        dtype: one of "u4", "u8", "f16", "f32"
    """
    import numpy as np
    if isinstance(data, np.ndarray):
        if shape is None:
            shape = list(data.shape)
        data = data.flatten().tolist()
    if dtype == "u4":
        return PackedTensor4(data, shape, scale, zero)
    elif dtype == "u8":
        return PackedTensor8(data, shape, scale, zero)
    elif dtype == "f16":
        return PackedTensor16(data, shape, scale, zero)
    elif dtype == "f32":
        return PackedTensor32(data, shape, scale, zero)
    else:
        raise ValueError(f"Unknown packed dtype: {dtype}")


def packed_tensor_from_f32(data, shape=None, dtype="u4"):
    """Create a packed tensor with auto-computed scale from f32 data.

    Args:
        data: numpy array or list of f32 values
        shape: optional shape (auto-detected from numpy array)
        dtype: one of "u4", "u8", "f16", "f32"
    """
    import numpy as np
    if isinstance(data, np.ndarray):
        if shape is None:
            shape = list(data.shape)
        data = data.flatten().tolist()
    if dtype == "u4":
        return PackedTensor4.from_f32_auto(data, shape)
    elif dtype == "u8":
        return PackedTensor8.from_f32_auto(data, shape)
    elif dtype == "f16":
        return PackedTensor16.from_f32_auto(data, shape)
    elif dtype == "f32":
        return PackedTensor32.from_f32_auto(data, shape)
    else:
        raise ValueError(f"Unknown packed dtype: {dtype}")


# Quantized tensors
QuantizedTensor = _core.PyQuantizedTensor

# Master weight optimizers
MasterWeightOptimizer4 = _core.PyMasterWeightOptimizer4
MasterWeightOptimizer8 = _core.PyMasterWeightOptimizer8
MasterWeightOptimizer16 = _core.PyMasterWeightOptimizer16
MasterWeightOptimizer32 = _core.PyMasterWeightOptimizer32

# Backend control
use_wgpu = _core.use_wgpu
use_cpu = _core.use_cpu
is_wgpu = _core.is_wgpu


def import_onnx(onnx_path: str, fnn_path: str):
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file

    Returns:
        Dictionary with model info (layers, input_shape, output_shape)
    """
    from fastnn.io.onnx import import_onnx as _import

    return _import(onnx_path, fnn_path)


def load_state_dict(model, state_dict):
    params = model.parameters()
    loaded = list(state_dict.values())
    if len(params) != len(loaded):
        raise ValueError(
            f"state_dict has {len(loaded)} params, model has {len(params)}"
        )
    for p, loaded_t in zip(params, loaded):
        p.copy_(loaded_t)
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


# Phase 0: New module classes
PReLU = _core.PReLU
AvgPool2d = _core.AvgPool2d

# Phase 1: DAG executor
DAGExecutor = _core.DAGExecutor

# Phase 3: New tensor operation functions
where = _core.where_
gather = _core.gather
repeat = _core.repeat
expand = _core.expand
fnn_slice = _core.slice
topk = _core.topk
leaky_relu = _core.leaky_relu
elu = _core.elu
softplus = _core.softplus
hardswish = _core.hardswish

import sys






_tensor_module = sys.modules.get("fastnn.tensor")
if _tensor_module is not None:
    _tensor_wrapper = _TensorModuleWrapper(_tensor_module)
    sys.modules["fastnn.tensor"] = _tensor_wrapper
    import fastnn

    fastnn.__dict__["tensor"] = _tensor_wrapper
