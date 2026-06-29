"""fastnn — High-Performance Neural Network Library in Rust with Python bindings.

AOT-compiled inference and training pipeline with 90+ IR opcodes, operator fusion,
per-channel weight quantization (U4/U8), and arena-based memory planning.
See https://github.com/PetrouilFan/fastnn for full documentation.
"""

import sys
import warnings

import numpy as np
import fastnn._core as _core
import fastnn.precision as precision

__version__ = "2.4.0"

# ---------------------------------------------------------------------------
# Exception hierarchy (from Rust side)
# ---------------------------------------------------------------------------
FastnnError = _core.FastnnError
ShapeError = _core.ShapeError
DtypeError = _core.DtypeError
DeviceError = _core.DeviceError
AutogradError = _core.AutogradError
OptimizerError = _core.OptimizerError
IoError = _core.IoError
CudaError = _core.CudaError

# ---------------------------------------------------------------------------
# Submodule imports — each submodule is the single source of truth for its
# domain.  `__init__.py` only re-exports; it never defines `X = _core.X`.
# ---------------------------------------------------------------------------
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
from fastnn.ops import (  # noqa: E402  — functional operations
    relu, gelu, sigmoid, tanh, silu, leaky_relu, elu, softplus, hardswish,
    softmax, log_softmax,
    fused_add_relu, fused_conv_bn_silu, fused_linear_relu, fused_linear_gelu,
    add, sub, mul, div, matmul, im2col,
    neg, abs, exp, log, sqrt, pow, clamp,
    argmax, argmin, cat, stack, sum, mean, max, min, maximum, minimum,
    where, where_, gather, repeat, expand, fnn_slice, topk,
    einsum, flash_attention, cumsum, erf,
    mse_loss, cross_entropy_loss, bce_with_logits, huber_loss,
)
from fastnn.losses import mse_loss, cross_entropy_loss, bce_with_logits, huber_loss  # noqa: F403
from fastnn.nn import (  # noqa: E402  — neural network modules
    Linear, Conv2d, Conv1d, Conv3d, ConvTranspose2d,
    LayerNorm, RMSNorm, GroupNorm, BatchNorm1d, BatchNorm2d,
    Dropout, Dropout2d, Embedding, Upsample,
    MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool2d,
    ReLU, GELU, Sigmoid, Tanh, SiLU, LeakyReLU, Softplus, Hardswish,
    Elu, Softmax, PReLU, Mish,
    Sequential, ModuleList, ResidualBlock,
    FusedConvBn, FusedConvBnRelu, FusedConvBnGelu,
    Flatten, PySequential, BasicBlock,
)
from fastnn.tensor import (  # noqa: E402
    Tensor, tensor, from_numpy as tensor_from_numpy,
    zeros, ones, full, eye, arange, linspace,
    rand, randn, randint, zeros_like, ones_like, full_like, _flatten,
)
from fastnn.parallel import DataParallel  # noqa: E402
from fastnn.optimizers import (  # noqa: E402
    SGD, Adam, AdamW, Muon, Lion, RMSprop,
    clip_grad_norm_, clip_grad_value_,
)
from fastnn.schedulers import (  # noqa: E402
    LRScheduler, StepLR, CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau,
)
from fastnn.io import (  # noqa: E402
    save as io_save,
    load as io_load,
    convert_from_pytorch,
    convert_from_onnx,
    DAGModel,
)
from fastnn.io.graph_builder import build_model_from_fnn  # noqa: E402
from fastnn.precision import (  # noqa: E402
    Precision, Quantizer, PrecisionConfig, QuantizationScheme,
)

# ---------------------------------------------------------------------------
# Rust-only utilities (not in any Python submodule)
# ---------------------------------------------------------------------------
compile_train_model = _core.compile_train_model
allocator_stats = _core.allocator_stats
list_registered_ops = getattr(_core, 'list_registered_ops', lambda: [])
batched_mlp_forward = _core.batched_mlp_forward
AotExecutor = _core.AotExecutor if hasattr(_core, 'AotExecutor') else None


def import_onnx(onnx_path: str, fnn_path: str, config=None):
    """Import an ONNX model and save it in fastnn format."""
    from fastnn.io.onnx import import_onnx as _import
    return _import(onnx_path, fnn_path, config=config)


def load_state_dict(model, state_dict):
    params = model.parameters()
    loaded = list(state_dict.values())
    if len(params) != len(loaded):
        raise ValueError(
            f"state_dict has {len(loaded)} params, model has {len(params)}"
        )
    for p, loaded_t in zip(params, loaded):
        p.copy_(loaded_t)


# ---------------------------------------------------------------------------
# Lazy-loaded submodules (via __getattr__)
# ---------------------------------------------------------------------------
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
    if name == "init":
        import fastnn.init
        return fastnn.init
    if name == "functional":
        import fastnn.functional
        return fastnn.functional
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Ensure 'models' is resolvable for 'from fastnn import *' across Python versions
try:
    import fastnn.models as models
except ImportError:
    pass


# ---------------------------------------------------------------------------
# NumPy patching for PyTensor.numpy()
# ---------------------------------------------------------------------------
_NUMPY_DTYPE_MAP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "bool": np.bool_,
    "f16": np.float16,
    "bf16": np.float32,  # lossy: bf16 stored as f32
}


def _patch_numpy(tensor_cls):
    _original_numpy = tensor_cls.numpy

    def _new_numpy(self):
        data = _original_numpy(self)
        shape = self.shape
        np_dtype = _NUMPY_DTYPE_MAP.get(self.dtype, np.float32)
        if self.dtype == "bf16":
            warnings.warn("bf16 dtype is lossily mapped to float32 during numpy conversion")
        return np.array(data, dtype=np_dtype).reshape(shape)

    tensor_cls.numpy = _new_numpy


_patch_numpy(_core.PyTensor)


# ---------------------------------------------------------------------------
# Callable tensor module wrapper (fastnn.tensor(...) works)
# ---------------------------------------------------------------------------
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


_tensor_module = sys.modules.get("fastnn.tensor")
if _tensor_module is not None:
    _tensor_wrapper = _TensorModuleWrapper(_tensor_module)
    sys.modules["fastnn.tensor"] = _tensor_wrapper
    import fastnn
    fastnn.__dict__["tensor"] = _tensor_wrapper


# ---------------------------------------------------------------------------
# __all__ — deduplicated, organized by domain
# ---------------------------------------------------------------------------
__all__ = [
    # Context managers and utilities
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "checkpoint",
    "compile_train_model",
    "import_onnx",
    "allocator_stats",
    "list_registered_ops",
    "batched_mlp_forward",
    # Tensor and factories
    "Tensor",
    "tensor",
    "tensor_from_numpy",
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
    # Layers and modules
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
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
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
    "Mish",
    "PReLU",
    "Softmax",
    "Sequential",
    "ModuleList",
    "FusedConvBn",
    "FusedConvBnRelu",
    "FusedConvBnGelu",
    "ResidualBlock",
    "Flatten",
    "PySequential",
    "BasicBlock",
    # Functional operations
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    "leaky_relu",
    "elu",
    "softplus",
    "hardswish",
    "softmax",
    "log_softmax",
    "fused_add_relu",
    "fused_conv_bn_silu",
    "fused_linear_relu",
    "fused_linear_gelu",
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
    "where",
    "gather",
    "repeat",
    "expand",
    "fnn_slice",
    "topk",
    "einsum",
    "flash_attention",
    "cumsum",
    "erf",
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
    # AOT executor
    "AotExecutor",
    # Precision system
    "precision",
    "Precision",
    "Quantizer",
    "PrecisionConfig",
    "QuantizationScheme",
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
