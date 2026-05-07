import numpy as np
import fastnn._core as _core
import struct

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
from fastnn.io import (  # noqa: E402, F403
    save as io_save,
    load as io_load,
    convert_from_pytorch,
    convert_from_onnx,
    MODEL_MAGIC,
    OPTIMIZER_MAGIC,
    MODEL_VERSION,
    OPTIMIZER_VERSION,
    write_tensor,
    read_tensor,
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


def save_model(model, path):
    params = model.parameters() if hasattr(model, "parameters") else []
    named = model.named_parameters() if hasattr(model, "named_parameters") else []
    if named:
        param_list = named
    else:
        param_list = [(f"param_{i}", p) for i, p in enumerate(params)]
    with open(path, "wb") as f:
        f.write(MODEL_MAGIC)
        f.write(struct.pack("<I", MODEL_VERSION))
        f.write(struct.pack("<Q", len(param_list)))
        for name, tensor in param_list:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            shape = tensor.shape
            f.write(struct.pack("<Q", len(shape)))
            for d in shape:
                f.write(struct.pack("<q", d))
            data = tensor.numpy().astype(np.float32).flatten()
            f.write(struct.pack("<Q", len(data)))
            f.write(data.tobytes())


def load_model(path):
    result = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MODEL_MAGIC:
            raise ValueError("Invalid file format: expected FNN magic bytes")
        version = struct.unpack("<I", f.read(4))[0]
        if version > MODEL_VERSION:
            raise ValueError(f"Unsupported format version: {version}")
        num_params = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_params):
            name_len = struct.unpack("<Q", f.read(8))[0]
            name = f.read(name_len).decode("utf-8")
            shape_len = struct.unpack("<Q", f.read(8))[0]
            shape = [struct.unpack("<q", f.read(8))[0] for _ in range(shape_len)]
            data_len = struct.unpack("<Q", f.read(8))[0]
            data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
            result[name] = tensor(data.copy(), list(shape))
    return result


def load_state_dict(model, state_dict):
    params = model.parameters()
    loaded = list(state_dict.values())
    if len(params) != len(loaded):
        raise ValueError(
            f"state_dict has {len(loaded)} params, model has {len(params)}"
        )
    for p, loaded_t in zip(params, loaded):
        p.copy_(loaded_t)


def save_state_dict(model, path):
    named = model.named_parameters() if hasattr(model, "named_parameters") else []
    params = model.parameters() if hasattr(model, "parameters") else []
    if named:
        param_list = named
    else:
        param_list = [(f"param_{i}", p) for i, p in enumerate(params)]
    with open(path, "wb") as f:
        f.write(MODEL_MAGIC)
        f.write(struct.pack("<I", MODEL_VERSION))
        f.write(struct.pack("<Q", len(param_list)))
        for name, tensor in param_list:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_bytes)))
            f.write(name_bytes)
            shape = tensor.shape
            f.write(struct.pack("<Q", len(shape)))
            for d in shape:
                f.write(struct.pack("<q", d))
            grad = tensor.grad
            if grad is not None:
                f.write(struct.pack("<B", 1))
                data = grad.numpy().astype(np.float32).flatten()
                f.write(struct.pack("<Q", len(data)))
                f.write(data.tobytes())
            else:
                f.write(struct.pack("<B", 0))
                f.write(struct.pack("<Q", 0))


def save_optimizer(opt, path):
    with open(path, "wb") as f:
        f.write(OPTIMIZER_MAGIC)
        f.write(struct.pack("<I", OPTIMIZER_VERSION))
        lr = getattr(opt, "lr", 0.0)
        f.write(struct.pack("<d", lr))
        betas = getattr(opt, "betas", (0.9, 0.999))
        f.write(struct.pack("<d", betas[0]))
        f.write(struct.pack("<d", betas[1]))
        eps = getattr(opt, "eps", 1e-8)
        f.write(struct.pack("<d", eps))
        wd = getattr(opt, "weight_decay", 0.0)
        f.write(struct.pack("<d", wd))
        params = getattr(opt, "params", [])
        n = len(params)
        f.write(struct.pack("<Q", n))
        for i in range(n):
            for state_tensor_name in ["m", "v", "v_hat"]:
                state_list = getattr(opt, state_tensor_name, None)
                if state_list and i < len(state_list):
                    data = state_list[i].numpy().astype(np.float32).flatten()
                    f.write(struct.pack("<B", 1))
                    f.write(struct.pack("<Q", len(data)))
                    f.write(data.tobytes())
                else:
                    f.write(struct.pack("<B", 0))
                    f.write(struct.pack("<Q", 0))
        step_list = getattr(opt, "step", None)
        if step_list:
            for s in step_list:
                f.write(struct.pack("<Q", s))


def load_optimizer(opt, path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != OPTIMIZER_MAGIC:
            raise ValueError("Invalid optimizer state file")
        struct.unpack("<I", f.read(4))[0]
        opt.lr = struct.unpack("<d", f.read(8))[0]
        b1 = struct.unpack("<d", f.read(8))[0]
        b2 = struct.unpack("<d", f.read(8))[0]
        if hasattr(opt, "betas"):
            opt.betas = (b1, b2)
        opt.eps = struct.unpack("<d", f.read(8))[0]
        if hasattr(opt, "weight_decay"):
            opt.weight_decay = struct.unpack("<d", f.read(8))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for i in range(n):
            for state_tensor_name in ["m", "v", "v_hat"]:
                state_list = getattr(opt, state_tensor_name, None)
                has_data = struct.unpack("<B", f.read(1))[0]
                data_len = struct.unpack("<Q", f.read(8))[0]
                if has_data and state_list and i < len(state_list):
                    data = np.frombuffer(f.read(data_len * 4), dtype=np.float32)
                    state_list[i].copy_(tensor(data.copy(), list(state_list[i].shape)))
                elif has_data:
                    f.read(data_len * 4)
        step_list = getattr(opt, "step", None)
        if step_list:
            for i in range(len(step_list)):
                step_list[i] = struct.unpack("<Q", f.read(8))[0]


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
