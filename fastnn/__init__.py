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

__all__ = [
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "checkpoint",
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


def tensor_from_array(arr):
    """Create tensor from numpy array efficiently (no intermediate list).

    Args:
        arr: numpy array (will be flattened and converted to f32)

    Returns:
        fastnn tensor
    """
    import numpy as np

    # Ensure float32 and C-contiguous
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    # Flatten and create tensor in one pass - no intermediate Python list
    flat = arr.flatten()
    return _core.tensor_from_data(
        flat.tolist(),  # Still creates list but more efficient than .tolist() on original
        list(arr.shape),
        device=None,
    )


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


class MaxPool2d:
    """Max pooling 2D."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if stride is not None else kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def __call__(self, x):
        # x shape: (batch, channels, height, width)
        # Use Rust implementation for performance
        import fastnn._core as _core

        # For now, only support symmetric kernel_size, stride, padding, dilation
        # Check if they are tuples and convert to single value if possible
        if isinstance(self.kernel_size, tuple):
            if self.kernel_size[0] != self.kernel_size[1]:
                raise NotImplementedError("Non-square kernel_size not supported yet")
            kernel_size = self.kernel_size[0]
        else:
            kernel_size = self.kernel_size

        if isinstance(self.stride, tuple):
            if self.stride[0] != self.stride[1]:
                raise NotImplementedError("Non-square stride not supported yet")
            stride = self.stride[0]
        else:
            stride = self.stride

        if isinstance(self.padding, tuple):
            if self.padding[0] != self.padding[1]:
                raise NotImplementedError("Non-square padding not supported yet")
            padding = self.padding[0]
        else:
            padding = self.padding

        if isinstance(self.dilation, tuple):
            if self.dilation[0] != self.dilation[1]:
                raise NotImplementedError("Non-square dilation not supported yet")
            dilation = self.dilation[0]
        else:
            dilation = self.dilation

        # Call the Rust implementation
        rust_maxpool = _core.MaxPool2d(kernel_size, stride, padding, dilation)
        return rust_maxpool(x)

    def train(self):
        pass

    def eval(self):
        pass


class Flatten:
    """Flatten layer."""

    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    def __call__(self, x):
        shape = x.shape
        ndim = len(shape)
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        end = end + 1
        new_shape = list(shape[:start])
        flattened_size = 1
        for dim in shape[start:end]:
            flattened_size *= dim
        new_shape.append(flattened_size)
        new_shape.extend(shape[end:])
        return x.view(new_shape)

    def train(self):
        pass

    def eval(self):
        pass


class PySequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def to_gpu(self, device_id):
        for layer in self.layers:
            if hasattr(layer, "to_gpu"):
                layer.to_gpu(device_id)

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()


class BasicBlock:
    """ResNet BasicBlock with skip connection."""

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample=None):
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

    def parameters(self):
        params = []
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2]:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        if self.downsample is not None:
            if hasattr(self.downsample, "parameters"):
                params.extend(self.downsample.parameters())
        return params

    def named_parameters(self):
        params = []
        for name, layer in [
            ("conv1", self.conv1),
            ("bn1", self.bn1),
            ("conv2", self.conv2),
            ("bn2", self.bn2),
        ]:
            if hasattr(layer, "named_parameters"):
                for n, p in layer.named_parameters():
                    params.append((f"{name}.{n}", p))
        if self.downsample is not None:
            if hasattr(self.downsample, "named_parameters"):
                for n, p in self.downsample.named_parameters():
                    params.append((f"downsample.{n}", p))
        return params

    def zero_grad(self):
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2]:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
        if self.downsample is not None:
            if hasattr(self.downsample, "zero_grad"):
                self.downsample.zero_grad()

    def train(self):
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2]:
            if hasattr(layer, "train"):
                layer.train()
        if self.downsample is not None:
            if hasattr(self.downsample, "train"):
                self.downsample.train()

    def eval(self):
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2]:
            if hasattr(layer, "eval"):
                layer.eval()
        if self.downsample is not None:
            if hasattr(self.downsample, "eval"):
                self.downsample.eval()


Sequential = PySequential
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
    import struct

    MAGIC = b"FNN\x00"
    VERSION = 1
    params = model.parameters() if hasattr(model, "parameters") else []
    named = model.named_parameters() if hasattr(model, "named_parameters") else []
    if named:
        param_list = named
    else:
        param_list = [(f"param_{i}", p) for i, p in enumerate(params)]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
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
    import struct

    MAGIC = b"FNN\x00"
    result = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("Invalid file format: expected FNN magic bytes")
        version = struct.unpack("<I", f.read(4))[0]
        if version > 1:
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
    import struct

    MAGIC = b"FNN\x00"
    VERSION = 1
    named = model.named_parameters() if hasattr(model, "named_parameters") else []
    params = model.parameters() if hasattr(model, "parameters") else []
    if named:
        param_list = named
    else:
        param_list = [(f"param_{i}", p) for i, p in enumerate(params)]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
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
    import struct

    MAGIC = b"FNO\x00"
    VERSION = 1
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        lr = opt.lr if hasattr(opt, "lr") else 0.0
        f.write(struct.pack("<d", lr))
        betas = opt.betas if hasattr(opt, "betas") else (0.9, 0.999)
        f.write(struct.pack("<d", betas[0]))
        f.write(struct.pack("<d", betas[1]))
        eps = opt.eps if hasattr(opt, "eps") else 1e-8
        f.write(struct.pack("<d", eps))
        wd = opt.weight_decay if hasattr(opt, "weight_decay") else 0.0
        f.write(struct.pack("<d", wd))
        n = len(opt.params) if hasattr(opt, "params") else 0
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
    import struct

    MAGIC = b"FNO\x00"
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
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


class LRScheduler:
    """Base class for learning rate schedulers.

    Modifies the optimizer's learning rate in-place via state_dict.

    Usage:
        >>> scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        >>> for epoch in range(100):
        ...     for batch_x, batch_y in loader:
        ...         ...
        ...     scheduler.step()
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = self._get_lr()
        self.last_epoch = -1

    def _get_lr(self):
        sd = self.optimizer.state_dict()
        return sd.get("lr", 0.01)

    def _set_lr(self, lr):
        sd = self.optimizer.state_dict()
        sd["lr"] = lr
        self.optimizer.load_state_dict(sd)

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self._set_lr(lr)
        return lr


class StepLR(LRScheduler):
    """Decays LR by gamma every step_size epochs.

    Args:
        optimizer: Optimizer to schedule.
        step_size: Number of epochs between LR decay.
        gamma: Multiplicative factor for LR decay.
    """

    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing LR schedule.

    Args:
        optimizer: Optimizer to schedule.
        T_max: Maximum number of epochs (half cycle).
        eta_min: Minimum learning rate.
    """

    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):
        import math

        return (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
        )


class ExponentialLR(LRScheduler):
    """Decays LR by gamma every epoch.

    Args:
        optimizer: Optimizer to schedule.
        gamma: Multiplicative factor for LR decay.
    """

    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self):
        return self.base_lr * self.gamma**self.last_epoch


class ReduceLROnPlateau:
    """Reduce LR when metric has stopped improving.

    Args:
        optimizer: Optimizer to schedule.
        mode: 'min' or 'max'.
        factor: Factor to multiply LR by.
        patience: Number of epochs with no improvement.
        min_lr: Minimum learning rate.
    """

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.current_lr = self._get_lr()

    def _get_lr(self):
        sd = self.optimizer.state_dict()
        return sd.get("lr", 0.01)

    def _set_lr(self, lr):
        sd = self.optimizer.state_dict()
        sd["lr"] = lr
        self.optimizer.load_state_dict(sd)
        self.current_lr = lr

    def step(self, metric):
        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                new_lr = self.current_lr * self.factor
                if new_lr < self.min_lr:
                    new_lr = self.min_lr
                self._set_lr(new_lr)
                return new_lr
        return self.current_lr


allocator_stats = _core.allocator_stats
list_registered_ops = _core.list_registered_ops
batched_mlp_forward = _core.batched_mlp_forward
