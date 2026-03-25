import numpy as np
import fastnn._core as _core

__version__ = "0.8.0"

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


def _patch_numpy(tensor_cls):
    _original_numpy = tensor_cls.numpy

    def _new_numpy(self):
        data = _original_numpy(self)
        shape = self.shape
        return np.array(data, dtype=np.float32).reshape(shape)

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


class AdaptiveAvgPool2d:
    """Adaptive average pooling 2D (output size must be (1,1))."""

    def __init__(self, output_size):
        if isinstance(output_size, (list, tuple)):
            output_size = tuple(output_size)
        if output_size != (1, 1):
            raise NotImplementedError("Only output size (1,1) is supported")
        self.output_size = output_size
        self._keepdim = True

    def __call__(self, x):
        # x shape: (batch, channels, height, width)
        # First reduce over width (dim=3)
        y = mean(x, dim=3, keepdim=self._keepdim)
        # Then reduce over height (dim=2)
        z = mean(y, dim=2, keepdim=self._keepdim)
        return z

    def train(self):
        pass

    def eval(self):
        pass


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
        # Flatten all dimensions except the first (batch dimension)
        # For input [batch, channels, 1, 1], output should be [batch, channels]
        # Compute the flattened size: batch_size * channels * 1 * 1
        batch_size = x.shape[0]
        flattened_size = 1
        for dim in x.shape[1:]:
            flattened_size *= dim

        # Use view to reshape
        return x.view([batch_size, flattened_size])

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
save_model = _core.save_model
load_model = _core.load_model
allocator_stats = _core.allocator_stats
list_registered_ops = _core.list_registered_ops
batched_mlp_forward = _core.batched_mlp_forward
