import numpy as np
import fastnn._core as _core

from fastnn.core import no_grad, set_seed, set_num_threads, set_default_device
from fastnn.data import DataLoader, Dataset, TensorDataset
from fastnn.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    CSVLogger,
)

__all__ = [
    "no_grad",
    "set_seed",
    "set_num_threads",
    "set_default_device",
    "DataLoader",
    "Dataset",
    "TensorDataset",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "CSVLogger",
    "models",
]


def __getattr__(name):
    if name == "models":
        import fastnn.models

        return fastnn.models
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        return np.array(data).reshape(shape)

    tensor_cls.numpy = _new_numpy

    # NOTE: Removed __getitem__ patch as it returns numpy arrays and breaks autograd.
    # Users should use .numpy()[idx] explicitly if they want numpy indexing.
    # def _new_getitem(self, idx):
    #     return self.numpy()[idx]
    # tensor_cls.__getitem__ = _new_getitem


_patch_numpy(_core.PyTensor)

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

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()


Sequential = PySequential
ModuleList = _core.ModuleList
SGD = _core.PySGD
Adam = _core.PyAdam
AdamW = _core.PyAdamW
save_model = _core.save_model
load_model = _core.load_model
allocator_stats = _core.allocator_stats
list_registered_ops = _core.list_registered_ops
batched_mlp_forward = _core.batched_mlp_forward
