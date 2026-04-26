import contextlib
import fastnn._core as _core
from typing import Any, List, Optional, Sequence, Tuple, Union

# Expose classes from _core
PyTransformerEncoder = _core.PyTransformerEncoder
Linear = _core.Linear
Sequential = _core.Sequential_
ModuleList = _core.ModuleList
MaxPool2d = _core.MaxPool2d
Conv2d = _core.Conv2d
LayerNorm = _core.LayerNorm
BatchNorm1d = _core.BatchNorm1d
Dropout = _core.Dropout
Embedding = _core.Embedding
ReLU = _core.ReLU
Gelu = _core.Gelu
GELU = _core.Gelu
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU

# Re-export basic tensor operations from _core
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

# Re-export additional layer classes from _core
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
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
BatchNorm2d = _core.BatchNorm2d
LeakyReLU = _core.LeakyReLU
Softplus = _core.Softplus
Hardswish = _core.Hardswish
cat = _core.cat
RMSprop = _core.PyRMSprop
Elu = _core.Elu
Mish = _core.Mish
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d

# Fused layers (if available)
try:
    FusedConvBnSilu = _core.PyFusedConvBnSilu
except AttributeError:
    FusedConvBnSilu = None

def tensor(data: Union[List, Tuple, Any], shape: Optional[Sequence[int]] = None,
           device: Optional[str] = None) -> Any:
    """Create a tensor from nested list/tuple data.
    
    Args:
        data: Nested list or tuple of numeric values.
        shape: Optional explicit shape. If not provided, inferred from data.
        device: Optional device string (e.g., "cpu", "gpu").
    
    Returns:
        A new tensor.
    
    Examples:
        >>> t = tensor([[1, 2], [3, 4]], shape=[2, 2])
        >>> t.shape
        [2, 2]
    """
    # Flatten data
    def _flatten(nested):
        result = []
        for item in nested:
            if isinstance(item, (list, tuple)):
                result.extend(_flatten(item))
            else:
                result.append(item)
        return result
    
    flat_data = _flatten(data)
    if shape is None:
        # Infer shape from data? Not possible without knowing nesting structure.
        # Require shape argument for safety.
        raise ValueError("shape argument is required for tensor()")
    return _core.tensor_from_data(flat_data, list(shape), device)


# Additional classes that may be needed
# Note: MultiHeadAttention is in Rust but not exposed via PyO3 yet
# We'll use the full TransformerEncoder which includes it


def sum(a, dim=None, keepdim=False):
    if dim is None:
        # Flatten then single sum instead of O(dims) dispatches
        flat = a.reshape([-1])
        return _core.sum(flat, 0, False)
    return _core.sum(a, dim, keepdim)


def mean(a, dim=None, keepdim=False):
    if dim is None:
        # Flatten then single mean
        flat = a.reshape([-1])
        return _core.mean(flat, 0, False)
    return _core.mean(a, dim, keepdim)


@contextlib.contextmanager
def no_grad():
    _core._no_grad_enter()
    try:
        yield
    finally:
        _core._no_grad_exit()


def set_seed(seed: int):
    _core._set_seed(seed)


def set_num_threads(n: int):
    _core._set_num_threads(n)


def set_default_device(device: str):
    """Set the default device for tensor creation.

    Args:
        device: Device string, e.g., "cpu", "gpu", "wgpu", "gpu:0"
    """
    _core._set_default_device(device)


def checkpoint(fn, inputs):
    """Enable gradient checkpointing to save memory during training.

    This is a placeholder implementation that returns the inputs as-is.
    In a full implementation, this would store the computation graph
    for recomputation during the backward pass.

    Args:
        fn: The function to checkpoint (currently not fully implemented)
        inputs: List of input tensors

    Returns:
        List of output tensors (currently just returns inputs)
    """
    # Note: PyO3 doesn't easily support passing Python callables to Rust.
    # The function is passed as a string name for now.
    # A full implementation would store the Python function and call it during backward.
    fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
    return _core.checkpoint(fn_name, inputs)
