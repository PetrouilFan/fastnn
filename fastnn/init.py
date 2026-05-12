"""Weight initialization functions, compatible with PyTorch's torch.nn.init."""

import math
import numpy as np
import fastnn
from fastnn import Tensor


__all__ = [
    'uniform_',
    'normal_',
    'constant_',
    'ones_',
    'zeros_',
    'xavier_uniform_',
    'xavier_normal_',
    'kaiming_uniform_',
    'kaiming_normal_',
    'calculate_gain',
    'orthogonal_',
    'eye_',
    'dirac_',
]


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor with values drawn from Uniform(a, b)."""
    with fastnn.no_grad():
        rng = np.random.default_rng()
        data = rng.uniform(a, b, tensor.shape).astype(np.float32).flatten().tolist()
        return fastnn.Tensor(data, tensor.shape)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with values drawn from Normal(mean, std)."""
    with fastnn.no_grad():
        rng = np.random.default_rng()
        data = rng.normal(mean, std, tensor.shape).astype(np.float32).flatten().tolist()
        return fastnn.Tensor(data, tensor.shape)


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor with constant value."""
    with fastnn.no_grad():
        data = np.full(tensor.shape, val, dtype=np.float32).flatten().tolist()
        return fastnn.Tensor(data, tensor.shape)


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor with ones."""
    return constant_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros."""
    return constant_(tensor, 0.0)


def calculate_gain(nonlinearity: str, param=None) -> float:
    """Return recommended gain value for the given nonlinearity.

    Follows PyTorch's calculate_gain.
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
                  'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns:
        return 1.0
    elif nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0 / 3.0
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 0.01
        return math.sqrt(2.0 / (1 + param ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4.0
    else:
        return 1.0


def _calculate_fan(tensor: Tensor) -> tuple:
    shape = tensor.shape
    ndim = len(shape)
    if ndim < 2:
        fan_in = 1
        fan_out = 1
    elif ndim == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        receptive_field_size = 1
        for s in shape[2:]:
            receptive_field_size *= s
        fan_in = shape[1] * receptive_field_size
        fan_out = shape[0] * receptive_field_size
    return fan_in, fan_out


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier uniform init."""
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier normal init."""
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0.0, mode: str = 'fan_in',
                      nonlinearity: str = 'leaky_relu') -> Tensor:
    """Fill tensor with Kaiming He uniform init."""
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0.0, mode: str = 'fan_in',
                     nonlinearity: str = 'leaky_relu') -> Tensor:
    """Fill tensor with Kaiming He normal init."""
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with orthogonal matrix."""
    with fastnn.no_grad():
        shape = tensor.shape
        flat_shape = (shape[-2], shape[-1])
        if len(shape) > 2:
            raise NotImplementedError("orthogonal_ for >2D tensors not yet implemented")
        rng = np.random.default_rng()
        q, r = np.linalg.qr(rng.normal(0.0, 1.0, flat_shape))
        q *= np.sign(np.diag(r))
        q *= gain
        return fastnn.Tensor(q.flatten().tolist(), shape)


def eye_(tensor: Tensor) -> Tensor:
    """Fill 2D tensor with identity matrix."""
    with fastnn.no_grad():
        n = min(tensor.shape)
        eye_data = [0.0] * (tensor.shape[0] * tensor.shape[1])
        for i in range(n):
            eye_data[i * tensor.shape[1] + i] = 1.0
        return fastnn.Tensor(eye_data, tensor.shape)


def dirac_(tensor: Tensor) -> Tensor:
    """Fill tensor with Dirac delta function (for Conv layers)."""
    with fastnn.no_grad():
        shape = tensor.shape
        if len(shape) < 2:
            raise ValueError("dirac_ requires at least 2D tensor")
        out_channels = shape[0]
        in_channels = shape[1]
        data = np.zeros(shape, dtype=np.float32)
        min_dim = min(out_channels, in_channels)
        for i in range(min_dim):
            if len(shape) == 2:
                data[i, i] = 1.0
            elif len(shape) == 3:
                data[i, i, shape[2] // 2] = 1.0
            elif len(shape) == 4:
                data[i, i, shape[2] // 2, shape[3] // 2] = 1.0
            elif len(shape) == 5:
                data[i, i, shape[2] // 2, shape[3] // 2, shape[4] // 2] = 1.0
        return fastnn.Tensor(data.flatten().tolist(), shape)
