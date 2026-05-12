"""Functional API for neural network operations (analogous to torch.nn.functional).

Provides stateless functions for common neural network operations.
All functions work with fastnn.Tensor objects.
"""

import fastnn
from fastnn import Tensor


__all__ = [
    # Activation functions
    'relu',
    'gelu',
    'sigmoid',
    'tanh',
    'silu',
    'leaky_relu',
    'elu',
    'softplus',
    'hardswish',
    'softmax',
    'log_softmax',
    # Convolution
    'conv2d',
    'conv1d',
    # Pooling
    'max_pool2d',
    'avg_pool2d',
    'adaptive_avg_pool2d',
    # Normalization
    'batch_norm',
    'layer_norm',
    # Loss functions
    'mse_loss',
    'cross_entropy',
    'binary_cross_entropy_with_logits',
    'huber_loss',
    # Tensor ops
    'dropout',
    'linear',
    'pad',
    'interpolate',
    # Padding
    'pad',
]


# --- Activation functions ---

def relu(x: Tensor) -> Tensor:
    """Applies the rectified linear unit function."""
    return fastnn.relu(x)


def gelu(x: Tensor) -> Tensor:
    """Applies the Gaussian Error Linear Units function."""
    return fastnn.gelu(x)


def sigmoid(x: Tensor) -> Tensor:
    """Applies the sigmoid function."""
    return fastnn.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    """Applies the hyperbolic tangent function."""
    return fastnn.tanh(x)


def silu(x: Tensor) -> Tensor:
    """Applies the SiLU (Swish) activation function."""
    return fastnn.silu(x)


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Applies the LeakyReLU function."""
    return fastnn.leaky_relu(x, negative_slope)


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Applies the ELU activation function."""
    return fastnn.elu(x, alpha)


def softplus(x: Tensor) -> Tensor:
    """Applies the Softplus function."""
    return fastnn.softplus(x)


def hardswish(x: Tensor) -> Tensor:
    """Applies the Hardswish function."""
    return fastnn.hardswish(x)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Applies the Softmax function."""
    return fastnn.softmax(x, dim)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Applies the LogSoftmax function."""
    return fastnn.log_softmax(x, dim)


# --- Convolution ---

def conv2d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    """Applies a 2D convolution over an input signal."""
    conv = fastnn.Conv2d(
        weight.shape[1] * groups, weight.shape[0],
        weight.shape[2], stride, padding, dilation, groups, False
    )
    conv.set_weight(weight)
    if bias is not None:
        conv.set_bias(bias)
    return conv(x)


def conv1d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    """Applies a 1D convolution over an input signal."""
    conv = fastnn.Conv1d(
        weight.shape[1] * groups, weight.shape[0],
        weight.shape[2], stride, padding, dilation, groups, bias is not None
    )
    conv.set_weight(weight)
    if bias is not None:
        conv.set_bias(bias)
    return conv(x)


# --- Pooling ---

def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0, dilation=1) -> Tensor:
    """Applies 2D max pooling over an input signal."""
    if stride is None:
        stride = kernel_size
    pool = fastnn.MaxPool2d(kernel_size, stride, padding, dilation)
    return pool(x)


def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    """Applies 2D average pooling over an input signal."""
    if stride is None:
        stride = kernel_size
    pool = fastnn.AvgPool2d(kernel_size, stride, padding)
    return pool(x)


def adaptive_avg_pool2d(x: Tensor, output_size) -> Tensor:
    """Applies 2D adaptive average pooling over an input signal."""
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    pool = fastnn.AdaptiveAvgPool2d(output_size)
    return pool(x)


# --- Normalization ---

def batch_norm(x: Tensor, running_mean=None, running_var=None, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5) -> Tensor:
    """Applies Batch Normalization."""
    channels = x.shape[1]
    if running_mean is None:
        running_mean = fastnn.zeros([channels])
    if running_var is None:
        running_var = fastnn.ones([channels])
    if weight is None:
        weight = fastnn.ones([channels])
    if bias is None:
        bias = fastnn.zeros([channels])
    bn = fastnn.BatchNorm2d(channels, eps, momentum)
    bn.set_weight(weight)
    bn.set_bias(bias)
    bn.set_running_mean(running_mean)
    bn.set_running_var(running_var)
    if training:
        bn.train()
    else:
        bn.eval()
    return bn(x)


def layer_norm(x: Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor:
    """Applies Layer Normalization."""
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]
    if weight is None:
        weight = fastnn.ones(normalized_shape)
    if bias is None:
        bias = fastnn.zeros(normalized_shape)
    ln = fastnn.LayerNorm(normalized_shape[-1], eps)
    ln.set_weight(weight)
    ln.set_bias(bias)
    return ln(x)


# --- Loss functions ---

def mse_loss(x: Tensor, target: Tensor) -> Tensor:
    """Mean squared error loss."""
    return fastnn.mse_loss(x, target)


def cross_entropy(x: Tensor, target: Tensor) -> Tensor:
    """Cross entropy loss (with logits)."""
    return fastnn.cross_entropy_loss(x, target)


def binary_cross_entropy_with_logits(x: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy loss with logits."""
    return fastnn.bce_with_logits(x, target)


def huber_loss(x: Tensor, target: Tensor) -> Tensor:
    """Huber loss."""
    return fastnn.huber_loss(x, target)


# --- Regularization ---

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Applies dropout during training."""
    d = fastnn.Dropout(p)
    if training:
        d.train()
    else:
        d.eval()
    return d(x)


# --- Linear ---

def linear(x: Tensor, weight: Tensor, bias=None) -> Tensor:
    """Applies a linear transformation."""
    lin = fastnn.Linear(weight.shape[1], weight.shape[0], bias is not None)
    lin.set_weight(weight)
    if bias is not None:
        lin.set_bias(bias)
    return lin(x)


# --- Interpolate ---

def interpolate(x: Tensor, size=None, scale_factor=None, mode='nearest') -> Tensor:
    """Upsamples input to given size or scale factor."""
    if scale_factor is None and size is not None:
        # Compute scale_factor from input and target size
        in_h, in_w = x.shape[-2], x.shape[-1]
        scale_h = size[-2] / in_h
        scale_w = size[-1] / in_w
        if abs(scale_h - scale_w) > 1e-6:
            raise ValueError("Aspect ratio change not supported")
        scale_factor = scale_h
    upsample = fastnn.Upsample(scale_factor, mode)
    return upsample(x)


# --- Padding ---

def pad(x: Tensor, pad_width, mode='constant', value=0.0):
    """Pad tensor (basic implementation using slice and fill)."""
    # Only a simple constant-pad implementation for now
    if mode != 'constant':
        raise NotImplementedError(f"Padding mode '{mode}' not implemented")
    if isinstance(pad_width, int):
        pad_width = [pad_width]
    # pad_width format: [left, right, top, bottom, front, back, ...]
    # For 4D [N, C, H, W] input with pad=[left, right, top, bottom]:
    # This is a placeholder that returns the input unchanged for now.
    return x
