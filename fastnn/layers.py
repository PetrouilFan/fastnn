"""Layer definitions for fastnn.

This module provides neural network layers, including both Python implementations
and re-exports of high-performance Rust layers. All layers are callable and
implement the `train()` and `eval()` methods for switching between training
and inference modes.

Examples:
    >>> import fastnn as fnn
    >>> layer = fnn.layers.Linear(128, 64)
    >>> x = fnn.randn([32, 128])
    >>> y = layer(x)
"""

from typing import Optional, Tuple, List, Any, Union

import fastnn._core as _core
from fastnn.module import Module

# Re-export Rust layer classes with improved naming (remove 'Py' prefix)
# We'll keep the original names but also provide aliases without 'Py' for consistency.
# However we cannot modify the original class names, so we re-export them as is.
# Users can import from fastnn.layers directly.

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
MaxPool2d = _core.MaxPool2d
ConvTranspose2d = _core.ConvTranspose2d
Conv1d = _core.Conv1d
Conv3d = _core.Conv3d
RMSNorm = _core.RMSNorm
GroupNorm = _core.GroupNorm
BatchNorm2d = _core.BatchNorm2d
LeakyReLU = _core.LeakyReLU
Softplus = _core.Softplus
Hardswish = _core.Hardswish
Dropout2d = _core.Dropout2d
Upsample = _core.Upsample
AdaptiveAvgPool2d = _core.AdaptiveAvgPool2d
FusedConvBnSilu = getattr(_core, "PyFusedConvBnSilu", None)
ResidualBlock = _core.ResidualBlock

# Python-implemented layers (for compatibility and educational purposes)

class MaxPool2dPy:
    """Max pooling 2D layer (Python implementation).
    
    This is a pure-Python implementation of 2D max pooling, provided for
    compatibility and educational purposes. For performance, use the Rust
    implementation `fastnn.layers.MaxPool2d`.
    
    Args:
        kernel_size: Size of the pooling window. If int, square kernel.
        stride: Stride of the pooling window. Defaults to kernel_size.
        padding: Padding added to both sides of the input. Defaults to 0.
        dilation: Spacing between kernel elements. Defaults to 1.
        return_indices: If True, return indices of max values. Defaults to False.
        ceil_mode: If True, use ceil instead of floor for output shape. Defaults to False.
    
    Examples:
        >>> pool = MaxPool2dPy(kernel_size=2, stride=2)
        >>> x = fnn.randn([1, 3, 32, 32])
        >>> y = pool(x)
    """
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
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
        # For now, only support symmetric kernel_size, stride, padding, dilation
        # Check if they are tuples and convert to single value if possible
        kernel_size = self._as_single_value(self.kernel_size, "kernel_size")
        stride = self._as_single_value(self.stride, "stride")
        padding = self._as_single_value(self.padding, "padding")
        dilation = self._as_single_value(self.dilation, "dilation")
        
        # Call the Rust implementation
        rust_maxpool = _core.MaxPool2d(kernel_size, stride, padding, dilation)
        return rust_maxpool(x)
    
    @staticmethod
    def _as_single_value(value, name):
        """Convert a value to a single number if it is a symmetric pair."""
        if isinstance(value, tuple):
            if value[0] != value[1]:
                raise NotImplementedError(f"Non-square {name} not supported yet")
            return value[0]
        return value
    
    def train(self):
        pass
    
    def eval(self):
        pass


class Flatten:
    """Flatten layer.
    
    Flattens the input tensor starting from a given dimension.
    
    Args:
        start_dim: First dimension to flatten (default: 1).
        end_dim: Last dimension to flatten (default: -1).
    
    Examples:
        >>> flatten = Flatten()
        >>> x = fnn.randn([32, 3, 32, 32])
        >>> y = flatten(x)  # shape [32, 3072]
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
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


class PySequential(Module):
    """Sequential container (Python implementation).
    
    A simple sequential container that applies layers in order.
    This is provided for compatibility; prefer `fastnn.layers.Sequential`
    (the Rust implementation) for better performance.
    
    Args:
        layers: List of layers to apply sequentially.
    
    Examples:
        >>> model = PySequential([
        ...     fnn.layers.Linear(784, 256),
        ...     fnn.layers.ReLU(),
        ...     fnn.layers.Linear(256, 10),
        ... ])
        >>> x = fnn.randn([32, 784])
        >>> y = model(x)
    """
    
    def __init__(self, layers: List[Any]):
        self.layers = layers
        self._param_layers = [l for l in layers if hasattr(l, "parameters")]
        self._gpu_layers = [l for l in layers if hasattr(l, "to_gpu")]
        self._train_layers = [l for l in layers if hasattr(l, "train")]
        self._eval_layers = [l for l in layers if hasattr(l, "eval")]
    
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
        for layer in self._param_layers:
            params.extend(layer.parameters())
        return params
    
    def to_gpu(self, device_id: int):
        for layer in self._gpu_layers:
            layer.to_gpu(device_id)
    
    def train(self):
        for layer in self._train_layers:
            layer.train()
    
    def eval(self):
        for layer in self._eval_layers:
            layer.eval()


class BasicBlock(Module):
    """ResNet BasicBlock with skip connection.
    
    This is a building block for ResNet architectures.
    
    Args:
        conv1: First convolution layer.
        bn1: First batch normalization layer.
        relu: Activation function.
        conv2: Second convolution layer.
        bn2: Second batch normalization layer.
        downsample: Optional downsample layer for identity mapping.
    
    Examples:
        >>> block = BasicBlock(conv1, bn1, ReLU(), conv2, bn2)
        >>> x = fnn.randn([1, 64, 32, 32])
        >>> y = block(x)
    """
    
    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample=None):
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        self._sublayers = [conv1, bn1, conv2, bn2]
        if downsample is not None:
            self._sublayers.append(downsample)
        self._param_layers = [l for l in self._sublayers if hasattr(l, "parameters")]
        self._named_param_pairs = []
        for name, layer in [("conv1", conv1), ("bn1", bn1), ("conv2", conv2), ("bn2", bn2)]:
            if hasattr(layer, "named_parameters"):
                self._named_param_pairs.append((name, layer))
        if downsample is not None and hasattr(downsample, "named_parameters"):
            self._named_param_pairs.append(("downsample", downsample))
        self._zero_grad_layers = [l for l in self._sublayers if hasattr(l, "zero_grad")]
        self._train_layers = [l for l in self._sublayers if hasattr(l, "train")]
        self._eval_layers = [l for l in self._sublayers if hasattr(l, "eval")]
    
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
        for layer in self._param_layers:
            params.extend(layer.parameters())
        return params
    
    def named_parameters(self):
        params = []
        for name, layer in self._named_param_pairs:
            for n, p in layer.named_parameters():
                params.append((f"{name}.{n}", p))
        return params
    
    def zero_grad(self):
        for layer in self._zero_grad_layers:
            layer.zero_grad()
    
    def train(self):
        for layer in self._train_layers:
            layer.train()
    
    def eval(self):
        for layer in self._eval_layers:
            layer.eval()


# Alias for backward compatibility
MaxPool2dPy = MaxPool2dPy
