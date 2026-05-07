"""Layer definitions for fastnn.

This module provides Python-implemented neural network layers.
For high-performance Rust layers, import directly from `fastnn` (e.g., `fastnn.Linear`).

Examples:
    >>> import fastnn as fnn
    >>> from fastnn.layers import Flatten, PySequential
    >>> layer = fnn.Linear(128, 64)  # Rust implementation
    >>> x = fnn.randn([32, 128])
    >>> y = layer(x)
"""

from typing import Optional, Tuple, List, Any, Union

import math
import numpy as np
import fastnn._core as _core
from fastnn.module import Module


class _BaseModule(Module):
    """Base class for modules with common layer iteration logic."""
    def parameters(self):
        params = []
        for layer in self._param_layers:
            params.extend(layer.parameters())
        return params

    def train_mode(self):
        for layer in self._train_layers:
            layer.train_mode()

    def eval_mode(self):
        for layer in self._eval_layers:
            layer.eval_mode()


# Python-implemented layers (for compatibility and educational purposes)

class MaxPool2dPy(Module):
    """Max pooling 2D layer (Python implementation).
    
    This is a pure-Python implementation of 2D max pooling, provided for
    compatibility and educational purposes. For performance, use the Rust
    implementation `fastnn.MaxPool2d`.
    
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
        # Process kernel_size to scalar
        if isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise NotImplementedError(f"Non-square kernel_size not supported yet")
            kernel_size_scalar = kernel_size[0]
        else:
            kernel_size_scalar = kernel_size
        
        # Process stride to scalar
        if stride is None:
            stride = kernel_size
        if isinstance(stride, tuple):
            if stride[0] != stride[1]:
                raise NotImplementedError(f"Non-square stride not supported yet")
            stride_scalar = stride[0]
        else:
            stride_scalar = stride
        
        # Process padding to scalar
        if isinstance(padding, tuple):
            if padding[0] != padding[1]:
                raise NotImplementedError(f"Non-square padding not supported yet")
            padding_scalar = padding[0]
        else:
            padding_scalar = padding
        
        # Process dilation to scalar
        if isinstance(dilation, tuple):
            if dilation[0] != dilation[1]:
                raise NotImplementedError(f"Non-square dilation not supported yet")
            dilation_scalar = dilation[0]
        else:
            dilation_scalar = dilation
        
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        self._kernel_size_scalar = kernel_size_scalar
        self._stride_scalar = stride_scalar
        self._padding_scalar = padding_scalar
        self._dilation_scalar = dilation_scalar
        
        # Pre-create Rust object
        self._rust_maxpool = _core.MaxPool2d(
            self._kernel_size_scalar,
            self._stride_scalar,
            self._padding_scalar,
            self._dilation_scalar
        )
    
    def parameters(self):
        return []
    
    def __call__(self, x):
        return self._rust_maxpool(x)


class Flatten(Module):
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
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def parameters(self):
        return []
    
    def __call__(self, x):
        shape = x.shape
        ndim = len(shape)
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        end = end + 1
        new_shape = list(shape[:start])
        flattened_size = math.prod(shape[start:end])
        new_shape.append(flattened_size)
        new_shape.extend(shape[end:])
        return x.view(new_shape)


class PySequential(_BaseModule):
    """Sequential container (Python implementation).
    
    A simple sequential container that applies layers in order.
    This is provided for compatibility; prefer `fastnn.Sequential`
    (the Rust implementation) for better performance.
    
    Args:
        layers: List of layers to apply sequentially.
    
    Examples:
        >>> model = PySequential([
        ...     fnn.Linear(784, 256),
        ...     fnn.ReLU(),
        ...     fnn.Linear(256, 10),
        ... ])
        >>> x = fnn.randn([32, 784])
        >>> y = model(x)
    """
    
    def __init__(self, layers: List[Any]):
        self.layers = layers
        self._param_layers = []
        self._gpu_layers = []
        self._train_layers = []
        self._eval_layers = []
        for l in layers:
            if hasattr(l, "parameters"):
                self._param_layers.append(l)
            if hasattr(l, "to_gpu"):
                self._gpu_layers.append(l)
            if hasattr(l, "train"):
                self._train_layers.append(l)
            if hasattr(l, "eval"):
                self._eval_layers.append(l)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def to_gpu(self, device_id: int):
        for layer in self._gpu_layers:
            layer.to_gpu(device_id)


class BasicBlock(_BaseModule):
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
        
        # Single pass to initialize layer lists
        self._param_layers = []
        self._zero_grad_layers = []
        self._train_layers = []
        self._eval_layers = []
        for l in self._sublayers:
            if hasattr(l, "parameters"):
                self._param_layers.append(l)
            if hasattr(l, "zero_grad"):
                self._zero_grad_layers.append(l)
            if hasattr(l, "train"):
                self._train_layers.append(l)
            if hasattr(l, "eval"):
                self._eval_layers.append(l)
        
        # Named parameters setup
        self._named_param_pairs = []
        for name, layer in [("conv1", conv1), ("bn1", bn1), ("conv2", conv2), ("bn2", bn2)]:
            if hasattr(layer, "named_parameters"):
                self._named_param_pairs.append((name, layer))
        if downsample is not None and hasattr(downsample, "named_parameters"):
            self._named_param_pairs.append(("downsample", downsample))
    
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
    
    def named_parameters(self):
        params = []
        for name, layer in self._named_param_pairs:
            for n, p in layer.named_parameters():
                params.append((f"{name}.{n}", p))
        return params
    
    def zero_grad(self):
        for layer in self._zero_grad_layers:
            layer.zero_grad()
    
    def train_mode(self):
        for layer in self._train_layers:
            layer.train_mode()
    
    def eval_mode(self):
        for layer in self._eval_layers:
            layer.eval_mode()
    
    # Backward compatibility
    train = train_mode
    eval = eval_mode


# Alias for backward compatibility
MaxPool2dPy = MaxPool2dPy
