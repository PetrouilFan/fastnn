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

from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import fastnn._core as _core
from fastnn.module import Module


def _to_scalar(value: Union[int, Tuple[int, int]], name: str) -> int:
    """Convert a tuple or int to a scalar value.
    
    Args:
        value: Either an int or a tuple of ints.
        name: Name of the parameter (for error messages).
    
    Returns:
        The scalar value.
    
    Raises:
        NotImplementedError: If tuple values are not equal (non-square).
    """
    if isinstance(value, tuple):
        if value[0] != value[1]:
            raise NotImplementedError(f"Non-square {name} not supported yet")
        return value[0]
    return value


class _BaseModule(Module):
    """Base class for modules with common layer iteration logic."""
    __slots__ = ('_param_layers', '_zero_grad_layers', '_train_layers', 
                 '_eval_layers', '_gpu_layers', '_named_param_pairs')
    
    def __init__(self):
        super().__init__()
        self._param_layers = []
        self._zero_grad_layers = []
        self._train_layers = []
        self._eval_layers = []
        self._gpu_layers = []
        self._named_param_pairs = []
    
    def _register_layer(self, layer: Any, name: Optional[str] = None) -> None:
        """Register a layer for parameter iteration, training mode, etc."""
        has_params = hasattr(layer, "parameters")
        has_zero_grad = hasattr(layer, "zero_grad")
        has_train = hasattr(layer, "train_mode")
        has_eval = hasattr(layer, "eval_mode")
        has_gpu = hasattr(layer, "to_gpu")
        has_named_params = name and hasattr(layer, "named_parameters")
        
        if has_params:
            self._param_layers.append(layer)
        if has_zero_grad:
            self._zero_grad_layers.append(layer)
        if has_train:
            self._train_layers.append(layer)
        if has_eval:
            self._eval_layers.append(layer)
        if has_gpu:
            self._gpu_layers.append(layer)
        if has_named_params:
            self._named_param_pairs.append((name, layer))
    
    def parameters(self) -> List[Any]:
        params = []
        for layer in self._param_layers:
            params.extend(layer.parameters())
        return params
    
    def named_parameters(self) -> List[Tuple[str, Any]]:
        params = []
        for name, layer in self._named_param_pairs:
            for n, p in layer.named_parameters():
                params.append((f"{name}.{n}", p))
        return params
    
    def zero_grad(self):
        for layer in self._zero_grad_layers:
            layer.zero_grad()
    
    def train_mode(self) -> None:
        super().train_mode()
        for layer in self._train_layers:
            layer.train_mode()
        for layer in self._param_layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval_mode(self) -> None:
        super().eval_mode()
        for layer in self._eval_layers:
            layer.eval_mode()
        for layer in self._param_layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def train(self) -> None:
        self.train_mode()
    
    def eval(self) -> None:
        self.eval_mode()
    
    def to_gpu(self, device_id: int) -> None:
        for layer in self._gpu_layers:
            layer.to_gpu(device_id)
    
    def state_dict(self) -> Dict[str, Any]:
        result = {}
        for name, param in self.named_parameters():
            result[name] = param
        return result
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        loaded_keys = set()
        for name_prefix, layer in self._named_param_pairs:
            for param_name, param in layer.named_parameters():
                full_name = f"{name_prefix}.{param_name}"
                loaded_keys.add(full_name)
                if full_name not in state_dict:
                    raise KeyError(f"Missing key in state_dict: {full_name}")
                src = state_dict[full_name]
                if hasattr(src, 'numpy'):
                    src_arr = src.numpy()
                else:
                    src_arr = np.asarray(src)
                data = src_arr.flatten().tolist()
                new_param = _core.PyTensor(data, list(param.shape))
                setter = f"set_{param_name}"
                if hasattr(layer, setter):
                    getattr(layer, setter)(new_param)
        extra_keys = set(state_dict.keys()) - loaded_keys
        if extra_keys:
            raise KeyError(f"Unexpected keys in state_dict: {extra_keys}")


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
    __slots__ = ('return_indices', 'ceil_mode', '_kernel_size_scalar', 
                 '_stride_scalar', '_padding_scalar', '_dilation_scalar', '_rust_maxpool')
    
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        # Process parameters to scalars using helper
        kernel_size_scalar = _to_scalar(kernel_size, "kernel_size")
        
        stride = kernel_size if stride is None else stride
        stride_scalar = _to_scalar(stride, "stride")
        
        padding_scalar = _to_scalar(padding, "padding")
        dilation_scalar = _to_scalar(dilation, "dilation")
        
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

    def named_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

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
    __slots__ = ('start_dim', 'end_dim')
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def parameters(self):
        return []

    def named_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        shape = x.shape
        ndim = len(shape)
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        new_shape = list(shape[:start])
        flattened_size = math.prod(shape[start:end + 1])
        new_shape.append(flattened_size)
        new_shape.extend(shape[end + 1:])
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
    __slots__ = ('layers',)
    
    def __init__(self, layers: List[Any]):
        super().__init__()
        self.layers = layers
        for i, l in enumerate(layers):
            # Only register with name if layer has named_parameters
            if hasattr(l, "named_parameters"):
                self._register_layer(l, str(i))
            else:
                self._register_layer(l)
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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
    __slots__ = ('conv1', 'bn1', 'relu', 'conv2', 'bn2', 'downsample')
    
    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample=None):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        
        # Register layers with names for named_parameters using a loop
        layers_to_register = [
            (conv1, "conv1"),
            (bn1, "bn1"),
            (conv2, "conv2"),
            (bn2, "bn2"),
        ]
        if downsample is not None:
            layers_to_register.append((downsample, "downsample"))
        
        for layer, name in layers_to_register:
            self._register_layer(layer, name)
    
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
