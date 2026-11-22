"""fastnn — an inference-only neural-network runtime.

The public surface is :mod:`fastnn.inference`: prepare an artifact once, load it
on an explicit device, and run sessions that own their cache and their KV state.
Tensors cross the boundary as NumPy arrays.

There is no autograd, optimizer, training loop, data loader or wgpu backend.
"""

from fastnn import inference

__all__ = ["inference"]
