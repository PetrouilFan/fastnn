import contextlib
import fastnn._core as _core

# Utility functions that wrap _core with Python-friendly interfaces


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


def maximum(a, b):
    """Element-wise maximum of two tensors.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Tensor containing element-wise maximum
    """
    return a.maximum(b)


def minimum(a, b):
    """Element-wise minimum of two tensors.

    Args:
        a: First tensor
        b: Second tensor

    Returns:
        Tensor containing element-wise minimum
    """
    return a.minimum(b)


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
