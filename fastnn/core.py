import contextlib
import fastnn._core as _core

# Expose classes from _core
PyTransformerEncoder = _core.PyTransformerEncoder
Linear = _core.Linear
Conv2d = _core.Conv2d
LayerNorm = _core.LayerNorm
BatchNorm1d = _core.BatchNorm1d
Dropout = _core.Dropout
Embedding = _core.Embedding
ReLU = _core.ReLU
Gelu = _core.Gelu
Sigmoid = _core.Sigmoid
Tanh = _core.Tanh
SiLU = _core.SiLU

# Additional classes that may be needed
# Note: MultiHeadAttention is in Rust but not exposed via PyO3 yet
# We'll use the full TransformerEncoder which includes it


def sum(a, dim=None, keepdim=False):
    if dim is None:
        # For full sum, sum all dimensions with keepdim=True to preserve shape for backward
        # Then squeeze all dimensions
        result = a
        for i in range(a.ndim):
            result = _core.sum(result, 0, True)  # keepdim=True
        # Now squeeze to get scalar
        while result.ndim > 0:
            result = _core.sum(result, 0, False)
        return result
    return _core.sum(a, dim, keepdim)


def mean(a, dim=None, keepdim=False):
    if dim is None:
        result = _core.mean(a)
        while result.ndim > 0:
            result = _core.mean(result)
        return result
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
