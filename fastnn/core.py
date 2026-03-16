import contextlib
import fastnn._core as _core


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
