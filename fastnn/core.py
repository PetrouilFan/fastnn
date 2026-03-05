import contextlib
import fastnn._core as _core


def sum(a, dim=None, keepdim=False):
    if dim is None:
        result = _core.sum(a)
        while result.ndim > 0:
            result = _core.sum(result)
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
    _core._set_default_device(device)
