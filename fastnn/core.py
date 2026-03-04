import contextlib
import fastnn._core as _core


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
