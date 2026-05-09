import sys
import pytest
import numpy as np
import fastnn as fnn
from tests.test_utils import assert_shape_equal


@pytest.mark.parametrize("factory, shape, dtype, expected_fill", [
    (fnn.zeros, [3, 4], "f32", 0.0),
    (fnn.ones, [2, 3], "f32", 1.0),
    (fnn.full, [2, 2], "f32", 5.0),
])
def test_creation_ops(factory, shape, dtype, expected_fill):
    if factory == fnn.full:
        t = factory(shape, expected_fill, dtype=dtype)
    else:
        t = factory(shape, dtype=dtype)
    assert_shape_equal(t, shape)
    assert t.dtype == dtype
    assert np.allclose(t.numpy(), expected_fill)


def test_arange():
    t = fnn.arange(0.0, 5.0, 1.0)
    assert_shape_equal(t, [5])
    assert np.allclose(t.numpy(), [0, 1, 2, 3, 4])


def test_linspace():
    t = fnn.linspace(0.0, 1.0, 5)
    assert_shape_equal(t, [5])
    expected = np.linspace(0.0, 1.0, 5)
    assert np.allclose(t.numpy(), expected)


def test_eye():
    t = fnn.eye(3)
    assert_shape_equal(t, [3, 3])
    expected = np.eye(3)
    assert np.allclose(t.numpy(), expected)


def test_add():
    a = fnn.tensor([1.0, 2.0, 3.0], [3])
    b = fnn.tensor([4.0, 5.0, 6.0], [3])
    c = a + b
    assert np.allclose(c.numpy(), [5.0, 7.0, 9.0])


def test_sub():
    a = fnn.tensor([5.0, 6.0, 7.0], [3])
    b = fnn.tensor([1.0, 2.0, 3.0], [3])
    c = a - b
    assert np.allclose(c.numpy(), [4.0, 4.0, 4.0])


def test_mul():
    a = fnn.tensor([2.0, 3.0, 4.0], [3])
    b = fnn.tensor([5.0, 6.0, 7.0], [3])
    c = a * b
    assert np.allclose(c.numpy(), [10.0, 18.0, 28.0])


def test_matmul():
    a = fnn.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = fnn.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = a @ b
    expected = np.array([[19, 22], [43, 50]])
    assert np.allclose(c.numpy(), expected)


def test_view():
    t = fnn.zeros([2, 3, 4])
    v = t.view([6, 4])
    assert_shape_equal(v, [6, 4])


def test_reshape():
    t = fnn.zeros([2, 3, 4])
    r = t.reshape([6, 4])
    assert_shape_equal(r, [6, 4])


def test_transpose():
    t = fnn.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert_shape_equal(tr, [3, 2])


def test_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0])
    t = fnn.tensor(arr.tolist(), arr.shape)
    result = t.numpy()
    assert np.allclose(arr, result)


def test_tensor_from_numpy():
    arr = np.array([1.0, 2.0, 3.0])
    t = fnn.tensor(arr.tolist(), list(arr.shape))
    assert np.allclose(t.numpy(), arr)


def test_memory_pool_reuse():
    import gc
    import json
    gc.collect()

    def allocs():
        return json.loads(fnn.allocator_stats())["num_allocs"]

    # First allocation: pool creates a new block.
    t1 = fnn.zeros([1000], dtype="f32")
    del t1
    gc.collect()
    n_first = allocs()

    # Second allocation of same size: pool may reuse.
    t2 = fnn.zeros([1000], dtype="f32")
    n_second = allocs()
    del t2
    gc.collect()
    assert n_second >= n_first, (
        f"pool reuse check: num_allocs decreased from {n_first} to {n_second}"
    )

    # Different size: should trigger at most one new allocation.
    t3 = fnn.zeros([2000], dtype="f32")
    n_third = allocs()
    assert n_third <= n_second + 1, (
        f"expected at most one new allocation for larger tensor: "
        f"{n_second} -> {n_third}"
    )
    del t3
    gc.collect()
