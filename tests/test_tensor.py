import pytest
import numpy as np
import fastnn as fnn


def test_zeros():
    t = fnn.zeros([3, 4], dtype="f32")
    assert t.shape == [3, 4]
    assert t.dtype == "f32"


def test_ones():
    t = fnn.ones([2, 3], dtype="f32")
    assert t.shape == [2, 3]
    assert np.allclose(t.numpy(), 1.0)


def test_full():
    t = fnn.full([2, 2], 5.0, dtype="f32")
    assert t.shape == [2, 2]
    assert np.allclose(t.numpy(), 5.0)


def test_arange():
    t = fnn.arange(0.0, 5.0, 1.0)
    assert t.shape == [5]
    assert np.allclose(t.numpy(), [0, 1, 2, 3, 4])


def test_linspace():
    t = fnn.linspace(0.0, 1.0, 5)
    assert t.shape == [5]
    expected = np.linspace(0.0, 1.0, 5)
    assert np.allclose(t.numpy(), expected)


def test_eye():
    t = fnn.eye(3)
    assert t.shape == [3, 3]
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
    assert v.shape == [6, 4]


def test_reshape():
    t = fnn.zeros([2, 3, 4])
    r = t.reshape([6, 4])
    assert r.shape == [6, 4]


def test_transpose():
    t = fnn.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert tr.shape == [3, 2]


def test_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0])
    t = fnn.tensor(arr.tolist(), arr.shape)
    result = t.numpy()
    assert np.allclose(arr, result)


def test_tensor_from_numpy():
    arr = np.array([1.0, 2.0, 3.0])
    t = fnn.tensor(arr.tolist(), list(arr.shape))
    assert np.allclose(t.numpy(), arr)
