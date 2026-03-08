import pytest
import fastnn as fnn


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_add(benchmark, size):
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    benchmark(lambda: a + b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_mul(benchmark, size):
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    benchmark(lambda: a * b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_relu(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.relu, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_sigmoid(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.sigmoid, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_tanh(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.tanh, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000), (3162, 3162)])
def test_gelu(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.gelu, x)


@pytest.mark.parametrize(
    "sizes",
    [
        (128, 256, 256, 128),
        (256, 512, 512, 256),
        (512, 512, 512, 512),
        (512, 1024, 1024, 512),
        (1024, 1024, 1024, 1024),
    ],
)
def test_matmul(benchmark, sizes):
    m, k, k2, n = sizes
    assert k == k2
    a = fnn.rand([m, k])
    b = fnn.rand([k, n])
    benchmark(lambda: a @ b)


@pytest.mark.parametrize(
    "config",
    [
        (32, 256, 512),
        (32, 512, 1024),
        (32, 1024, 2048),
        (128, 256, 512),
        (128, 512, 1024),
    ],
)
def test_linear(benchmark, config):
    batch_size, in_features, out_features = config
    linear = fnn.Linear(in_features, out_features)
    x = fnn.rand([batch_size, in_features])
    benchmark(linear, x)


@pytest.mark.parametrize(
    "config",
    [
        (1, 32, 32, 32, 3),
        (1, 64, 32, 32, 3),
        (1, 64, 64, 64, 3),
        (1, 128, 64, 64, 3),
    ],
)
def test_conv2d(benchmark, config):
    batch, channels, height, width, kernel = config
    conv = fnn.Conv2d(channels, channels, kernel)
    x = fnn.rand([batch, channels, height, width])
    benchmark(conv, x)


@pytest.mark.parametrize("size", [(1000, 1000), (3162, 3162)])
def test_sum(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.sum, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000), (3162, 3162)])
def test_mean(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.mean, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000), (3162, 3162)])
def test_max(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.max, x, 1)


def test_zeros(benchmark):
    benchmark(fnn.zeros, [1000, 1000])


def test_ones(benchmark):
    benchmark(fnn.ones, [1000, 1000])


def test_rand(benchmark):
    benchmark(fnn.rand, [1000, 1000])


def test_to_numpy(benchmark):
    x = fnn.rand([1000, 1000])
    benchmark(x.numpy)
