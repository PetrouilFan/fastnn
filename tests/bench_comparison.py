import pytest
import fastnn as fnn
import torch


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_add_fastnn(benchmark, size):
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    benchmark(lambda: a + b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_add_torch(benchmark, size):
    a = torch.rand(size)
    b = torch.rand(size)
    benchmark(lambda: a + b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_mul_fastnn(benchmark, size):
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    benchmark(lambda: a * b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_mul_torch(benchmark, size):
    a = torch.rand(size)
    b = torch.rand(size)
    benchmark(lambda: a * b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_relu_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.relu, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_relu_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.nn.functional.relu, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_fused_add_relu_fastnn(benchmark, size):
    a = fnn.rand(list(size))
    b = fnn.rand(list(size))
    benchmark(fnn.fused_add_relu, a, b)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_fused_add_relu_torch(benchmark, size):
    a = torch.rand(size)
    b = torch.rand(size)
    benchmark(lambda: torch.nn.functional.relu(a + b))


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_sigmoid_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.sigmoid, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_sigmoid_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.sigmoid, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_tanh_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.tanh, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_tanh_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.tanh, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_gelu_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.gelu, x)


@pytest.mark.parametrize("size", [(100, 100), (1000, 1000)])
def test_gelu_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.nn.functional.gelu, x)


@pytest.mark.parametrize(
    "sizes",
    [
        (128, 256, 256, 128),
        (256, 512, 512, 256),
        (512, 1024, 1024, 512),
    ],
)
def test_matmul_fastnn(benchmark, sizes):
    m, k, k2, n = sizes
    assert k == k2
    a = fnn.rand([m, k])
    b = fnn.rand([k, n])
    benchmark(lambda: a @ b)


@pytest.mark.parametrize(
    "sizes",
    [
        (128, 256, 256, 128),
        (256, 512, 512, 256),
        (512, 1024, 1024, 512),
    ],
)
def test_matmul_torch(benchmark, sizes):
    m, k, k2, n = sizes
    assert k == k2
    a = torch.rand(m, k)
    b = torch.rand(k, n)
    benchmark(lambda: a @ b)


@pytest.mark.parametrize(
    "config",
    [
        (32, 256, 512),
        (32, 512, 1024),
        (128, 256, 512),
    ],
)
def test_linear_fastnn(benchmark, config):
    batch_size, in_features, out_features = config
    linear = fnn.Linear(in_features, out_features)
    x = fnn.rand([batch_size, in_features])
    benchmark(linear, x)


@pytest.mark.parametrize(
    "config",
    [
        (32, 256, 512),
        (32, 512, 1024),
        (128, 256, 512),
    ],
)
def test_linear_torch(benchmark, config):
    batch_size, in_features, out_features = config
    linear = torch.nn.Linear(in_features, out_features)
    x = torch.rand(batch_size, in_features)
    benchmark(linear, x)


@pytest.mark.parametrize(
    "config",
    [
        (1, 32, 32, 32, 3),
        (1, 64, 64, 64, 3),
    ],
)
def test_conv2d_fastnn(benchmark, config):
    batch, channels, height, width, kernel = config
    conv = fnn.Conv2d(channels, channels, kernel)
    x = fnn.rand([batch, channels, height, width])
    benchmark(conv, x)


@pytest.mark.parametrize(
    "config",
    [
        (1, 32, 32, 32, 3),
        (1, 64, 64, 64, 3),
    ],
)
def test_conv2d_torch(benchmark, config):
    batch, channels, height, width, kernel = config
    conv = torch.nn.Conv2d(channels, channels, kernel)
    x = torch.rand(batch, channels, height, width)
    benchmark(conv, x)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_sum_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.sum, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_sum_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.sum, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_mean_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.mean, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_mean_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.mean, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_max_fastnn(benchmark, size):
    x = fnn.rand(list(size))
    benchmark(fnn.max, x, 1)


@pytest.mark.parametrize("size", [(1000, 1000)])
def test_max_torch(benchmark, size):
    x = torch.rand(size)
    benchmark(torch.max, x, 1)
