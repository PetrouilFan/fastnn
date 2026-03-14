import time
import fastnn as fnn
import torch


def benchmark_conv2d():
    configs = [
        (1, 32, 32, 32, 3),  # (batch, channels, height, width, kernel)
        (1, 64, 64, 64, 3),
    ]

    for config in configs:
        batch, channels, height, width, kernel = config
        print(f"\nBenchmarking Conv2d with config: {config}")

        # fastnn
        conv_fnn = fnn.Conv2d(channels, channels, kernel)
        x_fnn = fnn.rand([batch, channels, height, width])

        # warmup
        for _ in range(10):
            _ = conv_fnn(x_fnn)

        # benchmark fastnn
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            _ = conv_fnn(x_fnn)
        end = time.time()
        fnn_time = (end - start) / iterations * 1000  # ms

        # PyTorch
        conv_torch = torch.nn.Conv2d(channels, channels, kernel)
        x_torch = torch.rand(batch, channels, height, width)

        # warmup
        for _ in range(10):
            _ = conv_torch(x_torch)

        # benchmark PyTorch
        start = time.time()
        for _ in range(iterations):
            _ = conv_torch(x_torch)
        end = time.time()
        torch_time = (end - start) / iterations * 1000  # ms

        print(f"  fastnn: {fnn_time:.2f} ms")
        print(f"  PyTorch: {torch_time:.2f} ms")
        print(f"  Ratio (fastnn/torch): {fnn_time / torch_time:.2f}x")


if __name__ == "__main__":
    benchmark_conv2d()
