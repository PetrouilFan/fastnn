#!/usr/bin/env python3
"""Simple benchmark FastNN vs PyTorch for Conv2d."""

import time
import fastnn
import torch
import numpy as np

def benchmark_conv(batch_size=1, in_channels=32, out_channels=64, height=64, width=64, iters=100):
    """Benchmark Conv2d."""
    print(f"Benchmarking Conv2d({in_channels}->{out_channels}) on {batch_size}x{in_channels}x{height}x{width}")

    # PyTorch version
    torch_conv = torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
    x_torch = torch.randn(batch_size, in_channels, height, width)
    x_torch.requires_grad_(False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.bias.requires_grad_(False)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            out = torch_conv(x_torch)

    # Benchmark PyTorch
    torch_times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            out = torch_conv(x_torch)
            t1 = time.perf_counter()
            torch_times.append((t1 - t0) * 1000)

    torch_median = np.median(torch_times)
    print(f"  PyTorch median time: {torch_median:.2f} ms")

    # FastNN version
    fastnn_conv = fastnn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    x_fastnn = fastnn.randn([batch_size, in_channels, height, width])
    x_fastnn.requires_grad_(False)
    params = fastnn_conv.parameters()
    params[0].requires_grad_(False)  # weight
    if len(params) > 1:
        params[1].requires_grad_(False)  # bias

    # Warmup
    for _ in range(3):
        _ = fastnn_conv(x_fastnn)

    # Benchmark FastNN
    fastnn_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fastnn_conv(x_fastnn)
        t1 = time.perf_counter()
        fastnn_times.append((t1 - t0) * 1000)

    fastnn_median = np.median(fastnn_times)
    print(f"  FastNN median time: {fastnn_median:.2f} ms")

    speedup = torch_median / fastnn_median if fastnn_median > 0 else 0.0
    print(f"  Speedup: {speedup:.2f}x")

    return torch_median, fastnn_median, speedup

if __name__ == "__main__":
    benchmark_conv()