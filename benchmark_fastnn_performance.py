#!/usr/bin/env python3
"""Benchmark FastNN Conv2d performance for YOLO-style operations."""

import time
import fastnn
import numpy as np

def benchmark_conv(batch_size=1, in_channels=64, out_channels=128, height=32, width=32, iters=100):
    """Benchmark Conv2d performance."""
    print(f"Benchmarking Conv2d({in_channels}->{out_channels}) on {batch_size}x{in_channels}x{height}x{width}")

    conv = fastnn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    x = fastnn.randn([batch_size, in_channels, height, width])

    # Warmup
    for _ in range(10):
        _ = conv(x)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = conv(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median_time = np.median(times)
    fps = 1000.0 / median_time
    print(f"  Median time: {median_time:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")

    return median_time, fps

def benchmark_conv_bn_silu(batch_size=1, in_channels=64, out_channels=128, height=32, width=32, iters=100):
    """Benchmark Conv2d + BatchNorm + SiLU separate operations."""
    print(f"Benchmarking Conv2d({in_channels}->{out_channels}) + BN + SiLU on {batch_size}x{in_channels}x{height}x{width}")

    conv = fastnn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    bn = fastnn.BatchNorm2d(out_channels)
    silu = fastnn.SiLU()

    x = fastnn.randn([batch_size, in_channels, height, width])

    # Warmup
    for _ in range(10):
        out = conv(x)
        out = bn(out)
        out = silu(out)

    # Benchmark
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = conv(x)
        out = bn(out)
        out = silu(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    median_time = np.median(times)
    fps = 1000.0 / median_time
    print(f"  Median time: {median_time:.2f} ms")
    print(f"  Throughput: {fps:.1f} FPS")

    return median_time, fps

def main():
    print("=" * 60)
    print("FastNN YOLO-style Operation Performance")
    print("=" * 60)

    configs = [
        (1, 32, 64, 64, 64),   # Small feature maps
        (1, 64, 128, 32, 32),  # Medium feature maps
        (1, 128, 256, 16, 16), # Large channels, small spatial
    ]

    print("\nConv2d Only:")
    print("-" * 30)
    for batch, in_ch, out_ch, h, w in configs:
        benchmark_conv(batch, in_ch, out_ch, h, w)
        print()

    print("Conv2d + BatchNorm + SiLU:")
    print("-" * 30)
    for batch, in_ch, out_ch, h, w in configs:
        benchmark_conv_bn_silu(batch, in_ch, out_ch, h, w)
        print()

    print("=" * 60)
    print("Performance Analysis:")
    print("- FastNN uses SIMD-accelerated kernels")
    print("- Memory-efficient tensor operations")
    print("- Optimized for inference workloads")
    print()
    print("For YOLO11n full model:")
    print("- 88 Conv layers with BN and SiLU activations")
    print("- Multi-scale feature fusion (Concat operations)")
    print("- Complex graph with skip connections")
    print()
    print("Next steps:")
    print("1. Complete ONNX graph reconstruction")
    print("2. Implement operator fusion for Conv+BN+SiLU")
    print("3. Add memory pooling for intermediates")
    print("4. Optimize execution order and dependencies")

if __name__ == "__main__":
    main()