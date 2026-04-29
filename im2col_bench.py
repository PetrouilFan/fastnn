#!/usr/bin/env python3
"""Benchmark optimized im2col implementation vs baseline for Conv2d scenarios."""

import time
import fastnn
import torch
import numpy as np
import tracemalloc

def baseline_conv2d_im2col(x, w, bias, stride=1, padding=1, dilation=1):
    """Baseline im2col + GEMM implementation in PyTorch."""
    batch, in_ch, h, w = x.shape
    out_ch, _, kh, kw = w.shape

    # Compute output dimensions
    oh = (h + 2 * padding - kh) // stride + 1
    ow = (w + 2 * padding - kw) // stride + 1

    col_rows = batch * oh * ow
    col_cols = in_ch * kh * kw

    # im2col
    col = torch.zeros(col_rows, col_cols, dtype=torch.float32)
    for row in range(col_rows):
        n = row // (oh * ow)
        sp = row % (oh * ow)
        sph = sp // ow
        spw = sp % ow
        for ic in range(in_ch):
            for ky in range(kh):
                for kx in range(kw):
                    ih = sph * stride + ky
                    iw = spw * stride + kx
                    col_col = (ic * kh + ky) * kw + kx
                    if ih >= padding and ih < padding + h and iw >= padding and iw < padding + w:
                        xih = ih - padding
                        xiw = iw - padding
                        col[row, col_col] = x[n, ic, xih, xiw]

    # GEMM: col @ w.view(out_ch, -1).t() + bias
    w_flat = w.view(out_ch, -1)
    result = torch.mm(col, w_flat.t())
    if bias is not None:
        result += bias.view(1, -1)
    result = result.view(batch, oh, ow, out_ch).permute(0, 3, 1, 2)
    return result

def benchmark_im2col(batch_size=1, in_channels=32, out_channels=64, height=64, width=64, kernel_size=3, stride=1, padding=1, iters=10):
    """Benchmark im2col optimized vs baseline."""
    print(f"Benchmarking Conv2d({in_channels}->{out_channels}, k={kernel_size}) on {batch_size}x{in_channels}x{height}x{width}")

    # FastNN optimized
    fastnn_conv = fastnn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    x_fastnn = fastnn.randn([batch_size, in_channels, height, width])
    x_fastnn.requires_grad_(False)
    params = fastnn_conv.parameters()
    params[0].requires_grad_(False)  # weight

    # Baseline inputs (random, same shapes)
    x_torch = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32)
    w_torch = torch.randn(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32)
    bias_torch = None

    # Warmup FastNN
    for _ in range(3):
        _ = fastnn_conv(x_fastnn)

    # Warmup baseline
    for _ in range(3):
        _ = baseline_conv2d_im2col(x_torch, w_torch, bias_torch, stride=stride, padding=padding)

    # Benchmark FastNN time
    fastnn_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fastnn_conv(x_fastnn)
        t1 = time.perf_counter()
        fastnn_times.append((t1 - t0) * 1000)

    fastnn_median = np.median(fastnn_times)
    print(f"  FastNN optimized median time: {fastnn_median:.2f} ms")

    # Benchmark baseline time
    baseline_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = baseline_conv2d_im2col(x_torch, w_torch, bias_torch, stride=stride, padding=padding)
        t1 = time.perf_counter()
        baseline_times.append((t1 - t0) * 1000)

    baseline_median = np.median(baseline_times)
    print(f"  Baseline median time: {baseline_median:.2f} ms")

    speedup = baseline_median / fastnn_median if fastnn_median > 0 else 0.0
    print(f"  Speedup: {speedup:.2f}x")

    # Memory usage (peak during execution)
    tracemalloc.start()
    _ = fastnn_conv(x_fastnn)
    current, peak_fastnn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = baseline_conv2d_im2col(x_torch, w_torch, bias_torch, stride=stride, padding=padding)
    current, peak_baseline = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  FastNN peak memory: {peak_fastnn / 1024 / 1024:.2f} MB")
    print(f"  Baseline peak memory: {peak_baseline / 1024 / 1024:.2f} MB")

    return fastnn_median, baseline_median, speedup, peak_fastnn, peak_baseline

def main():
    scenarios = [
        (1, 32, 64, 64, 64, 3, 1, 1),  # Small
        (1, 64, 128, 32, 32, 3, 1, 1), # Medium
        (1, 128, 256, 16, 16, 3, 1, 1), # Large channels
        (2, 64, 128, 64, 64, 3, 1, 1), # Batch 2
        (1, 64, 64, 64, 64, 1, 1, 0),  # 1x1 conv
    ]

    for batch, in_ch, out_ch, h, w, k, s, p in scenarios:
        benchmark_im2col(batch, in_ch, out_ch, h, w, k, s, p)
        print()

if __name__ == "__main__":
    main()