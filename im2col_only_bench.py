#!/usr/bin/env python3
"""Benchmark im2col extraction and GEMM separately from full Conv2d.

This script isolates the two main steps of a convolution:
1) im2col extraction: transforms input tensor into a column matrix.
2) GEMM: matrix multiplication of the column matrix with the flattened weight.

It uses fastnn's im2col function and matmul to time each step individually,
then compares with the full Conv2d forward pass to identify the bottleneck.
"""

import time
import fastnn
import numpy as np


def compute_output_dimensions(height, width, kernel_size, stride, padding):
    """Compute output height and width for a Conv2d."""
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    return out_h, out_w


def benchmark_im2col_and_gemm(
    batch_size=1,
    in_channels=32,
    out_channels=64,
    height=64,
    width=64,
    kernel_size=3,
    stride=1,
    padding=1,
    iters=10,
    warmup=3,
):
    """Benchmark im2col extraction and GEMM separately."""
    print(
        f"Benchmarking im2col + GEMM breakdown for Conv2d({in_channels}->{out_channels}, k={kernel_size}) "
        f"on {batch_size}x{in_channels}x{height}x{width}"
    )

    # Create input tensor (no gradient needed)
    x = fastnn.randn([batch_size, in_channels, height, width])
    x.requires_grad_(False)

    # Create weight tensor (out_channels, in_channels, kernel_size, kernel_size)
    weight = fastnn.randn([out_channels, in_channels, kernel_size, kernel_size])
    weight.requires_grad_(False)

    # Compute output spatial dimensions
    out_h, out_w = compute_output_dimensions(height, width, kernel_size, stride, padding)

    # ---------- Benchmark im2col extraction ----------
    # Warmup
    for _ in range(warmup):
        _ = fastnn.im2col(x, kernel_size, stride, padding)

    # Timed runs
    im2col_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        col = fastnn.im2col(x, kernel_size, stride, padding)
        t1 = time.perf_counter()
        im2col_times.append((t1 - t0) * 1000)  # ms
    im2col_median = np.median(im2col_times)
    print(f"  im2col extraction median: {im2col_median:.2f} ms")

    # ---------- Prepare for GEMM ----------
    # Flatten weight to (out_channels, in_channels*kernel_size*kernel_size) and transpose
    weight_flat = weight.reshape([out_channels, -1])
    weight_t = weight_flat.transpose(0, 1)  # shape (in_channels*kh*kw, out_channels)

    # Ensure we have the col_matrix from the last im2col call
    # (col is already defined from last iteration)

    # Warmup matmul
    for _ in range(warmup):
        _ = fastnn.matmul(col, weight_t)

    # Timed runs for GEMM
    gemm_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fastnn.matmul(col, weight_t)
        t1 = time.perf_counter()
        gemm_times.append((t1 - t0) * 1000)
    gemm_median = np.median(gemm_times)
    print(f"  GEMM (matmul) median: {gemm_median:.2f} ms")

    # ---------- Benchmark full Conv2d for reference ----------
    conv = fastnn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    # Use the same weight for fair comparison
    conv.set_weight(weight)
    # Ensure no gradient tracking on conv weight
    conv_w = conv.parameters()[0]
    conv_w.requires_grad_(False)

    # Warmup
    for _ in range(warmup):
        _ = conv(x)

    # Timed runs
    conv_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = conv(x)
        t1 = time.perf_counter()
        conv_times.append((t1 - t0) * 1000)
    conv_median = np.median(conv_times)
    print(f"  Full Conv2d median: {conv_median:.2f} ms")

    # ---------- Analysis ----------
    total_separate = im2col_median + gemm_median
    print(f"  Sum of separate steps: {total_separate:.2f} ms")
    overhead_ratio = conv_median / total_separate if total_separate > 0 else 0.0
    print(f"  Conv2d overhead vs separate: {overhead_ratio:.2f}x")

    # Determine bottleneck
    if im2col_median > gemm_median:
        bottleneck = "im2col extraction"
        bottleneck_time = im2col_median
        other_time = gemm_median
    else:
        bottleneck = "GEMM"
        bottleneck_time = gemm_median
        other_time = im2col_median

    ratio = bottleneck_time / other_time if other_time > 0 else float("inf")
    print(
        f"  Bottleneck: {bottleneck} ({bottleneck_time:.2f} ms vs {other_time:.2f} ms, {ratio:.1f}x)"
    )


def main():
    scenarios = [
        (1, 32, 64, 64, 64, 3, 1, 1),   # Small
        (1, 64, 128, 32, 32, 3, 1, 1),  # Medium
        (1, 128, 256, 16, 16, 3, 1, 1), # Large channels
        (2, 64, 128, 64, 64, 3, 1, 1),  # Batch 2
        (1, 64, 64, 64, 64, 1, 1, 0),   # 1x1 conv
    ]

    print("=" * 60)
    print("FastNN Im2col + GEMM Breakdown Benchmark")
    print("=" * 60)

    for batch, in_ch, out_ch, h, w, k, s, p in scenarios:
        benchmark_im2col_and_gemm(batch, in_ch, out_ch, h, w, k, s, p)
        print()

    print("Analysis:")
    print(
        "- im2col extraction: transforms input into column matrix (memory-bound)"
    )
    print(
        "- GEMM: matrix multiplication of column matrix with weight (compute-bound)"
    )
    print(
        "- If im2col is the bottleneck, optimizations like vectorized loads or parallelization help."
    )
    print(
        "- If GEMM is the bottleneck, using optimized BLAS or higher compute throughput helps."
    )


if __name__ == "__main__":
    main()
