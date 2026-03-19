#!/usr/bin/env python3
"""Test memory access patterns."""

import time
import torch
import numpy as np
import fastnn as fnn


def test_memory_access():
    """Test memory access patterns."""

    print("=" * 80)
    print("Memory Access Pattern Analysis")
    print("=" * 80)

    # Test different tensor sizes
    sizes = [
        (100, 100),  # Small
        (500, 500),  # Medium
        (1000, 1000),  # Large
    ]

    for size in sizes:
        print(f"\nTensor size: {size}")
        print("-" * 40)

        numel = size[0] * size[1]
        nbytes = numel * 4  # f32 = 4 bytes

        # Create tensors
        x_fnn = fnn.randn(size)
        x_torch = torch.randn(size)

        print(f"  Elements: {numel:,}")
        print(f"  Memory: {nbytes / 1024:.1f} KB")

        # Test ReLU
        print("  Testing ReLU...")

        # fastnn
        times_fnn = []
        for _ in range(100):
            start = time.perf_counter()
            _ = fnn.relu(x_fnn)
            end = time.perf_counter()
            times_fnn.append(end - start)
        mean_fnn = np.mean(times_fnn) * 1e6

        # PyTorch
        times_torch = []
        for _ in range(100):
            start = time.perf_counter()
            _ = torch.relu(x_torch)
            end = time.perf_counter()
            times_torch.append(end - start)
        mean_torch = np.mean(times_torch) * 1e6

        speedup = mean_torch / mean_fnn
        print(
            f"  ReLU - fastnn: {mean_fnn:8.2f} μs, PyTorch: {mean_torch:8.2f} μs, Speedup: {speedup:.2f}x"
        )

        # Calculate memory bandwidth
        total_bytes = nbytes * 2  # Read + Write
        bw_fnn = total_bytes / (mean_fnn * 1e-6) / 1e9  # GB/s
        bw_torch = total_bytes / (mean_torch * 1e-6) / 1e9

        print(
            f"  Memory Bandwidth - fastnn: {bw_fnn:.2f} GB/s, PyTorch: {bw_torch:.2f} GB/s"
        )


if __name__ == "__main__":
    test_memory_access()
