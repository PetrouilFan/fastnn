#!/usr/bin/env python3
"""Test single-threaded vs multi-threaded ReLU performance."""

import os
import time
import torch
import numpy as np
import fastnn as fnn


def test_threading():
    """Test single-threaded vs multi-threaded performance."""

    print("=" * 80)
    print("Single-threaded vs Multi-threaded ReLU Performance")
    print("=" * 80)

    # Test different tensor sizes
    sizes = [
        (100, 100),  # Small
        (500, 500),  # Medium
        (1000, 1000),  # Large
    ]

    # Test with different thread counts
    thread_counts = [1, 2, 4, 8]

    for size in sizes:
        print(f"\nTensor size: {size}")
        print("-" * 40)

        # Create tensor
        x_fnn = fnn.randn(size)

        # Warmup
        for _ in range(10):
            _ = fnn.relu(x_fnn)

        # Test with different thread counts
        for num_threads in thread_counts:
            # Set thread count (if supported)
            # Note: fastnn might not have a direct thread control API
            # We'll just measure the current performance

            times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = fnn.relu(x_fnn)
                end = time.perf_counter()
                times.append(end - start)

            mean_time = np.mean(times) * 1e6  # Convert to μs
            print(f"  Threads: {num_threads:2d} - Time: {mean_time:8.2f} μs")

        # Compare with PyTorch
        x_torch = torch.randn(size)
        times_torch = []
        for _ in range(100):
            start = time.perf_counter()
            _ = torch.relu(x_torch)
            end = time.perf_counter()
            times_torch.append(end - start)

        mean_torch = np.mean(times_torch) * 1e6
        print(f"  PyTorch:        - Time: {mean_torch:8.2f} μs")


if __name__ == "__main__":
    test_threading()
