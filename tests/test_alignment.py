#!/usr/bin/env python3
"""Test memory alignment."""

import torch
import numpy as np
import fastnn as fnn


def test_alignment():
    """Test memory alignment of tensors."""

    print("=" * 80)
    print("Memory Alignment Analysis")
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

        # Create tensors
        x_fnn = fnn.randn(size)
        x_torch = torch.randn(size)

        # Get data pointers
        # Note: This is for analysis only
        # We can't directly get the pointer from fastnn tensors
        # but we can infer alignment from performance

        print("  Testing alignment effects...")

        # Test with different alignment scenarios
        # (We can't directly control alignment, but we can measure performance)

        import time

        # Measure ReLU performance
        times_fnn = []
        for _ in range(100):
            start = time.perf_counter()
            _ = fnn.relu(x_fnn)
            end = time.perf_counter()
            times_fnn.append(end - start)

        times_torch = []
        for _ in range(100):
            start = time.perf_counter()
            _ = torch.relu(x_torch)
            end = time.perf_counter()
            times_torch.append(end - start)

        mean_fnn = np.mean(times_fnn) * 1e6
        mean_torch = np.mean(times_torch) * 1e6

        print(f"  fastnn:  {mean_fnn:8.2f} μs")
        print(f"  PyTorch: {mean_torch:8.2f} μs")
        print(f"  Ratio:   {mean_fnn / mean_torch:.2f}x")

        # Check if performance suggests alignment issues
        if mean_fnn / mean_torch > 5:
            print("  ⚠️  Severe performance gap - likely alignment or cache issue")
        elif mean_fnn / mean_torch > 2:
            print("  ⚠️  Moderate performance gap - possible optimization opportunity")
        else:
            print("  ✓ Performance is reasonable")


if __name__ == "__main__":
    test_alignment()
