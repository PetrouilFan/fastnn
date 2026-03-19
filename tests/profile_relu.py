#!/usr/bin/env python3
"""Profile fastnn ReLU performance on large tensors."""

import time
import torch
import numpy as np
import fastnn as fnn


def profile_relu():
    """Profile ReLU performance characteristics."""

    print("=" * 80)
    print("ReLU Performance Profiling - Large Tensors")
    print("=" * 80)

    # Test different tensor sizes
    sizes = [
        (100, 100),  # Small
        (500, 500),  # Medium
        (1000, 1000),  # Large
        (2000, 2000),  # Very large
    ]

    results = []

    for size in sizes:
        print(f"\nTensor size: {size}")
        print("-" * 40)

        numel = size[0] * size[1]
        nbytes = numel * 4  # f32 = 4 bytes

        # Create tensors
        x_fnn = fnn.randn(size)
        x_torch = torch.randn(size)

        # Warmup
        for _ in range(10):
            _ = fnn.relu(x_fnn)
            _ = torch.relu(x_torch)

        # Benchmark fastnn
        times_fnn = []
        for _ in range(100):
            start = time.perf_counter()
            _ = fnn.relu(x_fnn)
            end = time.perf_counter()
            times_fnn.append(end - start)

        # Benchmark PyTorch
        times_torch = []
        for _ in range(100):
            start = time.perf_counter()
            _ = torch.relu(x_torch)
            end = time.perf_counter()
            times_torch.append(end - start)

        # Calculate statistics
        mean_fnn = np.mean(times_fnn) * 1e6  # Convert to μs
        mean_torch = np.mean(times_torch) * 1e6
        speedup = mean_torch / mean_fnn

        print(f"fastnn:  {mean_fnn:8.2f} μs (mean)")
        print(f"PyTorch: {mean_torch:8.2f} μs (mean)")
        print(f"Speedup: {speedup:8.2f}x")
        print(f"Memory:  {nbytes / 1024:.1f} KB")

        results.append(
            {
                "size": size,
                "numel": numel,
                "nbytes": nbytes,
                "mean_fnn": mean_fnn,
                "mean_torch": mean_torch,
                "speedup": speedup,
            }
        )

    # Analysis
    print("\n" + "=" * 80)
    print("Performance Analysis")
    print("=" * 80)

    # Calculate memory bandwidth
    print("\nMemory Bandwidth Analysis:")
    print("-" * 40)

    for r in results:
        # ReLU reads input and writes output
        total_bytes = r["nbytes"] * 2  # Read + Write
        bw_fnn = total_bytes / (r["mean_fnn"] * 1e-6) / 1e9  # GB/s
        bw_torch = total_bytes / (r["mean_torch"] * 1e-6) / 1e9

        print(f"Size {r['size']}:")
        print(f"  fastnn:  {bw_fnn:.2f} GB/s")
        print(f"  PyTorch: {bw_torch:.2f} GB/s")
        print(f"  Ratio:   {bw_torch / bw_fnn:.2f}x")

    # Identify scaling issues
    print("\nScaling Analysis:")
    print("-" * 40)

    if len(results) >= 2:
        # Check if performance scales linearly with size
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            size_ratio = curr["numel"] / prev["numel"]
            time_ratio_fnn = curr["mean_fnn"] / prev["mean_fnn"]
            time_ratio_torch = curr["mean_torch"] / prev["mean_torch"]

            print(f"Size {prev['size']} -> {curr['size']}:")
            print(f"  Size increase: {size_ratio:.2f}x")
            print(f"  fastnn time increase:  {time_ratio_fnn:.2f}x")
            print(f"  PyTorch time increase: {time_ratio_torch:.2f}x")

            # Check for overhead
            if time_ratio_fnn > size_ratio * 1.2:
                print(f"  ⚠️  fastnn has {time_ratio_fnn / size_ratio:.2f}x overhead")

    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)

    # Find the largest performance gap
    worst = max(results, key=lambda x: x["mean_fnn"] / x["mean_torch"])
    print(f"\nLargest performance gap at size {worst['size']}:")
    print(
        f"  fastnn is {worst['mean_fnn'] / worst['mean_torch']:.2f}x slower than PyTorch"
    )

    # Check if it's memory-bound or compute-bound
    avg_bw_ratio = np.mean([r["mean_fnn"] / r["mean_torch"] for r in results])
    if avg_bw_ratio > 10:
        print("\n⚠️  Severe memory bandwidth limitation detected.")
        print("   Consider:")
        print("   1. Memory alignment (10-15% improvement)")
        print("   2. More aggressive prefetching")
        print("   3. Non-temporal stores for large arrays")
    elif avg_bw_ratio > 3:
        print("\n⚠️  Moderate performance gap detected.")
        print("   Consider:")
        print("   1. Parallelization overhead reduction")
        print("   2. Cache optimization")
    else:
        print("\n✓ Performance is competitive.")


if __name__ == "__main__":
    profile_relu()
