#!/usr/bin/env python3
"""Benchmark fastnn.matmul vs torch.matmul for 1024x1024 matrices."""

import time
import fastnn
import torch
import numpy as np

def benchmark_matmul(size=1024, iters=100):
    """Benchmark matmul for square matrices."""
    print(f"Benchmarking matmul for {size}x{size} matrices with {iters} iterations")
    
    # Create random matrices on CPU
    a_torch = torch.randn(size, size, device='cpu')
    b_torch = torch.randn(size, size, device='cpu')
    a_torch.requires_grad_(False)
    b_torch.requires_grad_(False)
    
    a_fastnn = fastnn.randn([size, size])
    b_fastnn = fastnn.randn([size, size])
    a_fastnn.requires_grad_(False)
    b_fastnn.requires_grad_(False)
    
    # Warmup for PyTorch
    with torch.no_grad():
        for _ in range(10):
            _ = torch.matmul(a_torch, b_torch)
    
    # Benchmark PyTorch
    torch_times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = torch.matmul(a_torch, b_torch)
            t1 = time.perf_counter()
            torch_times.append((t1 - t0) * 1000)
    
    torch_median = np.median(torch_times)
    print(f"  PyTorch median time: {torch_median:.2f} ms")
    
    # Warmup for FastNN
    for _ in range(10):
        _ = fastnn.matmul(a_fastnn, b_fastnn)
    
    # Benchmark FastNN
    fastnn_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = fastnn.matmul(a_fastnn, b_fastnn)
        t1 = time.perf_counter()
        fastnn_times.append((t1 - t0) * 1000)
    
    fastnn_median = np.median(fastnn_times)
    print(f"  FastNN median time: {fastnn_median:.2f} ms")
    
    speedup = torch_median / fastnn_median if fastnn_median > 0 else 0.0
    print(f"  Speedup: {speedup:.2f}x")
    
    return torch_median, fastnn_median, speedup

if __name__ == "__main__":
    benchmark_matmul()
