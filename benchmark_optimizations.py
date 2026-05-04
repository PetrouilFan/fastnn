#!/usr/bin/env python3
"""Benchmark various FastNN optimizations."""

import time
import fastnn as nn
import numpy as np

def benchmark(name, fn, iters=50, warmup=5):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    median = np.median(times)
    print(f"{name:40s}: {median:7.2f} ms")
    return median

print("=" * 70)
print("FastNN Optimization Benchmarks")
print("=" * 70)

# ============================================================
# 1. BatchNorm2d kernel dispatch
# ============================================================
print("\n--- BatchNorm2d Kernel Dispatch ---")
for batch, ch, h, w in [(1, 64, 32, 32), (1, 256, 16, 16), (4, 512, 8, 8)]:
    print(f"\nConfig: {batch}x{ch}x{h}x{w}")
    x = nn.randn([batch, ch, h, w])
    
    def bn_train():
        bn = nn.BatchNorm2d(ch)
        bn.train()
        return bn(x)
    
    def bn_eval():
        bn = nn.BatchNorm2d(ch)
        bn.eval()
        bn(x)  # running stats
        return bn(x)
    
    benchmark("BatchNorm2d train", bn_train)
    benchmark("BatchNorm2d eval", bn_eval)

# ============================================================
# 2. Elementwise operations with SIMD
# ============================================================
print("\n--- Elementwise SIMD ---")
for shape in [[1, 512, 256], [1, 1024, 512], [4, 2048, 1024]]:
    print(f"\nConfig: {shape}")
    x = nn.randn(shape)
    y = nn.randn(shape)
    
    def add_op():
        return x + y
    
    def mul_op():
        return x * y
    
    def relu_op():
        return nn.relu(x)
    
    benchmark("Add (elementwise)", add_op)
    benchmark("Mul (elementwise)", mul_op)
    benchmark("ReLU (activation)", relu_op)

# ============================================================
# 3. RMSNorm kernel
# ============================================================
print("\n--- RMSNorm Kernel ---")
for shape in [[1, 256, 512], [1, 512, 1024], [4, 1024, 2048]]:
    print(f"\nConfig: {shape}")
    x = nn.randn(shape)
    
    def rmsnorm():
        rms = nn.RMSNorm(shape[-1])
        rms.eval()
        return rms(x)
    
    benchmark("RMSNorm", rmsnorm)

# ============================================================
# 4. Conv2d kernels (various sizes)
# ============================================================
print("\n--- Conv2d Kernels ---")
configs = [
    (1, 32, 64, 64, 64),
    (1, 64, 128, 32, 32),
    (1, 128, 256, 16, 16),
]
for batch, in_ch, out_ch, h, w in configs:
    print(f"\nConfig: {batch}x{in_ch}x{h}x{w} -> {batch}x{out_ch}x{h}x{w}")
    x = nn.randn([batch, in_ch, h, w])
    
    def conv():
        c = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        return c(x)
    
    benchmark("Conv2d 3x3", conv)

# ============================================================
# 5. Training backward pass (workspace reuse)
# ============================================================
print("\n--- Training Backward Pass ---")

# Simplified - just forward pass shows kernel performance
for shape in [[1, 64, 32, 32], [1, 128, 16, 16]]:
    print(f"\nConfig: {shape}")
    x = nn.randn(shape)
    conv = nn.Conv2d(shape[1], 128, 3, padding=1)
    
    def forward():
        return conv(x)
    
    benchmark("Conv2d forward", forward, iters=50)

# ============================================================
# 6. Broadcasting operations (tensor iterator fix)
# ============================================================
print("\n--- Broadcasting (TensorIterator) ---")
for shape in [[1024, 1], [512, 512], [128, 128, 128]]:
    print(f"\nConfig: {shape}")
    x = nn.randn(shape)
    y = nn.randn([shape[0], 1] if len(shape) == 2 else [1] + shape[1:])
    
    def broadcast_add():
        return x + y
    
    benchmark("Broadcast add", broadcast_add)

print("\n" + "=" * 70)
print("Benchmarks complete!")