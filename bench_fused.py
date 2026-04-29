#!/usr/bin/env python3
"""Benchmark fused_conv_bn_silu vs separate Conv2d+BatchNorm2d+SiLU."""

import time
import fastnn
import numpy as np

def benchmark_fused_vs_separate(batch_size=1, in_ch=32, out_ch=64, h=64, w=64, iters=20):
    print(f"Benchmark fused conv+bn+silu ({in_ch}->{out_ch}, {h}x{w})")

    # Create input
    x = fastnn.randn([batch_size, in_ch, h, w])

    # Create parameters for fused path (random)
    weight = fastnn.randn([out_ch, in_ch, 3, 3])
    bias = fastnn.randn([out_ch])
    bn_w = fastnn.randn([out_ch])
    bn_b = fastnn.randn([out_ch])
    bn_mean = fastnn.randn([out_ch])
    bn_var = fastnn.add(fastnn.rand([out_ch]), fastnn.full([out_ch], 0.1))  # positive variance

    # Create scalar tensors for the remaining params
    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], 1e-5)

    # Warmup fused
    for _ in range(5):
        _ = fastnn._core.fused_conv_bn_silu(x, weight, bias, bn_w, bn_b, bn_mean, bn_var, stride_t, padding_t, dilation_t, groups_t, eps_t)

    # Benchmark fused
    fused_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out_fused = fastnn._core.fused_conv_bn_silu(x, weight, bias, bn_w, bn_b, bn_mean, bn_var, stride_t, padding_t, dilation_t, groups_t, eps_t)
        t1 = time.perf_counter()
        fused_times.append((t1 - t0) * 1000)
    fused_med = np.median(fused_times)
    print(f"  Fused median: {fused_med:.2f} ms")

    # Separate ops: Conv2d + BatchNorm2d + SiLU
    conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
    bn = fastnn.BatchNorm2d(out_ch, eps=1e-5)
    silu = fastnn.SiLU()
    # Use eval mode to use running stats; still tracks? For timing we don't want training overhead
    conv.eval()
    bn.eval()

    # Warmup separate
    for _ in range(5):
        out = conv(x)
        out = bn(out)
        out = silu(out)

    # Benchmark separate
    sep_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = conv(x)
        out = bn(out)
        out = silu(out)
        t1 = time.perf_counter()
        sep_times.append((t1 - t0) * 1000)
    sep_med = np.median(sep_times)
    print(f"  Separate median: {sep_med:.2f} ms")

    speedup = sep_med / fused_med if fused_med > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    return fused_med, sep_med, speedup

if __name__ == "__main__":
    benchmark_fused_vs_separate()
