#!/usr/bin/env python3
"""Benchmark FastNN fused kernels vs separate operations."""

import time
import fastnn as nn
import numpy as np

def benchmark(name, fn, iters=100, warmup=10):
    """Generic benchmark helper."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    median_time = np.median(times)
    fps = 1000.0 / median_time
    print(f"{name:45s}: {median_time:6.2f} ms  ({fps:6.1f} FPS)")
    return median_time

def main():
    print("=" * 70)
    print("FastNN Fused Kernel Benchmarks")
    print("=" * 70)
    
    configs = [
        (1, 32, 64, 64, 64),   # Small
        (1, 64, 128, 32, 32),  # Medium
        (1, 128, 256, 16, 16), # Large
        (1, 256, 512, 8, 8),   # Extra large
    ]
    
    for batch, in_ch, out_ch, h, w in configs:
        print(f"\nConfig: {batch}x{in_ch}x{h}x{w} -> {batch}x{out_ch}x{h}x{w}")
        print("-" * 70)
        
        # Setup data
        np.random.seed(42)
        w_data = np.random.randn(out_ch, in_ch, 3, 3).flatten().tolist()
        b_data = np.random.randn(out_ch).tolist()
        bn_w = np.random.randn(out_ch).tolist()
        bn_b = np.random.randn(out_ch).tolist()
        bn_mean = np.random.randn(out_ch).tolist()
        bn_var = np.random.rand(out_ch).tolist()
        x_data = np.random.randn(batch, in_ch, h, w).flatten().tolist()
        
        x = nn.tensor(x_data, [batch, in_ch, h, w])
        
        # === SiLU (FusedConvBn) ===
        def separate_silu():
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
            conv.set_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            conv.set_bias(nn.tensor(b_data, [out_ch]))
            
            bn = nn.BatchNorm2d(out_ch)
            bn.set_weight(nn.tensor(bn_w, [out_ch]))
            bn.set_bias(nn.tensor(bn_b, [out_ch]))
            bn.set_running_mean(nn.tensor(bn_mean, [out_ch]))
            bn.set_running_var(nn.tensor(bn_var, [out_ch]))
            bn.eval()
            
            return bn(nn.SiLU()(conv(x)))
        
        def fused_silu():
            fused = nn.FusedConvBn(in_ch, out_ch, 3, padding=1, bias=True)
            fused.set_conv_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            fused.set_conv_bias(nn.tensor(b_data, [out_ch]))
            fused.set_bn_weight(nn.tensor(bn_w, [out_ch]))
            fused.set_bn_bias(nn.tensor(bn_b, [out_ch]))
            fused.set_bn_running_mean(nn.tensor(bn_mean, [out_ch]))
            fused.set_bn_running_var(nn.tensor(bn_var, [out_ch]))
            fused.eval()
            return fused(x)
        
        print("SiLU activation:")
        benchmark("Separate (Conv+BN+SiLU)", separate_silu)
        benchmark("Fused (FusedConvBn)", fused_silu)
        
        # === ReLU ===
        def separate_relu():
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
            conv.set_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            conv.set_bias(nn.tensor(b_data, [out_ch]))
            
            bn = nn.BatchNorm2d(out_ch)
            bn.set_weight(nn.tensor(bn_w, [out_ch]))
            bn.set_bias(nn.tensor(bn_b, [out_ch]))
            bn.set_running_mean(nn.tensor(bn_mean, [out_ch]))
            bn.set_running_var(nn.tensor(bn_var, [out_ch]))
            bn.eval()
            
            return nn.ReLU()(bn(conv(x)))
        
        def fused_relu():
            fused = nn.FusedConvBnRelu(in_ch, out_ch, 3, padding=1, bias=True)
            fused.set_conv_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            fused.set_conv_bias(nn.tensor(b_data, [out_ch]))
            fused.set_bn_weight(nn.tensor(bn_w, [out_ch]))
            fused.set_bn_bias(nn.tensor(bn_b, [out_ch]))
            fused.set_bn_running_mean(nn.tensor(bn_mean, [out_ch]))
            fused.set_bn_running_var(nn.tensor(bn_var, [out_ch]))
            fused.eval()
            return fused(x)
        
        print("\nReLU activation:")
        benchmark("Separate (Conv+BN+ReLU)", separate_relu)
        benchmark("Fused (FusedConvBnRelu)", fused_relu)
        
        # === GELU ===
        def separate_gelu():
            conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True)
            conv.set_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            conv.set_bias(nn.tensor(b_data, [out_ch]))
            
            bn = nn.BatchNorm2d(out_ch)
            bn.set_weight(nn.tensor(bn_w, [out_ch]))
            bn.set_bias(nn.tensor(bn_b, [out_ch]))
            bn.set_running_mean(nn.tensor(bn_mean, [out_ch]))
            bn.set_running_var(nn.tensor(bn_var, [out_ch]))
            bn.eval()
            
            return nn.GELU()(bn(conv(x)))
        
        def fused_gelu():
            fused = nn.FusedConvBnGelu(in_ch, out_ch, 3, padding=1, bias=True)
            fused.set_conv_weight(nn.tensor(w_data, [out_ch, in_ch, 3, 3]))
            fused.set_conv_bias(nn.tensor(b_data, [out_ch]))
            fused.set_bn_weight(nn.tensor(bn_w, [out_ch]))
            fused.set_bn_bias(nn.tensor(bn_b, [out_ch]))
            fused.set_bn_running_mean(nn.tensor(bn_mean, [out_ch]))
            fused.set_bn_running_var(nn.tensor(bn_var, [out_ch]))
            fused.eval()
            return fused(x)
        
        print("\nGELU activation:")
        benchmark("Separate (Conv+BN+GELU)", separate_gelu)
        benchmark("Fused (FusedConvBnGelu)", fused_gelu)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()