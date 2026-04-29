#!/usr/bin/env python3
"""Debug fused_conv_bn_silu by comparing intermediates with separate ops."""

import fastnn
import numpy as np

def debug_fused_vs_separate():
    """Compare each intermediate stage with tiny deterministic tensors."""
    print("=== Debug: Fused vs Separate (deterministic inputs) ===")
    np.random.seed(0)
    fastnn.set_seed(0)

    # Tiny config for easy manual verification
    batch, in_ch, out_ch, h, w = 1, 2, 2, 3, 3  # 3x3 input, 3x3 kernel -> 3x3 output (padding=1)
    # Actually with padding=1, input 3x3 -> output 3x3

    # Use simple deterministic values: all ones for input, weight=1, bias=0, bn params simple
    x_val = np.ones((batch, in_ch, h, w), dtype=np.float32)
    x = fastnn.tensor(x_val.flatten().tolist(), list(x_val.shape))

    # Weight: ones
    w_val = np.ones((out_ch, in_ch, 3, 3), dtype=np.float32)
    weight = fastnn.tensor(w_val.flatten().tolist(), list(w_val.shape))

    # Bias: zeros (optional)
    bias = fastnn.zeros([out_ch])

    # BN params: weight=1, bias=0, mean=0, var=1 (so BN is y = x * 1 + 0 = x)
    bn_w = fastnn.ones([out_ch])
    bn_b = fastnn.zeros([out_ch])
    bn_mean = fastnn.zeros([out_ch])
    bn_var = fastnn.full([out_ch], 1.0)  # var=1, eps=1e-5 -> inv_std ~ 1
    eps_val = 1e-5

    # Scalar params
    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], eps_val)

    # Fused
    fused = fastnn._core.fused_conv_bn_silu(
        x, weight, bias, bn_w, bn_b, bn_mean, bn_var,
        stride_t, padding_t, dilation_t, groups_t, eps_t
    )
    fused_np = fused.numpy()
    print(f"Fused output shape: {fused_np.shape}")
    print(f"Fused output (first few): {fused_np.flat[:10]}")

    # Separate
    conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
    conv.set_weight(weight)
    conv.eval()
    bn = fastnn.BatchNorm2d(out_ch, eps=eps_val)
    bn.set_weight(bn_w)
    bn.set_bias(bn_b)
    bn.set_running_mean(bn_mean)
    bn.set_running_var(bn_var)
    bn.eval()
    silu = fastnn.SiLU()

    sep1 = conv(x)
    sep2 = bn(sep1)
    sep3 = silu(sep2)
    sep_np = sep3.numpy()
    print(f"Separate output (first few): {sep_np.flat[:10]}")

    # Also print conv output (before BN+SiLU)
    conv_np = sep1.numpy()
    print(f"Conv output (first few): {conv_np.flat[:10]}")

    # Compare
    diff = np.abs(fused_np - sep_np).max()
    print(f"\nMax abs diff (fused vs separate): {diff:.2e}")
    if diff < 1e-3:
        print("✅ Forward match")
    else:
        print("❌ Mismatch")

    # Additional check: if BN is identity (weight=1, bias=0, mean=0, var=1) and SiLU is non-linear,
    # then fused should equal silu(conv). Because BN: (x - 0)/sqrt(1+eps) * 1 + 0 ≈ x
    # For small eps, inv_std ≈ 1. So sep ≈ silu(conv).
    silu_of_conv_np = 1 / (1 + np.exp(-conv_np))
    diff2 = np.abs(fused_np - silu_of_conv_np).max()
    print(f"Max abs diff (fused vs silu(conv)): {diff2:.2e}")

if __name__ == "__main__":
    debug_fused_vs_separate()
