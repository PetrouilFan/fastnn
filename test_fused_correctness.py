#!/usr/bin/env python3
"""Correctness test for fused_conv_bn_silu: forward equivalence using same parameters."""

import fastnn
import numpy as np

def test_forward_equivalence_deterministic():
    """Use constant tensors to verify algorithm correctness."""
    print("=== Deterministic Forward Test (ones) ===")
    batch, in_ch, out_ch, h, w = 1, 2, 2, 3, 3
    x = fastnn.ones([batch, in_ch, h, w])
    weight = fastnn.ones([out_ch, in_ch, 3, 3])
    bias = fastnn.zeros([out_ch])
    bn_w = fastnn.ones([out_ch])
    bn_b = fastnn.zeros([out_ch])
    bn_mean = fastnn.zeros([out_ch])
    bn_var = fastnn.full([out_ch], 1.0)

    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], 1e-5)

    fused = fastnn._core.fused_conv_bn_silu(
        x, weight, bias, bn_w, bn_b, bn_mean, bn_var,
        stride_t, padding_t, dilation_t, groups_t, eps_t
    )

    # Separate
    conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
    conv.set_weight(weight)
    conv.eval()
    bn = fastnn.BatchNorm2d(out_ch, eps=1e-5)
    bn.set_weight(bn_w)
    bn.set_bias(bn_b)
    bn.set_running_mean(bn_mean)
    bn.set_running_var(bn_var)
    bn.eval()
    silu = fastnn.SiLU()
    sep = silu(bn(conv(x)))

    diff = np.abs(fused.numpy() - sep.numpy()).max()
    print(f"Max abs diff: {diff:.2e}")
    assert diff < 1e-4, f"Fused forward mismatch: diff={diff}"
    print("✅ Passed")

def test_forward_equivalence_random():
    """Use shared random parameters to verify numerical match."""
    print("\n=== Random Forward Test (shared parameters) ===")
    batch, in_ch, out_ch, h, w = 1, 32, 64, 64, 64
    x = fastnn.randn([batch, in_ch, h, w])
    weight = fastnn.randn([out_ch, in_ch, 3, 3])
    bias = fastnn.randn([out_ch])
    bn_w = fastnn.randn([out_ch])
    bn_b = fastnn.randn([out_ch])
    bn_mean = fastnn.randn([out_ch])
    bn_var = fastnn.full([out_ch], 0.1)  # positive constant

    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], 1e-5)

    fused = fastnn._core.fused_conv_bn_silu(
        x, weight, bias, bn_w, bn_b, bn_mean, bn_var,
        stride_t, padding_t, dilation_t, groups_t, eps_t
    )

    conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
    conv.set_weight(weight)
    conv.set_bias(bias)
    conv.eval()
    bn = fastnn.BatchNorm2d(out_ch, eps=1e-5)
    bn.set_weight(bn_w)
    bn.set_bias(bn_b)
    bn.set_running_mean(bn_mean)
    bn.set_running_var(bn_var)
    bn.eval()
    silu = fastnn.SiLU()
    sep = silu(bn(conv(x)))

    diff = np.abs(fused.numpy() - sep.numpy()).max()
    rel = diff / (np.abs(sep.numpy()).mean() + 1e-8)
    print(f"Max abs diff: {diff:.2e}, rel: {rel:.2e}")
    assert diff < 1e-3, f"Fused forward mismatch: diff={diff}"
    print("✅ Passed")

def test_multiple_shapes():
    """Test fused kernel on various shapes that meet constraints."""
    print("\n=== Multiple Shapes Test ===")
    configs = [
        (1, 8, 16, 16, 16),
        (1, 16, 32, 32, 32),
        (2, 8, 8, 8, 8),
    ]
    for batch, in_ch, out_ch, h, w in configs:
        x = fastnn.randn([batch, in_ch, h, w])
        weight = fastnn.randn([out_ch, in_ch, 3, 3])
        bias = fastnn.randn([out_ch])
        bn_w = fastnn.randn([out_ch])
        bn_b = fastnn.randn([out_ch])
        bn_mean = fastnn.randn([out_ch])
        bn_var = fastnn.full([out_ch], 0.1)

        stride_t = fastnn.full([], 1)
        padding_t = fastnn.full([], 1)
        dilation_t = fastnn.full([], 1)
        groups_t = fastnn.full([], 1)
        eps_t = fastnn.full([], 1e-5)

        fused = fastnn._core.fused_conv_bn_silu(
            x, weight, bias, bn_w, bn_b, bn_mean, bn_var,
            stride_t, padding_t, dilation_t, groups_t, eps_t
        )
        conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
        conv.set_weight(weight)
        conv.set_bias(bias)
        conv.eval()
        bn = fastnn.BatchNorm2d(out_ch, eps=1e-5)
        bn.set_weight(bn_w)
        bn.set_bias(bn_b)
        bn.set_running_mean(bn_mean)
        bn.set_running_var(bn_var)
        bn.eval()
        silu = fastnn.SiLU()
        sep = silu(bn(conv(x)))

        diff = np.abs(fused.numpy() - sep.numpy()).max()
        print(f"  shape {x.shape} -> diff {diff:.2e}")
        assert diff < 1e-3, f"Shape {x.shape} failed: diff={diff}"
    print("✅ All shapes passed")

if __name__ == "__main__":
    test_forward_equivalence_deterministic()
    test_forward_equivalence_random()
    test_multiple_shapes()
    print("\n✅ All forward correctness tests passed!")
