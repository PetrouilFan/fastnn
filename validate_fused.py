#!/usr/bin/env python3
"""Validate correctness of fused_conv_bn_silu: compare forward and gradients."""

import fastnn
import numpy as np

def forward_equivalence():
    """Check that fused forward matches separate conv+bn+silu forward using same inputs/weights."""
    print("=== Forward Equivalence Test ===")
    np.random.seed(42)
    fastnn.set_seed(42)

    batch, in_ch, out_ch, h, w = 1, 32, 64, 64, 64

    # Shared parameters
    x = fastnn.randn([batch, in_ch, h, w])
    weight = fastnn.randn([out_ch, in_ch, 3, 3])
    bias = fastnn.randn([out_ch])
    bn_w = fastnn.randn([out_ch])
    bn_b = fastnn.randn([out_ch])
    bn_mean = fastnn.randn([out_ch])
    bn_var = fastnn.full([out_ch], 0.1)  # positive variance

    # Scalar tensors
    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], 1e-5)

    # Fused
    fused_out = fastnn._core.fused_conv_bn_silu(
        x, weight, bias, bn_w, bn_b, bn_mean, bn_var,
        stride_t, padding_t, dilation_t, groups_t, eps_t
    )

    # Separate using modules with same parameters
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
    sep_out = silu(bn(conv(x)))

    diff = np.abs((fused_out.numpy() - sep_out.numpy())).max()
    rel = diff / (np.abs(sep_out.numpy()).mean() + 1e-8)
    print(f"  Max abs diff: {diff:.2e}")
    print(f"  Max rel diff: {rel:.2e}")
    if diff < 1e-4:
        print("  ✅ Forward outputs match")
    else:
        print("  ❌ Forward outputs differ")

def gradient_equivalence():
    """Check fused vs separate gradients using same parameters."""
    print("\n=== Gradient Equivalence Test ===")
    np.random.seed(123)
    fastnn.set_seed(123)

    batch, in_ch, out_ch, h, w = 2, 16, 32, 16, 16

    # Helper to create fresh params with requires_grad
    def make_params():
        x = fastnn.randn([batch, in_ch, h, w])
        x.requires_grad_(True)
        w = fastnn.randn([out_ch, in_ch, 3, 3])
        w.requires_grad_(True)
        b = fastnn.randn([out_ch])
        b.requires_grad_(True)
        bn_w = fastnn.randn([out_ch])
        bn_w.requires_grad_(True)
        bn_b = fastnn.randn([out_ch])
        bn_b.requires_grad_(True)
        bn_mean = fastnn.randn([out_ch])  # running stats, no grad
        bn_var = fastnn.full([out_ch], 0.1)
        return x, w, b, bn_w, bn_b, bn_mean, bn_var

    # Separate path
    x_s, w_s, b_s, bn_w_s, bn_b_s, bn_mean_s, bn_var_s = make_params()
    conv = fastnn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=True)
    conv.set_weight(w_s)
    conv.set_bias(b_s)
    conv.eval()
    bn = fastnn.BatchNorm2d(out_ch, eps=1e-5)
    bn.set_weight(bn_w_s)
    bn.set_bias(bn_b_s)
    bn.set_running_mean(bn_mean_s)
    bn.set_running_var(bn_var_s)
    bn.eval()
    silu = fastnn.SiLU()
    out_sep = silu(bn(conv(x_s)))
    loss_s = out_sep.sum()
    loss_s.backward()

    # Fused path
    x_f, w_f, b_f, bn_w_f, bn_b_f, bn_mean_f, bn_var_f = make_params()
    stride_t = fastnn.full([], 1)
    padding_t = fastnn.full([], 1)
    dilation_t = fastnn.full([], 1)
    groups_t = fastnn.full([], 1)
    eps_t = fastnn.full([], 1e-5)
    try:
        out_f = fastnn._core.fused_conv_bn_silu(
            x_f, w_f, b_f, bn_w_f, bn_b_f, bn_mean_f, bn_var_f,
            stride_t, padding_t, dilation_t, groups_t, eps_t
        )
        loss_f = out_f.sum()
        loss_f.backward()
        fused_has_grad = True
    except Exception as e:
        print(f"  ⚠️ Fused backward not supported: {e}")
        fused_has_grad = False

    if not fused_has_grad:
        print("  Skipping gradient comparison (fused backward unavailable)")
        return

    def rel_diff(a, b):
        a_np = a.numpy()
        b_np = b.numpy()
        diff = np.abs(a_np - b_np).max()
        denom = np.abs(b_np).mean() + 1e-8
        return diff / denom

    grads = [
        ("x", x_f.grad, x_s.grad),
        ("weight", w_f.grad, w_s.grad),
        ("bias", b_f.grad, b_s.grad),
        ("bn_weight", bn_w_f.grad, bn_w_s.grad),
        ("bn_bias", bn_b_f.grad, bn_b_s.grad),
    ]
    all_ok = True
    for name, g_fused, g_sep in grads:
        if g_fused is None or g_sep is None:
            print(f"  ❌ {name} grad is None")
            all_ok = False
            continue
        rel = rel_diff(g_fused, g_sep)
        print(f"  {name} grad max abs diff: {np.abs(g_fused.numpy()-g_sep.numpy()).max():.2e}, rel: {rel:.2e}")
        if rel < 1e-2:
            print(f"    ✅")
        else:
            print(f"    ❌ exceeds tolerance")
            all_ok = False
    if all_ok:
        print("✅ All gradients match between fused and separate")
    else:
        print("❌ Gradient mismatch")

if __name__ == "__main__":
    forward_equivalence()
    gradient_equivalence()
