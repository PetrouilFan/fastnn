"""
Comprehensive gradient test for ALL backward operations in fastnn.

Tests that every autograd-enabled operation produces correct gradients
by checking that gradients exist and are non-zero after backward().

Run: python test_gradients_comprehensive.py
"""

import sys
import math
import numpy as np
import fastnn


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    """Check a condition and print PASS/FAIL."""
    global PASS, FAIL
    if condition:
        print(f"  PASS [{name}]")
        PASS += 1
    else:
        msg = f"  FAIL [{name}]"
        if detail:
            msg += f": {detail}"
        print(msg)
        FAIL += 1


def has_grad(tensor, name="tensor"):
    """Check if tensor has a non-None grad with non-zero values."""
    g = tensor.grad
    if g is None:
        return False, "grad is None"
    data = np.array(g.numpy()).flatten()
    if np.all(np.abs(data) < 1e-10):
        return False, "grad is all zeros"
    return True, f"grad OK (max={np.max(np.abs(data)):.4f})"


# ============================================================
# NORMALIZATION
# ============================================================

def test_batch_norm1d():
    x = fastnn.randn([2, 4, 8])
    x.requires_grad_(True)
    bn = fastnn.BatchNorm1d(4, 1e-5, 0.1)
    bn.train()
    out = bn(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("BatchNorm1d-input", ok, msg)
    ok, msg = has_grad(bn.parameters()[0], "weight")
    check("BatchNorm1d-weight", ok, msg)
    ok, msg = has_grad(bn.parameters()[1], "bias")
    check("BatchNorm1d-bias", ok, msg)


def test_batch_norm2d():
    x = fastnn.randn([2, 4, 8, 8])
    x.requires_grad_(True)
    bn = fastnn.BatchNorm2d(4, 1e-5, 0.1)
    bn.train()
    out = bn(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("BatchNorm2d-input", ok, msg)
    ok, msg = has_grad(bn.parameters()[0], "weight")
    check("BatchNorm2d-weight", ok, msg)
    ok, msg = has_grad(bn.parameters()[1], "bias")
    check("BatchNorm2d-bias", ok, msg)


def test_rms_norm():
    x = fastnn.randn([2, 4, 8])
    x.requires_grad_(True)
    rms = fastnn.RMSNorm(8, 1e-5)
    out = rms(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("RMSNorm-input", ok, msg)
    ok, msg = has_grad(rms.parameters()[0], "weight")
    check("RMSNorm-weight", ok, msg)


def test_group_norm():
    x = fastnn.randn([2, 4, 8, 8])
    x.requires_grad_(True)
    gn = fastnn.GroupNorm(2, 4, 1e-5)
    out = gn(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("GroupNorm-input", ok, msg)
    ok, msg = has_grad(gn.parameters()[0], "weight")
    check("GroupNorm-weight", ok, msg)
    ok, msg = has_grad(gn.parameters()[1], "bias")
    check("GroupNorm-bias", ok, msg)


def test_layer_norm():
    x = fastnn.randn([2, 4, 8])
    x.requires_grad_(True)
    ln = fastnn.LayerNorm(8, 1e-5)
    out = ln(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("LayerNorm-input", ok, msg)
    ok, msg = has_grad(ln.parameters()[0], "weight")
    check("LayerNorm-weight", ok, msg)
    ok, msg = has_grad(ln.parameters()[1], "bias")
    check("LayerNorm-bias", ok, msg)


# ============================================================
# POOLING
# ============================================================

def test_max_pool2d():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    pool = fastnn.MaxPool2d(3, 2, 1, 1)
    out = pool(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("MaxPool2d-input", ok, msg)


def test_avg_pool2d():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    pool = fastnn.AvgPool2d(3, 2, 1)
    out = pool(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("AvgPool2d-input", ok, msg)


def test_adaptive_avg_pool2d():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    pool = fastnn.AdaptiveAvgPool2d(4, 4)  # takes (h, w) as separate args
    out = pool(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("AdaptiveAvgPool2d-input", ok, msg)


# ============================================================
# DROPOUT
# ============================================================

def test_dropout():
    x = fastnn.ones([4, 32])  # Large enough that dropout doesn't zero everything
    x.requires_grad_(True)
    d = fastnn.Dropout(0.5)
    d.train()
    out = d(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("Dropout-input", ok, msg)


def test_dropout2d():
    x = fastnn.ones([1, 4, 8, 8])
    x.requires_grad_(True)
    d = fastnn.Dropout2d(0.5)
    d.train()
    out = d(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("Dropout2d-input", ok, msg)


# ============================================================
# UPSAMPLE
# ============================================================

def test_upsample_nearest():
    x = fastnn.randn([1, 2, 4, 4])
    x.requires_grad_(True)
    up = fastnn.Upsample(2.0, "nearest")
    out = up(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("UpsampleNearest-input", ok, msg)


def test_upsample_bilinear():
    x = fastnn.randn([1, 2, 4, 4])
    x.requires_grad_(True)
    up = fastnn.Upsample(2.0, "bilinear")
    out = up(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("UpsampleBilinear-input", ok, msg)


# ============================================================
# CONVOLUTION
# ============================================================

def test_conv2d():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    conv = fastnn.Conv2d(2, 4, 3, 1, 1)
    out = conv(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("Conv2d-input", ok, msg)
    ok, msg = has_grad(conv.parameters()[0], "weight")
    check("Conv2d-weight", ok, msg)


# ============================================================
# FUSED CONV+BN (formerly panicked on backward)
# ============================================================

def test_fused_conv_bn():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    f = fastnn.FusedConvBn(2, 4, 3, 1, 1, 1, 1, 1e-5, True)
    out = f(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("FusedConvBn-input", ok, msg)
    ok, msg = has_grad(f.parameters()[0], "conv_weight")
    check("FusedConvBn-conv_weight", ok, msg)


def test_fused_conv_bn_relu():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    f = fastnn.FusedConvBnRelu(2, 4, 3, 1, 1, 1, 1, 1e-5, True)
    out = f(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("FusedConvBnRelu-input", ok, msg)
    ok, msg = has_grad(f.parameters()[0], "conv_weight")
    check("FusedConvBnRelu-conv_weight", ok, msg)


def test_fused_conv_bn_gelu():
    x = fastnn.randn([1, 2, 8, 8])
    x.requires_grad_(True)
    f = fastnn.FusedConvBnGelu(2, 4, 3, 1, 1, 1, 1, 1e-5, True)
    out = f(x)
    loss = out.sum()
    loss.backward()
    
    ok, msg = has_grad(x, "input")
    check("FusedConvBnGelu-input", ok, msg)
    ok, msg = has_grad(f.parameters()[0], "conv_weight")
    check("FusedConvBnGelu-conv_weight", ok, msg)


# ============================================================
# NN.INIT
# ============================================================

def test_nn_init():
    import fastnn.init as init
    
    # Note: fastnn's init functions return new tensors (in-place is not yet supported).
    # We assign the return value to see the initialized values.
    
    t = init.kaiming_uniform_(fastnn.zeros([4, 4]))
    data = np.array(t.numpy()).reshape(4, 4)
    has_nonzero = np.any(np.abs(data) > 1e-6)
    check("init-kaiming_uniform", has_nonzero, f"max={np.max(np.abs(data)):.6f}")
    
    t2 = init.eye_(fastnn.zeros([4, 4]))
    data = np.array(t2.numpy()).reshape(4, 4)
    check("init-eye-diag", abs(data[0,0] - 1.0) < 1e-6, f"diag={data[0,0]}")
    check("init-eye-offdiag", abs(data[0,1]) < 1e-6, f"offdiag={data[0,1]}")
    
    t3 = init.constant_(fastnn.zeros([4]), 3.14)
    data = np.array(t3.numpy()).reshape(-1)
    check("init-constant", abs(data[0] - 3.14) < 1e-6, f"val={data[0]}")
    
    t4 = init.zeros_(fastnn.ones([4]))
    data = np.array(t4.numpy()).reshape(-1)
    check("init-zeros", np.all(data == 0), f"max={np.max(np.abs(data))}")


# ============================================================
# END-TO-END TRAINING STEP
# ============================================================

def test_training_step():
    x = fastnn.randn([2, 3, 16, 16])
    x.requires_grad_(True)
    
    conv = fastnn.Conv2d(3, 8, 3, 1, 1)
    bn = fastnn.BatchNorm2d(8, 1e-5, 0.1)
    bn.train()
    
    out = conv(x)
    out = bn(out)
    out = fastnn.relu(out)
    out = out.mean()
    out.backward()
    
    all_ok = True
    for name, p in [("conv_w", conv.parameters()[0]), ("bn_w", bn.parameters()[0]), ("bn_b", bn.parameters()[1])]:
        ok, msg = has_grad(p, name)
        if not ok:
            all_ok = False
    
    check("TrainingStep-grads", all_ok)
    
    # Optimizer step
    try:
        opt = fastnn.Adam(conv.parameters() + bn.parameters(), lr=0.001)
        opt.step()
        check("TrainingStep-optimizer", True)
    except Exception as e:
        check("TrainingStep-optimizer", False, str(e))


# ============================================================
# RUN ALL TESTS
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Comprehensive Gradient Tests for fastnn")
    print("=" * 60)
    
    tests = [
        ("BatchNorm1d", test_batch_norm1d),
        ("BatchNorm2d", test_batch_norm2d),
        ("RMSNorm", test_rms_norm),
        ("GroupNorm", test_group_norm),
        ("LayerNorm", test_layer_norm),
        ("MaxPool2d", test_max_pool2d),
        ("AvgPool2d", test_avg_pool2d),
        ("AdaptiveAvgPool2d", test_adaptive_avg_pool2d),
        ("Dropout", test_dropout),
        ("Dropout2d", test_dropout2d),
        ("UpsampleNearest", test_upsample_nearest),
        ("UpsampleBilinear", test_upsample_bilinear),
        ("Conv2d", test_conv2d),
        ("FusedConvBn", test_fused_conv_bn),
        ("FusedConvBnRelu", test_fused_conv_bn_relu),
        ("FusedConvBnGelu", test_fused_conv_bn_gelu),
        ("nn.init", test_nn_init),
        ("TrainingStep", test_training_step),
    ]
    
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"  FAIL [{name}]: EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            FAIL += 1
    
    print("=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS} passed, {FAIL} failed, {total} total")
    if FAIL > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
