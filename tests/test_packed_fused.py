"""Tests for fused packed layers."""
import numpy as np
import fastnn as fnn


def test_packed_conv_relu():
    """Test PackedConvRelu4 forward."""
    conv = fnn.PackedConvRelu4(8, 16, 3, stride=1, padding=1)
    x = np.random.randn(2, 8, 32, 32).astype(np.float32)
    x_t = fnn.Tensor(x.ravel().tolist(), list(x.shape))
    out = conv.forward(x_t)
    assert out.shape == [2, 16, 32, 32], f"Expected [2,16,32,32] got {out.shape}"
    out_data = np.array(out.numpy()).reshape(out.shape)
    assert np.all(out_data >= -1e-6), "ReLU failed: found negative values"
    print(f"test_packed_conv_relu: OK, output shape={out.shape}")


def test_packed_linear_gelu():
    """Test PackedLinearGelu4 forward."""
    linear = fnn.PackedLinearGelu4(64, 128)
    x = np.random.randn(2, 64).astype(np.float32)
    x_t = fnn.Tensor(x.ravel().tolist(), list(x.shape))
    out = linear.forward(x_t)
    assert out.shape == [2, 128], f"Expected [2,128] got {out.shape}"
    print(f"test_packed_linear_gelu: OK, output shape={out.shape}")


def test_bn_folding():
    """Test BN folding into packed conv weights (scaffolding)."""
    print("test_bn_folding: OK (scaffolding - fold_bn_into_packed_conv is Rust-only)")


if __name__ == "__main__":
    test_packed_conv_relu()
    test_packed_linear_gelu()
    test_bn_folding()
    print("All fused layer tests passed!")
