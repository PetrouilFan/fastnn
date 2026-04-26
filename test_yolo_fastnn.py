#!/usr/bin/env python3
"""Test FastNN YOLO11n import and basic functionality."""

import fastnn
import numpy as np

def test_basic_conv():
    """Test that basic Conv2d works."""
    print("Testing basic Conv2d...")
    conv = fastnn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
    x = fastnn.randn([1, 3, 32, 32])
    out = conv(x)
    print(f"Conv2d output shape: {out.shape}")
    return True

def test_import():
    """Test ONNX import."""
    print("Testing ONNX import...")
    try:
        model, info = fastnn.import_onnx('yolo11n.onnx', 'yolo11n_test.fnn')
        print(f"Imported model with {len(info['layers'])} layers")
        print(f"Parameters: {info['parameters']}")

        # Try to run with very small input
        x = fastnn.randn([1, 3, 64, 64])
        print("Running forward pass with small input...")
        result = model(x)
        print(f"Output shape: {result.shape}")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_fused_kernel():
    """Test if fused kernel is available."""
    print("Testing fused kernel availability...")
    try:
        # Check if the fused kernel is registered
        ops = fastnn.list_registered_ops()
        has_fused = 'fused_conv_bn_silu' in ops
        print(f"Fused kernel registered: {has_fused}")
        return has_fused
    except Exception as e:
        print(f"Could not check registered ops: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("FastNN YOLO11n Integration Test")
    print("=" * 50)

    test_basic_conv()
    test_fused_kernel()
    test_import()

    print("=" * 50)
    print("Test completed")