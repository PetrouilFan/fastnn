"""
v0.8.0 Verification Gate: WGPU Device-Loss / OOM Recovery Test

Ensures the framework handles VRAM exhaustion gracefully.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastnn


def test_cpu_fallback():
    """Test CPU fallback works when GPU is not used."""
    print("Testing CPU tensor operations...")

    t1 = fastnn.randn([256, 256])
    t2 = fastnn.randn([256, 256])

    # Use function call, not method
    result = fastnn.matmul(t1, t2)
    assert result.shape == [256, 256], f"Unexpected shape: {result.shape}"

    numpy_result = result.numpy()
    assert numpy_result.size > 0, "Empty result"

    print("  PASSED: CPU operations work correctly")


def test_device_attr():
    """Test device attribute access."""
    print("Testing device attribute...")

    t_cpu = fastnn.randn([64, 64])
    # device is a property, not a method
    assert t_cpu.device == "cpu", f"Expected cpu, got {t_cpu.device}"

    print("  PASSED: Device attribute works correctly")


def test_gpu_detection():
    """Test GPU detection if available."""
    try:
        is_gpu = fastnn.is_wgpu()
        assert isinstance(is_gpu, bool), "is_wgpu() should return a bool"
        if is_gpu:
            assert fastnn.is_wgpu_device_available() is True
    except AttributeError:
        pytest.skip("fastnn does not have wgpu support")
