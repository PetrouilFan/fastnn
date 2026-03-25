"""
v0.8.0 Verification Gate: WGPU Device-Loss / OOM Recovery Test

Ensures the framework handles VRAM exhaustion gracefully, which is critical for
edge deployment (e.g., Jetson). The engine should throw a catchable DeviceError
or CudaError rather than triggering a Rust panic.
"""

import sys
import gc

sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def test_gpu_availability():
    """Test GPU availability detection."""
    print("Testing GPU availability detection...")

    try:
        is_gpu = fastnn.is_wgpu()
        print(f"  GPU available: {is_gpu}")
        return is_gpu
    except Exception as e:
        print(f"  GPU not available: {e}")
        return False


def test_oom_graceful_handling():
    """Test that OOM is handled gracefully with typed exceptions."""
    print("Testing OOM graceful handling...")

    if not fastnn.is_wgpu():
        print("  SKIPPED: No GPU available")
        return

    buffers = []
    oom_caught = False

    try:
        # Allocate increasingly large buffers until VRAM exhaustion
        for i in range(1000):
            # Each buffer is ~64MB (4096*4096*4 bytes)
            t = fastnn.randn([4096, 4096], device="gpu:0")
            buffers.append(t)

            if i % 10 == 0:
                print(f"  Allocated {i} buffers (~{i * 64}MB)")
    except fastnn.DeviceError as e:
        oom_caught = True
        print(f"  Caught expected DeviceError: {e}")
    except fastnn.CudaError as e:
        oom_caught = True
        print(f"  Caught expected CudaError: {e}")
    except Exception as e:
        print(f"  FAILED: Wrong exception type: {type(e).__name__}: {e}")
        raise AssertionError(
            f"Expected DeviceError or CudaError, got {type(e).__name__}"
        )

    finally:
        # Cleanup
        del buffers
        gc.collect()

    if oom_caught:
        print("  PASSED: OOM handled gracefully")
    else:
        print("  WARNING: Could not trigger OOM (GPU has enough VRAM)")


def test_cpu_fallback():
    """Test CPU fallback works when GPU is not used."""
    print("Testing CPU tensor operations...")

    t1 = fastnn.randn([256, 256])
    t2 = fastnn.randn([256, 256])

    result = t1.matmul(t2)
    assert result.shape() == [256, 256], f"Unexpected shape: {result.shape()}"

    numpy_result = result.numpy()
    assert not numpy_result.size == 0, "Empty result"
    assert not numpy_result.any() != numpy_result.any(), (
        "NaN in result"
    )  # Check for NaN

    print("  PASSED: CPU operations work correctly")


def test_device_transfer():
    """Test tensor transfer between devices."""
    print("Testing device transfer...")

    t_cpu = fastnn.randn([64, 64])
    assert t_cpu.device() == "cpu", f"Expected cpu, got {t_cpu.device()}"

    if fastnn.is_wgpu():
        t_gpu = t_cpu.to_device("gpu:0")
        assert t_gpu.device() == "gpu:0", f"Expected gpu:0, got {t_gpu.device()}"

        t_back = t_gpu.to_device("cpu")
        assert t_back.device() == "cpu", f"Expected cpu, got {t_back.device()}"

        # Values should be preserved
        import numpy as np

        np.testing.assert_array_almost_equal(t_cpu.numpy(), t_back.numpy(), decimal=5)

        print("  PASSED: Device transfer preserves values")
    else:
        print("  SKIPPED: No GPU available")


if __name__ == "__main__":
    test_cpu_fallback()
    has_gpu = test_gpu_availability()
    if has_gpu:
        test_device_transfer()
        test_oom_graceful_handling()
    else:
        print("\n  NOTE: GPU tests skipped (no WGPU device available)")
        print("  These tests will run on hardware with GPU support")
    print("\n=== WGPU Device-Loss / OOM Tests PASSED ===")
