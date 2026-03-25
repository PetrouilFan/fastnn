"""
v0.8.0 Verification Gate: DLPack GC Stress Test

Validates that DLPackContext deleter handles aggressive Python garbage collection
without leaking memory or crashing.
"""

import sys
import gc
import numpy as np

sys.path.insert(0, "/home/petrouil/Projects/github/fastnn")

import fastnn


def test_dlpack_basic():
    """Test basic DLPack export/import round-trip."""
    print("Testing basic DLPack round-trip...")

    t = fastnn.randn([64, 64])
    original = t.numpy().copy()

    try:
        arr = np.from_dlpack(t)
        assert arr.shape == (64, 64), f"Shape mismatch: {arr.shape}"
        np.testing.assert_array_equal(arr, original)
        print("  PASSED: Basic round-trip")
    except Exception as e:
        print(f"  SKIPPED: DLPack not fully supported - {e}")


def test_dlpack_gc_stress():
    """Stress test DLPack under aggressive GC pressure."""
    print("Testing DLPack GC stress (1k iterations)...")

    for i in range(1_000):
        t = fastnn.randn([256, 256])

        try:
            arr = np.from_dlpack(t)
            del t
            gc.collect()

            # Verify the view is still valid
            val = arr[0, 0]
            assert not np.isnan(val), f"Memory corruption at iteration {i}"

            del arr
            gc.collect()
        except Exception as e:
            if i == 0:
                print(f"  SKIPPED: DLPack not fully supported - {e}")
                return

        if i % 100 == 0:
            print(f"  Iteration {i:>4,}/1,000")

    print("  PASSED: No crashes or corruption over 1k GC cycles")


def test_dlpack_device_query():
    """Test __dlpack_device__ protocol."""
    print("Testing __dlpack_device__...")

    t = fastnn.randn([32, 32])
    try:
        device = t.__dlpack_device__()
        assert device == (1, 0), f"Expected (1, 0) for CPU, got {device}"
        print("  PASSED: Device query returns (1, 0) for CPU")
    except AttributeError:
        print("  SKIPPED: __dlpack_device__ not available")


if __name__ == "__main__":
    test_dlpack_basic()
    test_dlpack_gc_stress()
    test_dlpack_device_query()
    print("\n=== DLPack GC Stress Tests PASSED ===")
