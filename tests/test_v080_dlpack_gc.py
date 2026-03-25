"""
v0.8.0 Verification Gate: DLPack GC Stress Test

Validates that DLPackContext deleter handles aggressive Python garbage collection
without leaking memory or crashing. The Arc<Storage> must stay alive while NumPy
holds the capsule, and drop correctly when both are released.
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

    arr = np.from_dlpack(t)
    assert arr.shape == (64, 64), f"Shape mismatch: {arr.shape}"
    np.testing.assert_array_equal(arr, original)

    print("  PASSED: Basic round-trip")


def test_dlpack_gc_stress():
    """Stress test DLPack under aggressive GC pressure."""
    print("Testing DLPack GC stress (10k iterations)...")

    for i in range(10_000):
        t = fastnn.randn([1024, 1024])
        arr = np.from_dlpack(t)

        # Deliberately drop the Rust tensor while NumPy still holds the view
        # The DLPackContext Arc<Storage> must keep memory alive
        del t
        gc.collect()

        # Verify the view is still valid (not freed memory)
        val = arr[0, 0]
        assert not np.isnan(val), f"Memory corruption at iteration {i}"

        del arr
        gc.collect()

        if i % 1000 == 0:
            print(f"  Iteration {i:>5,}/10,000")

    print("  PASSED: No crashes or corruption over 10k GC cycles")


def test_dlpack_multiple_refs():
    """Test multiple DLPack exports from same tensor."""
    print("Testing multiple DLPack refs to same tensor...")

    t = fastnn.randn([128, 128])
    arr1 = np.from_dlpack(t)
    arr2 = np.from_dlpack(t)

    # Both should reference the same underlying memory
    np.testing.assert_array_equal(arr1, arr2)

    del t
    gc.collect()

    # Both views should still be valid
    _ = arr1[0, 0]
    _ = arr2[0, 0]

    del arr1, arr2
    gc.collect()

    print("  PASSED: Multiple refs handled correctly")


def test_dlpack_device_query():
    """Test __dlpack_device__ protocol."""
    print("Testing __dlpack_device__...")

    t = fastnn.randn([32, 32])
    device = t.__dlpack_device__()
    assert device == (1, 0), f"Expected (1, 0) for CPU, got {device}"

    print("  PASSED: Device query returns (1, 0) for CPU")


if __name__ == "__main__":
    test_dlpack_basic()
    test_dlpack_gc_stress()
    test_dlpack_multiple_refs()
    test_dlpack_device_query()
    print("\n=== DLPack GC Stress Tests PASSED ===")
