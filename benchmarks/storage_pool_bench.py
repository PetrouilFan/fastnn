"""Benchmark storage pool allocation patterns."""
import time
import fastnn as fnn
import gc

def benchmark_tensor_creation(num_iters=1000):
    """Benchmark creating and destroying tensors (pool hit/miss)."""
    # Warmup
    for _ in range(100):
        t = fnn.randn([256, 256])
        del t
    gc.collect()
    
    # Benchmark: create tensors in loop (should hit pool)
    start = time.perf_counter()
    for _ in range(num_iters):
        fnn.randn([256, 256])
    elapsed = time.perf_counter() - start
    return elapsed / num_iters * 1000  # ms per creation

def benchmark_zeros_vs_empty(num_iters=1000):
    """Benchmark zeros (zero-fill) vs empty (no fill)."""
    # Benchmark zeros
    start = time.perf_counter()
    for _ in range(num_iters):
        fnn.zeros([256, 256])
    zeros_time = time.perf_counter() - start
    
    # Benchmark empty (if available, otherwise skip)
    try:
        start = time.perf_counter()
        for _ in range(num_iters):
            fnn.empty([256, 256])
        empty_time = time.perf_counter() - start
        empty_ms = empty_time / num_iters * 1000
    except AttributeError:
        empty_ms = None
    
    return zeros_time / num_iters * 1000, empty_ms

if __name__ == "__main__":
    print("Benchmarking storage pool allocation...")
    create_time = benchmark_tensor_creation()
    print(f"  Tensor creation (pool reuse): {create_time:.3f} ms/creation")
    
    zeros_ms, empty_ms = benchmark_zeros_vs_empty()
    print(f"  zeros() (zero-fill): {zeros_ms:.3f} ms")
    if empty_ms:
        print(f"  empty() (no fill): {empty_ms:.3f} ms")
        print(f"  Ratio zeros/empty: {zeros_ms/empty_ms:.1f}x")
    
    print("\n✓ Storage pool benchmark completed")
