"""Benchmark storage pool allocation patterns."""
from tests.benchmark_utils import BenchmarkTimer
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
    timer = BenchmarkTimer(warmup=0, iterations=num_iters, unit="ms")
    def create_tensor():
        fnn.randn([256, 256])
    return timer.value(create_tensor)


def benchmark_zeros_vs_empty(num_iters=1000):
    """Benchmark zeros (zero-fill) vs empty (no fill)."""
    timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    
    # Benchmark zeros
    def create_zeros():
        fnn.zeros([256, 256])
    zeros_ms = timer.value(create_zeros)
    
    # Benchmark empty (if available, otherwise skip)
    try:
        def create_empty():
            fnn.empty([256, 256])
        empty_ms = timer.value(create_empty)
    except AttributeError:
        empty_ms = None
    
    return zeros_ms, empty_ms


if __name__ == "__main__":
    print("Benchmarking storage pool allocation...")
    create_ms = benchmark_tensor_creation()
    print(f"  Tensor creation (pool reuse): {create_ms:.3f} ms/creation")
    
    zeros_ms, empty_ms = benchmark_zeros_vs_empty()
    print(f"  zeros() (zero-fill): {zeros_ms:.3f} ms")
    if empty_ms:
        print(f"  empty() (no fill): {empty_ms:.3f} ms")
        print(f"  Ratio zeros/empty: {zeros_ms/empty_ms:.1f}x")
    
    print("\n✓ Storage pool benchmark completed")
