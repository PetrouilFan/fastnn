"""Benchmark storage pool allocation patterns."""
from tests.benchmark_utils import BenchmarkTimer
import fastnn as fnn
import gc


def benchmark_tensor_creation(num_iters=1000):
    """Benchmark creating and destroying tensors (pool hit/miss)."""    
    # Benchmark: create tensors in loop (should hit pool)
    timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    def create_tensor():
        fnn.randn([256, 256])
    return timer.measure(create_tensor)


def benchmark_zeros_vs_empty(num_iters=1000):
    """Benchmark zeros (zero-fill) vs empty (no fill)."""
    # Benchmark zeros (new timer per benchmark)
    zeros_timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    def create_zeros():
        fnn.zeros([256, 256])
    zeros_result = zeros_timer.measure(create_zeros)
    
    # Benchmark empty (new timer per benchmark)
    try:
        empty_timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
        def create_empty():
            fnn.empty([256, 256])
        empty_result = empty_timer.measure(create_empty)
    except AttributeError:
        empty_result = None
    
    return zeros_result, empty_result


if __name__ == "__main__":
    print("Benchmarking storage pool allocation...")
    create_result = benchmark_tensor_creation()
    print(f"  Tensor creation (pool reuse): {create_result.mean:.3f} ± {create_result.std:.3f} ms/creation")
    
    zeros_result, empty_result = benchmark_zeros_vs_empty()
    print(f"  zeros() (zero-fill): {zeros_result.mean:.3f} ± {zeros_result.std:.3f} ms")
    if empty_result:
        print(f"  empty() (no fill): {empty_result.mean:.3f} ± {empty_result.std:.3f} ms")
        print(f"  Ratio zeros/empty: {zeros_result.mean/empty_result.mean:.1f}x")
    
    print("\n✓ Storage pool benchmark completed")
