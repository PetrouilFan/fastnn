"""Benchmark tensor creation from various sources."""
from tests.benchmark_utils import BenchmarkTimer
import numpy as np
import fastnn as fnn


def benchmark_tensor_from_numpy(num_iters=1000):
    """Benchmark tensor creation from numpy array."""
    arr = np.random.randn(32, 32).astype(np.float32)
    shape = list(arr.shape)
    
    timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    def create_from_numpy():
        fnn.tensor(arr, shape)
    return timer.value(create_from_numpy)


def benchmark_tensor_from_list(num_iters=1000):
    """Benchmark tensor creation from Python list (slow path)."""
    lst = [float(i) for i in range(1024)]
    shape = [32, 32]
    
    timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    def create_from_list():
        fnn.tensor(lst, shape)
    return timer.value(create_from_list)


def benchmark_collate_fastnn(num_iters=500):
    """Benchmark collate with fastnn tensors (fast path)."""
    from fastnn.data import default_collate
    tensors = [fnn.randn([3, 32, 32]) for _ in range(16)]
    
    timer = BenchmarkTimer(warmup=10, iterations=num_iters, unit="ms")
    def run_collate():
        batch = [t for t in tensors[:2]]
        default_collate(batch)
    return timer.value(run_collate)


if __name__ == "__main__":
    print("Benchmarking tensor creation:")
    np_time = benchmark_tensor_from_numpy()
    print(f"  From numpy array: {np_time:.3f} ms/creation")
    
    list_time = benchmark_tensor_from_list()
    print(f"  From Python list: {list_time:.3f} ms/creation")
    
    print("\nBenchmarking collate (2 tensors, 3x32x32):")
    collate_time = benchmark_collate_fastnn()
    print(f"  Collate fastnn tensors: {collate_time:.3f} ms/collate")
    
    print("\n✓ Benchmarks completed")
