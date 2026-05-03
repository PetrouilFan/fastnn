"""Benchmark tensor creation from various sources."""
import time
import numpy as np
import fastnn as fnn

def benchmark_tensor_from_numpy(num_iters=1000):
    """Benchmark tensor creation from numpy array."""
    arr = np.random.randn(32, 32).astype(np.float32)
    shape = list(arr.shape)
    
    # Warmup
    for _ in range(10):
        fnn.tensor(arr, shape)
    
    start = time.perf_counter()
    for _ in range(num_iters):
        fnn.tensor(arr, shape)
    elapsed = time.perf_counter() - start
    return elapsed / num_iters * 1000  # ms

def benchmark_tensor_from_list(num_iters=1000):
    """Benchmark tensor creation from Python list (slow path)."""
    lst = [float(i) for i in range(1024)]
    shape = [32, 32]
    
    # Warmup
    for _ in range(10):
        fnn.tensor(lst, shape)
    
    start = time.perf_counter()
    for _ in range(num_iters):
        fnn.tensor(lst, shape)
    elapsed = time.perf_counter() - start
    return elapsed / num_iters * 1000  # ms

def benchmark_collate_fastnn(num_iters=500):
    """Benchmark collate with fastnn tensors (fast path)."""
    import fastnn as fnn
    tensors = [fnn.randn([3, 32, 32]) for _ in range(16)]
    
    # Warmup
    for _ in range(10):
        from fastnn.data import default_collate
        batch = [t for t in tensors[:2]]
        default_collate(batch)
    
    start = time.perf_counter()
    for _ in range(num_iters):
        batch = [t for t in tensors[:2]]
        default_collate(batch)
    elapsed = time.perf_counter() - start
    return elapsed / num_iters * 1000  # ms

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
