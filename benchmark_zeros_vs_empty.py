import fastnn as ft
import time

def benchmark_zeros_vs_empty():
    """Benchmark Tensor.zeros() vs Tensor.empty()"""
    
    sizes = [
        (10,),  # Small
        (100,),  # Small-medium
        (1000,),  # Medium
        (10000,),  # Large
        (100000,),  # Very large
        (1000000,),  # Huge
    ]
    
    print("=" * 60)
    print("Benchmark: zeros() vs empty()")
    print("=" * 60)
    print(f"{'Size':<20} {'zeros() ms':<15} {'empty() ms':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for size in sizes:
        numel = 1
        for s in size:
            numel *= s
        nbytes = numel * 4  # f32
        
        # Benchmark zeros()
        start = time.perf_counter()
        for _ in range(100):
            t = ft._core.zeros(shape=list(size))
        zeros_time = (time.perf_counter() - start) * 1000 / 100
        
        # Benchmark empty()
        start = time.perf_counter()
        for _ in range(100):
            t = ft._core.empty(shape=list(size))
        empty_time = (time.perf_counter() - start) * 1000 / 100
        
        speedup = zeros_time / empty_time if empty_time > 0 else float('inf')
        size_str = f"({numel:,})"
        print(f"{size_str:<20} {zeros_time:<15.4f} {empty_time:<15.4f} {speedup:<10.2f}x")
    
    print("=" * 60)

if __name__ == "__main__":
    benchmark_zeros_vs_empty()
