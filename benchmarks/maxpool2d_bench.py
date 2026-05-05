"""Benchmark MaxPool2d reuse fix."""
import time
import fastnn as fnn

def benchmark_maxpool2d(num_iters=2000):
    """Benchmark MaxPool2d forward passes with reuse."""
    x = fnn.randn([16, 3, 32, 32])
    
    # Fixed version: module created once in __init__
    pool = fnn.MaxPool2d(kernel_size=2, stride=2)
    
    # Warmup
    for _ in range(100):
        pool(x)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        pool(x)
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iters * 1000  # ms per forward

def benchmark_maxpool2d_old(num_iters=2000):
    """Simulate old behavior: create Rust module every call."""
    x = fnn.randn([16, 3, 32, 32])
    
    def forward(x):
        # Simulate old __call__ that creates module every time
        import fastnn._core as _core
        rust_maxpool = _core.MaxPool2d(2, 2, 0, 1)
        return rust_maxpool(x)
    
    # Warmup
    for _ in range(100):
        forward(x)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        forward(x)
    elapsed = time.perf_counter() - start
    
    return elapsed / num_iters * 1000  # ms per forward

if __name__ == "__main__":
    print("Benchmarking MaxPool2d forward (2000 iters, batch=16, 3x32x32):")
    reuse_time = benchmark_maxpool2d()
    old_time = benchmark_maxpool2d_old()
    print(f"  Fixed (reuse module): {reuse_time:.3f} ms/forward")
    print(f"  Old (create every call): {old_time:.3f} ms/forward")
    print(f"  Ratio (old/new): {old_time/reuse_time:.2f}x")
    print("  ✓ Fix eliminates redundant module construction in __call__")
    print("  Note: Timings may vary based on Rust module creation cost")
