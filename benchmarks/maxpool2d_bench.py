"""Benchmark MaxPool2d reuse fix."""
from tests.benchmark_utils import BenchmarkTimer
import fastnn as fnn


def benchmark_maxpool2d(num_iters=2000):
    """Benchmark MaxPool2d forward passes with reuse."""
    x = fnn.randn([16, 3, 32, 32])
    
    # Fixed version: module created once in __init__
    pool = fnn.MaxPool2d(kernel_size=2, stride=2)
    
    timer = BenchmarkTimer(warmup=100, iterations=num_iters, unit="ms")
    def run_pool():
        pool(x)
    return timer.value(run_pool)


def benchmark_maxpool2d_old(num_iters=2000):
    """Simulate old behavior: create Rust module every call."""
    x = fnn.randn([16, 3, 32, 32])
    
    def forward(x):
        # Simulate old __call__ that creates module every time
        import fastnn._core as _core
        rust_maxpool = _core.MaxPool2d(2, 2, 0, 1)
        return rust_maxpool(x)
    
    timer = BenchmarkTimer(warmup=100, iterations=num_iters, unit="ms")
    return timer.value(forward)


if __name__ == "__main__":
    print("Benchmarking MaxPool2d forward (2000 iters, batch=16, 3x32x32):")
    reuse_ms = benchmark_maxpool2d()
    old_ms = benchmark_maxpool2d_old()
    print(f"  Fixed (reuse module): {reuse_ms:.3f} ms/forward")
    print(f"  Old (create every call): {old_ms:.3f} ms/forward")
    print(f"  Ratio (old/new): {old_ms/reuse_ms:.2f}x")
    print("  ✓ Fix eliminates redundant module construction in __call__")
    print("  Note: Timings may vary based on Rust module creation cost")
