"""Benchmark Python hot-path: Sequential and BasicBlock."""
from tests.benchmark_utils import BenchmarkTimer
import fastnn as fnn


def benchmark_sequential(num_iters=100):
    """Benchmark PySequential vs Rust Sequential."""
    # Create a simple MLP in Python
    layers = [
        fnn.Linear(128, 256),
        fnn.ReLU(),
        fnn.Linear(256, 128),
        fnn.ReLU(),
    ]
    py_seq = fnn.PySequential(layers)
    x = fnn.randn([32, 128])
    
    # Warmup
    for _ in range(5):
        py_seq(x)
    
    timer = BenchmarkTimer(warmup=0, iterations=num_iters, unit="ms")
    def run_py():
        py_seq(x)
    py_time = timer.value(run_py)
    
    # Rust Sequential
    rust_seq = fnn.Sequential(layers)
    def run_rust():
        rust_seq(x)
    rust_time = timer.value(run_rust)
    
    return py_time, rust_time


def benchmark_basicblock(num_iters=50):
    """Benchmark Python BasicBlock vs Rust ResidualBlock."""
    # Python BasicBlock
    conv1 = fnn.Conv2d(64, 64, 3, padding=1)
    bn1 = fnn.BatchNorm2d(64)
    conv2 = fnn.Conv2d(64, 64, 3, padding=1)
    bn2 = fnn.BatchNorm2d(64)
    py_block = fnn.BasicBlock(conv1, bn1, fnn.ReLU(), conv2, bn2)
    x = fnn.randn([32, 64, 32, 32])
    
    # Warmup
    for _ in range(3):
        py_block(x)
    
    timer = BenchmarkTimer(warmup=0, iterations=num_iters, unit="ms")
    def run_py_block():
        py_block(x)
    py_time = timer.value(run_py_block)
    
    # Rust ResidualBlock
    try:
        rust_block = fnn.ResidualBlock(
            64, 64, 3, 1, 1,  # conv1 params
            64, 64, 3, 1, 1,  # conv2 params
            None
        )
        def run_rust_block():
            rust_block(x)
        rust_time = timer.value(run_rust_block)
    except Exception as e:
        print(f"  Rust ResiduablBlock error: {e}")
        rust_time = None
    
    return py_time, rust_time


if __name__ == "__main__":
    print("Benchmarking Sequential (100 iters, batch=32, 128->256->128):")
    py_seq_ms, rust_seq_ms = benchmark_sequential()
    print(f"  Python PySequential: {py_seq_ms:.3f} ms/forward")
    print(f"  Rust Sequential: {rust_seq_ms:.3f} ms/forward")
    print(f"  Speedup: {py_seq_ms/rust_seq_ms:.1f}x")
    
    print("\nBenchmarking BasicBlock (50 iters, batch=32, 64x32x32):")
    py_block_ms, rust_block_ms = benchmark_basicblock()
    print(f"  Python BasicBlock: {py_block_ms:.3f} ms/forward")
    if rust_block_ms:
        print(f"  Rust ResidualBlock: {rust_block_ms:.3f} ms/forward")
        print(f"  Speedup: {py_block_ms/rust_block_ms:.1f}x")
    else:
        print("  Rust ResidualBlock not available or not compatible")
    
    print("\n✓ Benchmarks completed")
