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
    
    # Benchmark Python Sequential with built-in warmup
    py_timer = BenchmarkTimer(warmup=5, iterations=num_iters, unit="ms")
    def run_py():
        py_seq(x)
    py_result = py_timer.measure(run_py)
    
    # Rust Sequential (new timer per benchmark)
    rust_seq = fnn.Sequential(layers)
    rust_timer = BenchmarkTimer(warmup=5, iterations=num_iters, unit="ms")
    def run_rust():
        rust_seq(x)
    rust_result = rust_timer.measure(run_rust)
    
    return py_result, rust_result


def benchmark_basicblock(num_iters=50):
    """Benchmark Python BasicBlock vs Rust ResidualBlock."""
    # Python BasicBlock
    conv1 = fnn.Conv2d(64, 64, 3, padding=1)
    bn1 = fnn.BatchNorm2d(64)
    conv2 = fnn.Conv2d(64, 64, 3, padding=1)
    bn2 = fnn.BatchNorm2d(64)
    py_block = fnn.BasicBlock(conv1, bn1, fnn.ReLU(), conv2, bn2)
    x = fnn.randn([32, 64, 32, 32])
    
    # Benchmark Python BasicBlock with built-in warmup
    py_timer = BenchmarkTimer(warmup=5, iterations=num_iters, unit="ms")
    def run_py_block():
        py_block(x)
    py_result = py_timer.measure(run_py_block)
    
    # Rust ResidualBlock (new timer per benchmark)
    try:
        rust_block = fnn.ResidualBlock(
            64, 64, 3, 1, 1,  # conv1 params
            64, 64, 3, 1, 1,  # conv2 params
            None
        )
        rust_timer = BenchmarkTimer(warmup=5, iterations=num_iters, unit="ms")
        def run_rust_block():
            rust_block(x)
        rust_result = rust_timer.measure(run_rust_block)
    except Exception as e:
        print(f"  Rust ResidualBlock error: {e}")
        rust_result = None
    
    return py_result, rust_result


if __name__ == "__main__":
    print("Benchmarking Sequential (100 iters, batch=32, 128->256->128):")
    py_seq_res, rust_seq_res = benchmark_sequential()
    print(f"  Python PySequential: {py_seq_res.mean:.3f} ± {py_seq_res.std:.3f} ms/forward")
    print(f"  Rust Sequential: {rust_seq_res.mean:.3f} ± {rust_seq_res.std:.3f} ms/forward")
    print(f"  Speedup: {py_seq_res.mean/rust_seq_res.mean:.1f}x")
    
    print("\nBenchmarking BasicBlock (50 iters, batch=32, 64x32x32):")
    py_block_res, rust_block_res = benchmark_basicblock()
    print(f"  Python BasicBlock: {py_block_res.mean:.3f} ± {py_block_res.std:.3f} ms/forward")
    if rust_block_res:
        print(f"  Rust ResidualBlock: {rust_block_res.mean:.3f} ± {rust_block_res.std:.3f} ms/forward")
        print(f"  Speedup: {py_block_res.mean/rust_block_res.mean:.1f}x")
    else:
        print("  Rust ResidualBlock not available or not compatible")
    
    print("\n✓ Benchmarks completed")
