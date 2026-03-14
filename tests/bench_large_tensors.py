"""Benchmark CPU vs GPU performance on large tensors (500x500+)."""

import fastnn as fnn
import time
import statistics


def benchmark(name, fn, warmup=5, iterations=30):
    """Benchmark a function and return mean time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1e6)  # Convert to microseconds

    mean_time = statistics.mean(times)
    return mean_time


def run_benchmarks(device, sizes):
    """Run benchmarks on the specified device."""
    results = {}

    # Add/Mul operations
    for size in sizes:
        label = f"Add {size[0]}x{size[1]}"
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)
        results[label] = benchmark(label, lambda: a + b)
        print(f"  {label}: {results[label]:.1f}μs")

    for size in sizes:
        label = f"Mul {size[0]}x{size[1]}"
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)
        results[label] = benchmark(label, lambda: a * b)
        print(f"  {label}: {results[label]:.1f}μs")

    # ReLU
    for size in sizes:
        label = f"ReLU {size[0]}x{size[1]}"
        a = fnn.rand(list(size), device=device)
        results[label] = benchmark(label, lambda: fnn.relu(a))
        print(f"  {label}: {results[label]:.1f}μs")

    # MatMul with different sizes
    matmul_sizes = [
        (500, 500, 500),
        (1000, 1000, 1000),
        (2048, 2048, 2048),
    ]
    for m, k, n in matmul_sizes:
        label = f"MatMul {m}x{k}x{n}"
        a = fnn.rand([m, k], device=device)
        b = fnn.rand([k, n], device=device)
        results[label] = benchmark(label, lambda: a @ b)
        print(f"  {label}: {results[label]:.1f}μs")

    return results


def compare_performance(cpu_results, gpu_results):
    """Compare CPU and GPU performance."""
    print("\n" + "=" * 80)
    print("Performance Comparison (CPU vs GPU)")
    print("=" * 80)
    print(
        f"{'Operation':<25} {'CPU (μs)':<15} {'GPU (μs)':<15} {'Speedup':<12} {'Status'}"
    )
    print("-" * 80)

    for op, cpu_time in cpu_results.items():
        gpu_time = gpu_results.get(op, None)
        if gpu_time:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            if speedup > 1:
                speedup_str = f"{speedup:.2f}x"
                status = "✅ GPU faster" if speedup > 2 else "⚠️  Close"
            else:
                speedup_str = f"{1 / speedup:.2f}x"
                status = "❌ CPU faster"
            print(
                f"{op:<25} {cpu_time:<15.1f} {gpu_time:<15.1f} {speedup_str:<12} {status}"
            )


if __name__ == "__main__":
    print("=" * 80)
    print("fastnn Large Tensor Benchmark (CPU vs GPU)")
    print("Testing tensors 500x500 and larger")
    print("=" * 80)

    # Define test sizes
    sizes = [
        (500, 500),
        (1000, 1000),
        (2048, 2048),
    ]

    # CPU benchmarks
    print("\n" + "=" * 80)
    print("CPU Benchmarks")
    print("=" * 80)
    cpu_results = run_benchmarks("cpu", sizes)

    # GPU benchmarks
    print("\n" + "=" * 80)
    print("GPU Benchmarks")
    print("=" * 80)
    gpu_results = run_benchmarks("gpu", sizes)

    # Compare
    compare_performance(cpu_results, gpu_results)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
