"""Final CPU vs GPU performance comparison for large tensors."""

import fastnn as fnn
import time
import statistics


def benchmark_single(op_name, device, size, iterations=5):
    """Benchmark a single operation."""
    print(f"  Testing {op_name} on {device} (size {size[0]}x{size[1]})...")

    if op_name == "add":
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)

        def op():
            return a + b
    elif op_name == "mul":
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)

        def op():
            return a * b
    elif op_name == "relu":
        a = fnn.rand(list(size), device=device)

        def op():
            return fnn.relu(a)
    elif op_name.startswith("matmul"):
        m, k, n = size
        a = fnn.rand([m, k], device=device)
        b = fnn.rand([k, n], device=device)

        def op():
            return a @ b
    else:
        return None

    # Warmup
    for _ in range(3):
        op()

    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        op()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = statistics.mean(times) * 1000
    return avg_ms


def main():
    print("=" * 70)
    print("fastnn CPU vs GPU Performance Comparison - Large Tensors")
    print("=" * 70)

    test_cases = [
        ("Add", (500, 500)),
        ("Add", (1000, 1000)),
        ("Add", (2048, 2048)),
        ("Mul", (500, 500)),
        ("Mul", (1000, 1000)),
        ("ReLU", (500, 500)),
        ("ReLU", (1000, 1000)),
        ("matmul", (500, 500, 500)),
        ("matmul", (1000, 1000, 1000)),
    ]

    results = []

    print("\nRunning benchmarks...")
    print("-" * 70)

    for op_name, size in test_cases:
        cpu_time = benchmark_single(op_name, "cpu", size)
        gpu_time = benchmark_single(op_name, "gpu", size)

        if cpu_time and gpu_time:
            results.append(
                {
                    "operation": op_name,
                    "size": size,
                    "cpu_ms": cpu_time,
                    "gpu_ms": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
                }
            )

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS (time in milliseconds)")
    print("=" * 70)
    print(
        f"{'Operation':<15} {'Size':<20} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}"
    )
    print("-" * 70)

    for r in results:
        size_str = str(r["size"])
        speedup = r["speedup"]
        if speedup > 1:
            status = "✅ GPU faster"
        else:
            status = f"❌ CPU faster ({1 / speedup:.1f}x)"
        print(
            f"{r['operation']:<15} {size_str:<20} {r['cpu_ms']:<12.1f} {r['gpu_ms']:<12.1f} {speedup:<10.2f}x {status}"
        )

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Current GPU Implementation Issues:
1. Data Transfer Overhead: Every GPU operation involves:
   - CPU → GPU copy (input data)
   - GPU compute
   - GPU → CPU copy (output data)
   - CPU → GPU copy (result tensor)

2. This makes GPU operations SLOWER than CPU for small/medium tensors.

3. Expected Performance with Optimized Implementation:
   - Memory-bound ops (add, mul, relu): 2-5x speedup on GPU
   - Compute-bound ops (matmul): 10-50x speedup on GPU

4. Current Status: GPU is functional but NOT optimized for performance yet.
   """)


if __name__ == "__main__":
    main()
