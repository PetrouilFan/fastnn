"""Simple GPU benchmark comparison."""

import fastnn as fnn
import time


def benchmark_op(name, op, device, warmup=3, iterations=20):
    """Benchmark a single operation."""
    print(f"  Benchmarking {name} on {device}...")

    # Warmup
    for _ in range(warmup):
        op()

    # Time measurement
    start = time.perf_counter()
    for _ in range(iterations):
        op()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / iterations) * 1000
    return avg_time_ms


def main():
    print("=" * 60)
    print("fastnn CPU vs GPU Benchmark (Simple Operations)")
    print("=" * 60)

    operations = []

    # Test 1: Add (100x100)
    sizes = [(100, 100), (1000, 1000)]
    for size in sizes:
        w, h = size
        # CPU
        a_cpu = fnn.rand([w, h], device="cpu")
        b_cpu = fnn.rand([w, h], device="cpu")
        cpu_time = benchmark_op(f"Add {w}x{h}", lambda: a_cpu + b_cpu, "cpu")

        # GPU
        a_gpu = fnn.rand([w, h], device="gpu")
        b_gpu = fnn.rand([w, h], device="gpu")
        gpu_time = benchmark_op(f"Add {w}x{h}", lambda: a_gpu + b_gpu, "gpu")

        operations.append(("Add", f"{w}x{h}", cpu_time, gpu_time))

    # Test 2: ReLU (100x100)
    for size in sizes:
        w, h = size
        # CPU
        a_cpu = fnn.rand([w, h], device="cpu")
        cpu_time = benchmark_op(f"ReLU {w}x{h}", lambda: fnn.relu(a_cpu), "cpu")

        # GPU
        a_gpu = fnn.rand([w, h], device="gpu")
        gpu_time = benchmark_op(f"ReLU {w}x{h}", lambda: fnn.relu(a_gpu), "gpu")

        operations.append(("ReLU", f"{w}x{h}", cpu_time, gpu_time))

    # Test 3: MatMul (128x256x128)
    for m, k, n in [(128, 256, 128)]:
        # CPU
        a_cpu = fnn.rand([m, k], device="cpu")
        b_cpu = fnn.rand([k, n], device="cpu")
        cpu_time = benchmark_op(f"MatMul {m}x{k}x{n}", lambda: a_cpu @ b_cpu, "cpu")

        # GPU
        a_gpu = fnn.rand([m, k], device="gpu")
        b_gpu = fnn.rand([k, n], device="gpu")
        gpu_time = benchmark_op(f"MatMul {m}x{k}x{n}", lambda: a_gpu @ b_gpu, "gpu")

        operations.append(("MatMul", f"{m}x{k}x{n}", cpu_time, gpu_time))

    # Print results
    print("\n" + "=" * 60)
    print("Results (time in milliseconds, lower is better)")
    print("=" * 60)
    print(
        f"{'Operation':<15} {'Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<10} {'Speedup':<10}"
    )
    print("-" * 60)

    for op_name, size, cpu_time, gpu_time in operations:
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1 / speedup:.2f}x"
        print(
            f"{op_name:<15} {size:<12} {cpu_time:<12.2f} {gpu_time:<10.2f} {speedup_str:<10}"
        )

    print("\n" + "=" * 60)
    print("Note: GPU overhead may make small tensors slower than CPU.")
    print("Larger tensors show better GPU utilization.")
    print("=" * 60)


if __name__ == "__main__":
    main()
