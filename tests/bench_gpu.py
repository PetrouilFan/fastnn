"""Benchmark script comparing CPU vs GPU performance for fastnn operations."""

import fastnn as fnn
import time
import statistics


def benchmark(name, fn, warmup=10, iterations=100):
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
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    return mean_time, std_time


def run_benchmarks(device, device_label):
    """Run benchmarks on the specified device."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarks on {device_label}")
    print(f"{'=' * 70}")

    results = {}

    # Add
    for size in [(100, 100), (1000, 1000)]:
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)
        mean, _ = benchmark("add", lambda: a + b)
        results[f"Add {size[0]}x{size[1]}"] = mean
        print(f"Add {size[0]}x{size[1]}: {mean:.1f}μs")

    # Mul
    for size in [(100, 100), (1000, 1000)]:
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)
        mean, _ = benchmark("mul", lambda: a * b)
        results[f"Mul {size[0]}x{size[1]}"] = mean
        print(f"Mul {size[0]}x{size[1]}: {mean:.1f}μs")

    # ReLU
    for size in [(100, 100), (1000, 1000)]:
        x = fnn.rand(list(size), device=device)
        mean, _ = benchmark("relu", lambda: fnn.relu(x))
        results[f"ReLU {size[0]}x{size[1]}"] = mean
        print(f"ReLU {size[0]}x{size[1]}: {mean:.1f}μs")

    # FusedAddReLU
    for size in [(100, 100), (1000, 1000)]:
        a = fnn.rand(list(size), device=device)
        b = fnn.rand(list(size), device=device)
        mean, _ = benchmark("fused_add_relu", lambda: fnn.fused_add_relu(a, b))
        results[f"FusedAddReLU {size[0]}x{size[1]}"] = mean
        print(f"FusedAddReLU {size[0]}x{size[1]}: {mean:.1f}μs")

    # MatMul
    for sizes in [(128, 256, 256, 128), (256, 512, 512, 256), (512, 1024, 1024, 512)]:
        m, k, k2, n = sizes
        a = fnn.rand([m, k], device=device)
        b = fnn.rand([k, n], device=device)
        mean, _ = benchmark("matmul", lambda: a @ b)
        results[f"MatMul {m}x{k}x{n}"] = mean
        print(f"MatMul {m}x{k}x{n}: {mean:.1f}μs")

    # GELU
    for size in [(100, 100), (1000, 1000)]:
        x = fnn.rand(list(size), device=device)
        mean, _ = benchmark("gelu", lambda: fnn.gelu(x))
        results[f"GELU {size[0]}x{size[1]}"] = mean
        print(f"GELU {size[0]}x{size[1]}: {mean:.1f}μs")

    # Sigmoid
    for size in [(100, 100), (1000, 1000)]:
        x = fnn.rand(list(size), device=device)
        mean, _ = benchmark("sigmoid", lambda: fnn.sigmoid(x))
        results[f"Sigmoid {size[0]}x{size[1]}"] = mean
        print(f"Sigmoid {size[0]}x{size[1]}: {mean:.1f}μs")

    # Tanh
    for size in [(100, 100), (1000, 1000)]:
        x = fnn.rand(list(size), device=device)
        mean, _ = benchmark("tanh", lambda: fnn.tanh(x))
        results[f"Tanh {size[0]}x{size[1]}"] = mean
        print(f"Tanh {size[0]}x{size[1]}: {mean:.1f}μs")

    # Linear
    for config in [(32, 256, 512), (32, 512, 1024), (128, 256, 512)]:
        batch_size, in_features, out_features = config
        linear = fnn.Linear(in_features, out_features)
        # Move linear weights to device
        if device == "gpu":
            # Note: Linear layer weights are created on CPU by default
            # For proper GPU benchmarking, we'd need to move them
            pass
        x = fnn.rand([batch_size, in_features], device=device)
        mean, _ = benchmark("linear", lambda: linear(x))
        results[f"Linear {batch_size}x{in_features}x{out_features}"] = mean
        print(f"Linear {batch_size}x{in_features}x{out_features}: {mean:.1f}μs")

    # Sum
    size = (1000, 1000)
    x = fnn.rand(list(size), device=device)
    mean, _ = benchmark("sum", lambda: fnn.sum(x, 1))
    results[f"Sum {size[0]}x{size[1]}"] = mean
    print(f"Sum {size[0]}x{size[1]}: {mean:.1f}μs")

    # Mean
    x = fnn.rand(list(size), device=device)
    mean, _ = benchmark("mean", lambda: fnn.mean(x, 1))
    results[f"Mean {size[0]}x{size[1]}"] = mean
    print(f"Mean {size[0]}x{size[1]}: {mean:.1f}μs")

    # Max
    x = fnn.rand(list(size), device=device)
    mean, _ = benchmark("max", lambda: fnn.max(x, 1))
    results[f"Max {size[0]}x{size[1]}"] = mean
    print(f"Max {size[0]}x{size[1]}: {mean:.1f}μs")

    return results


def compare_performance(cpu_results, gpu_results):
    """Compare CPU and GPU performance."""
    print(f"\n{'=' * 70}")
    print("Performance Comparison (CPU vs GPU)")
    print(f"{'=' * 70}")
    print(f"{'Operation':<30} {'CPU (μs)':<15} {'GPU (μs)':<15} {'Speedup':<10}")
    print("-" * 70)

    for op, cpu_time in cpu_results.items():
        gpu_time = gpu_results.get(op, None)
        if gpu_time:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1 / speedup:.2f}x"
            print(f"{op:<30} {cpu_time:<15.1f} {gpu_time:<15.1f} {speedup_str:<10}")


if __name__ == "__main__":
    print("fastnn CPU vs GPU Benchmark")
    print("=" * 70)

    try:
        # Run CPU benchmarks
        cpu_results = run_benchmarks("cpu", "CPU")

        # Run GPU benchmarks
        try:
            gpu_results = run_benchmarks("gpu", "GPU")
            compare_performance(cpu_results, gpu_results)
        except Exception as e:
            print(f"\nGPU benchmarks failed: {e}")
            print("This is expected if GPU support is not yet fully implemented.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
