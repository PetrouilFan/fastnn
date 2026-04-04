"""
PyTorch GEMV benchmark for comparison with fastnn packed precision.
Measures torch.nn.Linear forward pass (matrix × vector + bias).
"""

import torch
import torch.nn as nn
import time
import subprocess


def bench_fn(fn, warmup=10, iters=200):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # median


def bench_linear(m, k, dtype_str="float32", iters=200):
    """Benchmark torch.nn.Linear forward pass."""
    if dtype_str == "float32":
        dtype = torch.float32
    elif dtype_str == "float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

    linear = nn.Linear(k, m, bias=False).to(dtype)
    x = torch.randn(k, dtype=dtype)

    def fn():
        return linear(x)

    ms = bench_fn(fn, warmup=10, iters=iters)
    gflops = (2 * m * k) / (ms / 1000) / 1e9
    return ms, gflops


def bench_int8_dynamic(m, k, iters=200):
    """Benchmark PyTorch dynamic int8 quantized linear."""
    linear = nn.Linear(k, m, bias=False)
    qlinear = torch.ao.quantization.quantize_dynamic(
        linear, {nn.Linear}, dtype=torch.qint8
    )
    x = torch.randn(k)

    def fn():
        return qlinear(x)

    ms = bench_fn(fn, warmup=10, iters=iters)
    gflops = (2 * m * k) / (ms / 1000) / 1e9
    return ms, gflops


def bench_int8_static(m, k, iters=200):
    """Benchmark PyTorch static int8 quantized linear (via int8 GEMM)."""
    # Use the lower-level int8 matmul directly
    # Weight is [m, k] as int8, activation is [k] as int8
    weight = torch.randint(-128, 127, (m, k), dtype=torch.int8)
    x = torch.randn(k)

    # PyTorch doesn't expose int8 GEMV directly in Python easily
    # We simulate: dequantize to f32 and do f32 matmul
    # This is what happens internally for dynamic quantization
    weight_f = weight.to(torch.float32)

    def fn():
        return torch.mv(weight_f, x)

    ms = bench_fn(fn, warmup=10, iters=iters)
    gflops = (2 * m * k) / (ms / 1000) / 1e9
    return ms, gflops


def main():
    torch.set_num_threads(1)  # Single-threaded for fair comparison
    n_threads = torch.get_num_threads()

    print(f"PyTorch {torch.__version__}, threads={n_threads}")

    # Get CPU info
    try:
        cpu_info = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in cpu_info.stdout.split("\n"):
            if "Model name" in line:
                print(f"CPU: {line.strip()}")
                break
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    print()
    sizes = [(256, 256), (512, 512), (1024, 1024), (4096, 4096)]

    print("=== torch.nn.Linear (f32) — baseline ===")
    print(f"{'M×K':<12} {'ms':>10} {'GFLOP/s':>10}")
    for m, k in sizes:
        ms, gflops = bench_linear(m, k, "float32")
        print(f"{m}×{k:<7} {ms:>10.3f} {gflops:>10.2f}")

    print()
    print("=== torch.nn.Linear (float16) ===")
    print(f"{'M×K':<12} {'ms':>10} {'GFLOP/s':>10} {'vs f32':>8}")
    for m, k in sizes:
        ms_f32, _ = bench_linear(m, k, "float32")
        ms_f16, gflops = bench_linear(m, k, "float16")
        print(f"{m}×{k:<7} {ms_f16:>10.3f} {gflops:>10.2f} {ms_f32 / ms_f16:>7.1f}x")

    print()
    print("=== Dynamic int8 quantized (torch.ao.quantization) ===")
    print(f"{'M×K':<12} {'ms':>10} {'GFLOP/s':>10} {'vs f32':>8}")
    for m, k in sizes:
        ms_f32, _ = bench_linear(m, k, "float32")
        try:
            ms_q, gflops = bench_int8_dynamic(m, k)
            print(f"{m}×{k:<7} {ms_q:>10.3f} {gflops:>10.2f} {ms_f32 / ms_q:>7.1f}x")
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"{m}×{k:<7} {'ERROR':>10} {str(e)[:40]}")

    print()
    print("=" * 60)
    print("Compare against fastnn Rust benchmark output.")
    print("PyTorch uses single thread for fair comparison.")
    print("=" * 60)


if __name__ == "__main__":
    main()
