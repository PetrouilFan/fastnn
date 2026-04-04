"""
PyTorch vs fastnn head-to-head benchmark.
PyTorch uses all available CPU threads (default).
fastnn uses rayon (all threads by default).
"""

import torch
import torch.nn as nn
import time
import subprocess
import gc
import os  # noqa: F401


def bench_fn(fn, warmup=20, iters=500):
    """Benchmark, return min of 3 median measurements."""
    medians = []
    for _trial in range(3):
        gc.collect()  # Ensure clean state between trials
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        times.sort()
        medians.append(times[len(times) // 2])
    return min(medians)  # best of 3 medians


def bench_linear(m, k, dtype=torch.float32, iters=500):
    linear = nn.Linear(k, m, bias=False).to(dtype)
    x = torch.randn(k, dtype=dtype)

    def fn():
        return linear(x)

    ms = bench_fn(fn, warmup=20, iters=iters)
    gflops = (2 * m * k) / (ms / 1000) / 1e9
    return ms, gflops


def bench_int8_dynamic(m, k, iters=500):
    linear = nn.Linear(k, m, bias=False)
    qlinear = torch.ao.quantization.quantize_dynamic(
        linear, {nn.Linear}, dtype=torch.qint8
    )
    x = torch.randn(k)

    def fn():
        return qlinear(x)

    ms = bench_fn(fn, warmup=20, iters=iters)
    gflops = (2 * m * k) / (ms / 1000) / 1e9
    return ms, gflops


def run_fastnn_bench():
    """Run the Rust benchmark and parse output."""
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        ["cargo", "run", "--release", "--bin", "packed_bench"],
        capture_output=True,
        text=True,
        cwd=bench_dir,
        timeout=300,
    )
    return result.stdout


def main():
    n_threads = torch.get_num_threads()
    print(f"PyTorch {torch.__version__}, threads={n_threads}")

    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True)
        for line in cpu.stdout.split("\n"):
            if "Model name" in line:
                print(f"CPU: {line.split(':')[1].strip()}")
                break
            if "CPU(s):" in line and "NUMA" not in line:
                print(f"Cores: {line.split(':')[1].strip()}")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    sizes = [(256, 256), (512, 512), (1024, 1024), (4096, 4096)]

    print("\n" + "=" * 70)
    print("PYTORCH RESULTS (all threads)")
    print("=" * 70)

    print(
        f"\n{'M×K':<12} {'f32 ms':>10} {'f32 GFLOP/s':>12} {'int8 ms':>10} {'int8 GFLOP/s':>12} {'int8 vs f32':>12}"
    )
    for m, k in sizes:
        ms_f32, gf_f32 = bench_linear(m, k, torch.float32)
        ms_q8, gf_q8 = bench_int8_dynamic(m, k)
        print(
            f"{m}×{k:<7} {ms_f32:>10.3f} {gf_f32:>12.2f} {ms_q8:>10.3f} {gf_q8:>12.2f} {ms_f32 / ms_q8:>11.1f}x"
        )

    print("\n" + "=" * 70)
    print("FASTNN RESULTS (all threads via rayon)")
    print("=" * 70)
    fastnn_output = run_fastnn_bench()
    # Print just the GEMV section
    in_gemv = False
    for line in fastnn_output.split("\n"):
        if (
            line.startswith("GEMV ")
            or line.startswith("dtype")
            or line.startswith("F32x1")
            or line.startswith("F16x2")
            or line.startswith("U8x4")
            or line.startswith("U4x8")
        ):
            print(line)
            in_gemv = True
        elif in_gemv and line.strip() == "":
            break

    print("\n" + "=" * 70)
    print("COMBINED COMPARISON (all threads)")
    print("=" * 70)
    print("See above for absolute numbers. Key insight:")
    print("fastnn U8x4/U4x8 achieve competitive GFLOP/s with 4-8x memory savings.")


if __name__ == "__main__":
    main()
