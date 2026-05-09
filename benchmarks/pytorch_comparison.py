"""Benchmark comparing fastnn vs PyTorch on key operations.

Usage:
    uv run python benchmarks/pytorch_comparison.py
    uv run python benchmarks/pytorch_comparison.py --output results.json
    uv run python benchmarks/pytorch_comparison.py --markdown     # CI step summary
    uv run python benchmarks/pytorch_comparison.py --fast         # quick CI check

Outputs a comparison table and optionally a JSON file.
"""

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Tuple

import numpy as np

import fastnn as fnn

# ---------------------------------------------------------------------------
# Detect PyTorch availability
# ---------------------------------------------------------------------------

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if not _HAS_TORCH:
    print("PyTorch not installed. Please run: uv pip install torch")
    sys.exit(0)

import torch

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

_ITER_DEFAULTS = {
    "tiny": 500,
    "small": 200,
    "medium": 100,
    "large": 50,
    "train": 30,
}

_ITER_FAST = {
    "tiny": 50,
    "small": 30,
    "medium": 20,
    "large": 10,
    "train": 5,
}

_WARMUP = 5


@dataclass
class BenchResult:
    op: str
    fastnn_ms: float
    torch_ms: float
    speedup: float
    fastnn_std: float = 0.0
    torch_std: float = 0.0
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _measure(fn: Callable, warmup: int, iters: int) -> Tuple[float, float]:
    """Measure mean and std of fn() execution time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def _bench(
    name: str,
    fastnn_fn: Callable,
    torch_fn: Callable,
    warmup: int,
    iters: int,
    **params,
) -> BenchResult:
    fnn_ms, fnn_std = _measure(fastnn_fn, warmup, iters)
    th_ms, th_std = _measure(torch_fn, warmup, iters)
    speedup = th_ms / fnn_ms if fnn_ms > 0 else 0.0
    return BenchResult(
        op=name,
        fastnn_ms=round(fnn_ms, 3),
        torch_ms=round(th_ms, 3),
        speedup=round(speedup, 3),
        fastnn_std=round(fnn_std, 3),
        torch_std=round(th_std, 3),
        params=params,
    )


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def bench_matmul(iters):
    results = []
    for size in [256, 512, 1024]:
        cat = "small" if size <= 256 else ("medium" if size <= 512 else "large")
        n = iters.get(cat, 100)
        a_fnn = fnn.randn([size, size])
        b_fnn = fnn.randn([size, size])
        a_th = torch.randn(size, size)
        b_th = torch.randn(size, size)
        results.append(
            _bench(
                f"matmul_{size}x{size}",
                lambda a=a_fnn, b=b_fnn: fnn.matmul(a, b),
                lambda a=a_th, b=b_th: torch.matmul(a, b),
                _WARMUP,
                n,
                shape=f"({size},{size})",
            )
        )
    return results


def bench_elementwise(iters):
    results = []
    n = iters.get("tiny", 500)
    size = 1024
    a_fnn = fnn.randn([size, size])
    b_fnn = fnn.randn([size, size])
    a_th = torch.randn(size, size)
    b_th = torch.randn(size, size)

    ops = [
        ("add", lambda: a_fnn + b_fnn, lambda: a_th + b_th),
        ("mul", lambda: a_fnn * b_fnn, lambda: a_th * b_th),
    ]
    unary_ops = [
        ("relu", lambda: fnn.relu(a_fnn), lambda: torch.nn.functional.relu(a_th)),
        ("gelu", lambda: fnn.gelu(a_fnn), lambda: torch.nn.functional.gelu(a_th)),
        ("sigmoid", lambda: fnn.sigmoid(a_fnn), lambda: torch.sigmoid(a_th)),
        ("tanh", lambda: fnn.tanh(a_fnn), lambda: torch.tanh(a_th)),
        ("silu", lambda: fnn.silu(a_fnn), lambda: torch.nn.functional.silu(a_th)),
        ("exp", lambda: fnn.exp(a_fnn), lambda: torch.exp(a_th)),
        ("neg", lambda: fnn.neg(a_fnn), lambda: -a_th),
    ]
    for name, fn_fnn, fn_th in ops + unary_ops:
        results.append(_bench(name, fn_fnn, fn_th, _WARMUP, n))
    return results


def bench_softmax_reductions(iters):
    results = []
    n = iters.get("tiny", 500)
    size = 1024
    a_fnn = fnn.randn([size, size])
    a_th = torch.randn(size, size)

    results.append(
        _bench(
            "softmax",
            lambda: fnn.softmax(a_fnn, dim=1),
            lambda: torch.softmax(a_th, dim=1),
            _WARMUP,
            n,
        )
    )
    results.append(
        _bench(
            "sum",
            lambda: fnn.sum(a_fnn, dim=1),
            lambda: a_th.sum(dim=1),
            _WARMUP,
            n,
        )
    )
    results.append(
        _bench(
            "mean",
            lambda: fnn.mean(a_fnn, dim=1),
            lambda: a_th.mean(dim=1),
            _WARMUP,
            n,
        )
    )
    return results


def bench_linear(iters):
    n = iters.get("small", 200)
    batch, in_features, out_features = 32, 512, 256
    fnn_lin = fnn.Linear(in_features, out_features)
    x_fnn = fnn.randn([batch, in_features])
    th_lin = torch.nn.Linear(in_features, out_features)
    x_th = torch.randn(batch, in_features)
    return [
        _bench(
            "linear_512x256",
            lambda: fnn_lin(x_fnn),
            lambda: th_lin(x_th),
            _WARMUP,
            n,
            batch=batch,
        )
    ]


def bench_conv2d(iters):
    n = iters.get("medium", 100)
    fnn_conv = fnn.Conv2d(3, 16, 3, padding=1)
    x_fnn = fnn.randn([16, 3, 64, 64])
    th_conv = torch.nn.Conv2d(3, 16, 3, padding=1)
    x_th = torch.randn(16, 3, 64, 64)

    results = [
        _bench(
            "conv2d_3x3_3to16",
            lambda: fnn_conv(x_fnn),
            lambda: th_conv(x_th),
            _WARMUP,
            n,
            batch=16,
            in_channels=3,
            out_channels=16,
        )
    ]

    # Larger conv
    fnn_conv2 = fnn.Conv2d(64, 128, 3, padding=1)
    x_fnn2 = fnn.randn([8, 64, 32, 32])
    th_conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
    x_th2 = torch.randn(8, 64, 32, 32)
    results.append(
        _bench(
            "conv2d_3x3_64to128",
            lambda: fnn_conv2(x_fnn2),
            lambda: th_conv2(x_th2),
            _WARMUP,
            n,
            batch=8,
            in_channels=64,
            out_channels=128,
        )
    )
    return results


def bench_normalization(iters):
    results = []
    n = iters.get("medium", 100)

    # LayerNorm
    norm_shape = 1024
    fnn_ln = fnn.LayerNorm(norm_shape)
    x_fnn = fnn.randn([32, norm_shape])
    th_ln = torch.nn.LayerNorm(norm_shape)
    x_th = torch.randn(32, norm_shape)
    results.append(
        _bench(
            "layernorm_1024",
            lambda: fnn_ln(x_fnn),
            lambda: th_ln(x_th),
            _WARMUP,
            n,
        )
    )

    # BatchNorm2d
    fnn_bn = fnn.BatchNorm2d(64)
    x_fnn_bn = fnn.randn([16, 64, 32, 32])
    th_bn = torch.nn.BatchNorm2d(64)
    x_th_bn = torch.randn(16, 64, 32, 32)
    results.append(
        _bench(
            "batchnorm2d_64ch",
            lambda: fnn_bn(x_fnn_bn),
            lambda: th_bn(x_th_bn),
            _WARMUP,
            n,
        )
    )
    return results


def bench_training_step(iters):
    n = iters.get("train", 30)
    in_dim, hidden, out_dim = 128, 64, 10
    batch = 32

    # fastnn
    fnn_model = fnn.Sequential([
        fnn.Linear(in_dim, hidden),
        fnn.ReLU(),
        fnn.Linear(hidden, out_dim),
    ])
    fnn_opt = fnn.Adam(fnn_model.parameters(), lr=1e-3)
    x_fnn = fnn.randn([batch, in_dim])
    y_fnn = fnn.randint(0, out_dim, [batch])

    # torch
    import torch.nn as nn

    th_model = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )
    th_opt = torch.optim.Adam(th_model.parameters(), lr=1e-3)
    x_th = torch.randn(batch, in_dim)
    y_th = torch.randint(0, out_dim, (batch,))

    def fnn_step():
        fnn_opt.zero_grad()
        pred = fnn_model(x_fnn)
        loss = fnn.cross_entropy_loss(pred, y_fnn)
        loss.backward()
        fnn_opt.step()

    def th_step():
        th_opt.zero_grad()
        pred = th_model(x_th)
        loss = nn.functional.cross_entropy(pred, y_th)
        loss.backward()
        th_opt.step()

    return [
        _bench(
            "train_step_mlp",
            fnn_step,
            th_step,
            _WARMUP,
            n,
            batch=batch,
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
        )
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_BENCHMARKS = [
    ("MatMul", bench_matmul),
    ("Elementwise", bench_elementwise),
    ("Softmax & Reductions", bench_softmax_reductions),
    ("Linear", bench_linear),
    ("Conv2d", bench_conv2d),
    ("Normalization", bench_normalization),
    ("Training Step", bench_training_step),
]

# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _color(s: float) -> str:
    if s >= 1.5:
        return f"{_GREEN}{s:.2f}x{_RESET}"
    elif s >= 1.0:
        return f"{_YELLOW}{s:.2f}x{_RESET}"
    else:
        return f"{_RED}{s:.2f}x{_RESET}"


def _print_cat(results: List[BenchResult], title: str):
    if not results:
        return
    print(f"\n  {_BOLD}{title}{_RESET}")
    header = f"  {'Operation':<30} {'fastnn (ms)':>16} {'PyTorch (ms)':>16} {'Speedup':>10}"
    sep = f"  {'-'*30} {'-'*16} {'-'*16} {'-'*10}"
    print(header)
    print(sep)
    for r in results:
        op = r.op[:29]
        fnn_s = f"{r.fastnn_ms:>8.3f} ±{r.fastnn_std:<5.3f}"
        th_s = f"{r.torch_ms:>8.3f} ±{r.torch_std:<5.3f}"
        print(f"  {op:<30} {fnn_s:>16} {th_s:>16} {_color(r.speedup):>10}")


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _md_cat(results: List[BenchResult], title: str) -> str:
    if not results:
        return ""
    lines = [
        f"### {title}",
        "",
        f"| {'Operation':<30} | {'fastnn (ms)':<18} | {'PyTorch (ms)':<18} | {'Speedup':<10} |",
        f"| {'-'*30} | {'-'*18} | {'-'*18} | {'-'*10} |",
    ]
    for r in results:
        op = r.op[:29]
        fnn_s = f"{r.fastnn_ms:.3f} ± {r.fastnn_std:.3f}"
        th_s = f"{r.torch_ms:.3f} ± {r.torch_std:.3f}"
        sp = f"{r.speedup:.2f}x"
        lines.append(f"| {op:<30} | {fnn_s:<18} | {th_s:<18} | {sp:<10} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark fastnn vs PyTorch")
    parser.add_argument("--output", type=str, help="Path to write JSON results")
    parser.add_argument("--markdown", action="store_true", help="Print markdown (for CI)")
    parser.add_argument("--fast", action="store_true", help="Fewer iterations (CI check)")
    args = parser.parse_args()

    iters = _ITER_FAST if args.fast else _ITER_DEFAULTS

    print(f"{_BOLD}fastnn vs PyTorch — Speed & Latency Comparison{_RESET}")
    print(f"  Mode: {'fast' if args.fast else 'standard'}")
    print(f"  Warmup: {_WARMUP} iters")
    print()

    all_results: List[BenchResult] = []
    md_sections = []

    for title, bench_fn in _ALL_BENCHMARKS:
        try:
            cat_results = bench_fn(iters)
        except Exception as e:
            print(f"\n  {_RED}SKIPPED {title}: {e}{_RESET}")
            continue
        _print_cat(cat_results, title)
        if args.markdown:
            md_sections.append(_md_cat(cat_results, title))
        all_results.extend(cat_results)

    if not all_results:
        print(f"\n  {_RED}No benchmarks completed successfully{_RESET}")
        sys.exit(1)

    # Overall summary
    total_fnn = sum(r.fastnn_ms for r in all_results)
    total_th = sum(r.torch_ms for r in all_results)
    avg_speedup = total_th / total_fnn if total_fnn > 0 else 0

    print(f"\n  {_BOLD}Overall Summary{_RESET}")
    print(f"  Total fastnn:  {total_fnn:.2f} ms")
    print(f"  Total PyTorch: {total_th:.2f} ms")
    print(f"  Avg speedup:   {avg_speedup:.2f}x")

    # Build markdown (always, for JSON output)
    md_header = "## ⚡ fastnn vs PyTorch Benchmark Comparison\n"
    md_summary = (
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Total fastnn | {total_fnn:.2f} ms |\n"
        f"| Total PyTorch | {total_th:.2f} ms |\n"
        f"| Avg Speedup | {avg_speedup:.2f}x |\n\n"
    )
    md_body = "\n".join(s for s in md_sections)
    full_md = md_header + md_summary + md_body

    if args.markdown:
        print(f"\n{_YELLOW}--- Markdown (copy for CI step summary) ---{_RESET}\n")
        print(full_md)

    # JSON
    if args.output:
        data = {
            "metadata": {
                "mode": "fast" if args.fast else "standard",
                "warmup": _WARMUP,
                "num_benchmarks": len(all_results),
            },
            "summary": {
                "total_fastnn_ms": round(total_fnn, 3),
                "total_torch_ms": round(total_th, 3),
                "avg_speedup": round(avg_speedup, 3),
            },
            "results": [asdict(r) for r in all_results],
            "markdown": full_md,
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Results written to {args.output}")

    # Warn if fastnn is slower overall (non-fatal - CI runners are noisy)
    if avg_speedup < 1.0:
        print(f"\n  {_RED}WARNING: fastnn is slower than PyTorch overall{_RESET}")
    else:
        print(f"\n  {_GREEN}fastnn is {avg_speedup:.2f}x faster than PyTorch overall{_RESET}")

    print(f"\n  {_GREEN}Benchmarks complete{_RESET}")


if __name__ == "__main__":
    main()
