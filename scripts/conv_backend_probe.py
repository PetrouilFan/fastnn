#!/usr/bin/env python3
"""Probe optional Conv backends for YOLO CPU shapes.

Roadmap Task C1: compare fastnn/OpenBLAS Conv+SiLU isolated timings against
PyTorch CPU Conv2d+SiLU (oneDNN/MKLDNN when PyTorch enables it). This is a
benchmark/probe only; it does not integrate any backend into fastnn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SHAPES = [
    {"name": "stem_f16_c3_sp25600", "n": 1, "c": 3, "h": 320, "w": 320, "f": 16, "kh": 3, "kw": 3, "stride": 2, "padding": 1},
    {"name": "yolo_f32_c16_sp6400", "n": 1, "c": 16, "h": 80, "w": 80, "f": 32, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
    {"name": "yolo_f32_c32_sp1600", "n": 1, "c": 32, "h": 40, "w": 40, "f": 32, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
    {"name": "yolo_f64_c64_sp400", "n": 1, "c": 64, "h": 20, "w": 20, "f": 64, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
    {"name": "yolo_f64_c64_sp1600", "n": 1, "c": 64, "h": 40, "w": 40, "f": 64, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
    {"name": "yolo_f80_c80_sp1600", "n": 1, "c": 80, "h": 40, "w": 40, "f": 80, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
    {"name": "yolo_f128_c128_sp100", "n": 1, "c": 128, "h": 10, "w": 10, "f": 128, "kh": 3, "kw": 3, "stride": 1, "padding": 1},
]

FASTNN_RE = re.compile(
    r"^(?P<name>\S+)\s+gemm=\((?P<m>\d+),(?P<k>\d+),(?P<spatial>\d+)\)\s+"
    r"total=(?P<total>[0-9.]+)ms\s+phased=(?P<phased>[0-9.]+)ms\s+"
    r"im2col=(?P<im2col>[0-9.]+)ms\(\s*(?P<im2col_pct>[0-9.]+)%\)\s+"
    r"gemm=(?P<gemm>[0-9.]+)ms\(\s*(?P<gemm_pct>[0-9.]+)%\)\s+"
    r"silu=(?P<silu>[0-9.]+)ms\(\s*(?P<silu_pct>[0-9.]+)%\)"
)


def _bench_torch_shape(shape: dict[str, Any], *, warmup: int, iters: int, threads: int, channels_last: bool) -> dict[str, Any]:
    getattr(torch, "set_num_threads")(threads)
    getattr(torch, "manual_seed")(0)
    torch_float32 = getattr(torch, "float32")
    x = torch.randn((shape["n"], shape["c"], shape["h"], shape["w"]), dtype=torch_float32)
    w = torch.randn((shape["f"], shape["c"], shape["kh"], shape["kw"]), dtype=torch_float32)
    b = torch.randn((shape["f"],), dtype=torch_float32)
    if channels_last:
        torch_channels_last = getattr(torch, "channels_last")
        x = x.contiguous(memory_format=torch_channels_last)
        w = w.contiguous(memory_format=torch_channels_last)
    y = None
    with torch.no_grad():
        for _ in range(warmup):
            y = F.silu(F.conv2d(x, w, b, stride=shape["stride"], padding=shape["padding"]))
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = F.silu(F.conv2d(x, w, b, stride=shape["stride"], padding=shape["padding"]))
            times.append((time.perf_counter() - t0) * 1000.0)
    assert y is not None
    return {
        "name": shape["name"],
        "backend": "torch_conv2d_silu_channels_last" if channels_last else "torch_conv2d_silu_contiguous",
        "threads": threads,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "output_shape": list(y.shape),
    }


def _run_fastnn_phase(args: argparse.Namespace) -> dict[str, Any]:
    cmd = ["cargo", "run", "--release"]
    if args.fastnn_openblas:
        cmd.extend(["--features", "openblas"])
    cmd.extend(["--example", "conv_phase_bench", "--", str(args.fastnn_iters)])
    env = os.environ.copy()
    if args.openblas_threads is not None:
        env["OPENBLAS_NUM_THREADS"] = str(args.openblas_threads)
    proc = subprocess.run(cmd, cwd=args.cwd, env=env, text=True, capture_output=True)
    rows = []
    for line in proc.stdout.splitlines():
        m = FASTNN_RE.match(line.strip())
        if not m:
            continue
        gd = m.groupdict()
        rows.append({
            "name": gd["name"],
            "backend": "fastnn_conv_phase_openblas" if args.fastnn_openblas else "fastnn_conv_phase_matrixmultiply",
            "total_ms": float(gd["total"]),
            "phased_ms": float(gd["phased"]),
            "im2col_ms": float(gd["im2col"]),
            "gemm_ms": float(gd["gemm"]),
            "silu_ms": float(gd["silu"]),
            "gemm": [int(gd["m"]), int(gd["k"]), int(gd["spatial"])],
        })
    return {
        "ok": proc.returncode == 0 and bool(rows),
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50, help="Torch iterations per shape")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", default="1,2,4", help="Comma-separated torch thread counts")
    ap.add_argument("--fastnn-iters", type=int, default=50)
    ap.add_argument("--fastnn-openblas", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--openblas-threads", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-fastnn", action="store_true")
    args = ap.parse_args()
    args.cwd = Path.cwd()

    thread_counts = [int(x.strip()) for x in args.threads.split(",") if x.strip()]
    torch_rows = []
    for threads in thread_counts:
        for channels_last in (False, True):
            for shape in SHAPES:
                row = _bench_torch_shape(shape, warmup=args.warmup, iters=args.iters, threads=threads, channels_last=channels_last)
                torch_rows.append(row)
                print(f"{row['backend']} threads={threads} {shape['name']} mean_ms={row['mean_ms']:.4f}")

    fastnn = None if args.skip_fastnn else _run_fastnn_phase(args)
    fastnn_by_name = {r["name"]: r for r in (fastnn or {}).get("rows", [])}

    best_torch: dict[str, dict[str, Any]] = {}
    for shape in SHAPES:
        candidates = [r for r in torch_rows if r["name"] == shape["name"]]
        best_torch[shape["name"]] = min(candidates, key=lambda r: r["mean_ms"])

    comparisons = []
    for name, torch_row in best_torch.items():
        fast = fastnn_by_name.get(name)
        comp = {"name": name, "best_torch": torch_row, "fastnn": fast}
        if fast:
            comp["torch_vs_fastnn_total_ratio"] = torch_row["mean_ms"] / fast["total_ms"]
            comp["torch_beats_fastnn_by_pct"] = (1.0 - torch_row["mean_ms"] / fast["total_ms"]) * 100.0
        comparisons.append(comp)

    print("best comparisons")
    for comp in comparisons:
        fast = comp.get("fastnn")
        if fast:
            print(
                f"  {comp['name']}: fastnn_total={fast['total_ms']:.4f}ms "
                f"best_torch={comp['best_torch']['mean_ms']:.4f}ms "
                f"backend={comp['best_torch']['backend']} threads={comp['best_torch']['threads']} "
                f"torch_beats_by={comp['torch_beats_fastnn_by_pct']:.1f}%"
            )
        else:
            print(f"  {comp['name']}: fastnn unavailable best_torch={comp['best_torch']['mean_ms']:.4f}ms")

    result = {
        "config": {
            "iters": args.iters,
            "warmup": args.warmup,
            "threads": thread_counts,
            "fastnn_iters": args.fastnn_iters,
            "fastnn_openblas": args.fastnn_openblas,
            "openblas_threads": args.openblas_threads,
            "torch_mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "torch_version": torch.__version__,
        },
        "torch_rows": torch_rows,
        "fastnn": fastnn,
        "comparisons": comparisons,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"json {args.json}")
    return 0 if fastnn is None or fastnn.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
