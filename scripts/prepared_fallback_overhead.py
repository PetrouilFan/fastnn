#!/usr/bin/env python3
"""Benchmark fastnn prepared fallback overhead on a YOLO ONNX model.

This tool compares the default AOT forward path against prepared-plan opt-in
fallback paths on the same input. It is intended to quantify whether prepared
runtime scaffolding is cheap enough before adding kernel/layout substitutions.

It requires fastnn to be built with the `prepared-plan` feature.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import numpy as np

import fastnn as fnn
from scripts.yolo_compare_fastnn_pytorch import _make_fastnn_executor


def _time_method(
    name: str,
    method: Callable,
    inputs: dict,
    output_name: str,
    baseline: np.ndarray,
    warmup: int,
    iters: int,
) -> dict:
    for _ in range(warmup):
        method(inputs)

    times: list[float] = []
    last = None
    for _ in range(iters):
        start = time.perf_counter()
        out = method(inputs)
        times.append((time.perf_counter() - start) * 1000.0)
        last = out[output_name].numpy()

    assert last is not None
    diff = np.abs(baseline - last)
    return {
        "name": name,
        "iters": iters,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "max_abs_vs_forward": float(diff.max()),
        "mean_abs_vs_forward": float(diff.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=Path("yolov8n.onnx"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    executor, input_name, output_name = _make_fastnn_executor(args.onnx)
    x = np.random.default_rng(args.seed).standard_normal((1, 3, 320, 320), dtype=np.float32)
    inputs = {input_name: fnn.tensor(x, list(x.shape))}

    baseline = executor.forward(inputs)[output_name].numpy()
    methods: list[tuple[str, Callable]] = [("forward", executor.forward)]

    for name in ("forward_prepared_fallback", "forward_prepared_arena_fallback"):
        if hasattr(executor, name):
            methods.append((name, getattr(executor, name)))

    results = [
        _time_method(name, method, inputs, output_name, baseline, args.warmup, args.iters)
        for name, method in methods
    ]

    by_name = {item["name"]: item for item in results}
    forward_mean = by_name["forward"]["mean_ms"]
    for item in results:
        item["overhead_vs_forward_ms"] = item["mean_ms"] - forward_mean
        item["overhead_vs_forward_pct"] = (
            100.0 * (item["mean_ms"] - forward_mean) / forward_mean if forward_mean else 0.0
        )

    payload = {
        "model": str(args.onnx),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    print("prepared fallback overhead")
    print(f"  model: {args.onnx}")
    print(f"  warmup: {args.warmup}")
    print(f"  iters: {args.iters}")
    for item in results:
        print(
            "  {name}: mean={mean_ms:.3f} ms median={median_ms:.3f} ms "
            "overhead={overhead_vs_forward_ms:+.3f} ms ({overhead_vs_forward_pct:+.2f}%) "
            "max_abs={max_abs_vs_forward:.6g}".format(**item)
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
