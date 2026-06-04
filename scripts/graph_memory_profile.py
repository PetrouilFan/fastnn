#!/usr/bin/env python3
"""Overlay fastnn runtime profile timings with compiled-graph memory stats.

The profiler is intentionally model-agnostic: it compiles an ONNX graph through
fastnn.AotExecutor, collects static memory traffic from ``memory_stats()``, runs
one instrumented ``profile()`` pass, and emits JSON that ranks kernels by elapsed
time plus estimated static bytes-per-ms.  Static memory_stats currently exposes
kernel traffic by aggregate type and kernel counts, so per-kernel bytes are an
estimate apportioned by instruction count rather than exact per-node accounting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load_make_executor():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from yolo_compare_fastnn_pytorch import _make_fastnn_executor

    return _make_fastnn_executor


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _coerce_elapsed_ns(entry: dict[str, Any]) -> float:
    return float(entry.get("elapsed_ns", entry.get("elapsed", 0.0)))


def build_memory_profile(
    memory_stats: dict[str, Any],
    profile_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Combine ``memory_stats()`` output with raw ``profile()`` entries.

    Per-kernel static traffic is estimated from aggregate kernel read/write bytes
    apportioned by the static call count per kernel.  Copy/fill/write-constant
    bytes remain separate because they are not emitted as profiled kernels today.
    """

    profiled: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_ms": 0.0})
    for entry in profile_entries:
        kernel = str(entry.get("kernel_name", "<unknown>"))
        profiled[kernel]["count"] += 1.0
        profiled[kernel]["total_ms"] += _coerce_elapsed_ns(entry) / 1_000_000.0

    static_counts = {
        str(row.get("kernel", row.get("kernel_name", "<unknown>"))): int(row.get("count", 0))
        for row in memory_stats.get("top_kernels_by_count", [])
    }
    if not static_counts:
        static_counts = {kernel: int(max(1.0, values["count"])) for kernel, values in profiled.items()}

    kernel_static_total = int(memory_stats.get("kernel_read_bytes", 0)) + int(
        memory_stats.get("kernel_write_bytes", 0)
    )
    total_static_calls = sum(static_counts.values())

    special_static_bytes = {
        "memcopy": int(memory_stats.get("memcpy_bytes", 0)),
        "memcpy": int(memory_stats.get("memcpy_bytes", 0)),
        "write_const": int(memory_stats.get("write_const_bytes", 0)),
        "fill": int(memory_stats.get("fill_bytes", 0)),
    }

    kernel_rows: list[dict[str, Any]] = []
    profiled_special_kinds: set[str] = set()
    for kernel, values in profiled.items():
        static_count = static_counts.get(kernel, int(max(1.0, values["count"])))
        if kernel in special_static_bytes:
            static_bytes = special_static_bytes[kernel]
            if kernel in {"memcopy", "memcpy"}:
                profiled_special_kinds.add("memcpy")
            else:
                profiled_special_kinds.add(kernel)
        else:
            static_bytes = int(round(kernel_static_total * static_count / total_static_calls)) if total_static_calls else 0
        total_ms = float(values["total_ms"])
        mean_ms = total_ms / values["count"] if values["count"] else 0.0
        bytes_per_ms = static_bytes / total_ms if total_ms > 0.0 else 0.0
        kernel_rows.append(
            {
                "kernel_name": kernel,
                "profile_count": int(values["count"]),
                "static_count": static_count,
                "total_ms": total_ms,
                "mean_ms": mean_ms,
                "static_bytes": static_bytes,
                "bytes_per_ms": bytes_per_ms,
                "suspected_memory_bound": static_bytes > 0 and (mean_ms <= 1.0 or bytes_per_ms >= 1_000_000),
            }
        )

    kernel_rows.sort(key=lambda row: (row["static_bytes"], row["total_ms"]), reverse=True)

    unprofiled = [
        {"kind": "memcpy", "bytes": int(memory_stats.get("memcpy_bytes", 0))},
        {"kind": "write_const", "bytes": int(memory_stats.get("write_const_bytes", 0))},
        {"kind": "fill", "bytes": int(memory_stats.get("fill_bytes", 0))},
    ]
    unprofiled = [row for row in unprofiled if row["bytes"] and row["kind"] not in profiled_special_kinds]

    profiled_static = sum(int(row["static_bytes"]) for row in kernel_rows)
    profiled_kernel_static = sum(
        int(row["static_bytes"])
        for row in kernel_rows
        if row["kernel_name"] not in special_static_bytes
    )
    estimated_static = int(memory_stats.get("estimated_static_traffic_bytes", 0))

    return {
        "summary": {
            "profiled_total_ms": sum(float(row["total_ms"]) for row in kernel_rows),
            "estimated_static_traffic_bytes": estimated_static,
            "profiled_static_traffic_bytes": profiled_static,
            "profiled_kernel_static_bytes": profiled_kernel_static,
            "unprofiled_static_traffic_bytes": sum(int(row["bytes"]) for row in unprofiled),
            "bytes_per_profiled_ms": profiled_static / max(sum(float(row["total_ms"]) for row in kernel_rows), 1.0e-12),
            "arena_size": int(memory_stats.get("arena_size", 0)),
        },
        "kernels": kernel_rows,
        "unprofiled_static_traffic": unprofiled,
        "memory_stats": memory_stats,
    }


def _input_shape_from_executor(executor: Any, input_name: str, user_shape: str | None) -> list[int]:
    if user_shape:
        return [int(part) for part in user_shape.split(",") if part]
    stats = executor.memory_stats()
    _ = stats  # Keeps this helper explicit: memory_stats does not expose input shape today.
    raise ValueError("--input-shape is required for standalone graph memory profiling")


def _print_summary(model: Path, payload: dict[str, Any], top: int) -> None:
    summary = payload["summary"]
    print("Graph memory profile")
    print(f"  model: {model}")
    print(f"  profiled total: {summary['profiled_total_ms']:.3f} ms")
    print(f"  estimated static traffic: {_fmt_bytes(int(summary['estimated_static_traffic_bytes']))}")
    print(f"  profiled kernel static traffic: {_fmt_bytes(int(summary['profiled_kernel_static_bytes']))}")
    print(f"  unprofiled copy/const traffic: {_fmt_bytes(int(summary['unprofiled_static_traffic_bytes']))}")
    print("  top kernels by estimated static bytes:")
    for row in payload["kernels"][:top]:
        marker = " memory-bound?" if row["suspected_memory_bound"] else ""
        print(
            f"    {row['kernel_name']}: {row['total_ms']:.3f} ms, "
            f"{_fmt_bytes(int(row['static_bytes']))}, {row['bytes_per_ms'] / (1024 * 1024):.2f} MiB/ms{marker}"
        )
    if payload["unprofiled_static_traffic"]:
        print("  unprofiled static traffic:")
        for row in payload["unprofiled_static_traffic"]:
            print(f"    {row['kind']}: {_fmt_bytes(int(row['bytes']))}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile ONNX graph timing overlaid with fastnn memory traffic stats.")
    ap.add_argument("--onnx", default="yolov8n.onnx")
    ap.add_argument("--input-shape", default="1,3,320,320", help="Comma-separated input tensor shape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", dest="json_out", default=None, help="Write combined profile JSON to this path")
    args = ap.parse_args(argv)

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"error: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2

    try:
        make_executor = _load_make_executor()
        executor, input_name, _output_name = make_executor(onnx_path)
        input_shape = _input_shape_from_executor(executor, input_name, args.input_shape)
        rng = np.random.default_rng(args.seed)
        x = rng.random(tuple(input_shape), dtype=np.float32)
        import fastnn as fnn

        fx = fnn.tensor(x, input_shape)
        memory_stats = dict(executor.memory_stats())
        prof_result = executor.profile({input_name: fx})
        payload = build_memory_profile(memory_stats, list(prof_result["profile"]))
    except Exception as exc:
        print(f"error: graph memory profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    _print_summary(onnx_path, payload, args.top)
    if args.json_out is not None:
        try:
            _atomic_write_json(Path(args.json_out), payload)
        except Exception as exc:
            print(f"error: failed to write JSON to {args.json_out}: {exc}", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
