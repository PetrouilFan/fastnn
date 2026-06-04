#!/usr/bin/env python3
"""Report the prepared-plan composition of a YOLO-style ONNX model.

Compiles the model through `fastnn.AotExecutor` (sharing the same ONNX
loading logic as `scripts/yolo_compare_fastnn_pytorch.py`) and calls
`executor.prepared_stats()` to print a small per-instruction-kind summary.

Requires the Python extension to have been built with
`maturin develop --features prepared-plan`. If the feature is missing,
`prepared_stats` will be absent (or raise) and this script exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_make_executor():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from yolo_compare_fastnn_pytorch import _make_fastnn_executor
    return _make_fastnn_executor


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
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


def _print_summary(onnx_path: Path, stats: dict) -> None:
    total = int(stats.get("total", 0))
    conv2d = int(stats.get("conv2d", 0))
    matmul = int(stats.get("matmul", 0))
    generic = int(stats.get("generic", 0))
    static_weight_bindings = int(stats.get("static_weight_bindings", 0))
    constant_arena_entries = int(stats.get("constant_arena_entries", 0))
    constant_arena_bytes = int(stats.get("constant_arena_bytes", 0))
    packed_fp32_conv_candidates = int(stats.get("packed_fp32_conv_candidates", 0))
    packed_fp32_conv_candidate_flops = int(stats.get("packed_fp32_conv_candidate_flops", 0))
    transposed_fp32_conv_entries = int(stats.get("transposed_fp32_conv_entries", 0))
    transposed_fp32_conv_bytes = int(stats.get("transposed_fp32_conv_bytes", 0))
    print("YOLO prepared stats")
    print(f"  model: {onnx_path}")
    print(f"  total instructions: {total}")
    print(f"  conv2d: {conv2d}")
    print(f"  matmul: {matmul}")
    print(f"  generic (other): {generic}")
    print(f"  static weight bindings: {static_weight_bindings}")
    print(f"  constant arena entries: {constant_arena_entries}")
    print(f"  constant arena bytes: {constant_arena_bytes}")
    print(f"  packed fp32 conv candidates: {packed_fp32_conv_candidates}")
    print(f"  packed fp32 conv candidate flops: {packed_fp32_conv_candidate_flops}")
    print(f"  transposed fp32 conv entries: {transposed_fp32_conv_entries}")
    print(f"  transposed fp32 conv bytes: {transposed_fp32_conv_bytes}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Print prepared-plan instruction counts for a YOLO ONNX model.",
    )
    ap.add_argument(
        "--onnx",
        default="yolov8n.onnx",
        help="Path to the ONNX model (default: yolov8n.onnx in CWD).",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="Optional path; write the raw prepared_stats() dict here as JSON.",
    )
    args = ap.parse_args(argv)

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"error: ONNX file not found: {onnx_path}", file=sys.stderr)
        return 2

    try:
        make_executor = _load_make_executor()
    except Exception as exc:
        print(f"error: failed to import fastnn executor helper: {exc}", file=sys.stderr)
        return 3

    try:
        executor, _input_name, _output_name = make_executor(onnx_path)
    except Exception as exc:
        print(f"error: failed to build AotExecutor from {onnx_path}: {exc}", file=sys.stderr)
        return 4

    if not hasattr(executor, "prepared_stats"):
        print(
            "error: AotExecutor has no 'prepared_stats' method; "
            "rebuild the Python extension with --features prepared-plan.",
            file=sys.stderr,
        )
        return 5

    try:
        stats = executor.prepared_stats()
    except Exception as exc:
        print(f"error: prepared_stats() failed: {exc}", file=sys.stderr)
        return 6

    if not isinstance(stats, dict):
        print(f"error: prepared_stats() returned non-dict: {type(stats).__name__}", file=sys.stderr)
        return 7

    _print_summary(onnx_path, stats)

    if args.json_out is not None:
        try:
            _atomic_write_json(Path(args.json_out), dict(stats))
        except Exception as exc:
            print(f"error: failed to write JSON to {args.json_out}: {exc}", file=sys.stderr)
            return 8

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
