#!/usr/bin/env python3
"""Report model-agnostic compiled-graph memory efficiency statistics.

This script compiles an ONNX model through fastnn.AotExecutor and calls
AotExecutor.memory_stats().  Unlike the YOLO-specific profiling scripts, the
metrics are static compiler/runtime properties: arena pressure, slot reuse,
physical copy bytes, WriteConst bytes, and instruction mix.  Use it to find
broad memory/layout optimisation targets before choosing a kernel/backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_make_executor():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from yolo_compare_fastnn_pytorch import _make_fastnn_executor

    return _make_fastnn_executor


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _print_summary(model: Path, stats: dict[str, Any]) -> None:
    arena = int(stats.get("arena_size", 0))
    logical = int(stats.get("logical_slot_bytes", 0))
    physical = int(stats.get("physical_slot_bytes", 0))
    saved = int(stats.get("slot_reuse_saved_bytes", 0))
    traffic = int(stats.get("estimated_static_traffic_bytes", 0))
    kernel_read = int(stats.get("kernel_read_bytes", 0))
    kernel_write = int(stats.get("kernel_write_bytes", 0))
    memcpy = int(stats.get("memcpy_bytes", 0))
    write_const = int(stats.get("write_const_bytes", 0))
    fill = int(stats.get("fill_bytes", 0))

    reuse_ratio = (saved / logical * 100.0) if logical else 0.0
    traffic_vs_arena = (traffic / arena) if arena else 0.0

    print("Graph memory stats")
    print(f"  model: {model}")
    print(f"  instructions: {int(stats.get('instructions', 0))}")
    print(f"  arena size: {_fmt_bytes(arena)}")
    print(f"  logical slot bytes: {_fmt_bytes(logical)}")
    print(f"  physical slot bytes by offset: {_fmt_bytes(physical)}")
    print(f"  slot reuse saved: {_fmt_bytes(saved)} ({reuse_ratio:.1f}% of logical)")
    print(
        "  alias reuse: "
        f"{int(stats.get('alias_groups', 0))} groups, "
        f"{int(stats.get('aliased_nodes', 0))} aliased nodes"
    )
    print(f"  estimated static traffic: {_fmt_bytes(traffic)} ({traffic_vs_arena:.2f}x arena)")
    print(f"    kernel reads: {_fmt_bytes(kernel_read)}")
    print(f"    kernel writes: {_fmt_bytes(kernel_write)}")
    print(f"    memcpy: {_fmt_bytes(memcpy)}")
    print(f"    write_const: {_fmt_bytes(write_const)}")
    print(f"    fill: {_fmt_bytes(fill)}")
    print(
        "  instruction mix: "
        f"call_kernel={int(stats.get('call_kernel_count', 0))}, "
        f"memcpy={int(stats.get('memcpy_count', 0))}, "
        f"write_const={int(stats.get('write_const_count', 0))}, "
        f"fill={int(stats.get('fill_count', 0))}"
    )

    kernels = stats.get("top_kernels_by_count", [])
    if kernels:
        print("  top kernels by count:")
        for row in kernels[:10]:
            print(f"    {row['kernel']}: {row['count']}")

    instructions = stats.get("top_instructions_by_static_bytes", [])
    if instructions:
        print("  top instructions by static bytes:")
        for row in instructions[:10]:
            node = ""
            if row.get("node_id") is not None:
                node = f" node={int(row['node_id'])}"
            if row.get("node_name"):
                node += f" name={row['node_name']}"
            shape = ""
            if row.get("input_shapes") and row.get("output_shape"):
                shape = f" inputs={row['input_shapes']} output={row['output_shape']}"
            print(
                f"    #{int(row['instruction_index'])} {row['kernel_name']} "
                f"({row['kind']}): {_fmt_bytes(int(row['static_bytes']))}"
                f"{node}{shape}"
            )

    write_consts = stats.get("top_write_consts_by_size", [])
    if write_consts:
        print("  largest WriteConst instructions:")
        for row in write_consts[:10]:
            prepared = ""
            if row.get("prepared_static_role"):
                prepared = (
                    f" prepared={row['prepared_static_role']}"
                    f" consumer=#{int(row['prepared_consumer_instruction_index'])}"
                    f" input={int(row['prepared_input_index'])}"
                    f" constant={row['prepared_constant_name']}"
                )
            print(
                f"    #{int(row['instruction_index'])}: "
                f"{_fmt_bytes(int(row['write_bytes']))} "
                f"dst_offset={int(row['dst_offset'])}"
                f"{prepared}"
            )

    aliases = stats.get("top_alias_groups", [])
    if aliases:
        print("  largest alias groups:")
        for row in aliases[:5]:
            nodes = row.get("nodes", [])
            preview = ",".join(str(n) for n in nodes[:8])
            suffix = "..." if len(nodes) > 8 else ""
            print(
                f"    offset={row['offset']} size={_fmt_bytes(int(row['size']))} "
                f"nodes=[{preview}{suffix}]"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Print compiled-graph memory/traffic stats for an ONNX model.",
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
        help="Optional path; write the raw memory_stats() dict here as JSON.",
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

    if not hasattr(executor, "memory_stats"):
        print(
            "error: AotExecutor has no 'memory_stats' method; rebuild the Python extension.",
            file=sys.stderr,
        )
        return 5

    try:
        raw_stats = executor.memory_stats()
    except Exception as exc:
        print(f"error: memory_stats() failed: {exc}", file=sys.stderr)
        return 6

    stats = dict(raw_stats)
    _print_summary(onnx_path, stats)

    if args.json_out is not None:
        try:
            _atomic_write_json(Path(args.json_out), stats)
        except Exception as exc:
            print(f"error: failed to write JSON to {args.json_out}: {exc}", file=sys.stderr)
            return 7

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
