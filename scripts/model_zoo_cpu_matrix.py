#!/usr/bin/env python3
"""Run a small CPU model-zoo benchmark and bottleneck classification matrix.

This script is intentionally model-agnostic.  It iterates over a configurable
list of model entries, attempts to (1) download/export a small ONNX artifact
into a local cache, (2) build a ``fastnn.AotExecutor`` from that artifact, and
(3) record export/import status, AOT success/failure, fastnn forward timing,
ONNX Runtime timing, PyTorch timing where applicable, a single profile pass
aggregated by kernel, ``memory_stats()`` summary, instruction-level
hotspots when ``graph_memory_profile`` helpers support them, and an accuracy
check (max-abs vs the PyTorch reference) when both backends succeed.

Failures are recorded per model and never abort the rest of the matrix so
that one broken import does not hide findings from the rest of the zoo.

Usage::

    .venv/bin/python scripts/model_zoo_cpu_matrix.py \
        --json /tmp/fastnn_model_zoo_cpu_matrix.json \
        --models yolov8n,yolo11n,resnet18,resnet50

Output is a JSON document keyed by model with one record each.  The CLI
also prints a small human-readable summary at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Generic AOT executor builder
# ---------------------------------------------------------------------------


def _format_attr_value(value: Any) -> str:
    """Render an attribute value as a string for the AotExecutor node dict."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return ",".join(
            str(int(v)) if float(v).is_integer() else str(float(v))
            for v in value.flatten()
        )
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _build_aot_executor(
    onnx_path: Path, family: str | None = None
) -> tuple[Any, str, str]:
    """Build a ``fastnn.AotExecutor`` from an ONNX file.

    YOLO graphs reuse the existing ``yolo_compare_fastnn_pytorch._make_fastnn_executor``
    helper because the Ultralytics DFL Reshape/Concat path requires the
    Shape/Gather/Add/Sub/Mul/Div constant-folding that helper does and that
    this script does not want to maintain in two places.

    Other model families (torchvision classifiers today) are loaded through
    the in-script ``_build_generic_aot_executor`` helper, which adds a
    Gemm weight transpose for ``transB=1`` so the FC head in legacy
    torchvision ONNX exports feeds the right-shape weight to fastnn's
    MatMul kernel.
    """

    if family == "yolo":
        from yolo_compare_fastnn_pytorch import _make_fastnn_executor

        return _make_fastnn_executor(onnx_path)
    return _build_generic_aot_executor(onnx_path)


def _build_generic_aot_executor(onnx_path: Path) -> tuple[Any, str, str]:
    """Build a ``fastnn.AotExecutor`` for non-YOLO ONNX graphs.

    Today the only addition over a plain node-dict translation is a
    ``Gemm`` weight transpose when ``transB=1``; torchvision's legacy
    PyTorch ONNX export emits a final ``Gemm`` for the FC head with
    ``transB=1`` and weight shape ``(out, in)`` which would otherwise
    feed the wrong-shape weight into fastnn's MatMul kernel.
    """

    import onnx
    from onnx import numpy_helper
    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [
        i.name for i in model.graph.input if i.name not in initializer_names
    ]
    output_names = [o.name for o in model.graph.output]
    if not input_names:
        raise RuntimeError("ONNX model has no non-initializer input")
    if not output_names:
        raise RuntimeError("ONNX model has no outputs")

    input_shapes: dict[str, list[int]] = {}
    for i in model.graph.input:
        if i.name in input_names:
            tensor_type = i.type.tensor_type
            if tensor_type.HasField("shape"):
                input_shapes[i.name] = [
                    int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1
                    for d in tensor_type.shape.dim
                ]

    initializers_by_name = {init.name: init for init in model.graph.initializer}

    params: dict[str, Any] = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        arr = arr_raw.astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))

    nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        attrs = _extract_attrs(node)

        if node.op_type == "Gemm" and len(node.input) >= 2:
            transB = int(attrs.get("transB", 0))
            if transB == 1 and node.input[1] in params and node.input[1] in initializers_by_name:
                w_init = initializers_by_name[node.input[1]]
                w = numpy_helper.to_array(w_init)
                w_t = np.ascontiguousarray(w.T)
                params[node.input[1]] = fnn.tensor(
                    w_t.astype(np.float32, copy=False), list(w_t.shape)
                )
                attrs["transB"] = 0

        item: dict[str, Any] = {
            "name": name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
        }
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            item[k] = _format_attr_value(v)
        nodes.append(item)

    return (
        fnn.AotExecutor(nodes, params, input_names, output_names, input_shapes=input_shapes),
        input_names[0],
        output_names[0],
    )


# ---------------------------------------------------------------------------
# Timing + accuracy helpers
# ---------------------------------------------------------------------------


def _timer(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    times: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "iters": iters,
        "warmup": warmup,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Per-model record construction
# ---------------------------------------------------------------------------


def _tensor_shape(value_info: Any) -> list[int | None]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return []
    return [
        int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else None
        for d in tensor_type.shape.dim
    ]


def _conv_class(groups: int, input_shape: list[int | None], weight_shape: list[int]) -> str:
    out_channels = weight_shape[0] if len(weight_shape) >= 1 else None
    per_group_in = weight_shape[1] if len(weight_shape) >= 2 else None
    in_channels = input_shape[1] if len(input_shape) >= 2 else None
    kh = weight_shape[2] if len(weight_shape) >= 3 else None
    kw = weight_shape[3] if len(weight_shape) >= 4 else None
    if (
        groups > 1
        and in_channels is not None
        and out_channels is not None
        and groups == in_channels
        and groups == out_channels
        and per_group_in == 1
    ):
        return "depthwise"
    if groups > 1:
        return "grouped"
    if kh == 1 and kw == 1:
        return "pointwise"
    return "standard"


def _conv_shape_metadata(model: Any) -> dict[str, Any]:
    """Summarize Conv shape classes from an inferred ONNX graph."""

    import onnx
    from onnx import numpy_helper

    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | None]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[value.name] = _tensor_shape(value)
    initializers = {init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer}

    by_class: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "estimated_flops": 0.0, "examples": []}
    )
    by_shape: dict[str, dict[str, Any]] = {}
    total_flops = 0.0
    conv_count = 0

    for idx, node in enumerate(inferred.graph.node):
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        conv_count += 1
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        groups = int(attrs.get("group", 1))
        weight = initializers.get(node.input[1])
        weight_shape = list(weight.shape) if weight is not None else []
        input_shape = shapes.get(node.input[0], [])
        output_shape = shapes.get(node.output[0], []) if node.output else []
        strides_raw = attrs.get("strides", [])
        pads_raw = attrs.get("pads", [])
        strides = [int(v) for v in strides_raw] if strides_raw is not None else []
        pads = [int(v) for v in pads_raw] if pads_raw is not None else []
        klass = _conv_class(groups, input_shape, weight_shape)
        if len(output_shape) >= 4 and len(weight_shape) >= 4:
            n = output_shape[0] or 1
            out_c = output_shape[1] or weight_shape[0]
            out_h = output_shape[2] or 0
            out_w = output_shape[3] or 0
            k_per_group = weight_shape[1] * weight_shape[2] * weight_shape[3]
            flops = float(2 * n * out_c * out_h * out_w * k_per_group)
        else:
            flops = 0.0
        total_flops += flops
        row = {
            "node_index": idx,
            "name": node.name or f"Conv_{idx}",
            "class": klass,
            "groups": groups,
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "output_shape": output_shape,
            "strides": strides,
            "pads": pads,
            "estimated_flops": flops,
        }
        bucket = by_class[klass]
        bucket["count"] += 1
        bucket["estimated_flops"] += flops
        if len(bucket["examples"]) < 5:
            bucket["examples"].append(row)
        shape_key = json.dumps(
            {
                "class": klass,
                "groups": groups,
                "input": input_shape,
                "weight": weight_shape,
                "output": output_shape,
                "strides": strides,
                "pads": pads,
            },
            sort_keys=True,
        )
        shape_bucket = by_shape.setdefault(
            shape_key,
            {**row, "count": 0, "total_estimated_flops": 0.0},
        )
        shape_bucket["count"] += 1
        shape_bucket["total_estimated_flops"] += flops

    class_rows = {
        k: {
            "count": int(v["count"]),
            "estimated_flops": float(v["estimated_flops"]),
            "flop_fraction": float(v["estimated_flops"] / total_flops) if total_flops else 0.0,
            "examples": v["examples"],
        }
        for k, v in sorted(by_class.items())
    }
    repeated_shapes = sorted(
        by_shape.values(),
        key=lambda r: (float(r["total_estimated_flops"]), int(r["count"])),
        reverse=True,
    )[:20]
    return {
        "conv_count": conv_count,
        "estimated_flops": float(total_flops),
        "by_class": class_rows,
        "top_repeated_shapes": repeated_shapes,
    }


def _concat_shape_metadata(model: Any) -> dict[str, Any]:
    """Summarize Concat operations from an inferred ONNX graph.

    Focuses on channel-axis (axis=1 in NCHW) concatenations feeding Conv2d,
    which are the pattern targeted by segmented concat planning.  Returns
    per-concat detail including input/output shapes, static byte estimates,
    and whether the output feeds a Conv node.
    """

    import onnx
    from onnx import numpy_helper

    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | None]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        shapes[value.name] = _tensor_shape(value)
    initializers = {init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer}

    output_to_node: dict[str, int] = {}
    concat_nodes: list[tuple[int, Any]] = []
    for idx, node in enumerate(inferred.graph.node):
        for out in node.output:
            output_to_node[out] = idx
        if node.op_type == "Concat":
            concat_nodes.append((idx, node))

    conv_input_to_node: dict[str, int] = {}
    for idx, node in enumerate(inferred.graph.node):
        if node.op_type == "Conv" and node.input:
            conv_input_to_node[node.input[0]] = idx

    BYTE_PER_ELEM = 4  # float32

    def _static_bytes_for_shape(shape: list[int | None]) -> int:
        if not shape or any(d is None or d <= 0 for d in shape):
            return 0
        n = 1
        for d in shape:
            n *= d
        return n * BYTE_PER_ELEM

    by_axis: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_static_bytes": 0, "examples": []}
    )
    total_concat_count = 0
    channel_axis_concat_count = 0
    channel_axis_feeds_conv_count = 0
    large_channel_axis_concat_count = 0
    LARGE_THRESHOLD_BYTES = 32 * 1024  # 32 KiB

    for idx, node in concat_nodes:
        total_concat_count += 1
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        axis = int(attrs.get("axis", 0))
        input_shapes = [shapes.get(inp, []) for inp in node.input]
        output_shape = shapes.get(node.output[0], []) if node.output else []
        output_bytes = _static_bytes_for_shape(output_shape)
        input_bytes = [_static_bytes_for_shape(s) for s in input_shapes]

        feeds_conv = node.output[0] in conv_input_to_node
        is_channel_axis = axis == 1
        if is_channel_axis:
            channel_axis_concat_count += 1
            if feeds_conv:
                channel_axis_feeds_conv_count += 1
            if output_bytes >= LARGE_THRESHOLD_BYTES:
                large_channel_axis_concat_count += 1

        bucket = by_axis[axis]
        bucket["count"] += 1
        bucket["total_static_bytes"] += output_bytes
        if len(bucket["examples"]) < 5:
            bucket["examples"].append({
                "node_index": idx,
                "name": node.name or f"Concat_{idx}",
                "axis": axis,
                "num_inputs": len(node.input),
                "input_shapes": input_shapes,
                "output_shape": output_shape,
                "output_static_bytes": output_bytes,
                "input_static_bytes": input_bytes,
                "feeds_conv": feeds_conv,
                "feeds_conv_node": conv_input_to_node.get(node.output[0]) if feeds_conv else None,
            })

    large_concat_details = []
    for idx, node in concat_nodes:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        axis = int(attrs.get("axis", 0))
        if axis != 1:
            continue
        output_shape = shapes.get(node.output[0], []) if node.output else []
        output_bytes = _static_bytes_for_shape(output_shape)
        if output_bytes < LARGE_THRESHOLD_BYTES:
            continue
        feeds_conv = node.output[0] in conv_input_to_node
        input_shapes = [shapes.get(inp, []) for inp in node.input]
        large_concat_details.append({
            "node_index": idx,
            "name": node.name or f"Concat_{idx}",
            "axis": axis,
            "num_inputs": len(node.input),
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "output_static_bytes": output_bytes,
            "feeds_conv": feeds_conv,
            "feeds_conv_node": conv_input_to_node.get(node.output[0]) if feeds_conv else None,
        })
    large_concat_details.sort(key=lambda r: r["output_static_bytes"], reverse=True)

    by_axis_summary = {
        str(k): {
            "count": int(v["count"]),
            "total_static_bytes": int(v["total_static_bytes"]),
            "examples": v["examples"],
        }
        for k, v in sorted(by_axis.items())
    }
    return {
        "concat_count": total_concat_count,
        "channel_axis_concat_count": channel_axis_concat_count,
        "channel_axis_feeds_conv_count": channel_axis_feeds_conv_count,
        "large_channel_axis_concat_count": large_channel_axis_concat_count,
        "by_axis": by_axis_summary,
        "large_channel_axis_concats": large_concat_details[:10],
    }


def _load_onnx(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path))
    op_counter: dict[str, int] = defaultdict(int)
    for node in model.graph.node:
        op_counter[node.op_type] += 1
    metadata = {
        "path": str(onnx_path),
        "size_bytes": int(onnx_path.stat().st_size),
        "num_nodes": int(len(model.graph.node)),
        "num_initializers": int(len(model.graph.initializer)),
        "inputs": [
            {
                "name": i.name,
                "shape": [
                    d.dim_value if d.HasField("dim_value") else None
                    for d in i.type.tensor_type.shape.dim
                ],
            }
            for i in model.graph.input
            if i.name not in {init.name for init in model.graph.initializer}
        ],
        "outputs": [
            {
                "name": o.name,
                "shape": [
                    d.dim_value if d.HasField("dim_value") else None
                    for d in o.type.tensor_type.shape.dim
                ],
            }
            for o in model.graph.output
        ],
        "op_counts": dict(sorted(op_counter.items())),
    }
    try:
        metadata["conv_shape_summary"] = _conv_shape_metadata(model)
    except Exception as exc:
        metadata["conv_shape_summary_error"] = f"{type(exc).__name__}: {exc}"
    try:
        metadata["concat_shape_summary"] = _concat_shape_metadata(model)
    except Exception as exc:
        metadata["concat_shape_summary_error"] = f"{type(exc).__name__}: {exc}"
    return metadata

def _summarize_profile(profile: list[dict[str, Any]], top: int) -> list[dict[str, Any]]:
    per_kernel: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_ms": 0.0})
    for entry in profile:
        kernel = str(entry.get("kernel_name", "<unknown>"))
        per_kernel[kernel]["count"] += 1.0
        per_kernel[kernel]["total_ms"] += float(entry.get("elapsed_ns", 0)) / 1_000_000.0
    rows = [
        {
            "kernel_name": k,
            "count": int(v["count"]),
            "total_ms": float(v["total_ms"]),
            "mean_ms": float(v["total_ms"] / max(v["count"], 1.0)),
        }
        for k, v in per_kernel.items()
    ]
    rows.sort(key=lambda r: r["total_ms"], reverse=True)
    return rows[:top]


def _summarize_memory_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "instructions": int(stats.get("instructions", 0)),
        "arena_size": int(stats.get("arena_size", 0)),
        "logical_slot_bytes": int(stats.get("logical_slot_bytes", 0)),
        "physical_slot_bytes": int(stats.get("physical_slot_bytes", 0)),
        "slot_reuse_saved_bytes": int(stats.get("slot_reuse_saved_bytes", 0)),
        "estimated_static_traffic_bytes": int(stats.get("estimated_static_traffic_bytes", 0)),
        "kernel_read_bytes": int(stats.get("kernel_read_bytes", 0)),
        "kernel_write_bytes": int(stats.get("kernel_write_bytes", 0)),
        "memcpy_bytes": int(stats.get("memcpy_bytes", 0)),
        "write_const_bytes": int(stats.get("write_const_bytes", 0)),
        "fill_bytes": int(stats.get("fill_bytes", 0)),
        "call_kernel_count": int(stats.get("call_kernel_count", 0)),
        "memcpy_count": int(stats.get("memcpy_count", 0)),
        "write_const_count": int(stats.get("write_const_count", 0)),
        "fill_count": int(stats.get("fill_count", 0)),
        "alias_groups": int(stats.get("alias_groups", 0)),
        "aliased_nodes": int(stats.get("aliased_nodes", 0)),
        "top_kernels_by_count": list(stats.get("top_kernels_by_count", []))[:10],
        "top_instructions_by_static_bytes": list(
            stats.get("top_instructions_by_static_bytes", [])
        )[:10],
        "top_write_consts_by_size": list(stats.get("top_write_consts_by_size", []))[:10],
        "top_alias_groups": list(stats.get("top_alias_groups", []))[:5],
    }


def _summarize_instruction_hotspots(
    memory_stats: dict[str, Any],
    profile: list[dict[str, Any]],
    top: int,
) -> list[dict[str, Any]]:
    """Best-effort instruction-level hotspot ranking.

    The script does not import graph_memory_profile.py to keep dependencies
    tight; it just re-implements the trivial rank-by-static-bytes table
    enriched with per-kernel mean elapsed time.  Useful as a quick smoke
    that exercises the same data ``graph_memory_profile`` consumes.
    """

    kernel_timing: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "total_ms": 0.0}
    )
    for entry in profile:
        kernel = str(entry.get("kernel_name", "<unknown>"))
        kernel_timing[kernel]["count"] += 1.0
        kernel_timing[kernel]["total_ms"] += float(entry.get("elapsed_ns", 0)) / 1_000_000.0
    rows: list[dict[str, Any]] = []
    for row in memory_stats.get("top_instructions_by_static_bytes", []):
        kernel = str(row.get("kernel_name", row.get("kind", "<unknown>")))
        timing = kernel_timing.get(kernel, {"count": 0.0, "total_ms": 0.0})
        static_bytes = int(
            row.get(
                "static_bytes",
                int(row.get("read_bytes", 0)) + int(row.get("write_bytes", 0)),
            )
        )
        mean_ms = float(timing["total_ms"] / max(timing["count"], 1.0)) if timing["count"] else 0.0
        rows.append(
            {
                "instruction_index": int(row.get("instruction_index", -1)),
                "kind": str(row.get("kind", "")),
                "kernel_name": kernel,
                "node_id": row.get("node_id"),
                "static_bytes": static_bytes,
                "estimated_mean_ms": mean_ms,
                "suspected_memory_bound": static_bytes > 0
                and (mean_ms <= 1.0 or static_bytes / max(mean_ms, 1e-9) >= 1_000_000),
            }
        )
    rows.sort(key=lambda r: (r["static_bytes"], r["estimated_mean_ms"]), reverse=True)
    return rows[:top]


def _accuracy(ref: np.ndarray, cand: np.ndarray) -> dict[str, float]:
    if ref.shape != cand.shape:
        return {
            "shape_mismatch": True,
            "ref_shape": list(ref.shape),
            "cand_shape": list(cand.shape),
        }
    diff = np.abs(ref - cand)
    return {
        "shape_mismatch": False,
        "max_abs": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs": float(np.mean(diff)) if diff.size else 0.0,
        "ref_sum": float(np.sum(ref)) if ref.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-model handlers
# ---------------------------------------------------------------------------


def _export_yolo_ultralytics(entry: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    onnx_path = cache_dir / f"{entry['key']}.onnx"
    pt_path = cache_dir / f"{entry['key']}.pt"
    info: dict[str, Any] = {
        "source": "ultralytics",
        "pt_path": str(pt_path) if pt_path.exists() else entry.get("pt"),
        "onnx_path": str(onnx_path),
    }
    if not onnx_path.is_file():
        pt = entry.get("pt")
        if pt is None or not Path(pt).is_file():
            model = YOLO(entry["ultralytics_name"])
        else:
            model = YOLO(pt)
        info["pt_path"] = str(Path(model.ckpt_path) if hasattr(model, "ckpt_path") else pt)
        exported = model.export(
            format="onnx",
            imgsz=entry["imgsz"],
            opset=12,
            simplify=False,
            dynamic=False,
            half=False,
            device="cpu",
            verbose=False,
        )
        exported_path = Path(exported)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if exported_path.resolve() != onnx_path.resolve():
            onnx_path.write_bytes(exported_path.read_bytes())
    info["onnx_existed"] = onnx_path.is_file() and "export" not in info
    return info


def _export_torchvision(entry: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    import torch
    from torchvision import models

    onnx_path = cache_dir / f"{entry['key']}.onnx"
    info: dict[str, Any] = {
        "source": "torchvision",
        "factory": entry["torchvision_factory"],
        "imgsz": entry["imgsz"],
        "onnx_path": str(onnx_path),
    }
    if not onnx_path.is_file():
        factory = getattr(models, entry["torchvision_factory"])
        model = factory(weights=None)
        model.eval()
        x = torch.randn(1, 3, entry["imgsz"], entry["imgsz"])
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        # dynamo=False selects the legacy TorchScript-based exporter which
        # works for the CNN classifiers we target.  The new dynamo exporter
        # currently requires onnxscript >= 0.5 and trips on ResNet18 export
        # under onnx 1.21 due to a version_converter axes_input_to_attribute
        # assertion.
        try:
            torch.onnx.export(
                model,
                x,
                str(onnx_path),
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=None,
                dynamo=False,
            )
        except Exception as exc:
            info["export_error"] = f"{type(exc).__name__}: {exc}"
            raise
    return info


def _make_input(entry: dict[str, Any], seed: int) -> np.ndarray:
    shape = [1, 3, entry["imgsz"], entry["imgsz"]]
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _try_fastnn(
    entry: dict[str, Any],
    onnx_path: Path,
    x: np.ndarray,
    ref_outputs: dict[str, np.ndarray],
    warmup: int,
    iters: int,
    profile_top: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        executor, in_name, out_name = _build_aot_executor(onnx_path, family=entry["family"])
    except Exception as exc:
        info["aot_error"] = f"{type(exc).__name__}: {exc}"
        info["aot_traceback"] = traceback.format_exc(limit=3)
        return info
    info["aot_input_name"] = in_name
    info["aot_output_name"] = out_name

    import fastnn as fnn

    fx = fnn.tensor(x, list(x.shape))
    try:
        fy = executor.forward({in_name: fx})[out_name].numpy()
    except Exception as exc:
        info["forward_error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["output_shape"] = list(fy.shape)
    info["output_sum"] = float(np.sum(fy))

    # Accuracy vs the first reference we can match on shape.
    for ref_name, ref_y in ref_outputs.items():
        if ref_y.shape == fy.shape:
            info["accuracy_vs_" + ref_name] = _accuracy(ref_y, fy)
            break

    info["forward_speed"] = _timer(
        lambda: executor.forward({in_name: fx}), warmup, iters
    )

    if hasattr(executor, "memory_stats"):
        try:
            info["memory_stats_summary"] = _summarize_memory_stats(dict(executor.memory_stats()))
        except Exception as exc:
            info["memory_stats_error"] = f"{type(exc).__name__}: {exc}"
    else:
        info["memory_stats_error"] = "AotExecutor has no memory_stats()"

    if hasattr(executor, "profile"):
        try:
            prof = executor.profile({in_name: fx})
            info["profile_top_kernels"] = _summarize_profile(prof["profile"], profile_top)
            info["profile_total_ms"] = float(
                sum(float(e.get("elapsed_ns", 0)) for e in prof["profile"]) / 1_000_000.0
            )
            if "memory_stats_summary" in info:
                info["profile_instruction_hotspots"] = _summarize_instruction_hotspots(
                    dict(executor.memory_stats()),
                    prof["profile"],
                    profile_top,
                )
        except Exception as exc:
            info["profile_error"] = f"{type(exc).__name__}: {exc}"

    return info


def _try_onnxruntime(
    entry: dict[str, Any], onnx_path: Path, x: np.ndarray, warmup: int, iters: int
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import onnxruntime as ort
    except Exception as exc:
        info["ort_error"] = f"{type(exc).__name__}: {exc}"
        return info
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        info["ort_session_error"] = f"{type(exc).__name__}: {exc}"
        return info
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    info["ort_input_name"] = in_meta.name
    info["ort_output_name"] = out_meta.name
    try:
        y = sess.run([out_meta.name], {in_meta.name: x})[0]
    except Exception as exc:
        info["ort_forward_error"] = f"{type(exc).__name__}: {exc}"
        return info
    info["output_shape"] = list(y.shape)
    info["output_sum"] = float(np.sum(y))
    info["ort_speed"] = _timer(
        lambda: sess.run(None, {in_meta.name: x}), warmup, iters
    )
    return info


def _try_pytorch(
    entry: dict[str, Any], x: np.ndarray, ref_outputs: dict[str, np.ndarray],
    warmup: int, iters: int,
) -> dict[str, Any]:
    info: dict[str, Any] = {}
    if entry["family"] == "yolo":
        try:
            from ultralytics import YOLO
        except Exception as exc:
            info["pytorch_error"] = f"ultralytics import: {type(exc).__name__}: {exc}"
            return info
        try:
            import torch

            model = YOLO(entry["pt"]) if entry.get("pt") and Path(entry["pt"]).is_file() else YOLO(entry["ultralytics_name"])
            try:
                model.fuse()
            except Exception:
                pass
            net = model.model.eval().cpu()
            xt = torch.from_numpy(x)
            with torch.no_grad():
                y = net(xt)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y_np = _to_numpy(y)
            info["output_shape"] = list(y_np.shape)
            info["output_sum"] = float(np.sum(y_np))
            ref_outputs["pytorch"] = y_np
            torch.set_num_threads(1)
            info["pytorch_speed"] = _timer(lambda: net(xt), warmup, iters)
        except Exception as exc:
            info["pytorch_error"] = f"{type(exc).__name__}: {exc}"
            return info
    elif entry["family"] == "torchvision":
        try:
            import torch
            from torchvision import models as tvm

            factory = getattr(tvm, entry["torchvision_factory"])
            net = factory(weights=None).eval().cpu()
            xt = torch.from_numpy(x)
            with torch.no_grad():
                y = net(xt)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y_np = _to_numpy(y)
            info["output_shape"] = list(y_np.shape)
            info["output_sum"] = float(np.sum(y_np))
            ref_outputs["pytorch"] = y_np
            torch.set_num_threads(1)
            info["pytorch_speed"] = _timer(lambda: net(xt), warmup, iters)
        except Exception as exc:
            info["pytorch_error"] = f"{type(exc).__name__}: {exc}"
            return info
    return info


# ---------------------------------------------------------------------------
# Default model list
# ---------------------------------------------------------------------------


DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "key": "yolov8n",
        "family": "yolo",
        "ultralytics_name": "yolov8n.pt",
        "pt": "yolov8n.pt",
        "imgsz": 320,
        "description": "YOLOv8n detection baseline (regression anchor)",
    },
    {
        "key": "yolo11n",
        "family": "yolo",
        "ultralytics_name": "yolo11n.pt",
        "pt": "yolo11n.pt",
        "imgsz": 320,
        "description": "YOLO11n detection (newer Ultralytics architecture)",
    },
    {
        "key": "yolo11l",
        "family": "yolo",
        "ultralytics_name": "yolo11l.pt",
        "pt": "yolo11l.pt",
        "imgsz": 320,
        "description": "YOLO11l detection (larger YOLO curiosity comparison)",
        "optional": True,
    },
    {
        "key": "resnet18",
        "family": "torchvision",
        "torchvision_factory": "resnet18",
        "imgsz": 224,
        "description": "ResNet18 ImageNet classifier (4x bottleneck blocks)",
    },
    {
        "key": "resnet50",
        "family": "torchvision",
        "torchvision_factory": "resnet50",
        "imgsz": 224,
        "description": "ResNet50 ImageNet classifier (deeper bottleneck blocks)",
    },
    {
        "key": "mobilenet_v2",
        "family": "torchvision",
        "torchvision_factory": "mobilenet_v2",
        "imgsz": 224,
        "description": "MobileNetV2 (depthwise + pointwise conv stack)",
    },
    {
        "key": "mobilenet_v3_small",
        "family": "torchvision",
        "torchvision_factory": "mobilenet_v3_small",
        "imgsz": 224,
        "description": "MobileNetV3-small (depthwise + SE + h-swish)",
    },
    {
        "key": "efficientnet_b0",
        "family": "torchvision",
        "torchvision_factory": "efficientnet_b0",
        "imgsz": 224,
        "description": "EfficientNet-B0 (depthwise + SE + SiLU)",
    },
]


def _resolve_entry(entry: dict[str, Any], selected: set[str] | None) -> dict[str, Any] | None:
    if selected is None:
        return entry
    return entry if entry["key"] in selected else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _classify_bottleneck(record: dict[str, Any]) -> str:
    """Quick bottleneck class label for a per-model record.

    The classification is intentionally coarse; per-kernel drill-downs are
    left to the full JSON.
    """

    fastnn_block = record.get("fastnn", {}) or {}
    if fastnn_block.get("aot_error"):
        return "aot_import_failure"
    if "forward_speed" not in fastnn_block:
        if record.get("export_error"):
            return "export_failure"
        if record.get("aot_error"):
            return "aot_import_failure"
        return "no_fastnn_timing"
    mem = fastnn_block.get("memory_stats_summary") or {}
    arena = mem.get("arena_size", 0)
    traffic = mem.get("estimated_static_traffic_bytes", 0)
    write_const = mem.get("write_const_bytes", 0)
    kernel_read = mem.get("kernel_read_bytes", 0)
    kernel_write = mem.get("kernel_write_bytes", 0)
    memcpy = mem.get("memcpy_bytes", 0)
    if arena and traffic / arena >= 4.0:
        return "memory_traffic_bound"
    if arena and write_const / max(arena, 1) >= 1.5:
        return "write_const_heavy"
    if arena and memcpy / max(arena, 1) >= 0.4:
        return "memcpy_heavy"
    if kernel_read and kernel_write and kernel_read + kernel_write > arena * 2:
        return "kernel_io_bound"
    return "compute_or_unclassified"


def _print_summary(results: dict[str, dict[str, Any]]) -> None:
    print("\n=== Model zoo CPU matrix summary ===")
    header = (
        f"{'model':<22} {'aot':<5} {'fastnn ms':>10} {'onnxrt ms':>10} "
        f"{'pytorch ms':>11} {'arena':>10} {'traffic/arena':>14} {'class':<26}"
    )
    print(header)
    for key, record in sorted(results.items()):
        fastnn_block = record.get("fastnn", {}) or {}
        ort_block = record.get("onnxruntime", {}) or {}
        pt_block = record.get("pytorch", {}) or {}
        aot = "ok" if fastnn_block.get("forward_speed") else "fail"
        fast = (fastnn_block.get("forward_speed") or {}).get("mean_ms")
        ort = (ort_block.get("ort_speed") or {}).get("mean_ms")
        pt = (pt_block.get("pytorch_speed") or {}).get("mean_ms")
        mem = fastnn_block.get("memory_stats_summary") or {}
        arena = mem.get("arena_size", 0)
        traffic = mem.get("estimated_static_traffic_bytes", 0)
        ratio = traffic / arena if arena else 0.0
        cls = _classify_bottleneck(record)
        def _fmt(v: float | None) -> str:
            return f"{v:10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}"
        print(
            f"{key:<22} {aot:<5} {_fmt(fast)} {_fmt(ort)} {_fmt(pt):>11} "
            f"{arena:>10d} {ratio:>14.2f} {cls:<26}"
        )

    print("\n=== Concat breadth summary ===")
    concat_header = (
        f"{'model':<22} {'concat#':>7} {'chan_axis':>9} {'chan->Conv':>10} "
        f"{'large_chan':>10} {'top_large_bytes':>16}"
    )
    print(concat_header)
    for key, record in sorted(results.items()):
        meta = record.get("onnx_metadata", {}) or {}
        cs = meta.get("concat_shape_summary") or {}
        total = cs.get("concat_count", 0)
        chan = cs.get("channel_axis_concat_count", 0)
        chan_conv = cs.get("channel_axis_feeds_conv_count", 0)
        large = cs.get("large_channel_axis_concat_count", 0)
        large_list = cs.get("large_channel_axis_concats") or []
        top_bytes = large_list[0]["output_static_bytes"] if large_list else 0
        err = meta.get("concat_shape_summary_error")
        if err:
            print(f"{key:<22} {'ERROR':>7}  {err}")
        else:
            print(
                f"{key:<22} {total:>7d} {chan:>9d} {chan_conv:>10d} "
                f"{large:>10d} {top_bytes:>16d}"
            )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2, default=str)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run a small CPU model-zoo benchmark matrix.  Records per-model "
            "fastnn AOT, ORT, and PyTorch timing, profile, memory_stats, and "
            "accuracy.  Failures are recorded per model and never abort the run."
        )
    )
    ap.add_argument(
        "--models",
        default=",".join(
            entry["key"] for entry in DEFAULT_MODELS if not entry.get("optional")
        ),
        help=(
            "Comma-separated subset of model keys to run. Default excludes "
            "optional larger models; include yolo11l explicitly for the large YOLO probe."
        ),
    )
    ap.add_argument("--cache-dir", type=Path, default=Path("/tmp/fastnn-zoo"))
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--profile-top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--skip-fastnn", action="store_true", help="Skip fastnn AOT timing/profile"
    )
    ap.add_argument(
        "--skip-ort", action="store_true", help="Skip ONNX Runtime timing"
    )
    ap.add_argument(
        "--skip-pytorch", action="store_true", help="Skip PyTorch reference timing"
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="Write a machine-readable summary JSON to this path",
    )
    ap.add_argument(
        "--pt-yolov8n",
        default="yolov8n.pt",
        help="Path to local yolov8n.pt if cached; otherwise downloaded by Ultralytics",
    )
    ap.add_argument(
        "--pt-yolo11n",
        default="yolo11n.pt",
        help="Path to local yolo11n.pt if cached; otherwise downloaded by Ultralytics",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    selected = {m.strip() for m in args.models.split(",") if m.strip()}

    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Override pt paths from CLI for the two YOLO entries when given.
    model_list: list[dict[str, Any]] = []
    for entry in DEFAULT_MODELS:
        if entry["key"] == "yolov8n" and args.pt_yolov8n:
            entry = {**entry, "pt": args.pt_yolov8n}
        if entry["key"] == "yolo11n" and args.pt_yolo11n:
            entry = {**entry, "pt": args.pt_yolo11n}
        if entry["key"] in selected:
            model_list.append(entry)

    results: dict[str, dict[str, Any]] = {}
    for entry in model_list:
        key = entry["key"]
        record: dict[str, Any] = {
            "key": key,
            "family": entry["family"],
            "description": entry.get("description", ""),
            "imgsz": entry["imgsz"],
        }
        # 1) Export
        try:
            if entry["family"] == "yolo":
                export_info = _export_yolo_ultralytics(entry, cache_dir)
            elif entry["family"] == "torchvision":
                export_info = _export_torchvision(entry, cache_dir)
            else:
                raise RuntimeError(f"unknown family: {entry['family']}")
            record["export"] = export_info
            onnx_path = Path(export_info["onnx_path"])
        except Exception as exc:
            record["export_error"] = f"{type(exc).__name__}: {exc}"
            record["export_traceback"] = traceback.format_exc(limit=4)
            print(f"[{key}] export FAILED: {record['export_error']}")
            record["bottleneck_class"] = _classify_bottleneck(record)
            results[key] = record
            continue
        try:
            record["onnx_metadata"] = _load_onnx(onnx_path)
        except Exception as exc:
            record["onnx_metadata_error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{key}] ONNX ready: {onnx_path} ({record.get('onnx_metadata', {}).get('size_bytes', '?')} bytes)")

        # 2) PyTorch reference (if requested)
        x = _make_input(entry, args.seed)
        ref_outputs: dict[str, np.ndarray] = {}
        if not args.skip_pytorch:
            pt_info = _try_pytorch(entry, x, ref_outputs, args.warmup, args.iters)
            record["pytorch"] = pt_info
            if "pytorch_error" in pt_info:
                print(f"[{key}] pytorch FAILED: {pt_info['pytorch_error']}")
            else:
                print(
                    f"[{key}] pytorch mean_ms={pt_info['pytorch_speed']['mean_ms']:.3f}"
                )

        # 3) ONNX Runtime (if requested)
        if not args.skip_ort:
            ort_info = _try_onnxruntime(entry, onnx_path, x, args.warmup, args.iters)
            record["onnxruntime"] = ort_info
            if "ort_speed" in ort_info:
                print(
                    f"[{key}] onnxruntime mean_ms={ort_info['ort_speed']['mean_ms']:.3f}"
                )
            elif (
                "ort_error" in ort_info
                or "ort_session_error" in ort_info
                or "ort_forward_error" in ort_info
            ):
                err = (
                    ort_info.get("ort_error")
                    or ort_info.get("ort_session_error")
                    or ort_info.get("ort_forward_error")
                )
                print(f"[{key}] onnxruntime FAILED: {err}")

        # Capture ORT output tensor for accuracy (cheap, single forward).
        if "onnxruntime" in record and "output_shape" in record["onnxruntime"]:
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
                in_meta = sess.get_inputs()[0]
                out_meta = sess.get_outputs()[0]
                y_ref = sess.run([out_meta.name], {in_meta.name: x})[0]
                ref_outputs["onnxruntime"] = y_ref
            except Exception:
                pass

        # 4) fastnn AOT
        if not args.skip_fastnn:
            fastnn_info = _try_fastnn(
                entry, onnx_path, x, ref_outputs, args.warmup, args.iters, args.profile_top
            )
            record["fastnn"] = fastnn_info
            if "forward_speed" in fastnn_info:
                print(
                    f"[{key}] fastnn mean_ms={fastnn_info['forward_speed']['mean_ms']:.3f}"
                )
            elif "aot_error" in fastnn_info:
                print(f"[{key}] fastnn AOT FAILED: {fastnn_info['aot_error']}")
            elif "forward_error" in fastnn_info:
                print(f"[{key}] fastnn forward FAILED: {fastnn_info['forward_error']}")

        record["bottleneck_class"] = _classify_bottleneck(record)
        results[key] = record

    _print_summary(results)

    summary = {
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "cache_dir": str(cache_dir),
        "models": results,
    }
    if args.json_out is not None:
        try:
            _atomic_write_json(args.json_out, summary)
            print(f"json {args.json_out}")
        except Exception as exc:
            print(f"error: failed to write JSON: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
