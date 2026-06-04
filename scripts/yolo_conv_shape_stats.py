#!/usr/bin/env python3
"""Summarize Conv/Conv+SiLU shapes in a YOLO ONNX model.

This is a planning/profiling utility for fastnn PreparedPlan Conv work. It does
not execute the model; it uses ONNX shape inference and initializers to report
Conv geometry, GEMM shape estimates, repeated-shape counts, and fused-SiLU-like
Conv -> Sigmoid -> Mul patterns.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper, shape_inference


def _value_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    value_infos = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for vi in value_infos:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for d in tt.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                dims.append(int(d.dim_value))
            else:
                ok = False
                break
        if ok:
            shapes[vi.name] = dims
    return shapes


def _initializer_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    return {init.name: list(numpy_helper.to_array(init).shape) for init in model.graph.initializer}


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for a in node.attribute:
        if a.ints:
            attrs[a.name] = [int(x) for x in a.ints]
        elif a.HasField("i"):
            attrs[a.name] = int(a.i)
        elif a.HasField("f"):
            attrs[a.name] = float(a.f)
        elif a.HasField("s"):
            attrs[a.name] = a.s.decode("utf-8", errors="replace")
    return attrs


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for inp in node.input:
            consumers[inp].append(node)
    return consumers


def _is_conv_silu(node: onnx.NodeProto, consumers: dict[str, list[onnx.NodeProto]]) -> bool:
    if not node.output:
        return False
    conv_out = node.output[0]
    conv_consumers = consumers.get(conv_out, [])
    sigmoids = [n for n in conv_consumers if n.op_type == "Sigmoid"]
    for sig in sigmoids:
        if not sig.output:
            continue
        sig_out = sig.output[0]
        for mul in consumers.get(sig_out, []):
            if mul.op_type == "Mul" and conv_out in mul.input:
                return True
    return False


def conv_rows(path: Path) -> list[dict[str, Any]]:
    model = onnx.load(str(path))
    inferred = shape_inference.infer_shapes(model)
    shapes = _value_shapes(inferred)
    init_shapes = _initializer_shapes(inferred)
    consumers = _consumer_map(inferred)

    rows: list[dict[str, Any]] = []
    for idx, node in enumerate(inferred.graph.node):
        if node.op_type != "Conv":
            continue
        attrs = _attrs(node)
        x_name = node.input[0]
        w_name = node.input[1] if len(node.input) > 1 else ""
        y_name = node.output[0] if node.output else ""
        input_shape = shapes.get(x_name)
        weight_shape = init_shapes.get(w_name) or shapes.get(w_name)
        output_shape = shapes.get(y_name)
        if not input_shape or not weight_shape or not output_shape:
            continue
        n, c, h, w = input_shape
        f, wc, kh, kw = weight_shape
        _n2, _f2, oh, ow = output_shape
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        dilations = attrs.get("dilations", [1, 1])
        groups = int(attrs.get("group", 1))
        stride = int(strides[0]) if strides else 1
        pad = int(pads[0]) if pads else 0
        dilation = int(dilations[0]) if dilations else 1
        gemm_m = f
        gemm_k = wc * kh * kw
        gemm_n = oh * ow
        rows.append(
            {
                "index": idx,
                "name": node.name or f"Conv_{idx}",
                "input": [n, c, h, w],
                "weight": [f, wc, kh, kw],
                "output": output_shape,
                "stride": stride,
                "padding": pad,
                "dilation": dilation,
                "groups": groups,
                "activation": "silu" if _is_conv_silu(node, consumers) else None,
                "gemm": [gemm_m, gemm_k, gemm_n],
                "flops": 2 * gemm_m * gemm_k * gemm_n,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    shape_counts: Counter[tuple[Any, ...]] = Counter()
    examples: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            tuple(row["input"]),
            tuple(row["weight"]),
            row["stride"],
            row["padding"],
            row["dilation"],
            row["groups"],
            row["activation"],
        )
        shape_counts[key] += 1
        examples.setdefault(key, row)

    grouped = []
    for key, count in shape_counts.items():
        ex = examples[key]
        grouped.append(
            {
                "count": count,
                "input": ex["input"],
                "weight": ex["weight"],
                "output": ex["output"],
                "stride": ex["stride"],
                "padding": ex["padding"],
                "dilation": ex["dilation"],
                "groups": ex["groups"],
                "activation": ex["activation"],
                "gemm": ex["gemm"],
                "flops_each": ex["flops"],
                "flops_total": ex["flops"] * count,
            }
        )
    grouped.sort(key=lambda r: (r["flops_total"], r["count"]), reverse=True)
    return {
        "total_conv": len(rows),
        "conv_silu": sum(1 for r in rows if r["activation"] == "silu"),
        "groups": grouped,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("yolov8n.onnx"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"ONNX file not found: {args.onnx}")
    rows = conv_rows(args.onnx)
    summary = summarize(rows)
    result = {"model": str(args.onnx), "summary": summary, "rows": rows}
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("YOLO Conv shape stats")
    print(f"  model: {args.onnx}")
    print(f"  total conv: {summary['total_conv']}")
    print(f"  conv+silu patterns: {summary['conv_silu']}")
    print("  top grouped shapes:")
    for g in summary["groups"][: args.top]:
        print(
            "    "
            f"count={g['count']:2d} act={g['activation'] or '-':4s} "
            f"input={g['input']} weight={g['weight']} stride={g['stride']} pad={g['padding']} "
            f"gemm={g['gemm']} total_gflop={g['flops_total']/1e9:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
