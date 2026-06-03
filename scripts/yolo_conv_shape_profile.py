#!/usr/bin/env python3
"""Profile YOLO conv kernels by ONNX Conv shape.

Maps fastnn AotExecutor profile entries back to ONNX Conv nodes and reports
per-node/per-shape timing. This is intended to guide CPU conv backend work after
ConvSiLU fusion.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from yolo_compare_fastnn_pytorch import _export_yolo_pt_to_onnx, _make_fastnn_executor  # noqa: E402


def _shape_from_vi(vi: onnx.ValueInfoProto) -> list[int | str]:
    if not vi.type.HasField("tensor_type"):
        return []
    out: list[int | str] = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def _to_int_shape(shape: list[int | str]) -> list[int] | None:
    if all(isinstance(x, int) and x > 0 for x in shape):
        return [int(x) for x in shape]
    return None


def _conv_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for a in node.attribute:
        if a.name in {"strides", "pads", "dilations", "kernel_shape"}:
            attrs[a.name] = [int(v) for v in a.ints]
        elif a.name == "group":
            attrs[a.name] = int(a.i)
    attrs.setdefault("strides", [1, 1])
    attrs.setdefault("pads", [0, 0, 0, 0])
    attrs.setdefault("dilations", [1, 1])
    attrs.setdefault("group", 1)
    return attrs


def load_conv_meta(onnx_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    inferred = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    shapes: dict[str, list[int | str]] = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shapes[vi.name] = _shape_from_vi(vi)
    initializers = {init.name: numpy_helper.to_array(init) for init in inferred.graph.initializer}

    convs: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(inferred.graph.node):
        if node.op_type != "Conv":
            continue
        attrs = _conv_attrs(node)
        w = initializers.get(node.input[1])
        weight_shape = list(map(int, w.shape)) if w is not None else []
        in_shape = _to_int_shape(shapes.get(node.input[0], []))
        out_shape = _to_int_shape(shapes.get(node.output[0], []))
        c_per_group = 0
        if weight_shape:
            f, c_per_group, kh, kw = weight_shape
            groups = int(attrs["group"])
            c = c_per_group * groups
        else:
            f = c = kh = kw = groups = 0
        if out_shape and len(out_shape) == 4:
            n, _, oh, ow = out_shape
            spatial = oh * ow
        else:
            n = spatial = oh = ow = 0
        k = c_per_group * kh * kw if weight_shape else 0
        m = f
        n_gemm = spatial
        strides = attrs["strides"]
        pads = attrs["pads"]
        dil = attrs["dilations"]
        kind = f"{kh}x{kw}/s{strides[0]}x{strides[1]}/g{groups}"
        meta = {
            "onnx_index": idx,
            "name": node.name,
            "input": node.input[0],
            "output": node.output[0],
            "input_shape": in_shape,
            "output_shape": out_shape,
            "weight_shape": weight_shape,
            "f": f,
            "c": c,
            "c_per_group": c_per_group if weight_shape else 0,
            "kh": kh,
            "kw": kw,
            "groups": groups,
            "strides": strides,
            "pads": pads,
            "dilations": dil,
            "kind": kind,
            "gemm_m": m,
            "gemm_k": k,
            "gemm_n": n_gemm,
            "gemm_flops": 2 * m * k * n_gemm,
        }
        convs.append(meta)
        by_name[node.name] = meta
    return convs, by_name


def summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "total_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": values[len(values) // 2],
        "min_ms": values[0],
        "max_ms": values[-1],
        "total_ms": sum(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=ROOT / "yolov8n.pt")
    ap.add_argument("--onnx", type=Path, default=Path("/tmp/fastnn-yolo-verify/yolov8n.onnx"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--profile-runs", type=int, default=10)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    onnx_path = args.onnx
    if not onnx_path.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        _export_yolo_pt_to_onnx(str(args.model), onnx_path, imgsz=320)
    convs, conv_by_name = load_conv_meta(onnx_path)
    executor, input_name, _output_name = _make_fastnn_executor(onnx_path)
    x = np.random.default_rng(0).random((1, 3, 320, 320), dtype=np.float32)
    import fastnn as fnn
    fx = fnn.tensor(x, list(x.shape))

    for _ in range(args.warmup):
        executor.forward({input_name: fx})

    node_times: dict[str, list[float]] = collections.defaultdict(list)
    node_kernels: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    unmapped: list[dict[str, Any]] = []
    for _ in range(args.profile_runs):
        prof = executor.profile({input_name: fx})["profile"]
        conv_profile_entries: list[dict[str, Any]] = []
        for e in prof:
            kernel = e.get("kernel_name", "")
            if kernel not in {"conv2d", "conv2d_silu", "conv2d_relu", "conv2d_gelu"}:
                continue
            conv_profile_entries.append(e)

        if len(conv_profile_entries) == len(convs):
            # The current AotExecutor profile may not expose node names. YOLO's
            # compiled conv instructions are in ONNX Conv execution order, so
            # zip by order when counts match exactly.
            for conv, e in zip(convs, conv_profile_entries):
                ms = float(e["elapsed_ns"]) / 1e6
                node_times[conv["name"]].append(ms)
                node_kernels[conv["name"]][str(e.get("kernel_name", ""))] += 1
            continue

        # Fallback for builds that do expose node names.
        for e in conv_profile_entries:
            name = e.get("node_name", "")
            ms = float(e["elapsed_ns"]) / 1e6
            if name in conv_by_name:
                node_times[name].append(ms)
                node_kernels[name][str(e.get("kernel_name", ""))] += 1
            else:
                unmapped.append(dict(e))

    rows: list[dict[str, Any]] = []
    for conv in convs:
        times = node_times.get(conv["name"], [])
        if not times:
            continue
        s = summarize(times)
        flops = conv["gemm_flops"]
        gflops = (flops / 1e9) / (s["mean_ms"] / 1e3) if s["mean_ms"] else 0.0
        row = {**conv, **s, "gflops_est": gflops}
        row["kernels"] = dict(node_kernels.get(conv["name"], {}))
        rows.append(row)

    class_times: dict[tuple[Any, ...], list[float]] = collections.defaultdict(list)
    for row in rows:
        key = (row["kind"], row["f"], row["c"], row["gemm_n"])
        class_times[key].extend(node_times[row["name"]])
    class_rows = []
    for key, times in class_times.items():
        kind, f, c, spatial = key
        s = summarize(times)
        class_rows.append({"kind": kind, "f": f, "c": c, "spatial": spatial, **s})
    class_rows.sort(key=lambda r: r["total_ms"], reverse=True)
    rows.sort(key=lambda r: r["mean_ms"], reverse=True)

    out = {
        "onnx_path": str(onnx_path),
        "profile_runs": args.profile_runs,
        "num_onnx_convs": len(convs),
        "num_profiled_convs": len(rows),
        "unmapped_count": len(unmapped),
        "total_conv_mean_ms": sum(r["mean_ms"] for r in rows),
        "top_nodes": rows[: args.top],
        "top_classes": class_rows[: args.top],
    }
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(f"ONNX: {onnx_path}")
    print(f"Conv nodes: ONNX={len(convs)} profiled={len(rows)} unmapped_entries={len(unmapped)} runs={args.profile_runs}")
    print(f"Sum of per-node mean conv time: {out['total_conv_mean_ms']:.3f} ms")
    print("\nTop shape classes (aggregate over same kind/f/c/spatial):")
    for r in class_rows[: args.top]:
        print(
            f"  total={r['total_ms']:.3f}ms mean={r['mean_ms']:.3f}ms count={r['count']:3d} "
            f"{r['kind']} f={r['f']} c={r['c']} spatial={r['spatial']}"
        )
    print("\nTop conv nodes:")
    for r in rows[: args.top]:
        print(
            f"  mean={r['mean_ms']:.3f}ms med={r['median_ms']:.3f}ms gf/s={r['gflops_est']:.1f} "
            f"kernels={r['kernels']} {r['kind']} W={r['weight_shape']} out={r['output_shape']} "
            f"GEMM=({r['gemm_m']}x{r['gemm_k']})*({r['gemm_k']}x{r['gemm_n']}) {r['name']}"
        )
    if unmapped:
        print("\nUnmapped conv profile entry sample:")
        for e in unmapped[:5]:
            print(" ", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
