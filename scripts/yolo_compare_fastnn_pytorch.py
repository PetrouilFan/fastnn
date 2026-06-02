#!/usr/bin/env python3
"""Compare YOLO ONNX inference through fastnn against PyTorch/ONNX Runtime.

This script is intentionally diagnostic: it exports/loads a YOLO ONNX model,
tries to compile it with fastnn's AOT ONNX path, compares raw logits against
PyTorch and ONNX Runtime, and reports speed for each runnable backend.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def _timer(fn, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "iters": iters,
    }


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _export_yolo_pt_to_onnx(pt_path: str, onnx_path: Path, imgsz: int) -> Any:
    from ultralytics import YOLO

    model = YOLO(pt_path)
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=False,
        dynamic=False,
        half=False,
        device="cpu",
        verbose=False,
    )
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(exported_path.read_bytes())
    return model


def _onnx_metadata(onnx_path: Path) -> dict[str, Any]:
    import collections
    import onnx

    model = onnx.load(str(onnx_path))
    ops = collections.Counter(node.op_type for node in model.graph.node)
    return {
        "inputs": [
            [i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]]
            for i in model.graph.input
        ],
        "outputs": [
            [o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]]
            for o in model.graph.output
        ],
        "num_nodes": len(model.graph.node),
        "ops": dict(sorted(ops.items())),
    }


def _run_onnxruntime(onnx_path: Path, x: np.ndarray) -> tuple[np.ndarray, Any, str]:
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y = sess.run([output_name], {input_name: x})[0]
    return y, sess, input_name


def _run_pytorch(model: Any, x: np.ndarray) -> tuple[np.ndarray, Any]:
    import torch

    torch.set_num_threads(1)
    model.model.eval()
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y = model.model(xt)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return _to_numpy(y), xt


def _format_attr_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        return ",".join(str(int(v)) if float(v).is_integer() else str(float(v)) for v in value.flatten())
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _make_fastnn_executor(onnx_path: Path):
    import onnx
    from onnx import numpy_helper
    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializer_names]
    output_names = [o.name for o in model.graph.output]
    known_shapes: dict[str, list[int]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = []
        tensor_type = vi.type.tensor_type
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1)
            known_shapes[vi.name] = dims
    input_shapes = {}
    for i in model.graph.input:
        if i.name in input_names:
            dims = []
            for d in i.type.tensor_type.shape.dim:
                dims.append(int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else -1)
            input_shapes[i.name] = dims

    params = {}
    const_values: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        const_values[init.name] = arr_raw
        arr = arr_raw.astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))

    nodes = []
    for idx, node in enumerate(model.graph.node):
        name = node.name or f"{node.op_type}_{idx}"
        attrs = _extract_attrs(node)
        if node.op_type == "Constant" and node.output:
            value = attrs.get("value")
            if not isinstance(value, np.ndarray):
                for attr in node.attribute:
                    if attr.name == "value":
                        value = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
                        break
            if isinstance(value, np.ndarray):
                # Preserve shape constants as numeric tensors consumable by the Rust converter.
                const_values[node.output[0]] = value
                arr = value.astype(np.float32, copy=False) if value.dtype.kind == "f" else value.astype(np.float32)
                params[node.output[0]] = fnn.tensor(arr, list(arr.shape))
        elif node.output:
            out_name = node.output[0]
            try:
                if node.op_type == "Shape" and node.input[0] in known_shapes:
                    const_values[out_name] = np.asarray(known_shapes[node.input[0]], dtype=np.int64)
                elif node.op_type == "Gather" and node.input[0] in const_values and node.input[1] in const_values:
                    axis = int(attrs.get("axis", 0))
                    const_values[out_name] = np.take(const_values[node.input[0]], const_values[node.input[1]].astype(np.int64), axis=axis)
                elif node.op_type in {"Add", "Sub", "Mul", "Div"} and len(node.input) >= 2:
                    if node.input[0] in const_values and node.input[1] in const_values:
                        a = const_values[node.input[0]]
                        b = const_values[node.input[1]]
                        if node.op_type == "Add":
                            const_values[out_name] = a + b
                        elif node.op_type == "Sub":
                            const_values[out_name] = a - b
                        elif node.op_type == "Mul":
                            const_values[out_name] = a * b
                        elif node.op_type == "Div":
                            const_values[out_name] = np.floor_divide(a, b)
                if out_name in const_values and out_name not in params:
                    arr_raw = np.asarray(const_values[out_name])
                    arr = arr_raw.astype(np.float32, copy=False)
                    params[out_name] = fnn.tensor(arr, list(arr.shape))
            except Exception:
                pass
        item = {
            "name": name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
        }
        if node.output and node.output[0] in known_shapes:
            item["output_shape"] = ",".join(str(d) for d in known_shapes[node.output[0]])
        if node.op_type == "Slice" and len(node.input) >= 4:
            data_shape = known_shapes.get(node.input[0])
            starts = const_values.get(node.input[1])
            ends = const_values.get(node.input[2])
            axes = const_values.get(node.input[3])
            if data_shape and starts is not None and ends is not None and axes is not None:
                axis = int(np.asarray(axes).reshape(-1)[0])
                if axis < 0:
                    axis += len(data_shape)
                start = int(np.asarray(starts).reshape(-1)[0])
                end = int(np.asarray(ends).reshape(-1)[0])
                if 0 <= axis < len(data_shape):
                    dim_size = data_shape[axis]
                    if dim_size > 0:
                        if start < 0:
                            start += dim_size
                        if end < 0:
                            end += dim_size
                        start = max(0, min(start, dim_size))
                        end = max(start, min(end, dim_size))
                    item["starts"] = str(start)
                    item["ends"] = str(end)
                    item["axes"] = str(axis)
                    out_shape = list(data_shape)
                    if 0 <= axis < len(out_shape) and end >= start:
                        out_shape[axis] = end - start
                        item["output_shape"] = ",".join(str(d) for d in out_shape)
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            item[k] = _format_attr_value(v)
        if node.op_type == "Resize" and len(node.input) >= 3:
            # Fast path for Ultralytics nearest-neighbor 2x upsampling. The
            # exported graph supplies scales as [1,1,2,2]; make this explicit
            # so the Rust converter does not have to infer it from Constant nodes.
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    item["scale_h"] = str(int(scales[2]))
                    item["scale_w"] = str(int(scales[3]))
        nodes.append(item)

    return fnn.AotExecutor(nodes, params, input_names, output_names, input_shapes=input_shapes), input_names[0], output_names[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="yolov8n.pt", help="Ultralytics YOLO .pt model to export/use as PyTorch oracle")
    ap.add_argument("--onnx", default="/tmp/fastnn-yolo-verify/yolov8n.onnx")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="Print fastnn per-kernel profile from one measured run")
    ap.add_argument("--profile-top", type=int, default=20)
    args = ap.parse_args()

    np.random.seed(args.seed)
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"export/load YOLO: {args.pt} -> {onnx_path}")
    model = _export_yolo_pt_to_onnx(args.pt, onnx_path, args.imgsz)
    meta = _onnx_metadata(onnx_path)
    print("onnx_metadata", json.dumps(meta, sort_keys=True))

    x = np.random.default_rng(args.seed).random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)

    results: dict[str, Any] = {"onnx": meta}

    y_torch, x_torch = _run_pytorch(model, x)
    results["pytorch_output_shape"] = list(y_torch.shape)
    results["pytorch_speed"] = _timer(lambda: model.model(x_torch), args.warmup, args.iters)
    print("pytorch", json.dumps({"shape": list(y_torch.shape), "speed": results["pytorch_speed"]}, sort_keys=True))

    y_ort, ort_sess, ort_input = _run_onnxruntime(onnx_path, x)
    results["onnxruntime_output_shape"] = list(y_ort.shape)
    results["onnxruntime_vs_pytorch"] = {
        "max_abs": float(np.max(np.abs(y_ort - y_torch))),
        "mean_abs": float(np.mean(np.abs(y_ort - y_torch))),
    }
    results["onnxruntime_speed"] = _timer(lambda: ort_sess.run(None, {ort_input: x}), args.warmup, args.iters)
    print("onnxruntime", json.dumps({
        "shape": list(y_ort.shape),
        "vs_pytorch": results["onnxruntime_vs_pytorch"],
        "speed": results["onnxruntime_speed"],
    }, sort_keys=True))

    try:
        import fastnn as fnn
        executor, fastnn_input, fastnn_output = _make_fastnn_executor(onnx_path)
        fx = fnn.tensor(x, list(x.shape))
        fy = executor.forward({fastnn_input: fx})[fastnn_output].numpy()
        results["fastnn_output_shape"] = list(fy.shape)
        results["fastnn_vs_pytorch"] = {
            "max_abs": float(np.max(np.abs(fy - y_torch))),
            "mean_abs": float(np.mean(np.abs(fy - y_torch))),
        }
        results["fastnn_speed"] = _timer(lambda: executor.forward({fastnn_input: fx}), args.warmup, args.iters)
        if args.profile:
            prof_result = executor.profile({fastnn_input: fx})
            profile_entries = prof_result["profile"]
            profile_by_kernel: dict[str, dict[str, float]] = {}
            for entry in profile_entries:
                kernel = entry["kernel_name"]
                elapsed_ns = float(entry["elapsed_ns"])
                bucket = profile_by_kernel.setdefault(kernel, {"count": 0.0, "total_ms": 0.0})
                bucket["count"] += 1.0
                bucket["total_ms"] += elapsed_ns / 1_000_000.0
            profile_top = sorted(
                (
                    {
                        "kernel_name": k,
                        "count": int(v["count"]),
                        "total_ms": v["total_ms"],
                        "mean_ms": v["total_ms"] / max(v["count"], 1.0),
                    }
                    for k, v in profile_by_kernel.items()
                ),
                key=lambda row: row["total_ms"],
                reverse=True,
            )[: args.profile_top]
            results["fastnn_profile_top"] = profile_top
            print("fastnn_profile_top", json.dumps(profile_top, sort_keys=True))
        print("fastnn", json.dumps({
            "shape": list(fy.shape),
            "vs_pytorch": results["fastnn_vs_pytorch"],
            "speed": results["fastnn_speed"],
        }, sort_keys=True))
    except Exception as exc:
        results["fastnn_error"] = f"{type(exc).__name__}: {exc}"
        print("fastnn_error", results["fastnn_error"])

    print("summary", json.dumps(results, sort_keys=True))
    return 0 if "fastnn_error" not in results else 2


if __name__ == "__main__":
    raise SystemExit(main())
