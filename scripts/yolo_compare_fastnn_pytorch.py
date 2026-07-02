#!/usr/bin/env python3
"""Compare YOLO ONNX inference through fastnn against PyTorch/ONNX Runtime.

Extends the original single-model script to loop over multiple model
variants (yolo26n, yolo26l) and data types (f32, u4, u8, f8, f8r, f4),
reporting accuracy (logit error vs PyTorch), speed, and optional per-kernel
profiling per (model, dtype) pair.

Usage::

    # Default: yolo26n + yolo26l, all dtypes
    python scripts/yolo_compare_fastnn_pytorch.py

    # Specific models + dtypes
    python scripts/yolo_compare_fastnn_pytorch.py \\
        --models yolo26n --dtype f32,u8 --imgsz 640 --iters 50

    # With profile + machine-readable output
    python scripts/yolo_compare_fastnn_pytorch.py \\
        --profile --profile-json /tmp/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Known model metadata
# ---------------------------------------------------------------------------

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "yolo11n": {
        "pt_name": "yolo11n.pt",
    },
    "yolo11l": {
        "pt_name": "yolo11l.pt",
    },
}

VALID_DTYPES = ("f32", "u4", "u8", "i4", "i8", "f8", "f8r", "f4")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def _dtype_to_quantize(dtype: str) -> int | str | None:
    if dtype == "f32":
        return None
    if dtype == "u4" or dtype == "u8":
        return dtype  # unsigned → pass as string for WeightDtype::U4/U8
    if dtype == "i4":
        return 4  # signed → pass as int for WeightDtype::I4
    if dtype == "i8":
        return 8  # signed → pass as int for WeightDtype::I8
    return dtype  # "f8", "f8r", "f4" — passed as string for Phase 2


# ---------------------------------------------------------------------------
# Model download & export
# ---------------------------------------------------------------------------


def _download_hf_onnx(model_name: str, onnx_path: Path) -> None:
    import urllib.request

    spec = MODEL_SPECS[model_name]
    url = f"https://huggingface.co/{spec['hf_repo']}/resolve/main/{spec['hf_file']}"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, str(onnx_path))
    print(f"  saved -> {onnx_path}")


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


# ---------------------------------------------------------------------------
# ONNX metadata
# ---------------------------------------------------------------------------


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
        "num_params": len(model.graph.initializer),
        "ops": dict(sorted(ops.items())),
    }


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# fastnn AOT executor builder
# ---------------------------------------------------------------------------


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


def _make_fastnn_executor(onnx_path: Path, dtype: str):
    import onnx
    from onnx import numpy_helper

    import fastnn as fnn
    from fastnn.io.onnx import _extract_attrs

    model = onnx.load(str(onnx_path))
    initializer_names = {init.name for init in model.graph.initializer}
    input_names = [i.name for i in model.graph.input if i.name not in initializer_names]
    output_names = [o.name for o in model.graph.output]

    input_shapes = {}
    for i in model.graph.input:
        if i.name in input_names:
            dims = []
            for d in i.type.tensor_type.shape.dim:
                dv = d.dim_value if d.HasField("dim_value") else 0
                dims.append(int(dv) if dv > 0 else 1)
            input_shapes[i.name] = dims

    params = {}
    const_values: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        arr_raw = numpy_helper.to_array(init)
        const_values[init.name] = arr_raw
        arr = arr_raw.astype(np.float32, copy=False)
        params[init.name] = fnn.tensor(arr, list(arr.shape))

    known_shapes: dict[str, list[int]] = {}
    try:
        shape_model = onnx.shape_inference.infer_shapes(model)
        for vi in list(shape_model.graph.input) + list(shape_model.graph.value_info) + list(shape_model.graph.output):
            dims = []
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                for d in tt.shape.dim:
                    dv = d.dim_value if d.HasField("dim_value") else 0
                    dims.append(int(dv) if dv > 0 else -1)
                known_shapes[vi.name] = dims
    except Exception:
        pass

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
        item: dict[str, Any] = {
            "name": name,
            "op_type": node.op_type,
            "inputs": ",".join(node.input),
            "outputs": ",".join(node.output),
        }
        if node.op_type == "Slice" and len(node.input) >= 4:
            starts = const_values.get(node.input[1])
            ends = const_values.get(node.input[2])
            axes = const_values.get(node.input[3])
            if starts is not None and ends is not None and axes is not None:
                item["starts"] = str(int(np.asarray(starts).reshape(-1)[0]))
                item["ends"] = str(int(np.asarray(ends).reshape(-1)[0]))
                axes_arr = np.asarray(axes).reshape(-1)
                item["axes"] = str(int(axes_arr[0]))
        for k, v in attrs.items():
            if k == "value" or k in item:
                continue
            item[k] = _format_attr_value(v)
        if node.op_type == "Resize" and len(node.input) >= 3:
            scales_name = node.input[2]
            if scales_name in params:
                scales = np.asarray(params[scales_name].numpy(), dtype=np.float32).reshape(-1)
                if scales.size >= 4:
                    item["scale_h"] = str(int(scales[2]))
                    item["scale_w"] = str(int(scales[3]))
        nodes.append(item)

    quantize = _dtype_to_quantize(dtype)
    return fnn.AotExecutor(nodes, params, input_names, output_names, input_shapes=input_shapes, quantize=quantize), input_names[0], output_names[0]


# ---------------------------------------------------------------------------
# Single-model × single-dtype run
# ---------------------------------------------------------------------------


def _run_fastnn(onnx_path: Path, dtype: str, x: np.ndarray, y_torch: np.ndarray, args) -> dict[str, Any]:
    import fastnn as fnn

    result: dict[str, Any] = {"dtype": dtype}

    executor, fastnn_input, fastnn_output = _make_fastnn_executor(onnx_path, dtype=dtype)

    if args.scales_json and dtype in ("u4", "u8"):
        scales_text = Path(args.scales_json).read_text()
        executor.apply_calibration(scales_text)

    fx = fnn.tensor(x, list(x.shape))
    fy = executor.forward({fastnn_input: fx})[fastnn_output].numpy()
    result["nan_check"] = {"has_nan": bool(np.isnan(fy).any()), "has_inf": bool(np.isinf(fy).any()), "min": float(fy.min()), "max": float(fy.max())}
    result["output_shape"] = list(fy.shape)

    result["vs_pytorch"] = {
        "max_abs": float(np.max(np.abs(fy - y_torch))),
        "mean_abs": float(np.mean(np.abs(fy - y_torch))),
    }

    result["speed"] = _timer(lambda: executor.forward({fastnn_input: fx}), args.warmup, args.iters)

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
        profile_summary = sorted(
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
        )
        result["profile_top"] = profile_summary[: args.profile_top]
        result["profile_summary"] = profile_summary
        result["profile_raw"] = profile_entries

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare YOLO (yolo26n/l) inference across dtypes: fastnn vs PyTorch vs ONNX Runtime"
    )
    ap.add_argument("--models", default="yolo11n,yolo11l", help="Comma-separated model names")
    ap.add_argument("--dtype", default="f32", help=f"Comma-separated dtypes to test ({','.join(VALID_DTYPES)}, default: f32)")
    ap.add_argument("--pt", default=None, help="Override .pt path (only for single-model mode)")
    ap.add_argument("--onnx", default=None, help="Override ONNX path (only for single-model mode)")
    ap.add_argument("--hf", action="store_true", default=False, help="Download ONNX from HuggingFace instead of exporting from .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="Print fastnn per-kernel profile from one measured run")
    ap.add_argument("--profile-top", type=int, default=20)
    ap.add_argument("--profile-json", type=Path, default=None, help="Write machine-readable results to this JSON path")
    ap.add_argument("--scales-json", type=Path, default=None, help="Path to calibration scales JSON (from calibrate_yolo.py)")
    args = ap.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    dtypes = [d.strip() for d in args.dtype.split(",")]
    for d in dtypes:
        if d not in VALID_DTYPES:
            print(f"error: unknown dtype '{d}' (valid: {','.join(VALID_DTYPES)})")
            return 1

    np.random.seed(args.seed)
    x = np.random.default_rng(args.seed).random((1, 3, args.imgsz, args.imgsz), dtype=np.float32)

    all_results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\n{'=' * 70}")
        print(f"  MODEL: {model_name}")
        print(f"{'=' * 70}")

        cache_dir = Path(f"/tmp/fastnn-yolo-verify/{model_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        onnx_path = cache_dir / "model.onnx"

        if args.hf:
            if not onnx_path.exists():
                _download_hf_onnx(model_name, onnx_path)
            meta = _onnx_metadata(onnx_path)
            print(f"  ONNX from HF: {onnx_path}")
            print(f"  metadata: {json.dumps(meta, sort_keys=True)}")

            # When using HF ONNX, there's no PyTorch model to compare against.
            # Use ONNX Runtime as the ground truth instead.
            y_ort, ort_sess, ort_input = _run_onnxruntime(onnx_path, x)
            y_torch = y_ort
            ref_label = "onnxruntime"
        else:
            pt_name = MODEL_SPECS[model_name]["pt_name"]
            if not onnx_path.exists():
                print(f"  exporting {pt_name} -> {onnx_path}")
                _export_yolo_pt_to_onnx(pt_name, onnx_path, args.imgsz)
            meta = _onnx_metadata(onnx_path)
            print(f"  ONNX from ultralytics export: {onnx_path}")
            print(f"  metadata: {json.dumps(meta, sort_keys=True)}")

            print(f"  loading {pt_name} for PyTorch reference ...")
            from ultralytics import YOLO
            pt_model = YOLO(pt_name)
            y_torch, x_torch = _run_pytorch(pt_model, x)
            y_ort, ort_sess, ort_input = _run_onnxruntime(onnx_path, x)
            ref_label = "pytorch"

        model_results: dict[str, Any] = {
            "onnx_metadata": meta,
            "ref_label": ref_label,
            "ref_output_shape": list(y_torch.shape),
        }

        if not args.hf:
            model_results["onnxruntime_vs_pytorch"] = {
                "max_abs": float(np.max(np.abs(y_ort - y_torch))),
                "mean_abs": float(np.mean(np.abs(y_ort - y_torch))),
            }
            model_results["pytorch_speed"] = _timer(lambda: pt_model.model(x_torch), args.warmup, args.iters)
            model_results["onnxruntime_speed"] = _timer(lambda: ort_sess.run(None, {ort_input: x}), args.warmup, args.iters)

        dtypes_results: list[dict[str, Any]] = []
        for dtype in dtypes:
            print(f"\n  --- dtype={dtype} ---")
            try:
                dr = _run_fastnn(onnx_path, dtype, x, y_torch, args)
                dtypes_results.append(dr)
                print(f"    vs_{ref_label}: max_abs={dr['vs_pytorch']['max_abs']:.6f}  mean_abs={dr['vs_pytorch']['mean_abs']:.6f}")
                print(f"    speed: {dr['speed']['mean_ms']:.2f} ms")
                if args.profile and "profile_top" in dr:
                    for entry in dr["profile_top"]:
                        print(f"      {entry['kernel_name']:40s}  count={entry['count']:4d}  total={entry['total_ms']:8.2f}ms  mean={entry['mean_ms']:8.2f}ms")
            except Exception as exc:
                import traceback
                print(f"    ERROR: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                dtypes_results.append({"dtype": dtype, "error": f"{type(exc).__name__}: {exc}"})

        model_results["dtypes"] = dtypes_results
        all_results[model_name] = model_results

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    for model_name, mr in all_results.items():
        print(f"\n  {model_name}:")
        print(f"    ref: {mr['ref_label']}  shape={mr['ref_output_shape']}")
        for dr in mr["dtypes"]:
            dtype = dr["dtype"]
            if "error" in dr:
                print(f"    {dtype:6s}: ERROR {dr['error']}")
            else:
                err = dr["vs_pytorch"]
                spd = dr["speed"]
                print(f"    {dtype:6s}: max_abs={err['max_abs']:.6f}  mean_abs={err['mean_abs']:.6f}  "
                      f"speed={spd['mean_ms']:.2f}ms ({spd['min_ms']:.2f}-{spd['max_ms']:.2f})  "
                      f"nan={dr['nan_check']['has_nan']} inf={dr['nan_check']['has_inf']}")

    if args.profile_json is not None:
        args.profile_json.parent.mkdir(parents=True, exist_ok=True)
        args.profile_json.write_text(json.dumps(all_results, indent=2, sort_keys=True))
        print(f"\nprofile_json -> {args.profile_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
