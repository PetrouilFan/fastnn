#!/usr/bin/env python3
"""Concurrency test runner for fastnn YOLO quantization paths."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_profile_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _build_fastnn_profile(profile: dict[str, Any], output_json: Path) -> Path:
    import fastnn
    from fastnn.io import onnx as fastnn_onnx

    onnx_path = Path(profile["onnx_path"])
    out_path = Path(profile["fastnn_json_path"])
    calibration = profile.get("calibration_json_path")
    calibration_data = None
    if calibration:
        calibration_data = Path(calibration)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fastnn_onnx.prepare_from_onnx(
        str(onnx_path),
        str(output_json),
        calibration=str(calibration_data) if calibration_data else None,
    )
    return out_path


def _run_fastnn_profile(profile: dict[str, Any], x: np.ndarray):
    import fastnn

    input_name = profile["input_name"]
    output_name = profile["output_name"]
    prepared_path = Path(profile["fastnn_json_path"])
    extra_args: dict[str, Any] = {}
    if "quantization_bit_width" in profile:
        extra_args["quantization_bit_width"] = int(profile["quantization_bit_width"])
    if "use_quantized_paths" in profile:
        extra_args["use_quantized_paths"] = bool(profile["use_quantized_paths"])
    executor = fastnn.AotExecutor.from_prepared(
        str(prepared_path),
        input_names=[input_name],
        output_names=[output_name],
        **extra_args,
    )
    def _forward() -> np.ndarray:
        out = executor.run({input_name: x})
        if isinstance(out, dict):
            return _to_numpy(list(out.values())[0])
        return _to_numpy(out)
    return _forward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_json", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("/tmp/yolo_quant_profile.json"))
    args = parser.parse_args()
    profile = _load_profile_json(args.profile_json)

    # Prepare fastnn profile if missing.
    if not Path(profile["fastnn_json_path"]).exists():
        _build_fastnn_profile(profile, Path(profile["fastnn_json_path"]))

    model = None
    onnx_path = Path(profile["onnx_path"])
    if not onnx_path.exists():
        model = _export_yolo_pt_to_onnx(profile["pt_path"], onnx_path, int(profile["imgsz"]))

    x = np.load(profile["input_npy_path"]) if Path(profile["input_npy_path"]).exists() else np.zeros(
        tuple(profile["input_shape"]), dtype=np.float32
    )

    results: dict[str, Any] = {
        "profile": profile["name"],
        "quantization_bit_width": profile.get("quantization_bit_width"),
        "use_quantized_paths": profile.get("use_quantized_paths", False),
    }

    try:
        fastnn_fn = _run_fastnn_profile(profile, x)
        results["fastnn_speed"] = _timer(fastnn_fn, args.warmup, args.iters)
    except Exception as exc:
        results["fastnn_error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"profile_complete {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
