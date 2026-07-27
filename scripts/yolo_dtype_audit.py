#!/usr/bin/env python3
"""Locate the first YOLO Conv output that diverges for each fastnn dtype."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.yolo_coco_map import (
    MODEL_SPECS,
    VALID_DTYPES,
    _build_fastnn_executor,
    _export_yolo_pt_to_onnx,
    _load_coco_hf,
    _preprocess_pil,
)


def _conv_outputs(onnx_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return [node.output[0] for node in model.graph.node if node.op_type == "Conv" and node.output]


def _run(
    onnx_path: Path,
    dtype: str,
    outputs: tuple[str, ...],
    x: np.ndarray,
) -> dict[str, np.ndarray]:
    import fastnn as fnn

    executor, input_name, _ = _build_fastnn_executor(
        onnx_path, dtype, extra_outputs=outputs
    )
    tensor = fnn.tensor(x, list(x.shape))
    values = executor.forward({input_name: tensor})
    result = {name: np.asarray(values[name].numpy()).copy() for name in outputs}
    del values, tensor, executor
    gc.collect()
    return result


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(actual)
    nonfinite = int(actual.size - np.count_nonzero(finite))
    if nonfinite:
        return {"nonfinite": nonfinite, "elements": int(actual.size)}
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    ref64 = reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(ref64 * ref64)))
    return {
        "nonfinite": 0,
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "reference_rms": ref_rms,
        "normalized_rmse": rmse / max(ref_rms, 1e-12),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="yolo11n")
    parser.add_argument("--dtypes", default="all")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dtypes = list(VALID_DTYPES) if args.dtypes == "all" else [
        item.strip() for item in args.dtypes.split(",") if item.strip()
    ]
    unknown = sorted(set(dtypes) - set(VALID_DTYPES))
    if unknown:
        parser.error(f"unknown dtypes: {','.join(unknown)}")
    if "f32" in dtypes:
        dtypes.remove("f32")

    cache_dir = Path("/tmp/fastnn-yolo-coco") / args.model
    cache_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cache_dir / f"model-{args.imgsz}.onnx"
    if not onnx_path.exists():
        _export_yolo_pt_to_onnx(
            MODEL_SPECS[args.model]["pt_name"], onnx_path, args.imgsz
        )

    outputs = tuple(_conv_outputs(onnx_path))
    if not outputs:
        raise RuntimeError(f"{onnx_path} contains no Conv outputs")

    _, _, _, images = _load_coco_hf(1)
    x, _, _ = _preprocess_pil(images[0], args.imgsz)
    x = x[np.newaxis]

    print(f"Collecting F32 references for {len(outputs)} Conv outputs")
    reference = _run(onnx_path, "f32", outputs, x)
    report: dict[str, Any] = {
        "model": args.model,
        "image_size": args.imgsz,
        "seed": args.seed,
        "threshold": args.threshold,
        "conv_outputs": len(outputs),
        "dtypes": {},
    }

    for dtype in dtypes:
        print(f"Auditing {dtype}")
        actual = _run(onnx_path, dtype, outputs, x)
        layers: list[dict[str, Any]] = []
        first_failure: dict[str, Any] | None = None
        worst: dict[str, Any] | None = None
        for index, name in enumerate(outputs):
            metrics = _metrics(actual[name], reference[name])
            entry = {"index": index, "name": name, **metrics}
            layers.append(entry)
            score = float("inf") if metrics["nonfinite"] else metrics["normalized_rmse"]
            if worst is None or score > worst["score"]:
                worst = {"index": index, "name": name, "score": score}
            if first_failure is None and (
                metrics["nonfinite"] or metrics["normalized_rmse"] > args.threshold
            ):
                first_failure = entry
        report["dtypes"][dtype] = {
            "first_failure": first_failure,
            "worst": worst,
            "layers": layers,
        }
        print(f"  first_failure={first_failure}")
        del actual
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
