#!/usr/bin/env python3
"""Locate the first GPT-2 MatMul output that diverges under grouped W4A8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

import fastnn as fnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fnn", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--quantize", choices=("w4a8-g32", "w4a8-g64", "w4a8-g128"), default="w4a8-g128")
    parser.add_argument("--token", type=int, default=42)
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor(value: np.ndarray):
    return fnn.tensor(value, list(value.shape))


def state_names() -> list[tuple[str, str]]:
    return [
        (f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}")
        for layer in range(12)
        for kind in ("key", "value")
    ]


def linear_outputs(path: Path) -> list[str]:
    model = onnx.load(str(path), load_external_data=False)
    return [
        node.output[0]
        for node in model.graph.node
        if node.op_type in {"Gemm", "MatMul"} and node.output
    ]


def augmented_onnx(source: Path, outputs: list[str], destination: Path) -> Path:
    model = onnx.load(str(source), load_external_data=True)
    existing = {value.name for value in model.graph.output}
    known = {
        value.name: value
        for value in (*model.graph.value_info, *model.graph.output, *model.graph.input)
    }
    for name in outputs:
        if name not in existing:
            model.graph.output.append(known.get(name, helper.make_empty_tensor_value_info(name)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return destination


def feeds(token_id: int) -> dict[str, np.ndarray]:
    values = {
        "input_ids": np.array([[token_id]], dtype=np.int64),
        "attention_mask": np.ones((1, 1), dtype=np.int64),
    }
    values.update({name: np.zeros((1, 12, 0, 64), dtype=np.float32) for name, _ in state_names()})
    return values


def run_fastnn(args: argparse.Namespace, outputs: list[str], quantize: str | None) -> dict[str, np.ndarray]:
    model = fnn.build_model_from_fnn(
        str(args.fnn),
        symbolic_dim_bounds={"batch_size": 1, "past_sequence_length": args.max_context},
        quantize=quantize,
        diagnostic_outputs=outputs,
    )
    bindings = dict(state_names())
    model.configure_state(
        bindings,
        {name: tensor(np.zeros((1, 12, 0, 64), dtype=np.float32)) for name in bindings},
    )
    session = model.create_session()
    model_inputs = {name: tensor(value) for name, value in feeds(args.token).items() if name not in bindings}
    result = session.prefill(model_inputs, True)
    return {name: np.asarray(result[name].numpy()).copy() for name in outputs}


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if actual.shape != reference.shape:
        return {"shape_mismatch": [list(actual.shape), list(reference.shape)]}
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
    args = parse_args()
    outputs = linear_outputs(args.onnx)
    if not outputs:
        raise RuntimeError("ONNX graph has no Gemm or MatMul outputs")

    augmented = augmented_onnx(args.onnx, outputs, args.output.with_suffix(".onnx"))
    ort_session = ort.InferenceSession(str(augmented), providers=["CPUExecutionProvider"])
    ort_values = ort_session.run(outputs, feeds(args.token))
    ort_reference = dict(zip(outputs, ort_values, strict=True))

    f32 = run_fastnn(args, outputs, None)
    quantized = run_fastnn(args, outputs, args.quantize)
    layers = []
    first_failure = None
    worst = None
    for index, name in enumerate(outputs):
        f32_vs_ort = metrics(f32[name], ort_reference[name])
        w4a8_vs_f32 = metrics(quantized[name], f32[name])
        entry = {"index": index, "name": name, "f32_vs_ort": f32_vs_ort, "w4a8_vs_f32": w4a8_vs_f32}
        layers.append(entry)
        score = float("inf") if "normalized_rmse" not in w4a8_vs_f32 else w4a8_vs_f32["normalized_rmse"]
        if worst is None or score > worst["score"]:
            worst = {"index": index, "name": name, "score": score}
        if first_failure is None and score > args.threshold:
            first_failure = entry

    report = {
        "quantize": args.quantize,
        "token": args.token,
        "threshold": args.threshold,
        "matmul_outputs": len(outputs),
        "first_failure": first_failure,
        "worst": worst,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("quantize", "matmul_outputs", "first_failure", "worst")}, indent=2))
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
