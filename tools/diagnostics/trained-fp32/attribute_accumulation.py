#!/usr/bin/env python3
"""Prove which arithmetic order produced a captured fn3 matmul result.

``src/fn3/backend/cpu/kernels/matmul.rs`` documents one ``f32`` accumulator,
ascending ``k``, no fused multiply-add. This script recomputes selected
projections with exactly that order in NumPy (elementwise ``f32`` multiply and
add, vectorised over ``M`` and ``N`` only) and compares the result bit for bit
with the fn3 probe. It also computes a float64 reference and NumPy's float32
BLAS dot, so the fn3-vs-ORT difference can be read as accumulation order rather
than as a wrong value.

For ``Gemm`` the importer composes ``MatMul`` then ``Add`` (bias), and this
script follows that order: bias is added once after the whole accumulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

# (phase, layer, projection, activation slot, weight initializer)
CASES = [
    (
        "prefill",
        0,
        "attn.c_attn",
        "/transformer/h.0/ln_1/Add_1_output_0",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.bias",
    ),
    (
        "prefill",
        2,
        "mlp.c_proj (column 447, the massive-activation outlier)",
        "/transformer/h.2/mlp/act/Mul_3_output_0",
        "transformer.h.2.mlp.c_proj.weight",
        "transformer.h.2.mlp.c_proj.bias",
    ),
    (
        "prefill",
        11,
        "mlp.c_fc",
        "/transformer/h.11/ln_2/Add_1_output_0",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_fc.bias",
    ),
]


def load_index(root: Path, phase: str) -> dict[str, np.ndarray]:
    index = json.loads((root / phase / "index.json").read_text())
    entries = index.get("tensors") or index.get("captured")
    values = {}
    for entry in entries:
        path = root / phase / entry["file"]
        data = np.load(path) if path.suffix == ".npy" else np.fromfile(path, dtype="<f4")
        values[entry["name"]] = data.reshape(entry["shape"])
    return values


def initializers(model_path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    model = onnx.load(str(model_path), load_external_data=False)
    found = {}
    for tensor in model.graph.initializer:
        if tensor.name in wanted:
            found[tensor.name] = numpy_helper.to_array(tensor)
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"missing initializers: {sorted(missing)}")
    return found


def sequential_f32(rows: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """out[m, n] = ((0 + x[m,0]*w[n,0]) + x[m,1]*w[n,1]) + ... in f32."""
    rows = rows.astype(np.float32).reshape(-1, rows.shape[-1])
    weight = weight.astype(np.float32)
    accumulator = np.zeros((rows.shape[0], weight.shape[0]), dtype=np.float32)
    for k in range(rows.shape[1]):
        product = (rows[:, k : k + 1] * weight[:, k].reshape(1, -1)).astype(np.float32)
        accumulator = (accumulator + product).astype(np.float32)
    return accumulator


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    delta = a - r
    return {
        "elements": int(a.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "bit_identical": int(
            np.count_nonzero(
                np.asarray(actual, dtype=np.float32).view(np.uint32)
                == np.asarray(reference, dtype=np.float32).view(np.uint32)
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn3", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {"cases": []}
    for phase, layer, label, slot, weight_name, bias_name in CASES:
        fn3 = load_index(args.fn3, phase)
        ort = load_index(args.ort, phase)
        tensors = initializers(args.model, {weight_name, bias_name})
        rows = fn3[slot].reshape(-1, fn3[slot].shape[-1])
        # The Gemm input is the flattened [seq, hidden] reshape of the captured
        # activation, which is already [1, seq, hidden] here.
        weight = tensors[weight_name]
        bias = tensors[bias_name]
        if weight.shape[0] != rows.shape[1]:
            raise SystemExit(f"{weight_name}: {weight.shape} does not contract {rows.shape}")

        sequential = sequential_f32(rows, weight.T) + bias.astype(np.float32)
        blas = (rows.astype(np.float32) @ weight.astype(np.float32)) + bias.astype(np.float32)
        exact = (
            rows.astype(np.float64) @ weight.astype(np.float64)
            + bias.astype(np.float64)
        )

        # Recover the same output from the probe by name. The caller knows the
        # slot from the case label's prefix.
        if "c_attn" in label:
            output_name = f"/transformer/h.{layer}/attn/c_attn/Gemm_output_0"
        elif "c_proj" in label:
            output_name = f"/transformer/h.{layer}/mlp/c_proj/Gemm_output_0"
        else:
            output_name = f"/transformer/h.{layer}/mlp/c_fc/Gemm_output_0"
        fn3_output = fn3[output_name].reshape(-1, weight.shape[1])
        ort_output = ort[output_name].reshape(-1, weight.shape[1])

        case = {
            "phase": phase,
            "layer": layer,
            "projection": label,
            "input": slot,
            "weight": weight_name,
            "shape": list(fn3_output.shape),
            "fn3_vs_sequential_f32": compare(fn3_output, sequential),
            "fn3_vs_float64": compare(fn3_output, exact),
            "ort_vs_float64": compare(ort_output, exact),
            "ort_vs_blas_f32": compare(ort_output, blas),
            "sequential_f32_vs_float64": compare(sequential, exact),
            "blas_f32_vs_float64": compare(blas, exact),
        }
        report["cases"].append(case)
        print(
            f"{phase} h.{layer} {label}: fn3==sequential "
            f"{case['fn3_vs_sequential_f32']['bit_identical']}/{case['fn3_vs_sequential_f32']['elements']} "
            f"(max {case['fn3_vs_sequential_f32']['max_abs']:.3e}); "
            f"fn3 vs f64 rms {case['fn3_vs_float64']['rms']:.3e}, "
            f"ORT vs f64 rms {case['ort_vs_float64']['rms']:.3e}"
        )

    output = args.out or args.fn3 / "accumulation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
