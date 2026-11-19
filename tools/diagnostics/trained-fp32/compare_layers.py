#!/usr/bin/env python3
"""Per-stage comparison of the fn3 and ORT trained-GPT-2 FP32 runs.

Inputs are the two probe directories:

* ``--fn3``: written by ``tests/fn3_trained_layers.rs`` (one subdirectory per
  phase, ``index.json`` plus raw little-endian f32 files);
* ``--ort``: written by ``extract_intermediates.py`` (``.npy`` files).

Both sides execute the same ONNX graph with graph optimisations disabled; only
the implementation differs, so a stage delta is an implementation delta.

Every metric keeps the absolute error and the scale relative to it visible:

* ``max_abs`` / ``mean_abs``: the suite policy's metrics;
* ``mean_signed``: the common-mode component the suite ``mean_abs`` bound is
  sensitive to;
* ``centered_rms``: the ranking-relevant spread after removing that component;
* ``relative_rms``: ``rms(delta) / rms(reference)``;
* ``cosine`` and ``kl_reference_to_actual`` for logits.

The logits section additionally projects the final hidden-state delta through
the tied embedding matrix in float64. That separates "the lm_head matmul is
wrong" from "an upstream hidden-state delta is amplified by the large
common-mode logit offset".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODEL_SHA256 = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d"
ATOL = 5e-4
RTOL = 1e-4
MEAN_ABS_BOUND = 5e-5


def load_f32(index_path: Path, root: Path) -> dict[str, np.ndarray]:
    index = json.loads(index_path.read_text())
    entries = index.get("tensors") or index.get("captured")
    tensors = {}
    for entry in entries:
        path = root / entry["file"]
        if path.suffix == ".npy":
            values = np.load(path).reshape(-1)
        else:
            values = np.fromfile(path, dtype="<f4")
        tensors[entry["name"]] = values.reshape(entry["shape"])
    return tensors


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    if a.size != r.size:
        raise SystemExit(f"size mismatch: {a.size} vs {r.size}")
    delta = a - r
    rms_reference = float(np.sqrt(np.mean(r * r)))
    rms_delta = float(np.sqrt(np.mean(delta * delta)))
    centered = delta - delta.mean()
    row_mean = None
    # A per-row mean is meaningful when the last axis is a feature/vocabulary
    # axis; that is where GPT-2's large common-mode logit offset lives.
    if actual.ndim >= 2 and actual.shape[-1] > 8:
        rows = delta.reshape(-1, actual.shape[-1])
        row_mean = {
            "mean_of_row_mean": float(rows.mean(axis=1).mean()),
            "std_of_row_mean": float(rows.mean(axis=1).std()),
        }
    return {
        "elements": int(a.size),
        "rms_reference": rms_reference,
        "rms_delta": rms_delta,
        "relative_rms": rms_delta / rms_reference if rms_reference else None,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "mean_signed": float(delta.mean()),
        "std_delta": float(delta.std()),
        "centered_rms": float(np.sqrt(np.mean(centered * centered))),
        "centered_max_abs": float(np.max(np.abs(centered))),
        "cosine": float(
            np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r))
        )
        if np.linalg.norm(a) and np.linalg.norm(r)
        else None,
        "finite": bool(np.isfinite(a).all()),
        "row_mean": row_mean,
    }


def logits_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    result = metrics(actual, reference)
    a = actual.astype(np.float64).reshape(-1)
    r = reference.astype(np.float64).reshape(-1)
    delta = a - r
    bound = ATOL + RTOL * np.abs(r)
    violations = int(np.count_nonzero(np.abs(delta) > bound))
    order = np.argsort(-r)
    actual_order = np.argsort(-a)
    p = np.exp(r - r.max())
    p /= p.sum()
    q = np.exp(a - a.max())
    q /= q.sum()
    ref_top5 = [int(i) for i in order[:5]]
    actual_top5 = [int(i) for i in actual_order[:5]]
    result.update(
        {
            "elementwise_bound": {"atol": ATOL, "rtol": RTOL},
            "elementwise_violations": violations,
            "elementwise_passed": violations == 0,
            "mean_abs_bound": MEAN_ABS_BOUND,
            "mean_abs_within_bound": bool(np.mean(np.abs(delta)) <= MEAN_ABS_BOUND),
            "argmax": int(np.argmax(a)),
            "reference_argmax": int(np.argmax(r)),
            "top1_agree": bool(np.argmax(a) == np.argmax(r)),
            "top1_margin_reference": float(r[order[0]] - r[order[1]]),
            "top1_margin_actual": float(a[actual_order[0]] - a[actual_order[1]]),
            "top5_reference": ref_top5,
            "top5_actual": actual_top5,
            "top5_overlap": len(set(ref_top5) & set(actual_top5)),
            "kl_reference_to_actual": float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300)))),
        }
    )
    return result


def load_embedding(model_path: Path) -> np.ndarray:
    model = onnx.load(str(model_path), load_external_data=False)
    for initializer in model.graph.initializer:
        if initializer.name == "transformer.wte.weight":
            # The graph reaches the logits through an explicit Transpose node;
            # the token embedding is stored [vocab, hidden].
            return numpy_helper.to_array(initializer).astype(np.float64).T
    raise SystemExit("transformer.wte.weight not found")


def stage_order(name: str) -> tuple:
    """Stable model order: embedding, then layer, then position inside the block."""
    if name == "/transformer/Add_output_0":
        return (0, 0, 0)
    if name.startswith("present."):
        layer = int(name.split(".")[1])
        return (1, layer, 99)
    if not name.startswith("/transformer/h."):
        return (3, 0, 0)
    rest = name[len("/transformer/h."):]
    layer_text, _, tail = rest.partition("/")
    order = [
        "/ln_1/ReduceMean_output_0",
        "/ln_1/ReduceMean_1_output_0",
        "/ln_1/Add_output_0",
        "/ln_1/Sqrt_output_0",
        "/ln_1/Div_output_0",
        "/ln_1/Mul_output_0",
        "/ln_1/Add_1_output_0",
        "/attn/c_attn/Gemm_output_0",
        "/attn/Mul_output_0",
        "/attn/Mul_1_output_0",
        "/attn/MatMul_output_0",
        "/attn/Add_output_0",
        "/attn/Softmax_output_0",
        "/attn/MatMul_1_output_0",
        "/attn/c_proj/Gemm_output_0",
        "/Add_output_0",
        "/ln_2/Add_1_output_0",
        "/mlp/c_fc/Gemm_output_0",
        "/mlp/act/Mul_output_0",
        "/mlp/act/Add_output_0",
        "/mlp/act/Tanh_output_0",
        "/mlp/act/Add_1_output_0",
        "/mlp/act/Mul_3_output_0",
        "/mlp/c_proj/Gemm_output_0",
        "/Add_1_output_0",
    ]
    position = order.index("/" + tail) if "/" + tail in order else 50
    return (1, int(layer_text), position)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn3", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report: dict = {
        "tolerance": {"atol": ATOL, "rtol": RTOL, "mean_abs_bound": MEAN_ABS_BOUND},
        "model": str(args.model),
        "model_sha256_expected": MODEL_SHA256,
        "phases": {},
    }
    for phase in ("prefill", "decode1", "decode2"):
        fn3_root = args.fn3 / phase
        ort_root = args.ort / phase
        fn3 = load_f32(fn3_root / "index.json", fn3_root)
        ort = load_f32(ort_root / "index.json", ort_root)
        names = sorted(set(fn3) & set(ort), key=stage_order)
        missing = sorted((set(fn3) | set(ort)) - (set(fn3) & set(ort)))
        stages = {}
        for name in names:
            a, r = fn3[name], ort[name]
            if a.shape != r.shape:
                raise SystemExit(f"{phase}/{name}: shape {a.shape} vs {r.shape}")
            entry = metrics(a, r)
            if name == "logits":
                entry = logits_metrics(a, r)
            stages[name] = entry
        report["phases"][phase] = {"missing_on_one_side": missing, "stages": stages}

    embedding = load_embedding(args.model)

    for phase, entry in report["phases"].items():
        stages = entry["stages"]
        hidden_name = "/transformer/h.11/Add_1_output_0"
        normed_name = "/transformer/ln_f/Add_1_output_0"
        if hidden_name not in stages or "logits" not in stages:
            continue
        fn3_normed = load_f32(args.fn3 / phase / "index.json", args.fn3 / phase)[normed_name]
        ort_normed = load_f32(args.ort / phase / "index.json", args.ort / phase)[normed_name]
        # [seq, hidden] @ [hidden, vocab] in float64: is the lm_head arithmetic
        # consistent with the normed hidden state each side produced?
        fn3_projection = fn3_normed.reshape(-1, fn3_normed.shape[-1]).astype(np.float64) @ embedding
        ort_projection = ort_normed.reshape(-1, ort_normed.shape[-1]).astype(np.float64) @ embedding
        fn3_logits = load_f32(args.fn3 / phase / "index.json", args.fn3 / phase)["logits"].reshape(-1, 50257)
        ort_logits = load_f32(args.ort / phase / "index.json", args.ort / phase)["logits"].reshape(-1, 50257)
        delta_h = (
            fn3_normed.reshape(-1, fn3_normed.shape[-1]).astype(np.float64)
            - ort_normed.reshape(-1, ort_normed.shape[-1]).astype(np.float64)
        )
        predicted_delta = delta_h @ embedding
        observed_delta = fn3_logits.astype(np.float64) - ort_logits.astype(np.float64)
        common_delta = delta_h.mean(axis=0)
        common_projection = common_delta @ embedding
        entry["lm_head_attribution"] = {
            "note": (
                "projection is float64; fn3 and ORT logits are float32. "
                "residual is fn3/ORT lm_head float32 rounding, not the stage delta"
            ),
            "fn3_normed_vs_f64_projection": metrics(fn3_projection, fn3_logits),
            "ort_normed_vs_f64_projection": metrics(ort_projection, ort_logits),
            "predicted_delta_vs_observed_delta": metrics(predicted_delta, observed_delta),
            "observed_delta_constant_component": float(observed_delta.mean()),
            "predicted_delta_constant_component": float(predicted_delta.mean()),
            "common_delta_projection_std_over_vocab": float(common_projection.std()),
            "common_delta_projection_mean": float(common_projection.mean()),
            "delta_h_mean_abs": float(np.abs(delta_h).mean()),
            "delta_h_rms": float(np.sqrt(np.mean(delta_h * delta_h))),
        }

    output = args.out or args.fn3 / "layers.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")

    # Compact human-readable table.
    header = (
        f"{'stage':<52} {'mean_abs':>10} {'max_abs':>10} {'signed':>10} "
        f"{'centered':>10} {'rel_rms':>10} {'cos':>14}"
    )
    for phase, entry in report["phases"].items():
        print(f"\n== {phase} ==")
        print(header)
        for name, stage in entry["stages"].items():
            if not name.startswith("/transformer/h.") and name != "/transformer/Add_output_0":
                continue
            print(
                f"{name:<52} {stage['mean_abs']:>10.3e} {stage['max_abs']:>10.3e} "
                f"{stage['mean_signed']:>+10.3e} {stage['centered_rms']:>10.3e} "
                f"{stage['relative_rms']:>10.3e} {stage['cosine']:>14.12f}"
            )
        for name in ("/transformer/ln_f/Add_1_output_0", "logits"):
            stage = entry["stages"].get(name)
            if stage is None:
                continue
            print(
                f"{name:<52} {stage['mean_abs']:>10.3e} {stage['max_abs']:>10.3e} "
                f"{stage['mean_signed']:>+10.3e} {stage['centered_rms']:>10.3e} "
                f"{stage['relative_rms']:>10.3e} {stage['cosine']:>14.12f}"
            )
        logits = entry["stages"]["logits"]
        print(
            f"  logits: violations {logits['elementwise_violations']}, "
            f"mean_abs {logits['mean_abs']:.3e} (bound {MEAN_ABS_BOUND:.0e}: "
            f"{'pass' if logits['mean_abs_within_bound'] else 'FAIL'}), "
            f"top1 {logits['argmax']}/{logits['reference_argmax']} agree "
            f"{logits['top1_agree']}, top5 {logits['top5_overlap']}/5, "
            f"KL {logits['kl_reference_to_actual']:.3e}"
        )
        attribution = entry.get("lm_head_attribution")
        if attribution:
            print(
                "  lm_head attribution: predicted vs observed delta "
                f"rms {attribution['predicted_delta_vs_observed_delta']['rms_delta']:.3e}, "
                f"relative residual "
                f"{attribution['predicted_delta_vs_observed_delta']['relative_rms']:.3e}; "
                f"common delta {attribution['observed_delta_constant_component']:+.3e} "
                f"predicted {attribution['predicted_delta_constant_component']:+.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
