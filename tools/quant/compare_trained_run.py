#!/usr/bin/env python3
"""Compare the release-build fn3 trained-GPT-2 run with ONNX Runtime.

Inputs:
  * ``--fn3``: directory written by the ignored Rust test
    ``trained_gpt2_fp32_prefill_and_decode_match_ort``;
  * ``--ort``: directory written by ``tools/quant/ort_reference.py``;
  * ``--fixtures``: the committed ``fixtures/parity/trained-gpt2`` suite.

The script first proves the ORT reference reproduces the committed fixture
bytes, then reports the fn3-vs-ORT metrics.  It keeps the documented tolerance
(``atol = 5e-4``, ``rtol = 1e-4``) and reports the suite-level ``mean_abs``
bound separately, so a failure is visible instead of being tuned away.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ATOL = 5e-4
RTOL = 1e-4
MEAN_ABS_BOUND = 5e-5


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    delta = actual - reference
    bound = ATOL + RTOL * np.abs(reference)
    violations = np.count_nonzero(np.abs(delta) > bound)
    centered = delta - delta.mean()
    p = np.exp(reference - reference.max())
    p /= p.sum()
    q = np.exp(actual - actual.max())
    q /= q.sum()
    top1 = int(np.argmax(actual))
    ref_top1 = int(np.argmax(reference))
    actual_top5 = {int(i) for i in np.argsort(-actual)[:5]}
    ref_top5 = {int(i) for i in np.argsort(-reference)[:5]}
    order = np.argsort(-reference)
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "mean_abs_within_suite_bound": bool(np.mean(np.abs(delta)) <= MEAN_ABS_BOUND),
        "elementwise_violations": int(violations),
        "elementwise_passed": bool(violations == 0),
        "centered_max_abs": float(np.max(np.abs(centered))),
        "centered_rmse": float(np.sqrt(np.mean(centered * centered))),
        "kl_reference_to_actual": float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300)))),
        "argmax": top1,
        "reference_argmax": ref_top1,
        "top1_agree": bool(top1 == ref_top1),
        "top1_margin": float(reference[order[0]] - reference[order[1]]),
        "top5_overlap": len(actual_top5 & ref_top5),
        "reference_top5": [int(i) for i in order[:5]],
        "actual_top5": [int(i) for i in np.argsort(-actual)[:5]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn3", required=True, type=Path)
    parser.add_argument("--ort", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report: dict = {"tolerance": {"atol": ATOL, "rtol": RTOL, "mean_abs_bound": MEAN_ABS_BOUND}}
    steps = [
        ("prefill.p13.last_logits", "fn3.prefill.p13.last_logits.f32", "ort.prefill.p13.last_logits.npy", "prefill.p13.last_logits"),
        ("decode.step1.logits", "fn3.decode.step1.logits.f32", "ort.decode.step1.logits.npy", "decode.step1.logits"),
        ("decode.step2.logits", "fn3.decode.step2.logits.f32", "ort.decode.step2.logits.npy", None),
    ]
    for label, fn3_name, ort_name, fixture_name in steps:
        fn3 = np.fromfile(args.fn3 / fn3_name, dtype="<f4")
        ort = np.load(args.ort / ort_name).reshape(-1).astype(np.float32)
        entry = {
            "fn3": str(args.fn3 / fn3_name),
            "ort": str(args.ort / ort_name),
            "fn3_vs_ort": metrics(fn3, ort),
        }
        if fixture_name is not None:
            payload = (args.fixtures / "tensors" / f"{fixture_name}.f32").read_bytes()
            committed = np.frombuffer(payload, dtype="<f4")
            entry["fixture"] = fixture_name
            entry["ort_reproduces_committed_fixture_exactly"] = bool(
                np.array_equal(ort, committed) and sha256(payload) == sha256(ort.tobytes())
            )
            entry["fn3_vs_committed"] = metrics(fn3, committed)
        report[label] = entry
        m = entry["fn3_vs_ort"]
        print(
            f"{label}: max_abs {m['max_abs']:.3e} mean_abs {m['mean_abs']:.3e} "
            f"violations {m['elementwise_violations']} "
            f"mean_abs_bound {m['mean_abs_within_suite_bound']} "
            f"top1 {m['argmax']}/{m['reference_argmax']} top5 {m['top5_overlap']}/5 "
            f"KL {m['kl_reference_to_actual']:.3e}"
        )
        if fixture_name:
            print(
                f"  ORT == committed fixture: "
                f"{entry['ort_reproduces_committed_fixture_exactly']}"
            )
    output = args.out or args.fn3 / "trained_parity.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
