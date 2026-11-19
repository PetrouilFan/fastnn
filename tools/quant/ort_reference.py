#!/usr/bin/env python3
"""Independent ONNX Runtime reference for the trained GPT-2 prefill+decode.

Writes raw little-endian f32 logits and KV payloads into --out.  Nothing here
imports FastNN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13,
          4874, 2402, 257]
DECODE = [640, 11]
LAYERS = 12
KV_HEADS = 12
HEAD_DIM = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def session(model: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )


def run(sess, tokens, position_ids, past):
    feeds = {
        "input_ids": np.asarray([tokens], dtype=np.int64),
        "attention_mask": np.ones(
            (1, int(past[0][0].shape[2]) + len(tokens)), dtype=np.int64
        ),
        "position_ids": np.asarray([position_ids], dtype=np.int64),
    }
    for layer in range(LAYERS):
        feeds[f"past_key_values.{layer}.key"] = past[layer][0]
        feeds[f"past_key_values.{layer}.value"] = past[layer][1]
    names = ["logits"] + [
        f"present.{layer}.{kind}" for layer in range(LAYERS) for kind in ("key", "value")
    ]
    values = sess.run(names, feeds)
    logits = values[0]
    present = [
        (values[1 + 2 * layer], values[2 + 2 * layer]) for layer in range(LAYERS)
    ]
    return logits, present


def empty_past():
    return [
        (
            np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
            np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
        )
        for _ in range(LAYERS)
    ]


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    top1 = int(np.argmax(actual))
    ref_top1 = int(np.argmax(reference))
    order = np.argsort(-reference.ravel())
    top5 = set(int(i) for i in order[:5])
    actual_top5 = set(int(i) for i in np.argsort(-actual.ravel())[:5])
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "reference_rms": float(np.sqrt(np.mean(ref * ref))),
        "argmax": top1,
        "reference_argmax": ref_top1,
        "top1_agree": top1 == ref_top1,
        "top5_overlap": len(top5 & actual_top5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sess = session(args.model)

    logits, present = run(sess, PROMPT, list(range(len(PROMPT))), empty_past())
    np.save(args.out / "ort.prefill.p13.logits.npy", logits)
    np.save(args.out / "ort.prefill.p13.last_logits.npy", logits[:, -1, :])

    report = {
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "model_sha256": sha256(args.model),
        "provider": sess.get_providers(),
        "prompt": PROMPT,
        "decode": DECODE,
        "prefill": {
            "shape": list(logits.shape),
            "argmax": int(np.argmax(logits[0, -1, :])),
            "top5": [int(i) for i in np.argsort(-logits[0, -1, :])[:5]],
        },
        "steps": [],
    }

    past = present
    for index, token in enumerate(DECODE, start=1):
        total = len(PROMPT) + index
        logits, present = run(
            sess, [token], [total - 1], past
        )
        np.save(args.out / f"ort.decode.step{index}.logits.npy", logits[:, -1, :])
        for layer in range(LAYERS):
            np.save(
                args.out / f"ort.decode.step{index}.present.{layer}.key.npy",
                present[layer][0],
            )
            np.save(
                args.out / f"ort.decode.step{index}.present.{layer}.value.npy",
                present[layer][1],
            )
        report["steps"].append(
            {
                "step": index,
                "token": token,
                "past": total - 1,
                "shape": list(logits.shape),
                "argmax": int(np.argmax(logits[0, 0, :])),
                "top5": [int(i) for i in np.argsort(-logits[0, 0, :])[:5]],
                "cache_extent": int(present[0][0].shape[2]),
            }
        )
        past = present

    (args.out / "ort.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
