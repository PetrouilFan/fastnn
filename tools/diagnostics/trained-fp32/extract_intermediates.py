#!/usr/bin/env python3
"""Materialise per-layer intermediates from the pinned trained GPT-2 with ORT.

The committed parity suite stores only the final logits. This script rebuilds
the same reference run with the intermediate values promoted to graph outputs,
so ``compare_layers.py`` can attribute an fn3-vs-ORT difference to a stage.

Reference conditions are the fixture producer's, not this script's choice:

* ONNX Runtime CPU, ``ORT_DISABLE_ALL``, one intra-op and one inter-op thread,
  sequential execution (``fixtures/parity/trained-gpt2/manifest.json``);
* the pinned 498,126,358-byte graph, whose SHA-256 is checked before the run;
* the fixture prompt and the two cached decode tokens.

The graph is modified in memory only: nothing is written except the requested
``.npy`` payloads and ``report.json``. That keeps the run comparable with the
unmodified ORT reference, and the script proves the logits still match the
committed bytes exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

MODEL_SHA256 = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d"
MODEL_BYTES = 498_126_358
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13,
          4874, 2402, 257]
DECODE = [640, 11]
LAYERS = 12
KV_HEADS = 12
HEAD_DIM = 64

# Stages captured for every block. Keep in sync with
# tests/fn3_trained_layers.rs::wanted().
EVERY_LAYER = [
    "/ln_1/Add_1_output_0",
    "/attn/c_attn/Gemm_output_0",
    "/attn/Softmax_output_0",
    "/attn/MatMul_1_output_0",
    "/attn/c_proj/Gemm_output_0",
    "/Add_output_0",
    "/ln_2/Add_1_output_0",
    "/mlp/c_fc/Gemm_output_0",
    "/mlp/act/Mul_3_output_0",
    "/mlp/c_proj/Gemm_output_0",
    "/Add_1_output_0",
]
LAYER_ZERO = [
    # LayerNorm decomposition: mean, variance, epsilon add, sqrt, div, scale.
    "/ln_1/ReduceMean_output_0",
    "/ln_1/ReduceMean_1_output_0",
    "/ln_1/Add_output_0",
    "/ln_1/Sqrt_output_0",
    "/ln_1/Div_output_0",
    "/ln_1/Mul_output_0",
    # tanh-GELU decomposition.
    "/mlp/act/Mul_output_0",
    "/mlp/act/Add_output_0",
    "/mlp/act/Tanh_output_0",
    "/mlp/act/Add_1_output_0",
    # Attention scores: scaled Q/K, QK^T, masked scores.
    "/attn/Mul_output_0",
    "/attn/Mul_1_output_0",
    "/attn/MatMul_output_0",
    "/attn/Add_output_0",
]


def wanted(name: str) -> bool:
    if name in ("/transformer/Add_output_0", "/transformer/ln_f/Add_1_output_0", "logits"):
        return True
    if name.startswith("present."):
        return True
    if not name.startswith("/transformer/h."):
        return False
    rest = name[len("/transformer/h."):]
    layer, _, tail = rest.partition("/")
    tail = "/" + tail
    if not layer.isdigit():
        return False
    if tail in EVERY_LAYER:
        return True
    return layer == "0" and tail in LAYER_ZERO


def selected_names() -> list[str]:
    names = ["/transformer/Add_output_0", "/transformer/ln_f/Add_1_output_0", "logits"]
    for layer in range(LAYERS):
        for suffix in EVERY_LAYER:
            names.append(f"/transformer/h.{layer}{suffix}")
        body = f"present.{layer}"
        names.extend([f"{body}.key", f"{body}.value"])
    for suffix in LAYER_ZERO:
        names.append(f"/transformer/h.0{suffix}")
    return sorted(set(names))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def augmented_graph(model_path: Path) -> bytes:
    model = onnx.load(str(model_path), load_external_data=False)
    existing = {output.name for output in model.graph.output}
    known = {value.name: value for value in model.graph.value_info}
    added = []
    for name in selected_names():
        if name in existing:
            continue
        value = known.get(name)
        if value is None:
            value = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        else:
            value = onnx.ValueInfoProto()
            value.CopyFrom(known[name])
        model.graph.output.append(value)
        added.append(name)
    print(f"promoted {len(added)} intermediates to graph outputs")
    return model.SerializeToString()


def session(model_bytes: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )


def empty_past():
    return [
        (
            np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
            np.zeros((1, KV_HEADS, 0, HEAD_DIM), dtype=np.float32),
        )
        for _ in range(LAYERS)
    ]


def run(sess: ort.InferenceSession, tokens: list[int], position_ids: list[int], past):
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
    names = [output.name for output in sess.get_outputs()]
    values = sess.run(names, feeds)
    outputs = dict(zip(names, values))
    logits = outputs["logits"]
    present = [
        (outputs[f"present.{layer}.key"], outputs[f"present.{layer}.value"])
        for layer in range(LAYERS)
    ]
    return logits, present, outputs


def save_phase(out: Path, label: str, outputs: dict, names: list[str], keep: list[str]):
    phase = out / label
    phase.mkdir(parents=True, exist_ok=True)
    index = []
    for name in names:
        value = np.asarray(outputs[name])
        if value.dtype == np.float32 and name in keep:
            relative = name.replace("/", "_").replace(".", "_") + ".npy"
            np.save(phase / relative, value)
            index.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "file": relative,
                    "elements": int(value.size),
                }
            )
    (phase / "index.json").write_text(
        json.dumps({"phase": label, "tensors": index}, indent=2) + "\n"
    )
    print(f"{label}: wrote {len(index)} tensors to {phase}")


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual = actual.astype(np.float64).reshape(-1)
    reference = reference.astype(np.float64).reshape(-1)
    delta = actual - reference
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "mean_signed": float(np.mean(delta)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("fixtures/parity/trained-gpt2"),
        help="committed suite used to prove the logits are still the reference bytes",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    size = args.model.stat().st_size
    digest = sha256_file(args.model)
    if size != MODEL_BYTES or digest != MODEL_SHA256:
        raise SystemExit(
            f"not the pinned trained graph: {size} bytes, sha256 {digest}"
        )

    sess = session(augmented_graph(args.model))
    keep = set(selected_names())
    report = {
        "model": str(args.model),
        "model_bytes": size,
        "model_sha256": digest,
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "numpy": np.__version__,
        "provider": sess.get_providers(),
        "graph_optimization": "ORT_DISABLE_ALL",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "prompt": PROMPT,
        "decode": DECODE,
        "phases": {},
    }

    logits, present, outputs = run(sess, PROMPT, list(range(len(PROMPT))), empty_past())
    save_phase(args.out, "prefill", outputs, [o.name for o in sess.get_outputs()], keep)
    phase_report = {
        "shape": list(logits.shape),
        "argmax": int(np.argmax(logits[0, -1])),
        "top5": [int(i) for i in np.argsort(-logits[0, -1])[:5]],
        "logits_sha256": hashlib.sha256(
            logits[:, -1, :].astype("<f4").tobytes()
        ).hexdigest(),
    }
    fixture = args.fixtures / "tensors/prefill.p13.last_logits.f32"
    if fixture.is_file():
        committed = np.frombuffer(fixture.read_bytes(), dtype="<f4").reshape(1, 1, -1)
        phase_report["committed_fixture"] = str(fixture)
        phase_report["committed_metrics"] = metrics(logits[:, -1, :], committed)
    report["phases"]["prefill"] = phase_report

    past = present
    for index, token in enumerate(DECODE, start=1):
        total = len(PROMPT) + index
        logits, present, outputs = run(sess, [token], [total - 1], past)
        label = f"decode{index}"
        save_phase(args.out, label, outputs, [o.name for o in sess.get_outputs()], keep)
        phase_report = {
            "token": token,
            "past": total - 1,
            "shape": list(logits.shape),
            "argmax": int(np.argmax(logits[0, 0])),
            "top5": [int(i) for i in np.argsort(-logits[0, 0])[:5]],
            "cache_extent": int(present[0][0].shape[2]),
            "logits_sha256": hashlib.sha256(
                logits[:, 0, :].astype("<f4").tobytes()
            ).hexdigest(),
        }
        fixture = args.fixtures / f"tensors/decode.step{index}.logits.f32"
        if fixture.is_file():
            committed = np.frombuffer(fixture.read_bytes(), dtype="<f4").reshape(1, 1, -1)
            phase_report["committed_fixture"] = str(fixture)
            phase_report["committed_metrics"] = metrics(logits[:, 0, :], committed)
        report["phases"][label] = phase_report
        past = present

    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["phases"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
