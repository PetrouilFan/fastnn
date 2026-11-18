#!/usr/bin/env python3
"""Measure CPU prefill/decode-shaped executions of the tiny random GPT-2 graph.

FastNN 2.6.0 at a2c0759 cannot compile the stock dynamic-shape artifact (see
``tiny_gpt2_static.py``), so this benchmark builds static-shape variants for
each ``(tokens, past)`` workload. Timings are therefore whole-graph CPU AOT
execution on the real 257-node GPT-2 graph, but not evidence for the dynamic
stateful path.

Example::

    .venv/bin/python tools/baseline/bench_tiny_gpt2.py \
        --source-fnn /tmp/opencode/fn3-01/tiny-random-gpt2.fnn \
        --onnx /tmp/opencode/fn3-01/tiny-random-gpt2.onnx \
        --json docs/reports/fn3-01/raw/cpu-prefill-decode.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_gpt2_static import make_variant  # noqa: E402

import fastnn  # noqa: E402


def tensor(value: np.ndarray):
    return fastnn.tensor(value, list(value.shape))


def make_inputs(tokens: int, past: int, seed: int = 20261118):
    rng = np.random.default_rng(seed)
    ids = np.arange(11, 11 + tokens, dtype=np.int64)[None, :]
    mask = np.ones((1, tokens + past), dtype=np.int64)
    positions = np.arange(past, past + tokens, dtype=np.int64)[None, :]
    inputs = {
        "input_ids": tensor(ids),
        "attention_mask": tensor(mask),
        "position_ids": tensor(positions),
    }
    for layer in range(2):
        for kind in ("key", "value"):
            value = (rng.standard_normal((1, 2, past, 64), dtype=np.float32) * 0.1)
            inputs[f"past_key_values.{layer}.{kind}"] = tensor(value)
    return inputs, ids, mask, positions


def measure(model, inputs, warmup: int, reps: int) -> dict:
    for _ in range(warmup):
        model.forward(inputs)
    samples_ms = []
    for _ in range(reps):
        start = time.perf_counter_ns()
        model.forward(inputs)
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)
    ordered = sorted(samples_ms)

    model.enable_runtime_telemetry(True)
    model.forward(inputs)
    telemetry = model.runtime_telemetry() or {}
    model.enable_runtime_telemetry(False)

    return {
        "samples_ms": [round(value, 6) for value in samples_ms],
        "min_ms": round(ordered[0], 6),
        "p50_ms": round(float(statistics.median(ordered)), 6),
        "p95_ms": round(float(np.percentile(ordered, 95)), 6),
        "max_ms": round(ordered[-1], 6),
        "mean_ms": round(float(statistics.fmean(ordered)), 6),
        "telemetry": {
            key: int(telemetry.get(key, 0))
            for key in (
                "arena_bytes",
                "rss_bytes",
                "instruction_count",
                "write_const_count",
                "dispatch_ns",
                "total_ns",
            )
        },
    }


def parity(model, inputs, onnx_path: Path, ids, mask, positions) -> dict:
    import onnxruntime as ort

    actual = model.forward(inputs)
    feeds = {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": positions,
    }
    for name, value in inputs.items():
        if name.startswith("past_key_values"):
            feeds[name] = value.numpy()
    reference = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    logits = actual["logits"].numpy()
    logits_error = float(np.max(np.abs(logits - reference[0])))
    cache_errors = {}
    cache_nan = {}
    for index, name in enumerate(
        ["present.0.key", "present.0.value", "present.1.key", "present.1.value"]
    ):
        difference = np.abs(actual[name].numpy() - reference[index + 1])
        cache_nan[name] = int(np.isnan(difference).sum())
        cache_errors[name] = (
            float(np.max(difference)) if np.isfinite(difference).all() else None
        )
    finite = np.isfinite(logits)
    produced_nan = bool(np.isnan(logits).any() or any(cache_nan.values()))
    return {
        "logits_max_abs": float(logits_error) if np.isfinite(logits_error) else None,
        "logits_nan_count": int(np.isnan(logits).sum()),
        "logits_element_count": int(logits.size),
        "logits_argmax_last": int(np.argmax(logits[0, -1])) if finite[0, -1].all() else None,
        "ort_argmax_last": int(np.argmax(reference[0][0, -1])),
        "cache_max_abs": cache_errors,
        "cache_nan_count": cache_nan,
        "note": (
            "FastNN produced non-finite values; errors are null"
            if produced_nan
            else "logits and cache match ONNX Runtime within float32 noise"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fnn", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/opencode/fn3-01/variants"))
    parser.add_argument("--prefill-tokens", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--decode-pasts", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--prefill-warmup", type=int, default=3)
    parser.add_argument("--prefill-reps", type=int, default=20)
    parser.add_argument("--decode-warmup", type=int, default=3)
    parser.add_argument("--decode-reps", type=int, default=30)
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "kind": "static-shape tiny random GPT-2 CPU AOT",
        "warning": (
            "Dynamic-shape build fails at a2c0759; each workload uses a static-shape "
            "variant of the same 257-node graph and plain forward(), not stateful decode."
        ),
        "environment": {
            "python": platform.python_version(),
            "fastnn": getattr(fastnn, "__version__", "unknown"),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "prefill": [],
        "decode": [],
    }

    for tokens in args.prefill_tokens:
        variant = args.work_dir / f"prefill-p{tokens}-past1.fnn"
        info = make_variant(args.source_fnn, variant, tokens, past=1)
        model = fastnn.build_model_from_fnn(str(variant))
        inputs, ids, mask, positions = make_inputs(tokens, past=1)
        timing = measure(model, inputs, args.prefill_warmup, args.prefill_reps)
        record = {
            "workload": f"prefill {tokens} tokens + 1 cached",
            "tokens": tokens,
            "past": 1,
            "warmup": args.prefill_warmup,
            "reps": args.prefill_reps,
            "variant_changes": len(info["rewritten_reshapes"]),
            **timing,
        }
        if args.parity and args.onnx:
            record["parity"] = parity(model, inputs, args.onnx, ids, mask, positions)
        results["prefill"].append(record)
        print(json.dumps(record))

    for past in args.decode_pasts:
        variant = args.work_dir / f"decode-p1-past{past}.fnn"
        info = make_variant(args.source_fnn, variant, tokens=1, past=past)
        model = fastnn.build_model_from_fnn(str(variant))
        inputs, ids, mask, positions = make_inputs(tokens=1, past=past)
        timing = measure(model, inputs, args.decode_warmup, args.decode_reps)
        record = {
            "workload": f"decode 1 token after {past} cached",
            "tokens": 1,
            "past": past,
            "warmup": args.decode_warmup,
            "reps": args.decode_reps,
            "variant_changes": len(info["rewritten_reshapes"]),
            **timing,
        }
        if args.parity and args.onnx:
            record["parity"] = parity(model, inputs, args.onnx, ids, mask, positions)
        results["decode"].append(record)
        print(json.dumps(record))

    results["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    payload = json.dumps(results, indent=2, sort_keys=True, allow_nan=False)
    print(payload)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
