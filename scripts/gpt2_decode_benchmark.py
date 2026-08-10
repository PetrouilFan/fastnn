#!/usr/bin/env python3
"""Benchmark stateful one-token GPT-2 decode in fastnn and ONNX Runtime.

Each sample replays a valid sequence from an empty cache to the requested
incoming context length. Only the target token is timed; setup, reset, and
earlier cache-building tokens are excluded. Logits and all present K/V tensors
are checked against ONNX Runtime before latency is reported.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

import fastnn as fnn

STATE_PAIRS = [
    (f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}")
    for layer in range(12)
    for kind in ("key", "value")
]


def as_tensor(value: np.ndarray):
    return fnn.tensor(value, list(value.shape))


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    median = statistics.median(samples_ms)
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": median,
        "p95_ms": float(np.percentile(np.asarray(samples_ms), 95)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "tokens_per_second_from_median": 1000.0 / median,
    }


def telemetry_summary(samples: list[dict[str, int]]) -> dict[str, dict[str, float]]:
    if not samples:
        return {}
    fields = sorted(set.intersection(*(set(sample) for sample in samples)))
    return {
        field: {
            "median": float(statistics.median(sample[field] for sample in samples)),
            "p95": float(np.percentile([sample[field] for sample in samples], 95)),
        }
        for field in fields
    }


def initial_state() -> dict[str, np.ndarray]:
    return {
        name: np.zeros((1, 12, 0, 64), dtype=np.float32)
        for name, _ in STATE_PAIRS
    }


def run_fastnn_to_context(
    session: Any,
    context: int,
    *,
    timed: bool,
) -> tuple[dict[str, Any], float | None, dict[str, int] | None]:
    session.reset_state()
    output: dict[str, Any] | None = None
    elapsed_ms: float | None = None
    telemetry: dict[str, int] | None = None
    for step in range(context + 1):
        inputs = {
            "input_ids": as_tensor(np.asarray([[42 + (step % 7)]], dtype=np.int64)),
            "attention_mask": as_tensor(np.ones((1, step + 1), dtype=np.int64)),
        }
        start = time.perf_counter_ns() if timed and step == context else None
        output = (
            session.prefill(inputs, True)
            if step == 0
            else session.decode(inputs, True)
        )
        if start is not None:
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            telemetry = session.runtime_telemetry()
    assert output is not None
    return output, elapsed_ms, telemetry


def run_ort_to_context(
    ort_session: ort.InferenceSession,
    context: int,
    *,
    timed: bool,
) -> tuple[dict[str, np.ndarray], float | None]:
    state = initial_state()
    names = [output.name for output in ort_session.get_outputs()]
    output: dict[str, np.ndarray] | None = None
    elapsed_ms: float | None = None
    for step in range(context + 1):
        inputs = {
            "input_ids": np.asarray([[42 + (step % 7)]], dtype=np.int64),
            "attention_mask": np.ones((1, step + 1), dtype=np.int64),
            **state,
        }
        start = time.perf_counter_ns() if timed and step == context else None
        values = ort_session.run(None, inputs)
        if start is not None:
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        output = dict(zip(names, values, strict=True))
        state = {input_name: output[output_name] for input_name, output_name in STATE_PAIRS}
    assert output is not None
    return output, elapsed_ms


def load_pytorch_model(onnx_path: Path, threads: int):
    import onnx
    import torch
    from onnx import numpy_helper
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    graph = onnx.load(str(onnx_path))
    config = GPT2Config(
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_cache=True,
    )
    model = GPT2LMHeadModel(config).eval()
    state = {
        initializer.name: torch.from_numpy(numpy_helper.to_array(initializer).copy())
        for initializer in graph.graph.initializer
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["lm_head.weight"] or unexpected:
        raise RuntimeError(
            f"unexpected PyTorch state mapping: missing={missing}, unexpected={unexpected}"
        )
    return model


def run_pytorch_to_context(model: Any, context: int, *, timed: bool):
    import torch

    cache = None
    output = None
    elapsed_ms: float | None = None
    with torch.inference_mode():
        for step in range(context + 1):
            input_ids = torch.tensor([[42 + (step % 7)]], dtype=torch.long)
            attention_mask = torch.ones((1, step + 1), dtype=torch.long)
            start = time.perf_counter_ns() if timed and step == context else None
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            if start is not None:
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            cache = output.past_key_values
    assert output is not None
    return output, elapsed_ms


def benchmark_case(
    *,
    fnn_path: Path,
    ort_session: ort.InferenceSession,
    pytorch_model: Any | None,
    context: int,
    warmup: int,
    iterations: int,
    logits_atol: float,
    cache_atol: float,
) -> dict[str, Any]:
    model = fnn.build_model_from_fnn(
        str(fnn_path),
        symbolic_dim_bounds={"batch_size": 1, "past_sequence_length": context + 2},
    )
    empty = initial_state()
    bindings = dict(STATE_PAIRS)
    model.configure_state(
        bindings,
        {name: as_tensor(value) for name, value in empty.items()},
        None,
        {name: 2 for name in bindings},
    )
    session = model.create_session()
    session.enable_runtime_telemetry(True)

    actual, _, _ = run_fastnn_to_context(session, context, timed=False)
    expected, _ = run_ort_to_context(ort_session, context, timed=False)
    logits_max_abs = float(np.max(np.abs(actual["logits"].numpy() - expected["logits"])))
    cache_max_abs = max(
        float(np.max(np.abs(actual[output].numpy() - expected[output])))
        for _, output in STATE_PAIRS
    )
    fastnn_argmax = int(np.argmax(actual["logits"].numpy()[0, -1]))
    ort_argmax = int(np.argmax(expected["logits"][0, -1]))
    if (
        logits_max_abs > logits_atol
        or cache_max_abs > cache_atol
        or fastnn_argmax != ort_argmax
    ):
        raise AssertionError(
            f"context={context}: parity failure: logits={logits_max_abs}, "
            f"cache={cache_max_abs}, argmax={fastnn_argmax}/{ort_argmax}"
        )

    pytorch_parity = None
    if pytorch_model is not None:
        pytorch_output, _ = run_pytorch_to_context(pytorch_model, context, timed=False)
        pytorch_logits = pytorch_output.logits.detach().cpu().numpy()
        pytorch_logits_max_abs = float(np.max(np.abs(pytorch_logits - expected["logits"])))
        pytorch_cache_max_abs = 0.0
        for layer_index, layer in enumerate(pytorch_output.past_key_values.layers):
            for kind, tensor in (("key", layer.keys), ("value", layer.values)):
                reference = expected[f"present.{layer_index}.{kind}"]
                error = float(np.max(np.abs(tensor.detach().cpu().numpy() - reference)))
                pytorch_cache_max_abs = max(pytorch_cache_max_abs, error)
        pytorch_argmax = int(np.argmax(pytorch_logits[0, -1]))
        if (
            pytorch_logits_max_abs > logits_atol
            or pytorch_cache_max_abs > cache_atol
            or pytorch_argmax != ort_argmax
        ):
            raise AssertionError(
                f"context={context}: PyTorch parity failure: logits={pytorch_logits_max_abs}, "
                f"cache={pytorch_cache_max_abs}, argmax={pytorch_argmax}/{ort_argmax}"
            )
        pytorch_parity = {
            "logits_max_abs": pytorch_logits_max_abs,
            "cache_max_abs": pytorch_cache_max_abs,
            "argmax": [pytorch_argmax, ort_argmax],
        }

    for _ in range(warmup):
        run_fastnn_to_context(session, context, timed=False)
        run_ort_to_context(ort_session, context, timed=False)
        if pytorch_model is not None:
            run_pytorch_to_context(pytorch_model, context, timed=False)

    fastnn_ms = []
    fastnn_telemetry = []
    ort_ms = []
    pytorch_ms = []
    for _ in range(iterations):
        _, elapsed, telemetry = run_fastnn_to_context(session, context, timed=True)
        assert elapsed is not None
        assert telemetry is not None
        fastnn_ms.append(elapsed)
        fastnn_telemetry.append(telemetry)
        _, elapsed = run_ort_to_context(ort_session, context, timed=True)
        assert elapsed is not None
        ort_ms.append(elapsed)
        if pytorch_model is not None:
            _, elapsed = run_pytorch_to_context(pytorch_model, context, timed=True)
            assert elapsed is not None
            pytorch_ms.append(elapsed)

    fastnn_summary = latency_summary(fastnn_ms)
    ort_summary = latency_summary(ort_ms)
    pytorch_summary = latency_summary(pytorch_ms) if pytorch_ms else None
    result = {
        "context_tokens": context,
        "parity": {
            "logits_max_abs": logits_max_abs,
            "cache_max_abs": cache_max_abs,
            "argmax": [fastnn_argmax, ort_argmax],
        },
        "fastnn": fastnn_summary,
        "fastnn_runtime_telemetry": telemetry_summary(fastnn_telemetry),
        "onnxruntime": ort_summary,
        "median_speedup_fastnn_over_ort": ort_summary["median_ms"] / fastnn_summary["median_ms"],
    }
    if pytorch_summary is not None:
        result["pytorch"] = pytorch_summary
        result["pytorch_parity"] = pytorch_parity
        result["median_speedup_fastnn_over_pytorch"] = (
            pytorch_summary["median_ms"] / fastnn_summary["median_ms"]
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnn", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[0, 8, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--logits-atol", type=float, default=1e-3)
    parser.add_argument("--cache-atol", type=float, default=5e-4)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.threads <= 0:
        raise SystemExit("warmup must be non-negative; iterations and threads must be positive")
    if any(context < 0 for context in args.contexts):
        raise SystemExit("contexts must be non-negative")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort_session = ort.InferenceSession(
        str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    pytorch_model = None if args.skip_pytorch else load_pytorch_model(args.onnx, args.threads)

    result = {
        "schema_version": 1,
        "model": {"fnn": str(args.fnn), "onnx": str(args.onnx)},
        "contract": {
            "workload": "stateful single-token decode after valid cache-building replay",
            "timed_outputs": "logits plus all 24 present K/V tensors",
            "setup_reset_and_cache_building_in_timing": False,
            "warmup_sequences": args.warmup,
            "measured_sequences": args.iterations,
            "threads": args.threads,
            "ort_execution_mode": "sequential",
            "parity_thresholds": {
                "logits_max_abs": args.logits_atol,
                "cache_max_abs": args.cache_atol,
                "argmax_match": True,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "fastnn": getattr(fnn, "__version__", "unknown"),
            "onnxruntime": ort.__version__,
            "pytorch": None if pytorch_model is None else __import__("torch").__version__,
            "numpy": np.__version__,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS"),
        },
        "cases": [
            benchmark_case(
                fnn_path=args.fnn,
                ort_session=ort_session,
                pytorch_model=pytorch_model,
                context=context,
                warmup=args.warmup,
                iterations=args.iterations,
                logits_atol=args.logits_atol,
                cache_atol=args.cache_atol,
            )
            for context in args.contexts
        ],
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
