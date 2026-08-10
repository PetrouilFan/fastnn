#!/usr/bin/env python3
"""Compare fastnn runtime-owned GPT-2 KV sessions with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

import fastnn as fnn


@dataclass
class StepResult:
    seconds: float
    logits_max_abs: float
    logits_mean_abs: float
    cache_max_abs: float
    argmax_fastnn: int
    argmax_ort: int
    state_bytes: int
    logits: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnn", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-context", type=int, default=8)
    parser.add_argument("--quantize", default=None)
    parser.add_argument("--w4a8-calibration", type=Path)
    parser.add_argument("--atol", type=float, default=5e-4)
    return parser.parse_args()


def tensor(value: np.ndarray):
    return fnn.tensor(value, list(value.shape))


def state_names() -> list[tuple[str, str]]:
    return [
        (f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}")
        for layer in range(12)
        for kind in ("key", "value")
    ]


def make_model(args: argparse.Namespace):
    clip_ratios = None
    if args.w4a8_calibration is not None:
        report = json.loads(args.w4a8_calibration.read_text())
        if report.get("group_size") != int(str(args.quantize).removeprefix("w4a8-g")):
            raise ValueError("W4A8 calibration group size does not match --quantize")
        clip_ratios = {
            layer["name"]: layer["clip_ratios"]
            for layer in report.get("layers", [])
        }
        if not clip_ratios:
            raise ValueError("W4A8 calibration has no projection clip ratios")
    model = fnn.build_model_from_fnn(
        str(args.fnn),
        symbolic_dim_bounds={"batch_size": 1, "past_sequence_length": args.max_context},
        quantize=args.quantize,
        w4a8_clip_ratios=clip_ratios,
    )
    bindings = dict(state_names())
    initial = {
        input_name: tensor(np.zeros((1, 12, 0, 64), dtype=np.float32))
        for input_name in bindings
    }
    model.configure_state(bindings, initial, None, {name: 2 for name in bindings})
    return model


def run_sequence(args: argparse.Namespace, session, ort_session) -> list[StepResult]:
    output_names = [output.name for output in ort_session.get_outputs()]
    ort_state = {
        input_name: np.zeros((1, 12, 0, 64), dtype=np.float32)
        for input_name, _ in state_names()
    }
    results = []
    for step, token_id in enumerate(args.tokens, start=1):
        feeds = {
            "input_ids": np.array([[token_id]], dtype=np.int64),
            "attention_mask": np.ones((1, step), dtype=np.int64),
            **ort_state,
        }
        reference = dict(zip(output_names, ort_session.run(None, feeds), strict=True))
        inputs = {name: tensor(value) for name, value in feeds.items() if name not in ort_state}
        start = time.perf_counter()
        outputs = (
            session.prefill(inputs, True)
            if step == 1
            else session.decode(inputs, True)
        )
        elapsed = time.perf_counter() - start

        actual_logits = outputs["logits"].numpy()
        reference_logits = reference["logits"]
        cache_error = max(
            float(np.max(np.abs(reference[output_name] - outputs[output_name].numpy())))
            for _, output_name in state_names()
        )
        sizes = set(session.state_sizes().values())
        if len(sizes) != 1:
            raise AssertionError(f"inconsistent KV state sizes: {sorted(sizes)}")
        results.append(
            StepResult(
                seconds=elapsed,
                logits_max_abs=float(np.max(np.abs(actual_logits - reference_logits))),
                logits_mean_abs=float(np.mean(np.abs(actual_logits - reference_logits))),
                cache_max_abs=cache_error,
                argmax_fastnn=int(actual_logits[0, -1].argmax()),
                argmax_ort=int(reference_logits[0, -1].argmax()),
                state_bytes=sizes.pop(),
                logits=actual_logits.copy(),
            )
        )
        ort_state = {
            input_name: reference[output_name]
            for input_name, output_name in state_names()
        }
    return results


def main() -> None:
    args = parse_args()
    model = make_model(args)
    session = model.create_session()
    isolated = model.create_session()
    ort_session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    first = run_sequence(args, session, ort_session)
    session.reset_state()
    replay = run_sequence(args, session, ort_session)
    independent = run_sequence(args, isolated, ort_session)

    failures = []
    for index, result in enumerate(first, start=1):
        print(
            f"step={index} seconds={result.seconds:.6f} "
            f"logits_max_abs={result.logits_max_abs:.9g} "
            f"logits_mean_abs={result.logits_mean_abs:.9g} "
            f"cache_max_abs={result.cache_max_abs:.9g} "
            f"argmax={result.argmax_fastnn}/{result.argmax_ort} "
            f"state_bytes={result.state_bytes}"
        )
        if result.logits_max_abs > args.atol:
            failures.append(f"step {index} logits error exceeds {args.atol}")
        if result.cache_max_abs > args.atol:
            failures.append(f"step {index} cache error exceeds {args.atol}")
        if result.argmax_fastnn != result.argmax_ort:
            failures.append(f"step {index} argmax mismatch")

    replay_error = max(float(np.max(np.abs(a.logits - b.logits))) for a, b in zip(first, replay, strict=True))
    isolation_error = max(float(np.max(np.abs(a.logits - b.logits))) for a, b in zip(first, independent, strict=True))
    print(f"reset_replay_max_abs={replay_error:.9g}")
    print(f"session_isolation_max_abs={isolation_error:.9g}")
    print(f"session_steps={session.session_steps}")
    if replay_error != 0.0 or isolation_error != 0.0:
        failures.append("session replay or isolation is not deterministic")
    if failures:
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
