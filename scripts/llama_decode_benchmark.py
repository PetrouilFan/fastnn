#!/usr/bin/env python3
"""Parity-gated recurrent decode benchmark for Llama-family ONNX/FNN models.

The controller runs ONNX Runtime and fastnn in separate subprocesses so their
multi-gigabyte model allocations are not resident at the same time. ORT writes
reference logits and every present K/V tensor to a temporary NPZ file; fastnn
must match those outputs before its timings are accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

_STATE_RE = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    median = statistics.median(samples_ms)
    return {
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": median,
        "p95_ms": float(np.percentile(np.asarray(samples_ms), 95)),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "tokens_per_second_from_median": 1000.0 / median,
    }


def _rss_mib() -> float:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def _max_rss_mib() -> float:
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _model_contract(onnx_path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    input_names = {value.name for value in inputs}
    states: list[tuple[str, str]] = []
    state_shapes: dict[str, tuple[int, ...]] = {}
    for value in inputs:
        match = _STATE_RE.match(value.name)
        if match is None:
            continue
        layer, kind = int(match.group(1)), match.group(2)
        output_name = f"present.{layer}.{kind}"
        states.append((value.name, output_name))
        dims = value.type.tensor_type.shape.dim
        numeric = [int(dim.dim_value) if dim.dim_value else 0 for dim in dims]
        if len(numeric) != 4 or numeric[1] <= 0 or numeric[3] <= 0:
            raise ValueError(f"cannot infer recurrent state geometry for {value.name}: {numeric}")
        state_shapes[value.name] = (1, numeric[1], 0, numeric[3])
    states.sort(key=lambda pair: (int(_STATE_RE.match(pair[0]).group(1)), pair[0]))
    if not states:
        raise ValueError("model has no past_key_values.* recurrent inputs")
    output_names = [value.name for value in model.graph.output]
    missing = [output for _, output in states if output not in output_names]
    if missing:
        raise ValueError(f"missing recurrent outputs: {missing[:4]}")
    return {
        "input_names": sorted(input_names),
        "state_pairs": states,
        "state_shapes": state_shapes,
        "output_names": output_names,
        "layers": len(states) // 2,
    }


def _token_inputs(contract: dict[str, Any], step: int) -> dict[str, np.ndarray]:
    values = {"input_ids": np.asarray([[1 + (step % 31)]], dtype=np.int64)}
    if "attention_mask" in contract["input_names"]:
        values["attention_mask"] = np.ones((1, step + 1), dtype=np.int64)
    if "position_ids" in contract["input_names"]:
        values["position_ids"] = np.asarray([[step]], dtype=np.int64)
    return values


def _empty_state(contract: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(contract["state_shapes"][name], dtype=np.float32)
        for name, _ in contract["state_pairs"]
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 and right.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _compact_memory_stats(stats: dict[str, Any]) -> dict[str, Any]:
    scalar_keys = (
        "arena_size",
        "memory_plan_total_size",
        "logical_slot_bytes",
        "physical_slot_bytes",
        "slot_reuse_saved_bytes",
        "alias_groups",
        "aliased_nodes",
        "instructions",
        "call_kernel_count",
        "memcpy_count",
        "fill_count",
        "write_const_count",
        "write_const_bytes",
        "estimated_static_traffic_bytes",
    )
    compact = {key: stats[key] for key in scalar_keys if key in stats}
    compact["top_kernels_by_count"] = stats.get("top_kernels_by_count", [])[:20]
    compact["top_write_consts_by_size"] = stats.get("top_write_consts_by_size", [])[:10]
    return compact


def _run_ort_worker(args: argparse.Namespace) -> dict[str, Any]:
    import onnxruntime as ort

    contract = _model_contract(args.onnx)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - started
    rss_after_load = _rss_mib()
    cases = []
    for context in args.contexts:
        def replay(timed: bool) -> tuple[dict[str, np.ndarray], float | None]:
            state = _empty_state(contract)
            output = None
            elapsed = None
            for step in range(context + 1):
                inputs = {**_token_inputs(contract, step), **state}
                start = time.perf_counter_ns() if timed and step == context else None
                values = session.run(None, inputs)
                if start is not None:
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                output = dict(zip(contract["output_names"], values, strict=True))
                state = {
                    input_name: output[output_name]
                    for input_name, output_name in contract["state_pairs"]
                }
            assert output is not None
            return output, elapsed

        reference, _ = replay(False)
        np.savez(args.work_dir / f"ort-context-{context}.npz", **reference)
        for _ in range(args.warmup):
            replay(False)
        samples = []
        for _ in range(args.iterations):
            _, elapsed = replay(True)
            assert elapsed is not None
            samples.append(elapsed)
        cases.append({"context_tokens": context, "latency": _latency_summary(samples)})
    return {
        "runtime": "onnxruntime",
        "load_seconds": load_seconds,
        "rss_after_load_mib": rss_after_load,
        "peak_rss_mib": _max_rss_mib(),
        "cases": cases,
        "layers": contract["layers"],
    }


def _run_fastnn_worker(args: argparse.Namespace) -> dict[str, Any]:
    import fastnn as fnn

    contract = _model_contract(args.onnx)
    bound = max(args.contexts) + 2
    started = time.perf_counter()
    model = fnn.build_model_from_fnn(
        str(args.fnn),
        symbolic_dim_bounds={
            "batch_size": 1,
            "sequence_length": max(1, bound),
            "past_sequence_length": bound,
        },
    )
    load_seconds = time.perf_counter() - started
    rss_after_load = _rss_mib()
    initial_numpy = _empty_state(contract)
    initial = {
        name: fnn.tensor(value, list(value.shape)) for name, value in initial_numpy.items()
    }
    bindings = dict(contract["state_pairs"])
    sequence_axes = {}
    for name, shape in contract["state_shapes"].items():
        empty_axes = [axis for axis, extent in enumerate(shape) if extent == 0]
        if len(empty_axes) != 1:
            raise ValueError(
                f"state {name} must have exactly one empty sequence axis, got shape {shape}"
            )
        sequence_axes[name] = empty_axes[0]
    model.configure_state(bindings, initial, None, sequence_axes)
    session = model.create_session()

    def tensors(values: dict[str, np.ndarray]) -> dict[str, Any]:
        return {name: fnn.tensor(value, list(value.shape)) for name, value in values.items()}

    cases = []
    for context in args.contexts:
        def replay(
            target_session: Any, timed: bool
        ) -> tuple[dict[str, Any], float | None]:
            target_session.reset_state()
            output = None
            elapsed = None
            for step in range(context + 1):
                inputs = tensors(_token_inputs(contract, step))
                start = time.perf_counter_ns() if timed and step == context else None
                output = (
                    target_session.prefill(inputs, True)
                    if step == 0
                    else target_session.decode(inputs, True)
                )
                if start is not None:
                    elapsed = (time.perf_counter_ns() - start) / 1e6
            assert output is not None
            return output, elapsed

        actual, _ = replay(session, False)
        actual_numpy = {name: tensor.numpy() for name, tensor in actual.items()}
        isolated_session = model.create_session()
        isolated, _ = replay(isolated_session, False)
        isolated_numpy = {name: tensor.numpy() for name, tensor in isolated.items()}
        isolation_error = max(
            _max_abs(actual_numpy[name], isolated_numpy[name]) for name in actual_numpy
        )
        if isolation_error != 0.0:
            raise AssertionError(
                f"context={context} independent session replay diverged: {isolation_error}"
            )
        np.savez(args.work_dir / f"fastnn-context-{context}.npz", **actual_numpy)
        with np.load(args.work_dir / f"ort-context-{context}.npz") as expected:
            logits_error = _max_abs(actual_numpy["logits"], expected["logits"])
            cache_error = max(
                _max_abs(actual_numpy[output_name], expected[output_name])
                for _, output_name in contract["state_pairs"]
            )
            fastnn_argmax = int(np.argmax(actual_numpy["logits"][0, -1]))
            ort_argmax = int(np.argmax(expected["logits"][0, -1]))
        if (
            logits_error > args.logits_atol
            or cache_error > args.cache_atol
            or fastnn_argmax != ort_argmax
        ):
            raise AssertionError(
                f"context={context} parity failed: logits={logits_error}, "
                f"cache={cache_error}, argmax={fastnn_argmax}/{ort_argmax}"
            )
        for _ in range(args.warmup):
            replay(session, False)
        samples = []
        for _ in range(args.iterations):
            _, elapsed = replay(session, True)
            assert elapsed is not None
            samples.append(elapsed)
        cases.append(
            {
                "context_tokens": context,
                "latency": _latency_summary(samples),
                "parity": {
                    "logits_max_abs": logits_error,
                    "cache_max_abs": cache_error,
                    "argmax": [fastnn_argmax, ort_argmax],
                    "session_isolation_max_abs": isolation_error,
                },
            }
        )
    return {
        "runtime": "fastnn",
        "load_seconds": load_seconds,
        "rss_after_load_mib": rss_after_load,
        "peak_rss_mib": _max_rss_mib(),
        "cases": cases,
        "layers": contract["layers"],
        "prepared_stats": dict(model.prepared_stats()),
        "memory_stats": _compact_memory_stats(dict(model.memory_stats())),
    }


def _run_worker(args: argparse.Namespace) -> None:
    result = _run_ort_worker(args) if args.worker == "ort" else _run_fastnn_worker(args)
    args.worker_output.write_text(json.dumps(result, indent=2))


def _controller(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fastnn-llama-benchmark-") as temporary:
        work_dir = Path(temporary)
        results = {}
        for worker in ("ort", "fastnn"):
            output = work_dir / f"{worker}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker", worker,
                "--worker-output", str(output),
                "--work-dir", str(work_dir),
                "--fnn", str(args.fnn),
                "--onnx", str(args.onnx),
                "--contexts", *map(str, args.contexts),
                "--warmup", str(args.warmup),
                "--iterations", str(args.iterations),
                "--threads", str(args.threads),
                "--logits-atol", str(args.logits_atol),
                "--cache-atol", str(args.cache_atol),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "RAYON_NUM_THREADS": str(args.threads),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            subprocess.run(command, check=True, env=environment)
            results[worker] = json.loads(output.read_text())
    fastnn_cases = {case["context_tokens"]: case for case in results["fastnn"]["cases"]}
    ort_cases = {case["context_tokens"]: case for case in results["ort"]["cases"]}
    return {
        "model": {"fnn": str(args.fnn), "onnx": str(args.onnx)},
        "threads": args.threads,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "fastnn": {key: value for key, value in results["fastnn"].items() if key != "cases"},
        "onnxruntime": {key: value for key, value in results["ort"].items() if key != "cases"},
        "cases": [
            {
                "context_tokens": context,
                "parity": fastnn_cases[context]["parity"],
                "fastnn": fastnn_cases[context]["latency"],
                "onnxruntime": ort_cases[context]["latency"],
            }
            for context in args.contexts
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnn", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--contexts", type=int, nargs="+", default=[0, 8, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--logits-atol", type=float, default=1e-3)
    parser.add_argument("--cache-atol", type=float, default=1e-4)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--worker", choices=["ort", "fastnn"])
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    if args.worker and (args.worker_output is None or args.work_dir is None):
        parser.error("worker mode requires --worker-output and --work-dir")
    return args


def main() -> None:
    args = _parse_args()
    if args.worker:
        _run_worker(args)
        return
    result = _controller(args)
    text = json.dumps(result, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
