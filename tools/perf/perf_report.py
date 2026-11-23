#!/usr/bin/env python3
"""Measure one model and write a machine-readable performance report.

NOTE: this script still drives the archived 2.x Python facade
(`fastnn.build_model_from_fnn`, `fastnn.tensor_from_numpy`), which tag
`archive/3.0` removed. It is kept as the schema and discipline for the v3
harness (issue #106) and will be ported to the new `prepare`/`load`/`run`
API before it is used again.

NOTE: this script still drives the archived 2.x Python facade
(`fastnn.build_model_from_fnn`, `fastnn.tensor_from_numpy`), which tag
`archive/3.0` removed. It is kept as the schema and discipline for the v3
harness (issue #106) and will be ported to the new `prepare`/`load`/`run`
API before it is used again.

Unit tests answer "is this correct"; this answers "how fast, on which machine,
with how many threads, and does the output still match what it produced
before". A single number from a single run is not evidence, so the report
records the median and the 95th percentile of a series, the resident set before
and after, the model's own identity (size and SHA-256), the machine and the
commit, and a digest of every output tensor.

The schema is stable (`fastnn-perf/1`) so `compare_perf.py` can diff two
reports and a CI lane can refuse a regression.

Example:

    python3 tools/bench/perf_report.py \
        --model evit=/path/model.fnn \
        --input pixel_values=1,3,336,448 \
        --threads 8 --warmup 5 --repeat 30 \
        --out docs/reports/perf-evit.json

A report is only comparable to another made on the same machine with the same
thread count, which is why both are recorded and compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "fastnn-perf/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_mb() -> float:
    # ru_maxrss is a high-water mark in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_shape(text: str) -> list[int]:
    dims = [int(part) for part in text.split(",") if part.strip()]
    if not dims or any(dim <= 0 for dim in dims):
        raise SystemExit(f"invalid shape {text!r}; expected e.g. 1,3,336,448")
    return dims


def parse_model(spec: str) -> tuple[str, Path]:
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"invalid --model {spec!r}; expected NAME=PATH")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"model {name!r} does not exist: {resolved}")
    return name, resolved


def load_model(path: Path):
    import fastnn

    if path.suffix == ".fnn":
        return fastnn.build_model_from_fnn(str(path))
    # Anything else goes through the ONNX importer, which writes an .fnn
    # beside the source and then runs that.
    converted = path.with_suffix(".fnn")
    fastnn.convert_from_onnx(str(path), str(converted))
    return fastnn.build_model_from_fnn(str(converted))


def tensor_for(shape: list[int], seed: int):
    import numpy as np

    import fastnn

    rng = np.random.default_rng(seed)
    values = rng.standard_normal(shape, dtype=np.float32)
    return fastnn.tensor_from_numpy(values), values


def digest_outputs(result: dict) -> dict:
    import numpy as np

    outputs = {}
    for name, tensor in result.items():
        array = np.asarray(tensor.numpy(), dtype=np.float32)
        outputs[name] = {
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "sum": float(array.sum(dtype=np.float64)),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return outputs


def arena_bytes(executor) -> int | None:
    """Best-effort arena size from `memory_stats`, which is a text report."""
    try:
        stats = executor.memory_stats()
    except Exception:  # noqa: BLE001 - a missing feature must not fail a run
        return None
    if not isinstance(stats, dict):
        return None
    for key, value in stats.items():
        if "arena" in key.lower() and isinstance(value, (int, float)):
            return int(value)
    return None


def measure(model, inputs: dict, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        model.forward(inputs)
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        model.forward(inputs)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def compare_outputs(reference: dict, measured: dict) -> dict:
    """Compare two output digests and report agreement, not a tolerance guess."""
    differences = {}
    for name, entry in measured.items():
        previous = reference.get("outputs", {}).get(name)
        if previous is None:
            differences[name] = {"status": "absent-in-reference"}
            continue
        same_hash = previous.get("sha256") == entry["sha256"]
        differences[name] = {
            "hash_match": same_hash,
            "sum_delta": abs(entry["sum"] - previous.get("sum", 0.0)),
        }
    return differences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--input",
        required=True,
        metavar="NAME=SHAPE",
        help="e.g. pixel_values=1,3,336,448 (deterministic values from --seed)",
    )
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the default")
    parser.add_argument(
        "--dtype",
        default="f32",
        choices=("f32", "u8", "i8"),
        help="only f32 is reachable from the Python path today; see the report",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--out", metavar="FILE", help="write the report here")
    parser.add_argument(
        "--reference",
        metavar="FILE",
        help="earlier report to compare output digests against",
    )
    parser.add_argument(
        "--require-quiet",
        type=float,
        default=None,
        metavar="LOADAVG",
        help="refuse to measure when the 1-minute load average is above this",
    )
    args = parser.parse_args(argv)

    if args.dtype != "f32":
        # The Python facade has no quantized entry point; the Rust
        # `CompileTarget` route is not exposed. Saying so is better than
        # reporting a number for a path that did not run.
        raise SystemExit(
            f"--dtype {args.dtype} is not reachable from Python: fastnn exposes "
            "no quantizing builder. Report f32, or drive CompileTarget from Rust."
        )

    load = os.getloadavg()[0]
    if args.require_quiet is not None and load > args.require_quiet:
        raise SystemExit(
            f"machine is busy (1-minute load average {load:.2f} > "
            f"{args.require_quiet}); a measurement here would be noise"
        )

    import fastnn

    name, path = parse_model(args.model)
    input_name, _, shape_text = args.input.partition("=")
    shape = parse_shape(shape_text)

    if args.threads > 0:
        fastnn.set_num_threads(args.threads)

    rss_start = rss_mb()
    executor = load_model(path)
    tensor, _values = tensor_for(shape, args.seed)
    inputs = {input_name: tensor}
    rss_loaded = rss_mb()

    samples = measure(executor, inputs, args.warmup, args.repeat)
    rss_peak = rss_mb()
    outputs = digest_outputs(executor.forward(inputs))
    arena = arena_bytes(executor)

    report = {
        "schema": SCHEMA,
        "commit": commit(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpus": os.cpu_count(),
            "load_average_1m": round(load, 3),
            "python": platform.python_version(),
        },
        "threads": fastnn._core._get_num_threads(),
        "dtype": args.dtype,
        "model": {
            "name": name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "input": {"name": input_name, "shape": shape, "seed": args.seed},
        "warmup": args.warmup,
        "repeat": args.repeat,
        "timing_ms": {
            "min": round(min(samples), 3),
            "median": round(statistics.median(samples), 3),
            "p95": round(sorted(samples)[max(0, int(0.95 * len(samples)) - 1)], 3),
            "max": round(max(samples), 3),
        },
        "memory_mb": {
            "before_load": round(rss_start, 1),
            "after_load": round(rss_loaded, 1),
            "peak": round(rss_peak, 1),
        },
        "arena_bytes": arena,
        "outputs": outputs,
    }

    if args.reference:
        reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        if reference.get("model", {}).get("sha256") != report["model"]["sha256"]:
            raise SystemExit(
                "the reference report was made with a different model file; "
                "output digests would not be comparable"
            )
        report["accuracy"] = compare_outputs(reference, report)

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        print(f"wrote {destination}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    timing = report["timing_ms"]
    print(
        f"{name}: median {timing['median']:.2f} ms, p95 {timing['p95']:.2f} ms, "
        f"peak {report['memory_mb']['peak']:.1f} MB, {report['threads']} threads",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
