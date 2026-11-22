#!/usr/bin/env python3
"""Write an ONNX Runtime reference dump for the v3 parity test.

The Rust test (`tests/fn3_cnn_run.rs`) must not import a Python runtime: the
comparison input is generated here, ORT runs here, and the test reads the three
files this script writes. The dump is therefore reproducible from the model file
alone.

Usage:

    python tools/parity/ort_reference.py MODEL.onnx OUTPUT_DIR
    python tools/parity/ort_reference.py MODEL.onnx --time --threads 8

Files written into OUTPUT_DIR:

    input.bin   every graph input's values, f32 little-endian, in graph order
    output.bin  every graph output's values, f32 little-endian, in graph order
    meta.json   model sha256, ORT version, and each input/output's name, shape
                and dtype, in the same order as the .bin files

The input is `numpy.random.default_rng(0).standard_normal` cast to float32, per
input tensor in graph order, so two runs over the same model and the same
ONNX Runtime build produce byte-identical files. Inputs whose dtype is not
float32 are rejected: the v3 CNN path computes in f32 and the test asserts
against f32 operands.

With `--time` the script measures instead of dumping: it opens the model with
`SessionOptions.intra_op_num_threads = --threads`, warms `--warmup` forwards,
times `--iterations` more, and prints one JSON line with the engine, the thread
count, median/p95 milliseconds and the process's peak RSS. One process per
configuration is the point: `ru_maxrss` is then that configuration's number.

Exit code 0 means the dump was written (or the measurement ran); anything else
is an error with a message on stderr. ONNX Runtime's own logging is silenced to
WARNING so a CI log stays readable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info) -> list[int]:
    dims = value_info.type.tensor_type.shape.dim
    shape: list[int] = []
    for dim in dims:
        if not dim.HasField("dim_value"):
            raise SystemExit(
                f"{value_info.name}: symbolic extents are not supported by this dump "
                "(the v3 CNN path requires static shapes)"
            )
        shape.append(int(dim.dim_value))
    return shape


def dtype_of(value_info) -> str:
    return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type).lower()


def describe(values: list) -> list[dict]:
    return [
        {
            "name": value_info.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        for value_info, array in values
    ]


def machine() -> str:
    model = platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return model


def peak_rss_mib() -> float:
    # Linux reports KiB; macOS reports bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0 * 1024.0)


def time_mode(model: Path, threads: int, warmup: int, iterations: int) -> int:
    graph = onnx.load(str(model)).graph
    rng = np.random.default_rng(0)
    feeds: dict[str, np.ndarray] = {}
    for value_info in graph.input:
        if any(initializer.name == value_info.name for initializer in graph.initializer):
            continue
        feeds[value_info.name] = rng.standard_normal(
            size=tensor_shape(value_info), dtype=np.float64
        ).astype(np.float32)
    if not feeds:
        raise SystemExit("the model declares no feedable inputs")

    options = ort.SessionOptions()
    options.log_severity_level = 2  # WARNING
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    for _ in range(warmup):
        session.run(output_names, feeds)
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(output_names, feeds)
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[max(0, min(len(samples) - 1, int(np.ceil(0.95 * len(samples))) - 1))]
    print(
        json.dumps(
            {
                "engine": "ort",
                "machine": machine(),
                "threads": threads,
                "median_ms": round(median, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(samples[0], 2),
                "max_ms": round(samples[-1], 2),
                "iterations": iterations,
                "peak_rss_mib": round(peak_rss_mib(), 1),
                "onnxruntime": ort.__version__,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="self-contained .onnx model")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="directory to write the dump into (omitted with --time)",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="measure instead of dumping: warm forwards, then median/p95 latency",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="SessionOptions.intra_op_num_threads for --time (default 8)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="warm-up forwards (default 3)")
    parser.add_argument(
        "--iterations", type=int, default=30, help="timed forwards (default 30)"
    )
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2
    if args.time:
        return time_mode(args.model, args.threads, args.warmup, args.iterations)
    if args.output_dir is None:
        parser.error("OUTPUT_DIR is required unless --time is given")

    graph = onnx.load(str(args.model)).graph
    inputs: list[tuple[object, np.ndarray]] = []
    rng = np.random.default_rng(0)
    feeds: dict[str, np.ndarray] = {}
    for value_info in graph.input:
        # An initializer is also listed in `graph.input` in older exporters; skip
        # it, because it is a weight, not a feed.
        if any(initializer.name == value_info.name for initializer in graph.initializer):
            continue
        if dtype_of(value_info) != "float":
            raise SystemExit(
                f"input {value_info.name} is {dtype_of(value_info)}, not float32; this "
                "dump only generates f32 inputs"
            )
        shape = tensor_shape(value_info)
        values = rng.standard_normal(size=shape, dtype=np.float64).astype(np.float32)
        feeds[value_info.name] = values
        inputs.append((value_info, values))
    if not feeds:
        raise SystemExit("the model declares no feedable inputs")

    options = ort.SessionOptions()
    options.log_severity_level = 2  # WARNING
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]
    results = session.run(output_names, feeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "input.bin").open("wb") as handle:
        for _, values in inputs:
            handle.write(values.astype("<f4").tobytes(order="C"))
    with (args.output_dir / "output.bin").open("wb") as handle:
        for result in results:
            if result.dtype != np.float32:
                raise SystemExit(
                    f"an output is {result.dtype}, not float32; the parity test "
                    "compares f32 outputs"
                )
            handle.write(result.astype("<f4").tobytes(order="C"))

    meta = {
        "model": str(args.model),
        "model_sha256": sha256_of(args.model),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "seed": 0,
        "inputs": describe(inputs),
        "outputs": [
            {
                "name": value_info.name,
                "shape": list(result.shape),
                "dtype": str(result.dtype),
            }
            for value_info, result in zip(session.get_outputs(), results)
        ],
        "layout": {
            "input.bin": "each input's row-major f32 values, in the order of `inputs`",
            "output.bin": "each output's row-major f32 values, in the order of `outputs`",
        },
    }
    with (args.output_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")

    total = sum(result.size for result in results)
    print(
        f"wrote {args.output_dir}/input.bin, output.bin ({total} f32 values) and "
        f"meta.json for {args.model.name} with onnxruntime {ort.__version__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
