#!/usr/bin/env python3
"""Profile one ONNX Runtime session per node and aggregate the durations.

This is the ONNX Runtime side of `docs/v3/per-layer.md`. It opens the model on
`CPUExecutionProvider`, runs a few warm-ups and then a few timed forwards with
`SessionOptions.enable_profiling`, and reports:

- the per-op-type total per forward (median across the timed runs), so the
  classes line up with fastnn's segment labels;
- the per-node duration per forward (median), so one convolution can be compared
  with the same segment in a `FN3_CNN_TRACE=1` fastnn run.

Usage:

    python tools/parity/ort_profile.py MODEL.onnx --threads 1
    python tools/parity/ort_profile.py MODEL.onnx --threads 8 --runs 5 \
        --shape images=1,3,640,640 --json-out /tmp/ort-8t.json

`--shape NAME=D1,D2,...` is required only when the model declares a symbolic
input: ORT cannot allocate a feed with a symbolic extent.

The timing is wall-clock per node as ORT records it. A node's duration includes
the kernel work ORT attributes to it; it does not include the Python call or
data transfer. The script writes no model or fixture, so it is safe to run
against any local ONNX file.
"""

import argparse
import json
import statistics

import numpy as np
import onnx
import onnxruntime as ort


def parse_bindings(specs):
    bindings = {}
    for spec in specs:
        name, dims = spec.split("=", 1)
        bindings[name] = [int(value) for value in dims.split(",")]
    return bindings


def classify(op):
    """Group ORT's op spellings into the classes fastnn's labels use."""
    if op in ("Conv", "FusedConv", "ConvInteger", "QLinearConv"):
        return "conv"
    if op in ("MaxPool", "AveragePool", "GlobalAveragePool"):
        return "pool"
    if op.startswith("Resize") or op == "Upsample":
        return "resize"
    if op == "Concat":
        return "concat"
    if op == "Softmax":
        return "softmax"
    if op == "Transpose":
        return "transpose"
    if op == "Slice":
        return "slice"
    if op == "Split":
        return "split"
    if op == "Reshape":
        return "reshape"
    if op in ("Sigmoid", "HardSwish", "Relu", "Clip", "Mul", "Add", "Sub", "Div"):
        return "elementwise"
    return op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="name=d1,d2,... for a symbolic input; repeatable",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    bindings = parse_bindings(args.shape)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.enable_profiling = True
    session = ort.InferenceSession(
        args.model, sess_options=options, providers=["CPUExecutionProvider"]
    )

    feeds = {}
    for value in session.get_inputs():
        shape = bindings.get(value.name)
        if shape is None:
            if any(not isinstance(dim, int) for dim in value.shape):
                raise SystemExit(
                    f"input {value.name}{value.shape} is symbolic; pass --shape"
                )
            shape = list(value.shape)
        feeds[value.name] = np.zeros(shape, dtype=np.float32)

    for _ in range(args.warmup):
        session.run(None, feeds)
    for _ in range(args.runs):
        session.run(None, feeds)
    profile_path = session.end_profiling()

    with open(profile_path) as handle:
        events = json.load(handle)
    nodes = [event for event in events if event.get("cat") == "Node"]
    per_run = len(events) and (
        len(nodes) // (args.warmup + args.runs) if nodes else 0
    )
    if not per_run:
        raise SystemExit("the profile contains no node events")
    groups = [
        nodes[start : start + per_run]
        for start in range(0, len(nodes), per_run)
        if len(nodes[start : start + per_run]) == per_run
    ]
    # The first `warmup` groups are not measured: they may include one-time
    # kernel selection or cache warming.
    runs = groups[args.warmup : args.warmup + args.runs]
    if not runs:
        raise SystemExit("the profile has fewer runs than --warmup + --runs")

    type_totals = {}
    per_node = {}
    for run in runs:
        totals = {}
        for event in run:
            op = event.get("args", {}).get("op_name", event.get("name", "?"))
            cls = classify(op)
            milliseconds = event["dur"] / 1000.0
            totals[cls] = totals.get(cls, 0.0) + milliseconds
            per_node.setdefault(event.get("name", op), []).append(milliseconds)
        for cls, milliseconds in totals.items():
            type_totals.setdefault(cls, []).append(milliseconds)

    total = statistics.median(
        [sum(samples[index] for samples in type_totals.values()) for index in range(len(runs))]
    )
    print(f"model {args.model}, threads {args.threads}, runs {len(runs)}")
    print(f"{'class':<16} {'median_ms':>10} {'share':>7}")
    for cls, samples in sorted(
        type_totals.items(), key=lambda item: -statistics.median(item[1])
    ):
        milliseconds = statistics.median(samples)
        print(f"{cls:<16} {milliseconds:10.2f} {100 * milliseconds / total:6.1f}%")
    print(f"{'total':<16} {total:10.2f}")

    print("\ntop nodes:")
    for name, samples in sorted(per_node.items(), key=lambda item: -statistics.median(item[1]))[
        :15
    ]:
        print(f"  {statistics.median(samples):8.2f} ms  {name}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "threads": args.threads,
                    "total_ms": total,
                    "types": {
                        cls: statistics.median(samples)
                        for cls, samples in type_totals.items()
                    },
                    "nodes": {
                        name: statistics.median(samples)
                        for name, samples in per_node.items()
                    },
                },
                handle,
                indent=1,
            )


if __name__ == "__main__":
    main()
