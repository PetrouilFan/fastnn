#!/usr/bin/env python3
"""Create a static-shape variant of the tiny random GPT-2 FastNN artifact.

The stock ``onnx-community/tiny-random-gpt2-ONNX`` graph cannot be compiled by
FastNN 2.6.0 at a2c0759: several ``Reshape`` nodes use ``-1`` with a symbolic
element count, and the Rust converter rejects the resulting symbolic division.
This tool rewrites those ``Reshape`` shape initializers (and the recorded
symbolic output shapes) to concrete values for one ``(tokens, past)`` pair so the
same 257-node graph can still be executed and timed.

The result is a *static-shape workaround fixture*, not evidence that the
dynamic-shape path works. See docs/reports/fn3-01/ for the recorded failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fastnn
from fastnn.io import DTYPE_I64, read_fnn_header, read_fnn_parameters, write_fnn_file_v3


def _concrete_shape(node_name: str, tokens: int) -> list[int] | None:
    """Return the concrete target for a Reshape node that carried ``-1``."""
    # All four per-layer Reshape nodes flatten [1, tokens, width] to
    # [tokens, width]; only the trailing width differs.
    if node_name.endswith("/attn/c_attn/Reshape"):
        return [tokens, 128]
    if node_name.endswith("/attn/c_proj/Reshape"):
        return [tokens, 128]
    if node_name.endswith("/mlp/c_fc/Reshape"):
        return [tokens, 128]
    if node_name.endswith("/mlp/c_proj/Reshape"):
        return [tokens, 512]
    if node_name.endswith("/transformer/Reshape_2"):
        return [tokens, 1]
    return None


def _rewrite_dimension(value: str, tokens: int, past: int) -> str:
    table = {
        "Symbol(batch_size*sequence_length)": f"Known({tokens})",
        "Symbol(batch_size)": "Known(1)",
        "Symbol(sequence_length)": f"Known({tokens})",
        "Symbol(past_sequence_length + sequence_length)": f"Known({tokens + past})",
        "Symbol(past_sequence_length + 1)": f"Known({tokens + past})",
        "Symbol(past_sequence_length)": f"Known({past})",
    }
    return table.get(value, value)


def make_variant(source_fnn: Path, output_fnn: Path, tokens: int, past: int) -> dict:
    with source_fnn.open("rb") as handle:
        magic, version, header, num_params = read_fnn_header(handle)
        params = read_fnn_parameters(handle, num_params, version=version)

    if version < 3:
        raise SystemExit("static variant requires a v3 .fnn artifact")

    changed: list[dict] = []
    for node in header["graph"]["nodes"]:
        shape = node.get("output_shape", {}).get("shape", [])
        if shape:
            node["output_shape"]["shape"] = [
                _rewrite_dimension(value, tokens, past) for value in shape
            ]
        if node.get("opcode") != "Reshape" and node.get("op_type") != "Reshape":
            continue
        inputs = node.get("inputs", [])
        if isinstance(inputs, str):
            inputs = [part.strip() for part in inputs.split(",") if part.strip()]
        if len(inputs) < 2 or inputs[1] not in params:
            continue
        array = np.asarray(params[inputs[1]][0])
        if -1 not in array.tolist():
            continue
        concrete = _concrete_shape(node.get("name", ""), tokens)
        if concrete is None:
            raise SystemExit(f"unhandled Reshape -1 node: {node.get('name')}")
        new_name = f"{node.get('name')}.static_shape"
        params[new_name] = (np.asarray(concrete, dtype=np.int64), DTYPE_I64, [], [], list(np.shape(concrete)))
        comma = ",".join([inputs[0], new_name, *inputs[2:]])
        node["inputs"] = comma
        changed.append({"node": node.get("name"), "values": concrete})

    write_params = [
        (name, value[0], value[1], value[2], value[3], value[4])
        for name, value in params.items()
    ]
    output_fnn.parent.mkdir(parents=True, exist_ok=True)
    with output_fnn.open("wb") as handle:
        write_fnn_file_v3(handle, header, write_params, magic=magic, version=version)

    return {
        "source": str(source_fnn),
        "output": str(output_fnn),
        "tokens": tokens,
        "past": past,
        "rewritten_reshapes": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fnn", type=Path, required=True)
    parser.add_argument("--output-fnn", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--past", type=int, default=0)
    parser.add_argument("--build", action="store_true", help="Build the variant after writing")
    args = parser.parse_args()

    report = make_variant(args.source_fnn, args.output_fnn, args.tokens, args.past)
    if args.build:
        fastnn.build_model_from_fnn(str(args.output_fnn))
        report["built"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
