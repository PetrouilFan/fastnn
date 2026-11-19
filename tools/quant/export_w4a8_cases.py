#!/usr/bin/env python3
"""Export real GPT-2 projection weights and activations for the Rust W4A8 check.

The files are written outside the repository (``--out``).  The companion Rust
test (``rust_grouped_w4a8_matches_the_independent_reference``) reads them,
runs the actual 2.x `PackedTensor` quantizer and grouped W4A8 kernel, and
``tools/quant/check_rust_w4a8.py`` re-derives every byte with this independent
Python implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_w4a8 as e

CASES = [
    "h.0.attn.c_attn",
    "h.0.attn.c_proj",
    "h.0.mlp.c_fc",
    "h.0.mlp.c_proj",
    "lm_head",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = e.load_model(args.onnx)
    table = e.projection_table(model)
    captured: dict[str, np.ndarray] = {}

    def recording(x, name):
        captured.setdefault(name, np.asarray(x, dtype=np.float32).copy())
        weight, bias = table[name]
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out

    e.forward(model, e.PROMPT, recording)
    cases = []
    for name in CASES:
        weight, _bias = table[name]
        weight = np.asarray(weight, dtype=np.float32)
        activation = captured[name]
        weight_path = args.out / f"{name}.weight.f32"
        activation_path = args.out / f"{name}.activation.f32"
        weight_path.write_bytes(weight.tobytes())
        activation_path.write_bytes(activation.tobytes())
        cases.append(
            {
                "name": name,
                "n": int(weight.shape[0]),
                "k": int(weight.shape[1]),
                "m": int(activation.shape[0]),
                "weight": weight_path.name,
                "activation": activation_path.name,
            }
        )
        print(f"{name}: weight {weight.shape}, activation {activation.shape}")
    (args.out / "cases.json").write_text(json.dumps({"cases": cases, "prompt": e.PROMPT}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
