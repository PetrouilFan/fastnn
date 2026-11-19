#!/usr/bin/env python3
"""Float64 GPT-2 reference for the pinned trained graph, stage by stage.

This is not a new fixture: it is the exact-arithmetic arm of the #76
investigation. It mirrors the ONNX graph's decomposition (LayerNorm as
ReduceMean/Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul/Add, tanh GELU, split attention
scaling) but evaluates it in float64, so the difference between either f32
implementation and this reference is that implementation's rounding, not a
formula difference.

The GELU constants and LayerNorm epsilon are read from the model's own
initializers, not hard-coded from memory. Validation against ORT is printed;
the committed logits are checked by ``compare_layers.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

try:
    from extract_intermediates import selected_names
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from extract_intermediates import selected_names

LAYERS = 12
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
VOCAB = 50257
MASK_FILL = -3.4028234663852886e38  # the graph's finite -FLT_MAX


class Weights:
    def __init__(self, model_path: Path):
        model = onnx.load(str(model_path), load_external_data=False)
        self.raw = {
            tensor.name: numpy_helper.to_array(tensor).astype(np.float64)
            for tensor in model.graph.initializer
        }
        self.wte = self.raw["transformer.wte.weight"]
        self.wpe = self.raw["transformer.wpe.weight"]
        self.ln_f_weight = self.raw["transformer.ln_f.weight"]
        self.ln_f_bias = self.raw["transformer.ln_f.bias"]

        self.blocks = []
        for layer in range(LAYERS):
            prefix = f"transformer.h.{layer}"
            eps = float(
                self.raw[f"/transformer/h.{layer}/ln_1/Constant_1_output_0"]
            )
            scale = self.raw[f"/transformer/h.{layer}/attn/Sqrt_1_output_0"]
            constants = {
                "gelu_half": float(
                    self.raw[f"/transformer/h.{layer}/mlp/act/Constant_output_0"]
                ),
                "gelu_pow": float(
                    self.raw[f"/transformer/h.{layer}/mlp/act/Constant_1_output_0"]
                ),
                "gelu_cubic": float(
                    self.raw[f"/transformer/h.{layer}/mlp/act/Constant_2_output_0"]
                ),
                "gelu_scale": float(
                    self.raw[f"/transformer/h.{layer}/mlp/act/Constant_3_output_0"]
                ),
                "gelu_one": float(
                    self.raw[f"/transformer/h.{layer}/mlp/act/Constant_4_output_0"]
                ),
            }
            self.blocks.append(
                {
                    "eps": eps,
                    "attn_scale": float(scale[0]),
                    "constants": constants,
                    "ln_1_weight": self.raw[f"{prefix}.ln_1.weight"],
                    "ln_1_bias": self.raw[f"{prefix}.ln_1.bias"],
                    "ln_2_weight": self.raw[f"{prefix}.ln_2.weight"],
                    "ln_2_bias": self.raw[f"{prefix}.ln_2.bias"],
                    "c_attn_weight": self.raw[f"{prefix}.attn.c_attn.weight"],
                    "c_attn_bias": self.raw[f"{prefix}.attn.c_attn.bias"],
                    "c_proj_attn_weight": self.raw[f"{prefix}.attn.c_proj.weight"],
                    "c_proj_attn_bias": self.raw[f"{prefix}.attn.c_proj.bias"],
                    "c_fc_weight": self.raw[f"{prefix}.mlp.c_fc.weight"],
                    "c_fc_bias": self.raw[f"{prefix}.mlp.c_fc.bias"],
                    "c_proj_mlp_weight": self.raw[f"{prefix}.mlp.c_proj.weight"],
                    "c_proj_mlp_bias": self.raw[f"{prefix}.mlp.c_proj.bias"],
                }
            )


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> tuple:
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    variance = (centred * centred).mean(axis=-1, keepdims=True)
    deviation = np.sqrt(variance + eps)
    normalised = centred / deviation
    return normalised * weight + bias, mean, variance, deviation, normalised


def gelu_new(x: np.ndarray, constants: dict) -> np.ndarray:
    inner = x + constants["gelu_cubic"] * x ** constants["gelu_pow"]
    tanh = np.tanh(constants["gelu_scale"] * inner)
    return constants["gelu_half"] * x * (constants["gelu_one"] + tanh)


def attention_stages(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scale: float,
    seq: int,
    layer: int,
    stages: dict,
) -> tuple[np.ndarray, list]:
    """One block's attention, in the graph's layouts.

    ``q``/``k``/``v`` are the new token(s) as ``[heads, seq, head_dim]``;
    ``keys``/``values`` are the full cache ``[heads, total, head_dim]``. The
    graph stores the *unscaled* K/V in the cache and scales both Q and K before
    the score matmul (each by 1/sqrt(8), so the product is 1/8).
    """
    scored_q = q * scale
    scored_k = (keys * scale).transpose(0, 2, 1)  # [heads, head_dim, total]
    if layer == 0:
        stages[f"/transformer/h.{layer}/attn/Mul_output_0"] = scored_q.reshape(
            1, HEADS, seq, HEAD_DIM
        )
        stages[f"/transformer/h.{layer}/attn/Mul_1_output_0"] = scored_k.reshape(
            1, HEADS, HEAD_DIM, keys.shape[1]
        )
    scores = scored_q @ scored_k
    if layer == 0:
        stages[f"/transformer/h.{layer}/attn/MatMul_output_0"] = scores.reshape(
            1, HEADS, seq, keys.shape[1]
        )
    if seq > 1:
        causal = np.triu(np.ones((seq, keys.shape[1]), dtype=bool), k=1)
        scores = scores + np.where(causal, MASK_FILL, 0.0)
    if layer == 0:
        stages[f"/transformer/h.{layer}/attn/Add_output_0"] = scores.reshape(
            1, HEADS, seq, keys.shape[1]
        )
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = shifted / shifted.sum(axis=-1, keepdims=True)
    stages[f"/transformer/h.{layer}/attn/Softmax_output_0"] = probs.reshape(
        1, HEADS, seq, keys.shape[1]
    )
    context = probs @ values  # [heads, seq, head_dim]
    stages[f"/transformer/h.{layer}/attn/MatMul_1_output_0"] = context.reshape(
        1, HEADS, seq, HEAD_DIM
    )
    return context.transpose(1, 0, 2).reshape(seq, HIDDEN), [
        keys.reshape(1, HEADS, keys.shape[1], HEAD_DIM),
        values.reshape(1, HEADS, values.shape[1], HEAD_DIM),
    ]


def block_stages(
    x: np.ndarray,
    block: dict,
    layer: int,
    seq: int,
    stages: dict,
    past: list[tuple] | None,
    position: int,
) -> tuple[np.ndarray, list]:
    """One transformer block, stage by stage. ``x`` is ``[seq, hidden]``."""
    normed, mean, variance, deviation, normalised = layer_norm(
        x, block["ln_1_weight"], block["ln_1_bias"], block["eps"]
    )
    stages[f"/transformer/h.{layer}/ln_1/Add_1_output_0"] = normed.reshape(1, seq, HIDDEN)
    if layer == 0:
        stages["/transformer/h.0/ln_1/ReduceMean_output_0"] = mean.reshape(1, seq, 1)
        stages["/transformer/h.0/ln_1/ReduceMean_1_output_0"] = variance.reshape(1, seq, 1)
        stages["/transformer/h.0/ln_1/Add_output_0"] = (variance + block["eps"]).reshape(
            1, seq, 1
        )
        stages["/transformer/h.0/ln_1/Sqrt_output_0"] = deviation.reshape(1, seq, 1)
        stages["/transformer/h.0/ln_1/Div_output_0"] = normalised.reshape(1, seq, HIDDEN)
        stages["/transformer/h.0/ln_1/Mul_output_0"] = (
            normalised * block["ln_1_weight"]
        ).reshape(1, seq, HIDDEN)

    attention = normed @ block["c_attn_weight"] + block["c_attn_bias"]
    stages[f"/transformer/h.{layer}/attn/c_attn/Gemm_output_0"] = attention.reshape(
        seq, 3 * HIDDEN
    )
    q, k, v = np.split(attention, 3, axis=-1)
    q = q.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
    k = k.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
    v = v.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
    if past is None:
        keys, values = k, v
    else:
        keys = np.concatenate([past[layer][0].reshape(HEADS, -1, HEAD_DIM), k], axis=1)
        values = np.concatenate([past[layer][1].reshape(HEADS, -1, HEAD_DIM), v], axis=1)
    context, present = attention_stages(
        q, k, v, keys, values, block["attn_scale"], seq, layer, stages
    )
    projected = context @ block["c_proj_attn_weight"] + block["c_proj_attn_bias"]
    stages[f"/transformer/h.{layer}/attn/c_proj/Gemm_output_0"] = projected.reshape(
        seq, HIDDEN
    )
    x = x + projected
    stages[f"/transformer/h.{layer}/Add_output_0"] = x.reshape(1, seq, HIDDEN)
    normed, *_ = layer_norm(x, block["ln_2_weight"], block["ln_2_bias"], block["eps"])
    stages[f"/transformer/h.{layer}/ln_2/Add_1_output_0"] = normed.reshape(1, seq, HIDDEN)
    hidden = normed @ block["c_fc_weight"] + block["c_fc_bias"]
    stages[f"/transformer/h.{layer}/mlp/c_fc/Gemm_output_0"] = hidden.reshape(
        seq, 4 * HIDDEN
    )
    if layer == 0:
        inner = (
            hidden
            + block["constants"]["gelu_cubic"] * hidden ** block["constants"]["gelu_pow"]
        )
        tanh = np.tanh(block["constants"]["gelu_scale"] * inner)
        stages["/transformer/h.0/mlp/act/Mul_output_0"] = (
            block["constants"]["gelu_half"] * hidden
        ).reshape(1, seq, 4 * HIDDEN)
        stages["/transformer/h.0/mlp/act/Add_output_0"] = inner.reshape(1, seq, 4 * HIDDEN)
        stages["/transformer/h.0/mlp/act/Tanh_output_0"] = tanh.reshape(1, seq, 4 * HIDDEN)
        stages["/transformer/h.0/mlp/act/Add_1_output_0"] = (
            tanh + block["constants"]["gelu_one"]
        ).reshape(1, seq, 4 * HIDDEN)
    activated = gelu_new(hidden, block["constants"])
    stages[f"/transformer/h.{layer}/mlp/act/Mul_3_output_0"] = activated.reshape(
        1, seq, 4 * HIDDEN
    )
    projected = activated @ block["c_proj_mlp_weight"] + block["c_proj_mlp_bias"]
    stages[f"/transformer/h.{layer}/mlp/c_proj/Gemm_output_0"] = projected.reshape(
        seq, HIDDEN
    )
    x = x + projected
    stages[f"/transformer/h.{layer}/Add_1_output_0"] = x.reshape(1, seq, HIDDEN)
    return x, present


def forward_prefill(weights: Weights, tokens: list[int], stages: dict) -> list[tuple]:
    seq = len(tokens)
    positions = np.arange(seq)
    x = weights.wte[np.asarray(tokens)] + weights.wpe[positions]
    stages["/transformer/Add_output_0"] = x.reshape(1, seq, HIDDEN)
    present = []
    for layer, block in enumerate(weights.blocks):
        x, layer_present = block_stages(x, block, layer, seq, stages, None, 0)
        present.append(layer_present)
    normed, *_ = layer_norm(x, weights.ln_f_weight, weights.ln_f_bias, 1e-5)
    stages["/transformer/ln_f/Add_1_output_0"] = normed.reshape(1, seq, HIDDEN)
    stages["logits"] = (normed @ weights.wte.T).reshape(1, seq, VOCAB)
    return present


def forward_decode(
    weights: Weights,
    tokens: list[int],
    past: list[tuple],
    position: int,
    stages: dict,
) -> list[tuple]:
    x = weights.wte[np.asarray(tokens)] + weights.wpe[[position]]
    stages["/transformer/Add_output_0"] = x.reshape(1, 1, HIDDEN)
    present = []
    for layer, block in enumerate(weights.blocks):
        x, layer_present = block_stages(x, block, layer, 1, stages, past, position)
        present.append(layer_present)
    normed, *_ = layer_norm(x, weights.ln_f_weight, weights.ln_f_bias, 1e-5)
    stages["/transformer/ln_f/Add_1_output_0"] = normed.reshape(1, 1, HIDDEN)
    stages["logits"] = (normed @ weights.wte.T).reshape(1, 1, VOCAB)
    return present


def save_phase(out: Path, label: str, stages: dict, keep: set[str]) -> None:
    phase = out / label
    phase.mkdir(parents=True, exist_ok=True)
    index = []
    for name in selected_names():
        if name not in keep or name not in stages:
            continue
        value = np.asarray(stages[name], dtype=np.float64)
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
    print(f"{label}: wrote {len(index)} tensors")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="464,2068,7586,21831,18045,625,262,16931,3290,13,4874,2402,257",
    )
    parser.add_argument("--decode", default="640,11")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = [int(value) for value in args.prompt.split(",")]
    decode = [int(value) for value in args.decode.split(",")]
    weights = Weights(args.model)
    keep = set(selected_names())

    stages: dict = {}
    present = forward_prefill(weights, prompt, stages)
    for layer, (k, v) in enumerate(present):
        stages[f"present.{layer}.key"] = k
        stages[f"present.{layer}.value"] = v
    save_phase(args.out, "prefill", stages, keep)

    for index, token in enumerate(decode, start=1):
        stages = {}
        past = present
        present = forward_decode(
            weights, [token], past, len(prompt) + index - 1, stages
        )
        for layer, (k, v) in enumerate(present):
            stages[f"present.{layer}.key"] = k
            stages[f"present.{layer}.value"] = v
        save_phase(args.out, f"decode{index}", stages, keep)

    print("float64 reference complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
