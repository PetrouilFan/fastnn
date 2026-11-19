#!/usr/bin/env python3
"""Evaluate candidate weight formats on the pinned trained GPT-2 graph.

The measurement has three independent layers:

1. **Per-tensor error** of each candidate encoder against the real ONNX
   projection weights (shapes read from the graph, not assumed).
2. **End-to-end logits** from a NumPy GPT-2 forward whose only difference from
   the FP32 path is the candidate quantization.  The FP32 forward is first
   checked against the committed ONNX Runtime reference.
3. **Exact W4A8 kernel semantics** for the ``new`` and ``existing`` formats:
   per-token INT8 activations, per-group INT4 dot, integer zero-point
   correction, FP32 group accumulation.

Nothing imports FastNN.  The Rust grouped-W4A8 path is checked separately by
``tools/quant/check_rust_w4a8.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnx

import fn3_quant as fq

EOS = -3.4028234663852886e38
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257]
DECODE = [640, 11]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Model:
    wte: np.ndarray
    wpe: np.ndarray
    ln1: list[tuple[np.ndarray, np.ndarray]]
    ln2: list[tuple[np.ndarray, np.ndarray]]
    lnf: tuple[np.ndarray, np.ndarray]
    attn: list[tuple[np.ndarray, np.ndarray]]
    attn_proj: list[tuple[np.ndarray, np.ndarray]]
    fc: list[tuple[np.ndarray, np.ndarray]]
    mlp_proj: list[tuple[np.ndarray, np.ndarray]]

    @property
    def layers(self) -> int:
        return len(self.attn)


def initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    for tensor in model.graph.initializer:
        if tensor.name == name:
            if tensor.data_type != onnx.TensorProto.FLOAT:
                raise ValueError(f"{name} is not f32")
            return onnx.numpy_helper.to_array(tensor)
    raise KeyError(name)


def load_model(path: Path) -> Model:
    proto = onnx.load(str(path), load_external_data=False)
    layers = 12
    wte = initializer(proto, "transformer.wte.weight")
    wpe = initializer(proto, "transformer.wpe.weight")
    ln1, ln2 = [], []
    attn, attn_proj, fc, mlp_proj = [], [], [], []
    for layer in range(layers):
        prefix = f"transformer.h.{layer}"
        ln1.append(
            (
                initializer(proto, f"{prefix}.ln_1.weight"),
                initializer(proto, f"{prefix}.ln_1.bias"),
            )
        )
        ln2.append(
            (
                initializer(proto, f"{prefix}.ln_2.weight"),
                initializer(proto, f"{prefix}.ln_2.bias"),
            )
        )
        attn.append(
            (
                initializer(proto, f"{prefix}.attn.c_attn.weight"),
                initializer(proto, f"{prefix}.attn.c_attn.bias"),
            )
        )
        attn_proj.append(
            (
                initializer(proto, f"{prefix}.attn.c_proj.weight"),
                initializer(proto, f"{prefix}.attn.c_proj.bias"),
            )
        )
        fc.append(
            (
                initializer(proto, f"{prefix}.mlp.c_fc.weight"),
                initializer(proto, f"{prefix}.mlp.c_fc.bias"),
            )
        )
        mlp_proj.append(
            (
                initializer(proto, f"{prefix}.mlp.c_proj.weight"),
                initializer(proto, f"{prefix}.mlp.c_proj.bias"),
            )
        )
    return Model(
        wte=wte,
        wpe=wpe,
        ln1=ln1,
        ln2=ln2,
        lnf=(
            initializer(proto, "transformer.ln_f.weight"),
            initializer(proto, "transformer.ln_f.bias"),
        ),
        attn=attn,
        attn_proj=attn_proj,
        fc=fc,
        mlp_proj=mlp_proj,
    )


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def gelu_new(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


Linear = Callable[[np.ndarray, str], np.ndarray]


def forward(model: Model, tokens: list[int], linear: Linear) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.int64)
    seq = len(tokens)
    x = model.wte[tokens] + model.wpe[:seq]
    mask = np.triu(np.full((seq, seq), EOS, dtype=np.float64), k=1)
    heads, head_dim = 12, 64
    for index in range(model.layers):
        gamma, beta = model.ln1[index]
        h = layer_norm(x, gamma, beta)
        qkv = linear(h, f"h.{index}.attn.c_attn")
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(seq, heads, head_dim).transpose(1, 0, 2) * (1.0 / math.sqrt(head_dim))
        k = k.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, heads, head_dim).transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        context = (weights @ v).transpose(1, 0, 2).reshape(seq, heads * head_dim)
        x = x + linear(context, f"h.{index}.attn.c_proj")
        gamma, beta = model.ln2[index]
        h = layer_norm(x, gamma, beta)
        h = gelu_new(linear(h, f"h.{index}.mlp.c_fc"))
        x = x + linear(h, f"h.{index}.mlp.c_proj")
    gamma, beta = model.lnf
    x = layer_norm(x, gamma, beta)
    return linear(x, "lm_head")


def linear_fp32(model: Model) -> Linear:
    table = {
        f"h.{i}.attn.c_attn": model.attn[i] for i in range(model.layers)
    }
    table.update({f"h.{i}.attn.c_proj": model.attn_proj[i] for i in range(model.layers)})
    table.update({f"h.{i}.mlp.c_fc": model.fc[i] for i in range(model.layers)})
    table.update({f"h.{i}.mlp.c_proj": model.mlp_proj[i] for i in range(model.layers)})

    def linear(x, name):
        if name == "lm_head":
            return x @ model.wte.T
        weight, bias = table[name]
        return x @ weight + bias

    return linear


def projection_table(model: Model) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Every quantizable projection as ``[N, K]`` with its bias."""

    table: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "lm_head": (model.wte, None)
    }
    for index in range(model.layers):
        weight, bias = model.attn[index]
        table[f"h.{index}.attn.c_attn"] = (weight.T, bias)
        weight, bias = model.attn_proj[index]
        table[f"h.{index}.attn.c_proj"] = (weight.T, bias)
        weight, bias = model.fc[index]
        table[f"h.{index}.mlp.c_fc"] = (weight.T, bias)
        weight, bias = model.mlp_proj[index]
        table[f"h.{index}.mlp.c_proj"] = (weight.T, bias)
    return table


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class Scheme:
    name: str
    quantize: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    group_sizes: tuple[int, ...]
    scale_bytes: int
    zero_bytes: int
    dequantize: Callable[..., np.ndarray] = fq.dequantize
    matmul: Callable[..., np.ndarray] = fq.w4a8_matmul
    extra_bytes_per_group: int = 0
    bits: int = 4


SCHEMES = {
    "existing-w4a8": Scheme(
        "existing-w4a8",
        fq.quantize_int4_existing_w4a8,
        (32, 64, 128),
        scale_bytes=4,
        zero_bytes=4,
        dequantize=fq.dequantize_float_zero,
        matmul=fq.w4a8_matmul_float_zero,
        extra_bytes_per_group=4,
    ),
    "asym-int4": Scheme(
        "asym-int4",
        fq.quantize_int4_grouped,
        (32, 64, 128),
        scale_bytes=2,
        zero_bytes=2,
    ),
    "sym-int4-7": Scheme(
        "sym-int4-7",
        lambda w, g: fq.quantize_int4_symmetric(w, g, 7.0),
        (32, 64, 128),
        scale_bytes=2,
        zero_bytes=2,
    ),
    "sym-int4-8": Scheme(
        "sym-int4-8",
        lambda w, g: fq.quantize_int4_symmetric(w, g, 8.0),
        (32, 64, 128),
        scale_bytes=2,
        zero_bytes=2,
    ),
    "asym-int8": Scheme(
        "asym-int8",
        fq.quantize_int8_grouped,
        (32, 64, 128),
        scale_bytes=2,
        zero_bytes=2,
        bits=8,
    ),
    "sym-int8": Scheme(
        "sym-int8",
        fq.quantize_int8_symmetric,
        (64,),
        scale_bytes=2,
        zero_bytes=2,
        bits=8,
    ),
}


def tensor_metrics(weight: np.ndarray, dequant: np.ndarray) -> dict:
    delta = weight.astype(np.float64) - dequant.astype(np.float64)
    ref = weight.astype(np.float64)
    return {
        "elements": int(weight.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_frobenius": float(
            np.sqrt(np.sum(delta * delta)) / max(np.sqrt(np.sum(ref * ref)), 1e-30)
        ),
    }


def evaluate_tensors(model: Model, schemes: list[str], groups: list[int]) -> dict:
    table = projection_table(model)
    result = {}
    for scheme_name in schemes:
        scheme = SCHEMES[scheme_name]
        for group in groups:
            if group not in scheme.group_sizes:
                continue
            rows = []
            total_bytes = 0
            total_fp32 = 0
            for name, (weight, _bias) in table.items():
                codes, scales, zeros = scheme.quantize(weight, group)
                dequant = scheme.dequantize(codes, scales, zeros, group)
                metrics = tensor_metrics(weight, dequant)
                packed_row = fq.align_up(
                    (weight.shape[1] * scheme.bits + 7) // 8, 16
                )
                packed = weight.shape[0] * packed_row
                group_count = (weight.shape[1] + group - 1) // group
                meta = (
                    weight.shape[0]
                    * group_count
                    * (
                        scheme.scale_bytes
                        + scheme.zero_bytes
                        + scheme.extra_bytes_per_group
                    )
                )
                metrics.update(
                    {
                        "name": name,
                        "shape_nk": list(weight.shape),
                        "packed_bytes": int(packed),
                        "metadata_bytes": int(meta),
                        "total_bytes": int(packed + meta),
                    }
                )
                total_bytes += packed + meta
                total_fp32 += weight.size * 4
                rows.append(metrics)
            result[f"{scheme_name}-g{group}"] = {
                "group_size": group,
                "projections": rows,
                "summary": {
                    "max_relative_frobenius": max(
                        row["relative_frobenius"] for row in rows
                    ),
                    "mean_relative_frobenius": float(
                        np.mean([row["relative_frobenius"] for row in rows])
                    ),
                    "max_absolute": max(row["max_abs"] for row in rows),
                    "projection_bytes": int(total_bytes),
                    "fp32_projection_bytes": int(total_fp32),
                    "ratio_vs_fp32": float(total_bytes / total_fp32),
                },
            }
    return result


# ---------------------------------------------------------------------------
# Simulated inference
# ---------------------------------------------------------------------------


def quantized_linear(
    model: Model,
    scheme: Scheme,
    group: int,
    quantized_names: set[str],
    activation: str,
    overrides: dict[str, tuple[Scheme, int]] | None = None,
) -> Linear:
    table = projection_table(model)
    prepared = {}
    for name, (weight, _bias) in table.items():
        if name not in quantized_names:
            prepared[name] = ("f32", weight)
            continue
        chosen, chosen_group = (overrides or {}).get(name, (scheme, group))
        codes, scales, zeros = chosen.quantize(weight, chosen_group)
        if activation == "int8":
            prepared[name] = ("w4a8", (chosen, codes, scales, zeros, chosen_group))
        else:
            prepared[name] = (
                "dequant",
                chosen.dequantize(codes, scales, zeros, chosen_group),
            )

    biases = {}
    for name, (weight, bias) in table.items():
        if bias is not None:
            biases[name] = bias

    def linear(x, name):
        kind, value = prepared[name]
        if kind == "f32":
            out = x @ value.T
        elif kind == "dequant":
            out = x @ value.T
        else:
            chosen, codes, scales, zeros, chosen_group = value
            out = chosen.matmul(
                np.asarray(x, dtype=np.float32), codes, scales, zeros, chosen_group
            )
        bias = biases.get(name)
        if bias is not None:
            out = out + bias
        return out

    return linear


def logit_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    delta = actual - reference
    top1 = int(np.argmax(actual))
    ref_top1 = int(np.argmax(reference))
    actual_top5 = set(int(i) for i in np.argsort(-actual)[:5])
    ref_top5 = set(int(i) for i in np.argsort(-reference)[:5])
    centered = delta - delta.mean()
    p = np.exp(reference - reference.max())
    p /= p.sum()
    q = np.exp(actual - actual.max())
    q /= q.sum()
    order = np.argsort(-reference)
    margin = float(reference[order[0]] - reference[order[1]])
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "centered_max_abs": float(np.max(np.abs(centered))),
        "centered_rmse": float(np.sqrt(np.mean(centered * centered))),
        "reference_rms": float(np.sqrt(np.mean(reference * reference))),
        "reference_centered_std": float(np.std(reference)),
        "kl_reference_to_actual": float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300)))),
        "top1_margin": margin,
        "argmax": top1,
        "reference_argmax": ref_top1,
        "top1_agree": top1 == ref_top1,
        "top5_overlap": len(actual_top5 & ref_top5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.onnx)
    reference = np.load(args.reference).reshape(-1)
    baseline = forward(model, PROMPT, linear_fp32(model))[-1].reshape(-1)
    baseline_metrics = logit_metrics(baseline, reference)

    all_names = set(projection_table(model))
    no_lm_head = all_names - {"lm_head"}
    report: dict = {
        "onnx": str(args.onnx),
        "grammar": {
            "prompt": PROMPT,
            "reference": str(args.reference),
            "baseline_fp32_sim_vs_ort": baseline_metrics,
        },
        "tensor_error": evaluate_tensors(
            model,
            [
                "existing-w4a8",
                "asym-int4",
                "sym-int4-7",
                "sym-int4-8",
                "asym-int8",
                "sym-int8",
            ],
            [32, 64, 128],
        ),
        "end_to_end": {},
    }

    variants = [
        ("existing-w4a8-g64", "existing-w4a8", 64, "int8", all_names),
        ("existing-w4a8-g128", "existing-w4a8", 128, "int8", all_names),
        ("asym-int4-g32", "asym-int4", 32, "int8", all_names),
        ("asym-int4-g64", "asym-int4", 64, "int8", all_names),
        ("asym-int4-g128", "asym-int4", 128, "int8", all_names),
        ("asym-int4-g64-no-lm-head", "asym-int4", 64, "int8", no_lm_head),
        ("asym-int4-g64-weight-only", "asym-int4", 64, "weight", all_names),
        ("sym-int4-7-g64", "sym-int4-7", 64, "int8", all_names),
        ("sym-int4-8-g64", "sym-int4-8", 64, "int8", all_names),
        ("asym-int8-g32-weight-only", "asym-int8", 32, "weight", all_names),
        ("asym-int8-g64", "asym-int8", 64, "int8", all_names),
        ("sym-int8-g64", "sym-int8", 64, "int8", all_names),
        ("asym-int8-g64-weight-only", "asym-int8", 64, "weight", all_names),
        ("asym-int8-g128-weight-only", "asym-int8", 128, "weight", all_names),
        ("mixed-a", "asym-int4", 64, "int8", all_names),
        ("mixed-b", "asym-int4", 64, "int8", all_names),
        ("mixed-c", "asym-int4", 64, "int8", all_names),
    ]
    mixed_policies = {
        # Attention QKV and MLP expansion in INT4; the projections back into
        # the residual stream and the tied head in INT8.
        "mixed-a": {
            name: (SCHEMES["asym-int8"], 64)
            for name in all_names
            if name.endswith(("attn.c_proj", "mlp.c_proj")) or name == "lm_head"
        },
        # Only the tied head is protected.
        "mixed-b": {"lm_head": (SCHEMES["asym-int8"], 64)},
        # INT4 attention, INT8 MLP and head.
        "mixed-c": {
            name: (SCHEMES["asym-int8"], 64)
            for name in all_names
            if name.endswith(("mlp.c_fc", "mlp.c_proj")) or name == "lm_head"
        },
    }
    for label, scheme_name, group, activation, names in variants:
        scheme = SCHEMES[scheme_name]
        linear = quantized_linear(
            model, scheme, group, names, activation, mixed_policies.get(label)
        )
        logits = forward(model, PROMPT, linear)[-1].reshape(-1)
        report["end_to_end"][label] = {
            "scheme": scheme_name,
            "group_size": group,
            "activation": activation,
            "quantized_projections": len(names),
            "metrics": logit_metrics(logits, baseline),
            "vs_ort": logit_metrics(logits, reference),
        }
        m = report["end_to_end"][label]["metrics"]
        print(
            f"{label}: max_abs {m['max_abs']:.6g} centered_max {m['centered_max_abs']:.6g} "
            f"centered_rmse {m['centered_rmse']:.6g} kl {m['kl_reference_to_actual']:.6g} "
            f"top1 {m['top1_agree']} top5 {m['top5_overlap']}"
        )

    # Decode steps 1 and 2 with the chosen format are measured by re-running
    # the forward with the growing cache; a full recompute over the prefix is
    # exactly equal to cached decode, so it is the cleaner reference here.
    for label, scheme_name, group in [
        ("asym-int4-g64", "asym-int4", 64),
        ("existing-w4a8-g128", "existing-w4a8", 128),
        ("asym-int8-g64", "asym-int8", 64),
        ("asym-int8-g64-weight-only", "asym-int8", 64),
        ("asym-int8-g128-weight-only", "asym-int8", 128),
        ("mixed-a", "asym-int4", 64),
    ]:
        scheme = SCHEMES[scheme_name]
        activation = "weight" if label.endswith("weight-only") else "int8"
        linear = quantized_linear(
            model, scheme, group, all_names, activation, mixed_policies.get(label)
        )
        steps = []
        for step, token in enumerate(DECODE, start=1):
            tokens = PROMPT + DECODE[:step]
            logits = forward(model, tokens, linear)[-1]
            fp32 = forward(model, tokens, linear_fp32(model))[-1]
            steps.append(
                {
                    "step": step,
                    "token": token,
                    "metrics": logit_metrics(logits, fp32),
                }
            )
        report["end_to_end"][f"{label}-decode"] = {"steps": steps}
        for step in steps:
            m = step["metrics"]
            print(
                f"{label} decode {step['step']}: max_abs {m['max_abs']:.6g} "
                f"centered_max {m['centered_max_abs']:.6g} kl "
                f"{m['kl_reference_to_actual']:.6g} top1 {m['top1_agree']}"
            )

    (args.out / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["grammar"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
