#!/usr/bin/env python3
"""Generate independent parity fixtures for FastNN 3.0.

The generator is intentionally boring and deterministic:

* ONNX Runtime (CPU, single thread, graph optimisations disabled) produces the
  whole-model reference.
* A PyTorch ``GPT2LMHeadModel`` loaded from the pinned base checkpoint is an
  independent cross-check.  Its errors are recorded, not committed as truth.
* Small operator references are computed in float64 from float32 inputs and
  rounded once to float32; a float32 PyTorch self-check is recorded.

Nothing in the output depends on wall-clock time, absolute paths, environment
variables, or Python hash ordering.  See ``fixtures/parity/README.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

from model_pins import (
    TINY_BASE_ID,
    TINY_BASE_REVISION,
    TINY_GRAPH,
    TINY_ID,
    TINY_REVISION,
    TINY_TOKENIZER,
    TRAINED_GRAPH,
    TRAINED_ID,
    TRAINED_REVISION,
    TRAINED_TOKENIZER,
    trained_lock,
    tiny_lock,
)
from parity_format import (
    Suite,
    reset_suite_dir,
    sha256_file,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "fixtures" / "parity"
DEFAULT_CACHE = REPO_ROOT / "tools" / "parity" / ".cache" / "models"

# The 17-token sequence is the GPT-2 tokenizer encoding of
#   "The quick brown fox jumps over the lazy dog. Once upon a time, there was"
# The first 13 tokens are the prefill prompt; the remaining four are decode
# steps.  This lets cached decode be compared with a full recomputation of the
# same prefix at every step.
TOKENS = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612, 373]
PROMPT_TEXT = "The quick brown fox jumps over the lazy dog. Once upon a time, there was"
PROMPT_LENGTHS = (1, 4, 8, 13)
DECODE_TOKENS = TOKENS[13:]
MASK_FILL = float(np.float32(-3.4028234663852886e38))


# ---------------------------------------------------------------------------
# deterministic inputs


class SplitMix64:
    """Tiny fixed PRNG; independent of NumPy's RNG implementation history."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return value ^ (value >> 31)

    def uniform(self, low: float, high: float) -> float:
        unit = (self.next_u64() >> 11) / float(1 << 53)
        return low + (high - low) * unit


def rand_array(shape: tuple[int, ...], low: float, high: float, seed: int) -> np.ndarray:
    rng = SplitMix64(seed)
    values = [rng.uniform(low, high) for _ in range(int(np.prod(shape)))]
    return np.asarray(values, dtype=np.float32).reshape(shape)


# ---------------------------------------------------------------------------
# reference math


def as_f32(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).astype(np.float32)


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def layernorm64(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps) * weight + bias


def softmax64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def gelu_erf64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + torch.special.erf(torch.from_numpy(x) / math.sqrt(2.0)).numpy())


def gelu_tanh64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def tolerance(atol: float, rtol: float) -> dict[str, Any]:
    return {
        "atol": atol,
        "rtol": rtol,
        "form": "abs(actual - ref) <= atol + rtol * abs(ref)",
    }


def tensor_provenance(
    *,
    kind: str,
    call: str,
    tolerance_block: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "kind": kind,
        "call": call,
        "tolerance": tolerance_block,
    }
    if extra:
        provenance.update(extra)
    return provenance


# ---------------------------------------------------------------------------
# operator fixtures


def generate_ops(root: Path, environment: dict[str, Any]) -> dict[str, Any]:
    reset_suite_dir(root)
    suite = Suite(
        root,
        "ops",
        description=(
            "Small independent operator references for kernel conformance: "
            "embedding, LayerNorm, MatMul/Gemm, GELU, masked Softmax, and "
            "masked multi-head attention with an all-masked query row."
        ),
        provenance={
            "producer": "explicit float64 reference, rounded once to float32",
            "self_check": "PyTorch 2.15.0+cpu float32",
            "environment": environment,
            "mask_policy": {
                "fill_value": MASK_FILL,
                "fill_dtype": "f32",
                "all_masked_row": (
                    "A row whose additive mask is entirely -FLT_MAX produces a "
                    "uniform distribution.  This is the GPT-2 ONNX mask convention "
                    "and is stable under max-subtraction; it is not NaN."
                ),
            },
        },
    )
    cases: list[dict[str, Any]] = []

    # -- embedding ---------------------------------------------------------
    indices = np.asarray(
        [[0, 16, 3, 3, 7], [9, 1, 9, 0, 16]], dtype=np.int64
    )
    table = rand_array((17, 7), -2.0, 2.0, 0xE0BED)
    embedding = as_f32(table.astype(np.float64)[indices])
    torch_table = torch.from_numpy(table)
    torch_embedding = F.embedding(torch.from_numpy(indices), torch_table).numpy()
    tol = tolerance(0.0, 0.0)
    for name, array, dtype, prov in [
        ("embedding.indices", indices, "i64", tensor_provenance(kind="input", call="indices", tolerance_block=tol)),
        ("embedding.table", table, "f32", tensor_provenance(kind="input", call="embedding weight [17,7]", tolerance_block=tol)),
        ("embedding.out", embedding, "f32", tensor_provenance(kind="reference", call="Gather(axis=0, table, indices)", tolerance_block=tol)),
    ]:
        suite.add(name, array, dtype, provenance=prov)
    cases.append(
        {
            "operator": "embedding",
            "shapes": {"indices": [2, 5], "table": [17, 7], "out": [2, 5, 7]},
            "features": ["duplicate indices", "last table row", "zero row"],
            "self_check_max_abs": max_abs(embedding, torch_embedding),
        }
    )

    # -- LayerNorm ---------------------------------------------------------
    x = rand_array((2, 3, 5), -3.0, 3.0, 0x1A7E)
    x[1, 2, :] = 3.25  # zero-variance row exercises the epsilon path
    weight = rand_array((5,), 0.5, 1.5, 0xBEEF)
    bias = rand_array((5,), -0.25, 0.25, 0xCAFE)
    ln = as_f32(layernorm64(x, weight.astype(np.float64), bias.astype(np.float64), 1e-5))
    torch_ln = F.layer_norm(
        torch.from_numpy(x),
        (5,),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        1e-5,
    ).numpy()
    tol = tolerance(2e-5, 1e-5)
    for name, array, dtype, prov in [
        ("layernorm.input", x, "f32", tensor_provenance(kind="input", call="x [2,3,5]", tolerance_block=tol, extra={"contains_zero_variance_row": True})),
        ("layernorm.weight", weight, "f32", tensor_provenance(kind="input", call="weight [5]", tolerance_block=tol)),
        ("layernorm.bias", bias, "f32", tensor_provenance(kind="input", call="bias [5]", tolerance_block=tol)),
        ("layernorm.out", ln, "f32", tensor_provenance(kind="reference", call="LayerNorm(x, weight, bias, eps=1e-5)", tolerance_block=tol)),
    ]:
        suite.add(name, array, dtype, provenance=prov)
    cases.append(
        {
            "operator": "layernorm",
            "shape": [2, 3, 5],
            "epsilon": 1e-5,
            "features": ["zero-variance row", "odd hidden size 5"],
            "self_check_max_abs": max_abs(ln, torch_ln),
        }
    )

    # Model-shaped LayerNorm (hidden size 128, sequence 4).
    x_model = rand_array((1, 4, 128), -2.0, 2.0, 0x5150)
    w_model = rand_array((128,), 0.75, 1.25, 0x5151)
    b_model = rand_array((128,), -0.1, 0.1, 0x5152)
    ln_model = as_f32(layernorm64(x_model, w_model.astype(np.float64), b_model.astype(np.float64), 1e-5))
    for name, array, dtype, prov in [
        ("layernorm.model_shape.input", x_model, "f32", tensor_provenance(kind="input", call="x [1,4,128]", tolerance_block=tol)),
        ("layernorm.model_shape.weight", w_model, "f32", tensor_provenance(kind="input", call="weight [128]", tolerance_block=tol)),
        ("layernorm.model_shape.bias", b_model, "f32", tensor_provenance(kind="input", call="bias [128]", tolerance_block=tol)),
        ("layernorm.model_shape.out", ln_model, "f32", tensor_provenance(kind="reference", call="LayerNorm(x, weight, bias, eps=1e-5)", tolerance_block=tol)),
    ]:
        suite.add(name, array, dtype, provenance=prov)
    cases.append(
        {
            "operator": "layernorm",
            "shape": [1, 4, 128],
            "epsilon": 1e-5,
            "features": ["GPT-2 hidden size"],
            "self_check_max_abs": max_abs(
                ln_model,
                F.layer_norm(torch.from_numpy(x_model), (128,), torch.from_numpy(w_model), torch.from_numpy(b_model), 1e-5).numpy(),
            ),
        }
    )

    # -- MatMul and Gemm ---------------------------------------------------
    def add_matmul(
        label: str,
        a: np.ndarray,
        b: np.ndarray,
        *,
        bias: np.ndarray | None = None,
        atol: float,
    ) -> None:
        a64 = a.astype(np.float64)
        b64 = b.astype(np.float64)
        if bias is None:
            out64 = a64 @ b64
            call = f"MatMul A{list(a.shape)} @ B{list(b.shape)}"
            formula = "A @ B"
            inputs = [("a", a, "f32"), ("b", b, "f32")]
            torch_out = torch.matmul(torch.from_numpy(a), torch.from_numpy(b)).numpy()
        else:
            out64 = a64 @ b64.T + bias.astype(np.float64)
            call = f"Gemm A{list(a.shape)} @ W{list(b.shape)}.T + b{list(bias.shape)} (transB=1)"
            formula = "A @ W.T + b"
            inputs = [("a", a, "f32"), ("w", b, "f32"), ("bias", bias, "f32")]
            torch_out = (
                torch.matmul(torch.from_numpy(a), torch.from_numpy(b).T)
                + torch.from_numpy(bias)
            ).numpy()
        out = as_f32(out64)
        tol = tolerance(atol, 1e-5)
        for suffix, array, dtype in inputs:
            suite.add(
                f"matmul.{label}.{suffix}",
                array,
                dtype,
                provenance=tensor_provenance(kind="input", call=formula, tolerance_block=tol),
            )
        suite.add(
            f"matmul.{label}.out",
            out,
            "f32",
            provenance=tensor_provenance(kind="reference", call=call, tolerance_block=tol),
        )
        cases.append(
            {
                "operator": "gemm" if bias is not None else "matmul",
                "case": label,
                "shapes": {"a": list(a.shape), "b": list(b.shape), "out": list(out.shape)},
                "bias": bias is not None,
                "self_check_max_abs": max_abs(out, torch_out),
            }
        )

    add_matmul("2d_k3", rand_array((2, 3), -2, 2, 0xA001), rand_array((3, 4), -2, 2, 0xA002), atol=5e-6)
    add_matmul(
        "gemm_k13_bias",
        rand_array((7, 13), -2, 2, 0xA003),
        rand_array((5, 13), -2, 2, 0xA004),
        bias=rand_array((5,), -1, 1, 0xA005),
        atol=5e-6,
    )
    add_matmul(
        "batched_4d",
        rand_array((2, 3, 5, 7), -2, 2, 0xA006),
        rand_array((2, 3, 7, 11), -2, 2, 0xA007),
        atol=1e-5,
    )
    add_matmul(
        "model_proj",
        rand_array((1, 13, 128), -2, 2, 0xA008),
        rand_array((128, 8), -1, 1, 0xA009),
        atol=2e-5,
    )
    add_matmul("tail_k65", rand_array((3, 65), -2, 2, 0xA00A), rand_array((65, 7), -2, 2, 0xA00B), atol=2e-5)

    # -- GELU --------------------------------------------------------------
    gelu_input = np.concatenate(
        [np.linspace(-6.0, 6.0, 65, dtype=np.float32), np.asarray([-1e4, -10.0, 10.0, 1e4], dtype=np.float32)]
    )
    gelu_erf = as_f32(gelu_erf64(gelu_input))
    gelu_tanh = as_f32(gelu_tanh64(gelu_input))
    torch_erf = F.gelu(torch.from_numpy(gelu_input), approximate="none").numpy()
    torch_tanh = F.gelu(torch.from_numpy(gelu_input), approximate="tanh").numpy()
    tol = tolerance(2e-6, 1e-6)
    suite.add("gelu.input", gelu_input, "f32", provenance=tensor_provenance(kind="input", call="x [69]", tolerance_block=tol))
    for name, out, label, call, self_check in [
        ("gelu.erf.out", gelu_erf, "erf", "0.5*x*(1+erf(x/sqrt(2)))", torch_erf),
        ("gelu.tanh.out", gelu_tanh, "tanh", "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))", torch_tanh),
    ]:
        suite.add(
            name,
            out,
            "f32",
            provenance=tensor_provenance(kind="reference", call=call, tolerance_block=tol),
        )
        cases.append(
            {
                "operator": "gelu",
                "case": label,
                "shape": [69],
                "model_uses": label == "tanh",
                "self_check_max_abs": max_abs(out, self_check),
            }
        )

    # -- Softmax -----------------------------------------------------------
    def add_softmax_case(label: str, x: np.ndarray, mask: np.ndarray | None) -> None:
        scores = x.astype(np.float64) + (mask.astype(np.float64) if mask is not None else 0.0)
        out = as_f32(softmax64(scores))
        torch_scores = torch.from_numpy(scores.astype(np.float32))
        torch_out = torch.softmax(torch_scores, dim=-1).numpy()
        tol = tolerance(2e-6, 1e-6)
        suite.add(
            f"softmax.{label}.input",
            x,
            "f32",
            provenance=tensor_provenance(kind="input", call="logits", tolerance_block=tol),
        )
        if mask is not None:
            suite.add(
                f"softmax.{label}.mask",
                mask,
                "f32",
                provenance=tensor_provenance(
                    kind="input",
                    call="additive mask",
                    tolerance_block=tol,
                    extra={"fill_value": MASK_FILL, "all_masked_rows": int(np.count_nonzero(np.all(mask == MASK_FILL, axis=-1)))},
                ),
            )
        suite.add(
            f"softmax.{label}.out",
            out,
            "f32",
            provenance=tensor_provenance(kind="reference", call="Softmax(additive(logits, mask), axis=-1)", tolerance_block=tol),
        )
        cases.append(
            {
                "operator": "softmax",
                "case": label,
                "shape": list(x.shape),
                "masked": mask is not None,
                "self_check_max_abs": max_abs(out, torch_out),
            }
        )

    add_softmax_case("unmasked", rand_array((2, 3, 4, 5), -5.0, 5.0, 0x5001), None)
    causal = np.triu(np.full((1, 1, 4, 4), MASK_FILL, dtype=np.float32), k=1)
    add_softmax_case("causal", rand_array((1, 2, 4, 4), -5.0, 5.0, 0x5002), np.broadcast_to(causal, (1, 2, 4, 4)).copy())
    all_masked = np.zeros((1, 1, 2, 4), dtype=np.float32)
    all_masked[0, 0, 1, :] = MASK_FILL
    add_softmax_case("all_masked", rand_array((1, 1, 2, 4), -2.0, 2.0, 0x5003), all_masked)
    padded = np.zeros((1, 1, 3, 5), dtype=np.float32)
    padded[0, 0, :, 3:] = MASK_FILL
    add_softmax_case("padded", rand_array((1, 1, 3, 5), -4.0, 4.0, 0x5004), padded)

    # -- Masked multi-head attention --------------------------------------
    q = rand_array((2, 2, 4, 5), -1.0, 1.0, 0xA770)
    k = rand_array((2, 2, 4, 5), -1.0, 1.0, 0xA771)
    v = rand_array((2, 2, 4, 5), -1.0, 1.0, 0xA772)
    mask = np.tile(np.triu(np.full((1, 1, 4, 4), MASK_FILL, dtype=np.float32), k=1), (2, 1, 1, 1))
    # Sample 1, query row 2 is deliberately fully masked (for example a fully
    # padded query) to pin the all-masked policy in a realistic attention case.
    mask[1, 0, 2, :] = MASK_FILL
    head_dim = 5
    scores64 = (q.astype(np.float64) @ k.astype(np.float64).transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    probs64 = softmax64(scores64 + mask.astype(np.float64))
    out64 = probs64 @ v.astype(np.float64)
    probs = as_f32(probs64)
    attention_out = as_f32(out64)
    torch_scores = torch.matmul(torch.from_numpy(q), torch.from_numpy(k).transpose(-1, -2)) / math.sqrt(head_dim)
    torch_probs = torch.softmax(torch_scores + torch.from_numpy(mask), dim=-1)
    torch_out = torch.matmul(torch_probs, torch.from_numpy(v)).numpy()
    tol = tolerance(3e-6, 1e-6)
    for name, array, dtype, prov in [
        ("attention.q", q, "f32", tensor_provenance(kind="input", call="Q [B=2,H=2,S=4,D=5]", tolerance_block=tol)),
        ("attention.k", k, "f32", tensor_provenance(kind="input", call="K [B=2,H=2,S=4,D=5]", tolerance_block=tol)),
        ("attention.v", v, "f32", tensor_provenance(kind="input", call="V [B=2,H=2,S=4,D=5]", tolerance_block=tol)),
        (
            "attention.mask",
            mask,
            "f32",
            tensor_provenance(
                kind="input",
                call="additive causal + all-masked mask",
                tolerance_block=tol,
                extra={
                    "all_masked": {"batch": 1, "query": 2, "keys": 4},
                    "fill_value": MASK_FILL,
                },
            ),
        ),
        ("attention.probs", probs, "f32", tensor_provenance(kind="reference", call="Softmax(Q @ K.T / sqrt(5) + mask)", tolerance_block=tol)),
        ("attention.out", attention_out, "f32", tensor_provenance(kind="reference", call="probs @ V", tolerance_block=tol)),
    ]:
        suite.add(name, array, dtype, provenance=prov)
    all_masked_probs = probs[1, :, 2, :]
    cases.append(
        {
            "operator": "masked_multi_head_attention",
            "shape": {"q": [2, 2, 4, 5], "k": [2, 2, 4, 5], "v": [2, 2, 4, 5], "out": [2, 2, 4, 5]},
            "features": ["causal mask", "head dim 5", "all-masked query row"],
            "all_masked_row_probs": all_masked_probs.tolist(),
            "all_masked_row_uniform": bool(np.allclose(all_masked_probs, 0.25, atol=0.0, rtol=0.0)),
            "self_check_max_abs": max(max_abs(probs, torch_probs.numpy()), max_abs(attention_out, torch_out)),
        }
    )

    suite.extra["cases"] = cases
    for tensor in suite.tensors:
        tensor["provenance"].setdefault(
            "tool_versions",
            {"torch": torch.__version__, "numpy": np.__version__},
        )
    suite.write()
    return {
        "suite": "ops",
        "tensors": len(suite.tensors),
        "stored": sum(1 for tensor in suite.tensors if tensor["stored"]),
    }


# ---------------------------------------------------------------------------
# ONNX Runtime helpers


def make_session(path: Path, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def empty_past(layers: int, heads: int, head_dim: int) -> dict[str, np.ndarray]:
    return {
        f"past_key_values.{layer}.{kind}": np.zeros((1, heads, 0, head_dim), dtype=np.float32)
        for layer in range(layers)
        for kind in ("key", "value")
    }


def ort_forward(
    session: ort.InferenceSession,
    *,
    token_ids: list[int],
    past: dict[str, np.ndarray],
    position_ids: list[int],
) -> dict[str, np.ndarray]:
    past_length = next(iter(past.values())).shape[2] if past else 0
    input_ids = np.asarray([token_ids], dtype=np.int64)
    attention_mask = np.ones((1, past_length + len(token_ids)), dtype=np.int64)
    feeds = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": np.asarray([position_ids], dtype=np.int64),
        **past,
    }
    output_names = [output.name for output in session.get_outputs()]
    return dict(zip(output_names, session.run(None, feeds), strict=True))


def present_to_past(outputs: dict[str, np.ndarray], layers: int) -> dict[str, np.ndarray]:
    return {
        f"past_key_values.{layer}.{kind}": outputs[f"present.{layer}.{kind}"]
        for layer in range(layers)
        for kind in ("key", "value")
    }


def validate_graph(path: Path, expected: dict[str, Any], *, n_layers: int) -> None:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    opset = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if opset != [expected["opset"]]:
        raise SystemExit(f"{path}: opset {opset} != pinned {expected['opset']}")
    if len(graph.node) != expected["nodes"]:
        raise SystemExit(f"{path}: {len(graph.node)} nodes != pinned {expected['nodes']}")
    dtype_counts = {1: 0, 7: 0}
    for initializer in graph.initializer:
        if initializer.data_type in dtype_counts:
            dtype_counts[initializer.data_type] += 1
    if dtype_counts[1] != expected["initializers"]["f32"]:
        raise SystemExit(
            f"{path}: {dtype_counts[1]} f32 initializers != pinned "
            f"{expected['initializers']['f32']}"
        )
    if dtype_counts[7] != expected["initializers"]["i64"]:
        raise SystemExit(
            f"{path}: {dtype_counts[7]} i64 initializers != pinned "
            f"{expected['initializers']['i64']}"
        )
    input_names = {value.name for value in graph.input}
    output_names = {value.name for value in graph.output}
    needed_inputs = {"input_ids", "attention_mask", "position_ids"} | {
        f"past_key_values.{layer}.{kind}" for layer in range(n_layers) for kind in ("key", "value")
    }
    needed_outputs = {"logits"} | {
        f"present.{layer}.{kind}" for layer in range(n_layers) for kind in ("key", "value")
    }
    if not needed_inputs <= input_names:
        raise SystemExit(f"{path}: missing inputs {sorted(needed_inputs - input_names)}")
    if not needed_outputs <= output_names:
        raise SystemExit(f"{path}: missing outputs {sorted(needed_outputs - output_names)}")


def load_torch_model(path: Path) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(str(path), torch_dtype=torch.float32)
    model.eval()
    return model


def torch_forward(
    model: GPT2LMHeadModel,
    *,
    token_ids: list[int],
    position_ids: list[int],
    cache: Any | None,
) -> tuple[torch.Tensor, Any]:
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones((1, len(token_ids) + (cache.get_seq_length() if cache is not None else 0)), dtype=torch.long)
    position = torch.tensor([position_ids], dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
        )
    return output.logits, output.past_key_values


def torch_cache_arrays(cache: Any, layers: int) -> dict[str, np.ndarray]:
    arrays = {}
    for layer in range(layers):
        arrays[f"{layer}.key"] = cache.layers[layer].keys.numpy()
        arrays[f"{layer}.value"] = cache.layers[layer].values.numpy()
    return arrays


# ---------------------------------------------------------------------------
# tiny GPT-2 whole-model fixtures


def generate_tiny(root: Path, cache: Path, environment: dict[str, Any]) -> dict[str, Any]:
    model_dir = cache / "tiny-gpt2"
    hf_dir = cache / "tiny-gpt2-hf"
    graph_path = model_dir / "onnx" / "model.onnx"
    if not graph_path.is_file():
        raise SystemExit(f"missing {graph_path}; run tools/parity/fetch_models.py --suite tiny-gpt2")
    validate_graph(graph_path, TINY_GRAPH, n_layers=2)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoded = tokenizer.encode(PROMPT_TEXT)
    if encoded != TOKENS:
        raise SystemExit(f"pinned tokenizer changed the fixture token ids: {encoded}")
    if tokenizer.vocab_size != TINY_TOKENIZER["vocab_size"]:
        raise SystemExit("pinned tokenizer vocab size changed")

    torch_model = load_torch_model(hf_dir)
    torch.set_num_threads(1)
    session = make_session(graph_path, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    default_session = make_session(graph_path, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

    reset_suite_dir(root)
    suite = Suite(
        root,
        "tiny-gpt2",
        description=(
            "Whole-model F32 reference for the 2-layer, 2-head, head-dim-64 "
            "randomly-initialised GPT-2 ONNX graph.  Prefill at lengths 1/4/8/13, "
            "four cached decode steps, and cached-vs-full parity at every step."
        ),
        provenance={
            "model_id": TINY_ID,
            "revision": TINY_REVISION,
            "base_checkpoint": {"model_id": TINY_BASE_ID, "revision": TINY_BASE_REVISION},
            "graph": {
                "path": "onnx/model.onnx",
                "sha256": sha256_file(graph_path),
                "bytes": graph_path.stat().st_size,
                "geometry": TINY_GRAPH,
            },
            "tokenizer": TINY_TOKENIZER,
            "sequence": {
                "text": PROMPT_TEXT,
                "token_ids": TOKENS,
                "prefill_lengths": list(PROMPT_LENGTHS),
                "decode_token_ids": DECODE_TOKENS,
                "note": (
                    "TOKENS is tokenizer.encode(text). Prefill lengths are prefixes; "
                    "decode tokens are TOKENS[13:17], so each cached step is compared "
                    "with a full recomputation of the same prefix."
                ),
            },
            "producer": {
                "library": "onnxruntime",
                "version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "graph_optimization": "ORT_DISABLE_ALL",
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL",
            },
            "cross_check": {
                "library": "pytorch",
                "version": torch.__version__,
                "model_class": "transformers.GPT2LMHeadModel",
                "weights": "pinned base checkpoint model.safetensors",
            },
            "environment": environment,
            "tolerance_policy": {
                "elementwise": "abs(actual - ref) <= atol + rtol * abs(ref)",
                "model_logits": tolerance(1e-5, 1e-4),
                "kv_tensors": tolerance(1e-5, 1e-4),
                "sequence_scaling": (
                    "For S > 16 multiply atol by sqrt(S/16); rtol is unchanged. "
                    "The tiny-model fixtures stop at S=17, so the unscaled bound applies "
                    "to every committed tensor."
                ),
                "mean_abs": 2.5e-6,
                "failure": [
                    "shape or dtype mismatch",
                    "NaN or Inf where the reference is finite",
                    "any element outside the elementwise bound",
                    "mean_abs above the stated bound (systematic bias)",
                    "present-cache extent/shape mismatch",
                ],
                "argmax": (
                    "A different argmax is a failure only when the reference top-two "
                    "gap exceeds 2 * atol; ties inside the noise floor are not gated."
                ),
            },
        },
    )

    sequence = suite.extra.setdefault("sequence", {})
    sequence.update(
        {
            "text": PROMPT_TEXT,
            "token_ids": TOKENS,
            "prefill_lengths": list(PROMPT_LENGTHS),
            "decode_token_ids": DECODE_TOKENS,
        }
    )
    prefill_records = []
    pytorch_prefills: dict[int, Any] = {}
    noise_floor = {"onnxruntime_optimized_vs_disabled": [], "pytorch_vs_onnxruntime": [], "cached_vs_full": []}

    for length in PROMPT_LENGTHS:
        ids = TOKENS[:length]
        positions = list(range(length))
        outputs = ort_forward(
            session,
            token_ids=ids,
            past=empty_past(2, 2, 64),
            position_ids=positions,
        )
        logits = outputs["logits"]
        if logits.shape != (1, length, 50257):
            raise SystemExit(f"unexpected prefill logits shape {logits.shape} for length {length}")
        full_entry = suite.add(
            f"prefill.p{length}.logits",
            logits,
            "f32",
            stored=(length == 1),
            slice_axis=1,
            provenance=tensor_provenance(
                kind="model_output",
                call=f"prefill length {length}",
                tolerance_block=tolerance(1e-5, 1e-4),
                extra={
                    "tokens": ids,
                    "position_ids": positions,
                    "past_length": 0,
                    "dtype": "float32",
                    "producer": "onnxruntime",
                },
            ),
        )
        if length in (4, 13):
            suite.add(
                f"prefill.p{length}.last_logits",
                logits[:, -1:, :],
                "f32",
                provenance=tensor_provenance(
                    kind="model_output",
                    call=f"prefill length {length}, last position",
                    tolerance_block=tolerance(1e-5, 1e-4),
                    extra={
                        "tokens": ids,
                        "position_ids": positions,
                        "past_length": 0,
                        "derived_from": f"prefill.p{length}.logits[0,{length - 1},:]",
                        "dtype": "float32",
                        "producer": "onnxruntime",
                    },
                ),
            )

        present = {}
        for layer in range(2):
            for kind in ("key", "value"):
                name = f"present.{layer}.{kind}"
                array = outputs[name]
                suite.add(
                    f"prefill.p{length}.{name}",
                    array,
                    "f32",
                    provenance=tensor_provenance(
                        kind="cache_output",
                        call=f"prefill length {length}",
                        tolerance_block=tolerance(1e-5, 1e-4),
                        extra={
                            "layer": layer,
                            "kind": kind,
                            "tokens": ids,
                            "past_length": 0,
                            "cache_extent": int(array.shape[2]),
                            "dtype": "float32",
                            "producer": "onnxruntime",
                        },
                    ),
                )
                present[name] = {
                    "tensor": f"prefill.p{length}.{name}",
                    "shape": [int(value) for value in array.shape],
                    "cache_extent": int(array.shape[2]),
                }

        # Optimised ORT session cross-check (same graph, different kernel plan).
        optimized = ort_forward(
            default_session,
            token_ids=ids,
            past=empty_past(2, 2, 64),
            position_ids=positions,
        )
        noise_floor["onnxruntime_optimized_vs_disabled"].append(
            {"length": length, "logits_max_abs": max_abs(logits, optimized["logits"])}
        )

        # Independent PyTorch prefill.
        torch_logits, torch_cache = torch_forward(
            torch_model, token_ids=ids, position_ids=positions, cache=None
        )
        torch_arrays = torch_cache_arrays(torch_cache, 2)
        kv_error = max(
            max_abs(outputs[f"present.{layer}.{kind}"], torch_arrays[f"{layer}.{kind}"])
            for layer in range(2)
            for kind in ("key", "value")
        )
        logits_error = max_abs(logits, torch_logits.numpy())
        noise_floor["pytorch_vs_onnxruntime"].append(
            {"length": length, "logits_max_abs": logits_error, "kv_max_abs": kv_error}
        )
        pytorch_prefills[length] = torch_cache
        prefill_records.append(
            {
                "length": length,
                "logits_tensor": f"prefill.p{length}.logits",
                "logits_stored": length == 1,
                "logits_sha256": full_entry["sha256"],
                "present": present,
                "pytorch_cross_check": {
                    "logits_max_abs": logits_error,
                    "logits_mean_abs": mean_abs(logits, torch_logits.numpy()),
                    "kv_max_abs": kv_error,
                },
            }
        )
        print(
            f"prefill {length:2d}: logits {logits.shape} sha {full_entry['sha256'][:12]} "
            f"torch {logits_error:.3e}"
        )

    # -- cached decode -----------------------------------------------------
    past = present_to_past(
        ort_forward(session, token_ids=TOKENS[:13], past=empty_past(2, 2, 64), position_ids=list(range(13))),
        2,
    )
    decode_records = []
    torch_cache = pytorch_prefills[13]
    for step, token in enumerate(DECODE_TOKENS, start=1):
        total_length = 13 + step
        position = 13 + step - 1
        outputs = ort_forward(session, token_ids=[token], past=past, position_ids=[position])
        logits = outputs["logits"]
        if logits.shape != (1, 1, 50257):
            raise SystemExit(f"unexpected decode logits shape {logits.shape} at step {step}")
        suite.add(
            f"decode.step{step}.logits",
            logits,
            "f32",
            stored=(step == 1),
            provenance=tensor_provenance(
                kind="model_output",
                call=f"cached decode step {step}",
                tolerance_block=tolerance(1e-5, 1e-4),
                extra={
                    "input_token": token,
                    "tokens": TOKENS[:total_length],
                    "position_ids": [position],
                    "past_length": 13 + step - 1,
                    "total_length": total_length,
                    "dtype": "float32",
                    "producer": "onnxruntime",
                },
            ),
        )

        present = {}
        for layer in range(2):
            for kind in ("key", "value"):
                name = f"present.{layer}.{kind}"
                array = outputs[name]
                if array.shape != (1, 2, total_length, 64):
                    raise SystemExit(f"bad cache extent {array.shape} for {name} at step {step}")
                suite.add(
                    f"decode.step{step}.{name}",
                    array,
                    "f32",
                    provenance=tensor_provenance(
                        kind="cache_output",
                        call=f"cached decode step {step}",
                        tolerance_block=tolerance(1e-5, 1e-4),
                        extra={
                            "layer": layer,
                            "kind": kind,
                            "input_token": token,
                            "past_length": total_length - 1,
                            "cache_extent": total_length,
                            "dtype": "float32",
                            "producer": "onnxruntime",
                        },
                    ),
                )
                present[name] = {
                    "tensor": f"decode.step{step}.{name}",
                    "shape": [int(value) for value in array.shape],
                    "cache_extent": int(array.shape[2]),
                }
        cache_extent = outputs["present.0.key"].shape[2]
        if cache_extent != total_length:
            raise SystemExit(f"present cache extent {cache_extent} != {total_length}")

        # Full recomputation of the same prefix for cached-vs-full parity.
        full = ort_forward(
            session,
            token_ids=TOKENS[:total_length],
            past=empty_past(2, 2, 64),
            position_ids=list(range(total_length)),
        )
        suite.add(
            f"full.p{total_length}.logits",
            full["logits"],
            "f32",
            stored=False,
            slice_axis=1,
            provenance=tensor_provenance(
                kind="model_output",
                call=f"full recompute length {total_length}",
                tolerance_block=tolerance(1e-5, 1e-4),
                extra={
                    "tokens": TOKENS[:total_length],
                    "position_ids": list(range(total_length)),
                    "past_length": 0,
                    "dtype": "float32",
                    "producer": "onnxruntime",
                },
            ),
        )
        cached_last = logits[0, 0]
        full_last = full["logits"][0, -1]
        cached_full = {
            "max_abs": max_abs(cached_last, full_last),
            "mean_abs": mean_abs(cached_last, full_last),
            "full_logits_tensor": f"full.p{total_length}.logits",
        }
        noise_floor["cached_vs_full"].append({"step": step, "total_length": total_length, **cached_full})
        if step == 1:
            prefix_error = max_abs(full["logits"][:, :13, :], ort_forward(
                session, token_ids=TOKENS[:13], past=empty_past(2, 2, 64), position_ids=list(range(13))
            )["logits"])
            cached_full["prefix_vs_prefill_max_abs"] = prefix_error

        # Independent PyTorch cached step and full recomputation.
        torch_logits, torch_cache = torch_forward(
            torch_model,
            token_ids=[token],
            position_ids=[position],
            cache=torch_cache,
        )
        torch_arrays = torch_cache_arrays(torch_cache, 2)
        torch_kv_error = max(
            max_abs(outputs[f"present.{layer}.{kind}"], torch_arrays[f"{layer}.{kind}"])
            for layer in range(2)
            for kind in ("key", "value")
        )
        torch_logits_error = max_abs(logits, torch_logits.numpy())
        torch_full_logits, _ = torch_forward(
            torch_model,
            token_ids=TOKENS[:total_length],
            position_ids=list(range(total_length)),
            cache=None,
        )
        torch_cached_full = max_abs(
            torch_logits[0, -1].numpy(), torch_full_logits[0, -1].numpy()
        )
        noise_floor["pytorch_vs_onnxruntime"].append(
            {
                "step": step,
                "total_length": total_length,
                "logits_max_abs": torch_logits_error,
                "kv_max_abs": torch_kv_error,
            }
        )

        past = present_to_past(outputs, 2)
        decode_records.append(
            {
                "step": step,
                "input_token": token,
                "past_length": 13 + step - 1,
                "total_length": total_length,
                "cache_extent": cache_extent,
                "logits_tensor": f"decode.step{step}.logits",
                "logits_stored": step == 1,
                "present": present,
                "cached_vs_full": cached_full,
                "pytorch_cross_check": {
                    "logits_max_abs": torch_logits_error,
                    "kv_max_abs": torch_kv_error,
                    "cached_vs_full_max_abs": torch_cached_full,
                },
            }
        )
        print(
            f"decode  {step:2d}: cache {cache_extent:2d} cached-vs-full {cached_full['max_abs']:.3e} "
            f"torch {torch_logits_error:.3e}"
        )

    suite.extra["prefill"] = prefill_records
    suite.extra["decode"] = decode_records
    suite.extra["noise_floor"] = noise_floor
    for tensor in suite.tensors:
        tensor["provenance"].setdefault("model_id", TINY_ID)
        tensor["provenance"].setdefault("model_revision", TINY_REVISION)
        tensor["provenance"].setdefault(
            "tool_versions",
            {
                "onnxruntime": ort.__version__,
                "onnx": onnx.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        )
    suite.write()
    write_json(root / "model.lock.json", tiny_lock(cache))

    return {
        "suite": "tiny-gpt2",
        "tensors": len(suite.tensors),
        "stored": sum(1 for tensor in suite.tensors if tensor["stored"]),
        "bytes": sum(tensor["bytes"] for tensor in suite.tensors if tensor["stored"]),
        "max_cached_vs_full": max(item["max_abs"] for item in noise_floor["cached_vs_full"]),
        "max_pytorch_logits": max(item["logits_max_abs"] for item in noise_floor["pytorch_vs_onnxruntime"]),
    }


# ---------------------------------------------------------------------------
# trained GPT-2 fixtures (opt-in)


def generate_trained(root: Path, cache: Path, environment: dict[str, Any]) -> dict[str, Any]:
    model_dir = cache / "trained-gpt2"
    graph_path = model_dir / "onnx" / "model.onnx"
    if not graph_path.is_file():
        raise SystemExit(
            "missing trained model; run tools/parity/fetch_models.py --suite trained-gpt2 "
            "(~498 MB) or omit --include-trained"
        )
    validate_graph(graph_path, TRAINED_GRAPH, n_layers=12)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoded = tokenizer.encode(PROMPT_TEXT)
    if encoded != TOKENS:
        raise SystemExit(f"trained tokenizer changed the fixture token ids: {encoded}")
    session = make_session(graph_path, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

    reset_suite_dir(root)
    suite = Suite(
        root,
        "trained-gpt2",
        description=(
            "Opt-in whole-model reference for the real 124M-parameter GPT-2 ONNX "
            "graph.  Not a CI dependency: the graph is 498 MB and is fetched "
            "explicitly.  Provides a stable prefill and cached-decode sample for "
            "demos and benchmarking."
        ),
        provenance={
            "model_id": TRAINED_ID,
            "revision": TRAINED_REVISION,
            "graph": {
                "path": "onnx/model.onnx",
                "sha256": sha256_file(graph_path),
                "bytes": graph_path.stat().st_size,
                "geometry": TRAINED_GRAPH,
            },
            "tokenizer": TRAINED_TOKENIZER,
            "sequence": {
                "text": PROMPT_TEXT,
                "token_ids": TOKENS,
                "prefill_length": 13,
                "decode_token_ids": DECODE_TOKENS[:2],
            },
            "producer": {
                "library": "onnxruntime",
                "version": ort.__version__,
                "provider": "CPUExecutionProvider",
                "graph_optimization": "ORT_DISABLE_ALL",
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "execution_mode": "ORT_SEQUENTIAL",
            },
            "environment": environment,
            "tolerance_policy": {
                "elementwise": "abs(actual - ref) <= atol + rtol * abs(ref)",
                "model_logits": tolerance(5e-4, 1e-4),
                "kv_tensors": tolerance(1e-4, 1e-4),
                "sequence_scaling": (
                    "For S > 16 multiply atol by sqrt(S/16); rtol is unchanged."
                ),
                "mean_abs": 5e-5,
                "failure": [
                    "shape or dtype mismatch",
                    "NaN or Inf where the reference is finite",
                    "any element outside the elementwise bound",
                    "mean_abs above the stated bound",
                    "present-cache extent/shape mismatch",
                ],
            },
            "ci": False,
            "ci_reason": (
                "498,126,358-byte graph and the ONNX repository does not repeat the "
                "base GPT-2 MIT declaration; demos fetch it explicitly."
            ),
        },
    )
    suite.extra["sequence"] = {
        "text": PROMPT_TEXT,
        "token_ids": TOKENS,
        "prefill_length": 13,
        "decode_token_ids": DECODE_TOKENS[:2],
    }

    prefill = ort_forward(
        session,
        token_ids=TOKENS[:13],
        past=empty_past(12, 12, 64),
        position_ids=list(range(13)),
    )
    suite.add(
        "prefill.p13.logits",
        prefill["logits"],
        "f32",
        stored=False,
        slice_axis=1,
        provenance=tensor_provenance(
            kind="model_output",
            call="prefill length 13",
            tolerance_block=tolerance(5e-4, 1e-4),
            extra={
                "tokens": TOKENS[:13],
                "position_ids": list(range(13)),
                "past_length": 0,
                "dtype": "float32",
                "producer": "onnxruntime",
            },
        ),
    )
    suite.add(
        "prefill.p13.last_logits",
        prefill["logits"][:, -1:, :],
        "f32",
        provenance=tensor_provenance(
            kind="model_output",
            call="prefill length 13, last position",
            tolerance_block=tolerance(5e-4, 1e-4),
            extra={
                "tokens": TOKENS[:13],
                "position_ids": list(range(13)),
                "past_length": 0,
                "derived_from": "prefill.p13.logits[0,12,:]",
                "dtype": "float32",
                "producer": "onnxruntime",
            },
        ),
    )
    prefill_present = {}
    for layer in range(12):
        for kind in ("key", "value"):
            name = f"present.{layer}.{kind}"
            array = prefill[name]
            suite.add(
                f"prefill.p13.{name}",
                array,
                "f32",
                stored=False,
                provenance=tensor_provenance(
                    kind="cache_output",
                    call="prefill length 13",
                    tolerance_block=tolerance(1e-4, 1e-4),
                    extra={
                        "layer": layer,
                        "kind": kind,
                        "past_length": 0,
                        "cache_extent": int(array.shape[2]),
                        "dtype": "float32",
                        "producer": "onnxruntime",
                    },
                ),
            )
            prefill_present[name] = {"tensor": f"prefill.p13.{name}", "shape": [int(v) for v in array.shape]}

    past = present_to_past(prefill, 12)
    decode_records = []
    for step, token in enumerate(DECODE_TOKENS[:2], start=1):
        total_length = 13 + step
        outputs = ort_forward(session, token_ids=[token], past=past, position_ids=[13 + step - 1])
        logits = outputs["logits"]
        suite.add(
            f"decode.step{step}.logits",
            logits,
            "f32",
            stored=(step == 1),
            provenance=tensor_provenance(
                kind="model_output",
                call=f"cached decode step {step}",
                tolerance_block=tolerance(5e-4, 1e-4),
                extra={
                    "input_token": token,
                    "tokens": TOKENS[:total_length],
                    "position_ids": [13 + step - 1],
                    "past_length": total_length - 1,
                    "total_length": total_length,
                    "dtype": "float32",
                    "producer": "onnxruntime",
                },
            ),
        )
        present = {}
        for layer in range(12):
            for kind in ("key", "value"):
                name = f"present.{layer}.{kind}"
                array = outputs[name]
                if array.shape != (1, 12, total_length, 64):
                    raise SystemExit(f"bad trained cache extent {array.shape}")
                suite.add(
                    f"decode.step{step}.{name}",
                    array,
                    "f32",
                    stored=False,
                    provenance=tensor_provenance(
                        kind="cache_output",
                        call=f"cached decode step {step}",
                        tolerance_block=tolerance(1e-4, 1e-4),
                        extra={
                            "layer": layer,
                            "kind": kind,
                            "past_length": total_length - 1,
                            "cache_extent": total_length,
                            "dtype": "float32",
                            "producer": "onnxruntime",
                        },
                    ),
                )
                present[name] = {
                    "tensor": f"decode.step{step}.{name}",
                    "shape": [int(v) for v in array.shape],
                }
        full = ort_forward(
            session,
            token_ids=TOKENS[:total_length],
            past=empty_past(12, 12, 64),
            position_ids=list(range(total_length)),
        )
        cached_full = {
            "max_abs": max_abs(outputs["logits"][0, 0], full["logits"][0, -1]),
            "mean_abs": mean_abs(outputs["logits"][0, 0], full["logits"][0, -1]),
        }
        decode_records.append(
            {
                "step": step,
                "input_token": token,
                "past_length": total_length - 1,
                "total_length": total_length,
                "cache_extent": int(outputs["present.0.key"].shape[2]),
                "logits_tensor": f"decode.step{step}.logits",
                "present": present,
                "cached_vs_full": cached_full,
            }
        )
        past = present_to_past(outputs, 12)
        print(
            f"trained decode {step}: cache {total_length} cached-vs-full {cached_full['max_abs']:.3e}"
        )

    suite.extra["prefill"] = {
        "length": 13,
        "logits_tensor": "prefill.p13.logits",
        "last_logits_tensor": "prefill.p13.last_logits",
        "present": prefill_present,
    }
    suite.extra["decode"] = decode_records
    for tensor in suite.tensors:
        tensor["provenance"].setdefault("model_id", TRAINED_ID)
        tensor["provenance"].setdefault("model_revision", TRAINED_REVISION)
        tensor["provenance"].setdefault(
            "tool_versions",
            {
                "onnxruntime": ort.__version__,
                "onnx": onnx.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        )
    suite.write()
    write_json(root / "model.lock.json", trained_lock(cache))
    return {
        "suite": "trained-gpt2",
        "tensors": len(suite.tensors),
        "stored": sum(1 for tensor in suite.tensors if tensor["stored"]),
        "bytes": sum(tensor["bytes"] for tensor in suite.tensors if tensor["stored"]),
        "max_cached_vs_full": max(item["cached_vs_full"]["max_abs"] for item in decode_records),
    }


# ---------------------------------------------------------------------------
# entry point


def environment_block() -> dict[str, Any]:
    cpu_model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--suites",
        choices=["ci", "all", "ops", "tiny-gpt2", "trained-gpt2"],
        default="ci",
        help="'ci' = ops + tiny-gpt2; 'all' also regenerates trained-gpt2",
    )
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    environment = environment_block()
    print(json.dumps({"output": str(out), "cache": str(cache), "environment": environment}, sort_keys=True))
    results = []
    if args.suites in ("ci", "all", "ops"):
        results.append(generate_ops(out / "ops", environment))
    if args.suites in ("ci", "all", "tiny-gpt2"):
        results.append(generate_tiny(out / "tiny-gpt2", cache, environment))
    if args.suites in ("all", "trained-gpt2"):
        results.append(generate_trained(out / "trained-gpt2", cache, environment))
    print(json.dumps({"generated": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
