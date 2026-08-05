#!/usr/bin/env python3
"""Rank GPT-style Gemm projections by activation-aware grouped-I4 output error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--tokens", type=int, nargs="+", default=[42])
    parser.add_argument("--validation-tokens", type=int, nargs="+", default=[50, 51, 52])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state_names() -> list[str]:
    return [f"past_key_values.{layer}.{kind}" for layer in range(12) for kind in ("key", "value")]


def feeds(tokens: list[int]) -> dict[str, np.ndarray]:
    token_array = np.asarray([tokens], dtype=np.int64)
    values = {
        "input_ids": token_array,
        "attention_mask": np.ones_like(token_array),
    }
    values.update({name: np.zeros((1, 12, 0, 64), dtype=np.float32) for name in state_names()})
    return values


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def grouped_i4_dequantize(weight_kn: np.ndarray, group_size: int) -> np.ndarray:
    """Replicate fastnn's guarded asymmetric signed-I4 K-group quantizer."""
    if weight_kn.ndim != 2:
        raise ValueError(f"expected [K,N] weight, got {weight_kn.shape}")
    k, n = weight_kn.shape
    dequantized = np.empty_like(weight_kn, dtype=np.float32)
    for column in range(n):
        for start in range(0, k, group_size):
            end = min(start + group_size, k)
            group = weight_kn[start:end, column].astype(np.float32)
            minimum = float(np.min(group))
            maximum = float(np.max(group))
            value_range = maximum - minimum
            if value_range == 0.0:
                dequantized[start:end, column] = minimum
                continue
            initial_scale = value_range / 15.0
            low = minimum - initial_scale
            high = maximum + initial_scale
            scale = (high - low) / 15.0
            offset = low + 8.0 * scale
            codes = np.clip(np.rint((group - offset) / scale), -8.0, 7.0)
            dequantized[start:end, column] = codes * scale + offset
    return dequantized


def activation_weighted_grouped_i4_dequantize(
    weight_kn: np.ndarray,
    activation_statistics: np.ndarray,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose group clipping using diagonal importance or full calibration activations."""
    if activation_statistics.ndim == 1:
        if activation_statistics.shape != (weight_kn.shape[0],):
            raise ValueError("channel importance must have one value per K channel")
    elif activation_statistics.ndim == 2:
        if activation_statistics.shape[1] != weight_kn.shape[0]:
            raise ValueError("activation matrix K dimension must match the weight")
    else:
        raise ValueError("activation statistics must be [K] or [tokens,K]")
    k, n = weight_kn.shape
    dequantized = np.empty_like(weight_kn, dtype=np.float32)
    selected_ratios = np.empty((n, (k + group_size - 1) // group_size), dtype=np.float32)
    clip_ratios = np.linspace(0.70, 1.0, 31, dtype=np.float32)
    for column in range(n):
        for start in range(0, k, group_size):
            end = min(start + group_size, k)
            group = weight_kn[start:end, column].astype(np.float32)
            importance = None
            activation_group = None
            if activation_statistics.ndim == 1:
                importance = activation_statistics[start:end].astype(np.float64)
            else:
                activation_group = activation_statistics[:, start:end].astype(np.float64)
            minimum = float(np.min(group))
            maximum = float(np.max(group))
            if maximum == minimum:
                dequantized[start:end, column] = minimum
                selected_ratios[column, start // group_size] = 1.0
                continue
            midpoint = 0.5 * (minimum + maximum)
            half_range = 0.5 * (maximum - minimum)
            best_score = float("inf")
            best_values = None
            best_ratio = None
            for ratio in clip_ratios:
                clipped_low = midpoint - half_range * float(ratio)
                clipped_high = midpoint + half_range * float(ratio)
                initial_scale = (clipped_high - clipped_low) / 15.0
                low = clipped_low - initial_scale
                high = clipped_high + initial_scale
                scale = (high - low) / 15.0
                offset = low + 8.0 * scale
                codes = np.clip(np.rint((group - offset) / scale), -8.0, 7.0)
                candidate = codes * scale + offset
                delta = candidate.astype(np.float64) - group.astype(np.float64)
                if activation_group is not None:
                    projected_error = activation_group @ delta
                    score = float(np.mean(projected_error * projected_error))
                else:
                    assert importance is not None
                    score = float(np.sum(importance * delta * delta))
                if score < best_score:
                    best_score = score
                    best_values = candidate
                    best_ratio = ratio
            assert best_values is not None
            assert best_ratio is not None
            dequantized[start:end, column] = best_values
            selected_ratios[column, start // group_size] = best_ratio
    return dequantized, selected_ratios


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    rmse = float(np.sqrt(np.mean(delta * delta)))
    reference_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    return {
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "normalized_rmse": rmse / max(reference_rms, 1e-12),
    }


def main() -> int:
    args = parse_args()
    model = onnx.load(str(args.onnx), load_external_data=True)
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    projections = []
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        attrs = attributes(node)
        if int(attrs.get("transA", 0)) != 0:
            raise RuntimeError(f"unsupported transA=1 in {node.name}")
        raw_weight = np.asarray(initializers[node.input[1]], dtype=np.float32)
        weight_kn = raw_weight.T if int(attrs.get("transB", 0)) else raw_weight
        if weight_kn.ndim != 2:
            continue
        projections.append((node, weight_kn, float(attrs.get("alpha", 1.0))))
    if not projections:
        raise RuntimeError("model has no constant-weight Gemm projections")

    activation_names = list(dict.fromkeys(node.input[0] for node, _, _ in projections))
    existing_outputs = {value.name for value in model.graph.output}
    known = {
        value.name: value
        for value in (*model.graph.value_info, *model.graph.output, *model.graph.input)
    }
    for name in activation_names:
        if name not in existing_outputs:
            model.graph.output.append(known.get(name, helper.make_empty_tensor_value_info(name)))
    augmented = args.output.with_suffix(".onnx")
    augmented.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(augmented))
    session = ort.InferenceSession(str(augmented), providers=["CPUExecutionProvider"])
    def collect(tokens: list[int]) -> dict[str, np.ndarray]:
        state_pairs = [
            (f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}")
            for layer in range(12)
            for kind in ("key", "value")
        ]
        state = {
            input_name: np.zeros((1, 12, 0, 64), dtype=np.float32)
            for input_name, _ in state_pairs
        }
        requested = activation_names + [output_name for _, output_name in state_pairs]
        activation_runs = []
        for step, token in enumerate(tokens, start=1):
            step_feeds = {
                "input_ids": np.asarray([[token]], dtype=np.int64),
                "attention_mask": np.ones((1, step), dtype=np.int64),
                **state,
            }
            values = session.run(requested, step_feeds)
            activation_runs.append(values[: len(activation_names)])
            state = {
                input_name: values[len(activation_names) + index]
                for index, (input_name, _) in enumerate(state_pairs)
            }
        return {
            name: np.concatenate([run[index] for run in activation_runs], axis=0)
            for index, name in enumerate(activation_names)
        }

    activations = collect(args.tokens)
    validation_activations = collect(args.validation_tokens)

    layers = []
    for node, weight_kn, alpha in projections:
        activation = np.asarray(activations[node.input[0]], dtype=np.float32)
        flat = activation.reshape(-1, activation.shape[-1])
        validation_activation = np.asarray(
            validation_activations[node.input[0]], dtype=np.float32
        )
        validation_flat = validation_activation.reshape(
            -1, validation_activation.shape[-1]
        )
        if flat.shape[1] != weight_kn.shape[0]:
            raise RuntimeError(
                f"{node.name}: activation K={flat.shape[1]} does not match weight K={weight_kn.shape[0]}"
            )
        quantized_weight = grouped_i4_dequantize(weight_kn, args.group_size)
        optimized_weight, clip_ratios = activation_weighted_grouped_i4_dequantize(
            weight_kn,
            flat,
            args.group_size,
        )
        reference = alpha * (flat @ weight_kn)
        actual = alpha * (flat @ quantized_weight)
        optimized = alpha * (flat @ optimized_weight)
        validation_reference = alpha * (validation_flat @ weight_kn)
        validation_actual = alpha * (validation_flat @ quantized_weight)
        validation_optimized = alpha * (validation_flat @ optimized_weight)
        layer_metrics = metrics(actual, reference)
        optimized_layer_metrics = metrics(optimized, reference)
        weight_metrics = metrics(quantized_weight, weight_kn)
        optimized_weight_metrics = metrics(optimized_weight, weight_kn)
        validation_layer_metrics = metrics(validation_actual, validation_reference)
        optimized_validation_layer_metrics = metrics(
            validation_optimized, validation_reference
        )
        channel_rms = np.sqrt(np.mean(flat.astype(np.float64) ** 2, axis=0))
        layers.append(
            {
                "name": node.name or node.output[0],
                "activation": node.input[0],
                "weight": node.input[1],
                "shape": {"tokens": int(flat.shape[0]), "k": int(weight_kn.shape[0]), "n": int(weight_kn.shape[1])},
                "activation_rms": float(np.sqrt(np.mean(flat.astype(np.float64) ** 2))),
                "activation_channel_rms_max": float(np.max(channel_rms)),
                "weight_error": weight_metrics,
                "activation_weighted_output_error": layer_metrics,
                "optimized_weight_error": optimized_weight_metrics,
                "optimized_activation_weighted_output_error": optimized_layer_metrics,
                "validation_output_error": validation_layer_metrics,
                "optimized_validation_output_error": optimized_validation_layer_metrics,
                "clip_ratios": clip_ratios.reshape(-1).tolist(),
            }
        )
    layers.sort(
        key=lambda layer: layer["activation_weighted_output_error"]["normalized_rmse"],
        reverse=True,
    )
    report = {
        "group_size": args.group_size,
        "tokens": args.tokens,
        "validation_tokens": args.validation_tokens,
        "projection_count": len(layers),
        "worst": layers[0],
        "layers": layers,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    worst_summary = {
        key: value for key, value in layers[0].items() if key != "clip_ratios"
    }
    print(
        json.dumps(
            {
                "group_size": args.group_size,
                "projection_count": len(layers),
                "worst": worst_summary,
            },
            indent=2,
        )
    )
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
