#!/usr/bin/env python3
"""Independently verify the real Rust grouped-W4A8 path.

Reads the files produced by ``export_w4a8_cases.py`` and the ignored Rust test
``rust_grouped_w4a8_matches_the_independent_reference``, then:

1. unpacks the Rust ``u32`` words from scratch and compares the codes with this
   repository's independent Python encoder;
2. recomputes the dequantized weights and the exact W4A8 matmul, and compares
   them with the Rust outputs;
3. reports weight and matmul error against the original FP32 tensors.

Usage:
    FN3_W4A8_DIR=/tmp/opencode/fn3-68/w4a8 python3 tools/quant/check_rust_w4a8.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import fn3_quant as fq


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def read_u32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u4")


def read_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<i4")


def unpack_words(words: np.ndarray, n: int, k: int) -> np.ndarray:
    words_per_row = (k + 7) // 8
    assert words.size == n * words_per_row
    words = words.reshape(n, words_per_row).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, None, :]
    nibbles = (words[:, :, None] >> shifts) & 0xF
    signed = np.where(nibbles >= 8, nibbles.astype(np.int16) - 16, nibbles).astype(np.int8)
    return signed.reshape(n, words_per_row * 8)[:, :k]


def exact_grouped_matmul(
    act_codes: np.ndarray,
    act_scales: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """The 2.x kernel: integer dot, integer correction, one f32 affine."""

    m, k = act_codes.shape
    n = codes.shape[0]
    out = np.zeros((m, n), dtype=np.float32)
    for g, (start, end) in enumerate(fq.group_ranges(k, group_size)):
        a = act_codes[:, start:end].astype(np.float64)
        w = codes[:, start:end].astype(np.float64)
        dot = a @ w.T
        sum_a = a.sum(axis=1)[:, None]
        s_w = scales[:, g].astype(np.float64)[None, :]
        z_w = zeros[:, g].astype(np.float64)[None, :]
        s_a = act_scales.astype(np.float64)[:, None]
        out += (dot * s_w * s_a + z_w * s_a * sum_a).astype(np.float32)
    return out


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    scale = np.linalg.norm(reference.astype(np.float64))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_frobenius": float(np.linalg.norm(delta) / max(scale, 1e-30)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    directory = args.dir or Path(__import__("os").environ["FN3_W4A8_DIR"])
    manifest = json.loads((directory / "cases.json").read_text())
    rows = []
    for case in manifest["cases"]:
        name = case["name"]
        n, k, m = case["n"], case["k"], case["m"]
        weight = read_f32(directory / case["weight"]).reshape(n, k)
        activation = read_f32(directory / case["activation"]).reshape(m, k)
        for group in (32, 64, 128):
            prefix = f"{name}.g{group}"
            words = read_u32(directory / f"{prefix}.packed.u32")
            rust_scales = read_f32(directory / f"{prefix}.scales.f32").reshape(n, -1)
            rust_zeros = read_f32(directory / f"{prefix}.zeros.f32").reshape(n, -1)
            rust_dequant = read_f32(directory / f"{prefix}.dequant.f32").reshape(n, k)
            rust_sums = read_i32(directory / f"{prefix}.sums.i32").reshape(n, -1)
            rust_act_scales = read_f32(directory / f"{prefix}.act_scales.f32")
            rust_act_codes = np.fromfile(
                directory / f"{prefix}.act_codes.u8", dtype=np.uint8
            ).reshape(m, k).view(np.int8)
            rust_matmul = read_f32(directory / f"{prefix}.matmul.f32").reshape(m, n)

            codes = unpack_words(words, n, k)
            groups = (k + group - 1) // group
            py_codes, py_scales, py_zeros = fq.quantize_int4_existing_w4a8(weight, group)
            sums = np.zeros((n, groups), dtype=np.int64)
            for g, (start, end) in enumerate(fq.group_ranges(k, group)):
                sums[:, g] = codes[:, start:end].astype(np.int64).sum(axis=1)
            py_act_scale = np.maximum(np.abs(activation).max(axis=1) / 127.0, 1e-30)
            py_act_codes = np.clip(
                fq.round_half_away(activation / py_act_scale[:, None]), -128, 127
            ).astype(np.int8)
            # The Rust float-zero path needs the float zero, not the int zero.
            rust_dequant_ref = np.zeros((n, k), dtype=np.float32)
            for g, (start, end) in enumerate(fq.group_ranges(k, group)):
                rust_dequant_ref[:, start:end] = (
                    codes[:, start:end].astype(np.float32)
                    * rust_scales[:, g].astype(np.float32)[:, None]
                    + rust_zeros[:, g].astype(np.float32)[:, None]
                )
            python_matmul = exact_grouped_matmul(
                rust_act_codes,
                rust_act_scales,
                codes,
                rust_scales,
                rust_zeros,
                group,
            )
            fp32_matmul = activation @ weight.T
            dequant_matmul = activation @ rust_dequant_ref.T
            code_delta = codes.astype(np.int16) - py_codes.astype(np.int16)
            rows.append(
                {
                    "name": name,
                    "group_size": group,
                    "n": n,
                    "k": k,
                    "m": m,
                    "packed_bytes": int(words.size * 4),
                    "metadata_bytes": int(rust_scales.size * 4 + rust_zeros.size * 4 + rust_sums.size * 4),
                    "code_differences": int(np.count_nonzero(code_delta)),
                    "max_code_delta": int(np.max(np.abs(code_delta))),
                    "codes_within_one": bool(np.max(np.abs(code_delta)) <= 1),
                    "scale_max_abs_diff": float(
                        np.max(np.abs(rust_scales - py_scales.astype(np.float32)))
                    ),
                    "zero_max_abs_diff": float(np.max(np.abs(rust_zeros - py_zeros))),
                    "group_sums_match": bool(np.array_equal(sums, rust_sums.astype(np.int64))),
                    "rust_dequant_vs_independent": metrics(rust_dequant, rust_dequant_ref),
                    "dequant_vs_fp32": metrics(rust_dequant_ref, weight),
                    "activation_codes_match": bool(
                        np.array_equal(rust_act_codes, py_act_codes)
                    ),
                    "act_scale_max_abs_diff": float(
                        np.max(np.abs(rust_act_scales - py_act_scale.astype(np.float32)))
                    ),
                    "rust_matmul_vs_independent": metrics(rust_matmul, python_matmul),
                    "matmul_vs_fp32_weight": metrics(rust_matmul, fp32_matmul),
                    "matmul_vs_dequant_weight": metrics(rust_matmul, dequant_matmul),
                }
            )
            row = rows[-1]
            print(
                f"{name:20s} g{group:<3d} relfro {row['dequant_vs_fp32']['relative_frobenius']:.4f} "
                f"rust-vs-py dequant {row['rust_dequant_vs_independent']['max_abs']:.2e} "
                f"matmul {row['rust_matmul_vs_independent']['max_abs']:.2e} "
                f"code diffs {row['code_differences']}/{codes.size} sums {row['group_sums_match']}"
            )

    output = args.out or directory / "rust_w4a8.json"
    output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
