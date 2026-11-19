#!/usr/bin/env python3
"""Cross-check the Rust packer against the independent Python encoder.

`tests/fn3_vulkan_quant.rs::encoder_cross_check_dump` writes, for one real
`c_attn` weight, the source `[N, K]` values and the sections
`kernels::quant::encode_projection` produced (codes, scales, zero-points).
`tools/quant/fn3_quant.py` implements the same reference policy from scratch, so
this compares the two byte for byte rather than grading the Rust packer against
itself.

    FN3_QUANT_DUMP_DIR=/tmp/opencode/fn3-30-dump \\
      cargo test --features vulkan --test fn3_vulkan_quant encoder_cross_check_dump -- --nocapture
    python3 docs/reports/fn3-30/cross_check_encoder.py /tmp/opencode/fn3-30-dump
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools" / "quant"))

import fn3_quant  # noqa: E402


def main(directory: pathlib.Path) -> int:
    cases = sorted(directory.glob("*.shape.txt"))
    if not cases:
        print(f"no dumped cases under {directory}", file=sys.stderr)
        return 2
    failures = 0
    for shape_path in cases:
        stem = shape_path.name[: -len(".shape.txt")]
        n, k, groups = (int(value) for value in shape_path.read_text().split())
        base = directory / stem
        source = np.fromfile(base.with_suffix(".source.f32"), dtype="<f4").reshape(n, k)
        codes = np.fromfile(base.with_suffix(".codes.bin"), dtype=np.int8)
        scales = np.fromfile(base.with_suffix(".scales.bin"), dtype="<f2")
        zeros = np.fromfile(base.with_suffix(".zeros.bin"), dtype="<i2")

        expected_codes, expected_scales, expected_zeros = fn3_quant.quantize_int8_grouped(
            source
        )
        # The payload pads each code row to a multiple of 16 bytes.
        stride = (k + 15) // 16 * 16
        packed = np.zeros((n, stride), dtype=np.int8)
        packed[:, :k] = expected_codes
        same_codes = np.array_equal(codes.reshape(n, stride), packed)
        same_scales = np.array_equal(scales.reshape(n, groups), expected_scales)
        same_zeros = np.array_equal(zeros.reshape(n, groups), expected_zeros)
        code_differences = int(np.count_nonzero(codes.reshape(n, stride) != packed))
        scale_differences = int(np.count_nonzero(scales.reshape(n, groups) != expected_scales))
        zero_differences = int(np.count_nonzero(zeros.reshape(n, groups) != expected_zeros))

        ok = same_codes and same_scales and same_zeros
        failures += 0 if ok else 1
        print(
            f"{stem}: N={n} K={k} groups={groups} "
            f"codes={'match' if same_codes else f'{code_differences} differ'} "
            f"scales={'match' if same_scales else f'{scale_differences} differ'} "
            f"zeros={'match' if same_zeros else f'{zero_differences} differ'}"
        )
    print(f"{len(cases) - failures}/{len(cases)} cases byte-identical")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/fn3-30-dump")
    raise SystemExit(main(root))
