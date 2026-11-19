#!/usr/bin/env python3
"""Offline verifier for ``fixtures/quant`` (standard library + NumPy only).

    python3 tools/quant/verify_fixtures.py --root fixtures/quant

It checks every byte length, SHA-256, descriptor field, scale, zero-point,
packed hex string and expected dequantized value against the independent Python
decoder, and asserts the typed error for every malformed blob.  It also proves
that no payload file is unreferenced.

Pass ``--regenerate`` to rebuild the fixtures into a temporary directory and
diff them against the checkout byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import fn3_quant as fq

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_fixture(root: Path, entry: dict, errors: list[str]) -> None:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"missing {entry['path']}")
        return
    blob = path.read_bytes()
    if len(blob) != entry["bytes"]:
        errors.append(f"{entry['path']}: bytes {len(blob)} != {entry['bytes']}")
    if sha256(blob) != entry["sha256"]:
        errors.append(f"{entry['path']}: sha256 mismatch")
    try:
        tensors = fq.decode_blob(blob)
    except fq.QuantFormatError as error:
        errors.append(f"{entry['path']}: valid blob rejected: {error}")
        return
    declared = {tensor["name"]: tensor for tensor in entry["tensors"]}
    if set(declared) != {tensor.name for tensor in tensors}:
        errors.append(f"{entry['path']}: tensor names")
        return
    for tensor in tensors:
        want = declared[tensor.name]
        for field in (
            "n_rows",
            "k",
            "qbits",
            "group_size",
            "groups_per_row",
            "row_stride_bytes",
            "data_offset",
        ):
            if getattr(tensor, field) != want[field]:
                errors.append(f"{entry['path']}:{tensor.name}: {field}")
        if not np.array_equal(
            tensor.scales.reshape(-1).astype(np.float32),
            np.asarray(want["scales"], dtype=np.float32),
        ):
            errors.append(f"{entry['path']}:{tensor.name}: scales")
        if not np.array_equal(
            tensor.zeros.reshape(-1).astype(np.int16),
            np.asarray(want["zero_points"], dtype=np.int16),
        ):
            errors.append(f"{entry['path']}:{tensor.name}: zero points")
        payload = bytearray()
        for row in range(tensor.n_rows):
            payload += tensor.codes[row].astype(np.int8).tobytes()
            payload += b"\0" * (tensor.row_stride_bytes - tensor.k)
        if payload.hex() != want["packed_hex"]:
            errors.append(f"{entry['path']}:{tensor.name}: packed hex")
        expected = np.asarray(want["expected"], dtype=np.float32)
        actual = tensor.dequantize().reshape(-1)
        if not np.array_equal(actual, expected):
            errors.append(f"{entry['path']}:{tensor.name}: dequantized values")


def check_malformed(root: Path, entry: dict, errors: list[str]) -> None:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"missing {entry['path']}")
        return
    blob = path.read_bytes()
    if len(blob) != entry["bytes"]:
        errors.append(f"{entry['path']}: bytes")
    if sha256(blob) != entry["sha256"]:
        errors.append(f"{entry['path']}: sha256 mismatch")
    try:
        fq.decode_blob(blob)
    except fq.QuantFormatError as error:
        if error.kind != entry["expected_error"]:
            errors.append(
                f"{entry['path']}: error {error.kind} != {entry['expected_error']}"
            )
    else:
        errors.append(f"{entry['path']}: malformed blob decoded successfully")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "fixtures" / "quant")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    errors: list[str] = []
    if manifest.get("format") != fq.FORMAT_ID:
        errors.append("format id")
    if manifest.get("format_version") != fq.VERSION:
        errors.append("format version")
    for entry in manifest["fixtures"]:
        check_fixture(root, entry, errors)
    for entry in manifest["malformed"]:
        check_malformed(root, entry, errors)

    referenced = {entry["path"] for entry in manifest["fixtures"] + manifest["malformed"]}
    for path in sorted(root.rglob("*.fn3q")):
        relative = path.relative_to(root).as_posix()
        if relative not in referenced:
            errors.append(f"unreferenced payload {relative}")

    if args.regenerate:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "quant"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "quant" / "make_fixtures.py"),
                    "--out",
                    str(output),
                ],
                check=True,
                cwd=REPO_ROOT,
            )
            for relative in sorted(referenced | {"manifest.json"}):
                regenerated = output / relative
                committed = root / relative
                if not regenerated.is_file() or regenerated.read_bytes() != committed.read_bytes():
                    errors.append(f"regenerated {relative} differs")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        f"OK: {len(manifest['fixtures'])} fixtures, "
        f"{len(manifest['malformed'])} malformed cases, "
        f"{sum(entry['bytes'] for entry in manifest['fixtures'])} fixture bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
