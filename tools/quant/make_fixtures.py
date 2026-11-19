#!/usr/bin/env python3
"""Generate the committed ``fixtures/quant`` blobs and their manifest.

Run from the repository root:

    python3 tools/quant/make_fixtures.py

The output is deterministic.  ``tools/quant/verify_fixtures.py`` re-decodes
every blob with the independent Python reference; the Rust test
``tests/fn3_quant_format.rs`` re-decodes the same bytes and compares the values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

import fn3_quant as fq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "fixtures" / "quant"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fix_crcs(blob: bytearray) -> None:
    """Recompute both checksums so a semantic mutation is the next failure."""

    (
        _magic,
        _version,
        _header,
        _flags,
        _count,
        _dbytes,
        _gbytes,
        _doff,
        _goff,
        payload_offset,
        payload_bytes,
        _payload_crc,
        _meta_crc,
    ) = fq.HEADER.unpack_from(blob, 0)
    struct.pack_into("<I", blob, 56, zlib.crc32(blob[payload_offset : payload_offset + payload_bytes]) & 0xFFFFFFFF)
    struct.pack_into("<I", blob, 60, 0)
    struct.pack_into("<I", blob, 60, zlib.crc32(blob[:payload_offset]) & 0xFFFFFFFF)


def fix_payload_crc(blob: bytearray) -> None:
    (_, _, _, _, _, _, _, _, _, payload_offset, payload_bytes, _, _) = fq.HEADER.unpack_from(blob, 0)
    struct.pack_into(
        "<I",
        blob,
        56,
        zlib.crc32(blob[payload_offset : payload_offset + payload_bytes]) & 0xFFFFFFFF,
    )


def descriptor_offset(index: int = 0) -> int:
    return fq.HEADER_BYTES + index * fq.DESCRIPTOR_BYTES


def group_offset(blob: bytes, index: int = 0) -> int:
    (_, _, _, _, _, _, _, _, goff, _, _, _, _) = fq.HEADER.unpack_from(blob, 0)
    return goff + index * fq.GROUP_ENTRY_BYTES


def payload_offset(blob: bytes) -> int:
    return fq.HEADER.unpack_from(blob, 0)[9]


def tensor_entry(name: str, codes, scales, zeros) -> fq.TensorEntry:
    codes = np.asarray(codes, dtype=np.int8)
    return fq.TensorEntry(
        name=name,
        codes=codes,
        scales=np.asarray(scales, dtype=np.float16).reshape(
            1 if codes.ndim == 1 else codes.shape[0], -1
        ),
        zeros=np.asarray(zeros, dtype=np.int16).reshape(
            codes.shape[0], (codes.shape[1] + fq.GROUP_SIZE - 1) // fq.GROUP_SIZE
        ),
    )


def valid_fixtures() -> list[tuple[str, str, bytes]]:
    fixtures = []

    # 1. Two rows, one group each.  Row 0 walks the whole code range; row 1 is
    #    constant with a non-zero zero-point.
    codes = np.zeros((2, 64), dtype=np.int8)
    codes[0] = np.arange(64, dtype=np.int16) % 16 - 8
    codes[1] = -4
    blob = fq.encode_blob([tensor_entry("basic", codes, [0.5, 1.0], [0, -4])])
    fixtures.append(("basic", "two rows; full code range and a non-zero zero-point", blob))

    # 2. K = 70: a full group followed by a six-element tail group, and 10
    #    bytes of row-alignment padding.
    codes = np.zeros((1, 70), dtype=np.int8)
    codes[0, :64] = np.arange(64, dtype=np.int16) - 8
    codes[0, 64:] = [10, 20, 30, 40, 50, 60]
    blob = fq.encode_blob(
        [tensor_entry("tail_k70", codes, [0.25, 0.125], [0, 30])]
    )
    fixtures.append(("tail_k70", "K=70: full group plus a six-element tail", blob))

    # 3. A non-zero zero-point shifts every code so zero is not code 0.
    codes = np.zeros((1, 64), dtype=np.int8)
    codes[0] = np.arange(64, dtype=np.int16) - 32
    blob = fq.encode_blob([tensor_entry("zero_point", codes, [0.5], [-10])])
    fixtures.append(("zero_point", "non-zero integer zero-point", blob))

    # 4. Two tensors in one container, including K=8 (a group shorter than 64).
    short = np.arange(8, dtype=np.int8) + 1
    square = np.arange(64, dtype=np.int16).reshape(1, 64) % 16 - 8
    blob = fq.encode_blob(
        [
            tensor_entry("short_k8", short.reshape(1, 8), [1.0], [4]),
            tensor_entry("square", square, [0.25], [-8]),
        ]
    )
    fixtures.append(("two_tensors", "two tensors; one group shorter than 64", blob))
    return fixtures


def malformed_fixtures(valid: list[tuple[str, str, bytes]]) -> list[tuple[str, str, bytes]]:
    by_name = {name: blob for name, _description, blob in valid}
    base = bytearray(by_name["tail_k70"])
    two = bytearray(by_name["two_tensors"])
    out: list[tuple[str, str, bytes]] = []

    def emit(name: str, description: str, blob: bytearray, fix: str = "meta") -> None:
        if fix == "meta":
            fix_crcs(blob)
        elif fix == "payload":
            # The payload checksum lives in the metadata, so both change.
            fix_crcs(blob)
        out.append((name, description, bytes(blob)))

    # Header and framing.
    blob = bytearray(base)
    blob[0:4] = b"XXXX"
    emit("bad_magic", "magic is not FN3Q", blob, fix="none")
    blob = bytearray(base)
    struct.pack_into("<H", blob, 4, 2)
    emit("unsupported_version", "format version 2", blob, fix="none")
    blob = bytearray(base)
    struct.pack_into("<H", blob, 6, 32)
    emit("bad_header_size", "header_bytes is 32", blob, fix="none")
    blob = bytearray(base)
    struct.pack_into("<I", blob, 8, 1)
    emit("unsupported_flags", "flags bit 0 is set", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, 12, 0)
    emit("bad_tensor_count", "tensor_count is zero", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, 16, 0)
    emit("bad_descriptor_bytes", "descriptor_bytes is zero", blob)
    blob = bytearray(base)
    struct.pack_into("<Q", blob, 24, 0)
    emit("bad_descriptor_offset", "descriptor_offset is zero", blob)
    blob = bytearray(base)
    struct.pack_into("<Q", blob, 32, struct.unpack_from("<Q", blob, 32)[0] + 8)
    emit("bad_group_offset", "group_offset is not descriptor end", blob)
    blob = bytearray(base)
    struct.pack_into("<Q", blob, 40, struct.unpack_from("<Q", blob, 40)[0] + 16)
    emit("bad_payload_offset", "payload_offset is not canonical", blob)

    # Descriptor semantics.  DESCRIPTOR is <32s II HH III Q.
    offset = descriptor_offset()
    blob = bytearray(base)
    struct.pack_into("<H", blob, offset + 40, 4)
    emit("unsupported_qbits", "qbits is 4", blob)
    blob = bytearray(base)
    struct.pack_into("<H", blob, offset + 42, 32)
    emit("unsupported_group_size", "group_size is 32", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, offset + 44, 1)
    emit("bad_group_count", "groups_per_row is 1 for K=70", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, offset + 48, 1)
    emit("bad_group_index", "group_dir_index is not cumulative", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, offset + 52, 80 + 16)
    emit("bad_row_stride", "row_stride_bytes is 96 for K=70", blob)
    blob = bytearray(base)
    struct.pack_into("<Q", blob, offset + 56, struct.unpack_from("<Q", blob, offset + 56)[0] + 1)
    emit("misaligned_offset", "data_offset is not 16-byte aligned", blob)
    blob = bytearray(base)
    struct.pack_into("<Q", blob, offset + 56, struct.unpack_from("<Q", blob, offset + 56)[0] + 16)
    emit("bad_data_offset", "data_offset is aligned but not canonical", blob)
    blob = bytearray(base)
    blob[offset : offset + fq.NAME_BYTES] = b"A" * fq.NAME_BYTES
    emit("bad_name", "name is not NUL-terminated", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, offset + 32, 0)
    emit("zero_extent", "n_rows is zero", blob)
    blob = bytearray(two)
    struct.pack_into(
        "<32s",
        blob,
        descriptor_offset(1),
        struct.unpack_from("<32s", blob, descriptor_offset(0))[0],
    )
    emit("duplicate_name", "two tensors share a name", blob)

    # Group directory.
    blob = bytearray(base)
    first = group_offset(blob, 0)
    struct.pack_into("<I", blob, first + 8, 32)
    emit("group_gap_or_overlap", "the tail group starts inside the first", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, first + 8, 66)
    emit("group_gap_or_overlap_gap", "the tail group starts after a gap", blob)
    blob = bytearray(base)
    struct.pack_into("<I", blob, first + 12, 0)
    emit("bad_group_length", "the tail group has length zero", blob)

    # Payload contents.
    start = payload_offset(base)
    blob = bytearray(base)
    struct.pack_into("<H", blob, start + 80, 0)  # first scale of row 0
    emit("scale_zero", "a scale is zero", blob, fix="payload")
    blob = bytearray(base)
    struct.pack_into("<H", blob, start + 80, 0x7E00)  # f16 NaN
    emit("scale_nan", "a scale is NaN", blob, fix="payload")
    blob = bytearray(base)
    struct.pack_into("<H", blob, start + 80, 0xBC00)  # f16 -1.0
    emit("scale_negative", "a scale is negative", blob, fix="payload")
    blob = bytearray(base)
    struct.pack_into("<h", blob, start + 80 + 2 * 2, 200)  # first zero point
    emit("zero_point_out_of_range", "a zero-point is 200", blob, fix="payload")
    blob = bytearray(base)
    blob[start + 70] = 1  # first byte after K in row 0
    emit("nonzero_tail_padding", "row alignment padding is non-zero", blob, fix="payload")

    # Checksums and file framing.
    blob = bytearray(base[:-1])
    emit("trailing_or_truncated_bytes", "one byte is missing", blob, fix="none")
    blob = bytearray(base)
    blob[-1] ^= 0xFF
    emit("payload_checksum", "payload byte changed without a checksum update", blob, fix="none")
    blob = bytearray(base)
    blob[descriptor_offset()] ^= 0x01
    emit("meta_checksum", "descriptor byte changed without a checksum update", blob, fix="none")
    blob = bytearray(base)
    blob += b"\0"
    emit("trailing_bytes", "one byte is appended", blob, fix="none")
    return out


def tensor_manifest(entry: fq.ParsedTensor) -> dict:
    packed_row = entry.row_stride_bytes
    start = entry.data_offset
    packed = bytes(
        entry.codes.reshape(-1)
        .astype(np.int8)
        .tobytes()
    )
    # Row padding is not part of codes; reconstruct it for the hex dump.
    payload = bytearray()
    for row in range(entry.n_rows):
        payload += entry.codes[row].astype(np.int8).tobytes()
        payload += b"\0" * (packed_row - entry.k)
    return {
        "name": entry.name,
        "n_rows": entry.n_rows,
        "k": entry.k,
        "qbits": entry.qbits,
        "group_size": entry.group_size,
        "groups_per_row": entry.groups_per_row,
        "row_stride_bytes": entry.row_stride_bytes,
        "data_offset": entry.data_offset,
        "scales": [float(value) for value in entry.scales.reshape(-1)],
        "zero_points": [int(value) for value in entry.zeros.reshape(-1)],
        "packed_hex": bytes(payload).hex(),
        "expected": [float(value) for value in entry.dequantize().reshape(-1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v1").mkdir(exist_ok=True)
    (args.out / "malformed").mkdir(exist_ok=True)

    valid = valid_fixtures()
    manifest = {
        "format": fq.FORMAT_ID,
        "format_version": fq.VERSION,
        "python_reference": "tools/quant/fn3_quant.py",
        "scalar_types": "signed INT8 weights, IEEE binary16 scale, signed int16 zero-point",
        "group_size": fq.GROUP_SIZE,
        "data_alignment": fq.DATA_ALIGNMENT,
        "fixtures": [],
        "malformed": [],
    }
    for name, description, blob in valid:
        relative = f"v1/{name}.fn3q"
        (args.out / relative).write_bytes(blob)
        tensors = fq.decode_blob(blob)
        manifest["fixtures"].append(
            {
                "path": relative,
                "bytes": len(blob),
                "sha256": sha256(blob),
                "description": description,
                "tensors": [tensor_manifest(tensor) for tensor in tensors],
            }
        )

    for name, description, blob in malformed_fixtures(valid):
        relative = f"malformed/{name}.fn3q"
        (args.out / relative).write_bytes(blob)
        try:
            fq.decode_blob(blob)
        except fq.QuantFormatError as error:
            kind = error.kind
        else:
            raise AssertionError(f"malformed fixture {name} decoded successfully")
        manifest["malformed"].append(
            {
                "path": relative,
                "bytes": len(blob),
                "sha256": sha256(blob),
                "expected_error": kind,
                "description": description,
            }
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {len(manifest['fixtures'])} fixture blobs and "
        f"{len(manifest['malformed'])} malformed blobs to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
