#!/usr/bin/env python3
"""Independent reference for ``fn3-quant-format-v1``.

Standard library plus NumPy only.  Nothing here imports FastNN: this file is
the second implementation that the Rust decoder in ``src/fn3/quant/`` is checked
against, and it is also the encoder used to produce ``fixtures/quant/``.

Format summary (see ``docs/quantization/fn3-quant-format-v1.md`` for the
authoritative prose):

* one 64-byte header, followed by fixed 64-byte tensor descriptors, an 8-byte
  group directory, then a 16-byte-aligned payload;
* signed INT4 weights, eight nibbles per little-endian ``u32``, low nibble is
  the lower element index;
* groups of 64 elements along K, each with an IEEE binary16 scale and a signed
  ``i8`` zero-point in ``[-8, 7]``;
* ``w = (q - zero_point) * scale``; the integer zero-point is the code that
  represents 0.0;
* per tensor payload order is ``packed data | scales | zero points``.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"FN3Q"
VERSION = 1
HEADER_BYTES = 64
DESCRIPTOR_BYTES = 64
GROUP_ENTRY_BYTES = 8
GROUP_SIZE = 64
QUBITS = 8
NAME_BYTES = 32
DATA_ALIGNMENT = 16
FORMAT_ID = "fn3-quant-format-v1"

HEADER = struct.Struct("<4sHHIIIIQQQQII")
DESCRIPTOR = struct.Struct("<32sIIHHIIIQ")
GROUP_ENTRY = struct.Struct("<II")


class QuantFormatError(ValueError):
    """The blob is not a valid ``fn3-quant-format-v1`` object."""

    def __init__(self, kind: str, detail: str, **context: object) -> None:
        self.kind = kind
        self.detail = detail
        self.context = context
        super().__init__(f"{kind}: {detail}")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def round_half_away(values: np.ndarray) -> np.ndarray:
    """Round to nearest, ties away from zero (the format's one rounding rule)."""

    values = np.asarray(values, dtype=np.float64)
    return np.trunc(values + np.copysign(0.5, values))


def f16_scale(values: np.ndarray) -> np.ndarray:
    """Round f32 scales to binary16; the stored value is the one used to encode."""

    return np.asarray(values, dtype=np.float32).astype(np.float16)


def group_ranges(k: int, group_size: int = GROUP_SIZE) -> list[tuple[int, int]]:
    return [
        (start, min(start + group_size, k))
        for start in range(0, k, group_size)
    ]


def quantize_int4_grouped(
    weights: np.ndarray, group_size: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric signed-INT4 quantization with an integer zero-point.

    Returns ``(codes, scales, zero_points)`` where ``codes`` is ``int8 [N, K]``
    in ``[-8, 7]``, ``scales`` is the stored ``float16 [N, G]`` and
    ``zero_points`` is ``int8 [N, G]`` in ``[-8, 7]``.
    """

    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise ValueError("weights must be two-dimensional [N, K]")
    n, k = weights.shape
    if n == 0 or k == 0:
        raise ValueError("weights must not be empty")
    ranges = group_ranges(k, group_size)
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, len(ranges)), dtype=np.float16)
    zeros = np.zeros((n, len(ranges)), dtype=np.int8)

    for g, (start, end) in enumerate(ranges):
        block = weights[:, start:end]
        lo = block.min(axis=1)
        hi = block.max(axis=1)
        span = (hi - lo).astype(np.float32)
        degenerate = span == 0.0
        asym_scale = np.maximum(span / 15.0, 1e-30)
        zp_asym = -8.0 - lo / asym_scale
        # A signed code range cannot represent a group whose mean is more than
        # half the code range away from zero.  Those groups (and constant
        # groups) use the symmetric special case zero_point = 0.
        use_asym = (~degenerate) & (zp_asym >= -8.0) & (zp_asym <= 7.0)
        sym_abs = np.maximum(np.abs(lo), np.abs(hi))
        sym_scale = np.where(sym_abs == 0.0, 1.0, np.maximum(sym_abs / 7.0, 1e-6))
        scale = np.where(use_asym, asym_scale, sym_scale)
        scale16 = f16_scale(scale).astype(np.float32)
        if not np.all(np.isfinite(scale16)) or np.any(scale16 <= 0.0):
            raise ValueError("scale is not finite and positive")
        zp = np.where(use_asym, round_half_away(-8.0 - lo / scale16), 0.0)
        zp = np.clip(zp, -8, 7).astype(np.int8)
        q = round_half_away(block / scale16[:, None]) + zp.astype(np.int64)[:, None]
        codes[:, start:end] = np.clip(q, -8, 7).astype(np.int8)
        scales[:, g] = scale16.astype(np.float16)
        zeros[:, g] = zp
    return codes, scales, zeros


def quantize_int4_symmetric(
    weights: np.ndarray, group_size: int = GROUP_SIZE, divisor: float = 7.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric signed-INT4 baseline: ``q = round(w / (absmax/divisor))``."""

    weights = np.asarray(weights, dtype=np.float32)
    n, k = weights.shape
    ranges = group_ranges(k, group_size)
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, len(ranges)), dtype=np.float16)
    zeros = np.zeros((n, len(ranges)), dtype=np.int8)
    for g, (start, end) in enumerate(ranges):
        block = weights[:, start:end]
        absmax = np.abs(block).max(axis=1)
        scale = np.maximum(absmax / divisor, 1e-30)
        scale16 = f16_scale(scale).astype(np.float32)
        codes[:, start:end] = np.clip(
            round_half_away(block / scale16[:, None]), -8, 7
        ).astype(np.int8)
        scales[:, g] = scale16.astype(np.float16)
    return codes, scales, zeros


def quantize_int8_grouped(
    weights: np.ndarray, group_size: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric signed-INT8 weights with an integer zero-point.

    Same arithmetic as :func:`quantize_int4_grouped`, with 255 steps instead of
    15.  Codes are ``int8`` in ``[-128, 127]``.
    """

    weights = np.asarray(weights, dtype=np.float32)
    n, k = weights.shape
    ranges = group_ranges(k, group_size)
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, len(ranges)), dtype=np.float16)
    zeros = np.zeros((n, len(ranges)), dtype=np.int8)
    for g, (start, end) in enumerate(ranges):
        block = weights[:, start:end]
        lo = block.min(axis=1)
        hi = block.max(axis=1)
        span = (hi - lo).astype(np.float32)
        degenerate = span == 0.0
        asym_scale = np.maximum(span / 255.0, 1e-30)
        zp_asym = -128.0 - lo / asym_scale
        use_asym = (~degenerate) & (zp_asym >= -128.0) & (zp_asym <= 127.0)
        sym_abs = np.maximum(np.abs(lo), np.abs(hi))
        sym_scale = np.where(sym_abs == 0.0, 1.0, np.maximum(sym_abs / 127.0, 1e-6))
        scale = np.where(use_asym, asym_scale, sym_scale)
        scale16 = f16_scale(scale).astype(np.float32)
        if not np.all(np.isfinite(scale16)) or np.any(scale16 <= 0.0):
            raise ValueError("scale is not finite and positive")
        zp = np.where(use_asym, round_half_away(-128.0 - lo / scale16), 0.0)
        zp = np.clip(zp, -128, 127).astype(np.int8)
        q = round_half_away(block / scale16[:, None]) + zp.astype(np.int64)[:, None]
        codes[:, start:end] = np.clip(q, -128, 127).astype(np.int8)
        scales[:, g] = scale16.astype(np.float16)
        zeros[:, g] = zp
    return codes, scales, zeros


def quantize_int8_symmetric(
    weights: np.ndarray, group_size: int = GROUP_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric signed-INT8 weights: ``q = round(w / (absmax/127))``."""

    weights = np.asarray(weights, dtype=np.float32)
    n, k = weights.shape
    ranges = group_ranges(k, group_size)
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, len(ranges)), dtype=np.float16)
    zeros = np.zeros((n, len(ranges)), dtype=np.int8)
    for g, (start, end) in enumerate(ranges):
        block = weights[:, start:end]
        absmax = np.abs(block).max(axis=1)
        scale = np.maximum(absmax / 127.0, 1e-30)
        scale16 = f16_scale(scale).astype(np.float32)
        codes[:, start:end] = np.clip(
            round_half_away(block / scale16[:, None]), -127, 127
        ).astype(np.int8)
        scales[:, g] = scale16.astype(np.float16)
    return codes, scales, zeros


def quantize_int4_existing_w4a8(
    weights: np.ndarray, group_size: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 2.x grouped W4A8 encoder, reproduced independently.

    ``scale = (span + 2*span/15) / 15`` and ``zero = lo + 8*scale`` where
    ``lo = min - span/15``.  Codes round to nearest, ties away from zero and are
    clamped to ``[-8, 7]``.  The zero-point is an f32, not an integer.
    """

    weights = np.asarray(weights, dtype=np.float32)
    n, k = weights.shape
    ranges = group_ranges(k, group_size)
    codes = np.zeros((n, k), dtype=np.int8)
    scales = np.zeros((n, len(ranges)), dtype=np.float32)
    zeros = np.zeros((n, len(ranges)), dtype=np.float32)
    for g, (start, end) in enumerate(ranges):
        block = weights[:, start:end]
        lo = block.min(axis=1)
        hi = block.max(axis=1)
        span = hi - lo
        init = span / 15.0
        scale = (span + 2.0 * init) / 15.0
        zero = lo - init + 8.0 * scale
        safe = np.where(scale > 0.0, scale, 1.0)
        q = round_half_away((block - zero[:, None]) / safe[:, None])
        codes[:, start:end] = np.clip(q, -8, 7).astype(np.int8)
        scales[:, g] = np.where(scale > 0.0, scale, 1.0)
        zeros[:, g] = np.where(scale > 0.0, zero, lo)
    return codes, scales, zeros


def pack_int8_rows(codes: np.ndarray) -> bytes:
    """Pack ``int8 [N, K]`` codes into 16-byte-aligned rows, zero padded."""

    n, k = codes.shape
    row_stride = align_up(k, DATA_ALIGNMENT)
    out = bytearray(n * row_stride)
    for row in range(n):
        base = row * row_stride
        out[base : base + k] = np.asarray(codes[row], dtype=np.int8).tobytes()
    return bytes(out)


def unpack_int8_rows(payload: bytes, n: int, k: int) -> np.ndarray:
    row_stride = align_up(k, DATA_ALIGNMENT)
    if len(payload) != n * row_stride:
        raise QuantFormatError(
            "truncated-payload",
            f"expected {n * row_stride} packed bytes, found {len(payload)}",
        )
    codes = np.zeros((n, k), dtype=np.int8)
    for row in range(n):
        base = row * row_stride
        codes[row] = np.frombuffer(payload[base : base + k], dtype=np.int8)
    return codes


def dequantize(
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """``w = (q - z) * s`` in f32, with the stored binary16 scale widened."""

    n, k = codes.shape
    out = np.zeros((n, k), dtype=np.float32)
    for g, (start, end) in enumerate(group_ranges(k, group_size)):
        out[:, start:end] = (
            codes[:, start:end].astype(np.float32)
            - zeros[:, g].astype(np.float32)[:, None]
        ) * scales[:, g].astype(np.float32)[:, None]
    return out


def dequantize_float_zero(
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """The 2.x grouped W4A8 dequantization: ``w = q * s + zero``."""

    n, k = codes.shape
    out = np.zeros((n, k), dtype=np.float32)
    for g, (start, end) in enumerate(group_ranges(k, group_size)):
        out[:, start:end] = (
            codes[:, start:end].astype(np.float32) * scales[:, g].astype(np.float32)[:, None]
            + zeros[:, g].astype(np.float32)[:, None]
        )
    return out


@dataclass
class TensorEntry:
    name: str
    codes: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.codes.shape[0])

    @property
    def k(self) -> int:
        return int(self.codes.shape[1])

    @property
    def groups_per_row(self) -> int:
        return (self.k + GROUP_SIZE - 1) // GROUP_SIZE


def encode_blob(entries: list[TensorEntry]) -> bytes:
    """Encode a canonical v1 blob.  Used by the fixture generator."""

    if not entries:
        raise ValueError("a blob needs at least one tensor")
    names = set()
    for entry in entries:
        if not entry.name or len(entry.name.encode("ascii")) >= NAME_BYTES:
            raise ValueError(f"tensor name {entry.name!r} does not fit 32 bytes")
        if entry.name in names:
            raise ValueError(f"duplicate tensor name {entry.name!r}")
        names.add(entry.name)
        codes = np.asarray(entry.codes)
        if codes.ndim != 2 or entry.k == 0 or entry.n_rows == 0:
            raise ValueError(f"{entry.name}: empty or non-2D tensor")
        if codes.dtype != np.int8 or np.any(codes < -128) or np.any(codes > 127):
            raise ValueError(f"{entry.name}: codes must be signed int8")
        if entry.scales.shape != (entry.n_rows, entry.groups_per_row):
            raise ValueError(f"{entry.name}: scale shape")
        if entry.zeros.shape != entry.scales.shape:
            raise ValueError(f"{entry.name}: zero-point shape")
        if np.any(entry.zeros < -128) or np.any(entry.zeros > 127):
            raise ValueError(f"{entry.name}: zero-point outside [-128, 127]")

    descriptor_offset = HEADER_BYTES
    descriptor_bytes = len(entries) * DESCRIPTOR_BYTES
    group_offset = descriptor_offset + descriptor_bytes
    total_groups = sum(entry.groups_per_row for entry in entries)
    group_dir_bytes = total_groups * GROUP_ENTRY_BYTES
    payload_offset = align_up(group_offset + group_dir_bytes, DATA_ALIGNMENT)

    descriptors = bytearray()
    groups = bytearray()
    payload = bytearray()
    group_index = 0
    for entry in entries:
        if len(payload) % DATA_ALIGNMENT:
            raise AssertionError("payload cursor is not 16-byte aligned")
        data_offset = payload_offset + len(payload)
        packed = pack_int8_rows(entry.codes)
        payload += packed
        payload += entry.scales.astype("<f2").tobytes()
        payload += entry.zeros.astype("<i2").tobytes()
        row_stride = align_up(entry.k, DATA_ALIGNMENT)
        name = entry.name.encode("ascii").ljust(NAME_BYTES, b"\0")
        descriptors += DESCRIPTOR.pack(
            name,
            entry.n_rows,
            entry.k,
            QUBITS,
            GROUP_SIZE,
            entry.groups_per_row,
            group_index,
            row_stride,
            data_offset,
        )
        for g, (start, end) in enumerate(group_ranges(entry.k, GROUP_SIZE)):
            groups += GROUP_ENTRY.pack(start, end - start)
        group_index += entry.groups_per_row
        payload += b"\0" * (align_up(len(payload), DATA_ALIGNMENT) - len(payload))
    payload = bytes(payload)
    payload_bytes = len(payload)

    header = bytearray(
        HEADER.pack(
            MAGIC,
            VERSION,
            HEADER_BYTES,
            0,
            len(entries),
            descriptor_bytes,
            group_dir_bytes,
            descriptor_offset,
            group_offset,
            payload_offset,
            payload_bytes,
            zlib.crc32(payload) & 0xFFFFFFFF,
            0,
        )
    )
    meta = bytes(header) + bytes(descriptors) + bytes(groups)
    if len(meta) > payload_offset:
        raise AssertionError("metadata overruns the payload offset")
    meta = meta + b"\0" * (payload_offset - len(meta))
    crc = zlib.crc32(meta) & 0xFFFFFFFF
    header[60:64] = struct.pack("<I", crc)
    meta = bytes(header) + meta[HEADER_BYTES:]
    blob = meta + payload
    assert len(blob) == payload_offset + payload_bytes
    return blob


@dataclass
class ParsedTensor:
    name: str
    n_rows: int
    k: int
    qbits: int
    group_size: int
    groups_per_row: int
    row_stride_bytes: int
    data_offset: int
    codes: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray

    def dequantize(self) -> np.ndarray:
        return dequantize(self.codes, self.scales, self.zeros)


def decode_blob(blob: bytes) -> list[ParsedTensor]:
    """The independent reference decoder.  Raises QuantFormatError."""

    if len(blob) < HEADER_BYTES:
        raise QuantFormatError("truncated-header", f"{len(blob)} bytes")
    (
        magic,
        version,
        header_bytes,
        flags,
        tensor_count,
        descriptor_bytes,
        group_dir_bytes,
        descriptor_offset,
        group_offset,
        payload_offset,
        payload_bytes,
        payload_crc,
        meta_crc,
    ) = HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise QuantFormatError("bad-magic", repr(magic))
    if version != VERSION:
        raise QuantFormatError("unsupported-version", str(version))
    if header_bytes != HEADER_BYTES:
        raise QuantFormatError("bad-header-size", str(header_bytes))
    if flags != 0:
        raise QuantFormatError("unsupported-flags", hex(flags))
    if not 1 <= tensor_count <= 4096:
        raise QuantFormatError("bad-tensor-count", str(tensor_count))
    if descriptor_bytes != tensor_count * DESCRIPTOR_BYTES:
        raise QuantFormatError("bad-descriptor-bytes", str(descriptor_bytes))
    if descriptor_offset != HEADER_BYTES:
        raise QuantFormatError("bad-descriptor-offset", str(descriptor_offset))
    if group_offset != descriptor_offset + descriptor_bytes:
        raise QuantFormatError("bad-group-offset", str(group_offset))
    if group_dir_bytes % GROUP_ENTRY_BYTES:
        raise QuantFormatError("bad-group-dir-bytes", str(group_dir_bytes))
    if payload_offset != align_up(group_offset + group_dir_bytes, DATA_ALIGNMENT):
        raise QuantFormatError("bad-payload-offset", str(payload_offset))
    if len(blob) != payload_offset + payload_bytes:
        raise QuantFormatError(
            "trailing-or-truncated-bytes",
            f"file {len(blob)} != payload end {payload_offset + payload_bytes}",
        )
    meta = bytearray(blob[:payload_offset])
    stored_meta_crc = struct.unpack_from("<I", meta, 60)[0]
    meta[60:64] = b"\0\0\0\0"
    if (zlib.crc32(bytes(meta)) & 0xFFFFFFFF) != stored_meta_crc:
        raise QuantFormatError("meta-checksum", "metadata checksum mismatch")
    payload = blob[payload_offset:]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != payload_crc:
        raise QuantFormatError("payload-checksum", "payload checksum mismatch")

    total_groups = group_dir_bytes // GROUP_ENTRY_BYTES
    group_table = [
        GROUP_ENTRY.unpack_from(blob, group_offset + index * GROUP_ENTRY_BYTES)
        for index in range(total_groups)
    ]

    tensors: list[ParsedTensor] = []
    names: set[str] = set()
    expected_data_offset = payload_offset
    expected_group_index = 0
    for index in range(tensor_count):
        base = descriptor_offset + index * DESCRIPTOR_BYTES
        (
            raw_name,
            n_rows,
            k,
            qbits,
            group_size,
            groups_per_row,
            group_index,
            row_stride,
            data_offset,
        ) = DESCRIPTOR.unpack_from(blob, base)
        name_bytes = bytes(raw_name)
        nul = name_bytes.find(b"\0")
        if nul < 0:
            raise QuantFormatError("bad-name", f"tensor {index} is not NUL-terminated")
        name = name_bytes[:nul]
        if not name or any(byte < 0x20 or byte > 0x7E for byte in name):
            raise QuantFormatError("bad-name", f"tensor {index} has an empty/non-ASCII name")
        if name_bytes[nul + 1 :].strip(b"\0"):
            raise QuantFormatError("bad-name", f"{name!r} has bytes after the NUL")
        name = name.decode("ascii")
        if name in names:
            raise QuantFormatError("duplicate-name", name)
        names.add(name)
        if n_rows == 0 or k == 0:
            raise QuantFormatError("zero-extent", name)
        if qbits != QUBITS:
            raise QuantFormatError("unsupported-qbits", f"{name}: {qbits}")
        if group_size != GROUP_SIZE:
            raise QuantFormatError("unsupported-group-size", f"{name}: {group_size}")
        expected_groups = (k + GROUP_SIZE - 1) // GROUP_SIZE
        if groups_per_row != expected_groups:
            raise QuantFormatError(
                "bad-group-count", f"{name}: {groups_per_row} != {expected_groups}"
            )
        if group_index != expected_group_index:
            raise QuantFormatError("bad-group-index", name)
        expected_group_index += groups_per_row
        expected_stride = align_up(k, DATA_ALIGNMENT)
        if row_stride != expected_stride:
            raise QuantFormatError(
                "bad-row-stride", f"{name}: {row_stride} != {expected_stride}"
            )
        if data_offset % DATA_ALIGNMENT:
            raise QuantFormatError("misaligned-offset", f"{name}: {data_offset}")
        if data_offset != expected_data_offset:
            raise QuantFormatError(
                "bad-data-offset", f"{name}: {data_offset} != {expected_data_offset}"
            )
        data_bytes = n_rows * row_stride
        scale_bytes = n_rows * groups_per_row * 2
        zero_bytes = n_rows * groups_per_row * 2
        end = data_offset + data_bytes + scale_bytes + zero_bytes
        if end > payload_offset + payload_bytes:
            raise QuantFormatError("truncated-payload", name)
        expected_data_offset = align_up(end, DATA_ALIGNMENT)
        if blob[end:expected_data_offset].strip(b"\0"):
            raise QuantFormatError("nonzero-padding", f"{name}: after zero points")

        previous_end = 0
        for g in range(groups_per_row):
            start, length = group_table[group_index + g]
            if start != previous_end:
                raise QuantFormatError(
                    "group-gap-or-overlap",
                    f"{name}: group {g} starts at {start}, expected {previous_end}",
                )
            if length != min(GROUP_SIZE, k - start):
                raise QuantFormatError(
                    "bad-group-length", f"{name}: group {g} length {length}"
                )
            previous_end = start + length
        if previous_end != k:
            raise QuantFormatError("incomplete-groups", name)

        packed = blob[data_offset : data_offset + data_bytes]
        codes = unpack_int8_rows(packed, n_rows, k)
        # Bytes after k in each row are alignment padding and must be zero.
        for row in range(n_rows):
            row_base = row * row_stride
            if packed[row_base + k : row_base + row_stride].strip(b"\0"):
                raise QuantFormatError(
                    "nonzero-tail-padding", f"{name}: row {row}"
                )
        scales = np.frombuffer(
            blob[data_offset + data_bytes : data_offset + data_bytes + scale_bytes],
            dtype="<f2",
        ).reshape(n_rows, groups_per_row)
        zeros = np.frombuffer(
            blob[data_offset + data_bytes + scale_bytes : end], dtype="<i2"
        ).reshape(n_rows, groups_per_row)
        if not np.all(np.isfinite(scales.astype(np.float32))) or np.any(scales <= 0):
            raise QuantFormatError("bad-scale", name)
        if np.any(zeros < -128) or np.any(zeros > 127):
            raise QuantFormatError("zero-point-out-of-range", name)
        tensors.append(
            ParsedTensor(
                name=name,
                n_rows=n_rows,
                k=k,
                qbits=qbits,
                group_size=group_size,
                groups_per_row=groups_per_row,
                row_stride_bytes=row_stride,
                data_offset=data_offset,
                codes=codes,
                scales=scales,
                zeros=zeros,
            )
        )

    if expected_group_index != total_groups:
        raise QuantFormatError("unused-group-entries", str(total_groups - expected_group_index))
    if expected_data_offset != payload_offset + payload_bytes:
        raise QuantFormatError(
            "payload-tail",
            f"{expected_data_offset} != {payload_offset + payload_bytes}",
        )
    return tensors


def w4a8_matmul(
    activations: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """Exact kernel semantics with INT8 activations and a quantized weight.

    ``activations`` is ``[M, K]`` f32, ``codes``/``scales``/``zeros`` describe
    ``[N, K]`` weights.  Integer dot per group, integer zero-point correction,
    then one f32 scale multiply and an f32 accumulation across groups.
    """

    activations = np.asarray(activations, dtype=np.float32)
    m, k = activations.shape
    n = codes.shape[0]
    act_scale = np.maximum(np.abs(activations).max(axis=1) / 127.0, 1e-30)
    act_codes = np.clip(
        round_half_away(activations / act_scale[:, None]), -128, 127
    ).astype(np.int16)
    out = np.zeros((m, n), dtype=np.float32)
    for g, (start, end) in enumerate(group_ranges(k, group_size)):
        a = act_codes[:, start:end].astype(np.float64)
        w = codes[:, start:end].astype(np.float64)
        dot = a @ w.T  # exact in binary64 for these magnitudes
        correction = zeros[:, g].astype(np.float64)[None, :] * a.sum(axis=1)[:, None]
        combine = (dot - correction).astype(np.float32) * (
            scales[:, g].astype(np.float32)[None, :] * act_scale[:, None]
        )
        out += combine.astype(np.float32)
    return out


def w4a8_matmul_float_zero(
    activations: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int = GROUP_SIZE,
) -> np.ndarray:
    """The 2.x grouped W4A8 kernel: ``dot*s_w*s_a + zero_w*s_a*sum(q_a)``."""

    activations = np.asarray(activations, dtype=np.float32)
    m, k = activations.shape
    n = codes.shape[0]
    act_scale = np.maximum(np.abs(activations).max(axis=1) / 127.0, 1e-30)
    act_codes = np.clip(
        round_half_away(activations / act_scale[:, None]), -128, 127
    ).astype(np.int16)
    out = np.zeros((m, n), dtype=np.float32)
    for g, (start, end) in enumerate(group_ranges(k, group_size)):
        a = act_codes[:, start:end].astype(np.float64)
        w = codes[:, start:end].astype(np.float64)
        dot = a @ w.T
        s_w = scales[:, g].astype(np.float64)[None, :]
        z_w = zeros[:, g].astype(np.float64)[None, :]
        s_a = act_scale.astype(np.float64)[:, None]
        out += (dot * s_w * s_a + z_w * s_a * a.sum(axis=1)[:, None]).astype(
            np.float32
        )
    return out
