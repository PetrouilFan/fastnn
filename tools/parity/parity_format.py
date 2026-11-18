"""Write FastNN parity tensor fixtures.

A fixture suite is a directory containing:

* ``manifest.json`` -- language-neutral tensor index and provenance;
* ``tensors/<name>.<dtype>`` -- optional raw little-endian payloads;
* ``model.lock.json`` -- pinned upstream model files (optional per suite).

The manifest is byte-reproducible: it contains no timestamps, absolute paths,
or unordered maps.  Tensor payloads are always C-order (row-major) and
little-endian.  See ``fixtures/parity/README.md`` for the consumer contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORMAT_NAME = "fastnn-parity-tensors"
FORMAT_VERSION = 1
ENDIANNESS = "little"
LAYOUT = "row-major (C order)"

DTYPE_ITEMSIZE = {"f32": 4, "i64": 8}
_DTYPE_TO_NUMPY = {"f32": "<f4", "i64": "<i8"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def encode_array(array: np.ndarray, dtype: str) -> bytes:
    """Return the canonical little-endian C-order bytes for *array*."""
    if dtype not in DTYPE_ITEMSIZE:
        raise ValueError(f"unsupported fixture dtype: {dtype!r}")
    array = np.asarray(array)
    if dtype == "f32":
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"f32 payload from non-floating array {array.dtype}")
        encoded = array.astype("<f4", copy=False)
    else:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"i64 payload from non-integer array {array.dtype}")
        encoded = array.astype("<i8", copy=False)
    return np.ascontiguousarray(encoded).tobytes()


def _json_number(value: Any) -> int | float:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite value cannot be recorded in JSON: {number!r}")
        return number
    if isinstance(value, bool):
        return int(value)
    return value


def summarize(array: np.ndarray, *, argmax: bool = True) -> dict[str, Any]:
    """Compact, checksummable summary of a tensor."""
    array = np.asarray(array)
    flat = array.reshape(-1)
    count = int(flat.size)
    is_float = np.issubdtype(array.dtype, np.floating)
    summary: dict[str, Any] = {
        "count": count,
        "min": _json_number(np.min(array)) if count else None,
        "max": _json_number(np.max(array)) if count else None,
        "mean_abs": float(np.mean(np.abs(flat.astype(np.float64)))) if count else 0.0,
        "max_abs": float(np.max(np.abs(flat.astype(np.float64)))) if count else 0.0,
        "nan_count": int(np.count_nonzero(np.isnan(flat))) if is_float else 0,
        "inf_count": int(np.count_nonzero(np.isinf(flat))) if is_float else 0,
        "first": [_json_number(v) for v in flat[: min(8, count)]],
        "last": [_json_number(v) for v in flat[-min(8, count) :]] if count else [],
    }
    if argmax and count:
        summary["argmax_flat"] = int(np.argmax(flat))
    return summary


def _slice_summaries(
    array: np.ndarray, axis: int, *, argmax: bool
) -> list[dict[str, Any]]:
    items = []
    for index in range(array.shape[axis]):
        piece = np.take(array, index, axis=axis)
        piece = np.ascontiguousarray(piece)
        items.append(
            {
                "index": index,
                "shape": [int(size) for size in piece.shape],
                "sha256": sha256_bytes(encode_array(piece, "f32")),
                "summary": summarize(piece, argmax=argmax),
            }
        )
    return items


class Suite:
    """Collect tensors and write a deterministic manifest for one suite."""

    def __init__(
        self,
        root: Path,
        name: str,
        *,
        description: str,
        provenance: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.description = description
        self.provenance = provenance
        self.extra = extra or {}
        self.tensors: list[dict[str, Any]] = []
        self._names: set[str] = set()

    def add(
        self,
        name: str,
        array: np.ndarray,
        dtype: str,
        *,
        provenance: dict[str, Any],
        stored: bool = True,
        slice_axis: int | None = None,
        slice_argmax: bool = True,
    ) -> dict[str, Any]:
        if name in self._names:
            raise ValueError(f"duplicate tensor name in {self.name}: {name}")
        self._names.add(name)
        array = np.asarray(array)
        payload = encode_array(array, dtype)
        expected = int(np.prod(array.shape, dtype=np.int64)) * DTYPE_ITEMSIZE[dtype]
        if len(payload) != expected:
            raise AssertionError(
                f"{name}: encoded {len(payload)} bytes, expected {expected}"
            )
        entry: dict[str, Any] = {
            "name": name,
            "path": None,
            "dtype": dtype,
            "shape": [int(size) for size in array.shape],
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "stored": bool(stored),
            "summary": summarize(array),
            "provenance": provenance,
        }
        if stored:
            relative = Path("tensors") / f"{name}.{dtype}"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            entry["path"] = relative.as_posix()
        if slice_axis is not None:
            entry["slices"] = {
                "axis": int(slice_axis),
                "items": _slice_summaries(array, slice_axis, argmax=slice_argmax),
            }
        self.tensors.append(entry)
        return entry

    def write(self) -> None:
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "endianness": ENDIANNESS,
            "layout": LAYOUT,
            "suite": self.name,
            "description": self.description,
            "provenance": self.provenance,
            **self.extra,
            "tensors": self.tensors,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def reset_suite_dir(path: Path) -> None:
    """Remove a previously generated suite directory, never a parent."""
    path = Path(path)
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def hash_file_entry(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = Path(root) / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
        "verified": "local_sha256",
    }


def iter_tensor_names(manifest: dict[str, Any]) -> Iterable[str]:
    for tensor in manifest["tensors"]:
        yield tensor["name"]
