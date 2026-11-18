#!/usr/bin/env python3
"""Offline verifier for committed FastNN parity fixtures.

This script deliberately imports only the Python standard library.  It needs no
network, no NumPy, no PyTorch, and no ONNX Runtime, so it can run in CI from a
plain checkout:

    python tools/parity/verify_fixtures.py --root fixtures/parity

It checks manifests, byte lengths, SHA-256 digests, summaries, slice digests,
model locks, cross-references, and the absence of unreferenced payload files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterable

DTYPE_ITEMSIZE = {"f32": 4, "i64": 8}
DTYPE_STRUCT = {"f32": "<f", "i64": "<q"}
FORMAT_NAME = "fastnn-parity-tensors"
FORMAT_VERSION = 1


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.errors: list[str] = []
        self.tensor_count = 0
        self.stored_count = 0
        self.stored_bytes = 0

    def error(self, message: str) -> None:
        self.errors.append(message)

    def require(self, condition: bool, message: str) -> bool:
        if not condition:
            self.error(message)
        return condition

    # -- small helpers -----------------------------------------------------

    def _resolve(self, suite_dir: Path, relative: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise ValueError("empty path")
        candidate = (suite_dir / relative).resolve()
        try:
            candidate.relative_to(suite_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"path escapes suite directory: {relative!r}") from exc
        return candidate

    def _decode(self, payload: bytes, dtype: str) -> list[int | float]:
        item = DTYPE_ITEMSIZE[dtype]
        if len(payload) % item:
            raise ValueError(f"payload length {len(payload)} is not a multiple of {item}")
        fmt = DTYPE_STRUCT[dtype]
        count = len(payload) // item
        return [value[0] for value in struct.iter_unpack(fmt, payload[: count * item])]

    # -- summary checks ----------------------------------------------------

    def _check_summary(
        self,
        *,
        label: str,
        values: list[int | float],
        summary: dict[str, Any],
        dtype: str,
    ) -> None:
        if not isinstance(summary, dict):
            self.error(f"{label}: summary is not an object")
            return
        expected = {
            "count",
            "min",
            "max",
            "mean_abs",
            "max_abs",
            "nan_count",
            "inf_count",
            "first",
            "last",
        }
        missing = expected - set(summary)
        if missing:
            self.error(f"{label}: summary missing {sorted(missing)}")
            return
        if summary["count"] != len(values):
            self.error(f"{label}: summary count {summary['count']} != {len(values)}")
            return
        if not values:
            return
        floats = [float(value) for value in values]
        if dtype == "f32":
            nan_count = sum(1 for value in floats if math.isnan(value))
            inf_count = sum(1 for value in floats if math.isinf(value))
            finite = [value for value in floats if not math.isnan(value) and not math.isinf(value)]
        else:
            nan_count = inf_count = 0
            finite = floats
        if summary["nan_count"] != nan_count:
            self.error(f"{label}: nan_count {summary['nan_count']} != {nan_count}")
        if summary["inf_count"] != inf_count:
            self.error(f"{label}: inf_count {summary['inf_count']} != {inf_count}")
        if finite:
            if float(summary["min"]) != min(finite):
                self.error(f"{label}: min mismatch")
            if float(summary["max"]) != max(finite):
                self.error(f"{label}: max mismatch")
            max_abs = max(abs(value) for value in finite)
            if not math.isclose(float(summary["max_abs"]), max_abs, rel_tol=1e-9, abs_tol=1e-12):
                self.error(f"{label}: max_abs {summary['max_abs']} != {max_abs}")
            mean_abs = sum(abs(value) for value in finite) / len(finite)
            if not math.isclose(float(summary["mean_abs"]), mean_abs, rel_tol=1e-9, abs_tol=1e-12):
                self.error(f"{label}: mean_abs {summary['mean_abs']} != {mean_abs}")
            if "argmax_flat" in summary:
                argmax = max(range(len(finite)), key=lambda index: finite[index])
                if summary["argmax_flat"] != argmax:
                    self.error(f"{label}: argmax_flat {summary['argmax_flat']} != {argmax}")
        first = summary.get("first")
        last = summary.get("last")
        if not isinstance(first, list) or first != values[: len(first)]:
            self.error(f"{label}: first values mismatch")
        if not isinstance(last, list) or last != values[-len(last) :]:
            self.error(f"{label}: last values mismatch")

    def _slice_values(
        self, values: list[int | float], shape: list[int], axis: int, index: int
    ) -> list[int | float]:
        strides = [0] * len(shape)
        stride = 1
        for dim in range(len(shape) - 1, -1, -1):
            strides[dim] = stride
            stride *= shape[dim]
        outer_shape = shape[:axis] + shape[axis + 1 :]
        outer_strides = strides[:axis] + strides[axis + 1 :]
        result: list[int | float] = []
        index_tuple = [0] * len(outer_shape)
        while True:
            offset = index * strides[axis]
            for dim, value in enumerate(index_tuple):
                offset += value * outer_strides[dim]
            result.append(values[offset])
            dim = len(outer_shape) - 1
            while dim >= 0:
                index_tuple[dim] += 1
                if index_tuple[dim] < outer_shape[dim]:
                    break
                index_tuple[dim] = 0
                dim -= 1
            if dim < 0:
                break
        return result

    # -- tensor checks -----------------------------------------------------

    def check_tensor(self, suite_dir: Path, tensor: dict[str, Any]) -> None:
        self.tensor_count += 1
        name = tensor.get("name", "<unnamed>")
        label = f"{suite_dir.name}/{name}"
        required = {"name", "path", "dtype", "shape", "bytes", "sha256", "stored", "summary", "provenance"}
        missing = required - set(tensor)
        if missing:
            self.error(f"{label}: missing fields {sorted(missing)}")
            return
        dtype = tensor["dtype"]
        if dtype not in DTYPE_ITEMSIZE:
            self.error(f"{label}: unsupported dtype {dtype!r}")
            return
        shape = tensor["shape"]
        if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
            self.error(f"{label}: invalid shape {shape!r}")
            return
        count = math.prod(shape) if shape else 1
        expected_bytes = count * DTYPE_ITEMSIZE[dtype]
        if tensor["bytes"] != expected_bytes:
            self.error(f"{label}: bytes {tensor['bytes']} != {expected_bytes}")
        sha = tensor["sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            self.error(f"{label}: invalid sha256 {sha!r}")
            return
        stored = tensor["stored"]
        if not isinstance(stored, bool):
            self.error(f"{label}: stored must be boolean")
            return
        if stored:
            self.stored_count += 1
            self.stored_bytes += int(tensor["bytes"])
            if not isinstance(tensor["path"], str):
                self.error(f"{label}: stored tensor has no path")
                return
            try:
                path = self._resolve(suite_dir, tensor["path"])
            except ValueError as error:
                self.error(f"{label}: {error}")
                return
            if path.is_symlink():
                self.error(f"{label}: payload must not be a symlink")
                return
            if not path.is_file():
                self.error(f"{label}: missing payload {tensor['path']}")
                return
            if path.stat().st_size != tensor["bytes"]:
                self.error(
                    f"{label}: file size {path.stat().st_size} != {tensor['bytes']}"
                )
                return
            payload = path.read_bytes()
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != sha:
                self.error(f"{label}: sha256 {actual_sha} != {sha}")
                return
            values = self._decode(payload, dtype)
            self._check_summary(label=label, values=values, summary=tensor["summary"], dtype=dtype)
            if "slices" in tensor:
                self._check_slices(label, values, shape, tensor["slices"], dtype)
        else:
            if tensor["path"] is not None:
                self.error(f"{label}: summary-only tensor must have null path")
            # The full digest is still committed; it can be regenerated but not
            # verified from the checkout alone.  Validate its shape.
            if not isinstance(tensor["summary"], dict) or "count" not in tensor["summary"]:
                self.error(f"{label}: summary-only tensor lacks a summary block")

    def _check_slices(
        self,
        label: str,
        values: list[int | float],
        shape: list[int],
        slices: dict[str, Any],
        dtype: str,
    ) -> None:
        axis = slices.get("axis")
        items = slices.get("items")
        if not isinstance(axis, int) or not 0 <= axis < len(shape):
            self.error(f"{label}: invalid slice axis {axis!r}")
            return
        if not isinstance(items, list) or len(items) != shape[axis]:
            self.error(f"{label}: expected {shape[axis]} slices, got {len(items) if isinstance(items, list) else items!r}")
            return
        for item in items:
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < shape[axis]:
                self.error(f"{label}: invalid slice index {index!r}")
                continue
            slice_values = self._slice_values(values, shape, axis, index)
            item_label = f"{label}[axis {axis}={index}]"
            if item.get("shape") != ([dim for dim_index, dim in enumerate(shape) if dim_index != axis]):
                self.error(f"{item_label}: slice shape mismatch")
            if dtype == "f32":
                payload = struct.pack(f"<{len(slice_values)}f", *[float(v) for v in slice_values])
            else:
                payload = struct.pack(f"<{len(slice_values)}q", *[int(v) for v in slice_values])
            actual_sha = hashlib.sha256(payload).hexdigest()
            if item.get("sha256") != actual_sha:
                self.error(f"{item_label}: slice sha mismatch")
            self._check_summary(
                label=item_label, values=slice_values, summary=item.get("summary", {}), dtype=dtype
            )

    # -- manifest checks ---------------------------------------------------

    def check_manifest(self, suite_dir: Path) -> None:
        manifest_path = suite_dir / "manifest.json"
        if not manifest_path.is_file():
            self.error(f"{suite_dir}: missing manifest.json")
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.error(f"{manifest_path}: {error}")
            return
        if manifest.get("format") != FORMAT_NAME:
            self.error(f"{manifest_path}: format {manifest.get('format')!r}")
        if manifest.get("format_version") != FORMAT_VERSION:
            self.error(f"{manifest_path}: format_version {manifest.get('format_version')!r}")
        if manifest.get("endianness") != "little":
            self.error(f"{manifest_path}: endianness must be 'little'")
        if manifest.get("layout") != "row-major (C order)":
            self.error(f"{manifest_path}: layout must be 'row-major (C order)'")
        tensors = manifest.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            self.error(f"{manifest_path}: no tensors")
            return
        names: set[str] = set()
        payload_paths: set[str] = set()
        for tensor in tensors:
            name = tensor.get("name")
            if name in names:
                self.error(f"{manifest_path}: duplicate tensor {name!r}")
            names.add(name)
            if isinstance(tensor.get("path"), str):
                payload_paths.add(tensor["path"])
            # Provenance is required by the format contract.
            if not isinstance(tensor.get("provenance"), dict):
                self.error(f"{manifest_path}: {name!r} has no provenance block")
            self.check_tensor(suite_dir, tensor)

        # No stale payloads.
        tensor_dir = suite_dir / "tensors"
        if tensor_dir.is_dir():
            for path in sorted(tensor_dir.rglob("*")):
                if path.is_file() and path.relative_to(suite_dir).as_posix() not in payload_paths:
                    self.error(f"{suite_dir}: unreferenced payload {path.relative_to(suite_dir)}")

        self.check_cross_references(suite_dir, manifest, names)
        self.check_lock(suite_dir, manifest)

    def check_cross_references(
        self, suite_dir: Path, manifest: dict[str, Any], names: set[str]
    ) -> None:
        def require_tensor(value: Any, where: str) -> None:
            if not isinstance(value, str) or value not in names:
                self.error(f"{suite_dir}/{where}: unknown tensor {value!r}")

        for key in ("prefill", "decode"):
            records = manifest.get(key)
            if records is None:
                continue
            if isinstance(records, dict):
                records = [records]
            if not isinstance(records, list):
                self.error(f"{suite_dir}/{key}: expected object or list")
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                for field in ("logits_tensor", "last_logits_tensor", "full_logits_tensor"):
                    if field in record:
                        require_tensor(record[field], f"{key}.{field}")
                present = record.get("present", {})
                if isinstance(present, dict):
                    for entry in present.values():
                        if isinstance(entry, dict) and "tensor" in entry:
                            require_tensor(entry["tensor"], f"{key}.present")
                cached_full = record.get("cached_vs_full", {})
                if isinstance(cached_full, dict) and "full_logits_tensor" in cached_full:
                    require_tensor(cached_full["full_logits_tensor"], f"{key}.cached_vs_full")
        noise = manifest.get("noise_floor", {})
        if isinstance(noise, dict):
            for entry in noise.get("cached_vs_full", []):
                if isinstance(entry, dict) and "full_logits_tensor" in entry:
                    require_tensor(entry["full_logits_tensor"], "noise_floor.cached_vs_full")

    def check_lock(self, suite_dir: Path, manifest: dict[str, Any]) -> None:
        lock_path = suite_dir / "model.lock.json"
        if not lock_path.is_file():
            return
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self.error(f"{lock_path}: {error}")
            return
        if lock.get("format") != "fastnn-model-lock":
            self.error(f"{lock_path}: bad format")
        if lock.get("format_version") != 1:
            self.error(f"{lock_path}: bad format_version")
        revision = lock.get("revision", "")
        if not isinstance(revision, str) or len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            self.error(f"{lock_path}: revision is not a 40-character commit SHA")
        if not isinstance(lock.get("license"), dict):
            self.error(f"{lock_path}: missing license block")
        files = lock.get("files")
        if not isinstance(files, list) or not files:
            self.error(f"{lock_path}: missing files")
            return
        by_path = {}
        for entry in files:
            if not isinstance(entry, dict):
                self.error(f"{lock_path}: file entry is not an object")
                continue
            path = entry.get("path")
            sha = entry.get("sha256")
            size = entry.get("bytes")
            if not isinstance(path, str) or not isinstance(sha, str) or len(sha) != 64:
                self.error(f"{lock_path}: invalid file entry {entry!r}")
                continue
            if not isinstance(size, int) or size < 0:
                self.error(f"{lock_path}: invalid size for {path}")
            by_path[path] = entry
        graph = manifest.get("provenance", {}).get("graph", {})
        if isinstance(graph, dict) and "path" in graph:
            lock_graph = by_path.get(graph["path"])
            if not lock_graph:
                self.error(f"{lock_path}: graph {graph['path']!r} absent from lock")
            elif lock_graph["sha256"] != graph.get("sha256"):
                self.error(f"{lock_path}: graph sha does not match manifest")
            elif lock_graph["bytes"] != graph.get("bytes"):
                self.error(f"{lock_path}: graph size does not match manifest")
        tokenizer = manifest.get("provenance", {}).get("tokenizer", {})
        if isinstance(tokenizer, dict) and "tokenizer_json_sha256" in tokenizer:
            tokenizer_entry = by_path.get("tokenizer.json")
            if not tokenizer_entry:
                self.error(f"{lock_path}: tokenizer.json absent from lock")
            elif tokenizer_entry["sha256"] != tokenizer["tokenizer_json_sha256"]:
                self.error(f"{lock_path}: tokenizer sha does not match manifest")

    # -- driver ------------------------------------------------------------

    def run(self) -> int:
        if not self.root.is_dir():
            print(f"error: fixture root does not exist: {self.root}", file=sys.stderr)
            return 1
        manifests = sorted(self.root.glob("*/manifest.json"))
        if not manifests:
            print(f"error: no suite manifests under {self.root}", file=sys.stderr)
            return 1
        for manifest_path in manifests:
            self.check_manifest(manifest_path.parent)
        if self.errors:
            print(f"FAILED: {len(self.errors)} problem(s)", file=sys.stderr)
            for message in self.errors[:100]:
                print(f"  - {message}", file=sys.stderr)
            if len(self.errors) > 100:
                print(f"  ... {len(self.errors) - 100} more", file=sys.stderr)
            return 1
        print(
            f"OK: {len(manifests)} suites, {self.tensor_count} tensors "
            f"({self.stored_count} stored, {self.stored_bytes} bytes)"
        )
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "fixtures" / "parity",
    )
    args = parser.parse_args()
    return Checker(args.root).run()


if __name__ == "__main__":
    raise SystemExit(main())
