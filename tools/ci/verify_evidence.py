#!/usr/bin/env python3
"""Validate a physical-GPU evidence record produced by tools/ci/run-gpu-lane.sh.

This is the gate that keeps three different kinds of evidence apart:

* a *compile* check (no device),
* a *software-ICD* smoke run (lavapipe, CI-ONLY), and
* a *physical-GPU* run (the only thing this module accepts).

Rules enforced here, not by convention:

1. ``ci_only`` must be false and the device must not be a CPU/lavapipe device.
2. The record must name the device, its UUID, driver version and Vulkan API
   version, and it must include validation-layer results.
3. A record whose overall status is ``passed`` must contain a passed
   ``fastnn_vulkan_smoke`` check. A device probe alone is not backend evidence.
4. A ``not_verified`` record is truthful but is never accepted as evidence.

Usage:
    python3 tools/ci/verify_evidence.py RECORD.json [RECORD.json ...]
    python3 tools/ci/verify_evidence.py --require-pass RECORD.json
    python3 tools/ci/verify_evidence.py --json RECORD.json

Exit codes: 0 valid, 1 invalid or not acceptable as evidence.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
KIND = "fastnn-gpu-evidence"
LANE = "gpu-physical"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
API_RE = re.compile(r"^\d+\.\d+\.\d+$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SOFTWARE_MARKERS = ("lavapipe", "llvmpipe", "swiftshader", "software rasterizer")
STATUSES = ("passed", "failed", "not_verified")


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_record(record: dict) -> list[str]:
    """Return a list of validation errors; an empty list means valid."""
    errors: list[str] = []
    _require(record.get("schema") == SCHEMA, f"schema must be {SCHEMA}", errors)
    _require(record.get("kind") == KIND, f"kind must be {KIND!r}", errors)
    _require(record.get("lane") == LANE, f"lane must be {LANE!r}", errors)
    _require(record.get("ci_only") is False, "ci_only must be false: CI-only evidence is not GPU evidence", errors)

    generated = record.get("generated_utc")
    if not isinstance(generated, str):
        errors.append("generated_utc must be an ISO-8601 UTC string")
    else:
        try:
            _dt.datetime.fromisoformat(generated.replace("Z", "+00:00"))
        except ValueError:
            errors.append(f"generated_utc is not ISO-8601: {generated!r}")

    git = record.get("git") or {}
    _require(isinstance(git.get("head"), str) and bool(SHA_RE.match(git["head"])), "git.head must be a 40-hex commit", errors)
    _require(git.get("dirty") in (True, False), "git.dirty must be a boolean", errors)

    device = record.get("device") or {}
    _require(device.get("kind") == "physical", "device.kind must be 'physical'", errors)
    name = str(device.get("name") or "")
    _require(bool(name), "device.name must be non-empty", errors)
    lowered = " ".join(str(device.get(key) or "") for key in ("name", "driver_name", "icd")).lower()
    for marker in SOFTWARE_MARKERS:
        _require(marker not in lowered, f"device mentions software renderer {marker!r}; not physical-GPU evidence", errors)
    _require(isinstance(device.get("uuid"), str) and bool(UUID_RE.match(device["uuid"])), "device.uuid must be a canonical 8-4-4-4-12 UUID", errors)
    device_type = str(device.get("device_type") or "")
    _require(device_type.startswith("VK_PHYSICAL_DEVICE_TYPE_"), "device.device_type must be a VkPhysicalDeviceType", errors)
    _require(device_type != "VK_PHYSICAL_DEVICE_TYPE_CPU", "device.device_type is CPU: not physical-GPU evidence", errors)
    _require(bool(str(device.get("driver_version") or "")), "device.driver_version must be non-empty", errors)
    _require(isinstance(device.get("api_version"), str) and bool(API_RE.match(device["api_version"])), "device.api_version must look like 1.4.312", errors)

    validation = record.get("validation") or {}
    _require(validation.get("enabled") is True, "validation.enabled must be true", errors)
    _require(isinstance(validation.get("errors"), int) and validation["errors"] == 0, "validation.errors must be 0", errors)
    _require(isinstance(validation.get("warnings"), int), "validation.warnings must be an integer", errors)
    _require(validation.get("result") == "clean", "validation.result must be 'clean'", errors)

    timings = record.get("timings") or {}
    total = timings.get("total_seconds")
    _require(isinstance(total, (int, float)) and total > 0, "timings.total_seconds must be positive", errors)

    result = record.get("result") or {}
    status = result.get("status")
    _require(status in STATUSES, f"result.status must be one of {STATUSES}", errors)
    _require(bool(str(result.get("reason") or "")), "result.reason must explain the outcome", errors)

    checks = record.get("checks")
    _require(isinstance(checks, list) and bool(checks), "checks must be a non-empty list", errors)
    if isinstance(checks, list):
        names = {check.get("name") for check in checks if isinstance(check, dict)}
        for check in checks:
            if not isinstance(check, dict):
                errors.append("each check must be an object")
                continue
            _require(check.get("status") in ("passed", "failed", "not_run"), f"check {check.get('name')!r} has an invalid status", errors)
        if status == "passed":
            passed_smoke = any(
                isinstance(check, dict)
                and check.get("name") == "fastnn_vulkan_smoke"
                and check.get("status") == "passed"
                for check in checks
            )
            _require(passed_smoke, "a passed record must contain a passed 'fastnn_vulkan_smoke' check", errors)
        if status == "failed":
            errors.append("result.status is 'failed': a failed run is not evidence of correctness")
    return errors


def load_record(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("evidence record must be a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--require-pass", action="store_true", help="also require result.status == 'passed'")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable verdict")
    args = parser.parse_args(argv)

    verdicts = []
    for path in args.records:
        try:
            record = load_record(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            verdicts.append({"path": str(path), "valid": False, "errors": [f"cannot read record: {error}"]})
            continue
        errors = validate_record(record)
        if args.require_pass:
            status = (record.get("result") or {}).get("status")
            if status != "passed":
                errors.append(f"result.status is {status!r}; a physical-GPU evidence record must be passed")
        verdicts.append({"path": str(path), "valid": not errors, "errors": errors})

    ok = all(verdict["valid"] for verdict in verdicts)
    if args.json:
        print(json.dumps({"ok": ok, "records": verdicts}, indent=2))
    else:
        for verdict in verdicts:
            label = "VALID" if verdict["valid"] else "REJECTED"
            print(f"{label}: {verdict['path']}")
            for error in verdict["errors"]:
                print(f"  - {error}")
        if not ok:
            print(
                "Physical-GPU evidence must be produced by tools/ci/run-gpu-lane.sh and must "
                "name a physical device with a clean validation run. Software-ICD output is never accepted.",
                file=sys.stderr,
            )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
