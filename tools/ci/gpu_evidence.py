#!/usr/bin/env python3
"""Collect a physical-GPU evidence record for the gpu-physical lane.

This module is the implementation behind tools/ci/run-gpu-lane.sh; run that
script, not this file. It is deliberately separate from the bash entry point so
the device/UUID/API parsing is testable and cross-platform.

Honesty rules implemented here:

* A hosted CI runner is refused before any device work.
* The device must be a physical Vulkan device (deviceType != CPU and no
  software-renderer marker). A lavapipe-only machine produces no evidence.
* The fastnn Vulkan smoke test is the only check that can make the record
  "passed". When the backend does not exist yet (FN3-11 / #57), the record is
  "not_verified" and says so; the auxiliary spike run is labelled as raw
  Vulkan bring-up, never as fastnn backend evidence.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE_MARKERS = ("lavapipe", "llvmpipe", "swiftshader", "software rasterizer")
VALIDATION_ERROR_RE = re.compile(r"(VUID-|Validation Error|ValidationError)")
VALIDATION_SUMMARY_RE = re.compile(r"validation summary:\s*errors=(?P<errors>\d+)\s+warnings=(?P<warnings>\d+)", re.IGNORECASE)
UUID_BYTES_RE = re.compile(r"^[0-9a-f]{32}$")
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NOT_VERIFIED = 69


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        env={**os.environ, **(env or {})},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def _uuid_from_bytes(values: list[int]) -> str:
    raw = "".join(f"{value:02x}" for value in values)
    if len(raw) != 32:
        raise ValueError(f"deviceUUID must be 16 bytes, got {len(values)}")
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"


def _version(raw: int) -> str:
    return f"{(raw >> 22) & 0x7F}.{(raw >> 12) & 0x3FF}.{raw & 0xFFF}"


def _read_vulkaninfo(device_index: int | None) -> tuple[dict | None, str]:
    if shutil.which("vulkaninfo") is None:
        return None, "vulkaninfo is not installed"
    with tempfile.TemporaryDirectory() as temp:
        args = ["vulkaninfo", "-j"] if device_index is None else ["vulkaninfo", f"-j={device_index}"]
        result = _run(args, cwd=Path(temp), timeout=120)
        candidates = list(Path(temp).glob("VP_VULKANINFO_*.json"))
        if result.returncode != 0 or not candidates:
            return None, result.stdout.strip()[-2000:] or "vulkaninfo produced no JSON"
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        capabilities = data.get("capabilities") or {}
        device = capabilities.get("device") or {}
        properties = device.get("properties") or {}
        props = properties.get("VkPhysicalDeviceProperties") or {}
        vulkan11 = properties.get("VkPhysicalDeviceVulkan11Properties") or {}
        vulkan12 = properties.get("VkPhysicalDeviceVulkan12Properties") or {}
        profiles = data.get("profiles") or {}
        profile = next(iter(profiles.values()), {}) if profiles else {}
        uuid = ""
        if isinstance(vulkan11.get("deviceUUID"), list):
            try:
                uuid = _uuid_from_bytes(vulkan11["deviceUUID"])
            except ValueError:
                uuid = ""
        return (
            {
                "name": str(props.get("deviceName") or ""),
                "device_type": str(props.get("deviceType") or ""),
                "api_version": _version(int(props.get("apiVersion") or 0)),
                "instance_api_version": str(profile.get("api-version") or ""),
                "driver_name": str(vulkan12.get("driverName") or ""),
                "driver_info": str(vulkan12.get("driverInfo") or ""),
                "driver_id": str(vulkan12.get("driverID") or ""),
                "driver_version_raw": int(props.get("driverVersion") or 0),
                "uuid": uuid,
                "vendor_id": int(props.get("vendorID") or 0),
                "device_id": int(props.get("deviceID") or 0),
            },
            "",
        )


def _pick_device(requested_uuid: str) -> tuple[dict | None, str, int | None]:
    last_error = ""
    indices: list[int | None] = [None, 0, 1, 2, 3]
    for index in indices:
        device, error = _read_vulkaninfo(index)
        if device is None:
            last_error = error
            continue
        if not device["name"]:
            last_error = error or "vulkaninfo returned no device name"
            continue
        if requested_uuid and device["uuid"].lower() != requested_uuid.lower():
            continue
        return device, "", index
    return None, last_error or f"no Vulkan device matched UUID {requested_uuid!r}", None


def _is_physical(device: dict) -> bool:
    haystack = f"{device.get('name','')} {device.get('driver_name','')}".lower()
    if any(marker in haystack for marker in SOFTWARE_MARKERS):
        return False
    return device.get("device_type") != "VK_PHYSICAL_DEVICE_TYPE_CPU"


def _count_validation(output: str) -> tuple[int, int]:
    summary = VALIDATION_SUMMARY_RE.search(output)
    if summary:
        return int(summary.group("errors")), int(summary.group("warnings"))
    errors = len(VALIDATION_ERROR_RE.findall(output))
    warnings = len(re.findall(r"WARNING-", output))
    return errors, warnings


def _spike_run(device_uuid: str, timeout: int) -> dict:
    crate = ROOT / "spike" / "vk-bringup"
    binary = crate / "target" / "release" / "vk-bringup"
    started = time.monotonic()
    if not crate.exists():
        return {"status": "not_run", "reason": "spike/vk-bringup does not exist", "seconds": 0.0}
    if not binary.exists():
        build = _run(["cargo", "build", "--release"], cwd=crate, timeout=timeout)
        if build.returncode != 0:
            return {"status": "not_run", "reason": f"spike build failed: {build.stdout.strip()[-500:]}", "seconds": time.monotonic() - started}
    command = [str(binary), "--device-uuid", device_uuid, "--validation"]
    result = _run(command, env={"VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation"}, timeout=timeout)
    seconds = time.monotonic() - started
    errors, warnings = _count_validation(result.stdout)
    passed = result.returncode == 0 and errors == 0 and "result: PASS" in result.stdout
    return {
        "status": "passed" if passed else "failed",
        "reason": "raw Vulkan bring-up only; NOT fastnn backend evidence",
        "command": " ".join(command),
        "exit_code": result.returncode,
        "validation_errors": errors,
        "validation_warnings": warnings,
        "seconds": round(seconds, 3),
        "output_tail": result.stdout.strip().splitlines()[-12:],
    }


def collect(args: argparse.Namespace) -> int:
    if os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("FN3_ALLOW_HOSTED_GPU") != "1":
        print(
            "refusing to produce physical-GPU evidence on a hosted runner. "
            "Run tools/ci/run-gpu-lane.sh on a trusted local/protected machine; "
            "hosted CI must never execute untrusted code on a GPU.",
            file=sys.stderr,
        )
        return EXIT_FAIL

    started = time.monotonic()
    timings: dict[str, float] = {}
    checks: list[dict] = []
    commands: list[str] = []

    requested_uuid = (args.device_uuid or os.environ.get("FN3_VK_DEVICE_UUID") or "").strip()
    if requested_uuid and UUID_BYTES_RE.match(requested_uuid.replace("-", "")):
        raw = requested_uuid.replace("-", "")
        requested_uuid = f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"

    step = time.monotonic()
    device, error, index = _pick_device(requested_uuid)
    timings["device_probe_seconds"] = round(time.monotonic() - step, 3)
    commands.append("vulkaninfo -j" if index is None else f"vulkaninfo -j={index}")

    if device is None:
        print(f"no physical Vulkan device found: {error}", file=sys.stderr)
        record = {
            "schema": 1,
            "kind": "fastnn-gpu-evidence",
            "lane": "gpu-physical",
            "ci_only": False,
            "generated_utc": _now(),
            "host": {"hostname": socket.gethostname(), "os": platform.platform(), "arch": platform.machine()},
            "git": {
                "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
                "head": _git(["rev-parse", "HEAD"]),
                "dirty": bool(_git(["status", "--porcelain"])),
            },
            "device": {"kind": "none", "name": "", "uuid": "", "device_type": "", "driver_version": "", "api_version": ""},
            "validation": {"enabled": False, "errors": 0, "warnings": 0, "result": "not_run"},
            "checks": [{"name": "physical_device", "status": "failed", "detail": error}],
            "timings": {"total_seconds": round(time.monotonic() - started, 3)},
            "result": {"status": "not_verified", "reason": f"no physical Vulkan device: {error}"},
            "commands": ["vulkaninfo -j"],
            "auxiliary": {},
        }
        _write(args, record)
        return EXIT_NOT_VERIFIED

    physical = _is_physical(device)

    record: dict = {
        "schema": 1,
        "kind": "fastnn-gpu-evidence",
        "lane": "gpu-physical",
        "ci_only": False,
        "generated_utc": _now(),
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.platform(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        },
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head": _git(["rev-parse", "HEAD"]),
            "dirty": bool(_git(["status", "--porcelain"])),
        },
        "device": {
            "kind": "physical" if physical else "software",
            "name": device["name"],
            "uuid": device["uuid"],
            "device_type": device["device_type"],
            "driver_name": device["driver_name"],
            "driver_version": device["driver_info"] or str(device["driver_version_raw"]),
            "api_version": device["api_version"],
            "instance_api_version": device["instance_api_version"],
            "vendor_id": device["vendor_id"],
            "device_id": device["device_id"],
            "icd": os.environ.get("VK_ICD_FILENAMES", "loader default"),
            "enumeration_index": index,
        },
        "validation": {
            "enabled": False,
            "layer": "VK_LAYER_KHRONOS_validation",
            "errors": 0,
            "warnings": 0,
            "result": "not_run",
            "source": "none",
        },
        "checks": checks,
        "timings": timings,
        "result": {"status": "not_verified", "reason": ""},
        "commands": commands,
        "auxiliary": {},
    }

    if not physical:
        checks.append({"name": "physical_device", "status": "failed", "detail": f"{device['name']} is not a physical GPU"})
        record["result"] = {
            "status": "not_verified",
            "reason": f"selected Vulkan device {device['name']!r} ({device['device_type']}) is software/CPU; a physical GPU is required",
        }
        record["timings"]["total_seconds"] = round(time.monotonic() - started, 3)
        _write(args, record)
        return EXIT_NOT_VERIFIED

    checks.append({"name": "physical_device", "status": "passed", "detail": f"{device['name']} [{device['uuid']}]"})

    # The only check that can make this record "passed".
    feature_declared = bool(re.search(r"^\s*vulkan\s*=", (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.MULTILINE))
    smoke_test = ROOT / "tests" / "fn3_vulkan_smoke.rs"
    if feature_declared and smoke_test.exists():
        env = {
            "FN3_VK_REQUIRE_PHYSICAL": "1",
            "FN3_VK_DEVICE_UUID": device["uuid"],
        }
        command = [
            "cargo",
            "test",
            "--no-default-features",
            "--features",
            "vulkan",
            "--test",
            "fn3_vulkan_smoke",
            "--",
            "--test-threads=1",
            "--nocapture",
        ]
        step = time.monotonic()
        run = _run(command, env=env, timeout=args.timeout_seconds)
        seconds = time.monotonic() - step
        errors, warnings = _count_validation(run.stdout)
        # Validation is enabled in code (VulkanConfig::validated) and asserted by
        # the smoke test. Forcing VK_INSTANCE_LAYERS here would duplicate the
        # layer and make the loader emit 'env var VK_INSTANCE_LAYERS defined'
        # warnings, which the teardown test's zero-delta guard counts as a leak.
        # The two FN3_VK_* variables make the run demand the record's device.
        commands.append(
            f"FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_DEVICE_UUID={device['uuid']} " + " ".join(command)
        )
        smoke_passed = run.returncode == 0 and errors == 0
        checks.append(
            {
                "name": "fastnn_vulkan_smoke",
                "status": "passed" if smoke_passed else "failed",
                "detail": f"exit {run.returncode}, validation errors {errors}, warnings {warnings}",
                "reproduce": commands[-1],
                "output_tail": run.stdout.strip().splitlines()[-20:],
            }
        )
        record["validation"] = {
            "enabled": True,
            "layer": "VK_LAYER_KHRONOS_validation",
            "errors": errors,
            "warnings": warnings,
            "result": "clean" if errors == 0 else "errors",
            "source": "fastnn_vulkan_smoke",
        }
        timings["fastnn_vulkan_smoke_seconds"] = round(seconds, 3)
        if smoke_passed:
            record["result"] = {"status": "passed", "reason": "fastnn Vulkan smoke passed on the named physical device with validation enabled"}
        else:
            record["result"] = {"status": "failed", "reason": "fastnn Vulkan smoke failed or validation reported errors"}
    else:
        reason_parts = []
        if not feature_declared:
            reason_parts.append("Cargo.toml declares no 'vulkan' feature")
        if not smoke_test.exists():
            reason_parts.append("tests/fn3_vulkan_smoke.rs does not exist")
        checks.append(
            {
                "name": "fastnn_vulkan_smoke",
                "status": "not_run",
                "detail": "Vulkan backend not yet implemented (FN3-11 / #57): " + "; ".join(reason_parts),
            }
        )
        record["result"] = {
            "status": "not_verified",
            "reason": "Vulkan backend not yet implemented (FN3-11 / #57): " + "; ".join(reason_parts),
        }

    # Auxiliary, clearly labelled: raw Vulkan bring-up on the same device.
    if not args.no_spike:
        step = time.monotonic()
        spike = _spike_run(device["uuid"], args.timeout_seconds)
        timings["auxiliary_spike_seconds"] = round(time.monotonic() - step, 3)
        record["auxiliary"]["spike_vk_bringup"] = spike
        if spike.get("status") == "passed" and record["result"]["status"] != "passed":
            record["validation"] = {
                "enabled": True,
                "layer": "VK_LAYER_KHRONOS_validation",
                "errors": int(spike.get("validation_errors", 0)),
                "warnings": int(spike.get("validation_warnings", 0)),
                "result": "clean" if int(spike.get("validation_errors", 0)) == 0 else "errors",
                "source": "spike/vk-bringup auxiliary run (NOT fastnn backend evidence)",
            }
            checks.append(
                {
                    "name": "auxiliary_spike_bringup",
                    "status": "passed",
                    "detail": "raw Vulkan loader/compute/validation bring-up; proves nothing about the fastnn backend",
                }
            )

    record["timings"]["total_seconds"] = round(time.monotonic() - started, 3)
    _write(args, record)
    return {"passed": EXIT_OK, "failed": EXIT_FAIL}.get(record["result"]["status"], EXIT_NOT_VERIFIED)


def _write(args: argparse.Namespace, record: dict) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nevidence record: {out}")
    print(json.dumps({"result": record["result"], "device": record["device"], "validation": record["validation"]}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device-uuid", default="")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--no-spike", action="store_true")
    return collect(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
