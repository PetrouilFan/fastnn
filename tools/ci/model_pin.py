#!/usr/bin/env python3
"""Read the single consumer-side model pin for CI.

CI must never guess which model fixtures a run used. ``tools/ci/model-pin.json``
is the only file the pinned-model lane reads, and FN3-03 (#49) owns the
revision it points at (``fixtures/``).

Absence is never success. Every non-pinned state exits 69 (EX_UNAVAILABLE) with
a message naming the pin file and FN3-03 (#49), so the lane can report
NOT VERIFIED instead of pretending to pass.

Usage:
    python3 tools/ci/model_pin.py check [--pin PATH] [--json]
    python3 tools/ci/model_pin.py seal  [--pin PATH]   # fill fixtures_manifest_sha256
    python3 tools/ci/model_pin.py selftest

Exit codes: 0 pinned and consistent, 69 not verified (missing/unpinned),
1 invalid or inconsistent pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIN = ROOT / "tools" / "ci" / "model-pin.json"
OWNER = "FN3-03 (#49)"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
EXIT_OK = 0
EXIT_INVALID = 1
EXIT_NOT_VERIFIED = 69


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(pin_path: Path) -> tuple[int, str, dict]:
    """Return (exit_code, message, details)."""
    if not pin_path.exists():
        return (
            EXIT_NOT_VERIFIED,
            f"NOT VERIFIED: {_relative(pin_path)} is missing; the pinned model revision is owned by {OWNER}.",
            {"status": "missing", "pin": _relative(pin_path)},
        )
    try:
        pin = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return (EXIT_INVALID, f"INVALID PIN: {_relative(pin_path)}: {error}", {"status": "invalid"})
    if not isinstance(pin, dict):
        return (EXIT_INVALID, f"INVALID PIN: {_relative(pin_path)} must contain a JSON object", {"status": "invalid"})

    status = pin.get("status")
    details = {
        "status": status,
        "pin": _relative(pin_path),
        "revision": pin.get("revision"),
        "fixtures_manifest": pin.get("fixtures_manifest"),
    }
    if status != "pinned":
        return (
            EXIT_NOT_VERIFIED,
            f"NOT VERIFIED: {_relative(pin_path)} has status={status!r}, not 'pinned'; "
            f"no pinned model revision exists yet ({OWNER}).",
            details,
        )

    revision = pin.get("revision")
    if not isinstance(revision, str) or not REVISION_RE.match(revision):
        return (EXIT_INVALID, f"INVALID PIN: {_relative(pin_path)} revision must be a 40-hex commit, got {revision!r}", details)

    manifest_name = pin.get("fixtures_manifest")
    if not isinstance(manifest_name, str) or not manifest_name:
        return (EXIT_INVALID, f"INVALID PIN: {_relative(pin_path)} must name fixtures_manifest", details)
    manifest = ROOT / manifest_name
    if not manifest.exists():
        return (
            EXIT_INVALID,
            f"INVALID PIN: {_relative(pin_path)} is pinned but {manifest_name} does not exist; "
            f"{OWNER} must publish the fixtures manifest.",
            details,
        )

    expected = pin.get("fixtures_manifest_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return (
            EXIT_INVALID,
            f"INVALID PIN: {_relative(pin_path)} fixtures_manifest_sha256 must be a 64-hex digest; "
            "run 'python3 tools/ci/model_pin.py seal' after the fixtures are published.",
            details,
        )
    actual = _sha256(manifest)
    details["fixtures_manifest_sha256"] = actual
    if actual != expected:
        return (
            EXIT_INVALID,
            f"INVALID PIN: {manifest_name} sha256 {actual} does not match the pin {expected}; "
            "the fixtures changed without a new pin.",
            details,
        )

    # The manifest is a model lock, so it names the revision too. Two files that
    # disagree mean one of them was edited alone, and the lane would then fetch
    # one revision while claiming another.
    try:
        manifest_revision = json.loads(manifest.read_text(encoding="utf-8")).get("revision")
    except (OSError, json.JSONDecodeError) as error:
        return (EXIT_INVALID, f"INVALID PIN: {manifest_name}: {error}", details)
    if manifest_revision != revision:
        return (
            EXIT_INVALID,
            f"INVALID PIN: {manifest_name} pins revision {manifest_revision!r} but "
            f"{_relative(pin_path)} says {revision!r}; one of the two was edited alone.",
            details,
        )

    return (EXIT_OK, f"PINNED: {_relative(pin_path)} revision={revision} fixtures={manifest_name} sha256={actual}", details)


def seal(pin_path: Path) -> int:
    """Fill fixtures_manifest_sha256 from the manifest named by the pin."""
    if not pin_path.exists():
        print(f"cannot seal: {_relative(pin_path)} does not exist", file=sys.stderr)
        return EXIT_INVALID
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("status") != "pinned":
        print(f"cannot seal: {_relative(pin_path)} status is {pin.get('status')!r}, not 'pinned'", file=sys.stderr)
        return EXIT_INVALID
    manifest_name = pin.get("fixtures_manifest")
    if not isinstance(manifest_name, str) or not (ROOT / manifest_name).exists():
        print(f"cannot seal: fixtures manifest {manifest_name!r} does not exist", file=sys.stderr)
        return EXIT_INVALID
    pin["fixtures_manifest_sha256"] = _sha256(ROOT / manifest_name)
    pin_path.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"sealed {_relative(pin_path)} fixtures_manifest_sha256={pin['fixtures_manifest_sha256']}")
    return EXIT_OK


def selftest() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        missing = temp_path / "model-pin.json"
        code, message, _ = check(missing)
        if code != EXIT_NOT_VERIFIED or "missing" not in message or "#49" not in message:
            failures.append(f"missing pin: expected exit 69 naming the file and #49, got {code}: {message}")

        unpinned = temp_path / "unpinned.json"
        unpinned.write_text(json.dumps({"schema": 1, "status": "unpinned"}), encoding="utf-8")
        code, message, _ = check(unpinned)
        if code != EXIT_NOT_VERIFIED or "status='unpinned'" not in message:
            failures.append(f"unpinned: expected exit 69, got {code}: {message}")

        bad_revision = temp_path / "bad.json"
        bad_revision.write_text(
            json.dumps({"schema": 1, "status": "pinned", "revision": "main", "fixtures_manifest": "x.json"}),
            encoding="utf-8",
        )
        code, message, _ = check(bad_revision)
        if code != EXIT_INVALID or "40-hex" not in message:
            failures.append(f"bad revision: expected exit 1, got {code}: {message}")

        # A pinned pin with a matching manifest hash and revision must pass.
        manifest = temp_path / "manifest.json"
        manifest.write_text(
            json.dumps({"schema": 1, "revision": "0" * 40}), encoding="utf-8"
        )
        pinned = temp_path / "pinned.json"
        pinned.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "status": "pinned",
                    "revision": "0" * 40,
                    "fixtures_manifest": str(manifest),
                    "fixtures_manifest_sha256": _sha256(manifest),
                }
            ),
            encoding="utf-8",
        )
        code, message, _ = check(pinned)
        if code != EXIT_OK or not message.startswith("PINNED"):
            failures.append(f"pinned: expected exit 0, got {code}: {message}")

        manifest.write_text(
            json.dumps({"schema": 1, "revision": "0" * 40, "changed": True}),
            encoding="utf-8",
        )
        code, message, _ = check(pinned)
        if code != EXIT_INVALID or "changed without a new pin" not in message:
            failures.append(f"drifted manifest: expected exit 1, got {code}: {message}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("model pin selftest: 5 cases passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "seal"):
        command = sub.add_parser(name)
        command.add_argument("--pin", type=Path, default=DEFAULT_PIN)
        if name == "check":
            command.add_argument("--json", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args(argv)

    if args.command == "selftest":
        return selftest()
    if args.command == "seal":
        return seal(args.pin)

    code, message, details = check(args.pin)
    print(json.dumps({"code": code, "message": message, "details": details}, indent=2) if args.json else message)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
