#!/usr/bin/env python3
"""Record and render FastNN CI lane status so absence can never look like success.

Each lane job calls ``record`` with an explicit status. The ``lane-summary`` job
calls ``render``, which reads ``tools/ci/lanes.json`` and the GitHub ``needs``
context, then:

* renders every lane, including lanes that produced no record, as
  ``NOT VERIFIED`` with a reason;
* fails when a lane that applies to this tree today is not ``passed``;
* refuses to let a hosted job claim physical-GPU evidence;
* prints the exact local reproduction command for every failure.

Usage:
    python3 tools/ci/lane_status.py record --lane cpu --job cpu-rust \\
        --status passed --detail "..."
    python3 tools/ci/lane_status.py render --needs-json "$NEEDS_JSON"
    python3 tools/ci/lane_status.py resolve
    python3 tools/ci/lane_status.py selftest

Exit codes: 0 all applicable lanes passed, 1 a required lane did not pass or a
record is invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_evidence  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STATUSES = ("passed", "failed", "not_verified")
EXIT_OK = 0
EXIT_FAIL = 1
NOT_VERIFIED = "not_verified"
JOB_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def lanes_path(root: Path = ROOT) -> Path:
    return root / "tools" / "ci" / "lanes.json"


def load_lanes(root: Path = ROOT) -> list[dict]:
    data = json.loads(lanes_path(root).read_text(encoding="utf-8"))
    lanes = data.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise ValueError(f"{lanes_path(root)} must contain a non-empty 'lanes' list")
    names = [lane.get("name") for lane in lanes]
    if len(set(names)) != len(names):
        raise ValueError(f"{lanes_path(root)} has duplicate lane names: {names}")
    return lanes


def lane_by_name(name: str, root: Path = ROOT) -> dict:
    for lane in load_lanes(root):
        if lane.get("name") == name:
            return lane
    raise KeyError(f"unknown lane {name!r}; add it to {lanes_path(root)}")


def _cargo_feature_declared(root: Path, feature: str) -> bool:
    cargo = root / "Cargo.toml"
    if not cargo.exists():
        return False
    return re.search(rf"^\s*{re.escape(feature)}\s*=", cargo.read_text(encoding="utf-8"), re.MULTILINE) is not None


def _pin_status(root: Path, pin_file: str) -> str:
    pin = root / pin_file
    if not pin.exists():
        return "missing"
    try:
        data = json.loads(pin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    return str(data.get("status", "invalid")) if isinstance(data, dict) else "invalid"


def effective_required(lane: dict, root: Path = ROOT) -> tuple[bool, str]:
    """Return (required_now, human-readable reason)."""
    if lane.get("required") is True:
        return True, "lane is required for every run"
    condition = lane.get("required_when")
    if not condition:
        return False, "lane is optional and has no automatic trigger"

    reasons: list[str] = []
    satisfied = True
    if "cargo_feature" in condition:
        feature = str(condition["cargo_feature"])
        present = _cargo_feature_declared(root, feature)
        satisfied = satisfied and present
        reasons.append(f"Cargo feature {feature!r} is {'declared' if present else 'not declared'}")
    if "path" in condition:
        relative = str(condition["path"])
        present = (root / relative).exists()
        satisfied = satisfied and present
        reasons.append(f"{relative} {'exists' if present else 'does not exist'}")
    if "pin_status" in condition:
        expected = str(condition["pin_status"])
        pin_file = str(condition.get("pin_file", "tools/ci/model-pin.json"))
        actual = _pin_status(root, pin_file)
        satisfied = satisfied and actual == expected
        reasons.append(f"{pin_file} status is {actual!r}")
    return satisfied, "; ".join(reasons)


def status_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get("LANE_STATUS_DIR"):
        return Path(os.environ["LANE_STATUS_DIR"])
    return ROOT / "target" / "ci-lane-status"


def record_file(directory: Path, lane: str, job: str) -> Path:
    safe_job = JOB_RE.sub("-", job).strip("-") or "job"
    return directory / f"{lane}__{safe_job}.json"


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def command_record(args: argparse.Namespace) -> int:
    if args.status not in STATUSES:
        print(f"invalid status {args.status!r}; expected one of {STATUSES}", file=sys.stderr)
        return EXIT_FAIL
    lane = lane_by_name(args.lane)
    directory = status_dir(args.status_dir)
    directory.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []
    for raw in args.check or []:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            print(f"--check must be JSON: {raw!r}: {error}", file=sys.stderr)
            return EXIT_FAIL
        if not isinstance(parsed, dict) or "name" not in parsed or parsed.get("status") not in ("passed", "failed", "not_run"):
            print(f"--check needs name and status in passed|failed|not_run: {raw!r}", file=sys.stderr)
            return EXIT_FAIL
        checks.append(parsed)

    evidence = [str(Path(item)) for item in (args.evidence or [])]
    if args.status == "passed" and lane.get("evidence_required") is True:
        if not evidence:
            print(
                f"lane {args.lane!r} requires --evidence: a pass without a validated evidence record is not allowed",
                file=sys.stderr,
            )
            return EXIT_FAIL
        for item in evidence:
            try:
                errors = verify_evidence.validate_record(verify_evidence.load_record(Path(item)))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors = [f"cannot read evidence: {error}"]
            if errors:
                print(f"evidence record {item} is not acceptable: {errors}", file=sys.stderr)
                return EXIT_FAIL

    record = {
        "schema": 1,
        "lane": args.lane,
        "job": args.job,
        "status": args.status,
        "detail": args.detail,
        "checks": checks,
        "reproduce": args.reproduce or lane.get("reproduce"),
        "evidence": evidence,
        "duration_seconds": args.duration_seconds,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"])},
    }
    target = record_file(directory, args.lane, args.job)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    line = f"- `{args.lane}` / `{args.job}`: **{args.status.replace('_', ' ').upper()}** — {args.detail}"
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return EXIT_OK


def _merge_records(records: list[dict]) -> tuple[str, str]:
    statuses = [record.get("status") for record in records]
    if "failed" in statuses:
        merged = "failed"
    elif all(status == "passed" for status in statuses):
        merged = "passed"
    else:
        merged = NOT_VERIFIED
    detail = "; ".join(
        f"{record.get('job')}: {record.get('status')} ({record.get('detail')})" for record in records
    )
    return merged, detail


def _job_reported(job: str, recorded_jobs: set[str]) -> bool:
    # Matrix jobs record one file per matrix entry (packaging-ubuntu-latest),
    # while `needs` reports the base job id (packaging).
    return any(recorded == job or recorded.startswith(job + "-") for recorded in recorded_jobs)


def _job_fallback(lane: dict, needs: dict, recorded_jobs: set[str]) -> tuple[str, str] | None:
    """Compare mapped job results with the records actually produced."""
    observed = [(job, (needs.get(job) or {}).get("result")) for job in lane.get("jobs", []) if job in needs]
    if not observed:
        return None
    failed = [job for job, result in observed if result == "failure"]
    if failed:
        return "failed", f"job(s) {', '.join(failed)} failed (no passing status record for them)"
    cancelled = [job for job, result in observed if result == "cancelled"]
    if cancelled:
        return NOT_VERIFIED, f"job(s) {', '.join(cancelled)} were cancelled"
    unreported = [job for job, _ in observed if not _job_reported(job, recorded_jobs)]
    if not unreported:
        return None
    skipped = [job for job in unreported if (needs.get(job) or {}).get("result") == "skipped"]
    if len(skipped) == len(unreported):
        return NOT_VERIFIED, f"job(s) {', '.join(skipped)} were skipped and produced no status record"
    return NOT_VERIFIED, f"job(s) {', '.join(unreported)} produced no status record"


def _escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _load_records(directory: Path) -> dict[str, list[dict]]:
    records_by_lane: dict[str, list[dict]] = {}
    if not directory.exists():
        return records_by_lane
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"ignoring unreadable lane status {path}: {error}", file=sys.stderr)
            continue
        if isinstance(record, dict) and record.get("lane"):
            records_by_lane.setdefault(record["lane"], []).append(record)
    return records_by_lane


def _render(directory: Path, needs: dict) -> tuple[int, str, list[str]]:
    records_by_lane = _load_records(directory)
    rows: list[dict] = []
    failures: list[str] = []

    for lane in load_lanes():
        name = lane["name"]
        records = records_by_lane.get(name, [])
        recorded_jobs = {record.get("job") for record in records}
        fallback = _job_fallback(lane, needs, recorded_jobs)

        if records:
            status, detail = _merge_records(records)
            if fallback:
                extra_status, extra_detail = fallback
                detail = f"{detail}; {extra_detail}"
                if extra_status == "failed" or (extra_status == NOT_VERIFIED and status == "passed"):
                    status = extra_status
        elif fallback:
            status, detail = fallback
        else:
            status, detail = NOT_VERIFIED, "no status record was produced and no job result was reported"

        required, reason = effective_required(lane, ROOT)

        # Mechanical honesty guards.
        if lane.get("hosted") is True and "physical gpu" in " ".join(lane.get("proves", [])).lower():
            failures.append(f"lane {name!r} is hosted but claims physical-GPU proof; remove the claim or the job")
        if name == "gpu-physical" and status == "passed":
            evidence = [item for record in records for item in record.get("evidence", [])]
            if not evidence:
                status = NOT_VERIFIED
                detail = "lane claimed passed without an evidence record"
            else:
                for item in evidence:
                    try:
                        errors = verify_evidence.validate_record(verify_evidence.load_record(Path(item)))
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        errors = [f"cannot read evidence: {error}"]
                    if errors:
                        failures.append(f"gpu-physical evidence {item} is invalid: {errors}")

        if required and status != "passed":
            failures.append(f"required lane {name!r} is {status.upper()}: {detail or reason}")
        if lane.get("software_only") is True and status == "passed":
            detail = "CI-ONLY software ICD run; not physical GPU evidence. " + detail
        rows.append(
            {
                "lane": name,
                "title": lane.get("title", ""),
                "status": status,
                "required": required,
                "reason": reason,
                "detail": detail,
                "proves": " ".join(lane.get("proves", [])),
                "does_not_prove": " ".join(lane.get("does_not_prove", [])),
                "reproduce": lane.get("reproduce", ""),
                "records": records,
            }
        )

    lines = [
        "## FastNN CI lane status",
        "",
        "A lane is evidence only for what its **Proves** column says. "
        "**NOT VERIFIED** means the check did not run or produced no record — it is not a pass.",
        "",
        "### Status",
        "",
        "| Lane | Status | Applies today | Detail |",
        "|---|---|---|---|",
    ]
    for row in rows:
        applies = ("yes — " if row["required"] else "no — ") + row["reason"]
        lines.append(
            f"| {_escape(row['lane'])} ({_escape(row['title'])}) | **{row['status'].replace('_', ' ').upper()}** | "
            f"{_escape(applies)} | {_escape(row['detail'])} |"
        )
    lines += ["", "### What each lane proves — and does not prove", "", "| Lane | Proves | Does NOT prove |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {_escape(row['lane'])} | {_escape(row['proves'])} | {_escape(row['does_not_prove'])} |")
    lines.append("")

    failed_rows = [row for row in rows if row["status"] == "failed"]
    if failed_rows:
        lines += ["### Reproduce failures locally", ""]
        for row in failed_rows:
            lines.append(f"- `{row['lane']}`: `{row['reproduce']}`")
            for record in row["records"]:
                for check in record.get("checks", []):
                    if check.get("status") == "failed" and check.get("reproduce"):
                        lines.append(f"  - {check.get('name')}: `{check['reproduce']}`")
        lines.append("")

    return (EXIT_FAIL if failures else EXIT_OK), "\n".join(lines), failures


def command_render(args: argparse.Namespace) -> int:
    needs: dict = {}
    if args.needs_json:
        try:
            needs = json.loads(args.needs_json)
        except json.JSONDecodeError as error:
            print(f"cannot parse --needs-json: {error}", file=sys.stderr)
            return EXIT_FAIL

    code, text, failures = _render(status_dir(args.status_dir), needs)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    for failure in failures:
        print(f"::error::{failure}", file=sys.stderr)
    if failures:
        print("\nLane gate failed; see the table above and tools/ci/README.md.", file=sys.stderr)
    return code


def command_resolve(_args: argparse.Namespace) -> int:
    resolved = [
        {"lane": lane["name"], "required_now": effective_required(lane, ROOT)[0], "reason": effective_required(lane, ROOT)[1]}
        for lane in load_lanes()
    ]
    print(json.dumps(resolved, indent=2))
    return EXIT_OK


def command_describe(args: argparse.Namespace) -> int:
    lane = lane_by_name(args.lane)
    required, reason = effective_required(lane, ROOT)
    print(f"LANE {lane['name']} — {lane.get('title', '')}")
    print(f"  PROVES: {' '.join(lane.get('proves', []))}")
    print(f"  DOES NOT PROVE: {' '.join(lane.get('does_not_prove', []))}")
    print(f"  APPLIES TODAY: {'yes' if required else 'no'} ({reason})")
    print(f"  REPRODUCE: {lane.get('reproduce', '')}")
    return EXIT_OK


def selftest() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)

        def render(needs: dict) -> tuple[int, str]:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code, text, _ = _render(directory, needs)
            return code, text

        def write(lane: str, job: str, status: str) -> None:
            record_file(directory, lane, job).write_text(
                json.dumps({"lane": lane, "job": job, "status": status, "detail": "selftest"}), encoding="utf-8"
            )

        def pass_all_required() -> dict:
            needs = {"cpu-rust": {"result": "success"}, "cpu-python": {"result": "success"}}
            for lane in load_lanes():
                if effective_required(lane)[0]:
                    for job in lane.get("jobs", []):
                        needs[job] = {"result": "success"}
                        write(lane["name"], job, "passed")
            return needs

        cpu = lane_by_name("cpu")
        if not effective_required(cpu)[0]:
            failures.append("cpu lane must be required")
        if lane_by_name("gpu-physical").get("hosted") is not False:
            failures.append("gpu-physical lane must declare hosted=false")
        if effective_required(lane_by_name("gpu-physical"))[0]:
            failures.append("gpu-physical lane must not apply to hosted CI")

        all_success = {"cpu-rust": {"result": "success"}, "cpu-python": {"result": "success"}}
        code, text = render(all_success)
        if code == 0:
            failures.append("render accepted a required lane with no records")
        if "**NOT VERIFIED**" not in text:
            failures.append("render did not print NOT VERIFIED for a missing required lane")

        needs = pass_all_required()
        code, _ = render(needs)
        if code != 0:
            failures.append("render rejected fully passed required lane records")

        record_file(directory, "cpu", "cpu-python").unlink()
        code, text = render({**needs, "cpu-python": {"result": "failure"}})
        if code == 0 or "**FAILED**" not in text:
            failures.append("render accepted a failed job with no record")

        write("cpu", "cpu-python", "passed")
        write("gpu-physical", "local", "passed")
        code, text = render(needs)
        gpu_row = next(line for line in text.splitlines() if line.startswith("| gpu-physical"))
        if "**PASSED**" in gpu_row:
            failures.append("gpu-physical rendered PASSED without an evidence record")

        write("vulkan-software-smoke", "software", "passed")
        code, text = render(needs)
        software_row = next(line for line in text.splitlines() if line.startswith("| vulkan-software-smoke"))
        if "physical gpu" not in software_row.lower():
            failures.append("software lane row does not state that it is not physical-GPU evidence")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return EXIT_FAIL
    print("lane status selftest: passed")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record")
    record.add_argument("--lane", required=True)
    record.add_argument("--job", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--detail", required=True)
    record.add_argument("--check", action="append", metavar="JSON")
    record.add_argument("--reproduce")
    record.add_argument("--evidence", action="append", metavar="PATH")
    record.add_argument("--duration-seconds", type=float)
    record.add_argument("--status-dir")

    render = sub.add_parser("render")
    render.add_argument("--status-dir")
    render.add_argument("--needs-json")

    sub.add_parser("resolve")
    describe = sub.add_parser("describe")
    describe.add_argument("--lane", required=True)
    sub.add_parser("selftest")

    args = parser.parse_args(argv)
    if args.command == "record":
        return command_record(args)
    if args.command == "render":
        return command_render(args)
    if args.command == "resolve":
        return command_resolve(args)
    if args.command == "describe":
        return command_describe(args)
    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
