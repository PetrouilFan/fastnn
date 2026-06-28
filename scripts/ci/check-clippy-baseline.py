#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_FILE = REPO_ROOT / "ci" / "clippy-baseline.txt"

if not BASELINE_FILE.exists():
    print(f"missing baseline file: {BASELINE_FILE}", file=sys.stderr)
    sys.exit(1)

cmd = [
    "cargo",
    "clippy",
    "--lib",
    "--tests",
    "--message-format=json",
    "--",
    "-D",
    "warnings",
]

proc = subprocess.run(
    cmd,
    cwd=REPO_ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

entries = set()
noise_lines = []
for line in proc.stdout.splitlines():
    stripped = line.strip()
    if not stripped:
        continue
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        noise_lines.append(line)
        continue

    if payload.get("reason") != "compiler-message":
        continue

    message = payload.get("message", {})
    level = message.get("level")
    if level not in {"warning", "error"}:
        continue

    code = (message.get("code") or {}).get("code") or level
    rendered_message = (message.get("message") or "").strip()
    span_path = "<unknown>"
    for span in message.get("spans", []):
        if span.get("is_primary") and span.get("file_name"):
            span_path = span["file_name"]
            break
    entries.add(f"{code}|{span_path}|{rendered_message}")

baseline_entries = {
    line.strip()
    for line in BASELINE_FILE.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}

new_entries = sorted(entries - baseline_entries)
retired_entries = sorted(baseline_entries - entries)

if retired_entries:
    print("Clippy baseline entries no longer needed:")
    for entry in retired_entries:
        print(f"  {entry}")

if new_entries:
    print("New clippy diagnostics outside the baseline:", file=sys.stderr)
    for entry in new_entries:
        print(f"  {entry}", file=sys.stderr)
    if noise_lines:
        print("", file=sys.stderr)
        print("Non-JSON cargo/clippy output:", file=sys.stderr)
        for line in noise_lines:
            print(line, file=sys.stderr)
    sys.exit(1)

print(f"clippy baseline check passed ({len(entries)} tracked diagnostics, no new ones).")
if proc.returncode not in (0, 101):
    print(f"note: cargo clippy exited with unexpected status {proc.returncode}", file=sys.stderr)
