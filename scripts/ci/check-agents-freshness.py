#!/usr/bin/env python3
"""Reject volatile or broken repository-local AGENTS.md guidance."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_LINES = 100
VERSION_RE = re.compile(r"(?<![A-Za-z0-9_])v?\d+\.\d+(?:\.\d+)?(?![A-Za-z0-9_])")
VOLATILE_COUNT_RE = re.compile(
    r"\b\d+[\s-]+(?:source\s+)?(?:files?|modules?|lines?|loc)\b", re.IGNORECASE
)
SOURCE_LINE_RE = re.compile(r"\b(?:lines?|loc)\s+\d+(?:\s*[-–]\s*\d+)?\b", re.IGNORECASE)
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
PATH_PREFIXES = (
    ".github/",
    "benches/",
    "docs/",
    "examples/",
    "fastnn/",
    "scripts/",
    "src/",
    "tests/",
)
ROOT_PATHS = {"Cargo.toml", "Makefile", "pyproject.toml", "pytest.ini"}
DISALLOWED_COMMANDS = ("cargo tarpaulin", "python setup.py", "pip install -e .")


def tracked_agent_guides() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "*AGENTS.md"], cwd=ROOT, text=True
    )
    return [ROOT / line for line in output.splitlines() if line]


def candidate_path(token: str) -> str | None:
    token = token.strip().rstrip(".,:;)")
    if token in ROOT_PATHS or token.startswith(PATH_PREFIXES):
        if any(char in token for char in "*{}[]"):
            return None
        return token
    return None


def check_guide(path: Path) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []

    line_count = len(text.splitlines())
    if line_count > MAX_LINES:
        failures.append(f"{relative}: {line_count} lines exceeds {MAX_LINES}")

    if match := VERSION_RE.search(text):
        failures.append(f"{relative}: volatile version literal {match.group(0)!r}")
    if match := VOLATILE_COUNT_RE.search(text):
        failures.append(f"{relative}: volatile count claim {match.group(0)!r}")
    if match := SOURCE_LINE_RE.search(text):
        failures.append(f"{relative}: volatile source-line claim {match.group(0)!r}")

    lower = text.lower()
    for command in DISALLOWED_COMMANDS:
        if command in lower:
            failures.append(f"{relative}: disallowed command {command!r}")

    for token in BACKTICK_RE.findall(text):
        referenced = candidate_path(token)
        if referenced is not None and not (ROOT / referenced).exists():
            failures.append(f"{relative}: missing referenced path {referenced}")

    return failures


def main() -> int:
    guides = tracked_agent_guides()
    if not guides:
        print("no tracked AGENTS.md guides found", file=sys.stderr)
        return 1

    failures = [failure for path in guides for failure in check_guide(path)]
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    print(f"checked {len(guides)} AGENTS.md guides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
