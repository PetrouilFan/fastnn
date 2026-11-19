#!/usr/bin/env python3
"""Fail when the release version drifts between the files that declare it.

Rust and Python spell a prerelease differently: Cargo/SemVer writes
``3.0.0-alpha.1`` and PEP 440 writes ``3.0.0a1``. Both spellings therefore have
to be normalised before they are compared, and every place that stores the
version is checked -- not merely the three that were edited by habit.

Checked locations:

* ``Cargo.toml`` ``[package] version`` -- the Rust crate version;
* ``Cargo.lock`` and ``fuzz/Cargo.lock`` ``[[package]] fastnn`` -- a lockfile
  that lags the manifest makes a rebuilt wheel disagree with its source;
* ``pyproject.toml`` ``[project] version`` -- the wheel version;
* ``uv.lock`` ``[[package]] fastnn`` -- the Python environment's view;
* ``fastnn/__init__.py`` ``__version__`` -- what ``import fastnn`` reports;
* ``CHANGELOG.md`` -- a release without notes is incomplete.

``--tag vX.Y.Z[-PRERELEASE]`` additionally requires the tag to match the
Cargo version exactly, which is what ``.github/workflows/release.yml`` needs.

Usage:

    python3 scripts/ci/check-versions.py
    python3 scripts/ci/check-versions.py --tag v3.0.0-alpha.1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")
PEP440_RE = re.compile(r"^(\d+\.\d+\.\d+)((?:a|b|rc)\d+)?$")
PRERELEASE_RE = re.compile(r"^(alpha|beta|rc)\.?(\d+)$")


def normalise(version: str) -> str | None:
    """Return the PEP 440 spelling of a SemVer or PEP 440 version.

    ``3.0.0-alpha.1`` -> ``3.0.0a1``; ``3.0.0a1`` -> ``3.0.0a1``;
    ``2.6.0`` -> ``2.6.0``. ``None`` means the input is neither spelling.
    """
    version = version.strip()
    if match := PEP440_RE.match(version):
        return version
    match = SEMVER_RE.match(version)
    if not match:
        return None
    major, minor, patch, pre = match.groups()
    base = f"{major}.{minor}.{patch}"
    if not pre:
        return base
    # SemVer allows dot-separated identifiers; only the PEP 440 alpha/beta/rc
    # forms are meaningful here and anything else is refused rather than
    # guessed at.
    pre_match = PRERELEASE_RE.match(pre)
    if not pre_match:
        return None
    kind, number = pre_match.groups()
    return f"{base}{ {'alpha': 'a', 'beta': 'b', 'rc': 'rc'}[kind] }{number}"


def cargo_version(text: str) -> str | None:
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def lock_version(text: str, package: str) -> str | None:
    pattern = re.compile(
        r'^\[\[package\]\]\s*\nname = "' + re.escape(package) + r'"\s*\nversion = "([^"]+)"',
        re.MULTILINE,
    )
    match = pattern.search(text)
    return match.group(1) if match else None


def python_version(text: str) -> str | None:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag",
        default="",
        help="release tag to validate (e.g. v3.0.0-alpha.1); the v prefix is optional",
    )
    parser.add_argument(
        "--print-normalised",
        action="store_true",
        help="print the PEP 440 spelling of the Cargo version and exit (for wheel names)",
    )
    parser.add_argument(
        "--root",
        default="",
        help="repository root to inspect (default: this checkout); used by the drift test",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else ROOT

    if args.print_normalised:
        cargo = cargo_version((root / "Cargo.toml").read_text(encoding="utf-8"))
        normalised = normalise(cargo) if cargo else None
        if normalised is None:
            print(f"Cargo.toml version {cargo!r} is not SemVer or PEP 440", file=sys.stderr)
            return 1
        print(normalised)
        return 0

    sources: list[tuple[str, str | None]] = []

    def add(relative: str, value: str | None) -> None:
        sources.append((relative, value))

    add("Cargo.toml", cargo_version((root / "Cargo.toml").read_text(encoding="utf-8")))
    add(
        "Cargo.lock",
        lock_version((root / "Cargo.lock").read_text(encoding="utf-8"), "fastnn"),
    )
    add(
        "fuzz/Cargo.lock",
        lock_version((root / "fuzz" / "Cargo.lock").read_text(encoding="utf-8"), "fastnn"),
    )
    add("pyproject.toml", cargo_version((root / "pyproject.toml").read_text(encoding="utf-8")))
    add(
        "uv.lock",
        lock_version((root / "uv.lock").read_text(encoding="utf-8"), "fastnn"),
    )
    add(
        "fastnn/__init__.py",
        python_version((root / "fastnn" / "__init__.py").read_text(encoding="utf-8")),
    )

    failures: list[str] = []
    canonical = sources[0][1]
    if canonical is None:
        failures.append("Cargo.toml has no [package] version")
    canonical_normalised = normalise(canonical) if canonical else None
    if canonical and canonical_normalised is None:
        failures.append(f"Cargo.toml version {canonical!r} is not SemVer or PEP 440")

    print(f"Cargo version:              {canonical}")
    print(f"Python-normalised version:  {canonical_normalised}")
    for relative, value in sources[1:]:
        if value is None:
            failures.append(f"{relative}: no fastnn version found")
            print(f"  {relative:<22} MISSING")
            continue
        normalised = normalise(value)
        ok = normalised is not None and normalised == canonical_normalised
        print(f"  {relative:<22} {value:<18} {'ok' if ok else 'DRIFT'}")
        if normalised is None:
            failures.append(f"{relative}: {value!r} is not SemVer or PEP 440")
        elif not ok:
            failures.append(
                f"{relative}: {value!r} normalises to {normalised!r}, expected "
                f"{canonical_normalised!r} (from Cargo.toml {canonical!r})"
            )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if canonical and f"## [{canonical}]" not in changelog:
        failures.append(f"CHANGELOG.md has no '## [{canonical}]' entry")
    else:
        print(f"  {'CHANGELOG.md':<22} '## [{canonical}]' present")

    if args.tag:
        tag_version = args.tag[1:] if args.tag.startswith("v") else args.tag
        if tag_version != canonical:
            failures.append(
                f"tag {args.tag!r} names {tag_version!r}, Cargo.toml says {canonical!r}; "
                "a prerelease tag must use the Cargo spelling (v3.0.0-alpha.1)"
            )
        elif normalise(tag_version) != canonical_normalised:
            failures.append(f"tag {args.tag!r} does not normalise to {canonical_normalised!r}")
        else:
            print(f"  {'tag':<22} {args.tag} ok")

    if failures:
        print("\nVERSION DRIFT:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"\nVERSIONS AGREE: {canonical} == {canonical_normalised} in every checked file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
