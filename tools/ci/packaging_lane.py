#!/usr/bin/env python3
"""Lane: packaging — build a wheel, install it in a clean venv, smoke it.

The wheel is installed into a fresh virtual environment and imported from a
temporary directory outside the checkout, so an editable install cannot pass by
accident. This proves packaging works; it says nothing about the Rust test
suites or GPU support.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANE = "packaging"


def _run(command: list[str], *, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env={**os.environ, **(env or {})},
        text=True,
        check=False,
    )


def _fail(job: str, detail: str, reproduce: str, checks: list[dict], status_dir: str | None) -> int:
    print(f"\nREPRODUCE: {reproduce}", file=sys.stderr)
    print(f"PACKAGING FAILED: {detail}", file=sys.stderr)
    _record(job, "failed", detail, checks, reproduce, status_dir)
    return 1


def _record(job: str, status: str, detail: str, checks: list[dict], reproduce: str, status_dir: str | None) -> None:
    command = [
        sys.executable,
        str(ROOT / "tools" / "ci" / "lane_status.py"),
        "record",
        "--lane",
        LANE,
        "--job",
        job,
        "--status",
        status,
        "--detail",
        detail,
        "--reproduce",
        reproduce,
    ]
    for check in checks:
        command += ["--check", json.dumps(check)]
    if status_dir:
        command += ["--status-dir", status_dir]
    subprocess.run(command, cwd=str(ROOT), check=False)


def _version_from_cargo() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else ""


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="packaging")
    parser.add_argument("--out", default=str(ROOT / "target" / "ci-packaging"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--status-dir", default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    dist = out / "dist"
    venv = out / "venv"
    started = time.monotonic()
    checks: list[dict] = []
    reproduce = "python3 tools/ci/packaging_lane.py"

    print("LANE packaging — wheel build/install smoke")
    print("  PROVES: a wheel builds, installs into a clean venv outside the checkout, imports and computes.")
    print("  DOES NOT PROVE: Rust tests, model loading, GPU support, or manylinux compliance.")

    version = _version_from_cargo()
    if not version:
        return _fail(args.job, "could not read the crate version from Cargo.toml", reproduce, checks, args.status_dir)

    if not args.skip_build:
        maturin = os.environ.get("MATURIN") or shutil.which("maturin")
        if not maturin:
            return _fail(
                args.job,
                "maturin is not installed",
                "python3 -m pip install 'maturin>=1.7,<2.0' && python3 tools/ci/packaging_lane.py",
                checks,
                args.status_dir,
            )
        dist.mkdir(parents=True, exist_ok=True)
        result = _run([maturin, "build", "--release", "--out", str(dist)], cwd=ROOT)
        if result.returncode != 0:
            return _fail(args.job, "maturin build failed", reproduce, checks, args.status_dir)
        checks.append({"name": "wheel_build", "status": "passed"})

    wheels = sorted(dist.glob("fastnn-*.whl"))
    if len(wheels) != 1:
        return _fail(
            args.job,
            f"expected exactly one fastnn wheel in {dist}, found {[w.name for w in wheels]}",
            reproduce,
            checks,
            args.status_dir,
        )
    wheel = wheels[0]
    print(f"wheel: {wheel}")
    if not wheel.name.startswith(f"fastnn-{version}-"):
        return _fail(
            args.job,
            f"wheel {wheel.name} does not match crate version {version}",
            reproduce,
            checks,
            args.status_dir,
        )
    checks.append({"name": "wheel_version", "status": "passed"})

    if venv.exists():
        shutil.rmtree(venv)
    result = _run([sys.executable, "-m", "venv", str(venv)], cwd=ROOT)
    if result.returncode != 0:
        return _fail(args.job, "python -m venv failed", reproduce, checks, args.status_dir)
    python = _venv_python(venv)
    checks.append({"name": "clean_venv", "status": "passed"})

    # runtime dependency, then the wheel itself without resolving the index
    result = _run([str(python), "-m", "pip", "install", "--quiet", "numpy"], cwd=ROOT)
    if result.returncode != 0:
        return _fail(args.job, "installing numpy into the venv failed", reproduce, checks, args.status_dir)
    result = _run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-index", "--no-deps", "--force-reinstall", str(wheel)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        return _fail(args.job, "installing the wheel into the clean venv failed", reproduce, checks, args.status_dir)
    checks.append({"name": "wheel_install", "status": "passed"})

    with tempfile.TemporaryDirectory() as temp:
        result = _run(
            [
                str(python),
                str(ROOT / "tools" / "ci" / "wheel_smoke.py"),
                "--expect-version",
                version,
                "--source-root",
                str(ROOT),
            ],
            cwd=Path(temp),
        )
    if result.returncode != 0:
        return _fail(
            args.job,
            "the installed wheel failed the outside-the-tree smoke",
            f"python3 tools/ci/packaging_lane.py --job {args.job}",
            checks,
            args.status_dir,
        )
    checks.append({"name": "wheel_smoke", "status": "passed"})

    elapsed = round(time.monotonic() - started, 3)
    detail = f"wheel {wheel.name} built, installed in a clean venv and smoke-tested from outside the source tree ({elapsed}s)"
    print(f"\nPACKAGING PASSED: {detail}")
    _record(args.job, "passed", detail, checks, reproduce, args.status_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
