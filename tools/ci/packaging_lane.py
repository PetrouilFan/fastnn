#!/usr/bin/env python3
"""Lane: packaging — build a wheel, install it in a clean venv, smoke it.

The wheel is installed into a fresh virtual environment and imported from a
temporary directory outside the checkout, so an editable install cannot pass by
accident. When the pinned tiny-GPT-2 ONNX model is present, the smoke also runs
a real `fastnn.fn3` prepare/load/prefill/decode and asserts the pinned top-5
tokens; without the model the step is recorded as `not_run` with its reason,
never as a pass.

`--require-inference` changes that contract for releases: the pinned model must
be present (path from `FASTNN_FN3_ONNX_MODEL`, digest from
`fixtures/parity/tiny-gpt2/model.lock.json`) and the inference must pass, or the
lane fails with a named error. The release workflow uses it; local runs keep the
non-strict default. This proves packaging works; it says nothing about the Rust
test suites or GPU support.
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
import tomllib
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
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (ROOT / "Cargo.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else ""


def _python_version() -> str:
    """The PEP 440 spelling of the Cargo version (the wheel's version)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci" / "check-versions.py"), "--print-normalised"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _declared_dependencies() -> set[str]:
    """Requirement names from `[project].dependencies`, PEP 503-normalised."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = set()
    for requirement in project.get("dependencies", []):
        name = re.split(r"[<>=!~ ;\[]", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def _installed_distributions(python: Path) -> set[str]:
    result = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    distributions = {item["name"].lower().replace("_", "-") for item in json.loads(result.stdout)}
    # `venv` installs these to bootstrap pip; they are not wheel dependencies.
    return distributions - {"pip", "setuptools", "wheel"}


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _pinned_model() -> tuple[str, str]:
    """The pinned graph's sha256 and its documented download URL, from the lock."""
    lock = json.loads((ROOT / "fixtures" / "parity" / "tiny-gpt2" / "model.lock.json").read_text(encoding="utf-8"))
    entry = next(item for item in lock["files"] if item["path"] == "onnx/model.onnx")
    url = f"https://huggingface.co/{lock['model_id']}/resolve/{lock['revision']}/onnx/model.onnx"
    return entry["sha256"], url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="packaging")
    parser.add_argument("--out", default=str(ROOT / "target" / "ci-packaging"))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--status-dir", default=None)
    parser.add_argument(
        "--require-inference",
        action="store_true",
        help="fail unless the pinned model is present and its inference passes (release gate)",
    )
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dist = out / "dist"
    venv = out / "venv"
    started = time.monotonic()
    checks: list[dict] = []
    reproduce = "python3 tools/ci/packaging_lane.py" + (
        " --require-inference" if args.require_inference else ""
    )

    print("LANE packaging — wheel build/install smoke")
    print("  PROVES: versions agree everywhere, a wheel builds, installs into a clean venv")
    print("          outside the checkout with only its declared dependencies, the fastnn.fn3")
    print("          surface imports, and (when the pinned model is present) a real")
    print("          prepare/load/prefill/decode reproduces the pinned top-5 tokens.")
    if args.require_inference:
        print("  STRICT: --require-inference — a missing or mismatching pinned model FAILS this lane.")
    print("  DOES NOT PROVE: Rust tests, GPU support, or manylinux compliance.")

    version = _version_from_cargo()
    if not version:
        return _fail(args.job, "could not read the crate version from Cargo.toml", reproduce, checks, args.status_dir)
    wheel_version = _python_version()
    if not wheel_version:
        return _fail(
            args.job,
            "could not derive the PEP 440 wheel version from Cargo.toml",
            reproduce,
            checks,
            args.status_dir,
        )
    result = _run([sys.executable, str(ROOT / "scripts" / "ci" / "check-versions.py")], cwd=ROOT)
    if result.returncode != 0:
        return _fail(args.job, "version drift between manifests, lockfiles and the facade", reproduce, checks, args.status_dir)
    checks.append({"name": "versions_agree", "status": "passed"})

    model = Path(
        os.environ.get(
            "FASTNN_FN3_ONNX_MODEL",
            "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx",
        )
    ).expanduser()
    try:
        expected_sha256, model_url = _pinned_model()
    except Exception as error:  # noqa: BLE001 - a broken lock must fail the lane by name
        return _fail(
            args.job,
            f"cannot read the pinned model lock fixtures/parity/tiny-gpt2/model.lock.json: {error}",
            reproduce,
            checks,
            args.status_dir,
        )
    if args.require_inference and not model.is_file():
        return _fail(
            args.job,
            f"strict wheel inference requires the pinned tiny-GPT-2 ONNX at {model}; "
            f"fetch {model_url} and verify sha256 {expected_sha256} "
            "(fixtures/parity/tiny-gpt2/model.lock.json)",
            reproduce,
            checks,
            args.status_dir,
        )

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
    # maturin writes the PEP 440 spelling (3.0.0a1), not the SemVer spelling
    # (3.0.0-alpha.1), so the filename is checked against the normalised value.
    if not wheel.name.startswith(f"fastnn-{wheel_version}-"):
        return _fail(
            args.job,
            f"wheel {wheel.name} does not match {wheel_version} (Cargo.toml {version})",
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

    declared = _declared_dependencies()
    installed = _installed_distributions(python)
    unexpected = installed - declared - {"fastnn"}
    if unexpected:
        return _fail(
            args.job,
            f"the clean venv contains undeclared distributions: {sorted(unexpected)} "
            f"(declared: {sorted(declared)})",
            reproduce,
            checks,
            args.status_dir,
        )
    checks.append(
        {
            "name": "declared_dependencies_only",
            "status": "passed",
            "distributions": sorted(installed),
        }
    )

    report_path = out / "wheel-inference.json"
    report_path.unlink(missing_ok=True)
    smoke_command = [
        str(python),
        str(ROOT / "tools" / "ci" / "wheel_smoke.py"),
        "--expect-version",
        wheel_version,
        "--source-root",
        str(ROOT),
        "--model",
        str(model),
        "--expect-model-sha256",
        expected_sha256,
        "--report",
        str(report_path),
    ]
    if args.require_inference:
        smoke_command.append("--require-inference")
    with tempfile.TemporaryDirectory() as temp:
        result = _run(smoke_command, cwd=Path(temp))
    if result.returncode != 0:
        return _fail(
            args.job,
            "the installed wheel failed the outside-the-tree smoke",
            f"python3 tools/ci/packaging_lane.py --job {args.job}",
            checks,
            args.status_dir,
        )
    checks.append({"name": "wheel_smoke", "status": "passed"})

    inference = {}
    if report_path.is_file():
        try:
            inference = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            inference = {}
    if inference.get("status") == "passed":
        checks.append(
            {
                "name": "wheel_inference",
                "status": "passed",
                "top5": inference.get("top5"),
                "model_sha256": inference.get("model_sha256"),
            }
        )
    elif inference.get("status") == "failed":
        return _fail(
            args.job,
            f"the installed wheel's real inference failed: {inference.get('reason', 'no reason recorded')}",
            reproduce,
            checks,
            args.status_dir,
        )
    else:
        reason = inference.get(
            "reason",
            "the pinned tiny-GPT-2 ONNX model was absent, so prepare/load/prefill/decode did not run",
        )
        if args.require_inference:
            return _fail(
                args.job,
                f"strict wheel inference did not run although --require-inference was given: {reason}",
                reproduce,
                checks,
                args.status_dir,
            )
        print(f"\nWHEEL INFERENCE: NOT RUN — {reason}")
        checks.append({"name": "wheel_inference", "status": "not_run", "reason": reason})

    elapsed = round(time.monotonic() - started, 3)
    inference_detail = (
        f"real inference top-5 {inference.get('top5')}"
        if inference.get("status") == "passed"
        else "pinned-model inference NOT RUN (model absent)"
    )
    if args.require_inference:
        inference_detail = f"strict {inference_detail}"
    detail = (
        f"wheel {wheel.name} built, installed in a clean venv with only its declared "
        f"dependencies and smoke-tested from outside the source tree; {inference_detail} ({elapsed}s)"
    )
    print(f"\nPACKAGING PASSED: {detail}")
    _record(args.job, "passed", detail, checks, reproduce, args.status_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
