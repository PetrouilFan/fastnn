#!/usr/bin/env python3
"""Build/install a wheel outside the checkout and replay CPU inference.

Prepare from the locked tiny-GPT-2 ONNX in one process, then replay prefill and
cached decode twice in fresh processes given only the .fnn3: once with FP32
weights and once with ``--weight-int8``. --require-inference is the release
gate: a missing fixture, failed prepare, or failed replay fails the lane.
Without it a missing fixture is explicitly not_run.

PROVES: the wheel builds, installs outside the checkout, imports, prepares the
pinned tiny model, and replays both precision modes from the artifact alone;
the INT8 arm packs 8 of 13 MatMuls and matches the pinned logits.

DOES NOT PROVE: GPU support, the Rust test suites, trained-model INT8, long
context, or anything about latency. The model fixture itself remains readable
on disk; isolation is by construction, not a sandbox.
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
import tomllib
from dataclasses import dataclass
from pathlib import Path

import wheel_smoke

ROOT = Path(__file__).resolve().parents[2]
LANE = "packaging"


def _run(command: list[str], *, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
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


def _installed_distributions(python: Path, *, cwd: Path) -> set[str]:
    result = subprocess.run(
        [str(python), "-I", "-m", "pip", "list", "--format=json"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    distributions = {item["name"].lower().replace("_", "-") for item in json.loads(result.stdout)}
    # `venv` installs these to bootstrap pip; they are not wheel dependencies.
    return distributions - {"pip", "setuptools", "wheel"}


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


@dataclass(frozen=True)
class ModelPin:
    size: int
    sha256: str
    url: str


def _required_model_reason(model: Path | None, pin: ModelPin) -> str:
    location = f"at {model}" if model is not None else f"(set {wheel_smoke.MODEL_ENV})"
    return (
        f"strict wheel inference requires the pinned tiny-GPT-2 ONNX {location}; "
        f"fetch {pin.url} and verify sha256 {pin.sha256} "
        "(fixtures/parity/tiny-gpt2/model.lock.json)"
    )


def _absent_model_reason(model: Path | None) -> str:
    if model is None:
        return (
            f"{wheel_smoke.MODEL_ENV} is not set; the pinned tiny-GPT-2 ONNX was not "
            "selected and no inference was attempted"
        )
    return f"pinned tiny-GPT-2 ONNX not present at {model}; set {wheel_smoke.MODEL_ENV}"


def _pinned_model() -> ModelPin:
    """The pinned ONNX graph's identity and documented download URL."""
    lock = json.loads((ROOT / "fixtures" / "parity" / "tiny-gpt2" / "model.lock.json").read_text(encoding="utf-8"))
    entry = next(item for item in lock["files"] if item["path"] == "onnx/model.onnx")
    url = f"https://huggingface.co/{lock['model_id']}/resolve/{lock['revision']}/onnx/model.onnx"
    return ModelPin(entry["bytes"], entry["sha256"], url)


def _phase(command: list[str], *, phase: str, cwd: Path, env: dict) -> dict:
    report = cwd.parent / f"{phase}.json"
    report.unlink(missing_ok=True)
    result = _run([*command, "--report", str(report)], cwd=cwd, env=env)
    try:
        record = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        raise ValueError(f"{phase} produced no valid evidence (exit {result.returncode})")
    if result.returncode or record.get("status") != "passed":
        raise ValueError(f"{phase} failed: {record.get('reason', f'exit {result.returncode}')}")
    return record


def _artifact_inference(worker: list[str], scratch: Path, model: Path, pin: ModelPin) -> dict:
    """CPU wheel cases: FP32 and INT8-weight replay of the same prepared artifact.

    Both replays are separate processes started from a directory holding only
    the artifact. INT8 is a runtime option; the artifact bytes do not change.
    """
    env = os.environ.copy()
    env.pop(wheel_smoke.MODEL_ENV, None)
    replay_dir = scratch / "replay"
    replay_dir.mkdir()
    quant_dir = scratch / "replay-int8"
    quant_dir.mkdir()
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=scratch) as directory:
        prepare_dir = Path(directory)
        staged = prepare_dir / "model.onnx"
        shutil.copyfile(model, staged)
        prepared = _phase(
            [*worker, "prepare", "--model", staged.name, "--expect-model-bytes", str(pin.size),
             "--expect-model-sha256", pin.sha256],
            phase="prepare", cwd=prepare_dir, env=env,
        )
        artifact = prepare_dir / wheel_smoke.ARTIFACT
        digest = prepared["artifact_sha256"]
        if wheel_smoke.sha256(artifact) != digest:
            raise ValueError("the prepared artifact digest changed before replay")
        shutil.copyfile(artifact, replay_dir / wheel_smoke.ARTIFACT)
        shutil.copyfile(artifact, quant_dir / wheel_smoke.ARTIFACT)

    # The staged ONNX is gone before launching either replay process. The
    # caller's fixture is never mutated and its path is not passed to replay.
    replayed = _phase(
        [*worker, "replay", "--expect-artifact-sha256", digest],
        phase="replay", cwd=replay_dir, env=env,
    )
    if replayed.get("artifact_only") is not True or replayed.get("artifact_sha256") != digest:
        raise ValueError("replay did not attest to artifact-only inference with the prepared digest")
    quantized = _phase(
        [*worker, "replay", "--weight-int8", "--expect-artifact-sha256", digest],
        phase="replay-int8", cwd=quant_dir, env=env,
    )
    if quantized.get("precision") != "int8-weights" or quantized.get("artifact_only") is not True:
        raise ValueError("the INT8 replay did not attest to artifact-only INT8-weight inference")
    if quantized.get("artifact_sha256") != digest:
        raise ValueError("the INT8 replay ran on a different artifact digest")
    if not quantized.get("weight_int8"):
        raise ValueError("the INT8 replay recorded no coverage report")
    return {**prepared, **replayed, "weight_int8_case": quantized}


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

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dist = out / "dist"
    checks: list[dict] = []
    reproduce = subprocess.list2cmdline(["python3", "tools/ci/packaging_lane.py", *(argv if argv is not None else sys.argv[1:])])
    report_path = out / "wheel-inference.json"
    report_path.unlink(missing_ok=True)

    def record_inference(status: str, **fields) -> None:
        report = {"case": wheel_smoke.CASE, "strict": args.require_inference, "status": status, **fields}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        checks.append({"name": "wheel_inference", **report})

    def fail(detail: str) -> int:
        return _fail(args.job, detail, reproduce, checks, args.status_dir)

    print("LANE packaging — wheel build/install smoke")
    print("  PROVES: versions agree everywhere, a wheel builds, installs into a clean venv")
    print("          outside the checkout with only its declared dependencies, the fastnn.inference")
    print("          surface imports, and (when the pinned model is present) a real")
    print("          CPU FP32 artifact-only replay reproduces pinned prefill/decode tokens.")
    if args.require_inference:
        print("  STRICT: --require-inference — a missing or mismatching pinned model FAILS this lane.")
    print("  DOES NOT PROVE: quantized inference, Rust tests, GPU support, or manylinux compliance.")

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

    # Inference is opt-in through the environment. A local run must never pick
    # up whatever model happens to sit at a fixed path on the developer machine.
    model_setting = os.environ.get(wheel_smoke.MODEL_ENV, "").strip()
    model = Path(model_setting).expanduser().resolve() if model_setting else None
    try:
        pin = _pinned_model()
    except Exception as error:  # noqa: BLE001 - a broken lock must fail the lane by name
        return _fail(
            args.job,
            f"cannot read the pinned model lock fixtures/parity/tiny-gpt2/model.lock.json: {error}",
            reproduce,
            checks,
            args.status_dir,
        )
    model_present = model is not None and model.is_file()
    if args.require_inference and not model_present:
        reason = _required_model_reason(model, pin)
        record_inference("failed", reason=reason)
        return fail(reason)

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
        # A wheel left by an earlier run (possibly naming an older version)
        # must not be installed or make the one-wheel check ambiguous.
        for stale in dist.glob("fastnn-*.whl"):
            stale.unlink()
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

    with tempfile.TemporaryDirectory(prefix="fastnn-wheel-") as directory:
        scratch = Path(directory).resolve()
        if scratch.is_relative_to(ROOT):
            return fail(f"temporary directory {scratch} is in the checkout; set TMPDIR/TEMP outside it")

        venv = scratch / "venv"
        if _run([sys.executable, "-I", "-m", "venv", str(venv)], cwd=scratch).returncode:
            return fail("python -m venv failed")
        python = _venv_python(venv)
        checks.append({"name": "clean_venv", "status": "passed"})

        pip = [str(python), "-I", "-m", "pip", "install", "--quiet"]
        if _run([*pip, "numpy"], cwd=scratch).returncode:
            return fail("installing numpy into the venv failed")
        if _run([*pip, "--no-index", "--no-deps", str(wheel)], cwd=scratch).returncode:
            return fail("installing the wheel into the clean venv failed")
        checks.append({"name": "wheel_install", "status": "passed"})

        try:
            installed = _installed_distributions(python, cwd=scratch)
        except (subprocess.CalledProcessError, ValueError) as error:
            return fail(f"cannot list installed dependencies: {error}")
        declared = _declared_dependencies() | {"fastnn"}
        if installed != declared:
            return fail(f"installed distributions {sorted(installed)} != expected {sorted(declared)}")
        checks.append({"name": "declared_dependencies_only", "status": "passed", "distributions": sorted(installed)})

        smoke = scratch / "wheel_smoke.py"
        shutil.copyfile(Path(wheel_smoke.__file__), smoke)
        worker = [str(python), "-I", str(smoke), "--expect-version", wheel_version]
        if _run([*worker, "import"], cwd=scratch).returncode:
            return fail("the installed wheel failed the outside-the-tree import smoke")
        checks.append({"name": "wheel_smoke", "status": "passed"})

        inference_status = "not_run"
        inference_fields: dict = {}
        if model_present:
            try:
                inference = _artifact_inference(worker, scratch, model, pin)
            except (OSError, ValueError, KeyError) as error:
                reason = f"{wheel_smoke.CASE} wheel inference failed: {error}"
                record_inference("failed", reason=reason)
                return fail(reason)
            del inference["status"]
            inference_status = "passed"
            inference_fields = inference
            quant = inference["weight_int8_case"]
            inference_detail = (
                f"CPU artifact-only replay top-5 {inference['top5']} (FP32) and "
                f"top-5 {quant['top5']} (INT8 weights, "
                f"{quant['weight_int8']['packed_matmuls']}/{quant['weight_int8']['packed_matmuls'] + quant['weight_int8']['fp32_matmuls']} packed)"
            )
        else:
            inference_reason = _absent_model_reason(model)
            inference_detail = f"CPU artifact-only replay NOT RUN: {inference_reason}"

    # The scratch directory (and any staging model) is gone before a success is
    # recorded, so a passed report can never describe a run that then failed.
    detail = f"wheel {wheel.name} installed and smoke-tested outside the checkout; {inference_detail}"
    try:
        if inference_status == "passed":
            record_inference("passed", **inference_fields)
        else:
            record_inference("not_run", reason=inference_reason)
        print(f"\nPACKAGING PASSED: {detail}")
        _record(args.job, "passed", detail, checks, reproduce, args.status_dir)
    except BaseException:
        try:
            record_inference("failed", reason="the lane failed after the inference attempt")
        except OSError:
            report_path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
