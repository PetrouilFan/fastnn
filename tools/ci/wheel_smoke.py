#!/usr/bin/env python3
"""Worker for packaging_lane.py, run by the installed wheel's Python (-I).

`prepare` accepts a pinned ONNX file. `replay` accepts only the prepared
artifact, in a fresh working directory without the ONNX or a source hint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
from importlib import import_module
from pathlib import Path

CASE = "tiny-gpt2-cpu-fp32"
MODEL_ENV = "FASTNN_FN3_ONNX_MODEL"
ARTIFACT = "tiny-gpt2.fnn3"
ARTIFACT_BYTES = 27_879_928
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
EXPECTED_TOP5 = [46737, 28597, 16571, 7868, 13]
EXPECTED_DECODE = [(640, 32493), (11, 11724), (612, 612)]
VOCAB = 50_257


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def installed_api(expect_version: str):
    import fastnn
    import fastnn._core
    import fastnn._core.inference as core_inference
    import fastnn.inference as fn3

    if sys.prefix == sys.base_prefix or not sys.flags.isolated:
        raise AssertionError("wheel smoke requires a fresh venv and isolated Python (-I)")
    site_packages = [Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")]
    for module in (fastnn, fastnn._core, fn3):
        location = Path(module.__file__).resolve()
        if not any(location.is_relative_to(site) for site in site_packages):
            raise AssertionError(f"{module.__name__} imported from {location}, not venv site-packages")
        print(f"{module.__name__} imported from: {location}")

    if fastnn.__version__ != expect_version:
        raise AssertionError(f"expected version {expect_version!r}, got {fastnn.__version__!r}")
    for obsolete in ("fastnn.fn3", "fastnn._core.fn3"):
        try:
            import_module(obsolete)
        except ModuleNotFoundError as exc:
            if exc.name != obsolete:
                raise
        else:
            raise AssertionError(f"obsolete inference module {obsolete} still imports")
    if fn3.Session is not core_inference.Session:
        raise AssertionError("fastnn.inference.Session is not fastnn._core.inference.Session")
    devices = fn3.devices()
    if not devices or devices[0].kind != "cpu":
        raise AssertionError(f"fastnn.inference.devices() = {devices!r}; CPU must be listed first")
    return fn3


def prepare(fn3, model: Path, expected_bytes: int, expected_sha256: str) -> dict:
    size = model.stat().st_size
    if size != expected_bytes:
        raise AssertionError(f"model size mismatch: {size} bytes, expected {expected_bytes}")
    digest = sha256(model)
    if digest != expected_sha256:
        raise AssertionError(f"model sha256 mismatch: {digest}, expected {expected_sha256}")
    print(f"pinned model: {model} ({size} bytes, sha256 {digest})")

    artifact = Path(ARTIFACT)
    report = fn3.prepare(model, artifact)
    if report.status != "created" or not report.wrote:
        raise AssertionError(f"expected a newly prepared artifact, got {report.status!r}")
    if artifact.stat().st_size != ARTIFACT_BYTES or report.size_bytes != ARTIFACT_BYTES:
        raise AssertionError(f"expected a {ARTIFACT_BYTES}-byte artifact, got {report.size_bytes}")
    artifact_sha256 = sha256(artifact)
    if report.sha256 != artifact_sha256:
        raise AssertionError("the prepare report digest does not match the produced artifact")
    print(f"prepare: {ARTIFACT} ({ARTIFACT_BYTES} bytes, sha256 {artifact_sha256})")
    return {
        "model_bytes": size,
        "model_sha256": digest,
        "artifact_bytes": ARTIFACT_BYTES,
        "artifact_sha256": artifact_sha256,
    }


def _load_artifact_only(fn3, artifact: Path):
    """Load with no source hint and prove the model records that observation."""
    loaded = fn3.load(artifact, device="cpu")
    if loaded.source is not None:
        raise AssertionError(f"artifact-only replay loaded with a source attached: {loaded.source!r}")
    return loaded


def top_k(logits, k: int, stage: str) -> list[int]:
    import numpy as np

    values = np.asarray(logits)
    if values.dtype != np.float32 or values.size != VOCAB or not np.isfinite(values).all():
        raise AssertionError(f"{stage}: expected {VOCAB} finite FP32 logits, got {values.shape} {values.dtype}")
    values = values.reshape(-1)
    # Highest logit first, with ties broken by the lowest token ID.
    return sorted(range(VOCAB), key=lambda i: (-float(values[i]), i))[:k]


def replay(fn3, expected_sha256: str, *, weight_int8: bool = False) -> dict:
    artifact = Path(ARTIFACT)
    if set(Path.cwd().iterdir()) != {artifact.absolute()} or artifact.is_symlink():
        raise AssertionError("replay must start in a directory containing only the regular .fnn3 artifact")
    if MODEL_ENV in os.environ:
        raise AssertionError(f"replay must not inherit {MODEL_ENV}")
    digest = sha256(artifact)
    if digest != expected_sha256:
        raise AssertionError("replay artifact sha256 differs from the prepare phase")
    precision = "int8-weights" if weight_int8 else "fp32"
    print(
        f"replay[{precision}]: cwd={Path.cwd()} entries={sorted(path.name for path in Path.cwd().iterdir())}; "
        f"{MODEL_ENV} unset; source=None; device=cpu"
    )

    # Intentionally no source= argument, no ONNX input, and no re-prepare.
    if weight_int8:
        loaded = fn3.load(artifact, device="cpu", weight_int8=True)
        if loaded.source is not None:
            raise AssertionError(f"artifact-only replay loaded with a source attached: {loaded.source!r}")
    else:
        loaded = _load_artifact_only(fn3, artifact)
    if loaded.sha256 != digest:
        raise AssertionError("the loaded model does not report the artifact digest that was verified")
    print(f"model: source={loaded.source!r} device={loaded.device.kind} sha256={loaded.sha256}")
    if loaded.device.kind != "cpu" or loaded.geometry.vocab_size != VOCAB:
        raise AssertionError("expected the pinned tiny-GPT-2 model on the explicit CPU device")
    if loaded.weight_int8 is not weight_int8:
        raise AssertionError(f"expected weight_int8={weight_int8}, got {loaded.weight_int8}")
    session = loaded.session()
    observed = top_k(session.prefill(PROMPT), 5, "prefill")
    if observed != EXPECTED_TOP5 or session.cache_len != len(PROMPT):
        raise AssertionError(f"prefill: top-5 {observed} != {EXPECTED_TOP5} or unexpected cache {session.cache_len}")
    print(f"prefill: tokens={len(PROMPT)} cache={session.cache_len} top-5={observed}")

    decoded = []
    for cache, (token, expected) in enumerate(EXPECTED_DECODE, start=len(PROMPT) + 1):
        observed_top1 = top_k(session.decode([token]), 1, f"decode({token})")[0]
        if observed_top1 != expected or session.cache_len != cache:
            raise AssertionError(
                f"decode({token}): top-1 {observed_top1} (expected {expected}), "
                f"cache {session.cache_len} (expected {cache})"
            )
        decoded.append({"token": token, "top1": observed_top1})
    print(f"decode: {decoded}, cache={session.cache_len}")

    record = {
        "device": "cpu",
        "precision": precision,
        "artifact_only": loaded.source is None,
        "artifact_sha256": digest,
        "prompt_tokens": PROMPT,
        "top5": observed,
        "decode": decoded,
        "cache_len": session.cache_len,
    }
    if weight_int8:
        report = session.weight_int8_report
        if report is None:
            raise AssertionError("an INT8 session completed a step without a report")
        if (report.packed_matmuls, report.fp32_matmuls) != (8, 5):
            raise AssertionError(
                f"INT8 coverage {report.packed_matmuls}/{report.packed_matmuls + report.fp32_matmuls}, "
                "expected 8/13 on the pinned tiny artifact"
            )
        if report.packed_payload != 417_792 or report.dense_label != "fp32-retained":
            raise AssertionError(
                f"INT8 byte report payload={report.packed_payload} label={report.dense_label!r}"
            )
        print(
            f"int8: packed={report.packed_matmuls}/{report.packed_matmuls + report.fp32_matmuls} "
            f"payload={report.packed_payload}B dense={report.dense_weight}B ({report.dense_label})"
        )
        record["weight_int8"] = {
            "packed_matmuls": report.packed_matmuls,
            "fp32_matmuls": report.fp32_matmuls,
            "packed_payload": report.packed_payload,
            "packed_source": report.packed_source,
            "dense_weight": report.dense_weight,
            "dense_label": report.dense_label,
        }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-version", required=True)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("import")
    prepare_parser = phases.add_parser("prepare")
    prepare_parser.add_argument("--model", required=True, type=Path)
    prepare_parser.add_argument("--expect-model-bytes", required=True, type=int)
    prepare_parser.add_argument("--expect-model-sha256", required=True)
    prepare_parser.add_argument("--report", required=True, type=Path)
    replay_parser = phases.add_parser("replay")
    replay_parser.add_argument("--expect-artifact-sha256", required=True)
    replay_parser.add_argument(
        "--weight-int8",
        action="store_true",
        help="opt into INT8 weights with FP32 activations; the source artifact is unchanged",
    )
    replay_parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        fn3 = installed_api(args.expect_version)
        if args.phase == "import":
            return 0
        if args.phase == "prepare":
            record = prepare(fn3, args.model, args.expect_model_bytes, args.expect_model_sha256)
        else:
            record = replay(fn3, args.expect_artifact_sha256, weight_int8=args.weight_int8)
    except Exception as error:  # phase-specific diagnostic for CI, including native exceptions
        print(f"WHEEL SMOKE {args.phase} FAILED: {error}", file=sys.stderr)
        if args.phase != "import":
            args.report.write_text(json.dumps({"status": "failed", "reason": str(error)}), encoding="utf-8")
        return 1

    args.report.write_text(json.dumps({"status": "passed", **record}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
