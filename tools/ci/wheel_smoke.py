#!/usr/bin/env python3
"""Smoke-test an installed FastNN wheel from outside the source tree.

Run with the wheel's virtualenv interpreter and a temporary working directory,
so a source-tree import (editable install) cannot masquerade as a wheel install.

Exits 0 only when:
  * fastnn imports from the virtualenv site-packages, not the checkout;
  * the version matches the expected version when one is given;
  * the 3.0 inference surface imports and lists at least the CPU device.

When the pinned tiny-GPT-2 ONNX model is present, the same process also runs a
real inference from the installed wheel: `prepare` -> `load` -> `prefill` ->
`decode`, and asserts the pinned 10-token prompt produces the top-5 token IDs
`46737, 28597, 16571, 7868, 13`. Without the model the inference step prints
and records a SKIP with the reason; a skip is never reported as a pass.

`--require-inference` makes a missing model a **failure** (exit 2) instead of a
skip. The release workflow uses it so a release wheel is never accepted on the
strength of an import alone. `--expect-model-sha256` additionally pins the
model bytes; the packaging lane passes the value from
`fixtures/parity/tiny-gpt2/model.lock.json`.

`--report <path>` writes a small JSON record (`passed`, `skipped` or `failed`
plus the observed top-5) so the packaging lane can record a truthful check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile

DEFAULT_ONNX = "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx"

#: The 10-token prompt used by the CLI, the Python quickstart and FN3-21.
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
EXPECTED_TOP5 = [46737, 28597, 16571, 7868, 13]
DECODE_TOKENS = [640, 11, 612]


def top_k(values, k: int = 5) -> list[int]:
    """Token IDs by descending logit, ties to the lowest ID (the CLI rule)."""
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    return order[:k]


def write_report(path: str | None, record: dict) -> None:
    if not path:
        return
    pathlib.Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_inference(model: pathlib.Path, expect_sha256: str = "") -> dict:
    import numpy as np

    import fastnn.inference as fn3

    digest = hashlib.sha256()
    with model.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    if expect_sha256 and sha256 != expect_sha256:
        raise AssertionError(
            f"model sha256 is {sha256}, expected {expect_sha256} "
            "(fixtures/parity/tiny-gpt2/model.lock.json)"
        )
    print(f"inference model: {model} ({model.stat().st_size} bytes, sha256 {sha256})")

    with tempfile.TemporaryDirectory(prefix="fastnn-wheel-inference-") as directory:
        artifact = pathlib.Path(directory) / "tiny-gpt2.fnn3"
        report = fn3.prepare(model, artifact)
        print(
            f"prepare: status={report.status} bytes={report.size_bytes} "
            f"sha256={report.sha256}"
        )
        if report.status not in ("created", "replaced", "already_valid"):
            raise AssertionError(f"unexpected prepare status {report.status!r}")
        if report.size_bytes != 27_879_928:
            raise AssertionError(
                f"artifact is {report.size_bytes} bytes, expected 27,879,928 "
                "(the pinned prepared artifact)"
            )

        loaded = fn3.load(artifact, source=model, device="cpu")
        geometry = loaded.geometry
        print(
            f"load: layers={geometry.layers} hidden={geometry.hidden_size} "
            f"kv_heads={geometry.kv_heads} head_dim={geometry.head_dim} "
            f"vocab={geometry.vocab_size}"
        )
        session = loaded.session()
        logits = np.asarray(session.prefill(PROMPT)).reshape(-1)
        if not np.isfinite(logits).all():
            raise AssertionError("prefill logits contain NaN or infinity")
        observed = top_k(logits)
        print(f"prefill: tokens={len(PROMPT)} cache={session.cache_len} top-5={observed}")
        if observed != EXPECTED_TOP5:
            raise AssertionError(
                f"prefill top-5 is {observed}, expected {EXPECTED_TOP5}; the installed "
                "wheel did not reproduce the pinned model's logits"
            )

        decoded = []
        for token in DECODE_TOKENS:
            step = np.asarray(session.decode([token])).reshape(-1)
            if not np.isfinite(step).all():
                raise AssertionError(f"decode({token}) logits contain NaN or infinity")
            decoded.append({"token": token, "top1": top_k(step, 1)[0]})
        if session.cache_len != len(PROMPT) + len(DECODE_TOKENS):
            raise AssertionError(
                f"cache is {session.cache_len}, expected "
                f"{len(PROMPT) + len(DECODE_TOKENS)} after {len(DECODE_TOKENS)} decode steps"
            )
        summary = ", ".join(f"{item['token']}->{item['top1']}" for item in decoded)
        print(
            f"decode: {len(DECODE_TOKENS)} cached steps ({summary}), "
            f"cache={session.cache_len}"
        )

    return {
        "status": "passed",
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256,
        "prompt_tokens": PROMPT,
        "top5": observed,
        "decode": decoded,
        "cache_len": len(PROMPT) + len(DECODE_TOKENS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--expect-version", default="")
    parser.add_argument("--source-root", default="")
    parser.add_argument(
        "--model",
        default=os.environ.get("FASTNN_FN3_ONNX_MODEL", DEFAULT_ONNX),
        help="pinned tiny-GPT-2 ONNX file; skipped with a printed reason when absent",
    )
    parser.add_argument(
        "--report",
        default="",
        help="write a JSON record of the inference step (passed/skipped/failed)",
    )
    parser.add_argument(
        "--require-inference",
        action="store_true",
        help="fail with exit 2 when the pinned model is absent instead of recording a skip",
    )
    parser.add_argument(
        "--expect-model-sha256",
        default="",
        help="require the model to hash to this value (from the committed model lock)",
    )
    args = parser.parse_args(argv)

    import fastnn

    location = pathlib.Path(fastnn.__file__).resolve()
    print(f"fastnn imported from: {location}")
    if args.source_root:
        source = pathlib.Path(args.source_root).resolve()
        # Build artifacts live under <source>/target, so only reject an import
        # from the source package itself, not from any path under the checkout.
        for forbidden in (source / "fastnn", source / "src"):
            if location == forbidden or forbidden in location.parents:
                print(
                    f"FAIL: fastnn was imported from the source tree ({forbidden}), "
                    "not an installed wheel",
                    file=sys.stderr,
                )
                return 1

    version = getattr(fastnn, "__version__", "")
    print(f"fastnn.__version__ = {version}")
    if args.expect_version and version != args.expect_version:
        print(
            f"FAIL: expected version {args.expect_version!r}, got {version!r}",
            file=sys.stderr,
        )
        return 1

    # The shipped Python surface is the 3.0 inference API. It must import
    # without a model and without a GPU.
    import fastnn.inference as fn3
    import fastnn._core.inference as core_inference
    from importlib import import_module

    for obsolete in ("fastnn.fn3", "fastnn._core.fn3"):
        try:
            import_module(obsolete)
        except ModuleNotFoundError as exc:
            if exc.name != obsolete:
                raise
        else:
            print(f"FAIL: obsolete inference module {obsolete} still imports", file=sys.stderr)
            return 1

    if fn3.Session is not core_inference.Session:
        print("FAIL: fastnn.inference.Session is not fastnn._core.inference.Session", file=sys.stderr)
        return 1
    devices = fn3.devices()
    if not devices or devices[0].kind != "cpu":
        print(
            f"FAIL: fn3.devices() = {devices!r}; the CPU device must always be listed",
            file=sys.stderr,
        )
        return 1
    print(f"inference smoke: fastnn.inference imported, devices={[d.id for d in devices]}")

    model = pathlib.Path(args.model).expanduser()
    if not model.is_file():
        reason = (
            f"pinned tiny-GPT-2 ONNX not present at {model}; set FASTNN_FN3_ONNX_MODEL "
            "or pass --model to run the installed-wheel prepare/load/prefill/decode smoke"
        )
        if args.require_inference:
            print(f"WHEEL INFERENCE: FAILED — {reason}", file=sys.stderr)
            print(
                "STRICT MODE: --require-inference was given, so a missing pinned model is a "
                "failure, not a skip. Fetch it from the documented URL in "
                "fixtures/parity/tiny-gpt2/model.lock.json and verify its sha256.",
                file=sys.stderr,
            )
            write_report(
                args.report,
                {"status": "failed", "reason": reason, "strict": True},
            )
            return 2
        print(f"WHEEL INFERENCE: SKIPPED — {reason}")
        write_report(args.report, {"status": "skipped", "reason": reason, "strict": False})
        print("WHEEL SMOKE: PASSED (without the pinned-model inference step)")
        return 0

    try:
        record = run_inference(model, args.expect_model_sha256)
    except Exception as error:  # noqa: BLE001 - the report needs the real message
        write_report(
            args.report,
            {"status": "failed", "reason": str(error), "strict": args.require_inference},
        )
        raise

    record["strict"] = args.require_inference
    write_report(args.report, record)
    print(f"WHEEL INFERENCE: PASSED — top-5 {record['top5']}")
    print("WHEEL SMOKE: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
