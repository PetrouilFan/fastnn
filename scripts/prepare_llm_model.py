#!/usr/bin/env python3
"""Reproducibly acquire, export, validate, and compile decoder-only HF models.

The source checkpoint is pinned to an immutable Hub revision. Export uses
Optimum's `text-generation-with-past` contract, then FastNN compiles the ONNX
model into an FNN artifact. A SHA-256 manifest records every produced file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODEL_ALIASES = {
    "tinyllama-1.1b": (
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    ),
    "smollm2-135m": (
        "HuggingFaceTB/SmolLM2-135M",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    ),
    "qwen2.5-0.5b": (
        "Qwen/Qwen2.5-0.5B",
        "060db6499f32faf8b98477b0a26969ef7d8b9987",
    ),
}


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _files_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def _resolve_model(args: argparse.Namespace) -> tuple[str, str]:
    if args.model in MODEL_ALIASES:
        model_id, pinned_revision = MODEL_ALIASES[args.model]
        return model_id, args.revision or pinned_revision
    local_model = Path(args.model).expanduser()
    if local_model.is_dir():
        fixture_manifest = local_model / "fastnn-fixture.json"
        if not fixture_manifest.is_file():
            raise SystemExit(
                "local --model directories require fastnn-fixture.json for reproducibility"
            )
        fixture = json.loads(fixture_manifest.read_text())
        revision = fixture.get("weights_sha256")
        if not isinstance(revision, str) or len(revision) != 64:
            raise SystemExit("local fixture manifest lacks a valid weights_sha256")
        return str(local_model.resolve()), f"local-sha256:{revision}"
    if not args.revision:
        raise SystemExit("custom --model requires an immutable --revision commit SHA")
    return args.model, args.revision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Known alias or Hugging Face model ID")
    parser.add_argument("--revision", help="Immutable Hub commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument(
        "--optimum-cli",
        default="optimum-cli",
        help="Optimum CLI executable from a transformers<5 export environment",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    model_id, revision = _resolve_model(args)
    root = args.output_dir.resolve()
    source_dir = root / "source"
    onnx_dir = root / "onnx"
    fnn_path = root / "model.fnn"
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    if not args.skip_download:
        local_model = Path(model_id)
        if local_model.is_dir():
            shutil.copytree(local_model, source_dir)
        else:
            _run(
                [
                    "hf",
                    "download",
                    model_id,
                    "--revision",
                    revision,
                    "--local-dir",
                    str(source_dir),
                ]
            )
    if not source_dir.exists():
        raise SystemExit(f"missing pinned source directory: {source_dir}")

    if not args.skip_export:
        if shutil.which(args.optimum_cli) is None and not Path(args.optimum_cli).exists():
            raise SystemExit(
                "optimum-cli is required; install `optimum` and `optimum-onnx` "
                "into the active environment"
            )
        _run(
            [
                args.optimum_cli,
                "export",
                "onnx",
                "--model",
                str(source_dir),
                "--task",
                "text-generation-with-past",
                "--dtype",
                "fp32",
                "--opset",
                str(args.opset),
                str(onnx_dir),
            ]
        )

    onnx_path = onnx_dir / "model.onnx"
    if not onnx_path.exists():
        candidates = sorted(onnx_dir.glob("*.onnx"))
        if len(candidates) != 1:
            raise SystemExit(f"expected one ONNX model in {onnx_dir}, found {candidates}")
        onnx_path = candidates[0]

    import onnx
    import fastnn as fnn

    model = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(model)
    conversion = fnn.convert_from_onnx(str(onnx_path), str(fnn_path))
    graph = conversion.get("graph", {})
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "task": "text-generation-with-past",
        "dtype": "fp32",
        "opset": args.opset,
        "onnx_path": str(onnx_path.relative_to(root)),
        "fnn_path": str(fnn_path.relative_to(root)),
        "graph_nodes": len(graph.get("nodes", [])),
        "graph_inputs": len(graph.get("inputs", [])),
        "graph_outputs": len(graph.get("outputs", [])),
        "elapsed_seconds": time.time() - started,
        "files": _files_manifest(root),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
