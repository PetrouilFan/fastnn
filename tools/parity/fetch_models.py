#!/usr/bin/env python3
"""Fetch and verify the models pinned by ``fixtures/parity/*/model.lock.json``.

Network is required only for this step.  CI consumes the committed fixtures
and never runs this script.

Examples:
    python tools/parity/fetch_models.py --suite tiny-gpt2
    python tools/parity/fetch_models.py --suite trained-gpt2
    python tools/parity/fetch_models.py --suite all --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "fixtures" / "parity"
DEFAULT_CACHE = REPO_ROOT / "tools" / "parity" / ".cache" / "models"

SUITE_LOCK = {
    "tiny-gpt2": FIXTURE_ROOT / "tiny-gpt2" / "model.lock.json",
    "trained-gpt2": FIXTURE_ROOT / "trained-gpt2" / "model.lock.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict[str, Any]) -> None:
    if not path.is_file():
        raise SystemExit(f"missing: {path}")
    size = path.stat().st_size
    if size != entry["bytes"]:
        raise SystemExit(
            f"size mismatch for {path}: {size} != pinned {entry['bytes']}"
        )
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise SystemExit(
            f"sha256 mismatch for {path}: {actual} != pinned {entry['sha256']}"
        )


def fetch_file(
    *,
    repo_id: str,
    revision: str,
    relative: str,
    destination_root: Path,
    entry: dict[str, Any],
    verify_only: bool,
) -> None:
    destination = destination_root / relative
    if not verify_only:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            filename=relative,
            revision=revision,
            local_dir=destination_root,
        )
    verify_file(destination, entry)
    print(f"verified {repo_id}@{revision[:12]} {relative} {entry['bytes']} bytes {entry['sha256']}")


def process_suite(suite: str, cache: Path, verify_only: bool, all_files: bool) -> None:
    lock_path = SUITE_LOCK[suite]
    if not lock_path.is_file():
        raise SystemExit(f"missing lock file: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    repo_id = lock["model_id"]
    revision = lock["revision"]
    destination_root = cache / suite

    selected = [
        entry
        for entry in lock["files"]
        if all_files or entry["role"] != "unused_variant"
    ]
    for entry in selected:
        fetch_file(
            repo_id=repo_id,
            revision=revision,
            relative=entry["path"],
            destination_root=destination_root,
            entry=entry,
            verify_only=verify_only,
        )

    reference = lock.get("reference_checkpoint")
    if reference and reference.get("files"):
        base_root = cache / f"{suite}-hf"
        for entry in reference["files"]:
            fetch_file(
                repo_id=reference["model_id"],
                revision=reference["revision"],
                relative=entry["path"],
                destination_root=base_root,
                entry=entry,
                verify_only=verify_only,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["tiny-gpt2", "trained-gpt2", "all"],
        default="tiny-gpt2",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify cached files without contacting the Hub",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="also verify every quantized variant listed in the lock (large)",
    )
    args = parser.parse_args()
    cache = args.cache.expanduser().resolve()
    suites = ["tiny-gpt2", "trained-gpt2"] if args.suite == "all" else [args.suite]
    for suite in suites:
        print(f"== {suite}")
        process_suite(suite, cache, args.verify_only, args.all_files)
    print("all pinned model files verified")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as error:
        print(
            f"error: {error.name!r} is missing; run tools/parity/regenerate.sh "
            "or install tools/parity/requirements.lock.txt in a Python 3.13 venv",
            file=sys.stderr,
        )
        raise SystemExit(1)
