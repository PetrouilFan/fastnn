"""Pinned upstream model metadata for FastNN 3.0 parity fixtures.

The checksums here are the source of truth.  ``fetch_models.py`` downloads
files from the pinned revision and verifies these values; ``generate_fixtures``
re-checks every file it reads and rewrites ``model.lock.json`` byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from parity_format import sha256_file, write_json

TINY_ID = "onnx-community/tiny-random-gpt2-ONNX"
TINY_REVISION = "90f61e71d6fa8e571d0ab0f95a637a5d7d8ed52f"
TINY_BASE_ID = "hyper-accel/tiny-random-gpt2"
TINY_BASE_REVISION = "26bf7041497544bbd014ffdc245f3c60c0b05179"

TRAINED_ID = "onnx-community/gpt2-ONNX"
TRAINED_REVISION = "1975fb9b4a2a92bd99f457407337ee25ab6cfde0"
TRAINED_BASE_ID = "openai-community/gpt2"
TRAINED_BASE_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

# Every sibling of the pinned revision, as reported by the Hugging Face API
# (``?blobs=true``) on 2026-11-18.  LFS payloads carry the upstream sha256;
# small non-LFS files were downloaded and hashed locally.
TINY_FILES = [
    ("onnx/model.onnx", 27897974, "e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1", True, "graph"),
    ("config.json", 969, "935121200917262375ee8a179461219005d4992767823a4f7460dbd4b648023b", False, "config"),
    ("generation_config.json", 119, "e7f6144153c0e1c0ccd6c327134b1bcc8f4e36e164d903935b08753947126636", False, "generation_config"),
    ("tokenizer.json", 3557680, "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e", False, "tokenizer"),
    ("tokenizer_config.json", 474, "70049a392a6fd4dd1343d076e1f0e64d98ae821287c80638f603364ae21e64d7", False, "tokenizer_support"),
    ("vocab.json", 798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7", False, "tokenizer_support"),
    ("merges.txt", 456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5", False, "tokenizer_support"),
    ("special_tokens_map.json", 438, "98412137ae43c77f8af52eb51b19c3536d3242cb55339167d841005fa94a23b7", False, "tokenizer_support"),
    ("README.md", 347, "799bdd318fb17ccb1d8d554875e410d9cbe8c97cc5cde54d7efc9d0401ed8685", False, "model_card"),
    ("quantize_config.json", 312, "70f61db8708568fa3ee6e66ecbcfdb8a89b9069feadff679dea00b097e521c98", False, "quantization_metadata"),
    (".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361", False, "repo_metadata"),
    ("onnx/model_bnb4.onnx", 27897993, "1a9d2b878d676d715f0f0d2ac8d466f2c6b59b8e15c381c5788c688f6355068e", True, "unused_variant"),
    ("onnx/model_fp16.onnx", 13983109, "931ff378adcc54584725bae42fe79e891097954c566075d99bd2263fc9e39352", True, "unused_variant"),
    ("onnx/model_int8.onnx", 32770730, "d479a74f7f74df91b77def2aee5e64cadf1c82d2b6c9d99fd68833009cf11a64", True, "unused_variant"),
    ("onnx/model_q4.onnx", 27897993, "1a9d2b878d676d715f0f0d2ac8d466f2c6b59b8e15c381c5788c688f6355068e", True, "unused_variant"),
    ("onnx/model_q4f16.onnx", 13983128, "452d5f7ecdf55abd896f8e6d0ee35d50ce35c6cf93f1fb1764d0f48aa484b348", True, "unused_variant"),
    ("onnx/model_quantized.onnx", 32770730, "d479a74f7f74df91b77def2aee5e64cadf1c82d2b6c9d99fd68833009cf11a64", True, "unused_variant"),
    ("onnx/model_uint8.onnx", 32770733, "648708f78f13747f478cd1475c1143bfdcd1b44c9a48b8250d63362edc9b6349", True, "unused_variant"),
]

TRAINED_FILES = [
    ("onnx/model.onnx", 498126358, "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d", True, "graph"),
    ("config.json", 896, "d23d70079d44f51bca546fd5bab31268642283337950681cf136ec94665d4a1b", False, "config"),
    ("generation_config.json", 119, "6588807eba4137ddf98463ec0a4e2adea525ecf72512ac703e1d70b6e443d40b", False, "generation_config"),
    ("tokenizer.json", 2107680, "a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8", False, "tokenizer"),
    ("tokenizer_config.json", 444, "acb93ce659b9d31eb7792367ef683668844dc0a7fc5d5a0873e87556b1c08b40", False, "tokenizer_support"),
    ("vocab.json", 798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7", False, "tokenizer_support"),
    ("merges.txt", 456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5", False, "tokenizer_support"),
    ("special_tokens_map.json", 99, "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad", False, "tokenizer_support"),
    ("quantize_config.json", 285, "8da6e4e50ef7c210ba66a98a4256e6d60e38cef8a9a9c74d327251aaf4e78767", False, "quantization_metadata"),
    (".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361", False, "repo_metadata"),
    ("onnx/model_bnb4.onnx", 498126377, "adf198f0dbed6f3d7338afcd7f75abe25e1140b3062ab24f802e03eb08a77068", True, "unused_variant"),
    ("onnx/model_fp16.onnx", 249252545, "5e3a317ecbf78f55e75a3a059adc208abf5a336b23f6b415c7f92271127925ee", True, "unused_variant"),
    ("onnx/model_int8.onnx", 279633549, "98545cbc4b922ac4779114143af17820b0b74da86f2d68a23cefa0faf376a8ae", True, "unused_variant"),
    ("onnx/model_q4.onnx", 498126377, "adf198f0dbed6f3d7338afcd7f75abe25e1140b3062ab24f802e03eb08a77068", True, "unused_variant"),
    ("onnx/model_q4f16.onnx", 249252564, "c4b2c1fb045d81a6c4266c9d4e77743c0c5a698d9faf63c6c8110d872328747c", True, "unused_variant"),
    ("onnx/model_quantized.onnx", 279633549, "98545cbc4b922ac4779114143af17820b0b74da86f2d68a23cefa0faf376a8ae", True, "unused_variant"),
    ("onnx/model_uint8.onnx", 279633573, "a53fd694214c4697009b86e0c86e9574a636cee951ddb26f4cf9a63f6238dd23", True, "unused_variant"),
]

TINY_BASE_FILES = [
    ("model.safetensors", 27845776, "41a64ede13b6f741a26b8cd2e2332796c4165cab81ec37056a6fe9336b8a492e", True, "pytorch_weights"),
    ("config.json", 946, "0b5ad0d47ad3a9a9409ffa05553dd0294f017eba5c3ab4d060f79ac809a0e477", False, "config"),
    ("generation_config.json", 119, "576b5ac34bfed3b58775e722b0f7d9c7d8fef43ee5a9980b85f6b87ae447cb09", False, "generation_config"),
    ("tokenizer.json", 2107680, "a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8", False, "tokenizer"),
    ("tokenizer_config.json", 444, "acb93ce659b9d31eb7792367ef683668844dc0a7fc5d5a0873e87556b1c08b40", False, "tokenizer_support"),
    ("vocab.json", 798156, "3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7", False, "tokenizer_support"),
    ("merges.txt", 456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5", False, "tokenizer_support"),
    ("special_tokens_map.json", 99, "6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad", False, "tokenizer_support"),
    ("README.md", 5174, "0336368f8274a5ed09f54cee8d56a851ecee96026d9f391a6735762c16608a60", False, "model_card"),
    (".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361", False, "repo_metadata"),
]

TINY_LICENSE = {
    "declared_spdx": None,
    "declaration": (
        "The pinned ONNX repository has no license field, no LICENSE file, and no "
        "license statement in README.md at the pinned revision."
    ),
    "base_model": TINY_BASE_ID,
    "base_revision": TINY_BASE_REVISION,
    "base_license": None,
    "base_declaration": (
        "The base checkpoint also declares no license; its model card is the "
        "auto-generated template with every field marked '[More Information Needed]'."
    ),
    "tokenizer_provenance": (
        "The tokenizer/vocab/merges files are GPT-2 BPE assets; the canonical "
        "openai-community/gpt2 repository declares MIT."
    ),
    "redistribution": (
        "FastNN does not commit or redistribute model weights. Committed fixtures are "
        "numeric outputs derived from the weights plus checksums of the weight files. "
        "Redistribution of the weight files themselves is unresolved."
    ),
    "ci_use": (
        "CI consumes only committed numeric fixtures and never downloads the model, so "
        "the undeclared weight license does not block CI. Regeneration is opt-in."
    ),
}

TRAINED_LICENSE = {
    "declared_spdx": None,
    "declaration": (
        "The pinned ONNX repository has no license field, no LICENSE file, and no "
        "README at the pinned revision."
    ),
    "base_model": TRAINED_BASE_ID,
    "base_revision": TRAINED_BASE_REVISION,
    "base_license": "MIT",
    "base_declaration": (
        "openai-community/gpt2 declares 'license: mit' in its Hugging Face model-card "
        "metadata. The ONNX repository is a conversion of those weights but does not "
        "repeat the license in its own metadata or files."
    ),
    "tokenizer_provenance": "GPT-2 BPE assets from the same GPT-2 family.",
    "redistribution": (
        "Weights are not committed. The trained ONNX graph is a 498,126,358-byte "
        "opt-in download; only a small numeric logits fixture is committed."
    ),
    "ci_use": (
        "Not a CI dependency: the graph is ~498 MB and the ONNX repository does not "
        "repeat the MIT declaration. Demos and benchmarks fetch it explicitly."
    ),
}

TINY_TOKENIZER = {
    "class": "GPT2Tokenizer",
    "vocab_size": 50257,
    "model_max_length": 1024,
    "tokenizer_json_sha256": "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e",
    "tokenizer_json_bytes": 3557680,
    "add_prefix_space": False,
    "pad_token": None,
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
}

TRAINED_TOKENIZER = {
    "class": "GPT2Tokenizer",
    "vocab_size": 50257,
    "model_max_length": 1024,
    "tokenizer_json_sha256": "a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8",
    "tokenizer_json_bytes": 2107680,
    "add_prefix_space": False,
    "pad_token": None,
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
}

TINY_GRAPH = {
    "opset": 14,
    "ir_version": 7,
    "nodes": 257,
    "initializers": {"f32": 24, "i64": 24},
    "layers": 2,
    "query_heads": 2,
    "kv_heads": 2,
    "head_dim": 64,
    "hidden_size": 128,
    "vocab_size": 50257,
    "activation": "gelu_new (tanh approximation)",
    "layer_norm_epsilon": 1e-05,
    "position_embedding": "learned, n_positions=1024",
    "cache_layout": "[batch, kv_heads, sequence, head_dim]",
    "mask_fill_value": -3.4028234663852886e38,
    "inputs": [
        "input_ids",
        "past_key_values.0.key",
        "past_key_values.0.value",
        "past_key_values.1.key",
        "past_key_values.1.value",
        "attention_mask",
        "position_ids",
    ],
    "outputs": [
        "logits",
        "present.0.key",
        "present.0.value",
        "present.1.key",
        "present.1.value",
    ],
}

TRAINED_GRAPH = {
    "opset": 14,
    "ir_version": 7,
    "nodes": 1282,
    "initializers": {"f32": 286, "i64": 564},
    "layers": 12,
    "query_heads": 12,
    "kv_heads": 12,
    "head_dim": 64,
    "hidden_size": 768,
    "vocab_size": 50257,
    "activation": "gelu_new (tanh approximation)",
    "layer_norm_epsilon": 1e-05,
    "position_embedding": "learned, n_positions=1024",
    "cache_layout": "[batch, kv_heads, sequence, head_dim]",
    "mask_fill_value": -3.4028234663852886e38,
}


def _files_payload(entries, *, local_root: Path | None, require_used: bool) -> list[dict[str, Any]]:
    files = []
    for path, size, sha, lfs, role in entries:
        entry: dict[str, Any] = {
            "path": path,
            "bytes": size,
            "sha256": sha,
            "role": role,
            "lfs": lfs,
            "verification": "upstream_lfs_sha256" if lfs else "upstream_metadata",
        }
        if local_root is not None:
            local = local_root / path
            if local.is_file():
                actual_size = local.stat().st_size
                actual_sha = sha256_file(local)
                if actual_size != size or actual_sha != sha:
                    raise SystemExit(
                        f"checksum mismatch for {local}: got {actual_size} bytes "
                        f"{actual_sha}, pinned {size} bytes {sha}"
                    )
                entry["verification"] = "local_sha256"
            elif require_used and role not in {"unused_variant"}:
                raise SystemExit(f"required model file missing from cache: {local}")
        files.append(entry)
    return sorted(files, key=lambda item: item["path"])


def _lock_payload(
    *,
    suite: str,
    model_id: str,
    revision: str,
    base_id: str,
    base_revision: str,
    files,
    base_files,
    license_block: dict[str, Any],
    tokenizer: dict[str, Any],
    graph: dict[str, Any],
    cache_dir: Path | None,
) -> dict[str, Any]:
    downloaded_root = cache_dir / suite if cache_dir is not None else None
    base_root = cache_dir / f"{suite}-hf" if cache_dir is not None else None
    payload = {
        "format": "fastnn-model-lock",
        "format_version": 1,
        "suite": suite,
        "model_id": model_id,
        "revision": revision,
        "hub_url": f"https://huggingface.co/{model_id}/tree/{revision}",
        "license": license_block,
        "tokenizer": tokenizer,
        "graph": graph,
        "files": _files_payload(files, local_root=downloaded_root, require_used=True),
    }
    if base_files:
        payload["reference_checkpoint"] = {
            "model_id": base_id,
            "revision": base_revision,
            "purpose": "independent PyTorch cross-check of the ONNX Runtime reference",
            "files": _files_payload(base_files, local_root=base_root, require_used=True),
        }
    return payload


def tiny_lock(cache_dir: Path | None = None) -> dict[str, Any]:
    return _lock_payload(
        suite="tiny-gpt2",
        model_id=TINY_ID,
        revision=TINY_REVISION,
        base_id=TINY_BASE_ID,
        base_revision=TINY_BASE_REVISION,
        files=TINY_FILES,
        base_files=TINY_BASE_FILES,
        license_block=TINY_LICENSE,
        tokenizer=TINY_TOKENIZER,
        graph=TINY_GRAPH,
        cache_dir=cache_dir,
    )


def trained_lock(cache_dir: Path | None = None) -> dict[str, Any]:
    return _lock_payload(
        suite="trained-gpt2",
        model_id=TRAINED_ID,
        revision=TRAINED_REVISION,
        base_id=TRAINED_BASE_ID,
        base_revision=TRAINED_BASE_REVISION,
        files=TRAINED_FILES,
        base_files=[],
        license_block=TRAINED_LICENSE,
        tokenizer=TRAINED_TOKENIZER,
        graph=TRAINED_GRAPH,
        cache_dir=cache_dir,
    )


def write_lock(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def main() -> None:
    """Small inspection helper: print a lock JSON to stdout."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=["tiny-gpt2", "trained-gpt2"])
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    payload = tiny_lock(args.cache) if args.suite == "tiny-gpt2" else trained_lock(args.cache)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
