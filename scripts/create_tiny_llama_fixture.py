#!/usr/bin/env python3
"""Create a deterministic tiny weighted Llama/GQA model for FastNN integration tests.

The generated Hugging Face directory is intentionally kept out of Git. Export it
with ``scripts/prepare_llm_model.py`` to exercise the same ONNX path as real
Llama-family models while keeping model construction reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    model = LlamaForCausalLM(config).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    weights = args.output / "model.safetensors"
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "architecture": "LlamaForCausalLM",
        "config": config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights_sha256": file_sha256(weights),
    }
    (args.output / "fastnn-fixture.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
