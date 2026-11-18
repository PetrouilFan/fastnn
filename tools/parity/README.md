# Parity fixture tooling

This directory owns the pinned-model and independent-reference pipeline for
FastNN 3.0.  The generated data lives in `fixtures/parity/`; the format contract
is documented in `fixtures/parity/README.md`.

## What is pinned

| Suite | Purpose | CI dependency |
| --- | --- | --- |
| `tiny-gpt2` | 2-layer, 2-head, head-dim-64 random GPT-2; prefill 1/4/8/13, four cached decode steps, operator fixtures | yes |
| `trained-gpt2` | real 124M-parameter GPT-2; prefill 13, two cached decode steps, stable demo/benchmark sample | no (498 MB graph) |

Exact revisions, every file's SHA-256 and size, tokenizer identity, graph
geometry, and license notes are in each suite's `model.lock.json`.

## One-command regeneration

```bash
tools/parity/regenerate.sh
```

This creates `tools/parity/.cache/venv` (Python 3.13), installs the exact
versions in `requirements.lock.txt` plus the CPU-only `torch` wheel from
`requirements-torch.lock.txt`, downloads/verifies the pinned model files,
regenerates `fixtures/parity/`, and runs the offline verifier.

For the trained variant:

```bash
tools/parity/regenerate.sh --include-trained
```

To prove byte-identical regeneration without touching the committed tree:

```bash
tools/parity/regenerate.sh --include-trained --out /tmp/parity-check
diff -r --exclude=README.md fixtures/parity /tmp/parity-check
```

`README.md` is hand-maintained documentation, not generator output; every other
file under `fixtures/parity/` is byte-identical across runs on the reference
machine.

`--cache <dir>` moves the git-ignored model/venv cache.  `--venv <dir>` reuses a
separate virtual environment.

## Individual steps

```bash
# 1. Fetch + verify only (network)
tools/parity/.cache/venv/bin/python tools/parity/fetch_models.py --suite tiny-gpt2

# 2. Generate (no network; reads the verified cache)
tools/parity/.cache/venv/bin/python tools/parity/generate_fixtures.py \
    --out fixtures/parity --suites ci

# 3. Verify offline (stdlib only; the CI entry point)
python3 tools/parity/verify_fixtures.py --root fixtures/parity
```

`generate_fixtures.py --suites` accepts `ci` (ops + tiny), `all`, `ops`,
`tiny-gpt2`, and `trained-gpt2`.

## Reproducibility

The generator contains no timestamps, absolute paths, or environment-dependent
ordering.  The reference environment is recorded in every manifest:

```text
Python 3.13.15, NumPy 2.5.5, ONNX 1.23.2, ONNX Runtime 1.32.0,
PyTorch 2.15.0+cpu, x86_64 AMD Ryzen 7 3700X
```

Two runs on the same machine are byte-identical (verified with `diff -r`).
Across CPU models, ONNX Runtime's MLAS kernels and PyTorch's vectorized math can
differ in the last FP32 bits.  The committed tolerance table accounts for that;
the report records the measured noise floor.  Summary-only tensors still carry
the full payload SHA-256, so a cross-machine regeneration that differs will fail
loudly rather than silently changing the gate.

## Model cache

```
tools/parity/.cache/
  venv/
  models/
    tiny-gpt2/          # onnx-community/tiny-random-gpt2-ONNX
    tiny-gpt2-hf/       # hyper-accel/tiny-random-gpt2 (PyTorch cross-check)
    trained-gpt2/       # onnx-community/gpt2-ONNX
    trained-gpt2-hf/    # openai-community/gpt2 (optional one-off cross-check)
```

The whole `.cache/` tree is git-ignored by `tools/parity/.gitignore`.
