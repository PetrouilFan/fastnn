# FN3-03 — model pinning and independent parity fixtures

**Status:** complete, committed on `fn3/49-parity`. The deliverable adds no Rust
source, no `Cargo.toml`, and no files under `tests/`.

This report is the handoff contract for FastNN 3.0 kernel and whole-model gates.
All reference numbers were produced outside FastNN by ONNX Runtime 1.32.0 (CPU)
and PyTorch 2.15.0+cpu, then cross-checked against each other.

Headline results:

* The pinned 2-layer tiny GPT-2 graph is 257 nodes, 2 Q/KV heads, head dim 64,
  24 F32 + 24 I64 initializers, and 27,897,974 bytes — exactly the geometry
  documented in `docs/roadmap/llm-runtime.md`.
* ONNX Runtime cached decode agrees with full recomputation to **5.96e-7**
  maximum absolute logit error across four steps and total lengths 14–17.
* PyTorch agrees with ONNX Runtime to **6.56e-7** maximum absolute logit error
  and **4.17e-7** maximum KV error.
* The 124M-parameter trained GPT-2 variant is pinned as an opt-in asset. Its
  cached-vs-full noise floor is **1.30e-4**, and a one-off PyTorch comparison
  agrees to **2.82e-4**.
* A fresh virtual environment with a fresh Hugging Face cache regenerated every
  committed fixture byte-identically (`diff -r --exclude=README.md`).

## 1. Branch, commit, base

| Item | Value |
| --- | --- |
| Worktree | `/home/petrouil/Projects/github/fastnn-wt-49-parity` |
| Branch | `fn3/49-parity` |
| Base SHA | `a2c0759fff8bdcb7c943e53442f3382b86e59b66` |
| Base branch tip at start | `a2c0759` (merge of PR #45) |
| Final branch HEAD | see the handoff message / `git log -1 --format=%H` |
| Relationship to main | branch only; no push, tag, merge, or publish |

`git rev-parse --abbrev-ref HEAD` and `git rev-parse HEAD` were checked before
any edit and matched the assignment.

`docs/roadmap/fn3-ledger.md` is **not present in this worktree or in base
`a2c0759`**. It was added later by an orchestrator commit on another ref
(`6d48abd65807576b9dbd91f4f1dee117c8803c2d`, “docs(fn3): add 3.0 coordination
ledger and wave-1 dispatch record”) and was read with
`git show 6d48abd:docs/roadmap/fn3-ledger.md`. It confirms that
`fixtures/**` and `tools/parity/**` are FN3-03-owned, that the baseline is
`a2c0759`, and that the alpha gate requires FP32 prefill plus several cached
decode steps compared to an independent reference. The ledger also records the
host Python as 3.14.9; this work uses a 3.13.15 venv because ONNX Runtime 1.32
has 3.13 wheels, while the stdlib-only verifier runs on the host 3.14.9.
`docs/reports/**` is nominally owned by FN3-01, but this one report path is
explicitly assigned to FN3-03 by issue #49.

## 2. Changed paths

```text
fixtures/parity/README.md
fixtures/parity/ops/manifest.json
fixtures/parity/ops/tensors/*.f32 | *.i64          (47 files)
fixtures/parity/tiny-gpt2/manifest.json
fixtures/parity/tiny-gpt2/model.lock.json
fixtures/parity/tiny-gpt2/tensors/*.f32            (36 files)
fixtures/parity/trained-gpt2/manifest.json
fixtures/parity/trained-gpt2/model.lock.json
fixtures/parity/trained-gpt2/tensors/*.f32         (2 files)
tools/parity/.gitignore
tools/parity/README.md
tools/parity/fetch_models.py
tools/parity/generate_fixtures.py
tools/parity/model_pins.py
tools/parity/parity_format.py
tools/parity/regenerate.sh
tools/parity/requirements.lock.txt
tools/parity/requirements-torch.lock.txt
tools/parity/verify_fixtures.py
docs/reports/fn3-03-parity-fixtures.md
```

Totals:

* 91 committed files under `fixtures/parity/` (including its README), ~2.1 MB.
* 10 tooling files under `tools/parity/`, plus this report.
* 1,415,488 stored tensor payload bytes (85 stored tensors out of 169 total).
* No model weights are committed. Root `.gitignore` was not edited; a
  `tools/parity/.gitignore` ignores the local `.cache/`.

## 3. Pinned CI model

| Field | Value |
| --- | --- |
| Hub repository | `onnx-community/tiny-random-gpt2-ONNX` |
| Revision | `90f61e71d6fa8e571d0ab0f95a637a5d7d8ed52f` |
| Last modified (upstream) | 2025-10-03T18:20:54Z |
| Graph used | `onnx/model.onnx`, 27,897,974 bytes, SHA-256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1` |
| Base checkpoint | `hyper-accel/tiny-random-gpt2` @ `26bf7041497544bbd014ffdc245f3c60c0b05179` |
| PyTorch weights | `model.safetensors`, 27,845,776 bytes, SHA-256 `41a64ede13b6f741a26b8cd2e2332796c4165cab81ec37056a6fe9336b8a492e` |
| License | **none declared** on either the ONNX repo or the base checkpoint |
| Tokenizer | GPT-2 BPE, `tokenizer.json` 3,557,680 bytes SHA-256 `1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e`, vocab 50,257, max length 1,024 |
| Graph geometry | opset 14, IR 7, 257 nodes, 2 layers, 2 Q heads, 2 KV heads, head dim 64, hidden 128, learned positions, LayerNorm ε=1e-5, `gelu_new` (tanh) |
| Cache contract | `past_key_values.{0,1}.{key,value}` / `present.*`, `[batch, 2, sequence, 64]`, F32 |
| Mask convention | additive `-FLT_MAX` (`-3.4028234663852886e38`) |

The 257-node, 24-F32-initializer, 24-I64-initializer inventory was verified by
loading the pinned graph with `onnx.load(..., load_external_data=False)` and by
the generator's preflight, which refuses to run if the node count, opset,
initializer counts, input names, or output names drift.

### File inventory (every upstream sibling at the pinned revision)

All 18 files are recorded in `fixtures/parity/tiny-gpt2/model.lock.json` with
size, SHA-256, role, and verification source.

| File | Bytes | SHA-256 | Role |
| --- | ---: | --- | --- |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` | repo metadata |
| `README.md` | 347 | `799bdd318fb17ccb1d8d554875e410d9cbe8c97cc5cde54d7efc9d0401ed8685` | model card |
| `config.json` | 969 | `935121200917262375ee8a179461219005d4992767823a4f7460dbd4b648023b` | config |
| `generation_config.json` | 119 | `e7f6144153c0e1c0ccd6c327134b1bcc8f4e36e164d903935b08753947126636` | generation config |
| `merges.txt` | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | tokenizer support |
| `onnx/model.onnx` | 27,897,974 | `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1` | graph (used) |
| `onnx/model_bnb4.onnx` | 27,897,993 | `1a9d2b878d676d715f0f0d2ac8d466f2c6b59b8e15c381c5788c688f6355068e` | unused variant |
| `onnx/model_fp16.onnx` | 13,983,109 | `931ff378adcc54584725bae42fe79e891097954c566075d99bd2263fc9e39352` | unused variant |
| `onnx/model_int8.onnx` | 32,770,730 | `d479a74f7f74df91b77def2aee5e64cadf1c82d2b6c9d99fd68833009cf11a64` | unused variant |
| `onnx/model_q4.onnx` | 27,897,993 | `1a9d2b878d676d715f0f0d2ac8d466f2c6b59b8e15c381c5788c688f6355068e` | unused variant |
| `onnx/model_q4f16.onnx` | 13,983,128 | `452d5f7ecdf55abd896f8e6d0ee35d50ce35c6cf93f1fb1764d0f48aa484b348` | unused variant |
| `onnx/model_quantized.onnx` | 32,770,730 | `d479a74f7f74df91b77def2aee5e64cadf1c82d2b6c9d99fd68833009cf11a64` | unused variant |
| `onnx/model_uint8.onnx` | 32,770,733 | `648708f78f13747f478cd1475c1143bfdcd1b44c9a48b8250d63362edc9b6349` | unused variant |
| `quantize_config.json` | 312 | `70f61db8708568fa3ee6e66ecbcfdb8a89b9069feadff679dea00b097e521c98` | quantization metadata |
| `special_tokens_map.json` | 438 | `98412137ae43c77f8af52eb51b19c3536d3242cb55339167d841005fa94a23b7` | tokenizer support |
| `tokenizer.json` | 3,557,680 | `1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e` | tokenizer (used) |
| `tokenizer_config.json` | 474 | `70049a392a6fd4dd1343d076e1f0e64d98ae821287c80638f603364ae21e64d7` | tokenizer support |
| `vocab.json` | 798,156 | `3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7` | tokenizer support |

Total repo payload: 214,788,722 bytes including all quantized variants. The used
payload is 27,897,974 bytes (graph) plus tokenizer assets.

### Base checkpoint inventory (PyTorch cross-check)

All 10 files are recorded in `model.lock.json` under `reference_checkpoint`.
The important ones:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `model.safetensors` | 27,845,776 | `41a64ede13b6f741a26b8cd2e2332796c4165cab81ec37056a6fe9336b8a492e` |
| `config.json` | 946 | `0b5ad0d47ad3a9a9409ffa05553dd0294f017eba5c3ab4d060f79ac809a0e477` |
| `tokenizer.json` | 2,107,680 | `a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8` |

The ONNX graph's 13 distinct model F32 initializers are bit-identical to the
corresponding `model.safetensors` tensors. The ONNX exporter deduplicated the
all-zero bias tensors: every `Gemm` points at layer 0's zero bias, including the
layer-0 MLP projection bias and all layer-1 biases. This is mathematically exact
but worth knowing when debugging node-level comparisons.

### License and handling — tiny model

* The ONNX repository has no license field, no `LICENSE` file, and no license
  statement in `README.md`.
* The base checkpoint also has no license; its model card is the auto-generated
  template with every field marked “More Information Needed”.
* The tokenizer/vocab/merges assets are GPT-2 BPE files from the GPT-2 family,
  whose canonical `openai-community/gpt2` repository declares MIT.
* **FastNN does not commit or redistribute the weights.** Committed content is
  checksums, numeric outputs, and provenance. Redistribution of the ONNX weight
  files themselves is unresolved and requires legal review before anyone does
  it.
* CI is unaffected because it never downloads the model.

Exact acquisition command:

```bash
python tools/parity/fetch_models.py --suite tiny-gpt2
```

This reads the committed lock, calls `hf_hub_download` at the pinned revision,
and verifies each file's size and SHA-256 before returning.

## 4. Pinned trained variant

### Evaluation

| Candidate | Revision | Merged graph | Notes |
| --- | --- | --- | --- |
| `onnx-community/gpt2-ONNX` | `1975fb9b4a2a92bd99f457407337ee25ab6cfde0` | `onnx/model.onnx`, 498,126,358 B, SHA-256 `42c1e92a…6c7c0d` | 12 layers / 12 heads / 768 hidden, config `_name_or_path = openai-community/gpt2`; same merged-cache input/output contract as the tiny graph |
| `Xenova/gpt2` | `bf2c7f02e0b826c60d03af341171bde20893da66` | `onnx/model.onnx`, 500,799,139 B, SHA-256 `d6aa1de0…8d0349` | Same family; also ships separate `decoder_model` / `decoder_with_past_model`; slightly larger, no extra numerical value for this contract |

`onnx-community/gpt2-ONNX` was chosen as the pin: it is smaller, its config
explicitly names `openai-community/gpt2`, and its merged graph matches the tiny
model's exact input/output naming and cache layout. `Xenova/gpt2` was evaluated
and is recorded here as the fallback; it was not downloaded.

### Pin

| Field | Value |
| --- | --- |
| Hub repository | `onnx-community/gpt2-ONNX` |
| Revision | `1975fb9b4a2a92bd99f457407337ee25ab6cfde0` |
| Graph | `onnx/model.onnx`, 498,126,358 B, SHA-256 `42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d` |
| Geometry | opset 14, IR 7, 1,282 nodes, 12 layers, 12 Q/KV heads, head dim 64, hidden 768, 286 F32 + 564 I64 initializers |
| Tokenizer | GPT-2 BPE, `tokenizer.json` 2,107,680 B, SHA-256 `a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8`, vocab 50,257 |
| Base weights | `openai-community/gpt2` @ `607a30d783dfa663caf39e06633721c8d4cfcd7e`, `model.safetensors` 548,105,171 B, SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| License | ONNX repo declares none; base GPT-2 declares **MIT** in Hugging Face model-card metadata |
| CI-safe? | **No** — 498 MB graph, and the ONNX repo does not repeat the MIT declaration. Opt-in only. |

The trained suite is marked `"ci": false` in its manifest and is not touched by
the default regeneration path. It is for demos, stable benchmarking, and
large-model spot checks.

One-off independent PyTorch check (not committed, not a CI dependency) against
the base GPT-2 checkpoint:

| Call | Logits max abs | KV max abs |
| --- | ---: | ---: |
| prefill 13 | 1.5258789e-4 | 1.6212463e-5 |
| cached decode step 1 (total 14) | 1.2207031e-4 | 2.1934509e-5 |
| cached decode step 2 (total 15) | 2.8228759e-4 | 2.3841858e-5 |

This validates that the ONNX conversion is numerically the same model. The
one-off checkpoint is in the git-ignored cache only; it is not in the trained
lock, so `fetch_models.py --suite trained-gpt2` downloads 498 MB, not 1 GB.

### Trained file inventory

All 17 upstream siblings are in
`fixtures/parity/trained-gpt2/model.lock.json`.

| File | Bytes | SHA-256 | Role |
| --- | ---: | --- | --- |
| `.gitattributes` | 1,519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` | repo metadata |
| `config.json` | 896 | `d23d70079d44f51bca546fd5bab31268642283337950681cf136ec94665d4a1b` | config |
| `generation_config.json` | 119 | `6588807eba4137ddf98463ec0a4e2adea525ecf72512ac703e1d70b6e443d40b` | generation config |
| `merges.txt` | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | tokenizer support |
| `onnx/model.onnx` | 498,126,358 | `42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d` | graph (used) |
| `onnx/model_bnb4.onnx` | 498,126,377 | `adf198f0dbed6f3d7338afcd7f75abe25e1140b3062ab24f802e03eb08a77068` | unused variant |
| `onnx/model_fp16.onnx` | 249,252,545 | `5e3a317ecbf78f55e75a3a059adc208abf5a336b23f6b415c7f92271127925ee` | unused variant |
| `onnx/model_int8.onnx` | 279,633,549 | `98545cbc4b922ac4779114143af17820b0b74da86f2d68a23cefa0faf376a8ae` | unused variant |
| `onnx/model_q4.onnx` | 498,126,377 | `adf198f0dbed6f3d7338afcd7f75abe25e1140b3062ab24f802e03eb08a77068` | unused variant |
| `onnx/model_q4f16.onnx` | 249,252,564 | `c4b2c1fb045d81a6c4266c9d4e77743c0c5a698d9faf63c6c8110d872328747c` | unused variant |
| `onnx/model_quantized.onnx` | 279,633,549 | `98545cbc4b922ac4779114143af17820b0b74da86f2d68a23cefa0faf376a8ae` | unused variant |
| `onnx/model_uint8.onnx` | 279,633,573 | `a53fd694214c4697009b86e0c86e9574a636cee951ddb26f4cf9a63f6238dd23` | unused variant |
| `quantize_config.json` | 285 | `8da6e4e50ef7c210ba66a98a4256e6d60e38cef8a9a9c74d327251aaf4e78767` | quantization metadata |
| `special_tokens_map.json` | 99 | `6f50ab5a5a509a1c309d6171f339b196a900dc9c99ad0408ff23bb615fdae7ad` | tokenizer support |
| `tokenizer.json` | 2,107,680 | `a6aa29bf8416d74ad795a73262b1aa3f985564ee338adbee7ffbc5861f78b6b8` | tokenizer (used) |
| `tokenizer_config.json` | 444 | `acb93ce659b9d31eb7792367ef683668844dc0a7fc5d5a0873e87556b1c08b40` | tokenizer support |
| `vocab.json` | 798,156 | `3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7` | tokenizer support |

Exact acquisition command:

```bash
python tools/parity/fetch_models.py --suite trained-gpt2
```

## 5. Fixture inventory

The language-neutral format is specified in `fixtures/parity/README.md`:
little-endian raw F32/I64 payloads, C-order, plus a JSON manifest with name,
dtype, shape, byte length, SHA-256, summary, slices, and provenance. Stored
tensors have a payload file; summary-only tensors keep the full payload SHA-256
and per-position digests but no bytes.

### Suite totals

| Suite | Tensors | Stored | Stored bytes | Manifest | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| `ops` | 47 | 47 | 29,096 | 72,063 B | isolated kernel conformance |
| `tiny-gpt2` | 46 | 36 | 984,336 | 218,543 B | whole-model prefill + cached decode |
| `trained-gpt2` | 76 | 2 | 402,056 | 170,947 B | opt-in demo/benchmark sample |

### Whole-model sequence

Tokenizer encoding of:

```text
The quick brown fox jumps over the lazy dog. Once upon a time, there was
```

```text
[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612, 373]
```

Prefill lengths 1/4/8/13 use prefixes. Decode feeds `640, 11, 612, 373` one at
a time with the returned `present.*` caches, absolute position ids, and an
all-ones attention mask of length `past + 1`. This means decode step *n* is also
the full recomputation reference for prefix length `13+n`.

### `tiny-gpt2` tensors

All names, shapes, and byte counts below are exact. `{L}` expands to the listed
prompt lengths; `{l}` is 0 or 1; `{k}` is `key`/`value`.

| Name pattern | Shape | Dtype | Bytes each | Stored | What it proves |
| --- | --- | --- | ---: | --- | --- |
| `prefill.p1.logits` | `[1,1,50257]` | f32 | 201,028 | yes | shortest prefill, full vocabulary |
| `prefill.p4.logits` | `[1,4,50257]` | f32 | 804,112 | no | four-position prefill; per-position digest |
| `prefill.p4.last_logits` | `[1,1,50257]` | f32 | 201,028 | yes | exact last-position values at length 4 |
| `prefill.p8.logits` | `[1,8,50257]` | f32 | 1,608,224 | no | eight-position prefill; per-position digest |
| `prefill.p13.logits` | `[1,13,50257]` | f32 | 2,613,364 | no | awkward length 13; per-position digest |
| `prefill.p13.last_logits` | `[1,1,50257]` | f32 | 201,028 | yes | exact values feeding decode step 1 |
| `prefill.p{L}.present.{l}.{k}` | `[1,2,L,64]` | f32 | 512 / 2,048 / 4,096 / 6,656 | yes | prefill KV state and extent |
| `decode.step1.logits` | `[1,1,50257]` | f32 | 201,028 | yes | first cached decode step, full vocabulary |
| `decode.step{2,3,4}.logits` | `[1,1,50257]` | f32 | 201,028 | no | successive decode logits; full SHA + summary |
| `decode.step{n}.present.{l}.{k}` | `[1,2,13+n,64]` | f32 | 7,168 / 7,680 / 8,192 / 8,704 | yes | actual cache extent and appended bytes each step |
| `full.p{14,15,16,17}.logits` | `[1,N,50257]` | f32 | 2.8–3.4 MB | no | full-recompute reference at each decode length |

Prefill KV sizes: L=1 → 512 B, L=4 → 2,048 B, L=8 → 4,096 B, L=13 → 6,656 B.
Decode KV sizes: step 1 (extent 14) → 7,168 B, step 2 (15) → 7,680 B,
step 3 (16) → 8,192 B, step 4 (17) → 8,704 B.

Stored logits are exactly `p1`, `p4` last position, `p13` last position, and
`decode.step1`. Everything else keeps the full payload SHA-256 plus per-position
slice digests. Regeneration produces the full bytes.

### `ops` tensors

| Case | Tensor names | Shapes | Bytes total | What it proves |
| --- | --- | --- | ---: | --- |
| embedding | `embedding.indices`, `.table`, `.out` | `[2,5]`, `[17,7]`, `[2,5,7]` | 836 | I64 gather, duplicate indices, last row |
| layernorm | `layernorm.{input,weight,bias,out}` | `[2,3,5]`, `[5]`, `[5]`, `[2,3,5]` | 280 | ε path, zero-variance row, odd hidden 5 |
| layernorm.model_shape | `layernorm.model_shape.{input,weight,bias,out}` | `[1,4,128]`, `[128]`, `[128]`, `[1,4,128]` | 5,120 | GPT-2 hidden width 128 |
| matmul.2d_k3 | `matmul.2d_k3.{a,b,out}` | `[2,3]`, `[3,4]`, `[2,4]` | 104 | odd K=3 |
| matmul.gemm_k13_bias | `matmul.gemm_k13_bias.{a,w,bias,out}` | `[7,13]`, `[5,13]`, `[5]`, `[7,5]` | 644 | Gemm `transB=1`, bias, odd K=13 |
| matmul.batched_4d | `matmul.batched_4d.{a,b,out}` | `[2,3,5,7]`, `[2,3,7,11]`, `[2,3,5,11]` | 4,008 | 4-D batched MatMul, tail K=7 |
| matmul.model_proj | `matmul.model_proj.{a,b,out}` | `[1,13,128]`, `[128,8]`, `[1,13,8]` | 11,168 | hidden width 128, odd M=13 |
| matmul.tail_k65 | `matmul.tail_k65.{a,b,out}` | `[3,65]`, `[65,7]`, `[3,7]` | 2,684 | non-power-of-two K tail |
| gelu | `gelu.input`, `gelu.erf.out`, `gelu.tanh.out` | `[69]` each | 828 | exact/erf and `gelu_new` tanh variants |
| softmax.unmasked | `softmax.unmasked.{input,out}` | `[2,3,4,5]` | 960 | arbitrary-axis softmax |
| softmax.causal | `softmax.causal.{input,mask,out}` | `[1,2,4,4]` | 384 | causal additive mask |
| softmax.all_masked | `softmax.all_masked.{input,mask,out}` | `[1,1,2,4]` | 96 | fully masked row → uniform |
| softmax.padded | `softmax.padded.{input,mask,out}` | `[1,1,3,5]` | 180 | padding mask |
| attention | `attention.{q,k,v,mask,probs,out}` | Q/K/V/out `[2,2,4,5]`, mask/probs `[2,2,4,4]` | 1,600 | causal mask, head dim 5, all-masked query row |

All operator tensors are stored. The manifest records the float64 reference and
the float32 PyTorch self-check for every case.

### `trained-gpt2` tensors

| Name pattern | Shape | Dtype | Bytes each | Stored | What it proves |
| --- | --- | --- | ---: | --- | --- |
| `prefill.p13.logits` | `[1,13,50257]` | f32 | 2,613,364 | no | real GPT-2 prefill, per-position digest |
| `prefill.p13.last_logits` | `[1,1,50257]` | f32 | 201,028 | yes | stable last-position values |
| `prefill.p13.present.{0..11}.{key,value}` | `[1,12,13,64]` | f32 | 39,936 | no | 12-layer KV state and extent |
| `decode.step1.logits` | `[1,1,50257]` | f32 | 201,028 | yes | first trained decode step |
| `decode.step2.logits` | `[1,1,50257]` | f32 | 201,028 | no | second decode step |
| `decode.step{n}.present.{0..11}.{key,value}` | `[1,12,13+n,64]` | f32 | 43,008 / 46,080 | no | cache extent, full SHA + summary |

The trained suite is 76 tensors, 402,056 stored bytes, and is explicitly not a
CI dependency.

## 6. Tool versions and regeneration command

### Reference environment

| Tool | Version |
| --- | --- |
| Python | 3.13.15 |
| uv | 0.12.19 |
| NumPy | 2.5.5 |
| ONNX | 1.23.2 |
| ONNX Runtime | 1.32.0 (CPUExecutionProvider) |
| PyTorch | 2.15.0+cpu |
| Transformers | 5.23.0 |
| CPU | AMD Ryzen 7 3700X, x86_64 |
| OS | Linux |

All versions are pinned in `tools/parity/requirements.lock.txt` and
`tools/parity/requirements-torch.lock.txt` (split because `torch==2.15.0+cpu`
must come from the PyTorch CPU wheel index).

### Exact verified command

```bash
cd /home/petrouil/Projects/github/fastnn-wt-49-parity
HF_HOME=/tmp/opencode/parity-fresh-hf \
  tools/parity/regenerate.sh \
    --venv /tmp/opencode/parity-fresh/venv \
    --cache /tmp/opencode/parity-fresh/models \
    --out /tmp/opencode/parity-fresh/out \
    --include-trained
```

Observed result:

```text
+ uv venv --python 3.13 ...
+ uv pip install torch from the CPU wheel index
+ uv pip install tools/parity/requirements.lock.txt
verified onnx-community/tiny-random-gpt2-ONNX@90f61e71d6fa ...
verified hyper-accel/tiny-random-gpt2@26bf70414975 ...
verified onnx-community/gpt2-ONNX@1975fb9b4a2a ...
all pinned model files verified
...
OK: 3 suites, 169 tensors (85 stored, 1415488 bytes)
done: /tmp/opencode/parity-fresh/out
```

Then:

```bash
diff -r --exclude=README.md fixtures/parity /tmp/opencode/parity-fresh/out
# no output
```

This was a genuinely fresh virtual environment and a fresh `HF_HOME`; it
downloaded 31 pinned files (565,422,411 bytes fetched, of which 553,870,108
bytes are the three model graphs/checkpoints) and verified every SHA-256.
Two additional same-machine regenerations into
`/tmp/opencode/regen-a` and `/tmp/opencode/regen-b` were byte-identical.

`README.md` is the only committed file under `fixtures/parity/` that is not
generator output; it is hand-maintained documentation. Every manifest, lock,
and payload is reproduced byte-for-byte.

### Offline CI entry point

```bash
python3 tools/parity/verify_fixtures.py --root fixtures/parity
# OK: 3 suites, 169 tensors (85 stored, 1415488 bytes)
```

The verifier is standard-library-only. It was run with the system Python 3.14
(no NumPy/ONNX/Torch installed) and completed in ~0.2 s. It validates:

* manifest format/version/endianness/layout;
* required fields and unique names;
* byte length = product(shape) × itemsize;
* payload SHA-256;
* summary statistics and first/last values;
* per-row slice SHA-256 for stored tensors;
* model-lock revision, file inventory, graph checksum, tokenizer checksum;
* cross-references from workload records to tensors;
* unreferenced payload files.

Corruption test: flipping one byte in
`tiny-gpt2/tensors/prefill.p1.logits.f32` produces exit 1 with the expected
SHA-256 mismatch. The verifier does **not** need the model cache.

## 7. Tolerance table and measured noise floor

### Gate policy

Every tensor's `provenance.tolerance` is machine-readable. The general form is:

```text
abs(actual - ref) <= atol + rtol * abs(ref)
```

Recommended gates for a correct FP32 implementation:

| Workload | atol | rtol | mean_abs bound | Measured reference noise |
| --- | ---: | ---: | ---: | --- |
| tiny GPT-2 prefill/decode logits | 1e-5 | 1e-4 | 2.5e-6 | 6.56e-7 (ORT↔PyTorch), 5.96e-7 (cached↔full) |
| tiny GPT-2 KV tensors | 1e-5 | 1e-4 | 2.5e-6 | 4.17e-7 (ORT↔PyTorch) |
| trained GPT-2 logits | 5e-4 | 1e-4 | 5e-5 | 2.82e-4 (ORT↔PyTorch), 1.30e-4 (cached↔full) |
| trained GPT-2 KV | 1e-4 | 1e-4 | 5e-5 | 2.38e-5 (ORT↔PyTorch) |
| embedding | exact | exact | 0 | 0 (integer-index gather) |
| LayerNorm | 2e-5 | 1e-5 | — | 3.58e-7 (float64 ref ↔ PyTorch f32) |
| MatMul/Gemm, K ≤ 16 | 5e-6 | 1e-5 | — | 9.54e-7 |
| MatMul, K = 65/128 | 2e-5 | 1e-5 | — | 4.77e-6 |
| GELU (erf and tanh) | 2e-6 | 1e-6 | — | 4.77e-7 |
| Softmax probabilities | 2e-6 | 1e-6 | — | 5.96e-8 |
| Attention probabilities/output | 3e-6 | 1e-6 | — | 5.96e-8 |

The tiny-model bound is ~15× the measured ORT↔PyTorch noise, leaving room for
kernel-layout differences without hiding a real error. The trained bound is
~1.8× its measured PyTorch cross-check noise, because 12 layers accumulate more
rounding than two.

### Sequence-length scaling

For total sequence length `S > 16`, multiply `atol` by

```text
sqrt(S / 16)
```

and keep `rtol` unchanged. This reflects the growing number of terms in the
attention/softmax and residual accumulation. All committed tiny fixtures stop at
`S = 17`, so the unscaled bound applies directly. For operator cases keyed by
`K`, the same rule uses `sqrt(max(1, K/64))`; the committed cases are covered by
their per-tensor atol.

### What counts as a failure

1. Shape or dtype mismatch.
2. NaN or Inf where the reference is finite.
3. Any element outside the elementwise bound.
4. `mean_abs` above the stated bound (systematic bias, not just one outlier).
5. Present-cache extent or shape mismatch (`[1, kv_heads, total_length, 64]`).
6. A different argmax **only** when the reference top-two gap is larger than
   `2 * atol`; ties inside the noise floor are not gated.
7. A stored tensor whose SHA-256 does not match, or a summary-only tensor whose
   regenerated full SHA-256 does not match.

### Measured reference agreement

Tiny GPT-2, per cached decode step:

| Step | Total length | Cache extent | Cached vs full max abs | Mean abs | PyTorch cached vs full max abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14 | 14 | 5.3644180e-7 | 6.5875e-8 | 5.3644180e-7 |
| 2 | 15 | 15 | 5.9604645e-7 | 5.6939e-8 | 6.2584877e-7 |
| 3 | 16 | 16 | 5.9604645e-7 | 6.3148e-8 | 4.7683716e-7 |
| 4 | 17 | 17 | 4.7683716e-7 | 5.7676e-8 | 5.9604645e-7 |

The full recomputation's first 13 positions match the independent length-13
prefill **exactly** (`0.0`), which is the expected causal invariant.

Independent implementation checks:

| Comparison | Max abs |
| --- | ---: |
| ORT default optimisations vs ORT disabled (prefill 1/4/8/13) | 5.3644180e-7 |
| ORT vs PyTorch logits (all prefill + decode calls) | 6.5565109e-7 |
| ORT vs PyTorch KV (all layers/key/value) | 4.1723251e-7 |
| PyTorch cached vs full recomputation | 6.2584877e-7 |

Trained GPT-2:

| Comparison | Max abs | Mean abs |
| --- | ---: | ---: |
| ORT cached vs full, step 1 (total 14) | 5.3405762e-5 | 1.3837e-5 |
| ORT cached vs full, step 2 (total 15) | 1.2970e-4 | 2.6035e-5 |
| PyTorch vs ORT logits, prefill 13 | 1.5259e-4 | 2.09e-5 |
| PyTorch vs ORT logits, decode 1 | 1.2207e-4 | — |
| PyTorch vs ORT logits, decode 2 | 2.8229e-4 | — |
| PyTorch vs ORT KV | 2.3842e-5 | — |

This is the noise floor downstream workers should expect. The v2.6 audit
reported `5.96e-7` for the same tiny model and three cache steps; the committed
fixtures reproduce that number, which is a useful independent consistency check
on the new generator.

### All-masked rows

The GPT-2 ONNX graph masks with finite `-FLT_MAX`, not `-inf`. A fully masked
softmax row therefore becomes uniform:

```text
softmax([-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX]) = [0.25, 0.25, 0.25, 0.25]
```

This is pinned in `ops/softmax.all_masked` and `ops/attention`. An
implementation that uses `-inf` and produces NaN is not conformant to this
fixture. If a future model uses `-inf`, add a separate fixture rather than
changing this policy silently.

## 8. Commands, skipped checks, and risks

### Commands actually run and outcomes

| Command | Outcome |
| --- | --- |
| `git rev-parse --abbrev-ref HEAD` / `HEAD` | `fn3/49-parity`, `a2c0759…` — matched assignment |
| `gh issue view 49 --repo PetrouilFan/fastnn` | Read spec |
| `test -f docs/roadmap/fn3-ledger.md` | Missing from base/worktree |
| `git log --all -- docs/roadmap/fn3-ledger.md`; `git show 6d48abd:…` | Found later orchestrator commit; read ownership and alpha gates |
| `curl https://huggingface.co/api/models/...` | Retrieved revisions, sizes, LFS SHA-256 values |
| `uv venv --python 3.13 tools/parity/.cache/venv` | Python 3.13.15 |
| `uv pip install ... numpy onnx onnxruntime huggingface_hub tokenizers` | Installed pinned versions |
| `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` | 2.15.0+cpu |
| `uv pip install transformers safetensors` | 5.23.0 / 0.8.0 |
| `python tools/parity/generate_fixtures.py --suites ops` | 47 tensors, 29,096 bytes |
| `python tools/parity/generate_fixtures.py --suites tiny-gpt2` | 46 tensors, 984,336 stored bytes |
| `python tools/parity/generate_fixtures.py --suites trained-gpt2` | 76 tensors, 402,056 stored bytes |
| `python tools/parity/generate_fixtures.py --suites all` | All three suites, no errors |
| `python3 tools/parity/verify_fixtures.py --root fixtures/parity` | Pass: 169 tensors / 85 stored |
| Verifier with one flipped payload byte | Fail: SHA-256 mismatch, exit 1 |
| `diff -r /tmp/opencode/regen-a /tmp/opencode/regen-b` | Byte-identical |
| `tools/parity/regenerate.sh --out /tmp/opencode/regen-script` | Pass (CI suites) |
| `tools/parity/regenerate.sh --include-trained --out /tmp/opencode/regen-script-all` | Pass; full tree identical except static README |
| Fresh venv + fresh `HF_HOME` full regeneration | Pass; `FRESH-ENV FULL TREE BYTE-IDENTICAL` |
| `bash -n tools/parity/regenerate.sh` | Syntax OK |
| `python3 -m py_compile tools/parity/*.py` | Syntax OK |
| One-off PyTorch trained cross-check | 1.53e-4 / 1.22e-4 / 2.82e-4 logits, 2.38e-5 KV |

Model downloads and checksums: every downloaded file's SHA-256 and size is in
`fixtures/parity/tiny-gpt2/model.lock.json` (18 upstream siblings + 10 base
checkpoint files) and `fixtures/parity/trained-gpt2/model.lock.json` (17
upstream siblings). The fresh run verified all 32 files it fetched.

### Skipped checks and why

* **No Rust/cargo checks.** No Rust source, `Cargo.toml`, or `tests/*.rs` was
  changed. The fixture work is Python/Markdown/binary-data only.
* **No `--all-files` download of quantized variants.** The locks record the
  upstream LFS SHA-256 for all 14 unused variants (182,074,416 bytes for tiny
  plus 2,333,658,534 bytes for trained, ~2.52 GB combined). They are not needed
  numerically; verifying them would add several gigabytes to regeneration.
* **No cross-ISA/cross-OS regeneration.** Only x86_64 Linux on an AMD Ryzen
  7 3700X was available. ONNX Runtime MLAS and PyTorch vectorized kernels can
  differ in the last FP32 bits on other CPUs. The tolerance table is designed
  for that, but byte-identical regeneration is proven only for this machine.
* **No trained model in CI.** The 498 MB graph and the undeclared ONNX-repo
  license make it opt-in by design.
* **No direct ledger file at the base SHA.** The ledger was added later on
  another ref (`6d48abd`) and was read with `git show`; its alpha gate is
  “FP32 prefill and several cached decode steps, CPU and named physical Vulkan
  device, logits compared to an independent reference.” The CPU half is
  satisfied here; the Vulkan half belongs to FN3-51/FN3-11/FN3-59–61.
* **No redistribution of weights.** Only checksums and numeric outputs are
  committed.

### Known risks

1. **Tiny-model license is undeclared.** The ONNX repo and base checkpoint both
   lack a license. FastNN does not redistribute the weights, but anyone who
   wants to redistribute the ONNX files needs legal review. The numeric
   fixtures are derived outputs, not weights.
2. **Byte identity is machine-specific.** Same-machine runs are byte-identical.
   A different CPU may change low FP32 bits and therefore manifest SHA-256
   values. The report's tolerances and noise floor are the portability contract;
   regeneration on a new machine should be treated as a new reference unless it
   matches within tolerance.
3. **Summary-only tensors are not elementwise-verifiable offline.** Full logits
   for p4/p8/p13 and decode steps 2–4 are not committed; their full SHA-256 and
   per-position digests are. CI can verify stored tensors exactly and verify
   summary statistics for summary-only tensors. To debug an elementwise failure,
   regenerate the suite (the recipe is deterministic on this machine).
4. **Trained suite has no committed PyTorch reference.** It is ORT-only by
   design; the one-off PyTorch comparison above is evidence, not a CI gate. The
   tiny suite has a reproducible PyTorch cross-check.
5. **`-FLT_MAX` policy.** Implementations that treat a fully masked row as NaN
   will fail the attention/softmax fixtures. This is intentional and documented.
6. **Upstream availability.** The pinned revisions are immutable in normal Hub
   operation, but the repos could be deleted or made private. `fetch_models.py`
   will fail loudly rather than substitute a different revision.

### Evidence paths

* Format contract: `fixtures/parity/README.md`
* Machine-readable provenance/tolerances: `fixtures/parity/*/manifest.json`
* Model revisions, all file checksums, licenses: `fixtures/parity/*/model.lock.json`
* Generator: `tools/parity/generate_fixtures.py`
* Independent verifier: `tools/parity/verify_fixtures.py`
* One-command reproduction: `tools/parity/regenerate.sh`
* Environment pins: `tools/parity/requirements.lock.txt`,
  `tools/parity/requirements-torch.lock.txt`
