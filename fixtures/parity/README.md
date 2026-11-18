# FastNN parity fixtures

This directory is the independent numerical reference used by FastNN 3.0 kernel
and whole-model gates.  The payloads were generated outside the FastNN runtime;
no FastNN code participated in producing them.

Two reference implementations are used:

* **ONNX Runtime 1.32.0 (CPU)** for the whole GPT-2 graphs, with graph
  optimisations disabled, one intra-op thread, one inter-op thread, and
  sequential execution.
* **PyTorch 2.15.0+cpu** as an independent cross-check.  The tiny GPT-2 graph
  was checked against a `transformers.GPT2LMHeadModel` loaded from the pinned
  base checkpoint; the per-operator fixtures were computed in float64 from
  float32 inputs and checked against a float32 PyTorch implementation.

The committed bytes are **not** model weights.  Model files are fetched into a
git-ignored cache and are identified by SHA-256 in `model.lock.json`.

## Layout

```text
fixtures/parity/
  README.md
  ops/
    manifest.json
    tensors/<name>.<dtype>
  tiny-gpt2/
    manifest.json
    model.lock.json
    tensors/<name>.<dtype>
  trained-gpt2/            # opt-in, not a CI dependency
    manifest.json
    model.lock.json
    tensors/<name>.<dtype>
```

`tensors/` contains only tensors whose manifest entry has `"stored": true`.
Summary-only tensors still carry the SHA-256 of their full payload so a
regenerated run can be compared exactly without committing ~200 KB per logit
position.

## Tensor encoding

* **dtype** is `"f32"` (IEEE-754 binary32) or `"i64"` (two's-complement signed).
* **endianness** is little-endian.
* **layout** is row-major (C order), with no padding or strides.
* **byte length** is `product(shape) * itemsize`, with `itemsize` 4 for `f32`
  and 8 for `i64`.
* A scalar has shape `[]` and 1 element.
* Payload files are `tensors/<tensor name>.f32` or `.i64`.

For example, a tensor with shape `[1, 13, 50257]` and dtype `f32` has
`1 * 13 * 50257 * 4 = 2,613,364` bytes.  Element `(b, s, v)` starts at byte
`((b * 13 + s) * 50257 + v) * 4`.

## Manifest

Every suite has a `manifest.json`:

```json
{
  "format": "fastnn-parity-tensors",
  "format_version": 1,
  "endianness": "little",
  "layout": "row-major (C order)",
  "suite": "tiny-gpt2",
  "description": "human-readable purpose",
  "provenance": { "...suite-level provenance..." },
  "tensors": [
    {
      "name": "prefill.p4.logits",
      "path": null,
      "dtype": "f32",
      "shape": [1, 4, 50257],
      "bytes": 804112,
      "sha256": "ff478444...",
      "stored": false,
      "summary": { "...": "..." },
      "slices": {
        "axis": 1,
        "items": [
          {
            "index": 3,
            "shape": [1, 50257],
            "sha256": "...",
            "summary": { "...": "..." }
          }
        ]
      },
      "provenance": {
        "kind": "model_output",
        "model_id": "onnx-community/tiny-random-gpt2-ONNX",
        "model_revision": "90f61e...",
        "tokens": [464, 2068, 7586, 21931],
        "position_ids": [0, 1, 2, 3],
        "past_length": 0,
        "dtype": "float32",
        "tolerance": { "atol": 1e-05, "rtol": 0.0001, "form": "..." },
        "tool_versions": { "onnxruntime": "1.32.0", "torch": "2.15.0+cpu" }
      }
    }
  ]
}
```

Required tensor fields:

| Field | Meaning |
| --- | --- |
| `name` | Unique within the suite.  Stable across regenerations. |
| `path` | Relative payload path, or `null` for summary-only. |
| `dtype` | `"f32"` or `"i64"`. |
| `shape` | Row-major dimensions. |
| `bytes` | Exact payload size in bytes. |
| `sha256` | SHA-256 of the complete canonical payload. |
| `stored` | Whether `path` is present in this checkout. |
| `summary` | Compact checksum/statistics block (see below). |
| `provenance` | Model revision, call inputs, dtype, tolerance, tool versions. |
| `slices` | Optional per-row digests along one axis (used for logits). |

`summary` fields are `count`, `min`, `max`, `mean_abs`, `max_abs`, `nan_count`,
`inf_count`, `first` (up to eight values), `last` (up to eight values), and
`argmax_flat` for the operator fixtures and tensor statistics.  For logits the
`argmax_flat` is the token id because the last axis is the vocabulary.

`provenance.tolerance` applies to that tensor.  Suite-level
`provenance.tolerance_policy` explains sequence-length scaling and failure
criteria.

## Consuming from Rust

Add `serde_json` and read the bytes directly; no FastNN dependency is required.
The following loads one stored tensor and checks its digest:

```rust
use serde_json::Value;
use sha2::{Digest, Sha256};

fn load_f32(root: &std::path::Path, manifest: &Value, name: &str)
    -> (Vec<f32>, Vec<usize>)
{
    let tensor = manifest["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == name)
        .expect("tensor present");
    assert_eq!(tensor["dtype"], "f32");
    assert_eq!(tensor["stored"], true);
    let relative = tensor["path"].as_str().expect("stored path");
    let bytes = std::fs::read(root.join(relative)).expect("payload");
    assert_eq!(bytes.len(), tensor["bytes"].as_u64().unwrap() as usize);
    let digest = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(digest, tensor["sha256"].as_str().unwrap());
    let shape: Vec<usize> = tensor["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    (values, shape)
}
```

For a summary-only tensor, compare a computed tensor's SHA-256, `mean_abs`,
`max_abs`, `argmax_flat`, and `slices[].sha256` against the manifest.  To get the
full bytes, run `tools/parity/generate_fixtures.py` with the model cache
populated.

## Sequence and decode protocol

The fixture sequence is the GPT-2 tokenizer encoding of:

```text
The quick brown fox jumps over the lazy dog. Once upon a time, there was
```

Token ids:

```text
[464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257, 640, 11, 612, 373]
```

* Prefill inputs are prefixes of lengths 1, 4, 8, and 13.
* Cached decode steps feed tokens 13, 14, 15, and 16 (ids `640, 11, 612, 373`)
  one at a time, with the returned `present.*` tensors fed back as
  `past_key_values.*` and `position_ids` set to the absolute position.
* `attention_mask` is all ones with length `past_length + new_tokens`.
* Every decode step is compared with a full recomputation over
  `tokens[0:total_length]` at the same total length.  The comparison metrics are
  in `decode[].cached_vs_full` and the reference noise floor is in
  `noise_floor`.
* Cache tensors use `[batch, kv_heads, sequence, head_dim]`; `cache_extent` is
  the live sequence length (`14`, `15`, `16`, `17` in the tiny suite).

### Mask policy

The GPT-2 ONNX graph builds an additive mask with `-FLT_MAX` (the exact f32
value `-3.4028234663852886e38`), not `-inf`.  A fully masked softmax row
therefore becomes **uniform**, because max-subtraction cancels the constant.
This is the pinned policy in `ops/softmax.all_masked` and
`ops/attention`:

```text
softmax([-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX]) = [0.25, 0.25, 0.25, 0.25]
```

Do not implement an all-masked row as NaN unless a fixture explicitly says so.

## Operator fixture map

| Case | Shapes | What it proves |
| --- | --- | --- |
| `embedding` | indices `[2,5]`, table `[17,7]`, out `[2,5,7]` | I64 gather, duplicate indices, last row |
| `layernorm` | `[2,3,5]`, hidden 5 | zero-variance row, epsilon, odd hidden size |
| `layernorm.model_shape` | `[1,4,128]` | GPT-2 hidden size |
| `matmul.2d_k3` | `[2,3] @ [3,4]` | odd K |
| `matmul.gemm_k13_bias` | `[7,13] @ [5,13].T + [5]` | Gemm `transB=1`, odd K, bias |
| `matmul.batched_4d` | `[2,3,5,7] @ [2,3,7,11]` | 4-D batched MatMul, tail K |
| `matmul.model_proj` | `[1,13,128] @ [128,8]` | GPT-2 hidden width, odd M |
| `matmul.tail_k65` | `[3,65] @ [65,7]` | non-power-of-two K tail |
| `gelu.erf` | `[69]` | exact/erf GELU |
| `gelu.tanh` | `[69]` | `gelu_new` tanh GELU (the variant this GPT-2 graph uses) |
| `softmax.unmasked` | `[2,3,4,5]` | arbitrary-axis softmax |
| `softmax.causal` | `[1,2,4,4]` | causal additive mask |
| `softmax.all_masked` | `[1,1,2,4]` | fully masked row -> uniform |
| `softmax.padded` | `[1,1,3,5]` | padding mask |
| `attention` | Q/K/V `[2,2,4,5]` | causal mask, two heads, head dim 5, all-masked query row |

## Tolerances

The detailed table and measured noise floors are in
`docs/reports/fn3-03-parity-fixtures.md`.  The machine-readable policy is in
each manifest's `provenance.tolerance_policy`.  In short, for FP32 outputs:

```text
abs(actual - ref) <= atol + rtol * abs(ref)
```

* Tiny GPT-2 logits and KV: `atol = 1e-5`, `rtol = 1e-4`.
* Trained GPT-2 logits: `atol = 5e-4`, `rtol = 1e-4`; KV `atol = 1e-4`.
* Operator fixtures: see each tensor's `provenance.tolerance`.
* For total sequence length `S > 16`, multiply `atol` by `sqrt(S/16)`.
* A different argmax is only a failure when the reference top-two gap is larger
  than `2 * atol`.

## Offline verification

From a clean checkout, with no network and no Python packages:

```bash
python3 tools/parity/verify_fixtures.py --root fixtures/parity
```

This validates manifest structure, byte lengths, SHA-256 digests, summaries,
per-row slice digests, model lock consistency, and cross-references.  It is the
CI entry point.

## Regeneration

```bash
tools/parity/regenerate.sh                 # CI suites: ops + tiny-gpt2
tools/parity/regenerate.sh --include-trained
```

To regenerate into a temporary tree and prove the committed bytes:

```bash
tools/parity/regenerate.sh --include-trained --out /tmp/parity-check
diff -r --exclude=README.md fixtures/parity /tmp/parity-check
```

`README.md` is hand-maintained documentation; every other file in this
directory (manifests, locks, and payloads) is generator output.

## Licences and redistribution

* `onnx-community/tiny-random-gpt2-ONNX` declares no license.  Its base
  checkpoint `hyper-accel/tiny-random-gpt2` also declares none.  The tokenizer
  assets are the GPT-2 BPE files whose canonical repository is MIT.
* `onnx-community/gpt2-ONNX` declares no license of its own; it is a conversion
  of `openai-community/gpt2`, which declares MIT in its model-card metadata.
* **We commit neither set of weights.**  We commit checksums, numeric outputs,
  and provenance.  Redistribution of the ONNX weight files themselves is
  unresolved for the tiny model and should be treated as opt-in for the trained
  model.  CI never downloads a model.

See `tiny-gpt2/model.lock.json` and `trained-gpt2/model.lock.json` for the full
file inventory, sizes, SHA-256 values, tokenizer identity, graph geometry, and
license notes.
