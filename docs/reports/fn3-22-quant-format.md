# FN3-22 (#68): choose and validate one quantization format

**Issue:** [#68](https://github.com/PetrouilFan/fastnn/issues/68), backlog
FN3-22, workstream Quantization, milestone stable.
**Owner:** #68 (format and reference arithmetic); #69/#70 own backend production
paths.
**Branch:** `fn3/68-quant-format`, base `672cec9faf089ef0840179f6edfdf33779b576a1`
on `rewrite/fastnn-3`.

## Decision

**`fn3-quant-format-v1` is W8A16: signed INT8 weight codes in groups of 64 along
K, an IEEE binary16 scale and an int16 integer zero-point per group, and FP32
activations.** The weight-only INT8 path is the only candidate that preserved
top-1 on the pinned trained GPT-2 graph at both cached decode positions; W8A8
added activation-quantization noise that changed the decode distribution
(KL 1.14 at step 2), and every pure 4-bit candidate changed the decode top-1.
The complete byte layout and the #69/#70 contract are in
[`docs/quantization/fn3-quant-format-v1.md`](../quantization/fn3-quant-format-v1.md).

This is a deliberate departure from the existing grouped-W4A8 path. The
evidence is in §4–§5; the short version is:

| Candidate | Decode step 1 top-1 | Decode step 2 top-1 | Decode KL (s1 / s2) | Projection storage vs FP32 |
| --- | --- | --- | --- | ---: |
| Existing grouped W4A8, G=64 | yes | **no** | 0.976 / 4.611 | 17.2% |
| Existing grouped W4A8, G=128 | **no** | **no** | 1.997 / 5.986 | 14.8% |
| Asymmetric INT4, G=64 | **no** | **no** | 2.026 / 5.427 | 14.1% |
| Symmetric INT4 (÷7), G=64 | yes | **no** | 1.006 / 5.944 | 14.1% |
| INT4 + shared column offset (diagnostic) | yes | yes | 0.029 / 0.161 | ~14% |
| W8A8, G=64 | yes | yes | 0.091 / 1.137 | 26.6% |
| **W8A16, G=64 (chosen)** | **yes** | **yes** | **0.0016 / 0.0128** | **26.6%** |

The INT4 column-offset diagnostic is the only 4-bit variant that kept top-1; it
needs a rank-1 K-vector field per tensor that is not in v1, and it is still 10×
worse in decode KL than W8A16. It is recorded as a possible `v2`, not shipped.

## 1. Scope and method

Three independent measurements were used, in this order:

1. **The real fn3 path.** A release build imports and prepares
   `onnx-community/gpt2-ONNX` and runs a 13-token prefill plus two cached decode
   steps on CPU. This proves the FP32 graph and provides the fn3-vs-ORT parity
   numbers.
2. **The existing Rust quantization path.** The 2.x `PackedTensor<I4x8>` K-grouped
   quantizer and the grouped W4A8 CPU kernel were run on the real trained
   projection weights and real activation rows. A from-scratch Python
   unpack/dequant/matmul re-derived every result, so the Rust path is not graded
   against itself.
3. **Independent end-to-end simulation.** A NumPy GPT-2 forward (validated
   against ORT to 9.0e-5 max on the same prompt) swaps in each candidate's real
   quantize/dequantize arithmetic and, for W4A8, the exact per-token INT8
   activation quantization and integer-dot structure. The quantized path cannot
   yet run *through fn3* because #69/#70 implement it; that is stated wherever
   those numbers appear.

No tolerance was widened. Where the FP32 fn3 run exceeds a documented bound, it
is reported in §3.4 rather than hidden.

## 2. The trained GPT-2 pin

`fixtures/parity/README.md` and `fixtures/parity/trained-gpt2/model.lock.json`
already carried the trained suite as opt-in (`"ci": false`). This issue verified
the actual bytes and added the fetch record in
`fixtures/parity/model.lock.json`.

| Field | Value |
| --- | --- |
| Model | `onnx-community/gpt2-ONNX` |
| Revision | `1975fb9b4a2a92bd99f457407337ee25ab6cfde0` |
| Graph | `onnx/model.onnx` |
| Size | **498,126,358 bytes** (verified with `stat`) |
| SHA-256 | **`42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d`** (computed locally) |
| Config SHA-256 | `d23d70079d44f51bca546fd5bab31268642283337950681cf136ec94665d4a1b` |
| Geometry | 12 layers, hidden 768, MLP 3072, 12 query / 12 KV heads, head dim 64, vocab 50,257, `gelu_new`, LayerNorm ε 1e-5 |
| License | base `openai-community/gpt2` declares MIT; the ONNX repo declares none; no weights are committed |
| CI | opt-in, never downloaded by CI |

The local digests match the committed per-suite lock; the new top-level index
records the download commands and the bytes actually hashed. Only checksums,
commands and numeric fixtures are committed.

### 2.1 ORT reference provenance

`tools/quant/ort_reference.py` runs ONNX Runtime 1.23.2 CPU with graph
optimisations disabled, one intra-op and one inter-op thread, and sequential
execution. It produced the 13-token prefill and two cached decode steps. On the
committed suite:

* ORT reproduced `fixtures/parity/trained-gpt2/tensors/prefill.p13.last_logits.f32`
  **byte for byte** (`max_abs 0.0`, SHA-256 equal);
* ORT reproduced `decode.step1.logits.f32` **byte for byte**;
* decode step 2 has no committed payload, so its ORT output is the reference for
  the fn3 comparison only.

### 2.2 fn3 release run

Commands (release build, CPU, serialized model runs under
`flock /tmp/opencode/fn3-bench.lock`):

```bash
cargo build --release --no-default-features --features cli
./target/release/fn3-infer prepare \
  --onnx /tmp/opencode/fn3-68/models/trained-gpt2/onnx/model.onnx \
  --out /tmp/opencode/fn3-68/trained-gpt2.fnn3
FN3_TRAINED_ONNX=... cargo test --release --no-default-features \
  --test fn3_quant_format -- --ignored --nocapture
python3 tools/quant/compare_trained_run.py \
  --fn3 /tmp/opencode/fn3-68/fn3 --ort /tmp/opencode/fn3-68/ort \
  --fixtures fixtures/parity/trained-gpt2
```

The prepared artifact is 498,075,184 bytes with SHA-256
`2626606ee4f079b43524f27f739be9414fa32fb52e9eff4b8b97e50cb095fa19`; prepare
verified all 852 weight payloads against the ONNX source bytes.

| Position | max_abs | mean_abs | Elementwise bound (atol 5e-4, rtol 1e-4) | top-1 | top-5 | KL |
| --- | ---: | ---: | --- | --- | --- | ---: |
| prefill p13 last | 1.640e-4 | 7.273e-5 | pass, 0 violations | 640 = 640 | 5/5 | 3.5e-13 |
| decode step 1 | 9.537e-5 | 1.743e-5 | pass, 0 violations | 11 = 11 | 5/5 | 5.7e-11 |
| decode step 2 | 2.975e-4 | 9.302e-5 | pass, 0 violations | 262 = 262 | 5/5 | 1.2e-9 |

The `fn3-infer run` top-5 for the 13-token prompt is
`640, 3862, 981, 1661, 2435`, exactly ORT's. The cached decode steps return
`11` and `262` at extents 14 and 15, exactly ORT.

### 2.3 Finding: suite-level mean_abs

The trained manifest's `provenance.tolerance_policy.mean_abs` is `5e-5`. The
fn3 prefill (`7.273e-5`) and decode step 2 (`9.302e-5`) exceed it; decode step 1
passes. The elementwise tolerance — the hardness gate — passes everywhere with
zero violations. This is a systematic-bias delta in the fn3 FP32 graph, not a
quantization result, and it is reported here instead of being tuned away. The
#62 gate for the tiny suite is unaffected; the trained suite is opt-in. Owner
action if the trained suite becomes a gate: investigate the LayerNorm/softmax
accumulation order and decide whether the `5e-5` bound is the intended policy.

## 3. What exists today: grouped W4A8

### 3.1 Code and arithmetic

The 2.x path is `src/packed_tensor.rs` (`from_f32_k_grouped_i4`), the compiler
passes `src/compiler/passes/quantization.rs` and
`activation_quantization.rs`, and the CPU kernel
`src/backend/cpu/microkernels/gemm.rs`
(`gemm_cpu_flat_i8_i4x8_grouped_scalar_impl` / `_per_token`). Its actual
semantics, read from the code and reproduced independently:

* signed I4 codes, eight per little-endian `u32`, low nibble first, range
  `[-8, 7]`, round-half-away;
* K groups of 32, 64 or 128;
* per `(row, group)` an **f32** scale and an **f32** zero offset, with guard
  padding: `scale = (span + 2·span/15)/15`, `zero = min − span/15 + 8·scale`;
  dequant `w = q·scale + zero`;
* a precomputed `i32` group sum per `(row, group)`;
* per-token symmetric INT8 activations, `scale = max|x|/127`;
* integer dot per group, then the four-term affine
  `s_w·s_a·Σq_wq_a + s_w·z_a·Σq_w + z_w·s_a·Σq_a + k·z_w·z_a`
  (the last two vanish for zero activation zero-point).

### 3.2 The Rust path against an independent implementation

`tools/quant/export_w4a8_cases.py` exported five real projections and the real
activation rows that feed them from the FP32 forward. The release-mode ignored
test `rust_grouped_w4a8_matches_the_independent_reference` ran the actual Rust
quantizer and kernel. `tools/quant/check_rust_w4a8.py` unpacked the Rust words
from scratch and recomputed everything.

| Case | G | Weight rel. Frobenius | MatMul vs FP32 weight | MatMul vs dequant weight | Rust vs Python MatMul | Code diffs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c_attn `[2304,768]` | 32 | 0.0936 | 0.0566 | 0.0064 | 9.5e-7 | 1 / 1,769,472 |
| c_attn `[2304,768]` | 64 | 0.1101 | 0.0665 | 0.0064 | 9.5e-7 | 1 / 1,769,472 |
| c_attn `[2304,768]` | 128 | 0.1263 | 0.0762 | 0.0064 | 9.5e-7 | 1 / 1,769,472 |
| attn.c_proj `[768,768]` | 64 | 0.1027 | 0.0650 | 0.0090 | 1.9e-6 | 1 / 589,824 |
| mlp.c_fc `[3072,768]` | 64 | 0.1105 | 0.0553 | 0.0098 | 7.2e-7 | 0 / 2,359,296 |
| mlp.c_proj `[768,3072]` | 64 | 0.1233 | 0.0926 | 0.0292 | 7.6e-6 | 2 / 2,359,296 |
| lm_head `[50257,768]` | 64 | 0.1047 | 0.0319 | 0.0129 | 6.1e-5 | 5 / 38,597,376 |

Dequantized bytes agree **bit for bit** with the independent dequant of the Rust
packed words (`max_abs 0.0`). The handful of differing codes are all ±1 and come
from the Rust encoder multiplying by the f32 reciprocal `1/scale` where the
Python reference divides by `scale`; that is a tie-rounding difference, not a
layout difference. Group sums match exactly. This is the evidence that the
existing path is understood and reproducibly measured, not trusted by assertion.

### 3.3 Per-tensor error over all 49 trained projections

The Python encoder was also run on every trained projection (49 matrices,
including the tied `lm_head`) for the existing and candidate formats:

| Encoder | G | Mean rel. Frobenius | Max rel. Frobenius | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| Existing W4A8 | 32 | 0.0908 | 0.0991 | 0.655 |
| Existing W4A8 | 64 | 0.1052 | 0.1233 | 0.667 |
| Existing W4A8 | 128 | 0.1200 | 0.1486 | 0.686 |
| Asymmetric INT4, exact span | 32 | 0.0822 | 0.0862 | 0.599 |
| Asymmetric INT4, exact span | 64 | 0.0938 | 0.1024 | 0.599 |
| Asymmetric INT4, exact span | 128 | 0.1060 | 0.1191 | 0.605 |
| Symmetric INT4 (÷7) | 64 | 0.1130 | 0.1278 | 1.212 |
| Symmetric INT4 (÷8) | 64 | 0.1043 | 0.1202 | 2.132 |
| **INT8, asymmetric/symmetric fallback** | **64** | **0.0055** | **0.0064** | **0.0354** |

The existing W4A8 encoder's guard padding makes its per-tensor error worse than
a plain exact-span INT4 encoder at the same group size (10.5% vs 9.4% mean at
G=64). That is a real, avoidable loss; it is not the main reason for rejecting
4-bit, but it is part of why “keep what exists” was not automatically the right
answer.

### 3.4 End-to-end logits (independent NumPy GPT-2 forward)

Reference: the FP32 NumPy forward, itself checked against ORT at
`max_abs 9.0e-5`, `centered_max_abs 2.5e-5`, top-1/top-5 exact. The table reports
prefill p13 and the two cached positions as full recomputations over the prefix
(equal to cached decode). `max_abs` is dominated by a large common-mode logit
offset in this graph (all logits are around −50, centered std 3.3); `centered`
metrics and KL are the ranking-relevant quantities. KL is
`KL(reference ‖ actual)` in nats.

| Variant | Prefill KL | Prefill top-1 / top-5 | s1 KL | s1 top-1 / top-5 | s2 KL | s2 top-1 / top-5 |
| --- | ---: | --- | ---: | --- | ---: | --- |
| Existing W4A8 G=32 | 0.0026 | yes / 4 | 0.712 | no / 3 | 4.318 | no / 0 |
| Existing W4A8 G=64 | 0.0115 | yes / 4 | 0.976 | yes / 3 | 4.611 | **no** / 1 |
| Existing W4A8 G=128 | 2.375 | **no** / 1 | 1.997 | no / 2 | 5.986 | no / 2 |
| Asymmetric INT4 G=32 | 0.0090 | yes / 3 | 2.103 | no / 1 | 5.202 | no / 1 |
| Asymmetric INT4 G=64 | 0.246 | yes / 1 | 2.026 | no / 1 | 5.427 | no / 0 |
| Asymmetric INT4 G=128 | 0.097 | yes / 2 | 3.044 | no / 1 | 6.990 | no / 1 |
| Symmetric INT4 ÷7 G=64 | 0.651 | yes / 2 | 1.006 | yes / 2 | 5.944 | **no** / 2 |
| Symmetric INT4 ÷8 G=64 | 0.215 | yes / 3 | — | — | — | — |
| INT4 + column offset G=32 (diagnostic) | 0.0005 | yes / 5 | 0.0298 | yes / 4 | 0.161 | yes / 4 |
| W8A8 G=64 | 5.3e-4 | yes / 5 | 0.0912 | yes / 4 | 1.137 | yes / 2 |
| **W8A16 G=64 (chosen)** | **6.7e-6** | **yes / 5** | **0.00163** | **yes / 4** | **0.0128** | **yes / 3** |
| W8A16 G=128 | 3.0e-5 | yes / 5 | 0.00174 | yes / 4 | 0.0127 | yes / 4 |

The decode top-1 is the discriminator. Every pure 4-bit configuration flips at
least one of the two decode top-1s (W4A8 G=64 flips step 2; G=128 flips step 1;
all asymmetric INT4 variants flip both). The 4-bit column-offset variant keeps
top-1 but requires extra format metadata; even then its step-2 KL is 0.161,
12.6× the chosen format.

The reference top-1 margins are 7.56 (prefill), 2.22 (step 1) and 1.03 (step 2);
the reference top-5 boundary gap is 1.20 at step 2. The step-2 top-5 overlap of
3/5 for the chosen format is therefore within the documented margin rule, not a
tolerance exception: `centered_max_abs` is 0.774 and the reference 5th-place gap
is 1.20, so the gap is below `2 × centered_max_abs`. The full per-position
numbers are in `/tmp/opencode/fn3-68/eval/eval.json` (regenerable with
`tools/quant/eval_w4a8.py`).

### 3.5 Verdict on the existing path

The grouped W4A8 arithmetic is sound and efficient, and the Rust implementation
is reproduced independently. It is not accurate enough on the pinned trained
graph: the per-tensor error is 10–12% relative at G=64/128, and the end-to-end
decode distribution changes enough to flip top-1. The issue asked for evidence
before choosing; this is that evidence.

## 4. Chosen format

### 4.1 Parameters

| Property | v1 value |
| --- | --- |
| Identifier | `fn3-quant-format-v1` |
| Weight type | signed INT8 (`[-128, 127]`), one byte per element |
| Weight granularity | groups of **64** along K |
| Scale | IEEE binary16, finite and > 0, one per `(row, group)` |
| Zero-point | signed int16 in `[-128, 127]`, one per `(row, group)` |
| Dequant | `w = (q − z) · s` in f32 |
| Activation | **FP32**; no activation quantization, no Q/DQ |
| Accumulation | f32 per group; integer `q − z` in i32 |
| Container | 64 B header + 64 B descriptors + 8 B/group directory + 16 B-aligned payload |
| Row layout | `align_up(k, 16)` bytes, zero padded after `k` |
| Tail | final group is `1..=64` elements; `k < 64` is one group |
| Checksums | IEEE CRC-32 over metadata and payload |
| Nibble interleaving | **none in v1** (8-bit only; 4-bit is a future version) |
| Alignment | payload and every tensor's code section at 16 B |

### 4.2 Why INT8 W8A16

* **Top-1 is preserved on both decode positions.** Every pure INT4 variant
  flipped at least one; W8A8 preserved top-1 but changed the distribution
  (KL 1.14), and W8A16 is two orders of magnitude closer (KL 0.0128).
* **The residual error is dominated by the format, not the kernel.** Per-tensor
  relative Frobenius is ≤ 0.64% for every projection; the worst end-to-end
  centered RMSE is 0.191 and the worst centered max is 0.774 on logits with a
  centered std of 3.3.
* **It proves the packed-layout and decoder story without also asking the
  activation-quantization story to be right.** #69/#70 get one precision
  contract, one group size, one dequantization rule. Activation quantization is
  a separate decision with its own evidence; W8A8's decode KL is not good enough
  to bundle it in.
* **It is a real compression and bandwidth win.** Projection weights are 26.6%
  of FP32 (3.77× smaller); decode is bandwidth-bound, and the packed weights are
  consumed without an FP32 copy.
* **It keeps the implementation cheap on both backends.** The kernel multiplies
  FP32 activations by small integer codes in 64-element groups. There is no
  activation-quantization calibration, no per-token rescaling, no Q/DQ pass, and
  no integer activation path to match between CPU and Vulkan.

G=64 was chosen over G=128 because it has the better prefill KL (6.7e-6 vs
3.0e-5) and centered max (0.417 vs 0.461), with a 0.8%-of-FP32 storage penalty
and a 64-element group that matches the attention head width and a 64-byte
INT8 block.

### 4.3 What is deliberately not in v1

* **No 4-bit payload.** The evaluated candidates are in §3; a future 4-bit
  format would need either the column-offset rank-1 correction or a better
  per-group encoder, plus a new version.
* **No activation quantization.** W8A8 is measured and rejected for the starter.
* **No per-tensor mode.** Every tensor uses the same group size and scale/zero
  dtypes. A blob that says otherwise is rejected, not silently accepted.
* **No nibble order.** The descriptor's `qbits` field exists to reject a 4-bit
  blob with `UnsupportedQBits`, not to accept it.

## 5. Mixed precision, Q/DQ placement, fallback

Quantized: the five projection families `c_attn`, `attn.c_proj`, `mlp.c_fc`,
`mlp.c_proj` and `lm_head` (a packed copy of the tied `wte`). FP32: embeddings
and position table, LayerNorm, GELU, softmax, attention score/value MatMuls,
residual adds, all biases, and any MatMul whose weight is not a 2-D F32 constant
or whose RHS is dynamic. There are **no Q/DQ nodes**: the weight stays packed,
the activation stays FP32, and dequantization is the per-group multiply inside
the projection kernel.

Fallback rules (must be visible in prepared-artifact metadata):

* `K < 64` or `numel < 4096` → FP32 (metadata would dominate);
* non-2-D weight, non-F32 weight, dynamic RHS → FP32;
* weight shared with a non-MatMul consumer without a permitted packed copy →
  FP32;
* any other tensor shape still works: tails are part of the format, not a
  fallback.

On the trained model all 49 projections qualify; the only “fallback” is the
intentional retention of the FP32 `wte` table for the embedding Gather, with a
separate packed copy for `lm_head`.

## 6. Accuracy gate for #69 / #70

The candidate must meet all of these on the pinned trained prompt (prefill 13,
decode tokens 640 and 11), with the FP32 NumPy forward as the reference:

| Metric | Bound | Chosen format measured |
| --- | ---: | --- |
| Per-projection relative Frobenius | ≤ 0.008 | max 0.0064 |
| Prefill KL | ≤ 1e-3 | 6.7e-6 |
| Decode KL, each step | ≤ 0.02 | 0.00163 / 0.0128 |
| Centered max abs, any position | ≤ 0.8 | 0.417 / 0.364 / 0.774 |
| Centered RMSE, any position | ≤ 0.25 | 0.107 / 0.088 / 0.191 |
| Top-1 agreement | 100% where the reference margin > 1.0 | 3/3 (margins 7.56 / 2.22 / 1.03) |
| Top-5 | ≥ 4/5 where the 5th-place gap > 2×centered max | 5/5, 4/5, 3/5 (step 2 gap 1.20 < 1.55) |
| Elementwise logits | max_abs ≤ 1.5 | 1.42 / 0.87 / 1.45 (common-mode dominated) |

A regression in any top-1 with a reference margin above 1.0 is a failure even if
the aggregate KL passes. A backend that dequantizes into a tile scratch buffer
is still required to meet the same numbers; the format does not relax because an
implementation chose a different inner loop.

## 7. Storage

| Quantity | Bytes | Vs FP32 |
| --- | ---: | ---: |
| FP32 projection weights (49 matrices) | 494,128,128 | 100% |
| v1 packed projections (codes + metadata) | 131,252,784 | 26.6% (3.77× smaller) |
| Metadata share | 7,720,752 (4 B per 64 weights) | 6.25% of the INT8 payload |
| Whole model, FP32 embeddings retained | 289,279,144 | 58.1% (1.72× smaller) |
| Whole model, `lm_head` left FP32 | 248,269,432 | 49.9% (2.00× smaller) |

The metadata is `f16 scale + i16 zero-point = 4 bytes` per 64 weights. The whole
model is dominated by the FP32 token-embedding table (154,389,504 B), which is
kept for the Gather. Quantizing the embedding table itself would need a
dequantizing gather and is out of scope. The `lm_head` packed copy costs
41,009,712 B; leaving it FP32 saves that copy at the cost of the largest decode
matmul.

## 8. Fixtures and the independent reference

`tools/quant/fn3_quant.py` is a from-scratch encoder/decoder (stdlib + NumPy).
`tools/quant/make_fixtures.py` wrote 4 valid blobs and 31 malformed blobs into
`fixtures/quant/`; `tools/quant/verify_fixtures.py --regenerate` rebuilt them and
required a byte-for-byte match. `tests/fn3_quant_format.rs` parses the same files
with the Rust reader and compares every field, every code byte, every scale,
every zero-point and every dequantized value exactly.

Valid fixtures:

| File | What it proves |
| --- | --- |
| `v1/basic.fn3q` | 2×64, full code range, a non-zero zero-point, constant row |
| `v1/tail_k70.fn3q` | K=70: full group + six-element tail, 10 bytes of row padding |
| `v1/zero_point.fn3q` | a non-zero zero-point shifts the whole code range |
| `v1/two_tensors.fn3q` | two tensors, cumulative group directory, K=8 short group |

The README gives the hand arithmetic for each (for example `basic` row 0 starts
at byte `0xF8` for code −8 and value −4.0 at scale 0.5; `tail_k70`'s last value
is `(60−30)·0.125 = 3.75`). The manifest carries the full expected tensor so the
check is not only a prose claim.

## 9. Malformed metadata

31 blobs, 27 distinct typed errors. The Rust and Python readers share the
`kind()` strings; the test asserts the exact string per file:

```text
bad-magic, unsupported-version, bad-header-size, unsupported-flags,
bad-tensor-count, bad-descriptor-bytes, bad-descriptor-offset, bad-group-offset,
bad-group-dir-bytes, bad-payload-offset, trailing-or-truncated-bytes,
meta-checksum, payload-checksum, bad-name, duplicate-name, zero-extent,
unsupported-qbits, unsupported-group-size, bad-group-count, bad-group-index,
bad-row-stride, misaligned-offset, bad-data-offset, truncated-payload,
nonzero-padding, nonzero-tail-padding, group-gap-or-overlap, bad-group-length,
incomplete-groups, bad-scale, zero-point-out-of-range, unused-group-entries,
payload-tail, output-too-small
```

The parser checks lengths and offsets before forming a slice, checks both CRCs
before trusting metadata, validates every scale and zero-point at load, and
returns `QuantError` (never panics). The `output-too-small` case is the only
caller-error variant; it is exercised separately from the blob matrix.

## 10. #72 coordination note

Quantized weights are **device-resident from prepare**. Prepare packs on the
host, uploads the v1 payload (codes, scales, zero-points) once with the prepared
artifact, and the resident session binds those buffers directly. There is no
per-op upload, no per-token weight staging and no host readback during decode.

Dequantization happens **inside the GEMM kernel**: the packed INT8 codes remain
in device memory; the activation tile is FP32; the kernel accumulates
`x·(q−z)` per 64-element group in FP32, multiplies by the binary16 scale widened
on device, and accumulates groups in FP32. A dequantized weight buffer is never
written. This is the integer-storage / FP32-accumulate choice: `q` and `z` stay
integer for the dot, and only the per-group affine is floating point. It avoids
the bandwidth of an FP32 weight copy and keeps the resident path free of Q/DQ
nodes. A backend without a fused affine may dequantize a *tile*, not a whole
tensor, and must account for that scratch in its residency budget; the on-disk
format and the accuracy gate do not change.

## 11. Commands and outcomes

All model runs were wrapped in `flock /tmp/opencode/fn3-bench.lock`; observed
`/proc/loadavg` before/after was 6.59→6.27 (prepare) and 5.62 (run), on a shared
16-core host with other owners building concurrently.

| Command | Outcome |
| --- | --- |
| `git rev-parse HEAD` | `672cec9faf089ef0840179f6edfdf33779b576a1` (matches base) |
| `curl` trained graph/config + `sha256sum` | 498,126,358 B, `42c1e92a…`; config `d23d7007…` |
| `cargo build --release --no-default-features --features cli` | success, 1m35s |
| `fn3-infer prepare` (trained) | 1330 ops, 852 weights, artifact 498,075,184 B, all weights verified |
| `fn3-infer run --tokens <13> --device cpu` | prefill 5.93 s; top-5 exactly ORT; argmax 640 |
| `ort_reference.py` | reproduces both committed trained payloads byte for byte |
| `cargo test --release --no-default-features --test fn3_quant_format -- --ignored` (trained) | pass; prefill max_abs 1.64e-4, decode1 9.54e-5, decode2 2.98e-4 |
| `compare_trained_run.py` | 0 elementwise violations at all three positions; mean_abs bound exceeded at prefill/step2 as reported |
| `export_w4a8_cases.py` | 5 real projections + real activations |
| `cargo test --release … rust_grouped_w4a8 -- --ignored` | pass; 15 (case, group) runs |
| `check_rust_w4a8.py` | dequant bit-exact; MatMul vs independent ≤ 6.1e-5; codes ±1 ≤ 6 elements/tensor |
| `eval_w4a8.py` | full per-tensor + end-to-end table in `eval.json` |
| `make_fixtures.py` | 4 valid + 31 malformed blobs |
| `verify_fixtures.py --regenerate` | `OK: 4 fixtures, 31 malformed cases, 1072 fixture bytes` |
| `cargo test --no-default-features --test fn3_quant_format` | 4 passed, 2 ignored |
| `cargo fmt --check` | clean |
| `cargo test --lib` | **674 passed, 0 failed** (673 + the new CRC test) |
| `cargo clippy --tests --no-default-features` | clean |
| `cargo clippy --tests` (default features) | clean |

## 12. Skips, blockers, open findings

* **No quantized fn3 run.** #69/#70 have not implemented the production path;
  the end-to-end quantized numbers are an independent NumPy simulation, clearly
  labelled. The Rust W4A8 path *was* run for real and independently checked.
* **No ONNX Runtime MatMulNBits reference.** ORT's MatMulNBits needs a
  pre-quantized ONNX graph; the committed lock records the same size and SHA-256 for
  `model_q4.onnx` and `model_bnb4.onnx` (`adf198f0…`, 498,126,377 B), so the
  repository's "q4" variant is not a clean grouped-int4 reference. A from-scratch grouped-int4 unpack/matmul was
  used instead, as the issue permits.
* **No Vulkan run.** Quantized Vulkan is #70; #72 owns the resident rewrite.
  The format note for #72 is §10.
* **FP32 mean_abs delta** on trained prefill and decode step 2 (§2.3): a real
  finding, not tuned away. It does not affect the format decision.
* **No 4-bit production format.** The best 4-bit result (column offset) is
  documented, but it needs a field v1 does not have and still loses 10× on
  decode KL.
* **License.** The trained ONNX repository declares no license of its own; the
  base GPT-2 weights are MIT. Nothing is redistributed here. The suite remains
  `"ci": false`.

## 13. Artifacts

| Path | Purpose |
| --- | --- |
| `docs/quantization/fn3-quant-format-v1.md` | authoritative byte layout and #69/#70 contract |
| `src/fn3/quant/` | Rust decoder/validator and typed errors |
| `fixtures/quant/` | hand-checkable valid and malformed blobs |
| `fixtures/parity/model.lock.json` | additive trained fetch record |
| `tools/quant/fn3_quant.py` | independent encoder/decoder and candidate quantizers |
| `tools/quant/make_fixtures.py` | deterministic fixture generator |
| `tools/quant/verify_fixtures.py` | offline verifier, `--regenerate` byte check |
| `tools/quant/ort_reference.py` | ORT reference for the trained run |
| `tools/quant/compare_trained_run.py` | fn3-vs-ORT report |
| `tools/quant/eval_w4a8.py` | candidate per-tensor and end-to-end evaluation |
| `tools/quant/export_w4a8_cases.py` | real weights/activations for the Rust check |
| `tools/quant/check_rust_w4a8.py` | independent verification of the Rust W4A8 path |
| `tests/fn3_quant_format.rs` | format gate + opt-in trained/W4A8 evidence |
