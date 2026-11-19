# Artifact-only INT8 weights (opt-in)

Pass `--weight-int8` to `fastnn-infer run` or `generate`, or set
`SessionOptions::weight_int8` in Rust. Without the flag the FP32 path is
unchanged. Deployment still needs only the existing `.fnn3` artifact: the file
carries FP32 weights, its size is unchanged, and no conversion tool or ONNX file
is involved. Activations, accumulators, the KV cache and logits stay FP32.

## What is packed, and what falls back

The session walks each live plan and treats a `MatMul` right-hand side as a
candidate only when it is a static 2-D FP32 model weight with `K >= 64` and at
least 4096 elements (the format's §6 policy). Each admitted weight is packed on
first use by the existing `PackedWeight` encoder and cached by model weight ID.
The step then routes:

| Operand | Route |
|---|---|
| admitted static weight | fused INT8 kernel (`matmul_quantized` on CPU, `launch_projection` on Vulkan) |
| dynamic RHS (attention products, the computed LM-head transpose) | the same backend's FP32 kernel |
| weight below the policy floor | FP32 |
| non-FP32 activation, rank < 2, contraction mismatch, strided operand | FP32, with the typed reason |
| weight the encoder rejects (non-finite value, unrepresentable scale) | FP32, reason `packing: …` |
| Vulkan activation or output that is not device-resident / whole-buffer | FP32 |

Nothing is approximated: an instruction either runs the packed kernel or the
backend's FP32 kernel. There is no CPU fallback on a Vulkan session. Requesting
`--weight-int8` on a Vulkan backend whose session is host-staged is rejected at
session creation rather than silently degrading.

`--steps` prints one line per `MatMul` with its route, so coverage is auditable
per step:

```text
weight-int8: packed-matmuls=48/73 fp32-matmuls=25 packed-payload=90243072B \
  dense-weight=...B model-weight=...B route/pack/upload=...
  matmul[7] model.layers.0.attn.c_attn.weight: packed
  matmul[9] ...: dynamic-rhs
```

`packed-payload` is the packed byte total this session owns (codes + scales +
zero-points), not RSS, VRAM allocation size or file size. `dense-weight` is the
byte total of model weights some live consumer still reads as FP32, so
`model-weight - dense-weight` is the dense residency this session avoided.

## Residency on Vulkan

Packing and the one-time device upload happen before the step's submission
opens, so a packed projection is three device buffers bound by handle inside the
resident submission. Before recording, the session collects every instruction
input and graph output of the live plan; a weight is prepared densely only when
a consumer other than a packed `MatMul` RHS reads it. A weight that is both
packed and dense keeps both representations (host FP32 stays owned by the
model), and a later plan that needs the dense form prepares it then. A
proven packed-only consumer therefore never reaches `prepare_persistent`, and
`VulkanBackend::check_staged_input` still rejects any late host upload.

## Correctness on the pinned artifacts

```sh
cargo build --release --no-default-features --features cli,vulkan --bin fastnn-infer
python3 tools/quant/check_artifact_int8.py target/release/fastnn-infer \
  /tmp/opencode/fn3-artifact-exec/tiny-gpt2.fnn3 \
  /tmp/opencode/fn3-68/trained-gpt2.fnn3
```

The script checks both SHA-256 pins, runs artifact-only (no `--onnx`), and
compares every one of the 50,257 logits of one 10-token prefill plus two
teacher-forced decode tokens (4874, 2402) on CPU and on the GTX 1080 Ti
(`471b2bda-51e1-6451-e692-ff5d1e146f1e`). Vulkan runs take
`flock /tmp/opencode/fn3-measure.lock`, require the pinned UUID, a physical
device, `FN3_VK_VALIDATION=1`, zero validation errors and no
`VK_INSTANCE_LAYERS`. Logits are the CLI's four-decimal rendering.

| Artifact | Packed / total per step (CPU & Vulkan) | Max FP32 → INT8 abs (CPU / Vulkan) | Max CPU vs Vulkan INT8 abs |
|---|---:|---:|---:|
| Tiny | 8 / 13 | 0.00870 / 0.00870 | 0.00010 |
| Trained GPT-2 | 48 / 73 | 0.81710 / 0.81710 | 0.00020 |

Argmax and top-five membership matched on every tested step. The trained arm's
max RMS error was 0.67608. These are raw logits over three steps: a correctness
result, not a quality, long-context or generation-fidelity guarantee.

The trained artifact's 25 fallbacks are exactly the graph's dynamic operands:
24 attention products (per layer, the score `Mul_1` and the value transpose
that feeds `present.N.value`) and `transformer.wte.weight_transposed` (the tied
LM head). The 48 packed ones are `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and
`mlp.c_proj` for each of the 12 layers. The LM head is the largest single
projection and stays FP32 because its RHS is an instruction output, not a
weight; supporting it would need a packed copy of the transposed embedding
table rather than a transpose at run time.

## Weight bytes: dense residency avoided

Both arms print their counters; `shared-weight=0B` on both artifacts means no
weight is both packed and dense, so the two numbers do not double-count.

| Artifact | Model weights | Dense under INT8 | Packed payload | INT8 weight bytes | Avoided |
|---|---:|---:|---:|---:|---:|
| Tiny | 27,833,180 B | 26,260,132 B | 417,792 B | 26,677,924 B | 1,155,256 B (4.2%) |
| Trained GPT-2 | 497,764,984 B | 158,022,256 B | 90,243,072 B | 248,265,328 B | 249,499,656 B (50.1%) |

On the trained artifact the dense remainder is the token embedding, the
position table, every bias and every LayerNorm parameter, because none of those
is a packed `MatMul` RHS. These are API-level bytes (`Storage::len` for dense
weights, `PackedProjection::payload_bytes` for packed), not RSS, VRAM, allocator
or file size; the `.fnn3` file is unchanged in both arms.

## Timing (separate from correctness; validation off)

Three cold processes per arm, medians quoted, same GPU, same prompt and decode
tokens, `run --steps --top-k 1`, `FN3_VK_VALIDATION=0`, no `VK_INSTANCE_LAYERS`,
serialized with `flock /tmp/opencode/fn3-measure.lock`. The FP32 arm prepares all
dense weights during session setup; the INT8 arm packs and uploads inside the
first step's `route/pack/upload` counter, and that time is part of its prefill
wall.

| Artifact / mode | Prefill | Decode 1 | Decode 2 | First pack+upload | Process wall |
|---|---:|---:|---:|---:|---:|
| Tiny FP32 | 9.15 ms | 5.81 ms | 5.56 ms | — | 0.370 s |
| Tiny INT8 | 46.09 ms | 5.93 ms | 5.71 ms | 33.97 ms | 0.375 s |
| Trained FP32 | 98.67 ms | 63.52 ms | 62.31 ms | — | 3.725 s |
| Trained INT8 | 1.06 s | 67.42 ms | 66.93 ms | 899.89 ms | 4.387 s |

```sh
env -u VK_INSTANCE_LAYERS -u FN3_VK_STAGED FN3_VK_VALIDATION=0 \
  flock /tmp/opencode/fn3-measure.lock \
  fastnn-infer run --model <artifact> --device vulkan:<uuid> \
    --tokens 464,2068,7586,21831,18045,625,262,16931,3290,13 \
    --decode 4874,2402 --top-k 1 --steps [--weight-int8]
```

No speedup is claimed. Steady-state decode is 2% slower on the tiny artifact and
6–7% slower on the trained one (67.4/66.9 ms against 63.5/62.3 ms), so the packed
path does not beat the FP32 matmul at this shape. The first call costs 0.90 s of
packing on the trained model and 34 ms on the tiny one; a later artifact-embedded
packed section would remove that cost without changing the dispatch path or the
steady-state result.

## Limits

- Packing is per session and per weight, from the loaded FP32 artifact; the
  artifact format is unchanged and holds no packed data.
- Only static 2-D FP32 weights are candidates. A dynamic RHS, a weight below
  the size floor, a weight whose encoder rejects it, and any operand layout the
  fused kernel rejects stay FP32 with a printed reason. A weight that is packed
  and also read by a non-`MatMul` consumer keeps both copies and shows up in
  `shared-weight`.
- One GPU, one batch of 1, a 10-token prefill and two decode steps. No long
  context, no KV-capacity sweep, no CPU timing, no comparison against the
  standalone `QuantCpuBackend` thread budget (the session path is serial).
- The reported bytes are API counters. Device allocation rounding, staging
  buffers, the plan cache and the KV cache are outside them.
