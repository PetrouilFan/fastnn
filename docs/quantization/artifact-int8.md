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
