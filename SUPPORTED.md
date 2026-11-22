# Supported surface

This is the evidence matrix for the `v3` branch, as of 2026-11-22. It is a
snapshot of what is implemented and proven, not a plan for what will be. The
contract it measures against is [docs/v3/scope.md](docs/v3/scope.md); the
architecture decision is [docs/v3/adr-0001-one-runtime.md](docs/v3/adr-0001-one-runtime.md).

The headline is deliberately blunt: **the runtime supports the decoder operator
set only.** There is no `Conv`, `MaxPool`/`AveragePool`, `BatchNormalization`,
`Resize`, `Pad` or `TopK`, so no CNN or vision family can import, and there is no
W8A8 execution path. Every CNN family below is `planned`.

Legend:

- `full` — every documented shape/dtype combination for the op is implemented.
  A request outside the kernel's documented contract is rejected with a typed
  error, never approximated.
- `partial` — a named subset is implemented; requests outside it are rejected.
- `—` — not implemented on that backend.
- Model status is `planned`, `imports`, `runs` or `gated`, defined in
  [scope.md §2](docs/v3/scope.md#2-what-supported-means). The mode table uses
  the same words applied to execution.

## Model families

| Family | Example | Status | Evidence today | Required for the next status |
|---|---|---|---|---|
| GPT-2-style decoder | tiny GPT-2 (28 MB), trained GPT-2 (498 MB) | `runs` | Strict import (257/257 nodes), `.fnn3` prepare/load, CPU prefill and cached decode, KV parity against the pinned fixtures, session-local weight-only INT8; FP32 logits parity gate exists but requires the external trained graph | `gated`: commit the zoo entry and licence, run the accuracy gate in default CI, publish the metric |
| Llama-style decoder | Llama-family export with RoPE and grouped-query attention | `planned` | IR has `Sin`, `Cos`, `Trilu`, `Gather`, `ScatterND` and a GQA-shaped KV interface; no architecture profile and no parity gate | `imports`: architecture profile and a pinned export in the 13–17 opset range; then `runs`: RoPE/KV parity; then `gated`: token and perplexity gate |
| YOLO11/26 detection | YOLO11n, YOLO26n | `planned` | No `Conv`, `MaxPool`/`AveragePool`, `Resize`, `Pad` or `TopK` | `imports`: CNN frontend plus the documented detection adapter (#104); then `runs`: F32 detection parity; then `gated`: COCO 500-image mAP gate (#105) |
| EfficientViT depth | EfficientViT | `planned` | No convolution, normalization or resize ops | `imports`: CNN frontend; then `runs`: F32 depth parity; then `gated`: abs-rel/RMSE/delta gate |
| ResNet/MobileNet classification | ResNet50, MobileNetV4 | `planned` | No `Conv`, `BatchNormalization`, `MaxPool` or `GlobalAveragePool` | `imports`: CNN frontend; then `runs`: F32 top-1 parity; then `gated`: ImageNet-subset top-1/top-5 gate |
| ViT classification | ViT | `planned` | No patch-embed `Conv`; an ONNX `LayerNorm` node is not mapped | `imports`: frontend coverage; then `runs`; then `gated` on ImageNet-subset top-1/top-5 |

No family is `gated` today. The only runnable path is a GPT-2-shaped decoder on
CPU (and, with a `vulkan` build, on Vulkan for the registered op subset).

## Operators

The table is the full `OpId` registry (`src/fn3/ops/mod.rs`), in discriminant
order. "ONNX input" is the operator type the frontend maps to that id; an
em-dash means the id exists in the IR but no ONNX node maps to it.

| `OpId` | Code | ONNX input | CPU | Vulkan | Notes |
|---|---:|---|:--:|:--:|---|
| `Gather` | 1 | `Gather` | full | full | Embedding lookup and shape indexing; `batch_dims = 0` only |
| `MatMul` | 2 | `MatMul`, `Gemm`\* | full | full | Rank ≥ 2, batched with broadcast batch dims |
| `Add` | 3 | `Add`, `Gemm`\* | full | full | Broadcasting |
| `Sub` | 4 | `Sub` | full | full | Broadcasting |
| `Mul` | 5 | `Mul` | full | full | Broadcasting |
| `Div` | 6 | `Div` | full | full | Broadcasting |
| `Neg` | 7 | `Neg` | full | full | |
| `Pow` | 8 | `Pow` | full | full | Broadcasting |
| `Sqrt` | 9 | `Sqrt` | full | full | |
| `Erf` | 10 | `Erf` | full | full | Exact GELU building block |
| `Tanh` | 11 | `Tanh` | full | full | |
| `Gelu` | 12 | — | full | full | Exact (`erf`) form; no ONNX mapping |
| `Relu` | 13 | `Relu` | full | full | |
| `Softmax` | 14 | `Softmax` | full | partial | Vulkan: f32, trailing axis, contiguous |
| `LayerNorm` | 15 | — | full | full | Trailing axis only; no ONNX mapping |
| `ReduceMean` | 16 | `ReduceMean` | full | full | Attribute `axes`; `keepdims` 0 or 1 |
| `ReduceSum` | 17 | `ReduceSum` | full | full | Attribute `axes` only; the opset-13 axes input is rejected |
| `ArgMax` | 18 | `ArgMax` | full | partial | Vulkan: f32/i32 input, i64 output; i64 input rejected. `select_last_index = 0` only |
| `Transpose` | 19 | `Transpose` | full | partial | Vulkan: 4- and 8-byte elements; `Bool` rejected |
| `Reshape` | 20 | `Reshape` | full | partial | `allowzero = 0` only; Vulkan `Bool` rejected |
| `Flatten` | 21 | `Flatten` | full | partial | Vulkan `Bool` rejected |
| `Squeeze` | 22 | `Squeeze` | full | partial | Vulkan `Bool` rejected |
| `Unsqueeze` | 23 | `Unsqueeze` | full | partial | Vulkan `Bool` rejected |
| `Concat` | 24 | `Concat` | full | partial | Vulkan: one axis, at most 16 inputs, 4/8-byte elements |
| `Slice` | 25 | `Slice` | full | partial | Bounds must be statically known; Vulkan: one axis, 4/8-byte |
| `Split` | 26 | `Split` | full | partial | Vulkan: one axis, at most 16 outputs, 4/8-byte |
| `Shape` | 27 | `Shape` | full | full | `start`/`end` attributes not implemented |
| `ConstantOfShape` | 28 | `ConstantOfShape` | full | full | Fill value lifted to a scalar constant input |
| `Expand` | 29 | `Expand` | full | partial | Vulkan `Bool` rejected |
| `Range` | 30 | `Range` | full | full | f32 and i64 |
| `Cast` | 31 | `Cast` | full | full | The only sanctioned dtype-changing op |
| `Equal` | 32 | `Equal` | full | full | `Bool` output |
| `Less` | 33 | `Less` | full | full | `Bool` output |
| `Greater` | 34 | `Greater` | full | full | `Bool` output |
| `Where` | 35 | `Where` | full | partial | Vulkan: f32/i32/i64 values; `Bool` value operand rejected |
| `Trilu` | 36 | `Trilu` | full | — | Triangular masks |
| `Sin` | 37 | `Sin` | full | full | RoPE building block |
| `Cos` | 38 | `Cos` | full | full | RoPE building block |
| `ScatterND` | 39 | `ScatterND` | full | — | Indexed scatter into a copy |
| `MaskedAttention` | 40 | — | — | full | Reserved fused id; CPU composes attention from primitives, Vulkan has the fused kernel. ONNX `Attention`, `MultiHeadAttention` and `GroupQueryAttention` are rejected, not mapped |

\* `Gemm` is not an op id. It is accepted only in the identity form
(`alpha = beta = 1`, `transA = transB = 0`) and composed as `MatMul` + `Add`.

Frontend restrictions that matter in practice:

- `Gather` accepts `batch_dims = 0` only.
- `Reshape` accepts `allowzero = 0` only.
- `ArgMax` accepts `select_last_index = 0` only.
- `Shape` does not implement the opset-15 `start`/`end` attributes.
- `Slice` requires `starts`, `ends`, `axes` and `steps` to be statically known.
- `ReduceSum` accepts the attribute form of `axes`; an axes input is rejected.
- `ConstantOfShape` accepts its fill value as an attribute and lifts it to a
  scalar input, which is the CPU kernel's contract.

The following ONNX operators have no mapping and are rejected by name, with the
node index, node name and operator type. The list is not exhaustive; it is the
set the target families need first: `Conv`, `ConvTranspose`, `MaxPool`,
`AveragePool`, `GlobalAveragePool`, `BatchNormalization`, `Resize`, `Pad`,
`TopK`, `GatherElements`, `Sigmoid`, `LeakyRelu`, `Exp`, `Log`, `Abs`, `Max`,
`Min`, `Identity`, `Constant`, `Dropout`, `If`, `Loop`, `Scan`,
`QuantizeLinear`, `DequantizeLinear`.

`If`, `Loop` and `Scan` are reported as control flow, not as generic unmapped
operators. External tensor data, sparse initializers and local functions are
rejected by name as well.

The missing CNN operators, including `Conv`, the pool family,
`BatchNormalization`, `Resize`, `Pad` and `TopK`, are planned under #102 and
#104. None is implemented, and no row above claims otherwise.

## Dtypes and execution modes

| Mode | Status | Weights | Activations | In the artifact | What may be claimed |
|---|---|---|---|---|---|
| `F32` | `runs` (decoder; CPU, Vulkan optional) | f32 | f32 | Yes | The reference mode for every family; decoder logits have a manual accuracy gate |
| Weight-only INT8 (`weight_int8`) | `runs` (decoder; opt-in, session-local) | int8 codes in groups of 64, f16 scales, i16 zero points | f32 | No; packed on first use per session | A weight-compression option with coverage diagnostics and a manual accuracy gate (top-1/KL); never a latency claim and not a v3 execution mode |
| `W8A8` | `planned` (#103) | int8 per-channel | u8 per-tensor | Planned | Nothing until the #105 gates pass |
| `W4A8` | `planned` (#103) | 4-bit | 8-bit | Planned | Nothing; the archived report records 4-bit candidates losing detections, measured on the archived runtime ([report](docs/archive/reports/fn3-22-quant-format.md)) |
| `F16`, `BF16`, `FP8`, `U4`, `I4`, `F4` | Not planned | — | — | — | No public dtype menu, no storage tag standing in for a compute mode |

The rules behind the table:

- **Storage is not a compute mode.** A dtype tag without a kernel is
  `UnsupportedDType`, not a dequantize-and-run-in-F32 path.
- **Weight-only compression is an artifact-size feature.** It is never claimed
  as latency. Today it is session-local, so the artifact does not change; in v3
  it is packed into the artifact (#99, #103).
- **Quantized execution keeps integer space.** W8A8, when it exists, must fuse
  requantization, bias and activation and must not materialize FP32 weight or
  Q/DQ scratch buffers ([AGENTS.md](AGENTS.md)).
- Any measured accuracy or performance claim needs the gate and the machine,
  thread count, median, p95 and memory recorded (#105, #106).

## Known limitations

Capability gaps:

- No CNN/vision frontend: `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`,
  `BatchNormalization`, `Resize`, `Pad`, `TopK`, `GatherElements`, `Sigmoid` and
  the other operators listed above are absent. YOLO11/26, EfficientViT,
  ResNet/MobileNet and ViT cannot import, let alone run.
- The decoder profile is GPT-2-shaped only. It requires a learned position
  table and the `past_key_values.{i}` / `present.{i}` interface; Llama-style
  RoPE and other decoder interfaces are not detected.
- ONNX `LayerNorm` nodes are not mapped; the pinned GPT-2 export works because
  it decomposes LayerNorm and GELU into primitives.
- No shape buckets and no CNN spatial symbols. The dynamic contract is seven
  decoder symbols with one envelope and a 2^24 per-extent cap. A request
  outside the envelope is `DimOutOfEnvelope`.
- No W8A8 or W4A8 path, no calibration, no quantized dtypes in the IR, and QDQ
  import is not implemented.
- No fused-segment planner, no layout assignment and no precision assignment:
  the plan is a specialized instruction list executed per op (#98).
- No CPU kernel provider: the CNN path does not exist at all; oneDNN/MLAS/
  XNNPACK are planned (#101).

Runtime and packaging gaps:

- Threads are a process-global Rayon pool with a per-call chunk budget (with
  the `parallel` feature), not a session-owned pool. `FASTNN_CPU_THREADS` caps
  the process; the budget can be narrowed per call.
- The `.fnn3` artifact is decoder-shaped format version 1: graph, geometry,
  envelope and weights. It has no per-target plans, no quantization metadata and
  no mmap; weights are read into memory. Two sessions share `Arc` weights in one
  process, not across processes.
- Plans are built and cached per session; they are not stored in the artifact.
- The CLI has `prepare`, `inspect`, `run`, `generate` and `diagnose`. There is no
  `doctor`, `bench`, `explain` or `convert` yet (#107, #104).
- The Python API has `prepare`, `inspect`, `load`, `devices`, and session
  `prefill`/`decode`. There is no `run`, `benchmark` or `explain` facade yet.
- Vulkan is decoder-first and partial: `Trilu` and `ScatterND` have no kernel,
  and twelve of its registered ops are partial, not full. CNN Vulkan is later
  than CPU CNN.
- Physical-GPU evidence is manual and per-device. The CI GPU lane uses a
  software Vulkan implementation and is explicitly not physical-GPU evidence
  (`tools/ci/lanes.json`).

Evidence gaps:

- No model zoo is committed. The pinned tiny GPT-2 has import tests that skip
  when the file is absent; the trained GPT-2 parity gate is `#[ignore]` and
  needs an external 498 MB graph.
- The only accuracy gates are FP32 decoder logits and weight-only INT8 decoder
  logits; both are manual/opt-in. There is no COCO mAP, depth, ImageNet or
  perplexity result in this tree, and no W8A8 accuracy result. KV parity is
  covered by `tests/fn3_e2e.rs`, but only when the external tiny model is
  present.
- No performance claim in this tree is v3 evidence. Numbers in
  [docs/archive/](docs/archive/README.md) were measured on the archived runtime
  and must be cited as such, never as current results.

## How to check this file

Every row is backed by code or a test in this tree:

- Operator ids and support answers: `src/fn3/ops/mod.rs`,
  `src/fn3/backend/cpu/support.rs`, `src/fn3/backend/vulkan/kernels/`.
- Frontend mapping and rejections: `src/fn3/import/graph.rs`,
  `src/fn3/import/error.rs`, `tests/fn3_import.rs`.
- Decoder path: `tests/fn3_session.rs`, `tests/fn3_e2e.rs`,
  `tests/fn3_model_format.rs`, and `tests/fn3_trained_parity.rs` for the gate.
- Weight-only INT8: `tests/fn3_trained_int8.rs`, `tests/fn3_quant_cpu.rs`,
  `src/fn3/model/weight_int8.rs`.
- Lane honesty: `tools/ci/lanes.json`, `tests/fn3_ci_smoke.rs`.

When a row changes, this file and
[docs/v3/scope.md](docs/v3/scope.md) change in the same pull request.
