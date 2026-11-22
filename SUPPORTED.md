# Supported surface

This is the evidence matrix for the `v3` branch, as of 2026-11-22. It is a
snapshot of what is implemented and proven, not a plan for what will be. The
contract it measures against is [docs/v3/scope.md](docs/v3/scope.md); the
architecture decision is [docs/v3/adr-0001-one-runtime.md](docs/v3/adr-0001-one-runtime.md).

The headline is deliberately blunt: **the runtime executes the decoder operator
set only.** The frontend now maps and shapes the CNN operator set (`Conv`, the
pool family, `BatchNormalization`, `Resize`, `Pad`, `Clip`, `Sigmoid`, `Max`,
`Min`, `TopK`) into the typed IR, so an Ultralytics YOLOv8 graph imports with
every extent resolved; **no backend executes those ops yet**, and there is no
W8A8 execution path. In the vocabulary of this file the twelve new ops are
`imports`, not `runs`, and every CNN family stays `planned` until a pinned zoo
entry and an executor exist.

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
| YOLO11/26 detection | YOLO11n, YOLO26n | `planned` | Ultralytics YOLOv8n ([1,3,320,320] → [1,84,2100]) and YOLOv8s ([1,3,640,640] → [1,84,8400]) import into the typed IR: 262 source nodes, 240 typed nodes (22 `Constant` nodes folded into weights), 398 values, every value fully static, proven by `tests/fn3_cnn_import.rs` when the files are present. No execution kernel, no zoo entry/hash, no artifact round trip, so this is not yet `imports` | `imports`: commit the zoo entry (hash + licence) and round-trip the graph through an artifact; then `runs`: F32 detection parity; then `gated`: COCO 500-image mAP gate (#105) |
| EfficientViT depth | EfficientViT | `planned` | The CNN ops import (`Conv`, `BatchNormalization`, `Resize`, `Pad`, pooling); no execution kernel | `imports`: a pinned export in the zoo and an artifact round trip; then `runs`: F32 depth parity; then `gated`: abs-rel/RMSE/delta gate |
| ResNet/MobileNet classification | ResNet50, MobileNetV4 | `planned` | `Conv`, `BatchNormalization`, `MaxPool` and `GlobalAveragePool` import; no execution kernel | `imports`: a pinned export in the zoo and an artifact round trip; then `runs`: F32 top-1 parity; then `gated`: ImageNet-subset top-1/top-5 gate |
| ViT classification | ViT | `planned` | Patch-embed `Conv` imports; an ONNX `LayerNorm` node is still not mapped (the pinned exports decompose it) | `imports`: frontend coverage; then `runs`; then `gated` on ImageNet-subset top-1/top-5 |

No family is `gated` today. The only runnable path is a GPT-2-shaped decoder on
CPU (and, with a `vulkan` build, on Vulkan for the registered op subset). The CNN
ops are `imports`: the typed IR is the only consumer, and the CPU backend reports
them `Unsupported` with the owner of the missing kernel.

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
| `Conv` | 41 | `Conv` | — | — | `imports`. N-dimensional NCHW/NCDHW, `group`, `dilations`, `strides`, `pads`, `auto_pad`; `kernel_shape` is required by the shape rule |
| `MaxPool` | 42 | `MaxPool` | — | — | `imports`. `kernel_shape`, `strides`, `pads`, `dilations`, `auto_pad`, `ceil_mode`; the optional indices output is rejected |
| `AveragePool` | 43 | `AveragePool` | — | — | `imports`. As `MaxPool`, plus `count_include_pad` |
| `GlobalAveragePool` | 44 | `GlobalAveragePool` | — | — | `imports`. Spatial axes collapse to extent 1 |
| `BatchNorm` | 45 | `BatchNormalization` | — | — | `imports`. Inference form only: one output, 1-D `scale`/`B`/`mean`/`var`, `epsilon`; `training_mode = 1` is rejected |
| `Resize` | 46 | `Resize` | — | — | `imports`. 4-D nearest/linear, constant `scales` or `sizes`; the `roi` operand must be empty and `tf_crop_and_resize`/cubic are rejected |
| `Pad` | 47 | `Pad` | — | — | `imports`. Constant/reflect/edge, attribute or constant input pads, negative pads crop; `wrap` is rejected |
| `Clip` | 48 | `Clip` | — | — | `imports`. Attribute or constant min/max; a runtime bound is rejected |
| `Sigmoid` | 49 | `Sigmoid` | — | — | `imports` |
| `Maximum` | 50 | `Max` | — | — | `imports`. Two operands with broadcasting; the variadic ONNX form is rejected |
| `Minimum` | 51 | `Min` | — | — | `imports`. As `Maximum` |
| `TopK` | 52 | `TopK` | — | — | `imports`. Axis, `largest`/`sorted`, attribute or constant k; two outputs (values and i64 indices) |

\* `Gemm` is not an op id. It is accepted only in the identity form
(`alpha = beta = 1`, `transA = transB = 0`) and composed as `MatMul` + `Add`.

Frontend restrictions that matter in practice:

- The default-domain opset range is 12..=17. Opset 12 is the last revision with
  the `split` attribute form of `Split`; both that and the opset-13 input form
  are implemented.
- `Constant` nodes are folded into the weight table, not executed.
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
set the target families need next: `ConvTranspose`, `GatherElements`,
`LeakyRelu`, `Exp`, `Log`, `Abs`, `Identity`, `Dropout`, `If`, `Loop`, `Scan`,
`QuantizeLinear`, `DequantizeLinear`.

`If`, `Loop` and `Scan` are reported as control flow, not as generic unmapped
operators. External tensor data, sparse initializers and local functions are
rejected by name as well.

The CNN operators are mapped and shaped under #102/#104 but have no execution
kernel: every CPU and Vulkan cell above is `—`, and `Backend::supports` returns
`Unsupported` naming phase 2. Import is not execution, and no row above claims
otherwise.

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

- No CNN/vision executor: `Conv`, the pool family, `BatchNormalization`,
  `Resize`, `Pad`, `Clip`, `Sigmoid`, `Max`, `Min` and `TopK` import into the
  typed IR, but no backend has a kernel. YOLO, EfficientViT, ResNet/MobileNet and
  ViT cannot run, and a family is not `imports` until its pinned zoo entry
  round-trips through an artifact.
- `ConvTranspose`, `GatherElements` and `LeakyRelu` remain unmapped.
- The typed-IR import requires every extent to be provable. A model whose
  intermediate shape depends on a runtime value is rejected by name; the
  decoder-shaped `import_onnx_model` keeps its symbolic-path contract.
- The 3.0 artifact format version 1 stores only `axes`, `keep_dims` and
  `epsilon` from `OpParams`. A CNN graph cannot round-trip through it yet; that
  is phase 2 work, so no CNN row claims `imports`.
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
- No CPU kernel provider: the CNN execution path does not exist at all; oneDNN/
  MLAS/XNNPACK are planned (#101).

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
- Typed IR and shape rules: `src/fn3/ir/`, `tests/fn3_cnn_import.rs`.
- Decoder path: `tests/fn3_session.rs`, `tests/fn3_e2e.rs`,
  `tests/fn3_model_format.rs`, and `tests/fn3_trained_parity.rs` for the gate.
- Weight-only INT8: `tests/fn3_trained_int8.rs`, `tests/fn3_quant_cpu.rs`,
  `src/fn3/model/weight_int8.rs`.
- Lane honesty: `tools/ci/lanes.json`, `tests/fn3_ci_smoke.rs`.

When a row changes, this file and
[docs/v3/scope.md](docs/v3/scope.md) change in the same pull request.
