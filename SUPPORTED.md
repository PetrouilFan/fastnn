# Supported surface

This is the evidence matrix for the `v3` branch, as of 2026-11-23. It is a
snapshot of what is implemented and proven, not a plan for what will be. The
contract it measures against is [docs/v3/scope.md](docs/v3/scope.md); the
architecture decision is [docs/v3/adr-0001-one-runtime.md](docs/v3/adr-0001-one-runtime.md).

The headline is deliberately blunt: **the v3 runtime executes the CNN operator
set end to end on the CPU, and the decoder operator set has import and kernel
coverage but no KV/prefill/decode path until phase 6 (#108).** The version-2
artifact round trip exists: `tests/fn3_artifact_v2.rs` prepares YOLOv8n, loads
it with no ONNX source, runs it and compares bit for bit with an in-memory
session. The frontend maps and shapes `Conv`, the pool family,
`BatchNormalization`, `Resize`, `Pad`, `Clip`, `Sigmoid`, `Max`, `Min` and
`TopK` into the typed IR, and the v3 segment runtime executes all twelve on
scalar NCHW kernels. An optional oneDNN CPU engine (`--features cpu-onednn`)
additionally executes 4-D convolution and 2-D matmul, folds an
activation or a residual sum into the producing convolution, and keeps internal
conv-to-conv edges in a channels-last layout; see
[docs/v3/backends.md](docs/v3/backends.md). There is no W8A8 execution path:
`prepare --precision w8a8` refuses with the target's ISA and a measurement (on
this AVX2 host oneDNN's AVX2 int8 convolution is only 0.97-1.20x the blocked
f32 path, because without VNNI the two-instruction dot product has no
MAC-throughput advantage over FMA). In the vocabulary of this file the twelve CNN ops are `runs` on the
CPU, while a model *family* still needs a committed zoo entry with a hash and
licence before it may be called `gated` (see §2 of
[scope.md](docs/v3/scope.md#2-what-supported-means)).

## Model families

| Family | Example | Status | Evidence today | Required for the next status |
|---|---|---|---|---|
| GPT-2-style decoder | tiny GPT-2 (28 MB), trained GPT-2 (498 MB) | `planned` | The v3 frontend imports the graph (opset 12..=22, batched attention `MatMul`, `Range`/`Cast` folding), but the v3 session has no KV cache, prefill or decode: that path is phase 6 (#108). | `imports`: an artifact-first decoder plan; then `runs`: prefill/decode with KV parity; then `gated`: the logits gate |
| Llama-style decoder | Llama-family export with RoPE and grouped-query attention | `planned` | The IR has `Sin`, `Cos`, `Trilu`, `Gather`, `ScatterND`, and the decoder geometry profile still accepts `KvCacheConfig`; no architecture profile, no KV execution. | `imports`: architecture profile and a pinned export; then `runs`: RoPE/KV parity |
| YOLO detection | YOLOv8n, YOLOv8s, YOLO11n | `runs` | YOLOv8n ([1,3,320,320] → [1,84,2100]), YOLOv8s ([1,3,640,640] → [1,84,8400]) and stock YOLO11n (421 nodes, opset 22, with `--shape images=1,3,640,640`) import into the typed IR and run through a prepared artifact; the ORT parity gate is 0 of 705,600 outputs outside tolerance on YOLO11n. No zoo entry/hash/licence is committed. | `gated`: commit the zoo entry (hash + licence) and the COCO 500-image mAP gate (#105) |
| EfficientViT depth | EfficientViT | `planned` | The CNN ops import and execute; there is no pinned export, parity result or depth metric. | `imports`: a pinned export in the zoo; then `runs`: F32 depth parity; then `gated`: abs-rel/RMSE/delta gate |
| ResNet/MobileNet classification | ResNet50, MobileNetV4 | `planned` | `Conv`, `BatchNormalization`, `MaxPool` and `GlobalAveragePool` import and execute; there is no pinned export or accuracy result. | `imports`: a pinned export in the zoo; then `runs`: F32 top-1 parity; then `gated`: ImageNet-subset gate |
| ViT classification | ViT | `planned` | Patch-embed `Conv` and the rest of the CNN set execute; an ONNX `LayerNorm` node is still not mapped (the pinned exports decompose it). | `imports`: frontend coverage; then `runs`; then `gated` on ImageNet-subset top-1/top-5 |

No family is `gated` today. The CNN path is the one with end-to-end execution
and parity evidence on CPU; the decoder has import and kernel coverage but no
session path yet (#108).

## Operators

The table is the full `OpId` registry (`src/fn3/ops/mod.rs`), in discriminant
order. "ONNX input" is the operator type the frontend maps to that id; an
em-dash means the id exists in the IR but no ONNX node maps to it. The Vulkan
column is code inventory: the backend compiles behind `vulkan`, but no v3
session reaches it until phase 6 (#108), so none of those rows is a runnable
target today.

| `OpId` | Code | ONNX input | CPU | Vulkan | Notes |
|---|---:|---|:--:|:--:|---|
| `Gather` | 1 | `Gather` | full | full | Embedding lookup and shape indexing; `batch_dims = 0` only |
| `MatMul` | 2 | `MatMul`, `Gemm`\* | full | full | Rank ≥ 2, batched with broadcast batch dims; the optional oneDNN engine also runs the 2-D `f32` form |
| `Add` | 3 | `Add`, `Gemm`\* | full | full | Broadcasting. The planner folds `add(conv, other)` (same shape and dtype) into the convolution as a sum, with an optional activation after it |
| `Sub` | 4 | `Sub` | full | full | Broadcasting |
| `Mul` | 5 | `Mul` | full | full | Broadcasting. `mul(x, sigmoid(x))` is folded into the producing convolution as a swish epilogue |
| `Div` | 6 | `Div` | full | full | Broadcasting |
| `Neg` | 7 | `Neg` | full | full | |
| `Pow` | 8 | `Pow` | full | full | Broadcasting |
| `Sqrt` | 9 | `Sqrt` | full | full | |
| `Erf` | 10 | `Erf` | full | full | Exact GELU building block |
| `Tanh` | 11 | `Tanh` | full | full | |
| `Gelu` | 12 | — | full | full | Exact (`erf`) form; no ONNX mapping |
| `Relu` | 13 | `Relu` | full | full | |
| `Softmax` | 14 | `Softmax` | full | partial | CPU: a contiguous trailing-axis tensor uses a row-wise three-pass kernel over slices (no per-element index, no allocation, bit-identical to the general path). Vulkan: f32, trailing axis, contiguous |
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
| `Conv` | 41 | `Conv` | runs | — | N-dimensional NCHW/NCDHW scalar kernels; the optional oneDNN engine (`cpu-onednn`) additionally runs 4-D `f32` with `group`, `dilations`, `strides`, explicit pads and bias, and claims the fused forms `conv+activation`, `conv+sum` and `conv+sum+activation` |
| `MaxPool` | 42 | `MaxPool` | runs (scalar) | — | N-D `f32`: `kernel_shape`, `strides`, dilation, explicit pads, `ceil_mode`. `MaxPool` is separable in 2-D, so the max is taken per row first. The optional indices output is rejected |
| `AveragePool` | 43 | `AveragePool` | runs (scalar) | — | As `MaxPool`, plus `count_include_pad` |
| `GlobalAveragePool` | 44 | `GlobalAveragePool` | runs (scalar) | — | Spatial axes collapse to extent 1 |
| `BatchNorm` | 45 | `BatchNormalization` | runs (scalar) | — | Inference form only: one output, 1-D `scale`/`B`/`mean`/`var`, `epsilon`; `training_mode = 1` is rejected |
| `Resize` | 46 | `Resize` | runs (scalar) | — | 4-D nearest/linear, constant `scales` or `sizes`; the `roi` operand is accepted and ignored (its values are defined only for `tf_crop_and_resize`, which is rejected), and cubic is rejected |
| `Pad` | 47 | `Pad` | runs (scalar) | — | Constant/reflect/edge, attribute or constant input pads, negative pads crop; `wrap` is rejected |
| `Clip` | 48 | `Clip` | runs (scalar) | — | Attribute or constant min/max; a runtime bound is rejected |
| `Sigmoid` | 49 | `Sigmoid` | runs (scalar) | — | Folded into the producing convolution when it is the `sigmoid(x)` of `mul(x, sigmoid(x))`; a standalone sigmoid runs scalar |
| `Maximum` | 50 | `Max` | runs (scalar) | — | Two operands with broadcasting; the variadic ONNX form is rejected |
| `Minimum` | 51 | `Min` | runs (scalar) | — | As `Maximum` |
| `TopK` | 52 | `TopK` | runs (scalar) | — | Axis, `largest`/`sorted`, attribute or constant k; two outputs (values and i64 indices) |

\* `Gemm` is not an op id. It is accepted only in the identity form
(`alpha = beta = 1`, `transA = transB = 0`) and composed as `MatMul` + `Add`.

Frontend restrictions that matter in practice:

- The default-domain opset range is 12..=22. Opset 12 is the last revision with
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

The CNN operators are mapped and shaped under #102/#104 and execute on the CPU:
the twelve scalar NCHW kernels in `src/fn3/runtime/kernels/` are the reference,
and the optional oneDNN engine executes the 4-D convolution and 2-D matmul
rows; pooling is scalar on every engine, because the provider's plain-NCHW path
resolves to its reference implementation on x86 without AVX-512 (see
[docs/v3/backends.md](docs/v3/backends.md) §3). Those two rows are also the only
ones the engine claims on AArch64, where it is built and tested but slower than
ONNX Runtime (2.24x at one thread on a Raspberry Pi 5 — see
[docs/v3/aarch64.md](docs/v3/aarch64.md)); the scalar kernels are the portable
path on both targets. The Vulkan column stays `—`; a Vulkan session rejects those ops
instead of staging them to the CPU.

## Dtypes and execution modes

| Mode | Status | Weights | Activations | In the artifact | What may be claimed |
|---|---|---|---|---|---|
| `F32` | `runs` (decoder; CPU, Vulkan optional) | f32 | f32 | Yes | The reference mode for every family; decoder logits have a manual accuracy gate |
| Weight-only INT8 (`weight_int8`) | archived with the 3.0 tree | — | — | No | Nothing; the v3 artifact has no weight-only payload |
| `W8A8` | `refused` (#103) | int8 per-channel (planned) | u8 per-tensor (planned) | No | Nothing yet: `w8a8` parses and `prepare` refuses it with the measured reason. On this AVX2 host oneDNN's int8 convolution is 0.97-1.20x the blocked f32 path (0.83-0.94 ms against 0.94-1.07 ms); without VNNI the dot-product sequence has no MAC-throughput advantage. The executor waits for VNNI/AMX hardware where it can be measured. On AArch64 the hardware *does* have `sdot` and it still loses: oneDNN 3.14.1 has no non-SVE int8 convolution kernel, and int8 measures 4-6.5x slower than f32 on a Cortex-A76 ([docs/v3/aarch64.md](docs/v3/aarch64.md)); W8A8 stays refused there too |
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

- The CNN operator set executes on the CPU and round-trips through the
  version-2 artifact, but no model family is `imports` yet: a family still
  needs a committed zoo entry with a hash and licence. See
  [docs/v3/backends.md](docs/v3/backends.md) for the oneDNN engine's subset.
- `ConvTranspose`, `GatherElements` and `LeakyRelu` remain unmapped.
- The typed-IR import requires every extent to be provable. A model whose
  intermediate shape depends on a runtime value is rejected by name; the
  decoder-shaped `import_onnx_model` keeps its symbolic-path contract.
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
- The fused-segment planner covers convolutions only (an activation, the ONNX
  SiLU and a residual sum); pool, matmul and pointwise chains are not fused, and
  there is no precision assignment. A residual add *after* an activation is not
  fused, because oneDNN's post-op order is sum-then-eltwise (#98).
- Layout assignment is a oneDNN policy (plain or channels-last on internal
  conv-to-conv edges) plus counted conversions; the scalar engine and the plan's
  slots stay NCHW.
- The optional oneDNN CPU engine (`cpu-onednn`) is the first provider: 4-D
  `f32` convolution and 2-D matmul, with explicit-pad
  resolution, output-channel or output-row block threading, a fused epilogue and
  a channels-last layout policy. MLAS/XNNPACK are not implemented (#101, #106).

Runtime and packaging gaps:

- Threads belong to the session: each `Session` builds its own `fn3-worker`
  pool sized by `Threading` (`Auto` = physical cores), and no process-global
  pool exists. The oneDNN engine is built with its sequential runtime and the
  session splits blocks itself.
- The `.fnn3` artifact is format version 2 (`FNN3ART2`): typed graph, one full
  plan per target, one contiguous 64-byte-aligned weight payload mapped
  read-only, quantization metadata (empty for f32) and a SHA-256 per section.
  Two sessions and two processes share the same weight pages; see
  [docs/v3/artifact.md](docs/v3/artifact.md).
- `load` decodes the plan stored in the artifact; it does not run the planner.
  A plan's oneDNN layout reorders are recorded when the engine opens, so the
  artifact carries the policy and the prepare-time list, not a worker-specific
  descriptor choice.
- The CLI has `prepare`, `load`, `run`, `inspect` and `doctor`. There is no
  `bench`, `explain` or `convert` yet.
- The Python API has `prepare`, `inspect`, `load`, `doctor`, `devices` and
  `targets`, and `Model.session().run({name: values})`. There is no
  `benchmark` or `explain` facade yet.
- Vulkan is compiled behind the `vulkan` feature but no session selects it in
  this tree; there is no Vulkan test and no packaging claim until phase 6
  (#108).
- Physical-GPU evidence is manual and per-device. The CI GPU lane uses a
  software Vulkan implementation and is explicitly not physical-GPU evidence
  (`tools/ci/lanes.json`).

Evidence gaps:

- No model zoo is committed. The YOLO gates need an external file plus an ONNX
  Runtime dump and skip with a printed command when either is absent; the stock
  YOLO11n gate needs `--shape images=1,3,640,640`.
- There is no COCO mAP, depth, ImageNet or perplexity result in this tree, and
  no W8A8 accuracy result. The only accuracy gates are F32 CNN parity against
  the scalar reference and against ONNX Runtime (`tests/fn3_cnn_run.rs`,
  `tests/fn3_onednn.rs`).
- No decoder prefill/decode gate exists after the pre-v3 engine was deleted
  (#117); that returns with phase 6 (#108).
- Performance evidence is [docs/v3/status.md](docs/v3/status.md) and
  [docs/v3/backends.md](docs/v3/backends.md), measured on the machine each
  table names. Numbers in [docs/archive/](docs/archive/README.md) were measured
  on the archived runtime and must be cited as such.

## How to check this file

Every row is backed by code or a test in this tree:

- Operator ids and support answers: `src/fn3/ops/mod.rs`,
  `src/fn3/runtime/kernels/mod.rs`, `src/fn3/backend/cpu/onednn/geometry.rs`.
- Frontend mapping and rejections: `src/fn3/import/graph.rs`,
  `src/fn3/import/error.rs`, `tests/fn3_import.rs`.
- Typed IR and shape rules: `src/fn3/ir/`, `tests/fn3_cnn_import.rs`.
- CNN execution and parity: `tests/fn3_cnn_run.rs`, `tests/fn3_onednn.rs`,
  `tests/fn3_constant_folding.rs`, `tests/fn3_shape_binding.rs`.
- Artifact and CLI: `tests/fn3_artifact_v2.rs`, `tests/fn3_cli.rs`.
- Lane honesty: `tools/ci/lanes.json`.

When a row changes, this file and
[docs/v3/scope.md](docs/v3/scope.md) change in the same pull request.
