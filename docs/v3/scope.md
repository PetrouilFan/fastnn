# v3 scope and model contract

- **Status:** accepted
- **Date:** 2026-11-22
- **Issue:** #96
- **Authority:** [ADR-0001](adr-0001-one-runtime.md). The live, per-item matrix
  is [SUPPORTED.md](../../SUPPORTED.md).

This document defines what FastNN v3 runs, what it rejects, and how it fails. It
is a contract, not a roadmap: `planned` means the design reserves the place and
nothing more, and no row becomes a product claim until the evidence in
[§2](#2-what-supported-means) exists. The current state of each row is
maintained in [SUPPORTED.md](../../SUPPORTED.md), not here.

## 1. What v3 is

FastNN v3 is one compiler, one runtime and one artifact for inference only
(#94, [ADR-0001](adr-0001-one-runtime.md)):

- **One compiler.** A strict ONNX frontend produces a canonical, typed,
  layout-free IR with symbolic dimensions and source locations. Shape
  inference, constant folding and planning are explicit passes. The plan's unit
  is a fused segment, not an operator; layout and precision are plan data.
- **One runtime.** A `Session` owns its plan, arena and KV state and executes a
  prepared instruction list. It does not interpret a graph per forward. There is
  no process-global thread pool in the target design (#100).
- **One artifact.** `prepare` is the only step that requires ONNX. The artifact
  carries the graph, weights, per-target plans, quantization metadata and
  hashes; `load` and every session run from it alone (#99).

**Artifact-first inference** is the deployment rule: a production process needs
the artifact, not the source model, the `onnx` package or Python. The v3
`.fnn3` artifact (`FNN3ART2`, format version 2) carries the typed IR, one full
plan per target profile, one contiguous 64-byte-aligned weight payload mapped
read-only, quantization metadata and a SHA-256 per section. `load` maps it,
validates every hash and decodes the stored plan; it does not read ONNX and does
not re-plan. Two sessions, in one process or across processes, share the same
weight pages. The layout, budgets and mapping rules are in
[artifact.md](artifact.md); [`tests/fn3_artifact_v2.rs`](../../tests/fn3_artifact_v2.rs)
prepares YOLOv8n, loads it with no ONNX source and runs it bit for bit against
an in-memory session. Format version 1 (`FNN3PREP`) is refused with a typed
"re-prepare" error and has no compatibility reader (#99).

**Primary workloads** are CNN/vision models (detection, depth, classification)
and transformer decoders (prefill plus cached token-by-token decoding), on CPU
first and Vulkan where the decoder profile supports it (#94, #96, #108).

## 2. What "supported" means

Supported is an evidence level, not an intention and not "it once imported".
There are four states, and a family may only move forward by producing the
evidence named for that transition:

| State | Meaning | Evidence required to enter |
|---|---|---|
| `planned` | In scope; no import path is proven. | The family is listed here and has an owner issue. |
| `imports` | A pinned export imports to the IR and prepares an artifact. | A zoo entry with file hash and licence; `prepare` completes with zero import errors (warnings are listed); the artifact round-trips and `inspect` reads it. |
| `runs` | The prepared model executes in F32 on the named device. | `imports`, plus correct output shapes, finite outputs, and a recorded comparison against a pinned oracle (ONNX Runtime or upstream golden) using the budget in [§10](#10-accuracy-budgets). |
| `gated` | The family is a release gate. | `runs`, plus the accuracy suite executes in default CI on every change with sample count and confidence interval, and a performance record with machine, thread count and percentiles exists. |

A regression that fails a gate moves the family back to `runs`. Evidence is
never grandfathered, and a single-image or single-prompt result is a smoke test,
not a gate (#105). A model family with only `imports` is not supported and must
not be described as such.

The model zoo (#105) is the only place a family's status can be earned. As of
this revision no family is `gated`; see [SUPPORTED.md](../../SUPPORTED.md).

## 3. Model families

| Family | Representative | What v3 must prove | Status now |
|---|---|---|---|
| YOLO11/26 detection | YOLO11n, YOLO26n | COCO mAP@0.5 and mAP@0.5:0.95 on a 500-image subset; decode/NMS outside the graph | `planned` |
| EfficientViT depth | EfficientViT | abs-rel, RMSE and delta metrics against the F32 reference | `planned` |
| ResNet/MobileNet classification | ResNet50, MobileNetV4 | ImageNet-subset top-1/top-5 | `planned` |
| ViT classification | ViT | ImageNet-subset top-1/top-5 | `planned` |
| GPT-2-style decoder | tiny GPT-2, trained GPT-2 | logits parity, top-1/top-5, KV parity, long context | `runs` |
| Llama-style decoder | Llama-family export | RoPE, grouped-query KV layout, token/perplexity parity | `planned` |

The families above are the whole v3 scope. A model outside them may import by
accident, but it is not supported and gets no accuracy or performance claim.

## 4. ONNX input contract

The frontend is strict and Rust-native; Python is not in the import path (#104,
[`src/fn3/import/`](../../src/fn3/import/mod.rs)).

| Dimension | Accepted |
|---|---|
| Container | One self-contained model file. External tensor data, sparse initializers and local functions are rejected by name. |
| IR version | 4 through 10 inclusive. |
| Opset, default domain | 12 through 22 inclusive. Opsets 18–22 are accepted because the operators this importer implements did not change shape or attribute spelling across them, and a current Ultralytics YOLO export declares 22; an attribute a supported op does not understand is still rejected by name. Opset 12 is the last revision whose `Split` carries its sizes in the `split` attribute; both that form and the opset-13 input form are implemented. Opset 13 moved the axes of `Squeeze`/`Unsqueeze`/`Split`/`Slice` from attributes to inputs; the importer implements that input form. `ReduceMean` is read in its attribute form, which is the form defined through opset 17. |
| Domains | The default ONNX domain only. Any other domain is reported, not guessed. |
| Dtypes | `f32` compute, `i64` indices/shapes, `i32` auxiliary integers, `bool` masks. `f16`, `bf16`, `f64`, quantized and string types are rejected. |
| Architecture | The decoder entry point requires one GPT-2-shaped profile: token and learned position tables, per-layer projections, `past_key_values.{i}` inputs, `present.{i}` outputs and a `logits` output. The typed-graph entry point has no architecture profile; any graph the operator table covers converts, with every extent proven or a node-named error. |

Widening the opset range or adding an architecture profile is a deliberate
change to this table, tested against a pinned export. It is not a side effect of
a parser change (#104). The current decoder profile is detected structurally;
a graph that does not expose the required interface is rejected, not guessed.

## 5. Operator policy

The IR names operations with a stable `OpId` and validates attributes, rank,
dtype and shape before execution ([`src/fn3/ops/mod.rs`](../../src/fn3/ops/mod.rs)).
The rules are (#94, #96):

1. **One ONNX node maps to one `OpId`**, with three documented exceptions:
   `Gemm` becomes `MatMul` + `Add` and is accepted only in the identity form
   (`alpha = beta = 1`, `transA = transB = 0`); `ConstantOfShape` lifts its fill
   value into a scalar constant input; `Constant` is folded into the weight
   table and produces no runtime op. Any other composition is an error, not an
   approximation ([`src/fn3/import/graph.rs`](../../src/fn3/import/graph.rs)).
2. **No fallback returns an input.** An unmapped ONNX operator, an unsupported
   attribute, a dtype mismatch or an unimplemented reserved op is an error that
   names the node index, node name and ONNX op type. Substituting an input, a
   zero or a dequantized approximation is forbidden.
3. **Control flow is rejected.** `If`, `Loop` and `Scan` are outside the model
   contract, not lowered or unrolled.
4. **The ONNX attention operators are rejected**, not mapped to the reserved
   fused `MaskedAttention` id. That id is a backend contract used by the Vulkan
   decoder profile; the importer reports the owning issue instead of guessing a
   decomposition.
5. **Adding an operator is a traced change.** It needs an IR id, shape
   inference, importer mapping, planner handling, a kernel on each backend that
   claims it, a test at each boundary, and an update to
   [SUPPORTED.md](../../SUPPORTED.md).

The current operator table is in [SUPPORTED.md](../../SUPPORTED.md). The CNN
operator set (`Conv`, the pool family, `BatchNormalization`, `Resize`, `Pad`,
`Clip`, `Sigmoid`, `Max`, `Min`, `TopK`) imports into the typed IR but is not
executed; executor work is phase 2. A family reaches `imports` only after its
pinned zoo entry round-trips through an artifact, exactly as §2 requires.

## 6. Dynamic shapes

A dynamic dimension is a named symbol with declared inclusive bounds; it is not
an unchecked runtime value. The current decoder contract has seven symbols
(`batch`, `seq`, `past`, `total`, `n_heads`, `n_kv_heads`, `head_dim`), a
per-model envelope, and a hard policy cap of 2^24 per extent
([`src/fn3/shape.rs`](../../src/fn3/shape.rs)). A request outside the envelope is
`DimOutOfEnvelope`, naming the symbol, the request and the bound.

v3 adds **shape buckets** (#98): a bucket is a concrete shape specialization
the planner compiles and caches, selected from the model's declared set. The
rules are:

- A bucket is a plan cache key, not an input transformation. The runtime never
  pads, crops or resizes an input to make it fit a bucket.
- If no bucket matches, the planner builds a **bounded fallback** plan for the
  requested shape from the IR. The fallback is bounded by the same envelope and
  by plan budgets, and its construction is reported by the planned `explain`
  (#107), not hidden.
- If the requested shape is outside the envelope, the request fails with
  `DimOutOfEnvelope`. There is no unbounded dynamic-shape execution and no
  silent re-plan per operator.

Today only the envelope exists: one plan per concrete environment, cached by
`PlanId`. Buckets, CNN spatial symbols and the fallback plan are not implemented.

## 7. Quantization and QDQ

Precision is a planner decision, not a tensor tag ([ADR-0001](adr-0001-one-runtime.md),
#103):

- **Execution modes** are F32, W8A8 and optional W4A8. There is no public
  nine-dtype menu and no F16/BF16/FP8 mode.
- **QDQ is an import annotation, not a runtime op (planned, #104).**
  `QuantizeLinear` and `DequantizeLinear` patterns are normalized into
  per-tensor/per-channel scales and zero points on the IR; they never execute as
  Q/DQ pairs and are never converted to F32 around a kernel (#103).
- **Integer-space execution is preserved.** A quantized segment computes in
  integer space with fused requantization, bias and activation. FP32 scratch
  buffers that merely re-create the unquantized graph are a defect, not an
  implementation choice.
- **Weight-only packing is separate.** It is an artifact-size feature and is
  never a latency claim. Today's `weight_int8` option packs eligible static
  `MatMul` weights per session with F32 activations; the artifact is unchanged
  and other instructions stay F32. It is not W8A8.
- **QDQ that cannot be normalized is rejected at import** with a node-level
  error. A calibration failure is explicit and recoverable; it never produces a
  silently mixed plan.

Today the IR has no quantized dtype, QDQ operators are unmapped and rejected,
and no W8A8 path exists. The only quantized work is the session-local
weight-only option above.

## 8. Devices and CPU features

- **`device="auto"` is the only request allowed to choose (#96).** Every explicit
  `cpu` or `vulkan` request either resolves to that device or fails with
  `DeviceUnavailable`, naming the request and listing the devices this build can
  execute on. Falling back from Vulkan to CPU would turn a Vulkan measurement
  into an unlabelled CPU run and is forbidden
  ([`src/fn3/device.rs`](../../src/fn3/device.rs)).
- **CPU is always available.** The portable scalar path is the floor and must
  build with `--no-default-features`. AVX2 kernels are selected at runtime after
  feature detection; the binary does not require `-C target-cpu=native` or a
  global target-feature flag ([`src/fn3/backend/cpu/simd.rs`](../../src/fn3/backend/cpu/simd.rs)).
- **CPU feature sets are explicit plan data (planned, #98/#99).** A v3 plan
  records the instruction set its segments require (for example AVX2 or
  AVX-VNNI). A host that lacks a required feature rejects that plan and may build
  the portable fallback; it never executes a substitute kernel silently.
- **Kernel providers do not change the contract.** oneDNN is the primary x86
  provider and MLAS the designated fallback; XNNPACK becomes relevant only if
  ARM ships as a first-class target (#101). Their absence changes the plan and
  its speed, not the model contract or the public API.
- **Vulkan is optional and decoder-first.** It is a non-default build feature;
  requesting it in a build without it fails before any work. CNN Vulkan is later
  than CPU CNN (#96). Software-ICD output is CI evidence and never counts as
  physical-GPU evidence ([`src/fn3/backend/vulkan/AGENTS.md`](../../src/fn3/backend/vulkan/AGENTS.md)).

Threads are not in this section because they are not a device property. The
target rule is that a session owns its workers (`Auto` = physical cores,
`Fixed(n)` = exactly `n`, no global pool, #100). Today the CPU backend uses a
process-global Rayon pool with a per-call chunk budget; that is a known
limitation, not the v3 contract.

## 9. Errors

Every user-visible failure is one of a small, typed set (#96, #110). No failure
is a panic, a silent default, or a partially executed model.

| Failure | Detected at | Type | Message contract |
|---|---|---|---|
| Unsupported operator | Import (`prepare` and `doctor`) | `ImportProblemKind::UnmappedOp`, `UnimplementedOpId`, `ControlFlow` | Node index, node name, ONNX op type; every independent problem is reported, up to the diagnostic cap |
| Backend cannot run an op or shape | Plan preflight, then execution | `Fn3Error::UnsupportedOp` | Op, backend, reason and owning issue |
| Dynamic shape outside the envelope | Session creation or step | `Fn3Error::DimOutOfEnvelope` | Symbol, requested extent, inclusive bounds |
| Shape/rank/count budget exceeded | Import, load or session | `ImportProblemKind::LimitExceeded`, `Fn3Error::CapacityExceeded`, `Fn3Error::InvalidShape` | What was requested, what the limit is |
| Out of memory | Allocation | `Fn3Error::Allocation`, `AllocationSizeUnknown` | Byte request when known; a failed step leaves the KV extent and step counter unchanged |
| Device unavailable | Device resolution | `Fn3Error::DeviceUnavailable` | The request, the reason and every device this build can run; CLI exit 4 |
| Invalid artifact | Load | `FormatErrorKind` (`BadMagic`, `ObsoleteFormat`, `UnsupportedVersion`, `HashMismatch`, `MalformedAt`, `Truncated`, ...) | Section and offset; no partial read; CLI exit 3 |
| Invalid model or attributes | Import, plan or execution | Mismatch variants of `ImportProblemKind` and `Fn3Error` (`DTypeMismatch`, `ShapeMismatch`, `RankMismatch`, `ArityMismatch`) | Expected versus found, with the op named |

**The rule that matters most:** an unsupported model fails **before execution**
with a node-level message. Import validates every node and returns the independent
problems in file order, up to the diagnostic cap; a failed import writes no
artifact. Loading validates the artifact before allocating; a failed load opens no
session. Plan preflight asks the backend `admits` for concrete operand shapes and
dtypes; a rejected plan never runs. If a kernel rejects a shape that could not be
known earlier, it fails before writing an output. A model is never partially
executed and then reported as unsupported at token 400.

`fastnn-infer doctor model.onnx` (and `fastnn::inference::doctor`) is the
preflight form of the same checks: it imports, runs plan preflight and reports
unsupported nodes, dynamic-shape limits, dtype problems and expected memory
without writing an artifact or opening a session. `prepare` and `doctor`
therefore reject the same models; only `prepare` writes the artifact.

## 10. Accuracy budgets

A budget is a threshold for a gate, not a measurement. The decoder logits gate
and the weight-only INT8 decoder gate are the only implemented accuracy gates;
both are manual and opt-in. Everything else in this table is policy owned by
#105 and is not evidence until it runs.

| Family | Metric | Budget |
|---|---|---|
| GPT-2-style decoder, F32 | Elementwise logits versus the pinned oracle | `atol = 5e-4`, `rtol = 1e-4`, zero violations |
| GPT-2-style decoder, F32 | Aggregate logits | mean absolute delta `<= 5e-5 + 5e-6 * mean reference magnitude`; centred RMS `<= 5e-5 * reference RMS` |
| GPT-2-style decoder, F32 | Token agreement | top-1 equal and top-5 overlap 5/5 on the gate prompts |
| YOLO11/26 | COCO 500-image mAP@0.5 and mAP@0.5:0.95 | F32 within 0.2 point of the pinned oracle; W8A8/W4A8 within 0.5 point of F32; at least 99% detection agreement at IoU 0.5 |
| EfficientViT depth | abs-rel, RMSE, delta1 | F32 abs-rel and RMSE within 1% relative of the oracle; delta1 at most 0.2 point below; quantized modes within 1% abs-rel of F32 and delta1 at most 0.2 point below F32 |
| ResNet/MobileNet/ViT | ImageNet-subset top-1 and top-5 | F32 within 0.2 point of the oracle; quantized modes within 0.5 point of F32 |
| GPT-2-style decoder, quantized | top-1/top-5 and perplexity | Same top-1 and top-5 agreement as F32; perplexity within 1% of F32 |

The F32 decoder gate is implemented by
[`tests/fn3_trained_parity.rs`](../../tests/fn3_trained_parity.rs); the
weight-only INT8 gate is [`tests/fn3_trained_int8.rs`](../../tests/fn3_trained_int8.rs).
Both need an external trained model (the F32 gate the ONNX graph, the INT8 gate
the prepared artifact) and neither is part of the default test run, so even these
budgets are currently manual gates. No performance number appears in this
document: performance evidence belongs to #106 and must name the machine, thread
count, median, p95 and memory.

## 11. Adapters

The generic importer maps ONNX semantics only (#104). It must not branch on a
model name, producer string, file name or metadata to change a mapping, and it
must not contain per-model graph surgery. A family-level profile (for example the
decoder geometry contract) is allowed only when it is named, documented,
structural and tested.

When a common export genuinely needs a graph rewrite, it is an **adapter**: a
named transform with a declared input/output contract, its own tests, a pinned
model in the zoo, and a report that says it ran. Adapters are never applied
silently, and `doctor` (#104) reports the ones a model would need.
No adapter exists today; the importer accepts or rejects the graph as exported.

## 12. Non-goals

The following are outside v3 and are not planned behind another name (#96):

- Training, fine-tuning, autograd, optimizers, losses, data loaders and
  schedulers.
- Arbitrary control flow: `If`, `Loop`, `Scan`, subgraphs and local functions.
- Custom operator domains and vendor libraries.
- Every ONNX operator. The operator table is intentionally small.
- CUDA, ROCm, Metal, WebGPU and any wgpu backend.
- Distributed or multi-node inference, model sharding and serving frameworks.
- Tokenization and detokenization. The API takes token IDs.
- Detection decode, NMS and other post-processing inside the graph. If fastnn
  ships them, they are a separate, tested post-process adapter.
- A public dtype menu. F16, BF16, FP8, U4, I4 and F4 are not execution modes.
- Compatibility with the 2.x executor, `.fnn`, `.fnnc` or the 2.x API. The old
  surface is archived at tag `archive/3.0`; see
  [the archive note](../archive/README.md).

## 13. Changing this contract

A change to this document is a product decision, not a documentation edit. A
capability moves from *planned* to *imports*/*runs*/*gated* only with the
evidence in [§2](#2-what-supported-means), and the corresponding row in
[SUPPORTED.md](../../SUPPORTED.md) changes in the same pull request. A negative
result is recorded and the work closes; it is not retold as a win
([`AGENTS.md`](../../AGENTS.md)).
