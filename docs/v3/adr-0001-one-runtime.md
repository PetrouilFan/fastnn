# ADR-0001: one runtime, one IR, one plan, one artifact

- **Status:** accepted
- **Date:** 2026-11-22
- **Supersedes:** `docs/roadmap/fn3-architecture.md`, the 3.1 roadmap (#81)
- **Related:** epic #94, children #95–#111

## Context

The tree at `archive/3.0` carries three overlapping surfaces: the 2.x AOT
executor with its `.fnn` format and nine-dtype menu, the fn3 decoder runtime
with its own artifact and planner, and 3.0 compatibility naming that exposes
both. CNNs can only run on the 2.x path; decoders can only run on fn3.

Measured against ONNX Runtime on the same machine and thread count, that
arrangement loses badly: YOLO11n by about 4.7x, YOLO26n by about 5.3x,
EfficientViT by about 2.5x, with worse scaling from one to all cores. The
quantized formats do not pay for themselves: U8 runs slower than F32 on YOLO
with slightly more RSS, I8 duplicates U8, and the 4-bit formats lose
detections. Two numbers explain it:

1. **The execution unit is one operator.** Both runtimes dispatch a `match` on
   an op code per forward. Fusion is a graph rewrite that produces a new fused
   op code, which then needs its own hand-written kernel. So the kernel surface
   grows as patterns x dtypes x instruction sets, most combinations fall back to
   a dequantize-and-do-it-in-F32 path, and the winner is still an unwritten
   kernel.
2. **Layout and precision are not planned.** A 1x1 convolution is a GEMM whose
   left operand is NCHW; without an assigned blocked layout it runs through
   im2col in and out of cache. 1x1 convolutions are the majority of EfficientViT's
   conv time. Similarly, a dtype tag on a tensor is not a compute mode: if a
   kernel does not exist for the tag, the graph quietly dequantizes.

The supporting problems are structural rather than local: CNN import requires
Python plus the `onnx` package and a JSON round trip into Rust; weights are
repacked at runtime behind slot-keyed caches; the thread pool is a process-global
Rayon pool shared by both runtimes, so a session cannot own its parallelism;
and there are two incompatible artifact formats, neither memory-mapped.

## Decision

FastNN v3 is a clean-sheet inference runtime. The 2.x surface is deleted, not
deprecated, and no compatibility layer or dual API is retained.

1. **One IR.** A canonical, typed graph with logical (layout-free) tensor types,
   symbolic dimensions, source locations, an op registry with validated
   attributes, and explicit shape-inference and constant-folding passes. No
   backend layout, no dtype tag standing in for a compute mode, no protobuf or
   Python object in the IR.
2. **The plan's atom is a fused segment, not an operator.** A segment is a
   subgraph bound to one implementation: a library primitive with fused
   post-ops, or one custom kernel. Fusion becomes data the planner emits, so
   adding a pattern does not add a kernel per dtype and instruction set.
3. **Layout is plan data.** Every value slot in the plan carries a layout
   (NCHW, blocked, NHWC, ...). The planner propagates layouts and emits explicit,
   counted reorders. `explain` prints them.
4. **Precision is plan data.** A segment carries F32, W8A8 or W4A8 chosen against
   an accuracy budget from calibration. No runtime dequantize conversions, no
   public nine-dtype menu. Weight-only packing is an artifact-size feature and is
   never claimed as a latency or memory win.
5. **A session owns its threads, arena and KV state.** No process-global pool, no
   import-time pinning, no nested oversubscription. `Auto` means physical cores;
   `Fixed(n)` means exactly `n` workers; multiple sessions coexist with shared
   read-only weights and isolated activations.
6. **Import is Rust-native and there is one artifact.** The strict protobuf
   frontend already in the decoder path is extended with the CNN operator set.
   One artifact carries the graph, per-target plans, memory-mappable weights,
   quantization metadata and hashes. Loading is map plus deserialize; two
   sessions share one weight set.
7. **One high-level API.** `prepare`, `load`, `run`, `explain`, `doctor`,
   `benchmark`, mirrored by the CLI. No public name refers to the old executor.

## Kernel providers

The planner emits requirements (layout, precision, fused post-op list, tile
budget) and never names a library. A provider that cannot satisfy a requirement
reports it as unsupported and the planner falls back to a slower segment.

- **oneDNN** is the primary x86 provider: it exposes layouts, post-ops,
  reorders, prepacking and int8 requantization as first-class primitives, which
  is exactly the vocabulary the planner speaks. It also makes true W8A8
  reachable on AVX2-only hosts.
- **MLAS** is the designated fallback if oneDNN's build or packaging blocks a
  platform.
- **XNNPACK** becomes relevant only if ARM ships as a first-class target.
- **ONNX Runtime is a test oracle, never a backend.**

Custom kernels remain for what a library cannot express well: attention, KV
handling, normalization, and a portable scalar path that keeps every model
runnable.

## Consequences

Positive: the kernel surface grows with *implementations*, not with patterns
times dtypes; layout and precision decisions become inspectable and testable
before execution; one artifact removes the ONNX-at-load-time requirement and the
weight copies; sessions become composable in a library or a server; and the
decoder work already invested in strict import, sessions, KV and Vulkan becomes
the decoder profile of the same runtime instead of a parallel product.

Negative: a real rewrite, measured in months not days, with a period where the
CNN path is slower than the old one because the scalar fallback runs first.
oneDNN adds a C++ build and an optional wheel extra. Deleting the old surface
removes working features (eager training-era API, nine dtypes, `.fnn`) before
their replacements exist, which is acceptable only because there are no users.

Mitigation: the scalar segment executor is built and validated against ONNX
Runtime numerics *before* oneDNN is integrated, so performance work never has to
double as correctness work. The v3 branch merges back to `main` as soon as both
CNNs run correctly, rather than living as a long-running rewrite branch.

## Alternatives rejected

- **Fix the 2.x path incrementally.** The multiplicative kernel surface is the
  problem; adding kernels makes it larger, not smaller. The 3.1 work already
  demonstrated this: each dtype needed its own path and still lost to ORT.
- **Keep both surfaces and unify later.** Two runtimes is what produced the
  current split, duplicated planners and shared global state.
- **Use ONNX Runtime as the backend.** It would meet the latency target by
  removing the product.
- **Hand-write VNNI int8 kernels.** The reference host has AVX2 only, so the
  kernel could not be validated where it is developed, and oneDNN already ships
  the AVX2 and VNNI paths.
- **Long-lived rewrite branch.** Rewrites that never merge are rewrites that
  never ship.

## What is kept

The decoder runtime's hard-won parts move into the unified core rather than
being rewritten: strict, budgeted protobuf decoding with byte-level errors;
tensor, shape, storage and alignment primitives; the artifact framing and its
bounds checks; session and KV ownership; the Vulkan platform, kernels and
resident execution; capability detection and SIMD helpers; and the adversarial
test corpus (malformed artifacts, truncated payloads, rank and size budgets).

## What is deleted

The 2.x AOT executor, compiler passes, graph IR, eager tensor/nn API, ONNX
converter, `.fnn` I/O, packed-tensor and dtype machinery, the process-global
thread configuration, the per-op CNN kernels as a primary path, the Python
training-era module system and its graph builder, and every test that asserts
the deleted behaviour.

## Invariants that survive the rewrite

- Inference only: no autograd, optimizer, training loop or wgpu backend.
- Backend support is a promise, not a hint: no silent fallback, no
  "unknown op returns its first input".
- Untrusted input is bounds-checked: artifact readers, shape expressions and
  quantization metadata fail with typed errors, never panics.
- Numerics are deterministic for a given plan; thread count does not change
  results.
- A performance claim needs a before-and-after measurement with machine, thread
  count, median, p95, RSS and peak. A negative result is documented, not hidden.
