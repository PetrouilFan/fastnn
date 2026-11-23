# Architecture Improvement Backlog

This backlog complements the source-layout reorganization. It is intentionally
ordered by reduction of semantic and operational risk, not by file size.

## P0 — define the runtime contracts

### Recoverable error model

Public eager execution and Python bindings should return domain errors rather
than panic on compilation, execution, unsupported types, or invalid user input.
Introduce fallible Rust APIs and map them to the Python exception hierarchy.
Keep panics only for proven internal invariant violations.

### Diagnostics instead of embedded debug output

Move production `eprintln!` debug traces, NaN dumps, and debug-triggered panics
behind a typed diagnostics API and an opt-in feature/configuration. Diagnostics
must expose structured pass, graph, arena, and dispatch events; logs are a
rendering of that data, not the only interface.

### Canonical compiler pipeline

Move pass order and compilation policy out of `GraphExecutor` into an explicit
compiler pipeline with typed options. The backend consumes a compiled graph and
lowers it; it does not own graph optimization policy.

### Dtype and quantization redesign

Execute `dtype-redesign.md` before further quantized-kernel expansion. Dtype,
packing, quantization metadata, and compile policy must become orthogonal typed
concepts with explicit integer accumulator/requantization contracts.
The CPU execution extension is specified in
[`cpu-low-bit-engine.md`](cpu-low-bit-engine.md): storage format, decode family,
workload phase, and ISA requirements are separate compiler concepts, and a
measured W4 path gates more experimental sub-4-bit families.

## P1 — remove duplicate semantic ownership

### One compiler authority

Rust owns compiled-IR inference, graph rewrites, quantization, memory planning,
and lowering. Python import code is an adapter. Establish operation-level parity,
then delete duplicated Python shape inference and graph optimizer semantics.

### Typed IR attributes

Replace `HashMap<String, String>` operation attributes with opcode-specific typed
attribute structures or a validated typed attribute map. Kernel lowering must
not parse untyped string conventions repeatedly.

### Separate mathematical semantics from execution strategy

IR opcodes express mathematical operations. Quantized representation, kernel
selection, packing, and backend routing are compiler/lowering concerns. This is
required for clear CPU/WGPU ownership and predictable optimization.

### One serialization design

Define a single owner and explicit versioned formats for model artifacts and
compiled artifacts. Redesign freely; there are no external compatibility
requirements. Do not retain parallel Python/Rust format implementations without
an intentional cross-language contract.

## P2 — simplify public and build surfaces

### Simplify the Python API

Reduce `fastnn/__init__.py` to a small explicit facade. Remove the callable
module wrapper, unnecessary dynamic imports, duplicate re-exports, and aliases
that do not justify their maintenance cost.

### Simplify feature/configuration policy

Define a small set of supported Cargo feature bundles. Replace scattered
environment variables with typed compiler/runtime options. The CLI and Python
bindings should populate the same configuration objects.

### Delete unsupported and historical surfaces

For each `NotImplementedError`, panic-only API, compatibility shim, obsolete
pass, and experimental path: implement it, mark it explicitly experimental, or
delete it. Do not retain public-looking interfaces that fail deep in execution.

### Decide WGPU scope

Choose whether WGPU is a first-class backend with its own lowering/runtime
contract or a deliberately limited experimental backend. Avoid an ambiguous
host-staged fallback architecture presented as general GPU execution.

## P3 — make correctness and performance measurable

### Property and differential testing

Add generated small-graph differential tests, ONNX Runtime comparisons for the
supported operator set, graph-rewrite idempotence tests, shape/execution
agreement checks, and integer quantization reference tests.

Low-bit benchmarks must separate decode from prefill and distinguish storage,
microkernel, operator, and end-to-end wins. Report payload width and effective
bits including metadata, scratch allocation, cache behavior where available,
quality delta, and AOT code size. A model-file-size win alone does not promote a
format to production support.

### Structured profiling

Expose compilation pass timings, graph size before/after rewrites, fusion and
quantization decisions, arena reuse, kernel dispatch, and fallback paths through
one diagnostics/profiling subsystem.

### Test architecture

After source ownership stabilizes, organize tests by behavior and use explicit
harnesses for Rust integration-test discovery. Tests should assert public
behavior and numerical contracts, not private file layout.

## Sequencing

1. Error model, diagnostics policy, canonical pipeline, dtype redesign.
2. IR split and typed attributes.
3. Quantization rewrite and Python/Rust compiler consolidation.
4. Prepared plan, PyO3, CPU control-plane, and serialization redesign.
5. Public API, feature/configuration, WGPU scope, and test-tree simplification.

## Concrete audit findings

### CI and test gates

- CI must never translate segmentation faults, aborts, or missing commands into
  successful results. Remove the current exit-code exception in the Python job.
- Add a required full Rust integration-test job alongside focused test targets.
- Release uses abort-on-panic, so public-path panics are host-process failures,
  not recoverable library errors.

### Dtype and quantization hazards

- Add a direct-output U4/U8 test: executor output sizing has a packed-format
  special case that currently treats U4/U8 differently from other packed types.
- Replace ambiguous packed-format `DType::size()` semantics with explicit logical
  bit-width and exact storage-byte calculation APIs.
- Replace `PackedWord`'s FP32-centric generic unpack/pack/dot contract with
  separate packed storage, integer compute, and float compute contracts.
- Consolidate `PackedTensor` and `QuantizedTensor` unless their audit establishes
  truly distinct roles; the current conversion collapses blockwise metadata into
  a global representation.
- Merge the two activation quantization rewrite implementations into the
  canonical quantization pipeline.

### Build and API surface

- Remove orphan Cargo feature gates or define them, and consolidate overlapping
  OpenBLAS/BLAS configuration into one documented backend policy.
- Replace the callable `fastnn.tensor` module wrapper with ordinary explicit
  Python module/function semantics.
- Define operation support once and generate a capability matrix covering IR,
  eager, CPU, WGPU, autograd, ONNX, and quantized support.
- Concentrate unsafe allocation, byte reinterpretation, arena views, SIMD access,
  and DLPack logic behind narrow safe APIs with sanitizer/Miri-friendly tests.
- Define explicit graph kinds for inference, training-forward, backward, and
  optimizer/update graphs so pass eligibility is not inferred from opcodes.
- Centralize numerical policy: accumulation types, overflow, NaN/Inf behavior,
  reduction ordering, determinism, RNG, and CPU/WGPU tolerance.
