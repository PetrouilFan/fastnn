# Codebase Reorganization Roadmap

## Status

fastnn currently has no external users or downstream projects. The migration may
break public paths, Python names, CLI behavior, serialized formats, and internal
contracts when that produces a cleaner architecture. Preserve numerical
correctness and the targeted test suite unless a change deliberately replaces
the associated behavior.

## Goals

- Make ownership, dependency direction, and entry points obvious from the directory tree.
- Establish clean Rust and Python module boundaries without retaining obsolete
  facade paths merely for compatibility.
- Isolate unsafe CPU arena access, compiler lowering, kernel dispatch, compatibility code, and file-format contracts.
- Establish one authority per compiler concern; duplicate Python and Rust behavior requires explicit parity coverage.
- Replace stale AGENTS.md inventories with concise, source-verified subsystem guidance.

The broader engineering changes are tracked in
`docs/roadmap/architecture-improvements.md`. The dtype/quantization redesign is
specified in `docs/roadmap/dtype-redesign.md` and precedes further quantized
kernel expansion. CPU low-bit storage/compute-family policy is evaluated in
`docs/roadmap/cpu-low-bit-engine.md`.

## Non-goals

- Reducing line count by compressing code or arbitrarily minimizing files.
- Making unreviewed numerical or performance changes incidental to a move.
- Activating reserved prepared-plan features while reorganizing code.
- Removing duplicate Python graph implementations before operation-level parity tests establish the Rust replacement.

## Contract-impact inventory

| Contract | Current owner | Refactor rule | Required proof |
|---|---|---|---|
| Rust crate modules and root re-exports | `src/lib.rs` | Replace paths if a clean module boundary requires it; update internal callers together. | Rust integration tests. |
| Graph API | `src/ir/builder.rs`, `src/ir/node.rs` | Rework types and paths when it clarifies ownership. | IR, shape-inference, quantized-pipeline tests. |
| Tensor/eager API | `src/tensor/`, `src/nn/`, `src/optim/` | Change signatures only as part of an explicit redesign. | Rust library and Python API tests. |
| Python extension module | `src/python/mod.rs`, `pyproject.toml` | Registration and names may change with a deliberate binding redesign. | Build/install plus Python modular API smoke tests. |
| Python facade | `fastnn/__init__.py` | Remove or rename façade aliases when a cleaner API warrants it. | Python import/API tests. |
| CLI | `src/bin/runtime.rs`, `Cargo.toml` | Reorganize or replace CLI behavior with the owning subsystem. | CLI smoke tests after relevant moves. |
| Python CLI | `pyproject.toml` | Revisit entry points only with their Python I/O owner. | CLI help/smoke test. |
| `.fnn` tensor/model files | `src/io/serialize.rs`, `fastnn/io/` | Consolidate format ownership and version deliberately; legacy readers are optional. | New-format round-trip tests. |
| `.fnnc` executable plans | `src/backend/mod.rs`, `src/backend/runtime.rs` | Redesign serialization if prepared-plan architecture benefits. | Prepared-plan execution test. |
| Graph serialization | `src/ir/node.rs` | Redesign/remove serialization if graph ownership becomes clearer. | Graph execution test. |
| Kernel names and instruction routing | CPU backend lowering/dispatch | Rename as part of a coherent lowering/dispatch redesign. | Quantized pipeline and CPU dispatch tests. |
| Features | `Cargo.toml` | Simplify feature combinations when they obscure ownership. | Targeted affected-feature checks. |

## Known compatibility defects to resolve separately

These are correctness/documentation tasks, not mechanical moves:

- `Cargo.toml` and `pyproject.toml` declare version `2.5.0`, while `fastnn/__init__.py` declares `__version__ = "2.4.0"`.
- Existing AGENTS.md files contain stale line counts, file counts, module counts, and version claims. They are not reliable architectural sources.
- Python and Rust both implement graph optimization and shape inference. Their equivalence is not established by the current tree.

## Dependency direction

Target direction:

```text
storage + dtypes
      ↓
tensor + packed_tensor
      ↓
nn + optim + autograd
      ↓
ir
      ↓
compiler passes
      ↓
backend lowering + execution
      ↓
python bindings / CLI / import facades
```

Allowed exceptions must be explicit and narrow:

- `autograd` may construct IR backward graphs but should not own backend dispatch.
- `backend` may consume compiler memory-plan types during the transition, but compiler passes should not depend on backend execution details except shared plan data that is extracted to a neutral module.
- Python bindings are adapters over Rust public or explicitly designated bridge APIs; they should not become a second compiler implementation.

Current coupling requiring attention:

| Coupling | Evidence | Refactor consequence |
|---|---|---|
| Backend and compiler are mutually coupled | CPU/backend files import compiler memory planning; compiler quantization and memory planning import backend types. | Extract neutral plan/metadata types or establish a one-way interface before broad backend splits. |
| Eager tensor and autograd depend on IR/backend | tensor and autograd import `ir` and `backend`. | Preserve behavior while isolating eager compatibility from graph compilation paths. |
| `GraphBuilder` owns both graph construction and execution cache | `src/ir/builder.rs` imports autograd, backend, compiler, and IR. | Split builder operations from `compile.rs`; keep cache ownership explicit. |
| PyO3 binding files share one textual namespace | `src/python/mod.rs` uses seven `include!` files. | Replace with normal modules and per-domain registration functions; preserve names. |
| Python I/O performs compiler-like transformations | `fastnn/io/graph_optimizer.py` and `shape_inference.py`. | Do not relocate blindly; resolve semantic authority with parity tests. |

## Compiler ownership matrix

| Responsibility | Current Rust implementation | Current Python implementation | Target authority | Required decision / proof |
|---|---|---|---|---|
| `.fnn` tensor/model serialization | `src/io/serialize.rs` | `fastnn/io/__init__.py`, `serialization.py` | Undecided | Define format owner and cross-language fixture tests before consolidation. |
| `.fnnc` plan serialization | `src/backend/mod.rs`, `runtime.rs` | None found | Rust | Freeze legacy fixture before `prepared` or backend moves. |
| IR graph serialization | `src/ir/node.rs` | ONNX numeric graph dictionaries | Rust for native IR | Preserve bincode compatibility; define ONNX dictionary adapter boundary. |
| ONNX import/mapping | `src/onnx/` | `fastnn/io/onnx.py` | Undecided | Build operator support matrix and import/execute parity tests. |
| PyTorch export/import | Rust export support exists | `fastnn/io/export.py` | Python facade, Rust runtime | Define supported graph contract and round-trip tests. |
| Shape inference | `src/compiler/passes/shape_inference.rs` | `fastnn/io/shape_inference.py` | Rust for compiled IR | Python remains import-time compatibility until operation-level parity is proven. |
| Constant folding | `src/compiler/passes/constant_folding.rs` | `fastnn/io/graph_optimizer.py` | Rust for compiled IR | Add parity corpus before delegating/removing Python folding. |
| Arithmetic simplification | `src/compiler/passes/arithmetic_simplify.rs` | Partly Python graph optimizer | Rust | State Python import-time exceptions explicitly. |
| DCE | `src/compiler/passes/dead_code_elimination.rs` | `graph_optimizer.py` | Rust for compiled IR | Compare graph outputs/node liveness on shared fixtures. |
| Fusion | `src/compiler/passes/fusion/` | `fuse_silu`, `fuse_conv_bn` in Python I/O | Split by boundary | Rust owns executable-IR fusion; Python may normalize imported format only after parity decisions. |
| Weight quantization | `src/compiler/passes/quantization.rs` | `fastnn/precision.py`, `io/convert.py` | Rust for executable graphs | Verify U4/U8 integer-space behavior and format metadata compatibility. |
| Activation/gradient quantization and Q/DQ pruning | Rust activation/gradient/QDQ passes | Python calibration/precision tools | Rust for executable graph rewrite | Priority parity area; do not introduce a second rewrite path. |
| Calibration data collection | `src/compiler/passes/calibration.rs` | `calibrate.py`, `act_calibrate.py`, `profiler.py` | Mixed | Python may collect observations; Rust owns graph rewrite and compiled representation. |
| Memory planning | `src/compiler/passes/memory_planning.rs` | None found | Rust | Keep backend interface stable; test dynamic-shape tightening. |
| Backend lowering and execution | `src/backend/` | `dag_model.py` wraps `_core.AotExecutor` | Rust | Python remains a thin adapter. |

## Structural work order

1. Add the AGENTS freshness guard.
2. Establish the recoverable error model, typed diagnostics policy, canonical compiler pipeline, and canonical dtype schema.
3. Split IR types, graph storage, builder operation families, compile/cache, and builder tests; remove obsolete paths instead of retaining compatibility facades.
4. Replace stringly IR attributes with typed, opcode-owned attributes.
5. Rework quantization around the canonical dtype model; enforce integer U4/U8 accumulator and requantization contracts.
   Establish typed storage/decode/kernel-family capability selection and a
   measured W4 production baseline before adding production INT3/INT2/ternary
   formats.
6. Separate autograd compatibility stubs, tape metadata, and backward-rule families.
7. Split prepared-plan data, constants, construction, and execution; redesign its serialized representation if the new ownership warrants it.
8. Replace PyO3 `include!` with normal modules and registration functions, then simplify the Python facade.
9. Split CPU backend control plane: buffer safety, caches, lowering, normal dispatch, persistent dispatch, and operation-family adapters.
10. Resolve Python/Rust compiler ownership through parity tests, then delete duplicate graph logic.
11. Simplify feature/configuration policy, decide WGPU scope, and reorganize tests after source boundaries stabilize.

## Target module boundaries

```text
src/ir/
  mod.rs                  # compatibility facade
  types.rs                # NodeId, DimExpr, IrDType, TensorType, ShapeEnv
  opcode.rs               # Opcode and opcode-local data
  graph.rs                # IRNode, ComputeGraph, caches, mutation
  builder/
    mod.rs inputs.rs compile.rs shape.rs tests.rs
    ops_{elementwise,shape,reduce,nn,quantization,training}.rs

src/autograd/
  mod.rs compat.rs tape.rs
  backward/{mod,elementwise,linear,conv,norm,reduce,shape,training}.rs

src/backend/prepared/
  mod.rs types.rs constants.rs build.rs execute.rs

src/backend/cpu/
  mod.rs buffer.rs cache.rs lower.rs dispatch.rs dispatch_persistent.rs
  kernels/{elementwise,reduce,matmul,conv,scalar}.rs
  microkernels/

src/python/
  mod.rs registration.rs
  bindings/{tensor,factories,ops,optim,io,trainer}.rs
  bindings/nn/
```

## Verification gates

Every mechanical move must pass the nearest applicable checks before the next move:

```bash
cargo fmt --check
cargo test --lib
cargo test --test quantized_pipeline -- --test-threads=1
cargo test --test optim_test -- --test-threads=1
cargo test --test autograd_gradient_checks -- --test-threads=1
cargo test --test cpu_arena_telemetry -- --test-threads=1
uv run pytest tests/ -m "not slow" -v
```

For CPU/backend changes also compare existing allocator telemetry and selected CPU benchmarks. Serialization-bearing changes require new-format round trips, not legacy-load fixtures. For Python binding changes, build/install the extension and run the relevant API smoke tests.

## AGENTS.md policy

Each guide must state only local ownership, invariants, source-of-truth files, and targeted validation. Prohibited volatile content includes exact LOC/file counts, copied package versions, source line ranges, exhaustive API inventories, and claims not checked by tests or source.

A future freshness check must reject nonexistent referenced paths, disallowed commands, mismatched version literals, and numeric LOC/file-count claims. It should not pretend to validate free-form architecture prose.
