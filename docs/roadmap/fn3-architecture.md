# FastNN 3.0 architecture decision

**Status:** accepted for the alpha candidate, 2026-11-18.
**Owner:** FN3-02 (#48). This file is the authority for the 3.0 core contract.
**Base of record:** `a2c0759fff8bdcb7c943e53442f3382b86e59b66` (crate version 2.6.0).

The 3.0 candidate lives in `src/fn3/`. It is a new inference-only core built
beside the 2.x tree so parallel workers never collide with 2.x paths. FN3-19
(#65) promotes and renames the tree when the 2.x modules are deleted; until then
`fn3` is a temporary name and the two trees share only `Cargo.toml`.

## 1. Scope decision (do not re-litigate)

FastNN 3.0 is **inference-only**. In scope: tensor metadata and storage, a
portable CPU reference, strict import, a versioned prepared-model format, bounded
dynamic shapes, Vulkan compute, session-owned KV state, an inference CLI and a
minimal Python inference API.

Out of scope for 3.0, permanently: autograd, optimizers, training loops,
training-state serialization, and the wgpu backend. Quantization is out of scope
for the alpha and is only *reserved*: an op-id range and a format dtype-code
space exist so a later milestone does not renumber anything.

The 2.x tree remains the historical reference. It is frozen at the base commit
above; its behaviour, including the numerical reference values recorded in
`docs/roadmap/llm-runtime.md` (produced at v2.6.0 `e63eda8`), is evidence for
parity work and is not a source of 3.0 design constraints.

### 1.1 Supersession table

`docs/roadmap/codebase-reorganization.md` remains authoritative for the 2.x tree
while it exists. Its training/autograd/quantization/wgpu gates do **not** apply
to the 3.0 candidate:

| 2.x gate (source) | 3.0 resolution | Why |
|---|---|---|
| `cargo fmt --check` (reorg "Verification gates") | **Applies** | Formatting is tree-independent. |
| `cargo test --lib` (reorg gates) | **Applies to 2.x only** | Proves the frozen tree still compiles while `src/fn3/` coexists. It is not a 3.0 behavioural gate. |
| `cargo test --test quantized_pipeline` (reorg gates) | **Superseded** | No quantized execution in 3.0 alpha. |
| `cargo test --test optim_test`, `--test autograd_gradient_checks`, `tests/training.rs` (reorg gates) | **Superseded** | Autograd, optimizers and training are deleted from 3.0 by FN3-19 (#65). |
| `cargo test --test cpu_arena_telemetry` and "allocator/performance comparison" for CPU changes (reorg gates, `src/backend/cpu/AGENTS.md`) | **Superseded for `src/fn3/`** | The 3.0 CPU backend has no arena, no persistent scratch and no allocator telemetry. Its evidence is deterministic reference numerics (`tests/fn3_contracts.rs`) plus the FN3-04 benchmark harness (#50). |
| `uv run pytest tests/ -m "not slow"` and Python facade/extension gates (reorg gates) | **Superseded for the alpha** | 3.0 alpha has no Python surface. FN3-18 (#64) re-enables Python tests for the new minimal API; the 2.x Python tests stay frozen with the 2.x tree. |
| Structural work order 2–4: recoverable error model, IR type split, typed opcode attributes (reorg) | **Superseded** | 3.0 does not extend the 2.x IR. `src/fn3/error.rs` and `src/fn3/ops/` are new contracts. |
| Work order 5: quantized dtype redesign, U4/U8 integer-space gates (reorg, `dtype-redesign.md`) | **Deferred past alpha** | No quantized dtype variant, no quantized op id. Ranges `0x0100..=0x01FF` are reserved. |
| Work order 6: split autograd compatibility/tape/backward rules (reorg) | **Superseded** | Autograd is removed, not refactored. |
| Work order 7: prepared-plan redesign (reorg, `.fnnc`) | **Replaced** | `src/fn3/model` defines `ExecutionPlan`; FN3-09 (#55) defines the versioned format. `.fnnc` is legacy-only. |
| Work order 8: replace PyO3 `include!` with modules (reorg) | **Superseded** | 2.x Python is frozen. FN3-18 (#64) adds a new binding module for the 3.0 API. |
| Work order 9: split CPU backend control plane (reorg, `src/backend/cpu/AGENTS.md`) | **Superseded** | `src/fn3/backend/cpu/` is a new, deliberately small reference backend. |
| Work order 10: Python/Rust compiler ownership parity (reorg) | **Superseded** | 3.0 has no Python compiler. |
| Work order 11: feature policy and WGPU scope decision (reorg) | **Resolved** | wgpu is out of 3.0. Vulkan (#57) is the accelerated path. `src/fn3/` declares no features of its own; FN3-11 adds exactly one, `vulkan`. |
| Contract-impact rows: `.fnn`, `.fnnc`, IR graph serialization (reorg) | **Superseded** | 3.0 defines its own versioned prepared-model format (#55). Legacy readers are optional and never gate 3.0. |
| AGENTS freshness policy (reorg) | **Applies** | Guides must not cite nonexistent paths or volatile counts. |

Everything in `AGENTS.md` (no unjustified `unsafe`, no silent numerical
changes, closest-guide updates) applies unchanged to `src/fn3/`.

## 2. Module tree and dependency direction

```text
src/fn3/
  mod.rs                 facade, re-exports, temporary-name notice
  dtype.rs               DType: f32 | i64 | i32 | bool; size_bytes, is_compute, is_index
  shape.rs               Shape, Strides, DimBound, ShapeEnvelope, LiveDims, Symbol
  error.rs               Fn3Error (one enum, actionable context)
  device.rs              DeviceId, DeviceRequest, DeviceResolveError, resolution
  storage.rs             AlignedBuffer (64-byte), Arc-shared Storage
  tensor.rs              TensorMeta, TensorView, TensorMut, Tensor
  ops/
    mod.rs               OpId registry, OpFamily, OpMetadata
  backend/
    mod.rs               Backend trait, Capabilities, CapabilityFeatures, Support, OpParams
    cpu/mod.rs           CpuBackend: Add (broadcast) + MatMul rank-2 reference
  model/
    mod.rs               ExecutionPlan, Instruction, ModelWeights, PreparedModel,
                         Session, KvCacheConfig, KvCacheState
```

Dependency direction is a DAG; no module may import a module above it:

```text
dtype   device   ops
  \       |      /
        error
          |
     shape  storage
          \   /
         tensor
           |
        backend
           |
         model
```

`src/fn3/` must not import any 2.x module (`crate::tensor`, `crate::ir`,
`crate::backend`, `crate::compiler`, ...). The reverse is also forbidden: 2.x
code must not depend on `fn3`. The only shared file is `src/lib.rs`, which
declares `pub mod fn3;` unconditionally.

## 3. Contract summary

| Contract | Type | Invariant |
|---|---|---|
| dtypes | `DType` | `F32` is the only compute dtype; `I64` indices/shape; `I32` aux; `Bool` masks. Unknown codes are rejected with `UnknownDTypeCode` naming the reader. No implicit conversion except the explicit `OpId::Cast`. |
| shape | `Shape` | Ordered extents; rank 0 = 1 element; zero extents allowed; `numel`/`byte_len` checked (u64 **and** usize) → `ShapeOverflow`. |
| strides | `Strides` | Elements, signed. Positive and non-overlapping for writable views; zero only for broadcast-declared read-only views; negative always rejected → `InvalidStride`. |
| storage | `AlignedBuffer`/`Storage` | 64-byte aligned, zeroed, fixed length. Shared reads via `Arc`; writes require exclusive ownership (`Storage::try_make_mut` → `StorageShared` otherwise). No interior mutability. |
| views | `TensorMeta`/`TensorView`/`TensorMut` | One metadata type; offset in elements; broadcast views are read-only; device affinity lives in metadata. |
| device | `DeviceId`/`DeviceRequest` | `Cpu` or `Vulkan(PhysicalDeviceId)`. Explicit requests resolve to exactly that device or fail. Only `Auto` may choose the CPU. |
| capabilities | `Capabilities`/`Support` | dtypes, `max_buffer_bytes`, `max_shape_rank`, workgroup/thread geometry, feature flags. `Support::{Full, Partial, Unsupported}`. |
| ops | `OpId` | `#[repr(u16)]`, explicit stable discriminants, `ALL`, `name()`, `family()`, `is_composite()`, `from_u16`, reserved ranges. |
| plans | `ExecutionPlan` | Ordered instructions over flat value slots; no read/write aliasing, no duplicate output slot, indices in range. |
| model | `PreparedModel` | Immutable, `Send + Sync`, `Arc` plan + `Arc` weights + device + envelope + optional KV geometry. |
| session | `Session` | Owns KV buffers, live extents, step counter, phase. `Send`, deliberately `!Sync`. Failed step changes neither cache extent nor counter. |

### 3.1 The no-silent-fallback rule

This is a release gate for the alpha, not a guideline. Every case below has a
pinned contract test in `tests/fn3_contracts.rs`.

1. **Device.** An explicit `DeviceRequest::{Cpu, Vulkan, VulkanNamed}` either
   yields that device or `DeviceResolveError`. The message names the request and
   lists available devices. `DeviceRequest::Auto` is the only request allowed to
   choose. `Tensor::zeros` on a device without a backend reports
   `DeviceUnavailable`; it does not allocate host memory.
2. **Operation.** A backend that does not implement an op returns
   `UnsupportedOp { op, backend, reason }`. It must not return an input, a zero,
   or a decomposed approximation. `Support::Partial` names its subset and the op
   still rejects everything outside it.
3. **Shape.** A request outside the alpha contract (for example rank-3 matmul)
   is `UnsupportedOp` with the owner named, never a silently reduced rank.
4. **Dtype.** A kernel rejects a dtype it does not compute on
   (`DTypeMismatch`/`UnsupportedDType`); it never casts.
5. **Session.** A capacity overflow is detected before any mutation
   (`check_capacity`); a failed step leaves live extents and the counter
   unchanged.
6. **Skeleton.** An unimplemented entry point returns `Fn3Error::NotImplemented`
   naming its owner. It never panics and never returns a placeholder tensor.

## 4. Op id table

Discriminants are serialized plan data and capability-table keys. Never
renumber, never reuse.

| Code | Op | Family | Composite | Alpha CPU |
|---:|---|---|---|---|
| 1 | `gather` | data movement | no | unsupported (#53) |
| 2 | `matmul` | linear algebra | no | **full**, rank 2 |
| 3 | `add` | elementwise | no | **full**, broadcast |
| 4 | `sub` | elementwise | no | unsupported (#53) |
| 5 | `mul` | elementwise | no | unsupported (#53) |
| 6 | `div` | elementwise | no | unsupported (#53) |
| 7 | `neg` | elementwise | no | unsupported (#53) |
| 8 | `pow` | elementwise | no | unsupported (#53) |
| 9 | `sqrt` | elementwise | no | unsupported (#53) |
| 10 | `erf` | elementwise | no | unsupported (#53) |
| 11 | `tanh` | elementwise | no | unsupported (#53) |
| 12 | `gelu` | elementwise | yes | unsupported (#53) |
| 13 | `relu` | elementwise | no | unsupported (#53) |
| 14 | `softmax` | normalization | yes | unsupported (#53) |
| 15 | `layernorm` | normalization | yes | unsupported (#53) |
| 16 | `reduce_mean` | reduction | no | unsupported (#53) |
| 17 | `reduce_sum` | reduction | no | unsupported (#53) |
| 18 | `argmax` | reduction | no | unsupported (#53) |
| 19 | `transpose` | data movement | no | unsupported (#53) |
| 20 | `reshape` | data movement | no | unsupported (#53) |
| 21 | `flatten` | data movement | no | unsupported (#53) |
| 22 | `squeeze` | data movement | no | unsupported (#53) |
| 23 | `unsqueeze` | data movement | no | unsupported (#53) |
| 24 | `concat` | data movement | no | unsupported (#53) |
| 25 | `slice` | data movement | no | unsupported (#53) |
| 26 | `split` | data movement | no | unsupported (#53) |
| 27 | `shape` | shape | no | unsupported (#53) |
| 28 | `constant_of_shape` | shape | no | unsupported (#53) |
| 29 | `expand` | data movement | no | unsupported (#53) |
| 30 | `range` | shape | no | unsupported (#53) |
| 31 | `cast` | elementwise | no | unsupported (#53) |
| 32 | `equal` | comparison | no | unsupported (#53) |
| 33 | `less` | comparison | no | unsupported (#53) |
| 34 | `greater` | comparison | no | unsupported (#53) |
| 35 | `where` | comparison | no | unsupported (#53) |
| 36 | `trilu` | comparison | no | unsupported (#53) |
| 37 | `sin` | elementwise | no | unsupported (#53) |
| 38 | `cos` | elementwise | no | unsupported (#53) |
| 39 | `scatter_nd` | comparison | no | unsupported (#53) |
| 40 | `masked_attention` | attention | yes | unsupported (#61) |

Reserved ranges (decoded as `None` today, must not be reassigned):

| Range | Reservation |
|---|---|
| `0x0100..=0x01FF` | quantized ops (post-alpha) |
| `0x0200..=0x02FF` | fused/attention variants (#61) |
| `0x0300..=0x03FF` | structural/planner ops (#56) |

There is intentionally no `Embedding` id: an embedding lookup is `gather` with
`axis = 0` over an `F32` table and `I64` indices, which keeps the ONNX mapping
one-to-one.

## 5. Reserved paths per downstream issue

Parallel workers must only create/modify the paths reserved for their issue.
Files not listed here belong to FN3-02 (#48) until FN3-19 (#65) promotes the
tree; changing a shared contract requires coordination with the core owner.

| Issue | Owner | Reserved paths |
|---|---|---|
| #52 FN3-06 | tensor + host storage | extend `src/fn3/shape.rs`, `src/fn3/storage.rs`, `src/fn3/tensor.rs` (contract-compatible only); `tests/fn3_tensor_storage.rs` |
| #53 FN3-07 | CPU reference | `src/fn3/backend/cpu/kernels/` (new), `src/fn3/backend/cpu/support.rs` (new), `tests/fn3_cpu_reference.rs` |
| #54 FN3-08 | ONNX import + weights | `src/fn3/import/` (new), `src/fn3/model/weights.rs` (new), `tests/fn3_onnx_import.rs` |
| #55 FN3-09 | prepared-model format | `src/fn3/model/format/` (new), `tests/fn3_model_format.rs` |
| #56 FN3-10 | bounded specialisation + plans | `src/fn3/model/plan.rs` (new), `src/fn3/model/specialize.rs` (new), `tests/fn3_plan.rs` |
| #57 FN3-11 | Vulkan memory/dispatch/sync | `src/fn3/backend/vulkan/` (new), `Cargo.toml` feature `vulkan` + deps, `tests/fn3_vulkan_smoke.rs` |
| #58 FN3-12 | session KV + prefill/decode | `src/fn3/model/kv.rs`, `src/fn3/model/session.rs` (new), `tests/fn3_session.rs` |
| #59 FN3-13 | Vulkan transforms/embedding/pointwise | `src/fn3/backend/vulkan/kernels/{transform,embedding,pointwise}.rs` |
| #60 FN3-14 | Vulkan matmul | `src/fn3/backend/vulkan/kernels/matmul.rs` |
| #61 FN3-15 | Vulkan norm/reduce/attention | `src/fn3/backend/vulkan/kernels/{norm,reduce,attention}.rs` |
| #62 FN3-16 | end-to-end pinned model | `tests/fn3_e2e/` |
| #63 FN3-17 | inference CLI | `src/fn3/cli/` (new), `src/bin/fn3-infer.rs` (new), `tests/fn3_cli.rs` |
| #64 FN3-18 | minimal Python API | `src/fn3/python/` (new, behind the existing `python` feature), `fastnn/fn3/` (new), `tests/test_fn3_api.py` |
| #66 FN3-20 | CI | `.github/workflows/`, `tests/fn3_ci_smoke.rs` |

`src/fn3/model/mod.rs` and `src/fn3/backend/cpu/mod.rs` stay single files owned by
#48; the issues above split them into the listed submodules when their work
lands. `src/fn3/mod.rs`, `error.rs`, `dtype.rs`, `device.rs`, `ops/`, and
`backend/mod.rs` are core-owned for the whole alpha.

## 6. Extension points per owner

### CPU owner (#53)

1. Add the kernel under `src/fn3/backend/cpu/kernels/`.
2. Add the id to `CpuBackend::supports` with the honest `Support` value.
3. Add a match arm in `CpuBackend::execute_op`; call `check_arity`, validate
   dtypes with `TensorMeta::require_dtype`, and validate shapes with explicit
   expected/actual strings.
4. Write every output element or return an error. Follow the reference order in
   `Add`/`MatMul` (`f32` accumulation in ascending `k`) so faster kernels stay
   bit-comparable.
5. Index storage through `TensorMeta::{offset, strides}`; do not assume
   contiguity unless you checked `is_contiguous`.

### Importer owner (#54)

- Map each ONNX node to exactly one `OpId`. An unmapped node is an error naming
  the node, domain, opset and input types. Never map to an input as a fallback.
- Decode initializer dtype codes with `DType::from_code(code, "typed weights")`
  and keep the original bytes; build `ModelWeights` only after every initializer
  validated.
- Attribute payloads that `OpParams` cannot express are a blocking gap: extend
  `OpParams` in coordination with #48 rather than dropping the attribute.

### Vulkan owners (#57, #59, #60, #61)

- FN3-11 declares `feature = "vulkan"` in `Cargo.toml`, adds
  `#[cfg(feature = "vulkan")] pub mod vulkan;` to `src/fn3/backend/mod.rs`, and
  appends enumerated devices in `src/fn3/device.rs::available_devices`.
- Implement `Backend` with real `Capabilities` read from the physical device
  (dtypes, `max_buffer_bytes`, workgroup geometry). `supports` must stay honest;
  a missing kernel is `Unsupported`, not a CPU fallback.
- Vulkan tensors must be allocated by the Vulkan allocator. `Tensor::from_storage`
  documents that host-backed storage tagged as Vulkan is a metadata-only
  test/validation path; a Vulkan backend must reject it.
- `MaskedAttention` (40) is the reserved fused id for #61. Until then it reports
  `Unsupported`.

### Runtime owners (#55, #56, #58)

- #55 owns the bytes: version, dtype codes, tensor table, instruction table.
  `OpId::code()`/`from_u16` and `DType::code`/`from_code` are the only stable
  numbers.
- #56 owns plan construction and specialisation. Output slots are pre-allocated
  by the planner; `ExecutionPlan::execute` never allocates.
- #58 owns the KV append protocol. Extend `KvCacheState` with transactional
  append: validate all extents, then commit, then advance `step`. `prefill`
  moves `Empty → Ready`; `decode` requires `Ready`. A failed step must be a
  no-op for `cache_len()` and `step()`.

## 7. Evidence map

| Rule | Test |
|---|---|
| checked element/byte/stride arithmetic | `shape_element_count_overflow_is_rejected`, `shape_byte_length_overflow_is_rejected`, `stride_span_overflow_is_rejected`, `zero_extent_short_circuits_multiplication` |
| scalar and zero-size shapes | `rank_zero_is_a_one_element_scalar`, `zero_size_tensors_are_legal_and_execute` |
| stride rejection and broadcast views | `negative_strides_are_rejected`, `zero_strides_are_rejected_unless_broadcast_declared`, `overlapping_strides_are_rejected`, `broadcast_views_are_read_only` |
| device affinity and no fallback | `device_mismatch_is_rejected_by_the_cpu_backend`, `explicit_device_requests_never_resolve_to_cpu`, `allocation_on_a_missing_device_is_not_substituted` |
| actionable device errors | `explicit_vulkan_request_names_the_request_and_available_devices`, `named_vulkan_request_lists_available_adapters` |
| op-id stability | `op_ids_are_stable_and_total`, `gpt2_op_set_is_present` |
| unsupported ops are reported | `unsupported_ops_are_reported_not_approximated`, `partial_support_is_not_full_support`, `plan_preflight_rejects_unsupported_instructions` |
| working CPU ops through the plan path | `cpu_add_broadcasts_through_the_backend`, `cpu_matmul_is_rank_two_and_honours_strides`, `two_op_plan_runs_add_then_matmul` |
| storage aliasing rule | `storage_is_aligned_and_casts_are_checked`, `shared_storage_cannot_be_written` |
| dynamic bounds | `envelope_rejects_out_of_bounds_and_names_axis_and_bound`, `exact_envelope_accepts_its_own_dims_and_rejects_another_total` |
| model/session ownership | `prepared_model_is_shareable_and_sessions_own_their_state`, `session_phase_rules_and_atomic_failures`, `kv_cache_capacity_is_checked_before_mutation` |

## 8. Consequences and open items

- `src/fn3/` is a second implementation during the alpha. It is kept honest by
  importing nothing from 2.x and by owning its own tests.
- No Python surface exists for 3.0 yet. `tests/` Python suites still target 2.x
  and must keep passing for the frozen tree until FN3-19 removes it.
- Quantization dtype codes are not assigned. `DType::from_code` rejects
  `4..=255`; the first quantized milestone must add codes and a `DType` variant
  together.
- FN3-19 (#65) must move this file's module tree, drop the `fn3` prefix, delete
  the superseded 2.x gates from CI, and update `AGENTS.md` files in the same
  change.
