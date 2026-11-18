# FN3-14 (#60) — Vulkan FP32 matmul and projections

**Workstream:** GPU matmul · **Milestone:** alpha · **Tracking issue:** #46
**Branch:** `fn3/60-gpu-matmul` · **Base:** `27fc60d9d4b42a3b0e1e20e883daecb5cfa93871`
**Depends on:** #49/#53 (numerical contract), #57 (platform layer)
**Status of this report:** every number below was measured on the named
workstation on 2026-11-18, or is explicitly marked as not run. It is a
measurement on one device, not a general performance claim.

## 1. What landed

| File | Role |
|---|---|
| `src/fn3/backend/vulkan/kernels/matmul.rs` (new content) | kernel declaration, registration, shape/stride validation and the launch path |
| `src/fn3/backend/vulkan/shaders/matmul_tiled.comp` (new) | GLSL source |
| `src/fn3/backend/vulkan/shaders/matmul_tiled.comp.spv` (new) | committed SPIR-V (`glslc --target-env=vulkan1.1 -O`) |
| `src/fn3/backend/vulkan/shaders/matmul_tiled.comp.spvasm` (new) | committed disassembly, the reviewable record of the compiled module |
| `tests/fn3_vulkan_matmul.rs` (new) | conformance, tail/batch/stride coverage, rejection, evidence and timing |
| `docs/reports/fn3-14-gpu-matmul.md` | this report |

No shared file was modified. The kernel-registration site in
`kernels/mod.rs::KernelSet::alpha` already called `matmul::register`, so this
change only had to fill in that module.

### Kernel design

- **16x16 output tile per workgroup, 16-deep shared-memory stage.** 256
  invocations (`local_size = [16,16,1]`), two 16x17 `float` shared arrays
  (2,176 B, far below the 49,152 B limit). The `+1` padding keeps consecutive
  rows off one shared-memory bank.
- **One dispatch per op.** The column tile is `gl_WorkGroupID.x`, the row tile
  is `y`, and the flattened batch slice is `z`. The residency record's
  `workgroups` triple therefore describes the whole operation instead of the
  last of several launches.
- **Batch broadcast in the shader.** Six right-aligned batch extents and two
  sets of effective batch strides are push constants; unused axes are extent 1
  with stride 0, so the decomposition loop is branch-free. An operand axis with
  extent 1 under a larger output extent gets an effective stride of 0, exactly
  as the CPU reference computes it in `batch_strides`.
- **No operand copy.** Both operands are addressed as
  `base + row * row_stride + k * k_stride` with metadata strides, so a
  transposed right-hand side (`Gemm(transB=1) = MatMul(a, w^T)`), an
  offset-window view and an explicitly broadcast read-only view are read in
  place. Only the view's addressed window is uploaded (`required_elements -
  offset` elements), not the whole parent allocation.
- **Accumulation order.** One `f32` accumulator per output element, ascending
  `k`, starting at `+0.0`. A K tail is zero-filled in shared memory and
  `x + 0.0 == x` exactly, so padding cannot perturb the sum. The only source of
  difference from the CPU reference is the driver contracting `acc + a*b` into
  an FMA; measured differences are in the tables below.
- **112-byte push-constant block** (`m, n, k, a_base, b_base, out_base,
  a_row_stride, a_k_stride, b_k_stride, b_col_stride, batch_dims[6],
  a_batch_strides[6], b_batch_strides[6]`), below the 128-byte guaranteed
  `maxPushConstantsSize` and well below this device's 256.
- **One submission per op:** staging write → device-buffer copy → transfer→
  compute barriers → dispatch → compute→transfer barrier → readback copy →
  fence wait. Buffers come from `alloc_buffer` (chunk suballocator), never a
  direct `vkAllocateMemory`; the residency record is appended after the wait.

### Support verdict

| Field | Value | Why |
|---|---|---|
| `Support` | **`Full`** | The whole rank >= 2 contract: arbitrary M/N/K including tails, right-aligned batch broadcast, strided/offset/broadcast read-only operands, empty product. Matches the CPU verdict for `MatMul`. |
| `dtypes` | `[DType::F32]` | The only alpha compute dtype; a non-f32 request is rejected by `require_dtype`, never cast. |
| `features` | `CapabilityFeatures::default()` | `broadcasting` is documented as the *elementwise* flag (owned by #59), `masked_attention`/`quantized`/`persistent_weights` are not matmul. Matmul batch broadcasting is part of this op's own contract, so claiming a capability flag would overstate what the core can decompose. |

Rejected rather than approximated, each naming FN3-14 (#60) where the kernel is
the owner:

| Request | Error |
|---|---|
| rank-1 lhs or rhs | `UnsupportedOp`, reason names #60 and the rank-1 contract exclusion |
| > 6 broadcast batch axes | `UnsupportedOp` (cannot come from a legal rank <= 8 plan) |
| strided (non-contiguous) output | `UnsupportedOp` |
| broadcast (read-only) output | `UnsupportedOp`; note the core rejects it in `TensorMut::new` first |
| zero-stride contraction | `InvalidStride`, the same error the CPU reference returns |
| negative / > u32 stride, > u32 element span | `InvalidStride` / `UnsupportedOp` |
| workgroup count over a device axis limit | `UnsupportedOp` naming batch, M, N and the limit |
| stray attributes (`axes`, `keep_dims`, `epsilon`) | `InvalidShape` |

## 2. Numerical results

All comparisons are GPU (`VulkanBackend::execute_op`) against the portable
[`CpuBackend`] for identical logical operands, at the tolerance recorded in
`fixtures/parity/ops/manifest.json` (`atol + rtol*|ref|`). `max_abs`/`mean_abs`
are reported in `f32` units.

### 2.1 Fixture shapes (the acceptance gate)

| Case | Shapes | Elements | `max_abs` | `mean_abs` | Tolerance (`atol + rtol|ref|`) |
|---|---|---:|---:|---:|---|
| `matmul.2d_k3` | `[2,3] @ [3,4]` | 8 | 1.136e-7 | 2.165e-8 | 5e-6 + 1e-5 |
| `matmul.gemm_k13_bias` | `[7,13] @ [5,13].T` | 35 | 4.768e-7 | 1.792e-7 | 5e-6 + 1e-5 |
| `matmul.batched_4d` | `[2,3,5,7] @ [2,3,7,11]` | 330 | 9.537e-7 | 1.294e-7 | 1e-5 + 1e-5 |
| `matmul.model_proj` | `[1,13,128] @ [128,8]` | 104 | 2.861e-6 | 7.214e-7 | 2e-5 + 1e-5 |
| `matmul.tail_k65` | `[3,65] @ [65,7]` | 21 | 2.861e-6 | 7.940e-7 | 2e-5 + 1e-5 |

The same results checked against the stored float64-derived payload (the check
`tests/fn3_cpu_reference.rs` uses for the CPU backend), so GPU and CPU are both
inside the published fixture bound:

| Fixture tensor | `max_abs` | `mean_abs` | CPU `max_abs` from the #53 report |
|---|---:|---:|---:|
| `matmul.2d_k3.out` | 7.823e-8 | 9.779e-9 | 1.92e-7 |
| `matmul.gemm_k13_bias.out` (+ bias on host) | 9.537e-7 | 2.401e-7 | 9.54e-7 |
| `matmul.batched_4d.out` | 9.537e-7 | 1.131e-7 | 9.54e-7 |
| `matmul.model_proj.out` | 5.722e-6 | 1.004e-6 | 4.77e-6 |
| `matmul.tail_k65.out` | 5.722e-6 | 1.064e-6 | 3.81e-6 |

The Gemm bias add stays on the host here because Vulkan `Add` with [5]
broadcasting is FN3-13 (#59)'s kernel; the matrix product itself is the GPU's.

### 2.2 Real model shapes

| Case | Shapes | Elements | `max_abs` | `mean_abs` | Tolerance |
|---|---|---:|---:|---:|---|
| Prefill projection | `[13,128] @ [128,384]` | 4,992 | 5.722e-6 | 7.692e-7 | 2.828e-5 + 1e-5 |
| Decode `lm_head` | `[1,1,128] @ [128,50257]` | 50,257 | 2.921e-6 | 3.505e-7 | 1.414e-5 + 1e-4 |
| Attention scores | `[1,2,17,64] @ [1,2,64,17]` | 578 | 1.431e-6 | 2.180e-7 | 2e-5 + 1e-4 |

`atol` follows the fixture policy `2e-5 * sqrt(K/64)` where the CPU parity suite
uses it, and `1e-5 * sqrt(K/64)` for the decode projection.

### 2.3 Tails, degeneracy, batches and strides (GPU vs CPU)

| Case | `max_abs` | `mean_abs` | Tolerance |
|---|---:|---:|---|
| `K = 1, 2, 3, 13, 65, 127, 128, 129, 130` (`M=5, N=7`) | 0 … 5.722e-6 | 0 … 1.836e-6 | `2e-5*sqrt(max(K,1)/64)` + 1e-5 |
| `M=1, K=1, N=1` | 0 | 0 | 2e-5 + 1e-5 |
| `M=1, K=17, N=64` | 9.537e-7 | 2.036e-7 | 2e-5 + 1e-5 |
| `M=64, K=17, N=1` | 2.086e-6 | 2.999e-7 | 2e-5 + 1e-5 |
| `M=1, K=129, N=3` | 3.815e-6 | 1.907e-6 | 2.839e-5 + 1e-5 |
| `M=5, K=1, N=1` | 0 | 0 | 2e-5 + 1e-5 |
| `M=1, K=7, N=1` | 2.384e-7 | 2.384e-7 | 2e-5 + 1e-5 |
| rank-3 `[2,3,4] @ [4,5]` (batch from lhs) | 1.192e-7 | 1.757e-8 | 2e-5 + 1e-5 |
| rank-3 `[2,3,4] @ [2,4,5]` | 1.192e-7 | 1.589e-8 | 2e-5 + 1e-5 |
| rank-3 `[1,3,4] @ [2,4,5]` (lhs broadcast) | 1.192e-7 | 1.850e-8 | 2e-5 + 1e-5 |
| rank-3 `[2,3,4] @ [1,4,5]` (rhs broadcast) | 5.960e-8 | 8.941e-9 | 2e-5 + 1e-5 |
| rank-4 `[2,1,3,4] @ [1,5,4,6]` (two-sided) | 1.192e-7 | 1.461e-8 | 2e-5 + 1e-5 |
| rank-4 `[2,1,3,4] @ [4,6]` | 1.192e-7 | 1.697e-8 | 2e-5 + 1e-5 |
| transposed lhs `[4,9]` over stored `[9,4]` | 1.192e-7 | 2.173e-8 | 2e-5 + 1e-5 |
| transposed rhs `[9,6]` over stored `[6,9]` | 1.192e-7 | 3.811e-8 | 2e-5 + 1e-5 |
| offset window lhs at `+37` elements | 1.192e-7 | 3.345e-8 | 2e-5 + 1e-5 |
| broadcast lhs rows `[4,9]` strides `[0,1]` | 1.192e-7 | 2.670e-8 | 2e-5 + 1e-5 |
| broadcast rhs columns `[9,6]` strides `[1,0]` | 1.192e-7 | 3.353e-8 | 2e-5 + 1e-5 |
| empty product `[2,0] @ [0,3]` | 0 | 0 | exact (+0.0) |

The K sweep covers the 16-deep shared tile boundary from both sides, and the
`M`/`N` sweep covers a single row and a single column.

## 3. Device timings

**Measured on this workstation only:** NVIDIA GeForce GTX 1080 Ti, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, driver 580.196.07, device Vulkan API
1.4.312, compute queue family 2, validation layer active. These are not general
performance claims and say nothing about another GPU or driver.

### 3.1 Per-op end-to-end device time (alpha host-staged tensors)

One submission per op: upload → dispatch → readback. `gpu_ns` is the device
timestamp pair around that submission, so it includes the staging copies — the
alpha `Tensor` owns host storage on every device (#52/#48).

| Shape | Workgroups | Upload bytes | Readback bytes | Device time | Host wall time |
|---|---|---:|---:|---:|---:|
| `[13,128] @ [128,384]` (prefill) | `[24, 1, 1]` | 203,264 | 19,968 | 33,632 ns (33.6 µs) | 131,950 ns |
| `[1,1,128] @ [128,50257]` (decode `lm_head`) | `[3142, 1, 1]` | 25,732,096 | 201,028 | 2,128,736 ns (2.129 ms) | 2,196,152 ns |
| `[1,2,17,64] @ [1,2,64,17]` (attention scores) | `[2, 2, 2]` | 17,408 | 2,312 | 12,640 ns (12.6 µs) | 84,270 ns |

### 3.2 Kernel-only decode timing

Device-resident path (`alloc_buffer`/`upload`, then one `dispatch` per
submission, no host staging), 20 samples after 3 warm-ups; the downloaded
result is checked against the CPU reference (3.815e-6 max_abs, tolerance
1.414e-5 + 1e-4) so the timed dispatch is a correct dispatch:

| Quantity | Value |
|---|---|
| Shape | `[1,128] @ [128,50257]`, 12.87 MFLOP |
| Dispatch geometry | `[3142, 1, 1]`, 16x16 tile |
| Device time, median | **144,448 ns** (min 143,136, max 145,408) |
| Achieved throughput, median | **89.07 GFLOP/s** (min-based 89.89) |

The decode projection is bandwidth-bound here: it streams 25.7 MB of weights
per token, so ~178 GB/s effective is close to what a 16x16 tile can sustain on
this part. A wider tile or a column-major weight layout is a performance
question for #71/#72, not a correctness one.

## 4. Device-execution evidence

For `[13,128] @ [128,384]`, the residency record asserts and reports:

```text
[fn3vk][evidence] workgroups=[24, 1, 1] submissions=1 bindings=3 upload_bytes=203264
readback_bytes=19968 device_local_bytes=223232 gpu_ns=Some(32864)
gpu_ticks=Some((1795006632226143232, 1795006632226176096)) host_ns=107580
```

`proves_device_execution()` is true (one submission, all workgroup axes
non-zero, three device-local bindings, no input read back); `inputs_read_back`
is false and `ResidencySummary::input_readbacks` is 0. The test also asserts
`record.kernel == "matmul.tiled.f32"`, `record.op == MatMul`,
`record.backend == "vulkan"`, and the exact upload/readback byte counts.

## 5. Validation

Validation is enabled in code (`VulkanConfig::validated()`); `VK_INSTANCE_LAYERS`
was **not** set, so the zero-delta guard counts exactly one layer instance.
Every `[fn3vk][validation][...]` line that is not a loader `INFO` message:

```text
[fn3vk][validation] batched broadcast sweep: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] decode timing: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] degenerate M/N sweep: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] empty contraction: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] evidence [13,128]@[128,384]: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] fixture cases: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] K boundary sweep: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] real model shapes: errors=0 warnings=0 (opened with errors=0 warnings=0)
[fn3vk][validation] strided/broadcast operands: errors=0 warnings=0 (opened with errors=0 warnings=0)
```

**0 errors and 0 warnings**, with a zero delta from `open` in every test. The
run also contains 1,520 loader `INFO` lines (layer/ICD enumeration, device
selection); none is a validation message.

## 6. Commands and outcomes

All runs with `CARGO_BUILD_JOBS=4` on the workstation above.

| Command | Outcome |
|---|---|
| `cargo fmt --check` | **PASS** (exit 0) |
| `cargo build --no-default-features --features vulkan` | **PASS**; only the two pre-existing warnings (2.x `augment_runtime_telemetry` is never used, unused `fnv` dependency) |
| `cargo test --features vulkan --test fn3_vulkan_matmul -- --nocapture --test-threads=1` | **PASS**: `17 passed; 0 failed` (9 #60 tests + 8 fixture-loader tests in the same binary), 1.65 s |
| `cargo test --no-default-features --features vulkan --lib fn3::backend::vulkan` | **PASS**: `30 passed; 0 failed` (includes 4 new `kernels::matmul` unit tests) |
| `cargo test --features vulkan --test fn3_vulkan_smoke -- --test-threads=1` | **13 passed; 1 failed** — see §7 |
| `spirv-val --target-env vulkan1.1` on each committed `.spv` | **PASS** for `matmul_tiled.comp.spv` and `smoke_add.comp.spv` (SPIRV-Tools v2026.4) |
| `src/fn3/backend/vulkan/shaders/compile.sh` (glslc 2026.4) | **PASS**: both shaders recompile **byte-identically**; `git status` shows no change to the tracked `smoke_add.comp.spv`, so the new `.spv`/`.spvasm` are the compiler's committed output |

The physical device was pinned with `FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e`;
selection is by UUID, never by index, because the loader also enumerates
lavapipe at index 1.

## 7. One shared file must change: `tests/fn3_vulkan_smoke.rs`

`tests/fn3_vulkan_smoke.rs` is owned by #57 and must not be edited by this
change, but it contains a "not implemented yet" assertion that #60 necessarily
invalidates. The failing test is
`unsupported_ops_and_shapes_are_reported_not_approximated`:

```text
thread 'unsupported_ops_and_shapes_are_reported_not_approximated' panicked at
tests/fn3_vulkan_smoke.rs:614:9:
matmul must be unsupported
```

Minimal required edit (no behavioural change to #57's own coverage):

1. Remove `(OpId::MatMul, "#60"),` from the `for (op, owner)` loop at line 609.
2. Replace the block at lines 618-635 — which expects
   `execute_op(OpId::MatMul, &[], &mut [], ..)` to return `UnsupportedOp` —
   with an arity check, since MatMul is now registered:

   ```rust
   let error = backend
       .execute_op(OpId::MatMul, &[], &mut [], &OpParams::default())
       .expect_err("matmul needs two inputs");
   assert!(
       matches!(error, Fn3Error::ArityMismatch { op: OpId::MatMul, .. }),
       "{error:?}"
   );
   ```

After that edit the suite is 14/14 — verified experimentally by applying exactly
that edit in the working tree and reverting it afterwards (`14 passed; 0
failed`). The committed change does not touch the file. The other smoke
assertion that broke while this change was in progress —
`capabilities_match_the_physical_device` requiring
`!features.broadcasting` — was fixed **without** touching the file by not
claiming the elementwise-broadcasting capability flag, which is #59's to set.
Everything else in the smoke suite (13 tests) passes unchanged.

## 8. Skipped checks

| Check | Why |
|---|---|
| lavapipe / software-ICD lane for this kernel | not run: this issue's evidence must be physical-device evidence, and the software lane cannot satisfy it. The kernel's only device-specific assumption is `maxPushConstantsSize >= 112` (guaranteed by the spec) and a 2,176-byte shared tile, both checked against the device limits at registration. |
| #71/#72 performance variants (wider tiles, vectorised loads, fence ring) | out of scope; see §9. |
| End-to-end pinned-model run | FN3-16 (#62). |
| CI `gpu-physical` lane record | not run here; the local commands in §6 are quoted verbatim and name the device, UUID and driver. |
| `cargo test --features vulkan` (whole workspace) | not run: the 2.x test tree is slow and unrelated; the required gates are listed in §6 and the frozen-tree `cargo test --lib` is unchanged by this diff. |

## 9. Known risks and remaining work

- **Output must be contiguous.** The kernel writes one contiguous row-major
  window and rejects a strided output with `UnsupportedOp`. Plan-allocated
  outputs always are; a future device-resident `Storage` (#52/#48) that hands
  the backend a strided output would need either a copy in the backend or a
  storage-offset write path.
- **32-bit element addressing.** Windows and strides are `u32`; the check
  cannot trigger for a legal buffer on this device (`maxStorageBufferRange` is
  4,294,967,295 bytes = under `u32::MAX` elements) but it is a real limit on a
  device with a larger storage range, and it fails loudly rather than wrapping.
- **FMA contraction.** The shader's `acc + a*b` is contracted by the driver,
  so results differ from the CPU's separate multiply/add by up to ~1 ulp per
  term. Measured differences are one to two orders of magnitude inside every
  published tolerance; this is not a re-specification of the accumulation order
  (still one `f32` accumulator over ascending `k`).
- **The `[1,128] @ [128,50257]` op is host-staged in the model path** until
  device-resident tensor storage lands: 25.7 MB is uploaded per token, which is
  why the end-to-end device time (2.13 ms) is 15x the kernel-only time
  (144 µs). That is the alpha tensor-storage limitation recorded by #57, not a
  matmul regression.
- **Performance is a baseline, not a target.** 89 GFLOP/s and ~178 GB/s
  effective weight streaming on a GTX 1080 Ti. #71 (CPU microkernels) and #72
  (kernel variants, fence ring, persistent device buffers) own the next step:
  a 32x32 or 64x64 tile with 2-4 outputs per thread would cut the weight traffic
  per output, and a column-major weight layout would let a single workgroup read
  coalesced columns. Any such variant must stay behind `OpId::MatMul` and prove
  agreement with this path before replacing it.
- **Quantized specialization** is coordinated with #24 and out of the alpha;
  the kernel registers `f32` only and rejects every other dtype.
