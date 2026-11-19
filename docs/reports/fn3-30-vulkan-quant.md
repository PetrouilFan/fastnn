# FN3-24 (#70): W8A16 quantized inference on the Vulkan resident path

**Issue:** [#70](https://github.com/PetrouilFan/fastnn/issues/70), backlog FN3-24,
workstream GPU quantization, milestone stable.
**Branch:** `fn3/70-quant-vulkan`.
**Worktree:** `/home/petrouil/Projects/github/fastnn-wt-70-quant-vulkan`.
**Base:** `6389186d4599471c4cb3ebc1156a5c1b872cd423` (`fn3(quant): the trained
fixture and W4A8 evidence are integrated`, audited integration HEAD).
**Format:** `fn3-quant-format-v1`, locked by #68 — signed INT8 weights in groups
of 64 along K, a binary16 scale and an int16 zero-point per `(row, group)`, FP32
activations, no nibble interleaving, no Q/DQ pair.
**Resident contract:** #72 (`docs/reports/fn3-26-vulkan-residency.md` §10).

No remote writes were made: no push, tag, pull request, merge, release,
publication, issue comment or close. Model weights are not committed.

---

## 0. Commits

```text
<HEAD> fn3(vulkan): W8A16 coverage, accuracy and timing evidence (#70)
f57678c fn3(vulkan): fused W8A16 grouped projection kernel
6389186 base: fn3(quant): the trained fixture and W4A8 evidence are integrated
```

Every commit carries `Refs #70, #68, #72`.

---

## 1. What was built

One fused projection kernel and its family registration. Nothing outside the
issue's ownership changed: two additive lines in
`src/fn3/backend/vulkan/kernels/mod.rs` (`pub mod quant;` and a
`KernelSet::quantized()` facade), three new files (the kernel module, the GLSL
source and its committed SPIR-V), one new integration test, and this report.

| File | Status | Purpose |
|---|---|---|
| `src/fn3/backend/vulkan/kernels/quant.rs` | new, 1,3xx lines | kernel declaration, admission rule, registration table, packer, packed device payload, launch |
| `src/fn3/backend/vulkan/shaders/quant_matmul.comp` | new | the fused grouped-INT8 projection |
| `src/fn3/backend/vulkan/shaders/quant_matmul.comp.spv` | new | committed SPIR-V (`spirv-val --target-env vulkan1.1` clean) |
| `src/fn3/backend/vulkan/shaders/quant_matmul.comp.spvasm` | new | committed disassembly, the review artifact |
| `src/fn3/backend/vulkan/kernels/mod.rs` | +17 lines | `pub mod quant;` and the `quantized()` registration seam |
| `tests/fn3_vulkan_quant.rs` | new | kernel parity, selection, resident proof, end-to-end accuracy, measurement |
| `docs/reports/fn3-30-vulkan-quant.md`, `docs/reports/fn3-30/` | new | this report and the raw data |

### 1.1 The kernel

```text
y[m, n] = sum_g ( sum_{k in g} x[m, k] * (code[n, k] - zero[n, g]) ) * scale[n, g]
```

* **One FP32 accumulator per 64-element group**, ascending `k`; the binary16
  scale is applied **once per group** with one FP32 multiply, then the group is
  added to the row's accumulator. `code - zero` stays integer for the whole dot,
  exactly as `docs/quantization/fn3-quant-format-v1.md` §5 requires.
* **No dequantized weight buffer.** The code byte is unpacked from a `uint`
  word and multiplied against the FP32 activation inside the group loop, so the
  only floating-point work besides the dot is 2 metadata conversions and one
  multiply per group.
* **No half-precision type and no 8/16-bit storage extension.** Codes are bound
  as `uint` (four codes per word, which the format's 16-byte row alignment
  guarantees), scales and zero-points as `uint` (two values per word). The half
  is widened to `f32` by integer bit assembly (`f16_to_f32`), including the
  subnormal case, so the kernel needs nothing beyond Vulkan 1.1 core — the GTX
  1080 Ti's feature set is not a constraint and no `VkPhysicalDeviceFeatures`
  flag had to be enabled. That matters because device creation is platform code
  (#73's file, and `device.rs` before it): the issue needed **no** platform
  change to run.
* **Geometry.** 64 invocations per workgroup; `x` is the 64-column output tile
  (the batch-1 decode's output-channel split: `ceil(50257/64) = 786` workgroups
  for `lm_head`), `y` is a 16-row activation tile (prefill `M = 13` is one
  tile). Each invocation owns one output column and walks that weight row, so
  every packed word is read once per workgroup and reused across the 16 rows;
  the activation reads are one address for all 64 invocations (broadcast from
  L1). 28 push-constant bytes, five storage bindings, 4 KB of registers, no
  shared memory, no `barrier()`.
* **Tails.** `K` need not be a multiple of 64: the final group's `words` is
  `ceil(k_len/4)`, the payload's row padding is zero by the format's rules, and
  `x` beyond `K` is a select of a clamped (never out-of-bounds) activation
  address. `K = 8`, `K = 70` and `K = 65` are tested.
* **Numerics.** The only allowed difference from the scalar reference is
  fusing a group's multiply-add into an FMA, i.e. at most one rounding per term.
  Measured against the reference: **≤ 2.4e-7 maximum absolute** and **≤ 9e-7
  relative Frobenius** across every fixture and every pinned shape.

### 1.2 The packed payload is device-resident from prepare

`PackedProjection::prepare` allocates three device-local buffers and moves each
section once, in the prepare phase. The launch binds those `Arc<Buffer>`
handles; there is no host copy of the codes, scales or zero-points anywhere in
the launch path, so **a per-op weight upload is unrepresentable**, not merely
unused. This is the #72 §10 contract implemented without asking the platform for
anything new:

| #72 surface | How #70 uses it |
|---|---|
| `VulkanBackend::alloc_buffer` / `upload` | the three prepare transfers |
| `check_staged_input(None, bytes)` | attributes the prepare bytes to `staged_upload_bytes[Prepare]` so the phase counters prove the "zero weight bytes after prepare" claim |
| `Job` / `begin_step` / `end_step` | one dispatch inside the step's single submission; the kernel records its own compute→compute barriers over all five bindings |
| `ExecutionRecord` | one record per quantized op, with five binding snapshots and zero upload/readback bytes |

No counter had to be added. One *gap* is reported in §8: the packed binds are
not counted by `ResidencyStats::persistent_hits` (they are not `Tensor`s in the
persistent table), so the proof reads the ledger's per-record bindings instead.

### 1.3 Why the family is not in `KernelSet`

`KernelSet` holds **one kernel per `OpId`**, and the alpha has no quantized op
id: `MatMul` is the FP32 tiled product, so a second `MatMul` registration would
*replace* it rather than sit beside it. A quantized projection is selected by
the packed weight a node binds, not by its op. The family therefore registers in
`QuantKernelSet`, whose semantics mirror `KernelSet` exactly:

| `KernelSet` | `QuantKernelSet` |
|---|---|
| `register` replaces by `OpId` | `register` replaces by `QuantFamily` |
| `supports(op)` is the op-level verdict | `supports(family)` is the family-level verdict |
| `admits(op, shapes, dtypes, params)` refines a `Partial` with a rule and can only narrow it | `admits(family, &ProjectionRequest)` refines a `Partial` with a rule and can only narrow it |
| `Full` / `Unsupported` ignore the rule | same |
| `alpha()` is the registration point | `alpha()` is the registration point; `KernelSet::quantized()` exposes it |

The registration is **`Support::Partial`**, and `supports(OpId::MatMul)` is
untouched: it keeps meaning "the FP32 kernel", never "everything including
W8A16". A unit test (`the_family_sits_beside_the_fp32_matmul_registration`)
asserts exactly that, and `admit_projection` implements the format's fallback
list as the concrete rule:

| Condition (format §7) | Verdict |
|---|---|
| activation not F32 | `Partial`: v1 never quantizes activations |
| weight not F32 | `Partial` |
| right-hand side computed at run time | `Partial`: a dynamic RHS keeps FP32 |
| shared weight, packed copy not permitted | `Partial` |
| weight rank ≠ 2 | `Partial` |
| `K < 64` | `Partial`: metadata would dominate |
| `N * K < 4096` | `Partial` |
| activation rank < 2, batch axis > 1, or `K` mismatch | `Partial` |
| everything else | `Full` |

`launch_projection` validates its own hard preconditions (contiguous windows,
matching extents, same device, 32-bit index space, dispatch limits) and returns
`UnsupportedOp` naming `FN3-24 (#70)` for anything outside the kernel's shape
contract. The policy thresholds are a *selection* decision; the kernel itself is
exact for `K = 8` and `N = 1`, which is why the committed format fixtures can be
run through it.

### 1.4 The integration seam this issue could not make

The plan cannot yet *select* the kernel, and that is a core-contract change this
issue does not own (see §8). `tests/fn3_vulkan_quant.rs` supplies the adapter a
future planner change will replace: a `Backend` decorator that intercepts
`MatMul` when its right-hand side resolves to a prepared packed weight and
delegates every other instruction to the real Vulkan backend. The GPU work is
the production kernel; the test computes nothing except the independent
reference used to grade it.

---

## 2. Coverage

The pinned graph's `MatMul` inventory, classified by `admit_projection` from the
plan's own slots (`SlotRole::Weight` vs `InstructionOutput`). Regenerate with
`cargo test --features vulkan --test fn3_vulkan_quant coverage_of_the_pinned_graphs -- --nocapture`.

### 2.1 Tiny GPT-2 (hidden 128, 2 layers, vocab 50,257)

| # | MatMul | RHS | Decision |
|---|---|---|---|
| 1 | `[13,128] x [128,384]` `h.0.attn.c_attn.weight` | weight | **quantized** |
| 2 | `[1,2,13,64] x [1,2,64,13]` attention scores | `mul` output | fallback: dynamic RHS |
| 3 | `[1,2,13,13] x [1,2,13,64]` attention values | `concat` output | fallback: dynamic RHS |
| 4 | `[13,128] x [128,128]` `h.0.attn.c_proj.weight` | weight | **quantized** |
| 5 | `[13,128] x [128,512]` `h.0.mlp.c_fc.weight` | weight | **quantized** |
| 6 | `[13,512] x [512,128]` `h.0.mlp.c_proj.weight` | weight | **quantized** |
| 7-12 | the same six for `h.1` | | 4 quantized, 2 fallback |
| 13 | `[1,13,128] x [128,50257]` `lm_head` | `transpose` output | fallback: dynamic RHS |

Decode `(1,13)` is the same table with `M = 1`. Totals per step: **8 quantized,
5 fallback** (4 attention products + the tied head). Running `Packed` head
routing makes it 9/4.

**`lm_head` and the embeddings.** The embedding table (`Gather`) stays FP32 —
v1 is a weight format for projections and the README's coverage table says so.
`lm_head` is the *same* `wte` matrix read through a `Transpose` instruction, so
the plan hands `MatMul` an instruction output, not a weight: the format's own
rule ("a weight that is not a 2-D F32 constant at prepare time falls back")
declines it. The packed copy *does* exist and is correct (`[vocab, hidden]` is
the table's own layout, no transpose needed); what is missing is the plan-level
fold that hoists a transpose of a constant out of the step. That is #72 §9's
"caching the transposed embedding matrix", plan work, not backend work. This
report measures both arms so the opportunity is sized rather than asserted.

### 2.2 Trained GPT-2 (hidden 768, 12 layers, vocab 50,257)

PENDING-TRAINED-COVERAGE

### 2.3 Storage

PENDING-STORAGE

---

## 3. Resident proof

`resident_quantized_projections_upload_nothing`, physical GTX 1080 Ti, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, `FN3_VK_REQUIRE_PHYSICAL=1`,
`VulkanConfig::validated()`, `VK_INSTANCE_LAYERS` unset.
PENDING-RESIDENT

---

## 4. Parity and accuracy

PENDING-ACCURACY

---

## 5. Measurement

PENDING-PERF

---

## 6. Verification commands and their real outcomes

PENDING-COMMANDS

---

## 7. Skipped checks

PENDING-SKIPS

---

## 8. Platform changes this issue needed but did not make

PENDING-PLATFORM

---

## 9. Open decisions

PENDING-DECISIONS
