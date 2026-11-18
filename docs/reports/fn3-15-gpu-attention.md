# FN3-15 — Vulkan normalization, reductions and attention

**Status:** implemented on `fn3/61-gpu-attention`, physical-GPU evidence captured.
**Issue:** [#61](https://github.com/PetrouilFan/fastnn/issues/61), tracking #46.
**Branch point / base SHA:** `27fc60d9d4b42a3b0e1e20e883daecb5cfa93871`
(`git rev-parse HEAD` before the first edit; the work started there).
**Rebased onto:** `rewrite/fastnn-3` at `c6af6cd` before handoff, which contains
the #60 merge (`2835c32`, FP32 `MatMul` registered `Full`) and the smoke-test
invariant rewrite (`4d188e8`). The rebase was conflict-free; no file outside
this issue's ownership was touched.
**Branch HEAD:** see `git log -1 --format=%H` on `fn3/61-gpu-attention`.
**Remote writes:** none. Local commits only, message trailer `Refs #61, #46`.

This report owns the FN3-15 kernel family: `LayerNorm`, `Softmax`,
`ReduceMean`, `ReduceSum`, `ArgMax`, and the reserved fused `MaskedAttention`
(id 40). The numerical oracle is the merged CPU reference (FN3-07 / #53); the
platform layer is FN3-11 / #57 and was not modified.

## 1. What shipped

| Op | Kernel | Asset | Arithmetic contract |
|---|---|---|---|
| `LayerNorm` | `norm.layernorm.f32` | `shaders/fn3_layernorm.comp` | mean, **biased** variance, both ascending in one `f32` accumulator; `(x-mean)/sqrt(var+eps)*w+b`; eps from `OpParams` |
| `Softmax` | `norm.softmax.last.f32` | `shaders/fn3_softmax_last.comp` | trailing axis; max subtraction before `exp`; ascending sum; division per element |
| `ReduceMean` / `ReduceSum` | `reduce.sum_mean.f32` | `shaders/fn3_reduce_sum_mean.comp` | any axis set, either `keep_dims`; ascending reduced-coordinate scan into one `f32` accumulator; mean divides by the reduced count |
| `ArgMax` | `reduce.argmax.f32_i32` | `shaders/fn3_argmax.comp` | one axis, either `keep_dims`; first occurrence wins; `I64` output |
| `MaskedAttention` | `attention.fused.f32` | `shaders/fn3_attention_fused.comp` | `MatMul(q,kᵀ) → Div(√D) → Add(mask) → Softmax → MatMul(probs,v)` in one dispatch; GQA head grouping; finite `-FLT_MAX` masks; optional causal flag; optional `probs` output |

One invocation owns one row (norm/softmax) or one output element (reductions);
one workgroup owns one `(batch, head, query)` row for attention. This is a
deliberate choice: the summation order is the reference order rather than a
tree order, which makes `LayerNorm`, `Softmax`, `ReduceSum` and the attention
softmax bit-comparable with the CPU reference at the shapes the pinned graph
produces. The cost is intra-row parallelism, not correctness.

All five kernels go through one launch helper (`kernels/norm.rs::execute_staged`):
allocate through the suballocator, lease staging, record
`upload → barrier → dispatch → barrier → readback` as **one** submission, wait
once, read back, append one `ExecutionRecord`. The synchronous
`Submission::wait` model is unchanged and no dependency was added.

### `MaskedAttention` contract (id 40)

Inputs (3 or 4) and outputs (1 or 2):

```text
q     [batch, heads,    queries, head_dim]  f32, contiguous
k, v  [batch, kv_heads, keys,    head_dim]  f32, contiguous   (heads % kv_heads == 0)
mask  [batch|1, heads|1, queries, keys]     f32 additive, optional
out   [batch, heads,    queries, head_dim]
probs [batch, heads,    queries, keys]      optional second output
```

Query head `h` reads KV head `h / (heads / kv_heads)` (grouped-query; the
identity when equal). `OpParams` is the alpha attribute carrier and has exactly
one boolean, so the fusion's `causal` attribute rides in `keep_dims`; `axes`
must be empty and `epsilon` must be 0 (the scale is fixed at
`1/sqrt(head_dim)`, applied as a **division**, as the graph does). `keep_dims`
never changes the output shape. The causal rule replaces the effective mask with
`-FLT_MAX` above the diagonal (shifted by `keys - queries` for cached decode)
instead of adding a second mask term: adding two `-FLT_MAX` values saturates to
`-inf`, and `-inf - (-inf)` would make max subtraction produce NaN.

Masked-row semantics: a query row whose effective mask is entirely the finite
`-FLT_MAX` is **exactly uniform** `1/keys` and finite. That is the pinned GPT-2
convention and a conformance requirement, not a tolerance.

Without a second output the probabilities never leave shared memory: the fused
path materialises no `S × S_kv` tensor. When requested, `probs` is written from
the shared row.

## 2. Changed paths

| Path | Lines | Note |
|---|---:|---|
| `src/fn3/backend/vulkan/kernels/norm.rs` | 798 | `LayerNorm` + `Softmax` + shared launch plumbing |
| `src/fn3/backend/vulkan/kernels/reduce.rs` | 589 | `ReduceMean` / `ReduceSum` / `ArgMax` |
| `src/fn3/backend/vulkan/kernels/attention.rs` | 465 | fused `MaskedAttention` |
| `tests/fn3_vulkan_attention.rs` | 1872 | new integration suite (12 tests + the reused fixture-module tests) |
| `shaders/fn3_layernorm.comp` (+ `.spv` 2596 B, `.spvasm`) | 72 | new |
| `shaders/fn3_softmax_last.comp` (+ `.spv` 2024 B, `.spvasm`) | 64 | new |
| `shaders/fn3_reduce_sum_mean.comp` (+ `.spv` 3476 B, `.spvasm`) | 84 | new |
| `shaders/fn3_argmax.comp` (+ `.spv` 3328 B, `.spvasm`) | 95 | new |
| `shaders/fn3_attention_fused.comp` (+ `.spv` 5948 B, `.spvasm`) | 158 | new |
| `docs/reports/fn3-15-gpu-attention.md` | — | this report |

No shared file was edited. `kernels/mod.rs`, `smoke.rs`, the platform layer,
`Cargo.toml`/`Cargo.lock`, fixtures, tools, CI and roadmap docs are untouched.
GLSL and SPIR-V are committed together; `shaders/compile.sh` already globs
`*.comp`, so the new sources are picked up without editing it.

## 3. Registered `Support` verdicts

| Op | Verdict | Registered reason (abridged) |
|---|---|---|
| `LayerNorm` | **`Full`** | The fn3 LayerNorm contract is "normalize the trailing axis with rank-1 weight/bias"; the kernel implements it in full. Multi-axis normalization and strided views are rejected by name. |
| `Softmax` | **`Partial`** | `f32` trailing-axis softmax of a contiguous operand. The CPU reference also accepts an arbitrary axis; that case is rejected by name. |
| `ReduceMean` | **`Full`** | Any axis set, either `keep_dims`, contiguous `f32` — the whole fn3 reduction contract. |
| `ReduceSum` | **`Full`** | Same. |
| `ArgMax` | **`Partial`** | `f32`/`i32` inputs, one axis, either `keep_dims`, `I64` output. `i64` inputs need `shaderInt64`, which the FN3-11 (#57) device does not enable, so they are rejected by name. |
| `MaskedAttention` | **`Full`** | The complete fused contract documented in §1: rank-4 contiguous `f32`, `heads % kv_heads == 0`, additive `[B\|1,H\|1,S,S_kv]` mask, optional causal attribute, `keys <= 4096`, optional `probs`. Requests outside it are rejected by name. |

Registering `MaskedAttention` also advertises
`Capabilities::features.masked_attention = true` (the platform layer derives the
flag from a `Full` registration); the other flags stay false. `Full` for the
fused id is the same reading the CPU reference uses for its own ops: `Full`
means "the whole documented surface of this kernel", and the bound
(`keys <= 4096`, driven by the 16 KiB shared-memory score row) is part of the
documented contract and is enforced loudly, never approximated. `Softmax` and
`ArgMax` stay `Partial` because their contract is shared with the CPU reference
and is wider than the Vulkan subset.

Every rejection is `Fn3Error::UnsupportedOp { backend: "vulkan", reason }` with
`(owner: FN3-15 / #61)` in the text. Examples captured from the run:

```text
[fn3vk][reject] op=softmax backend=vulkan reason=this kernel implements the trailing axis only (the fn3 softmax contract and the pinned graph); axis 1 of [2, 3, 4] was requested (owner: FN3-15 / #61)
[fn3vk][reject] op=masked_attention backend=vulkan reason=4097 keys exceed the 4096-key shared-memory score row of this kernel; split the sequence (owner: FN3-15 / #61)
[fn3vk][reject] op=masked_attention backend=vulkan reason=grouped-query attention requires n_kv_heads to divide n_heads; q declares 3 head(s) and k/v declare 2 (owner: FN3-15 / #61)
[fn3vk][reject] op=softmax backend=vulkan reason=softmax input is a strided view (strides[1, 3]), and the FN3-15 kernels require contiguous operands; materialise it with `to_contiguous` first (owner: FN3-15 / #61)
[fn3vk][reject] op=argmax backend=vulkan reason=this kernel compares f32 and i32 payloads; a i64 input would need shaderInt64 (or a wide comparison), which FN3-11 (#57) does not enable (owner: FN3-15 / #61)
```

## 4. Numerical results

Device: **NVIDIA GeForce GTX 1080 Ti**, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, driver 580.196.07, device Vulkan
1.4.312, compute queue family 2, `maxComputeSharedMemorySize 49152`,
`maxMemoryAllocationSize 4292870144`. Selection is by UUID, never by index (the
loader also enumerates lavapipe).

### 4.1 Committed `ops` fixtures

Tolerances are the fixture's own `provenance.tolerance` rows; the comparison
uses the fixture predicates (`check`, `check_summary`), not a hand-picked bound.

| Fixture | Shape | max_abs | mean_abs | atol | rtol |
|---|---|---:|---:|---:|---:|
| `layernorm.out` | `[2,3,5]` | **2.384e-7** | 3.617e-8 | 2e-5 | 1e-5 |
| `layernorm.model_shape.out` | `[1,4,128]` | **2.384e-7** | 4.769e-8 | 2e-5 | 1e-5 |
| `softmax.unmasked.out` | `[2,3,4,5]` | **5.960e-8** | 1.149e-8 | 2e-6 | 1e-6 |
| `softmax.causal.out` | `[1,2,4,4]` | **5.960e-8** | 4.626e-9 | 2e-6 | 1e-6 |
| `softmax.padded.out` | `[1,1,3,5]` | **5.960e-8** | 5.076e-9 | 2e-6 | 1e-6 |
| `softmax.all_masked.out` | `[1,1,2,4]` | **7.451e-9** | 1.164e-9 | 2e-6 | 1e-6 |
| `attention.probs` (fused) | `[2,2,4,4]` | **5.960e-8** | 4.657e-9 | 3e-6 | 1e-6 |
| `attention.out` (fused) | `[2,2,4,5]` | **5.960e-8** | 9.034e-9 | 3e-6 | 1e-6 |

The masked softmax cases are the graph's own form: the additive mask is added
with the FN3-11 smoke `Add` kernel and then `Softmax` runs, exactly as
`Add → Softmax` in the graph.

### 4.2 Comparison with `CpuBackend`

| Case | Elements | max_abs | mean_abs | Bound |
|---|---:|---:|---:|---|
| LayerNorm `[2,3,5]` | 30 | 2.384e-7 | 2.497e-8 | 2e-5 + 1e-5·\|ref\| |
| LayerNorm `[1,4,128]` | 512 | 2.384e-7 | 4.012e-8 | 2e-5 + 1e-5·\|ref\| |
| LayerNorm `[1,13,128]` (real prefill) | 1664 | 2.384e-7 | 3.434e-8 | 2e-5 + 1e-5·\|ref\| |
| Softmax unmasked (fixture input) | 120 | 5.960e-8 | 3.528e-9 | 2e-6 + 1e-6·\|ref\| |
| Softmax causal (Add + Softmax) | 32 | 5.960e-8 | 2.991e-9 | 2e-6 + 1e-6·\|ref\| |
| Softmax padded (Add + Softmax) | 15 | 5.960e-8 | 5.774e-9 | 2e-6 + 1e-6·\|ref\| |
| Softmax all-masked (Add + Softmax) | 8 | 2.980e-8 | 3.958e-9 | 2e-6 + 1e-6·\|ref\| |
| Attention fixture probs | 64 | 5.960e-8 | 3.260e-9 | 3e-6 + 1e-6·\|ref\| |
| Attention fixture out | 80 | 8.941e-8 | 8.545e-9 | 3e-6 + 1e-6·\|ref\| |
| Attention `S=17`, `head_dim=64` probs | 578 | 5.960e-8 | 2.981e-9 | 3.092e-6 + 1e-6·\|ref\| |
| Attention `S=17`, `head_dim=64` out | 2176 | 1.192e-7 | 1.469e-8 | 3.092e-6 + 1e-6·\|ref\| |
| GQA (4 query / 2 KV heads) probs | 200 | 5.960e-8 | 9.653e-9 | 3e-6 + 1e-6·\|ref\| |
| GQA (4 query / 2 KV heads) out | 320 | 1.788e-7 | 2.647e-8 | 3e-6 + 1e-6·\|ref\| |
| Causal `S=33` (GQA 2/1, no mask input) probs | 2178 | 5.960e-8 | 1.722e-9 | 4.308e-6 + 1e-6·\|ref\| |
| Causal `S=33` (GQA 2/1, no mask input) out | 1056 | 1.192e-7 | 1.476e-8 | 4.308e-6 + 1e-6·\|ref\| |
| Cached decode `keys=5` out | 128 | 8.941e-8 | 1.858e-8 | 3e-6 + 1e-6·\|ref\| |
| Cached decode `keys=17` out | 128 | 5.960e-8 | 9.361e-9 | 3.092e-6 + 1e-6·\|ref\| |
| Cached decode `keys=33` out | 128 | 2.980e-8 | 1.111e-8 | 4.308e-6 + 1e-6·\|ref\| |
| `ReduceSum` axes `[-1]` | 6 | **0.0** | 0.0 | exact |
| `ReduceSum` axes `[1,2]` keep_dims | 2 | **0.0** | 0.0 | exact |
| `ReduceMean` axes `[-1]` | 6 | 1.490e-8 | 2.794e-9 | 1e-6 + 1e-6·\|ref\| |
| `ReduceMean` axes `[1]` keep_dims | 10 | 1.192e-7 | 2.421e-8 | 1e-6 + 1e-6·\|ref\| |
| `ReduceMean` all axes | 1 | 7.451e-9 | 7.451e-9 | 1e-6 + 1e-6·\|ref\| |
| `ArgMax` f32 (ties) / i32 | — | **0 (indices equal)** | — | exact |

Sequence scaling for `S > 16` is the fixture policy `atol *= sqrt(S/16)`, rtol
unchanged (applied to the `S=17` and `S=33` bounds above).

`ReduceSum` is bit-exact: the accumulation order is the reference order.
`ReduceMean` differs by at most one or two ULP, and **only** because Vulkan
requires 2.5 ULP (not IEEE) accuracy for `OpFDiv`; the GPU lowers the mean's
`sum / count` to a reciprocal multiply. The sum itself is identical — proven by
the exact `ReduceSum` rows on the same data. The published bound is 1e-6.

### 4.3 All-masked rows (the conformance requirement)

Observed, exact and finite element by element:

```text
[fn3vk][contract] softmax all-masked row = [0.25, 0.25, 0.25, 0.25] (exact, finite)
[fn3vk][contract] attention all-masked probs row (batch 1, head 0) = [0.25, 0.25, 0.25, 0.25]
[fn3vk][contract] attention all-masked probs row (batch 1, head 1) = [0.25, 0.25, 0.25, 0.25]
```

The assertions are `assert_eq!` per element plus `is_finite()` on every element
of the whole output tensor. A NaN or a non-uniform row fails.

## 5. Device evidence, allocation sizes and timings

Every op records one residency record with `backend = "vulkan"`, one submission,
the dispatch geometry, every binding, the transfer sizes and device timestamps.
The suite asserts `proves_device_execution()` and `inputs_read_back == false`
on every record, and `input_readbacks == 0` on the summary, in **every** test
that executes an op. The two tests that only check verdicts or rejections
(`unsupported_requests_are_rejected_by_name`, `verdicts_are_three_valued_and_honest`)
have no execution to prove: the first instead asserts that a rejected request
leaves the residency ledger empty (`executions == 0`), which is the stronger
statement for that case. Representative records:

```text
[fn3vk][evidence] op=layernorm        kernel=norm.layernorm.f32     submissions=1 workgroups=[1, 1, 1]   bindings=4 upload_bytes=3072   readback_bytes=2048  device_local_bytes=5120   host_ns=61680  gpu_ns=Some(21760)
[fn3vk][evidence] op=softmax          kernel=norm.softmax.last.f32  submissions=1 workgroups=[1, 1, 1]   bindings=2 upload_bytes=480    readback_bytes=480   device_local_bytes=960    host_ns=77041  gpu_ns=Some(8992)
[fn3vk][evidence] op=reduce_mean      kernel=reduce.sum_mean.f32    submissions=1 workgroups=[1, 1, 1]   bindings=3 upload_bytes=236    readback_bytes=24    device_local_bytes=260    host_ns=49460  gpu_ns=Some(12448)
[fn3vk][evidence] op=argmax           kernel=reduce.argmax.f32_i32  submissions=1 workgroups=[1, 1, 1]   bindings=3 upload_bytes=112    readback_bytes=16    device_local_bytes=128    host_ns=43860  gpu_ns=Some(8352)
[fn3vk][evidence] op=masked_attention kernel=attention.fused.f32    submissions=1 workgroups=[17, 2, 1] bindings=7 upload_bytes=27316  readback_bytes=11016 device_local_bytes=38332  host_ns=91830  gpu_ns=Some(16864)
```

`device_local_bytes` is the sum of the op's device buffers, i.e. the
intermediate allocation footprint (`q`, `k`, `v`, mask, parameter block and
outputs; a suppressed `probs` output is a 4-byte scratch instead of an
`S × S_kv` tensor). `backend.memory_stats().live_device_local_bytes == 0` is
asserted after each test: every op-scoped buffer is returned to the allocator.

Real-shape timings, 20 runs per shape per run, three runs on this workstation
(the values are the min/median range across those runs). One sample is a whole
op round trip — **one submission = upload + dispatch + readback** — because the
alpha `Tensor` is host-staged (`host_staged = true`):

| Shape | device `gpu_ns` min / median | `host_ns` min / median |
|---|---:|---:|
| LayerNorm `[1,13,128]` | 23.87–24.03 µs / 24.06–24.16 µs | 59.05–72.57 µs / 60.00–76.76 µs |
| Attention `[1,2,17,64]` (S=17, D=64, causal) | 15.20–15.30 µs / 15.26–15.33 µs | 50.71–60.58 µs / 51.63–68.56 µs |

The device side is stable to well under 1 µs; the host side moves by ~20 % with
machine load because it includes allocation, staging and the fence wait.

**These are measurements on this device only** (GTX 1080 Ti, driver 580.196.07)
and are not a cross-device or cross-backend claim. They are also not throughput
numbers: each sample is a whole op round trip with one submission, and the
per-row kernels deliberately trade intra-row parallelism for bit-comparable
accumulation order (§1).

## 6. Validation summary

Validation is enabled in code (`VulkanConfig::validated()`); `VK_INSTANCE_LAYERS`
was **not** set (that would duplicate the layer). Every test captures
`validation_counts()` before opening the device and asserts it is unchanged
after teardown. The `fn3_vulkan_attention` run ends with:

```text
[fn3vk] backend=vulkan adapter="NVIDIA GeForce GTX 1080 Ti" type=DISCRETE_GPU uuid=471b2bda-51e1-6451-e692-ff5d1e146f1e api=1.4.312 driver=580.196.7.0 queue_family=2 validation=true software=false maxMemoryAllocationSize=4292870144 maxStorageBufferRange=4294967295 chunk_cap=536870912
test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.86s
```

- `[fn3vk][validation][ERROR]` lines: **0**
- `[fn3vk][validation][WARN…]` lines: **0**
- `validation_counts()` delta per test: **0 errors, 0 warnings**
  (`assert_validation_clean` passed in every test)

19 tests = 12 in this file + the 7 `fn3_fixtures` unit tests that ride along with
`mod fn3_fixtures;`.

## 7. Commands and outcomes

All runs on the physical GPU, `CARGO_BUILD_JOBS=4`,
`FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e`, no
`VK_INSTANCE_LAYERS`.

| # | Command | Outcome |
|---|---|---|
| 1 | `cargo fmt --check` | **clean** (after `cargo fmt` on the new files; no other file changed) |
| 2 | `cargo build --no-default-features --features vulkan` | **ok** (only the pre-existing `augment_runtime_telemetry` dead-code and `fnv` unused-dependency warnings from the 2.x tree) |
| 3 | `cargo test --features vulkan --test fn3_vulkan_attention -- --nocapture --test-threads=1` | **19 passed, 0 failed** |
| 4 | `cargo test --features vulkan --test fn3_vulkan_smoke -- --test-threads=1` | **14 passed, 0 failed** (after the `rewrite/fastnn-3` rebase that brings the invariant rewrite; on the pre-rebase branch point the two old snapshot assertions failed, see §8) |
| 5 | `cargo test --no-default-features --features vulkan --lib fn3::backend::vulkan` | **41 passed, 0 failed** |
| 6 | `spirv-val --target-env vulkan1.1 <each .spv>` | **ok** for all five modules (`fn3_layernorm`, `fn3_softmax_last`, `fn3_reduce_sum_mean`, `fn3_argmax`, `fn3_attention_fused`); `.spvasm` committed alongside |
| 7 | validation | **0 errors, 0 warnings** (§6) |
| — | `cargo test --no-default-features --features vulkan --test fn3_contracts` | **43 passed** (the CPU `MaskedAttention` verdict is unchanged: still `Unsupported`) |
| — | `cargo test --features vulkan --test fn3_vulkan_matmul -- --test-threads=1` | **17 passed** (#60's suite still green with the FN3-15 registrations in the same kernel set) |

### Shader compilation

```bash
cd src/fn3/backend/vulkan/shaders
for s in fn3_layernorm fn3_softmax_last fn3_reduce_sum_mean fn3_argmax fn3_attention_fused; do
  glslc --target-env=vulkan1.1 -fshader-stage=compute -O "$s.comp" -o "$s.comp.spv"
  spirv-val --target-env vulkan1.1 "$s.comp.spv"
  spirv-dis "$s.comp.spv" -o "$s.comp.spvasm"
done
```

`kernels/reflect.rs` agrees with every committed module at registration:
`LocalSize 64 1 1`, declared bindings, and push-constant sizes 12 (LayerNorm)
and 8 (Softmax), 0 for the three parameter-buffer kernels. The run prints:

```text
[fn3vk] registered kernel "norm.layernorm.f32" (local_size=[64, 1, 1], 4 binding(s), 12 push-constant bytes, ...)
[fn3vk] registered kernel "norm.softmax.last.f32" (local_size=[64, 1, 1], 2 binding(s), 8 push-constant bytes, ...)
[fn3vk] registered kernel "reduce.sum_mean.f32" (local_size=[64, 1, 1], 3 binding(s), 0 push-constant bytes, ...)
[fn3vk] registered kernel "reduce.argmax.f32_i32" (local_size=[64, 1, 1], 3 binding(s), 0 push-constant bytes, ...)
[fn3vk] registered kernel "attention.fused.f32" (local_size=[64, 1, 1], 7 binding(s), 0 push-constant bytes, ...)
```

## 8. Shared-file note: `tests/fn3_vulkan_smoke.rs` (resolved upstream)

This branch started at `27fc60d`, where FN3-11's smoke suite pinned the
*scaffold-era* verdicts (Softmax and MaskedAttention must be `Unsupported`,
`features.masked_attention` must be false). Registering this issue's kernels
made those two snapshot assertions fail, which is the expected consequence of
closing the gap — not a defect in the kernels.

The integrator rewrote them as invariants in `4d188e8` on `rewrite/fastnn-3`:

* an op that is not `Full` must name its owner, and one such op must still
  exist;
* `features.masked_attention` is asserted **equal to**
  `backend.supports(OpId::MaskedAttention).is_full()`, so a `Full` fused kernel
  makes the flag true (it is, and it does);
* `features.broadcasting` is only checked for the one-way implication that an
  advertised broadcast capability implies a `Full` `Add`.

This branch was rebased onto `rewrite/fastnn-3` (conflict-free; my paths are
disjoint), so the smoke suite is **14/14** without editing that file. I did not
modify `tests/fn3_vulkan_smoke.rs` at any point; the temporary local patch used
before the rebase to prove 14/14 was reverted byte-identically
(`git diff --quiet -- tests/fn3_vulkan_smoke.rs`).

## 9. Known risks and open items

1. **`OpParams` has no boolean for the fused attention's causal flag.** The flag
   rides in `keep_dims`, is documented in `kernels/attention.rs` and enforced
   (`axes` must be empty, `epsilon` must be 0). When #56 replaces `OpParams`
   with typed payloads, move `causal` to a named field.
2. **`keys <= 4096` is part of the documented fused contract.** The score row
   lives in 16 KiB of shared memory — the Vulkan minimum
   `maxComputeSharedMemorySize`, so the shader compiles on any conformant
   device. Longer contexts are rejected by name and the limit is stated in the
   registered `KernelSpec::shape`. Note the trade-off of the `Full` verdict:
   `Capabilities` has no field for a key-extent bound, so a planner that trusts
   `Full` without checking the spec string will learn the limit from the
   rejection. A tiled online softmax (flash-style) is the follow-up that removes
   the bound.
3. **Vulkan division accuracy.** `OpFDiv` is 2.5 ULP, so `ReduceMean` and the
   softmax/attention divisions can differ from IEEE by 1–2 ULP. Measured maxima
   are in §4.2; all are far inside the fixture tolerances. If bit-exact means
   are ever required, the divisor must be applied as a correctly rounded
   multiply by an exactly representable reciprocal (only possible for powers of
   two).
4. **Intra-row parallelism is traded for bit-comparable order.** A row is one
   invocation (norm/softmax) or one workgroup with a serial softmax phase
   (attention). For the pinned model (hidden 128, ≤ 17 keys) this is
   microseconds; a large-batch or long-context deployment wants the tiled
   variant. Timings in §5 include the host round trip and are not throughput
   numbers.
5. **Alpha host-staged tensors.** Every op uploads inputs and downloads the
   output because `Tensor` storage is host-backed (#52/#48). The device-resident
   path exists (`execute_staged` already allocates device buffers and could be
   reused without the transfers); residency records report `host_staged = true`
   and the byte counts, so the overhead is visible.
6. **`ArgMax` `i64` inputs** need `shaderInt64`, not enabled by the platform
   layer. Rejected by name; the pinned graph contains no `ArgMax`.
7. **Unverified on a second adapter.** No lavapipe run is claimed here as
   evidence (the task forbids using it for the gate). The kernels use no
   optional features, and the shared-memory footprint is the Vulkan minimum, so
   a portable build is plausible but untested on another ICD.
8. **Not in scope for this issue (owned by #58/#62):** cached-decode *state*
   layout and the whole-model prefill/decode comparison across context lengths.
   The fused kernel was exercised at query/key extents 1/5, 1/17, 1/33 (decode),
   4/4, 5/5 (GQA), 17/17 and 33/33 (prefill-like), which is the kernel-level
   part of that gate; the end-to-end comparison must still be produced by #62.
9. **`MaskedAttention` mask head broadcast** accepts `[B|1, H|1, S, S_kv]`. A
   per-KV-head mask (`H_kv < H`) is rejected by name rather than silently
   broadcast; add it when a model needs it.

## 10. Reproducing

```bash
cd /home/petrouil/Projects/github/fastnn-wt-61-gpu-attention
U=471b2bda-51e1-6451-e692-ff5d1e146f1e
CARGO_BUILD_JOBS=4 cargo fmt --check
CARGO_BUILD_JOBS=4 cargo build --no-default-features --features vulkan
FN3_VK_DEVICE_UUID=$U CARGO_BUILD_JOBS=4 cargo test --features vulkan \
  --test fn3_vulkan_attention -- --nocapture --test-threads=1
FN3_VK_DEVICE_UUID=$U CARGO_BUILD_JOBS=4 cargo test --features vulkan \
  --test fn3_vulkan_smoke -- --test-threads=1        # 14/14
CARGO_BUILD_JOBS=4 cargo test --no-default-features --features vulkan \
  --lib fn3::backend::vulkan
CARGO_BUILD_JOBS=4 cargo test --no-default-features --features vulkan \
  --test fn3_contracts
FN3_VK_DEVICE_UUID=$U CARGO_BUILD_JOBS=4 cargo test --features vulkan \
  --test fn3_vulkan_matmul -- --test-threads=1
```
