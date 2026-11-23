# FN3-13 — Vulkan pointwise, transform and embedding kernels

**Status:** complete in the worktree; committed on `fn3/59-gpu-pointwise`, not
pushed. One upstream test needs a coordinated change (see §9).

## 1. Branch, commits, base

| Item | Value |
|---|---|
| Worktree | `/home/petrouil/Projects/github/fastnn-wt-59-gpu-pointwise` |
| Branch | `fn3/59-gpu-pointwise` |
| Assigned base SHA | `27fc60d` (verified with `git rev-parse HEAD` before any edit) |
| Integration base after rebase | `c6af6cd` (`rewrite/fastnn-3`, includes `2835c32` merge of #60 and `4d188e8` smoke-test invariants) |
| Feature commit | `fc36218` — `feat(fn3): Vulkan pointwise, transform and embedding kernels` |
| Report commit | the commit that adds this file |
| Remote writes | none: no push, tag, PR or merge |

The rebase onto `rewrite/fastnn-3` was done on the orchestrator's explicit
instruction ("You may bring the fixed test into your branch with
`git rebase rewrite/fastnn-3`"). It applied with no conflict; the changed paths
are disjoint from that branch's.

## 2. Changed paths

`git show --numstat fc36218`:

```text
   203 added   24 removed  src/fn3/backend/vulkan/kernels/embedding.rs
  1409 added   44 removed  src/fn3/backend/vulkan/kernels/pointwise.rs
  1680 added   41 removed  src/fn3/backend/vulkan/kernels/transform.rs
  1904 added    0 removed  tests/fn3_vulkan_pointwise.rs

 16 new GLSL sources (`.comp`, 26–116 lines each, 975 lines total)
 16 new committed SPIR-V modules (`.comp.spv`)
 16 new disassemblies (`.comp.spvasm`, for review, as `compile.sh` produces)
```

New shaders, all under `src/fn3/backend/vulkan/shaders/`:

```text
pointwise_binary.comp   pointwise_unary.comp    pointwise_where.comp
pointwise_compare.comp  cast_to_f32.comp        cast_to_i32.comp
cast_to_i64.comp        cast_to_bool.comp       transform_permute.comp
transform_block_copy.comp  transform_expand.comp  transform_fill.comp
transform_shape.comp    transform_range_f32.comp  transform_range_i64.comp
embedding_gather.comp
```

No file outside the paths assigned to #59 was modified. `KERNEL_DTYPES` was
**not** appended to: the capability table stays `[DType::F32]` (see §8).

## 3. Registered ops and verdicts

`KernelSet::alpha()` order is unchanged (`smoke`, `pointwise`, `transform`,
`embedding`, `matmul`, `norm`, `reduce`, `attention`). The pointwise `Add`
registration replaces the FN3-11 smoke kernel and sets
`features.broadcasting = true`, so `Capabilities::features.broadcasting` is now
`true` and `supports(Add)` is `Full`.

`Full` is claimed only where the kernel covers the whole documented contract of
the op; every `Partial` names its missing case and the owning issue (#59).
`MatMul` is untouched (#60 owns it).

| op | kernel | verdict |
|---|---|---|
| `Add` `Sub` `Mul` `Div` `Pow` | `pointwise.binary.f32` | `Full` (broadcast) |
| `Neg` `Sqrt` `Tanh` `Relu` `Sin` `Cos` `Erf` `Gelu` | `pointwise.unary.f32` | `Full` |
| `Equal` `Less` `Greater` | `pointwise.compare` | `Full` (f32/i32/i64; `Equal` also bool) |
| `Cast` | `cast.to.{f32,i32,i64,bool}` | `Full` (all 16 dtype pairs) |
| `Shape` | `transform.shape` | `Full` |
| `ConstantOfShape` | `transform.fill` | `Full` (all dtypes) |
| `Range` | `transform.range.{f32,i64}` | `Full` |
| `Gather` | `embedding.gather` | `Full` (any axis, i64/i32 indices, all element widths) |
| `Where` | `pointwise.where` | `Partial` — `Bool` value operand rejected (#59) |
| `Transpose` `Reshape` `Flatten` `Squeeze` `Unsqueeze` `Expand` | `transform.permute` / `transform.expand` | `Partial` — 4- and 8-byte elements; `Bool` tensor rejected (#59) |
| `Slice` | `transform.block_copy` | `Partial` — one sliced axis per call; `Bool` rejected (#59) |
| `Split` `Concat` | `transform.block_copy` | `Partial` — one axis, ≤ 16 pieces, 4- and 8-byte elements; `Bool` rejected (#59) |

The `Bool` gap is a platform limit, not a shortcut: core Vulkan cannot write a
single byte atomically without `storageBuffer8BitAccess`, which the FN3-11
platform layer deliberately does not enable. Where a `Bool` **output** is part
of the op (`Equal`/`Less`/`Greater`, `Cast` to `Bool`, `Gather` of a `Bool`
table) one invocation packs four elements into one 32-bit word, so those ops do
cover it.

Design notes worth knowing:

- One block-copy kernel covers `Slice`, `Split` and `Concat` through an
  `outer x axis x inner` decomposition, so the three ops cannot disagree about
  layouts.
- `Pow` reproduces C `pow`'s negative-base semantics. GLSL's `pow` is undefined
  for `x < 0`, and the pinned graph's `gelu_new` cube (`x^3`) is negative for
  negative activations; without this branch the activation would be NaN.
- `Range` i64 is computed with 32-bit word arithmetic (`umulExtended`); the
  platform layer enables no optional shader feature, so `shaderInt64` is not
  available. The launch validates on the host that every value fits `i64`.

## 4. Measured results against the CPU reference

Every op was executed twice from identical bytes: once through
`CpuBackend::execute_op`, once through `VulkanBackend::execute_op`, and compared
with the bound stated below. "exact" means `assert_eq!` on the output window
bytes (max_abs = mean_abs = 0).

| op family | case | max_abs | mean_abs | bound used | basis |
|---|---|---:|---:|---|---|
| `Add` | `[3,1,5]` op `[4,5]` → `[3,4,5]` | 0 | 0 | exact | one IEEE-754 op |
| `Add` | prime tail 100,003 | 0 | 0 | exact | |
| `Sub` | broadcast | 0 | 0 | exact | |
| `Mul` | broadcast | 0 | 0 | exact | |
| `Mul` | vector × scalar `[1]` | 0 | 0 | exact | |
| `Div` | broadcast | 9.537e-7 | 6.518e-8 | atol 1e-6 + rtol 1e-6 | GPU division is 1 ulp off |
| `Pow` | negative bases, `x^2`/`x^3` | 1.907e-6 | 2.645e-7 | atol 1e-5 + rtol 1e-5 | GPU `pow` ≈ 2 ulp vs libm |
| `Pow` | cubed prime tail 100,003 (bases ±4) | 6.104e-5 | 1.203e-5 | atol 1e-5 + rtol 1e-5 | values up to 64, relative ≈ 2e-7 |
| `Neg` | 100,003 | 0 | 0 | exact | |
| `Relu` | 100,003 | 0 | 0 | exact | |
| `Sqrt` | 100,003 (positive) | 2.384e-7 | 1.735e-8 | atol 1e-6 + rtol 1e-6 | GPU sqrt 1 ulp |
| `Tanh` | 100,003, \|x\|≤6 | 1.192e-7 | 4.395e-9 | atol 1e-6 + rtol 1e-6 | |
| `Sin` | 100,003, \|x\|≤12 | 7.078e-7 | 1.627e-7 | atol 1e-6 + rtol 1e-6 | |
| `Cos` | 100,003, \|x\|≤12 | 7.749e-7 | 1.818e-7 | atol 1e-6 + rtol 1e-6 | |
| `Erf` | 100,003, \|x\|≤5 | 5.383e-7 | 5.445e-8 | atol 1e-6 | A&S 7.1.26 |
| `Erf` | 9 high-precision table points | ≤ 5.4e-7 | — | atol 1e-6 | |
| `Gelu` | 100,003, \|x\|≤6 | 4.768e-7 | 7.625e-8 | atol 2e-6 + rtol 1e-6 | parity GELU row |
| `Gelu` | **fixture `gelu.erf.out`** (69) | 4.768e-7 | 7.545e-8 | atol 2e-6 + rtol 1e-6 | `fn3_fixtures` |
| `Where` | cond `[2,1,4]`, x `[2,3,4]`, y `[1]` (f32) | 0 | 0 | exact | |
| `Where` | i64 `[4]` + scalar | 0 | 0 | exact | |
| `Equal`/`Less`/`Greater` | f32 broadcast `[2,3]` op `[3]` | 0 | 0 | exact | |
| `Equal`/`Less`/`Greater` | i64 extremes incl. `i64::MIN/MAX`, `2^40` | 0 | 0 | exact | |
| `Equal`/`Less`/`Greater` | i32 | 0 | 0 | exact | |
| `Less` | prime tail 100,003 bool | 0 | 0 | exact | 4-byte packing tail |
| `Cast` | all 16 dtype pairs (incl. NaN, ±inf, 1e30, `i64::MIN/MAX`) | 0 | 0 | exact | |
| `Gather` | **fixture `embedding.out`** `[17,7]`×`[2,5]` | 0 | 0 | exact (fixture atol=rtol=0) | `fn3_fixtures` |
| `Gather` | axis 1 of `[2,3,4]`, 5 i64 indices incl. negatives | 0 | 0 | exact | |
| `Gather` | i64 rank-1 table + i32 indices | 0 | 0 | exact | |
| `Gather` | Bool table, 5 indices, negative index | 0 | 0 | exact | |
| `Transpose` | `[2,0,1]` on `[2,3,4]` | 0 | 0 | exact | |
| `Transpose` | default (reverse) on `[2,3,4]` | 0 | 0 | exact | |
| `Transpose` | 4-D i64 `[0,1,3,2]` | 0 | 0 | exact | |
| `Reshape` | `[-1,4]`, `[0,3,-1]` | 0 | 0 | exact | |
| `Flatten` | axis 2 | 0 | 0 | exact | |
| `Squeeze` | `[1,3,1,4]` → `[3,4]`; axes input | 0 | 0 | exact | |
| `Unsqueeze` | `[3,4]` → `[1,3,4,1]` | 0 | 0 | exact | |
| `Slice` | axis 1, start 1, open end, step 2 | 0 | 0 | exact | |
| `Slice` | axis 0 reversed (`i64::MAX`,`i64::MIN`,-1) | 0 | 0 | exact | |
| `Split` | `[2,9]` axis 1 sizes `[3,5,1]` | 0 | 0 | exact | |
| `Concat` | 3 pieces axis 0; 3 pieces axis 1 | 0 | 0 | exact | |
| `Expand` | `[1,3]` and `[3]` → `[2,3]` | 0 | 0 | exact | |
| `Shape` | `[2,3,4]` → i64 `[2,3,4]` | 0 | 0 | exact | |
| `ConstantOfShape` | f32 (−FLT_MAX), i64, Bool | 0 | 0 | exact | |
| `Range` | f32 (0,10,2.5), (1,−1,−0.5); i64 (5,0,−2), (0,8,3), (−4,4,1) | 0 | 0 | exact | |

Reading the table: the pointwise arithmetic that is *one* IEEE-754 operation is
bit-identical. The transcendental families are within 1–13 f32 ulp of the
platform libm, which is what a GPU can promise without a software `f64` path.
`Erf` uses Abramowitz & Stegun 7.1.26; its 5.4e-7 measured error is well inside
the model-facing GELU bound (4.77e-7 measured on the fixture against a 2e-6
bound).

## 5. Evidence: no silent host fallback

Every test asserts, for every executed op:

```text
record.backend == "vulkan"
record.proves_device_execution()      (>=1 submission, non-zero dispatch,
                                       >=1 device-local binding, no input readback)
record.submissions == 1
record.inputs_read_back == false
record.device_local_bytes > 0
record.readback_bytes > 0
```

`ResidencySummary::input_readbacks` stays 0 for the whole run. The only host
reads of tensor data are the ones the alpha storage model forces (the inputs are
host-backed and must be uploaded) plus the *validation* reads documented in §6;
no output element is computed on the host.

Odd sizes that exercise the workgroup tail: 100,003 elements (100,003 % 64 = 3)
for the binary, unary, compare and pow cases, and a 5-element `Bool` output
(5 mod 4 = 1) for the packed compare/gather paths.

## 6. What runs on the host, by design

The issue asks for this boundary to be stated:

- Output **shapes** are resolved on the host from the pre-allocated output
  tensor and from the small integer tensors ONNX uses as shape inputs
  (`Reshape` target, `Squeeze`/`Unsqueeze` axes, `Slice` starts/ends/steps,
  `Split` sizes, `Expand` target, `ConstantOfShape` shape, `Range` scalars).
  This is exactly the CPU reference's job and re-deriving it on the device would
  duplicate the rules without computing a single output element on the GPU.
- `Gather` **bounds-checks every index** on the host before dispatch, so a bad
  index is `Fn3Error::OutOfBounds` (the CPU's error) and appends no record. A
  shader cannot return an `Fn3Error`; this is the acceptance criterion "bad
  indices have defined behavior".
- `ConstantOfShape`'s fill value and `Range`'s start/delta are parameter
  scalars, passed as push constants; `Shape`'s extents are metadata passed as
  push constants and written by a real dispatch.
- Everything else — every output element of every op — is produced by a
  compute dispatch on the physical device.

## 7. Validation and shaders

Physical GPU used for every run, selected by UUID, never by index:

```text
[fn3vk][test] device="NVIDIA GeForce GTX 1080 Ti"
  uuid=471b2bda-51e1-6451-e692-ff5d1e146f1e driver=580.196.7.0
  queue_family=2 validation=true
```

Validation is enabled in code (`VulkanConfig::validated()`); no
`VK_INSTANCE_LAYERS` is passed on any command line. The per-test guard prints,
verbatim, eleven times (once per test):

```text
[fn3vk][validation] delta_errors=0 delta_warnings=0 total_errors=0 total_warnings=0
```

So the run is **0 errors, 0 warnings**, and no test leaked validation messages
into a later test (the counters are process-global and monotonic).

Shaders: `compile.sh` compiled and `spirv-val --target-env vulkan1.1` validated
every module. 18 committed `.spv` files exist (16 from #59, `smoke_add` from
#57, `matmul_tiled` from #60); all 18 pass `spirv-val`.

## 8. Commands and actual results

| # | Command | Result |
|---|---|---|
| 0 | `git rev-parse HEAD` (before edits) | `27fc60d9d4b42a3b0e1e20e883daecb5cfa93871` — matches the assignment |
| 1 | `cargo fmt --check` | clean, no diff |
| 2 | `CARGO_BUILD_JOBS=4 cargo build --no-default-features --features vulkan` | `Finished`; only the two pre-existing warnings (`unused dependency fnv` in Cargo.toml, dead `augment_runtime_telemetry` in the frozen 2.x tree) |
| 3 | `CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_pointwise -- --nocapture --test-threads=1` | `test result: ok. 18 passed; 0 failed` (11 #59 tests + 7 `fn3_fixtures` self-tests), 0/0 validation |
| 4 | `CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_smoke -- --test-threads=1` | `test result: ok. 14 passed; 0 failed` |
| 5 | `CARGO_BUILD_JOBS=4 cargo test --no-default-features --features vulkan --lib fn3::backend::vulkan` | `test result: ok. 30 passed; 0 failed` |
| 6 | `CARGO_BUILD_JOBS=4 cargo test --no-default-features --test fn3_contracts` | `test result: ok. 43 passed; 0 failed` (extra sanity: core untouched) |
| 7 | `bash src/fn3/backend/vulkan/shaders/compile.sh` | all 18 `.comp` compiled and validated (see the SIGPIPE note in §9); no committed artifact changed except #59's |
| 8 | `spirv-val --target-env vulkan1.1` on every `.spv` | 18/18 valid |
| 9 | `git rebase rewrite/fastnn-3` | `Successfully rebased`; no conflict |

Not run, and why:

- `cargo test --test fn3_cpu_reference` and the 2.x suites: no CPU or 2.x file
  changed. The CPU backend is used as the oracle by the #59 test instead.
- The lavapipe (`FN3_VK_TEST_SOFTWARE=1`) lane: this test file is a
  physical-device gate and has no software lane by design; the platform-level
  software lane is `fn3_vulkan_smoke`'s.
- `cargo clippy`: not part of the project's gates, and the workspace has
  pre-existing clippy noise in the frozen 2.x tree that would drown the signal.

## 9. Files another issue must change

1. **`tests/fn3_vulkan_smoke.rs` — the "unsupported op" probe selects a
   `Partial` op** (owner: #57 / integrator; I did not edit it).

   ```rust
   let unsupported = OpId::ALL.iter().copied()
       .find(|op| !backend.supports(*op).is_full())
       .expect("alpha must not claim Full support for every op id");
   ...
   let error = backend.execute_op(unsupported, &[], &mut [], &OpParams::default())
       .expect_err("an op with no Vulkan kernel must be rejected");
   assert!(matches!(error, Fn3Error::UnsupportedOp { .. }));
   ```

   `Partial` means "a registered kernel implements a named subset", so asking
   for it with zero operands correctly yields `ArityMismatch` (or a shape error),
   not `UnsupportedOp`. Today the first non-`Full` id is `Softmax` (14), which is
   `Unsupported`, so the test passes — but #61 will make 14–18 `Full` and the
   probe will then land on `Slice` (25), which #59 registers as `Partial`, and
   the smoke test will fail for a reason that is not a defect. The fix is one
   line, matching the test's own comment ("an op with no Vulkan kernel"):

   ```rust
   .find(|op| !backend.supports(*op).is_supported())
   ```

   I hit exactly this failure before making `Gather` genuinely `Full`
   (`Gather` is id 1, so a `Partial` `Gather` was selected); making `Gather`
   cover `Bool` tables was the right kernel fix, but it only moves the problem
   to #61's integration.

2. **`src/fn3/backend/vulkan/shaders/compile.sh` exits 141** (owner: #57).
   `set -euo pipefail` plus `glslc --version | head -n 1` makes the script die
   with SIGPIPE *after* compiling and validating everything. A CI lane that
   checks the exit status would report failure for a successful build. One-line
   fix: `glslc --version | head -n 1 || true` (and the same for `spirv-val
   --version`). I left the script alone because the #59 brief reserved only new
   files under `shaders/`.

3. **`Capabilities::dtypes` remains `[DType::F32]`** (owner: #48/#62). The #59
   brief explicitly said not to append to `KERNEL_DTYPES`, so I did not — but
   `Gather` consumes `I64`/`I32` indices, `Shape` produces `I64`, `Cast` covers
   all 16 pairs, and `ConstantOfShape` fills any dtype. Those are structural
   dtypes, not compute dtypes, and the op-level `OpKernel::dtypes` records them
   accurately. A planner that gates *structural* ops on
   `capabilities().supports_dtype(I64)` would refuse work the kernels can do.
   Recommend a decision in #48 (for example a separate
   `structural_dtypes` field, or documenting `dtypes` as compute-only).

## 10. Known risks and remaining work

- **`Where` uses 140 push-constant bytes** (four 8-entry `uint` extent arrays
  plus three scalars). The GTX 1080 Ti allows 256, so it registers; a device
  with the guaranteed-minimum 128 bytes would fail at `open` with the size and
  the limit named. Trimming it would need a uniform-buffer parameter block,
  which is not worth it before the alpha gate.
- **`Slice` moves one axis per dispatch.** The pinned graph's slices are
  single-axis (`/transformer/Slice` on the KV sequence axis). Multi-axis slices
  are rejected, not approximated, and the `Partial` verdict says so. A
  multi-dispatch chain would be a #59 follow-up if a real model needs it.
- **`Split`/`Concat` cap at 16 pieces** because one submission uses one
  transient descriptor set per dispatch (the platform pool holds 64). The
  pinned graph uses three-way splits and two-way concats.
- **`Erf` is an approximation** (A&S 7.1.26, measured 5.4e-7 max absolute).
  `GELU` uses it and passes the parity fixture at 4.77e-7 against a 2e-6 bound.
  If a future op needs a correctly rounded `erf`, it needs a higher-order
  polynomial, not a different contract.
- **`Sqrt`/`Div`/`Pow` are 1–2 ulp off libm** on this GPU. The measured bounds
  above are stated in the test, and the model-level parity tolerance
  (`atol 1e-5 + rtol 1e-4`) is comfortably wider.
- **`Gather`'s host-side index validation reads the index tensor on the host**
  (it is host-backed anyway in the alpha storage model). This is a validation
  read, not a device readback, and `inputs_read_back` stays false; #52/#48's
  device-resident storage will need a GPU-side bounds story (a clamp plus a
  status word, or a two-pass check) when it lands.
- **No chaining yet.** Each op is one submission with an upload and a readback,
  because the alpha `Tensor` is host-backed. The platform layer's
  `alloc_buffer`/`dispatch` path is ready for #62's device-resident chains.
