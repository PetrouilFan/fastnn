# FN3-10 (#56) — bounded shape specialisation and execution plans

**Branch** `fn3/56-planner`
**Base** `456f3a35d26ae6f202d4bffac0abf448b45887f0` (verified with
`git rev-parse HEAD` before the first edit)
**Tracking issue** #46 · **Issue** #56 · **Status** implementation complete, local
commits only, no remote write

---

## 1. What landed

The runtime planner: an imported graph plus a concrete shape environment becomes
an immutable, bounded, executable [`Plan`] whose every slot has a concrete shape
and whose integer shape arithmetic has been folded into static constants.

```text
ModelImport ──► resolve   substitute LiveDims, fold the integer shape algebra
            ──► build     liveness, slot table, constants, instructions
            ──► preflight Support::require_full, capabilities, mask constants
            ──► Plan      ExecutionPlan + slots + constants + identity + bytes
```

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/model/plan.rs` | 1433 | mine (new, reserved by ADR §5) |
| `src/fn3/model/specialize.rs` | 2687 | mine (new, reserved by ADR §5) |
| `tests/fn3_planner.rs` | 1047 | mine (new) |
| `src/fn3/model/mod.rs` | +7 | shared: two `pub mod` lines, two `pub use` blocks |
| `src/fn3/AGENTS.md` | +4 | closest guide: the two new `model/` modules in the ownership list |
| `docs/reports/fn3-10-planner.md` | this file | mine |

Total: 5,167 new lines of implementation and tests plus this report, 11 inserted
lines in two shared files and no other change anywhere.

`src/fn3/model/mod.rs` is the shared source file touched (plus four lines in
`src/fn3/AGENTS.md` so its ownership list names the new modules, as that guide's
own policy requires). The `mod.rs` change is exactly this, inserted between
`pub mod weights;` and `use std::cell::Cell;`:

```rust
pub mod plan;
pub mod specialize;
pub mod weights;

pub use plan::{
    BoundValue, Constant, ConstantData, FoldedNode, MaskConstant, MaskRole, Plan, PlanBound,
    PlanCache, PlanId, SlotPlan, SlotRole,
};
pub use specialize::{PlanAttr, Planner};
pub use weights::{WeightId, WeightSet, WeightSource, WeightTensor};
```

Nothing else in that file moved, so #55's parallel edit should merge as a
adjacent-line conflict at worst.

## 2. The public API #58 / #62 / #63 should call

```rust
use fastnn::fn3::model::{Plan, PlanBound, PlanCache, Planner, PlanId};
use fastnn::fn3::{CpuBackend, LiveDims, ShapeEnvelope};

let imported = import_onnx_file(path)?;
let backend = CpuBackend::new();
let planner = Planner::new(&imported, &backend);

// The model's own bounds: fixed head geometry, free batch/seq/past/total.
let envelope: ShapeEnvelope = planner.envelope(/*batch*/ 1, /*seq*/ 13,
                                               /*past*/ 17, /*total*/ 18)?;

// The bound every admissible shape must respect (plans the worst corners).
let bound: PlanBound = planner.bound(&envelope, 1)?;

// One plan per concrete environment; deterministic and content-addressed.
let prefill: Plan = planner.plan(&envelope, LiveDims::new(1, 4, 0, 2, 2, 64)?)?;
let decode:  Plan = planner.plan(&envelope, LiveDims::new(1, 1, 17, 2, 2, 64)?)?;
bound.admits(&prefill)?;   // CapacityExceeded naming the exceeded figure
bound.admits(&decode)?;

// Allocation is a pure function of the plan.
let mut slots = prefill.materialise(&model_weights, DeviceId::Cpu)?;
let ids = prefill.input("input_ids").expect("declared graph input");
slots[ids.slot() as usize] = Some(tokens);
prefill.execute(&backend, &mut slots)?;
let logits = prefill.output("logits").expect("declared graph output");
let tensor = slots[logits.slot() as usize].as_ref().expect("the plan writes it");

// Reuse across calls: one envelope, one backend, key (batch, seq, past).
let mut cache = PlanCache::new(envelope, backend.name(), /*capacity*/ 8)?;
let same = cache.plan(&planner, &envelope, dims)?;   // Arc<Plan>, ptr-equal
```

| Need | API |
|---|---|
| specialise | `Planner::plan(&ShapeEnvelope, LiveDims) -> Result<Plan, Fn3Error>` |
| build the envelope | `Planner::envelope(max_batch, max_seq, max_past, max_total)` |
| prove a shape set | `Planner::bound(&ShapeEnvelope, batch) -> PlanBound`, `PlanBound::admits(&Plan)` |
| reuse | `PlanCache::{new, plan, get, clear, hits, misses, evictions}` |
| execute | `Plan::materialise(&ModelWeights, DeviceId)`, `Plan::execute(&dyn Backend, &mut [Option<Tensor>])` |
| name a slot | `Plan::input(name)`, `Plan::output(name)`, `Plan::slot_of(name)`, `Plan::slots()` |
| cache key | `Plan::identity() -> PlanId` (128-bit, stable, content-only) |
| budget | `Plan::{value_count, peak_bytes, allocation_bytes, weight_bytes}` |
| evidence | `Plan::{instructions, folded, folded_node(onnx_index), constants, mask_constants, dims, envelope, backend_name}` |

`Plan::materialise` leaves graph-input slots `None` (the caller owns them),
clones weight slots (an `Arc` bump, not a byte copy), copies folded constants and
zero-allocates instruction outputs. Alpha storage is host memory on every device
(`Tensor::from_storage`), so this is the CPU path; device-resident allocation
stays with #58.

## 3. Planner design

### 3.1 Resolution

`Survey::resolve` walks `GraphIr` in file order, which the importer guarantees is
topological, and maintains two per-value tables:

* `shapes: Vec<Option<Shape>>` — the resolved concrete shape;
* `data: Vec<Option<ConstantData>>` — the known integer payload, if any.

Leaves are resolved from their declaration: a graph input's `ValueShape` has its
`Dim::Symbol`s replaced by `LiveDims`; a weight's `ValueShape` likewise, checked
against the weight's own element count so a declared scalar and a one-element
payload cannot disagree. A `Dim::Unknown(reason)` is a hard error naming the
value, the axis and the importer's reason — never a guess.

Node outputs are *recomputed* from op semantics. There is no code path that copies
the importer's inferred shape into the plan: that is why the 24 values the
importer could not resolve resolve here. As a second safety net, whenever the
importer's inference *is* fully static after substitution, the planner compares it
with its own result and refuses the plan if the two disagree. That check catches
semantic drift between #54 and #56 instead of letting it become wrong memory
addressing; it fires on no value of the pinned graph.

Shape rules are implemented for all 40 `OpId`s (the `match` is deliberately
exhaustive, so adding an op id without a shape rule is a compile error).

### 3.2 The 24 unresolved values

The importer leaves exactly 24 values with one `Dim::Unknown`: the eight
`Reshape`s that flatten `[batch, seq, hidden]` to `[batch*seq, hidden]` before
each `Gemm`, plus the sixteen `MatMul`/`Add` results that inherit that axis.
`Reshape` target resolution (ONNX `allowzero = 0`) divides the input element
count by the product of the known target extents, so `[-1, 128]` on
`[1, 4, 128]` becomes `[4, 128]` and `[-1, 128]` on `[1, 13, 128]` becomes
`[13, 128]`. `tests/fn3_planner.rs::pinned_importer_unresolved_values_all_resolve_in_the_plan`
asserts the import still has exactly 24 such values and that every one of them has
a slot with a concrete shape in the plan for `seq = 13`.

### 3.3 Folding rule

A node folds when its result is determined at build time:

* `Shape` folds whenever its operand's *metadata* is resolved (it needs no data);
* every other foldable op folds when all of its operands are known constants.

The foldable set is the shape/control algebra: `Shape`, `Gather`, `Slice`,
`Concat`, `Unsqueeze`, `Squeeze`, `Reshape`, `Flatten`, `Expand`,
`ConstantOfShape`, `Range`, `Cast`, `Equal`, `Less`, `Greater`, `Where`, `Add`,
`Sub`, `Mul`, `Div`, `Neg`, `Pow` — and only when the output dtype is `I64`,
`I32` or `Bool`.

**Floating-point arithmetic is never folded.** `f32` work stays on the backend so
there is exactly one numerical path and the planner cannot become a second,
subtly different reference implementation. That is also why the additive mask is
not materialised as a plan constant: it stays a runnable subgraph, and only the
small integer tensors that drive its shape are folded.

A folded node produces a `Constant` and has **no instruction at all**. If an
integer *arithmetic* node (`MatMul`, `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Neg`,
`Sqrt`, `Erf`, `Tanh`, `Gelu`, `Relu`, `Softmax`, `LayerNorm`, `ReduceMean`,
`ReduceSum`) survives folding with a non-`f32` operand, the plan is refused with
`UnsupportedOp { backend: "planner", reason: "… Owner: FN3-10 (#56) for the fold,
FN3-07 (#53) for an integer kernel if a fold is impossible" }`. That guard is the
release condition: an integer `Add` can never reach a kernel.

### 3.4 The four integer-arithmetic nodes, folded

The pinned exporter computes `total`, `past` and the causal index range with `i64`
`Add`/`Sub`. In the specialised plan for `seq = 4, past = 9`:

| ONNX | node | folded value | evidence |
|---:|---|---|---|
| 17 | `/transformer/Add_1` | `seq + past = total` | `plan.folded_node(17)` |
| 21 | `/transformer/Sub` | `total - seq = past` | `plan.folded_node(21)` |
| 27 | `/transformer/Add_3` | `seq + past = total` | `plan.folded_node(27)` |
| 44 | `/transformer/Add_2` | `Range(0, seq) + 1` | `plan.folded_node(44)` |

The *values* are pinned through live constants whose content depends on those
results, because dead constants are pruned (node 17's own result is consumed only
by node 21, which also folds):

* `/transformer/Concat_4_output_0` = `[seq, total - seq]` = `[4, 9]` — the mask's
  past extent is the fold of the `Sub`;
* `/transformer/Concat_2_output_0` = `[4, 4]` — the causal-mask shape;
* `/transformer/h.0/attn/c_attn/Concat_1_output_0` = `[1, 4, 384]` — the reshape
  target of the fused QKV projection, one of the 24 previously unresolved values;
* `/transformer/Cast_2_output_0` = the `16`-element causal index mask, exactly
  `column <= row`, which is the whole folded `Range → Add → Reshape → Less` chain.

Both tests also assert that no instruction in any plan touches a non-`f32` slot
with an arithmetic op.

### 3.5 Slot table, allocation and liveness

Every value that a surviving instruction reads or writes, plus every graph
output, gets exactly one slot, in `GraphIr` value order. Each slot records its
name, dtype, resolved shape and a `SlotRole`:

| Role | Meaning | `materialise` |
|---|---|---|
| `GraphInput` | `GraphIr` input | leaves `None` |
| `Weight { id }` | initializer | clones the `Arc<Tensor>` |
| `Constant { index }` | folded | materialises from `Plan::constants` |
| `InstructionOutput { instruction }` | written by that instruction | `Tensor::zeros` |

Weights that only fed folded nodes are pruned, which is why the plan references
27,832,996 of the import's 27,833,180 weight bytes (34 of the 50 weight entries).

`peak_bytes` is a real liveness figure, not a sum: resident slots (graph inputs,
weight handles at zero cost, constants, graph outputs) are live for the whole
step and every other value is released after the instruction that last reads it.
An output is added before the inputs of the same instruction are released, so the
figure covers a real executor rather than an optimistic pool.

## 4. Plan identity

`PlanId` is a 128-bit FNV-1a digest over a canonical little-endian encoding of:

1. the string `fastnn.fn3.plan.v1` and the backend name;
2. all seven envelope bounds, and all seven concrete `LiveDims`;
3. every slot in index order: dtype code, rank, extents, role tag, value name,
   and the weight handle / constant index / instruction index;
4. every constant: dtype, shape and exact payload bytes;
5. every instruction: op code, input and output slot lists in order, attribute
   axes, `keep_dims`, and `epsilon.to_bits()`;
6. every graph input and output: name, slot, dtype and shape.

No hash map is iterated, no pointer or address is written, and floats enter the
digest as bits, so the identity is stable across processes and machines. It is a
fingerprint, not a cryptographic hash (this crate has no crypto dependency);
`Plan` derives `PartialEq` and `PlanCache::clear` exists for the one thing an
identity cannot see — the weights or the device changing.

Evidence: building the same plan twice produces `plan1 == plan2` and equal
identities, and `(1,0)`, `(2,0)`, `(13,0)`, `(1,1)`, `(1,17)` all produce
different identities.

## 5. Bounds: declared, computed, measured

`Planner::bound` plans the maximal element(s) of the admissible set
`{(past, seq) : past + seq = total, inside the envelope}` — one corner when
`past.max + seq.max <= total.max`, otherwise the two ends of the
`past + seq = total.max` frontier — and takes the maximum of every figure.

For the pinned model with `envelope(1, 13, 17, 18)`:

```text
corners: (seq 1, past 17, total 18) and (seq 13, past 5, total 18)
<= 210 value slots
<= 144 instructions
<= 28,507,141 peak bytes
<= 29,652,789 allocation bytes
   27,832,996 weight bytes (Arc-shared, resident)
```

Every prefill and decode plan, measured:

| seq | past | total | instructions | slots | peak bytes | allocated bytes | identity |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 1 | 144 | 210 | 25,943,873 | 26,029,563 | `13978fd5ad6b906f2aa8719826bc9613` |
| 4 | 0 | 4 | 144 | 210 | 26,579,360 | 26,923,296 | `59842a3dccfa4b0806fd2824e4199866` |
| 8 | 0 | 8 | 144 | 210 | 27,426,816 | 28,117,824 | `3588744d8c9fb818b998545a07be42e2` |
| 13 | 0 | 13 | 144 | 210 | 28,486,361 | 29,615,619 | `5f993dbc02f490241e06489b59a025ee` |
| 1 | 0 | 1 | 144 | 210 | 25,943,873 | 26,029,563 | `13978fd5ad6b906f2aa8719826bc9613` |
| 1 | 1 | 2 | 144 | 210 | 25,947,981 | 26,035,821 | `fa6cf06acdd2222ca84ec988d057d118` |
| 1 | 2 | 3 | 144 | 210 | 25,952,089 | 26,042,079 | `dd1768c87853ee44e5a6dc7d098a6b90` |
| 1 | 3 | 4 | 144 | 210 | 25,956,197 | 26,048,337 | `3a9c045b038830aef3144a2411696025` |
| 1 | 4 | 5 | 144 | 210 | 25,960,305 | 26,054,595 | `f713ef4c03081728d6533a2c22886503` |
| 1 | 5 | 6 | 144 | 210 | 25,964,413 | 26,060,853 | `a4e7bd50eee35aea08bbe20ad25c922c` |
| 1 | 6 | 7 | 144 | 210 | 25,968,521 | 26,067,111 | `21dc026674f1064639df96d3253ff0fa` |
| 1 | 7 | 8 | 144 | 210 | 25,973,141 | 26,073,369 | `1e3429b573a0e2595a17f9025a70c241` |
| 1 | 8 | 9 | 144 | 210 | 25,978,273 | 26,079,627 | `0b77c0672d63eef462f9f1ea85b5f87f` |
| 1 | 9 | 10 | 144 | 210 | 25,983,405 | 26,085,885 | `a83de8ea5f421ae09b8b55b3b392bc5f` |
| 1 | 10 | 11 | 144 | 210 | 25,988,537 | 26,092,143 | `e5eb12715287d7c8700c3ade202e947b` |
| 1 | 11 | 12 | 144 | 210 | 25,993,669 | 26,098,401 | `0244a6adaeaa4e0697afcf710eba81f4` |
| 1 | 12 | 13 | 144 | 210 | 25,998,801 | 26,104,659 | `cf8a2a1aaf1997734c04e849fae2dfc1` |
| 1 | 13 | 14 | 144 | 210 | 26,003,933 | 26,110,917 | `4cef003e8cdc0ac0e07b5cc1a5ba1051` |
| 1 | 14 | 15 | 144 | 210 | 26,009,065 | 26,117,175 | `29bd78cfa37953b85bab9b6f9f231c53` |
| 1 | 15 | 16 | 144 | 210 | 26,014,197 | 26,123,433 | `e65aa7fcc682f836b13d063b727b0263` |
| 1 | 16 | 17 | 144 | 210 | 26,019,329 | 26,129,691 | `63442a1bf91ac49e941dff0a7f13ebc1` |
| 1 | 17 | 18 | 144 | 210 | 26,024,461 | 26,135,949 | `d0545f1588c7f1b49a81d6d70565fbda` |

Every row is admitted by `PlanBound::admits`. The instruction count and slot
count are identical for every shape (a folding or pruning decision that depended
on `seq` would make the bound unsound), and the growth in bytes is exactly the
`total`-linear term.

**What the peak actually is.** Peak − 25,731,584 = logits plus the small integer
plumbing, and 25,731,584 = `128 × 50,257 × 4`: the plan materialises the
`Transpose` of the token embedding table (`wte`) because the exporter feeds
`lm_head` as `Transpose(wte)`. That single instruction output is 99.2 % of the
peak at `seq = 1` (25,731,584 of 25,943,873 bytes). Removing it needs `MatMul` with `transpose_b` (see §7), and it
is the largest available memory win on the whole path. It is reported, not hidden.

## 6. Backend integration

The planner never assumes. For every instruction it calls
`backend.supports(op).require_full(op, backend.name())`; `Partial` is a plan
failure, because a plan is a promise for one concrete shape and "partially
supported" is not a promise it can keep. The error keeps the op, the backend and
the owner from the backend's own reason string and appends the instruction
position. It then checks `Capabilities` for every slot: `check_dtype`,
`check_rank` (8 on the CPU) and `check_buffer` (16 GiB on the CPU).

That integration is tested with two honest test backends over the real graph: one
reports `Partial` for `Softmax` and the plan fails with
`UnsupportedOp { op: Softmax, backend: "partial-test", reason: "only partially
supported: …" }`, the other reports `Unsupported` for `MatMul` and the plan fails
naming `MatMul`.

`MaskedAttention` (id 40) has a shape rule derived from #61's documented fused
contract (`q [b,h,q,d] × k [b,vh,keys,d] → out = q`, second output `probs`), so a
graph that used the fused id would plan against a backend that reports it `Full`
(Vulkan) and be rejected by the CPU plan, which is the honest outcome. The pinned
export does not use the fused id: it decomposes attention and the plan runs that
decomposition, exactly like the CPU reference.

## 7. The `OpParams` gap, and what #48 should add

### 7.1 What the planner does today

`PlanAttr` in `specialize.rs` is the typed view of the alpha attribute carrier,
and `PlanAttr::encode()` is the only code in the planner that writes `OpParams`.
Every instruction is `PlanAttr::decode(op, node.params)?.encode()`, so the typed
view and the encoding cannot drift; a field the op does not define is rejected
rather than copied through.

| Op | `OpParams` fields | Meaning |
|---|---|---|
| `Gather`, `Concat`, `Split` | `axes = [axis]` | the one axis |
| `Softmax` | `axes = [axis]` | softmax axis (may be negative) |
| `ArgMax` | `axes = [axis]`, `keep_dims` | axis and `keepdims` |
| `Transpose` | `axes = perm` | full permutation |
| `ReduceMean`, `ReduceSum` | `axes`, `keep_dims` | normalised non-negative axes |
| `Trilu` | `axes = [upper]` | `0` = lower, non-zero = upper |
| `LayerNorm` | `axes`, `epsilon` | normalised axes, epsilon |
| `MaskedAttention` | `keep_dims` | `true` = causal; `axes` empty, `epsilon` 0 |
| every other op | all neutral | any non-neutral field is rejected |

A unit test round-trips every variant (`decode(encode(x)) == x` and
`encode(decode(p)) == p`) and pins that `Attention { causal: true }` encodes to
exactly `keep_dims = true, axes = [], epsilon = 0.0` — #61's workaround, isolated
to two lines.

### 7.2 The variant I want #48 to add

One enum behind the `OpParams` fields instead of three generic ones:

```rust
// requested addition to `src/fn3/backend/mod.rs` (FN3-02 / #48).
#[derive(Debug, Clone, PartialEq)]
pub enum OpAttributes {
    /// No attributes; the only legal payload for most ops.
    None,
    /// `MatMul`; removes the materialised 25.7 MB `wte` transpose (§5).
    MatMul { transpose_a: bool, transpose_b: bool },
    /// Fused attention.
    Attention {
        causal: bool,
        scale: Option<f32>,
        window: Option<u32>,
        layout: AttentionLayout,
    },
    Softmax { axis: i32 },
    Reduce { axes: Vec<u32>, keep_dims: bool },
    Transpose { perm: Vec<u32> },
    /// `Flatten.axis`, which `OpParams` cannot carry at all today.
    Flatten { axis: u32 },
    LayerNorm { axes: Vec<u32>, epsilon: f32 },
    Trilu { upper: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpParams {
    pub attributes: OpAttributes,
}
```

Replacing three loosely typed fields with one enum makes "an op must reject an
attribute it does not understand" a type-level property: an `Add` with
`OpAttributes::Attention { causal: true }` is unrepresentable rather than a
runtime check. It also removes the `keep_dims` overload, gives `MatMul` the
transpose flags that let the planner delete the largest scratch allocation on the
whole path, and closes `Flatten.axis`, which the importer accepts but drops.

### 7.3 The second, smaller gap: `Flatten.axis`

`src/fn3/import/graph.rs::plan_flatten` accepts an `axis` attribute and computes
the output shape with it, but the plan carries no attribute for `Flatten`, so
this planner can only plan the ONNX default (`axis = 1`). A non-default axis makes
the planner's resolved shape disagree with the import's inference, and the
declared-vs-resolved check refuses the plan. The pinned graph has no `Flatten`, so
this is a documented limitation, not a silence.

## 8. Verification

Every command below was run in this worktree with `CARGO_BUILD_JOBS=4` and the
pinned model at `FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`.

| Command | Result |
|---|---|
| `cargo fmt --check` | exit 0, clean |
| `cargo build --no-default-features` | exit 0 |
| `cargo build` | exit 0 |
| `cargo test --test fn3_planner -- --test-threads=1` | **11 passed**, 0 failed |
| `cargo test --test fn3_import -- --test-threads=1` | **27 passed** |
| `cargo test --test fn3_contracts -- --test-threads=1` | **43 passed** |
| `cargo test --test fn3_cpu_reference -- --test-threads=1` | **43 passed** |
| `cargo test --lib -- --test-threads=1` | **649 passed** |
| `cargo test --test fn3_ci_smoke -- --test-threads=1` | **5 passed** |
| `cargo clippy --all-targets --no-default-features` | 0 findings in `plan.rs`, `specialize.rs`, `tests/fn3_planner.rs` |

The model file's SHA-256 is
`e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`
(27,897,974 bytes), which is the value in `fixtures/parity/tiny-gpt2/model.lock.json`
and in `src/fn3/backend/cpu/support.rs`. The task brief's hash differs by a
transposition (`…c4ef7b6a…` instead of `…c4aef7b6…`); the file on disk matches the
lock file, and the test additionally checks the FNV-1a fingerprint
`0xc294_d3ab_837f_b1be` that `tests/fn3_import.rs` pins, so a wrong graph fails
rather than silently changing the numbers.

### 8.1 The plan actually runs

`pinned_prefill_executes_and_matches_the_committed_fixtures` plans, allocates,
executes and compares against the committed ORT/PyTorch reference. No `Session`
is involved, so #58 was not needed: `Plan::materialise` + `Plan::execute` is
enough, which is exactly the seam #58 wraps.

| Case | reference | elements | max_abs | mean_abs | tolerance (FN3-03) |
|---|---|---:|---:|---:|---|
| `prefill.p1` | `prefill.p1.logits.f32` | 50,257 | **5.96e-7** | **9.77e-8** | atol 1e-5, rtol 1e-4, mean_abs 2.5e-6 |
| `prefill.p4.last` | `prefill.p4.last_logits.f32` | 50,257 | **5.16e-7** | **7.90e-8** | atol 1e-5, rtol 1e-4, mean_abs 2.5e-6 |

Both are ~17× inside `atol` and ~25× inside `mean_abs`; argmax matches the
fixture. The comparison is elementwise against
`abs(actual − ref) <= atol + rtol·|ref|` and asserts finiteness, so an all-masked
row could not hide a NaN.

## 9. What I did not reach, and why

* **No decode execution against the decode fixtures.** The fixtures for cached
  decode need the KV cache append protocol, which is #58's (`Session::decode`
  returns `NotImplemented` today). What this issue can prove — and does — is that
  every decode step builds a bounded, fully concrete plan. There is no honest way
  to turn `present.*` back into `past_key_values.*` without writing #58's session
  in my files.
* **No `prefill.p8` / `prefill.p13` logits comparison.** The fixtures for those
  shapes are summary-only (not committed), which the FN3-03 report documents; the
  plan builds for them and is admitted by the bound, but there is no elementwise
  reference to compare against offline.
* **No Vulkan plan.** `Plan` is backend-agnostic and preflights against any
  `Backend`; nothing in this issue was run on a device. The only Vulkan-specific
  thing here is the `MaskedAttention` shape rule, which is derived from #61's
  documented contract and is not exercised by the pinned graph.
* **No `.fnnc`/prepared-model serialization.** #55 owns the bytes; `Plan` and
  `PlanId` are the in-memory contract it can serialize against.
* **The un-foldable-integer-arithmetic rejection and the unresolved-value
  rejection are not reachable from the pinned graph.** That is the point: every
  value resolves and every integer node folds. Both paths are implemented and
  named (they produce `UnsupportedOp`/`InvalidShape` naming #56), and the
  `Shape`-only fold rule is the one non-obvious case that would have hit them: a
  naive "all operands must be constant" rule fails immediately on node 0, which
  reads `input_ids`. I hit that during development and fixed the rule rather than
  weakening it. Honest statement: these two refusals have unit-level coverage but
  no end-to-end fixture, because a graph that triggers them is a graph the pinned
  model is not.

## 10. Known risks

1. **`peak_bytes` assumes a pooling executor.** `materialise` allocates every
   output slot at once, so a caller that uses it directly allocates
   `allocation_bytes`, not `peak_bytes`. `peak_bytes` is the figure a buffer pool
   can meet and the figure the bound promises; #58 should say which one it uses.
2. **Weight bytes are excluded from `peak_bytes`** (reported as `weight_bytes`)
   because they are `Arc`-shared and resident. A caller that copies weights instead
   of sharing them is outside the bound.
3. **Digit-level dependence on the importer's attribute encoding.** `PlanAttr`
   decodes `Gather`/`Concat`/`Split`/`Softmax` as exactly one axis, `Trilu` as
   `axes[0]`, and so on. If #54 changes an encoding, `PlanAttr::decode` starts
   rejecting (loudly, at plan build) rather than mis-executing; the round-trip unit
   test is the tripwire.
4. **`Flatten` is default-axis only** (§7.3).
5. **The plan re-does the 25.7 MB `wte` transpose on every step**, because the
   scale of a plan is per-step. Correct but expensive; fixing it needs the
   `MatMul::transpose_b` attribute and is a #62/#71 concern, not a correctness one.
6. **`Reshape` with `0` copies the input extent positionally**, matching the
   `allowzero = 0` form the importer accepts. `allowzero = 1` cannot reach the
   planner because the importer rejects it.
7. **The bound is proven by monotonicity** of the resolved extents in
   `(batch, seq, past)`. That holds for every decoder graph whose shapes are
   products of the seven symbols with non-negative powers. A graph with a
   subtraction inside a shape (e.g. `seq - 1` as an extent) could break it; the
   planner does not currently allow such a shape to be produced silently, but the
   argument is documented rather than machine-checked. The test suite checks all
   22 planned shapes against the bound, which is the empirical half.

## 11. Remaining work for other owners

* **#48**: add `OpParams::attributes: OpAttributes` (§7.2). The two-line
  `PlanAttr::{decode, encode}` attention arm and the `Flatten` limitation both
  disappear with it.
* **#58**: `Session` should hold `Arc<Plan>` from `PlanCache` and use
  `Plan::materialise` + `Plan::execute`; KV append turns `present.*` into the next
  step's `past_key_values.*`. The plan interface already separates the two.
* **#62**: the end-to-end gate can compare against the same fixtures this report
  uses; `Plan::output("logits")` is the seam.
* **#63**: `Planner::envelope` + `PlanCache` is the CLI's shape configuration
  surface; `PlanBound` is what "this model cannot run a 4096-token prompt" should
  quote.
* **#67**: the `wte` transpose (§5, §7.2) is the first memory/performance item
  this work exposes.
