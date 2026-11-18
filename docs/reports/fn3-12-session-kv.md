# FN3-12 (#58) — session-owned KV state, prefill and cached decode

**Branch** `fn3/58-session`
**Base** `90a4e5960f042623cd4bbba06813fff659f12727` (verified with
`git rev-parse HEAD` before the first edit)
**Tracking issue** #46 · **Issue** #58 · **Status** implementation complete on
the CPU reference backend; the full-model Vulkan path is implemented but blocked
by two cross-issue preflight gaps (reported in §9). Local commits only, no remote
write.

---

## 1. What landed

The missing half of the alpha: a session owns bounded key/value buffers, runs a
specialised plan for `past = 0` (prefill) or for the current extent (decode),
appends the newly computed `present` slice and commits the extent and the
counters only after the whole step succeeded.

```text
Session::with_runtime(&model, &import, &backend)
  prefill(tokens) ─► plan(seq = tokens.len(), past = 0)  ─┐
  decode(tokens)  ─► plan(seq = tokens.len(), past = live) │ PlanCache: one build per
                                                            │ distinct (seq, past)
        ─► materialise slots ─► bind ids/positions/mask/past KV
        ─► plan.execute(backend, slots)
        ─► copy logits[last] ─► validate present[*]
        ─► append at the live extent, all layers ─► advance extents and counters
```

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/model/kv.rs` | 714 | mine (new; ADR §5 reserves it) — `KvCacheConfig`, `KvCacheLayout`, `KvCacheState`, the append protocol |
| `src/fn3/model/session.rs` | 1203 | mine (new; ADR §5 reserves it) — `Session`, `SessionPhase`, `prefill`/`decode`/`resume`, plan cache, slot pool |
| `tests/fn3_session.rs` | 1645 | mine (new) |
| `src/fn3/model/mod.rs` | +26 / −410 | shared: two `pub mod` lines, two `pub use` blocks, the moved `Session`/KV declarations, one removed device gate (§9.2), the moved unit test |
| `src/fn3/AGENTS.md` | +13 / −2 | closest guide: `model/kv.rs` and `model/session.rs` in the ownership list |
| `docs/reports/fn3-12-session-kv.md` | this file | mine |

Nothing else changed: no backend, kernel, import, format, plan, fixture, tool,
CI or roadmap file. `cargo fmt --check` is clean.

`mod.rs` keeps its facade role: `Session`, `SessionPhase`, `KvCacheConfig`,
`KvCacheLayout`, `KvCacheState` and `MAX_KV_LAYERS` keep their public paths and
are re-exported from the new modules; `ExecutionPlan`, `ModelWeights` and
`PreparedModel` stay where they were.

## 2. KV layout and bounds arithmetic

One `KvCacheState` per layer, two buffers per state, fixed for the session's
life:

```text
keys    [batch, n_kv_heads, capacity_tokens, head_dim]   f32, contiguous
values  [batch, n_kv_heads, capacity_tokens, head_dim]   f32, contiguous
live_len                                                 tokens committed
```

Pinned geometry (`DecoderGeometry::kv_cache_config(17, 1)`):

| Figure | Value |
|---|---|
| layers / batch / KV heads / head_dim | 2 / 1 / 2 / 64 |
| elements per buffer | `1 × 2 × 17 × 64 = 2,176` |
| bytes per buffer | `2,176 × 4 = 8,704` |
| bytes per layer (keys + values) | `17,408` |
| whole session cache | `2 × 17,408 = 34,816` |

Bounds arithmetic, all checked:

- `capacity_tokens` must satisfy `1 ≤ capacity ≤ envelope.total.max()`
  (`KvCacheConfig::validate`); `layers ≤ MAX_KV_LAYERS = 1024`; `batch`,
  `n_kv_heads`, `head_dim` and `capacity_tokens` must be positive.
- An append of `n` tokens requires `live_len + n ≤ capacity_tokens`
  (`checked_add`; overflow is `ShapeOverflow`) and is otherwise
  `CapacityExceeded { what: "KV cache live extent", requested, capacity }`.
  That is the exact variant `tests/fn3_contracts.rs` already pins.
- The decoder plan is only legal while `past + seq ≤ envelope.total.max()`;
  the planner enforces it (`DimOutOfEnvelope`), and the session enforces
  `past + seq ≤ position-table rows` (`CapacityExceeded`) because a learned
  position table is finite (1024 rows for the pinned model).
- Token *t* of head *h* starts at element `((b·heads + h)·capacity + t)·head_dim`
  in the buffer and at `((b·heads + h)·n + t)·head_dim` in a `present` slice.
  Both ends of every row are checked before the first byte is written.

The session does **not** feed a strided view of the cache to the graph.
`live_keys()`/`live_values()` build a contiguous
`[batch, n_kv_heads, live_len, head_dim]` copy (8.7 KB per buffer at 17 tokens)
because the Vulkan kernels reject strided operands. The append itself reads a
zero-copy strided view of the plan's `present` output and writes the new rows at
the live extent.

## 3. The public API #62 / #63 / #64 should call

```rust
use fastnn::fn3::model::{Session, SessionPhase};

// One session per conversation. The backend is the one and only executor; its
// device must equal the model's (DeviceMismatch otherwise).
let mut session = Session::with_runtime(&model, &import, &backend)?;

// Prompt: returns the logits of the final prompt position, [1, 1, vocab]
// (generally [batch, 1, vocab]) as a fresh tensor owned by the caller.
let logits = session.prefill(&prompt)?;

// One cached step: same return shape, appends one token to every layer.
let next = session.decode(&[token])?;

// State a caller may inspect:
session.phase();                 // Empty | Prefilling | Ready
session.step();                  // successful decode calls
session.tokens_consumed();       // prefill + decode tokens
session.cache_len();             // live extent (tokens)
session.caches();                // &[KvCacheState] (read-only)
session.kv_bytes();              // 34,816 for the pinned model
session.plan_hits(); session.plan_misses(); session.plan_evictions();
session.slot_allocations(); session.slot_reuses(); session.slot_bytes_allocated();
session.last_plan();             // Option<PlanId>
session.last_plan_dims();        // Option<LiveDims>
session.reset();                 // extents and counters to zero, buffers kept
```

`Session::resume(&keys, &values, tokens)` seeds the cache from an externally
computed prefix (a persisted prompt cache). It computes nothing, and it is what
makes the GPU decode test in §7 possible.

`Session::new(&model)` remains for a model with one fixed-shape plan; `prefill`
and `decode` on such a session return `SessionState` naming
`Session::with_runtime`.

`KvCacheState` (`src/fn3/model/kv.rs`) is public for callers that read the cache:

```rust
cache.config(); cache.keys(); cache.values(); cache.capacity(); cache.live_len();
cache.check_capacity(n); cache.check_append(n);
cache.check_present(&keys, &values) -> Result<u64>;      // tokens
cache.append(&keys, &values) -> Result<u64>;             // new extent
cache.live_keys(); cache.live_values();                  // contiguous copies
cache.allocated_bytes(); cache.reset();
```

## 4. What `prefill` and `decode` return, and what "step" counts

- **`prefill` returns the logits of the final prompt position only**, shape
  `[1, 1, vocab]` for the pinned model (`[batch, 1, vocab]` generally), copied
  out of the plan's `[1, seq, vocab]` output. It is a fresh allocation, so the
  caller may keep it while the session continues; no session scratch is handed
  out.
- **`decode` returns the logits of the last new position**, same shape. A
  multi-token decode is one plan at `seq = tokens.len()`, one append, one step.
- **`step()` counts successful `decode` calls.** Not tokens, not prefills. A
  13-token prefill followed by four one-token decodes reports `step() == 4`,
  `tokens_consumed() == 17`, `cache_len() == 17`.
- `phase` is `Empty` before a prompt, `Prefilling` only while a step is running
  (no call is legal then), `Ready` after a committed prefill, resume or decode.
  A failed prefill returns the session to `Empty`; a failed decode returns it to
  `Ready`.

## 5. Atomicity

The step commit order is the contract (module docs of `session.rs`):

```text
plan ─ capacity check (all layers) ─ materialise ─ bind ─ execute ─
logits copy-out ─ present validation (all layers) ─ append (all layers) ─ counters
```

Everything fallible happens before the first cache byte is written. For the
multi-layer commit the session calls `check_append` **and** `check_present` for
every layer, then appends every layer; `KvCacheState::append` recomputes and
bounds-checks every destination and source row before it borrows a buffer
mutably, so an append that passed its checks cannot fail halfway.

Evidence (`capacity_and_mid_plan_failures_change_nothing`):

| Injected failure | Result |
|---|---|
| decode at `live_len == capacity == 17` | `CapacityExceeded { what: "KV cache live extent", requested: 18, capacity: 17 }`; extent, `step`, `tokens_consumed`, phase and `kv_bytes` unchanged |
| prefill of 14 tokens (capacity 17, envelope `seq ≤ 13`) | `DimOutOfEnvelope`; extent 0, `step` 0, tokens 0, phase `Empty`, plan cache empty |
| backend rejects instruction 37 of a 144-instruction prefill | `UnsupportedOp`; 38 of 144 instructions ran; extent 0, `step` 0, tokens 0, phase `Empty` |
| same backend, disarmed | identical prefill commits; exactly 144 `execute_op` calls |
| backend rejects instruction 5 of a decode | extent 4, `step` 0, tokens 4, phase `Ready` unchanged; a retried decode commits to extent 5, `step` 1 |

## 6. Plan cache and allocation behaviour

- The plan cache is **per session** (`PlanCache`, capacity
  `min(2 × capacity_tokens + 1, 64) = 35` for the pinned model). It is state: a
  session's hit/miss/eviction counters must not be shared, which is why two
  sessions do not share one cache. Plans are immutable and content-identical
  across sessions — the isolation test asserts two sessions report the same
  `PlanId` for the same shapes.
- Measured (CPU, one prefill 13 + four decode steps): `hits 0`, `misses 5`,
  `evictions 0`, `cached 5` — one build per distinct `(seq, past)`, which is
  what specialisation means: a decode at `past = 13` is a different plan from
  `past = 14`. A repeated round over the same five shapes (reset, prefill,
  decode ×4) is served with **`hits 5`, `misses 5`**: after the first round there
  is no recompilation per step.
- Slot pool: instruction-output buffers are keyed by `(name, dtype, shape)` and
  carried from the previous step; only one generation is retained, so the pool
  is bounded by one plan's outputs. Measured over the same round:
  379 allocations, 361 reuses, 30,168,442 bytes allocated in total (dominated by
  the `seq = 13` prefill; a decode step allocates only the shapes that grew).
  Weights are `Arc`-shared, never copied; folded constants share their
  allocation and are only re-tagged for the model's device.
- The KV cache is allocated once per session (`KvCacheState::new`) and never
  reallocated: `reset()` reuses the buffers and leaves the plan cache warm.

## 7. Verification

Model: `/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`,
27,897,974 bytes, SHA-256
`e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1` (measured
with `sha256sum`; it matches `fixtures/parity/tiny-gpt2/model.lock.json`, and the
FNV-1a fingerprint `0xc294d3ab837fb1be` that `tests/fn3_planner.rs` pins). The
digest quoted in the task text has a transposed pair (`…c4ef7b6…`); the file on
disk matches the lock file, not that line.

Tolerances: tiny logits/KV `atol 1e-5`, `rtol 1e-4`, `mean_abs ≤ 2.5e-6`;
argmax is a failure only when the reference top-two gap exceeds `2 × atol`.

### 7.1 Prefill, CPU reference (`CpuBackend`, debug build)

| Fixture | measured `max_abs` | measured `mean_abs` |
|---|---:|---:|
| `prefill.p1.logits` (stored, full `[1,1,50257]`) | 5.9604645e-7 | 9.7683845e-8 |
| `prefill.p4.last_logits` | 5.159527e-7 | 7.895393e-8 |
| `prefill.p8.logits` | summary-only; not asserted (§8) | — |
| `prefill.p13.last_logits` | 5.364418e-7 | 1.0014763e-7 |
| `prefill.p1.present.{0,1}.{key,value}` | 1.49e-7 … 2.83e-7 | 2.68e-8 … 7.66e-8 |
| `prefill.p4.present.{0,1}.{key,value}` | 1.49e-7 … 3.28e-7 | 2.66e-8 … 7.75e-8 |
| `prefill.p8.present.{0,1}.{key,value}` | 1.79e-7 … 4.17e-7 | 2.77e-8 … 8.05e-8 |
| `prefill.p13.present.{0,1}.{key,value}` | 1.79e-7 … 4.17e-7 | 2.65e-8 … 7.46e-8 |

### 7.2 Cached decode, CPU reference

Single session: `prefill(tokens[0..13])`, then four one-token decodes.
Extents after each step: **14, 15, 16, 17** (asserted exactly).

| Fixture | measured `max_abs` | measured `mean_abs` |
|---|---:|---:|
| `decode.step1.logits` (stored) | 6.109476e-7 | 1.0241756e-7 |
| `decode.step1.present.*` | 1.79e-7 … 4.17e-7 | 2.60e-8 … 7.53e-8 |
| `decode.step2.logits` (summary-only) | first/last eight ≤ 1.79e-7, min/max match, argmax 11 = recorded 11 | — |
| `decode.step2.present.*` | 1.79e-7 … 4.17e-7 | 2.59e-8 … 7.41e-8 |
| `decode.step3.logits` (summary-only) | first/last eight ≤ 2.68e-7, min/max match, argmax 612 = recorded 612 | — |
| `decode.step3.present.*` | 1.79e-7 … 4.17e-7 | 2.67e-8 … 7.46e-8 |
| `decode.step4.logits` (summary-only) | first/last eight ≤ 2.98e-7, min/max match, argmax 373 = recorded 373 | — |
| `decode.step4.present.*` | 1.79e-7 … 4.17e-7 | 2.66e-8 … 7.43e-8 |

**Cached-vs-full.** The `full.p{14..17}.logits` payloads are summary-only, so the
reference was *regenerated*: a second `PreparedModel` with the wider envelope
`seq ≤ 17, past = 0, total ≤ 17` prefills the whole 14/15/16/17-token sequence in
a fresh session and its last row is compared with the cached decode logits for
the same position.

| Comparison | measured `max_abs` | `mean_abs` | fixture's ORT cached-vs-full |
|---|---:|---:|---:|
| cached `decode` vs full recompute, p14 | 0.0 | 0.0 | 5.36e-7 |
| p15 | 0.0 | 0.0 | 5.96e-7 |
| p16 | 0.0 | 0.0 | 5.96e-7 |
| p17 | 0.0 | 0.0 | 4.77e-7 |

The two paths are **bit-identical** here, because the CPU reference evaluates the
same operands in the same order: the appended cache bytes are exact copies of the
`present` slice, so the attention arithmetic sees the same `k`/`v` rows. This is
stronger than the gate but it is *not* independent evidence about the arithmetic;
the independent evidence is the fixture comparison above. As a second,
independent check the full recompute's last row is compared with the recorded
`full.pN.logits` slice summary:

| Recorded reference | measured first/last/min/max `max_abs` | argmax |
|---|---:|---:|
| `full.p14.logits[13]` | 3.05e-7 | 24671 = recorded |
| `full.p15.logits[14]` | 2.83e-7 | 11 = recorded |
| `full.p16.logits[15]` | 1.79e-7 | 612 = recorded |
| `full.p17.logits[16]` | 2.38e-7 | 373 = recorded |

### 7.3 Session behaviour, CPU reference

| Property | Evidence |
|---|---|
| isolation | two sessions have distinct key/value pointers on every layer; `prefill(4)` on one and `prefill(13)` + `decode` on the other leave each extent, `step` and `tokens_consumed` separate; a third session reproduces the first's logits byte for byte and the same `PlanId`; a model clone shares one `Arc` per weight |
| no hidden device | `CountingBackend` records **288 `execute_op` calls for 2 steps (144 + 144 instructions)** and **0 operands tagged for another device** |
| device rule | a model prepared for `vulkan(0)` is rejected by `CpuBackend` with `DeviceMismatch { expected: vulkan(0), actual: cpu }` before any allocation or planning |
| no-NaN | every logits and KV element of every prefill/decode step is asserted finite; the plan's mask constants are checked finite for five shapes |
| all-masked rows | the pinned plan executed with a fully zero `attention_mask` keeps the first attention softmax row **uniform `1/4`** (`/transformer/h.0/attn/Softmax_output_0`) and its logits finite |
| capacity | 17 live tokens is the boundary; the 18th is `CapacityExceeded` and changes nothing |
| atomicity | table in §5; the cumulative *work* counters (plans built, buffers allocated) legitimately advance on a failed step and are documented as diagnostics, not state |

### 7.4 Physical GPU (`--features vulkan`)

Device: **NVIDIA GeForce GTX 1080 Ti**, UUID
`471b2bda-51e1-6451-e692-ff5d1e146f1e`, discrete GPU, driver **580.196.7.0**,
Vulkan API 1.4.312, queue family 2, selected by UUID (lavapipe is enumerated as
the second adapter and never used). Validation is enabled **in code**
(`VulkanConfig::validated()`); `VK_INSTANCE_LAYERS` is not set.
`FN3_VK_REQUIRE_PHYSICAL=1` makes a missing device a failure, not a skip.

Validation summary for the whole run: **0 errors, 0 warnings** (absolute, and the
per-test zero-delta assertion passed).

Cached decode of four steps on the device, session resumed from the committed
13-token reference prefix:

| Fixture | measured `max_abs` | measured `mean_abs` |
|---|---:|---:|
| `decode.step1.logits` | 7.748604e-7 | 1.2772918e-7 |
| `decode.step1.present.*` | 2.68e-7 … 4.77e-7 | 3.19e-9 … 7.19e-9 |
| `decode.step2` logits (summary-only) | first/last eight ≤ 2.98e-7, argmax 11 = recorded | — |
| `decode.step2.present.*` | 2.98e-7 … 4.77e-7 | 6.40e-9 … 1.23e-8 |
| `decode.step3` logits (summary-only) | first/last eight ≤ 1.64e-7, argmax 612 = recorded | — |
| `decode.step3.present.*` | 2.98e-7 … 4.77e-7 | 8.39e-9 … 1.57e-8 |
| `decode.step4` logits (summary-only) | first/last eight ≤ 3.05e-7, argmax 373 = recorded | — |
| `decode.step4.present.*` | 2.98e-7 … 4.77e-7 | 1.04e-8 … 1.90e-8 |

Extents: 14, 15, 16, 17 (asserted). `plan hits 0, misses 4` (each `past` is a new
shape).

**Residency counters proving the GPU run dispatched, not the host:**

| Counter | Value |
|---|---:|
| executions (one per instruction) | **576** (`4 × 144`) |
| queue submissions | **576** |
| records proving device execution | **576** |
| device-local bindings | **1,484 / 1,484** |
| input readbacks | **0** |
| `host_staged` | **true** on every record |
| upload / readback | 318,020,376 B / 104,359,960 B |
| workgroups, all records | non-zero |
| validation errors / warnings | **0 / 0** |

Every record was additionally checked for `backend == "vulkan"`,
`device == vulkan(0)`, ≥ 1 submission, a non-zero dispatch, a device-local
binding and `inputs_read_back == false`.

**No silent CPU fallback.** The production path refuses the pinned graph on
Vulkan with a typed error before executing anything:

```text
op transpose (axis permutation view) is unsupported by backend vulkan:
only partially supported: 4- and 8-byte elements (F32/I32/I64); a Bool tensor
is rejected with UnsupportedOp naming #59 (at plan instruction 1, op transpose)
```

and the test asserts the residency ledger is **empty (0 records)**, the cache
extent is 0, `step` is 0 and the phase is `Empty` after that refusal. Nothing
ran on the host.

## 8. Commands, with real outcomes

All runs from `/home/petrouil/Projects/github/fastnn-wt-58-session` with
`CARGO_BUILD_JOBS=4` and
`FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`.

| Command | Result |
|---|---|
| `cargo fmt --check` | **clean** |
| `cargo build --no-default-features` | **ok** (pre-existing 2.x `unused dependency fnv` manifest warning) |
| `cargo build` | **ok** |
| `cargo build --no-default-features --features vulkan` | **ok** |
| `cargo test --test fn3_session -- --nocapture --test-threads=1` | **8 passed**, 0 failed (118 s) |
| `cargo test --test fn3_planner` | **11 passed** |
| `cargo test --test fn3_import` | **27 passed** |
| `cargo test --test fn3_cpu_reference` | **43 passed** |
| `cargo test --test fn3_model_format` | **38 passed** |
| `cargo test --test fn3_contracts` | **42 passed, 1 failed** — the stale `NotImplemented` assertion (§9.3) |
| `cargo test --lib` | **683 passed**, 0 failed |
| `cargo test --features vulkan --test fn3_session -- --nocapture --test-threads=1` (GTX 1080 Ti by UUID) | **10 passed**, 0 failed (120 s), validation 0/0 |

Raw logs: `/tmp/opencode/fn3-58-logs/cpu-session.log`,
`/tmp/opencode/fn3-58-logs/gpu-session.log`.

Skipped, and why:

- `cargo test --features vulkan --lib` and the other `fn3_vulkan_*` suites are
  not re-run here: they are #57/#60/#61's gates and nothing this change touches
  is on their path.
- Full-model **prefill on the GPU** and any GPU run of a model as prepared by
  `PreparedModel::new` without a test scaffold are blocked (§9).
- The stale contract test was not edited: `tests/fn3_contracts.rs` is outside
  this issue's ownership (§9.3).

## 9. Blockers found (cross-issue), and exactly what is needed

### 9.1 No Vulkan `Plan` can be built for the pinned graph

`Planner::plan` calls `Support::require_full` for every instruction
(`src/fn3/model/specialize.rs`, `Survey::finish`), and the Vulkan structural
kernels report `Partial`:

```text
transpose, reshape, split, concat, slice  -> partial (4- and 8-byte elements; Bool rejected, #59)
softmax                                   -> partial (trailing axis only, #61)
```

For the *concrete* pinned graph every such instruction is inside the documented
subset (instruction 1 is `transpose` of an **f32** weight; no Bool is ever
transposed), but the verdict is op-level. `check_capabilities` additionally
rejects the plan because `KERNEL_DTYPES` is `&[DType::F32]` while the graph has
`i64` inputs and folded `i64`/`i32`/`bool` constants — even though
`embedding.gather` documents `i64`/`i32` indices.

**What I need** (either one unblocks the production GPU session):

1. Shape-aware preflight in `src/fn3/model/specialize.rs` (#56 core): accept
   `Support::Partial` when the plan's concrete slots satisfy the kernel's
   documented subset, e.g.

   ```rust
   // in Survey::finish, replacing require_full for Partial verdicts
   if let Support::Partial { reason } = backend.supports(op) {
       backend.admits_concrete(op, &[SlotPlan], instruction)?;
   }
   ```
   — needs a new `Backend` method (core-owned) such as
   `fn admits(&self, op: OpId, inputs: &[SlotPlan], params: &OpParams) -> Result<(), Fn3Error>`;

2. or make the structural kernels `Full` for what they already implement and
   add `I64`/`I32`/`Bool` to `KERNEL_DTYPES` (`src/fn3/backend/vulkan/kernels/`,
   #59/#57 owned).

Until then the **production** GPU path correctly refuses, and the GPU decode
evidence in §7.4 uses a **test-only** adapter in `tests/fn3_session.rs`
(`RelaxedPreflight`) that answers `Full` where the inner backend says `Partial`
and widens the capability dtype list, then delegates execution unchanged to
`VulkanBackend`. It cannot compute anything and cannot hide a rejection — an
unsupported concrete shape still fails inside the Vulkan kernel — and the test
first asserts the unrelaxed production refusal with an empty ledger.

### 9.2 A prefill with an empty past cannot execute on Vulkan

The graph's `present = Concat(past, new)` receives a zero-element past tensor at
`past = 0`. The Vulkan runtime refuses zero-byte buffers:

```text
invalid shape in vulkan runtime: fn3.concat.in: refusing a zero-byte Vulkan
buffer; a zero-element tensor must be handled before allocation
```

So even with §9.1 resolved, `Session::prefill` cannot run on the device until
`Concat` (or the buffer layer) handles a zero-element input — `#57`/`#59` owned.
Cached decode with `past > 0` is unaffected, which is why the GPU evidence covers
four real decode steps and not a device prefill.

### 9.3 One stale assertion in `tests/fn3_contracts.rs` (owned by #48)

`session_phase_rules_and_atomic_failures` asserts that
`Session::new(&model).prefill(&[1, 2])` returns
`Fn3Error::NotImplemented { what: "Session::prefill", owner: "FN3-12 (#58)" }`.
That entry point is now implemented, and a session built by `Session::new` binds
no shape planner, so the honest answer is `SessionState` naming
`Session::with_runtime`. I did not edit the file: it is outside this issue's
ownership. The required patch is:

```rust
    let error = session.prefill(&[1, 2]).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::SessionState {
            action: "prefill",
            ..
        }
    ));
    assert!(error.to_string().contains("with_runtime"), "{error}");
```

The other 42 contract tests pass unchanged, including
`prepared_model_is_shareable_and_sessions_own_their_state` and
`kv_cache_capacity_is_checked_before_mutation`.

### 9.4 One intentional change in `src/fn3/model/mod.rs`

`PreparedModel::new` used to reject any KV cache configuration on a non-CPU
device (`DeviceResolveError::unavailable_device`). That gate is now removed: the
alpha cache is host memory on every device, which is staging, not a CPU compute
path, and a Vulkan-tagged session must be constructible. It is a one-hunk change
inside the KV capability gate and it is documented on the constructor. No test
pinned the old behaviour.

## 10. Known risks

- **Host-staged cache and slots.** Every session buffer is host memory tagged for
  the model's device, exactly like every other alpha tensor. The GPU per-step
  cost is dominated by staging (318 MB uploaded and 104 MB read back over four
  decode steps of a 27.9 MB model), not by arithmetic. Device-resident KV needs
  a `Storage` variant owned by #48/#52; the session's allocation point is one
  function (`kv::allocate`) so that change is local.
- **`live_keys`/`live_values` copy.** One contiguous copy of the live prefix per
  layer and per step (68 KB at 17 tokens for two layers). Required because the
  Vulkan kernels reject strided operands; a device-resident cache would hand out
  a buffer instead.
- **Batch 1.** The session serves one sequence; `with_runtime` rejects
  `batch > 1` with a named error rather than silently using row 0.
- **Plan-cache capacity** is `min(2 × capacity + 1, 64)` per session. Beyond
  that, the smallest key is evicted and rebuilt, which is correct but not free.
  A workload with a very large `capacity_tokens` and interleaved shapes will see
  rebuilds; the counters make it visible.
- **`resume` is a data-in path.** It trusts the caller's prefix bytes; it checks
  shape, dtype, device, extent and envelope, not provenance. A wrong prefix
  produces wrong logits, not an error.
- **The GPU evidence's preflight relaxation** (§9.1) is a test scaffold, not a
  supported configuration. It must be deleted when either fix in §9.1 lands, and
  the production refusal test flipped into a positive assertion.
- **`step` semantics** are decode calls, not tokens. A consumer that wants a
  token count must read `tokens_consumed()`.

## 11. Remaining work for other owners

| Owner | Work |
|---|---|
| #56 (core) | shape-aware preflight for `Support::Partial` (§9.1); or |
| #59 / #57 | make `Transpose`/`Reshape`/`Split`/`Concat`/`Slice` `Full` for their documented subset and add the index dtypes to `KERNEL_DTYPES` |
| #57 / #59 | accept a zero-element tensor at `Concat` (or skip it) so `prefill` can run on a device (§9.2) |
| #48 | apply the one-test patch in §9.3 |
| #62 | replace the `RelaxedPreflight` scaffold with the production path once §9.1/§9.2 land; run prefill + decode of the pinned model on the GTX 1080 Ti and assert the same residency counters |
| #63 / #64 | call `Session::with_runtime` + `prefill`/`decode`; a CLI needs a tokenizer and a `plan_hits`/`tokens_consumed` line for observability |
| #48 / #52 | device-resident `Storage` so the KV cache and value slots stop staging through host memory |
