# FN3-27 — hardening the loaders (loader/parser half)

**Status:** work complete on the branch named below; integrated by the
orchestrator. **Runtime hardening (device loss, unavailable features, repeated
reset, session error recovery, release-build behaviour) is deliberately not in
this report** — it is a separate follow-up gated on #72, as the #73 brief
requires.

| Item | Value |
|---|---|
| Issue | #73 (FN3-27), parent #46 |
| Branch | `fn3/73-hardening` |
| Worktree | `/home/petrouil/Projects/github/fastnn-wt-73-hardening` |
| Base | `672cec9faf089ef0840179f6edfdf33779b576a1` (`git rev-parse HEAD` before the first edit) |
| Scope | `src/fn3/import/**`, `src/fn3/model/format/**`, `fuzz/**`, `tests/fn3_import.rs`, `tests/fn3_model_format.rs`, `tests/fn3_malformed.rs`, `tests/corpus/fn3/**` |
| Model used | `/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`, sha256 `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`, 27,897,974 bytes — checked with `sha256sum` against `fixtures/parity/tiny-gpt2/model.lock.json` (`files[].path == "onnx/model.onnx"`), not copied from the brief |

## 1. Parse-surface inventory

Both readers take untrusted bytes. The table lists every place a length, offset,
count or product from the file is used, the check that guards it, and who owns
the invariant.

### 1.1 ONNX protobuf reader — `src/fn3/import/proto.rs`

| Surface | Line | Check present | Owner |
|---|---|---|---|
| `Reader::remaining` | 83 | `bytes.len() - pos`; `pos <= len` holds because only `take` advances it and `take` rejects `len > remaining` | proto |
| `varint` | 98 | at most 10 bytes; a 10th byte `> 1` is `VarintTooLong` (no u64 wrap) | proto |
| `tag` | 124 | field number 0 rejected; `u64 -> u32` via `try_from` | proto |
| `take` | 137 | `checked_add` **and** `bytes.get(pos..end)`; never slices before both succeed | proto |
| `bytes` (length-delimited) | 154 | `u64 -> usize` via `try_from`, then `take` | proto |
| `sub` (nested message) | 181 | `MAX_DEPTH = 8`, checked before descending; subgraphs are marked, never parsed | proto |
| `skip` (unknown field) | 196 | wire types 0/1/2/5 only; groups (3/4/6/7) are `UnsupportedWireType` | proto |
| `text` (name/domain/dim_param) | 247 | length checked against `MAX_STRING_BYTES = 4096` **before** the slice and before UTF-8 validation | **new** — proto |
| `room_for` + `repeated_*` | 267–472 | per-list budget checked before each push (packed and unpacked) | **new** — proto |
| `decode_model` | 684 | `ir_version != 0`; opset imports capped at `MAX_OPSET_IMPORTS = 16` | proto |
| `decode_graph` | 767 | nodes ≤ `MAX_NODES`, initializers ≤ `MAX_WEIGHTS`, inputs/outputs/value_info ≤ `MAX_VALUES` | **new** — proto |
| `decode_node` | 836 | inputs ≤ `MAX_OPERANDS`, outputs ≤ `MAX_OPERANDS`, attributes ≤ `MAX_ATTRIBUTES` | **new** — proto |
| `decode_attribute` | 888 | two payload fields of different kinds is `ConflictingPayload`; list payloads ≤ `MAX_PARAM_AXES`; a known field with the wrong wire type is an error | proto |
| `decode_tensor` | 1068 | `dims` ≤ `MAX_RANK`; typed payloads ≤ `MAX_TYPED_ELEMENTS`; `raw_data` borrows the input (no copy, no count) | **new** — proto |
| `decode_shape` / `decode_dimension` | 1213 / 1236 | declared shape rank ≤ `MAX_RANK`; both `dim_value` and `dim_param` resolves to the symbol | **new** — proto |

Not reachable through the schema (documented, unit-tested only): `DepthExceeded`.
The deepest legal chain is `ModelProto → GraphProto → ValueInfoProto → TypeProto
→ TypeProto.Tensor → TensorShapeProto → Dimension`, seven levels, and the limit
is eight. `proto.rs`'s own unit test sets `depth` directly to exercise it.

### 1.2 ONNX graph import — `src/fn3/import/graph.rs`

| Surface | Line | Check present | Owner |
|---|---|---|---|
| `build_model` entry | 3297 | model byte length ≤ `MAX_MODEL_BYTES = 8 GiB` before decoding one field | **new** — import |
| `register_nodes` | 3725 | before each node: value/node/weight table below its budget; stops and reports rather than cascading architecture errors | **new** — import |
| `push_value` / `add_weight` | 3629 / 3590 | duplicate names are renamed/reported; ids are `u32` only because the budget caps the table at 2^20 | import |
| `tensor_shape` | 1650 | negative extent is `NegativeExtent`; rank ≤ `MAX_RANK`; `Shape::numel` (core) is checked-multiply | import + core `shape.rs` |
| `tensor_bytes` | 1677 | segment → `SegmentedTensor`; `external_data` **or** `data_location == 1` → `ExternalTensorData`; `raw_data` + typed field → `AmbiguousTensorData`; the populated typed field must match the dtype **before** conversion; typed element count must equal `numel` **before** conversion | **changed** — import |
| `build_weight` | 1800 | `WeightTensor::new` re-derives the byte length from shape × dtype (core owner) | import + core `model/weights.rs` |
| `detect_geometry` | 4088 | layer/hidden/MLP/head/vocab via `u32::try_from`, reported when they do not fit; exact head-count divisibility and KV grouping | import |
| `detect_layer_norm_epsilon` | 4394 | a non-finite scalar in the epsilon slot is now an `Architecture` problem instead of being skipped (`NaN > 0.0` is false) | **changed** — import |
| `plan_slice` | 2258 | start/end/axes lengths cross-checked; **`steps.len() == starts.len()` added** — the extent loop indexed `steps[position]` | **fixed** — import |
| `plan_transpose` | 1918 | each axis bounds-checked; **the result must be a permutation** (`Shape::validate_permutation`) | **changed** — import |
| `AtomSeries` folding | 885–905 | materialised sequences ≤ `MAX_TRACKED_ELEMENTS = 4096`; larger sequences keep the length and drop per-element detail | import |
| `Diagnostics` | error.rs 731 | at most `MAX_PROBLEMS = 100` problems, plus an `omitted` count; a hostile file cannot make the *error* unbounded | import |

Products that are checked by their owner, not re-checked here:
`Shape::numel` and `Shape::byte_len` (`src/fn3/shape.rs`, checked multiply with
`Fn3Error::ShapeOverflow`), `WeightTensor::new` (payload length vs shape ×
dtype), `DimBound::new` / `ShapeEnvelope::validate` (`src/fn3/shape.rs`).

### 1.3 Artifact reader — `src/fn3/model/format/`

| Surface | Line | Check present | Owner |
|---|---|---|---|
| `wire::Reader::take` | wire.rs 131 | `len > remaining` first, then `checked_add`; the comment records why `end <= len` follows | format |
| `wire::Reader::count` | wire.rs 206 | policy cap **and** `count × min_entry_bytes <= remaining`, before any `Vec::with_capacity` | format |
| `wire::Reader::string` | wire.rs 184 | `MAX_STRING_BYTES = 4096` before the slice, UTF-8 validated before `String` | format |
| `wire::Reader::pad_to_8` | wire.rs 229 | padding must be present and zero (canonical encoding) | format |
| `wire::Reader::finish` / `zero_u32` | wire.rs 249 / 261 | exact consumption; reserved words must be zero | format |
| `PreparedHeader::parse` | artifact.rs 222 | min size, magic, version, header length, dtype encoding, reserved words, `file length == bytes.len()`, `payload length == file - 64`, payload ≤ `MAX_PAYLOAD_BYTES = 8 GiB` | format |
| `PreparedArtifact::from_bytes` | 436 | CRC-32 over the payload **before** any section is decoded | format |
| `read_directory` | 816 | section count must equal 6 (cap and fit proof first); tags must be `1..=6` in order; each length ≤ remaining; padding must be zero; the whole payload must be consumed | format |
| `decode_metadata` | 889 | string cap; exact section consumption | format |
| `decode_geometry` | 959 | KV count ≤ `MAX_KV_LAYERS`; position kind, epsilon flag and epsilon bits must be self-consistent | format |
| `decode_envelope` | 1047 | every bound through `DimBound::new`; exact section consumption | format |
| `decode_graph` | 1186 | values ≤ `MAX_VALUES`, inputs/outputs ≤ `MAX_VALUES`, nodes ≤ `MAX_NODES`, node operands/axes ≤ `MAX_PARAM_AXES`, value rank ≤ `MAX_RANK`; all handles bound-checked by `PreparedGraph::new` | format |
| `decode_weights` | 1365 | weights ≤ `MAX_WEIGHTS`, rank ≤ `MAX_RANK`, payload `take` before `to_vec`, `WeightTensor::new` validates length | format |
| `validate` / `check_geometry` | 569 / 584 | layer count ≤ `MAX_KV_LAYERS`; vocabulary ≤ `MAX_VOCAB_SIZE` and **non-zero (new)**; every fixed size non-zero; heads × head_dim == hidden; KV interface count == layer count; token/position table shapes; epsilon finite and positive; fixed geometry inside the declared envelope | **changed** — format |
| `PreparedGraph::check` | format/graph.rs 146 | every value/node/input/output handle in range; both directions of origin vs writer; unique non-empty names | format |

Every offset addition in the format reader is `checked_add`. The only
multiplication on a file-derived value in the format reader is
`count.saturating_mul(min_entry_bytes)` in `count`, which is a *proof* (saturating
is conservative in the reject direction) and never an allocation size.
`check_geometry` also multiplies `query_heads × head_dim`, both `u32`, which
cannot overflow `u64`.

### 1.4 Unchecked arithmetic that remains, and why

| Site | Why it is safe |
|---|---|
| `proto::Reader::remaining` (`len - pos`) | `pos` is advanced only by `take`, which rejects `len > remaining` |
| `Model::function_count += 1`, `string_data_count += 1`, `has_*` counters | one increment per decoded element, so bounded by the input size |
| `graph.rs` `index as u32` / `values.len() as u32` | the new 2^20 budget is the reason this cannot alias; before it, a > 2^32-node graph would have wrapped |
| `i32`/`i64` casts from varints (`data_type`, `data_location`, `int32_data`) | protobuf defines int32 as the truncation of the 64-bit varint; rejecting "out of range" values would reject the spec's own encoding of negative integers. Recorded as a deliberate non-check |
| `decode_weights` `payload.to_vec()` | bounded by the bytes actually present; total copies ≤ section length |

## 2. Defects found and fixed

1. **Panic: `Slice` with a short `steps` operand.** `plan_slice` cross-checked
   `starts`/`ends`/`axes` but not `steps`, then indexed `steps[position]` for
   `position < axes.len()`. A five-operand `Slice` with two starts and one step
   panicked with `index out of bounds: the len is 1 but the index is 1`
   (`src/fn3/import/graph.rs:2323` before the fix). Reproduced by temporarily
   removing the fix and running
   `a_slice_with_a_short_step_list_is_a_typed_error_not_a_panic`; now a
   `ShapeMismatch` naming both lengths. Fixed in `plan_slice`.
2. **Non-finite LayerNorm epsilon silently accepted.** The detector's range test
   (`value > 0.0 && value <= 0.01`) is false for `NaN`, so a corrupt epsilon was
   read as "this is not a LayerNorm" and the geometry carried `None`. A
   non-finite scalar in an `Add(ReduceMean(x), c)` is now an `Architecture`
   problem. `±Inf` and `NaN` are covered by
   `a_non_finite_layer_norm_scale_is_reported` and two corpus cases.
3. **Typed tensor payload converted before it was validated.**
   `tensor_bytes` built the whole `Vec<u8>` from `float_data`/`int64_data`/... and
   compared lengths afterwards, so a tensor declaring `[1]` with 16 M typed
   elements allocated `elements × size` before being rejected. The field is now
   paired with the dtype and its element count compared first; a wrong field is a
   named `TensorDataKind`. Covered by `a_typed_payload_of_the_wrong_kind_is_named`
   and the `onnx_int32_on_float` corpus case.
4. **A non-permutation `Transpose` was accepted.** `normalize_axis` bounds each
   entry but `perm = [0, 0, 1, 2]` passed, producing `OpParams.axes = [0,0,1,2]`
   for a non-transpose. Now `Shape::validate_permutation` is applied at import.
5. **Zero vocabulary was accepted by the artifact reader.** A `[0, hidden]`
   embedding table satisfied every existing check. Now an `Inconsistent` error.
6. **No parser budgets in the ONNX importer.** The importer had one budget
   (`MAX_TRACKED_ELEMENTS`); counts, ranks, string lengths and typed payloads were
   bounded only by the input size. `src/fn3/import/budget.rs` now holds the
   documented budgets and they equal the artifact's, so a model that imports is a
   model the format can write.

Not a defect, recorded because it was investigated: `MIN_VALUE_ENTRY = 11` in
`artifact.rs` looked too large for a value entry (name 4 + dtype 1 + shape 1 +
origin 1 = 7). It is correct: a weight or node origin carries a 4-byte handle, and
a graph-input origin is always accompanied by a 4-byte slot in the input table
that `PreparedGraph::check` requires. The node entry minimum was tightened from
31 to 35 because `check` rejects a node with no output.

## 3. Malformed-input matrix

Every case produces a typed error; none panics. The allocation figures are
measured by §4's counting allocator, not inferred from a struct layout.

| Case | Input | Expected | Observed |
|---|---|---|---|
| empty ONNX | 0 bytes | `Proto::MissingField` | as expected |
| truncated tag | `ir_version` + tag with no length | `Proto::Truncated` | as expected |
| hostile length | 23 bytes claiming 4 GiB of `raw_data` | `Proto::Truncated` | as expected, largest allocation **136 B** |
| 11-byte varint | `[0x08, 0xFF × 11]` | `Proto::VarintTooLong` | as expected |
| group wire type | field 30, wire 3 | `Proto::UnsupportedWireType` | as expected |
| wrong wire type | `ir_version` as length-delimited | `Proto::WrongWireType` | as expected |
| rank 65 tensor | 65 `dims` entries | `Proto::LimitExceeded` ("tensor rank", 64) | as expected |
| 5,000-byte name | `NodeProto.name` | `Proto::LimitExceeded` (4096) | as expected |
| 1,025 node inputs | `Concat` with 1,025 operands | `Proto::LimitExceeded` (1024) | as expected |
| 1,025 attributes | one node | `Proto::LimitExceeded` (1024) | as expected |
| 1,048,577 nodes | 2,097,161-byte file | `Proto::LimitExceeded` | as expected, largest allocation **125,829,120 B** (documented ceiling, asserted < 512 MiB) |
| negative extent | `dims = [-1, 4]` | `NegativeExtent` | as expected |
| external data | `external_data` set, bytes also present | `ExternalTensorData` | as expected |
| `data_location = 1` | payload present | `ExternalTensorData` | as expected |
| `int32_data` on FLOAT | 2 elements | `TensorDataKind` | as expected |
| Slice, 2 starts / 1 step | five operands | `ShapeMismatch` ("2 starts and 1 steps") | as expected; panicked before the fix |
| Slice, 1 start / 3 steps | five operands | `ShapeMismatch` | as expected |
| Transpose `[0,0,1,2]` | rank 4 | `ShapeMismatch` ("permutation") | as expected |
| NaN epsilon | `Add(ReduceMean, NaN)` | `Architecture` | as expected |
| maximum extent | `dims = [i64::MAX]` | `ShapeMismatch` (byte count overflows) | as expected |
| zero-element weight | `[0, 4]` | accepted, 0 bytes, round-trips | as expected |
| NaN/±Inf weight payload | `[3]` | accepted, bit-exact | as expected |
| artifact: empty | 0 bytes | `Truncated` | as expected |
| artifact: bad magic | `XNN3PREP` | `BadMagic` | as expected |
| artifact: truncated header | 63 bytes | `Truncated` | as expected |
| artifact: truncated payload | 8 bytes removed | `LengthMismatch` | as expected |
| artifact: flipped byte | last byte XOR 1 | `ChecksumMismatch` | as expected |
| artifact: future version | `2` | `UnsupportedVersion` | as expected |
| artifact: reserved flags | `1` | `ReservedField` | as expected |
| artifact: sections out of order | META/GEOM tags swapped | `Malformed` | as expected |
| artifact: overlapping sections | META length covers GEOM | `Malformed` | as expected |
| artifact: count at cap | 1,048,576 values, 984-byte file | `Truncated` | as expected, largest allocation **96 B** |
| artifact: count above cap | 1,048,577 values | `LimitExceeded` | as expected |
| artifact: section length `u64::MAX` | META directory entry | `Truncated` | as expected |
| artifact: NaN epsilon | GEOM bits patched | `Inconsistent` | as expected |
| artifact: infinite epsilon | GEOM bits patched | `Inconsistent` | as expected |
| artifact: zero vocabulary | GEOM field patched | `Inconsistent` | as expected |
| artifact: maximum extent | WGHT extent `u64::MAX` | `Contract` (checked overflow) | as expected |

Plus the deterministic sweeps and the corpus:

* **ONNX, 44-byte model**: 44 truncation prefixes (all refused) and 132
  single-byte flips (all refused, largest allocation 800 B).
* **Artifact**: every truncation prefix refused; 3 masks × every byte position
  rejected — **no single-byte flip produces a valid artifact**.
* **Corpus**: 28 committed byte files, each mapped to its error class (19,182
  bytes; 20,461 bytes on disk including the manifest).

## 4. Allocation-bound proof

`tests/fn3_malformed.rs` installs a counting `GlobalAlloc` (the only `unsafe` in
this change; see §6) and measures the largest single allocation and the total
during each refusal. The counters are reset under a mutex that every test in the
binary takes, so measurements do not mix.

| Refusal | Input size | Largest single allocation |
|---|---|---|
| ONNX `raw_data` claims 4 GiB | 23 B | **136 B** |
| Artifact value table claims 1,048,576 entries | 984 B | **96 B** |
| ONNX graph declares 1,048,577 nodes | 2,097,161 B | 125,829,120 B |

The first two are the important ones: the declared size is multi-gigabyte (or a
million elements) and the refusal costs less than 200 bytes, which is only
possible because the check runs before `Vec::with_capacity`. The third is the
honest worst case of the node budget: a node's encoding is two bytes and its
decoded form is 120 bytes, so a graph at the cap costs ~120 MiB. That is why
`MAX_NODES` is an absolute policy cap rather than a function of the input size;
the test asserts the documented ceiling so removing the cap fails here.

## 5. Fuzz targets

The 2.x tree has no ONNX or artifact target to port: its importer consumed a JSON
blob produced by Python, and `.fnnc` is raw `bincode`. The two 3.0 surfaces are new
and needed new targets. The 2.x harness pattern (bounded input, property asserted
on success, `test = false`) is reused.

| Target | Property |
|---|---|
| `fuzz/fuzz_targets/fn3_onnx_import.rs` | importing never panics; a successful import can be written to an artifact and read back canonically |
| `fuzz/fuzz_targets/fn3_artifact.rs` | `from_bytes` never panics; an accepted artifact re-encodes **byte-identically** (the format's canonicality promise) |

Run with:

```bash
cd fuzz
cargo +nightly fuzz run fn3_onnx_import -- -max_total_time=60 -print_final_stats=1
cargo +nightly fuzz run fn3_artifact     -- -max_total_time=60 -print_final_stats=1
```

All three campaigns ran on the host described in the ledger; libFuzzer drives one
target at a time and no other build was running during a campaign.

| Campaign | Seeds | Runs | Exec/s | Final coverage | Result |
|---|---|---|---|---|---|
| `fn3_artifact` | 15 malformed artifacts | 8,127,453 | 133,236 | `cov: 730`, `ft: 1082` | **no crash**, 61 s, peak RSS 457 MB |
| `fn3_onnx_import` (malformed seeds only) | 11 malformed models | 4,992,401 | 81,842 | not captured (only the `-print_final_stats` tail was kept) | **no crash**, 61 s, peak RSS 559 MB |
| `fn3_onnx_import` (with a valid decoder seed) | 11 malformed + `synthetic_gpt2.onnx` (5,031 B) | 952,233 | 15,610 | `cov: 4333`, `ft: 9197` | **no crash**, 61 s, peak RSS 447 MB |

The third run is the meaningful one for the importer: with only malformed seeds
libFuzzer never assembled a graph that reaches the plan stage, and its dictionary
stayed at protobuf fragments. Seeding one valid 5 KB decoder raised coverage to
`cov: 4333` and taught the fuzzer the real value names (the auto-dictionary
contains `transformer.h.0.attn.c_attn.weight` and `shape_3d`). It found no panic
in 952,233 executions. The short-`steps` panic in §2 was found by reading the
code, and its regression test is deterministic, so the committed corpus keeps
that case even if a future campaign is shorter.

`fuzz/corpus/` is gitignored by the existing `fuzz/.gitignore` (which also
ignores `target`, `artifacts` and `coverage`), so no seed is committed and a
campaign leaves `git status` clean. Reproduce the seed directory before a run
with the two committed sources:

```bash
mkdir -p fuzz/corpus/fn3_onnx_import fuzz/corpus/fn3_artifact
cp tests/corpus/fn3/onnx_*     fuzz/corpus/fn3_onnx_import/
cp tests/corpus/fn3/artifact_* fuzz/corpus/fn3_artifact/
cargo test --test fn3_import write_fuzz_seed -- --ignored
```

The last command regenerates the 5,031-byte valid decoder from the test encoder;
it is the seed that makes the third campaign reach the plan stage.

## 6. Unsafe inventory

**There are no `unsafe` blocks in the owned production paths.**
`src/fn3/import/**` and `src/fn3/model/format/**` contain none; `proto.rs`'s module
docs state the no-`unsafe` property, and a repository-wide grep for `unsafe` in
those two trees returns only that sentence. This is the invariant the #73 brief
asks to record, and it is why the malformed-input work is entirely about logic
and arithmetic rather than about pointer validity.

For the record, `unsafe` in the fn3 tree lives in `src/fn3/storage.rs` (the
aligned buffer), `src/fn3/python/session.rs` (PyO3) and the seven
`src/fn3/backend/vulkan/*.rs` files (ash FFI). None of them was modified by this
wave; they belong to the storage, Python and Vulkan owners. `src/fn3/tensor.rs`
and every file under `src/fn3/import/` and `src/fn3/model/format/` contain no
`unsafe` block at all.

The one new `unsafe` is test-only: `tests/fn3_malformed.rs` implements
`GlobalAlloc` to count allocations. Its `SAFETY:` comment covers the four
required points — it forwards every call to `System` with the caller's own layout
and pointer, adds only atomic counters, never dereferences, frees or moves, and
`alloc_zeroed`/`realloc` forward rather than reimplement. It cannot affect a
release build because the file is an integration test, and it does not change the
`src/fn3/AGENTS.md` invariant ("unsafe is confined to `storage.rs`"), which is
about the shipped tree.

### 6.1 Miri

`cargo +nightly miri` with the `miri` component installed for
`nightly-x86_64-unknown-linux-gnu` (`miri 0.1.0 (65ce8611f)`). Every command used
`MIRIFLAGS=-Zmiri-disable-isolation`, because the corpus tests read files from
`tests/corpus/fn3/` and Miri's default isolation blocks `std::fs`.

| Command | Result | Covered |
|---|---|---|
| `cargo +nightly miri test --lib fn3::import` | **21 passed** | proto reader helpers, error accumulation, the budget unit tests, every `decode_*` unit test |
| `cargo +nightly miri test --lib fn3::model::format` | **28 passed** | wire reader/writer, CRC, artifact header/section/weight unit tests |
| `cargo +nightly miri test --test fn3_import` | **42 passed, 1 ignored** | the synthetic encoder, every new malformed-ONNX case, the budgets, NaN/Inf payloads. The tests that need the 28 MB model **skipped** with their printed reason (nine `SKIP:` lines when the file is absent); they are re-run with the model in §8 |
| `cargo +nightly miri test --test fn3_model_format -- --skip every_single_byte_change_is_detected` | **45 passed, 1 filtered** | all artifact malformed cases: section boundaries, out-of-order/overlapping sections, counts, epsilon, zero vocabulary, maximum extent |
| `cargo +nightly miri test --test fn3_malformed -- --skip a_graph_of_a_million_nodes --skip every_truncation_and_single_byte_flip_of_an_artifact` | **6 passed, 2 filtered** | the 28 corpus classes, both allocation-bound tests, the ONNX 44-byte sweep |

**Not covered, with the reason:**

* `every_truncation_and_single_byte_flip_of_an_artifact` (≈3,000 full decodes) and
  the pre-existing `every_single_byte_change_is_detected` were filtered: under
  Miri the first had run for over eight minutes without finishing, and neither
  exercises `unsafe`. Their subject is the artifact reader's logic, which the 45
  unit/integration cases above do cover.
* `a_graph_of_a_million_nodes_is_refused_by_the_node_budget` allocates ~120 MiB
  and runs a million interpreter steps; it is filtered for runtime and memory.
* Nothing in these paths is `unsafe`, so Miri's value here is proving the *safe*
  code has no UB — no out-of-bounds index, no invalid enum value, no aliasing
  through the borrow of the input slice. It cannot prove an unsafe invariant
  because the owned paths have none; the `GlobalAlloc` in `tests/fn3_malformed.rs`
  is the one new `unsafe` and Miri runs the tests that use it.

One pre-existing test needed a Miri accommodation:
`a_hostile_count_fails_before_it_is_allocated` used a wall-clock bound (<100 ms)
as its proxy for "no allocation was attempted", and under Miri the same call takes
about 0.5 s. Its functional assertions (the `LimitExceeded` variant and the exact
requested/limit values) still run; only the timing guard is compiled out under
`cfg!(miri)`. §4's allocator measurement is the stronger, deterministic proof and
it runs on real hardware.

### 6.2 Other tooling

| Tool | Status | What it would have covered |
|---|---|---|
| ThreadSanitizer | not run | the loaders are single-threaded and take `&[u8]`; there is no shared mutable state to race |
| AddressSanitizer | **run**: `cargo fuzz` builds every target with `-Zsanitizer=address` by default, so all three campaigns were ASan-instrumented | heap/stack overflows in the readers; no report in ~14 M executions |
| LeakSanitizer | available under the same ASan run; no leak is observable from a reader that takes `&[u8]` and returns owned values | leaks |
| Valgrind | not installed for this toolchain; not needed given ASan | uninitialised reads |
| `cargo miri` on the 2.x tree or Vulkan backend | out of scope | other owners' files |

The allocator instrumentation in `tests/fn3_malformed.rs` is not a sanitizer: it
reports the *size* of the largest allocation, which is the property the issue asks
about (a declared multi-GB count), where ASan reports *invalid access*.


## 7. Cross-cutting findings (reported, not edited)

1. **`Shape::validate_permutation` is a useful core check that the planner does
   not apply.** The importer now calls it for `Transpose`, but a plan built from
   a hand-made `GraphNode` can still carry a non-permutation `OpParams.axes`.
   Suggested owner: `src/fn3/model/plan.rs` / `specialize.rs` (planner wave).
2. **`DecoderGeometry` is not validated by its constructor.** The artifact and
   the importer both check it, but `DecoderGeometry` itself is a plain struct, so
   a future caller can build one with `vocab_size = 0` or `heads × head_dim ≠
   hidden`. The `kv_cache_config` arithmetic is checked, but the geometry
   invariants are duplicated in two readers. Suggested owner: `src/fn3/model/`
   (session/model wave).
3. **The ONNX importer has no streaming/incremental path**, so `MAX_MODEL_BYTES`
   cannot prevent the caller from holding an 8 GiB buffer. That is documented in
   `budget.rs` rather than hidden; a future mmap/Slice reader would move the cap
   into the file layer. Not scheduled.
4. **`fuzz/Cargo.lock` was stale**: it still listed `fnv` as a `fastnn`
   dependency after that crate was dropped from the root manifest. `cargo fuzz`
   refreshed it (one package removed). Kept because it is inside this wave's
   `fuzz/**` ownership and the fuzz crate needs a lock consistent with the root.
5. **Runtime hardening is out of scope here.** Device loss, unavailable
   features, repeated `reset`, session error recovery and release-build behaviour
   are the other half of #73 and are a separate follow-up after #72 lands. This
   wave did not touch `session.rs`, `storage.rs`, `tensor.rs`, `backend/**`,
   `cli/**` or `python/**`, and the report makes no claim about them.

## 8. Command results

Every line is an observed result from this worktree, `CARGO_BUILD_JOBS=3`.

| Check | Command | Result |
|---|---|---|
| Formatting | `cargo fmt --check` | clean |
| Library tests | `cargo test --lib` | **676 passed** (673 before this wave; +3 budget unit tests) |
| fn3 library tests | `cargo test --lib fn3` | **169 passed** |
| ONNX import suite | `cargo test --test fn3_import` | **42 passed, 1 ignored** (27 before; +15 new, +1 generator marked `#[ignore]`) |
| Artifact suite | `cargo test --test fn3_model_format` | **46 passed** (38 before; +8 new) |
| Malformed-input suite | `cargo test --test fn3_malformed -- --test-threads=1` | **8 passed** (new file) |
| Clippy, default features | `cargo clippy --all-targets -- -D warnings` | clean, exit 0 |
| Clippy, `fn3` features | `cargo clippy --all-targets --features fn3 -- -D warnings` | clean, exit 0 |
| No-default build | `cargo build --no-default-features` | ok (the `fn3` tree builds without the Python feature) |
| Miri, 5 commands | see §6.1 | 142 test executions passed, 1 ignored, 3 filtered, 0 failures |
| Fuzz, 3 campaigns | see §5 | 14,072,087 executions, 0 crashes |

Measured runtimes: `fn3_malformed` 0.74 s (including the 2 MB node-budget case),
`fn3_import` 0.01 s without the pinned model and 0.23 s with it,
`fn3_model_format` 0.05 s without and 1.13 s with it. Miri: 3.2 s + 22.5 s +
145.2 s + 105.8 s + 37.8 s. Fuzz: 61 s per campaign plus one
sanitizer-instrumented build.

The pinned-model halves of `fn3_import` and `fn3_model_format` were re-run with
`FASTNN_FN3_ONNX_MODEL` pointing at the verified 27,897,974-byte file:
**42 passed / 46 passed**, so the new budgets do not reject the real model and the
pinned inventory, geometry, weight layout, byte round trip and artifact
round-trip assertions all still hold. Without the variable those tests print their
skip reason, which is the pre-existing, documented behaviour of a repository that
does not redistribute the model.

## 9. Limitations

* Miri covers the fn3 importer and format unit tests and the malformed-input
  corpus, not the whole workspace: the 2.x tree and the Vulkan backend need
  GPU hardware and are outside this wave.
* The fuzz campaigns were short (`-max_total_time=60` each), as the brief
  specifies for a smoke run. They prove the targets work and found nothing in
  that window; they do not prove the absence of a bug that needs longer.
* The `DepthExceeded` proto error is not reachable through the ONNX schema (see
  §1.1); it is covered by a unit test, not by the corpus.
* The 8 GiB model cap is asserted by a unit test on the comparison function, not
  by allocating an 8 GiB buffer.
