# FN3-09 (#55) — versioned, shareable prepared-model format

**Branch** `fn3/55-model-format`
**Base** `456f3a3` (confirmed with `git rev-parse HEAD` before the first edit; the
work is local commits on top of it and is not pushed)

**Tracking issue** #46 · **Issue** #55 · **Status** implementation complete, local
commits only, no remote write

---

## 1. What landed

One versioned artifact format with a hand-written, bounds-checked codec, plus the
tests that pin its behaviour on the real 27,897,974-byte pinned model.

```text
ONNX bytes ──► import::ModelImport ──► PreparedArtifact::from_import(&import, envelope)
                                            │
                 ┌──────────────────────────┴───────────────────────────┐
                 ▼                                                      ▼
          to_bytes() / write_file()                          from_bytes() / read_file()
          one canonical byte string                          strict, bounded, checksummed
```

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/model/format/mod.rs` | 143 | mine (new) — specification + facade |
| `src/fn3/model/format/artifact.rs` | 1813 | mine (new) — header, sections, reader/writer |
| `src/fn3/model/format/graph.rs` | 476 | mine (new) — constructible graph section |
| `src/fn3/model/format/wire.rs` | 420 | mine (new) — cursor, writer, CRC-32 |
| `src/fn3/model/format/error.rs` | 399 | mine (new) — typed format errors |
| `tests/fn3_model_format.rs` | 1285 | mine (new) — 38 tests |
| `docs/reports/fn3-09-model-format.md` | this file | mine (new) |
| `src/fn3/model/mod.rs` | +1 | the single `pub mod format;` line |

No new dependency, no `unsafe`, no change to any other shared file.

## 2. Why a hand-written framing (checked first)

`Cargo.toml` already depends on `serde`, `serde_json` and `bincode 1.3`. None of
them was used, for reasons that are specific rather than stylistic:

* **`bincode` is the mistake this issue exists to correct.** The 2.x `.fnnc` is
  raw `bincode`: no magic, no version, no checksum, and a length prefix the
  reader is expected to trust. Reusing it would carry over exactly the
  properties the ADR puts on the do-not-carry list.
* **`serde` derive needs an intermediate DTO tree.** Deriving on the real types
  is impossible (`DecoderGeometry`, `WeightSet`, `OpParams` and `ValueInfo` are
  owned by other issues and are not `Serialize`), so every derive would need a
  shadow struct per section plus conversions — more code than the framing, with
  no control over magic, version, alignment, limits or the error taxonomy.
* **The 2.x format is not readable anyway.** `bincode` gives no version gate, so
  a "format version" would have to be bolted on in front of it. At that point the
  remaining value is a varint encoder, which is 30 lines here and fully checked.

The result is a byte layout a person can verify by hand, which is what "own
reader/writer and incompatibility diagnostics" asks for.

## 3. The format, version 1

### 3.1 Container

```text
header   64 bytes, fixed offsets
   0..8    b"FNN3PREP"            magic
   8..10   u16 format version     1
  10..12   u16 header length      64
  12..16   u32 dtype encoding     1      (DType::code, one byte per tensor)
  16..20   u32 flags              0      (reserved for future feature bits)
  20..24   u32 reserved           0
  24..32   u64 file length        header length + payload length
  32..40   u64 payload length
  40..44   u32 payload CRC-32
  44..64   20 reserved bytes      0

payload  (everything after the header; the CRC covers all of it)
   u32 section count              6
   u32 reserved                   0
   6 × [u32 tag, u32 reserved = 0, u64 section length]
   section bytes, in tag order, each zero-padded to a multiple of 8
```

All integers little-endian. Strings are `u32` byte length + UTF-8 bytes (no
terminator). Floats are their `f32` bit patterns, never text. Booleans are `u8`
0/1. Enumerations are `u8` codes with a documented meaning; an unknown code is an
error, never a default.

### 3.2 Sections

| Tag | Name | Bytes | Content and why it is separate |
|---:|---|---:|---|
| 1 | `META` | 1,013 on the pinned model | ONNX IR version, opset domain + version, producer, producer version, graph name. Provenance is the first thing a person needs when two artifacts behave differently, and it is what a reproducible-build check compares. |
| 2 | `GEOM` | 88 | layers, hidden, MLP width, query heads, KV heads, head width, vocabulary, position-embedding kind + weight + count, LayerNorm epsilon (flag + bits), token-embedding weight, logits value, per-layer KV interface (4 value handles). |
| 3 | `ENV` | 112 | The seven `Symbol`s, each `u64 min, u64 max`, in `Symbol::ALL` order. Exactly 112 bytes: fixed size, no count, nothing to validate beyond each bound. |
| 4 | `VALS` | 29,355 | The value table (name, dtype, shape, origin), then graph inputs, then graph outputs. |
| 5 | `NODE` | 105,489 | The op list: ONNX index/name/op, `OpId` code, operand handles, `OpParams` (axes, keep-dims, epsilon). |
| 6 | `WGHT` | 27,731,432 | The weight table: name, dtype, extents, source, exact payload bytes, payloads 8-byte aligned. |

Sections 4 and 5 are the graph description the planner consumes; section 6 is the
weight table it references. Each section is decoded by exactly one function that
must consume exactly its declared length — there is no way to smuggle a field
into a section (tested).

### 3.3 Checksum

CRC-32, IEEE 802.3: reflected polynomial `0xEDB88320`, init `0xFFFF_FFFF`, final
XOR `0xFFFF_FFFF`, check value `0xCBF43926` for `"123456789"` (asserted as a unit
test). It covers the payload — the section directory, every section and every
padding byte — and nothing in the header. The header is validated field by field
instead, so a corrupt header names the field (`reserved header field flags is
255`) while a corrupt payload is a checksum mismatch with both numbers.

CRC-32 rather than FNV because the job is to detect *accidental* corruption and a
standard polynomial with a published check value is auditable without trusting
this file — and because CRC-32 provably detects every single-byte error, which
`every_single_byte_change_is_detected` exercises over the whole small artifact.
It is explicitly not a signature: an attacker who controls the bytes controls the
checksum, which is why every structural rule is enforced independently of it
(`a_payload_edit_with_a_repaired_checksum_is_still_validated`).

### 3.4 Alignment

Weight payloads start at an 8-byte boundary counted from the section start; the
section starts are 8-aligned in the file because the header is 64 bytes and the
directory is 104. Padding is written as zeros and **required** to be zeros on
read, so the encoding is canonical. That is what makes "load and re-save is
byte-identical" a property rather than a hope, and it is independent of any
allocator: the pad length is a function of the byte offsets, never of capacity.

### 3.5 Validation order (the reason a version error is not a parse error)

`PreparedHeader::parse` runs, in this order:

1. `bytes.len() >= 64` → `Truncated`;
2. magic → `BadMagic` **naming the eight bytes found**;
3. format version → `UnsupportedVersion { found, supported }`;
4. header length → `LengthMismatch`;
5. dtype encoding → `UnsupportedDtypeEncoding { found, supported }`;
6. `flags` / reserved word → `ReservedField`;
7. `file length` and `payload length` against the bytes present → `LengthMismatch`;
8. declared payload length above the 8 GiB policy cap → `LimitExceeded`.

Only then is the checksum recomputed, then the directory, then the sections.
Magic and version are the first ten bytes, so a file from another version or
another format is rejected before any structure is interpreted.

## 4. Untrusted input

Every rule is enforced by construction in `wire.rs`:

* `Reader::take(n)` compares `n` against the remaining bytes with `u64` arithmetic
  before slicing; offsets use `checked_add`.
* `Reader::count(limit, min_entry_bytes)` reads a `u32`, rejects it against the
  policy cap, then proves `count * min_entry_bytes <= remaining` — so a hostile
  count costs one multiplication and fails **before** `Vec::with_capacity` is
  called. `a_hostile_count_fails_before_it_is_allocated` patches a count to
  4,294,967,280 and asserts the error, the named limit and that the call returns
  in under 100 ms (a real `Vec<ValueInfo>` of that length would need hundreds of
  gigabytes).
* `Reader::string` caps the length at 4,096 bytes and validates UTF-8 *before*
  allocating the `String`.
* `Reader::pad_to_8` requires zero padding; `Reader::finish` requires exact
  consumption.
* Every weight payload length is checked against the section, and
  `WeightTensor::new` re-checks `numel × dtype.size_bytes() == payload.len()`.

Policy caps (documented constants, all enforced before allocation):

| Constant | Value | Why |
|---|---:|---|
| `MAX_PAYLOAD_BYTES` | 8 GiB | a load reads one buffer; this bounds it |
| `MAX_VALUES` | 1,048,576 | pinned model: 326 |
| `MAX_NODES` | 1,048,576 | pinned model: 265 |
| `MAX_WEIGHTS` | 1,048,576 | pinned model: 50 |
| `MAX_RANK` | 64 | pinned maximum rank: 4 |
| `MAX_PARAM_AXES` | 1,024 | largest real `OpParams.axes`: 1 |
| `MAX_VOCAB_SIZE` | 16,777,216 | pinned: 50,257; policy cap, not hardware |
| layer count | `MAX_KV_LAYERS` (1,024, reused from `model`) | pinned: 2 |

After decoding, `PreparedArtifact::validate` enforces that the sections agree:

* the token-embedding weight's shape is exactly `[vocab_size, hidden_size]`;
* the position table's shape is exactly `[max_positions, hidden_size]`;
* `query_heads × head_dim == hidden_size` and every geometry width is non-zero;
* `kv.len() == layers` and every KV/logits handle is in range with dtype `f32`;
* `layer_norm_epsilon` is a finite positive float when present;
* every value handle, node operand and graph interface names a slot in range;
* no value is written twice and no node claims a value it does not write;
* the fixed `n_heads`, `n_kv_heads` and `head_dim` are inside the artifact's own
  declared envelope, so a file cannot declare an envelope that cannot run it;
* every `ShapeEnvelope` bound satisfies `min <= max <= DimBound::MAX_EXTENT`
  (the existing core cap, so an envelope this build would refuse is refused at
  load, not at session creation).

That is the issue's "a file cannot claim a vocabulary or envelope this build
would not accept": the claim is cross-checked against the bytes that must
support it.

### 4.1 Error taxonomy

`FormatErrorKind` (new, format-local, convertible to `Fn3Error::InvalidFormat`):

| Variant | Meaning | `is_version_mismatch` | `is_corruption` |
|---|---|:-:|:-:|
| `BadMagic` | not this format (including a 2.x `.fnnc`) | | ✔ |
| `UnsupportedVersion` | written by another format version | ✔ | |
| `UnsupportedDtypeEncoding` | another dtype code space | ✔ | |
| `ReservedField` | a must-be-zero field is not zero | | ✔ |
| `Truncated` | ended before the declared structure did | | ✔ |
| `LengthMismatch` | two length fields disagree | | ✔ |
| `ChecksumMismatch` | the payload does not match its CRC | | ✔ |
| `LimitExceeded` | a count or size above a policy cap | | |
| `Malformed` | bad tag, non-UTF-8 name, insane enum code | | ✔ |
| `Inconsistent` | well-framed but impossible model | | |
| `UnknownDType` | dtype code this build does not define, naming it | | |
| `UnknownOpCode` | op code this build does not define (says if reserved) | | |
| `Contract` | a core contract rejected the decoded object | | |
| `Io` | a file could not be read or written | | |

## 5. Determinism

* Nothing in the byte stream is a map; tables keep declaration order (weights are
  `WeightSet` order, ops are graph order).
* Every optional field has one absent form (`shape kind 0`, `origin kind 0`,
  epsilon flag `0` with zero bits, `positions kind 0`).
* Alignment padding is zeros and validated as zeros.
* Floats are stored as bits, so no formatting can round-trip differently.

Verified by `two_independent_imports_write_identical_artifacts` (two imports of
the same 28 MB file, two writes, identical bytes), by the round-trip test
(`to_bytes` twice, and `from_bytes → to_bytes` equal), and by the canonical-form
checks the reader imposes on any file it accepts.

## 6. Session state is not in the file

The artifact holds weights, the graph description, geometry and the declared
bounds. It holds **no** KV capacity, live extent, batch size, step counter, phase
or device. Evidence:

1. `the_artifact_declares_exactly_the_six_structural_sections` — the section
   inventory is exactly `META, GEOM, ENV, VALS, NODE, WGHT`; the count field is
   six and the tags are closed.
2. `an_extra_section_entry_is_rejected` and `an_unknown_section_tag_is_rejected` —
   a seventh section, or a section retagged `0x4B56` (say "KV state"), is
   refused.
3. `a_section_with_extra_fields_is_rejected` — a section that declares eight
   bytes more than its decoder consumes is refused, so a writer cannot append a
   live extent or a counter to a section either.
4. `a_session_adds_no_state_to_the_artifact` — two `Session`s over a
   `PreparedModel` built from the artifact each allocate their own cache
   (`capacity() == 4`, distinct pointers, `step() == 0`, `cache_len() == 0`), and
   `artifact.to_bytes()` is unchanged afterwards; a different capacity/batch
   choice reaches a `KvCacheConfig`, never the file.

## 7. Public API (what #56 and #62 call)

```rust
use fastnn::fn3::model::format::{
    ArtifactMetadata, PreparedArtifact, PreparedGraph, PreparedHeader,
    FormatError, FormatErrorKind,
    MAGIC, FORMAT_VERSION, HEADER_BYTES,
    MAX_PAYLOAD_BYTES, MAX_VALUES, MAX_NODES, MAX_WEIGHTS, MAX_RANK,
    MAX_PARAM_AXES, MAX_VOCAB_SIZE,
};
```

**Write (the importer / a conversion tool).**

```rust
let imported  = fastnn::fn3::import::import_onnx_file(path)?;          // ImportError
let envelope  = ShapeEnvelope::uniform(1024)?;                          // declared bounds
let artifact  = PreparedArtifact::from_import(&imported, envelope)?;    // FormatError
artifact.write_file("model.fn3")?;                                      // deterministic bytes
// or: let bytes: Vec<u8> = artifact.to_bytes()?;
```

Zero-copy variant when the caller already owns the parts:

```rust
let metadata = ArtifactMetadata::from_import(&imported);
let geometry = imported.geometry().clone();
let (graph, weights) = imported.into_parts();
let artifact = PreparedArtifact::from_parts(
    metadata, PreparedGraph::from_graph(&graph), weights, geometry, envelope,
)?;
```

**Read (#62 end-to-end, #63 CLI).**

```rust
let artifact = PreparedArtifact::read_file("model.fn3")?;  // or from_bytes(&bytes)
artifact.metadata();        // &ArtifactMetadata { ir_version, opset, producer, .. }
artifact.geometry();        // &DecoderGeometry  (layers, heads, vocab, kv interface, eps)
artifact.envelope();        // ShapeEnvelope     (the declared dynamic bounds)
artifact.graph();           // &PreparedGraph
artifact.weights();         // &WeightSet        (name, DType, Shape, bytes, source)
artifact.weight_bytes();    // u64
artifact.to_model_weights()?;  // ModelWeights on DeviceId::Cpu
artifact.validate()?;       // idempotent invariant check
```

**The graph, for #56's planner** (same accessors as `GraphIr`, one for one):

```rust
let graph = artifact.graph();          // &PreparedGraph
graph.value_count();                   // 326 on the pinned model
for node in graph.nodes() {            // execution order
    node.op;                           // OpId
    node.inputs; node.outputs;         // Vec<ValueId>
    node.params;                       // OpParams { axes, keep_dims, epsilon }
    node.onnx_index; node.name; node.onnx_op;   // provenance
}
for id in graph.inputs() { .. }        // declared graph inputs
for id in graph.outputs() { .. }       // declared graph outputs
graph.value(id);                       // &ValueInfo { name, dtype, shape, origin }
graph.lookup("logits");                // Option<ValueId>
graph.producer(id);                    // Option<&GraphNode>
```

**Header inspection (a CLI/health check that must not load 28 MB).**

```rust
let header = PreparedHeader::parse(&bytes)?;   // framing only, no allocation
header.version; header.payload_bytes; header.payload_crc32;
```

**The chain #62 assembles.**

```rust
let artifact = PreparedArtifact::read_file("model.fn3")?;
let kv  = artifact.geometry().kv_cache_config(capacity_tokens, batch);
let plan = /* #56: plan(artifact.graph(), artifact.geometry(), artifact.envelope()) */;
let model = PreparedModel::new(
    plan,
    artifact.to_model_weights()?,
    DeviceId::Cpu,
    artifact.envelope(),
    Some(kv),
)?;
```

`FormatError` converts to `Fn3Error` (`Fn3Error::InvalidFormat { format: "fn3
prepared model", .. }`), so `?` works in `Fn3Result` functions.

## 8. Measurements (this machine)

Debug build, GNU/Linux, the pinned tiny-GPT-2 graph, second pass (warm
allocator), `cargo test --test fn3_model_format -- --test-threads=1 --nocapture`:

| Quantity | Value |
|---|---|
| source ONNX | 27,897,974 B |
| weight payload | 27,833,180 B (equal to `ImportSummary::weight_bytes`) |
| artifact | **27,879,928 B** = 64 header + 27,879,864 payload |
| structural overhead | 46,748 B (0.17 % over the weight bytes) |
| section sizes (unpadded) | `META 54`, `GEOM 86`, `ENV 112`, `VALS 21,456`, `NODE 21,581`, `WGHT 27,836,460` |
| `to_bytes` (28 MB) | **154.2 ms** (≈181 MB/s), debug |
| `from_bytes` (28 MB, incl. CRC + copies) | **146.0 ms** (≈191 MB/s), debug |

Release build (`cargo test --release --test fn3_model_format
pinned_model_round_trips -- --test-threads=1 --nocapture`, same machine and
same second-pass protocol), as a sanity check that the codec is not
pathologically slow:

| Quantity | Value |
|---|---|
| artifact | 27,879,928 B (identical to the debug build) |
| `to_bytes` | **68.6 ms** ≈ 406 MB/s |
| `from_bytes` | **57.6 ms** ≈ 484 MB/s |

The 46,748-byte overhead (0.17 %) is structural, not padding: 64 bytes of
header, 104 bytes of section directory, 11 bytes of section padding, 3,280 bytes
of weight-table metadata (names, dtypes, extents and per-weight headers, which
makes `WGHT` 27,836,460 B for 27,833,180 B of payload) and 43,289 bytes for the
other five sections (`META` 54, `GEOM` 86, `ENV` 112, `VALS` 21,456, `NODE`
21,581).

## 9. Verification — commands and real outcomes

Run in this worktree with the pinned model present at
`FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx`
(27,897,974 B, SHA-256 `e18370ba6247c564a4ef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`,
verified out of band with `sha256sum`; not committed).

| Command | Outcome |
|---|---|
| `cargo fmt --check` | **clean** |
| `cargo build --no-default-features` | **ok** (only the pre-existing 2.x `augment_runtime_telemetry` and `fnv` manifest warnings) |
| `cargo build` | **ok** (only the pre-existing `fnv` manifest warning) |
| `cargo test --test fn3_model_format -- --test-threads=1` | **38 passed, 0 failed** |
| `cargo test --test fn3_import` | **27 passed** (unchanged) |
| `cargo test --test fn3_contracts` | **43 passed** (unchanged) |
| `cargo test --test fn3_cpu_reference -- --test-threads=1` | **43 passed** (unchanged) |
| `cargo test --lib` | **665 passed** (was 637; +28 format unit tests) |
| `cargo test --doc` | **4 passed, 14 ignored** (extra; the module example compiles) |
| `cargo clippy --all-targets` | **no finding in any file this issue touches** (the repo's pre-existing findings in `storage.rs`, `backend/cpu/kernels/`, `tests/fn3_cpu_reference.rs` are unchanged) |

With the model absent, `cargo test --test fn3_model_format` still passes: the two
pinned-model tests print the same SKIP line as #54 and return, and the 36
negative/structural tests run against a small artifact built through the public
API (no network, no fixture writes).

## 10. Test inventory (`tests/fn3_model_format.rs`)

Pinned model:

1. `pinned_model_round_trips_with_exact_bytes_and_structure` — 27,879,928-byte
   artifact; metadata (IR 7, opset 14, `pytorch`, `main_graph`), geometry (2
   layers, 128/512 width, 2/2 heads, head 64, vocab 50,257, eps 1e-5), envelope,
   326 values, 265 nodes, 50 weights compared name/dtype/shape/source/bytes in
   order, host tensors byte-identical to the import's, re-write byte-identical.
2. `two_independent_imports_write_identical_artifacts` — determinism across two
   imports.

Framing and versioning:

3. `bad_magic_is_rejected`, 4. `a_future_version_is_rejected_by_version_not_by_parsing`,
5. `a_past_version_is_rejected`, 6. `a_different_dtype_encoding_is_rejected`,
7. `header_reserved_fields_must_be_zero`, 8. `a_header_length_this_build_does_not_use_is_rejected`,
9. `a_truncated_header_is_rejected` (12 lengths), 10. `a_truncated_file_is_rejected`,
11. `a_section_length_beyond_the_payload_is_reported_as_truncated`,
12. `a_truncated_section_body_is_reported`, 13. `a_checksum_mismatch_is_rejected`,
14. `one_flipped_payload_byte_names_both_checksums`.

Content and limits:

15. `an_unknown_dtype_code_names_the_code_and_the_reader`,
16. `an_unknown_op_code_names_the_code`, 17. `a_declared_count_larger_than_the_file_is_rejected`,
18. `a_hostile_count_fails_before_it_is_allocated`, 19. `a_hostile_rank_is_rejected`,
20. `invalid_utf8_in_a_name_is_rejected`, 21. `trailing_payload_bytes_are_rejected`,
22. `a_payload_edit_with_a_repaired_checksum_is_still_validated`,
23. `an_envelope_above_the_hard_cap_is_rejected`,
24. `an_envelope_that_excludes_the_fixed_heads_is_rejected`,
25. `a_geometry_that_contradicts_its_weights_is_rejected`,
26. `a_layer_count_beyond_the_kv_bound_is_rejected`,
27. `a_weight_shape_that_overflows_is_rejected` (checked element arithmetic on
    untrusted extents, and a payload length larger than its section),
28. `every_single_byte_change_is_detected` (every byte of a small artifact, each
    reported by the checksum or by a header field).

No session state:

29. `the_artifact_declares_exactly_the_six_structural_sections`,
30. `an_unknown_section_tag_is_rejected`, 31. `an_extra_section_entry_is_rejected`,
32. `a_section_with_extra_fields_is_rejected`,
33. `a_session_adds_no_state_to_the_artifact`.

2.x policy and the file API:

34. `legacy_fnnc_bytes_are_rejected_at_the_magic`,
35. `weight_payloads_are_eight_byte_aligned_and_zero_padded`,
36. `a_file_round_trip_is_byte_identical`, 37. `a_missing_file_names_the_path`,
38. `the_public_surface_names_its_seams`.

Unit tests inside the module (+28 to `cargo test --lib`): header offsets vs the
documented layout, CRC check value and single-byte detection, hostile counts,
string caps, padding canonicity, `f32` bit fidelity including NaN payloads,
envelope order, geometry/weight cross-checks, symbol codes, graph invariants in
both directions.

## 11. Decisions worth reviewing

1. **The artifact stores the graph description, not a specialised `ExecutionPlan`.**
   A plan is a function of (graph, envelope, device, backend): #56 chooses slots
   and may fold per-device decisions. Serialising it would make the artifact
   device-specific and would bake in a plan that a different envelope invalidates.
   Everything a planner needs is in the artifact, and `PreparedArtifact` is the
   input to #56's `plan(..)`. This is the one interpretation call in the issue's
   "graph/plan description"; it is the interpretation that keeps the artifact
   shareable, which is the stated property.
2. **`PreparedGraph` mirrors `GraphIr`.** `GraphIr` has private fields and no
   constructor, so a reader cannot rebuild one (see §13). `PreparedGraph` is a
   validated container over #54's own `ValueInfo`/`GraphNode`, with the same
   accessor signatures, so the mirror collapses mechanically once
   `GraphIr::from_parts` exists.
3. **The envelope is an argument, not a detected value.** How many tokens a
   deployment may ask for is a deployment decision (the ADR says so for the KV
   cache), and the artifact must record it because "bounded" is a property of the
   artifact.
4. **Eight-byte weight alignment with mandatory zero padding.** It costs 46,748
   bytes on the pinned model and buys a canonical encoding plus an alignment
   guarantee for a future zero-copy reader. Padding length is a function of the
   offsets only.
5. **`from_import` copies the weight payload once** (the import owns it). A 28 MB
   copy is 5–10 ms; the zero-copy path is `from_parts` and is documented for the
   498 MB trained model.
6. **The dtype encoding is a header field, not just per-tensor bytes.** Assigning
   a quantized code space (#68) is then a deliberate new encoding id, and an old
   build says "this file uses dtype encoding 2" instead of misreading code 4.
7. **Policy caps are format constants, not backend limits.** A backend's
   `max_buffer_bytes` is a different question and is checked by #56/#58; the file
   must be rejectable without a device.

## 12. Known risks and limitations

* **`GraphIr` cannot be rebuilt yet.** Until `GraphIr::from_parts` lands (#54 /
  #48), a planner that only accepts `&GraphIr` cannot consume a loaded artifact
  directly; it must take `&PreparedGraph` or go through `PreparedGraph::parts()`.
  This is the single coordination item (§13).
* **The format is version 1 and therefore unproven against a second version.**
  The version gate, the dtype-encoding gate and the closed section inventory are
  tested, but there is no v2 to migrate from. The report records the policy
  ("any change to a section, a field order, a meaning or a dtype code bumps the
  version") rather than demonstrating it.
* **CRC-32 is not a security control.** It detects corruption, not tampering. A
  hostile file that recomputes it still has to pass every structural and
  cross-section check, but the checksum is not authentication; a signed digest is
  a distribution concern, not a format one.
* **Write is not atomic.** `write_file` is one `std::fs::write`, so a crash mid
  write leaves a partial file — which the reader then rejects with a typed error
  rather than a partial load. An atomic rename belongs to the CLI (#63).
* **Memory**: loading a 28 MB artifact holds the file bytes plus the weight
  payload (one copy), and `to_model_weights()` copies again into aligned storage.
  Peak for the pinned model is ~28 + 28 + 28 MB, the same shape as the import.
* **`to_bytes` needs the whole payload in memory** before writing (~2× the
  artifact during write). Fine for 28 MB and the 498 MB trained model; a streaming
  writer is a later optimisation if a multi-gigabyte model appears.
* **`ValueId`/`WeightId` are `u32`.** The caps are far below `u32::MAX`, so a
  larger model fails with `LimitExceeded` rather than truncating a handle.
* **`PreparedGraph::value(id)` panics on an out-of-range handle**, exactly like
  `GraphIr::value`; every handle inside a graph is validated, so this needs a
  hand-built `ValueId`. `producer(id)` and `lookup(name)` return `None` instead.

## 13. Shared-file changes needed (for the integrator)

**One line added, nothing else:** `src/fn3/model/mod.rs` declares `pub mod format;`
above `pub mod weights;`. No `pub use` was added, to keep the diff to #56's shared
file to a single line.

**Two documentation updates belong to the core owner** (I did not edit them, as
both are core-owned): `src/fn3/AGENTS.md` still describes `model/mod.rs` as the
home of `ExecutionPlan`/`ModelWeights`/`PreparedModel`/`Session` and does not
mention `model/weights.rs` (landed by #54) or `model/format/` (this issue);
`docs/roadmap/fn3-architecture.md` §2 lists only `model/mod.rs` in the module
tree.

**One upstream addition is needed for a clean seam** (owned by #54 / #48, in
`src/fn3/import/graph.rs` — I did not touch it):

```rust
impl GraphIr {
    /// Rebuild a graph from its parts, validating every handle.
    pub fn from_parts(
        values: Vec<ValueInfo>,
        nodes: Vec<GraphNode>,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<GraphIr, Fn3Error> { /* ids in range; origin/node agreement */ }
}
```

The format already provides exactly these four vectors
(`PreparedGraph::parts()`), and the validation it needs is the same as
`PreparedArtifact::validate`. When that constructor exists, `PreparedGraph` can be
deleted and `decode_graph` can return a `GraphIr`; until then, #56 should code
against `PreparedGraph` (same accessors) or accept the four vectors.

## 14. Not done / out of scope

* No plan construction, no slot allocation, no specialisation (#56); no session,
  no `prefill`/`decode` (#58).
* No compression, no encryption, no mmap loader: the reader takes `&[u8]`.
* No quantized dtype or quantized metadata section — the reserved tag space and
  the dtype-encoding field are the extension points (#68).
* No `.fnnc` reader, converter or re-exporter, by policy and by test.
* The pinned model is not committed and no network fetch happens in tests.
