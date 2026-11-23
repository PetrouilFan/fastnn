# The version-2 prepared-model artifact

- **Status:** implemented
- **Format:** `FNN3ART2`, version 2
- **Reader:** [`src/fn3/model/format/`](../../src/fn3/model/format/)
- **Tests:** [`tests/fn3_artifact_v2.rs`](../../tests/fn3_artifact_v2.rs),
  [`tests/fn3_model_format.rs`](../../tests/fn3_model_format.rs)

One file carries everything a deployment executes: the typed IR, one plan per
target profile, one contiguous memory-mappable weight payload, quantization
metadata and the hashes that prove the bytes. `prepare` is the only step that
needs ONNX. `load` maps the file, validates it and decodes a plan; it never
re-plans and never reads the source model.

There is one format. Version 1 (`FNN3PREP`) was decoder-shaped, stored only
three `OpParams` fields and had no target plans or mapping. It is **not read**:
a v1 file is [`FormatErrorKind::ObsoleteFormat`] and its message says to
re-prepare the source ONNX. There is no compatibility parser and no partial
read.

## Layout

```text
header      128 bytes, fixed
directory   128 bytes, 5 × 24-byte entries
sections    each starts on a 64-byte boundary, zero-padded to one
  GRAPH     typed IR, provenance, interface names
  PLANS     one full plan per target profile
  WEIGHTS   one contiguous 64-byte-aligned payload plus its table
  QUANT     per-tensor scale/zero-point/axis metadata (empty for f32)
  HASHES    SHA-256 of HEADER, GRAPH, PLANS, WEIGHTS and QUANT
```

The payload begins at byte 256 (`HEADER_BYTES + DIRECTORY_BYTES`, already
64-aligned). All integers are little-endian. Strings are a `u32` byte length
followed by UTF-8. Floats are their `f32` bit pattern. Booleans are `u8` 0/1.
Enumerations are `u8` codes; an unknown code is an error, never a default.

### Header (offset 0, 128 bytes)

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic `FNN3ART2` |
| 8 | 2 | format version = 2 |
| 10 | 2 | header length = 128 |
| 12 | 4 | section count = 5 |
| 16 | 4 | target profile count (1..=8) |
| 20 | 4 | flags = 0 |
| 24 | 4 | dtype encoding = 1 (one `DType::code` byte per tensor) |
| 28 | 4 | reserved = 0 |
| 32 | 32 | source ONNX SHA-256 |
| 64 | 8 | file length |
| 72 | 8 | payload length = file length − 128 |
| 80 | 8 | directory offset = 128 |
| 88 | 8 | directory length = 128 |
| 96 | 8 | WEIGHTS section offset |
| 104 | 8 | WEIGHTS section length |
| 112 | 8 | weight payload offset (absolute) |
| 120 | 8 | weight payload length |

Every reserved word is required to be zero. A nonzero one is
`ReservedField { field, value, offset }`, not a silently ignored field.

### Directory (offset 128, 128 bytes)

```text
u32 section count = 5
u32 reserved = 0
5 × { u32 tag, u32 reserved = 0, u64 absolute offset, u64 length }
```

Tags are `1 GRAPH`, `2 PLANS`, `3 WEIGHTS`, `4 QUANT`, `5 HASHES`, in that
order. The reader requires the first section to start at 256, every offset to
equal the 64-byte-aligned end of the previous section, and the last aligned end
to equal the file length. A section length of zero, an offset that is not
64-aligned, an out-of-order tag or a length past the file is a
`MalformedAt`/`Truncated` error naming the section and absolute byte offset.

### GRAPH

Provenance (`ir_version`, opset domain/version, producer, producer version,
graph name), then the typed `Graph`:

- every value: id, optional name, dtype, rank, each extent as
  `DimExpr::Const(u64)` or `DimExpr::Symbol(u8)`, optional producer node;
- every node: id, `OpId` u16, input ids, output ids, the **full** `OpParams`
  (all 28 fields and five enums), `SourceLoc` (source index, source op spelling,
  optional output name);
- the graph interface: input names then output names, in declaration order.

The reader checks that ids equal their table position, ranks and element counts
fit, every node is arity-valid and topological, producers agree in both
directions, and an interface name resolves to exactly one value (and to a value
with no producer for an input).

### PLANS

A target profile count, then one entry per target:

- target name and precision (`f32` = code 0);
- layout policy (`plain` = 0, `chained` = 1);
- arena bytes and peak live bytes;
- one slot per graph value: source kind (`arena` or `weight`), weight id when
  applicable, arena offset and byte length;
- graph interface as slot ids in declaration order;
- every segment: op or fused kind, anchor `OpId`, engine (`scalar`/`onednn`),
  operands, results, full `OpParams`, `SourceLoc`, and for a fused segment its
  activation, residual operand and folded op chain;
- the fusion report;
- layout conversions: segment, slot, direction and byte count, plus the total;
- provider fallbacks: segment, op and reason.

The reader cross-checks the plan against the graph and the weight table before
returning it: one slot per value with the same type and byte count, weight ids
in range with matching dtype and payload length, arena offsets aligned and
inside the arena, interface names in order, operand/result ids in range, fusion
entries that point at fused segments, and a reorder byte sum equal to its list.
A plan is decoded, not rebuilt: `load` does not run the planner. A oneDNN
layout conversion is chosen when an engine opens (it depends on the descriptor
the library picks and on the block split), so the artifact records the layout
policy and the prepare-time reorder list; `Session::from_artifact` may add the
engine's actual conversions to the in-session plan. It never changes the file.

### WEIGHTS

```text
u32 weight count
u32 reserved = 0
u64 payload offset from the section start (64-byte aligned)
u64 payload length
count × { u64 offset, u64 length, u32 dtype, u32 reserved = 0 }
zero padding to a 64-byte boundary
payload; each weight starts on a 64-byte boundary
```

Offsets are relative to the payload start. The table is canonical: the payload
offset equals `align64(table end)`, each entry offset equals
`align64(previous end)`, and the last aligned end equals the declared payload
length. Every length is a whole number of its dtype's elements and inside the
payload. The payload is one contiguous run, so one mapping covers every weight.

### QUANT

`u8 precision, u8 reserved, u16 reserved, u32 tensor count` followed by entries
when a future precision uses them. For f32, the count must be zero: an f32
artifact with quantization entries is `Inconsistent`. The entry format reserves
`tensor id`, `scale`, `zero point` and an optional axis; a reader of a future
precision must bump the format version.

### HASHES

```text
u32 count = 5
u32 reserved = 0
5 × { u32 tag, u32 reserved = 0, 32-byte SHA-256 }
```

The five entries are `0 HEADER`, `1 GRAPH`, `2 PLANS`, `3 WEIGHTS`, `4 QUANT`.
`HASHES` does not hash itself. A hash covers the exact section bytes, excluding
inter-section padding; the padding is separately required to be zero, so a
single flipped padding byte is also rejected. `Artifact::open` hashes the weight
payload from the file in fixed 1 MiB reads **before** it maps, so corruption is
reported before a weight page is faulted in.

## From a mapped weight to `&[f32]`

`Artifact::open` calls `memmap2::MmapOptions::map` (read-only) once. The mapping
is page-aligned; the payload starts 64 bytes into a 64-byte-aligned section, and
every weight starts 64 bytes into the payload. A decoded `ArtifactWeight`
therefore carries an absolute file offset that is 64-byte aligned in the
mapping.

`Artifact::weight_bytes_at(id)` returns
`&mapping[offset .. offset + len]` with no copy. `Artifact::weight_f32(id)`
checks that the dtype is `f32` and the length is a whole number of elements, then
uses `bytemuck::try_cast_slice` (which re-checks length and alignment). There is
no re-alignment, repacking or staging buffer: the returned slice borrows the
mapping, which the `Artifact` owns through an `Arc<Mmap>`. Every session opened
on the artifact holds an `Arc<Artifact>`, so the mapping outlives every weight
slice the session hands to a kernel.

The unsafe operations are:

- `MmapOptions::map`: the file is opened read-only, the mapping is never written
  through, and `Artifact` keeps the `Arc<Mmap>` alive for every derived slice;
- the `&[f32]` cast: the table proves the byte length is a multiple of four, the
  mapping and payload offsets prove 64-byte alignment, the slice is immutable,
  and the borrow is tied to `&self`/the owning `Arc`.

`Artifact::contains_address(ptr, len)` answers whether a pointer lies inside the
mapping; the sharing test uses it to prove that two sessions use the same
address for the same weight.

## `prepare` / `load` / `run` / `inspect` / `doctor`

```text
prepare(onnx, artifact, target, precision)  import -> plan -> serialize; atomic write
load(artifact, threading)                   mmap -> hashes -> stored plan -> Session
run(session, inputs)                        Session::run (existing segment executor)
inspect(artifact)                           header/graph/plans/hashes; no weight payload
doctor(onnx)                                import + support + memory; no execution
```

`target` is `cpu` (the default: oneDNN where the build has it, fused), 
`cpu-reference` (scalar, one segment per node, no fusion) or `all` (both, in
that order; the first is the default plan).

`inspect` reads the header, directory, `GRAPH`, `PLANS`, `QUANT`, `HASHES` and
the 64-aligned `WEIGHTS` table prefix only. It constructs no `Mmap`, reads no
weight byte and reports `mapped_bytes: 0` plus `bytes_read` as the evidence; its
`WEIGHTS` hash is printed as `stored (payload not read)` rather than claimed as
verified. `Artifact::open` (the `load` path) still verifies every hash,
including the payload.

`doctor` returns a report even when import fails: every independent
`ImportProblem` becomes a finding with the source node rendered as
`node 7 (onnx "Conv", output "y")`. It also reports dynamic symbols (the typed
plan requires static extents), non-f32 graph inputs, backend support verdicts
and the expected resident memory (`weights + arena`, with peak live bytes), all
before any artifact is written or any segment executes.

## Budgets

| Constant | Value | Applies to |
|---|---:|---|
| `MAX_PAYLOAD_BYTES` | 8 GiB | `payload length`, artifact size |
| `MAX_VALUES` | 2^20 | value table, graph inputs/outputs, quant entries |
| `MAX_NODES` | 2^20 | node table, plan segments, reorders, fusions, fallbacks |
| `MAX_WEIGHTS` | 2^20 | weight table |
| `MAX_RANK` | 64 | every tensor type in `GRAPH` |
| `MAX_PARAM_AXES` | 1024 | every `OpParams` vector, fused chain |
| `MAX_TARGETS` | 8 | target profiles |
| `MAX_INSPECT_SECTION_BYTES` | 256 MiB | `inspect` reading `GRAPH`/`PLANS`/`QUANT` |

Every count is checked against its cap and the remaining bytes before a `Vec`
is allocated. Every offset addition is `checked_add`. A malformed file returns
`FormatErrorKind::MalformedAt { section, offset, detail }`,
`Truncated { what, needed, available, offset }`,
`LengthMismatch { what, declared, actual, offset }`,
`HashMismatch { section, stored, computed }`, `LimitExceeded`, or
`Inconsistent`, never a panic and never a read past the mapping.

## Determinism and versioning

Tables keep declaration order; there is no `HashMap` in the byte stream; padding
is zeros and is required to be zeros on read. `to_bytes` twice is byte-identical,
and `from_bytes` → `to_bytes` is byte-identical for a decoded artifact.

A change that adds a section, reorders a field, changes a meaning or assigns a
new code **must** bump `FORMAT_VERSION`. A v1 file and a future version are both
refused before any structure is interpreted. There is no in-place compatibility
and no best-effort partial read.

## Measured load cost

A synthetic artifact with one 200,000,000-byte weight, prepared by
`tests/fn3_artifact_v2.rs` on this machine (Linux, debug test binary; `VmRSS`
from `/proc/self/status`):

| Measurement | Bytes |
|---|---:|
| artifact file | 200,001,344 |
| weight payload | 200,000,000 |
| RSS before `Artifact::open` | 7,102,464 |
| RSS after `open` | 8,318,976 |
| RSS after two sessions | 8,318,976 |
| growth | 1,216,512 (0.6% of the payload) |

Two sessions share the same weight address; the assertion is growth
`< weight_bytes / 4`, and the printed numbers above are the evidence, not a
claim derived from the code.
