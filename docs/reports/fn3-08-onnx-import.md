# FN3-08 (#54) — strict ONNX import and typed weights

**Branch** `fn3/54-onnx-import-b`
**Base** `27fc60d` (verified with `git rev-parse HEAD` before any edit); the work
is one local commit on top of it and is not pushed

**Tracking issue** #46 · **Issue** #54 · **Status** implementation complete, local
commits only, no remote write

---

## 1. What landed

A self-contained ONNX importer for the 3.0 core: a bounds-checked protobuf wire
reader, an operator mapping into `OpId`s, a symbolic static-analysis pass, a typed
weight table, and architecture detection — with no new dependency, no `unsafe`,
and no Python at runtime.

```text
bytes ──► proto.rs   bounds-checked protobuf wire reader (no dependency)
      ──► graph.rs   ONNX node → OpId mapping, attributes, dtype/shape inference
      ──► weights.rs name + DType + Shape + exact payload bytes
      ──► ModelImport { graph, weights, geometry, summary, warnings }
```

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/import/mod.rs` | 171 | mine (new) |
| `src/fn3/import/error.rs` | 825 | mine (new) |
| `src/fn3/import/proto.rs` | 1367 | mine (new) |
| `src/fn3/import/graph.rs` | 4365 | mine (new) |
| `src/fn3/model/weights.rs` | 426 | mine (new, reserved by the ADR) |
| `tests/fn3_import.rs` | 1418 | mine (new) |
| `docs/reports/fn3-08-onnx-import.md` | 405 | mine (new) |
| `src/fn3/mod.rs` | +1 | the single `pub mod import;` line |

Total: 8,572 new lines of code and tests plus this report, 1 line changed in a
shared file.

## 2. Why not a port

`src/onnx/` has no decoder: it consumes a JSON blob produced by the Python `onnx`
package (`fastnn/io/onnx.py`). That path is on the 2.x "do not carry" list and
3.0 must not require Python at runtime, so this is a new wire-format reader, not
a port. The inventory of `src/onnx/converter.rs` was read to make sure no
*behaviour* was lost (typed constants, symbolic input names, bounded shape
inputs); none of its code or its fallback semantics were reused.

## 3. Parser and IR architecture

### 3.1 Wire reader (`proto.rs`, private)

* `Reader<'a>` over one message: `bytes`/`pos`/`base`/`depth`.
* Every read is checked: `varint` (≤ 10 bytes), `tag` (field 0 rejected), `take`
  (`checked_add` + `get`), `bytes` (length prefix validated against the remaining
  bytes *before* slicing), `sub` (depth limit `MAX_DEPTH = 8`).
* Unknown fields are skipped by wire type 0/1/2/5. Groups (3/4/6/7) are a typed
  error: ONNX never emits them, so a group means the bytes are not an ONNX model.
* Known fields with the wrong wire type are errors, not skips.
* Two payload fields in one `AttributeProto` is `ConflictingPayload`.
* Nested `GraphProto` inside an attribute is **marked, never parsed**, so a
  hostile file cannot recurse.
* Decoded messages (`Model`, `Graph`, `Node`, `Attribute`, `AttributeValue`,
  `Tensor`, `ValueInfo`, `TensorType`, `Dim`) borrow names and weight bytes from
  the input; decoding the 28 MB file copies no weight payload.

### 3.2 Graph IR (`graph.rs`, public)

| Type | Purpose |
|---|---|
| `ModelImport` | the whole result: `ir_version`, `producer`, `opset`, `graph_name`, `graph`, `weights`, `geometry`, `summary`, `warnings` (getters + `into_parts` + `to_model_weights`) |
| `GraphIr` | `values()` , `value(id)`, `lookup(name)`, `find(name)`, `nodes()`, `inputs()`, `outputs()`, `producer(id)`, `value_count()` |
| `GraphNode` | `onnx_index`, `name`, `onnx_op`, `op: OpId`, `inputs: Vec<ValueId>`, `outputs: Vec<ValueId>`, `params: OpParams` |
| `ValueId` / `ValueInfo` | stable value handle; `name`, `dtype: DType`, `shape: ValueShape`, `origin: ValueOrigin` |
| `ValueOrigin` | `GraphInput` \| `Weight(WeightId)` \| `Node(u32)` |
| `Dim` | `Known(u64)` \| `Symbol(fn3::shape::Symbol)` \| `Unknown(String)` (the string is the reason) |
| `ValueShape` | `Unknown` \| `Dims(Vec<Dim>)`, with `rank()`, `is_static()`, `static_shape()`, `numel()`, `render()` |
| `DecoderGeometry` | `layers`, `hidden_size`, `mlp_hidden_size`, `query_heads`, `kv_heads`, `head_dim`, `vocab_size`, `positions`, `layer_norm_epsilon`, `token_embedding`, `logits`, `kv: Vec<KvInterface>`, plus `kv_cache_config(capacity, batch)` |
| `ImportSummary` | ONNX node count, fn3 node count, value count, initializer / attribute-constant counts, weight bytes, ONNX op census, `OpId` census, dtype census, `count()`/`count_onnx()` |
| `ImportWarning` / `ImportWarningKind` | `DeclaredShape { value, declared, inferred }` |
| `MIN_IR_VERSION`/`MAX_IR_VERSION` (4..=10), `MIN_OPSET`/`MAX_OPSET` (13..=17) | the accepted envelope |

`ModelImport` owns everything (no lifetime), derives `PartialEq` and is
`Send + Sync`, so a caller can drop the 28 MB buffer and a test can compare two
imports for equality.

### 3.3 Weight table (`src/fn3/model/weights.rs`)

`WeightTensor { name, DType, Shape, Vec<u8>, WeightSource }` with a checked
constructor (`numel * dtype.size_bytes() == bytes.len()`), `to_tensor()` (host,
CPU, exact bytes) and `WeightSet { entries in declaration order, BTreeMap name
index }` with `get(WeightId)`, `find(name)`, `id_of(name)`, `iter_ids()`,
`initializers()`, `total_bytes()`, `to_model_weights() -> ModelWeights`. Order is
initializer order, then lifted attribute constants in node order; duplicate names
are rejected rather than shadowed.

### 3.4 Static analysis

Dtypes are inferred for every value and always checked against the ONNX operator's
own rule. Shapes are tracked symbolically, including through the integer payloads
that shape arithmetic flows through: `Shape → Gather → Concat → Slice →
Unsqueeze/Squeeze → Reshape`, plus `Add/Sub/Mul` on integer series, so
`Reshape` targets like `[-1, 128]` and `[batch, 1, seq, total]` resolve instead of
collapsing to "unknown".

Two reducible identities are encoded because the decoder contract names them:
`past + seq = total`, `total - seq = past`, `total - past = seq`. `-1` in a
`Reshape` target is inferred by cancelling identical factors between the input and
the target (which is what makes `[batch, seq, 128] → [batch, seq, -1, 64]` give
`2` without inventing algebra); when cancellation is not enough the axis stays
`Dim::Unknown` with the expression that could not be reduced.

Everything that cannot be proven is `Dim::Unknown(reason)`; the importer never
guesses an extent. A declared `value_info` shape that contradicts a fully static
inference is a **warning** (exporter `dim_param` naming is advisory), while a
graph input/output dtype or rank contradiction is an **error**.

### 3.5 Diagnostics

`ImportError { problems: Vec<ImportProblem> }`, in file order, each with an
optional `NodeRef { index, name, op_type }`. 36 distinct `ImportProblemKind`
variants (wire, version, opset, domain, dtype, external data, segment, ambiguous
payload, length, functions, training info, sparse initializer, missing name,
operand count, attributes, unmapped op, unimplemented op id, control flow,
subgraph, undefined value, duplicate value, not-topological, dtype mismatch, index
dtype, shape mismatch, unsupported type, architecture, io, truncation). The list
is capped at `MAX_PROBLEMS = 100` with an explicit `TooManyProblems { omitted }`
entry, so a truncated report can never look complete.
`From<ImportError> for Fn3Error` maps to `Fn3Error::InvalidFormat { format: "onnx
model", detail }` with the full multi-problem text.

## 4. Inventory comparison — pinned model

`onnx-community/tiny-random-gpt2-ONNX` @ `90f61e71…d52f`, 27,897,974 bytes,
SHA-256 `e18370ba…141a1` (verified by
`python3 tools/parity/fetch_models.py --suite tiny-gpt2 --verify-only` in the
`#49` worktree, the only place the cache exists on this machine). The model is
**not** committed; the test also pins an FNV-1a 64-bit fingerprint
(`0xc294d3ab837fb1be`) so a different 28 MB graph at the same path is noticed.

| Quantity | Issue / lock file | Imported | Match |
|---|---|---|---|
| bytes | 27,897,974 | 27,897,974 | ✅ |
| IR version | 7 | 7 | ✅ |
| opset (default domain) | 14 | 14 | ✅ |
| producer | `pytorch 2.8.0` | `pytorch` / `2.8.0` | ✅ |
| graph name | `main_graph` | `main_graph` | ✅ |
| ONNX nodes | 257 | 257 | ✅ |
| initializers | 48 (24 F32 + 24 I64) | 48 (24 + 24) | ✅ |
| initializer payload | 27,833,172 B | 27,833,172 B | ✅ |
| lifted attribute constants | — | 2 (8 B) | ✅ |
| total weight bytes | — | 27,833,180 B | ✅ |
| fn3 nodes | — | 265 (8 `Gemm` → `MatMul`+`Add`) | ✅ |
| values | — | 326 | ✅ |
| layers / hidden / mlp | 2 / 128 / — | 2 / 128 / 512 | ✅ |
| query heads / KV heads / head dim | 2 / 2 / 64 | 2 / 2 / 64 | ✅ |
| vocab | 50,257 | 50,257 | ✅ |
| positions | learned, 1024 | `Learned { wpe, 1024 }` | ✅ |
| LayerNorm eps | 1e-5 | 1e-5 (structural detection) | ✅ |
| inputs / outputs | 7 / 5 | 7 / 5 | ✅ |

ONNX operator census — **exactly** the 27 types in the issue:

```text
Unsqueeze 37  Gather 35  Reshape 25  Concat 23  Shape 20  Add 19  Mul 12
Transpose 11  ReduceMean 10  Gemm 8  Pow 7  Sub 7  Div 5  MatMul 5  Sqrt 5
Where 5  Slice 4  Cast 3  ConstantOfShape 2  Equal 2  Expand 2  Softmax 2
Split 2  Squeeze 2  Tanh 2  Less 1  Range 1
```

No `Erf`, no `Gelu` node (asserted). fn3 census: `Gather 35, MatMul 13, Add 27,
Sub 7, Mul 12, Div 5, Pow 7, Sqrt 5, Tanh 2, Softmax 2, ReduceMean 10,
Transpose 11, Reshape 25, Squeeze 2, Unsqueeze 37, Concat 23, Slice 4, Split 2,
Shape 20, ConstantOfShape 2, Expand 2, Range 1, Cast 3, Equal 2, Less 1, Where 5`
= 265. Every one of the 257 ONNX nodes maps; every mapped id is in `OpId::ALL`
and the CPU backend reports it as supported (`every_mapped_node_has_an_implemented_op_id`).

Shapes resolve exactly for the whole transformer chain, e.g. `[batch, seq, 128]`
embedding, `[seq, total]` mask concat, `[batch, 1, seq, total]` additive mask,
`[batch, 2, seq, total]` scores/softmax, `[batch, 2, seq, 64]` head output,
`[batch, 2, total, 64]` `present.*`, `[batch, seq, 50257]` logits. 24 of 326
values carry one unresolved axis: the `[batch * seq, hidden]` flatten that feeds
each `Gemm`, which the seven-symbol set cannot express. That number is pinned by a
test so a regression to "everything unknown" cannot pass silently.

## 5. Mapping decisions (as fixed by #53 / #48)

| ONNX | fn3 | Evidence |
|---|---|---|
| `Gather(axis=0)` | `Gather` | embedding is `Gather`; no `Embedding` id |
| `Gemm` | `MatMul` + `Add` | 8 nodes, all `alpha=beta=1, transA=transB=0`, weight `[in,out]`; any scaling/transposition is a named rejection |
| `ConstantOfShape` | `ConstantOfShape(shape, scalar)` | the `value` tensor is lifted into a scalar constant registered as `WeightSource::AttributeConstant`; both fills (`-FLT_MAX`, `0.0`) asserted |
| `Reshape` | `Reshape` | `allowzero=0` semantics; `allowzero=1` is a named rejection (the kernel has no mode) |
| `Trilu` | `Trilu` | `upper` goes into `OpParams.axes[0]`, as the kernel documents |
| `Split` | `Split` | sizes input + `axis` in `axes[0]`, N outputs |
| `Slice` | `Slice` | 3–5 inputs; symbolic bounds handled |
| `ReduceMean/Softmax/Transpose/Concat/Gather/Split/Flatten/ArgMax` | same | attributes in `OpParams.axes`, `keep_dims` |
| `Cast` | `Cast` | target dtype from the `to` attribute, no implicit cast elsewhere |

## 6. Verification — commands and real outcomes

Model present (`FASTNN_FN3_ONNX_MODEL=<#49 cache>`):

| Command | Outcome |
|---|---|
| `cargo fmt --check` | **OK** |
| `cargo build --no-default-features` | **OK** (only the pre-existing `fnv` unused-dependency manifest warning) |
| `cargo build` | **OK** |
| `cargo test --test fn3_import -- --test-threads=1` | **27 passed, 0 failed** |
| `cargo test --test fn3_contracts` | **43 passed** (unchanged) |
| `cargo test --test fn3_cpu_reference -- --test-threads=1` | **43 passed** (unchanged) |
| `cargo test --lib` | **637 passed** (was 583 + new unit tests) |
| `cargo test --doc` | **3 passed** (extra; the module example compiles) |
| `cargo clippy --all-targets` | no warning in the files this issue touches (28 pre-existing elsewhere) |

Model absent (`cargo test --test fn3_import`): **27 passed**, every pinned-model
test prints
`SKIP: the pinned 27,897,974-byte ONNX model is not present. Set FASTNN_FN3_ONNX_MODEL or run python3 tools/parity/fetch_models.py --suite tiny-gpt2.`
and returns without failing.

### Test coverage in `tests/fn3_import.rs`

1. `pinned_model_imports_with_the_documented_inventory` — byte length, FNV
   fingerprint, IR/opset/producer/graph name, 257/265/48/2/326/27,833,180,
   24+24 dtypes, the exact 27-op census, no `Erf`/`Gelu`, every node with inputs
   and outputs.
2. `pinned_model_geometry_is_detected_and_validated` — all nine geometry fields,
   learned positions, eps, KV config, nine exact shape-chain assertions, and the
   24-unresolved-axis pin.
3. `every_mapped_node_has_an_implemented_op_id` — `OpId::ALL` + CPU `supports` +
   no `MaskedAttention` in a real import.
4. `integer_arithmetic_is_preserved_rather_than_cast` — the exact four `i64`
   `Add`/`Sub` nodes (17, 21, 27, 44).
5. `pinned_model_ir_is_self_consistent` — value handles in range, producers exist,
   `OpParams` carrier per op, unique names, zero shape warnings.
6. `reimporting_the_same_bytes_is_identical` — `ModelImport` equality plus weight
   layout equality (byte-identical re-import).
7. `pinned_weights_cover_every_byte_of_the_declared_layout` — per-weight
   `numel × size == bytes`, initializer count, 2 attribute constants, every
   weight present in the value table, `to_model_weights()` builds all tensors.
8. `constant_of_shape_lifts_its_value_attribute_into_a_constant_input` — 2 nodes,
   two inputs, origin `Weight`, `AttributeConstant`, 4 bytes, dtype agreement,
   fills `-FLT_MAX` and `0.0`.
9. `reshape_keeps_allowzero_zero_semantics_and_rejects_one` — 25 reshapes with
   `(data, shape)` and no axis attribute; `allowzero=1` rejected by name.
10. `a_synthetic_decoder_imports_with_fully_static_shapes` — an 11-initializer,
    9-node GPT-2 decoder built in memory, every value fully static.
11. `gemm_becomes_matmul_then_add_with_the_matmul_plus_bias_shape` — `[4,8]×[8,24]`
    → `MatMul [4,24]` then `Add` with bias `[24]` → `[4,24]`, same
    `onnx_index`; lm head `[4,16]` reshaped to `[1,4,16]`.
12. `a_constant_of_shape_attribute_value_becomes_a_scalar_weight` — synthetic,
    exact fill bytes.
13. Negative, one named error each: unsupported operator (`LSTM`), attention
    operator (`MultiHeadAttention` → `MaskedAttention`, owner `#61`), external
    tensor data, missing opset, unsupported opset 18, control flow (`If`),
    subgraph attribute, `FLOAT16` weight, `DOUBLE` input, unknown attribute,
    `Gemm` scaling/`transB`, duplicate value, unsorted (non-topological) graph,
    non-decoder graph, truncated initializer.
14. `truncated_and_corrupted_input_never_panics` — 11 truncated prefixes of the
    real 28 MB file plus 12 deterministic byte flips; every case returns (typed
    error or model), never panics.

## 7. The seam for #55 and #56

**#55 (prepared-model format)** should consume:

```rust
let imported = fastnn::fn3::import::import_onnx_file(path)?;   // ImportError
imported.ir_version();                 // i64
imported.opset();                      // &OpsetImport { domain, version }
imported.weights();                    // &WeightSet
for (id, weight) in imported.weights().iter_ids() {
    weight.name();                     // &str, exactly as the file spelled it
    weight.dtype();                    // DType  (DType::code() is the stable number)
    weight.shape();                    // &Shape (dims: &[u64])
    weight.bytes();                    // &[u8], exact payload
    weight.source();                   // Initializer | AttributeConstant
}
imported.summary();                    // &ImportSummary for a header/health check
```

Weight *order* is the declaration order (file order, then lifted constants in node
order), so writing the table back is deterministic. `ModelImport` is `PartialEq`.

**#56 (planner)** should consume:

```rust
let graph = imported.graph();          // &GraphIr
for node in graph.nodes() {            // already in execution order
    node.op;                           // OpId
    node.inputs;                       // Vec<ValueId>
    node.outputs;                      // Vec<ValueId>
    node.params;                       // OpParams (axes / keep_dims / epsilon)
    node.onnx_index; node.name; node.onnx_op;   // provenance for diagnostics
}
for id in graph.inputs() { .. }        // declared graph inputs
for id in graph.outputs() { .. }       // declared graph outputs
let value = graph.value(id);           // dtype + ValueShape
value.shape.static_shape();            // Option<Shape> when fully static
value.shape.dims();                    // &[Dim]  (Known | Symbol | Unknown(reason))
graph.lookup("logits");                // Option<ValueId>
let weights = imported.to_model_weights()?;   // ModelWeights on CPU
```

**#58 (session)** should use `geometry().kv` (past/present value handles per
layer), `geometry().kv_cache_config(capacity, batch)` and
`geometry().layer_norm_epsilon`.

**#62 (end-to-end)** can assert `imported.summary()`.

## 8. Decisions worth reviewing

1. **`i64` arithmetic is preserved, not rejected or cast.** The pinned exporter
   computes `total`, `past` and the causal index range with `i64` `Add`/`Sub`
   (nodes 17, 21, 27, 44). Casting them to `f32` would be a silent numerical
   change, so the mapping keeps `i64`. **Consequence:** the alpha CPU arithmetic
   kernels (`elementwise::binary`) are documented as `f32`-only, so those four
   nodes will fail at execution until #53 accepts integer `Add`/`Sub` (or #56
   folds the shape arithmetic during specialisation). This is recorded in the
   test `integer_arithmetic_is_preserved_rather_than_cast` and is the one runtime
   gap the import surfaces. `Equal`/`Less`/`Where`/`Trilu` already accept `i64`.
2. **"Reject non-F32 compute" is three rules**, documented in `import/mod.rs`:
   unrepresentable element types, operand dtype disagreement, and a non-`i64`
   index/shape operand. A flat "compute must be `f32`" rule would reject the
   pinned graph itself.
3. **Declared intermediate shapes are advisory.** The pinned export names the
   mask axis `"past_sequence_length + 1"` for a value that is `past + seq` (the
   graph builds the causal mask from the prompt length). Mapping that string to
   `Symbol::Total` is documented at `symbol_for`, and a static declared/inferred
   disagreement is a warning, not an error — errors are reserved for the graph
   interface (input/output dtype and rank).
4. **Opset 13..=17.** Opset 18 moves `ReduceMean.axes` from an attribute to an
   input, which this mapping would silently drop; the range is pinned and the
   rejection names the reason.
5. **Topological order is required, not repaired.** ONNX requires a sorted graph;
   a use-before-define that has a later producer is reported as
   `NotTopological { value, producer_index }` rather than being reordered.
6. **`ValueOrigin::GraphInput` is used as a placeholder** for a poisoned value
   after a node-level diagnostic. Such a `ModelImport` is never returned (the
   error path discards it), and the code documents this.

## 9. Known risks / limitations

* **Architecture support is deliberately narrow**: a GPT-2-shaped decoder with
  `past_key_values.{i}.{key,value}` inputs and `present.{i}` outputs. A
  prefill-only export, a non-GPT-2 decoder, or a model with different weight
  naming fails with `ImportProblemKind::Architecture` naming the missing tensor.
  This is the issue's "keep arbitrary ONNX models out of scope".
* **String tensors and sparse tensors** are rejected implicitly (dtype) and
  explicitly (`SparseInitializer`); no model in scope uses them.
* **`Shape` slicing attributes** (`start`/`end`, opset 15+) are rejected rather
  than implemented; the pinned export uses `Slice` on the shape instead.
* **`Slice` with a symbolic `step != 1`** leaves an axis unknown (the extent
  formula for a symbolic count is not implemented); the pinned graph only uses
  step 1.
* **`Shape::default_strides` defect** (reported by #53): a weight with an interior
  or trailing extent-1 axis (`[2, 1]`, `[1, 3, 1]`) cannot be turned into a
  `Tensor` yet, so `WeightTensor::to_tensor` would return the core error
  unchanged. No pinned weight has such a shape, and the fix belongs to the shape
  contract.
* **Memory**: `WeightSet` copies the 27.8 MB of payload once (from the borrowed
  decode window) and `to_model_weights()` copies it again into aligned storage.
  Peak for the pinned model is roughly 28 MB (file) + 28 MB (weights) + 28 MB
  (tensors) + metadata.
* `GraphIr::lookup` is a linear name scan (326 values); `ModelImport` keeps a
  `BTreeMap` internally for construction. If #56 needs many lookups it should
  build its own index.
* Tracked element sequences (shape payloads) are bounded by
  `MAX_TRACKED_ELEMENTS = 4096`: a declared extent larger than that keeps its
  length but loses the per-element detail instead of materialising a huge vector.
  This bounds the memory a hostile declared shape can cost.

## 10. Shared-file change needed (for the integrator)

`src/fn3/model/weights.rs` is reserved for this issue by the ADR, but
`src/fn3/model/mod.rs` (owned by #48) does not declare it, and this issue must not
edit that file. The file is therefore wired in from the import module with a
`#[path]` declaration:

```rust
// src/fn3/import/mod.rs
#[path = "../model/weights.rs"]
mod weights;
pub use weights::{WeightId, WeightSet, WeightSource, WeightTensor};
```

The one-line follow-up, when the core owner is ready:

```rust
// src/fn3/model/mod.rs
pub mod weights;   // then `pub use weights::{WeightId, WeightSet, WeightSource, WeightTensor};`
```

and in `src/fn3/import/mod.rs` replace the two `#[path]` lines with

```rust
pub use crate::fn3::model::weights::{WeightId, WeightSet, WeightSource, WeightTensor};
```

Nothing else moves: the file already uses fully qualified
`crate::fn3::model::ModelWeights`, so it compiles unchanged from either location.
This is the only coordination item.

## 11. Not done / out of scope

* No planning, slot allocation, bound specialisation or execution (#56), no
  serialization (#55), no session (#58) — by design.
* No `tests/fn3_fixtures/**` or `fixtures/**` writes; the parity fixtures were
  read only.
* The pinned model is not committed (it has no declared license) and no network
  fetch happens in tests.
* `#62`'s end-to-end run is not attempted here; the `i64` arithmetic gap above is
  the expected first obstacle.
