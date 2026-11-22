# FN3-18 (#64) — the minimal Python inference API

**Branch** `fn3/64-python`
**Base** `c5b69c221e6088cb10ea51878b03ebec951320d4` (verified with
`git rev-parse HEAD` before the first edit)
**Tracking issue** #46 · **Issue** #64 · **Status** implementation complete on
the CPU reference backend; the Vulkan branch compiles but is not exercised (the
production session path still refuses the pinned graph on Vulkan — see §11).
Local commits only, no remote write.

---

## 1. What landed

A new Python surface, `fastnn.fn3`, over the 3.0 runtime: prepare an ONNX file
into a versioned artifact, load it on an explicit device, open independent
sessions, prefill and decode, and read the session's own counters. It is a thin
adaptation of `crate::fn3`: every number comes from `Session::with_runtime` →
`prefill`/`decode`, the same path #58 proved against the committed ORT fixtures.

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/python/mod.rs` | 104 | mine (new; ADR §5 reserves `src/fn3/python/`) — module docs, registration |
| `src/fn3/python/errors.rs` | 452 | mine (new) — 15 exception classes and the `Fn3Error` → `PyErr` mapping |
| `src/fn3/python/device.rs` | 238 | mine (new) — `Device`, `devices()`, request parsing and resolution |
| `src/fn3/python/model.rs` | 1,357 | mine (new) — `prepare`, `inspect`, `load`, `Model`, geometry/envelope/metadata/report types |
| `src/fn3/python/session.rs` | 749 | mine (new) — `OwnedSession`, `Session`, `SessionStats`, `Logits`, token parsing |
| `src/fn3/python/sha256.rs` | 294 | mine (new) — streaming SHA-256 for file identity, with FIPS check-value tests |
| `fastnn/fn3/__init__.py` | 242 | mine (new) — the user-facing package |
| `tests/test_fn3_python.py` | 759 | mine (new) — 27 tests |
| `src/python/mod.rs` | +12 | shared: one `#[path]` module declaration and one `fn3::register(m)?` call |
| `fastnn/__init__.py` | +11 | shared: expose `fastnn.fn3` and add it to `__all__` |
| `docs/reports/fn3-18-python-facade.md` | this file | mine |

Nothing else changed: no `src/fn3/{shape,storage,tensor,device,error,dtype}.rs`,
no `ops/`, `backend/`, `import/`, `model/`, `src/lib.rs`, no manifest, no
existing Rust test, no fixture, tool, workflow or roadmap file. `cargo fmt
--check` is clean and `cargo clippy` reports nothing in these files.

**The binding lives in `src/fn3/python/` but is declared as a module of
`crate::python`** (`#[path = "../fn3/python/mod.rs"] pub mod fn3;`). That is the
only way to honour both constraints at this base: the ADR reserves
`src/fn3/python/` for #64, and `src/fn3/` must stay free of PyO3 while only
`crate::python` is compiled under the `python` feature. The module is registered
as `fastnn._core.fn3`, in `sys.modules` under that name, and wrapped by
`fastnn/fn3/`.

## 2. The Python API, complete

```python
fastnn.fn3.devices() -> list[Device]
fastnn.fn3.prepare(onnx, artifact, *, max_total=None, max_seq=None,
                   max_past=None, overwrite=False) -> PrepareReport
fastnn.fn3.inspect(artifact) -> ArtifactInfo
fastnn.fn3.load(artifact, source=None, *, device=None, capacity=None) -> Model

Model.path / source / sha256 / size_bytes / device / geometry / envelope /
      metadata / capacity / kv_bytes / weight_count / weight_bytes /
      value_count / node_count
Model.session() -> Session

Session.prefill(tokens) -> Logits          # [batch, 1, vocab]
Session.decode(tokens) -> Logits           # [batch, 1, vocab]
Session.reset() -> None
Session.phase / step / tokens_consumed / cache_len / capacity / kv_bytes /
        device / instance / model_instance
Session.stats() -> SessionStats

Logits.shape / dtype / numel / nbytes / data        # data is `bytes`
Logits.to_numpy() -> numpy.ndarray                  # zero-extra-copy view
Logits.__array__                                    # np.asarray(logits)
Logits.argmax() -> int                              # greedy next token
Logits.tolist() -> list

SessionStats.phase / step / tokens_consumed / cache_len / capacity / kv_bytes /
             plan_hits / plan_misses / plan_evictions / plan_cache_len /
             slot_allocations / slot_reuses / slot_bytes_allocated /
             last_plan / last_plan_dims / cache_addresses

PrepareReport.path / status / wrote / size_bytes / sha256 / payload_crc32 /
              weight_bytes / geometry / envelope / metadata / elapsed_ms /
              verified / replaced_differences
ArtifactInfo.path / size_bytes / sha256 / format_version / payload_bytes /
             payload_crc32 / weight_bytes / value_count / node_count /
             weight_count / geometry / envelope / metadata / elapsed_ms
Device.id / kind / index / name            (str(device) == device.id)
Geometry.layers / hidden_size / mlp_hidden_size / query_heads / kv_heads /
         head_dim / vocab_size / max_positions / layer_norm_epsilon /
         kv_bytes_per_token
Envelope.batch / seq / past / total / n_heads / n_kv_heads / head_dim  # (min, max)
        .as_dict()
Metadata.ir_version / opset_version / opset_domain / producer /
         producer_version / graph_name
```

`fastnn/fn3/__init__.py` adds only what Rust cannot do well: `os.PathLike`
coercion, docstrings, `__all__`, and the `source`/`device`/`capacity` keyword
defaults. The classes and exceptions are the extension's own objects
(`fastnn.fn3.Session is fastnn._core.fn3.Session`), asserted by a test.

### The example that was actually run

```python
import numpy as np
import fastnn.fn3 as fn3

ONNX = "/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx"
PROMPT = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257]

report = fn3.prepare(ONNX, "model.fn3prep", overwrite=True)
print(report.status, report.size_bytes, report.sha256, report.geometry)
# created 27879928 dc03b3c2… Geometry(layers=2, hidden=128, heads=2/2, …)

model = fn3.load("model.fn3prep", ONNX, device="cpu")     # explicit device
session = model.session()
logits = session.prefill(PROMPT)
print(int(logits.argmax()), session.cache_len)             # 257 13

for token in [640, 11, 612, 373]:
    logits = session.decode([token])
    print(int(logits.argmax()), session.cache_len)         # 24671 14 / 11 15 / 612 16 / 373 17
```

Output of `final_demo.py` on this machine (release extension, CPU):

```text
prepare      500 ms  status=created size=27879928 sha256=dc03b3c241c52549fadeb73e289b79760e1a11c74162e6c4ccccbb1a4a1ecf2f
                        crc32=0xddf840ab weights=27833180 verified=True
                        Geometry(layers=2, hidden=128, heads=2/2, head_dim=64, vocab=50257, max_positions=1024)
                        Envelope(batch=1..=1, seq=1..=1024, past=0..=1023, total=1..=1024, n_heads=2..=2, n_kv_heads=2..=2, head_dim=64..=64)
load         204 ms  Model(#1 path="/tmp/opencode/fn3-64/final.fn3prep" device=cpu capacity=1024 Geometry(…))
                        sha256=dc03b3c241c52549... kv_bytes/session=2097152
prefill      333 ms  13 tokens -> predicted next token 257 cache_len=13 step=0
                        vs ORT fixture: max_abs=5.3644180e-07 mean_abs=1.0014762e-07
decode 1     200 ms  fed   640 -> predicted next token 24671  cache_len=14 step=1 tokens=14  vs fixture max_abs=6.1094761e-07
decode 2     205 ms  fed    11 -> predicted next token    11  cache_len=15 step=2 tokens=15
decode 3     173 ms  fed   612 -> predicted next token   612  cache_len=16 step=3 tokens=16
decode 4     170 ms  fed   373 -> predicted next token   373  cache_len=17 step=4 tokens=17
state            SessionStats(phase="ready", cache_len=17/1024, step=4, tokens=17, plans=0 hits/5 misses/0 evicted)
                 last_plan=03c61dc7541be69f7fd5dc64b6b4111a dims=(1, 1, 16, 17, 2, 2, 64)
                 cache_addresses=['0x1f6624c0', '0x1f762540']
```

### The same numbers from a Rust session

`FASTNN_FN3_ONNX_MODEL=… cargo test --test fn3_session cached_decode -- --nocapture`
(unchanged, #58's test, run in this worktree):

```text
decode.step1: 50257 logits, max_abs 6.109476e-7 (element 31254: 0.2435193 vs 0.2435187), mean_abs 1.0241756e-7
decode.step1 (summary): summary-only, first/last/max_abs 2.30968e-7, argmax 24671 (recorded 24671)
decode.step2 (summary): summary-only, first/last/max_abs 1.7881393e-7, argmax 11 (recorded 11)
decode.step3 (summary): summary-only, first/last/max_abs 2.682209e-7, argmax 612 (recorded 612)
decode.step4 (summary): summary-only, first/last/max_abs 2.9802322e-7, argmax 373 (recorded 373)
```

Python: `decode.step1` max_abs `6.1094761e-07`, argmax `24671, 11, 612, 373`,
extents `14..17`, plans `0 hits / 5 misses`. Prefill `p13`: Python
`5.3644180e-07 / 1.0014762e-07`, #58 recorded `5.364418e-7 / 1.0014763e-7`. The
figures agree to the last printed digit because it is the same engine.

## 3. Prepare: "I wrote it" versus "it was already there and valid"

`prepare(onnx, artifact, ...)` imports, builds `PreparedArtifact`, serialises it
once, and then decides:

| Destination state | `status` | `wrote` | `verified` |
|---|---|:-:|:-:|
| absent | `created` | ✔ | ✔ |
| byte-identical artifact | `already_valid` | ✘ | ✔ |
| a *different* valid artifact, no `overwrite` | — | | `Fn3ModelExistsError` with `.differences` |
| a different valid artifact, `overwrite=True` | `replaced` | ✔ | ✔ |
| not a valid artifact, no `overwrite` | — | | `Fn3FormatError` (the reader's own error) |
| not a valid artifact, `overwrite=True` | `replaced` | ✔ | ✔ |

`verified` means "the file on disk is byte-identical to the artifact this call
built": for `already_valid` that is the digest comparison, for a write it is a
read-back, a re-parse, a re-validation and a digest comparison. A torn write is
therefore an error, never a silent success.

The report carries the geometry, the declared window, the file size, the
SHA-256 of the file (comparable with `sha256sum`) and the artifact's own payload
CRC-32. `status` is checked against `wrote`/`verified` in the tests.

The window defaults are derived from the model, not guessed: `max_total =
min(position rows, 1024)`, `max_seq = max_total`, `max_past = max_total - 1`.
Every bound is validated by `Planner::envelope` before the artifact exists, so
an artifact this build could not honour never reaches a file.

`inspect(path)` is the read-only counterpart: it parses, validates, and reports
the same figures without a device, a session or a source.

## 4. Load: why `source=` is still required, and what it buys

`Session::with_runtime(&model, &import, &backend)` borrows a `ModelImport`, and
`Planner` consumes the importer's `GraphIr`. At this base a `PreparedArtifact`
cannot produce one: `GraphIr` has no public constructor and the format's
`PreparedGraph` is a mirror of it (`docs/reports/fn3-09-model-format.md` §12–13).
So `load(artifact, source=onnx)` feeds the planner from the ONNX source while the
**artifact remains the authority**: its weights (`to_model_weights`), geometry,
envelope and capacity are what run.

That seam is not trusted blindly. Before anything executes, `load` refuses a
source that disagrees with the artifact in *any* of:

* geometry (layers, widths, heads, head width, vocab, positions, epsilon);
* the value table (first differing value, by name/dtype/shape);
* the operation table (first differing node, by op/onnx op/arity);
* the weight table (every name, dtype, shape and payload byte).

The failure is `Fn3ModelError` and names the first difference, e.g.:

```text
the ONNX source and the prepared artifact disagree (weight 47
`transformer.h.1.mlp.c_proj.weight`: the payload bytes differ (262144 bytes
each), so the source is a different export or a modified file); the artifact is
what would run, so loading both would execute a model that is half of each file
— regenerate the artifact from this source with fastnn.fn3.prepare
```

A test flips one byte inside a weight payload of a copy of the pinned ONNX and
asserts the refusal. Removing the `source` argument is one upstream addition
(`GraphIr::from_parts` plus an import-side constructor), owned by #62.

## 5. Devices: enumeration, explicit requests and the no-fallback rule

`devices()` returns what this build can execute on, in preference order: the
CPU first, then every Vulkan adapter the loader enumerates (a `vulkan`-feature
build). On this build it is exactly `[Device(id="cpu")]`.

Accepted requests:

| Python | `DeviceRequest` |
|---|---|
| `None`, `"auto"`, `""` | `Auto` (the only request that may choose) |
| `"cpu"` | `Cpu` |
| `"vulkan"`, `"vulkan(0)"`, `"vulkan:"` | `Vulkan { index: None }` |
| `"vulkan:1"` | `Vulkan { index: Some(1) }` |
| a `Device` from `devices()` | `VulkanNamed { name }` for an adapter, `Cpu` for the CPU |

Matching a `Device` **by adapter name** is deliberate: a list captured on one
machine must not silently select a different adapter with the same index on
another. Keywords are case-insensitive; anything else is `Fn3DeviceError`.

The device is resolved **before any file is read**, so a wrong request costs
nothing and cannot be masked by later work. A request that cannot be honoured
raises, naming the request, the reason and the devices that exist:

```text
Fn3DeviceError: requested device vulkan is unavailable: this build has no
Vulkan backend (the `vulkan` feature is not enabled); available devices: [cpu]
   .requested == "vulkan"   .available == ["cpu"]
```

`DeviceRequest::Auto` alone maps to `cpu` here, and that mapping is visible
(`Model.device`, `Session.device`). There is no code path that returns CPU
results for an explicit Vulkan request. If a build enumerates a Vulkan adapter,
`load` opens it with `VulkanBackend::open(&Vulkan { index })` — which itself
errors rather than falling back — and any refusal downstream (for example the
known #59/#61 preflight gap for the pinned graph) surfaces as
`Fn3UnsupportedError`/`Fn3DeviceError`, never as a host-computed result.

## 6. Errors

Every core failure maps to a class a caller can branch on; the message is the
core's own, so the Rust and Python surfaces say the same thing. Attributes
carry the same values for callers that must not parse prose.

| Class | Base | Raised for | Attributes |
|---|---|---|---|
| `Fn3Error` | `Exception` | base of all of the below | — |
| `Fn3ModelError` | `Fn3Error` | model-level failure (e.g. `load` without `source`) | `kind` |
| `Fn3ModelFileError` | `Fn3ModelError` | missing/unreadable/unwritable file, or an ONNX I/O problem | `path`, `kind` |
| `Fn3FormatError` | `Fn3ModelError` | bad magic, version, checksum, framing, limits, inconsistency | `kind`, `version_mismatch`, `corruption` |
| `Fn3ImportError` | `Fn3ModelError` | ONNX import problems (all of them, in file order) | `problems`, `problem_count` |
| `Fn3ModelExistsError` | `Fn3ModelFileError` | the destination differs and `overwrite` is false | `path`, `differences` |
| `Fn3DeviceError` | `Fn3Error` | an explicit device cannot be honoured | `requested`, `available`, `reason` (and `actual` for a mismatch) |
| `Fn3EnvelopeError` | `Fn3Error` | a dimension outside the artifact's window | `symbol`, `requested`, `min`, `max` |
| `Fn3CapacityError` | `Fn3Error` | a bounded resource would be exceeded | `what`, `requested`, `capacity` |
| `Fn3SessionError` | `Fn3Error` | the phase forbids the call, or the wrong constructor was used | `action`, `phase`, `detail` |
| `Fn3TokenError` | `Fn3Error` | a token id (or element) is not a valid id for this model | `token`, `vocab_size` |
| `Fn3ShapeError` | `Fn3Error` | shapes, strides, ranks, indices, storage | — |
| `Fn3TypeError` | `Fn3Error` | a dtype the operation cannot compute on (the path never casts) | — |
| `Fn3UnsupportedError` | `Fn3Error` | no kernel for this op, or an unimplemented entry point | `op`, `backend`, `reason` / `what`, `owner` |
| `Fn3ExecutionError` | `Fn3Error` | anything else, message intact | — |

`FormatError.kind` is one of `io`, `bad_magic`, `unsupported_version`,
`unsupported_dtype_encoding`, `reserved_field`, `truncated`, `length_mismatch`,
`checksum_mismatch`, `limit_exceeded`, `malformed`, `inconsistent`,
`unknown_dtype`, `unknown_op_code`, `contract` — an exhaustive match over the
format module's enum, so a new variant cannot be silently unmapped.

The six cases the issue names, as they surface (measured):

| Case | Class | Message |
|---|---|---|
| missing file | `Fn3ModelFileError` | `cannot read "/nope/x.fn3prep": No such file or directory (os error 2); check the path and its permissions` |
| bad magic | `Fn3FormatError` (`bad_magic`) | `not a FastNN 3 prepared model: the file starts with 46 4e 4e 32 …, the magic is 46 4e 4e 33 …` |
| future version | `Fn3FormatError` (`unsupported_version`) | `this file is prepared-model format version 2, but this build reads version 1 only` |
| unavailable device | `Fn3DeviceError` | `requested device vulkan is unavailable: …; available devices: [cpu]` |
| sequence beyond the envelope | `Fn3EnvelopeError` | `dimension seq = 14 is outside the declared envelope [1, 13] (inclusive)` |
| exceeded capacity | `Fn3CapacityError` | `capacity exceeded for KV cache live extent: requested 5, capacity 4` |

No panic crosses the boundary: the binding contains no `unwrap`, `expect`,
`panic!` or unchecked index on caller input (the only `unsafe` is the
`OwnedSession` lifetime extension in §7), and every fallible core call is
mapped. Python API misuse that no Rust error describes (a non-path argument, a
non-sequence of tokens) is a builtin `TypeError` where a builtin is the right
class, and `Fn3TokenError` where the value is a token id problem.

## 7. Sessions, copies and lifetimes

**Isolation.** Each `Model.session()` allocates its own `KvCacheState` per
layer, its own plan cache and its own counters. `SessionStats.cache_addresses`
exposes the per-layer key-buffer addresses (diagnostic only) precisely so the
property is observable from Python; the tests assert the addresses of two
sessions are disjoint, that extents/steps/tokens advance independently, that the
same prompt in two sessions produces bit-identical logits, and that both report
the same `PlanId` (plans are content-identical; sessions are not).

**Reset.** `reset()` clears the extents and the counters and returns to
`empty`, keeping the buffers and the warm plan cache. The test asserts the
second round over the same shapes is bit-identical and is served with
`plan misses 5 / hits 5` — the same figures #58 measured for the Rust session.

**Copies.** `prefill`/`decode` return the logits as one immutable `bytes` copy
of the session's fresh output tensor. `Logits.to_numpy()` and
`np.asarray(logits)` give a read-only `float32` view over that copy (one copy
in total, not two); `np.array(logits)` copies if a writable array is wanted.
A test drops the session, runs `gc.collect()`, and reads the array again,
which proves nothing borrows session scratch. The `data` getter hands out the
same `bytes` object, so `np.frombuffer(logits.data, "<f4")` is also zero-copy.

**Ownership and the one `unsafe` block.** `Session<'model>` borrows its model,
import and backend for its whole life. `OwnedSession` stores the three owning
`Arc`s *next to* the session and extends the three borrows to `'static`; the
`Arc` pointees are heap allocations that do not move, the struct is never
cloned, and `session` is the first field so it is dropped before the `Arc`s.
Nothing hands a `'static` reference out of the file. The alternative — leaking
28 MB of model per session — is not acceptable, and the code carries a `SAFETY:`
argument at both the helper and the call site.

## 8. Concurrency and the GIL

* **`prefill`, `decode`, `reset`** release the GIL (`Python::detach`) for
  planning, execution, the logits copy-out and the cache append. The closure
  captures only `&Mutex<OwnedSession>` and a `Vec<u32>`, both `Send`; no Python
  object is touched while detached.
* **`prepare`, `inspect`, `load`** release the GIL for the ONNX import, the
  artifact build/serialise/parse/validate, the SHA-256 pass, the file reads and
  the weight materialisation. Only argument handling and the final report
  construction hold it.
* **The session is serialised.** A `parking_lot::Mutex` guards the mutable
  session, so two threads that call the *same* session wait for each other
  rather than corrupting state or raising "already borrowed". Read-only
  getters take the fast path (`try_lock`) and, if a step is running, **wait with
  the GIL released** so a polling thread cannot freeze the interpreter for the
  length of a step.
* **Different sessions run in parallel.** A test prefills two sessions in two
  threads and asserts identical results; another starts four threads on one
  session and asserts exactly one prefill wins, the losers get
  `Fn3SessionError`, and the session remains usable.
* **Evidence for the release**: in one measured run a busy Python counter
  thread advanced 4.1 million iterations during a 0.30 s prefill. A call that
  held the GIL would have blocked it completely.
  `test_prefill_releases_the_gil` asserts this, with a threshold of 1,000
  iterations and a floor on the measured step length so the assertion cannot
  pass vacuously.

## 9. Verification — commands and real outcomes

Environment: `uv venv .venv --python 3.14` with `numpy 2.5.5`, `pytest 9.2.0`,
`maturin`; `CARGO_BUILD_JOBS=12`; Rust 1.100.0. The 2.x Python suite needs the
pins its CI lane installs (`tools/ci/python-lane.sh`: `onnx==1.17.0`,
`onnxruntime==1.20.1`), so it was run in a second `uv` venv with Python 3.12
using the same abi3 extension.

| Command | Result |
|---|---|
| `git rev-parse HEAD` before the first edit | `c5b69c221e6088cb10ea51878b03ebec951320d4` = base |
| `cargo fmt --check` | **clean** |
| `cargo build` | **ok** (pre-existing `unused dependency fnv` manifest warning only) |
| `cargo build --no-default-features` | **ok** |
| `cargo build --features vulkan` | **ok** (the Vulkan branch of `open_backend` compiles) |
| `cargo build --no-default-features --features vulkan` | **ok** |
| `cargo test --lib` | **686 passed**, 0 failed (683 at #58 + 3 SHA-256 unit tests) |
| `maturin develop --release` | ok; `python -c "import fastnn"` ok; `python -c "import fastnn.fn3"` ok; `import fastnn._core.fn3` ok |
| `python -m pytest tests/test_fn3_python.py -q` | **27 passed** in ~11 s |
| `python -m pytest tests/ -m "not slow" -q` (3.12, onnx/onnxruntime pinned) | **396 passed, 1 skipped, 2 deselected, 1 xfailed** |
| the same, `--ignore=tests/test_fn3_python.py` | **369 passed, 1 skipped, 2 deselected, 1 xfailed** — the 2.6.0 baseline exactly, so the only delta is the 27 new tests |
| `cargo test --test fn3_session cached_decode -- --nocapture` | passed; printed the numbers quoted in §2 |
| `maturin build --release` + install into a fresh Python 3.12 venv, run from `/tmp` | ok: `import fastnn`, `fastnn.fn3`, and the full prepare/load/prefill/decode demo from the wheel outside the checkout |

The artifact is 27,879,928 B (64-byte header + 27,879,864-byte payload), i.e.
the exact size #55 measured, with 326 values / 265 nodes / 50 weights and
27,833,180 weight bytes. Its SHA-256 is
`dc03b3c241c52549fadeb73e289b79760e1a11c74162e6c4ccccbb1a4a1ecf2f` and its
header CRC-32 `0xddf840ab`; `prepare` and `inspect` report the same values as
`hashlib.sha256` on the file. The pinned ONNX itself verified as
`e18370ba…141a1` (27,897,974 B) before any run.

### Test inventory (`tests/test_fn3_python.py`, 27 tests)

Surface: namespace reachable from `fastnn` and from the dotted extension name;
the class/exception objects are the extension's; the error hierarchy is
catchable at each level; docstrings exist.

Devices: CPU first with a readable description; an unknown spelling is typed;
`vulkan`, `vulkan:0`, `vulkan:7` all raise with `.available == ["cpu"]` and the
request in the message (or the test skips if the build really has a Vulkan
device).

Prepare: the artifact's size, digest, geometry, envelope, provenance and table
counts; `created` vs `already_valid` vs `replaced`; refusal of a different
artifact with `.differences`; the file untouched by a refusal; `PathLike`
accepted, a directory rejected; accumulated import problems; missing files.

Format: bad magic, future version, corrupt payload, truncated file — each with
its `.kind`, `version_mismatch`, `corruption`, and a message naming both
versions where that applies; `load` refuses the same bytes; `load` without
`source`; a modified-but-importable source refused with the weight named.

Run: prefill 13 against `prefill.p13.last_logits` (atol 1e-5, mean 2.5e-6),
decode step 1 against `decode.step1.logits`, the recorded predicted tokens
`24671, 11, 612, 373`, extents `14..17`, `step`/`tokens_consumed`, plan
counters, `last_plan_dims`, shape/dtype of the array, read-only view, logits
outliving the session, `tolist`, and a NumPy integer array accepted as tokens.

Sessions: isolation (addresses, extents, steps, identical logits, equal plan
id), reset (state cleared, logits reproduced, plans reused 5/5), envelope and
capacity refusals leaving state unchanged, phase errors, token validation,
GIL release, two sessions in parallel, one session serialised.

## 10. What I skipped, and why

* **Vulkan execution.** This build has no Vulkan device (the feature is off),
  so the Python tests exercise only the honest-refusal path. The `vulkan`
  branch of `open_backend` compiles under `--features vulkan`, but the
  production session path still refuses the pinned graph for the reasons #58
  documented (§9.1/§9.2 of `fn3-12-session-kv.md`), and #62 owns flipping that
  into a positive GPU run. A test that asserted GPU numbers here would be a lie.
* **The 2.x suite in the 3.14 venv.** `tests/test_w4a8_activation_audit.py`
  imports `onnx`, which the project's `uv.lock` does not contain; the CI lane
  installs `onnx==1.17.0`/`onnxruntime==1.20.1` explicitly. I reproduced the
  lane's environment in a 3.12 venv (same abi3 extension) rather than change
  the project's declared dependencies.
* **Removing 2.x Python entry points.** `fastnn/__init__.py` keeps every 2.x
  name; only `fn3` was added. Marking them deprecated at import time would
  either pollute every 2.x test's output or change `AotExecutor` from a class
  into a wrapper function, and 2.x tests inspect that class. The deprecation
  note lives in the module comment instead; deletion is #65.
* **`Session.resume`.** The Rust API supports seeding a cache from an external
  prefix. It needs a defined NumPy→tensor contract (dtype, shape, per-layer
  order) and has no consumer at this base; it belongs with the CLI's persisted
  prompt cache (#63) or a follow-up, not in the alpha minimum.
* **A `fastnn.fn3.run(...)` one-shot helper and sampling (temperature/top-k).**
  Greedy `Logits.argmax()` plus NumPy covers the alpha; sampling is a policy,
  not an engine feature, and belongs above this layer.
* **`docs/roadmap/**` updates.** The ledger and ADR are integrator-owned; this
  report names what they should say (§12).

## 11. Known risks

* **The two-file load.** `load` needs the artifact *and* the ONNX source until
  `GraphIr::from_parts` (or an import-side constructor) lands. Both are parsed,
  so a 28 MB model costs two reads (~500 ms prepare, ~200 ms load here). The
  source is cross-checked against the artifact, so this is a performance and
  ergonomics wart, not a correctness hole — but it is the one thing a user will
  notice.
* **`prepare` writes non-atomically.** It uses `PreparedArtifact::write_file`
  (one `std::fs::write`), then verifies by reading back. A crash mid-write
  leaves a partial file that every reader rejects with a typed error
  (`length_mismatch`/`checksum_mismatch`), never a partial load. An atomic
  temp-file-and-rename belongs with the CLI (#63), which owns the write policy.
* **`Model`/`Session` pin their device and backend for life.** There is no
  "move to another device" operation, by design (a silent move is the thing the
  no-fallback rule forbids). A caller re-loads on the other device.
* **The session mutex is not reentrant.** A Python signal handler or a callback
  that re-enters the same session from the thread that is stepping it would
  deadlock; nothing in this API calls back into Python during a step, so no
  path exists today.
* **`cache_addresses` are diagnostic.** Allocators reuse freed addresses; the
  claim is only that *two live sessions* never share a buffer, which the tests
  assert for live sessions.
* **`verified` reads the file back.** For the 498 MB trained model that is a
  full re-read after every write. It is the price of "a reported success is a
  verified one"; `already_valid` short-circuits it.
* **`Logits` copies 201 KB per step.** That is the documented ownership rule
  (a result outlives the session). A zero-copy view into a session buffer would
  have to be invalidated by the next step, which is a worse contract for a
  Python caller.

## 12. Remaining work for other owners

| Owner | Work |
|---|---|
| #62 | `GraphIr::from_parts` (or `ModelImport` from an artifact) so `load` needs only the artifact, and so the CLI does too. Then delete the `source` argument and the cross-check, and flip the Vulkan refusal into a positive GPU test. |
| #63 | The CLI shares `Session::with_runtime`/`prefill`/`decode` with this binding, so the two paths cannot drift. Its `prepare` should reuse the same status vocabulary (`created`/`already_valid`/`replaced`) and its writer should add the atomic rename. |
| #65 | Delete every 2.x Python export from `fastnn/__init__.py` (training, autograd, `AotExecutor`, `compile_train_model`, the 2.x exception classes) and the 2.x pytest files; `fastnn.fn3` and `fastnn._core.fn3` are the survivors. Also delete `src/python/**` if the remaining bindings move; `src/fn3/python/` must keep its `#[path]` declaration wherever `crate::python` ends up, or move under the promoted `python` module. |
| #65 | Update `src/fn3/AGENTS.md` and the ADR §5 row: the Python binding is `src/fn3/python/` (declared from `src/python/mod.rs`), the Python package is `fastnn/fn3/`, and the test file is `tests/test_fn3_python.py` (the ADR says `tests/test_fn3_api.py`; the task for #64 specified `test_fn3_python.py`). |
| #67 | Package gate: the wheel built here installs and runs outside the checkout, but the release workflow must (a) build with `maturin build`, (b) install into a clean venv, and (c) run a CPU smoke test that prepares, loads, prefills and decodes — the demo in §2 is that test. `fastnn/fn3/` must be included by `python-source = "."`; verified by installing the wheel. |
| #66 | Add the Python 3.12 lane's pins to whatever runs `tests/test_fn3_python.py`; it needs only `numpy`, and it skips (with a printed reason) when the pinned model is absent. |
