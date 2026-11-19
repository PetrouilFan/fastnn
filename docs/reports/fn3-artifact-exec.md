# FN3 artifact-only execution

**Branch** `fn3/artifact-exec` · **base** `672cec9faf089ef0840179f6edfdf33779b576a1`
(`rewrite/fastnn-3`, "the alpha candidate is complete") · **no remote writes**.

Six commits, all `Refs #55, #63, #64`. The first five hashes are stable; the
last row is this file's own commit:

| Commit | Summary | Paths (lines +/−) |
|---|---|---|
| `4604d70` | import: assemble a planner input from validated parts | `src/fn3/import/graph.rs` +77/−0, `src/fn3/import/mod.rs` +2/−2 |
| `a87a0b2` | format: turn a prepared artifact into the planner input | `src/fn3/model/format/artifact.rs` +38/−2, `…/format/graph.rs` +18/−5 |
| `8a9a810` | cli: run and generate from the artifact alone | `src/fn3/cli/args.rs` +56/−8, `src/fn3/cli/infer.rs` +72/−34 |
| `957e5ac` | python: load an artifact with no ONNX source | `fastnn/fn3/__init__.py` +9/−7, `src/fn3/python/model.rs` +45/−43 |
| `c7b60c3` | test: prove artifact-only execution on the pinned model | `tests/fn3_cli.rs` +207, `tests/fn3_model_format.rs` +117, `tests/test_fn3_python.py` +64/−9 |
| this commit | docs: prepare once, run the artifact; `--onnx` is verification | `docs/quickstart-fn3.md` +94/−31, `docs/reports/fn3-artifact-exec.md` +280 |

Before this branch, a prepared `.fnn3` artifact stored everything a model needs —
graph, weights, geometry, envelope — but could not run on its own: `fn3-infer
run` and `generate` required `--onnx`, and `fastnn.fn3.load` required
`source=`. `--onnx` was never data the plan needed; it was a source-integrity
check, and the planner was fed from the ONNX import because `GraphIr` had no
constructor.

This branch makes the artifact the input everywhere: Rust, CLI and Python.

## 1. What changed, in one table

| Surface | Before | After |
|---|---|---|
| `fn3-infer run` | `--model` + `--onnx` required | `--model` alone; `--onnx` optional verification |
| `fn3-infer generate` | same | same |
| `fn3-infer prepare` | `--onnx` required | unchanged (`prepare` is the only ONNX reader) |
| `--verify-source` | — | explicit request for the source check; requires `--onnx` |
| `fastnn.fn3.load(artifact)` | raised a model error demanding `source=` | loads, plans and runs |
| `load(artifact, source=onnx)` | verified, then ran | unchanged: verifies, then runs |
| Rust | `Planner`/`Session` needed an imported model | `PreparedArtifact::read_file` → `into_model_import` → plan → `PreparedModel` → `Session` |

The CLI header line moved with it:

```text
  onnx: /…/model.onnx (sha256 e18370ba…, verified: metadata, geometry and 50 weight payloads match)
```

became

```text
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
```

and, with `--onnx`, `source: <path> (sha256 …, verified against the artifact: metadata, geometry and 50 weight payloads match)`.

## 2. The seam: one constructor, no second planner

`Session::with_runtime` and `Planner` take `&ModelImport`, and `ModelImport`
has private fields, so an artifact could not become a planner input without an
import-side constructor. The minimum additive change, in its own commit:

* `GraphIr::from_parts(values, nodes, inputs, outputs)` — the constructor
  `docs/reports/fn3-09-model-format.md` §12-13 and `src/fn3/model/format/graph.rs`
  already named as the follow-up. It performs no validation; the caller owns the
  invariants, and `PreparedGraph::check` is what validates them on the read path.
* `ModelProvenance { ir_version, opset, producer, producer_version, graph_name }`
  and `ModelImport::from_parts(graph, weights, geometry, provenance)`.
  `summary` is recomputed from the graph and weights; every field is derivable
  because a node keeps its ONNX provenance and a weight keeps its source, so an
  artifact reproduces the original import's `ImportSummary` exactly (asserted).

On the format side, both additive:

* `PreparedGraph::into_graph_ir(self) -> GraphIr`
* `PreparedArtifact::into_model_import(self) -> ModelImport` and the borrowing
  `to_model_import(&self)` (same value, one copy).

Nothing is re-planned: `Planner`, `Survey`, `PlanCache` and `Session` are used
as they are, and the artifact's graph, weights, geometry and envelope are what
run (with `--onnx`, the import's bytes are proved identical first). `--onnx`,
when given, still goes through the existing `verify_pair` (provenance, geometry
and every weight byte) before anything executes.

**Ownership note.** `src/fn3/import/graph.rs` is not in this workstream's file
list; it is also the only place a `ModelImport` can be built. The change is
isolated in commit 1 (≈80 lines, purely additive, no importer behaviour
touched) so the orchestrator can drop it and take the #62 owner's version
instead if that lands first. `GraphIr::from_parts` and `ModelImport::from_parts`
are the names both briefs use.

The planner was **not** duplicated, and no private field of `Session`,
`PreparedModel` or `PreparedArtifact` was exposed.

## 3. CLI behaviour

```
exit 0  success
exit 2  usage      (incl. `--verify-source` with no `--onnx`)
exit 3  model      artifact unreadable / bad magic / future version / truncated /
                   corrupt / `--onnx` disagrees
exit 4  device     unchanged, still resolved before any file is read
exit 5  inference  unchanged
```

* `prepare` still requires `--onnx` and still writes + reads back.
* `run`/`generate` always print `error:` and `next:`; every before/after message
  is in §6.
* Device refusal is untouched: an explicit `--device vulkan` is still resolved
  before the artifact is read and never becomes CPU.

## 4. Python

`load(artifact)` is the primary path. `source=` remains accepted and keeps the
`check_source_agrees` proof (provenance, geometry, every value, every node,
every weight byte). `Model.source` is `None` for an artifact-only load and the
path otherwise.

With no source, the artifact's weight table moves into the import and is copied
once into `ModelWeights` — fewer copies than the two-file path had
(artifact + import + weights), because the artifact is consumed rather than
cloned.

## 5. The proof

Every command was run on this host. The artifact-only run used a *copy* of the
pinned ONNX that was renamed away before the run, so the shared pinned model was
never moved out from under the other workers.

```bash
BASE=/tmp/opencode/fn3-artifact-exec
mkdir -p "$BASE/in"
cp /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx "$BASE/in/model.onnx"
sha256sum "$BASE/in/model.onnx"
# e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1

fn3-infer prepare --onnx "$BASE/in/model.onnx" --out "$BASE/tiny-gpt2.fnn3"
# wrote 27,879,928 bytes, sha256 33cf4090…ca30122, crc32 0xe02e9330, verified by read-back

mv "$BASE/in/model.onnx" "$BASE/in/model.onnx.moved-away"
test -e "$BASE/in/model.onnx" || echo "the ONNX path used by prepare does not exist"
```

Then, with the source gone:

```bash
./target/release/fn3-infer run \
  --model "$BASE/tiny-gpt2.fnn3" \
  --tokens 464,2068,7586,21831,18045,625,262,16931,3290,13 \
  --device cpu --top-k 5
```

```text
fn3-infer inference
  artifact: /tmp/opencode/fn3-artifact-exec/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
  device: requested cpu, resolved cpu, backend cpu (reference implementation)
  execution: cpu on cpu
  prefill: seq=10 past=0 cache=10 plan=e6e4191a3d58f621d9f9438e1b6f6cc8 plan-cache=0hit/1miss
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9519 max=1.0020 mean=0.000426
  top-5: 46737 (1.0020), 28597 (0.9756), 16571 (0.9148), 7868 (0.9042), 13 (0.8694)
  predicted next token: 46737 (logit 1.0020)
```

`[46737, 28597, 16571, 7868, 13]` is the same top-5, the same logits summary and
the **same plan id** as the two-file path, for prompt
`464,2068,7586,21831,18045,625,262,16931,3290,13`.

The same equality holds for cached decode (plan ids `d968f0e1…`, `86826574…`,
`cbc06edc…` and tokens `24671, 11, 612`), for greedy generation
(`[46737, 46737, 46737]`), and on the physical GPU where the artifact-only
prefill/decode plan ids (`bc15e5df…`, `c35b87ea…`, `484c7ad6…`, `b18c41f9…`)
match the documented two-file run exactly.

Repeatable versions of this proof:

* `tests/fn3_cli.rs::the_artifact_runs_alone_with_its_onnx_moved_away` — prepares
  from a copy, renames the copy away, runs and generates from the artifact,
  compares the top-5 to the two-file run, then requires `--onnx` on the moved
  path to fail as a model error.
* `tests/fn3_model_format.rs::an_artifact_alone_plans_and_runs_with_no_onnx_source`
  — drops the `ModelImport`, writes the bytes, reads with
  `PreparedArtifact::read_file`, plans and runs, and asserts **bit-identical**
  logits vs the import path, plus the metadata/summary round-trip identity.
* `tests/test_fn3_python.py::test_artifact_only_load_works_in_a_clean_process` —
  a fresh interpreter, an empty cwd, `FASTNN_FN3_ONNX_MODEL` removed, artifact
  path as its only argument.

## 6. Error paths, before and after

Artifact-only still rejects every bad file through the existing typed errors,
with the same codes. These were captured with **no ONNX present**:

| Case | Exit | Message |
|---|---:|---|
| truncated file | 3 | `…truncated.fnn3 is not a usable prepared model: file length: the file declares 27879928, but 100000 is present` then `next: the file is truncated, corrupt or inconsistent; re-run fn3-infer prepare from the source ONNX` |
| flipped payload byte | 3 | `… payload checksum mismatch: the header stores crc32 0xe02e9330, the payload hashes to 0xc6b2cba5; the file is corrupt` then `next: the payload does not match its checksum; …` |
| future format version | 3 | `… this file is prepared-model format version 2, but this build reads version 1 only` then `next: this file uses format version 2; this build reads version 1 only. …` |
| missing `--onnx` for `--verify-source` | 2 | `--verify-source asks for a source check, but no source was given` then `next: pass the ONNX the artifact was prepared from with --onnx <model.onnx>, or drop --verify-source to run the artifact alone` |
| `--onnx` given and disagreeing | 3 | `…fnn3 and …flipped.onnx do not describe the same model: weight transformer.h.1.mlp.c_proj.weight differs` then `next: … without --onnx the artifact runs on its own` |
| `--onnx` given but missing | 3 | `cannot read ONNX model …: No such file or directory` then `next: --onnx is the optional source check; drop it to run the artifact alone, …` |

Source verification was not weakened: `verify_pair` is byte-for-byte the same
function and the same call, on the same inputs, whenever `--onnx` is passed.
`tests/fn3_cli.rs` still flips one weight byte of the source and requires
`EXIT_MODEL` + `do not describe the same model` + a named `weight `payload.

## 7. Test results

All counts are from the runs below, with `CARGO_BUILD_JOBS=3` and
`FASTNN_FN3_ONNX_MODEL` pointing at the pinned 27,897,974-byte model
(sha256 `e18370ba…141a1`, matching `fixtures/parity/tiny-gpt2/model.lock.json`).

| Command | Result |
|---|---|
| `cargo test --test fn3_cli -- --test-threads=1` | **8 passed** (6 before + 2 new) |
| `cargo test --test fn3_model_format` | **39 passed** (38 + 1) |
| `cargo test --test fn3_e2e -- --test-threads=1` | **5 passed** (CPU) |
| `cargo test --test fn3_session` | **8 passed** |
| `cargo test --test fn3_planner` | **11 passed** |
| `cargo test --test fn3_contracts` | **43 passed** |
| `cargo test --test fn3_cpu_reference` | **43 passed** |
| `cargo test --test fn3_tensor_storage` | **36 passed** |
| `cargo test --lib` | **673 passed** |
| `cargo fmt --check` | clean |
| `cargo clippy --all-targets -- -D warnings` | 0 findings |
| `cargo clippy --all-targets --no-default-features --features cli -- -D warnings` | 0 findings |
| `cargo clippy --all-targets --features fn3 -- -D warnings` | 0 findings |
| `cargo build --no-default-features` / `cargo build` / `--features vulkan` | ok |
| `uv run pytest tests/test_fn3_python.py` (`maturin develop --release`) | **29 passed** (27 + 2, one rewritten) |
| `cargo test --features vulkan --test fn3_vulkan_smoke` | not run (not a gate for this workstream; no Vulkan source touched) |

One real CLI proof line from the gate run:

```text
the_artifact_runs_alone_with_its_onnx_moved_away ... ok
a_truncated_artifact_fails_as_a_model_error_with_no_onnx_present ... ok
```

Clippy note: the only finding introduced by this branch was
`clippy::trim_split_whitespace` in the new CLI test helper; it was fixed rather
than allowed.

## 8. Skips and their reasons

* `fn3_cli::an_explicit_vulkan_request_is_never_turned_into_cpu` prints
  `SKIP: this test asserts the build-without-vulkan message` when compiled with
  `--features vulkan`; in the default and `cli` runs it executes. (It was not
  skipped in the runs reported above.)
* `fn3_cli::vulkan_either_executes_or_reports_the_known_blocker` only exists in
  a `vulkan` build and was not part of these counts.
* Every pinned-model test skips with a printed reason when the 27,897,974-byte
  ONNX is absent. It was present for every run above, so none skipped.
* The Python numerical tests skip when the ONNX is absent (they need it for
  `prepare`, not for `load`). It was present.
* `cargo test --features vulkan --test fn3_vulkan_smoke`,
  `fn3_vulkan_matmul` and `fn3_vulkan_attention` were not run: this branch
  touches no Vulkan code, and the physical-GPU generation in §5 is the
  end-to-end GPU evidence for artifact-only execution.

## 9. Decisions left to the orchestrator

1. **The import-side constructor.** `src/fn3/import/graph.rs` was outside the
   listed ownership. It is required (see §2) and isolated in commit 1. If the
   #62 owner also adds `GraphIr::from_parts` / `ModelImport::from_parts`, keep
   one — the signatures used here are the ones named in the #62 report and in
   the ledger. Nothing else in this branch depends on the file's internals.
2. **`--verify-source` is an intent flag, not a second mode.** With `--onnx`,
   verification always happens (the pre-branch behaviour); `--verify-source`
   exists so a script can *require* the check and get exit 2 if the source was
   omitted. The alternative — making bare `--onnx` mean "run from ONNX and skip
   verification" — would have changed existing behaviour and was rejected.
3. **Historical reports are now stale, and were not edited** because they are
   not this workstream's files: `docs/reports/fn3-09-model-format.md` §12-13,
   `docs/reports/fn3-17-cli-quickstart.md` ("Artifact-only `run` — blocked on
   `GraphIr::from_parts`"), `docs/reports/fn3-18-python-facade.md` (the `#62`
   follow-up row), and `docs/internals/performance-roadmap.md` line 16. They
   record what was true at that base; this report supersedes the claim.
4. **The ledger's #54 finding** (`docs/roadmap/fn3-ledger.md`, "The additive
   `GraphIr::from_parts` should land with its first consumer (#62)") is now
   satisfied by this branch's commit 1; the ledger is the orchestrator's file
   and was not edited.
