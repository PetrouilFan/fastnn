# FN3-17 (#63) — inference CLI and offline quickstart

**Branch** `fn3/63-cli`
**Base** `c5b69c221e6088cb10ea51878b03ebec951320d4` (verified with
`git rev-parse HEAD` before the first edit)
**Tracking issue** #46 · **Issue** #63 · **Status** implementation complete,
local commits only, no remote write

---

## 1. What landed

One CLI (`fn3-infer`) over the merged model path: strict ONNX import and the
versioned prepared artifact (`#54`/`#55`), the planner (`#56`), the session with
KV cache (`#58`) and the CPU/Vulkan backends (`#53`/`#57`–`#61`). No second
runtime, no second model format, no second plan builder.

| Path | Lines | Ownership |
|---|---:|---|
| `src/fn3/cli/mod.rs` | 222 | mine (new) — entry point, dispatch, `CliError`, writers, exit codes |
| `src/fn3/cli/args.rs` | 727 | mine (new) — hand-written parser, help, per-command arguments |
| `src/fn3/cli/prepare.rs` | 291 | mine (new) — import → artifact, atomic write, read-back verification |
| `src/fn3/cli/inspect.rs` | 235 | mine (new) — header, optional section summary, typed format errors |
| `src/fn3/cli/infer.rs` | 856 | mine (new) — `run`/`generate`, device reporting, bounds, logits, steps |
| `src/fn3/cli/devices.rs` | 557 | mine (new) — device grammar, UUID/index resolution, backend open, `diagnose` |
| `src/fn3/cli/digest.rs` | 213 | mine (new) — dependency-free SHA-256 (standard vectors + 28 MB file check) |
| `src/bin/fn3-infer.rs` | 15 | mine (new) — three-line binary wrapper |
| `tests/fn3_cli.rs` | 432 | mine (new) — 6 integration tests, pinned-model end-to-end |
| `src/fn3/mod.rs` | +1 | shared: one `pub mod cli;` line |
| `Cargo.toml` | +8 | shared: one `[[bin]] fn3-infer` entry with `required-features = ["cli"]` |
| `docs/quickstart-fn3.md` | this issue | mine (new) |
| `docs/reports/fn3-17-cli-quickstart.md` | this file | mine |

No backend, model, import, format, plan, session, Python, fixture, roadmap or
CI file was touched. The `cli` feature already existed for `fastnn-runtime`;
`#50`'s `fn3-bench` and `#57`'s `vulkan` feature are unchanged.

## 2. The one seam this issue could not close: `--onnx` alongside `--model`

The artifact is the authoritative model, but the current session entry point is:

```rust
Session::with_runtime(&model, &import, &backend)
```

and `import` is a `ModelImport`. `PreparedArtifact` carries a `PreparedGraph`
(the format's constructible mirror of `GraphIr`), not a `ModelImport`:
`GraphIr` has private fields and no `from_parts`, so a loaded artifact cannot
rebuild the type the planner consumes. `docs/reports/fn3-09-model-format.md`
§12/§13 recorded this as the one coordination item; it is still open.

The CLI therefore takes both paths **explicitly**:

```text
prepare --onnx model.onnx --out model.fnn3   # artifact: weights, geometry, graph, bounds, CRC
run     --model model.fnn3 --onnx model.onnx # artifact is authoritative; ONNX feeds the planner
```

`run` does not silently trust the ONNX. Before a backend is opened or a plan is
built it compares, in `src/fn3/cli/infer.rs::verify_pair`:

* `ArtifactMetadata::from_import(&import)` against the artifact's META
  (IR version, opset, producer, producer version, graph name);
* `import.geometry()` against the artifact's GEOM;
* every weight name, dtype, shape and payload byte (50 weights, 27,833,180 B on
  the pinned model).

A mismatch is exit 3 and names the first differing weight. This is a
deliberate alpha seam, not a hidden fallback: when `GraphIr::from_parts` lands
(#54/#48 follow-up), `--onnx` can become optional and `run --model` alone will
be the offline form. The quickstart states the current requirement.

## 3. CLI surface

```text
fn3-infer prepare --onnx <model.onnx> --out <model.fnn3>
                  [--max-seq N] [--max-past N] [--max-total N] [--force]
fn3-infer inspect <model.fnn3> [--sections|--full]
fn3-infer inspect --model <model.fnn3> [--sections]
fn3-infer run --model <model.fnn3> --onnx <model.onnx> --tokens <ids>
              [--decode <ids>] [--device <request>] [--capacity N]
              [--top-k N] [--steps]
fn3-infer generate --model <model.fnn3> --onnx <model.onnx> --tokens <ids>
                   --max-new-tokens N [--device <request>] [--capacity N]
                   [--top-k N] [--steps]
fn3-infer diagnose [--artifact <model.fnn3>] [--device <request>]
fn3-infer help [prepare|inspect|run|generate|diagnose]
fn3-infer --version
```

* `--tokens` / `--decode` accept comma/whitespace-separated unsigned token IDs
  or `@file` (one list per file). They are token IDs, never text.
* `--device` grammar: `auto`, `cpu`, `vulkan`, `vulkan:<index>`,
  `vulkan:<uuid>` (32 hex digits, dashed or not). There is no adapter-name
  form; a UUID is stable across enumeration order.
* `--capacity` defaults to the artifact envelope's `total.max()` and may not
  exceed it. `--top-k` defaults to 5. `--steps` prints token, cache length,
  plan id, cumulative plan hits/misses, slot counters and wall time.

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | success |
| 1 | reserved for an internal invariant; not produced by any user input path |
| 2 | usage: unknown command/flag/topic, missing or malformed value |
| 3 | model/input: missing file, bad magic, future version, corrupt payload, ONNX/artifact mismatch, invalid envelope |
| 4 | device: unavailable, build without the requested backend, loader/ICD cannot open |
| 5 | inference: envelope or KV capacity exceeded, unsupported op, invalid logits |

Every failure is two lines: `error: <what>` and `next: <what to do>`. No user
input is unwrapped, indexed without a bounds check, or allowed to panic.

## 4. Device and backend reporting

An explicit request never becomes a CPU run. `OpenedBackend::open` either opens
the named device or returns exit 4 with the request and the available list.
`auto` is the only request that chooses, and it chooses the CPU because the CPU
is first in `available_devices()`.

In a build without the `vulkan` feature, `--device vulkan` fails **before any
model file is read**:

```text
error: this build was compiled without the Vulkan feature, so no Vulkan device can be selected; the request is not silently turned into a CPU run
next: rebuild with `cargo build --release --no-default-features --features cli,vulkan`, or pass `--device cpu`; `fn3-infer diagnose` shows the devices in this build
```

With the feature, a Vulkan request is handed to `VulkanBackend::open` directly
instead of being pre-resolved through the core device list, because that list is
empty when the loader cannot create an instance and its generic reason would
hide the real driver error. A deliberately broken ICD
(`VK_ICD_FILENAMES=/nonexistent/icd.json VK_DRIVER_FILES=/nonexistent/icd.json`)
produces exit 4 with:

```text
error: cannot open the requested Vulkan device: device unavailable: requested device vulkan is unavailable: no Vulkan device is usable: Vulkan loader unavailable: vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER (-9): requesting Vulkan 1.1 with layers [] and extensions []. Install a Vulkan ICD (`vulkan-icd-loader` and the GPU driver package), or point VK_ICD_FILENAMES at a valid ICD; `vulkaninfo --summary` shows what the loader sees; available devices: [cpu]
next: install or repair the Vulkan ICD/GPU driver, or run `--device cpu`; `fn3-infer diagnose` shows the loader's view
```

`diagnose` prints three separate facts and a legend that says they are not the
same thing: *enumerated* (loader listed it), *opened* (backend built device and
pipelines), *ran the model* (this command did not execute inference). On this
host the matrix showed the physical GTX 1080 Ti enumerated and opened, and the
lavapipe software adapter enumerated but refused by policy unless
`FN3_VK_ALLOW_SOFTWARE=1` is set explicitly. A GPU that is advertised but not
opened, or opened but not run, is explicitly not evidence of a working path.

**Vulkan prefill status (honest).** The full decoder plan still cannot be built
on Vulkan. `run --device vulkan` opens the GTX 1080 Ti, then fails before any
KV byte is written:

```text
error: the vulkan backend cannot build the full decoder plan on vulkan(0): op transpose (axis permutation view) is unsupported by backend vulkan: only partially supported: 4- and 8-byte elements (F32/I32/I64); a Bool tensor is rejected with UnsupportedOp naming #59 (at plan instruction 1, op transpose)
next: full-model Vulkan execution is a known alpha blocker: the kernels exist, but the session's complete decoder plan is not yet validated on Vulkan. Run `--device cpu` for a working inference path; `fn3-infer diagnose` shows what the GPU did open
```

That is exit 5, not a fake success and not a CPU substitution. The expected
failure on machines without a usable driver is exit 4, shown above.

## 5. Determinism

* `run` reports logits and the argmax; `generate` repeats argmax until
  `--max-new-tokens` is reached.
* Tie-break is the lowest token ID (`value > best`, not `>=`), so equal logits
  cannot flip between runs.
* Non-finite logits are refused (exit 5) instead of being sorted into a
  distribution.
* `generate` has **no sampling path at all**: no temperature, top-k, top-p or
  RNG. A seeded sampler would need a pinned RNG-stream contract to be
  reproducible across versions; that is not in the alpha, and the help text
  says so.
* Step timings are wall-clock around the call. The Vulkan backend records
  per-dispatch `gpu_ns`, but `Session` does not aggregate a device time in this
  build; `--steps` prints that limitation rather than inventing a number.

## 6. Tokenisation decision: token IDs only, deliberately

The CLI does not ship a tokenizer and does not approximate one. GPT-2 text
input is byte-level BPE with a specific pre-tokenizer regex, byte-to-unicode
map, 50,257-entry vocabulary, 50,000 merges and the `<|endoftext|>` special
token. An approximate tokenizer would produce a different ID sequence, and the
CLI would report different logits while looking correct. That is worse than no
tokenizer.

What is implemented: exact token IDs, comma/whitespace separated or from a file,
range-checked against the artifact's vocabulary (50,257 here). The input line
says `input: token IDs (this alpha has no text tokenizer)`.

What an exact text path would need, all against the pinned assets already in the
parity cache:

* `vocab.json` (798,156 B, sha256 `3ba3c3109ff33976c4bd966589c11ee14fcaa1f4c9e5e154c2ed7f99d80709e7`);
* `merges.txt` (456,318 B, sha256 `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`);
* or `tokenizer.json` (3,557,680 B, sha256 `1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e`);
* the GPT-2 pre-tokenization pattern and byte encoder, plus special-token
  handling;
* conformance tests comparing every produced ID with the reference implementation
  on a fixed corpus before the flag is exposed.

Until those exist, `--tokens` is the only input and the quickstart calls that
out. The Python workflow (#64) must use the same rule.

## 7. Error coverage

| Case | Real command | Exit | Named next step |
|---|---|---:|---|
| missing artifact | `inspect /tmp/.../nope.fnn3` | 3 | `fn3-infer prepare ...` |
| foreign magic | 64-byte `FNN2PREP` file | 3 | explains `.fnnc` is never read |
| future version | 64-byte `FNN3PREP` version 2 | 3 | use matching tool or re-prepare |
| unavailable device | `--device vulkan` in a no-vulkan build | 4 | rebuild with `cli,vulkan` |
| unopenable driver | same with broken `VK_ICD_FILENAMES` | 4 | install/repair ICD or use CPU |
| wrong ONNX | one weight byte flipped in a copy | 3 | names the weight and says re-prepare |
| sequence > envelope | 129 prompt IDs against `seq <= 128` | 5 | `--max-seq 129` or shorten |
| KV capacity | 10-token prompt with `--capacity 5` | 5 | `--capacity 10` |
| token out of vocabulary | `--tokens 50257` | 5 | reference tokenizer, no BPE guessing |
| Vulkan plan blocker | `--device vulkan` on the GTX 1080 Ti | 5 | known alpha blocker; use CPU |

## 8. Offline use, model acquisition and checksums

The CLI performs no network access. `prepare` reads the ONNX; `run`/`generate`
read the artifact and the ONNX; `inspect` reads the artifact; `diagnose` reads
the artifact if given and probes the Vulkan loader. Once the pinned ONNX and the
artifact are on disk, the whole flow works with the network disconnected.

Acquisition uses the existing parity fetcher (not part of the CLI and not run by
these tests):

```text
python3 tools/parity/fetch_models.py --suite tiny-gpt2
```

Pinned model actually present on this host:

```text
path    /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
bytes   27,897,974
sha256  e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1
```

**Hash discrepancy, resolved against the committed lock.** The issue brief
states `e18370ba6247c564c4aef7b...`; the file on disk and
`fixtures/parity/tiny-gpt2/model.lock.json` both say
`e18370ba6247c564a4aef7b...` (one nibble: `a4`, not `c4`). `sha256sum` on the
27,897,974-byte file agrees with the lock and with the CLI. The brief's `c4` is
a typo; the quickstart prints the verified `a4` digest. The CLI's SHA-256 was
also checked against the standard `""`, `"abc"` and two-block vectors, and
against `sha256sum` for the 28 MB model.

The artifact's own checksum is CRC-32 over the payload (the format's
corruption check), printed by `prepare`/`inspect` next to its SHA-256.

## 9. Verification gates

All commands were run in `/home/petrouil/Projects/github/fastnn-wt-63-cli` with
`CARGO_BUILD_JOBS=12`.

| Gate | Command | Result |
|---|---|---|
| formatting | `cargo fmt --check` | clean |
| default build | `cargo build` | PASS (`Finished dev`; the bin is feature-gated and not built) |
| minimal build | `cargo build --no-default-features` | PASS |
| Vulkan lib build | `cargo build --no-default-features --features vulkan` | PASS |
| CLI release build | `cargo build --release --no-default-features --features cli --bin fn3-infer` | PASS |
| Vulkan CLI build | `cargo build --release --no-default-features --features cli,vulkan --bin fn3-infer` | PASS |
| CLI unit tests | `cargo test --lib fn3::cli` | 24 passed |
| CLI integration | `FASTNN_FN3_ONNX_MODEL=... cargo test --test fn3_cli -- --test-threads=1` | 6 passed, 0 failed (57.41 s) |
| library tests | `cargo test --lib` | 707 passed, 0 failed |
| contract tests | `cargo test --test fn3_contracts` | 43 passed, 0 failed |
| session tests | `FASTNN_FN3_ONNX_MODEL=... cargo test --test fn3_session -- --test-threads=1` | 8 passed, 0 failed (121.67 s) |

The exact user-facing outputs are in `docs/quickstart-fn3.md`, filled from the
release binary on the pinned model. The Vulkan outputs are the real failure on
this host, not a stub.

## 10. Skipped and remaining work

* **Text tokenization** — intentionally not implemented; see §6.
* **Artifact-only `run`** — blocked on `GraphIr::from_parts`; `--onnx` is
  required and verified. This is the only reason `run` needs two model files.
* **Full-model Vulkan** — open blocker reproduced honestly (§4); CPU is the
  working path.
* **Sampling** — not offered; greedy only until an RNG-stream contract exists.
* **Streaming header parse** — `PreparedHeader::parse` validates the declared
  file length, so it needs a slice of the whole file. `inspect` reads the file
  once but decodes no sections unless `--sections`; a streaming header API is a
  format follow-up. `inspect` never builds a session or copies weights.
* **Device time** — wall-clock only; see §5.
* **No GPU run to show** — `diagnose` shows the GTX 1080 Ti opened, but `run`
  cannot build the full plan. No Vulkan logits are claimed.
