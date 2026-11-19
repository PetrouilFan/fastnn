# Artifact-only INT8 weights (opt-in)

Pass `--weight-int8` to `fastnn-infer run` or `generate`, set
`SessionOptions::weight_int8` in Rust, or pass `weight_int8=True` to
`fastnn.inference.load` in Python. Without the flag the FP32 path is
unchanged. Deployment still needs only the existing `.fnn3` artifact: the file
carries FP32 weights, its size is unchanged, and no conversion tool or ONNX file
is involved. Activations, accumulators, the KV cache and logits stay FP32.

From Python the same option is a keyword on `fastnn.inference.load`, and the
per-step report is a session property:

```python
import fastnn.inference as fn3

model = fn3.load("model.fnn3", device="cpu", weight_int8=True)
session = model.session()
session.prefill(tokens)

report = session.weight_int8_report       # None when the option is off
print(report.packed_matmuls, report.fp32_matmuls)
print(report.packed_payload, report.packed_source)
print(report.dense_weight, report.dense_label)   # fp32-retained on the CPU
print(report.shared_weight, report.model_weight)
print(report.route_pack_upload)           # milliseconds
print(report.packed_prepare, report.staged_host_bytes, report.late_persistent)
print(report.as_dict())                   # the CLI's own field names

for route in report.routes:
    print(route.instruction, route.rhs, route.route)   # packed / fallback code
```

The report is a read-only snapshot of the last completed step and is `None`
before the first step. Its attribute names are the CLI labels with `-` written
as `_`; `as_dict()` uses the labels themselves (`packed-matmuls`,
`route/pack/upload`, ...). The three traffic fields are `None` on the CPU, where
the CLI prints no `weight-int8-traffic` line, and carry
`packed-prepare`/`staged-host-bytes`/`late-persistent` on a resident Vulkan
session. `weight_int8` must be a `bool`; anything else is a `TypeError`. A
host-staged Vulkan session raises `Fn3UnsupportedError` from `model.session()`,
exactly like the CLI's typed refusal, and never runs on the CPU.

This is a runtime path, not an artifact section. `fn3-quant-format-v1.md` §6 is
the fallback policy reused here; §8's packed tensor sections are read from an
artifact when it has them, and the pinned artifacts used below have none. The
session therefore packs from the FP32 weights it already loaded, and the format
and its defaults are untouched.

## What is packed, and what falls back

The session walks each live plan and treats a `MatMul` right-hand side as a
candidate only when it is a static 2-D FP32 model weight with `K >= 64` and at
least 4096 elements (the §6 policy). Each admitted weight is packed on first use
by the existing `PackedWeight` encoder and cached by model weight ID. The step
then routes:

| Operand | Route |
|---|---|
| admitted static weight | fused INT8 kernel (`matmul_quantized` on CPU, `launch_projection` on Vulkan) |
| computed RHS: attention products, the tied LM-head transpose | the same backend's FP32 kernel, code `computed-rhs` |
| folded-constant RHS | FP32, code `constant-rhs` (constants are integer-domain here) |
| graph-input RHS | FP32, code `graph-input-rhs` |
| weight below the policy floor / other §6 rule | FP32, the policy's own code (`k-below-min`, `numel-below-min`, …) |
| non-FP32 activation, rank < 2, contraction mismatch, strided view | FP32, code `activation-layout` |
| non-contiguous or non-FP32 output | FP32, code `output-layout` |
| Vulkan operand not resident/whole, or a packed buffer/dispatch limit | FP32, code `vulkan-operands` |
| weight the encoder rejects (non-finite value, unrepresentable scale) | FP32, code `packing` |

No instruction is partially approximated: each one runs wholly on the packed
kernel or wholly on the backend's FP32 kernel. There is no CPU fallback on a
Vulkan session. Requesting `--weight-int8` on a host-staged Vulkan session is
refused at session creation with a typed error, and the CLI's hint names the
fix rather than suggesting a CPU run.

`--steps` prints one line per `MatMul` with its route code, plus the byte and
traffic counters, so coverage is auditable per step:

```text
weight-int8: packed-matmuls=48/73 fp32-matmuls=25 packed-payload=90243072B \
  packed-source=339738624B dense-weight=158022256B shared-weight=0B \
  model-weight=497764984B route/pack/upload=898.52ms
weight-int8-traffic: packed-prepare=90243072B staged-host-bytes=52B late-persistent=0B
  matmul[25] transformer.h.0.attn.c_attn.weight: packed
  matmul[46] /transformer/h.0/attn/Mul_1_output_0: computed-rhs
```

`dense-weight` is Vulkan's label: the FP32 bytes some live consumer still reads
from a device buffer. On the CPU nothing is uploaded, so the same field is
printed as `fp32-retained` — weights kept in host memory. `packed-source` is
what the packed set replaced, so the exact byte reduction is
`packed-source - packed-payload`. All of these are API-level counters
(`Storage::len`, `PackedProjection::payload_bytes`), not RSS, VRAM, allocator
figures or file size.

## Residency on Vulkan

Admission is decided from the live operands **before** any bytes are encoded or
uploaded: the activation and output must be resident whole buffers, and the
packed buffers and dispatch grid must fit the device. A projection that cannot
be bound leaves no packed copy behind.

Packing and the one-time upload happen before the step's submission opens, so a
packed projection is three device buffers bound by handle inside the resident
submission. Before recording, the session collects every instruction input and
graph output of the live plan; a weight is prepared densely only when a consumer
other than a packed `MatMul` RHS reads it. A proven packed-only weight is
therefore never added to the persistent table, and `check_staged_input` — which
guards *dense* persistent weights against a late host upload — never sees it. A
weight that is both packed and dense keeps both representations and appears in
`shared-weight`; a later plan that needs the dense form prepares it then.

Packed sections are counted as prepare traffic (`packed-prepare`), not as staged
host operands: a resident INT8 run of the trained artifact reports 90,243,072 B
prepared once, 52 B of staged host bytes (the two token IDs of that prompt) and
zero late persistent uploads.

## Correctness on the pinned artifacts

```sh
cargo build --release --no-default-features --features cli,vulkan --bin fastnn-infer
python3 tools/quant/check_artifact_int8.py target/release/fastnn-infer \
  /tmp/opencode/fn3-artifact-exec/tiny-gpt2.fnn3 \
  /tmp/opencode/fn3-68/trained-gpt2.fnn3
```

The script checks both SHA-256 pins, runs artifact-only (no `--onnx`), and
compares every one of the 50,257 logits of one 10-token prefill plus two
teacher-forced decode tokens (4874, 2402) on CPU and on the GTX 1080 Ti
(`471b2bda-51e1-6451-e692-ff5d1e146f1e`). Vulkan runs take
`flock /tmp/opencode/fn3-measure.lock`, require the pinned UUID, a physical
device, `FN3_VK_VALIDATION=1`, zero validation errors and no
`VK_INSTANCE_LAYERS`. Logits are the CLI's four-decimal rendering.

The script fails, not skips, on any of: a missing artifact or SHA mismatch, a
missing `source: none (artifact-only` header, a non-physical or non-matching
device, any `[validation][ERROR]`, a coverage count other than the expected
8/13 or 48/73, an argmax or top-five set that differs, or a per-element
difference above the tolerance (0.01 for tiny, 0.85 for trained, 0.001 for
CPU-vs-Vulkan INT8).

| Artifact | Packed / total per step (CPU & Vulkan) | Max FP32 → INT8 abs (CPU / Vulkan) | Max CPU vs Vulkan INT8 abs |
|---|---:|---:|---:|
| Tiny | 8 / 13 | 0.00870 / 0.00870 | 0.00010 |
| Trained GPT-2 | 48 / 73 | 0.81710 / 0.81710 | 0.00020 |

Argmax and top-five membership matched on every tested step. The trained arm's
max RMS error was 0.67608. These are raw logits over three steps: a correctness
result, not a quality, long-context or generation-fidelity guarantee.

The trained artifact's 25 fallbacks are exactly the graph's dynamic operands:
24 attention products (per layer, the score `Mul_1` and the value transpose
that feeds `present.N.value`) and `transformer.wte.weight_transposed` (the tied
LM head). The 48 packed ones are `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and
`mlp.c_proj` for each of the 12 layers. The LM head is the largest single
projection and stays FP32 because its RHS is an instruction output, not a
weight; supporting it would need a packed copy of the transposed embedding
table rather than a transpose at run time.

## Weight bytes: exact reduction, and what stays dense

`shared-weight=0B` on both artifacts means no weight is both packed and dense,
so the two figures describe disjoint weights. They do not together cover every
model weight: `dense-weight + packed-source` is 4,104 B below `model-weight` on
the trained artifact. That is why the reduction is computed as
`packed-source - packed-payload` rather than as a complement that would absorb
whatever is in neither set.

| Artifact | Model | FP32 kept (`dense-weight`) | Packed source | Packed payload | Reduction |
|---|---:|---:|---:|---:|---:|
| Tiny | 27,833,180 B | 26,260,132 B | 1,572,864 B | 417,792 B | 1,155,072 B of packed source (73.4%) |
| Trained GPT-2 | 497,764,984 B | 158,022,256 B | 339,738,624 B | 90,243,072 B | 249,495,552 B of packed source (73.4%) |

On the trained artifact the FP32 remainder is the token embedding, the position
table, every bias and every LayerNorm parameter, because none of those is a
packed `MatMul` RHS. Against the whole model the trained reduction is 249.5 MB
(50.1%); against the packed set it is 73.4%, which is the format's size ratio.

Packed buffers are **per session**: N concurrent INT8 sessions hold N × packed
payload on the device, while the host FP32 weights are shared `Arc` storage in
the immutable model. Dense device buffers are per backend's persistent table.

## Timing (separate from correctness; validation off)

Three cold processes per arm, same GPU, same prompt and decode tokens,
`run --steps --top-k 1`, `FN3_VK_VALIDATION=0`, no `VK_INSTANCE_LAYERS`,
serialized with `flock /tmp/opencode/fn3-measure.lock`. Each cell is
min / median / max over the three runs; the FP32 arm prepares all dense weights
during session setup, while the INT8 arm packs and uploads inside the first
step's `route/pack/upload` counter, which is part of its prefill wall.

| Artifact / mode | Prefill | Decode 1 | Decode 2 | First pack+upload | Process wall |
|---|---:|---:|---:|---:|---:|
| Tiny FP32 | 8.80 / 8.86 / 9.00 ms | 5.76 / 5.76 / 5.79 ms | 5.50 / 5.51 / 5.69 ms | — | 0.364 / 0.366 / 0.368 s |
| Tiny INT8 | 44.11 / 44.37 / 44.95 ms | 5.87 / 5.88 / 6.07 ms | 5.68 / 5.69 / 5.80 ms | 32.71 / 32.94 / 33.70 ms | 0.372 / 0.373 / 0.374 s |
| Trained FP32 | 96.40 / 98.22 / 103.07 ms | 62.68 / 63.42 / 63.52 ms | 61.82 / 61.91 / 62.61 ms | — | 3.621 / 3.678 / 3.710 s |
| Trained INT8 | 1.06 / 1.06 / 1.06 s | 67.29 / 68.96 / 69.09 ms | 66.38 / 67.81 / 68.16 ms | 893.86 / 894.61 / 894.69 ms | 4.341 / 4.405 / 4.476 s |

```sh
env -u VK_INSTANCE_LAYERS -u FN3_VK_STAGED FN3_VK_VALIDATION=0 \
  flock /tmp/opencode/fn3-measure.lock \
  fastnn-infer run --model <artifact> --device vulkan:<uuid> \
    --tokens 464,2068,7586,21831,18045,625,262,16931,3290,13 \
    --decode 4874,2402 --top-k 1 --steps [--weight-int8]
```

No speedup is claimed. Steady-state decode is 2–3% slower on the tiny artifact
and 7–10% slower on the trained one (69.0/67.8 ms against 63.4/61.9 ms at the
medians), so the packed path does not beat the FP32 matmul at these shapes. The
first call costs about 0.89 s of packing on the trained model and 33 ms on the
tiny one; a later artifact-embedded packed section would remove that cost
without changing the dispatch path or the steady-state result.

## Tests

The behaviour is covered without the pinned model, so a missing fixture cannot
hide a regression:

- `src/fn3/model/weight_int8.rs`: a synthetic one-`MatMul` plan asserts the
  packed route, the report fields (`packed-payload`, `packed-source`,
  `dense-weight`, `shared-weight`), packed-vs-FP32 output within the format's
  error bound, the cached re-pack on a second step, and the closed fallback
  codes for a computed, graph-input and below-floor RHS.
- `src/fn3/model/session.rs`: a stub backend that names a Vulkan device but has
  no device allocator asserts that a staged Vulkan session is refused by the
  typed predicate, and that the CPU reference is accepted.
- `src/fn3/backend/vulkan/kernels/quant.rs`: `K = 70` exercises the compact-row
  to aligned-row conversion (zero padding per row, scales and zero-points
  copied).
- `src/fn3/cli/infer.rs`: the flag-specific hint fires only for the typed
  backend rejection.
- `tests/fn3_cli.rs`: the tiny artifact-only run, when the pinned ONNX is
  present, asserts the CPU report labels and route codes.
- `tests/fn3_session.rs`: `SessionOptions::weight_int8` exposes no report
  before a step and the pinned tiny artifact's 8/13 coverage, bytes and reset
  semantics after one; the core half of the Python hook.
- `tests/test_fn3_python.py` (`weight_int8_*`): the default emits no report; an
  INT8 load reports the CLI's coverage, byte and traffic fields (and `as_dict()`
  uses the CLI labels); the tiny artifact's logits stay within the published
  bound; a non-bool raises `TypeError`; and a staged Vulkan session raises
  `Fn3UnsupportedError`.
- `tests/fn3_vulkan_quant.rs`: a resident quantized prepare moves exactly the
  payload through `packed_prepare_bytes` and zero staged bytes.

## Automated gate

`tests/fn3_trained_int8.rs` is the opt-in automated gate for the trained
artifact on the pinned physical GPU. It is `#[ignore]` and enabled by
`FN3_TRAINED_INT8=1`; once enabled it fails, never skips, on a missing artifact,
a missing committed ORT reference, a build without `--features vulkan`, or a
non-physical/unmatched device. It reuses the ORT rows and length/SHA-256 checks
from `fn3_trained_parity.rs`, runs prefill plus decode tokens 640 and 11 on
Vulkan (resident, `weight_int8`) and CPU (`weight_int8`), asserts the 48/73
coverage, the 90,243,072-byte packed prepare traffic, zero input readbacks and
0 validation messages, and enforces the #68 logits bounds. Top-1 must match ORT
exactly; top-5 follows #68 §6's margin rule (the trained decode boundaries are
near-ties, so those two phases measure 4/5; the raw sets are printed). CPU INT8
and Vulkan INT8 must agree within 1e-3 max abs.

```sh
env -u VK_INSTANCE_LAYERS -u FN3_VK_STAGED \
  FN3_TRAINED_INT8=1 FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_VALIDATION=1 \
  FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
  flock /tmp/opencode/fn3-measure.lock \
  cargo test --release --features vulkan --test fn3_trained_int8 \
    -- trained_gpt2_int8_matches_ort_on_physical_vulkan --ignored --nocapture
```

`FN3_TRAINED_ARTIFACT` overrides the default
`/tmp/opencode/fn3-68/trained-gpt2.fnn3`. The gate does not read
`FN3_TRAINED_VULKAN`, so it can be run alone.

## Limits

- Packing is per session and per weight, from the loaded FP32 artifact; the
  artifact format is unchanged and holds no packed data.
- Only static 2-D FP32 weights are candidates. A dynamic RHS, a weight below
  the size floor, a weight whose encoder rejects it, and any operand layout the
  fused kernel rejects stay FP32 with a printed reason. A weight that is packed
  and also read by a non-`MatMul` consumer keeps both copies and shows up in
  `shared-weight`.
- One GPU, one batch of 1, a 10-token prefill and two decode steps. No long
  context, no KV-capacity sweep, no CPU timing, no comparison against the
  standalone `QuantCpuBackend` thread budget (the session path is serial).
- The reported bytes are API counters. Device allocation rounding, staging
  buffers, the plan cache and the KV cache are outside them.
