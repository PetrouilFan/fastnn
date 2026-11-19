# FastNN 3.0 inference quickstart

**Alpha.** One path: prepare a `.fnn3` artifact from ONNX once, then run token
IDs from that artifact alone on an explicit device. The artifact carries the
graph, the weights, the geometry and the declared window, so a deployment never
needs the ONNX it was prepared from. No network is used after the model file is
on disk.

The CLI takes **token IDs**, not text. This alpha does not ship a tokenizer and
does not approximate one; see [Token IDs](#token-ids) at the end.

Run the commands below from the repository root on the pinned tiny-GPT-2
model. The sample output uses the current CLI name; the original run is
recorded in [FN3-17 evidence](reports/fn3-17-cli-quickstart.md).

## 0. Build

```bash
cargo build --release --no-default-features --features cli
```

The binary is `target/release/fastnn-infer`. A Vulkan build is the same command with
`--features cli,vulkan`; the Vulkan section below shows that exact build.

## 1. The model

The pinned model is not committed (it has no declared licence). This
quickstart starts with the file already at the path below; on a fresh machine
the repository's parity fetcher
(`python3 tools/parity/fetch_models.py --suite tiny-gpt2`, not run here) is the
only model-acquisition step and the only model step that touches the network.
`cargo build` may fetch crate dependencies on a fresh machine; after the build
and the model file, everything below is offline.

Verify the file you have:

```bash
sha256sum /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
```

```text
e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1  /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
```

The same digest is in `fixtures/parity/tiny-gpt2/model.lock.json`. (The issue
brief spells one nibble as `c4`; the file and the committed lock agree on `a4`.)

```bash
MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
```

## 2. Prepare

`prepare` is the only command that reads ONNX. It runs once per model.

```bash
mkdir -p /tmp/opencode/fn3-cli/release
./target/release/fastnn-infer prepare \
  --onnx "$MODEL" \
  --out /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
```

```text
fastnn-infer prepare
  source: /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
    bytes:  27,897,974
    sha256: e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1
  imported: 265 fn3 ops from 257 ONNX nodes, 326 values, 50 weights (48 initializers, 27,833,180 bytes)
  geometry: layers=2 hidden=128 mlp=512 heads=2/2 x 64 vocab=50,257 positions=learned (1024 rows) epsilon=1e-5
  envelope: batch 1..=1, seq 1..=128, past 0..=1023, total 1..=1024
  wrote: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928
    sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
    sections: META, GEOM, ENV, VALS (326 values), NODE (265 ops), WGHT (50 weights, 27,833,180 bytes)
  verified: read back through PreparedArtifact::read_file (framing, checksum and sections)
```

The write is atomic and is read back through the strict reader before success is
reported. `--max-seq`, `--max-past` and `--max-total` change the declared bounds;
the defaults are 128 / 1023 / 1024 and match the model's 1024 learned positions.

## 3. Inspect

Header only (no sections decoded):

```bash
./target/release/fastnn-infer inspect /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
```

```text
fastnn-infer inspect
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:   27,879,928
    sha256:  33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    header:  magic FNN3PREP, version 1, header 64 B, dtype encoding 1
    payload: 27,879,864 bytes, crc32 0xe02e9330
    status:  framing valid; payload checksum not recomputed (pass --sections for a full decode)
```

Full decode of META, GEOM, ENV, VALS, NODE and WGHT:

```bash
./target/release/fastnn-infer inspect --sections /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
```

```text
fastnn-infer inspect
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:   27,879,928
    sha256:  33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    header:  magic FNN3PREP, version 1, header 64 B, dtype encoding 1
    payload: 27,879,864 bytes, crc32 0xe02e9330
    status:  framing valid; payload checksum verified below
    sections: META, GEOM, ENV, VALS (326 values), NODE (265 ops), WGHT (50 weights, 27,833,180 bytes)
    metadata: ir_version=7 opset=14 producer="pytorch" 2.8.0 graph="main_graph"
    geometry: layers=2 hidden=128 mlp=512 heads=2/2 head_dim=64 vocab=50,257
    model:    positions=learned (1024 rows) epsilon=1e-5 kv_layers=2 graph_inputs=7 graph_outputs=5
    envelope: batch 1..=1, seq 1..=128, past 0..=1023, total 1..=1024
    payload checksum: verified (crc32 matches)
```

## 4. Verify against the source (optional)

The artifact is self-sufficient. `--onnx` is an **optional** verification: when
you pass it, `run`/`generate` re-read that ONNX and refuse to continue unless
its provenance, geometry, every value, every operation and every weight byte
still match the artifact. `--verify-source` states that intent explicitly and
requires `--onnx`.

```bash
./target/release/fastnn-infer run \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --onnx "$MODEL" \
  --verify-source \
  --tokens 464,2068,7586,21831,18045,625,262,16931,3290,13 \
  --device cpu \
  --top-k 5
```

```text
fastnn-infer inference
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx (sha256 e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1, verified against the artifact: metadata, geometry and 50 weight payloads match)
  device: requested cpu, resolved cpu, backend cpu (reference implementation)
  envelope: batch 1..=1 seq 1..=128 past 0..=1023 total 1..=1024
  kv: capacity=1024 tokens, 2 layers x 2 KV heads x 64 head_dim = 2,097,152 bytes
  prompt: 10 token IDs [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
  input: token IDs (this alpha has no text tokenizer)
  session: phase=empty cache=0 kv_bytes=2,097,152
  execution: cpu on cpu
  prefill: seq=10 past=0 cache=10 plan=e6e4191a3d58f621d9f9438e1b6f6cc8 plan-cache=0hit/1miss wall=332.14ms
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9519 max=1.0020 mean=0.000426
  top-5: 46737 (1.0020), 28597 (0.9756), 16571 (0.9148), 7868 (0.9042), 13 (0.8694)
  argmax: 46737 (logit 1.0020)
  predicted next token: 46737 (logit 1.0020)
```

The verification is identical to the run in [section 5](#5-run-token-ids-on-the-cpu) —
same plan id, same logits — because it is the same model either way.

A source that disagrees is refused before any step runs (exit code 3), and the
message names the first difference:

```text
error: `/tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3` and `/tmp/opencode/fn3-artifact-exec/flipped.onnx` do not describe the same model: weight `transformer.h.1.mlp.c_proj.weight` differs
next: pass the exact ONNX the artifact was prepared from, or re-run `fastnn-infer prepare` with this ONNX; without `--onnx` the artifact runs on its own
```

`--verify-source` without `--onnx` is a usage error (exit code 2), so a script
cannot ask for a check that silently does not happen:

```text
error: `--verify-source` asks for a source check, but no source was given
next: pass the ONNX the artifact was prepared from with `--onnx <model.onnx>`, or drop `--verify-source` to run the artifact alone
```

## 5. Run token IDs on the CPU

From here on the artifact is the **only** input. It holds the graph, the 50
weight payloads, the geometry and the envelope; the ONNX file can be moved,
renamed or deleted without affecting any command below.

This is the 10-token prompt used in the examples. Token IDs are comma or
whitespace separated, or `@file`.

```bash
TOKENS=464,2068,7586,21831,18045,625,262,16931,3290,13

./target/release/fastnn-infer run \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens "$TOKENS" \
  --device cpu \
  --top-k 5
```

```text
fastnn-infer inference
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
  device: requested cpu, resolved cpu, backend cpu (reference implementation)
  envelope: batch 1..=1 seq 1..=128 past 0..=1023 total 1..=1024
  kv: capacity=1024 tokens, 2 layers x 2 KV heads x 64 head_dim = 2,097,152 bytes
  prompt: 10 token IDs [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
  input: token IDs (this alpha has no text tokenizer)
  session: phase=empty cache=0 kv_bytes=2,097,152
  execution: cpu on cpu
  prefill: seq=10 past=0 cache=10 plan=e6e4191a3d58f621d9f9438e1b6f6cc8 plan-cache=0hit/1miss wall=325.12ms
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9519 max=1.0020 mean=0.000426
  top-5: 46737 (1.0020), 28597 (0.9756), 16571 (0.9148), 7868 (0.9042), 13 (0.8694)
  argmax: 46737 (logit 1.0020)
  predicted next token: 46737 (logit 1.0020)
```

The model is a randomly initialised tiny GPT-2 used for numerical conformance,
not a useful text model; the predicted token is what the weights say, not a
quality claim. The output names the requested and resolved device and the
backend that actually executed.

## 6. Generate greedily

```bash
./target/release/fastnn-infer generate \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens "$TOKENS" \
  --max-new-tokens 3 \
  --device cpu \
  --top-k 3 \
  --steps
```

```text
fastnn-infer inference
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
  device: requested cpu, resolved cpu, backend cpu (reference implementation)
  envelope: batch 1..=1 seq 1..=128 past 0..=1023 total 1..=1024
  kv: capacity=1024 tokens, 2 layers x 2 KV heads x 64 head_dim = 2,097,152 bytes
  prompt: 10 token IDs [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
  input: token IDs (this alpha has no text tokenizer)
  session: phase=empty cache=0 kv_bytes=2,097,152
  execution: cpu on cpu
  timing: wall-clock around each call; the Vulkan backend records per-dispatch gpu_ns, but Session does not aggregate a device time in this build
  prefill: seq=10 past=0 cache=10 plan=e6e4191a3d58f621d9f9438e1b6f6cc8 plan-cache=0hit/1miss wall=332.25ms
    counters: slot_allocations=148 slot_reuses=0 plan_cache=1 phase=ready
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9519 max=1.0020 mean=0.000426
  top-3: 46737 (1.0020), 28597 (0.9756), 16571 (0.9148)
  argmax: 46737 (logit 1.0020)
  decode 1: token=46737 past=10 seq=1 cache=11 plan=d968f0e1457e7a6a925a18d38ab75bca plan-cache=0hit/1miss wall=216.87ms slots=295/1 phase=ready
  decode 2: token=46737 past=11 seq=1 cache=12 plan=86826574e44fb91c6337c96538a51e41 plan-cache=0hit/1miss wall=203.34ms slots=323/121 phase=ready
  decode 3: token=46737 past=12 seq=1 cache=13 plan=cbc06edcef4251404f5d6951d0ab17af plan-cache=0hit/1miss wall=208.31ms slots=351/241 phase=ready
  generated token IDs: [46737, 46737, 46737]
  predicted next token: 46737 (logit 1.1304)
```

Greedy argmax only: no temperature, top-k sampling, top-p or RNG. Ties go to the
lowest token ID. `--steps` adds the token, cache length, plan id, cumulative plan
hits/misses and wall time for every decode.

## 7. Decode explicit token IDs

`--decode` feeds the listed IDs one at a time through the cache. With the
13-token fixture prefix and three cached steps:

```bash
./target/release/fastnn-infer run \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens 464,2068,7586,21831,18045,625,262,16931,3290,13,4874,2402,257 \
  --decode 640,11,612 \
  --device cpu \
  --top-k 1
```

```text
fastnn-infer inference
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
  device: requested cpu, resolved cpu, backend cpu (reference implementation)
  envelope: batch 1..=1 seq 1..=128 past 0..=1023 total 1..=1024
  kv: capacity=1024 tokens, 2 layers x 2 KV heads x 64 head_dim = 2,097,152 bytes
  prompt: 13 token IDs [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 4874, 2402, 257]
  input: token IDs (this alpha has no text tokenizer)
  session: phase=empty cache=0 kv_bytes=2,097,152
  execution: cpu on cpu
  prefill: seq=13 past=0 cache=13 plan=5baea64ce096acaa9aa09e5f6e7caf41 plan-cache=0hit/1miss wall=370.12ms
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9061 max=1.1973 mean=0.001263
  top-1: 257 (1.1973)
  argmax: 257 (logit 1.1973)
  decode 1: token=640 cache=14
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9711 max=1.0029 mean=-0.000488
  top-1: 24671 (1.0029)
  argmax: 24671 (logit 1.0029)
  decode 2: token=11 cache=15
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9191 max=1.0606 mean=-0.000097
  top-1: 11 (1.0606)
  argmax: 11 (logit 1.0606)
  decode 3: token=612 cache=16
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9516 max=1.1293 mean=0.000377
  top-1: 612 (1.1293)
  argmax: 612 (logit 1.1293)
  predicted next token: 612 (logit 1.1293)
```

## 8. Vulkan: what this machine actually does

Build with Vulkan and diagnose first:

```bash
cargo build --release --no-default-features --features cli,vulkan
./target/release/fastnn-infer diagnose
```

```text
fastnn-infer 3.0.0-alpha.1 — build/device/backend diagnosis
  vulkan feature: enabled

device matrix:
  request          state        evidence
  cpu              ready        backend cpu; reference implementation
  vulkan(0)        enumerated   [0] NVIDIA GeForce GTX 1080 Ti uuid=471b2bda-51e1-6451-e692-ff5d1e146f1e compute=true software=false
[fn3vk] backend=vulkan adapter="NVIDIA GeForce GTX 1080 Ti" type=DISCRETE_GPU uuid=471b2bda-51e1-6451-e692-ff5d1e146f1e api=1.4.312 driver=580.196.7.0 queue_family=2 validation=false software=false maxMemoryAllocationSize=4292870144 maxStorageBufferRange=4294967295 chunk_cap=536870912
                   opened       adapter=NVIDIA GeForce GTX 1080 Ti software=false validation=false dtypes=[F32] max_buffer_bytes=4292870144
  vulkan(1)        enumerated   [1] llvmpipe (LLVM 23.1.4, 256 bits) uuid=6d657361-3236-2e32-2e37-2d6172636800 compute=true software=true
                   open-failed  device unavailable: requested device vulkan(index=1) is unavailable: adapter "llvmpipe (LLVM 23.1.4, 256 bits)" (6d657361-3236-2e32-2e37-2d6172636800) is a software implementation; software-ICD runs are CI-only evidence and must be allowed explicitly (VulkanConfig::with_software(true) or FN3_VK_ALLOW_SOFTWARE=1); available devices: [cpu, vulkan(0) (NVIDIA GeForce GTX 1080 Ti), vulkan(1) (llvmpipe (LLVM 23.1.4, 256 bits))]

legend: enumerated = the loader listed it; opened = the backend built its device and pipelines;
        ran the model = this command did not execute inference. An advertised GPU that is not opened,
        and an opened GPU that never ran the model, are not evidence of a working inference path.
```

`diagnose` distinguishes *enumerated*, *opened* and *ran the model*. It does not
claim a GPU result just because the loader lists a device.

The equivalent run command on a physical GPU is:

```bash
./target/release/fastnn-infer run \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens "$TOKENS" \
  --device vulkan \
  --top-k 3
```

On this host it opens the GTX 1080 Ti and runs the full decoder plan:

```text
[fn3vk] backend=vulkan adapter="NVIDIA GeForce GTX 1080 Ti" type=DISCRETE_GPU uuid=471b2bda-51e1-6451-e692-ff5d1e146f1e api=1.4.312 driver=580.196.7.0 queue_family=2 validation=false software=false maxMemoryAllocationSize=4292870144 maxStorageBufferRange=4294967295 chunk_cap=536870912
fastnn-infer inference
  artifact: /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3
    bytes:  27,879,928  sha256: 33cf409006d9d9538712486d45e9d3cdf1e3cf9176fdfe3c9af477630ca30122
    payload crc32: 0xe02e9330
  source: none (artifact-only: the artifact carries the graph, 50 weight payloads, the geometry and the envelope)
  device: requested vulkan, resolved vulkan(0), backend vulkan (adapter=NVIDIA GeForce GTX 1080 Ti software=false validation=false)
  envelope: batch 1..=1 seq 1..=128 past 0..=1023 total 1..=1024
  kv: capacity=1024 tokens, 2 layers x 2 KV heads x 64 head_dim = 2,097,152 bytes
  prompt: 10 token IDs [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13]
  input: token IDs (this alpha has no text tokenizer)
  session: phase=empty cache=0 kv_bytes=2,097,152
  execution: vulkan on vulkan(0)
  prefill: seq=10 past=0 cache=10 plan=bc15e5dfdd8171156113b43736e55066 plan-cache=0hit/1miss wall=183.10ms
  logits: shape=[1, 1, 50257] f32 finite=50,257 min=-0.9519 max=1.0020 mean=0.000426
  top-3: 46737 (1.0020), 28597 (0.9756), 16571 (0.9148)
  argmax: 46737 (logit 1.0020)
  predicted next token: 46737 (logit 1.0020)
```

The cached decode path runs on the same device (the full gate is
`tests/fn3_e2e.rs` with `--features vulkan`):

```bash
./target/release/fastnn-infer generate \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens "$TOKENS" \
  --max-new-tokens 3 \
  --device vulkan \
  --top-k 3 \
  --steps
```

```text
  decode 1: token=46737 past=10 seq=1 cache=11 plan=c35b87ea3285e6d2e486c099b8d2879e plan-cache=0hit/1miss wall=122.13ms slots=295/1 phase=ready
  decode 2: token=46737 past=11 seq=1 cache=12 plan=484c7ad6e270ab6c03373fb2e49c572b plan-cache=0hit/1miss wall=124.28ms slots=323/121 phase=ready
  decode 3: token=46737 past=12 seq=1 cache=13 plan=b18c41f9289b0daf0c333112b1f99f0c plan-cache=0hit/1miss wall=120.52ms slots=351/241 phase=ready
  generated token IDs: [46737, 46737, 46737]
```

An explicit Vulkan request never becomes a CPU run. If a plan contains an
operation the backend does not implement for these operand types, the command
fails with exit code 5 rather than falling back. On a machine without a usable
driver the failure is earlier and is exit code 4:

```bash
VK_ICD_FILENAMES=/nonexistent/icd.json VK_DRIVER_FILES=/nonexistent/icd.json \
./target/release/fastnn-infer run \
  --model /tmp/opencode/fn3-cli/release/tiny-gpt2.fnn3 \
  --tokens "$TOKENS" \
  --device vulkan
```

```text
[fn3vk][warn] Vulkan device discovery unavailable: Vulkan loader unavailable: vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER (-9): requesting Vulkan 1.1 with layers [] and extensions []
error: cannot open the requested Vulkan device: device unavailable: requested device vulkan is unavailable: no Vulkan device is usable: Vulkan loader unavailable: vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER (-9): requesting Vulkan 1.1 with layers [] and extensions []. Install a Vulkan ICD (`vulkan-icd-loader` and the GPU driver package), or point VK_ICD_FILENAMES at a valid ICD; `vulkaninfo --summary` shows what the loader sees; available devices: [cpu]
next: install or repair the Vulkan ICD/GPU driver, or run `--device cpu`; `fastnn-infer diagnose` shows the loader's view
```

## Token IDs

This alpha is exact about IDs and refuses to guess text. Different BPE tokens
produce different logits while looking plausible, so an approximate tokenizer
would be worse than none.

To add text later, implement exact GPT-2 byte-level BPE against the pinned
`vocab.json` (sha256 `3ba3c310…09e7`), `merges.txt` (`1ce16647…adc5`) or
`tokenizer.json` (`1fe93b61…0b9e`), including the GPT-2 pre-tokenization pattern,
byte encoder and `<|endoftext|>`, and prove the IDs match the reference on a
fixed corpus before exposing a flag. Until then:

* `--tokens` accepts IDs only;
* `--decode` accepts IDs only;
* the reference tokenizer is the tool to produce them.

The Python workflow (#64) uses the same artifact and the same token-ID rule.

## Offline

After step 1, every command reads local files only: `prepare` the ONNX,
`run`/`generate` the artifact (and the ONNX only when you ask it to verify one),
`inspect` the artifact, `diagnose` the artifact and the Vulkan loader. There is
no download, cache fill or registry lookup in the CLI.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | success |
| 2 | usage |
| 3 | model/input: missing artifact, bad magic, future version, truncated or corrupt artifact, `--onnx` mismatch |
| 4 | device |
| 5 | inference (bounds, capacity, unsupported op, invalid logits) |

Every error prints `error:` and a `next:` line naming the action that fixes it.
A truncated, corrupt or wrong-version artifact is refused with code 3 even when
no ONNX file exists anywhere: the artifact is checked on its own, before any
device work or plan is built.
