# #76 command log

Every model run was serialised under `/tmp/opencode/fn3-measure.lock`; each
entry records the observed `/proc/loadavg` and the `cargo`/`rustc` process
count. The host had other owners' builds running (load 1.5-5.1), so no wall
time in this investigation is a performance measurement.

## Model verification

```bash
stat -c '%s' /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
# 498126358
sha256sum /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
# 42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d
```

Tiny model, verified against `fixtures/parity/model.lock.json`:

```bash
stat -c '%s' /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
# 27897974
sha256sum /tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx
# e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1
```

## Reference

ORT 1.23.2 CPU, `ORT_DISABLE_ALL`, one intra-op and one inter-op thread,
sequential:

```bash
/tmp/opencode/fn3-68-venv/bin/python tools/quant/ort_reference.py \
  --model /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
  --out /tmp/opencode/trained-fp32/ort-123
```

Both committed logits are reproduced **byte for byte**:

```text
prefill.p13.last_logits bytes 201028 equal True sha 8653aebf200af28d
decode.step1.logits      bytes 201028 equal True sha 8b4e0a1093ab0344
```

## Existing trained test

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_TRAINED_OUT=/tmp/opencode/trained-fp32/fn3-existing
CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_quant_format trained_gpt2_fp32 -- --ignored --nocapture
```

Observed (loadavg 2.23 before / 4.34 after, 4 cargo/rustc processes):

```text
prefill.p13.last_logits: 50257 values, max_abs 1.6403198e-4, mean_abs 7.273026191275283e-5, top1 640 vs 640
decode.step1.logits:     50257 values, max_abs 9.536743e-5,  mean_abs 1.742523531586707e-5, top1 11 vs 11
```

## Parity test (CPU)

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_TRAINED_OUT=/tmp/opencode/trained-fp32/fn3-parity
CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_trained_parity -- --ignored --nocapture
```

Outcome: pass. `max_abs` / `mean_abs` / existing-bound result:

```text
cpu prefill:       max_abs 1.640e-4, mean_abs 7.273e-5 (existing 5e-5: FAIL; scale-aware 2.973e-4: PASS)
cpu decode.step1:  max_abs 9.537e-5, mean_abs 1.743e-5 (existing 5e-5: PASS; scale-aware 2.647e-4: PASS)
cpu decode.step2:  summary-only reference; argmax 262 (manifest 262)
```

## Per-instruction probe

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_PROBE_OUT=/tmp/opencode/trained-fp32/fn3-layers
CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_trained_layers -- --ignored --nocapture
```

Outcome:

```text
phase prefill: plan 8fedb78cddd78010178b7abdf3cc5a01 (777 instructions, 1284 slots), 173 captured tensors
phase decode1: plan 4ae0b20bc575ad4789b98026d30f458e (777 instructions, 1284 slots), 173 captured tensors
phase decode2: plan 2d77a7f0a9ff8a71c7cc392120e95f39 (777 instructions, 1284 slots), 173 captured tensors
```

## ORT intermediates

```bash
/tmp/opencode/fn3-68-venv/bin/python tools/diagnostics/trained-fp32/extract_intermediates.py \
  --model /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
  --out /tmp/opencode/trained-fp32/ort-layers
```

173 values per phase, and the promoted graph still reproduces the committed
last logits exactly (`max_abs 0.0`).

## Float64 reference

```bash
/tmp/opencode/fn3-68-venv/bin/python tools/diagnostics/trained-fp32/reference_float64.py \
  --model /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
  --out /tmp/opencode/trained-fp32/f64-layers
```

## Reductions

```bash
for pair in "fn3-layers ort-layers layers.json" \
            "fn3-layers f64-layers fn3_vs_f64.json" \
            "ort-layers f64-layers ort_vs_f64.json"; do
  set -- $pair
  /tmp/opencode/fn3-68-venv/bin/python tools/diagnostics/trained-fp32/compare_layers.py \
    --fn3 "/tmp/opencode/trained-fp32/$1" --ort "/tmp/opencode/trained-fp32/$2" \
    --model /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
    --out "/tmp/opencode/trained-fp32/$3"
done

/tmp/opencode/fn3-68-venv/bin/python tools/diagnostics/trained-fp32/attribute_accumulation.py \
  --fn3 /tmp/opencode/trained-fp32/fn3-layers \
  --ort /tmp/opencode/trained-fp32/ort-layers \
  --model /tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx \
  --out /tmp/opencode/trained-fp32/accumulation.json

/tmp/opencode/fn3-68-venv/bin/python tools/diagnostics/trained-fp32/summarize.py \
  --fn3 /tmp/opencode/trained-fp32/fn3-layers \
  --ort /tmp/opencode/trained-fp32/ort-layers \
  --f64 /tmp/opencode/trained-fp32/f64-layers \
  --parity /tmp/opencode/trained-fp32/fn3-parity/parity-report.json \
  --vulkan-parity /tmp/opencode/trained-fp32/fn3-parity-vulkan/parity-report.json \
  --out /tmp/opencode/trained-fp32/summary
```

## Tiny control

```bash
FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
  CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_e2e -- --nocapture --test-threads=1
```

```text
prefill.p13.last_logits [cpu]: max_abs 5.364418e-7, mean_abs 1.0014763e-7
decode.step1.logits      [cpu]: max_abs 6.109476e-7, mean_abs 1.0241756e-7
```

## Vulkan, staged (the arithmetic arm)

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_TRAINED_OUT=/tmp/opencode/trained-fp32/fn3-parity-vulkan
export FN3_TRAINED_VULKAN=1 FN3_VK_STAGED=1 FN3_VK_REQUIRE_PHYSICAL=1
export FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e
CARGO_BUILD_JOBS=2 cargo test --release --features vulkan \
  --test fn3_trained_parity -- --ignored --nocapture
```

Outcome: pass on the GTX 1080 Ti.

```text
vulkan prefill:      max_abs 1.869e-4, mean_abs 1.006e-4, centered_rms 2.200e-5, 0 violations
vulkan decode.step1: max_abs 1.564e-4, mean_abs 6.606e-5, centered_rms 2.161e-5, 0 violations
vulkan decode.step2: argmax 262 (manifest 262), top5 [262, 339, 618, 340, 257]
```

## Vulkan, resident (the defect)

The same command without `FN3_VK_STAGED` fails in `Session::prefill`:

```text
NotImplemented { what: "reading a device-resident tensor through the host slice API;
download it with Tensor::to_host", owner: "FN3-26 (#72)" }
```

The logging wrapper names the op:

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e
CARGO_BUILD_JOBS=2 cargo test --release --features vulkan \
  --test fn3_vulkan_trained_probe -- --ignored --nocapture
```

```text
FAILED op gather params OpParams { axes: [0], keep_dims: false, epsilon: 0.0 }:
  ["in0=[50257, 768] f32 resident=false contiguous=true offset=0",
   "in1=[1, 13] i64 resident=true contiguous=true offset=0",
   "out0=[1, 13, 768] f32 resident=true contiguous=true"]
```

## Static checks

```bash
cargo fmt --check
CARGO_BUILD_JOBS=2 cargo clippy --release --no-default-features \
  --test fn3_trained_parity --test fn3_trained_layers -- -D warnings
CARGO_BUILD_JOBS=2 cargo clippy --release --features vulkan \
  --test fn3_trained_parity --test fn3_vulkan_trained_probe -- -D warnings
```

All three exit 0.
