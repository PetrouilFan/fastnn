# FN3-30 (#76): trained GPT-2 FP32 error investigation

**Issue:** [#76](https://github.com/PetrouilFan/fastnn/issues/76), stable gate.
**Branch:** `fn3/trained-fp32`, base
`6389186d4599471c4cb3ebc1156a5c1b872cd423` (audited integration HEAD).
**Worktree:** `/home/petrouil/Projects/github/fastnn-wt-trained-fp32`.
**Commits:** `ea5e5f7` adds the tooling and the three ignored probes;
`0c2ac8f` adds this report and its reduced outputs. Later commits on the
branch are documentation-only and do not change the evidence. Every commit
carries `Refs #76, #68` and no auto-closing keyword.

## Decision summary

The trained FP32 excess is **not a defect in an fn3 op**. It is a **tolerance
policy calibrated to the wrong quantity**:

* every elementwise bound passes with **zero violations**; the highest
  observed ratio to `5e-4 + 1e-4·|ref|` over all captured stages is 0.70;
* the error that pushes `mean_abs` over 5e-5 is almost entirely a **common-mode
  translation of the whole logit vector**, which softmax cannot see (KL
  3.5e-13, 5.7e-11, 1.2e-9; top-1 and top-5 exact at every checked position);
* an independent **float64** evaluation of the same graph differs from ORT by
  **6.25e-5** `mean_abs` at the same last position, i.e. the 5e-5 bound is
  **below the reference implementation's own rounding error**;
* fn3's arithmetic is exactly the documented sequential-f32 accumulator: it is
  bit-identical to an independent reimplementation of that order on
  **79,872/79,872** elements, and its per-stage relative error against float64
  is 1e-7..5e-6, the same order as ORT's 1e-7..2e-6;
* the trained bound is **13.7x tighter relative to the output scale** than the
  tiny bound (1.01e-6 vs 1.39e-5), while the trained model has 6x the layers and
  6x the hidden width.

**Recommendation (owner decision, not applied here):** keep the elementwise
bound unchanged and replace the absolute `mean_abs` gate with

```text
mean_abs <= 5e-5 + 5e-6 * mean(|ref|)          # absolute + model scale
rms(delta - mean(delta)) / rms(ref - mean(ref)) <= 5e-5   # ranking-relevant, scale-free
```

Both are shown next to the existing 5e-5 result everywhere below. The current
bound is **not** loosened silently: the old check still fails in every output
and in the test report, and the fixture is **not** edited by this issue.

**One real defect was found on the way, outside the numerical path:** the
trained graph cannot run on the **resident** Vulkan path. `Gather` validates
indices on the host, and the trained graph feeds it a resident `Reshape`
output. The staged Vulkan path works and reproduces the same numerical finding.
The precise file, line, failing case and patch proposal are in
[§6](#6-vulkan). This is a stable-gate blocker for the default Vulkan path.

## 1. Reproduction

### 1.1 Environment and model

| Item | Value |
| --- | --- |
| Host | AMD Ryzen 7 3700X, 16 logical cores, 31 GiB RAM |
| Rust | rustc/cargo 1.100.0 |
| ORT | 1.23.2 CPU (`/tmp/opencode/fn3-68-venv`); it reproduces the committed ORT 1.32.0 payloads **byte for byte** |
| Python | 3.13.15, NumPy 2.3.4, ONNX 1.19.0 |
| Model | `onnx-community/gpt2-ONNX` @ `1975fb9b4a2a92bd99f457407337ee25ab6cfde0` |
| Graph | `onnx/model.onnx`, **498,126,358 B**, SHA-256 **`42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d`** |
| Geometry | 12 layers, hidden 768, MLP 3072, 12 Q/12 KV heads, head dim 64, vocab 50,257, `gelu_new`, LayerNorm eps 1e-5 |
| GPU | NVIDIA GTX 1080 Ti, UUID `471b2bda-51e1-6451-e692-ff5d1e146f1e`, driver 580.196.07 |

The digest and size were re-verified with `stat` and `sha256sum` before every
run. The model is not committed.

### 1.2 Reference payloads

The committed ORT payloads are the comparison target:

* `fixtures/parity/trained-gpt2/tensors/prefill.p13.last_logits.f32`
  (`[1,1,50257]`, SHA-256 starts `8653aebf...`);
* `fixtures/parity/trained-gpt2/tensors/decode.step1.logits.f32`
  (`[1,1,50257]`, SHA-256 starts `8b4e0a10...`).

Decode step 2 has **no committed payload**; its manifest entry is summary-only
(`argmax_flat = 262`). Its full reference is an ORT run at the same pinned
conditions, and the fn3 full logits are written for comparison.

ORT 1.23.2 reproduced both committed payloads exactly:

```text
prefill.p13.last_logits bytes 201028 equal True sha 8653aebf200af28d
decode.step1.logits      bytes 201028 equal True sha 8b4e0a1093ab0344
```

The graph was also promoted to expose 173 intermediate values. The promoted
graph still reproduces both committed payloads exactly (`max_abs 0.0`), so the
instrumentation does not change the reference.

### 1.3 The reported numbers reproduce

Release build, CPU, the existing ignored test:

```bash
export FN3_TRAINED_ONNX=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_TRAINED_OUT=/tmp/opencode/trained-fp32/fn3-existing
CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_quant_format trained_gpt2_fp32 -- --ignored --nocapture
```

```text
fn3 prefill: 13 tokens, cache 13, argmax 640
fn3 decode 1: token 640, extent 14, argmax 11
fn3 decode 2: token 11, extent 15, argmax 262
prefill.p13.last_logits: 50257 values, max_abs 1.6403198e-4, mean_abs 7.273026191275283e-5, top1 640 vs 640
decode.step1.logits:     50257 values, max_abs 9.536743e-5,  mean_abs 1.742523531586707e-5, top1 11 vs 11
```

This is exactly the #68 finding: prefill `1.640e-4 / 7.273e-5`, decode step 1
`9.537e-5 / 1.743e-5`; decode step 2 is `2.975e-4 / 9.302e-5` against ORT.
The elementwise bound `5e-4 + 1e-4·|ref|` has **0 violations** in every phase.

### 1.4 Tiny-model control

The tiny model (`tiny-random-gpt2`, SHA-256
`e18370ba...178141a1`, verified against `fixtures/parity/model.lock.json`) was
run through the same production path:

```bash
FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
  CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
  --test fn3_e2e -- --nocapture --test-threads=1
```

| Tiny model, CPU | max_abs | mean_abs | Bound (atol 1e-5, rtol 1e-4, mean 2.5e-6) |
| --- | ---: | ---: | --- |
| prefill p13 last | 5.364e-7 | 1.001e-7 | pass, 25x headroom |
| decode step 1 | 6.109e-7 | 1.024e-7 | pass, 24x headroom |
| decode step 2 (summary) | 1.788e-7 | — | argmax 11, recorded 11 |
| decode steps 3-4 (summary) | 2.68e-7 / 2.98e-7 | — | argmax 612 / 373, recorded |

The tiny model's logits are zero-centred: `mean(ref) = -6.6e-4`,
`mean(|ref|) = 0.1798`, `std = 0.2254`. The trained model's last-position logits
are **not**: `mean = -49.455`, `mean(|ref|) = 49.455`, `std = 3.3175`. That is
the whole difference the absolute bound ignores.

## 2. Where the error is, and where it is not

### 2.1 Per-layer growth (prefill, every block output, fn3 vs ORT)

`mean_abs` is the suite metric; `centered_rms` removes the one common-mode
component; `rel_rms = rms(delta)/rms(reference)`. The full machine-readable
table is in [`fn3-trained-fp32/layer-table.md`](fn3-trained-fp32/layer-table.md)
and [`layers-fn3-vs-ort.json`](fn3-trained-fp32/layers-fn3-vs-ort.json).

| Layer | fn3-ORT mean_abs | fn3-ORT max_abs | fn3-ORT centered_rms | fn3-ORT rel_rms | fn3-f64 rel_rms | ORT-f64 rel_rms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.051e-06 | 1.373e-04 | 2.183e-06 | 9.035e-07 | 8.101e-07 | 3.435e-07 |
| 1 | 1.130e-06 | 1.221e-04 | 2.038e-06 | 3.047e-07 | 3.362e-07 | 1.187e-07 |
| 2 | 1.997e-06 | 5.615e-03 | 5.628e-05 | 2.185e-06 | 1.897e-06 | 2.945e-07 |
| 3 | 2.281e-06 | 4.639e-03 | 4.652e-05 | 1.685e-06 | 1.527e-06 | 1.713e-07 |
| 4 | 2.473e-06 | 4.150e-03 | 4.167e-05 | 1.432e-06 | 1.345e-06 | 1.048e-07 |
| 5 | 2.898e-06 | 4.150e-03 | 4.172e-05 | 1.388e-06 | 1.269e-06 | 1.335e-07 |
| 6 | 3.292e-06 | 3.906e-03 | 3.934e-05 | 1.284e-06 | 1.234e-06 | 8.617e-08 |
| 7 | 3.778e-06 | 3.662e-03 | 3.700e-05 | 1.192e-06 | 1.190e-06 | 8.134e-08 |
| 8 | 4.417e-06 | 3.418e-03 | 3.470e-05 | 1.108e-06 | 1.156e-06 | 1.010e-07 |
| 9 | 5.361e-06 | 3.418e-03 | 3.494e-05 | 1.106e-06 | 1.154e-06 | 1.152e-07 |
| 10 | 6.617e-06 | 3.174e-03 | 3.321e-05 | 1.032e-06 | 1.069e-06 | 1.324e-07 |
| 11 | 9.609e-06 | 5.737e-03 | 6.054e-05 | 3.924e-06 | 3.914e-06 | 3.804e-07 |

The error first appears at the first arithmetic op (layer 0 LayerNorm:
`mean_abs 1.6e-8`, `rel_rms 2.7e-7`), grows monotonically and roughly linearly
with depth, and stays at **1e-6 relative** throughout. There is no stage where
it jumps by orders of magnitude. The large per-stage `max_abs` values come from
one known GPT-2 massive-activation channel (hidden index 447), not from a
broken element: at `h.2/mlp/c_proj` the reference is 2317.25, the absolute
error 5.6e-3, and the relative error **2.4e-6**. The same channel is 2465.76
after the residual and 371.92 at `h.11/Add_1`.

### 2.2 Stage decomposition, layer 0 and layer 11

Layer 0 shows the arithmetic starting from a tiny base:

| Stage | mean_abs | max_abs | signed | centered_rms | rel_rms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h.0/ln_1/Add_1_output_0` | 1.647e-08 | 2.384e-07 | +2.831e-10 | 2.842e-08 | 2.713e-07 |
| `h.0/attn/c_attn/Gemm_output_0` | 3.086e-07 | 8.106e-06 | +4.432e-09 | 5.454e-07 | 5.458e-07 |
| `h.0/attn/Softmax_output_0` | 3.456e-08 | 1.192e-06 | +4.741e-10 | 8.903e-08 | 4.611e-07 |
| `h.0/attn/MatMul_1_output_0` | 5.979e-08 | 9.537e-07 | +2.131e-09 | 9.050e-08 | 6.031e-07 |
| `h.0/attn/c_proj/Gemm_output_0` | 2.653e-07 | 8.583e-06 | -4.654e-09 | 5.984e-07 | 5.222e-07 |
| `h.0/Add_output_0` | 2.651e-07 | 8.583e-06 | -4.454e-09 | 5.981e-07 | 5.063e-07 |
| `h.0/ln_2/Add_1_output_0` | 8.623e-08 | 9.537e-07 | +3.649e-09 | 1.149e-07 | 7.409e-07 |
| `h.0/mlp/c_fc/Gemm_output_0` | 5.481e-07 | 1.621e-05 | +1.408e-08 | 7.618e-07 | 5.666e-07 |
| `h.0/mlp/act/Mul_3_output_0` | 1.044e-07 | 1.621e-05 | +2.274e-08 | 2.071e-07 | 9.711e-07 |
| `h.0/mlp/c_proj/Gemm_output_0` | 1.046e-06 | 1.373e-04 | +1.376e-08 | 2.179e-06 | 1.293e-06 |
| `h.0/Add_1_output_0` | 1.051e-06 | 1.373e-04 | +9.116e-09 | 2.183e-06 | 9.035e-07 |

Layer 11 shows the same arithmetic after twelve blocks, with no new failure
mode:

| Stage | mean_abs | max_abs | signed | centered_rms | rel_rms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h.11/ln_1/Add_1_output_0` | 3.412e-07 | 7.033e-06 | +4.686e-09 | 5.004e-07 | 1.716e-06 |
| `h.11/attn/c_attn/Gemm_output_0` | 1.292e-06 | 2.480e-05 | -4.455e-10 | 1.851e-06 | 1.613e-06 |
| `h.11/attn/Softmax_output_0` | 5.894e-08 | 2.325e-06 | -4.484e-10 | 1.650e-07 | 7.556e-07 |
| `h.11/attn/MatMul_1_output_0` | 4.533e-07 | 2.480e-05 | +4.806e-09 | 1.168e-06 | 1.693e-06 |
| `h.11/attn/c_proj/Gemm_output_0` | 2.726e-06 | 3.174e-03 | +2.230e-07 | 3.390e-05 | 1.232e-06 |
| `h.11/Add_output_0` | 8.050e-06 | 6.348e-03 | +5.929e-07 | 6.580e-05 | 4.924e-06 |
| `h.11/ln_2/Add_1_output_0` | 2.883e-07 | 9.298e-06 | +3.556e-09 | 4.407e-07 | 2.121e-06 |
| `h.11/mlp/c_fc/Gemm_output_0` | 1.294e-06 | 5.150e-05 | +1.784e-07 | 1.930e-06 | 1.488e-06 |
| `h.11/mlp/act/Mul_3_output_0` | 2.986e-07 | 5.150e-05 | +4.754e-08 | 7.539e-07 | 2.664e-06 |
| `h.11/mlp/c_proj/Gemm_output_0` | 5.301e-06 | 6.332e-04 | +6.751e-08 | 1.647e-05 | 4.134e-06 |
| `h.11/Add_1_output_0` | 9.609e-06 | 5.737e-03 | +6.540e-07 | 6.054e-05 | 3.924e-06 |
| `ln_f/Add_1_output_0` | 1.304e-06 | 7.668e-04 | -9.089e-08 | 1.175e-05 | 1.466e-06 |

### 2.3 The arithmetic is exactly the documented CPU contract

`src/fn3/backend/cpu/kernels/matmul.rs` documents one f32 accumulator, ascending
`k`, no FMA. `attribute_accumulation.py` recomputed three real projections with
that exact order in NumPy (elementwise f32 multiply/add, vectorised only over
`M`/`N`) and compared bit for bit:

| Projection | Elements | fn3 == sequential f32 | fn3 rms vs f64 | ORT rms vs f64 |
| --- | ---: | ---: | ---: | ---: |
| `h.0.attn.c_attn` (K=768, N=2304) | 29,952 | **29,952/29,952** | 4.112e-07 | 3.776e-07 |
| `h.2.mlp.c_proj` (K=3072, N=768) | 9,984 | **9,984/9,984** | 5.130e-05 | 5.128e-06 |
| `h.11.mlp.c_fc` (K=768, N=3072) | 39,936 | **39,936/39,936** | 5.498e-07 | 1.864e-06 |

All 79,872 elements are bit-identical to the documented order. The c_proj row
is less accurate than ORT (10x rms) because K=3072 and the GELU activations are
large, but it is still 2e-6 relative to the 2317-value output; the c_fc row is
**more** accurate than ORT. That is accumulation order, not a wrong value, and
it is not a systematic fn3 disadvantage.

### 2.4 Hypothesis checks

| Candidate | Evidence | Verdict |
| --- | --- | --- |
| LayerNorm epsilon | Graph constant `1e-5`; fn3 executes the same `Add` node; `ln_1` output `rel_rms 2.7e-7` (layer 0) and `1.5e-6` (layer 11). | no mismatch |
| Attention scaling | Graph constants `Sqrt_1 = Sqrt_2 = 0.35355338` (each 1/√8, product 1/8); both sides execute the same `Mul` nodes. Softmax `rel_rms 4.6e-7`..`1.0e-6`. | correct |
| GELU / `gelu_new` | Graph constants `0.5`, `3`, `0.044715`, `0.7978846`, `1`; both sides execute the same `Mul/Pow/Add/Tanh` decomposition; `tanh` `rel_rms 4.0e-7`. | correct |
| Softmax | Max-subtract, exp, ascending sum, divide; all-masked finite `-FLT_MAX` policy preserved; `rel_rms 4.6e-7`..`1.0e-6`. | correct |
| Transposes / layout | Every captured slot shape matches ORT exactly; `Q` is `[1,12,13,64]`, `Kᵀ` `[1,12,64,13]`, scores `[1,12,13,13]`, context `[1,12,13,64]`; no shape or stride mismatch. | correct |
| `lm_head` | The graph reaches it through `Transpose(wte)`. CPU fn3-vs-f64 projection error `mean_abs 3.5e-5`, ORT `8.8e-6`; the hidden-state delta projected through `wte` in float64 predicts the observed constant shift (`1.8508e-5` vs `1.8529e-5` prefill; `-2.696e-6` vs `-2.686e-6` decode 1; `9.273e-5` vs `9.228e-5` decode 2). | amplification, not a wrong matmul |
| Accumulation order | 79,872/79,872 bit-identical to the documented sequential order; per-stage relative error 1e-7..5e-6. | expected f32 |

### 2.5 The mechanism: a common-mode translation, not a ranking change

The last-position trained logits are dominated by a common mode:

| Phase | mean(ref) | mean(\|ref\|) | mean_abs | signed | std(delta) | centered_rms | rel_rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill p13 last | -49.455 | 49.455 | 7.273e-05 | -7.273e-05 | 2.200e-05 | 2.200e-05 | 1.533e-06 |
| decode step 1 | -42.931 | 42.931 | 1.743e-05 | -2.686e-06 | 2.170e-05 | 2.170e-05 | 5.082e-07 |
| decode step 2 | -108.459 | 108.459 | 9.302e-05 | +9.228e-05 | 4.622e-05 | 4.622e-05 | 9.511e-07 |

The signed mean flips sign between phases and is almost the whole `mean_abs` in
prefill and decode step 2. Projecting the final hidden-state delta through the
tied embedding in float64 predicts that constant component to better than 5%
(`1.8508e-5` vs `1.8529e-5`, `-2.696e-6` vs `-2.686e-6`, `9.273e-5` vs
`9.228e-5`). The common component projects almost uniformly over the
vocabulary: its standard deviation across the 50,257 logits is 1.66e-6
(prefill), 3.31e-6 (decode 1), 4.12e-6 (decode 2), one to two orders below its
mean. A constant added to every logit is invisible to softmax; the measured KL
is 3.5e-13, 5.7e-11 and 1.2e-9, and top-1/top-5 are exact.

The per-position breakdown shows the absolute error is not a fixed fn3 bias:

| Pos | ref mean | mean_abs | signed | centered_rms | rel_rms | KL | top-1 margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -39.98 | 2.841e-04 | +2.841e-04 | 2.769e-05 | 7.134e-06 | 4.4e-10 | 0.310 |
| 1 | -67.56 | 3.125e-05 | -7.172e-06 | 3.879e-05 | 5.835e-07 | 3.0e-10 | 0.113 |
| 2 | -72.74 | 4.778e-05 | -4.160e-05 | 4.005e-05 | 7.933e-07 | 4.1e-10 | 1.565 |
| 3 | -96.12 | 4.755e-05 | -4.227e-05 | 3.885e-05 | 5.970e-07 | 6.0e-10 | 1.043 |
| 4 | -137.29 | 7.891e-05 | -5.809e-05 | 7.873e-05 | 7.124e-07 | 7.7e-10 | 0.199 |
| 5 | -89.05 | 3.531e-05 | -1.773e-05 | 4.064e-05 | 4.975e-07 | 3.4e-10 | 1.372 |
| 6 | -97.97 | 3.330e-05 | +3.575e-06 | 4.169e-05 | 4.267e-07 | 9.6e-10 | 0.714 |
| 7 | -94.01 | 3.322e-05 | +8.149e-06 | 4.099e-05 | 4.443e-07 | 6.7e-10 | 0.270 |
| 8 | -101.31 | 3.459e-05 | +1.546e-05 | 4.055e-05 | 4.282e-07 | 5.2e-10 | 0.546 |
| 9 | -113.46 | 3.915e-05 | +1.989e-05 | 4.489e-05 | 4.325e-07 | 5.2e-10 | 0.682 |
| 10 | -123.51 | 3.947e-05 | +2.047e-05 | 4.539e-05 | 4.030e-07 | 8.0e-10 | 0.156 |
| 11 | -236.52 | 1.361e-04 | +1.288e-04 | 9.395e-05 | 6.739e-07 | 2.2e-11 | 7.145 |
| 12 | -49.46 | 7.273e-05 | -7.273e-05 | 2.200e-05 | 1.533e-06 | 3.5e-13 | 7.561 |

Every row's top-1 agrees. The largest per-position centered error (position 11,
9.4e-5) is 2.35e-5 of that row's logit standard deviation (4.00), and its KL is
2.2e-11.

### 2.6 The reference's own error exceeds the bound

An independent float64 evaluation of the same graph, stage by stage, gives the
absolute error scale that f32 cannot avoid:

| Last-position logits | mean_abs | max_abs | centered_rms | rel_rms |
| --- | ---: | ---: | ---: | ---: |
| fn3 vs ORT | 7.273e-05 | 1.640e-04 | 2.200e-05 | 1.533e-06 |
| fn3 vs float64 | 1.353e-04 | 2.267e-04 | 2.143e-05 | 2.763e-06 |
| **ORT vs float64** | **6.254e-05** | **8.674e-05** | **5.291e-06** | **1.266e-06** |

ORT itself is **6.25e-5** `mean_abs` away from an exact evaluation of the same
graph at the same position, i.e. **above the 5e-5 bound**. A maximally accurate
f32 implementation cannot satisfy a 5e-5 absolute bound against ORT unless it
reproduces ORT's rounding rather than the exact answer. The float64 reference
is not a new fixture; it is the exact-arithmetic arm of this investigation, and
it matches ORT to `rel_rms 1.27e-6` (centred: `5.29e-6` absolute), which is the
expected f32 agreement.

## 3. Separate metrics

### 3.1 CPU (release, committed reference)

| Phase | max_abs | mean_abs | existing 5e-5 | signed | centered_rms | rel_rms | violations | top1 | top5 | KL |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill p13 last | 1.640e-04 | 7.273e-05 | **FAIL** | -7.273e-05 | 2.200e-05 | 1.533e-06 | 0 | 640 = 640 | 5/5 | 3.504e-13 |
| decode step 1 | 9.537e-05 | 1.743e-05 | pass | -2.686e-06 | 2.170e-05 | 5.082e-07 | 0 | 11 = 11 | 5/5 | 5.662e-11 |
| decode step 2 | 2.975e-04 | 9.302e-05 | **FAIL** | +9.228e-05 | 4.622e-05 | 9.511e-07 | 0 | 262 = 262 | 5/5 | 1.195e-09 |

Full 13-position prefill (`prefill.p13.logits`, summary-only in the suite):
`mean_abs 7.027e-05`, `max_abs 5.493e-04`, `centered_rms 1.031e-04`,
`rel_rms 9.360e-07`, 0 violations, KL 1.496e-08.

Top-1 margins: 7.5605 (prefill), 2.2217 (decode 1), 1.03 (decode 2). The top-1
is not close: the smallest margin is 1.03 and the largest error is 2.98e-4.

### 3.2 KV cache

fn3 vs ORT over all 24 `present.*` tensors per phase (bound
`1e-4 + 1e-4·|ref|`):

| Phase | Violations | max_abs | mean_abs |
| --- | ---: | ---: | ---: |
| prefill | 0 / 239,616 | 2.861e-05 | 9.780e-07 |
| decode step 1 | 0 / 258,048 | 2.861e-05 | 1.011e-06 |
| decode step 2 | 0 / 276,480 | 2.861e-05 | 1.024e-06 |

### 3.3 Both bounds visible

| Phase | Reference mean(\|ref\|) | Existing 5e-5 | Proposed `5e-5 + 5e-6·mean\|ref\|` | Observed mean_abs |
| --- | ---: | --- | ---: | ---: |
| prefill p13 last | 49.455 | **FAIL** | 2.973e-04 (4.1x headroom) | 7.273e-05 |
| decode step 1 | 42.931 | pass | 2.647e-04 (15.2x) | 1.743e-05 |
| decode step 2 | 108.459 | **FAIL** | 5.923e-04 (6.4x) | 9.302e-05 |
| Vulkan staged prefill | 49.455 | **FAIL** | 2.973e-04 (3.0x) | 1.006e-04 |
| Vulkan staged decode 1 | 42.931 | **FAIL** | 2.647e-04 (4.0x) | 6.606e-05 |
| Vulkan staged decode 2 | 108.459 | **FAIL** | 5.923e-04 (7.0x) | 8.443e-05 |

The proposed ranking metric is `centered_rms / std(ref)`:

| Phase | centered_rms | std(ref) | ratio | Proposed 5e-5 |
| --- | ---: | ---: | ---: | --- |
| prefill last | 2.200e-05 | 3.3175 | 6.63e-06 | pass |
| decode 1 | 2.170e-05 | 2.9253 | 7.42e-06 | pass |
| decode 2 | 4.622e-05 | 3.4539 | 1.338e-05 | pass |
| Vulkan staged prefill | 2.200e-05 | 3.3175 | 6.63e-06 | pass |
| Vulkan staged decode 1 | 2.161e-05 | 2.9253 | 7.39e-06 | pass |
| Vulkan staged decode 2 | 4.611e-05 | 3.4539 | 1.335e-05 | pass |
| prefill per-position worst (pos 11) | 9.395e-05 | 4.001 | 2.348e-05 | pass |

## 4. Recommendation

### 4.1 Why the absolute 5e-5 is the wrong policy for this model

1. It is **below the reference's own error**: ORT is 6.25e-5 from float64 on
   the same tensor.
2. It is **not scale-aware**. The trained logits have a common mode of -49,
   -43 and -108; a 1e-6 relative error becomes 7e-5..1.4e-4 absolute. The tiny
   logits are zero-centred (`mean(|ref|) = 0.1798`), so the same absolute number
   is a different relative criterion: the trained bound is **13.7x tighter**
   relative to output scale (`5e-5/49.455 = 1.01e-6` vs
   `2.5e-6/0.1798 = 1.39e-5`).
3. It measures a quantity softmax removes. The signed error is 99-100% of
   `mean_abs` in prefill and decode step 2 and flips sign between phases; the
   centered error, KL and top-k are all far inside any meaningful bound.
4. It is at the edge of the reference noise it was calibrated against: the
   documented ORT↔PyTorch trained mean is 2.09e-5, 42% of the bound. A different
   valid f32 accumulation order (fn3's sequential reference) uses 1.4x the
   bound while staying at 1.5e-6 relative.

### 4.2 Proposed policy (owner decision)

Keep unchanged:

```text
abs(actual - ref) <= 5e-4 + 1e-4 * abs(ref)          # elementwise; 0 violations
NaN/Inf, shape/dtype, cache extent: unchanged
top-1: different only when the reference top-two gap exceeds 2*atol
```

Replace only the suite-level `mean_abs` clause with two clauses:

```text
mean_abs <= 5e-5 + 5e-6 * mean(|ref|)
rms(delta - mean(delta)) <= 5e-5 * rms(ref - mean(ref))
```

* The first is the same shape as the elementwise bound: an absolute floor plus
  a term proportional to the model's output scale. The 5e-6 coefficient is 3.4x
  the worst observed `mean_abs/mean(|ref|)` on CPU (1.47e-6) and 2.5x the worst
  on staged Vulkan (2.03e-6).
* The second is dimensionless. It removes the common mode that softmax
  removes, and measures the error that can change the distribution. The worst
  observed ratio is 2.35e-5 (prefill position 11), so 5e-5 has 2.1x headroom.
* Both are reported for every phase in
  [`summary.json`](fn3-trained-fp32/summary.json); the old 5e-5 result is kept
  as `mean_abs_within_existing_bound = false` in the test report.

**Do not** simply raise the absolute bound to 1.5e-4 or 3e-4: that would hide a
real absolute bias on a zero-centred model. **Do not** change the CPU matmul to
mimic ORT's blocked/FMA order solely to pass this bound: it would trade a
documented bit-reproducible reference for agreement with one implementation,
and the exact float64 result would still fail the 5e-5 check.

If the owner prefers no manifest change before the stable gate, the fallback is
an explicit, recorded exception for `trained-gpt2` with the elementwise gate,
KL/top-k and the two proposed numbers attached. That is weaker than changing
the policy because the next backend would need the same exception.

The machine-readable policy lives in
`fixtures/parity/trained-gpt2/manifest.json` `provenance.tolerance_policy`; this
issue **does not edit it**. The owner decision should update that field, the
`fixtures/parity/README.md` table, and `tools/parity/generate_fixtures.py` in
one change, then move the assertion into `tests/fn3_trained_parity.rs`.

## 5. Is there a numerical defect?

No. The evidence that would distinguish a defect from accumulation order is
all negative:

* no stage's relative error exceeds 5e-6 (the logits are 1.5e-6);
* every kernel's output matches an independent float64 evaluation at f32
  scale, and the CPU matmul matches the documented order bit for bit;
* the error's dominant component is a translation that the model's own output
  distribution makes invisible, not a shape-, layout- or formula-dependent
  error;
* ORT, PyTorch and fn3 each differ from exact by the same order; the
  ORT↔PyTorch trained noise floor was already 2.09e-5 mean / 2.82e-4 max;
* the sign and size of the translation change with phase and token position,
  and the staged Vulkan backend has a different translation with the same
  centred error.

The only numerical lever that would make fn3 pass the old bound is to reproduce
ORT's accumulation order. That is a performance/portability decision for #71,
not a correctness fix.

## 6. Vulkan

### 6.1 Staged Vulkan reproduces the finding

`FN3_VK_STAGED=1` runs the trained graph on the GTX 1080 Ti through the real
device kernels:

| Vulkan staged | max_abs | mean_abs | signed | centered_rms | rel_rms | violations | top1 | top5 | KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill p13 last | 1.869e-04 | 1.006e-04 | +1.006e-04 | 2.200e-05 | 2.078e-06 | 0 | 640 | 5/5 | 3.351e-13 |
| decode step 1 | 1.564e-04 | 6.606e-05 | +6.605e-05 | 2.161e-05 | 1.615e-06 | 0 | 11 | 5/5 | 7.613e-11 |
| decode step 2 | 2.747e-04 | 8.443e-05 | +8.316e-05 | 4.611e-05 | 8.763e-07 | 0 | 262 | 5/5 | 1.093e-09 |

The centred error is the **same** as CPU (2.200e-5 vs 2.200e-5 prefill,
2.161e-5 vs 2.170e-5 decode 1, 4.611e-5 vs 4.622e-5 decode 2). The common-mode
translation is **different and larger** (1.006e-4 vs 7.273e-5 prefill,
6.61e-5 vs 1.74e-5 decode 1), and Vulkan-vs-CPU centred RMS is 1.06e-5
(decode 1) / 1.42e-5 (decode 2). This is direct evidence that the absolute
`mean_abs` is accumulation-order-dependent, not a model property.

### 6.2 Real defect: resident Vulkan cannot run the trained graph

> **Resolved.** This section is the finding as recorded at the time. The defect
> is fixed in `14adee3` and narrowed in `95a44e9` (the planner folds a provably
> identity `Reshape` into its input slot only when the session owns a
> contiguous host i64 `input_ids`, so `Gather` validates the host tensor), the resident path is
> the default, and the trained resident and staged arms are byte-identical.
> See `docs/reports/fn3-3.0.0-release.md` §4 for the verification and
> `src/fn3/model/specialize.rs` for the fold. What follows is the historical
> diagnosis, kept because it names the exact mechanism and the rejected
> alternatives.

At the time of this investigation the default (resident) Vulkan path failed in
`Session::prefill`:

```text
NotImplemented { what: "reading a device-resident tensor through the host slice API;
download it with Tensor::to_host", owner: "FN3-26 (#72)" }
```

The logging wrapper names the instruction:

```text
FAILED op gather params OpParams { axes: [0], keep_dims: false, epsilon: 0.0 }:
  ["in0=[50257, 768] f32 resident=false contiguous=true offset=0",
   "in1=[1, 13] i64 resident=true contiguous=true offset=0",
   "out0=[1, 13, 768] f32 resident=true contiguous=true"]
```

* **File:** `src/fn3/backend/vulkan/kernels/embedding.rs:132`
  (`let index_values = read_i64_values(op, indices, "gather indices")?;`).
* **Mechanism:** `read_i64_values` is
  `src/fn3/backend/vulkan/kernels/transform.rs:239-247`; line 246 calls
  `view.as_i64_slice()`. A device-resident view returns the
  `Fn3Error::NotImplemented` constructed at `src/fn3/tensor.rs:1209`.
* **Failing case:** the pinned trained graph, node 12
  `Reshape(input_ids, [1,13])` → `Gather(transformer.wte.weight,
  /transformer/Reshape_output_0)`. The plan (probe output) has that identity
  Reshape as instruction 2, output slot 382, and the `Gather` as instruction 4.
  In a resident session the Reshape output is device-resident; `Gather` is
  documented to validate index values on the host. The tiny graph is unaffected
  because its `Gather` consumes the host `input_ids` graph input directly.
* **Why the staged path works:** every intermediate is host memory, so the
  index read succeeds.
* **Why a kernel-only download is not sufficient:** a resident step records the
  whole plan into one batched submission (`begin_step`/`end_step`); at the time
  `Gather` records, the producer `Reshape` has not executed. A host read of its
  output would be stale even if it were downloadable. The plan must not put a
  host-validated index chain on the device.

**Proposed patch (planner/session, not a kernel guess):**

1. **Primary:** fold the identity `Reshape` at plan time. When a surviving
   `Reshape` has the same resolved output shape as its input and is not a graph
   output, do not emit an instruction; point the output value's slot at the
   input slot. The slot table is built in
   `src/fn3/model/specialize.rs:1455-1496`; the instruction loop is
   `:1499-1538`. Restrict the fold to non-graph-output values, and record it in
   a comment/test because `Plan::slot_of` will no longer resolve the folded
   name. This removes the copy and leaves `Gather` reading the host
   `input_ids` slot.
2. **Fallback if aliasing is rejected:** mark every slot that is the index
   operand of a `Gather` (and any other host-validated op) as host-allocated in
   `StepRuntime::materialise`, and pass `batch = false` for a step whose plan
   contains an instruction-produced host index chain. That costs one submission
   per step on this graph and preserves correctness generally.
3. **Defensive:** in `launch_gather`, when `indices.is_device_resident()`,
   return a typed `UnsupportedOp` naming the planner contract ("Gather indices
   must be host-visible; the plan made them resident") instead of falling
   through to the generic `NotImplemented`. Do not silently `to_host()` inside
   a batched step.

**Regression cases:**

* A self-contained tiny synthetic graph `Reshape(input_ids, [1,13])` →
  `Gather(wte)` run on resident Vulkan must succeed. This does not need the
  trained model.
* `tests/fn3_trained_parity.rs` with `FN3_TRAINED_VULKAN=1` and no
  `FN3_VK_STAGED` must pass on the GTX 1080 Ti.
* A plan-level assertion that the wte `Gather`'s index slot is a `GraphInput`
  (or a host-designated slot), not an `InstructionOutput`, for the trained
  envelope.

Until the fix landed, the honest Vulkan arm was `FN3_VK_STAGED=1`; the resident
default was a stable-gate failure, not a skip. `14adee3` implements option 1
with the declared-shape and layout conditions described there, and keeps the
`Gather` bounds check.

## 7. Commands, invocations and outcomes

All model runs were serialised under `/tmp/opencode/fn3-measure.lock`; each
command printed `/proc/loadavg` and the `cargo`/`rustc` process count. The
background load from other owners' builds was 1.5-5.1 and is recorded in the
logs; no result here is a timing measurement.

| Command | Outcome |
| --- | --- |
| `git rev-parse HEAD` | `6389186d4599471c4cb3ebc1156a5c1b872cd423`, matches base |
| `sha256sum` / `stat` on the trained graph | 498,126,358 B, `42c1e92a...c7c0d`, matches lock |
| `python3 tools/quant/ort_reference.py --model ... --out ort-123` | ORT 1.23.2; both committed logits byte-identical |
| `cargo test --release --no-default-features --test fn3_quant_format trained_gpt2_fp32 -- --ignored --nocapture` | reproduced `1.640e-4 / 7.273e-5`, `9.537e-5 / 1.743e-5` |
| `cargo test --release --no-default-features --test fn3_trained_parity -- --ignored --nocapture` | CPU pass; full metrics in `parity-cpu.json` |
| `cargo test --release --no-default-features --test fn3_trained_layers -- --ignored --nocapture` | 173 tensors/phase, plan `8fedb78c...` prefill; 14.5 s |
| `python3 tools/diagnostics/.../extract_intermediates.py ...` | 173 intermediates; logits still equal committed bytes (`max_abs 0.0`) |
| `python3 tools/diagnostics/.../reference_float64.py ...` | 173 tensors/phase in float64 |
| `python3 tools/diagnostics/.../compare_layers.py ...` (three pairs) | `layers-fn3-vs-ort.json`, `layers-fn3-vs-f64.json`, `layers-ort-vs-f64.json` |
| `python3 tools/diagnostics/.../attribute_accumulation.py ...` | 79,872/79,872 bit-identical to sequential f32 |
| `python3 tools/diagnostics/.../summarize.py ...` | `summary.json`, `layer-table.md` |
| `FASTNN_FN3_ONNX_MODEL=tiny ... cargo test --release --test fn3_e2e` | tiny control `5.364e-7 / 1.001e-7`, all steps pass |
| `FN3_TRAINED_VULKAN=1 FN3_VK_STAGED=1 FN3_VK_DEVICE_UUID=... cargo test --release --features vulkan --test fn3_trained_parity` | staged Vulkan pass; `parity-vulkan-staged.json` |
| `FN3_TRAINED_VULKAN=1` (resident, no `FN3_VK_STAGED`) | **fails** with the `Gather` resident-index error above; fixed in `14adee3` |
| `cargo test --release --features vulkan --test fn3_vulkan_trained_probe -- --ignored --nocapture` | names `gather` and the resident `[1,13]` i64 index |
| `cargo fmt --check` | clean |
| `cargo clippy --release --no-default-features --test fn3_trained_parity --test fn3_trained_layers -- -D warnings` | clean |
| `cargo clippy --release --features vulkan --test fn3_trained_parity --test fn3_vulkan_trained_probe -- -D warnings` | clean |

### Skips and exact reasons

* The resident Vulkan parity arm **failed**, it was not skipped. The failure is
  §6.2 and is a stable-gate blocker.
* No check was run on CPU with `--no-default-features` beyond the release test
  builds above; the numerical path is feature-independent and the test is
  compiled in both configurations.
* The trained model is absent from a clean checkout by design. Every trained
  probe prints
  `SKIP: FN3_TRAINED_ONNX is not set; no per-layer/parity probe was attempted`
  and returns; the tiny `fn3_e2e` prints its own model-absent reason. Those
  skips are not counted as passes.
* Decode step 2 has no committed payload. Its reference is the ORT run in this
  report; the test checks its argmax against the manifest summary and writes the
  full logits for the Python metric.

## 8. Artifacts

Reduced outputs, committed:

| File | Contents |
| --- | --- |
| [`summary.json`](fn3-trained-fp32/summary.json) | every decision metric, both bounds, CPU/Vulkan, bound-scale comparison |
| [`layer-table.md`](fn3-trained-fp32/layer-table.md) | per-layer and logits tables |
| [`layers-fn3-vs-ort.json`](fn3-trained-fp32/layers-fn3-vs-ort.json) | 173 stages × 3 phases, all metrics |
| [`layers-fn3-vs-f64.json`](fn3-trained-fp32/layers-fn3-vs-f64.json) | fn3 against exact arithmetic |
| [`layers-ort-vs-f64.json`](fn3-trained-fp32/layers-ort-vs-f64.json) | ORT against exact arithmetic |
| [`accumulation.json`](fn3-trained-fp32/accumulation.json) | bit-equality and f64 comparisons for three projections |
| [`parity-cpu.json`](fn3-trained-fp32/parity-cpu.json) | the test's CPU report |
| [`parity-vulkan-staged.json`](fn3-trained-fp32/parity-vulkan-staged.json) | the test's staged Vulkan report |

The three logs that earlier revisions of this page linked — `tiny-e2e-cpu.log`,
`reproduce-existing-test.log` and `vulkan-resident-failure.log` — were produced
in a scratch directory and never committed, so the links are removed rather
than left dangling. The first two commands are in §7 and reproduce their
content; the third is preserved verbatim in §6.2.

Not committed (too large, and model-derived): the raw `.f32`/`.npy` tensors
(21 MB fn3 + 18 MB ORT + 35 MB float64) under
`/tmp/opencode/trained-fp32/{fn3-layers,ort-layers,f64-layers}`. They are
regenerated by `tools/diagnostics/trained-fp32/reproduce.sh`; the JSON contains
every metric computed from them.

## 9. Limitations

* ORT 1.23.2 was the local reference; the committed payloads were produced by
  ORT 1.32.0. 1.23.2 reproduces them byte for byte, so this is not a variable.
* The float64 arm is an independent reimplementation of the graph, not a
  committed fixture. It matches ORT to `rel_rms 1.27e-6` on the last logits and
  to 1e-7..1e-6 per stage; the common-mode difference is the expected f32
  rounding, not a formula difference.
* The per-layer probe executes the plan one instruction at a time through the
  same `CpuBackend`; it does not change the arithmetic, but it does keep every
  value alive, so it is not a memory benchmark.
* Vulkan was measured on the staged path because the resident path fails. The
  staged path is a real device execution (the kernels dispatch on the GTX 1080
  Ti), but its submission pattern is the alpha's, not #72's resident default.
* Wall-clock numbers are absent on purpose: #72 owns performance, the GPU clock
  is unpinned, and other owners were building during the runs.
