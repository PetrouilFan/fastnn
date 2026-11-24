# AArch64: Raspberry Pi 5

- **Board:** Raspberry Pi 5 Model B Rev 1.1, 4 x Cortex-A76, 7 GB, microSD root
- **CPU features:** `asimd`, `asimddp` (dot-product), no `i8mm`, no `sve`
- **OS:** Debian, kernel `6.18.34+rpt-rpi-2712`, gcc 14.2.0, cmake 3.31.6
- **Toolchain:** rustc 1.100.0 (rustup, stable, `--profile minimal`)
- **Model:** `yolov8n.onnx`, input `f32[1,3,320,320]`
- **ORT:** 1.32.0 from PyPI, the same version as the x86 baseline
- **Governor:** `performance` for every number here
- **Protocol:** identical to [per-layer.md](per-layer.md) — 3 warm-ups, then 3
  groups of 3 forwards per thread count, median of the groups.

This is the first time this tree has been built or measured on a non-x86 target.
The result is a negative one and it is recorded in full, because it changes
which backend work is worth doing.

## Headline

| threads | ORT | fastnn oneDNN | ratio | fastnn scalar |
|---:|---:|---:|---:|---:|
| 1 | 79.26 ms | 160.00 ms | **2.02x** | 2020.19 ms |
| 2 | — | 95.49 | — | 1058.57 |
| 4 | 31.83 | 72.81 | **2.29x** | 576.24 |
| 8 | — | 79.57 | — | 585.48 |

Four physical cores, no SMT: both sides flatten at four and are worse at eight.
The provider scales 1.68x at two workers and 2.20x at four; the scalar path
scales better (1.91x and 3.51x) but starts 12.6x behind, so it is not the
alternative.

## The scalar reference path

2020.19 / 1058.57 / 576.24 / 585.48 ms. It is 25x slower than ORT, which is the
reference path doing its job — it is the correctness oracle, not the product.

### The scalar kernels are as wide on NEON as on AVX2

Per layer, the Pi is a median **1.75x** slower than the Ryzen on the same scalar
convolution (range 1.39-2.00 across all 64 conv segments). The clock ratio is
about 1.8x (2.4 GHz against 4.2 GHz) and AVX2 is twice as wide as NEON, so a
kernel using the full width of both would be ~3.6x slower here. Being 1.75x
slower means the x86 scalar path is not getting eight lanes out of AVX2 either:
both targets auto-vectorise to about four.

## ONNX Runtime's baseline is MLAS, not ACL

| threads | conv | QuickGelu | total |
|---:|---:|---:|---:|
| 1 | 68.82 ms | 6.88 | **79.26** |
| 4 | 24.04 | 4.14 | **31.83** |

Two things worth recording. Convolution plus its fused SiLU is 95 % of the
one-thread forward here against 92 % on x86. And the aarch64 wheel is **not**
built with Arm Compute Library: `strings` on
`onnxruntime_pybind11_state.cpython-313-aarch64-linux-gnu.so` finds no
`ArmCompute`, `AclConvolution` or `acl_gemm` symbols, and the profile's node
names carry no `_nchwc_` infix. ORT's 15.9 GMAC/s comes from MLAS's hand-written
NEON kernels.

## The provider is 2.24x behind, and it is the kernel, not the engine

Matched over all 64 convolutions at one thread: **153.9 ms against ORT's
68.8 ms**, i.e. 2.24x, running at **7.1 GMAC/s against 15.9**. ORT is at ~83 % of
this core's f32 peak (2 FMA pipes x 4 lanes x 2.4 GHz = 19.2 GMAC/s); we are at
37 %. The loss is uniform, not concentrated in one class:

| layer | fastnn | ORT | ratio | GMAC/s ours | ORT |
|---|---:|---:|---:|---:|---:|
| `model.22/cv3.0/cv3.0.1` 80x80 3x3 40x40 | 13.83 ms | 5.14 | 2.69x | 6.7 | 17.9 |
| `model.22/cv3.0/cv3.0.0` | 11.15 | 4.15 | 2.69x | 6.6 | 17.8 |
| `model.22/cv2.0/cv2.0.0` | 8.91 | 3.43 | 2.60x | 6.6 | 17.2 |
| `model.3/conv` 64->64 40x40 | 4.73 | 1.94 | 2.44x | 6.2 | 15.2 |
| `model.1/conv` 32->32 80x80 | 3.40 | 2.38 | 1.43x | 8.7 | 12.4 |

The engine is not the cause. `probe_int8_speed` (an ignored `ffi` unit test that
builds one convolution directly against oneDNN, with no plan and no blocks)
reproduces the same throughput on this board:

| isolated conv, f32, one thread | ms | GMAC/s |
|---|---:|---:|
| 32x80x80, 32 out, 3x3 | 5.565 | 10.6 |
| 64x40x40, 64 out, 3x3 | 8.604 | 6.9 |
| 128x20x20, 128 out, 3x3 | 4.925 | 12.0 |
| 32x80x80, 32 out, 1x1 | 0.368 | 17.9 |

The same probe on the Ryzen runs ~59M-MAC shapes in 0.94-1.07 ms, i.e. ~55
GMAC/s, about 80 % of that core's peak. oneDNN's aarch64 f32 path is `jit_brgemm_conv`;
it reaches 35-60 % of this core's peak where MLAS reaches 83-93 %. **The provider
choice does not transfer from x86 to AArch64.**

## int8 has no home here either, despite `sdot`

The board reports `asimddp`, so it can execute `sdot`. It does not help: the
same probe's int8 paths are *4-6.5x slower* than f32.

| isolated conv, one thread | f32 blocked | int8 NHWC | int8 NCHW |
|---|---:|---:|---:|
| 32x80x80, 32 out, 3x3 | 5.565 ms | 23.611 | 1196.6 |
| 64x40x40, 64 out, 3x3 | 8.604 | 25.206 | 1177.7 |
| 128x20x20, 128 out, 3x3 | 4.925 | 32.277 | 1146.0 |
| 32x80x80, 32 out, 1x1 | 0.368 | 4.486 | 161.4 |

oneDNN 3.14.1's only aarch64 int8 convolution kernels are the SVE ones
(`jit_sve_512_x8s8s32x_convolution`) plus whatever KleidiAI exposes; the build
does compile KleidiAI (`src/cpu/aarch64/kleidiai_ops/kai/ops/gemm` appears in the
build log), but it is not reachable for these convolution configurations on a
non-SVE core. So hardware dot-product is not the same thing as a dot-product
kernel: **W8A8 would not pay on this board, and enabling it here would make the
product slower.**

## What this means

1. **Neither side uses ACL.** ORT's aarch64 wheel is MLAS; our oneDNN build is
   `AARCH64_USE_ACL=OFF` (the default, and `onednn-src` exposes no knob for it).
   ACL is the only path either side has to close its own gap, and it needs a
   vendored build with a prebuilt ACL >= 53.1 plus an accuracy gate because ACL
   enables fast-math. That is a project, not a flag.
2. **int8-on-ARM is closed as a performance idea** for this oneDNN version. The
   negative is measured on real hardware with the exact kernels that ship.
3. **AArch64 support stays correctness-first.** The scalar path is the portable
   one; the provider works and is 12.6x faster than scalar but is 2.2x behind
   ORT on f32 and worse on int8.
4. **Do not build ARM performance infrastructure** (custom NEON kernels, ACL
   wiring, ARM CI perf lanes) until something in (1) is funded. The x86 roadmap
   is unaffected.

## Portability defects this exercise found

- `simd::avx2_available` was dead code on every non-x86 target and warned;
  now `#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]`.
- `cargo test --lib --features cpu-onednn` **failed to link** on aarch64:
  Debian's `--as-needed` dropped the shared C++ runtime because it appears
  before the static oneDNN archive, leaving `operator new` and
  `__class_type_info` undefined. `build.rs` now names `-lstdc++` again at the
  end of the link line with `--no-as-needed`, for binaries and test targets.

## Reproduction

```bash
# on the board, from a synced tree
cargo build --release --no-default-features --features parallel,segment-timing \
    --test fn3_cnn_perf
FN3_CNN_PERF=onednn FN3_CNN_THREADS=4 cargo test --release \
    --no-default-features --features cpu-onednn,parallel,segment-timing \
    --test fn3_cnn_perf -- --nocapture

# the isolated convolution probe (no plan, no blocks, no engine)
cargo test --release --lib --no-default-features --features cpu-onednn \
    -- probe_int8_speed --ignored --nocapture

# ORT baseline, same version and protocol as x86
~/venv/bin/python tools/parity/ort_profile.py ~/.hermes/yolov8n.onnx \
    --threads 4 --runs 3 --json-out /tmp/ort-4t.json
```

The build takes 15 minutes on the board (`onednn-src` compiles oneDNN plus
KleidiAI); subsequent Rust-only rebuilds are about a minute.
