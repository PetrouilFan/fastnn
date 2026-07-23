# YOLO11n quantization baseline — 2026-07-23

## Scope

This is a CPU model-level quality and latency baseline for fastnn quantization formats. Results use YOLO11n at 640×640 on an AMD Ryzen 7 3700X with one PyTorch CPU thread. Latency measures model forward only; preprocessing, NMS, and COCO evaluation are excluded.

The dtype labels do not all represent the same activation contract. Interpret them using the explicit contract column rather than as weight formats alone.

## Reproducibility

- Branch: `refactor/navigation-foundation`
- Correctness fixes: `6d7214b`, `9db025f8`
- Reporting contract labels: `88c7049`
- Dataset: COCO val2017 through the deterministic Hugging Face stream prefix
- Image size: 640×640
- Confidence threshold: 0.001
- NMS IoU threshold: 0.6
- COCO-50 report: `.hermes/baselines/yolo11n-coco50-all-dtypes-9db025f8.json`
- COCO-500 report: `.hermes/baselines/yolo11n-coco500-f32-u8-88c7049.json`

## Corrected COCO-50 matrix

| Dtype | Execution contract | mAP@0.5 | mAP@0.5:0.95 | Mean latency | Weight storage | Assessment |
|---|---|---:|---:|---:|---:|---|
| PyTorch | W32A32 | 0.6279 | 0.4739 | 112.7 ms | 10.0 MB | Reference |
| F32 | W32A32 | 0.6279 | 0.4739 | 94.2 ms | 10.1 MB | Exact mAP parity |
| U8 | WU8A32-unpack | 0.6308 | 0.4790 | 93.4 ms | 2.7 MB | Best compressed smoke result |
| F8 | WF8A32-unpack | 0.5984 | 0.4330 | 96.3 ms | 2.7 MB | Moderate quality loss |
| F8R | WF8RA32-unpack | 0.5642 | 0.3933 | 100.2 ms | 2.7 MB | Inferior to F8 on this gate |
| I4CB | WI4-codebook-A32-unpack | 0.3395 | 0.2288 | 98.3 ms | 1.4 MB | Significant quality loss |
| F4 | WF4A32-unpack | 0.0761 | 0.0374 | 93.3 ms | 1.4 MB | Unacceptable quality |
| I8 (original, invalid) | WI8A8-dynamic | 0.0109 | 0.0072 | 613.6 ms | 2.7 MB | Corrupt activation packing; superseded |
| U4 | WU4A32-unpack | 0.0005 | 0.0001 | 93.9 ms | 1.4 MB | Unacceptable quality |
| I4 | WI4A4-dynamic | 0.00005 | 0.000009 | 762.6 ms | 1.4 MB | Slow and inaccurate |

The signed I4 result is W4A4, not the W4A8 MatMul path. It must not be cited as evidence for or against production W4A8.

## COCO-500 U8 validation

| Runtime | Contract | mAP@0.5 | mAP@0.5:0.95 | Delta mAP@0.5:0.95 | Mean latency | Weight storage |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | W32A32 | 0.5804 | 0.4211 | — | 109.4 ms | 10.0 MB |
| fastnn F32 | W32A32 | 0.5804 | 0.4211 | +0.0000 | 97.1 ms | 10.1 MB |
| fastnn U8 | WU8A32-unpack | 0.5767 | 0.4180 | -0.0031 | 99.7 ms | 2.7 MB |

On 500 images, U8 retains 99.26% of the F32 mAP@0.5:0.95 while reducing recorded weight storage by 73.4%. It is 1.10× faster than PyTorch but 2.6% slower than fastnn F32. The current U8 execution unpacks/dequantizes weights into a cached F32 view, so the storage reduction is real but this is not an integer U8 compute-speed result.

## COCO-500 corrected I8 validation

| Runtime | Contract | mAP@0.5 | mAP@0.5:0.95 | Delta mAP@0.5:0.95 | Mean latency | Weight storage |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | W32A32 | 0.5804 | 0.4211 | — | 109.7 ms | 10.0 MB |
| fastnn F32 | W32A32 | 0.5804 | 0.4211 | +0.0000 | 97.1 ms | 10.1 MB |
| fastnn I8 | WI8A8-dynamic | 0.5740 | 0.4162 | -0.0050 | 719.7 ms | 2.7 MB |

Corrected I8 retains 98.84% of F32 mAP@0.5:0.95 on 500 images. Its quality is therefore valid, but the scalar dynamic-quantization path is 7.41× slower than fastnn F32 and is not a performance candidate until a numerically equivalent SIMD implementation wins.

## Correctness fixes discovered by the gate

1. Unsigned asymmetric quantization used a signed code origin. U4/U8 offsets now start at the unsigned code origin. This removed full-output non-finite propagation and restored U8 quality.
2. The process-global packed-weight cache was not cleared between executor instances. Reused arena pointers could make I4CB consume stale I4 payloads. Both packed and dequantized caches are now cleared between executors.
3. The I8 im2col packer wrote scalar tail words beyond a pre-sized vector and its AVX2 path applied unsigned saturation to signed bytes. Correct scalar packing restores COCO-50 quality to 0.6277 mAP@0.5 and 0.4766 mAP@0.5:0.95. Corrected latency is 732.5 ms/image, so the current W8A8 path is accurate but not performant.

## First-divergent-layer audit

`scripts/yolo_dtype_audit.py` retains all 88 YOLO Conv outputs and compares each dtype with fastnn F32 on the same input. At a normalized-RMSE threshold of 10%:

| Dtype | First divergent Conv | Normalized RMSE |
|---|---:|---:|
| U8 | 33 | 10.20% |
| F8 | 10 | 13.26% |
| F8R | 1 | 11.63% |
| I8 before packing fix | 0 | 52.10% |
| I8 after packing fix | 12 | 12.62% |
| U4 | 0 | 21.97% |
| I4/W4A4 | 0 | 22.83% |
| F4 | 0 | 34.90% |
| I4CB | 0 | 20.49% |

Model-shaped differential tests now verify that direct I4 and I8 Conv outputs match independent convolution over the exact dequantized activation and weight operands, including K=27 packed tails. I4 remains inaccurate because the current model path dynamically quantizes activations to four bits; this does not evaluate W4A8. I8 is accurate after fixing its packer but remains slow because the corrupt AVX2 branch was removed in favor of scalar correctness.

## Current promotion decisions

- U8 is the compressed quality baseline for broader validation.
- F8 remains a functional lower-quality comparison.
- U4, F4, I4, I8, and I4CB are not promotion candidates at current quality.
- W4A8 YOLO remains unmeasured. The existing calibration script is stale and the Python `AotExecutor` currently exposes `apply_calibration()` but no calibration-collection API.

## Remaining gates

1. Validate U8 on full COCO val2017 before declaring a release threshold.
2. Record peak resident memory in addition to structural arena/weight accounting.
3. Restore a supported activation-calibration collection API.
4. Run explicit W4A8 Conv/YOLO parity and COCO gates.
5. Compare persistent packed execution without a cached F32 weight expansion before making integer-compute performance claims.
