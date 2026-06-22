# Fastnn U4/U8 Verification

## Working

- U4/U8 calibration + weight-packing pass compiles and runs in the test suite (270/270).
- YOLOv8n CPU YOLO inference completes without errors, and the new activation-quantization
  path now covers `Conv2d` as well as `MatMul`.
- Wired the `u4` and `u8` flags in the Python/ONNX side so the runtime can pick
  bit-width-specific paths.

## Not Working Yet

- Numeric distance against PyTorch is still ~210 max_abs with U8 and ~210 with U4.
- `activation_quantization` coverage alone is not enough: the runtime still has
  to actually dequant weights and execute SWAR kernels for `conv2d_u4`/`conv2d_u8`.

## Structural Gap

`PreparedConvKernelKind::FuturePackedI8/U4` for `conv2d_u4`/`conv2d_u8` means the
compiled plan is not guaranteed to call `gemm_packed_u8x4_fused`/`gemm_packed_u4x8_fused`.
There’s a bridge missing between the ONNX graph and the SWAR dispatch.

## Next Branch Plan (`yolo/u8-u4-rewrite`)

1. Map `conv2d_u4`/`conv2d_u8` instructions to the SWAR kernels in `prepared.rs`.
2. Ensure weight metadata includes symmetric/as zp/scale so dequantization inside
   the fused GEMM is satisfied.
3. Stop comparing raw logits vs PyTorch; use YOLO validation metrics (mAP, precision,
   recall) on a calibration-sized COCO eval run once data is available.
4. Regression benchmark: 5 seeds on 320×320, compare accuracy primary and latency
   secondary.

## Artifacts

- Benchmark scratch: `scripts/yolo_compare_fastnn_pytorch.py`, `scripts/yolo_quant_fast_triplet.py`
- Calibration scratch: `scripts/calibration_dataset.py`, `scripts/calibrate_yolo.py`
- Dump: `results/yolo_quant.json`, `results/yolo_openblas.json`, `results/yolo_openblas_2t.json`
