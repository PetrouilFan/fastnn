# fastnn Optimization Plan

Baseline (before fixes): yolo11n f32 ~1350ms on Ryzen 3700X (8C/16T)
Current: ~350ms
Target: match PyTorch MKL ~145ms

---

## Phase 0: AVX2 Direct Conv + im2col Fixes  ✅ DONE
**Speed impact: 1350ms → 350ms (74% reduction)**

### Phase 0a: Persistent view weight registration fix
- `build_persistent_prepared_weights` in `prepared.rs` called `register_quantized_slot` for ALL Conv2d weights, but this function rejects fp32 weights. Fixed by dispatching to `register_fp32_slot` for fp32.
- Fixed inconsistent benchmark timing (1340ms vs 6.3s), but no performance change.

### Phase 0b: im2col min/max scan over-iteration bug
- **Root cause**: `im2col_dispatch` iterated `col.iter()` over the FULL over-allocated thread-local buffer (3.6M elements, 14.7MB) instead of just the valid output (57K elements, 230KB). The kernel itself takes only 3μs.
- **Fix**: Precompute `h_out * w_out * c * kh * kw` and create a sub-slice `col[..col_elems]` for both the kernel and the min/max scan.
- **Result**: Depthwise im2col 150ms → 2-3ms. Per-group: 1.1ms → 50μs.

### Phase 0c: AVX2 direct_conv dispatch confirmed working
- `direct_conv3x3_f32` IS active for 3×3 stride-1 pad-1 groups=1 convs.
- Diagnostic: `[direct_conv3x3] AVX2 PATH ACTIVE` confirms.

### Key files changed
- `src/backend/cpu/im2col.rs` — fix min/max scan boundary
- `src/backend/prepared.rs` — fp32 weight registration for Conv2d

## Phase 1: Add direct_conv3x3 stride-2 support
**Files:** `direct_conv.rs`, `conv.rs`
- Current: stride-2 3×3 convs fall through to im2col+gemm (taking 40-50ms each)
- New: specialized AVX2 direct 3×3 stride-2 pad-1 kernel
- Dispatch in `conv2d_f32_im2col_gemm`: when kh=3, kw=3, stride=2, padding=1, dilation=1, groups=1
- Est: 30-50% of remaining time (targeting the 40-50ms stride-2 convs)

## Phase 2: matrixmultiply threading oversubscription
**Files:** `Cargo.toml`, `conv.rs`
- Remove `threading` from matrixmultiply features
- Add outer-loop parallelism via Rayon
- Est: 10-20% speedup

## Phase 3: Transpose weight layout [K,M]
**Files:** `conv_gemm.rs`, weight loading path
- Store weights as `[K, M]` row-major
- GEMM accesses both A and B with stride-1 in K dim
- Est: 20-30% GEMM speedup

## Phase 4: Custom AVX2 GEMM microkernel
**Files:** NEW `gemm_direct_avx2.rs`, `conv_gemm.rs`
- NR=8, MR=4 BLIS-style microkernel
- Fuse bias+activation
- Dispatch: small layers use custom, large use matrixmultiply
- Est: 2-4× GEMM improvement

## Phase 5: Fuse im2col with GEMM (small spatial)
**Files:** `direct_conv.rs`, `conv.rs`
- For spatial_size < threshold: direct compute without materializing col buffer
- Generalizes existing direct_conv3x3 to arbitrary kernel sizes
- Est: 20-50% for small-spatial layers

## Phase 6: Minor optimizations
- Packed ops: `OC_TILE 4→8` in `packed_conv.rs`
- Remove redundant zero-fill in `conv.rs:88`
- `assert!` → `debug_assert!` in `arena.rs`
- `aligned_packed_slice` → zero-copy view

---

## Test commands
```
maturin develop --release
python debug_bench3.py
python -m pytest tests/test_nn.py tests/test_onnx.py -x --no-header -q --tb=short
```
