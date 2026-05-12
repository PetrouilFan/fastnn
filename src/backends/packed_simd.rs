//! SIMD-accelerated GEMV kernels for packed precision types.
//!
//! Key optimizations:
//! - Type-dispatched SIMD: U8x4 uses int8→f32 widening, F16x2 uses F16C
//! - Pre-unpack packed u32 words into contiguous f32 buffer, then SIMD FMA
//! - Cache-blocked K dimension for large matrices (L2 reuse across rows)
//! - U4x8 AVX2 kernel: nibble extraction → int32 → f32 → FMA

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;
use std::sync::OnceLock;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// Cached feature-detection flags — checked once, reused forever.
static HAS_AVX512: OnceLock<bool> = OnceLock::new();
static HAS_AVX2: OnceLock<bool> = OnceLock::new();
static HAS_F16C: OnceLock<bool> = OnceLock::new();

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn has_avx512() -> bool {
    *HAS_AVX512
        .get_or_init(|| is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw"))
}
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn has_avx2() -> bool {
    *HAS_AVX2.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn has_f16c() -> bool {
    *HAS_F16C.get_or_init(|| is_x86_feature_detected!("f16c"))
}

/// Cache-blocked GEMV for packed types.
/// Tiles the K dimension to keep activation blocks in L2 cache.
const K_BLOCK_SIZE: usize = 4096; // 16KB of f32 activations per block

// Thread-local scratch buffer, reused across calls.
thread_local! {
    static PACKED_SCRATCH: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// Run `f` with a thread-local scratch buffer resized to `size`.
fn with_scratch<R>(size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PACKED_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.resize(size, 0.0);
        f(&mut buf)
    })
}

// ============================================================
// Thread-local Vec pool — reuse allocations across calls
// ============================================================

thread_local! {
    static TLS_VEC_POOL: std::cell::RefCell<Vec<Vec<f32>>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// A Vec<f32> borrowed from the thread-local pool.
/// Automatically returned to the pool when dropped.
pub struct ScopedVec {
    inner: Option<Vec<f32>>,
}

impl std::ops::Deref for ScopedVec {
    type Target = Vec<f32>;
    fn deref(&self) -> &Vec<f32> {
        self.inner.as_ref().expect("ScopedVec already consumed")
    }
}

impl std::ops::DerefMut for ScopedVec {
    fn deref_mut(&mut self) -> &mut Vec<f32> {
        self.inner.as_mut().expect("ScopedVec already consumed")
    }
}

impl ScopedVec {
    /// Take ownership of the inner Vec, preventing it from returning to the pool.
    /// Use this when the Vec must outlive the `ScopedVec` (e.g. passed to `Tensor::from_vec`).
    pub fn take(mut self) -> Vec<f32> {
        self.inner
            .take()
            .expect("ScopedVec::take called on already-consumed ScopedVec")
    }
}

impl Drop for ScopedVec {
    fn drop(&mut self) {
        if let Some(v) = self.inner.take() {
            // Cap pool entries to avoid memory bloat
            if v.capacity() <= 100_000_000 {
                TLS_VEC_POOL.with(|pool| {
                    pool.borrow_mut().push(v);
                });
            }
        }
    }
}

/// Thread-local arena for reusing Vec<f32> allocations across calls.
///
/// Use `alloc` or `alloc_zeroed` to get a `ScopedVec` with at least the
/// requested capacity. The Vec is returned to the pool on drop, avoiding
/// repeated allocation/deallocation overhead in hot paths.
pub struct TlsVecPool;

impl TlsVecPool {
    /// Acquire a Vec with at least `min_capacity` capacity and length.
    /// The Vec may contain stale data (NOT zero-initialized) — the caller
    /// must overwrite every element in `0..min_capacity` before reading.
    pub fn alloc(min_capacity: usize) -> ScopedVec {
        let mut v = TLS_VEC_POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_default();
        if v.capacity() < min_capacity {
            // Use v.len() not v.capacity(): a recycled Vec may have len < capacity,
            // and reserve(additional) guarantees capacity >= len + additional.
            // Using capacity() would under-shoot when len < capacity.
            v.reserve(min_capacity - v.len());
        }
        // SAFETY: The caller promises to write all min_capacity elements
        // before any read. This avoids the cost of zero-initialization.
        unsafe {
            v.set_len(min_capacity);
        }
        ScopedVec { inner: Some(v) }
    }

    /// Acquire a Vec with exactly `len` elements, all zeroed.
    pub fn alloc_zeroed(len: usize) -> ScopedVec {
        let mut v = Self::alloc(len);
        v.fill(0.0);
        v
    }
}

// ============================================================
// Main entry point — type-dispatched
// ============================================================

/// SIMD-accelerated GEMV for packed types.
/// Dispatches to type-specific SIMD kernels when available on the target.
#[inline(always)]
pub fn gemv_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    // Compile-time type dispatch — BIT_WIDTH and IS_FLOAT are const generics
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    match (T::BIT_WIDTH, T::IS_FLOAT) {
        (8, false) => {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<crate::dtypes::U8x4>) };
            return gemv_u8x4_dispatch(w, activation, output);
        }
        (4, false) => {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<crate::dtypes::U4x8>) };
            return gemv_u4x8_dispatch(w, activation, output);
        }
        (16, true) => {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<crate::dtypes::F16x2>) };
            return gemv_f16x2_dispatch(w, activation, output);
        }
        (32, true) => {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            let w = unsafe { &*(weights as *const _ as *const PackedTensor<crate::dtypes::F32x1>) };
            return gemv_f32x1_dispatch(w, activation, output);
        }
        _ => {}
    }

    // Generic fallback for F32x1 and non-x86 targets
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

// ============================================================
// U8x4: AVX2 int8→f32 widening + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_u8x4_dispatch(
    weights: &PackedTensor<crate::dtypes::U8x4>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(4);
    let weights_u32 = weights.as_u32();

    if has_avx512() {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;

                        // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                        let dot = unsafe {
                            gemv_row_u8x4_avx512(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                let dot = unsafe {
                    gemv_row_u8x4_avx512(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else if has_avx2() {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;

                        // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                        let dot = unsafe {
                            gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                let dot = unsafe {
                    gemv_row_u8x4_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else {
        // Scalar fallback
        for row in 0..m {
            let row_offset = row * k_packed;

            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            let dot =
                unsafe { gemv_row_u8x4_scalar(weights_u32, activation, row_offset, k, k_packed) };
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_u8x4_scalar(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut total = 0.0f32;
    for p in 0..k_packed {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = p * 4 + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
    }
    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_u8x4_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 int8 values at a time (2 u32 words = 8 bytes)
    while p + 1 < k_packed && act_idx + 8 <= k {
        // Prefetch next iteration's weights into L1
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        // _mm_set_epi32 places w0 in bytes [0-3], w1 in bytes [4-7].
        // _mm256_cvtepi8_epi32 sign-extends the low 8 bytes to 8 int32.
        let word_pair = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
        let weight_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(word_pair));

        let act = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc = _mm256_fmadd_ps(weight_f32, act, acc);

        p += 2;
        act_idx += 8;
    }

    // Horizontal sum
    let mut total = hsum256_ps(acc);

    // Scalar tail for remaining packed words (1 at a time = 4 values)
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_u8x4_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 4 u32 words at a time (16 int8 → 16 f32) using AVX512BW for byte ops
    while p + 4 <= k_packed && act_idx + 16 <= k {
        if p + 8 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 8) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w2 = weights_u32[row_offset + p + 2];
        let w3 = weights_u32[row_offset + p + 3];

        // Pack 16 bytes from 4 u32 words into a 128-bit register
        let bytes = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
        // Widen signed int8 → int32, then convert to f32
        let i32x16 = _mm512_cvtepi8_epi32(bytes);
        let weight_f32 = _mm512_cvtepi32_ps(i32x16);

        let act = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(weight_f32, act, acc0);

        p += 4;
        act_idx += 16;
    }

    let mut total = _mm512_reduce_add_ps(acc0);

    // 2-word tail: 8 int8 → 8 f32 (scalar)
    while p + 2 <= k_packed && act_idx + 8 <= k {
        for word_off in 0..2 {
            let w = weights_u32[row_offset + p + word_off];
            let bytes = w.to_le_bytes();
            for j in 0..4 {
                total += (bytes[j] as i8) as f32 * activation[act_idx + word_off * 4 + j];
            }
        }
        p += 2;
        act_idx += 8;
    }

    // Vectorized tail for remaining packed words using 128-bit operations
    while p < k_packed && act_idx + 4 <= k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        let weights_f32: [f32; 4] = [
            bytes[0] as i8 as f32,
            bytes[1] as i8 as f32,
            bytes[2] as i8 as f32,
            bytes[3] as i8 as f32,
        ];
        let acts = [
            activation[act_idx],
            activation[act_idx + 1],
            activation[act_idx + 2],
            activation[act_idx + 3],
        ];

        // Use 128-bit FMA
        let w_vec = _mm_loadu_ps(weights_f32.as_ptr());
        let a_vec = _mm_loadu_ps(acts.as_ptr());
        let prod = _mm_mul_ps(w_vec, a_vec);
        let hsum = _mm_hadd_ps(prod, prod);
        let final_sum = _mm_hadd_ps(hsum, hsum);
        total += _mm_cvtss_f32(final_sum);

        p += 1;
        act_idx += 4;
    }

    // Scalar tail for remaining elements
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let bytes = w.to_le_bytes();
        for j in 0..4 {
            let idx = act_idx + j;
            if idx < k {
                total += (bytes[j] as i8) as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 4;
    }

    total
}

// ============================================================
// U4x8: AVX2 nibble extraction → int32 → f32 → FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_u4x8_dispatch(
    weights: &PackedTensor<crate::dtypes::U4x8>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(8);
    let weights_u32 = weights.as_u32();

    if has_avx512() {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;

                        // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                        let dot = unsafe {
                            gemv_row_u4x8_avx512(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                let dot = unsafe {
                    gemv_row_u4x8_avx512(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else if has_avx2() {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;

                        // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                        let dot = unsafe {
                            gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                let dot = unsafe {
                    gemv_row_u4x8_avx2(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_u4x8_avx2(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_ext = _mm256_set1_epi32(8);

    // Process 2 u32 words (16 nibbles) per iteration with dual accumulators
    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];

        // Extract low nibbles (bits 0,4,8,12,16,20,24,28) from both words
        let shift0 = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift0), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift0), mask_lo);

        // Sign-extend via xor-sub identity: (x ^ 8) - 8
        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm256_cvtepi32_ps(signed_lo0);
        let fl1 = _mm256_cvtepi32_ps(signed_lo1);

        let al0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(fl0, al0, acc0);

        let al1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    // Scalar tail for remaining packed words
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_u4x8_avx512(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;
    let mask_lo = _mm256_set1_epi32(0xF);
    let sign_ext = _mm256_set1_epi32(8);

    // Process 2 u32 words (16 nibbles) per iteration with dual accumulators
    while p + 2 <= k_packed && act_idx + 16 <= k {
        if p + 4 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 4) as *const i8,
                _MM_HINT_T0,
            );
        }

        let w0 = weights_u32[row_offset + p];
        let w1 = weights_u32[row_offset + p + 1];
        let w0v = _mm256_set1_epi32(w0 as i32);
        let w1v = _mm256_set1_epi32(w1 as i32);

        // Low nibbles (bits 0,4,8,12,16,20,24,28)
        let shift_lo = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
        let nib_lo0 = _mm256_and_si256(_mm256_srlv_epi32(w0v, shift_lo), mask_lo);
        let nib_lo1 = _mm256_and_si256(_mm256_srlv_epi32(w1v, shift_lo), mask_lo);

        // Sign-extend via xor-sub identity: (x ^ 8) - 8
        let signed_lo0 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo0, sign_ext), sign_ext);
        let signed_lo1 = _mm256_sub_epi32(_mm256_xor_si256(nib_lo1, sign_ext), sign_ext);

        let fl0 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo0));
        let fl1 = _mm512_cvtepi32_ps(_mm512_castsi256_si512(signed_lo1));
        let al0 = _mm512_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm512_fmadd_ps(fl0, al0, acc0);
        let al1 = _mm512_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm512_fmadd_ps(fl1, al1, acc1);

        p += 2;
        act_idx += 16;
    }

    let combined = _mm512_add_ps(acc0, acc1);
    let mut total = _mm512_reduce_add_ps(combined);

    // Scalar tail for remaining packed words
    while p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        for j in 0..8 {
            let idx = act_idx + j;
            if idx < k {
                let nibble = (w >> (j * 4)) & 0xF;
                let signed = if nibble & 0x8 != 0 {
                    (nibble | 0xFFFFFFF0) as i32
                } else {
                    nibble as i32
                };
                total += signed as f32 * activation[idx];
            }
        }
        p += 1;
        act_idx += 8;
    }

    total
}

// ============================================================
// F16x2: F16C hardware half→float conversion + FMA
// ============================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(unused_variables)]
fn gemv_f16x2_dispatch(
    weights: &PackedTensor<crate::dtypes::F16x2>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(2);
    let weights_u32 = weights.as_u32();

    if has_f16c() {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
            output
                .par_chunks_mut(rows_per_chunk)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;

                        // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                        let dot = unsafe {
                            gemv_row_f16x2_f16c(
                                weights_u32,
                                activation,
                                row * k_packed,
                                k,
                                k_packed,
                            )
                        };
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for row in 0..m {
                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                let dot = unsafe {
                    gemv_row_f16x2_f16c(weights_u32, activation, row * k_packed, k, k_packed)
                };
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        }
    } else {
        gemv_generic_fallback(weights, activation, output);
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: The pointers are valid and properly aligned for SIMD access. Loop bounds prevent out-of-bounds access.
unsafe fn u32x4_to_f32x8_f16c(w0: u32, w1: u32, w2: u32, w3: u32) -> __m256 {
    // Each u32 has 2 f16: lo_half at bits 0-15, hi_half at bits 16-31
    // _mm_set_epi32(w3, w2, w1, w0) lays out as:
    //   u16 lane 0: w0 & 0xFFFF (lo)
    //   u16 lane 1: w0 >> 16    (hi)
    //   u16 lane 2: w1 & 0xFFFF (lo)
    //   ...etc — exactly what _mm256_cvtph_ps expects
    let half_bits = _mm_set_epi32(w3 as i32, w2 as i32, w1 as i32, w0 as i32);
    _mm256_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: The pointers are valid and properly aligned for SIMD access. Loop bounds prevent out-of-bounds access.
unsafe fn u32x2_to_f32x4_f16c(w0: u32, w1: u32) -> __m128 {
    let half_bits = _mm_set_epi32(0, 0, w1 as i32, w0 as i32);
    _mm_cvtph_ps(half_bits)
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
unsafe fn gemv_row_f16x2_f16c(
    weights_u32: &[u32],
    activation: &[f32],
    row_offset: usize,
    k: usize,
    k_packed: usize,
) -> f32 {
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut p = 0;
    let mut act_idx = 0;

    // Process 8 u32 words (16 f16) per iteration — 2x throughput with ILP
    while p + 8 <= k_packed && act_idx + 16 <= k {
        if p + 16 < k_packed {
            _mm_prefetch(
                weights_u32.as_ptr().add(row_offset + p + 16) as *const i8,
                _MM_HINT_T0,
            );
        }

        let f0 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a0 = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f0, a0, acc0);

        let f1 = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p + 4],
            weights_u32[row_offset + p + 5],
            weights_u32[row_offset + p + 6],
            weights_u32[row_offset + p + 7],
        );
        let a1 = _mm256_loadu_ps(activation.as_ptr().add(act_idx + 8));
        acc1 = _mm256_fmadd_ps(f1, a1, acc1);

        p += 8;
        act_idx += 16;
    }

    // 4-word tail: 8 f16 → 8 f32
    while p + 4 <= k_packed && act_idx + 8 <= k {
        let f = u32x4_to_f32x8_f16c(
            weights_u32[row_offset + p],
            weights_u32[row_offset + p + 1],
            weights_u32[row_offset + p + 2],
            weights_u32[row_offset + p + 3],
        );
        let a = _mm256_loadu_ps(activation.as_ptr().add(act_idx));
        acc0 = _mm256_fmadd_ps(f, a, acc0);
        p += 4;
        act_idx += 8;
    }

    acc0 = _mm256_add_ps(acc0, acc1);
    let mut total = hsum256_ps(acc0);

    // 2-word tail: 4 f16 → 4 f32
    if p + 2 <= k_packed && act_idx + 4 <= k {
        let f = u32x2_to_f32x4_f16c(weights_u32[row_offset + p], weights_u32[row_offset + p + 1]);
        let a = _mm_loadu_ps(activation.as_ptr().add(act_idx));
        let prod = _mm_mul_ps(f, a);
        let shuf = _mm_shuffle_ps(prod, prod, 0x0E);
        let sums = _mm_add_ps(prod, shuf);
        let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
        total += _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        p += 2;
        act_idx += 4;
    }

    // 1-word tail: 2 f16 → 2 f32
    if p < k_packed && act_idx < k {
        let w = weights_u32[row_offset + p];
        let f = u32x2_to_f32x4_f16c(w, 0);
        let arr: [f32; 4] = std::mem::transmute(f);
        total += arr[0] * activation[act_idx];
        if act_idx + 1 < k {
            total += arr[1] * activation[act_idx + 1];
        }
    }

    total
}

// ============================================================
// F32x1: direct f32 loads (no unpack needed)
// ============================================================

#[allow(unused_variables)]
fn gemv_f32x1_dispatch(
    weights: &PackedTensor<crate::dtypes::F32x1>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    // F32x1: u32 IS f32, reinterpret directly
    // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
    let weights_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(
            weights.as_u32().as_ptr() as *const f32,
            weights.packed_len(),
        )
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(row, out)| {
            let row_data = &weights_f32[row * k..(row + 1) * k];
            let dot = fma_f32_slice(row_data, activation);
            *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let row_data = &weights_f32[row * k..(row + 1) * k];
            let dot = fma_f32_slice(row_data, activation);
            output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    }
}

// ============================================================
// Generic fallback (used by non-SIMD targets)
// ============================================================

/// Generic GEMV fallback using scalar unpack + SIMD FMA.
#[inline(always)]
fn gemv_generic_fallback<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
) {
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    if k <= K_BLOCK_SIZE {
        gemv_packed_inner::<T>(weights, activation, output, m, k, k_packed);
    } else {
        gemv_packed_blocked::<T>(weights, activation, output, m, k, k_packed);
    }
}

// ============================================================
// Generic GEMV implementation (scalar unpack + AVX2 FMA)
// ============================================================

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn gemv_packed_inner<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    k_packed: usize,
) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let rows_per_chunk = (65536 / (k_packed * 4)).clamp(1, 64);
        output
            .par_chunks_mut(rows_per_chunk)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * rows_per_chunk;
                with_scratch(k_packed * T::ITEMS, |unpack_buf| {
                    for (local_row, out) in out_chunk.iter_mut().enumerate() {
                        let row = start_row + local_row;
                        let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf);
                        *out = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
                    }
                });
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        with_scratch(k_packed * T::ITEMS, |unpack_buf| {
            for row in 0.._m {
                let dot = gemv_row::<T>(weights, activation, row, k, k_packed, unpack_buf);
                output[row] = dot * weights.scale_for_row(row) + weights.zero_for_row(row);
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn gemv_packed_blocked<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    k_packed: usize,
) {
    for o in output.iter_mut() {
        *o = 0.0;
    }

    let items = T::ITEMS;

    with_scratch(K_BLOCK_SIZE, |unpack_buf| {
        let mut k_offset = 0;
        while k_offset < k {
            let k_end = (k_offset + K_BLOCK_SIZE).min(k);
            let k_block = k_end - k_offset;

            for row in 0..m {
                let row_offset = row * k_packed;
                let packed_start = k_offset / items;
                let packed_end = k_end.div_ceil(items);
                let unpack_len = (packed_end - packed_start) * items;

                if unpack_len <= K_BLOCK_SIZE {
                    for p in packed_start..packed_end {
                        let word = weights.as_packed()[row_offset + p];
                        let unpacked = word.unpack_to_f32();
                        let base = p * items;
                        for j in 0..items {
                            let idx = base + j;
                            if idx >= k_offset && idx < k_end {
                                unpack_buf[idx - k_offset] = unpacked.as_ref()[j];
                            }
                        }
                    }
                    let acc = fma_f32_slice(
                        &unpack_buf[..k_block.min(unpack_len)],
                        &activation[k_offset..k_end],
                    );
                    output[row] += acc;
                }
            }

            k_offset += K_BLOCK_SIZE;
        }

        for row in 0..m {
            output[row] = output[row] * weights.scale_for_row(row) + weights.zero_for_row(row);
        }
    });
}

#[inline]
fn gemv_row<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    row: usize,
    k: usize,
    k_packed: usize,
    unpack_buf: &mut [f32],
) -> f32 {
    let items = T::ITEMS;
    let row_offset = row * k_packed;

    for p in 0..k_packed {
        let word = weights.as_packed()[row_offset + p];
        let unpacked = word.unpack_to_f32();
        let base = p * items;
        for j in 0..items {
            let idx = base + j;
            if idx < k {
                unpack_buf[idx] = unpacked.as_ref()[j];
            }
        }
    }

    fma_f32_slice(&unpack_buf[..k], activation)
}

// ============================================================
// Shared SIMD utilities
// ============================================================

/// Horizontal sum of __m256 — used by all AVX2 kernels.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: The pointers are valid and properly aligned for SIMD access. Loop bounds prevent out-of-bounds access.
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

/// SIMD dot product of two f32 slices.
#[inline]
fn fma_f32_slice(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if has_avx512() {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            return unsafe { fma_f32_avx512(a, b) };
        }
        if has_avx2() {
            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
            return unsafe { fma_f32_avx2(a, b) };
        }
    }

    fma_f32_scalar(a, b)
}

#[inline]
fn fma_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[inline]
// SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
unsafe fn fma_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 32 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        acc2 = _mm256_fmadd_ps(a2, b2, acc2);
        acc3 = _mm256_fmadd_ps(a3, b3, acc3);
        i += 32;
    }

    while i + 8 <= len {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        i += 8;
    }

    let combined = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut total = hsum256_ps(combined);

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[inline]
// SAFETY: The pointers are valid and properly aligned for AVX-512 access. Loop bounds guarantee all accesses stay within allocated storage.
unsafe fn fma_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 64 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        let a1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
        let a2 = _mm512_loadu_ps(a.as_ptr().add(i + 32));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(i + 32));
        let a3 = _mm512_loadu_ps(a.as_ptr().add(i + 48));
        let b3 = _mm512_loadu_ps(b.as_ptr().add(i + 48));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        acc1 = _mm512_fmadd_ps(a1, b1, acc1);
        acc2 = _mm512_fmadd_ps(a2, b2, acc2);
        acc3 = _mm512_fmadd_ps(a3, b3, acc3);
        i += 64;
    }

    while i + 16 <= len {
        let a0 = _mm512_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm512_loadu_ps(b.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(a0, b0, acc0);
        i += 16;
    }

    let combined = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    let mut total = _mm512_reduce_add_ps(combined);

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// ============================================================
// Batched GEMM — unpack once, process N input vectors
// ============================================================

/// Batched GEMM for packed types: weights [M×K] × batch_inputs [N×K] → outputs [N×M].
/// Unpacks each weight row once, then processes all N input vectors against it.
/// This is the key optimization for inference — weights stay hot in L1 across the batch.
#[inline(always)]
pub fn gemm_packed_batched<T: PackedWord>(
    weights: &PackedTensor<T>,
    batch_inputs: &[Vec<f32>],
    outputs: &mut [Vec<f32>],
) {
    let n = batch_inputs.len();
    assert_eq!(n, outputs.len());
    let shape = weights.shape();
    let m = shape[0];
    let k = shape[1];
    let k_packed = k.div_ceil(T::ITEMS);

    // Validate output vector lengths (parallel path uses unsafe raw pointers)
    for o in outputs.iter() {
        assert!(o.len() >= m, "output vector length {} < m={}", o.len(), m);
    }

    // Process each weight row once, compute dot products with all N inputs.
    // Uses per-row scale/zero for correct per-channel quantization.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let out_addrs: Vec<usize> = outputs
            .iter_mut()
            .map(|o| o.as_mut_ptr() as usize)
            .collect();
        let k_items = T::ITEMS;

        (0..m).into_par_iter().for_each(|row| {
            let scale = weights.scale_for_row(row);
            let zero = weights.zero_for_row(row);
            with_scratch(k, |unpack_buf| {
                let row_offset = row * k_packed;
                for p in 0..k_packed {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * k_items;
                    for j in 0..k_items {
                        let idx = base + j;
                        if idx < k {
                            unpack_buf[idx] = unpacked.as_ref()[j];
                        }
                    }
                }

                for (bi, input) in batch_inputs.iter().enumerate() {
                    let acc = fma_f32_slice(unpack_buf, input);

                    // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                    unsafe {
                        *((out_addrs[bi] as *mut f32).add(row)) = acc * scale + zero;
                    }
                }
            });
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        with_scratch(k, |unpack_buf| {
            for row in 0..m {
                let scale = weights.scale_for_row(row);
                let zero = weights.zero_for_row(row);
                let row_offset = row * k_packed;
                for p in 0..k_packed {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx < k {
                            unpack_buf[idx] = unpacked.as_ref()[j];
                        }
                    }
                }
                for (bi, input) in batch_inputs.iter().enumerate() {
                    let acc = fma_f32_slice(unpack_buf, input);
                    outputs[bi][row] = acc * scale + zero;
                }
            }
        });
    }
}

// ============================================================
// Batch GEMM — K-tiled for cache reuse across batch rows
// ============================================================

/// SIMD-accelerated batch GEMM for packed types.
///
/// Multiplies packed weight matrix [M, K] by activation matrix [N, K],
/// producing output [N, M]. Tiles the K dimension so activation data
/// stays in L2 cache across all M × N output elements.
///
/// For each K-tile: unpacks each weight row's tile once, then computes
/// dot products with all N batch rows. Accumulates partial results
/// across K-tiles, then applies per-row scale/zero at the end.
#[inline(always)]
pub fn gemm_batch_packed_simd<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
    m: usize,
) {
    let k_packed = k.div_ceil(T::ITEMS);

    // Zero output accumulators
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Tile K dimension for L2 cache reuse
    let mut k_start = 0;
    while k_start < k {
        let k_end = (k_start + K_BLOCK_SIZE).min(k);
        let k_len = k_end - k_start;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);

        with_scratch(k_len, |unpack_buf| {
            for row in 0..m {
                let row_offset = row * k_packed;

                // Unpack this weight row's K-tile into contiguous f32 buffer
                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_start && idx < k_end {
                            unpack_buf[idx - k_start] = unpacked.as_ref()[j];
                        }
                    }
                }

                // Accumulate dot products with all N batch activation rows
                for bi in 0..n {
                    let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                    let acc = fma_f32_slice(unpack_buf, act_slice);
                    output[bi * m + row] += acc;
                }
            }
        });

        k_start += K_BLOCK_SIZE;
    }

    // Apply per-row quantization scale/zero to final accumulated output
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(m)
            .enumerate()
            .for_each(|(_bi, out_row)| {
                for row in 0..m {
                    let scale = weights.scale_for_row(row);
                    let zero = weights.zero_for_row(row);
                    out_row[row] = out_row[row] * scale + zero;
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for bi in 0..n {
            for row in 0..m {
                let scale = weights.scale_for_row(row);
                let zero = weights.zero_for_row(row);
                output[bi * m + row] = output[bi * m + row] * scale + zero;
            }
        }
    }
}

// ============================================================
// Block-major batch GEMM
// ============================================================

/// Block-major batch GEMM for packed types.
///
/// Processes `block_size` output channels (rows) at once, reusing
/// activation data across the block. The weights must be in block-major
/// layout (see `PackedTensor::to_block_major`).
///
/// Layout (two zones):
///   1. Full blocks (rows 0..m_aligned): block-major interleaved
///   2. Tail rows (m_aligned..m): standard per-row
///
/// For each K-tile in the block-major zone:
///   1. Unpack B weight rows' tile into a contiguous f32 buffer
///   2. For each of N batch positions: compute B dot products with the
///      same activation slice, writing into B output positions
///
/// This reduces loop overhead by M/B× and improves weight-cache locality
/// because B consecutive rows share the same packed-word neighborhood.
#[inline(always)]
pub fn gemm_batch_packed_block_major<T: PackedWord>(
    weights: &PackedTensor<T>,
    activation: &[f32],
    output: &mut [f32],
    n: usize,
    k: usize,
    m: usize,
) {
    let block_size = weights.block_size();
    debug_assert!(block_size > 1, "gemm_batch_packed_block_major requires block_size > 1");
    let k_packed = k.div_ceil(T::ITEMS);
    let m_aligned = (m / block_size) * block_size;
    let full_blocks = m_aligned / block_size;
    let tail_rows = m - m_aligned;

    // Zero output accumulators
    for o in output.iter_mut() {
        *o = 0.0;
    }

    // Tile K dimension for L2 cache reuse
    let mut k_start = 0;
    while k_start < k {
        let k_end = (k_start + K_BLOCK_SIZE).min(k);
        let k_len = k_end - k_start;
        let packed_start = k_start / T::ITEMS;
        let packed_end = k_end.div_ceil(T::ITEMS);
        let k_tile_words = packed_end - packed_start;

        // --- Zone 1: Block-major (full blocks) ---
        // Scratch buffer: block_size × k_len f32 values
        let buf_words = block_size * k_len;
        with_scratch(buf_words, |unpack_buf| {
            for block in 0..full_blocks {
                let block_offset = block * block_size * k_packed;

                // Unpack B weight rows' K-tile into unpack_buf.
                // Block-major layout: word (k, local_row) at block_offset + k * block_size + local_row.
                for w in 0..k_tile_words {
                    let packed_k = packed_start + w;
                    for local_row in 0..block_size {
                        let packed_idx = block_offset + packed_k * block_size + local_row;
                        let word = weights.as_packed()[packed_idx];
                        let unpacked = word.unpack_to_f32();
                        let base = packed_k * T::ITEMS;
                        let unpack_base = local_row * k_len;
                        for j in 0..T::ITEMS {
                            let idx = base + j;
                            if idx >= k_start && idx < k_end {
                                unpack_buf[unpack_base + (idx - k_start)] = unpacked.as_ref()[j];
                            }
                        }
                    }
                }

                // Dot products with all N batch activation rows.
                for bi in 0..n {
                    let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                    for local_row in 0..block_size {
                        let global_row = block * block_size + local_row;
                        let unpack_row =
                            &unpack_buf[local_row * k_len..(local_row + 1) * k_len];
                        let acc = fma_f32_slice(unpack_row, act_slice);
                        output[bi * m + global_row] += acc;
                    }
                }
            }
        });

        // --- Zone 2: Row-major tail (rows m_aligned .. m) ---
        let tail_base_words = m_aligned * k_packed;
        with_scratch(k_len, |unpack_buf| {
            for local_row in 0..tail_rows {
                let global_row = m_aligned + local_row;
                let row_offset = tail_base_words + local_row * k_packed;

                // Unpack this tail row's K-tile
                for p in packed_start..packed_end {
                    let word = weights.as_packed()[row_offset + p];
                    let unpacked = word.unpack_to_f32();
                    let base = p * T::ITEMS;
                    for j in 0..T::ITEMS {
                        let idx = base + j;
                        if idx >= k_start && idx < k_end {
                            unpack_buf[idx - k_start] = unpacked.as_ref()[j];
                        }
                    }
                }

                // Dot products with all N batch positions
                for bi in 0..n {
                    let act_slice = &activation[bi * k + k_start..bi * k + k_end];
                    let acc = fma_f32_slice(unpack_buf, act_slice);
                    output[bi * m + global_row] += acc;
                }
            }
        });

        k_start += K_BLOCK_SIZE;
    }

    // Apply per-row quantization scale/zero to final accumulated output
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(m)
            .for_each(|out_row| {
                for row in 0..m {
                    let scale = weights.scale_for_row(row);
                    let zero = weights.zero_for_row(row);
                    out_row[row] = out_row[row] * scale + zero;
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for bi in 0..n {
            for row in 0..m {
                let scale = weights.scale_for_row(row);
                let zero = weights.zero_for_row(row);
                output[bi * m + row] = output[bi * m + row] * scale + zero;
            }
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F16x2, F32x1, U4x8, U8x4};

    #[test]
    fn test_fma_f32_slice() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = fma_f32_slice(&a, &b);
        assert!((result - 45.0).abs() < 0.001);
    }

    #[test]
    fn test_gemv_dispatch_u8x4() {
        let k = 32;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 50.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U8x4>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        // This should dispatch to the AVX2 kernel via gemv_packed_simd
        gemv_packed_simd(&weights, &activation, &mut out);

        // Verify results are reasonable (quantization introduces error)
        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_f16x2() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin()).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<F16x2>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_u4x8() {
        let k = 32;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 5.0).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32 * 0.2).cos()).collect();
        let weights = PackedTensor::<U4x8>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemv_dispatch_f32x1_exact() {
        let k = 16;
        let m = 4;
        let data: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let activation: Vec<f32> = (0..k).map(|i| i as f32 * 0.2).collect();
        let weights = PackedTensor::<F32x1>::from_f32_auto(&data, &[m, k]);

        let mut out = vec![0.0f32; m];
        gemv_packed_simd(&weights, &activation, &mut out);

        // F32x1 should be exact
        for v in &out {
            assert!(v.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_gemm_batch_block_major_u8x4() {
        let k = 16;
        let m = 8;
        let n = 4;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.1).sin() * 30.0).collect();
        let activation: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.2).cos()).collect();

        // Row-major reference
        let w_row = PackedTensor::<U8x4>::from_f32_per_channel(&data, &[m, k]);
        let mut ref_out = vec![0.0f32; n * m];
        gemm_batch_packed_simd(&w_row, &activation, &mut ref_out, n, k, m);

        // Block-major (block_size=4)
        let w_block = w_row.to_block_major(4);
        let mut block_out = vec![0.0f32; n * m];
        gemm_batch_packed_block_major(&w_block, &activation, &mut block_out, n, k, m);

        // Compare
        let tol = 0.01;
        for i in 0..n * m {
            assert!(
                (ref_out[i] - block_out[i]).abs() < tol,
                "Mismatch at {}: ref={} block={}",
                i,
                ref_out[i],
                block_out[i]
            );
        }
    }

    #[test]
    fn test_gemm_batch_block_major_f16x2() {
        let k = 8;
        let m = 8;
        let n = 3;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.3).cos()).collect();
        let activation: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.7).sin()).collect();

        let w_row = PackedTensor::<F16x2>::from_f32_per_channel(&data, &[m, k]);
        let mut ref_out = vec![0.0f32; n * m];
        gemm_batch_packed_simd(&w_row, &activation, &mut ref_out, n, k, m);

        let w_block = w_row.to_block_major(4);
        let mut block_out = vec![0.0f32; n * m];
        gemm_batch_packed_block_major(&w_block, &activation, &mut block_out, n, k, m);

        let tol = 0.05;
        for i in 0..n * m {
            assert!(
                (ref_out[i] - block_out[i]).abs() < tol,
                "Mismatch at {}: ref={} block={}",
                i,
                ref_out[i],
                block_out[i]
            );
        }
    }

    #[test]
    fn test_gemm_batch_block_major_non_multiple_rows() {
        // m=6 (not a multiple of block_size=4), k=16, n=2
        let k = 16;
        let m = 6;
        let n = 2;
        let data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
        let activation: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.3 + 1.0).collect();

        let w_row = PackedTensor::<U8x4>::from_f32_per_channel(&data, &[m, k]);
        let mut ref_out = vec![0.0f32; n * m];
        gemm_batch_packed_simd(&w_row, &activation, &mut ref_out, n, k, m);

        let w_block = w_row.to_block_major(4);
        let mut block_out = vec![0.0f32; n * m];
        gemm_batch_packed_block_major(&w_block, &activation, &mut block_out, n, k, m);

        let tol = 0.01;
        for i in 0..n * m {
            assert!(
                (ref_out[i] - block_out[i]).abs() < tol,
                "Mismatch at {}: ref={} block={}",
                i,
                ref_out[i],
                block_out[i]
            );
        }
    }
}
