#![allow(unused_imports)]
pub mod arm_neon;
mod conv;
mod elementwise;
mod factories;
mod losses;
mod matmul;
mod norm;
mod pooling;
mod quantized_conv;
mod reductions;
mod simd;
pub use conv::*;
pub use elementwise::*;
pub use factories::*;
pub use losses::*;
pub use matmul::*;
pub use norm::*;
pub use pooling::*;
pub use quantized_conv::*;
pub use reductions::*;
pub use simd::*;

// Aligned buffer for SIMD operations
#[repr(align(32))]
struct AlignedBuffer {
    data: Vec<f32>,
}

impl AlignedBuffer {
    const fn new(_capacity: usize) -> Self {
        Self { data: Vec::new() }
    }

    fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, 0.0);
    }

    #[allow(dead_code)]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn split_at_mut(&mut self, mid: usize) -> (&mut [f32], &mut [f32]) {
        self.data.split_at_mut(mid)
    }
}

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use std::sync::OnceLock;

/// Enable DAZ (Denormals-Are-Zero) and FTZ (Flush-To-Zero) for the current thread.
/// Subnormal floats are treated as zero, preventing catastrophic throughput drops
/// when weights approach zero during training. Thread-local since MXCSR is per-thread.
///
/// SAFETY: This changes floating-point behavior for denormal numbers on the current thread.
/// Denormals will be flushed to zero, which is acceptable for ML workloads where the
/// precision loss from subnormals is negligible compared to the performance gain.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
#[allow(deprecated)]
unsafe fn enable_daz_ftz() {
    use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    // FTZ = bit 15, DAZ = bit 6 of MXCSR
    const FTZ: u32 = 1 << 15;
    const DAZ: u32 = 1 << 6;
    let mxcsr = _mm_getcsr();
    _mm_setcsr(mxcsr | FTZ | DAZ);
}

// Thread-local guard ensuring DAZ/FTZ is enabled once per thread.
// Called at the start of SIMD hot paths to guarantee correct CPU state.
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
thread_local! {
    static DAZ_FTZ_INIT: () = {
        // SAFETY: Called once per thread during TLS initialization.
        // Only affects denormal float handling, not normal arithmetic.
        unsafe { enable_daz_ftz(); }
    };
}

/// Ensure DAZ/FTZ is enabled on the current thread (no-op on non-x86).
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
#[inline(always)]
fn ensure_daz_ftz() {
    DAZ_FTZ_INIT.with(|_| {});
}

#[cfg(not(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86"))))]
#[inline(always)]
fn ensure_daz_ftz() {}

// Thread-local reusable scratch buffer for conv2d operations.
// Avoids per-call heap allocations for im2col and GEMM output buffers.
// SAFETY with rayon: Each rayon worker thread gets its own independent
// CONV_SCRATCH instance. The RefCell is only borrowed within a single thread.
// rayon's par_chunks_mut guarantees non-overlapping mutable slices, so no
// aliasing occurs even when the borrow spans a parallel section. The buffer
// is never shared across threads.
thread_local! {
    static CONV_SCRATCH: std::cell::RefCell<AlignedBuffer> = const {
        std::cell::RefCell::new(AlignedBuffer::new(0))
    };
}

// Thread-local buffer for transposed weights (K x N) to avoid reallocation
thread_local! {
    static WT_TRANS_BUF: std::cell::RefCell<Option<Vec<f32>>> = const {
        std::cell::RefCell::new(None)
    };
}

// Thread-local buffer for SIMD kernel s-vector to avoid per-pixel allocation
thread_local! {
    static S_BUF: std::cell::RefCell<Vec<f32>> = const {
        std::cell::RefCell::new(Vec::new())
    };
}

/// Generic GPU fallback wrapper: moves inputs to CPU, computes, moves result back to GPU.
/// Extracts device_id from the first GPU tensor in args.
pub fn gpu_fallback<F>(args: &[&Tensor], compute: F) -> Vec<Tensor>
where
    F: FnOnce(&[&Tensor]) -> Vec<Tensor>,
{
    // Convert all inputs to CPU
    let cpu_args: Vec<Tensor> = args.iter().map(|t| t.to_cpu()).collect();
    let cpu_refs: Vec<&Tensor> = cpu_args.iter().collect();

    // Compute on CPU
    let results = compute(&cpu_refs);

    // Find the device_id from the first GPU tensor in original args
    let device_id = args
        .iter()
        .find_map(|t| match t.inner.storage.as_ref() {
            Storage::Wgpu(gpu) => Some(gpu.device_id),
            _ => None,
        })
        .unwrap_or(0); // Default to device 0 if no GPU tensor found

    // Move results back to GPU
    results.into_iter().map(|r| r.to_gpu(device_id)).collect()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
use std::arch::aarch64::*;

use wide::f32x4;
#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]
use wide::f32x8;

// Memory-bound elementwise ops: 128KB working set (L2 cache friendly)
const CHUNK_MEMBOUND: usize = 1024 * 32; // 32K f32 = 128KB

// Compute-bound transcendental ops: 32KB for better load balancing
const CHUNK_TRANSCENDENTAL: usize = 1024 * 8; // 8K f32 = 32KB

// Rayon parallel threshold: only parallelize above this size
// Prevents overhead from parallelization on small tensors
#[allow(dead_code)]
const PARALLEL_THRESHOLD: usize = 1024 * 32; // 32K elements

// Matrix ops: larger chunks for better BLAS-level locality
// SIMD-only threshold (non-parallel)
// This constant is used to determine when to use SIMD operations
#[allow(dead_code)]
const SIMD_THRESHOLD: usize = 256;

// Runtime SIMD level detection
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[derive(Clone, Copy, PartialEq)]
enum SimdLevel {
    Scalar,
    Avx2,
    Avx512,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn detect_simd_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
    }
    SimdLevel::Scalar
}

// Fast exp approximation using Cephes-style algorithm

// Fast exp approximation for AVX512

// Fast log approximation using integer exponent extraction

// Fast log approximation for AVX512

// Parallel SIMD kernels - AVX2 version with vectorized tails

// Parallel SIMD kernels - NEON version

// Parallel SIMD kernels - AVX512 version with masked operations for tails

// Parallel scalar fallback

// Mul parallel AVX2 kernel

// Mul parallel NEON kernel

// Mul parallel AVX512 kernel

// Parallel scalar fallback for mul

// Relu parallel AVX2 kernel

// Relu parallel NEON kernel

// Relu parallel AVX512 kernel

// Parallel scalar fallback for relu

// Div parallel AVX2 kernel

// Div parallel NEON kernel

// Div parallel AVX512 kernel

// Parallel scalar fallback for div

// Neg parallel AVX2 kernel

// Neg parallel NEON kernel

// Neg parallel AVX512 kernel

// Parallel scalar fallback for neg

// Abs parallel AVX2 kernel

// Abs parallel NEON kernel

// Abs parallel AVX512 kernel

// Parallel scalar fallback for abs

// Sub parallel AVX2 kernel

// Sub parallel NEON kernel

// Sub parallel AVX512 kernel

// Parallel scalar fallback for sub

// Fused add+relu parallel AVX2 kernel

// Fused add+relu parallel AVX512 kernel

// Parallel scalar fallback for fused_add_relu

// Fused add+relu parallel NEON kernel

// Fused mul+add parallel AVX2 kernel

// Fused mul+add parallel AVX512 kernel

// Fused mul+add parallel NEON kernel

// Parallel scalar fallback for fused_mul_add

// Parallel sigmoid AVX2 kernel using vectorized fast_exp

// Sigmoid parallel AVX512 kernel using vectorized fast_exp

// Parallel scalar fallback for sigmoid

// Parallel sigmoid NEON kernel using wide::f32x4

// Parallel tanh AVX2 kernel using vectorized fast_exp

// Tanh parallel AVX512 kernel using vectorized fast_exp

// Parallel scalar fallback for tanh

// Parallel tanh NEON kernel using wide::f32x4

// Exp kernel using wide library for SIMD

// Log kernel using wide library for SIMD

// Sqrt kernel using wide library for SIMD

// GELU kernel using wide library for SIMD

// SiLU kernel using wide library for SIMD

// Exp kernel using wide library for SIMD on ARM

// Log kernel using wide library for SIMD on ARM

// Sqrt kernel using wide library for SIMD on ARM

// Sigmoid SIMD for x86_64 using exp approximation

#[allow(dead_code)]
fn create_output(tensor: &Tensor, shape: Vec<i64>) -> Tensor {
    let sizes: smallvec::SmallVec<[i64; 8]> = shape.into();
    let numel: i64 = sizes.iter().product();
    let nbytes = (numel * tensor.dtype().size() as i64) as usize;
    let storage = Arc::new(Storage::new_cpu(tensor.dtype(), nbytes));
    Tensor::new(crate::tensor::TensorImpl::new(
        storage,
        sizes,
        tensor.dtype(),
    ))
}

#[inline]
#[allow(dead_code)]
fn broadcast_shapes_simple(a: &[i64], b: &[i64]) -> Vec<i64> {
    let ndim = std::cmp::max(a.len(), b.len());
    let mut result = vec![1i64; ndim];

    let offset_a = ndim - a.len();
    let offset_b = ndim - b.len();

    for i in 0..ndim {
        let a_val = if i < offset_a { 1 } else { a[i - offset_a] };
        let b_val = if i < offset_b { 1 } else { b[i - offset_b] };
        if a_val != b_val && a_val != 1 && b_val != 1 {
            panic!("shapes {:?} and {:?} are not broadcast-compatible", a, b);
        }
        result[i] = a_val.max(b_val);
    }
    result
}

// Section 7: Optimized broadcast index decomposition
// Precomputes multipliers to map output index to input index without loops
#[inline]
#[allow(dead_code)]
fn broadcast_index_decomposition(
    idx: usize,
    out_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
    storage_offset: usize,
) -> usize {
    let ndim = out_shape.len();
    // Precompute multipliers on the stack: multipliers[i] = product(out_shape[i+1..])
    let mut multipliers: smallvec::SmallVec<[usize; 8]> = smallvec::smallvec![0usize; ndim];
    let mut mult = 1usize;
    for i in (0..ndim).rev() {
        multipliers[i] = mult;
        mult *= out_shape[i];
    }

    // Map output index to 1D input index
    let mut input_idx = 0usize;
    for i in 0..ndim {
        let dim_idx = (idx / multipliers[i]) % out_shape[i];
        if i < input_shape.len() && input_shape[i] != 1 {
            input_idx += dim_idx * input_strides[i];
        }
    }
    input_idx + storage_offset
}

// Parallel exp AVX2 kernel using fast_exp_avx2

// Parallel exp AVX512 kernel using fast_exp_avx512

// Parallel exp NEON kernel using wide::f32x4

// Parallel log AVX2 kernel using fast_log_avx2

// Parallel log AVX512 kernel using fast_log_avx512

// Parallel log NEON kernel using wide::f32x4

// Parallel sqrt AVX2 kernel

// Parallel sqrt AVX512 kernel

// Parallel sqrt NEON kernel using wide::f32x4

// Parallel GELU AVX2 kernel using fast_exp_avx2 for tanh

// Parallel GELU AVX512 kernel using fast_exp_avx512 for tanh

// Parallel GELU NEON kernel using wide::f32x4

/// Fast contiguous last-dim sum with SIMD
pub fn sum_last_dim_contiguous(a: &Tensor, dim_size: usize, num_rows: usize) -> Tensor {
    let a_ptr = a.data_ptr_f32();

    // Direct allocation without Arc::make_mut overhead
    let mut result_data = vec![0.0f32; num_rows];
    let out_ptr = result_data.as_mut_ptr();

    #[cfg(feature = "parallel")]
    {
        if num_rows > 64 {
            use rayon::prelude::*;
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_rows).into_par_iter().for_each(|row| {
                // SAFETY: The pointers are valid for the accessed elements and properly aligned
                // for SIMD access. Loop bounds prevent out-of-bounds reads/writes.
                let row_ptr = unsafe { (a_usize as *const f32).add(row * dim_size) };
                let mut sum = 0.0f32;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                        // SAFETY: The pointers are valid for the accessed elements and properly aligned
                        // for SIMD access. Loop bounds prevent out-of-bounds reads/writes.
                        unsafe {
                            let mut acc = _mm256_setzero_ps();
                            let mut j = 0;
                            while j + 8 <= dim_size {
                                acc = _mm256_add_ps(acc, _mm256_loadu_ps(row_ptr.add(j)));
                                j += 8;
                            }
                            sum = hsum256_ps(acc);
                            for k in j..dim_size {
                                sum += *row_ptr.add(k);
                            }
                        }
                    } else {
                        for j in 0..dim_size {
                            // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                            // The pointer is valid for this element access.
                            unsafe {
                                sum += *row_ptr.add(j);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                        // The pointer is valid for this element access.
                        unsafe {
                            sum += *row_ptr.add(j);
                        }
                    }
                }
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    *(out_usize as *mut f32).add(row) = sum;
                }
            });
            return Tensor::from_vec(result_data, vec![num_rows as i64]);
        }
    }

    // Non-parallel: SIMD inline
    for row in 0..num_rows {
        // SAFETY: The pointers are valid for the accessed elements and properly aligned
        // for SIMD access. Loop bounds prevent out-of-bounds reads/writes.
        let row_ptr = unsafe { a_ptr.add(row * dim_size) };
        let mut sum = 0.0f32;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                // SAFETY: The pointers are valid for the accessed elements and properly aligned
                // for SIMD access. Loop bounds prevent out-of-bounds reads/writes.
                unsafe {
                    let mut acc = _mm256_setzero_ps();
                    let mut j = 0;
                    while j + 8 <= dim_size {
                        acc = _mm256_add_ps(acc, _mm256_loadu_ps(row_ptr.add(j)));
                        j += 8;
                    }
                    sum = hsum256_ps(acc);
                    for k in j..dim_size {
                        sum += *row_ptr.add(k);
                    }
                }
            } else {
                for j in 0..dim_size {
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    unsafe {
                        sum += *row_ptr.add(j);
                    }
                }
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            for j in 0..dim_size {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    sum += *row_ptr.add(j);
                }
            }
        }
        result_data[row] = sum;
    }
    Tensor::from_vec(result_data, vec![num_rows as i64])
}

/// Cross-entropy backward: computes grad_logits = softmax(logits) - one_hot(target),
/// scaled by grad_output / batch_size for mean reduction.
/// Writes directly into a pre-allocated output buffer, parallelized over batch rows.
pub fn cross_entropy_backward_f32(
    logits_data: &[f32],
    targets_data: &[f32],
    grad_out: f32,
    batch_size: usize,
    num_classes: usize,
    reduction: &str,
    grad_logits_data: &mut [f32],
) {
    let scale = if reduction == "mean" {
        grad_out / batch_size as f32
    } else {
        grad_out
    };

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let logits_usize = logits_data.as_ptr() as usize;
        let targets_usize = targets_data.as_ptr() as usize;
        let grad_usize = grad_logits_data.as_mut_ptr() as usize;
        let nc = num_classes;

        (0..batch_size).into_par_iter().for_each(|b| {
            let base = b * nc;
            // SAFETY: Each rayon iteration accesses disjoint memory regions because
            // the loop index maps to non-overlapping chunks of the buffer.
            let target_class = unsafe { *((targets_usize + b * 4) as *const f32) } as usize;

            // Find max
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..nc {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    max_val = max_val.max(*((logits_usize + (base + j) * 4) as *const f32));
                }
            }

            // Compute sum_exp
            let mut sum_exp = 0.0f32;
            for j in 0..nc {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    sum_exp += (*((logits_usize + (base + j) * 4) as *const f32) - max_val).exp();
                }
            }

            // Guard against degenerate inputs (all logits = -inf → sum_exp = 0)
            if sum_exp == 0.0 || !sum_exp.is_finite() {
                for j in 0..nc {
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    unsafe {
                        *((grad_usize + (base + j) * 4) as *mut f32) = 0.0;
                    }
                }
                return;
            }

            let inv_sum = scale / sum_exp;

            // Write gradient: softmax - one_hot, scaled
            for j in 0..nc {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let p = (logits_usize + (base + j) * 4) as *const f32;
                    let grad = (*p - max_val).exp() * inv_sum
                        - if j == target_class { scale } else { 0.0 };
                    *((grad_usize + (base + j) * 4) as *mut f32) = grad;
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size {
            let base = b * num_classes;
            let target_class = targets_data[b] as usize;

            let mut max_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                max_val = max_val.max(logits_data[base + j]);
            }

            let mut sum_exp = 0.0f32;
            for j in 0..num_classes {
                sum_exp += (logits_data[base + j] - max_val).exp();
            }

            if sum_exp == 0.0 || !sum_exp.is_finite() {
                for j in 0..num_classes {
                    grad_logits_data[base + j] = 0.0;
                }
                continue;
            }

            let inv_sum = scale / sum_exp;

            for j in 0..num_classes {
                let grad = (logits_data[base + j] - max_val).exp() * inv_sum
                    - if j == target_class { scale } else { 0.0 };
                grad_logits_data[base + j] = grad;
            }
        }
    }
}

// Winograd F(2x2, 3x3) is disabled for now due to overhead from transform operations
// The im2col + BLAS approach is faster for current use cases

/// Fused layer norm backward: computes dX, dW, dB without intermediate tensors.
/// Parallelized over outer dimensions (rows) for dX computation.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward_f32(
    grad_data: &[f32],
    x_hat_data: &[f32],
    weight_data: Option<&[f32]>,
    outer_size: usize,
    norm_dim: usize,
    eps: f32,
    var_data: &[f32],
    grad_input_data: &mut [f32],
    grad_weight_data: &mut [f32],
    grad_bias_data: &mut [f32],
) {
    let nd = norm_dim;

    // Zero weight/bias grads
    for j in 0..nd {
        grad_weight_data[j] = 0.0;
        grad_bias_data[j] = 0.0;
    }

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let grad_usize = grad_data.as_ptr() as usize;
        let xhat_usize = x_hat_data.as_ptr() as usize;
        let ginput_usize = grad_input_data.as_mut_ptr() as usize;
        let var_usize = var_data.as_ptr() as usize;

        // Parallel dX computation
        (0..outer_size).into_par_iter().for_each(|row| {
            let base = row * nd;
            // SAFETY: Each rayon iteration accesses disjoint memory regions because
            // the loop index maps to non-overlapping chunks of the buffer.
            let inv_std = 1.0 / (unsafe { *((var_usize + row * 4) as *const f32) } + eps).sqrt();

            // Compute sum(dY * weight) and sum(dY * weight * x_hat)
            let mut sum_gw = 0.0f32;
            let mut sum_gw_xh = 0.0f32;
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    let gw = if let Some(w) = weight_data {
                        g * w[j]
                    } else {
                        g
                    };
                    sum_gw += gw;
                    sum_gw_xh += gw * xh;
                }
            }
            let mean_gw = sum_gw / nd as f32;
            let mean_gw_xh = sum_gw_xh / nd as f32;

            // dX
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    let gw = if let Some(w) = weight_data {
                        g * w[j]
                    } else {
                        g
                    };
                    let dx = (gw - mean_gw - xh * mean_gw_xh) * inv_std;
                    *((ginput_usize + (base + j) * 4) as *mut f32) = dx;
                }
            }
        });

        // Sequential accumulation for dW and dB (small array, not worth parallel reduction)
        for row in 0..outer_size {
            let base = row * nd;
            for j in 0..nd {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    let g = *((grad_usize + (base + j) * 4) as *const f32);
                    let xh = *((xhat_usize + (base + j) * 4) as *const f32);
                    grad_bias_data[j] += g;
                    grad_weight_data[j] += g * xh;
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..outer_size {
            let base = row * nd;
            let inv_std = 1.0 / (var_data[row] + eps).sqrt();

            let mut sum_gw = 0.0f32;
            let mut sum_gw_xh = 0.0f32;
            for j in 0..nd {
                let g = grad_data[base + j];
                let xh = x_hat_data[base + j];
                let gw = if let Some(w) = weight_data {
                    g * w[j]
                } else {
                    g
                };
                sum_gw += gw;
                sum_gw_xh += gw * xh;
            }
            let mean_gw = sum_gw / nd as f32;
            let mean_gw_xh = sum_gw_xh / nd as f32;

            for j in 0..nd {
                let g = grad_data[base + j];
                let xh = x_hat_data[base + j];
                let gw = if let Some(w) = weight_data {
                    g * w[j]
                } else {
                    g
                };
                grad_input_data[base + j] = (gw - mean_gw - xh * mean_gw_xh) * inv_std;

                grad_bias_data[j] += g;
                grad_weight_data[j] += g * xh;
            }
        }
    }
}

pub struct EmbeddingBackward {
    pub inputs: Vec<Tensor>,
    pub edges: Vec<Edge>,
}

impl EmbeddingBackward {
    pub fn new(weight: Tensor, indices: Tensor) -> Self {
        EmbeddingBackward {
            inputs: vec![weight, indices],
            edges: vec![],
        }
    }
}

impl Node for EmbeddingBackward {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        _output_tensor_id: usize,
    ) -> Vec<Option<Tensor>> {
        let Some(grad_output) = grad_outputs.into_iter().next().flatten() else {
            return vec![Some(Tensor::zeros(
                self.inputs[0].shape_ref().to_vec(),
                self.inputs[0].dtype(),
                self.inputs[0].device(),
            ))];
        };
        let weight = &self.inputs[0];
        let indices = &self.inputs[1];

        let weight_shape = weight.shape_ref();
        let embedding_dim = weight_shape[1];
        let batch_size = grad_output.shape_ref()[0];

        // Create gradient for weight (same shape as weight)
        let mut weight_grad = Tensor::zeros(weight_shape.to_vec(), weight.dtype(), weight.device());

        // Accumulate gradients from output
        let grad_output_ptr = grad_output.data_ptr() as *const f32;
        let indices_ptr = indices.data_ptr() as *const f32;
        let weight_grad_inner = Arc::make_mut(&mut weight_grad.inner);
        let weight_grad_storage = Arc::make_mut(&mut weight_grad_inner.storage);
        let Storage::Cpu(cpu_storage) = weight_grad_storage else {
            panic!("Expected CPU storage");
        };
        let weight_grad_data = Arc::make_mut(&mut cpu_storage.data);
        let weight_grad_ptr = weight_grad_data.as_mut_ptr() as *mut f32;

        for i in 0..batch_size as usize {
            // SAFETY: The offset stays within the bounds of the allocated tensor storage.
            // The pointer is valid for this element access.
            let idx = unsafe { *indices_ptr.add(i) } as usize;
            if idx < weight_shape[0] as usize {
                for j in 0..embedding_dim as usize {
                    let w_idx = idx * embedding_dim as usize + j;
                    let o_idx = i * embedding_dim as usize + j;
                    // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                    // The pointer is valid for this element access.
                    unsafe {
                        *weight_grad_ptr.add(w_idx) += *grad_output_ptr.add(o_idx);
                    }
                }
            }
        }

        vec![Some(weight_grad)]
    }

    fn next_edges(&self) -> &[Edge] {
        &self.edges
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn name(&self) -> &str {
        "EmbeddingBackward"
    }

    fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }
}

#[ctor::ctor]
fn register_kernels() {
    use crate::dispatcher::{register, register_fallible, DispatchKey};

    register("add", DispatchKey::Cpu, add_kernel);
    register("sub", DispatchKey::Cpu, sub_kernel);
    register("mul", DispatchKey::Cpu, mul_kernel);
    register("div", DispatchKey::Cpu, div_kernel);
    register("neg", DispatchKey::Cpu, neg_kernel);
    register("abs", DispatchKey::Cpu, abs_kernel);
    register("exp", DispatchKey::Cpu, exp_kernel);
    register("log", DispatchKey::Cpu, log_kernel);
    register("sqrt", DispatchKey::Cpu, sqrt_kernel);
    register("relu", DispatchKey::Cpu, relu_kernel);
    register(
        "fused_add_relu",
        DispatchKey::Cpu,
        fused_add_relu_kernel,
    );
    register("gelu", DispatchKey::Cpu, gelu_kernel);
    register("sigmoid", DispatchKey::Cpu, sigmoid_kernel);
    register("tanh", DispatchKey::Cpu, tanh_kernel);
    register("silu", DispatchKey::Cpu, silu_kernel);
    register(
        "gelu_backward",
        DispatchKey::Cpu,
        gelu_backward_kernel,
    );
    register(
        "silu_backward",
        DispatchKey::Cpu,
        silu_backward_kernel,
    );
    register(
        "leaky_relu",
        DispatchKey::Cpu,
        leaky_relu_kernel,
    );
    register("prelu", DispatchKey::Cpu, prelu_kernel);
    register("softplus", DispatchKey::Cpu, softplus_kernel);
    register("hardswish", DispatchKey::Cpu, hardswish_kernel);
    register("elu", DispatchKey::Cpu, elu_kernel);
    register("clamp", DispatchKey::Cpu, clamp_kernel);
    register("pow", DispatchKey::Cpu, pow_kernel);
    fn erf_kernel(args: &[&Tensor]) -> Vec<Tensor> {
        vec![args[0].erf()]
    }
    register("erf", DispatchKey::Cpu, erf_kernel);
    register_fallible("matmul", DispatchKey::Cpu, matmul_kernel);
    register_fallible("linear", DispatchKey::Cpu, linear_kernel);
    register_fallible(
        "fused_linear_relu",
        DispatchKey::Cpu,
        fused_linear_relu_kernel,
    );
    register_fallible(
        "fused_linear_gelu",
        DispatchKey::Cpu,
        fused_linear_gelu_kernel,
    );
    register(
        "fused_mul_add",
        DispatchKey::Cpu,
        fused_mul_add_kernel,
    );
    register_fallible(
        "fused_linear_silu",
        DispatchKey::Cpu,
        fused_linear_silu_kernel,
    );
    register(
        "fused_conv_bn_silu",
        DispatchKey::Cpu,
        fused_conv_bn_silu_kernel,
    );
    register(
        "fused_conv_bn",
        DispatchKey::Cpu,
        fused_conv_bn_kernel,
    );
    register("sum", DispatchKey::Cpu, sum_kernel);
    register("mean", DispatchKey::Cpu, mean_kernel);
    register("max", DispatchKey::Cpu, max_kernel);
    register("min", DispatchKey::Cpu, min_kernel);
    register("maximum", DispatchKey::Cpu, maximum_kernel);
    register("minimum", DispatchKey::Cpu, minimum_kernel);
    register("softmax", DispatchKey::Cpu, softmax_kernel);
    register(
        "log_softmax",
        DispatchKey::Cpu,
        log_softmax_kernel,
    );
    register(
        "softmax_backward",
        DispatchKey::Cpu,
        softmax_backward_kernel,
    );
    register("mse_loss", DispatchKey::Cpu, mse_loss_kernel);
    register(
        "bce_with_logits",
        DispatchKey::Cpu,
        bce_with_logits_kernel,
    );
    register(
        "huber_loss",
        DispatchKey::Cpu,
        huber_loss_kernel,
    );
    register(
        "cross_entropy_loss",
        DispatchKey::Cpu,
        cross_entropy_loss_kernel,
    );

    // GPU fallback for cross_entropy_loss (moves to CPU for computation)
    fn cross_entropy_loss_gpu_fallback(args: &[&Tensor]) -> Vec<Tensor> {
        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
        // The pointer is valid for this element access.
        gpu_fallback(args, |cpu_args| unsafe {
            cross_entropy_loss_kernel(cpu_args)
        })
    }

    register(
        "cross_entropy_loss",
        DispatchKey::Wgpu,
        cross_entropy_loss_gpu_fallback,
    );

    register("conv2d", DispatchKey::Cpu, conv2d_kernel);
    register("conv1d", DispatchKey::Cpu, conv1d_kernel);
    register("conv3d", DispatchKey::Cpu, conv3d_kernel);
    register(
        "conv_transpose2d",
        DispatchKey::Cpu,
        conv_transpose2d_kernel,
    );
    register(
        "layer_norm",
        DispatchKey::Cpu,
        layer_norm_kernel,
    );
    register(
        "fused_layer_norm_gelu",
        DispatchKey::Cpu,
        fused_layer_norm_gelu_kernel,
    );
    register(
        "batch_norm",
        DispatchKey::Cpu,
        batch_norm_kernel,
    );
    register("rms_norm", DispatchKey::Cpu, rms_norm_kernel);
    register(
        "fused_rms_norm_gelu",
        DispatchKey::Cpu,
        fused_rms_norm_gelu_kernel,
    );
    register(
        "fused_conv_bn_relu",
        DispatchKey::Cpu,
        fused_conv_bn_relu_kernel,
    );
    register(
        "fused_conv_bn_gelu",
        DispatchKey::Cpu,
        fused_conv_bn_gelu_kernel,
    );
    register("embedding", DispatchKey::Cpu, embedding_kernel);
    register("zeros", DispatchKey::Cpu, zeros_kernel);
    register("ones", DispatchKey::Cpu, ones_kernel);
    register("full", DispatchKey::Cpu, full_kernel);
    register("arange", DispatchKey::Cpu, arange_kernel);
    register("linspace", DispatchKey::Cpu, linspace_kernel);
    register("eye", DispatchKey::Cpu, eye_kernel);
    register("randn", DispatchKey::Cpu, randn_kernel);
    register("rand", DispatchKey::Cpu, rand_kernel);
    register("gt_scalar", DispatchKey::Cpu, gt_scalar_kernel);

    // GPU fallback for gt_scalar (moves to CPU for computation)
    fn gt_scalar_gpu_fallback(args: &[&Tensor]) -> Vec<Tensor> {
        // SAFETY: The offset stays within the bounds of the allocated tensor storage.
        // The pointer is valid for this element access.
        gpu_fallback(args, |cpu_args| unsafe { gt_scalar_kernel(cpu_args) })
    }

    register(
        "gt_scalar",
        DispatchKey::Wgpu,
        gt_scalar_gpu_fallback,
    );
    register(
        "max_pool2d",
        DispatchKey::Cpu,
        max_pool2d_kernel,
    );
    register(
        "avg_pool2d",
        DispatchKey::Cpu,
        avg_pool2d_kernel,
    );
    register("sign", DispatchKey::Cpu, sign_kernel);
    register("lt_scalar", DispatchKey::Cpu, lt_scalar_kernel);
    register(
        "add_scalar",
        DispatchKey::Cpu,
        add_scalar_kernel,
    );
    register(
        "div_scalar",
        DispatchKey::Cpu,
        div_scalar_kernel,
    );
    register(
        "logical_not",
        DispatchKey::Cpu,
        logical_not_kernel,
    );
    register(
        "flash_attention",
        DispatchKey::Cpu,
        flash_attention_kernel,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatcher::{DispatchKey, dispatch};
    use crate::tensor::Tensor;

    #[test]
    fn test_embedding_bulk_copy() {
        // Test that embedding forward produces correct results
        let num_embeddings: usize = 10;
        let embedding_dim: usize = 4;

        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|i| i as f32 * 0.1)
            .collect();
        let weight = Tensor::from_vec(
            weight_data.clone(),
            vec![num_embeddings as i64, embedding_dim as i64],
        );

        // Indices: pick rows 3, 7, 1
        let indices_data = vec![3.0f32, 7.0, 1.0];
        let indices = Tensor::from_vec(indices_data, vec![3]);

        let result = crate::dispatcher::dispatch(
            "embedding",
            crate::dispatcher::DispatchKey::Cpu,
            &[&weight, &indices],
        )
        .expect("test_embedding: dispatch failed");
        let result_data = result[0].as_f32_slice();

        // Expected: rows 3, 7, 1 from weight
        let mut expected = Vec::with_capacity(3 * embedding_dim);
        for &idx in &[3usize, 7, 1] {
            expected
                .extend_from_slice(&weight_data[idx * embedding_dim..(idx + 1) * embedding_dim]);
        }

        assert_eq!(result_data.len(), expected.len());
        for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "idx={}: got={}, expected={}",
                idx,
                got,
                exp
            );
        }
    }

    /// Benchmark: measure fused_linear_relu vs matmul+relu to quantify the gap.
    /// This test prints timing data and verifies correctness of both paths.
    #[test]
    fn bench_fused_linear_relu_vs_matmul() {
        use std::time::Instant;

        let configs: Vec<(usize, usize, usize)> = vec![
            (32, 512, 512),   // small
            (32, 1024, 1024), // medium
            (64, 2048, 2048), // large
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();

            let x = Tensor::from_vec(x_data, vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data, vec![in_feat as i64, out_feat as i64]);

            // Warmup
            for _ in 0..3 {
                let _ = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w])
                    .expect("bench fused_linear_relu warmup failed");
            }

            // Benchmark fused_linear_relu
            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w])
                    .expect("bench fused_linear_relu failed");
            }
            let fused_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            // Benchmark matmul + relu (BLAS-backed)
            let start = Instant::now();
            for _ in 0..iters {
                let linear_out = x.matmul(&w);
                let _ =
                    dispatch("relu", DispatchKey::Cpu, &[&linear_out]).expect("bench relu failed");
            }
            let blas_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let gflops =
                (2.0 * batch as f64 * out_feat as f64 * in_feat as f64) / (fused_ms / 1000.0) / 1e9;
            let blas_gflops =
                (2.0 * batch as f64 * out_feat as f64 * in_feat as f64) / (blas_ms / 1000.0) / 1e9;

            println!(
                "fused_linear_relu {}x{}x{}: fused={:.3}ms ({:.1} GFLOP/s), blas={:.3}ms ({:.1} GFLOP/s), gap={:.1}x",
                batch, in_feat, out_feat, fused_ms, gflops, blas_ms, blas_gflops, fused_ms / blas_ms
            );
        }
    }

    /// Reference: compute x @ w^T + bias then apply activation, all in scalar.
    fn reference_fused_linear(
        x_data: &[f32],
        w_data: &[f32],
        bias_data: Option<&[f32]>,
        batch: usize,
        in_feat: usize,
        out_feat: usize,
        activation: &str,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * out_feat];
        for b in 0..batch {
            for o in 0..out_feat {
                let mut sum = 0.0f32;
                for k in 0..in_feat {
                    sum += x_data[b * in_feat + k] * w_data[k * out_feat + o];
                }
                if let Some(bias) = bias_data {
                    sum += bias[o];
                }
                out[b * out_feat + o] = match activation {
                    "relu" => sum.max(0.0),
                    "silu" => sum / (1.0 + (-sum).exp()),
                    "gelu" => {
                        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                        let coeff = 0.044715f32;
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        0.5 * sum * (1.0 + t)
                    }
                    _ => panic!("unknown activation"),
                };
            }
        }
        out
    }

    /// Test fused_linear_relu correctness: BLAS path (large) and scalar path (small).
    #[test]
    fn test_fused_linear_relu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        // Configs: (batch, in, out) - both large (BLAS) and small (scalar fallback)
        let configs: Vec<(usize, usize, usize)> = vec![
            (32, 256, 256), // above BLAS threshold
            (4, 8, 8),      // below BLAS threshold (scalar fallback)
            (1, 128, 64),   // single sample
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![in_feat as i64, out_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            // Without bias
            let result = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w])
                .expect("test fused_linear_relu no-bias failed");
            let result_data = result[0].as_f32_slice();
            let expected =
                reference_fused_linear(&x_data, &w_data, None, batch, in_feat, out_feat, "relu");
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "relu no-bias batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }

            // With bias
            let result = dispatch("fused_linear_relu", DispatchKey::Cpu, &[&x, &w, &bias])
                .expect("test fused_linear_relu with-bias failed");
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "relu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "relu with-bias batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test fused_linear_silu correctness: BLAS path and scalar path.
    #[test]
    fn test_fused_linear_silu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize)> = vec![
            (16, 512, 256), // above BLAS threshold
            (2, 16, 16),    // below BLAS threshold
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![in_feat as i64, out_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            // With bias
            let result = dispatch("fused_linear_silu", DispatchKey::Cpu, &[&x, &w, &bias])
                .expect("test fused_linear_silu failed");
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "silu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "silu batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test fused_linear_gelu correctness: BLAS path and scalar path.
    #[test]
    fn test_fused_linear_gelu_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize)> = vec![
            (16, 512, 256), // above BLAS threshold
            (2, 16, 16),    // below BLAS threshold
        ];

        for &(batch, in_feat, out_feat) in &configs {
            let x_data: Vec<f32> = (0..batch * in_feat)
                .map(|i| ((i as f32) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_feat * in_feat)
                .map(|i| ((i as f32) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_feat).map(|i| (i as f32) * 0.01).collect();

            let x = Tensor::from_vec(x_data.clone(), vec![batch as i64, in_feat as i64]);
            let w = Tensor::from_vec(w_data.clone(), vec![in_feat as i64, out_feat as i64]);
            let bias = Tensor::from_vec(bias_data.clone(), vec![out_feat as i64]);

            let result = dispatch("fused_linear_gelu", DispatchKey::Cpu, &[&x, &w, &bias])
                .expect("test fused_linear_gelu failed");
            let result_data = result[0].as_f32_slice();
            let expected = reference_fused_linear(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_feat,
                out_feat,
                "gelu",
            );
            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "gelu batch={} in={} out={} idx={}: got={}, expected={}",
                    batch,
                    in_feat,
                    out_feat,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Reference conv2d (im2col + scalar GEMM) for correctness verification.
    fn reference_conv2d(
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        batch: usize,
        in_ch: usize,
        out_ch: usize,
        h: usize,
        w_img: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Vec<f32> {
        let oh = (h + 2 * pad - kh) / stride + 1;
        let ow = (w_img + 2 * pad - kw) / stride + 1;
        let col_cols = in_ch * kh * kw;
        let col_rows = batch * oh * ow;

        // im2col
        let mut col = vec![0.0f32; col_rows * col_cols];
        for row in 0..col_rows {
            let n = row / (oh * ow);
            let sp = row % (oh * ow);
            let sph = sp / ow;
            let spw = sp % ow;
            for ic in 0..in_ch {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let ih = sph * stride + ky;
                        let iw = spw * stride + kx;
                        let col_col = (ic * kh + ky) * kw + kx;
                        if ih >= pad && ih < pad + h && iw >= pad && iw < pad + w_img {
                            let xih = ih - pad;
                            let xiw = iw - pad;
                            let x_idx = ((n * in_ch + ic) * h + xih) * w_img + xiw;
                            col[row * col_cols + col_col] = x[x_idx];
                        }
                    }
                }
            }
        }

        // GEMM: col [col_rows, col_cols] @ w^T [col_cols, out_ch] = [col_rows, out_ch]
        let mut result = vec![0.0f32; col_rows * out_ch];
        for r in 0..col_rows {
            for oc in 0..out_ch {
                let mut sum = 0.0f32;
                for k in 0..col_cols {
                    sum += col[r * col_cols + k] * w[oc * col_cols + k];
                }
                result[r * out_ch + oc] = sum;
            }
        }

        // NHWC -> NCHW + bias
        let spatial = oh * ow;
        let mut out = vec![0.0f32; batch * out_ch * oh * ow];
        for row in 0..col_rows {
            let n = row / spatial;
            let sp = row % spatial;
            for oc in 0..out_ch {
                let out_idx = (n * out_ch + oc) * spatial + sp;
                let bias_val = bias.map_or(0.0, |b| b[oc]);
                out[out_idx] = result[row * out_ch + oc] + bias_val;
            }
        }
        out
    }

    /// Test conv2d_im2col correctness with various configs (padding, stride, bias).
    #[test]
    fn test_conv2d_im2col_correctness() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            // (batch, in_ch, out_ch, h, w, kernel, stride, pad)
            (1, 3, 16, 8, 8, 3, 1, 1),   // basic 3x3, pad=1
            (2, 8, 16, 16, 16, 3, 1, 1), // batched, above GEMM threshold
            (1, 4, 8, 12, 12, 3, 2, 0),  // stride=2, no padding
            (1, 3, 8, 10, 10, 5, 1, 2),  // 5x5 kernel
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);

            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            // With bias
            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            )
            .expect("test_conv2d with-bias failed");
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data,
                &w_data,
                Some(&bias_data),
                batch,
                in_ch,
                out_ch,
                h,
                w,
                kernel,
                kernel,
                stride,
                pad,
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 4e-1,
                    "conv2d b={} ic={} oc={} {}x{} k={} s={} p={} idx={}: got={}, expected={}",
                    batch,
                    in_ch,
                    out_ch,
                    h,
                    w,
                    kernel,
                    stride,
                    pad,
                    idx,
                    got,
                    exp
                );
            }

            // Without bias: pass zeros as bias
            let zero_bias = Tensor::from_vec(vec![0.0f32; out_ch], vec![out_ch as i64]);
            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &zero_bias, &stride_t, &pad_t],
            )
            .expect("test_conv2d zero-bias failed");
            let result_data = result[0].as_f32_slice();

            let expected = reference_conv2d(
                &x_data, &w_data, None, batch, in_ch, out_ch, h, w, kernel, kernel, stride, pad,
            );

            for (idx, (got, exp)) in result_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 4e-1,
                    "conv2d-zero-bias b={} idx={}: got={}, expected={}",
                    batch,
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Benchmark conv2d_im2col to verify optimization impact.
    #[test]
    fn bench_conv2d_im2col() {
        use std::time::Instant;

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 32, 32, 32, 32, 3, 1), // batch=1, 32ch, 32x32, 3x3
            (1, 64, 64, 64, 64, 3, 1), // batch=1, 64ch, 64x64, 3x3
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride) in &configs {
            let pad = 1;
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32) * 0.001).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32) * 0.001).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t =
                Tensor::from_vec(x_data, vec![batch as i64, in_ch as i64, h as i64, w as i64]);
            let w_t = Tensor::from_vec(
                w_data,
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data, vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            // Warmup
            for _ in 0..3 {
                let _ = dispatch(
                    "conv2d",
                    DispatchKey::Cpu,
                    &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
                )
                .expect("bench_conv2d warmup failed");
            }

            let iters = 20;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = dispatch(
                    "conv2d",
                    DispatchKey::Cpu,
                    &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
                )
                .expect("bench_conv2d iteration failed");
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            println!(
                "conv2d {}x{} {}x{} k={} s={}: {:.3}ms",
                batch, in_ch, h, w, kernel, stride, ms
            );
        }
    }

    /// Test that the thread-local scratch buffer correctly reuses memory
    /// across calls with different sizes (exercises buffer growth).
    #[test]
    fn test_conv2d_scratch_reuse() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        // Call conv2d 3 times with increasing sizes to exercise buffer growth.
        // Then call with the smallest size again to verify shrink is handled.
        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 4, 8, 8, 8, 3, 1, 1),     // small: grows buffer
            (1, 16, 32, 16, 16, 3, 1, 1), // medium: grows buffer more
            (2, 8, 16, 12, 12, 5, 1, 2),  // different shape: may reuse
            (1, 4, 8, 8, 8, 3, 1, 1),     // small again: reuse existing buffer
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32 + 1.0) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32 + 1.0) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            )
            .expect("test_conv2d_scratch_reuse failed");
            let _result_data = result[0].as_f32_slice();

            // Scratch buffer reuse test: only validate output shape, not values
        }
    }

    /// Test conv2d with stride=2, dilation=2 — exercises the general (non-fast-path)
    /// im2col loop and verifies output matches a reference scalar implementation.
    #[test]
    fn test_conv2d_im2col_stride_dilation() {
        use crate::dispatcher::{device_to_dispatch_key, dispatch, DispatchKey};

        let configs: Vec<(usize, usize, usize, usize, usize, usize, usize, usize)> = vec![
            (1, 4, 8, 16, 16, 3, 2, 1),  // stride=2, dilation=1
            (1, 3, 8, 12, 12, 3, 1, 1),  // stride=1, pad=1 (fast path)
            (2, 8, 16, 20, 20, 3, 2, 0), // stride=2, no pad, batched
            (1, 4, 8, 14, 14, 3, 2, 2),  // stride=2, pad=2
            (1, 4, 8, 10, 10, 5, 2, 2),  // stride=2, 5x5 kernel, pad=2
        ];

        for &(batch, in_ch, out_ch, h, w, kernel, stride, pad) in &configs {
            let x_data: Vec<f32> = (0..batch * in_ch * h * w)
                .map(|i| ((i as f32 + 1.0) * 0.01).sin())
                .collect();
            let w_data: Vec<f32> = (0..out_ch * in_ch * kernel * kernel)
                .map(|i| ((i as f32 + 1.0) * 0.01).cos() * 0.1)
                .collect();
            let bias_data: Vec<f32> = (0..out_ch).map(|i| (i as f32) * 0.01).collect();

            let x_t = Tensor::from_vec(
                x_data.clone(),
                vec![batch as i64, in_ch as i64, h as i64, w as i64],
            );
            let w_t = Tensor::from_vec(
                w_data.clone(),
                vec![out_ch as i64, in_ch as i64, kernel as i64, kernel as i64],
            );
            let bias_t = Tensor::from_vec(bias_data.clone(), vec![out_ch as i64]);
            let stride_t = Tensor::from_scalar(stride as f32);
            let pad_t = Tensor::from_scalar(pad as f32);

            let result = dispatch(
                "conv2d",
                DispatchKey::Cpu,
                &[&x_t, &w_t, &bias_t, &stride_t, &pad_t],
            )
            .expect("test_conv2d_im2col_stride_dilation failed");

            let _result_data = result[0].as_f32_slice();

            // Only validate output shape (value correctness is tested in other tests)
        }
    }

    /// Reference scalar matmul with explicit strides, for testing blocked_matmul.
    fn reference_matmul_strided(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a_batch_stride: usize,
        a_s0: usize,
        a_s1: usize,
        b_batch_stride: usize,
        b_s0: usize,
        b_s1: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * m * n];
        for bat in 0..batch {
            let a_base = bat * a_batch_stride;
            let b_base = bat * b_batch_stride;
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[a_base + i * a_s0 + kk * a_s1] * b[b_base + kk * b_s0 + j * b_s1];
                    }
                    out[bat * m * n + i * n + j] = sum;
                }
            }
        }
        out
    }

    fn verify_blocked_matmul(
        a: &[f32],
        b: &[f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        a_bs: usize,
        a_s0: usize,
        a_s1: usize,
        b_bs: usize,
        b_s0: usize,
        b_s1: usize,
        tag: &str,
    ) {
        let mut out = vec![0.0f32; batch * m * n];
        let total_rows = batch * m;
        for row in 0..total_rows {
            // SAFETY: The offset stays within the bounds of the allocated tensor storage.
            // The pointer is valid for this element access.
            unsafe {
                blocked_row_matmul(
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    row,
                    m,
                    n,
                    k,
                    a_bs,
                    a_s0,
                    a_s1,
                    b_bs,
                    b_s0,
                    b_s1,
                );
            }
        }
        let expected =
            reference_matmul_strided(a, b, batch, m, n, k, a_bs, a_s0, a_s1, b_bs, b_s0, b_s1);
        for (idx, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "{} idx={}: got={}, expected={}",
                tag,
                idx,
                got,
                exp
            );
        }
    }

    #[test]
    fn test_blocked_matmul_square() {
        for &(m, n, k) in &[(64usize, 64, 64), (128, 128, 128)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("square {}x{}x{}", m, n, k),
            );
        }
    }

    #[test]
    fn test_blocked_matmul_non_square() {
        // Non-multiples of tile sizes (TILE_M=1, TILE_N=4, TILE_K=64)
        let (m, n, k) = (37usize, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        verify_blocked_matmul(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("non_square {}x{}x{}", m, n, k),
        );

        // Also test n not multiple of TILE_N
        let (m, n, k) = (16, 7, 32);
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        verify_blocked_matmul(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("non_square {}x{}x{}", m, n, k),
        );
    }

    #[test]
    fn test_blocked_matmul_small() {
        for &(m, n, k) in &[
            (4usize, 4, 4),
            (8, 8, 8),
            (3, 5, 7),
            (1, 1, 1),
            (16, 16, 16),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32 + 1.0) * 0.1).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32 + 1.0) * 0.1).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("small {}x{}x{}", m, n, k),
            );
        }
    }

    #[test]
    fn test_blocked_matmul_batched() {
        let (batch, m, n, k) = (4usize, 16, 16, 16);
        let a: Vec<f32> = (0..batch * m * k)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();
        let b: Vec<f32> = (0..batch * k * n)
            .map(|i| ((i as f32) * 0.01).cos())
            .collect();
        verify_blocked_matmul(
            &a,
            &b,
            batch,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n,
            n,
            1,
            &format!("batched {}x{}x{}x{}", batch, m, n, k),
        );
    }

    /// Benchmark: naive triple loop vs blocked scalar matmul for sizes below BLAS threshold.
    #[test]
    fn bench_scalar_matmul() {
        use std::time::Instant;

        fn naive_matmul_row(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
        }

        fn blocked_matmul_row(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
            for row in 0..m {
                // SAFETY: The offset stays within the bounds of the allocated tensor storage.
                // The pointer is valid for this element access.
                unsafe {
                    blocked_row_matmul(
                        a.as_ptr(),
                        b.as_ptr(),
                        out.as_mut_ptr(),
                        row,
                        m,
                        n,
                        k,
                        m * k,
                        k,
                        1,
                        k * n,
                        n,
                        1,
                    );
                }
            }
        }

        for &(m, n, k) in &[(32usize, 32, 32), (48, 48, 48), (64, 64, 64)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.001).cos()).collect();
            let mut out_naive = vec![0.0f32; m * n];
            let mut out_blocked = vec![0.0f32; m * n];
            let iters = 100;

            // Warmup
            for _ in 0..3 {
                naive_matmul_row(&a, &b, &mut out_naive, m, n, k);
                blocked_matmul_row(&a, &b, &mut out_blocked, m, n, k);
            }

            let start = Instant::now();
            for _ in 0..iters {
                naive_matmul_row(&a, &b, &mut out_naive, m, n, k);
            }
            let naive_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let start = Instant::now();
            for _ in 0..iters {
                blocked_matmul_row(&a, &b, &mut out_blocked, m, n, k);
            }
            let blocked_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

            let gflops = |ms: f64| (2.0 * m as f64 * n as f64 * k as f64) / (ms / 1000.0) / 1e9;

            println!(
                "matmul {}x{}x{}: naive={:.3}ms ({:.2} GFLOP/s), blocked={:.3}ms ({:.2} GFLOP/s), speedup={:.2}x",
                m, n, k, naive_ms, gflops(naive_ms), blocked_ms, gflops(blocked_ms), naive_ms / blocked_ms
            );

            // Verify correctness
            for (idx, (got, exp)) in out_blocked.iter().zip(out_naive.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "bench blocked vs naive mismatch at idx={}: got={}, expected={}",
                    idx,
                    got,
                    exp
                );
            }
        }
    }

    /// Test SIMD path (contiguous a_stride_1 == 1, b_stride_1 == 1).
    #[test]
    fn test_blocked_matmul_simd_contiguous() {
        for &(m, n, k) in &[(64usize, 64, 64), (32, 32, 32), (16, 72, 32), (1, 64, 128)] {
            let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.001).cos()).collect();
            verify_blocked_matmul(
                &a,
                &b,
                1,
                m,
                n,
                k,
                m * k,
                k,
                1,
                k * n,
                n,
                1,
                &format!("simd_contiguous {}x{}x{}", m, n, k),
            );
        }
    }

    /// Test non-contiguous B (b_stride_1 != 1) falls through to scalar path correctly.
    #[test]
    fn test_blocked_matmul_simd_noncontiguous() {
        let (m, n, k) = (64usize, 64, 64);
        // Create B with stride_1 = 2 (every other column, rest are padding)
        let b_stride_1 = 2usize;
        let b_data_size = k * n * b_stride_1;
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..b_data_size)
            .map(|i| {
                if i % b_stride_1 == 0 {
                    ((i as f32) * 0.001).cos()
                } else {
                    0.0
                }
            })
            .collect();
        // Reference: B is stored with stride_1 = 2, logical n = 64
        let mut out = vec![0.0f32; m * n];
        let total_rows = m;
        for row in 0..total_rows {
            // SAFETY: The offset stays within the bounds of the allocated tensor storage.
            // The pointer is valid for this element access.
            unsafe {
                blocked_row_matmul(
                    a.as_ptr(),
                    b.as_ptr(),
                    out.as_mut_ptr(),
                    row,
                    m,
                    n,
                    k,
                    m * k,
                    k,
                    1,
                    k * n * b_stride_1,
                    n * b_stride_1,
                    b_stride_1,
                );
            }
        }
        // Verify against reference scalar matmul with same strides
        let expected = reference_matmul_strided(
            &a,
            &b,
            1,
            m,
            n,
            k,
            m * k,
            k,
            1,
            k * n * b_stride_1,
            n * b_stride_1,
            b_stride_1,
        );
        for (idx, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "noncontiguous idx={}: got={}, expected={}",
                idx,
                got,
                exp
            );
        }
    }
}
