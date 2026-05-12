//! CPU matmul kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use super::*;
use crate::autograd::{AutogradMeta, Edge, Node};
use crate::iterator::TensorIterator;
use crate::error::FastnnError;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;

pub fn matmul_kernel(args: &[&Tensor]) -> Result<Vec<Tensor>, FastnnError> {
    // Removed debug file writing that was causing issues on Windows

    if args.len() < 2 {
        return Err(FastnnError::InvalidArgument(format!(
            "matmul: expected at least 2 arguments, got {}", args.len()
        )));
    }

    let a = if args[0].is_contiguous() {
        args[0].clone()
    } else {
        args[0].contiguous()
    };
    let b = if args[1].is_contiguous() {
        args[1].clone()
    } else {
        args[1].contiguous()
    };

    let a_shape = a.shape_ref();
    let b_shape = b.shape_ref();

    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(FastnnError::Shape(format!(
            "matmul: both tensors must have at least 2 dimensions, got {} and {}",
            a_shape.len(), b_shape.len()
        )));
    }

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    // Both tensors are now contiguous (ensured above), so no transposed
    // stride detection is needed. The 3D flattening below also produces
    // contiguous tensors. This avoids a class of bugs where a 4D batched
    // tensor with transposed inner dims (e.g. from attention softmax
    // followed by transpose) is incorrectly detected as "transposed" at
    // the 4D level, causing the stride-based check to use a batch
    // dimension as the inner dimension and panic.

    // Both tensors are contiguous (ensured above), so B is never transposed.
    // Standard dimension compatibility check: B's second-to-last dim must equal A's last dim (k).
    if b_shape[b_shape.len() - 2] as i32 != k {
        return Err(FastnnError::Shape(format!(
            "matmul: A[{}, {}] @ B[{}, {}] - B second-to-last dim {} != k {}",
            m, k,
            b_shape[b_shape.len() - 2], b_shape[b_shape.len() - 1],
            b_shape[b_shape.len() - 2], k
        )));
    }
    // Use custom tiled matmul
    let batch_a = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch_b = if b_shape.len() > 2 {
        b_shape[..b_shape.len() - 2].iter().product::<i64>() as usize
    } else {
        1
    };
    let batch = batch_a.max(batch_b);

    // Save original shapes for output reshape
    let orig_a_shape = a_shape;

    // For N-D tensors (N > 3), flatten all batch dims into a single batch dim
    // by reshaping to 3D. This avoids incorrect batch stride calculations.
    let a_3d = if a_shape.len() > 3 {
        let flat_batch: i64 = a_shape[..a_shape.len() - 2].iter().product();
        a.reshape(vec![
            flat_batch,
            a_shape[a_shape.len() - 2],
            a_shape[a_shape.len() - 1],
        ])
    } else {
        a.clone()
    };
    let b_3d = if b_shape.len() > 3 {
        let flat_batch: i64 = b_shape[..b_shape.len() - 2].iter().product();
        b.reshape(vec![
            flat_batch,
            b_shape[b_shape.len() - 2],
            b_shape[b_shape.len() - 1],
        ])
    } else {
        b.clone()
    };

    // Validate inner dimensions on the (possibly flattened) 3D tensors.
    // This must run after batch flattening so shape indexing is unambiguous
    // for ND inputs (e.g. 4D batched attention tensors).
    let a_3d_shape = a_3d.shape_ref();
    let b_3d_shape = b_3d.shape_ref();
    let a_inner = a_3d_shape[a_3d_shape.len() - 1];
    let b_inner = b_3d_shape[b_3d_shape.len() - 2];
    if a_inner != b_inner {
        return Err(FastnnError::Shape(format!(
            "matmul: inner dimension mismatch: A[..{}] @ B[..{}]",
            a_inner, b_inner
        )));
    }

    let a = &a_3d;
    let b = &b_3d;
    let a_shape = a.shape_ref();
    let b_shape = b.shape_ref();
    let a_strides = a.strides();
    let b_strides = b.strides();

    let mut output_shape: smallvec::SmallVec<[i64; 4]> =
        smallvec::SmallVec::with_capacity(a_shape.len());
    if a_shape.len() > 2 {
        for i in 0..a_shape.len() - 2 {
            // If b has matching dimensions, use max (broadcasting)
            // If b is 2D (no batch dims), just use a's batch dims
            if b_shape.len() > 2 && i < b_shape.len() - 2 {
                output_shape.push(a_shape[i].max(b_shape[i]));
            } else {
                output_shape.push(a_shape[i]);
            }
        }
    }
    output_shape.push(m as i64);
    output_shape.push(n as i64);

    let mut output = Tensor::empty(output_shape.to_vec(), a.dtype(), a.device());

    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        return Err(FastnnError::Computation("matmul: expected CPU storage".into()));
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_rows = a_shape[a_shape.len() - 2] as usize;
    let a_cols = a_shape[a_shape.len() - 1] as usize;
    let b_cols = b_shape[b_shape.len() - 1] as usize;

    // Detect transposed matrices by checking strides
    // For row-major contiguous matrix [rows, cols], stride_0 = cols, stride_1 = 1
    // For transposed [rows, cols] stored as [cols, rows], stride_0 = 1, stride_1 = rows
    let a_stride_0 = a_strides[a.ndim() - 2];
    let a_stride_1 = a_strides[a.ndim() - 1];
    let b_stride_0 = b_strides[b.ndim() - 2];
    let b_stride_1 = b_strides[b.ndim() - 1];

    let a_batch_stride = if a.ndim() > 2 { a_strides[0] } else { 0 };
    let b_batch_stride = if b.ndim() > 2 { b_strides[0] } else { 0 };
    // Detect transposed matrices by checking strides
    // For row-major contiguous matrix [rows, cols], stride_0 = cols, stride_1 = 1
    // For transposed [rows, cols] stored as [cols, rows], stride_0 = 1, stride_1 = rows
    let a_is_transposed = a_stride_0 == 1 && a_stride_1 == a_rows as i64;
    let b_is_transposed = b_stride_0 == 1 && b_stride_1 == k as i64;

    // For BLAS, we need contiguous matrices or simple 2D transposition
    // A transposed matrix has stride_0 = 1 and stride_1 = original_rows
    let a_valid_for_blas = a.is_contiguous() || a_is_transposed;
    let b_valid_for_blas = b.is_contiguous() || b_is_transposed;

    // Normalise batch strides for broadcast semantics: when an operand has
    // a single batch (e.g. [1, k, n]), its stride to the next batch should
    // be 0 so that indexing repeats the same data across all output batches.
    let a_batch_stride = if a_batch_stride != 0 && batch_a == 1 { 0 } else { a_batch_stride };
    let b_batch_stride = if b_batch_stride != 0 && batch_b == 1 { 0 } else { b_batch_stride };

    // Reshape trick: For batched A [batch, m, k] @ 2D B [k, n], flatten to [batch*m, k] @ [k, n]
    // This enables a single BLAS call instead of looping over batch dimension
    let can_reshape_trick = batch > 1
        && b_shape.len() == 2  // B is 2D, not batched
        && a.is_contiguous()
        && b_valid_for_blas;

    // Batched 3D BLAS: both A and B are batched 3D contiguous tensors
    let can_batch_blas = batch > 1
        && a_shape.len() == 3
        && b_shape.len() == 3
        && a.is_contiguous()
        && b.is_contiguous();

    // Use BLAS for:
    // 1. Single batch with large enough matrices
    // 2. Batched operations with 2D weight (reshape trick)
    // 3. Batched 3D contiguous tensors (batched BLAS loop)
    let matrices_large = m as usize * n as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE
        || m as usize * k as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE
        || k as usize * n as usize >= MIN_BLAS_SIZE * MIN_BLAS_SIZE;
    let use_blas = (can_reshape_trick || batch == 1 || can_batch_blas)
        && matrices_large
        && a_valid_for_blas
        && b_valid_for_blas;

    if use_blas {
        if can_reshape_trick {
            // Reshape trick: treat [batch, m, k] as [batch*m, k]
            // Single BLAS call for entire batch
            let batch_m = batch * m as usize;
            // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, batch_m * k as usize) };
            // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
            let out_slice =
                // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
                unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_m * n as usize) };
            matmul_blas_with_transpose_into(
                a_slice,
                b_slice,
                out_slice,
                batch_m,
                k as usize,
                n as usize,
                a_is_transposed,
                b_is_transposed,
            );
        } else if can_batch_blas {
            // Batched 3D BLAS loop: process each batch element separately
            // The slice spans are based on the *actual* batch per operand, not
            // the max() broadcast batch, to avoid out-of-bounds access when
            // one side has batch=1 (broadcast semantics).
            let a_slice =
                // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
                unsafe { std::slice::from_raw_parts(a_ptr, batch_a * m as usize * k as usize) };
            let b_slice =
                // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
                unsafe { std::slice::from_raw_parts(b_ptr, batch_b * k as usize * n as usize) };
            let out_slice =
                // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
                unsafe { std::slice::from_raw_parts_mut(out_ptr, batch * m as usize * n as usize) };

            let m_usize = m as usize;
            let k_usize = k as usize;
            let n_usize = n as usize;
            let a_batch_elems = m_usize * k_usize;
            let b_batch_elems = k_usize * n_usize;
            let out_batch_elems = m_usize * n_usize;

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                out_slice
                    .par_chunks_mut(out_batch_elems)
                    .enumerate()
                    .for_each(|(bat, out_chunk)| {
                        let a_offset = if batch_a == 1 { 0 } else { bat * a_batch_elems };
                        let b_offset = if batch_b == 1 { 0 } else { bat * b_batch_elems };
                        matmul_blas_with_transpose_into(
                            &a_slice[a_offset..],
                            &b_slice[b_offset..],
                            out_chunk,
                            m_usize,
                            k_usize,
                            n_usize,
                            false,
                            false,
                        );
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for bat in 0..batch {
                    let a_offset = if batch_a == 1 { 0 } else { bat * a_batch_elems };
                    let b_offset = if batch_b == 1 { 0 } else { bat * b_batch_elems };
                    let out_offset = bat * out_batch_elems;
                    matmul_blas_with_transpose_into(
                        &a_slice[a_offset..],
                        &b_slice[b_offset..],
                        &mut out_slice[out_offset..],
                        m_usize,
                        k_usize,
                        n_usize,
                        false,
                        false,
                    );
                }
            }
        } else {
            // Single batch BLAS
            // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, a_rows * a_cols) };
            // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
            let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, k as usize * b_cols) };
            let out_slice =
                // SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
                unsafe { std::slice::from_raw_parts_mut(out_ptr, m as usize * n as usize) };
            matmul_blas_with_transpose_into(
                a_slice,
                b_slice,
                out_slice,
                m as usize,
                k as usize,
                n as usize,
                a_is_transposed,
                b_is_transposed,
            );
        }
    } else { unsafe {
        #[cfg(feature = "parallel")]
        {
            if batch > 1 || m as usize * n as usize > 10000 {
                parallel_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else {
                single_threaded_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            if m as usize <= 64 && n as usize <= 64 && k as usize <= 64 {
                small_matrix_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            } else {
                single_threaded_matmul(
                    a_ptr,
                    b_ptr,
                    out_ptr,
                    batch,
                    m,
                    n,
                    k,
                    a_batch_stride,
                    a_stride_0,
                    a_stride_1,
                    b_batch_stride,
                    b_stride_0,
                    b_stride_1,
                );
            }
        }
    } }

    // Reshape output back to original N-D shape if we flattened
    if orig_a_shape.len() > 3 {
        let mut final_shape: Vec<i64> = orig_a_shape[..orig_a_shape.len() - 2].to_vec();
        final_shape.push(m as i64);
        final_shape.push(n as i64);
        output = output.reshape(final_shape);
    }

    Ok(vec![output])
}

#[cfg(feature = "parallel")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn parallel_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let a_usize = a_ptr as usize;
    let b_usize = b_ptr as usize;
    let out_usize = out_ptr as usize;
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    let abs = a_batch_stride as usize;
    let as0 = a_stride_0 as usize;
    let as1 = a_stride_1 as usize;
    let bbs = b_batch_stride as usize;
    let bs0 = b_stride_0 as usize;
    let bs1 = b_stride_1 as usize;

    let num_threads = rayon::current_num_threads();
    let chunk_size = (total_rows / (num_threads * 4)).max(1);

    (0..total_rows)
        .into_par_iter()
        .chunks(chunk_size)
        .for_each(|row_chunk| {
            for &row_idx in &row_chunk {
                blocked_row_matmul(
                    a_usize as *const f32,
                    b_usize as *const f32,
                    out_usize as *mut f32,
                    row_idx,
                    m_usize,
                    n_usize,
                    k_usize,
                    abs,
                    as0,
                    as1,
                    bbs,
                    bs0,
                    bs1,
                );
            }
        });
}

/// Cache-blocked scalar matmul for one row of output: C[row, :] += A[row, :] @ B.
/// Tiles the K dimension so that a TILE_K × TILE_N block of B stays in L1 cache.
///
/// When strides are contiguous (a_stride_1 == 1, b_stride_1 == 1) and the
/// target supports AVX2, uses an 8-wide SIMD inner loop that broadcasts
/// A[i, kk] and FMADDs with the contiguous B[kk, j..j+8] row slice.
///
/// Cache math (scalar TILE_N=4):
///   B tile: 64 × 4 × 4 bytes = 1 KB
///   A tile: 1 × 64 × 4 bytes = 256 bytes
///   Total working set ≪ 32 KB L1 cache.
///
/// SIMD cache math (TILE_N_SIMD=8):
///   B tile: 64 × 8 × 4 bytes = 2 KB
///   Still well within L1.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn blocked_row_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    row: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch_stride: usize,
    a_stride_0: usize,
    a_stride_1: usize,
    b_batch_stride: usize,
    b_stride_0: usize,
    b_stride_1: usize,
) {
    const TILE_K: usize = 64;
    const TILE_N: usize = 4;

    let bat = row / m;
    let i = row % m;
    let a_off = bat * a_batch_stride + i * a_stride_0;
    let b_off = bat * b_batch_stride;
    let out_off = row * n;

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let use_simd = a_stride_1 == 1 && b_stride_1 == 1 && n >= 8;
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    let use_simd = false;

    // Zero output before accumulating. Both SIMD and scalar paths accumulate
    // across K-tiles (the outer `ko` loop), so the output must start at zero.
    // Previously the SIMD path skipped this, causing accumulation into
    // uninitialized/garbage memory when Tensor::empty was used (or into stale
    // values when the same buffer was reused across iterations as in the test).
    for j in 0..n {
        // SAFETY: Pointer arithmetic stays within bounds of the allocated tensor storage.
        unsafe {
            *out_ptr.add(out_off + j) = 0.0;
        }
    }

    // AVX2 SIMD fast path: 8-wide FMADD when A and B have contiguous columns.
    // For fixed (i, kk): C[i, j..j+8] += A[i, kk] * B[kk, j..j+8]
    // B[kk, j..j+8] is 8 contiguous floats — a correct plain load.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if use_simd {
        use std::arch::x86_64::*;

        const TILE_N_SIMD: usize = 8;
        let mut ko = 0;
        while ko < k {
            let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

            // SIMD tiles: 8 columns at a time
            let mut jo = 0;
            while jo + TILE_N_SIMD <= n {
                unsafe {
                    let mut acc = _mm256_setzero_ps();

                    let mut kk = ko;
                    while kk + 4 <= kend {
                        // Unroll 4 k-steps for ILP
                        let a0 = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let b0 = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a0, b0, acc);

                        let a1 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 1) * a_stride_1));
                        let b1 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 1) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a1, b1, acc);

                        let a2 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 2) * a_stride_1));
                        let b2 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 2) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a2, b2, acc);

                        let a3 = _mm256_set1_ps(*a_ptr.add(a_off + (kk + 3) * a_stride_1));
                        let b3 = _mm256_loadu_ps(b_ptr.add(b_off + (kk + 3) * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(a3, b3, acc);

                        kk += 4;
                    }
                    // Scalar tail for remaining kk
                    while kk < kend {
                        let av = _mm256_set1_ps(*a_ptr.add(a_off + kk * a_stride_1));
                        let bv = _mm256_loadu_ps(b_ptr.add(b_off + kk * b_stride_0 + jo));
                        acc = _mm256_fmadd_ps(av, bv, acc);
                        kk += 1;
                    }

                    // Accumulate into output (add to existing, not overwrite)
                    let out_v = _mm256_loadu_ps(out_ptr.add(out_off + jo));
                    _mm256_storeu_ps(out_ptr.add(out_off + jo), _mm256_add_ps(out_v, acc));
                }
                jo += TILE_N_SIMD;
            }

            // Scalar tail for remaining columns (< 8)
            while jo < n {
                let mut sum = 0.0f32;
                for kk in ko..kend {
                    sum += unsafe {
                        *a_ptr.add(a_off + kk * a_stride_1)
                            * b_ptr.add(b_off + kk * b_stride_0 + jo).read()
                    };
                }
                unsafe {
                    let p = out_ptr.add(out_off + jo);
                    *p += sum;
                }
                jo += 1;
            }

            ko += TILE_K;
        }
        return;
    }

    // Scalar blocked path (handles non-contiguous strides and small n)
    let mut ko = 0;
    while ko < k {
        let kend = if ko + TILE_K < k { ko + TILE_K } else { k };

        let mut jo = 0;
        while jo + TILE_N <= n {
            let mut acc = [0.0f32; TILE_N];

            let mut kk = ko;
            while kk < kend {
                let av = unsafe { *a_ptr.add(a_off + kk * a_stride_1) };
                for t in 0..TILE_N {
                    acc[t] +=
                        av * unsafe { *b_ptr.add(b_off + kk * b_stride_0 + (jo + t) * b_stride_1) };
                }
                kk += 1;
            }

            for t in 0..TILE_N {
                unsafe {
                    let p = out_ptr.add(out_off + jo + t);
                    *p += acc[t];
                }
            }
            jo += TILE_N;
        }

        // Tail: remaining columns
        while jo < n {
            let mut sum = 0.0f32;
            for kk in ko..kend {
                sum += unsafe {
                    *a_ptr.add(a_off + kk * a_stride_1)
                        * b_ptr.add(b_off + kk * b_stride_0 + jo * b_stride_1).read()
                };
            }
            unsafe {
                let p = out_ptr.add(out_off + jo);
                *p += sum;
            }
            jo += 1;
        }

        ko += TILE_K;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matrix_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn single_threaded_matmul(
    a_ptr: *const f32,
    b_ptr: *const f32,
    out_ptr: *mut f32,
    batch: usize,
    m: i32,
    n: i32,
    k: i32,
    a_batch_stride: i64,
    a_stride_0: i64,
    a_stride_1: i64,
    b_batch_stride: i64,
    b_stride_0: i64,
    b_stride_1: i64,
) {
    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let total_rows = batch * m_usize;

    for row in 0..total_rows {
        blocked_row_matmul(
            a_ptr,
            b_ptr,
            out_ptr,
            row,
            m_usize,
            n_usize,
            k_usize,
            a_batch_stride as usize,
            a_stride_0 as usize,
            a_stride_1 as usize,
            b_batch_stride as usize,
            b_stride_0 as usize,
            b_stride_1 as usize,
        );
    }
}

pub unsafe fn linear_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape_ref();
    let w_shape = w.shape_ref();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[0];

    let x_flat = x.reshape(vec![batch_size, in_features]);

    // w has shape [out_features, in_features]
    // We need to compute x_flat @ w.T (where w.T has shape [in_features, out_features])
    // The matmul kernel will detect that w is transposed by checking its strides
    // and use the appropriate BLAS transpose flag
    let w_t = w.transpose(0, 1);

    let mut result = (x_flat.matmul(&w_t)).reshape(
        x_shape[..x_shape.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(out_features))
            .collect(),
    );

    if let Some(b) = bias {
        result = result.add(b);
    }

    vec![result]
}

pub unsafe fn fused_linear_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape_ref();
    let w_shape = w.shape_ref();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[1];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_relu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    // Use BLAS for the GEMM when matrices are large enough, then fuse
    // bias + activation in a single parallel pass. This is 3-5x faster
    // than the scalar inner loop because BLAS uses SIMD + cache blocking.
    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        // GEMM: [batch, in] @ [in, out] = [batch, out]
        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            false,
        );

        // Parallel bias + relu pass over all output elements
        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let val = *((out_usize + idx * 4) as *const f32);
                    *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        // Scalar fallback for small matrices
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let total = batch_size * out_features;

            let x_usize = x_ptr as usize;
            let w_usize = w_ptr as usize;
            let out_usize = out_ptr as usize;

            (0..total).into_par_iter().for_each(|idx| {
                let batch_idx = idx / out_features;
                let out_idx = idx % out_features;

                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = k * out_features + out_idx;
                    let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                    let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                unsafe {
                    *((out_usize + idx * 4) as *mut f32) = sum.max(0.0);
                };
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for batch_idx in 0..batch_size {
                for out_idx in 0..out_features {
                    let mut sum = 0.0f32;
                    for k in 0..in_features {
                        let x_offset = batch_idx * in_features + k;
                        let w_offset = k * out_features + out_idx;
                        let x_val = unsafe { *x_ptr.add(x_offset) };
                        let w_val = unsafe { *w_ptr.add(w_offset) };
                        sum += x_val * w_val;
                    }

                    if let Some(b) = bias {
                        let b_ptr = b.data_ptr_f32();
                        sum += unsafe { *b_ptr.add(out_idx) };
                    }

                    unsafe {
                        *out_ptr.add(batch_idx * out_features + out_idx) = sum.max(0.0);
                    };
                }
            }
        }
    }

    vec![output]
}

pub unsafe fn fused_linear_silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape_ref();
    let w_shape = w.shape_ref();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[1];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_silu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;

    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            false,
        );

        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let val = *((out_usize + idx * 4) as *const f32);
                    *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let val = *((out_usize + idx * 4) as *const f32);
                        *((out_usize + idx * 4) as *mut f32) = val / (1.0 + (-val).exp());
                    }
                }
            }
        }
    } else {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let total = batch_size * out_features;

            let x_usize = x_ptr as usize;
            let w_usize = w_ptr as usize;
            let out_usize = out_ptr as usize;

            (0..total).into_par_iter().for_each(|idx| {
                let batch_idx = idx / out_features;
                let out_idx = idx % out_features;

                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = k * out_features + out_idx;
                    let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                    let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                // SiLU: x / (1.0 + (-x).exp())
                let silu_val = sum / (1.0 + (-sum).exp());
                unsafe {
                    *((out_usize + idx * 4) as *mut f32) = silu_val;
                };
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for batch_idx in 0..batch_size {
                for out_idx in 0..out_features {
                    let mut sum = 0.0f32;
                    for k in 0..in_features {
                        let x_offset = batch_idx * in_features + k;
                        let w_offset = k * out_features + out_idx;
                        let x_val = unsafe { *x_ptr.add(x_offset) };
                        let w_val = unsafe { *w_ptr.add(w_offset) };
                        sum += x_val * w_val;
                    }

                    if let Some(b) = bias {
                        let b_ptr = b.data_ptr_f32();
                        sum += unsafe { *b_ptr.add(out_idx) };
                    }

                    // SiLU: x / (1.0 + (-x).exp())
                    let silu_val = sum / (1.0 + (-sum).exp());
                    unsafe {
                        *out_ptr.add(batch_idx * out_features + out_idx) = silu_val;
                    };
                }
            }
        }
    }

    vec![output]
}

pub unsafe fn fused_linear_gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let w = args[1];
    let bias = if args.len() > 2 { Some(args[2]) } else { None };

    let x_shape = x.shape_ref();
    let w_shape = w.shape_ref();

    let batch_size: i64 = if x_shape.len() > 1 {
        x_shape[..x_shape.len() - 1].iter().product()
    } else {
        1
    };
    let in_features = x_shape[x_shape.len() - 1];
    let out_features = w_shape[1];

    let x_ptr = x.data_ptr_f32();
    let w_ptr = w.data_ptr_f32();
    debug_assert!(
        w.is_contiguous(),
        "fused_linear_gelu: weight tensor must be contiguous"
    );

    let output_shape: Vec<i64> = if x_shape.len() > 1 {
        let mut s = x_shape[..x_shape.len() - 1].to_vec();
        s.push(out_features);
        s
    } else {
        vec![out_features]
    };

    let mut output = Tensor::empty(output_shape.clone(), x.dtype(), x.device());
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let batch_size = batch_size as usize;
    let in_features = in_features as usize;
    let out_features = out_features as usize;
    const SQRT_2_OVER_PI: f32 = 0.7978846;
    const GELU_COEFF: f32 = 0.044715;
    let sqrt_2_over_pi = SQRT_2_OVER_PI;
    let coeff = GELU_COEFF;

    let use_blas = x.is_contiguous()
        && w.is_contiguous()
        && (batch_size >= MIN_BLAS_SIZE
            || in_features >= MIN_BLAS_SIZE
            || out_features >= MIN_BLAS_SIZE);

    if use_blas {
        let x_slice = unsafe { std::slice::from_raw_parts(x_ptr, batch_size * in_features) };
        let w_slice = unsafe { std::slice::from_raw_parts(w_ptr, out_features * in_features) };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(out_ptr, batch_size * out_features) };

        matmul_blas_with_transpose_into(
            x_slice,
            w_slice,
            out_slice,
            batch_size,
            in_features,
            out_features,
            false,
            false,
        );

        let total = batch_size * out_features;
        let out_usize = out_ptr as usize;

        if let Some(b) = bias {
            let b_ptr = b.data_ptr_f32();
            let b_usize = b_ptr as usize;
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| {
                    let out_idx = idx % out_features;
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    let out_idx = idx % out_features;
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32)
                            + *((b_usize + out_idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                }
            }
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..total).into_par_iter().for_each(|idx| unsafe {
                    let sum = *((out_usize + idx * 4) as *const f32);
                    let x3 = sum * sum * sum;
                    let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                    *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                });
            }
            #[cfg(not(feature = "parallel"))]
            {
                for idx in 0..total {
                    unsafe {
                        let sum = *((out_usize + idx * 4) as *const f32);
                        let x3 = sum * sum * sum;
                        let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                        *((out_usize + idx * 4) as *mut f32) = 0.5 * sum * (1.0 + t);
                    }
                }
            }
        }
    } else {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let total = batch_size * out_features;

            let x_usize = x_ptr as usize;
            let w_usize = w_ptr as usize;
            let out_usize = out_ptr as usize;

            (0..total).into_par_iter().for_each(|idx| {
                let batch_idx = idx / out_features;
                let out_idx = idx % out_features;

                let mut sum = 0.0f32;
                for k in 0..in_features {
                    let x_offset = batch_idx * in_features + k;
                    let w_offset = k * out_features + out_idx;
                    let x_val = unsafe { *((x_usize + x_offset * 4) as *const f32) };
                    let w_val = unsafe { *((w_usize + w_offset * 4) as *const f32) };
                    sum += x_val * w_val;
                }

                if let Some(b) = bias {
                    let b_ptr = b.data_ptr_f32();
                    sum += unsafe { *b_ptr.add(out_idx) };
                }

                let x3 = sum * sum * sum;
                let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                let gelu = 0.5 * sum * (1.0 + t);

                unsafe {
                    *((out_usize + idx * 4) as *mut f32) = gelu;
                };
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for batch_idx in 0..batch_size {
                for out_idx in 0..out_features {
                    let mut sum = 0.0f32;
                    for k in 0..in_features {
                        let x_offset = batch_idx * in_features + k;
                        let w_offset = k * out_features + out_idx;
                        let x_val = unsafe { *x_ptr.add(x_offset) };
                        let w_val = unsafe { *w_ptr.add(w_offset) };
                        sum += x_val * w_val;
                    }

                    if let Some(b) = bias {
                        let b_ptr = b.data_ptr_f32();
                        sum += unsafe { *b_ptr.add(out_idx) };
                    }

                    let x3 = sum * sum * sum;
                    let t = (sqrt_2_over_pi * (sum + coeff * x3)).tanh();
                    let gelu = 0.5 * sum * (1.0 + t);

                    unsafe {
                        *out_ptr.add(batch_idx * out_features + out_idx) = gelu;
                    };
                }
            }
        }
    }

    vec![output]
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline]
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    ensure_daz_ftz();
    let mut sum = 0.0f32;
    let mut i = 0;

    #[cfg(feature = "simd_avx512")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let mut acc0 = _mm512_setzero_ps();
                let mut acc1 = _mm512_setzero_ps();
                let mut acc2 = _mm512_setzero_ps();
                let mut acc3 = _mm512_setzero_ps();

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

                let acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
                sum += _mm512_reduce_add_ps(acc);

                while i + 16 <= len {
                    let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
                    acc0 = _mm512_fmadd_ps(a_vec, b_vec, _mm512_setzero_ps());
                    sum += _mm512_reduce_add_ps(acc0);
                    i += 16;
                }
            }
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

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

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let mut acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_mut_ptr() as *mut f32, acc);
                // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                sum += acc_arr.assume_init_ref().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let mut prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_mut_ptr() as *mut f32, prod);
                    // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                    sum += prod_arr.assume_init_ref().iter().sum::<f32>();
                    i += 8;
                }
            }
        }
    }

    #[cfg(not(feature = "simd_avx512"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();

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

                let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
                let mut acc_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                _mm256_storeu_ps(acc_arr.as_mut_ptr() as *mut f32, acc);
                // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                sum += acc_arr.assume_init_ref().iter().sum::<f32>();

                while i + 8 <= len {
                    let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
                    let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
                    let prod = _mm256_mul_ps(a_vec, b_vec);
                    let mut prod_arr = std::mem::MaybeUninit::<[f32; 8]>::uninit();
                    _mm256_storeu_ps(prod_arr.as_mut_ptr() as *mut f32, prod);
                    // SAFETY: All 8 lanes are initialized by _mm256_storeu_ps above.
                    sum += prod_arr.assume_init_ref().iter().sum::<f32>();
                    i += 8;
                }
            }
        }
    }

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

#[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
#[inline]
pub unsafe fn simd_dot_product(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= len {
        sum += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}
