//! Fast contiguous last-dim reduction helpers.
use crate::tensor::Tensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Fast contiguous last-dim sum with SIMD
pub fn sum_last_dim_contiguous(a: &Tensor, dim_size: usize, num_rows: usize) -> Tensor {
    let a_ptr = a.data_ptr_f32();

    let mut result_data = vec![0.0f32; num_rows];

    #[cfg(feature = "parallel")]
    {
        if num_rows > 64 {
            use rayon::prelude::*;
            let out_ptr = result_data.as_mut_ptr();
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_rows).into_par_iter().for_each(|row| {
                // SAFETY: `a_usize` is derived from a valid f32 pointer and `row * dim_size`
                // is in-bounds for the allocation.
                let row_ptr = unsafe { (a_usize as *const f32).add(row * dim_size) };
                let mut sum = 0.0f32;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                        // SAFETY: `row_ptr` is valid for `dim_size` f32 elements; AVX2
                        // loads/stores use unaligned instructions which are safe for any
                        // valid pointer.
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
                            // SAFETY: Same as above — `row_ptr` is valid for `dim_size` elements.
                            unsafe {
                                sum += *row_ptr.add(j);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for j in 0..dim_size {
                        // SAFETY: `row_ptr` is valid for `dim_size` elements.
                        unsafe {
                            sum += *row_ptr.add(j);
                        }
                    }
                }
                // SAFETY: `out_usize` is derived from a valid mutable f32 pointer and
                // `row` is within `num_rows`.
                unsafe {
                    *(out_usize as *mut f32).add(row) = sum;
                }
            });
            return Tensor::from_vec(result_data, vec![num_rows as i64]);
        }
    }

    for row in 0..num_rows {
        // SAFETY: `a_ptr` is valid for `num_rows * dim_size` f32 elements and
        // `row * dim_size` is in-bounds.
        let row_ptr = unsafe { a_ptr.add(row * dim_size) };
        let mut sum = 0.0f32;
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && dim_size >= 8 {
                // SAFETY: `row_ptr` is valid for `dim_size` f32 elements; AVX2
                // unaligned loads are safe for any valid pointer.
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
                    // SAFETY: `row_ptr` is valid for `dim_size` elements.
                    unsafe {
                        sum += *row_ptr.add(j);
                    }
                }
            }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            for j in 0..dim_size {
                // SAFETY: `row_ptr` is valid for `dim_size` elements.
                unsafe {
                    sum += *row_ptr.add(j);
                }
            }
        }
        result_data[row] = sum;
    }
    Tensor::from_vec(result_data, vec![num_rows as i64])
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
// SAFETY: `v` is a valid __m256 register value from an AVX2 intrinsic operation.
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let v8 = _mm256_hadd_ps(v, v);
    let v4 = _mm256_hadd_ps(v8, v8);
    let lo = _mm256_castps256_ps128(v4);
    let hi = _mm256_extractf128_ps(v4, 1);
    let sum = _mm_add_ss(lo, hi);
    _mm_cvtss_f32(sum)
}
