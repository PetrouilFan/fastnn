//! CPU elementwise kernels.

#![allow(unused_imports)]
#![allow(clippy::missing_safety_doc)]

use super::*;
use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose, matmul_blas_with_transpose_into,
    MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use half;
use std::sync::Arc;

const GELU_SQRT_2_OVER_PI: f32 = 0.7978846;
const GELU_COEFF: f32 = 0.044715;

pub unsafe fn add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let dtype = a.dtype();

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if dtype == DType::F32 && a_contig && b_contig && a_shape == b_shape {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            add_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            add_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        add_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        add_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    add_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_add_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_add_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_add_ps(a0, b0);
                        let r1 = _mm256_add_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_add_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vaddq_f32(a0, b0);
                        let r1 = vaddq_f32(a1, b1);
                        let r2 = vaddq_f32(a2, b2);
                        let r3 = vaddq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vaddq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) + *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    // Get raw byte pointers
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr();

    let out_usize = out_ptr as usize;
    iter.for_each_with_index(|idx, input_ptrs| unsafe {
        let a_val = match dtype {
            DType::F32 => *(input_ptrs[0].as_ptr() as *const f32),
            DType::F16 => half::f16::to_f32(*(input_ptrs[0].as_ptr() as *const half::f16)),
            DType::BF16 => half::bf16::to_f32(*(input_ptrs[0].as_ptr() as *const half::bf16)),
            _ => panic!("Unsupported dtype for add"),
        };
        let b_val = match dtype {
            DType::F32 => *(input_ptrs[1].as_ptr() as *const f32),
            DType::F16 => half::f16::to_f32(*(input_ptrs[1].as_ptr() as *const half::f16)),
            DType::BF16 => half::bf16::to_f32(*(input_ptrs[1].as_ptr() as *const half::bf16)),
            _ => panic!("Unsupported dtype for add"),
        };
        let sum = a_val + b_val;
        match dtype {
            DType::F32 => {
                *((out_usize as *mut f32).add(idx)) = sum;
            }
            DType::F16 => {
                *((out_usize as *mut half::f16).add(idx)) = half::f16::from_f32(sum);
            }
            DType::BF16 => {
                *((out_usize as *mut half::bf16).add(idx)) = half::bf16::from_f32(sum);
            }
            _ => {}
        }
    });

    vec![output]
}

pub unsafe fn sub_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sub_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sub_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        sub_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    sub_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_sub_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_sub_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_sub_ps(a0, b0);
                        let r1 = _mm256_sub_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_sub_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vsubq_f32(a0, b0);
                        let r1 = vsubq_f32(a1, b1);
                        let r2 = vsubq_f32(a2, b2);
                        let r3 = vsubq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vsubq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) - *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) - *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let out_usize = out_ptr as usize;
    iter.for_each_with_index(|idx, input_ptrs| unsafe {
        let a_val = *(input_ptrs[0].as_ptr() as *const f32);
        let b_val = *(input_ptrs[1].as_ptr() as *const f32);
        *((out_usize as *mut f32).add(idx)) = a_val - b_val;
    });

    vec![output]
}

pub unsafe fn mul_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.inner.sizes.as_slice();
    let b_shape = b.inner.sizes.as_slice();
    let a_contig = a.is_contiguous();
    let b_contig = b.is_contiguous();
    let a_numel = a.inner.numel() as usize;

    if a_contig && b_contig && a_shape == b_shape {
        let output_shape = a_shape.to_vec();
        let numel = a_numel;

        let mut output = Tensor::empty(output_shape, a.dtype(), a.device());

        let a_ptr = a.data_ptr_f32();
        let b_ptr = b.data_ptr_f32();

        let output_inner = Arc::make_mut(&mut output.inner);
        let output_storage = Arc::make_mut(&mut output_inner.storage);
        let Storage::Cpu(cpu_storage) = output_storage else {
            panic!("Expected CPU storage");
        };
        let out_data = Arc::make_mut(&mut cpu_storage.data);
        let out_ptr = out_data.as_mut_ptr() as *mut f32;

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            mul_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            mul_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        mul_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        mul_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    mul_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_mul_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_mul_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_mul_ps(a0, b0);
                        let r1 = _mm256_mul_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_mul_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vmulq_f32(a0, b0);
                        let r1 = vmulq_f32(a1, b1);
                        let r2 = vmulq_f32(a2, b2);
                        let r3 = vmulq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vmulq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx);
                    }
                }
            }
        }
        return vec![output];
    }

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let _numel = output_shape.iter().product::<i64>() as usize;
    let _a_ptr = a.data_ptr() as *const f32;
    let _b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let _a_storage_offset = a.inner.storage_offset as usize;
    let _b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let _out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let _a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let _b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let _a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let _b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

    let out_usize = out_ptr as usize;
    iter.for_each_with_index(|idx, input_ptrs| unsafe {
        let a_val = *(input_ptrs[0].as_ptr() as *const f32);
        let b_val = *(input_ptrs[1].as_ptr() as *const f32);
        *((out_usize as *mut f32).add(idx)) = a_val * b_val;
    });

    vec![output]
}

pub unsafe fn div_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let iter = TensorIterator::build_for_binary(a, b);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    let a_shape: Vec<i64> = a.inner.sizes.iter().copied().collect();
    let b_shape: Vec<i64> = b.inner.sizes.iter().copied().collect();
    let out_shape = output_shape.clone();

    let a_strides: Vec<i64> = a.inner.strides.iter().copied().collect();
    let b_strides: Vec<i64> = b.inner.strides.iter().copied().collect();
    let _a_storage_offset = a.inner.storage_offset as usize;
    let _b_storage_offset = b.inner.storage_offset as usize;

    // Convert to usize for the helper function (use SmallVec to avoid heap alloc)
    let _out_shape_usize: smallvec::SmallVec<[usize; 8]> =
        out_shape.iter().map(|&x| x as usize).collect();
    let _a_shape_usize: smallvec::SmallVec<[usize; 8]> =
        a_shape.iter().map(|&x| x as usize).collect();
    let _b_shape_usize: smallvec::SmallVec<[usize; 8]> =
        b_shape.iter().map(|&x| x as usize).collect();
    let _a_strides_usize: smallvec::SmallVec<[usize; 8]> =
        a_strides.iter().map(|&x| x as usize).collect();
    let _b_strides_usize: smallvec::SmallVec<[usize; 8]> =
        b_strides.iter().map(|&x| x as usize).collect();

    // Check if broadcasting is needed - only use parallel path when shapes are equal
    let needs_broadcast = a_shape != b_shape;
    let use_parallel = a.is_contiguous() && b.is_contiguous() && numel > 2048 && !needs_broadcast;

    if use_parallel {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            div_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            div_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        div_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        div_parallel_neon(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    div_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let result = _mm512_div_ps(a_vec, b_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_div_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let r0 = _mm256_div_ps(a0, b0);
                        let r1 = _mm256_div_ps(a1, b1);
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let result = _mm256_div_ps(a_vec, b_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));

                        let r0 = vdivq_f32(a0, b0);
                        let r1 = vdivq_f32(a1, b1);
                        let r2 = vdivq_f32(a2, b2);
                        let r3 = vdivq_f32(a3, b3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let result = vdivq_f32(a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) / *b_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) / *b_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        let out_usize = out_ptr as usize;
        iter.for_each_with_index(|idx, input_ptrs| unsafe {
            let a_val = *(input_ptrs[0].as_ptr() as *const f32);
            let b_val = *(input_ptrs[1].as_ptr() as *const f32);
            *((out_usize as *mut f32).add(idx)) = a_val / b_val;
        });
    }

    vec![output]
}

pub unsafe fn neg_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            neg_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            neg_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        neg_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    neg_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let result = _mm512_xor_ps(a_vec, _mm512_set1_ps(-0.0f32));
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let r0 = _mm256_xor_ps(a0, _mm256_set1_ps(-0.0f32));
                        let r1 = _mm256_xor_ps(a1, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), r0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), r1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let result = _mm256_xor_ps(a_vec, _mm256_set1_ps(-0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = -*a_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let r0 = vnegq_f32(a0);
                        let r1 = vnegq_f32(a1);
                        let r2 = vnegq_f32(a2);
                        let r3 = vnegq_f32(a3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let result = vnegq_f32(a_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = -*a_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = -*a_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = -val;
            }
        }
    }

    vec![output]
}

pub unsafe fn abs_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            abs_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            abs_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        abs_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        abs_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    abs_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));

                        let r0 = vabsq_f32(a0);
                        let r1 = vabsq_f32(a1);
                        let r2 = vabsq_f32(a2);
                        let r3 = vabsq_f32(a3);

                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);

                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let result = vabsq_f32(a_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = (*a_ptr.add(i)).abs();
                        i += 1;
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "aarch64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).abs();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.abs();
            }
        }
    }

    vec![output]
}

pub unsafe fn exp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            exp_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            exp_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).exp();
                            }
                        }
                    });
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                (0..num_chunks)
                    .into_par_iter()
                    .for_each(|chunk_idx| unsafe {
                        exp_parallel_neon(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).exp();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                exp_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                exp_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).exp();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.exp();
            }
        }
    }

    vec![output]
}

pub unsafe fn log_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            log_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            log_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).ln();
                            }
                        }
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).ln();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                log_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                log_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).ln();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.ln();
            }
        }
    }

    vec![output]
}

pub unsafe fn sqrt_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sqrt_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sqrt_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                *((out_usize + i * 4) as *mut f32) =
                                    (*((a_usize + i * 4) as *const f32)).sqrt();
                            }
                        }
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            *((out_usize + i * 4) as *mut f32) =
                                (*((a_usize + i * 4) as *const f32)).sqrt();
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                sqrt_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                sqrt_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).sqrt();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.sqrt();
            }
        }
    }

    vec![output]
}

pub unsafe fn relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let numel = a.numel() as usize;

    // Fast path: skip TensorIterator for contiguous tensors
    if a.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(a.shape_ref().to_vec(), a.dtype(), a.device());
        {
            let inner = Arc::make_mut(&mut output.inner);
            let storage = Arc::make_mut(&mut inner.storage);
            let Storage::Cpu(cpu_storage) = storage else {
                panic!()
            };
            let out_data = Arc::make_mut(&mut cpu_storage.data);
            let a_ptr = a.data_ptr() as *const f32;
            let out_ptr = out_data.as_mut_ptr() as *mut f32;

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let chunk_size = CHUNK_MEMBOUND;
                let num_chunks = numel.div_ceil(chunk_size);
                let a_usize = a_ptr as usize;
                let out_usize = out_ptr as usize;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                match SIMD_LEVEL.get_or_init(detect_simd_level) {
                    SimdLevel::Avx512 => {
                        (0..num_chunks)
                            .into_par_iter()
                            .for_each(|chunk_idx| unsafe {
                                relu_parallel_avx512(
                                    chunk_idx, chunk_size, numel, a_usize, out_usize,
                                );
                            });
                    }
                    SimdLevel::Avx2 => {
                        (0..num_chunks)
                            .into_par_iter()
                            .for_each(|chunk_idx| unsafe {
                                relu_parallel_avx2(
                                    chunk_idx, chunk_size, numel, a_usize, out_usize,
                                );
                            });
                    }
                    SimdLevel::Scalar => {
                        (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                            relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        relu_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                        relu_simd(a_slice, out_slice);
                    } else {
                        for idx in 0..numel {
                            unsafe {
                                *out_ptr.add(idx) = (*a_ptr.add(idx)).max(0.0);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = (*a_ptr.add(idx)).max(0.0);
                        }
                    }
                }
            }
        }
        return vec![output];
    }

    // Fallback: use TensorIterator for non-contiguous tensors
    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());
    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!()
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    for idx in 0..numel {
        unsafe {
            let val = *a_ptr.add(idx);
            *out_ptr.add(idx) = val.max(0.0);
        }
    }
    vec![output]
}

#[inline]
pub unsafe fn fused_add_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];

    let a_shape = a.shape_ref();
    let _b_shape = b.shape_ref();
    let output_shape = a_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && b.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_add_relu_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_add_relu_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        fused_add_relu_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    fused_add_relu_parallel_scalar(
                        chunk_idx, chunk_size, numel, a_usize, b_usize, out_usize,
                    );
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let zero = f32x4::ZERO;
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                let (a_chunks, a_rem) = a_slice.as_chunks::<4>();
                let (b_chunks, b_rem) = b_slice.as_chunks::<4>();
                let (out_chunks, out_rem) = out_slice.as_chunks_mut::<4>();

                for ((a_chunk, b_chunk), out_chunk) in a_chunks
                    .iter()
                    .zip(b_chunks.iter())
                    .zip(out_chunks.iter_mut())
                {
                    let a_vec = f32x4::from(*a_chunk);
                    let b_vec = f32x4::from(*b_chunk);
                    let sum = a_vec + b_vec;
                    let result = sum.max(zero);
                    *out_chunk = result.into();
                }
                for ((a_val, b_val), out_val) in
                    a_rem.iter().zip(b_rem.iter()).zip(out_rem.iter_mut())
                {
                    *out_val = (*a_val + *b_val).max(0.0);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let zero = _mm512_set1_ps(0.0f32);
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let sum = _mm512_add_ps(a_vec, b_vec);
                        let relu = _mm512_max_ps(sum, zero);
                        _mm512_storeu_ps(out_ptr.add(i), relu);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let sum = _mm256_add_ps(a_vec, b_vec);
                        let relu = _mm256_max_ps(sum, _mm256_set1_ps(0.0f32));
                        _mm256_storeu_ps(out_ptr.add(i), relu);
                        i += 8;
                    }
                    while i < numel {
                        let val = *a_ptr.add(i) + *b_ptr.add(i);
                        *out_ptr.add(i) = val.max(0.0);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let zero = _mm256_set1_ps(0.0f32);
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = _mm256_loadu_ps(a_ptr.add(i));
                        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
                        let b0 = _mm256_loadu_ps(b_ptr.add(i));
                        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
                        let sum0 = _mm256_add_ps(a0, b0);
                        let sum1 = _mm256_add_ps(a1, b1);
                        let relu0 = _mm256_max_ps(sum0, zero);
                        let relu1 = _mm256_max_ps(sum1, zero);
                        _mm256_storeu_ps(out_ptr.add(i), relu0);
                        _mm256_storeu_ps(out_ptr.add(i + 8), relu1);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let sum = _mm256_add_ps(a_vec, b_vec);
                        let relu = _mm256_max_ps(sum, zero);
                        _mm256_storeu_ps(out_ptr.add(i), relu);
                        i += 8;
                    }
                    while i < numel {
                        let val = *a_ptr.add(i) + *b_ptr.add(i);
                        *out_ptr.add(i) = val.max(0.0);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                            *out_ptr.add(idx) = val.max(0.0);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let b_slice = unsafe { std::slice::from_raw_parts(b_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };

                let (a_chunks, a_rem) = a_slice.as_chunks::<4>();
                let (b_chunks, b_rem) = b_slice.as_chunks::<4>();
                let (out_chunks, out_rem) = out_slice.as_chunks_mut::<4>();

                let zero = f32x4::ZERO;
                for ((a_chunk, b_chunk), out_chunk) in a_chunks
                    .iter()
                    .zip(b_chunks.iter())
                    .zip(out_chunks.iter_mut())
                {
                    let a_vec = f32x4::from(*a_chunk);
                    let b_vec = f32x4::from(*b_chunk);
                    let sum = a_vec + b_vec;
                    let result = sum.max(zero);
                    *out_chunk = result.into();
                }

                for ((a_val, b_val), out_val) in
                    a_rem.iter().zip(b_rem.iter()).zip(out_rem.iter_mut())
                {
                    *out_val = (*a_val + *b_val).max(0.0);
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                        *out_ptr.add(idx) = val.max(0.0);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx) + *b_ptr.add(idx);
                *out_ptr.add(idx) = val.max(0.0);
            }
        }
    }

    vec![output]
}

#[inline]
pub unsafe fn fused_mul_add_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let c = args[2];

    let a_shape = a.shape_ref();
    let output_shape = a_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;
    let b_ptr = b.data_ptr() as *const f32;
    let c_ptr = c.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && b.is_contiguous() && c.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let b_usize = b_ptr as usize;
            let c_usize = c_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_mul_add_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            fused_mul_add_parallel_avx2(
                                chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        fused_mul_add_parallel_scalar(
                            chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                        );
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    fused_mul_add_parallel_scalar(
                        chunk_idx, chunk_size, numel, a_usize, b_usize, c_usize, out_usize,
                    );
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a_vec = _mm512_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm512_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm512_loadu_ps(c_ptr.add(i));
                        let result = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm512_storeu_ps(out_ptr.add(i), result);
                        i += 16;
                    }
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Avx2 => unsafe {
                    let mut i = 0usize;
                    while i + 8 <= numel {
                        let a_vec = _mm256_loadu_ps(a_ptr.add(i));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(i));
                        let c_vec = _mm256_loadu_ps(c_ptr.add(i));
                        let result = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        _mm256_storeu_ps(out_ptr.add(i), result);
                        i += 8;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                },
                SimdLevel::Scalar => {
                    for idx in 0..numel {
                        unsafe {
                            *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx) + *c_ptr.add(idx);
                        }
                    }
                }
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                unsafe {
                    let mut i = 0usize;
                    while i + 16 <= numel {
                        let a0 = vld1q_f32(a_ptr.add(i));
                        let a1 = vld1q_f32(a_ptr.add(i + 4));
                        let a2 = vld1q_f32(a_ptr.add(i + 8));
                        let a3 = vld1q_f32(a_ptr.add(i + 12));
                        let b0 = vld1q_f32(b_ptr.add(i));
                        let b1 = vld1q_f32(b_ptr.add(i + 4));
                        let b2 = vld1q_f32(b_ptr.add(i + 8));
                        let b3 = vld1q_f32(b_ptr.add(i + 12));
                        let c0 = vld1q_f32(c_ptr.add(i));
                        let c1 = vld1q_f32(c_ptr.add(i + 4));
                        let c2 = vld1q_f32(c_ptr.add(i + 8));
                        let c3 = vld1q_f32(c_ptr.add(i + 12));
                        let r0 = vfmaq_f32(c0, a0, b0);
                        let r1 = vfmaq_f32(c1, a1, b1);
                        let r2 = vfmaq_f32(c2, a2, b2);
                        let r3 = vfmaq_f32(c3, a3, b3);
                        vst1q_f32(out_ptr.add(i), r0);
                        vst1q_f32(out_ptr.add(i + 4), r1);
                        vst1q_f32(out_ptr.add(i + 8), r2);
                        vst1q_f32(out_ptr.add(i + 12), r3);
                        i += 16;
                    }
                    while i + 4 <= numel {
                        let a_vec = vld1q_f32(a_ptr.add(i));
                        let b_vec = vld1q_f32(b_ptr.add(i));
                        let c_vec = vld1q_f32(c_ptr.add(i));
                        let result = vfmaq_f32(c_vec, a_vec, b_vec);
                        vst1q_f32(out_ptr.add(i), result);
                        i += 4;
                    }
                    while i < numel {
                        *out_ptr.add(i) = *a_ptr.add(i) * *b_ptr.add(i) + *c_ptr.add(i);
                        i += 1;
                    }
                }
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = *a_ptr.add(idx) * *b_ptr.add(idx) + *c_ptr.add(idx);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let a_val = *a_ptr.add(idx);
                let b_val = *b_ptr.add(idx);
                let c_val = *c_ptr.add(idx);
                *out_ptr.add(idx) = a_val * b_val + c_val;
            }
        }
    }

    vec![output]
}

pub unsafe fn gelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            gelu_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            gelu_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        let start = chunk_idx * chunk_size;
                        let end = std::cmp::min(start + chunk_size, numel);

                        for i in start..end {
                            unsafe {
                                let x = *((a_usize + i * 4) as *const f32);
                                let x3 = x * x * x;
                                let t = (GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh();
                                *((out_usize + i * 4) as *mut f32) = 0.5_f32 * x * (1.0_f32 + t);
                            }
                        }
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, numel);

                    for i in start..end {
                        unsafe {
                            let x = *((a_usize + i * 4) as *const f32);
                            let x3 = x * x * x;
                            let t = (GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh();
                            *((out_usize + i * 4) as *mut f32) = 0.5 * x * (1.0 + t);
                        }
                    }
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                gelu_simd(a_slice, out_slice);
            }
            #[cfg(all(feature = "simd", target_arch = "aarch64"))]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                gelu_simd(a_slice, out_slice);
            }
            #[cfg(not(any(
                all(feature = "simd", target_arch = "x86_64"),
                all(feature = "simd", target_arch = "aarch64")
            )))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        let x3 = x * x * x;
                        let t = (GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh();
                        *out_ptr.add(idx) = 0.5 * x * (1.0 + t);
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                let x3 = x * x * x;
                let t = (GELU_SQRT_2_OVER_PI * (x + GELU_COEFF * x3)).tanh();
                *out_ptr.add(idx) = 0.5_f32 * x * (1.0_f32 + t);
            }
        }
    }

    vec![output]
}

pub unsafe fn sigmoid_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // Use runtime feature detection for x86_64
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sigmoid_parallel_avx512(
                                chunk_idx, chunk_size, numel, a_usize, out_usize,
                            );
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            sigmoid_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        sigmoid_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    sigmoid_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = 1.0 / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

pub unsafe fn tanh_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // Use runtime feature detection for x86_64
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            match SIMD_LEVEL.get_or_init(detect_simd_level) {
                SimdLevel::Avx512 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            tanh_parallel_avx512(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Avx2 => {
                    (0..num_chunks)
                        .into_par_iter()
                        .for_each(|chunk_idx| unsafe {
                            tanh_parallel_avx2(chunk_idx, chunk_size, numel, a_usize, out_usize);
                        });
                }
                SimdLevel::Scalar => {
                    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                        tanh_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                    });
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                    tanh_parallel_scalar(chunk_idx, chunk_size, numel, a_usize, out_usize);
                });
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).tanh();
                    }
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                for idx in 0..numel {
                    unsafe {
                        *out_ptr.add(idx) = (*a_ptr.add(idx)).tanh();
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x.tanh();
            }
        }
    }

    vec![output]
}

pub unsafe fn silu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);

            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;

            // SiLU is transcendental, use scalar for parallel path
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = std::cmp::min(start + chunk_size, numel);

                for i in start..end {
                    unsafe {
                        let x = *((a_usize + i * 4) as *const f32);
                        *((out_usize + i * 4) as *mut f32) = x / (1.0 + (-x).exp());
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            #[cfg(feature = "simd")]
            {
                let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
                silu_simd(a_slice, out_slice);
            }
            #[cfg(not(feature = "simd"))]
            {
                for idx in 0..numel {
                    unsafe {
                        let x = *a_ptr.add(idx);
                        *out_ptr.add(idx) = x / (1.0 + (-x).exp());
                    }
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let x = *a_ptr.add(idx);
                *out_ptr.add(idx) = x / (1.0 + (-x).exp());
            }
        }
    }

    vec![output]
}

pub unsafe fn embedding_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let weight = args[0];
    let indices = args[1];

    let weight_shape = weight.shape_ref();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    let indices_shape = indices.shape_ref();
    let batch_size: i64 = indices_shape.iter().product();

    let output_shape: Vec<i64> = indices_shape
        .iter()
        .chain(std::iter::once(&embedding_dim))
        .copied()
        .collect();
    let mut output = Tensor::empty(output_shape.clone(), weight.dtype(), weight.device());

    let indices_ptr = indices.data_ptr() as *const f32;
    let weight_ptr = weight.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    for i in 0..batch_size as usize {
        let idx = unsafe { *indices_ptr.add(i) } as usize;
        if idx < num_embeddings as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    weight_ptr.add(idx * embedding_dim as usize),
                    out_ptr.add(i * embedding_dim as usize),
                    embedding_dim as usize,
                );
            }
        }
    }

    // Set up gradient tracking for embedding
    if weight.requires_grad() {
        let backward = EmbeddingBackward::new(weight.clone(), indices.clone());
        let mut meta = AutogradMeta::new_non_leaf(true);
        meta.grad_fn = Some(Arc::new(backward));
        Arc::make_mut(&mut output.inner).autograd_meta =
            Some(Arc::new(std::sync::Mutex::new(meta)));
    }

    vec![output]
}

pub unsafe fn clamp_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let min_val = args[1].item();
    let max_val = args[2].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice
                .par_iter_mut()
                .zip(a_slice.par_iter())
                .for_each(|(out, &val)| {
                    *out = val.clamp(min_val, max_val);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.clamp(min_val, max_val);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.clamp(min_val, max_val);
            }
        }
    }

    vec![output]
}

pub unsafe fn pow_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let exponent = args[1].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, numel) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, numel) };
            out_slice
                .par_iter_mut()
                .zip(a_slice.par_iter())
                .for_each(|(out, &val)| {
                    *out = val.powf(exponent);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for idx in 0..numel {
                unsafe {
                    let x = *a_ptr.add(idx);
                    *out_ptr.add(idx) = x.powf(exponent);
                }
            }
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = val.powf(exponent);
            }
        }
    }

    vec![output]
}

pub unsafe fn gt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let threshold = args[1].item();

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(numel);
                let a_p = a_usize as *const f32;
                let o_p = out_usize as *mut f32;
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        unsafe {
                            let thresh = _mm256_set1_ps(threshold);
                            let one = _mm256_set1_ps(1.0f32);
                            let zero = _mm256_set1_ps(0.0f32);
                            let mut i = start;
                            while i + 8 <= end {
                                let v = _mm256_loadu_ps(a_p.add(i));
                                let mask = _mm256_cmp_ps(v, thresh, _CMP_GT_OQ);
                                let r = _mm256_blendv_ps(zero, one, mask);
                                _mm256_storeu_ps(o_p.add(i), r);
                                i += 8;
                            }
                            for j in i..end {
                                *o_p.add(j) = if *a_p.add(j) > threshold { 1.0 } else { 0.0 };
                            }
                            return;
                        }
                    }
                }
                for j in start..end {
                    unsafe {
                        *o_p.add(j) = if *a_p.add(j) > threshold { 1.0 } else { 0.0 };
                    }
                }
            });
        }
    } else {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && numel >= 8 {
                unsafe {
                    let thresh = _mm256_set1_ps(threshold);
                    let one = _mm256_set1_ps(1.0f32);
                    let zero = _mm256_set1_ps(0.0f32);
                    let mut i = 0;
                    while i + 8 <= numel {
                        let v = _mm256_loadu_ps(a_ptr.add(i));
                        let mask = _mm256_cmp_ps(v, thresh, _CMP_GT_OQ);
                        let r = _mm256_blendv_ps(zero, one, mask);
                        _mm256_storeu_ps(out_ptr.add(i), r);
                        i += 8;
                    }
                    for j in i..numel {
                        *out_ptr.add(j) = if *a_ptr.add(j) > threshold { 1.0 } else { 0.0 };
                    }
                    return vec![output];
                }
            }
        }
        for idx in 0..numel {
            unsafe {
                *out_ptr.add(idx) = if *a_ptr.add(idx) > threshold {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }

    vec![output]
}

pub unsafe fn sign_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];

    let iter = TensorIterator::build_for_unary(a);
    let output_shape = iter.output_shape.to_vec();

    let mut output = Tensor::empty(output_shape.clone(), a.dtype(), a.device());

    let numel = output_shape.iter().product::<i64>() as usize;
    let a_ptr = a.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    if a.is_contiguous() && numel > 2048 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(numel);
                for j in start..end {
                    unsafe {
                        let val = *(a_usize as *const f32).add(j);
                        *(out_usize as *mut f32).add(j) = if val > 0.0 {
                            1.0
                        } else if val < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                }
            });
        }
    } else {
        for idx in 0..numel {
            unsafe {
                let val = *a_ptr.add(idx);
                *out_ptr.add(idx) = if val > 0.0 {
                    1.0
                } else if val < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }

    vec![output]
}

pub unsafe fn maximum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let out_shape = broadcast_shapes_simple(a.shape_ref(), b.shape_ref());
    let numel = out_shape.iter().product::<i64>() as usize;
    let mut output = Tensor::empty(out_shape.clone(), a.dtype(), a.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let a_strides = &a.inner.strides;
    let b_strides = &b.inner.strides;
    let out_strides = &output.inner.strides;
    let ndim = out_shape.len();
    let a_offset = a.inner.storage_offset as usize;
    let b_offset = b.inner.storage_offset as usize;
    let mut indices = vec![0i64; ndim];
    for _out_idx in 0..numel {
        let mut a_idx: usize = a_offset;
        let mut b_idx: usize = b_offset;
        for d in 0..ndim {
            let a_dim_idx = if d >= ndim - a.ndim() {
                d - (ndim - a.ndim())
            } else {
                usize::MAX
            };
            let b_dim_idx = if d >= ndim - b.ndim() {
                d - (ndim - b.ndim())
            } else {
                usize::MAX
            };
            if a_dim_idx != usize::MAX && a.shape_ref()[a_dim_idx] != 1 {
                a_idx += (indices[d] % a.shape_ref()[a_dim_idx]) as usize
                    * a_strides[a_dim_idx] as usize;
            }
            if b_dim_idx != usize::MAX && b.shape_ref()[b_dim_idx] != 1 {
                b_idx += (indices[d] % b.shape_ref()[b_dim_idx]) as usize
                    * b_strides[b_dim_idx] as usize;
            }
        }
        let av = unsafe { *a_data.as_ptr().add(a_idx) };
        let bv = unsafe { *b_data.as_ptr().add(b_idx) };
        let mut out_linear: usize = 0;
        for d in 0..ndim {
            out_linear += indices[d] as usize * out_strides[d] as usize;
        }
        unsafe {
            *out_ptr.add(out_linear) = if av > bv { av } else { bv };
        }
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < out_shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
    vec![output]
}

pub unsafe fn minimum_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let a = args[0];
    let b = args[1];
    let out_shape = broadcast_shapes_simple(a.shape_ref(), b.shape_ref());
    let numel = out_shape.iter().product::<i64>() as usize;
    let mut output = Tensor::empty(out_shape.clone(), a.dtype(), a.device());

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;
    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let a_strides = &a.inner.strides;
    let b_strides = &b.inner.strides;
    let out_strides = &output.inner.strides;
    let ndim = out_shape.len();
    let a_offset = a.inner.storage_offset as usize;
    let b_offset = b.inner.storage_offset as usize;
    let mut indices = vec![0i64; ndim];
    for _out_idx in 0..numel {
        let mut a_idx: usize = a_offset;
        let mut b_idx: usize = b_offset;
        for d in 0..ndim {
            let a_dim_idx = if d >= ndim - a.ndim() {
                d - (ndim - a.ndim())
            } else {
                usize::MAX
            };
            let b_dim_idx = if d >= ndim - b.ndim() {
                d - (ndim - b.ndim())
            } else {
                usize::MAX
            };
            if a_dim_idx != usize::MAX && a.shape_ref()[a_dim_idx] != 1 {
                a_idx += (indices[d] % a.shape_ref()[a_dim_idx]) as usize
                    * a_strides[a_dim_idx] as usize;
            }
            if b_dim_idx != usize::MAX && b.shape_ref()[b_dim_idx] != 1 {
                b_idx += (indices[d] % b.shape_ref()[b_dim_idx]) as usize
                    * b_strides[b_dim_idx] as usize;
            }
        }
        let av = unsafe { *a_data.as_ptr().add(a_idx) };
        let bv = unsafe { *b_data.as_ptr().add(b_idx) };
        let mut out_linear: usize = 0;
        for d in 0..ndim {
            out_linear += indices[d] as usize * out_strides[d] as usize;
        }
        unsafe {
            *out_ptr.add(out_linear) = if av < bv { av } else { bv };
        }
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < out_shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
    vec![output]
}

pub unsafe fn leaky_relu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let slope = args[1].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = if v > 0.0 { v } else { v * slope };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = if v > 0.0 { v } else { v * slope };
                }
            }
        }
        return vec![output];
    }

    // Fallback: TensorIterator for non-contiguous / small tensors
    let iter = TensorIterator::build_for_unary(x);
    let output_shape = iter.output_shape.to_vec();
    let mut output = Tensor::empty(output_shape, x.dtype(), x.device());
    let numel = output.numel() as usize;
    let a_ptr = x.data_ptr() as *const f32;
    let out_ptr = output.data_ptr_f32_mut();

    for i in 0..numel {
        unsafe {
            let v = *a_ptr.add(i);
            *out_ptr.add(i) = if v > 0.0 { v } else { v * slope };
        }
    }
    vec![output]
}

pub unsafe fn prelu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let weight = args[1];
    let w_data = weight.as_f32_slice();
    let x_shape = x.shape_ref();
    let numel = x.numel() as usize;
    let ndim = x.ndim();
    let w_numel = w_data.len();

    // Fast path: contiguous + large tensor + single weight
    if x.is_contiguous() && numel > 2048 && w_numel == 1 {
        let w = w_data[0];
        let mut output = Tensor::empty(x_shape.to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = if v > 0.0 { v } else { v * w };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = if v > 0.0 { v } else { v * w };
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let mut output = x.clone();
    let ptr = output.data_ptr_f32_mut();
    let mut indices = vec![0i64; ndim];
    for i in 0..numel {
        unsafe {
            let v = *ptr.add(i);
            let w_idx = if w_numel == 1 {
                0
            } else {
                indices[1] as usize % w_numel
            };
            // SAFETY: w_idx is computed as `indices[1] as usize % w_numel` (or 0 when w_numel==1),
            // which is always < w_data.len(). The debug_assert confirms this at runtime.
            debug_assert!(w_idx < w_data.len(), "leaky_relu weight index out of bounds");
            let w = *w_data.get_unchecked(w_idx);
            *ptr.add(i) = if v > 0.0 { v } else { v * w };
        }
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < x_shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }
    vec![output]
}

pub unsafe fn softplus_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let beta = args[1].item();
    let threshold = args[2].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        let bx = beta * v;
                        *(out_usize as *mut f32).add(i) = if bx > threshold {
                            v
                        } else {
                            (1.0_f32 + (bx).exp()).ln() / beta
                        };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    let bx = beta * v;
                    *out_ptr.add(i) = if bx > threshold {
                        v
                    } else {
                        (1.0_f32 + (bx).exp()).ln() / beta
                    };
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let mut output_data = vec![0.0f32; numel];
    let x_data = x.as_f32_slice();
    for i in 0..numel {
        let bx = beta * x_data[i];
        output_data[i] = if bx > threshold {
            x_data[i]
        } else {
            (1.0_f32 + (bx).exp()).ln() / beta
        };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn hardswish_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        let relu6 = v.clamp(0.0, 6.0);
                        *(out_usize as *mut f32).add(i) = v * relu6 / 6.0;
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    let relu6 = v.clamp(0.0, 6.0);
                    *out_ptr.add(i) = v * relu6 / 6.0;
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let mut output_data = vec![0.0f32; numel];
    let x_data = x.as_f32_slice();
    for i in 0..numel {
        let v = x_data[i];
        let relu6 = v.clamp(0.0, 6.0);
        output_data[i] = v * relu6 / 6.0;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn lt_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let threshold = args[1].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = if v < threshold { 1.0 } else { 0.0 };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = if v < threshold { 1.0 } else { 0.0 };
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = if x_data[i] < threshold { 1.0 } else { 0.0 };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn add_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let scalar = args[1].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = v + scalar;
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = v + scalar;
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = x_data[i] + scalar;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn div_scalar_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let scalar = args[1].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = v / scalar;
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = v / scalar;
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = x_data[i] / scalar;
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn logical_not_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_MEMBOUND;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) = if v == 0.0 { 1.0 } else { 0.0 };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = if v == 0.0 { 1.0 } else { 0.0 };
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = if x_data[i] == 0.0 { 1.0 } else { 0.0 };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn elu_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0];
    let alpha = args[1].item();
    let numel = x.numel() as usize;

    // Fast path: contiguous + large tensor
    if x.is_contiguous() && numel > 2048 {
        let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());
        let a_ptr = x.data_ptr_f32();
        let out_ptr = output.data_ptr_f32_mut();

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);
            let a_usize = a_ptr as usize;
            let out_usize = out_ptr as usize;
            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = ((chunk_idx + 1) * chunk_size).min(numel);
                for i in start..end {
                    unsafe {
                        let v = *(a_usize as *const f32).add(i);
                        *(out_usize as *mut f32).add(i) =
                            if v > 0.0 { v } else { alpha * (v.exp() - 1.0) };
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let v = *a_ptr.add(i);
                    *out_ptr.add(i) = if v > 0.0 { v } else { alpha * (v.exp() - 1.0) };
                }
            }
        }
        return vec![output];
    }

    // Fallback
    let numel = x.inner.numel() as usize;
    let x_data = x.as_f32_slice();
    let mut output_data = vec![0.0f32; numel];
    for i in 0..numel {
        output_data[i] = if x_data[i] > 0.0 {
            x_data[i]
        } else {
            alpha * (x_data[i].exp() - 1.0)
        };
    }
    vec![Tensor::from_vec(output_data, x.shape())]
}

pub unsafe fn gelu_backward_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let grad = args[0];
    let x = args[1];

    let output_shape = grad.shape_ref().to_vec();
    let mut output = Tensor::empty(output_shape.clone(), grad.dtype(), grad.device());

    let numel = grad.numel() as usize;

    let grad_ptr = grad.data_ptr() as *const f32;
    let x_ptr = x.data_ptr() as *const f32;

    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(cpu_storage) = output_storage else {
        panic!("Expected CPU storage");
    };
    let out_data = Arc::make_mut(&mut cpu_storage.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    const SQRT_2_OVER_PI: f32 = 0.7978846;
    const COEFF: f32 = 0.044715;

    if grad.is_contiguous() && x.is_contiguous() && numel > 256 {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            let chunk_size = CHUNK_TRANSCENDENTAL;
            let num_chunks = numel.div_ceil(chunk_size);

            let grad_usize = grad_ptr as usize;
            let x_usize = x_ptr as usize;
            let out_usize = out_ptr as usize;

            (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = std::cmp::min(start + chunk_size, numel);

                for i in start..end {
                    unsafe {
                        let x_val = *((x_usize + i * 4) as *const f32);
                        let grad_val = *((grad_usize + i * 4) as *const f32);

                        // Compute gelu'(x)
                        let x2 = x_val * x_val;
                        let x3 = x2 * x_val;
                        let inner = SQRT_2_OVER_PI * (x_val + COEFF * x3);
                        let t = inner.tanh();
                        let t2 = t * t;
                        let sech2 = 1.0 - t2;
                        let d_inner_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
                        let derivative = 0.5 * (1.0 + t) + 0.5 * x_val * sech2 * d_inner_dx;

                        *((out_usize + i * 4) as *mut f32) = grad_val * derivative;
                    }
                }
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..numel {
                unsafe {
                    let x_val = *x_ptr.add(i);
                    let grad_val = *grad_ptr.add(i);

                    // Compute gelu'(x)
                    let x2 = x_val * x_val;
                    let x3 = x2 * x_val;
                    let inner = SQRT_2_OVER_PI * (x_val + COEFF * x3);
                    let t = inner.tanh();
                    let t2 = t * t;
                    let sech2 = 1.0 - t2;
                    let d_inner_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
                    let derivative = 0.5 * (1.0 + t) + 0.5 * x_val * sech2 * d_inner_dx;

                    *out_ptr.add(i) = grad_val * derivative;
                }
            }
        }
    } else {
        // Fallback to TensorIterator for broadcasting
        let iter = TensorIterator::build_for_binary(grad, x);
        let out_usize = out_ptr as usize;

        iter.for_each_with_index(|idx, input_ptrs| {
            unsafe {
                let grad_val = *(input_ptrs[0].as_ptr() as *const f32);
                let x_val = *(input_ptrs[1].as_ptr() as *const f32);

                // Compute gelu'(x)
                let x2 = x_val * x_val;
                let x3 = x2 * x_val;
                let inner = SQRT_2_OVER_PI * (x_val + COEFF * x3);
                let t = inner.tanh();
                let t2 = t * t;
                let sech2 = 1.0 - t2;
                let d_inner_dx = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * x2);
                let derivative = 0.5 * (1.0 + t) + 0.5 * x_val * sech2 * d_inner_dx;

                *((out_usize as *mut f32).add(idx)) = grad_val * derivative;
            }
        });
    }

    vec![output]
}

/// Fused SiLUBackward kernel: computes grad_input = grad * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
/// This eliminates ~4 intermediate tensor allocations.
pub unsafe fn silu_backward_kernel(args: &[&Tensor]) -> Vec<Tensor> {
    let x = args[0]; // input tensor
    let s = args[1]; // sigmoid(x) (output of forward pass)
    let grad = args[2]; // gradient from next layer

    let numel = x.numel() as usize;
    let mut output = Tensor::empty(x.shape_ref().to_vec(), x.dtype(), x.device());

    // Input tensors are read-only, just get data pointers
    let x_ptr = x.data_ptr() as *const f32;
    let s_ptr = s.data_ptr() as *const f32;
    let grad_ptr = grad.data_ptr() as *const f32;

    // Output needs mutable access
    let output_inner = Arc::make_mut(&mut output.inner);
    let output_storage = Arc::make_mut(&mut output_inner.storage);
    let Storage::Cpu(out_cpu) = output_storage else {
        panic!("Expected CPU storage for output");
    };
    let out_data = Arc::make_mut(&mut out_cpu.data);
    let out_ptr = out_data.as_mut_ptr() as *mut f32;

    for i in 0..numel {
        unsafe {
            let x_val = *x_ptr.add(i);
            let s_val = *s_ptr.add(i);
            let g_val = *grad_ptr.add(i);
            // derivative = s * (1 + x * (1 - s))
            let derivative = s_val * (1.0 + x_val * (1.0 - s_val));
            *out_ptr.add(i) = g_val * derivative;
        }
    }

    vec![output]
}
