//! Macros for reducing boilerplate in tensor scalar operations.
//! These will be used by `src/tensor/indexing.rs` to eliminate
//! the 5x identical CPU fast-path patterns for gt_scalar,
//! lt_scalar, eq_scalar, add_scalar, div_scalar.

/// Implement a scalar comparison operation on CPU with
/// AVX2 SIMD fast path and scalar fallback.
///
/// Generates a function named `$fn_name` that compares each element
/// against a scalar using `$rust_op` and `$simd_cmp`,
/// producing 0.0 or 1.0.
///
/// Usage:
/// ```ignore
/// impl_scalar_op!(gt_scalar, _CMP_GT_OQ, >);
/// impl_scalar_op!(lt_scalar, _CMP_LT_OQ, <);
/// impl_scalar_op!(eq_scalar, _CMP_EQ_OQ, ==);
/// ```
#[macro_export]
macro_rules! impl_scalar_op {
    ($fn_name:ident, $simd_cmp:ident, $rust_op:tt) => {
        pub fn $fn_name(&self, threshold: f32) -> $crate::tensor::Tensor {
            if let Some((numel, a_ptr, out_ptr, output)) = cpu_f32_fast_path_setup(self) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let threshold_v = _mm256_set1_ps(threshold);
                            let one_v = _mm256_set1_ps(1.0);
                            let mut i = 0;
                            while i + 8 <= numel {
                                let av = _mm256_loadu_ps(a_ptr.add(i));
                                let cmp = _mm256_cmp_ps(av, threshold_v, $simd_cmp);
                                _mm256_storeu_ps(out_ptr.add(i), _mm256_and_ps(cmp, one_v));
                                i += 8;
                            }
                            for j in i..numel {
                                *out_ptr.add(j) = if *a_ptr.add(j) $rust_op threshold { 1.0 } else { 0.0 };
                            }
                        }
                    } else {
                        for i in 0..numel {
                            unsafe {
                                *out_ptr.add(i) = if *a_ptr.add(i) $rust_op threshold { 1.0 } else { 0.0 };
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for i in 0..numel {
                        unsafe { *out_ptr.add(i) = if *a_ptr.add(i) $rust_op threshold { 1.0 } else { 0.0 }; }
                    }
                }
                return output;
            }

            let scalar = Tensor::from_scalar(threshold);
            Tensor::exec_aot(&[self, &scalar], |g, ins| {
                vec![g.$fn_name(&ins[0], &ins[1])]
            })
            .expect(concat!("Tensor::", stringify!($fn_name), ": AOT execution failed"))
            .into_iter()
            .next()
            .unwrap()
        }
    };
}

/// Implement a scalar arithmetic operation on CPU with
/// AVX2 SIMD fast path, scalar fallback, and autograd support.
///
/// Generates a function named `$fn_name` that applies `$rust_op`
/// with `$simd_op` against a scalar value, supporting gradient tracking.
///
/// Usage:
/// ```ignore
/// impl_cpu_fast_path!(add_scalar, _mm256_add_ps, +, AddScalarBackward);
/// impl_cpu_fast_path!(div_scalar, _mm256_div_ps, /, DivScalarBackward);
/// ```
#[macro_export]
macro_rules! impl_cpu_fast_path {
    ($fn_name:ident, $simd_op:ident, $rust_op:tt, $backward_name:ident) => {
        pub fn $fn_name(&self, scalar: f32) -> $crate::tensor::Tensor {
            if let Some((numel, a_ptr, out_ptr, output)) = cpu_f32_fast_path_setup(self) {
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        unsafe {
                            let scalar_v = _mm256_set1_ps(scalar);
                            let mut i = 0;
                            while i + 8 <= numel {
                                let av = _mm256_loadu_ps(a_ptr.add(i));
                                _mm256_storeu_ps(out_ptr.add(i), $simd_op(av, scalar_v));
                                i += 8;
                            }
                            for j in i..numel {
                                *out_ptr.add(j) = *a_ptr.add(j) $rust_op scalar;
                            }
                        }
                    } else {
                        for i in 0..numel {
                            unsafe {
                                *out_ptr.add(i) = *a_ptr.add(i) $rust_op scalar;
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for i in 0..numel {
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) $rust_op scalar;
                        }
                    }
                }
                if autograd::is_grad_enabled() && self.requires_grad() {
                    let s = Tensor::from_scalar(scalar);
                    let inputs = vec![self.clone(), s.clone()];
                    return Self::attach_grad_fn(output, autograd::make_node_info(stringify!($backward_name), inputs));
                } else {
                    return output;
                }
            }

            let s = Tensor::from_scalar(scalar);
            let output = Tensor::exec_aot(&[self, &s], |g, ins| vec![g.$fn_name(&ins[0], &ins[1])])
                .expect(concat!("Tensor::", stringify!($fn_name), ": AOT execution failed"))
                .into_iter()
                .next()
                .unwrap();
            if autograd::is_grad_enabled() && self.requires_grad() {
                let _edges = autograd::make_edge(self);
                let inputs = vec![self.clone(), s.clone()];
                Self::attach_grad_fn(output, autograd::make_node_info(stringify!($backward_name), inputs))
            } else {
                output
            }
        }
    };
}
