use crate::backend::BufferSlice;

use super::{add_f32, arena, div_f32, microkernels, mul_f32, sub_f32, CpuBuffer};

/// Compute a fused binary op + activation over already-extracted f32 slices.
///
/// Uses SIMD microkernels for fused binary+activation when available (identified
/// by `kernel_name`), falling back to the generic closure loop. The scalar/Rayon
/// fallback retains modulo indexing so broadcast callers preserve their existing
/// semantics.
#[inline]
fn fused_binary_activation_dispatch_slices(
    kernel_name: &str,
    a: &[f32],
    b: &[f32],
    out_f32: &mut [f32],
    op: &(impl Fn(f32, f32) -> f32 + Sync),
    act: &(impl Fn(f32) -> f32 + Sync),
) {
    // SIMD fast path: chain elementwise op + activation using existing
    // SIMD microkernels. The `kernel_name` tells us which combination
    // to dispatch. Two SIMD passes is still 4-8x faster than scalar.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if microkernels::simd_avx2_available() && out_f32.len() >= 8 {
        // Binary op dispatch
        let binary_dispatched = if kernel_name.starts_with("add_") {
            add_f32(a, b, out_f32);
            true
        } else if kernel_name.starts_with("sub_") {
            sub_f32(a, b, out_f32);
            true
        } else if kernel_name.starts_with("mul_") {
            mul_f32(a, b, out_f32);
            true
        } else if kernel_name.starts_with("div_") {
            div_f32(a, b, out_f32);
            true
        } else {
            false
        };

        if binary_dispatched {
            // Activation dispatch — chain in-place on the output buffer.
            // We use raw pointer to avoid &[f32] + &mut [f32] aliasing.
            let len = out_f32.len();
            let ptr = out_f32.as_mut_ptr();
            if kernel_name.ends_with("relu_f32") {
                unsafe {
                    let v = std::slice::from_raw_parts(ptr, len);
                    microkernels::relu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                }
            } else if kernel_name.ends_with("gelu_f32") {
                unsafe {
                    let v = std::slice::from_raw_parts(ptr, len);
                    microkernels::gelu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                }
            } else if kernel_name.ends_with("silu_f32") {
                unsafe {
                    let v = std::slice::from_raw_parts(ptr, len);
                    microkernels::silu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                }
            }
            return;
        }
    }

    let out_len = out_f32.len();
    let a_len = a.len();
    let b_len = b.len();
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..out_len {
            let x = op(a[i % a_len], b[i % b_len]);
            out_f32[i] = act(x);
        }
    }
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out_f32[..out_len]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| {
                let x = op(a[i % a_len], b[i % b_len]);
                *o = act(x);
            });
    }
}

/// Helper: extract two f32 slices from the arena, broadcast-loop with a binary op
/// and activation function, and write the result to the output slice.
///
/// Disjoint input/output ranges are borrowed directly for both same-shape and
/// modulo-broadcast fallback semantics. If the output overlaps either input,
/// fall back to copied inputs so in-place/overlapping execution remains safe.
#[inline]
pub(super) fn fused_binary_activation_dispatch(
    kernel_name: &str,
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    out_start: usize,
    out_end: usize,
    op: impl Fn(f32, f32) -> f32 + Sync,
    act: impl Fn(f32) -> f32 + Sync,
) {
    if let [a_slice, b_slice] = input_slices {
        let output_slice = BufferSlice::new(out_start, out_end - out_start);
        arena::with_binary_f32_slices(arena, *a_slice, *b_slice, output_slice, |a, b, out_f32| {
            fused_binary_activation_dispatch_slices(kernel_name, a, b, out_f32, &op, &act);
        });
    }
}

#[cfg(test)]
mod elementwise_copy_reduction_tests {
    use super::*;

    fn arena_from_f32(values: &[f32]) -> CpuBuffer {
        CpuBuffer::new(bytemuck::cast_slice(values).to_vec())
    }

    fn read_f32s(arena: &CpuBuffer, start_elem: usize, len: usize) -> Vec<f32> {
        let start = start_elem * std::mem::size_of::<f32>();
        let end = start + len * std::mem::size_of::<f32>();
        let data = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&data[start..end]).to_vec()
    }

    fn f32_slice(start_elem: usize, len: usize) -> BufferSlice {
        BufferSlice::new(
            start_elem * std::mem::size_of::<f32>(),
            len * std::mem::size_of::<f32>(),
        )
    }

    #[test]
    fn elementwise_same_shape_fused_add_relu_correct() {
        let arena = arena_from_f32(&[
            -3.0, 1.0, 4.0, 6.0, // a
            1.0, -5.0, 2.0, -10.0, // b
            0.0, 0.0, 0.0, 0.0, // output
        ]);
        let inputs = vec![f32_slice(0, 4), f32_slice(4, 4)];
        let output = f32_slice(8, 4);

        fused_binary_activation_dispatch(
            "add_relu_f32",
            &inputs,
            &arena,
            output.offset,
            output.offset + output.size,
            |a, b| a + b,
            |x| x.max(0.0),
        );

        assert_eq!(read_f32s(&arena, 8, 4), vec![0.0, 0.0, 6.0, 0.0]);
    }

    #[test]
    fn elementwise_broadcast_fused_mul_silu_correct() {
        let arena = arena_from_f32(&[
            1.0, -2.0, 3.0, -4.0, // a
            0.5,  // b broadcasts by modulo
            0.0, 0.0, 0.0, 0.0, // output
        ]);
        let inputs = vec![f32_slice(0, 4), f32_slice(4, 1)];
        let output = f32_slice(5, 4);

        fused_binary_activation_dispatch(
            "mul_silu_f32",
            &inputs,
            &arena,
            output.offset,
            output.offset + output.size,
            |a, b| a * b,
            |x| x / (1.0 + (-x).exp()),
        );

        let got = read_f32s(&arena, 5, 4);
        let expected: Vec<f32> = [0.5_f32, -1.0, 1.5, -2.0]
            .iter()
            .map(|&x| x / (1.0 + (-x).exp()))
            .collect();
        for (actual, expected) in got.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn elementwise_same_shape_overlapping_output_uses_safe_result() {
        let arena = arena_from_f32(&[
            1.0, 2.0, 3.0, 4.0, // a also overlaps output below
            10.0, 20.0, 30.0, 40.0, // b
        ]);
        let inputs = vec![f32_slice(0, 4), f32_slice(4, 4)];
        let output = f32_slice(2, 4);

        fused_binary_activation_dispatch(
            "div_relu_f32",
            &inputs,
            &arena,
            output.offset,
            output.offset + output.size,
            |a, b| a / b,
            |x| x.max(0.0),
        );

        assert_eq!(read_f32s(&arena, 2, 4), vec![0.1, 0.1, 0.1, 0.1]);
    }
}
