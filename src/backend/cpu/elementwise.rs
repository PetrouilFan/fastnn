use crate::backend::BufferSlice;

use super::{add_f32, arena, div_f32, microkernels, mul_f32, sub_f32, telemetry, CpuBuffer};

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
/// Same-shape binary elementwise dispatch uses the arena helper so disjoint
/// input/output ranges can be borrowed directly without temporary copies. For
/// broadcast or mismatched-length inputs, fall back to the prior copy path to
/// preserve modulo/broadcast semantics exactly.
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
    if let [a_slice, b_slice] = &input_slices[..] {
        let output_slice = BufferSlice::new(out_start, out_end - out_start);

        if a_slice.size == output_slice.size && b_slice.size == output_slice.size {
            arena::with_binary_f32_slices(
                arena,
                *a_slice,
                *b_slice,
                output_slice,
                |a, b, out_f32| {
                    fused_binary_activation_dispatch_slices(kernel_name, a, b, out_f32, &op, &act);
                },
            );
            return;
        }

        let (a, b) = {
            let d = arena.data_mut();
            telemetry::record_arena_temp_copy(a_slice.size);
            telemetry::record_arena_temp_copy(b_slice.size);
            (
                bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size])
                    .to_vec(),
                bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size])
                    .to_vec(),
            )
        };
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };

        fused_binary_activation_dispatch_slices(kernel_name, &a, &b, out_f32, &op, &act);
    }
}
