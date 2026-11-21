use crate::backend::BufferSlice;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use super::{add_f32, div_f32, microkernels, mul_f32, sub_f32};
use super::{arena, CpuBuffer};

/// Compute a fused binary op + activation over already-extracted f32 slices.
///
/// Uses SIMD microkernels for fused binary+activation when available (identified
/// by `kernel_name`), falling back to the generic closure loop. The scalar/Rayon
/// fallback retains modulo indexing so broadcast callers preserve their existing
/// semantics.
#[inline]
fn fused_binary_activation_dispatch_slices(
    #[cfg_attr(
        not(all(feature = "simd", target_arch = "x86_64")),
        allow(unused_variables)
    )]
    kernel_name: &str,
    a: &[f32],
    b: &[f32],
    out_f32: &mut [f32],
    op: &(impl Fn(f32, f32) -> f32 + Sync),
    act: &(impl Fn(f32) -> f32 + Sync),
) {
    // Single-pass fused SIMD kernels: combine binary op + activation in one pass.
    // Supports same-shape, scalar broadcast, and tiled broadcast patterns.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if out_f32.len() >= 8
        && (a.len() == out_f32.len() || b.len() == out_f32.len())
        && microkernels::simd_avx2_available()
    {
        match kernel_name {
            "add_relu_f32" => {
                return unsafe { microkernels::fused_add_relu_f32_avx2(a, b, out_f32) }
            }
            "mul_relu_f32" => {
                return unsafe { microkernels::fused_mul_relu_f32_avx2(a, b, out_f32) }
            }
            "add_silu_f32" => {
                return unsafe { microkernels::fused_add_silu_f32_avx2(a, b, out_f32) }
            }
            "mul_silu_f32" => {
                return unsafe { microkernels::fused_mul_silu_f32_avx2(a, b, out_f32) }
            }
            "add_gelu_f32" => {
                return unsafe { microkernels::fused_add_gelu_f32_avx2(a, b, out_f32) }
            }
            _ => {}
        }
    }

    // Two-pass SIMD path: binary op then activation in-place.
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
                // SAFETY: The pointer `ptr` was obtained from `out_f32.as_mut_ptr()` and the
                // two created slices alias the same memory, which is sound because the SIMD
                // kernel requires a shared input and a mutable output that may alias.
                unsafe {
                    let v = std::slice::from_raw_parts(ptr, len);
                    microkernels::relu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                }
            } else if kernel_name.ends_with("gelu_f32") {
                // SAFETY: Same aliasing pattern as relu_f32 — the two raw slices point at the
                // same buffer, which the SIMD kernel accepts as valid input/output.
                unsafe {
                    let v = std::slice::from_raw_parts(ptr, len);
                    microkernels::gelu_f32_avx2(v, std::slice::from_raw_parts_mut(ptr, len));
                }
            } else if kernel_name.ends_with("silu_f32") {
                // SAFETY: Same aliasing pattern as relu_f32 — the two raw slices point at the
                // same buffer, which the SIMD kernel accepts as valid input/output.
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
    if out_len >= 4096 {
        use rayon::prelude::*;
        out_f32[..out_len]
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| {
                let x = op(a[i % a_len], b[i % b_len]);
                *o = act(x);
            });
    } else {
        for i in 0..out_len {
            let x = op(a[i % a_len], b[i % b_len]);
            out_f32[i] = act(x);
        }
    }
}

/// One contiguous span of the output, together with where each operand starts
/// and how far it advances per element (`1` when it is contiguous, `0` when it
/// repeats a single value across the span).
struct BroadcastRun {
    destination: usize,
    lhs: usize,
    rhs: usize,
    len: usize,
    lhs_step: usize,
    rhs_step: usize,
}

/// Decomposes a broadcast into contiguous runs.
///
/// `for_each_broadcast_index` decodes the coordinates of every output element,
/// which costs one division and one modulo per axis per element: a channel-wise
/// broadcast over `[1, 128, 22, 28]` paid six divisions for each of 78 848
/// elements and landed at a few GB/s. The runs carry the same information with
/// the coordinate math hoisted out of the element loop, so the inner loop is a
/// plain slice walk that the compiler can vectorise.
fn broadcast_runs(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    output_elements: usize,
) -> Vec<BroadcastRun> {
    let rank = output_shape.len();
    // Per output axis: the operand's own row-major multiplier, or zero when the
    // axis is broadcast (its extent is one) or absent from the operand.
    let multipliers = |shape: &[usize]| -> Vec<usize> {
        let pad = rank - shape.len();
        let mut strides = vec![0usize; rank];
        let mut running = 1usize;
        for axis in (0..shape.len()).rev() {
            strides[pad + axis] = running;
            running *= shape[axis];
        }
        (0..rank)
            .map(|axis| {
                let extent = if axis < pad { 1 } else { shape[axis - pad] };
                if extent == 1 && output_shape[axis] > 1 {
                    0
                } else {
                    strides[axis]
                }
            })
            .collect()
    };
    let lhs_multiplier = multipliers(lhs_shape);
    let rhs_multiplier = multipliers(rhs_shape);

    // Axes of extent one carry no information; dropping them keeps the odometer
    // from spending steps that change nothing.
    let axes: Vec<usize> = (0..rank).filter(|&axis| output_shape[axis] > 1).collect();

    if axes.is_empty() {
        // Every output element is the same value.
        return vec![BroadcastRun {
            destination: 0,
            lhs: 0,
            rhs: 0,
            len: output_elements.max(1),
            lhs_step: 0,
            rhs_step: 0,
        }];
    }

    // Merge trailing axes into one span while both operands keep advancing by
    // their existing step. The innermost multiplier of a row-major operand is
    // always one or zero, so the merged step stays one or zero.
    let mut len = output_shape[*axes.last().expect("axes is non-empty")];
    let lhs_step = lhs_multiplier[*axes.last().expect("axes is non-empty")];
    let rhs_step = rhs_multiplier[*axes.last().expect("axes is non-empty")];
    let mut odometer_axes = axes.len() - 1;
    while odometer_axes > 0 {
        let axis = axes[odometer_axes - 1];
        if lhs_multiplier[axis] != len * lhs_step || rhs_multiplier[axis] != len * rhs_step {
            break;
        }
        len *= output_shape[axis];
        odometer_axes -= 1;
    }

    let mut runs = Vec::with_capacity(output_elements / len.max(1));
    let mut coordinates = vec![0usize; odometer_axes];
    let mut lhs_base = 0usize;
    let mut rhs_base = 0usize;
    let mut destination = 0usize;
    loop {
        runs.push(BroadcastRun {
            destination,
            lhs: lhs_base,
            rhs: rhs_base,
            len,
            lhs_step,
            rhs_step,
        });
        destination += len;

        // Odometer carry: one pass per span instead of one pass per element.
        let mut index = odometer_axes;
        let mut advanced = false;
        while index > 0 {
            index -= 1;
            let axis = axes[index];
            coordinates[index] += 1;
            lhs_base += lhs_multiplier[axis];
            rhs_base += rhs_multiplier[axis];
            if coordinates[index] < output_shape[axis] {
                advanced = true;
                break;
            }
            lhs_base -= lhs_multiplier[axis] * output_shape[axis];
            rhs_base -= rhs_multiplier[axis] * output_shape[axis];
            coordinates[index] = 0;
        }
        if !advanced {
            return runs;
        }
    }
}

/// Applies `op` over precomputed runs.
fn apply_broadcast_runs(
    runs: &[BroadcastRun],
    lhs: &[f32],
    rhs: &[f32],
    output: &mut [f32],
    op: impl Fn(f32, f32) -> f32,
) {
    for run in runs {
        let span = &mut output[run.destination..run.destination + run.len];
        match (run.lhs_step, run.rhs_step) {
            (1, 1) => {
                let lhs_span = &lhs[run.lhs..run.lhs + run.len];
                let rhs_span = &rhs[run.rhs..run.rhs + run.len];
                for ((slot, &a), &b) in span.iter_mut().zip(lhs_span).zip(rhs_span) {
                    *slot = op(a, b);
                }
            }
            (1, 0) => {
                // A single value of the right operand covers the whole span, the
                // shape a channel or per-row scale takes.
                let b = rhs[run.rhs];
                for (slot, &a) in span.iter_mut().zip(&lhs[run.lhs..run.lhs + run.len]) {
                    *slot = op(a, b);
                }
            }
            (0, 1) => {
                let a = lhs[run.lhs];
                for (slot, &b) in span.iter_mut().zip(&rhs[run.rhs..run.rhs + run.len]) {
                    *slot = op(a, b);
                }
            }
            (0, 0) => {
                let value = op(lhs[run.lhs], rhs[run.rhs]);
                span.fill(value);
            }
            (lhs_step, rhs_step) => {
                for (offset, slot) in span.iter_mut().enumerate() {
                    *slot = op(
                        lhs[run.lhs + offset * lhs_step],
                        rhs[run.rhs + offset * rhs_step],
                    );
                }
            }
        }
    }
}

fn for_each_broadcast_index(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    output_elements: usize,
    mut write: impl FnMut(usize, usize, usize),
) {
    for flat_output_index in 0..output_elements {
        let mut remaining = flat_output_index;
        let mut lhs_index = 0usize;
        let mut rhs_index = 0usize;
        let mut lhs_stride = 1usize;
        let mut rhs_stride = 1usize;
        for output_axis in (0..output_shape.len()).rev() {
            let coordinate = remaining % output_shape[output_axis];
            remaining /= output_shape[output_axis];
            if let Some(lhs_axis) = output_axis.checked_sub(output_shape.len() - lhs_shape.len()) {
                if lhs_shape[lhs_axis] != 1 {
                    lhs_index += coordinate * lhs_stride;
                }
                lhs_stride *= lhs_shape[lhs_axis];
            }
            if let Some(rhs_axis) = output_axis.checked_sub(output_shape.len() - rhs_shape.len()) {
                if rhs_shape[rhs_axis] != 1 {
                    rhs_index += coordinate * rhs_stride;
                }
                rhs_stride *= rhs_shape[rhs_axis];
            }
        }
        write(flat_output_index, lhs_index, rhs_index);
    }
}

pub(super) fn broadcast_binary_dispatch(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    output_slice: BufferSlice,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> Result<(), String> {
    let [lhs_slice, rhs_slice] = input_slices else {
        return Err("broadcast binary op requires exactly two inputs".into());
    };
    let product = |shape: &[usize]| {
        shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
            .ok_or_else(|| "broadcast shape product overflows".to_string())
    };
    let lhs_elements = product(lhs_shape)?;
    let rhs_elements = product(rhs_shape)?;
    let output_elements = product(output_shape)?;
    let widths = [
        lhs_slice.size.checked_div(lhs_elements),
        rhs_slice.size.checked_div(rhs_elements),
        output_slice.size.checked_div(output_elements),
    ];
    let [lhs_width, rhs_width, output_width] = widths.map(Option::unwrap_or_default);
    if lhs_elements == 0
        || rhs_elements == 0
        || output_elements == 0
        || !matches!(lhs_width, 1 | 4)
        || !matches!(rhs_width, 1 | 4)
        || !matches!(output_width, 1 | 4)
    {
        return Err(format!(
            "broadcast semantic shapes disagree with storage: lhs {lhs_shape:?} bytes {}, rhs {rhs_shape:?} bytes {}, output {output_shape:?} bytes {}, expected [{}, {}, {}]",
            lhs_slice.size,
            rhs_slice.size,
            output_slice.size,
            lhs_elements * lhs_width,
            rhs_elements * rhs_width,
            output_elements * output_width,
        ));
    }
    for input_shape in [lhs_shape, rhs_shape] {
        if input_shape.len() > output_shape.len() {
            return Err("broadcast input rank exceeds output rank".into());
        }
        let rank_offset = output_shape.len() - input_shape.len();
        for (axis, &dimension) in input_shape.iter().enumerate() {
            let output_dimension = output_shape[rank_offset + axis];
            if dimension != 1 && dimension != output_dimension {
                return Err(format!(
                    "broadcast dimension {dimension} is incompatible with output dimension {output_dimension} at axis {}",
                    rank_offset + axis
                ));
            }
        }
    }

    if lhs_width != 4 || rhs_width != 4 || output_width != 4 {
        let read_boolean = |slice: BufferSlice, width: usize| -> Result<Vec<f32>, String> {
            let values = if width == 1 {
                unsafe { arena.view_u8(slice.offset, slice.size) }
                    .iter()
                    .map(|&value| value as f32)
                    .collect()
            } else {
                if !slice.offset.is_multiple_of(std::mem::align_of::<f32>()) {
                    return Err("four-byte boolean broadcast input is misaligned".into());
                }
                unsafe { arena.view_f32(slice.offset, slice.size) }.to_vec()
            };
            if values.iter().any(|value| !matches!(*value, 0.0 | 1.0)) {
                return Err("mixed-width broadcast operands are not canonical booleans".into());
            }
            Ok(values)
        };
        let lhs = read_boolean(*lhs_slice, lhs_width)?;
        let rhs = read_boolean(*rhs_slice, rhs_width)?;
        let mut result = vec![0.0f32; output_elements];
        let mut invalid_boolean = false;
        for_each_broadcast_index(
            lhs_shape,
            rhs_shape,
            output_shape,
            output_elements,
            |destination, lhs_index, rhs_index| {
                let lhs_value = lhs[lhs_index];
                let rhs_value = rhs[rhs_index];
                let value = op(lhs_value, rhs_value);
                if lhs_value > 1.0 || rhs_value > 1.0 || !(value == 0.0 || value == 1.0) {
                    invalid_boolean = true;
                } else {
                    result[destination] = value;
                }
            },
        );
        if invalid_boolean {
            return Err("one-byte broadcast operands are not canonical booleans".into());
        }
        if output_width == 1 {
            let output = unsafe { arena.view_u8_mut(output_slice.offset, output_slice.size) };
            for (destination, value) in output.iter_mut().zip(result) {
                *destination = value as u8;
            }
        } else {
            if !output_slice
                .offset
                .is_multiple_of(std::mem::align_of::<f32>())
            {
                return Err("four-byte boolean broadcast output is misaligned".into());
            }
            unsafe { arena.view_f32_mut(output_slice.offset, output_slice.size) }
                .copy_from_slice(&result);
        }
    } else {
        let runs = broadcast_runs(lhs_shape, rhs_shape, output_shape, output_elements);
        arena::with_binary_f32_slices(
            arena,
            *lhs_slice,
            *rhs_slice,
            output_slice,
            |lhs, rhs, output| {
                apply_broadcast_runs(&runs, lhs, rhs, output, op);
            },
        );
    }
    Ok(())
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

    #[test]
    fn broadcast_runs_cover_every_element_once_and_match_the_per_element_path() {
        // Real pairings from a CNN graph: a channel vector, a trailing spatial
        // plane, a token vector, a scalar, the same shape, a rank mismatch that
        // right-aligns, and a rank-0 operand.
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[1, 128, 22, 28], &[1, 128, 1, 1], &[1, 128, 22, 28]),
            (&[1, 128, 22, 28], &[1, 1, 22, 28], &[1, 128, 22, 28]),
            (&[16, 49, 49], &[1, 49, 1], &[16, 49, 49]),
            (&[1, 3, 336, 448], &[1], &[1, 3, 336, 448]),
            (&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]),
            (&[5], &[1, 5], &[1, 5]),
            (&[1, 7, 5], &[], &[1, 7, 5]),
            (&[1, 1, 1], &[4, 1, 3], &[1, 4, 3]),
        ];

        for &(lhs_shape, rhs_shape, output_shape) in cases {
            let lhs_len: usize = lhs_shape.iter().product();
            let rhs_len: usize = rhs_shape.iter().product::<usize>().max(1);
            let output_len: usize = output_shape.iter().product();
            let lhs: Vec<f32> = (0..lhs_len).map(|index| index as f32 * 0.5 - 3.0).collect();
            let rhs: Vec<f32> = (0..rhs_len)
                .map(|index| 1.0 + index as f32 * 0.25)
                .collect();

            // The per-element decoder is the reference.
            let mut expected = vec![f32::NAN; output_len];
            for_each_broadcast_index(
                lhs_shape,
                rhs_shape,
                output_shape,
                output_len,
                |destination, lhs_index, rhs_index| {
                    expected[destination] = lhs[lhs_index] * 2.0 + rhs[rhs_index];
                },
            );

            let runs = broadcast_runs(lhs_shape, rhs_shape, output_shape, output_len);
            let covered: usize = runs.iter().map(|run| run.len).sum();
            assert_eq!(
                covered, output_len,
                "runs must cover the output exactly once for {lhs_shape:?} {rhs_shape:?}"
            );

            let mut actual = vec![f32::NAN; output_len];
            apply_broadcast_runs(&runs, &lhs, &rhs, &mut actual, |a, b| a * 2.0 + b);
            assert_eq!(
                actual, expected,
                "run decomposition disagrees for {lhs_shape:?} {rhs_shape:?} -> {output_shape:?}"
            );
        }
    }
}
