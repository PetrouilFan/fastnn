use crate::backend::{BufferSlice, Instruction};

use super::{arena, telemetry, CpuBuffer};

/// Helper: build a `CallKernel` instruction for a scalar op.
///
/// All 6 scalar ops (gt, lt, eq, add, mul, div) share the same shape:
/// no extra params, no secondary output, no weight metadata.
#[inline]
pub(super) fn scalar_kernel_instruction(
    node_id: usize,
    kernel_name: &str,
    input_slices: Vec<BufferSlice>,
    output_slice: BufferSlice,
) -> Instruction {
    Instruction::CallKernel {
        node_id: Some(node_id),
        kernel_name: kernel_name.to_string(),
        input_slices,
        output_slice,
        secondary_output_slice: None,
        params: vec![],
        param_dims: None,
        weight_meta: None,
    }
}

/// Helper: dispatch a scalar op (gt, lt, eq, add, mul, div) at runtime.
///
/// Extracts data + scalar slices from the arena, casts to f32, resolves the
/// scalar value from the first element, and calls `op(data, scalar, output)`.
/// Generic closure `op` is monomorphized and inlined — zero overhead vs the
/// handwritten per-op blocks.
#[inline]
pub(super) fn scalar_op_dispatch(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    out_start: usize,
    out_end: usize,
    op: impl Fn(&[f32], f32, &mut [f32]),
) {
    if let [data_slice, scalar_slice] = &input_slices[..] {
        let scalar = arena::read_scalar_f32(arena, *scalar_slice);
        let output_slice = BufferSlice::new(out_start, out_end - out_start);
        arena::with_unary_f32_slices(arena, *data_slice, output_slice, |data, out| {
            op(data, scalar, out);
        });
    }
}

/// Helper: dispatch a unary f32 op at runtime.
///
/// Same-shape unary dispatch uses the arena helper so disjoint input/output
/// ranges can be borrowed directly without a temporary copy. If the output
/// aliases the input, the arena helper preserves correctness by copying the
/// input first. Mismatched lengths keep the previous pre-copy behavior.
#[inline]
pub(super) fn unary_op_dispatch(
    input_slices: &[BufferSlice],
    arena: &CpuBuffer,
    out_start: usize,
    out_end: usize,
    op: impl FnOnce(&[f32], &mut [f32]),
) {
    let output_slice = BufferSlice::new(out_start, out_end - out_start);

    if let Some(input_slice) = input_slices.first() {
        if input_slice.size == output_slice.size {
            arena::with_unary_f32_slices(arena, *input_slice, output_slice, op);
            return;
        }

        let input = {
            let d = arena.data_mut();
            let src = bytemuck::cast_slice::<_, f32>(
                &d[input_slice.offset..input_slice.offset + input_slice.size],
            );
            let mut buf = crate::backend::cpu::microkernels::TlsVecPool::alloc(src.len());
            buf.copy_from_slice(src);
            telemetry::record_arena_temp_copy(input_slice.size);
            buf
        };
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        op(&input, out_f32);
    } else {
        let input = crate::backend::cpu::microkernels::TlsVecPool::alloc(0);
        let out_f32 = {
            let d = arena.data_mut();
            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
        };
        op(&input, out_f32);
    }
}

#[cfg(test)]
mod scalar_dispatch_tests {
    use super::*;
    use crate::backend::cpu::add_scalar_f32;

    fn arena_from_f32(values: &[f32]) -> CpuBuffer {
        CpuBuffer::new(bytemuck::cast_slice(values).to_vec())
    }

    fn read_f32s(arena: &CpuBuffer, slice: BufferSlice) -> Vec<f32> {
        let end = slice.offset + slice.size;
        let data = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&data[slice.offset..end]).to_vec()
    }

    #[test]
    fn scalar_op_dispatch_disjoint_avoids_temp_copy() {
        telemetry::reset_cpu_telemetry();
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 10.0, 0.0, 0.0, 0.0]);
        let data = BufferSlice::new(0, 3 * std::mem::size_of::<f32>());
        let scalar = BufferSlice::new(3 * std::mem::size_of::<f32>(), std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            4 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );

        scalar_op_dispatch(
            &[data, scalar],
            &arena,
            output.offset,
            output.offset + output.size,
            add_scalar_f32,
        );

        assert_eq!(read_f32s(&arena, output), vec![11.0, 12.0, 13.0]);
        assert_eq!(telemetry::cpu_telemetry_snapshot().arena_temp_copies, 0);
    }

    #[test]
    fn scalar_op_dispatch_overlapping_input_falls_back_to_copy() {
        telemetry::reset_cpu_telemetry();
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 4.0, 10.0, 0.0]);
        let data = BufferSlice::new(0, 4 * std::mem::size_of::<f32>());
        let scalar = BufferSlice::new(4 * std::mem::size_of::<f32>(), std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            2 * std::mem::size_of::<f32>(),
            4 * std::mem::size_of::<f32>(),
        );

        scalar_op_dispatch(
            &[data, scalar],
            &arena,
            output.offset,
            output.offset + output.size,
            add_scalar_f32,
        );

        assert_eq!(read_f32s(&arena, output), vec![11.0, 12.0, 13.0, 14.0]);
        assert!(telemetry::cpu_telemetry_snapshot().arena_temp_copies >= 1);
    }
}

#[cfg(test)]
mod unary_dispatch_tests {
    use super::*;
    use crate::backend::cpu::relu_f32;

    fn arena_from_f32(values: &[f32]) -> CpuBuffer {
        CpuBuffer::new(bytemuck::cast_slice(values).to_vec())
    }

    fn read_f32s(arena: &CpuBuffer, slice: BufferSlice) -> Vec<f32> {
        let end = slice.offset + slice.size;
        let data = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&data[slice.offset..end]).to_vec()
    }

    #[test]
    fn unary_op_dispatch_disjoint_same_shape_avoids_temp_copy() {
        telemetry::reset_cpu_telemetry();
        let arena = arena_from_f32(&[-1.0, 2.0, -3.0, 0.0, 0.0, 0.0]);
        let input = BufferSlice::new(0, 3 * std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            3 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );

        unary_op_dispatch(
            &[input],
            &arena,
            output.offset,
            output.offset + output.size,
            relu_f32,
        );

        assert_eq!(read_f32s(&arena, output), vec![0.0, 2.0, 0.0]);
        assert_eq!(telemetry::cpu_telemetry_snapshot().arena_temp_copies, 0);
    }

    #[test]
    fn unary_op_dispatch_overlapping_input_falls_back_to_copy() {
        telemetry::reset_cpu_telemetry();
        let arena = arena_from_f32(&[-1.0, 2.0, -3.0, 4.0, 0.0, 0.0]);
        let input = BufferSlice::new(0, 4 * std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            2 * std::mem::size_of::<f32>(),
            4 * std::mem::size_of::<f32>(),
        );

        unary_op_dispatch(
            &[input],
            &arena,
            output.offset,
            output.offset + output.size,
            relu_f32,
        );

        assert_eq!(read_f32s(&arena, output), vec![0.0, 2.0, 0.0, 4.0]);
        assert!(telemetry::cpu_telemetry_snapshot().arena_temp_copies >= 1);
    }
}
