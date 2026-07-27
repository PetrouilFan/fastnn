use crate::backend::cpu::microkernels::TlsVecPool;
use crate::backend::cpu::telemetry::record_arena_temp_copy;
use crate::backend::BufferSlice;
use smallvec::SmallVec;

use super::CpuBuffer;

#[inline]
pub(super) fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

fn with_disjoint_output_bytes<R>(
    arena_bytes: &mut [u8],
    inputs: &[BufferSlice],
    output: BufferSlice,
    f: impl FnOnce(&[&[u8]], &mut [u8]) -> R,
) -> R {
    let output_end = checked_end(output);
    let (before_output, output_and_after) = arena_bytes.split_at_mut(output.offset);
    let (output_bytes, after_output) = output_and_after.split_at_mut(output.size);
    let input_bytes: SmallVec<[&[u8]; 8]> = inputs
        .iter()
        .map(|input| {
            let input_end = checked_end(*input);
            if input_end <= output.offset {
                &before_output[input.offset..input_end]
            } else {
                debug_assert!(input.offset >= output_end);
                let start = input.offset - output_end;
                &after_output[start..start + input.size]
            }
        })
        .collect();
    f(&input_bytes, output_bytes)
}

#[inline]
pub(super) fn with_unary_f32_slices<R>(
    arena: &CpuBuffer,
    input: BufferSlice,
    output: BufferSlice,
    f: impl FnOnce(&[f32], &mut [f32]) -> R,
) -> R {
    let input_end = checked_end(input);
    let output_end = checked_end(output);

    if !ranges_overlap(input.offset, input_end, output.offset, output_end) {
        let arena_bytes = arena.data_mut();
        assert_slice_in_bounds(arena_bytes.len(), input, input_end);
        assert_slice_in_bounds(arena_bytes.len(), output, output_end);

        with_disjoint_output_bytes(arena_bytes, &[input], output, |inputs, output| {
            let input_f32 = bytemuck::cast_slice::<_, f32>(inputs[0]);
            let output_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            f(input_f32, output_f32)
        })
    } else {
        let input_copy = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), input, input_end);
            let input_f32 = bytemuck::cast_slice::<_, f32>(&arena_bytes[input.offset..input_end]);
            let mut copy = TlsVecPool::alloc(input_f32.len());
            copy.copy_from_slice(input_f32);
            record_arena_temp_copy(input.size);
            copy
        };
        let output_f32 = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), output, output_end);
            bytemuck::cast_slice_mut::<_, f32>(&mut arena_bytes[output.offset..output_end])
        };
        f(&input_copy, output_f32)
    }
}

#[inline]
pub(super) fn with_binary_f32_slices<R>(
    arena: &CpuBuffer,
    a: BufferSlice,
    b: BufferSlice,
    output: BufferSlice,
    f: impl FnOnce(&[f32], &[f32], &mut [f32]) -> R,
) -> R {
    let a_end = checked_end(a);
    let b_end = checked_end(b);
    let output_end = checked_end(output);

    let output_overlaps_a = ranges_overlap(a.offset, a_end, output.offset, output_end);
    let output_overlaps_b = ranges_overlap(b.offset, b_end, output.offset, output_end);

    if !output_overlaps_a && !output_overlaps_b {
        let arena_bytes = arena.data_mut();
        assert_slice_in_bounds(arena_bytes.len(), a, a_end);
        assert_slice_in_bounds(arena_bytes.len(), b, b_end);
        assert_slice_in_bounds(arena_bytes.len(), output, output_end);

        with_disjoint_output_bytes(arena_bytes, &[a, b], output, |inputs, output| {
            let a_f32 = bytemuck::cast_slice::<_, f32>(inputs[0]);
            let b_f32 = bytemuck::cast_slice::<_, f32>(inputs[1]);
            let output_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            f(a_f32, b_f32, output_f32)
        })
    } else if output_overlaps_a && !output_overlaps_b {
        let a_copy = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), a, a_end);
            let a_f32 = bytemuck::cast_slice::<_, f32>(&arena_bytes[a.offset..a_end]);
            let mut copy = TlsVecPool::alloc(a_f32.len());
            copy.copy_from_slice(a_f32);
            record_arena_temp_copy(a.size);
            copy
        };

        let arena_bytes = arena.data_mut();
        assert_slice_in_bounds(arena_bytes.len(), b, b_end);
        assert_slice_in_bounds(arena_bytes.len(), output, output_end);

        with_disjoint_output_bytes(arena_bytes, &[b], output, |inputs, output| {
            let b_f32 = bytemuck::cast_slice::<_, f32>(inputs[0]);
            let output_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            f(&a_copy, b_f32, output_f32)
        })
    } else if output_overlaps_b && !output_overlaps_a {
        let b_copy = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), b, b_end);
            let b_f32 = bytemuck::cast_slice::<_, f32>(&arena_bytes[b.offset..b_end]);
            let mut copy = TlsVecPool::alloc(b_f32.len());
            copy.copy_from_slice(b_f32);
            record_arena_temp_copy(b.size);
            copy
        };

        let arena_bytes = arena.data_mut();
        assert_slice_in_bounds(arena_bytes.len(), a, a_end);
        assert_slice_in_bounds(arena_bytes.len(), output, output_end);

        with_disjoint_output_bytes(arena_bytes, &[a], output, |inputs, output| {
            let a_f32 = bytemuck::cast_slice::<_, f32>(inputs[0]);
            let output_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            f(a_f32, &b_copy, output_f32)
        })
    } else {
        let (a_copy, b_copy) = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), a, a_end);
            assert_slice_in_bounds(arena_bytes.len(), b, b_end);

            let a_f32 = bytemuck::cast_slice::<_, f32>(&arena_bytes[a.offset..a_end]);
            let b_f32 = bytemuck::cast_slice::<_, f32>(&arena_bytes[b.offset..b_end]);

            let mut a_copy = TlsVecPool::alloc(a_f32.len());
            a_copy.copy_from_slice(a_f32);
            record_arena_temp_copy(a.size);

            let mut b_copy = TlsVecPool::alloc(b_f32.len());
            b_copy.copy_from_slice(b_f32);
            record_arena_temp_copy(b.size);

            (a_copy, b_copy)
        };
        let output_f32 = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), output, output_end);
            bytemuck::cast_slice_mut::<_, f32>(&mut arena_bytes[output.offset..output_end])
        };
        f(&a_copy, &b_copy, output_f32)
    }
}

#[inline]
pub(super) fn with_nary_f32_slices<R>(
    arena: &CpuBuffer,
    inputs: &[BufferSlice],
    output: BufferSlice,
    f: impl FnOnce(&[&[f32]], &mut [f32]) -> R,
) -> R {
    let input_ends: SmallVec<[usize; 8]> = inputs.iter().copied().map(checked_end).collect();
    let output_end = checked_end(output);
    let output_overlaps_input =
        inputs
            .iter()
            .copied()
            .zip(input_ends.iter().copied())
            .any(|(input, input_end)| {
                ranges_overlap(input.offset, input_end, output.offset, output_end)
            });

    if !output_overlaps_input {
        let arena_bytes = arena.data_mut();
        for (&input, &input_end) in inputs.iter().zip(&input_ends) {
            assert_slice_in_bounds(arena_bytes.len(), input, input_end);
        }
        assert_slice_in_bounds(arena_bytes.len(), output, output_end);

        with_disjoint_output_bytes(arena_bytes, inputs, output, |inputs, output| {
            let input_f32: SmallVec<[&[f32]; 8]> = inputs
                .iter()
                .map(|input| bytemuck::cast_slice::<_, f32>(input))
                .collect();
            let output_f32 = bytemuck::cast_slice_mut::<_, f32>(output);
            f(&input_f32, output_f32)
        })
    } else {
        let input_copies = {
            let arena_bytes = arena.data_mut();
            inputs
                .iter()
                .copied()
                .zip(input_ends.iter().copied())
                .map(|(input, input_end)| {
                    assert_slice_in_bounds(arena_bytes.len(), input, input_end);
                    let input_f32 =
                        bytemuck::cast_slice::<_, f32>(&arena_bytes[input.offset..input_end]);
                    let mut copy = TlsVecPool::alloc(input_f32.len());
                    copy.copy_from_slice(input_f32);
                    record_arena_temp_copy(input.size);
                    copy
                })
                .collect::<SmallVec<[_; 8]>>()
        };
        let output_f32 = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), output, output_end);
            bytemuck::cast_slice_mut::<_, f32>(&mut arena_bytes[output.offset..output_end])
        };
        let input_refs: SmallVec<[&[f32]; 8]> = input_copies.iter().map(|copy| &copy[..]).collect();
        f(&input_refs, output_f32)
    }
}

#[inline]
pub(super) fn read_scalar_f32(arena: &CpuBuffer, scalar: BufferSlice) -> f32 {
    assert!(
        scalar.size >= std::mem::size_of::<f32>(),
        "scalar BufferSlice must contain at least one f32"
    );
    let scalar_end = checked_end(scalar);
    let arena_bytes = arena.data_mut();
    assert_slice_in_bounds(arena_bytes.len(), scalar, scalar_end);
    bytemuck::cast_slice::<_, f32>(&arena_bytes[scalar.offset..scalar_end])[0]
}

#[inline]
fn checked_end(slice: BufferSlice) -> usize {
    slice
        .offset
        .checked_add(slice.size)
        .expect("BufferSlice offset + size overflow")
}

#[inline]
fn assert_slice_in_bounds(arena_len: usize, slice: BufferSlice, end: usize) {
    debug_assert!(
        end <= arena_len,
        "BufferSlice out of bounds: offset={} size={} arena_len={}",
        slice.offset,
        slice.size,
        arena_len
    );
    debug_assert_eq!(
        slice.size % std::mem::size_of::<f32>(),
        0,
        "f32 BufferSlice size must be a multiple of 4"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::telemetry::{cpu_telemetry_snapshot, reset_cpu_telemetry};

    fn arena_from_f32(values: &[f32]) -> CpuBuffer {
        CpuBuffer::new(bytemuck::cast_slice(values).to_vec())
    }

    fn read_f32s(arena: &CpuBuffer, slice: BufferSlice) -> Vec<f32> {
        let end = slice.offset + slice.size;
        let data = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&data[slice.offset..end]).to_vec()
    }

    #[test]
    fn ranges_overlap_edge_cases() {
        assert!(ranges_overlap(0, 4, 2, 6));
        assert!(ranges_overlap(2, 6, 0, 4));
        assert!(ranges_overlap(0, 4, 0, 4));
        assert!(!ranges_overlap(0, 4, 4, 8));
        assert!(!ranges_overlap(4, 8, 0, 4));
        assert!(!ranges_overlap(0, 0, 0, 4));
        assert!(!ranges_overlap(0, 4, 4, 4));
        assert!(!ranges_overlap(usize::MAX - 1, usize::MAX, 0, 1));
    }

    #[test]
    fn disjoint_unary_input_output_direct_helper_result() {
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        let input = BufferSlice::new(0, 3 * std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            3 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );

        with_unary_f32_slices(&arena, input, output, |input, output| {
            for (dst, src) in output.iter_mut().zip(input) {
                *dst = *src + 1.0;
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn overlapping_unary_fallback_result() {
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
        let input = BufferSlice::new(0, 4 * std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            2 * std::mem::size_of::<f32>(),
            4 * std::mem::size_of::<f32>(),
        );

        with_unary_f32_slices(&arena, input, output, |input, output| {
            for (dst, src) in output.iter_mut().zip(input) {
                *dst = *src * 10.0;
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn disjoint_binary_helper_result() {
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0]);
        let a = BufferSlice::new(0, 3 * std::mem::size_of::<f32>());
        let b = BufferSlice::new(
            3 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );
        let output = BufferSlice::new(
            6 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );

        with_binary_f32_slices(&arena, a, b, output, |a, b, output| {
            for ((dst, lhs), rhs) in output.iter_mut().zip(a).zip(b) {
                *dst = *lhs + *rhs;
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn binary_single_output_overlap_copies_only_overlapping_operand() {
        reset_cpu_telemetry();

        let arena = arena_from_f32(&[
            1.0, 2.0, 3.0, 4.0, // a
            0.0, 0.0, // output tail before execution
            10.0, 20.0, 30.0, 40.0, // b
        ]);
        let a = BufferSlice::new(0, 4 * std::mem::size_of::<f32>());
        let output = BufferSlice::new(
            2 * std::mem::size_of::<f32>(),
            4 * std::mem::size_of::<f32>(),
        );
        let b = BufferSlice::new(
            6 * std::mem::size_of::<f32>(),
            4 * std::mem::size_of::<f32>(),
        );

        with_binary_f32_slices(&arena, a, b, output, |a, b, output| {
            for i in 0..output.len() {
                output[i] = a[i] + b[i];
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![11.0, 22.0, 33.0, 44.0]);
        let snapshot = cpu_telemetry_snapshot();
        assert_eq!(snapshot.arena_temp_copies, 1);
        assert_eq!(snapshot.arena_temp_copy_bytes, a.size as u64);

        reset_cpu_telemetry();
    }

    #[test]
    fn scalar_read() {
        let arena = arena_from_f32(&[1.0, 2.5, 3.0]);
        let scalar = BufferSlice::new(std::mem::size_of::<f32>(), std::mem::size_of::<f32>());

        assert_eq!(read_scalar_f32(&arena, scalar), 2.5);
    }

    #[test]
    fn disjoint_nary_input_output_direct_helper_result() {
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0]);
        let inputs = [
            BufferSlice::new(0, 3 * std::mem::size_of::<f32>()),
            BufferSlice::new(
                3 * std::mem::size_of::<f32>(),
                3 * std::mem::size_of::<f32>(),
            ),
        ];
        let output = BufferSlice::new(
            6 * std::mem::size_of::<f32>(),
            3 * std::mem::size_of::<f32>(),
        );

        with_nary_f32_slices(&arena, &inputs, output, |inputs, output| {
            for i in 0..output.len() {
                output[i] = inputs[0][i] + inputs[1][i];
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn overlapping_nary_fallback_result() {
        let arena = arena_from_f32(&[1.0, 2.0, 3.0, 4.0, 0.0]);
        let inputs = [BufferSlice::new(0, 4 * std::mem::size_of::<f32>())];
        let output = BufferSlice::new(std::mem::size_of::<f32>(), 4 * std::mem::size_of::<f32>());

        with_nary_f32_slices(&arena, &inputs, output, |inputs, output| {
            for (dst, src) in output.iter_mut().zip(inputs[0]) {
                *dst = *src * 2.0;
            }
        });

        assert_eq!(read_f32s(&arena, output), vec![2.0, 4.0, 6.0, 8.0]);
    }
}
