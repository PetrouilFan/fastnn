use crate::backend::cpu::microkernels::TlsVecPool;
use crate::backend::cpu::telemetry::record_arena_temp_copy;
use crate::backend::BufferSlice;

use super::CpuBuffer;

#[inline]
pub(super) fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
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

        // SAFETY: the input and output byte ranges were bounds-checked above and
        // proven disjoint, so creating one shared f32 slice and one mutable f32
        // slice from the arena cannot alias. Dispatch is single-threaded for a
        // CpuBuffer, matching CpuBuffer::data_mut's safety contract.
        unsafe {
            let input_f32 = bytes_as_f32_slice(arena_bytes.as_ptr().add(input.offset), input.size);
            let output_f32 =
                bytes_as_f32_slice_mut(arena_bytes.as_mut_ptr().add(output.offset), output.size);
            f(input_f32, output_f32)
        }
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

        // SAFETY: both input ranges are disjoint from output, so neither shared
        // input f32 slice can alias the mutable output f32 slice. The two input
        // slices may overlap each other, which is permitted for shared borrows.
        unsafe {
            let a_f32 = bytes_as_f32_slice(arena_bytes.as_ptr().add(a.offset), a.size);
            let b_f32 = bytes_as_f32_slice(arena_bytes.as_ptr().add(b.offset), b.size);
            let output_f32 =
                bytes_as_f32_slice_mut(arena_bytes.as_mut_ptr().add(output.offset), output.size);
            f(a_f32, b_f32, output_f32)
        }
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
    let input_ends: Vec<usize> = inputs.iter().copied().map(checked_end).collect();
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

        // SAFETY: every input byte range was bounds-checked and proven disjoint
        // from the mutable output byte range. Input ranges may overlap each
        // other, which is fine for shared f32 slices.
        unsafe {
            let input_f32: Vec<&[f32]> = inputs
                .iter()
                .map(|input| bytes_as_f32_slice(arena_bytes.as_ptr().add(input.offset), input.size))
                .collect();
            let output_f32 =
                bytes_as_f32_slice_mut(arena_bytes.as_mut_ptr().add(output.offset), output.size);
            f(&input_f32, output_f32)
        }
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
                .collect::<Vec<_>>()
        };
        let output_f32 = {
            let arena_bytes = arena.data_mut();
            assert_slice_in_bounds(arena_bytes.len(), output, output_end);
            bytemuck::cast_slice_mut::<_, f32>(&mut arena_bytes[output.offset..output_end])
        };
        let input_refs: Vec<&[f32]> = input_copies.iter().map(|copy| &copy[..]).collect();
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
    assert!(
        end <= arena_len,
        "BufferSlice out of bounds: offset={} size={} arena_len={}",
        slice.offset,
        slice.size,
        arena_len
    );
    assert_eq!(
        slice.size % std::mem::size_of::<f32>(),
        0,
        "f32 BufferSlice size must be a multiple of 4"
    );
}

#[inline]
unsafe fn bytes_as_f32_slice<'a>(ptr: *const u8, len_bytes: usize) -> &'a [f32] {
    std::slice::from_raw_parts(ptr.cast::<f32>(), len_bytes / std::mem::size_of::<f32>())
}

#[inline]
unsafe fn bytes_as_f32_slice_mut<'a>(ptr: *mut u8, len_bytes: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(ptr.cast::<f32>(), len_bytes / std::mem::size_of::<f32>())
}

#[cfg(test)]
mod tests {
    use super::*;

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
