//! Tensor metadata and host storage contract tests (FN3-06 / #52).
//!
//! These tests pin the foundation every backend and the importer build on:
//! checked element/byte/offset arithmetic on both integer widths, checked
//! offset-bearing windows over shared storage, the view families the alpha
//! supports, materialisation, the aliasing rule, and the scalar/empty edges.
//!
//! Every test uses only the public `fastnn::fn3` surface.

use fastnn::fn3::device::{available_devices, DeviceRequest, PhysicalDeviceId};
use fastnn::fn3::storage::{DTypeAlignment, StorageWindow};
use fastnn::fn3::{
    AlignedBuffer, Backend, CpuBackend, DType, DeviceId, Fn3Error, OpId, OpParams, Shape, Storage,
    Strides, Tensor, TensorMeta, TensorMut, TensorView,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cpu() -> DeviceId {
    DeviceId::Cpu
}

fn vulkan() -> DeviceId {
    DeviceId::Vulkan(PhysicalDeviceId::new(0))
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn storage_of(values: &[f32]) -> Storage {
    Storage::from_bytes(&f32_bytes(values)).unwrap()
}

fn shape(dims: &[u64]) -> Shape {
    Shape::from_dims(dims.iter().copied())
}

fn f32_view<'a>(
    storage: &'a Storage,
    dims: &[u64],
    strides: &[i64],
    offset: usize,
) -> TensorView<'a> {
    let meta = TensorMeta::strided(
        shape(dims),
        Strides::from_elements(strides.to_vec()),
        offset,
        DType::F32,
        cpu(),
    )
    .unwrap();
    TensorView::new(storage, meta).unwrap()
}

/// Row-major element indices of a view, computed independently of
/// `TensorMeta::checked_index` so the materialisation tests do not validate the
/// implementation against itself.
///
/// A rank-0 shape has one element; a zero extent has none; no division by a
/// zero extent is ever attempted.
fn logical_indices(dims: &[u64], strides: &[i64], offset: usize) -> Vec<usize> {
    let rank = dims.len();
    let total: usize = dims.iter().map(|&extent| extent as usize).product();
    let mut indices = Vec::with_capacity(total);
    let mut coordinates = vec![0usize; rank];
    for linear in 0..total {
        let mut remainder = linear;
        for axis in (0..rank).rev() {
            let extent = dims[axis] as usize;
            coordinates[axis] = remainder % extent;
            remainder /= extent;
        }
        let mut index = offset;
        for axis in 0..rank {
            index += coordinates[axis] * strides[axis] as usize;
        }
        indices.push(index);
    }
    indices
}

/// Materialise a strided view and prove its bytes are the logically indexed
/// source elements in row-major order.
fn assert_materialises(dims: &[u64], strides: &[i64], offset: usize, source: &[f32]) {
    let storage = storage_of(source);
    let expected: Vec<f32> = logical_indices(dims, strides, offset)
        .into_iter()
        .map(|index| source[index])
        .collect();
    let view = f32_view(&storage, dims, strides, offset);
    let contiguous = view.to_contiguous().unwrap();

    assert_eq!(contiguous.shape(), &shape(dims), "shape after materialise");
    assert!(
        contiguous.is_contiguous(),
        "materialised tensor is contiguous"
    );
    assert_eq!(contiguous.numel(), Ok(expected.len() as u64));
    assert_eq!(contiguous.as_f32_slice().unwrap(), expected.as_slice());

    let expected_bytes: Vec<u8> = expected
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let materialised_view = contiguous.as_view().unwrap();
    let start = materialised_view.meta().byte_offset().unwrap();
    let end = start + expected_bytes.len();
    assert_eq!(
        &materialised_view.storage_bytes()[start..end],
        expected_bytes.as_slice(),
        "materialised bytes must equal the logically indexed source bytes"
    );
}

/// A type aligned more strictly than a byte window's `offset` guarantees.
///
/// `AlignedBuffer` guarantees 64-byte alignment, so a window at byte offset 8 is
/// deterministically not 16-byte aligned. The payload is never read: only the
/// size and alignment of the type matter to the cast check.
#[repr(align(16))]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Align16([u8; 16]);

// SAFETY: `Align16` is a wrapper around a 16-byte `[u8; 16]` with an explicit
// 16-byte alignment. It has no padding (the payload is all bytes), every bit
// pattern is a valid value, and it contains no references or invalid niches,
// which is exactly what `Pod` requires.
unsafe impl bytemuck::Zeroable for Align16 {}
unsafe impl bytemuck::Pod for Align16 {}

// ---------------------------------------------------------------------------
// 1. Checked arithmetic: u64 boundaries, usize boundaries, offsets and strides.
// ---------------------------------------------------------------------------

#[test]
fn element_count_boundary_is_exact_at_u64_max() {
    // 3 * (u64::MAX / 3) == u64::MAX exactly: the largest product that does not
    // overflow.
    let third = u64::MAX / 3;
    assert_eq!(Shape::from_dims([third, 3]).numel(), Ok(u64::MAX));
    assert_eq!(Shape::from_dims([1, third, 3]).numel(), Ok(u64::MAX));
    assert_eq!(Shape::from_dims([u64::MAX]).numel(), Ok(u64::MAX));

    let error = Shape::from_dims([third + 1, 3]).numel().unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::ShapeOverflow {
            context: "Shape::numel",
            axis: Some(1),
            ..
        }
    ));
    assert!(error.to_string().contains("overflow"), "{error}");
}

#[test]
fn byte_length_boundary_is_exact_at_u64_max() {
    let quarter = u64::MAX / 4;
    let eighth = u64::MAX / 8;
    // The largest products that do not overflow u64. They only fit a usize on a
    // 64-bit target; a 32-bit target must report them, not truncate.
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(
            Shape::from_dims([quarter]).byte_len(DType::F32),
            Ok((quarter * 4) as usize)
        );
        assert_eq!(
            Shape::from_dims([eighth]).byte_len(DType::I64),
            Ok((eighth * 8) as usize)
        );
    }
    #[cfg(target_pointer_width = "32")]
    {
        assert!(Shape::from_dims([quarter]).byte_len(DType::F32).is_err());
        assert!(Shape::from_dims([eighth]).byte_len(DType::I64).is_err());
    }

    let f32_overflow = Shape::from_dims([quarter + 1])
        .byte_len(DType::F32)
        .unwrap_err();
    assert!(matches!(
        f32_overflow,
        Fn3Error::ShapeOverflow {
            context: "Shape::byte_len",
            ..
        }
    ));
    assert!(Shape::from_dims([eighth + 1]).byte_len(DType::I64).is_err());

    // One byte per element is the only dtype where u64::MAX elements can fit a
    // 64-bit usize.
    let bool_len = Shape::from_dims([u64::MAX]).byte_len(DType::Bool);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(bool_len, Ok(usize::MAX));
    #[cfg(target_pointer_width = "32")]
    {
        let error = bool_len.unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::ShapeOverflow {
                context: "Shape::byte_len",
                ..
            }
        ));
        assert!(error.to_string().contains("usize"), "{error}");
    }
}

#[test]
fn usize_width_is_checked_and_never_truncated() {
    let max_u32 = u32::MAX as u64;
    assert_eq!(
        Shape::from_dims([max_u32]).numel_usize(),
        Ok(u32::MAX as usize)
    );

    let beyond_u32 = Shape::from_dims([max_u32 + 1]);
    #[cfg(target_pointer_width = "32")]
    {
        let error = beyond_u32.numel_usize().unwrap_err();
        assert!(matches!(
            error,
            Fn3Error::ShapeOverflow {
                context: "Shape::numel_usize",
                ..
            }
        ));
        assert!(error.to_string().contains("usize"), "{error}");
        // A 1-byte dtype does not rescue a count that does not fit usize.
        assert!(beyond_u32.byte_len(DType::Bool).is_err());
    }
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(beyond_u32.numel_usize(), Ok(max_u32 as usize + 1));
        assert_eq!(Shape::from_dims([u64::MAX]).numel_usize(), Ok(usize::MAX));
    }
}

#[test]
fn stride_span_boundaries_are_exact() {
    // 1 + (2 - 1) * i64::MAX == 2^63: no overflow.
    let two = Shape::from_dims([2]);
    let largest = Strides::from_elements(vec![i64::MAX]);
    assert_eq!(largest.required_len(&two), Ok(1u64 << 63));

    // 1 + (3 - 1) * i64::MAX == u64::MAX: the largest span that does not
    // overflow.
    let three = Shape::from_dims([3]);
    assert_eq!(largest.required_len(&three), Ok(u64::MAX));
    assert_eq!(largest.required_bytes(&three, DType::Bool), Ok(usize::MAX));

    // One more element overflows the u64 span.
    let four = Shape::from_dims([4]);
    let error = largest.required_len(&four).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::ShapeOverflow {
            context: "Strides::required_len",
            axis: Some(0),
            ..
        }
    ));

    // A span of u64::MAX bytes only fits a 64-bit usize.
    let bytes = largest.required_bytes(&three, DType::Bool);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(bytes, Ok(usize::MAX));
    #[cfg(target_pointer_width = "32")]
    assert!(bytes.is_err());

    assert!(largest.required_bytes(&two, DType::F32).is_err());
    assert!(largest.required_len_usize(&four).is_err());
    let span_usize = largest.required_len_usize(&three);
    #[cfg(target_pointer_width = "64")]
    assert_eq!(span_usize, Ok(usize::MAX));
    #[cfg(target_pointer_width = "32")]
    assert!(span_usize.is_err());
}

#[test]
fn element_offset_boundaries_are_checked() {
    let near_max = usize::MAX - 1;
    // A scalar one byte from the end of the address space: offset + 1 ==
    // usize::MAX, representable on both widths.
    let meta = TensorMeta::strided(
        Shape::scalar(),
        Strides::from_elements(vec![]),
        near_max,
        DType::Bool,
        cpu(),
    )
    .unwrap();
    assert_eq!(meta.required_elements(), Ok(usize::MAX as u64));
    assert_eq!(meta.required_bytes(), Ok(usize::MAX));
    assert_eq!(meta.byte_offset(), Ok(near_max));
    assert_eq!(meta.checked_index(&[]), Ok(near_max));

    // Two elements at that offset overflow either the u64 sum (64-bit) or the
    // usize byte count (32-bit); both are ShapeOverflow, never a wrap.
    let error = TensorMeta::strided(
        Shape::from_dims([2]),
        Strides::from_elements(vec![1]),
        near_max,
        DType::Bool,
        cpu(),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::ShapeOverflow { .. }), "{error}");
}

#[test]
fn checked_index_names_the_axis_and_value() {
    let tensor = Tensor::from_f32(Shape::from_dims([2, 3]), &[0.0; 6]).unwrap();
    let meta = tensor.meta();

    let error = meta.checked_index(&[0, 3]).unwrap_err();
    assert!(matches!(error, Fn3Error::InvalidShape { .. }));
    let text = error.to_string();
    assert!(text.contains("axis 1"), "{text}");
    assert!(text.contains('3'), "{text}");

    assert!(matches!(
        meta.checked_index(&[0]),
        Err(Fn3Error::RankMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert_eq!(meta.checked_index(&[1, 2]), Ok(5));
    assert_eq!(meta.checked_index(&[0, 0]), Ok(0));
}

// ---------------------------------------------------------------------------
// 2. Offset-bearing byte windows over shared storage.
// ---------------------------------------------------------------------------

#[test]
fn window_checks_alignment_and_names_the_dtype() {
    let storage = storage_of(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // 24 bytes
    let window = storage.window(8, 8, DType::F32).unwrap();
    assert_eq!(window.offset(), 8);
    assert_eq!(window.len(), 8);
    assert_eq!(window.end(), 16);
    assert_eq!(window.element_offset(DType::F32), Ok(2));
    assert_eq!(window.as_slice::<f32>().unwrap().len(), 2);

    let error = storage.window(6, 8, DType::F32).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::StorageWindow {
            what: "Storage::window",
            offset: 6,
            len: 8,
            storage_len: 24,
            ..
        }
    ));
    let text = error.to_string();
    assert!(text.contains("f32"), "{text}");
    assert!(text.contains("4-byte"), "{text}");
    assert!(text.contains('6'), "{text}");

    // Bool's hint is one byte, so every offset is aligned.
    assert!(storage.window(1, 1, DType::Bool).is_ok());
    assert!(storage.window(1, 1, DType::I64).is_err());
}

#[test]
fn window_checks_bounds_and_never_wraps() {
    let storage = storage_of(&[0.0; 6]); // 24 bytes
    assert!(storage.window(0, 24, DType::F32).is_ok());
    assert!(storage.window(24, 0, DType::F32).is_ok());
    assert!(storage.window(25, 0, DType::Bool).is_err());
    assert!(storage.window(0, 25, DType::Bool).is_err());
    assert!(storage.window(20, 8, DType::F32).is_err());

    // usize::MAX with a non-zero length must be an overflow error, not a wrap
    // to an in-bounds window.
    let error = storage.window(usize::MAX, 1, DType::Bool).unwrap_err();
    assert!(matches!(error, Fn3Error::StorageWindow { .. }));
    assert!(error.to_string().contains("overflows"), "{error}");

    let error = storage.byte_range(usize::MAX, 1).unwrap_err();
    assert!(matches!(error, Fn3Error::StorageWindow { .. }));
    assert!(storage.byte_range(8, 8).unwrap() == &storage.as_bytes()[8..16]);
}

#[test]
fn window_views_share_the_allocation_and_respect_the_window() {
    let storage = storage_of(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let window = storage.window(8, 12, DType::F32).unwrap();
    let view = TensorView::from_window(&window, Shape::from_dims([3]), DType::F32, cpu()).unwrap();
    assert_eq!(view.meta().offset(), 2);
    assert_eq!(view.meta().byte_offset(), Ok(8));
    assert_eq!(view.as_f32_slice().unwrap(), &[2.0, 3.0, 4.0]);

    // The parent storage is long enough for a [5] tensor at offset 8, but the
    // window is not: the window is the promise that is enforced.
    let narrow = storage.window(8, 8, DType::F32).unwrap();
    let error =
        TensorView::from_window(&narrow, Shape::from_dims([3]), DType::F32, cpu()).unwrap_err();
    assert!(matches!(error, Fn3Error::StorageWindow { .. }), "{error}");
    assert!(
        error.to_string().contains("beyond the window end"),
        "{error}"
    );

    // A zero-element tensor may sit exactly at the end of the window.
    let empty_window = storage.window(24, 0, DType::F32).unwrap();
    let empty =
        TensorView::from_window(&empty_window, Shape::from_dims([0]), DType::F32, cpu()).unwrap();
    assert_eq!(empty.as_f32_slice().unwrap().len(), 0);
}

#[test]
fn window_tensors_share_storage_and_cannot_be_mutated_while_shared() {
    let storage = storage_of(&[0.0, 1.0, 2.0, 3.0]);
    let mut first = Tensor::from_window(
        storage.window(0, 8, DType::F32).unwrap(),
        Shape::from_dims([2]),
        DType::F32,
        cpu(),
    )
    .unwrap();
    let second = Tensor::from_window(
        storage.window(8, 8, DType::F32).unwrap(),
        Shape::from_dims([2]),
        DType::F32,
        cpu(),
    )
    .unwrap();

    assert!(first.shares_storage_with(&second));
    assert!(first.storage().ptr_eq(&storage));
    assert_eq!(first.as_f32_slice().unwrap(), &[0.0, 1.0]);
    assert_eq!(second.as_f32_slice().unwrap(), &[2.0, 3.0]);

    // A second tensor in the same allocation is a live reader: the write path
    // must be refused.
    assert!(matches!(
        first.as_mut_view(),
        Err(Fn3Error::StorageShared { .. })
    ));
    drop(second);
    drop(storage);
    assert!(first.as_mut_view().is_ok());
}

#[test]
fn a_window_holds_a_handle_and_prevents_mutation() {
    let mut storage = Storage::zeroed(16).unwrap();
    assert_eq!(storage.handle_count(), 1);
    let window: StorageWindow = storage.window(0, 16, DType::Bool).unwrap();
    assert_eq!(storage.handle_count(), 2);
    assert!(!storage.is_exclusive());
    assert!(matches!(
        storage.try_make_mut(),
        Err(Fn3Error::StorageShared {
            bytes: 16,
            handles: 2
        })
    ));
    drop(window);
    assert!(storage.try_make_mut().is_ok());
}

// ---------------------------------------------------------------------------
// 3. View semantics and matrix-free materialisation.
// ---------------------------------------------------------------------------

#[test]
fn every_supported_view_materialises_to_its_logical_order() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let cases: Vec<(Vec<u64>, Vec<i64>, usize)> = vec![
        (vec![3, 4], vec![4, 1], 0),  // contiguous
        (vec![4, 3], vec![1, 4], 0),  // transposed
        (vec![2, 3], vec![4, 1], 5),  // sliced, positive strides
        (vec![2, 2], vec![2, 5], 1),  // genuinely non-contiguous
        (vec![2, 2], vec![1, 4], 2),  // slice of a transpose
        (vec![3], vec![1], 9),        // rank-1 tail
        (vec![], vec![], 7),          // rank-0 scalar at an offset
        (vec![0, 3], vec![3, 1], 0),  // zero-sized
        (vec![3, 0], vec![0, 1], 0),  // zero extent with a zero stride
        (vec![0, 3], vec![3, 1], 12), // empty view at the end of storage
    ];
    for (dims, strides, offset) in cases {
        assert_materialises(&dims, &strides, offset, &source);
    }
}

#[test]
fn contiguous_and_rank_one_views_share_instead_of_copying() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let storage = storage_of(&source);

    let contiguous = f32_view(&storage, &[3, 4], &[4, 1], 0);
    assert!(contiguous.is_contiguous());
    let materialised = contiguous.to_contiguous().unwrap();
    assert!(
        materialised.storage().ptr_eq(&storage),
        "no copy for contiguous"
    );

    let tail = f32_view(&storage, &[3], &[1], 9);
    assert!(tail.is_contiguous());
    assert_eq!(tail.as_f32_slice().unwrap(), &[9.0, 10.0, 11.0]);
    assert_eq!(tail.meta().required_bytes(), Ok(48));
    assert!(tail.to_contiguous().unwrap().storage().ptr_eq(&storage));
}

#[test]
fn transposed_and_sliced_views_are_valid_but_not_one_slice() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let storage = storage_of(&source);

    let transposed = f32_view(&storage, &[4, 3], &[1, 4], 0);
    assert!(!transposed.is_contiguous());
    let error = transposed.as_f32_slice().unwrap_err();
    assert!(matches!(error, Fn3Error::NonContiguous { .. }));
    let text = error.to_string();
    assert!(text.contains("contiguous"), "{text}");
    assert!(text.contains("strides[1, 4]"), "{text}");
    assert_eq!(
        transposed.to_contiguous().unwrap().as_f32_slice().unwrap(),
        &[0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );

    let sliced = f32_view(&storage, &[2, 3], &[4, 1], 5);
    assert!(!sliced.is_contiguous());
    assert_eq!(sliced.meta().required_bytes(), Ok(48));
    assert_eq!(
        sliced.to_contiguous().unwrap().as_f32_slice().unwrap(),
        &[5.0, 6.0, 7.0, 9.0, 10.0, 11.0]
    );
}

#[test]
fn a_genuinely_non_contiguous_view_materialises_exactly() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let storage = storage_of(&source);
    let view = f32_view(&storage, &[2, 2], &[2, 5], 1);

    assert!(!view.is_contiguous());
    assert_eq!(view.meta().required_elements(), Ok(9));
    assert_eq!(view.meta().required_bytes(), Ok(36));
    assert_eq!(view.meta().checked_index(&[0, 0]), Ok(1));
    assert_eq!(view.meta().checked_index(&[1, 1]), Ok(8));
    assert_eq!(
        view.to_contiguous().unwrap().as_f32_slice().unwrap(),
        &[1.0, 6.0, 3.0, 8.0]
    );
}

#[test]
fn zero_sized_views_are_contiguous_without_division_or_underflow() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let storage = storage_of(&source);

    // Rank matches and an extent is zero: contiguous by definition, even with
    // strides that would be invalid for a non-empty view.
    let empty_shape = Shape::from_dims([0, 3]);
    assert!(empty_shape.is_contiguous(&Strides::from_elements(vec![100, 7])));
    let empty = f32_view(&storage, &[0, 3], &[100, 7], 0);
    assert!(empty.is_contiguous());
    assert_eq!(empty.numel(), Ok(0));
    assert_eq!(empty.meta().byte_len(), Ok(0));
    assert_eq!(empty.meta().required_bytes(), Ok(0));
    assert_eq!(empty.as_f32_slice().unwrap().len(), 0);
    let materialised = empty.to_contiguous().unwrap();
    assert_eq!(materialised.numel(), Ok(0));
    assert_eq!(materialised.storage().len(), 0);

    // A default-strided zero extent can contain a zero stride and still never
    // reach the overlap or contiguity arithmetic.
    let trailing = Shape::from_dims([3, 0]);
    assert_eq!(trailing.numel(), Ok(0));
    assert_eq!(trailing.default_strides().unwrap().as_slice(), &[0, 1]);
    Tensor::zeros(trailing.clone(), DType::F32, cpu()).unwrap();
    assert!(trailing.is_contiguous(&trailing.default_strides().unwrap()));
}

#[test]
fn scalar_views_have_one_element_and_a_real_byte_length() {
    let source: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let storage = storage_of(&source);
    let scalar = f32_view(&storage, &[], &[], 7);
    assert!(scalar.is_contiguous());
    assert_eq!(scalar.numel(), Ok(1));
    assert_eq!(scalar.meta().byte_len(), Ok(4));
    assert_eq!(scalar.meta().offset(), 7);
    assert_eq!(scalar.as_f32_slice().unwrap(), &[7.0]);
    assert_eq!(
        scalar.to_contiguous().unwrap().as_f32_slice().unwrap(),
        &[7.0]
    );
}

#[test]
fn broadcast_views_materialise_but_stay_read_only() {
    let storage = storage_of(&[1.0, 2.0]);
    let meta = TensorMeta::broadcast(
        Shape::from_dims([2, 2]),
        Strides::from_elements(vec![1, 0]),
        0,
        DType::F32,
        cpu(),
    )
    .unwrap();
    assert!(!meta.is_writable());
    let view = TensorView::new(&storage, meta).unwrap();
    assert!(!view.is_contiguous());
    assert_eq!(
        view.to_contiguous().unwrap().as_f32_slice().unwrap(),
        &[1.0, 1.0, 2.0, 2.0]
    );

    let mut buffer = AlignedBuffer::zeroed(8).unwrap();
    let error = TensorMut::new(&mut buffer, view.meta().clone()).unwrap_err();
    assert!(error.to_string().contains("read-only"), "{error}");
}

#[test]
fn invalid_strides_name_the_axis_and_value() {
    let square = Shape::from_dims([2, 2]);
    let cases: [(Vec<i64>, usize, i64); 4] = [
        (vec![-3, 1], 0, -3),
        (vec![2, -1], 1, -1),
        (vec![2, 0], 1, 0),
        (vec![1, 1], 1, 1),
    ];
    for (strides, axis, value) in cases {
        let error = Strides::from_elements(strides.clone())
            .validate_for(&square)
            .unwrap_err();
        assert!(
            matches!(
                error,
                Fn3Error::InvalidStride {
                    axis: got_axis,
                    stride: got_stride,
                    ..
                } if got_axis == axis && got_stride == value
            ),
            "expected axis {axis} stride {value}, got {error}"
        );
        let text = error.to_string();
        assert!(text.contains(&format!("axis {axis}")), "{text}");
        assert!(text.contains(&value.to_string()), "{text}");

        // The same rejection happens at metadata construction.
        let error = TensorMeta::strided(
            square.clone(),
            Strides::from_elements(strides),
            0,
            DType::F32,
            cpu(),
        )
        .unwrap_err();
        assert!(matches!(error, Fn3Error::InvalidStride { .. }));
    }

    // Zero on an extent-1 axis is still rejected: writable views never carry a
    // zero stride, broadcast or not.
    let singleton = Shape::from_dims([1, 2]);
    assert!(matches!(
        Strides::from_elements(vec![0, 1]).validate_for(&singleton),
        Err(Fn3Error::InvalidStride {
            axis: 0,
            stride: 0,
            ..
        })
    ));
}

// ---------------------------------------------------------------------------
// 4. Aliasing and mutation rules.
// ---------------------------------------------------------------------------

#[test]
fn mutation_requires_exclusive_ownership() {
    let mut storage = Storage::zeroed(32).unwrap();
    assert!(storage.is_exclusive());
    let second = storage.clone();
    assert_eq!(storage.handle_count(), 2);
    let error = storage.try_make_mut().unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::StorageShared {
            bytes: 32,
            handles: 2
        }
    ));
    let text = error.to_string();
    assert!(text.contains("shared"), "{text}");
    assert!(text.contains("2 handles"), "{text}");

    drop(second);
    assert!(storage.is_exclusive());
    storage.try_make_mut().unwrap();

    // `try_into_buffer` obeys the same rule.
    let shared = storage.clone();
    assert!(matches!(
        storage.try_into_buffer(),
        Err(Fn3Error::StorageShared { .. })
    ));
    drop(shared);
    assert!(Storage::zeroed(8).unwrap().try_into_buffer().is_ok());
}

#[test]
fn tensor_mutation_follows_storage_exclusivity() {
    let mut tensor = Tensor::from_f32(Shape::from_dims([2, 2]), &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let reader = tensor.storage().clone();
    assert!(matches!(
        tensor.as_mut_view(),
        Err(Fn3Error::StorageShared { .. })
    ));
    drop(reader);

    {
        let mut view = tensor.as_mut_view().unwrap();
        view.as_mut_slice::<f32>().unwrap()[0] = 9.0;
        assert_eq!(view.as_slice::<f32>().unwrap(), &[9.0, 2.0, 3.0, 4.0]);
    }
    assert_eq!(tensor.as_f32_slice().unwrap(), &[9.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_reader_in_the_same_allocation_forbids_every_write_path() {
    let mut source = Tensor::from_f32(Shape::from_dims([3]), &[1.0, 2.0, 3.0]).unwrap();
    let mut alias = Tensor::from_storage(source.storage().clone(), source.meta().clone()).unwrap();
    assert!(source.shares_storage_with(&alias));

    // This is the exact situation a backend must never write into: an output
    // window that overlaps a live input. The safe API refuses to hand out the
    // mutable window while either tensor (or a view of either) is alive.
    assert!(matches!(
        source.as_mut_view(),
        Err(Fn3Error::StorageShared { handles: 2, .. })
    ));
    assert!(matches!(
        alias.as_mut_view(),
        Err(Fn3Error::StorageShared { .. })
    ));

    drop(alias);
    assert!(source.as_mut_view().is_ok());
}

#[test]
fn distinct_allocations_with_equal_bytes_are_not_aliases() {
    let left = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    let right = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    assert_eq!(left.as_f32_slice().unwrap(), right.as_f32_slice().unwrap());
    assert!(!left.shares_storage_with(&right));
    assert!(!left.storage().ptr_eq(right.storage()));
    assert!(left.storage().ptr_eq(left.storage()));
}

// ---------------------------------------------------------------------------
// 5. Storage hardening: alignment, zero length, checked casts.
// ---------------------------------------------------------------------------

#[test]
fn every_allocation_is_64_byte_aligned_and_zeroed() {
    for len in [0usize, 1, 3, 63, 64, 65, 127, 128, 1000, 4096] {
        let buffer = AlignedBuffer::zeroed(len).unwrap();
        assert_eq!(buffer.as_ptr() as usize % 64, 0, "len {len}");
        assert_eq!(buffer.len(), len);
        assert!(buffer.as_bytes().iter().all(|&byte| byte == 0));
        assert_eq!(buffer.is_empty(), len == 0);
    }
    // Typed views start on their own alignment as well.
    let buffer = AlignedBuffer::zeroed(64).unwrap();
    assert_eq!(buffer.as_slice::<f64>().unwrap().as_ptr() as usize % 8, 0);
}

#[test]
fn zero_length_buffers_own_no_allocation_and_stay_sound() {
    let buffer = AlignedBuffer::zeroed(0).unwrap();
    assert!(buffer.is_empty());
    assert_eq!(buffer.as_ptr() as usize % 64, 0);
    assert!(buffer.as_bytes().is_empty());
    assert_eq!(buffer.as_slice::<f32>().unwrap().len(), 0);
    assert_eq!(AlignedBuffer::from_bytes(&[]).unwrap().len(), 0);
    assert_eq!(Storage::zeroed(0).unwrap().len(), 0);
    assert_eq!(Storage::from_bytes(&[]).unwrap().as_bytes(), &[] as &[u8]);
    // Dropping them must not deallocate anything; reaching this point without a
    // crash is the assertion.
}

#[test]
fn checked_casts_reject_partial_elements_and_zero_sized_types() {
    let six = AlignedBuffer::from_bytes(&[0u8; 6]).unwrap();
    let error = six.as_slice::<f32>().unwrap_err();
    assert!(matches!(error, Fn3Error::StorageCast { .. }));
    let text = error.to_string();
    assert!(text.contains("6 bytes"), "{text}");
    assert!(text.contains("f32"), "{text}");

    let mut seven = AlignedBuffer::from_bytes(&[0u8; 7]).unwrap();
    assert!(seven.as_slice::<u32>().is_err());
    assert!(seven.as_mut_slice::<u32>().is_err());

    let zero_sized = six.as_slice::<()>().unwrap_err();
    assert!(
        zero_sized.to_string().contains("zero-sized"),
        "{zero_sized}"
    );

    let eight = AlignedBuffer::from_bytes(&[0u8; 8]).unwrap();
    assert_eq!(eight.as_slice::<f32>().unwrap().len(), 2);
    assert_eq!(eight.as_slice::<u8>().unwrap().len(), 8);
    assert_eq!(eight.as_slice::<i64>().unwrap().len(), 1);
}

#[test]
fn checked_casts_validate_alignment_not_just_length() {
    // The base allocation is 64-byte aligned, so a 16-byte type is fine.
    let buffer = AlignedBuffer::zeroed(16).unwrap();
    assert_eq!(buffer.as_slice::<Align16>().unwrap().len(), 1);

    // A window at offset 8 is deliberately not 16-byte aligned: the cast must
    // fail even though the length is a whole number of elements.
    let storage = Storage::zeroed(32).unwrap();
    let window = storage.window(8, 16, DType::Bool).unwrap();
    let error = window.as_slice::<Align16>().unwrap_err();
    assert!(matches!(error, Fn3Error::StorageCast { .. }));
    assert!(error.to_string().contains("aligned"), "{error}");
}

#[test]
fn aligned_buffer_clone_is_a_distinct_copy() {
    let buffer = AlignedBuffer::from_bytes(&[1, 2, 3, 4]).unwrap();
    let copy = buffer.clone();
    assert_ne!(buffer.as_ptr(), copy.as_ptr());
    assert_eq!(buffer.as_bytes(), copy.as_bytes());
    assert_eq!(copy.as_ptr() as usize % 64, 0);
}

#[test]
fn alignment_hint_matches_the_element_size_for_the_alpha_dtypes() {
    for dtype in DType::ALL {
        assert_eq!(dtype.alignment_hint(), dtype.size_bytes());
        assert!(dtype.alignment_hint() > 0);
    }
}

// ---------------------------------------------------------------------------
// 6. Invalid devices: allocation and execution refuse, never substitute.
// ---------------------------------------------------------------------------

#[test]
fn allocation_on_a_device_without_a_backend_is_refused_for_every_dtype() {
    for dtype in DType::ALL {
        let error = Tensor::zeros(Shape::from_dims([4]), dtype, vulkan()).unwrap_err();
        assert!(matches!(error, Fn3Error::DeviceUnavailable(_)), "{error}");
        let text = error.to_string();
        assert!(text.contains("device unavailable"), "{text}");
        assert!(text.contains("vulkan(0)"), "{text}");
        assert!(text.contains("available devices: [cpu]"), "{text}");
    }
}

#[test]
fn a_device_with_no_backend_is_never_listed_as_available() {
    let available = available_devices();
    assert!(available.iter().any(|device| device.id.is_cpu()));
    assert!(available.iter().all(|device| device.id.is_cpu()));
    let error = DeviceRequest::Vulkan { index: Some(0) }
        .resolve_on_host()
        .unwrap_err();
    assert_eq!(error.requested(), "vulkan(index=0)");
    assert_eq!(error.available(), ["cpu"]);
    assert!(error.to_string().contains("available devices: [cpu]"));
}

#[test]
fn a_vulkan_tagged_host_tensor_is_metadata_only_and_cannot_execute() {
    // Metadata construction is allowed: plans are validated before a Vulkan
    // backend exists.
    let meta = TensorMeta::contiguous(Shape::from_dims([2]), DType::F32, vulkan()).unwrap();
    let host = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    let tagged = Tensor::from_storage(host.storage().clone(), meta).unwrap();
    assert_eq!(tagged.device(), vulkan());

    let mut out = Tensor::zeros(Shape::from_dims([2]), DType::F32, cpu()).unwrap();
    let lhs = tagged.as_view().unwrap();
    let rhs = host.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    let error = CpuBackend::new()
        .execute_op(
            OpId::Add,
            &[lhs, rhs],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap_err();
    assert!(matches!(error, Fn3Error::DeviceMismatch { .. }), "{error}");
    let text = error.to_string();
    assert!(text.contains("op add"), "{text}");
    assert!(text.contains("expected device cpu"), "{text}");
    assert!(text.contains("found vulkan(0)"), "{text}");
}

#[test]
fn materialising_a_strided_vulkan_view_allocates_nowhere_and_says_so() {
    let host = Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0; 6]).unwrap();
    let meta = TensorMeta::strided(
        Shape::from_dims([3, 2]),
        Strides::from_elements(vec![1, 3]),
        0,
        DType::F32,
        vulkan(),
    )
    .unwrap();
    let view = TensorView::new(host.storage(), meta).unwrap();
    let error = view.to_contiguous().unwrap_err();
    assert!(matches!(error, Fn3Error::DeviceUnavailable(_)), "{error}");
    assert!(error.to_string().contains("vulkan(0)"), "{error}");
}

// ---------------------------------------------------------------------------
// 7. Interop edges: scalar byte lengths, empty iteration, empty contiguity.
// ---------------------------------------------------------------------------

#[test]
fn scalar_byte_lengths_match_the_dtype_size() {
    assert_eq!(Shape::scalar().byte_len(DType::F32), Ok(4));
    assert_eq!(Shape::scalar().byte_len(DType::I32), Ok(4));
    assert_eq!(Shape::scalar().byte_len(DType::I64), Ok(8));
    assert_eq!(Shape::scalar().byte_len(DType::Bool), Ok(1));

    let f32_scalar = Tensor::from_f32(Shape::scalar(), &[2.5]).unwrap();
    assert_eq!(f32_scalar.storage().len(), 4);
    assert_eq!(f32_scalar.numel(), Ok(1));
    assert_eq!(f32_scalar.as_f32_slice().unwrap(), &[2.5]);

    let i64_scalar = Tensor::from_i64(Shape::scalar(), &[-7]).unwrap();
    assert_eq!(i64_scalar.storage().len(), 8);
    assert_eq!(i64_scalar.as_i64_slice().unwrap(), &[-7]);

    let mask_scalar = Tensor::from_mask(Shape::scalar(), &[true]).unwrap();
    assert_eq!(mask_scalar.storage().len(), 1);
    assert_eq!(mask_scalar.as_slice::<u8>().unwrap(), &[1]);
    assert_eq!(
        mask_scalar.to_contiguous().unwrap().shape(),
        &Shape::scalar()
    );
}

#[test]
fn empty_tensors_report_zero_everywhere_and_iterate_zero_times() {
    for dims in [
        vec![0u64],
        vec![0, 3],
        vec![3, 0],
        vec![2, 0, 4],
        vec![u64::MAX, 2, 0],
    ] {
        let empty = shape(&dims);
        assert_eq!(empty.numel(), Ok(0), "numel for {empty}");
        assert_eq!(empty.numel_usize(), Ok(0), "numel_usize for {empty}");
        assert_eq!(empty.rank(), dims.len());
        assert!(empty.is_contiguous(&empty.default_strides().unwrap()));

        for dtype in DType::ALL {
            assert_eq!(empty.byte_len(dtype), Ok(0), "{dtype} byte_len for {empty}");
            let tensor = Tensor::zeros(empty.clone(), dtype, cpu()).unwrap();
            assert_eq!(tensor.numel(), Ok(0));
            assert_eq!(tensor.storage().len(), 0);
            assert_eq!(tensor.as_view().unwrap().numel(), Ok(0));
            let materialised = tensor.to_contiguous().unwrap();
            assert_eq!(materialised.numel(), Ok(0));
            assert!(materialised.is_contiguous());
        }
    }

    // A zero-element tensor is contiguous for any rank-matching strides, so
    // contiguity never subtracts one from a zero extent or divides by it.
    let empty = Shape::from_dims([0, 3]);
    assert!(empty.is_contiguous(&Strides::from_elements(vec![i64::MAX, 1])));
    assert_eq!(Shape::from_dims([0, 3, u64::MAX]).numel(), Ok(0));
}

/// An axis of extent 1 addresses only coordinate 0, so it cannot overlap any
/// other axis. Regression: `check_no_overlap` used to multiply the covered
/// element count by every axis extent, including extent-1 axes, which made a
/// contiguous shape look overlapping whenever a preceding equal-stride axis
/// existed. `[2, 1, 4, 4]` (the canonical `[batch, 1, seq, seq]` attention mask)
/// has default strides `[16, 16, 4, 1]`, and axis 1 was rejected against a
/// coverage of 32 contributed by axis 0. `[N, 1]`, `[128, 1]`, `[1, 3, 1]` and
/// `[6, 1]` failed the same way, so this blocked every singleton axis.
#[test]
fn extent_one_axes_never_count_as_overlap() {
    for dims in [
        vec![2, 1, 4, 4],
        vec![1, 3, 1],
        vec![6, 1],
        vec![128, 1],
        vec![1],
        vec![1, 1, 1],
        vec![4, 1, 4, 1, 4],
        vec![2, 2, 1],
    ] {
        let shape = Shape::from_dims(dims.clone());
        let strides = shape.default_strides().unwrap();
        strides.validate_for(&shape).unwrap_or_else(|error| {
            panic!("default strides {strides} of contiguous {shape} were rejected: {error}")
        });
        assert!(
            shape.is_contiguous(&strides),
            "{shape} should be contiguous with its default strides"
        );
        // The same shape must also be constructible, which is what downstream
        // planner and attention code actually does.
        Tensor::zeros(shape.clone(), DType::F32, cpu())
            .unwrap_or_else(|error| panic!("could not allocate {shape}: {error}"));
    }

    // Ignoring extent-1 axes must not weaken real overlap detection: two axes
    // that genuinely address the same elements are still rejected, with and
    // without a singleton axis present.
    for (dims, bad) in [
        (vec![2, 2], vec![1, 1]),
        (vec![2, 1, 2], vec![1, 1, 1]),
        (vec![3, 1, 3], vec![1, 0, 1]),
        (vec![2, 4], vec![2, 1]),
    ] {
        let shape = Shape::from_dims(dims);
        let error = Strides::from_elements(bad.clone())
            .validate_for(&shape)
            .unwrap_err();
        assert!(
            matches!(error, Fn3Error::InvalidStride { .. }),
            "strides {bad:?} of {shape} must still be rejected, got {error}"
        );
    }
}
