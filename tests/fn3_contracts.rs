//! Contract tests for the FastNN 3.0 inference core (`src/fn3/`).
//!
//! These tests pin the behaviour downstream workers code against. They are
//! deliberately integration tests over the public `fastnn::inference` surface: if a
//! contract test needs to reach into a private item, the contract is missing.

use fastnn::inference::backend::{check_arity, Support};
use fastnn::inference::device::{available_devices, PhysicalDeviceId};
use fastnn::inference::model::{KvCacheConfig, KvCacheLayout, SessionPhase};
use fastnn::inference::shape::Symbol;
use fastnn::inference::{
    AlignedBuffer, Backend, Capabilities, CpuBackend, DType, DeviceId, DeviceRequest, DimBound,
    ExecutionPlan, Fn3Error, Instruction, LiveDims, ModelWeights, OpId, OpParams, PreparedModel,
    Session, Shape, ShapeEnvelope, Storage, Strides, Tensor, TensorMeta, TensorMut,
};

fn cpu() -> DeviceId {
    DeviceId::Cpu
}

fn vulkan() -> DeviceId {
    DeviceId::Vulkan(PhysicalDeviceId::new(0))
}

fn f32_shape(dims: [u64; 2]) -> Shape {
    Shape::from_dims(dims)
}

fn encoded_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn small_envelope() -> ShapeEnvelope {
    ShapeEnvelope::uniform(16).expect("16 is within DimBound::MAX_EXTENT")
}

// ---------------------------------------------------------------------------
// 1. Checked arithmetic: overflow is reported, never wrapped.
// ---------------------------------------------------------------------------

#[test]
fn shape_element_count_overflow_is_rejected() {
    let error = Shape::from_dims([u64::MAX, 2]).numel().unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::ShapeOverflow {
            context: "Shape::numel",
            axis: Some(_),
            ..
        }
    ));
    let text = error.to_string();
    assert!(text.contains("overflow"), "{text}");
    assert!(text.contains("18446744073709551615"), "{text}");
}

#[test]
fn shape_byte_length_overflow_is_rejected() {
    let error = Shape::from_dims([u64::MAX])
        .byte_len(DType::F32)
        .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::ShapeOverflow {
            context: "Shape::byte_len",
            ..
        }
    ));
}

#[test]
fn stride_span_overflow_is_rejected() {
    let shape = Shape::from_dims([u64::MAX]);
    let strides = Strides::from_elements(vec![3]);
    let error = strides.required_len(&shape).unwrap_err();
    assert!(matches!(error, Fn3Error::ShapeOverflow { .. }));
}

#[test]
fn zero_extent_short_circuits_multiplication() {
    // Would overflow if multiplied left to right; must report zero elements.
    let shape = Shape::from_dims([u64::MAX, 2, 0]);
    assert_eq!(shape.numel(), Ok(0));
    assert_eq!(shape.byte_len(DType::I64), Ok(0));
    assert_eq!(shape.default_strides().unwrap().required_len(&shape), Ok(0));
}

// ---------------------------------------------------------------------------
// 2. Scalar (rank 0) and zero-size shapes.
// ---------------------------------------------------------------------------

#[test]
fn rank_zero_is_a_one_element_scalar() {
    let scalar = Shape::scalar();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.numel(), Ok(1));
    assert_eq!(scalar.byte_len(DType::F32), Ok(4));
    assert_eq!(scalar.default_strides().unwrap().rank(), 0);
    assert_eq!(
        scalar.default_strides().unwrap().required_len(&scalar),
        Ok(1)
    );
    assert_eq!(scalar.to_string(), "[]");
}

#[test]
fn zero_size_tensors_are_legal_and_execute() {
    let shape = Shape::from_dims([0, 3]);
    assert_eq!(shape.numel(), Ok(0));

    let lhs = Tensor::zeros(shape.clone(), DType::F32, cpu()).unwrap();
    let rhs = Tensor::from_f32(Shape::from_dims([3]), &[1.0, 2.0, 3.0]).unwrap();
    let mut out = Tensor::zeros(shape.clone(), DType::F32, cpu()).unwrap();
    assert_eq!(out.storage().len(), 0);

    let lhs_view = lhs.as_view().unwrap();
    let rhs_view = rhs.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    CpuBackend::new()
        .execute_op(
            OpId::Add,
            &[lhs_view, rhs_view],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .expect("adding an empty tensor is a no-op, not an error");
    assert_eq!(out.numel(), Ok(0));
}

// ---------------------------------------------------------------------------
// 3. Alpha view rules: invalid strides are rejected with InvalidStride.
// ---------------------------------------------------------------------------

#[test]
fn negative_strides_are_rejected() {
    let shape = f32_shape([2, 2]);
    let error = Strides::from_elements(vec![-2, 1])
        .validate_for(&shape)
        .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::InvalidStride {
            axis: 0,
            stride: -2,
            ..
        }
    ));
    assert!(error.to_string().contains("negative"), "{error}");
}

#[test]
fn zero_strides_are_rejected_unless_broadcast_declared() {
    let shape = f32_shape([2, 2]);
    let strides = Strides::from_elements(vec![2, 0]);
    let error = strides.validate_for(&shape).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::InvalidStride {
            axis: 1,
            stride: 0,
            ..
        }
    ));
    assert!(error.to_string().contains("broadcast"), "{error}");
    strides
        .validate_broadcast_for(&shape)
        .expect("broadcast-declared views may use zero strides");
}

#[test]
fn overlapping_strides_are_rejected() {
    let shape = f32_shape([2, 2]);
    let error = Strides::from_elements(vec![1, 1])
        .validate_for(&shape)
        .unwrap_err();
    assert!(matches!(error, Fn3Error::InvalidStride { axis: 1, .. }));
    assert!(error.to_string().contains("overlap"), "{error}");
}

#[test]
fn tensor_meta_rejects_invalid_strides_at_construction() {
    let error = TensorMeta::strided(
        f32_shape([2, 2]),
        Strides::from_elements(vec![1, 1]),
        0,
        DType::F32,
        cpu(),
    )
    .unwrap_err();
    assert!(matches!(error, Fn3Error::InvalidStride { .. }));
}

#[test]
fn broadcast_views_are_read_only() {
    let meta = TensorMeta::broadcast(
        f32_shape([2, 2]),
        Strides::from_elements(vec![2, 0]),
        0,
        DType::F32,
        cpu(),
    )
    .unwrap();
    assert!(meta.is_broadcast());
    assert!(!meta.is_writable());

    let mut buffer = AlignedBuffer::zeroed(16).unwrap();
    let error = TensorMut::new(&mut buffer, meta).unwrap_err();
    assert!(error.to_string().contains("read-only"), "{error}");
}

#[test]
fn transpose_is_a_valid_non_contiguous_view() {
    let shape = f32_shape([2, 3]);
    let strides = shape.default_strides().unwrap();
    let transposed_shape = shape.permuted(&[1, 0]).unwrap();
    let transposed_strides = strides.permuted(&[1, 0]).unwrap();
    assert_eq!(transposed_shape, f32_shape([3, 2]));
    assert_eq!(transposed_strides.as_slice(), &[1, 3]);
    assert!(!transposed_shape.is_contiguous(&transposed_strides));
    assert_eq!(transposed_strides.required_len(&transposed_shape), Ok(6));
    transposed_strides.validate_for(&transposed_shape).unwrap();
    assert!(shape.is_permutation_of(&transposed_shape));
}

// ---------------------------------------------------------------------------
// 4. Device affinity: mismatch is reported by the executing backend.
// ---------------------------------------------------------------------------

fn cpu_tensor_tagged(device: DeviceId, values: &[f32]) -> Tensor {
    let host = Tensor::from_f32(Shape::from_dims([values.len() as u64]), values).unwrap();
    let meta = TensorMeta::contiguous(Shape::from_dims([values.len() as u64]), DType::F32, device)
        .unwrap();
    Tensor::from_storage(host.storage().clone(), meta).unwrap()
}

#[test]
fn device_mismatch_is_rejected_by_the_cpu_backend() {
    let lhs = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    let rhs = cpu_tensor_tagged(vulkan(), &[1.0, 2.0]);
    let mut out = Tensor::zeros(Shape::from_dims([2]), DType::F32, cpu()).unwrap();

    let lhs_view = lhs.as_view().unwrap();
    let rhs_view = rhs.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    let error = CpuBackend::new()
        .execute_op(
            OpId::Add,
            &[lhs_view, rhs_view],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap_err();

    assert!(matches!(error, Fn3Error::DeviceMismatch { .. }));
    assert_eq!(error.op(), Some(OpId::Add));
    let text = error.to_string();
    assert!(text.contains("op add"), "{text}");
    assert!(text.contains("expected device cpu"), "{text}");
    assert!(text.contains("found vulkan(0)"), "{text}");
}

#[test]
fn plan_execution_rejects_a_foreign_device_operand() {
    let lhs = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    let rhs = cpu_tensor_tagged(vulkan(), &[1.0, 2.0]);
    let out = Tensor::zeros(Shape::from_dims([2]), DType::F32, cpu()).unwrap();
    let plan =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
    let mut slots = vec![Some(lhs), Some(rhs), Some(out)];
    let error = plan.execute(&CpuBackend::new(), &mut slots).unwrap_err();
    assert!(matches!(error, Fn3Error::DeviceMismatch { .. }));
    assert!(
        slots[2].is_some(),
        "a failed instruction must not consume the output slot"
    );
}

#[test]
fn mismatched_devices_are_detected_between_operand_metas() {
    let cpu_meta = TensorMeta::contiguous(Shape::scalar(), DType::F32, cpu()).unwrap();
    let gpu_meta = TensorMeta::contiguous(Shape::scalar(), DType::F32, vulkan()).unwrap();
    let error = cpu_meta
        .check_same_device(OpId::MatMul, &gpu_meta)
        .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DeviceMismatch {
            op: Some(OpId::MatMul),
            expected: DeviceId::Cpu,
            actual: DeviceId::Vulkan(_)
        }
    ));
}

// ---------------------------------------------------------------------------
// 5. Explicit device requests fail loudly with requested + available text.
// ---------------------------------------------------------------------------

#[test]
fn explicit_vulkan_request_names_the_request_and_available_devices() {
    let error = DeviceRequest::Vulkan { index: Some(3) }
        .resolve_on_host()
        .expect_err("index 3 does not exist on this host");
    assert_eq!(error.requested(), "vulkan(index=3)");
    let available = error.available();
    // The CPU is always first, and the error always lists what this build can
    // execute on rather than collapsing to the CPU.
    assert_eq!(available.first().map(String::as_str), Some("cpu"));
    let text = error.to_string();
    assert!(text.contains("vulkan(index=3)"), "{text}");
    assert!(text.contains("available devices:"), "{text}");

    #[cfg(feature = "vulkan")]
    assert!(
        available.iter().any(|label| label.starts_with("vulkan(")),
        "a Vulkan-enabled build must list its adapters: {available:?}"
    );

    #[cfg(not(feature = "vulkan"))]
    {
        assert_eq!(available, ["cpu"]);
        assert!(text.contains("available devices: [cpu]"), "{text}");
        assert!(text.contains("no Vulkan backend"), "{text}");
    }
}

#[test]
fn named_vulkan_request_lists_available_adapters() {
    let error = DeviceRequest::VulkanNamed {
        name: "NVIDIA RTX 4090".to_string(),
    }
    .resolve_on_host()
    .expect_err("this host has no RTX 4090");
    let text = error.to_string();
    assert!(text.contains("NVIDIA RTX 4090"), "{text}");
    assert!(text.contains("available devices:"), "{text}");

    #[cfg(not(feature = "vulkan"))]
    assert!(text.contains("available devices: [cpu]"), "{text}");

    #[cfg(feature = "vulkan")]
    assert!(
        error
            .available()
            .iter()
            .any(|label| label.contains("llvmpipe") || label.contains("NVIDIA")),
        "a Vulkan-enabled build must list its adapters: {:?}",
        error.available()
    );
}

#[test]
fn explicit_device_requests_never_resolve_to_cpu() {
    let available = available_devices();
    for request in [
        DeviceRequest::Vulkan { index: None },
        DeviceRequest::Vulkan { index: Some(0) },
        DeviceRequest::Vulkan { index: Some(9999) },
        DeviceRequest::VulkanNamed {
            name: "anything".to_string(),
        },
    ] {
        assert!(request.is_explicit());
        let resolved = request.resolve(&available);
        // The one invariant that holds in every configuration: an explicit
        // request either reaches that device or fails. It never becomes CPU.
        assert_ne!(
            resolved,
            Ok(DeviceId::Cpu),
            "{request} must not fall back to the CPU"
        );

        // A request naming something this host does not have must be an error,
        // whichever build this is and whichever backends are enumerated.
        let must_fail = matches!(&request, DeviceRequest::VulkanNamed { .. })
            || matches!(&request, DeviceRequest::Vulkan { index: Some(9999) });
        if must_fail {
            assert!(resolved.is_err(), "{request} must not be substituted");
        }
    }
}

#[test]
fn auto_may_choose_the_cpu() {
    assert_eq!(DeviceRequest::Auto.resolve_on_host(), Ok(DeviceId::Cpu));
}

/// FN3-27 (#73): a build with no Vulkan backend refuses an explicit request at
/// runtime, by name, and never substitutes the CPU.
///
/// This test is the `backend-vulkan`-absent lane: it is compiled in the default
/// and `--no-default-features` configurations and asserts the exact refusal.
#[test]
fn a_build_without_the_vulkan_backend_refuses_an_explicit_request() {
    let available = available_devices();
    assert_eq!(
        available.first().map(|info| info.id),
        Some(DeviceId::Cpu),
        "the CPU is always enumerated first"
    );
    #[cfg(not(feature = "vulkan"))]
    {
        assert!(
            available.iter().all(|info| info.id.is_cpu()),
            "a build without the feature must enumerate no Vulkan adapter: {available:?}"
        );
        let error = DeviceRequest::Vulkan { index: None }
            .resolve(&available)
            .expect_err("an explicit Vulkan request must fail without the backend");
        let text = error.to_string();
        assert!(text.contains("vulkan"), "{text}");
        assert!(text.contains("no Vulkan backend"), "{text}");
        assert!(text.contains("feature is not enabled"), "{text}");
    }
    #[cfg(feature = "vulkan")]
    {
        // The absent-feature proof is compiled out; an unusable adapter in a
        // Vulkan-enabled build is covered by fn3_vulkan_smoke and fn3_e2e.
        assert!(
            available
                .iter()
                .any(|info| info.id.vulkan_index().is_some()),
            "a Vulkan-enabled build must enumerate its adapters: {available:?}"
        );
    }
    // The invariant that holds in both configurations.
    assert_ne!(
        DeviceRequest::Vulkan { index: None }.resolve(&available),
        Ok(DeviceId::Cpu)
    );
}

#[test]
fn allocation_on_a_missing_device_is_not_substituted() {
    let error = Tensor::zeros(Shape::from_dims([4]), DType::F32, vulkan()).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("device unavailable"), "{text}");
    assert!(text.contains("vulkan(0)"), "{text}");
    assert!(text.contains("available devices:"), "{text}");
    // Alpha tensor storage is host memory on every device, so an explicit
    // Vulkan allocation fails in both configurations. In a Vulkan-enabled
    // build the device is listed but still cannot own a tensor, and the reason
    // must say so instead of claiming the backend is missing.
    #[cfg(not(feature = "vulkan"))]
    assert!(text.contains("available devices: [cpu]"), "{text}");

    #[cfg(feature = "vulkan")]
    {
        assert!(text.contains("host storage only"), "{text}");
        assert!(
            text.contains("only the cpu device can own a tensor"),
            "{text}"
        );
        assert!(text.contains("device-resident storage"), "{text}");
        assert!(!text.contains("FN3-12"), "{text}");
        assert!(!text.contains("no backend for this device"), "{text}");
    }
}

// ---------------------------------------------------------------------------
// 6. OpId stability.
// ---------------------------------------------------------------------------

#[test]
fn op_ids_are_stable_and_total() {
    // Pinned discriminants: these are written into plans and capability tables.
    assert_eq!(OpId::Gather.code(), 1);
    assert_eq!(OpId::MatMul.code(), 2);
    assert_eq!(OpId::Add.code(), 3);
    assert_eq!(OpId::LayerNorm.code(), 15);
    assert_eq!(OpId::MaskedAttention.code(), 40);
    assert_eq!(OpId::Conv.code(), 41);
    assert_eq!(OpId::MaxPool.code(), 42);
    assert_eq!(OpId::AveragePool.code(), 43);
    assert_eq!(OpId::GlobalAveragePool.code(), 44);
    assert_eq!(OpId::BatchNorm.code(), 45);
    assert_eq!(OpId::Resize.code(), 46);
    assert_eq!(OpId::Pad.code(), 47);
    assert_eq!(OpId::Clip.code(), 48);
    assert_eq!(OpId::Sigmoid.code(), 49);
    assert_eq!(OpId::Maximum.code(), 50);
    assert_eq!(OpId::Minimum.code(), 51);
    assert_eq!(OpId::TopK.code(), 52);
    assert_eq!(OpId::ALL.len(), 52);

    let mut names = std::collections::HashSet::new();
    for op in OpId::ALL {
        assert_eq!(OpId::from_u16(op.code()), Some(op), "round trip for {op}");
        assert!(names.insert(op.name()), "duplicate op name {}", op.name());
        assert_eq!(op.to_string(), op.name());
        assert!(!op.metadata().description.is_empty());
    }
    assert_eq!(OpId::from_u16(0), None);
    assert_eq!(OpId::from_u16(53), None);
    assert!(OpId::is_reserved(*OpId::QUANTIZED_RANGE.start()));
    assert!(OpId::is_reserved(*OpId::STRUCTURAL_RANGE.end()));
    assert_eq!(OpId::from_u16(*OpId::QUANTIZED_RANGE.start()), None);
}

#[test]
fn gpt2_op_set_is_present() {
    let required = [
        OpId::Gather,
        OpId::MatMul,
        OpId::Add,
        OpId::Sub,
        OpId::Mul,
        OpId::Div,
        OpId::Neg,
        OpId::Pow,
        OpId::Sqrt,
        OpId::Erf,
        OpId::Tanh,
        OpId::Gelu,
        OpId::Relu,
        OpId::Softmax,
        OpId::LayerNorm,
        OpId::ReduceMean,
        OpId::ReduceSum,
        OpId::ArgMax,
        OpId::Transpose,
        OpId::Reshape,
        OpId::Flatten,
        OpId::Squeeze,
        OpId::Unsqueeze,
        OpId::Concat,
        OpId::Slice,
        OpId::Split,
        OpId::Shape,
        OpId::ConstantOfShape,
        OpId::Expand,
        OpId::Range,
        OpId::Cast,
        OpId::Equal,
        OpId::Less,
        OpId::Greater,
        OpId::Where,
        OpId::Trilu,
        OpId::Sin,
        OpId::Cos,
        OpId::ScatterND,
        OpId::MaskedAttention,
    ];
    let present: std::collections::HashSet<OpId> = OpId::ALL.into_iter().collect();
    for op in required {
        assert!(present.contains(&op), "missing op {op}");
    }
}

// ---------------------------------------------------------------------------
// 7. No silent fallback: unsupported requests are reported.
// ---------------------------------------------------------------------------

#[test]
fn unsupported_ops_are_reported_not_approximated() {
    let backend = CpuBackend::new();
    // `MaskedAttention` is a real, reserved id (40) that this backend genuinely
    // does not implement: it is the fused attention op owned by the Vulkan
    // attention worker (#61), and the CPU reference proves the same arithmetic
    // by decomposition. It is therefore the canonical honest-`Unsupported` case
    // now that the rest of the op matrix is implemented.
    let support = backend.supports(OpId::MaskedAttention);
    assert!(matches!(support, Support::Unsupported { .. }));
    assert!(!support.is_supported());
    assert!(support.reason().unwrap().contains("#61"));

    let input = Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0; 6]).unwrap();
    let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, cpu()).unwrap();
    let view = input.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    let error = backend
        .execute_op(
            OpId::MaskedAttention,
            &[view],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::UnsupportedOp {
            op: OpId::MaskedAttention,
            backend: "cpu",
            ..
        }
    ));
    let text = error.to_string();
    assert!(text.contains("masked_attention"), "{text}");
    assert!(text.contains("backend cpu"), "{text}");
    drop(out_view);

    // The classic 2.x bug: return the first input unchanged. The output must
    // still be the zero buffer the caller allocated.
    assert_eq!(out.as_f32_slice().unwrap(), &[0.0; 6]);
}

#[test]
fn partial_support_is_not_full_support() {
    let partial = Support::partial("rank-2 only");
    assert!(partial.is_supported());
    assert!(!partial.is_full());
    partial.require_supported(OpId::MatMul, "test").unwrap();
    let error = partial.require_full(OpId::MatMul, "test").unwrap_err();
    assert!(
        error.to_string().contains("only partially supported"),
        "{error}"
    );
}

#[test]
fn plan_preflight_rejects_unsupported_instructions() {
    let plan = ExecutionPlan::new(
        vec![
            Instruction::new(OpId::Add, vec![0, 1], vec![2]),
            Instruction::new(OpId::MaskedAttention, vec![2], vec![3]),
        ],
        4,
    )
    .unwrap();
    let error = plan.check_supported(&CpuBackend::new()).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::UnsupportedOp {
            op: OpId::MaskedAttention,
            ..
        }
    ));
    assert!(error.to_string().contains("at instruction 1"), "{error}");
}

#[test]
fn arity_and_attribute_errors_are_reported() {
    let backend = CpuBackend::new();
    let input = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
    let mut out = Tensor::zeros(Shape::from_dims([2]), DType::F32, cpu()).unwrap();
    let view = input.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    let error = backend
        .execute_op(
            OpId::Add,
            std::slice::from_ref(&view),
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap_err();
    assert!(matches!(error, Fn3Error::ArityMismatch { .. }));
    assert_eq!(check_arity(OpId::Add, 2, 1, 2, 1), Ok(()));

    let mut out_view = out.as_mut_view().unwrap();
    let error = backend
        .execute_op(
            OpId::Add,
            &[view.clone(), view],
            std::slice::from_mut(&mut out_view),
            &OpParams::default().with_axes([0]),
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("accepts no attributes"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// 8. Two working CPU ops through the Backend/plan path.
// ---------------------------------------------------------------------------

#[test]
fn cpu_add_broadcasts_through_the_backend() {
    let lhs = Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Tensor::from_f32(Shape::from_dims([3]), &[10.0, 20.0, 30.0]).unwrap();
    let mut out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, cpu()).unwrap();

    let lhs_view = lhs.as_view().unwrap();
    let rhs_view = rhs.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    CpuBackend::new()
        .execute_op(
            OpId::Add,
            &[lhs_view, rhs_view],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap();
    assert_eq!(
        out.as_f32_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn cpu_matmul_is_rank_two_and_honours_strides() {
    let lhs = Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_f32(Shape::from_dims([3, 2]), &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let mut out = Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, cpu()).unwrap();
    let lhs_view = lhs.as_view().unwrap();
    let rhs_view = rhs.as_view().unwrap();
    let mut out_view = out.as_mut_view().unwrap();
    CpuBackend::new()
        .execute_op(
            OpId::MatMul,
            &[lhs_view, rhs_view.clone()],
            std::slice::from_mut(&mut out_view),
            &OpParams::default(),
        )
        .unwrap();
    assert_eq!(out.as_f32_slice().unwrap(), &[58.0, 64.0, 139.0, 154.0]);

    // Batched matmul is now implemented (it is required by #53's acceptance
    // criteria), so the assertion is that the batch is actually contracted
    // rather than silently reduced to the first matrix.
    let batched = Tensor::from_f32(
        Shape::from_dims([2, 2, 3]),
        &[
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // batch 0: identity-ish 2x3
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // batch 1
        ],
    )
    .unwrap();
    let mut out3 = Tensor::zeros(Shape::from_dims([2, 2, 2]), DType::F32, cpu()).unwrap();
    let batched_view = batched.as_view().unwrap();
    let mut out3_view = out3.as_mut_view().unwrap();
    CpuBackend::new()
        .execute_op(
            OpId::MatMul,
            &[batched_view, rhs_view.clone()],
            std::slice::from_mut(&mut out3_view),
            &OpParams::default(),
        )
        .unwrap();
    assert_eq!(
        out3.as_f32_slice().unwrap(),
        &[
            7.0, 8.0, // batch 0 row 0 = 1 * rhs row 0
            9.0, 10.0, // batch 0 row 1 = 1 * rhs row 1
            11.0, 12.0, // batch 1 row 0 = 1 * rhs row 2
            27.0, 30.0, // batch 1 row 1 = rhs row 0 + rhs row 1 + rhs row 2
        ]
    );

    // A rank-1 operand is still an honest, named failure rather than an
    // implicit broadcast or a silent wrong answer.
    let vector = Tensor::from_f32(Shape::from_dims([3]), &[1.0, 2.0, 3.0]).unwrap();
    let vector_view = vector.as_view().unwrap();
    let mut out3_view = out3.as_mut_view().unwrap();
    let error = CpuBackend::new()
        .execute_op(
            OpId::MatMul,
            &[vector_view, rhs_view],
            std::slice::from_mut(&mut out3_view),
            &OpParams::default(),
        )
        .unwrap_err();
    assert!(matches!(error, Fn3Error::UnsupportedOp { .. }), "{error}");
    assert!(error.to_string().contains("rank"), "{error}");
}

#[test]
fn two_op_plan_runs_add_then_matmul() {
    let a = Tensor::from_f32(Shape::from_dims([2, 3]), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let bias = Tensor::from_f32(Shape::from_dims([3]), &[10.0, 20.0, 30.0]).unwrap();
    let add_out = Tensor::zeros(Shape::from_dims([2, 3]), DType::F32, cpu()).unwrap();
    let w = Tensor::from_f32(Shape::from_dims([3, 2]), &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
    let mm_out = Tensor::zeros(Shape::from_dims([2, 2]), DType::F32, cpu()).unwrap();

    let plan = ExecutionPlan::new(
        vec![
            Instruction::new(OpId::Add, vec![0, 1], vec![2]),
            Instruction::new(OpId::MatMul, vec![2, 3], vec![4]),
        ],
        5,
    )
    .unwrap();
    let model = PreparedModel::new(
        plan,
        ModelWeights::new(vec![]).unwrap(),
        cpu(),
        small_envelope(),
        None,
    )
    .unwrap();

    let mut slots = vec![Some(a), Some(bias), Some(add_out), Some(w), Some(mm_out)];
    model.execute(&CpuBackend::new(), &mut slots).unwrap();
    assert_eq!(
        slots[2].as_ref().unwrap().as_f32_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
    assert_eq!(
        slots[4].as_ref().unwrap().as_f32_slice().unwrap(),
        &[44.0, 55.0, 50.0, 61.0]
    );
}

#[test]
fn plan_rejects_slot_aliasing_and_out_of_range_slots() {
    let alias =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![0])], 2).unwrap_err();
    assert!(alias.to_string().contains("reads and writes"), "{alias}");

    let out_of_range =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![9])], 3).unwrap_err();
    assert!(matches!(out_of_range, Fn3Error::OutOfBounds { .. }));

    let plan =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
    let mut slots = vec![None];
    let error = plan.execute(&CpuBackend::new(), &mut slots).unwrap_err();
    assert!(matches!(error, Fn3Error::CapacityExceeded { .. }));
}

// ---------------------------------------------------------------------------
// 9. Storage: allocation diagnostics, alignment, checked casts, exclusive writes.
// ---------------------------------------------------------------------------

#[test]
fn allocation_diagnostics_distinguish_known_and_unknown_request_sizes() {
    let detail = "allocator refused the request";
    for bytes in [4096, 0] {
        let known = Fn3Error::Allocation {
            bytes,
            detail: detail.into(),
        };
        assert!(matches!(known, Fn3Error::Allocation { bytes: n, .. } if n == bytes));
        assert_eq!(
            known.to_string(),
            format!("failed to allocate {bytes} bytes: {detail}")
        );
    }

    let unknown = Fn3Error::AllocationSizeUnknown {
        detail: detail.into(),
    };
    assert!(matches!(unknown, Fn3Error::AllocationSizeUnknown { .. }));
    assert_eq!(
        unknown.to_string(),
        format!("allocation failed (request size unknown): {detail}")
    );
}

#[test]
fn storage_is_aligned_and_casts_are_checked() {
    let storage = Storage::from_bytes(&encoded_f32(&[1.0, 2.0, 3.0])).unwrap();
    assert_eq!(storage.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    assert!(storage.is_exclusive());

    let unaligned = Storage::from_bytes(&[0u8; 6]).unwrap();
    let error = unaligned.as_slice::<f32>().unwrap_err();
    assert!(matches!(error, Fn3Error::StorageCast { .. }));
    assert!(error.to_string().contains("6 bytes"), "{error}");
}

#[test]
fn shared_storage_cannot_be_written() {
    let mut storage = Storage::zeroed(16).unwrap();
    let shared = storage.clone();
    assert!(!storage.is_exclusive());
    let error = storage.try_make_mut().unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::StorageShared {
            bytes: 16,
            handles: 2
        }
    ));
    drop(shared);
    storage.try_make_mut().unwrap();
}

// ---------------------------------------------------------------------------
// 10. Dynamic bounds.
// ---------------------------------------------------------------------------

#[test]
fn envelope_rejects_out_of_bounds_and_names_axis_and_bound() {
    let envelope = ShapeEnvelope::uniform(8).unwrap();
    let values = LiveDims::new(2, 9, 0, 4, 2, 8).unwrap();
    let error = envelope.check(&values).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DimOutOfEnvelope {
            symbol: "seq",
            requested: 9,
            min: 1,
            max: 8
        }
    ));
    let text = error.to_string();
    assert!(text.contains("seq = 9"), "{text}");
    assert!(text.contains("[1, 8]"), "{text}");
}

#[test]
fn dim_bounds_are_validated_and_capped() {
    assert!(DimBound::new(5, 4).is_err());
    assert!(DimBound::new(0, DimBound::MAX_EXTENT + 1).is_err());
    assert_eq!(DimBound::exact(7).unwrap().max(), 7);
    assert!(DimBound::new(0, DimBound::MAX_EXTENT).is_ok());
}

#[test]
fn exact_envelope_accepts_its_own_dims_and_rejects_another_total() {
    let dims = LiveDims::new(1, 4, 2, 12, 3, 64).unwrap();
    let envelope = ShapeEnvelope::exact(&dims).unwrap();
    envelope.check(&dims).unwrap();
    envelope.check_consistency(&dims).unwrap();
    assert_eq!(envelope.get(Symbol::Total).max(), 6);

    let mut inconsistent = dims;
    inconsistent.total = 99;
    let error = envelope.check(&inconsistent).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DimOutOfEnvelope {
            symbol: "total",
            requested: 99,
            ..
        }
    ));
    assert!(dims.check_total().is_ok());
    assert!(inconsistent.check_total().is_err());
}

#[test]
fn grouped_query_geometry_is_validated() {
    let dims = LiveDims::new(1, 1, 0, 12, 5, 64).unwrap();
    let envelope = ShapeEnvelope::exact(&dims).unwrap();
    let error = envelope.check_consistency(&dims).unwrap_err();
    assert!(
        error.to_string().contains("multiple of n_kv_heads"),
        "{error}"
    );
}

// ---------------------------------------------------------------------------
// 11. Model and session ownership.
// ---------------------------------------------------------------------------

fn session_model() -> PreparedModel {
    let plan =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
    PreparedModel::new(
        plan,
        ModelWeights::new(vec![]).unwrap(),
        cpu(),
        small_envelope(),
        Some(KvCacheConfig {
            layers: 1,
            capacity_tokens: 8,
            batch: 1,
            n_kv_heads: 2,
            head_dim: 4,
            dtype: DType::F32,
            layout: KvCacheLayout::BatchHeadsCapacityHeadDim,
        }),
    )
    .unwrap()
}

#[test]
fn prepared_model_is_shareable_and_sessions_own_their_state() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    assert_send_sync::<PreparedModel>();
    assert_send::<Session<'static>>();

    let model = session_model();
    let first = Session::new(&model).unwrap();
    let second = Session::new(&model).unwrap();
    assert_eq!(first.cache_len(), 0);
    assert_eq!(first.step(), 0);
    assert_eq!(first.phase(), SessionPhase::Empty);
    assert_eq!(first.caches().len(), 1);
    assert_ne!(
        first.caches()[0].keys().storage().as_bytes().as_ptr(),
        second.caches()[0].keys().storage().as_bytes().as_ptr(),
        "sessions must not share KV buffers"
    );
    assert_eq!(first.caches()[0].capacity(), 8);
    assert_eq!(
        first.caches()[0].keys().shape(),
        &Shape::from_dims([1, 2, 8, 4])
    );
}

#[test]
fn session_phase_rules_and_atomic_failures() {
    let model = session_model();
    let mut session = Session::new(&model).unwrap();

    let error = session.decode(&[1]).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::SessionState {
            action: "decode",
            phase: "empty",
            ..
        }
    ));
    assert!(error.to_string().contains("prefill"), "{error}");

    let error = session.decode(&[]).unwrap_err();
    assert!(matches!(error, Fn3Error::SessionState { .. }));

    // A session built without a runtime cannot step a model. This used to be
    // `NotImplemented` because the whole path was unimplemented; FN3-12 (#58)
    // implemented it, so the honest answer is now that this particular session
    // has no executor and must be built with `Session::with_runtime`. The
    // assertion is deliberately about the reported action and the named
    // remedy, not about a message string, so it survives rewording.
    let error = session.prefill(&[1, 2]).unwrap_err();
    assert!(
        matches!(error, Fn3Error::SessionState { .. }),
        "expected a session-state error, got {error}"
    );
    let text = error.to_string();
    assert!(text.contains("prefill"), "{text}");
    assert!(
        text.contains("with_runtime"),
        "a hero session must name the constructor that would make it work: {text}"
    );
    // A failed step leaves the cache extent and the counter unchanged.
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
    assert_eq!(session.phase(), SessionPhase::Empty);

    session.reset();
    assert_eq!(session.cache_len(), 0);
    assert_eq!(session.step(), 0);
}

#[test]
fn kv_cache_capacity_is_checked_before_mutation() {
    let model = session_model();
    let session = Session::new(&model).unwrap();
    session.caches()[0].check_capacity(8).unwrap();
    let error = session.caches()[0].check_capacity(9).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::CapacityExceeded {
            what: "KV cache live extent",
            requested: 9,
            capacity: 8
        }
    ));
    assert_eq!(session.caches()[0].live_len(), 0);
}

#[test]
fn model_rejects_weights_on_another_device() {
    let plan =
        ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
    let foreign = cpu_tensor_tagged(vulkan(), &[1.0]);
    let weights = ModelWeights::new(vec![foreign]).unwrap();
    let error = PreparedModel::new(plan, weights, cpu(), small_envelope(), None).unwrap_err();
    assert!(matches!(
        error,
        Fn3Error::DeviceMismatch {
            expected: DeviceId::Cpu,
            actual: DeviceId::Vulkan(_),
            ..
        }
    ));
}

#[test]
fn backend_capabilities_are_declared() {
    let capabilities: Capabilities = CpuBackend::new().capabilities().clone();
    assert!(capabilities.supports_dtype(DType::F32));
    assert!(capabilities.supports_dtype(DType::I64));
    assert!(capabilities.supports_dtype(DType::I32));
    assert!(capabilities.supports_dtype(DType::Bool));
    assert!(capabilities.features.broadcasting);
    assert!(!capabilities.features.masked_attention);
    assert!(capabilities.supports_strided_views);
    assert_eq!(capabilities.max_workgroup_size, [1, 1, 1]);
    assert!(capabilities
        .check_buffer(capabilities.max_buffer_bytes)
        .is_ok());
    assert!(capabilities
        .check_buffer(capabilities.max_buffer_bytes + 1)
        .is_err());
    assert_eq!(CpuBackend::new().device(), DeviceId::Cpu);
}

#[test]
fn dtype_rejects_unknown_codes_with_context() {
    let error = DType::from_code(9, "typed weight header").unwrap_err();
    let text = error.to_string();
    assert!(text.contains('9'), "{text}");
    assert!(text.contains("typed weight header"), "{text}");
    let wrapped: Fn3Error = error.into();
    assert!(matches!(wrapped, Fn3Error::UnknownDTypeCode(_)));
    assert!(std::error::Error::source(&wrapped).is_some());
}

#[test]
fn dtype_helpers_are_explicit() {
    assert_eq!(DType::ALL.len(), 4);
    assert!(DType::F32.is_compute());
    assert!(DType::I64.is_index());
    assert!(DType::Bool.is_mask());
    assert!(DType::I32.is_integer());
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::F32.name(), "f32");
}
