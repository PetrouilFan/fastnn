//! FN3-11 Vulkan platform smoke tests.
//!
//! Run with:
//!
//! ```bash
//! CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_smoke -- --nocapture --test-threads=1
//! ```
//!
//! Physical-GPU tests skip cleanly with a `[fn3vk][skip]` message when no
//! non-software adapter is present. The lavapipe lane is gated behind
//! `FN3_VK_TEST_SOFTWARE=1` and every line it prints is prefixed
//! `[fn3vk][software-icd]`; it is CI evidence and never satisfies a
//! physical-device gate.
//!
//! The tests serialise on `GPU_LOCK` because physical-GPU runs are serialised by
//! the project's concurrency policy and because validation counts are
//! process-global.

#![cfg(feature = "vulkan")]

use std::sync::{Arc, Mutex};

use fastnn::fn3::backend::cpu::CpuBackend;
use fastnn::fn3::backend::vulkan::instance::validation_counts;
use fastnn::fn3::backend::vulkan::{
    adapters, available_devices as vulkan_devices, BufferDesc, KernelSpec, VulkanBackend,
    VulkanConfig,
};
use fastnn::fn3::backend::{Backend, OpParams};
use fastnn::fn3::device::{available_devices, DeviceId, DeviceRequest, PhysicalDeviceId};
use fastnn::fn3::dtype::DType;
use fastnn::fn3::error::Fn3Error;
use fastnn::fn3::model::{ExecutionPlan, Instruction};
use fastnn::fn3::ops::OpId;
use fastnn::fn3::shape::Shape;
use fastnn::fn3::storage::Storage;
use fastnn::fn3::tensor::{Tensor, TensorMeta};

/// Serialises physical-GPU use and validation-count assertions.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// True when this process must produce a real device result rather than skip.
///
/// `tools/ci/gpu_evidence.py` sets this: a record that names a physical device
/// must have been produced by a run that actually used one, so an incapable
/// host has to fail rather than quietly skip into a green suite.
fn require_physical() -> bool {
    std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
}

/// The adapter UUID the caller pinned, if any.
///
/// Selection is by UUID and never by index: the loader enumerates a software
/// ICD alongside the real GPU, so index 0 is not stable across machines.
fn pinned_uuid() -> Option<String> {
    let value = std::env::var("FN3_VK_DEVICE_UUID").ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

/// The first adapter that is a real (non-software) GPU with a compute queue.
///
/// When `FN3_VK_DEVICE_UUID` is set, only that device is eligible, so the run
/// and the evidence record always name the same physical device.
fn physical_adapter() -> Option<u32> {
    let pinned = pinned_uuid();
    adapters()
        .iter()
        .find(|adapter| {
            adapter.has_compute_queue()
                && !adapter.is_software()
                && pinned
                    .as_deref()
                    .map(|uuid| adapter.uuid_string().eq_ignore_ascii_case(uuid))
                    .unwrap_or(true)
        })
        .map(|adapter| adapter.index)
}

fn software_adapter() -> Option<u32> {
    adapters()
        .iter()
        .find(|adapter| adapter.has_compute_queue() && adapter.is_software())
        .map(|adapter| adapter.index)
}

fn open_backend(index: u32, config: VulkanConfig) -> Option<Arc<VulkanBackend>> {
    match VulkanBackend::open_with(&DeviceRequest::Vulkan { index: Some(index) }, config) {
        Ok(backend) => Some(backend),
        Err(error) => {
            eprintln!("[fn3vk][skip] could not open physical adapter {index}: {error}");
            None
        }
    }
}

fn open_physical() -> Option<Arc<VulkanBackend>> {
    let Some(index) = physical_adapter() else {
        if require_physical() {
            panic!(
                "FN3_VK_REQUIRE_PHYSICAL=1 but no physical Vulkan device with a compute queue is \
                 present{}, so this run cannot produce physical-GPU evidence",
                match pinned_uuid() {
                    Some(uuid) => format!(" (pinned uuid {uuid})"),
                    None => String::new(),
                }
            );
        }
        eprintln!(
            "[fn3vk][skip] no physical Vulkan device with a compute queue is present; \
             run the lavapipe lane with FN3_VK_TEST_SOFTWARE=1 for software evidence"
        );
        return None;
    };
    let backend = match open_backend(index, VulkanConfig::validated()) {
        Some(backend) => backend,
        None if require_physical() => {
            panic!("FN3_VK_REQUIRE_PHYSICAL=1 but adapter {index} could not be opened")
        }
        None => return None,
    };
    let adapter = backend.adapter();
    eprintln!(
        "[fn3vk][test] device=\"{}\" uuid={} api={} driver={} queue_family={} validation={}",
        adapter.name,
        adapter.uuid_string(),
        fastnn::fn3::backend::vulkan::device_info::format_version(adapter.api_version),
        adapter.driver_version_string(),
        backend.vulkan_device().queue_family(),
        backend.validation_enabled()
    );
    Some(backend)
}

/// A Vulkan-tagged tensor backed by host storage.
///
/// The alpha core has no device-resident `Storage` (owned by #52/#48), so this
/// is the documented `Tensor::from_storage` validation path. The backend
/// uploads it, dispatches on the GPU and downloads the result; the residency
/// record proves the dispatch.
fn host_tensor_on(device: DeviceId, shape: Shape, values: &[f32]) -> Tensor {
    let storage = Storage::from_bytes(bytemuck::cast_slice(values)).expect("host storage");
    let meta = TensorMeta::contiguous(shape, DType::F32, device).expect("tensor metadata");
    Tensor::from_storage(storage, meta).expect("tensor")
}

fn pattern(count: usize, seed: u64) -> Vec<f32> {
    (0..count)
        .map(|index| (((index as u64 * 7 + seed) % 13) as f32) - 6.0)
        .collect()
}

fn add_plan() -> ExecutionPlan {
    ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3)
        .expect("add plan is valid")
}

/// Run the CPU reference through the CPU backend for the same inputs.
fn cpu_add_reference(shape: &Shape, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let lhs = Tensor::from_f32(shape.clone(), lhs).expect("cpu lhs");
    let rhs = Tensor::from_f32(shape.clone(), rhs).expect("cpu rhs");
    let mut out = Tensor::zeros(shape.clone(), DType::F32, DeviceId::Cpu).expect("cpu out");
    {
        let lhs_view = lhs.as_view().expect("lhs view");
        let rhs_view = rhs.as_view().expect("rhs view");
        let mut out_view = out.as_mut_view().expect("out view");
        CpuBackend::new()
            .execute_op(
                OpId::Add,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default(),
            )
            .expect("cpu add");
    }
    out.as_f32_slice().expect("cpu result").to_vec()
}

#[test]
fn upload_dispatch_download_matches_cpu_add_with_an_odd_tail() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    // 100_003 is prime and 100_003 % 64 == 3: the final workgroup is mostly
    // inactive, which is the bounds-check path in the shader.
    let count = 100_003usize;
    let shape = Shape::from_dims([count as u64]);
    let lhs_values = pattern(count, 3);
    let rhs_values = pattern(count, 5);
    let zeros = vec![0.0f32; count];

    let lhs = host_tensor_on(device, shape.clone(), &lhs_values);
    let rhs = host_tensor_on(device, shape.clone(), &rhs_values);
    let out = host_tensor_on(device, shape.clone(), &zeros);
    let mut slots = vec![Some(lhs), Some(rhs), Some(out)];
    add_plan()
        .execute(&*backend, &mut slots)
        .expect("vulkan add through the plan path");

    let got = slots[2]
        .as_ref()
        .expect("output slot is present")
        .as_f32_slice()
        .expect("output is contiguous f32");
    let want = cpu_add_reference(&shape, &lhs_values, &rhs_values);
    assert_eq!(got.len(), want.len());
    let mut max_abs_error = 0.0f32;
    let mut exact = 0usize;
    for (index, (actual, expected)) in got.iter().zip(want.iter()).enumerate() {
        let error = (actual - expected).abs();
        if error > max_abs_error {
            max_abs_error = error;
        }
        if actual == expected {
            exact += 1;
        } else {
            panic!(
                "mismatch at {index}: gpu={actual} cpu={expected} (max_abs_error={max_abs_error})"
            );
        }
    }
    assert_eq!(max_abs_error, 0.0);
    assert_eq!(exact, count);

    let records = backend.residency().records();
    assert_eq!(records.len(), 1, "one op, one residency record");
    let record = &records[0];
    assert_eq!(record.backend, "vulkan");
    assert_eq!(record.op, OpId::Add);
    assert_eq!(record.device, device);
    assert!(record.submissions >= 1);
    assert!(
        record.proves_device_execution(),
        "record must prove device execution: {record:?}"
    );
    assert!(!record.inputs_read_back, "inputs must never be read back");
    assert_eq!(record.upload_bytes, 2 * count as u64 * 4);
    assert_eq!(record.readback_bytes, count as u64 * 4);
    assert_eq!(record.device_local_bytes, 3 * count as u64 * 4);

    let memory = backend.memory_stats();
    eprintln!(
        "[fn3vk][proof] op=add count={count} workgroups={:?} submissions={} \
         gpu_ns={:?} gpu_ticks={:?} host_ns={} max_abs_error={max_abs_error} exact={exact}/{count} \
         chunks={} reserved={} live={} peak_live={} allocations={}",
        record.workgroups,
        record.submissions,
        record.gpu_ns,
        record.gpu_ticks,
        record.host_ns,
        memory.chunks,
        memory.reserved_bytes,
        memory.live_bytes,
        memory.peak_live_bytes,
        memory.allocation_count
    );
    assert_eq!(
        memory.live_device_local_bytes, 0,
        "op-scoped device buffers are freed; host-visible staging may stay pooled"
    );

    let summary = backend.residency().summary();
    assert_eq!(summary.executions, 1);
    assert_eq!(summary.proven_device_executions, 1);
    assert_eq!(summary.input_readbacks, 0);

    drop(slots);
    drop(backend);
    let counts_after = validation_counts();
    assert_eq!(
        counts_after, counts_before,
        "validation layer reported an error or warning; see the [fn3vk][validation] lines above"
    );
    eprintln!(
        "[fn3vk][validation] errors={} warnings={} (delta 0/0 since the section opened)",
        counts_after.0, counts_after.1
    );
}

#[test]
fn device_buffer_round_trip_through_staging() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let count = 4099usize;
    let bytes = (count * 4) as u64;
    let values = pattern(count, 11);
    let host: Vec<u8> = bytemuck::cast_slice(&values).to_vec();

    let buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(bytes, "roundtrip"))
        .expect("device buffer");
    let upload = backend.upload(&buffer, &host).expect("upload");
    assert_eq!(upload.bytes, bytes);
    assert!(upload.submissions >= 1);
    let mut readback = vec![0u8; host.len()];
    let download = backend.download(&buffer, &mut readback).expect("download");
    assert_eq!(download.bytes, bytes);
    assert_eq!(readback, host, "device buffer round trip is byte-exact");

    let staging = backend.staging_stats();
    eprintln!(
        "[fn3vk][proof] staging round trip bytes={bytes} upload_submissions={} \
         download_submissions={} leases={} pooled_blocks={} pooled_bytes={} \
         created_blocks={}",
        upload.submissions,
        download.submissions,
        staging.leases,
        staging.pooled_blocks,
        staging.pooled_bytes,
        staging.created_blocks
    );
    assert!(staging.upload_bytes >= bytes);
    assert!(staging.readback_bytes >= bytes);

    drop(buffer);
    assert_eq!(backend.memory_stats().live_device_local_bytes, 0);
    drop(backend);
    assert_eq!(validation_counts(), counts_before);
}

#[test]
fn allocation_reuse_and_peak_byte_accounting() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let block = 1024 * 1024u64;
    let mut first_reserved = 0u64;
    let mut first_chunks = 0usize;
    for iteration in 0..20 {
        let a = backend
            .alloc_buffer(&BufferDesc::device_storage(block, "reuse.a"))
            .expect("a");
        let b = backend
            .alloc_buffer(&BufferDesc::device_storage(block, "reuse.b"))
            .expect("b");
        let c = backend
            .alloc_buffer(&BufferDesc::device_storage(block, "reuse.c"))
            .expect("c");
        let live = backend.memory_stats();
        assert_eq!(live.live_bytes, 3 * block);
        if iteration == 0 {
            first_reserved = live.reserved_bytes;
            first_chunks = live.chunks;
        } else {
            assert_eq!(
                live.reserved_bytes, first_reserved,
                "iteration {iteration} reserved new memory instead of reusing the chunk"
            );
            assert_eq!(live.chunks, first_chunks);
        }
        assert_eq!(
            live.peak_live_bytes,
            3 * block,
            "peak live bytes must not grow with churn: {live:?}"
        );
        assert!(
            live.largest_free_range >= 3 * block,
            "a fused free range must be available: {live:?}"
        );
        drop(a);
        drop(b);
        drop(c);
        assert_eq!(backend.memory_stats().live_bytes, 0);
    }
    let stats = backend.memory_stats();
    eprintln!(
        "[fn3vk][memory] chunks={} reserved={} live={} peak_live={} free={} \
         allocation_count={} free_count={} largest_free={} chunk_sizes={:?}",
        stats.chunks,
        stats.reserved_bytes,
        stats.live_bytes,
        stats.peak_live_bytes,
        stats.free_bytes,
        stats.allocation_count,
        stats.free_count,
        stats.largest_free_range,
        stats
            .chunk_list
            .iter()
            .map(|chunk| chunk.size)
            .collect::<Vec<_>>()
    );
    assert_eq!(stats.allocation_count, 60);
    assert_eq!(stats.free_count, 60);
    assert_eq!(stats.peak_live_bytes, 3 * block);
    assert!(stats.reserved_bytes <= 512 * 1024 * 1024);
    assert_eq!(
        stats.chunk_list.len(),
        1,
        "3 MiB fits one device-local chunk"
    );
}

#[test]
fn repeated_submissions_keep_handles_and_memory_balanced() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let count = 4099usize;
    let shape = Shape::from_dims([count as u64]);
    let lhs = host_tensor_on(device, shape.clone(), &pattern(count, 1));
    let rhs = host_tensor_on(device, shape.clone(), &pattern(count, 2));
    let mut out = host_tensor_on(device, shape.clone(), &vec![0.0f32; count]);
    let iterations = 100u32;
    for iteration in 0..iterations {
        {
            let lhs_view = lhs.as_view().expect("lhs view");
            let rhs_view = rhs.as_view().expect("rhs view");
            let mut out_view = out.as_mut_view().expect("out view");
            backend
                .execute_op(
                    OpId::Add,
                    &[lhs_view, rhs_view],
                    std::slice::from_mut(&mut out_view),
                    &OpParams::default(),
                )
                .expect("repeated add");
        }
        let memory = backend.memory_stats();
        assert_eq!(
            memory.live_device_local_bytes, 0,
            "iteration {iteration}: device buffers outlived the op"
        );
        assert_eq!(
            backend
                .vulkan_device()
                .ledger()
                .outstanding_count("device_buffer"),
            0,
            "iteration {iteration}: a device buffer handle leaked"
        );
        assert_eq!(
            backend.vulkan_device().ledger().outstanding_count("fence"),
            1,
            "iteration {iteration}: exactly one fence per thread submitter"
        );
        assert_eq!(
            backend
                .vulkan_device()
                .ledger()
                .outstanding_count("command_pool"),
            1,
            "iteration {iteration}: exactly one command pool per thread"
        );
    }
    let summary = backend.residency().summary();
    assert_eq!(summary.executions, u64::from(iterations));
    assert_eq!(summary.proven_device_executions, u64::from(iterations));
    assert_eq!(summary.input_readbacks, 0);
    assert_eq!(summary.submissions, u64::from(iterations));
    let staging = backend.staging_stats();
    eprintln!(
        "[fn3vk][proof] repeated submissions={iterations} residency_records={} \
         proven_device={} leases={} staging_blocks={} pooled_bytes={} \
         outstanding_handles={:?}",
        summary.executions,
        summary.proven_device_executions,
        staging.leases,
        staging.created_blocks,
        staging.pooled_bytes,
        backend.outstanding_handles()
    );

    // Every transient kind must be balanced while the backend is alive; only
    // the device, its pipeline cache and the per-thread submitter's persistent
    // handles remain.
    let outstanding = backend.outstanding_handles();
    for line in &outstanding {
        assert!(
            line.starts_with("device:")
                || line.starts_with("pipeline_cache:")
                || line.starts_with("command_pool:")
                || line.starts_with("command_buffer:")
                || line.starts_with("fence:")
                || line.starts_with("descriptor_pool:")
                || line.starts_with("query_pool:")
                || line.starts_with("pipeline:")
                || line.starts_with("pipeline_layout:")
                || line.starts_with("descriptor_set_layout:")
                || line.starts_with("shader_module:")
                || line.starts_with("staging_buffer"),
            "unexpected outstanding handle after {iterations} submissions: {line}"
        );
    }
    drop(out);
    drop(lhs);
    drop(rhs);
    drop(backend);
    assert_eq!(validation_counts(), counts_before);
}

#[test]
fn explicit_teardown_leaves_only_the_device_and_pipeline_cache() {
    let _guard = gpu_lock();
    let Some(index) = physical_adapter() else {
        eprintln!("[fn3vk][skip] no physical Vulkan device for the teardown test");
        return;
    };
    let counts_before = validation_counts();
    let instance = fastnn::fn3::backend::vulkan::instance::Instance::new(
        &fastnn::fn3::backend::vulkan::instance::InstanceConfig::with_validation(true),
    )
    .expect("instance");
    let config = fastnn::fn3::backend::vulkan::DeviceConfig::default();
    let device = fastnn::fn3::backend::vulkan::Device::open(Arc::clone(&instance), index, &config)
        .expect("device");
    let buffer = device
        .create_buffer(&BufferDesc::device_storage(64 * 1024, "teardown"))
        .expect("buffer");
    let submitter = fastnn::fn3::backend::vulkan::Submitter::new(&device).expect("submitter");
    drop(submitter);
    drop(buffer);
    let outstanding = device.ledger().outstanding();
    for line in &outstanding {
        assert!(
            line.starts_with("device:") || line.starts_with("pipeline_cache:"),
            "unexpected outstanding handle after explicit teardown: {line}"
        );
    }
    drop(device);
    drop(instance);
    let counts_after = validation_counts();
    assert_eq!(
        counts_after, counts_before,
        "validation reported a leak at device/instance destruction"
    );
    eprintln!(
        "[fn3vk][validation] errors={} warnings={} (delta 0/0 at device/instance destruction)",
        counts_after.0, counts_after.1
    );
}

#[test]
fn an_allocation_over_the_device_cap_is_actionable() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let limits = backend.vulkan_device().limits().clone();
    let adapter_name = backend.adapter().name.clone();
    let requested = limits.max_memory_allocation_size + 1;
    let error = backend
        .alloc_buffer(&BufferDesc::device_storage(requested, "too.large"))
        .expect_err("an allocation above maxMemoryAllocationSize must fail");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] size-over-limit: {text}");
    match &error {
        Fn3Error::Allocation { bytes, .. } => assert_eq!(*bytes, requested),
        other => panic!("expected Fn3Error::Allocation, found {other:?}"),
    }
    assert!(text.contains(&requested.to_string()), "{text}");
    assert!(
        text.contains(&limits.max_memory_allocation_size.to_string()),
        "{text}"
    );
    assert!(text.contains(&adapter_name), "{text}");
    // Nothing was allocated by the failed request.
    assert_eq!(backend.memory_stats().live_device_local_bytes, 0);
}

#[test]
fn an_over_large_storage_binding_is_rejected_before_allocation() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let limit = u64::from(backend.vulkan_device().limits().max_storage_buffer_range);
    let requested = limit + 4;
    let error = backend
        .alloc_buffer(&BufferDesc::device_storage(requested, "too.wide"))
        .expect_err("a storage buffer above maxStorageBufferRange must fail");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] storage-range-over-limit: {text}");
    assert!(text.contains(&requested.to_string()), "{text}");
    assert!(text.contains(&limit.to_string()), "{text}");
    assert!(
        text.contains("maxStorageBufferRange") || text.contains("storage buffer range"),
        "{text}"
    );
    assert_eq!(backend.memory_stats().live_device_local_bytes, 0);
}

#[test]
fn unsupported_ops_and_shapes_are_reported_not_approximated() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };

    // Alpha covers the pinned graph, not the whole id space, so the backend must
    // still be honest about something. This is deliberately not a snapshot of
    // which ops are missing: the kernel owners register them one family at a
    // time (#59, #60, #61), and a test that pinned today's gap would fail on
    // every one of those merges. The invariant is that a verdict which is not
    // `Full` names the issue that owns the gap, and that asking for such an op
    // is rejected by name rather than approximated.
    let unsupported = OpId::ALL
        .iter()
        .copied()
        // An op that is *registered* but partial is still executable, so asking
        // it for zero operands is an arity error, not a gap. The probe has to be
        // an op nobody implements at all, otherwise registering one more kernel
        // turns this test into a false failure.
        .find(|op| !backend.supports(*op).is_supported())
        .expect("alpha must have at least one op that no kernel implements");

    for op in OpId::ALL {
        let support = backend.supports(op);
        if support.is_full() {
            continue;
        }
        let reason = support
            .reason()
            .unwrap_or_else(|| panic!("{op} is not Full and names no owner: {support}"));
        assert!(reason.contains('#'), "{op} must name its owner: {reason}");
    }

    let error = backend
        .execute_op(unsupported, &[], &mut [], &OpParams::default())
        .expect_err("an op with no Vulkan kernel must be rejected");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] unsupported-op {unsupported}: {text}");
    assert!(
        matches!(
            error,
            Fn3Error::UnsupportedOp {
                backend: "vulkan",
                ..
            }
        ),
        "{error:?}"
    );
    assert!(text.contains("vulkan"), "{text}");
    assert!(text.contains('#'), "{text}");
    assert_eq!(backend.memory_stats().live_bytes, 0);
}

#[test]
fn explicit_device_requests_error_without_a_cpu_fallback() {
    let _guard = gpu_lock();
    let available = available_devices();
    let error = VulkanBackend::open(&DeviceRequest::Vulkan { index: Some(999) })
        .expect_err("index 999 does not exist");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] explicit-device-unavailable: {text}");
    assert!(matches!(error, Fn3Error::DeviceUnavailable(_)), "{error:?}");
    assert!(text.contains("vulkan(index=999)"), "{text}");
    assert!(text.contains("available devices:"), "{text}");
    if available
        .iter()
        .any(|info| info.id.vulkan_index().is_some())
    {
        assert!(text.contains("vulkan("), "{text}");
    }

    let error = VulkanBackend::open(&DeviceRequest::VulkanNamed {
        name: "No Such Adapter".to_string(),
    })
    .expect_err("named adapter does not exist");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] named-device-unavailable: {text}");
    assert!(text.contains("No Such Adapter"), "{text}");
    assert!(text.contains("available devices:"), "{text}");
}

#[test]
fn capabilities_match_the_physical_device() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let limits = backend.vulkan_device().limits().clone();
    let capabilities = backend.capabilities().clone();
    assert_eq!(
        capabilities.max_buffer_bytes,
        u64::from(limits.max_storage_buffer_range).min(limits.max_memory_allocation_size)
    );
    assert_eq!(
        capabilities.max_workgroup_size,
        limits.max_compute_work_group_size
    );
    assert_eq!(
        capabilities.max_threads,
        limits.max_compute_work_group_invocations
    );
    assert_eq!(capabilities.dtypes, &[DType::F32][..]);
    assert!(!capabilities.supports_strided_views);
    // Capability flags are derived from the registered kernels, so they must
    // agree with the verdicts instead of pinning the kernel set of the day.
    assert_eq!(
        capabilities.features.masked_attention,
        backend.supports(OpId::MaskedAttention).is_full(),
        "masked_attention must be advertised exactly when the fused kernel is registered"
    );
    if capabilities.features.broadcasting {
        assert!(
            backend.supports(OpId::Add).is_full(),
            "broadcasting is advertised but Add is not Full"
        );
    }
    assert_eq!(backend.name(), "vulkan");
    assert_eq!(Backend::device(&*backend), backend.device_id());
    assert!(!backend.is_software());

    let listed = vulkan_devices();
    let adapter = backend.adapter();
    assert!(listed.iter().any(|info| {
        info.id == DeviceId::Vulkan(PhysicalDeviceId::new(adapter.index))
            && info.name == adapter.name
    }));
    let all = available_devices();
    assert_eq!(all[0].id, DeviceId::Cpu);
    assert!(all.iter().any(|info| info.id == backend.device_id()));
    eprintln!("[fn3vk][capabilities] {:?}", capabilities);
}

#[test]
fn kernel_registration_rejects_a_spec_that_disagrees_with_its_spirv() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    static WRONG: KernelSpec = KernelSpec {
        name: "wrong.add.f32",
        spirv: include_bytes!("../src/fn3/backend/vulkan/shaders/smoke_add.comp.spv"),
        entry_point: "main",
        // The shader declares 3 storage buffers; this claim is wrong on
        // purpose so registration must refuse it.
        bindings: &[
            fastnn::fn3::backend::vulkan::BindingSpec::storage(0, "lhs"),
            fastnn::fn3::backend::vulkan::BindingSpec::storage(1, "rhs"),
        ],
        push_constant_bytes: 4,
        workgroup_size: [64, 1, 1],
        dtypes: &[DType::F32],
        shape: "same-shape contiguous f32 vectors",
        grid: fastnn::fn3::backend::vulkan::kernels::grid::elementwise,
    };
    let error = backend.kernel(&WRONG).expect_err("spec must not register");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] kernel-spec-mismatch: {text}");
    assert!(text.contains("wrong.add.f32"), "{text}");
    assert!(text.contains("bindings"), "{text}");
}

#[test]
fn software_icd_lane_is_labelled_and_matches_cpu() {
    if std::env::var("FN3_VK_TEST_SOFTWARE").as_deref() != Ok("1") {
        eprintln!(
            "[fn3vk][software-icd][skip] set FN3_VK_TEST_SOFTWARE=1 to run the lavapipe lane; \
             it is CI-only evidence and never satisfies the physical-GPU gate"
        );
        return;
    }
    let _guard = gpu_lock();
    let Some(index) = software_adapter() else {
        eprintln!("[fn3vk][software-icd][skip] no software Vulkan adapter is enumerated");
        return;
    };
    let counts_before = validation_counts();
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated().with_software(true),
    )
    .expect("software backend opens only when allowed");
    assert!(backend.is_software());
    eprintln!(
        "[fn3vk][software-icd][test] device=\"{}\" uuid={} validation={} \
         (CI-only evidence, not a physical-GPU result)",
        backend.adapter().name,
        backend.adapter().uuid_string(),
        backend.validation_enabled()
    );

    let device = backend.device_id();
    let count = 1003usize;
    let shape = Shape::from_dims([count as u64]);
    let lhs_values = pattern(count, 17);
    let rhs_values = pattern(count, 19);
    let lhs = host_tensor_on(device, shape.clone(), &lhs_values);
    let rhs = host_tensor_on(device, shape.clone(), &rhs_values);
    let out = host_tensor_on(device, shape.clone(), &vec![0.0f32; count]);
    let mut slots = vec![Some(lhs), Some(rhs), Some(out)];
    add_plan()
        .execute(&*backend, &mut slots)
        .expect("software add");
    let got = slots[2].as_ref().unwrap().as_f32_slice().unwrap().to_vec();
    let want = cpu_add_reference(&shape, &lhs_values, &rhs_values);
    let max_abs_error = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_abs_error, 0.0);
    let records = backend.residency().records();
    assert_eq!(records.len(), 1);
    assert!(records[0].proves_device_execution());
    eprintln!(
        "[fn3vk][software-icd][proof] op=add count={count} submissions={} workgroups={:?} \
         gpu_ns={:?} max_abs_error={max_abs_error}",
        records[0].submissions, records[0].workgroups, records[0].gpu_ns
    );

    // A software adapter must be refused unless explicitly allowed.
    let refused = VulkanBackend::open(&DeviceRequest::Vulkan { index: Some(index) })
        .expect_err("software adapter requires allow_software");
    let text = refused.to_string();
    eprintln!("[fn3vk][software-icd][failure] software-refused: {text}");
    assert!(text.contains("software"), "{text}");

    drop(backend);
    assert_eq!(validation_counts(), counts_before);
}

#[test]
fn no_device_is_reported_with_an_actionable_message() {
    let _guard = gpu_lock();
    if !vulkan_devices().is_empty() {
        eprintln!(
            "[fn3vk][skip] a Vulkan device is present; run this test in a process with no ICD              (for example VK_ICD_FILENAMES=/tmp/no-such-icd.json) to exercise the no-device path"
        );
        return;
    }
    let error = VulkanBackend::open(&DeviceRequest::Vulkan { index: None })
        .expect_err("no Vulkan device must not resolve to CPU");
    let text = error.to_string();
    eprintln!("[fn3vk][failure] no-device: {text}");
    assert!(matches!(error, Fn3Error::DeviceUnavailable(_)), "{error:?}");
    assert!(text.contains("vulkan"), "{text}");
    assert!(text.contains("available devices: [cpu]"), "{text}");
    assert!(
        text.contains("ICD") || text.contains("ICD_FILENAMES") || text.contains("COMPUTE"),
        "{text}"
    );
}

#[test]
fn software_adapter_cannot_be_selected_accidentally() {
    let _guard = gpu_lock();
    let Some(index) = software_adapter() else {
        return;
    };
    // The physical helper never picks it, and `open` refuses it by default.
    assert_ne!(physical_adapter(), Some(index));
    let error = VulkanBackend::open(&DeviceRequest::Vulkan { index: Some(index) })
        .expect_err("software must be opt-in");
    assert!(error.to_string().contains("software"));
}
