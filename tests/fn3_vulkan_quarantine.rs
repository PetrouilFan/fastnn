//! Resource quarantine for submissions whose GPU completion is unknown.
//!
//! A `vkWaitForFences` error or timeout leaves completion unknown. The backend
//! must then keep every in-flight resource (destination buffers and their
//! allocator ranges, staging blocks, command/descriptor pools and fences) in an
//! owned quarantine until a later wait proves the fence signalled, or until
//! teardown quiesces first. Nothing may be destroyed, freed or recycled early.
//!
//! # Lanes
//!
//! * `cargo test --features vulkan --test fn3_vulkan_quarantine` runs the
//!   no-injection case: a successful transfer leaves the quarantine empty and a
//!   normal teardown quiesces then destroys.
//! * `cargo test --features vulkan,vulkan-fault-injection --test
//!   fn3_vulkan_quarantine` additionally runs the deterministic pending cases:
//!   [`FaultPoint::FenceUnknown`] reports unknown completion *before* the real
//!   wait (the work may still be running), and [`FaultPoint::Idle`] makes
//!   teardown's `vkDeviceWaitIdle` fail so the documented leak path is taken.
//!
//! The post-completion fault (`FaultPoint::Fence` fires *after* the real wait)
//! stays in `tests/fn3_vulkan_faults.rs`, unchanged and separate.
//!
//! # Evidence
//!
//! Every assertion is a counter the backend already had or now exposes:
//! `MemoryStats` (allocations, frees, live bytes), `StagingStats` (pooled and
//! created blocks), `HandleLedger` (per-kind outstanding handles),
//! `quarantine_stats()` and the process-wide `teardown_totals()`.
//!
//! ```bash
//! flock /tmp/opencode/fn3-measure.lock env \
//!   FN3_VK_REQUIRE_PHYSICAL=1 FN3_VK_VALIDATION=1 \
//!   FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//!   cargo test --features vulkan,vulkan-fault-injection \
//!   --test fn3_vulkan_quarantine -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "vulkan")]

use std::sync::Mutex;

use fastnn::inference::backend::vulkan::instance::validation_counts;
use fastnn::inference::backend::vulkan::{
    adapters, teardown_totals, BufferDesc, QuarantineStats, VulkanBackend, VulkanConfig,
};
use fastnn::inference::device::DeviceRequest;

#[cfg(feature = "vulkan-fault-injection")]
use fastnn::inference::backend::vulkan::{FaultPoint, ReclaimReport};
#[cfg(feature = "vulkan-fault-injection")]
use fastnn::inference::Fn3Error;

/// Serialises physical-GPU use and the process-wide teardown counters.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn require_physical() -> bool {
    std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
}

fn pinned_uuid() -> Option<String> {
    let value = std::env::var("FN3_VK_DEVICE_UUID").ok()?;
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

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

fn open_physical() -> Option<std::sync::Arc<VulkanBackend>> {
    let Some(index) = physical_adapter() else {
        if require_physical() {
            panic!(
                "FN3_VK_REQUIRE_PHYSICAL=1 but no physical Vulkan device with a compute queue is \
                 present (pinned uuid {:?})",
                pinned_uuid()
            );
        }
        eprintln!("[fn3vk][quarantine][skip] no physical Vulkan device on this host");
        return None;
    };
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| panic!("opening adapter {index} failed: {error}"));
    assert!(
        !backend.is_software(),
        "the quarantine test needs a real GPU"
    );
    eprintln!(
        "[fn3vk][quarantine][device] name=\"{}\" uuid={} validation={}",
        backend.adapter().name,
        backend.adapter().uuid_string(),
        backend.validation_enabled()
    );
    Some(backend)
}

fn pattern(bytes: usize) -> Vec<u8> {
    (0..bytes).map(|index| (index % 251) as u8).collect()
}

#[cfg(feature = "vulkan-fault-injection")]
fn outstanding(backend: &VulkanBackend, kind: &str) -> u64 {
    backend.vulkan_device().ledger().outstanding_count(kind)
}

/// Every Vulkan kind an upload touches. Used to prove nothing was destroyed.
#[cfg(feature = "vulkan-fault-injection")]
const UPLOAD_KINDS: [&str; 6] = [
    "device_buffer",
    "staging_buffer",
    "command_pool",
    "command_buffer",
    "descriptor_pool",
    "fence",
];

// ---------------------------------------------------------------------------
// 1. Normal lane: a completed transfer leaves no quarantine and teardown
//    quiesces before destroying.
// ---------------------------------------------------------------------------

#[test]
fn a_completed_transfer_leaves_no_quarantine_and_teardown_quiesces() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let teardown_before = teardown_totals();
    let staging_before = backend.staging_stats();
    let payload = pattern(4096);
    let buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(4096, "quarantine-normal"))
        .expect("destination buffer");
    let upload = backend.upload(&buffer, &payload).expect("upload");
    assert_eq!(upload.bytes, 4096);
    let mut roundtrip = vec![0u8; payload.len()];
    let download = backend.download(&buffer, &mut roundtrip).expect("download");
    assert_eq!(download.bytes, payload.len() as u64);
    assert_eq!(roundtrip, payload, "the transfer must round-trip");

    assert_eq!(
        backend.quarantine_stats(),
        QuarantineStats::default(),
        "a completed transfer must not quarantine anything"
    );
    assert!(!backend.has_unknown_completion());
    let staging_after = backend.staging_stats();
    assert_eq!(
        staging_after.pooled_blocks,
        staging_before.pooled_blocks + 1,
        "a completed transfer returns its block to the pool"
    );
    assert_eq!(
        staging_after.created_blocks,
        staging_before.created_blocks + 1,
        "the upload and the download share one staging block"
    );
    drop(buffer);
    let memory = backend.memory_stats();
    assert_eq!(
        memory.live_bytes, staging_after.pooled_bytes,
        "dropping the only tensor buffer must return every tensor byte; only the pooled staging \
         block remains"
    );
    drop(backend);

    let teardown_after = teardown_totals();
    assert_eq!(
        teardown_after.quiesced_ok,
        teardown_before.quiesced_ok + 1,
        "normal teardown must quiesce"
    );
    assert_eq!(
        teardown_after.quiesced_failed, teardown_before.quiesced_failed,
        "normal teardown must not take the leak path"
    );
    assert_eq!(
        teardown_after.leaked_entries, teardown_before.leaked_entries,
        "normal teardown leaks no entry"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the completed path must raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] normal: round-trip 4096 B, quarantine empty, teardown quiesced \
         ({} ok, {} failed), validation {}/{}",
        teardown_after.quiesced_ok,
        teardown_after.quiesced_failed,
        validation_counts().0,
        validation_counts().1
    );
}

// ---------------------------------------------------------------------------
// 2. Injected lane: unknown completion keeps every in-flight resource owned,
//    and a later successful wait releases exactly once.
// ---------------------------------------------------------------------------

/// One failing upload with a warmed staging pool.
///
/// Returns the backend plus the counters taken just before the failure. The
/// warm-up proves the staging block came from the pool, so no allocation or
/// free is expected after the unknown wait.
#[cfg(feature = "vulkan-fault-injection")]
struct PendingUpload {
    backend: std::sync::Arc<VulkanBackend>,
    memory: fastnn::inference::backend::vulkan::MemoryStats,
    staging: fastnn::inference::backend::vulkan::StagingStats,
    ledger: Vec<(&'static str, u64)>,
    dst_allocation: u64,
}

#[cfg(feature = "vulkan-fault-injection")]
fn pending_upload() -> Option<PendingUpload> {
    let backend = open_physical()?;
    let payload = pattern(4096);
    // Warm the staging pool, then drop the buffer so the snapshot starts from a
    // clean per-kind ledger.
    let warm = backend
        .alloc_buffer(&BufferDesc::device_storage(4096, "quarantine-warm"))
        .expect("warm buffer");
    backend.upload(&warm, &payload).expect("warm upload");
    drop(warm);
    assert_eq!(
        backend.staging_stats().pooled_blocks,
        1,
        "the warm-up must leave one pooled block"
    );

    let dst = backend
        .alloc_buffer(&BufferDesc::device_storage(4096, "quarantine-dst"))
        .expect("destination buffer");
    let dst_allocation = dst.allocation_size();
    let memory = backend.memory_stats();
    let staging = backend.staging_stats();
    let ledger = UPLOAD_KINDS
        .iter()
        .map(|kind| (*kind, outstanding(&backend, kind)))
        .collect();

    backend.faults().arm(FaultPoint::FenceUnknown, 0);
    let error = backend
        .upload(&dst, &payload)
        .expect_err("the injected unknown fence wait must fail the upload");
    assert_eq!(
        backend.faults().fired_at(),
        Some(FaultPoint::FenceUnknown),
        "the failure must be the injected one"
    );
    assert!(
        matches!(error, Fn3Error::DeviceUnavailable(_)),
        "a terminal failure reports the device: {error:?}"
    );
    assert!(
        backend.vulkan_device().is_lost(),
        "unknown completion poisons the device"
    );

    // The caller's handle is dropped only after the quarantine took ownership.
    drop(dst);
    Some(PendingUpload {
        backend,
        memory,
        staging,
        ledger,
        dst_allocation,
    })
}

#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn an_unknown_fence_wait_quarantines_every_in_flight_resource() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(PendingUpload {
        backend,
        memory,
        staging,
        ledger,
        dst_allocation,
    }) = pending_upload()
    else {
        return;
    };

    // ---- still owned, nothing destroyed, nothing recycled -----------------
    let quarantine = backend.quarantine_stats();
    eprintln!("[fn3vk][quarantine] pending: {quarantine:?}");
    assert_eq!(quarantine.entries, 1, "one pending entry");
    assert_eq!(
        quarantine.pending_submissions, 1,
        "the entry carries its fence"
    );
    assert_eq!(
        quarantine.retained_submitters, 1,
        "command/descriptor pools and fence are owned"
    );
    assert_eq!(
        quarantine.retained_buffers, 2,
        "destination buffer and staging block are owned"
    );
    assert_eq!(quarantine.loose_buffers, 0);
    assert_eq!(quarantine.released_entries, 0);
    assert_eq!(quarantine.leaked_entries, 0);
    // The injected path returns before the wait is attempted, so completion was
    // never observed. The driver may still have finished by the time the entry
    // is captured, so this counter is reported, not asserted.
    eprintln!(
        "[fn3vk][quarantine] fence was unsignalled at capture: {}",
        quarantine.captured_unsignalled
    );

    let memory_after = backend.memory_stats();
    assert_eq!(
        memory_after.allocation_count, memory.allocation_count,
        "the failed upload allocated nothing new"
    );
    assert_eq!(
        memory_after.free_count, memory.free_count,
        "the failed upload freed no allocator range"
    );
    assert_eq!(
        memory_after.live_bytes, memory.live_bytes,
        "every live byte is still owned"
    );

    let staging_after = backend.staging_stats();
    assert_eq!(
        staging_after.pooled_blocks,
        staging.pooled_blocks - 1,
        "the staging pool must not take the leased block back"
    );
    assert_eq!(
        staging_after.created_blocks, staging.created_blocks,
        "no new staging block was created"
    );

    for (kind, before) in &ledger {
        assert_eq!(
            outstanding(&backend, kind),
            *before,
            "{kind} was destroyed before completion was proven"
        );
    }
    assert_eq!(
        outstanding(&backend, "device_buffer"),
        1,
        "the destination buffer is still owned after its handle dropped"
    );

    // ---- a later successful wait releases exactly once --------------------
    let reclaim = backend.reclaim_quarantine();
    assert_eq!(
        reclaim,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 2,
            pending: 0,
        },
        "the proven wait must release the entry and its two buffers"
    );
    let released = backend.quarantine_stats();
    assert_eq!(released.entries, 0, "the quarantine is empty");
    assert_eq!(released.released_entries, 1);
    assert_eq!(released.released_submitters, 1);
    assert_eq!(released.released_buffers, 2);
    assert_eq!(released.leaked_entries, 0);

    let memory_released = backend.memory_stats();
    assert_eq!(
        memory_released.allocation_count, memory.allocation_count,
        "releasing never allocates"
    );
    assert_eq!(
        memory_released.free_count,
        memory.free_count + 2,
        "the destination range and the staging range are each freed exactly once"
    );
    assert_eq!(
        memory_released.live_bytes,
        memory.live_bytes - dst_allocation - (staging.pooled_bytes),
        "only the two quarantined ranges left the allocator"
    );
    assert_eq!(
        outstanding(&backend, "device_buffer"),
        0,
        "the destination buffer was destroyed once"
    );
    assert_eq!(
        outstanding(&backend, "staging_buffer"),
        0,
        "the staging block was destroyed, not pooled"
    );
    for kind in ["command_pool", "command_buffer", "descriptor_pool", "fence"] {
        let before = ledger
            .iter()
            .find(|(name, _)| *name == kind)
            .map(|(_, count)| *count)
            .expect("tracked kind");
        assert_eq!(
            outstanding(&backend, kind),
            before - 1,
            "{kind}: exactly one submitter handle was destroyed"
        );
    }
    // A query pool is optional (an adapter without timestamps has none); it
    // is either absent before and after, or destroyed with the submitter.
    assert_eq!(
        outstanding(&backend, "query_pool"),
        0,
        "no query pool may outlive the quarantined submitter"
    );
    let staging_released = backend.staging_stats();
    assert_eq!(
        staging_released.pooled_blocks,
        staging.pooled_blocks - 1,
        "the released staging block is destroyed, not returned to the pool"
    );

    // ---- no double free on a second pass ----------------------------------
    let after_release: Vec<(&str, u64)> = UPLOAD_KINDS
        .iter()
        .map(|kind| (*kind, outstanding(&backend, kind)))
        .chain(std::iter::once((
            "query_pool",
            outstanding(&backend, "query_pool"),
        )))
        .collect();
    let again = backend.reclaim_quarantine();
    assert_eq!(again, ReclaimReport::default(), "nothing left to release");
    let memory_again = backend.memory_stats();
    assert_eq!(
        memory_again.free_count, memory_released.free_count,
        "a second reclaim must not free anything again"
    );
    assert_eq!(
        memory_again.allocation_count, memory_released.allocation_count,
        "a second reclaim must not allocate"
    );
    for (kind, count) in &after_release {
        assert_eq!(
            outstanding(&backend, kind),
            *count,
            "{kind} is unchanged by a second reclaim"
        );
    }
    eprintln!(
        "[fn3vk][quarantine] released: {released:?}, frees {} -> {}, live {} -> {}",
        memory.free_count,
        memory_released.free_count,
        memory.live_bytes,
        memory_released.live_bytes
    );

    drop(backend);
    assert_eq!(
        validation_counts(),
        counts_before,
        "the quarantine path must raise no validation message"
    );
}

// ---------------------------------------------------------------------------
// 3. Injected lane: teardown with a pending entry quiesces then destroys.
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn teardown_with_a_pending_entry_quiesces_then_destroys() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(PendingUpload { backend, .. }) = pending_upload() else {
        return;
    };
    assert_eq!(
        backend.quarantine_stats().entries,
        1,
        "teardown must start with a pending entry"
    );
    let teardown_before = teardown_totals();
    drop(backend);
    let teardown_after = teardown_totals();
    assert_eq!(
        teardown_after.quiesced_ok,
        teardown_before.quiesced_ok + 1,
        "teardown must quiesce, not leak"
    );
    assert_eq!(
        teardown_after.quiesced_failed, teardown_before.quiesced_failed,
        "a healthy device must not take the leak path"
    );
    assert!(
        teardown_after.released_entries > teardown_before.released_entries,
        "the pending entry is destroyed after the quiesce"
    );
    assert!(
        teardown_after.released_buffers > teardown_before.released_buffers + 1,
        "the destination and staging buffers are destroyed after the quiesce"
    );
    assert_eq!(
        teardown_after.leaked_entries, teardown_before.leaked_entries,
        "nothing is leaked when quiescing succeeds"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "quiesce-then-destroy must raise no validation message"
    );

    // The process and the physical device remain usable.
    let Some(fresh) = open_physical() else {
        return;
    };
    let payload = pattern(1024);
    let probe = fresh
        .alloc_buffer(&BufferDesc::device_storage(1024, "quarantine-probe"))
        .expect("probe buffer");
    fresh.upload(&probe, &payload).expect("probe upload");
    let mut roundtrip = vec![0u8; payload.len()];
    fresh
        .download(&probe, &mut roundtrip)
        .expect("probe download");
    assert_eq!(roundtrip, payload);
    drop(probe);
    drop(fresh);
    assert_eq!(
        validation_counts(),
        counts_before,
        "the probe backend must also raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] teardown pending: {} entry, {} buffer(s) destroyed after quiesce, \
         {} leaked",
        teardown_after.released_entries - teardown_before.released_entries,
        teardown_after.released_buffers - teardown_before.released_buffers,
        teardown_after.leaked_entries - teardown_before.leaked_entries
    );
}

// ---------------------------------------------------------------------------
// 4. Injected lane: a failed teardown quiesce leaks, never frees.
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_failed_teardown_quiesce_leaks_pending_resources() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(PendingUpload { backend, .. }) = pending_upload() else {
        return;
    };
    assert_eq!(backend.quarantine_stats().entries, 1);
    // The injected idle failure fires inside the backend's teardown, after the
    // pending entry exists. The injector is gone with the backend, so the
    // evidence is the teardown counter, not `fired_at()`.
    backend.faults().arm(FaultPoint::Idle, 0);
    let teardown_before = teardown_totals();
    drop(backend);
    let teardown_after = teardown_totals();
    assert_eq!(
        teardown_after.quiesced_failed,
        teardown_before.quiesced_failed + 1,
        "a failed quiesce is recorded, not silently ignored"
    );
    assert!(
        teardown_after.leaked_entries > teardown_before.leaked_entries,
        "the pending entry is deliberately leaked"
    );
    assert!(
        teardown_after.leaked_buffers > teardown_before.leaked_buffers + 1,
        "the destination and staging buffers are deliberately leaked"
    );
    assert_eq!(
        teardown_after.released_entries, teardown_before.released_entries,
        "a failed quiesce must not destroy a quarantined entry"
    );
    assert_eq!(
        teardown_after.released_buffers, teardown_before.released_buffers,
        "a failed quiesce must not free a quarantined buffer"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "leaking must not raise validation messages either"
    );
    println!(
        "[fn3vk][quarantine] teardown leak: {} entr(y/ies), {} buffer(s) leaked deliberately after \
         quiesce failure",
        teardown_after.leaked_entries - teardown_before.leaked_entries,
        teardown_after.leaked_buffers - teardown_before.leaked_buffers
    );
}
