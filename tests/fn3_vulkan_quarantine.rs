//! Resource quarantine for submissions whose GPU completion is unknown.
//!
//! A `vkWaitForFences` error or timeout leaves completion unknown. The backend
//! must then keep every in-flight resource (destination buffers and their
//! allocator ranges, staging blocks, command/descriptor pools and fences) in an
//! owned quarantine until a later wait proves the fence signalled, or until
//! teardown quiesces first. Nothing may be destroyed, freed or recycled early.
//!
//! # The shared-buffer rule
//!
//! A buffer can be bound by two per-thread submitters at once. Proving one
//! submission's fence does not prove the buffer is unused: the quarantine keeps
//! a device-wide reference count and frees a buffer only when the last
//! reference is released. The shared case below fails one submission while the
//! second is still running, reclaims the first, retires the second (its ticket
//! *joins* the already-quarantined id), then proves the second and frees the
//! buffer exactly once.
//!
//! # Lanes and cases
//!
//! * `cargo test --features vulkan --test fn3_vulkan_quarantine` (normal lane)
//!   runs `a_completed_transfer_leaves_no_quarantine_and_teardown_quiesces`.
//! * `cargo test --features vulkan,vulkan-fault-injection --test
//!   fn3_vulkan_quarantine` adds every injected case: pre-wait unknown
//!   completion, post-completion fence fault, implicit `Submission::drop`
//!   failure, `vkQueueSubmit` failure, command-buffer end-recording failure,
//!   the shared-buffer case, teardown with a pending entry, and the documented
//!   leak path when quiescing fails.
//!
//! The post-completion fault (`FaultPoint::Fence` fires *after* the real wait)
//! stays in `tests/fn3_vulkan_faults.rs`, unchanged and separate.
//!
//! # Evidence
//!
//! Every assertion is an observable: `MemoryStats` (allocations, frees, live
//! bytes), `StagingStats` (pooled and created blocks), `HandleLedger`
//! (per-kind outstanding handles), `quarantine_stats()`, the process-wide
//! `teardown_totals()`, and the process-wide destroyed counts for buffers,
//! allocator chunks and instances. The leak path must leave all destroy counts
//! unchanged.
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
use std::time::{Duration, Instant};

use fastnn::inference::backend::vulkan::instance::validation_counts;
use fastnn::inference::backend::vulkan::{
    adapters, destroyed_buffer_total, destroyed_chunk_total, destroyed_instance_total,
    teardown_totals, BufferDesc, QuarantineStats, VulkanBackend, VulkanConfig,
};
use fastnn::inference::device::DeviceRequest;

#[cfg(feature = "vulkan-fault-injection")]
use fastnn::inference::backend::vulkan::{FaultPoint, ReclaimReport, Submitter};
#[cfg(feature = "vulkan-fault-injection")]
use fastnn::inference::Fn3Error;

/// Serialises physical-GPU use and the process-wide counters.
static GPU_LOCK: Mutex<()> = Mutex::new(());

const PAYLOAD_BYTES: usize = 4096;

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

#[cfg(feature = "vulkan-fault-injection")]
fn assert_no_upload_handle_destroyed(backend: &VulkanBackend, before: &[(&'static str, u64)]) {
    for (kind, count) in before {
        assert_eq!(
            outstanding(backend, kind),
            *count,
            "{kind} was destroyed before completion was proven"
        );
    }
}

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
    let instances_before = destroyed_instance_total();
    let chunks_before = destroyed_chunk_total();
    let payload = pattern(PAYLOAD_BYTES);
    let buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-normal",
        ))
        .expect("destination buffer");
    let upload = backend.upload(&buffer, &payload).expect("upload");
    assert_eq!(upload.bytes, PAYLOAD_BYTES as u64);
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
        destroyed_instance_total(),
        instances_before + 1,
        "a healthy teardown destroys its instance"
    );
    assert!(
        destroyed_chunk_total() > chunks_before,
        "a healthy teardown frees the allocator's chunks"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the completed path must raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] normal: round-trip {} B, quarantine empty, teardown quiesced \
         ({} ok, {} failed), validation {}/{}",
        PAYLOAD_BYTES,
        teardown_after.quiesced_ok,
        teardown_after.quiesced_failed,
        validation_counts().0,
        validation_counts().1
    );
}

// ---------------------------------------------------------------------------
// Injected helpers
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
fn pending_upload(point: FaultPoint) -> Option<PendingUpload> {
    let backend = open_physical()?;
    let payload = pattern(PAYLOAD_BYTES);
    // Warm the staging pool, then drop the buffer so the snapshot starts from a
    // clean per-kind ledger.
    let warm = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-warm",
        ))
        .expect("warm buffer");
    backend.upload(&warm, &payload).expect("warm upload");
    drop(warm);
    assert_eq!(
        backend.staging_stats().pooled_blocks,
        1,
        "the warm-up must leave one pooled block"
    );

    let dst = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-dst",
        ))
        .expect("destination buffer");
    let dst_allocation = dst.allocation_size();
    let memory = backend.memory_stats();
    let staging = backend.staging_stats();
    let ledger = UPLOAD_KINDS
        .iter()
        .map(|kind| (*kind, outstanding(&backend, kind)))
        .collect();

    backend.faults().arm(point, 0);
    let error = backend
        .upload(&dst, &payload)
        .expect_err("the injected failure must fail the upload");
    assert_eq!(
        backend.faults().fired_at(),
        Some(point),
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

/// Assert a fully-owned pending entry and release it with one proof pass.
#[cfg(feature = "vulkan-fault-injection")]
fn assert_pending_then_release(point: FaultPoint, label: &str) {
    let counts_before = validation_counts();
    let Some(PendingUpload {
        backend,
        memory,
        staging,
        ledger,
        dst_allocation,
    }) = pending_upload(point)
    else {
        return;
    };
    let quarantine = backend.quarantine_stats();
    assert_eq!(quarantine.entries, 1, "{label}: one pending entry");
    assert_eq!(
        quarantine.pending_submissions, 1,
        "{label}: the entry carries its fence"
    );
    assert_eq!(
        quarantine.retained_submitters, 1,
        "{label}: command/descriptor pools and fence are owned"
    );
    assert_eq!(
        quarantine.retained_buffers, 2,
        "{label}: destination buffer and staging block are owned"
    );
    assert_eq!(quarantine.loose_buffers, 0);
    assert_eq!(quarantine.in_flight_buffers, 2);
    assert_eq!(quarantine.released_entries, 0);
    assert_eq!(quarantine.leaked_entries, 0);
    let memory_after = backend.memory_stats();
    assert_eq!(
        memory_after.allocation_count, memory.allocation_count,
        "{label}: the failed upload allocated nothing new"
    );
    assert_eq!(
        memory_after.free_count, memory.free_count,
        "{label}: the failed upload freed no allocator range"
    );
    assert_eq!(
        memory_after.live_bytes, memory.live_bytes,
        "{label}: every live byte is still owned"
    );
    let staging_after = backend.staging_stats();
    assert_eq!(
        staging_after.pooled_blocks,
        staging.pooled_blocks - 1,
        "{label}: the staging pool must not take the leased block back"
    );
    assert_no_upload_handle_destroyed(&backend, &ledger);

    let reclaim = backend.reclaim_quarantine();
    assert_eq!(
        reclaim,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 2,
            pending: 0,
        },
        "{label}: one proof wait releases the entry and both buffers"
    );
    let released = backend.quarantine_stats();
    assert_eq!(released.entries, 0, "{label}: the quarantine is empty");
    assert_eq!(released.retained_buffers, 0);
    assert_eq!(released.in_flight_buffers, 0);
    assert_eq!(released.released_entries, 1);
    assert_eq!(released.released_submitters, 1);
    assert_eq!(released.released_buffers, 2);
    let memory_released = backend.memory_stats();
    assert_eq!(
        memory_released.free_count,
        memory.free_count + 2,
        "{label}: each of the two ranges is freed exactly once"
    );
    assert_eq!(
        memory_released.live_bytes,
        memory.live_bytes - dst_allocation - staging.pooled_bytes,
        "{label}: only the two quarantined ranges left the allocator"
    );
    assert_eq!(outstanding(&backend, "device_buffer"), 0);
    assert_eq!(outstanding(&backend, "staging_buffer"), 0);
    for kind in ["command_pool", "command_buffer", "descriptor_pool", "fence"] {
        let before = ledger
            .iter()
            .find(|(name, _)| *name == kind)
            .map(|(_, count)| *count)
            .expect("tracked kind");
        assert_eq!(
            outstanding(&backend, kind),
            before - 1,
            "{label}: {kind} is destroyed exactly once"
        );
    }
    let again = backend.reclaim_quarantine();
    assert_eq!(
        again,
        ReclaimReport::default(),
        "{label}: a second reclaim frees nothing"
    );
    assert_eq!(
        backend.memory_stats().free_count,
        memory_released.free_count,
        "{label}: no double free"
    );
    drop(backend);
    assert_eq!(
        validation_counts(),
        counts_before,
        "{label}: the quarantine path must raise no validation message"
    );
    println!("[fn3vk][quarantine] {label}: released once, validation 0/0");
}

// ---------------------------------------------------------------------------
// 2. Injected lane: unknown completion keeps every in-flight resource owned.
// ---------------------------------------------------------------------------

#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn an_unknown_fence_wait_quarantines_every_in_flight_resource() {
    let _lock = gpu_lock();
    assert_pending_then_release(FaultPoint::FenceUnknown, "pre-wait unknown completion");
}

/// The post-completion fault: the real wait happened, so the entry is already
/// proven and a reclaim must release it without waiting for the GPU again.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_post_completion_fence_fault_releases_exactly_once_on_reclaim() {
    let _lock = gpu_lock();
    assert_pending_then_release(FaultPoint::Fence, "post-completion fence fault");
}

/// A token dropped without an explicit wait runs the implicit wait in
/// `Submission::drop`; with the pre-wait fault that wait fails and the whole
/// submission must be quarantined exactly like an explicit wait failure.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn an_implicit_submission_drop_wait_failure_quarantines() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let payload = pattern(PAYLOAD_BYTES);
    let dst = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-implicit-dst",
        ))
        .expect("destination buffer");
    let lease = backend
        .staging()
        .acquire(PAYLOAD_BYTES as u64)
        .expect("staging lease");
    lease.write(&payload).expect("staging write");
    let mut submitter = Submitter::new(backend.vulkan_device()).expect("submitter");
    let token = submitter
        .record(|recorder| {
            recorder.copy_buffer(lease.buffer(), 0, &dst, 0, PAYLOAD_BYTES as u64)?;
            Ok(())
        })
        .expect("record");
    // The submission is already queued; fail only the implicit wait.
    backend.faults().arm(FaultPoint::FenceUnknown, 0);
    drop(token);

    let quarantine = backend.quarantine_stats();
    assert_eq!(quarantine.entries, 1);
    assert_eq!(quarantine.pending_submissions, 1);
    assert_eq!(quarantine.retained_submitters, 1);
    assert_eq!(quarantine.retained_buffers, 2);
    assert_eq!(quarantine.in_flight_buffers, 2);
    let memory = backend.memory_stats();
    let reclaim = backend.reclaim_quarantine();
    assert_eq!(
        reclaim,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 2,
            pending: 0,
        }
    );
    assert_eq!(backend.memory_stats().free_count, memory.free_count + 2);
    drop(submitter);
    drop(lease);
    drop(dst);
    drop(backend);
    assert_eq!(
        validation_counts(),
        counts_before,
        "the implicit-wait path must raise no validation message"
    );
    println!("[fn3vk][quarantine] implicit Submission::drop wait failure: released once");
}

/// `vkQueueSubmit` fails before the driver saw the work: nothing is in flight,
/// so the submitter is quarantined with no fence or buffer and a reclaim must
/// not block on a never-submitted fence.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_failed_vkqueuesubmit_quarantines_only_the_submitter() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let payload = pattern(PAYLOAD_BYTES);
    let dst = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-submit-dst",
        ))
        .expect("destination buffer");
    backend.faults().arm(FaultPoint::Submit, 0);
    let error = backend
        .upload(&dst, &payload)
        .expect_err("the injected submit must fail");
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::Submit));
    assert!(matches!(error, Fn3Error::DeviceUnavailable(_)), "{error:?}");
    let quarantine = backend.quarantine_stats();
    assert_eq!(quarantine.entries, 1);
    assert_eq!(
        quarantine.pending_submissions, 0,
        "nothing reached the queue, so there is no fence to wait for"
    );
    assert_eq!(quarantine.retained_submitters, 1);
    assert_eq!(quarantine.retained_buffers, 0);
    assert_eq!(
        quarantine.loose_buffers, 0,
        "the returned staging block went back to the pool, not loose"
    );
    assert_eq!(
        backend.staging_stats().pooled_blocks,
        1,
        "the pool still owns the block; nothing is freed before teardown"
    );
    let memory = backend.memory_stats();
    let start = Instant::now();
    let reclaim = backend.reclaim_quarantine();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "a proof pass must not wait on a never-submitted fence"
    );
    assert_eq!(
        reclaim,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 0,
            pending: 0,
        }
    );
    assert_eq!(
        backend.memory_stats().free_count,
        memory.free_count,
        "nothing was freed before teardown"
    );
    drop(dst);
    assert_eq!(
        backend.quarantine_stats().loose_buffers,
        1,
        "the destination buffer is loose on the terminal device"
    );
    let teardown_before = teardown_totals();
    drop(backend);
    let teardown_after = teardown_totals();
    assert_eq!(teardown_after.quiesced_ok, teardown_before.quiesced_ok + 1);
    assert!(
        teardown_after.released_buffers > teardown_before.released_buffers + 1,
        "the two loose buffers are destroyed at teardown"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the submit-failure path must raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] vkQueueSubmit failure: submitter released without a fence wait, \
         loose buffers destroyed at teardown"
    );
}

/// `vkEndCommandBuffer` fails after recording but before submit: the
/// recording's references are proof-equivalently released and only the
/// submitter's handles are quarantined. The device itself stays usable.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn an_end_recording_failure_quarantines_only_the_submitter() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let payload = pattern(PAYLOAD_BYTES);
    let staging_before = backend.staging_stats();
    let dst = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-end-dst",
        ))
        .expect("destination buffer");
    backend.faults().arm(FaultPoint::EndRecording, 0);
    let error = backend
        .upload(&dst, &payload)
        .expect_err("the injected end-recording failure must fail the upload");
    assert_eq!(backend.faults().fired_at(), Some(FaultPoint::EndRecording));
    assert!(
        !backend.vulkan_device().is_lost(),
        "a recording that never reached the queue does not poison the device"
    );
    let quarantine = backend.quarantine_stats();
    assert_eq!(quarantine.entries, 1);
    assert_eq!(quarantine.pending_submissions, 0);
    assert_eq!(quarantine.retained_submitters, 1);
    assert_eq!(quarantine.retained_buffers, 0);
    assert_eq!(quarantine.loose_buffers, 0);
    assert_eq!(quarantine.in_flight_buffers, 0);
    let start = Instant::now();
    let reclaim = backend.reclaim_quarantine();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "a proof pass must not wait on a never-submitted fence"
    );
    assert_eq!(
        reclaim,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 0,
            pending: 0,
        }
    );
    let staging_after = backend.staging_stats();
    assert_eq!(
        staging_after.pooled_blocks,
        staging_before.pooled_blocks + 1,
        "the staging block returned to the pool on the healthy device"
    );
    let memory_before = backend.memory_stats();
    drop(dst);
    assert_eq!(
        backend.memory_stats().free_count,
        memory_before.free_count + 1,
        "the destination range is freed normally"
    );
    drop(backend);
    assert_eq!(
        validation_counts(),
        counts_before,
        "the end-recording path must raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] end-recording failure: submitter released, device {}",
        if error.to_string().contains("injected") {
            "not poisoned"
        } else {
            "state checked"
        }
    );
}

/// The critical shared-buffer case: two per-thread submitters bind one buffer.
/// The first fails, the second is still in flight, so the first reclaim must
/// retain the shared id. The second submission then retires (its ticket joins
/// the existing quarantine) and the next proof releases the shared buffer once.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_shared_buffer_survives_the_first_proof_and_frees_once_on_the_next() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let payload = pattern(PAYLOAD_BYTES);
    let shared = backend
        .alloc_buffer(&BufferDesc::device_storage(
            PAYLOAD_BYTES as u64,
            "quarantine-shared",
        ))
        .expect("shared buffer");
    let lease_a = backend
        .staging()
        .acquire(PAYLOAD_BYTES as u64)
        .expect("lease a");
    lease_a.write(&payload).expect("write a");
    let lease_b = backend
        .staging()
        .acquire(PAYLOAD_BYTES as u64)
        .expect("lease b");
    lease_b.write(&payload).expect("write b");
    let mut submitter_a = Submitter::new(backend.vulkan_device()).expect("submitter a");
    let mut submitter_b = Submitter::new(backend.vulkan_device()).expect("submitter b");
    let token_a = submitter_a
        .record(|recorder| {
            recorder.copy_buffer(lease_a.buffer(), 0, &shared, 0, PAYLOAD_BYTES as u64)?;
            Ok(())
        })
        .expect("record a");
    let token_b = submitter_b
        .record(|recorder| {
            recorder.copy_buffer(lease_b.buffer(), 0, &shared, 0, PAYLOAD_BYTES as u64)?;
            Ok(())
        })
        .expect("record b");

    // A fails before its wait; B is still an unproven, live submission.
    backend.faults().arm(FaultPoint::FenceUnknown, 0);
    token_a.wait().expect_err("A must fail");
    let pending = backend.quarantine_stats();
    assert_eq!(pending.entries, 1);
    assert_eq!(pending.retained_buffers, 2, "shared + lease a");
    assert_eq!(
        pending.in_flight_buffers, 3,
        "shared has two references; each lease one"
    );
    let memory = backend.memory_stats();
    let destroyed_before = destroyed_buffer_total();

    // First proof: A's entry is released, lease A is swept, but the shared id
    // is still referenced by B and must stay owned.
    let first = backend.reclaim_quarantine();
    assert_eq!(
        first,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 1,
            pending: 0,
        },
        "only A's private staging block may be freed"
    );
    assert_eq!(
        destroyed_buffer_total(),
        destroyed_before + 1,
        "the shared buffer must not be destroyed by A's proof"
    );
    let after_first = backend.quarantine_stats();
    assert_eq!(after_first.entries, 0);
    assert_eq!(
        after_first.retained_buffers, 1,
        "the shared buffer is still owned"
    );
    assert_eq!(
        after_first.in_flight_buffers, 2,
        "shared (B's reference) and lease B are still live"
    );
    assert_eq!(
        backend.memory_stats().free_count,
        memory.free_count + 1,
        "only the private staging range was freed"
    );

    // B is still in flight. Its implicit wait on the poisoned device fails, so
    // it retires and its ticket joins the already-quarantined shared id.
    drop(token_b);
    let joined = backend.quarantine_stats();
    assert_eq!(joined.entries, 1, "B's entry joined the quarantine");
    assert_eq!(
        joined.retained_buffers, 2,
        "the shared id is not duplicated; lease B is the second buffer"
    );
    assert_eq!(joined.in_flight_buffers, 2);

    // The next proof waits on B's fence and releases the last reference.
    let second = backend.reclaim_quarantine();
    assert_eq!(
        second,
        ReclaimReport {
            entries: 1,
            submitters: 1,
            buffers: 2,
            pending: 0,
        },
        "shared + lease B are freed by the second proof"
    );
    assert_eq!(
        destroyed_buffer_total(),
        destroyed_before + 3,
        "lease A, shared and lease B are each destroyed once"
    );
    let after_second = backend.quarantine_stats();
    assert_eq!(after_second.retained_buffers, 0);
    assert_eq!(after_second.in_flight_buffers, 0);
    assert_eq!(
        backend.memory_stats().free_count,
        memory.free_count + 3,
        "three ranges freed, no double free"
    );

    let third = backend.reclaim_quarantine();
    assert_eq!(third, ReclaimReport::default(), "nothing left to prove");
    assert_eq!(
        destroyed_buffer_total(),
        destroyed_before + 3,
        "a third reclaim frees nothing"
    );
    // The caller still holds `shared` and both leases; their drops are
    // tombstones and must not free anything again.
    drop(lease_a);
    drop(lease_b);
    drop(shared);
    assert_eq!(
        destroyed_buffer_total(),
        destroyed_before + 3,
        "a surviving Buffer/lease is a tombstone after the quarantine freed it"
    );
    drop(submitter_a);
    drop(submitter_b);
    drop(backend);
    assert_eq!(
        validation_counts(),
        counts_before,
        "the shared-buffer path must raise no validation message"
    );
    println!(
        "[fn3vk][quarantine] shared buffer: retained across A's proof, freed once after B's proof \
         (3 buffer destroys, 3 allocator frees)"
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
    let Some(PendingUpload { backend, .. }) = pending_upload(FaultPoint::FenceUnknown) else {
        return;
    };
    assert_eq!(
        backend.quarantine_stats().entries,
        1,
        "teardown must start with a pending entry"
    );
    assert_eq!(
        backend.quarantine_stats().pending_submissions,
        1,
        "the pending entry is a submitted, unproven fence"
    );
    let destroyed_before = destroyed_buffer_total();
    let chunks_before = destroyed_chunk_total();
    let instances_before = destroyed_instance_total();
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
        destroyed_buffer_total(),
        destroyed_before + 2,
        "quiesce-then-destroy frees exactly the two pending buffers"
    );
    assert!(
        destroyed_chunk_total() > chunks_before,
        "the allocator's chunks are freed after the quiesce"
    );
    assert_eq!(
        destroyed_instance_total(),
        instances_before + 1,
        "the instance is destroyed after the device"
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

/// The synthetic idle failure fires inside the backend teardown, with a
/// genuinely submitted entry still pending. Nothing may be destroyed: the
/// buffer, chunk and instance destroy counters must all be unchanged, and the
/// leaked entry stays owned so validation would flag an invalid destroy.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_failed_teardown_quiesce_leaks_pending_resources() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(PendingUpload { backend, .. }) = pending_upload(FaultPoint::FenceUnknown) else {
        return;
    };
    let pending = backend.quarantine_stats();
    assert_eq!(pending.entries, 1);
    assert_eq!(
        pending.pending_submissions, 1,
        "the leak test needs a genuinely submitted, unproven entry"
    );
    assert_eq!(pending.retained_submitters, 1);
    assert_eq!(pending.retained_buffers, 2);
    assert_eq!(pending.in_flight_buffers, 2);
    // The injected idle failure fires inside the backend's teardown, after the
    // pending entry exists. The injector is gone with the backend, so the
    // evidence is the teardown counter, not `fired_at()`.
    backend.faults().arm(FaultPoint::Idle, 0);
    let destroyed_buffers_before = destroyed_buffer_total();
    let destroyed_chunks_before = destroyed_chunk_total();
    let destroyed_instances_before = destroyed_instance_total();
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
        destroyed_buffer_total(),
        destroyed_buffers_before,
        "no vkDestroyBuffer may run on the leak path"
    );
    assert_eq!(
        destroyed_chunk_total(),
        destroyed_chunks_before,
        "no vkFreeMemory chunk may be freed on the leak path"
    );
    assert_eq!(
        destroyed_instance_total(),
        destroyed_instances_before,
        "vkDestroyInstance must not run over the leaked device"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "leaking must not raise validation messages either"
    );
    println!(
        "[fn3vk][quarantine] teardown leak: {} entr(y/ies), {} buffer(s) leaked deliberately after \
         quiesce failure; buffer/chunk/instance destroys 0",
        teardown_after.leaked_entries - teardown_before.leaked_entries,
        teardown_after.leaked_buffers - teardown_before.leaked_buffers
    );
}

// ---------------------------------------------------------------------------
// 5. Injected lane: a failed second wait-idle is still recorded.
// ---------------------------------------------------------------------------

/// Arm the idle fault with `skip = 1`, so the backend's first quiesce succeeds
/// and the `Device::drop` retry fails after `teardown_recorded` is already set.
/// The failure must appear in `teardown_totals()` and the instance must be
/// leaked with the device.
#[cfg(feature = "vulkan-fault-injection")]
#[test]
fn a_second_failed_quiesce_is_recorded_and_leaks_the_instance() {
    let _lock = gpu_lock();
    let counts_before = validation_counts();
    let Some(backend) = open_physical() else {
        return;
    };
    let buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(1024, "quarantine-second-idle"))
        .expect("buffer");
    drop(buffer);
    let destroyed_chunks_before = destroyed_chunk_total();
    let destroyed_instances_before = destroyed_instance_total();
    let teardown_before = teardown_totals();
    // First idle event: the backend teardown (skipped). Second: Device::drop.
    backend.faults().arm(FaultPoint::Idle, 1);
    drop(backend);
    let teardown_after = teardown_totals();
    assert_eq!(
        teardown_after.quiesced_ok,
        teardown_before.quiesced_ok + 1,
        "the backend boundary quiesced successfully"
    );
    assert_eq!(
        teardown_after.quiesced_failed,
        teardown_before.quiesced_failed + 1,
        "the failed second wait-idle is recorded even though teardown_recorded was set"
    );
    assert_eq!(
        destroyed_chunk_total(),
        destroyed_chunks_before,
        "the failed second quiesce must not free chunks"
    );
    assert_eq!(
        destroyed_instance_total(),
        destroyed_instances_before,
        "the failed second quiesce must not destroy the instance"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the retry leak must not raise validation messages"
    );
    println!(
        "[fn3vk][quarantine] second wait-idle failure: quiesced_ok +{}, quiesced_failed +{}, \
         chunks and instance leaked",
        teardown_after.quiesced_ok - teardown_before.quiesced_ok,
        teardown_after.quiesced_failed - teardown_before.quiesced_failed
    );
}
