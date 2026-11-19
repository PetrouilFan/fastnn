//! FN3-16 (#62): the Vulkan half of the end-to-end gate.
//!
//! Every assertion here is about a *physical* device: the run is skipped (with
//! a printed reason) when no non-software compute adapter is present, and
//! `FN3_VK_REQUIRE_PHYSICAL=1` turns that skip into a failure so a CI lane
//! cannot mistake "no GPU" for "passed". Selection is by UUID because the
//! loader enumerates a software ICD next to the real GPU.
//!
//! # What is proven
//!
//! 1. The **production** path — `Session::with_runtime` with no test adapter —
//!    builds a plan for the pinned graph and runs prefill at `past = 0`.
//! 2. Every instruction of every step leaves a residency record whose
//!    `proves_device_execution()` is true; the record count equals the plan's
//!    instruction count, `input_readbacks == 0`, and every record states
//!    `host_staged = true` (alpha tensor storage is host memory on every
//!    device, so the device path stages through it).
//! 3. A model prepared for Vulkan is never executed by the CPU, and an
//!    explicit Vulkan request on a host with no usable driver fails with an
//!    actionable error instead of running on the CPU.
//! 4. The validation layer reports zero errors and zero warnings for the run.

use std::sync::Mutex;

use fastnn::inference::backend::vulkan::instance::validation_counts;
use fastnn::inference::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::inference::device::DeviceRequest;
use fastnn::inference::model::SessionOptions;
use fastnn::inference::{Fn3Error, Session};

use super::harness::*;

/// Serialises physical-GPU use and the validation-count deltas.
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

/// The pinned physical adapter, selected by UUID when one is set.
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
                 present{}",
                match pinned_uuid() {
                    Some(uuid) => format!(" (pinned uuid {uuid})"),
                    None => String::new(),
                }
            );
        }
        eprintln!(
            "[fn3vk][skip] no physical Vulkan device with a compute queue is present; the device \
             gate must not run on a software ICD"
        );
        return None;
    };
    let backend = match VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    ) {
        Ok(backend) => backend,
        Err(error) => {
            if require_physical() {
                panic!(
                    "FN3_VK_REQUIRE_PHYSICAL=1 but adapter {index} could not be opened: {error}"
                );
            }
            eprintln!("[fn3vk][skip] could not open adapter {index}: {error}");
            return None;
        }
    };
    let adapter = backend.adapter();
    eprintln!(
        "[fn3vk][device] name=\"{}\" uuid={} api={} driver={} validation={} software={}",
        adapter.name,
        adapter.uuid_string(),
        fastnn::inference::backend::vulkan::device_info::format_version(adapter.api_version),
        adapter.driver_version_string(),
        backend.validation_enabled(),
        backend.is_software()
    );
    assert!(!backend.is_software(), "the device gate needs a real GPU");
    assert!(
        backend.validation_enabled(),
        "the gate runs with the validation layer enabled in code"
    );
    Some(backend)
}

/// Assert one proven execution per byte-producing instruction.
///
/// A plan can contain a zero-element instruction (the pinned prefill's
/// `ConstantOfShape` for the "future" half of the causal mask at
/// `total == seq`). It has no byte to write, so it is not an execution; the
/// census names it instead of pretending it dispatched. Every instruction that
/// does produce bytes must appear exactly once, per op, which is what the
/// per-op histogram check below enforces.
/// Which storage model the assertions describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageMode {
    /// Device-resident slots and KV (the #72 default).
    Resident,
    /// Host-staged slots and KV (the alpha path, `SessionOptions::staged()`).
    Staged,
}

fn assert_device_execution(
    backend: &VulkanBackend,
    census: &Census,
    label: &str,
    mode: StorageMode,
    steps: u64,
) {
    let ledger = backend.residency();
    let records = ledger.records();
    assert_eq!(
        records.len(),
        census.with_bytes,
        "{label}: one residency record per byte-producing instruction"
    );
    assert_eq!(
        census.total,
        census.with_bytes + census.zero_byte(),
        "{label}: every instruction is either byte-producing or a zero-element no-op"
    );
    let mut observed: std::collections::BTreeMap<fastnn::inference::OpId, usize> =
        std::collections::BTreeMap::new();
    for (op, count) in ledger.summary().ops {
        observed.insert(op, count as usize);
    }
    assert_eq!(
        observed, census.ops,
        "{label}: a byte-producing instruction was skipped or repeated"
    );
    let mut bindings = 0usize;
    let mut device_local = 0usize;
    let mut upload_bytes = 0u64;
    let mut readback_bytes = 0u64;
    let mut host_ns = 0u64;
    let mut device_ns = 0u64;
    let mut timed = 0usize;
    let mut any_resident = false;
    for record in &records {
        assert_eq!(record.backend, "vulkan", "{label}: record backend");
        assert_eq!(record.device, backend.device_id(), "{label}: record device");
        assert!(
            record.submissions >= 1 || record.batch.is_some(),
            "{label}: record {} neither submitted nor joined a batch",
            record.sequence
        );
        match mode {
            StorageMode::Resident => assert!(
                record.batch.is_some(),
                "{label}: record {} was not batched into its step",
                record.sequence
            ),
            StorageMode::Staged => assert!(
                record.batch.is_none(),
                "{label}: the staged arm must submit per op"
            ),
        }
        assert!(
            record.workgroups.iter().all(|count| *count > 0),
            "{label}: record {} dispatched no workgroup",
            record.sequence
        );
        assert!(
            record.bindings.iter().any(|binding| binding.device_local),
            "{label}: record {} has no device-local binding",
            record.sequence
        );
        assert!(
            !record.inputs_read_back,
            "{label}: record {} read an input back to the host",
            record.sequence
        );
        match mode {
            // The staged arm moves every operand through the host, exactly as
            // the alpha did, and says so on every record.
            StorageMode::Staged => assert!(
                record.host_staged,
                "{label}: the staged arm must stage every record (record {})",
                record.sequence
            ),
            // The resident arm binds device buffers; a record is host-staged
            // only when it consumes a graph input or a (host) constant.
            StorageMode::Resident => {
                any_resident = any_resident || !record.host_staged;
            }
        }
        assert!(
            record.proves_device_execution(),
            "{label}: record {} does not prove device execution",
            record.sequence
        );
        bindings += record.bindings.len();
        device_local += record
            .bindings
            .iter()
            .filter(|binding| binding.device_local)
            .count();
        upload_bytes += record.upload_bytes;
        readback_bytes += record.readback_bytes;
        host_ns += record.host_ns;
        if let Some(nanoseconds) = record.gpu_ns {
            device_ns += nanoseconds;
            timed += 1;
        }
    }
    assert_eq!(
        device_local, bindings,
        "{label}: every binding is device-local"
    );
    let summary = ledger.summary();
    assert_eq!(
        summary.executions, census.with_bytes as u64,
        "{label}: executions"
    );
    assert_eq!(
        summary.proven_device_executions, census.with_bytes as u64,
        "{label}: proven executions"
    );
    assert_eq!(summary.input_readbacks, 0, "{label}: input readbacks");
    match mode {
        // One command buffer per step: the plan's dispatches, the past-prefix
        // copies and the step's inputs all travel in one submission.
        StorageMode::Resident => assert_eq!(
            summary.submissions, steps,
            "{label}: one queue submission per step"
        ),
        StorageMode::Staged => assert!(
            summary.submissions >= census.with_bytes as u64,
            "{label}: submissions"
        ),
    }
    // ---- the storage-model assertions ------------------------------------
    let stats = backend.residency_stats();
    let resident_binds: u64 = stats.resident_binds.iter().sum();
    assert_eq!(
        stats.late_persistent_uploads, 0,
        "{label}: a prepared weight reached the host staging path"
    );
    match mode {
        StorageMode::Resident => {
            // Every step downloads exactly one row of logits per batch entry:
            // that is the only device -> host byte in a resident step. The
            // counter is the authoritative one: a resident download is an
            // explicit submission, not an op record.
            let expected_readback = steps * VOCAB as u64 * 4;
            assert_eq!(
                stats.readback_bytes, expected_readback,
                "{label}: a resident step may read back only the final logits ({} B read)",
                stats.readback_bytes
            );
            // Uploads are the per-step graph inputs plus host constants; a
            // weight or an intermediate would blow past this by orders of
            // magnitude (the alpha's per-step upload is ~80 MB). The counter
            // measures logical operand bytes and the record measures allocated
            // transfer bytes (padded to 4, plus parameter buffers), so both are
            // checked against the same budget.
            let budget = steps * 2 * 1024 * 1024;
            let staged_uploads: u64 = stats.staged_upload_bytes.iter().sum();
            assert!(
                upload_bytes <= budget && staged_uploads <= budget,
                "{label}: resident uploads {upload_bytes} B / {staged_uploads} B exceed the \
                 {budget} B input-and-constant budget"
            );
            assert!(
                any_resident,
                "{label}: no record was fully device-resident; the resident path is not active"
            );
            assert!(
                resident_binds > 0,
                "{label}: no operand was bound from device-resident storage"
            );
            assert!(
                stats.persistent_hits.iter().sum::<u64>() > 0,
                "{label}: no weight came from the persistent table"
            );
        }
        StorageMode::Staged => {
            assert!(
                !any_resident,
                "{label}: the staged arm bound a device-resident operand"
            );
            assert_eq!(
                resident_binds, 0,
                "{label}: the staged arm bound resident storage"
            );
            let staged_uploads: u64 = stats.staged_upload_bytes.iter().sum();
            assert!(
                staged_uploads > steps * 20 * 1024 * 1024,
                "{label}: the staged arm should move every intermediate again, uploaded \
                 {staged_uploads} B"
            );
            assert!(
                stats.readback_bytes > steps * 10 * 1024 * 1024,
                "{label}: the staged arm should read every output back, read {} B",
                stats.readback_bytes
            );
            assert!(
                stats.persistent_hits.iter().sum::<u64>() > 0,
                "{label}: even the staged arm binds persistent weights"
            );
        }
    }

    println!(
        "[fn3vk][residency] {label}: mode {mode:?} executions {} submissions {} proven {} \
         input_readbacks {} bindings {}/{} device-local, host_staged {} upload {} B, \
         readback {} B, resident_binds {resident_binds}, steps {steps}; plan {} \
         instructions ({} byte-producing, {} zero-element no-op{:?})",
        summary.executions,
        summary.submissions,
        summary.proven_device_executions,
        summary.input_readbacks,
        device_local,
        bindings,
        if mode == StorageMode::Staged {
            "true"
        } else {
            "mixed"
        },
        upload_bytes,
        readback_bytes,
        census.total,
        census.with_bytes,
        census.zero_byte(),
        census.zero_byte_ops
    );
    // Smoke timing, not a benchmark: a debug build with one fence wait per op
    // and a host round trip per operand. #50/#67 own real performance work.
    println!(
        "[fn3vk][timing] {label}: host submission wall {:.3} ms, device {} over {timed}/{} \
         records",
        host_ns as f64 / 1e6,
        if timed == 0 {
            "timestamps unavailable".to_string()
        } else {
            format!("{:.3} ms", device_ns as f64 / 1e6)
        },
        records.len()
    );
}

// ---------------------------------------------------------------------------
// 1. Prefill + four cached decode steps on the device
// ---------------------------------------------------------------------------

#[test]
fn vulkan_prefill_and_four_cached_decode_steps_prove_device_execution() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let (model, envelope) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();

    // Short prefills: the first two committed points.
    for seq in [1u64, 4] {
        backend.residency().clear();
        backend.reset_residency_stats();
        let outcome = check_prefill(
            &*backend,
            &import,
            &model,
            &envelope,
            &tokens,
            seq,
            SessionOptions::default(),
        );
        assert_device_execution(
            &backend,
            &outcome.census,
            &format!("prefill p{seq}"),
            StorageMode::Resident,
            1,
        );
    }

    // The production 13 + 4 chain. No test adapter is involved: the session's
    // own backend plans every step through `Backend::admits`.
    backend.residency().clear();
    backend.reset_residency_stats();
    let chain = run_chain(
        &*backend,
        &import,
        &model,
        &envelope,
        &tokens,
        FIXTURE_CONTEXT,
        SessionOptions::default(),
    );
    let mut expected = chain.prefill_census.clone();
    for step in &chain.steps {
        expected = expected.merge(&step.census);
    }
    assert_device_execution(
        &backend,
        &expected,
        "prefill p13 + four cached decode steps",
        StorageMode::Resident,
        1 + chain.steps.len() as u64,
    );
    assert_eq!(
        chain
            .steps
            .iter()
            .map(|step| step.extent)
            .collect::<Vec<_>>(),
        vec![14, 15, 16, 17]
    );
    assert_eq!(chain.kv_bytes, 34_816);
    println!(
        "[fn3vk][gate] prefill 13 + decode 14..17: {} proven executions over {} plan \
         instructions, plan misses {} evictions {}",
        expected.with_bytes, expected.total, chain.plan_misses, chain.plan_evictions
    );

    // A second session over the same prepared model produces the same bytes,
    // independently.
    backend.residency().clear();
    backend.reset_residency_stats();
    let mut second = Session::with_runtime(&model, &import, &*backend).expect("second session");
    let prefill = second.prefill(&tokens[..PROMPT]).expect("second prefill");
    assert_eq!(
        prefill.as_f32_slice().unwrap(),
        &chain.prefill[..],
        "two sessions over one model must produce identical prompt logits"
    );
    let logits = second
        .decode(&[tokens[PROMPT]])
        .expect("second session decode");
    assert_eq!(
        logits.as_f32_slice().unwrap(),
        &chain.steps[0].logits[..],
        "two sessions must produce identical decode logits"
    );
    assert_eq!(second.cache_len(), 14);
    let second_expected = census(&import, &*backend, &envelope, PROMPT as u64, 0).merge(&census(
        &import,
        &*backend,
        &envelope,
        1,
        PROMPT as u64,
    ));
    assert_device_execution(
        &backend,
        &second_expected,
        "second session prefill + decode",
        StorageMode::Resident,
        2,
    );

    // ---- the staged arm: the alpha path, still green ---------------------
    //
    // `SessionOptions::staged()` keeps every value slot and the whole KV cache
    // in host memory, so the same kernels stage every operand. This is the
    // comparison arm #72 promises not to delete, and it runs the same fixture
    // chain under the same per-op evidence rules.
    backend.residency().clear();
    backend.reset_residency_stats();
    let staged = run_chain(
        &*backend,
        &import,
        &model,
        &envelope,
        &tokens,
        FIXTURE_CONTEXT,
        SessionOptions::staged(),
    );
    let mut staged_expected = staged.prefill_census.clone();
    for step in &staged.steps {
        staged_expected = staged_expected.merge(&step.census);
    }
    assert_device_execution(
        &backend,
        &staged_expected,
        "staged arm: prefill p13 + four cached decode steps",
        StorageMode::Staged,
        1 + staged.steps.len() as u64,
    );
    assert_eq!(
        staged
            .steps
            .iter()
            .map(|step| step.extent)
            .collect::<Vec<_>>(),
        vec![14, 15, 16, 17]
    );
    assert_eq!(staged.kv_bytes, 34_816);
    assert_eq!(
        staged.prefill, chain.prefill,
        "the staged arm must produce the same logits as the resident arm"
    );
    for (resident, staged) in chain.steps.iter().zip(staged.steps.iter()) {
        assert_eq!(
            resident.logits, staged.logits,
            "resident and staged arms disagree on a decode step"
        );
    }
    println!(
        "[fn3vk][gate] staged arm: {} proven executions, byte-identical logits to the resident arm",
        staged_expected.with_bytes
    );

    let (errors, warnings) = validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "absolute validation counts must be zero"
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the run changed the validation counters; see the [fn3vk][validation] lines"
    );
    println!("[fn3vk][validation] absolute errors {errors} warnings {warnings}, delta 0/0");
}

// ---------------------------------------------------------------------------
// 2. A longer context, bounded
// ---------------------------------------------------------------------------

/// Prefill 13 then decode to a configurable context (default 32) on the device.
///
/// The point is that nothing about the device path is decode-only or
/// test-scaffolded: the same production session that ran 14..17 runs 18..32,
/// with a constant KV allocation and one proven execution per instruction.
#[test]
fn vulkan_long_chain_to_a_longer_context_stays_bounded() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let counts_before = validation_counts();
    let context = long_context();
    assert!(
        context > FIXTURE_CONTEXT,
        "FN3_E2E_CONTEXT={context} must exceed the fixture chain"
    );
    let device = backend.device_id();
    let (model, envelope) = prepared(&import, device, context);
    let tokens = token_ids();

    backend.residency().clear();
    backend.reset_residency_stats();
    let chain = run_chain(
        &*backend,
        &import,
        &model,
        &envelope,
        &tokens,
        context,
        SessionOptions::default(),
    );
    let mut expected = chain.prefill_census.clone();
    for step in &chain.steps {
        expected = expected.merge(&step.census);
    }
    assert_device_execution(
        &backend,
        &expected,
        &format!("chain to context {context}"),
        StorageMode::Resident,
        1 + chain.steps.len() as u64,
    );

    let extents: Vec<u64> = chain.steps.iter().map(|step| step.extent).collect();
    assert_eq!(extents, (14..=context).collect::<Vec<_>>());
    assert_eq!(
        chain.kv_bytes,
        LAYERS as u64 * 2 * KV_HEADS * context * HEAD_DIM * 4
    );
    assert_eq!(chain.plan_evictions, 0);
    assert!(chain.plan_cache_len <= 64);
    // The last step is not one of the committed decode fixtures; it has no
    // reference, so assert the properties the contract does guarantee.
    let last = chain.steps.last().expect("at least one decode step");
    assert_eq!(last.logits.len(), VOCAB);
    for (index, &value) in last.logits.iter().enumerate() {
        assert!(
            value.is_finite(),
            "context {context}: logit {index} is {value}"
        );
    }
    let argmax = argmax(&last.logits);
    assert!(argmax < VOCAB);
    println!(
        "[fn3vk][gate] long chain to {context}: extents 14..={context}, kv_bytes {} constant, \
         plan misses {} evictions {} cached {}, last argmax {}, {} proven executions over {} \
         instructions",
        chain.kv_bytes,
        chain.plan_misses,
        chain.plan_evictions,
        chain.plan_cache_len,
        argmax,
        expected.with_bytes,
        expected.total
    );

    let (errors, warnings) = validation_counts();
    assert_eq!((errors, warnings), (0, 0), "absolute validation counts");
    assert_eq!(validation_counts(), counts_before, "validation delta");
    println!("[fn3vk][validation] absolute errors {errors} warnings {warnings}, delta 0/0");
}

// ---------------------------------------------------------------------------
// 3. No usable driver: an explicit request fails, it does not fall back
// ---------------------------------------------------------------------------

/// Marker the child process prints once it has proved the failure is clean.
const NO_DRIVER_PROOF: &str = "[fn3e2e][no-driver]";
const NO_DRIVER_CHILD: &str = "FN3_E2E_NO_DRIVER_CHILD";
const NO_DRIVER_TEST: &str =
    "gpu::an_explicit_vulkan_request_without_a_usable_driver_fails_actionably";

#[test]
fn an_explicit_vulkan_request_without_a_usable_driver_fails_actionably() {
    if std::env::var(NO_DRIVER_CHILD).as_deref() == Ok("1") {
        // This process has no reachable ICD (the parent points the loader at a
        // nonexistent manifest). The only honest outcome is a typed failure.
        let available = fastnn::inference::device::available_devices();
        assert!(
            available.iter().all(|info| info.id.is_cpu()),
            "the child must not see a Vulkan adapter: {available:?}"
        );
        let error = VulkanBackend::open_with(
            &DeviceRequest::Vulkan { index: None },
            VulkanConfig::validated(),
        )
        .expect_err("with no ICD an explicit Vulkan request must fail");
        assert!(
            matches!(error, Fn3Error::DeviceUnavailable(_)),
            "expected DeviceUnavailable, found {error:?}"
        );
        let text = error.to_string();
        assert!(
            text.contains("vulkan"),
            "the error must name the request: {text}"
        );
        assert!(
            text.contains("ICD") || text.contains("loader") || text.contains("driver"),
            "the error must say how to fix it: {text}"
        );
        println!("{NO_DRIVER_PROOF} request=vulkan(index=None) error={text}");
        return;
    }

    let exe = std::env::current_exe().expect("the test binary path");
    let output = std::process::Command::new(exe)
        .args(["--exact", NO_DRIVER_TEST, "--nocapture", "--test-threads=1"])
        .env(NO_DRIVER_CHILD, "1")
        .env("VK_ICD_FILENAMES", "/nonexistent/fn3-e2e-no-icd.json")
        .env("VK_DRIVER_FILES", "/nonexistent/fn3-e2e-no-icd.json")
        .output()
        .expect("spawn the test binary");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "the no-driver child failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    let proof = stdout
        .lines()
        .chain(stderr.lines())
        .find(|line| line.contains(NO_DRIVER_PROOF))
        .unwrap_or_else(|| {
            panic!("the child passed without printing its proof:\n{stdout}\n{stderr}")
        });
    println!("no-driver: {proof}");
}

// ---------------------------------------------------------------------------
// 4. A bogus explicit request in the normal process
// ---------------------------------------------------------------------------

/// Even with a driver present, an explicit request that cannot be satisfied
/// fails with the request and the available adapters in the message. This is
/// the same rule the no-driver child proves at the loader level, without
/// changing the process environment.
#[test]
fn an_unsatisfiable_explicit_request_names_the_request_and_the_devices() {
    let _guard = gpu_lock();
    let Some(_backend) = open_physical() else {
        return;
    };
    let before = validation_counts();

    let error = VulkanBackend::open_with(
        &DeviceRequest::Vulkan {
            index: Some(u32::MAX),
        },
        VulkanConfig::validated(),
    )
    .expect_err("no adapter has that index");
    assert!(
        matches!(error, Fn3Error::DeviceUnavailable(_)),
        "expected DeviceUnavailable, found {error:?}"
    );
    let text = error.to_string();
    assert!(text.contains("vulkan"), "{text}");
    assert!(
        text.contains("available devices") || text.contains("no Vulkan device"),
        "the error must list what is available or why nothing is: {text}"
    );

    let error = VulkanBackend::open_with(
        &DeviceRequest::VulkanNamed {
            name: "FN3 e2e: an adapter that does not exist".to_string(),
        },
        VulkanConfig::validated(),
    )
    .expect_err("no adapter has that name");
    let text = error.to_string();
    assert!(
        text.contains("FN3 e2e"),
        "the error must name the request: {text}"
    );

    // Resolution itself never yields the CPU for an explicit request.
    let available = fastnn::inference::device::available_devices();
    for request in [
        DeviceRequest::Vulkan { index: None },
        DeviceRequest::Vulkan {
            index: Some(u32::MAX),
        },
        DeviceRequest::VulkanNamed {
            name: "FN3 e2e: an adapter that does not exist".to_string(),
        },
    ] {
        if let Ok(device) = request.resolve(&available) {
            assert!(!device.is_cpu(), "{request} resolved to the CPU");
        }
    }
    assert_eq!(validation_counts(), before, "a refused request ran nothing");
    println!("explicit request: refused without falling back; available {available:?}");
}

// ---------------------------------------------------------------------------
// 5. Zero-element operands at the buffer layer
// ---------------------------------------------------------------------------

/// A zero-element tensor is representable and unaddressable.
///
/// `past = 0` needs an empty concatenation input. Vulkan forbids a size-0
/// `vkCreateBuffer`, so the platform layer reserves one aligned word and keeps
/// the *logical* length at zero. This test exercises that allocation directly:
/// it must report `size == 0`, accept an empty transfer as a no-op, refuse any
/// non-empty access, and return every byte and handle on drop.
#[test]
fn a_zero_element_buffer_is_real_but_unaddressable() {
    use fastnn::inference::backend::vulkan::context::BufferDesc;

    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let before = backend.memory_stats();
    let handles_before = backend.outstanding_handles();

    let buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(0, "fn3.e2e.empty"))
        .expect("a zero-element operand must be allocatable");
    assert_eq!(buffer.size(), 0, "the logical length stays zero");
    assert!(buffer.is_empty());
    let reserved = buffer.allocation_size();
    assert!(
        reserved >= 1,
        "the dummy reservation is a real binding: {reserved} bytes"
    );

    // Empty transfers are legal and move nothing.
    backend
        .upload(&buffer, &[])
        .expect("a zero-byte upload is a no-op");
    let mut empty: [u8; 0] = [];
    backend
        .download(&buffer, &mut empty)
        .expect("a zero-byte download is a no-op");

    // Any addressable access is refused, with both numbers in the message.
    let error = backend
        .upload(&buffer, &[0u8; 4])
        .expect_err("the buffer has no addressable byte");
    assert!(
        matches!(error, Fn3Error::CapacityExceeded { .. }),
        "expected CapacityExceeded, found {error:?}"
    );
    assert!(error.to_string().contains("vulkan upload"), "{error}");

    drop(buffer);
    let after = backend.memory_stats();
    assert_eq!(
        after.live_bytes, before.live_bytes,
        "the dummy reservation must return to the allocator"
    );
    assert_eq!(
        backend.outstanding_handles(),
        handles_before,
        "no buffer or memory handle may leak"
    );
    println!(
        "zero-element buffer: size 0, reserved {reserved} B, empty transfers ok, 4-byte upload \
         refused, live_bytes {} -> {}",
        before.live_bytes, after.live_bytes
    );
    assert_eq!(
        validation_counts(),
        counts_before,
        "the zero-length path must not give the validation layer anything to report"
    );
}
