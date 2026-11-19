//! FN3-14 (#60): Vulkan FP32 matmul conformance, evidence and device timing.
//!
//! Run with:
//!
//! ```bash
//! CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_matmul \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! # What these tests prove
//!
//! 1. **Fixture parity against the CPU reference.** The five matmul fixtures
//!    from `fixtures/parity/ops` (`2d_k3`, `gemm_k13_bias` with its transposed
//!    weight, `batched_4d`, `model_proj`, `tail_k65`) are executed through the
//!    real [`Backend::execute_op`] path on the GPU and compared elementwise
//!    with [`CpuBackend`] at the tolerance recorded in the fixture manifest.
//!    The measured `max_abs`/`mean_abs` are printed; they are the numbers the
//!    hand-off report quotes.
//! 2. **Real model geometry.** Prefill `[13,128] @ [128,384]`, the decode
//!    `lm_head` projection `[1,128] @ [128,50257]` (the dominant single op) and
//!    the attention score geometry `[1,2,17,64] @ [1,2,64,17]` are checked
//!    against the CPU reference at the scaled tolerances from
//!    `docs/reports/fn3-03-parity-fixtures.md`.
//! 3. **Tails and boundaries.** `K` at the 16-deep tile boundary (`1, 2, 3, 13,
//!    65, 127, 128, 129, 130`), `M`/`N` of 1, rank-3/rank-4 batches with
//!    one-sided and two-sided batch broadcast, transposed, offset-window,
//!    explicit zero-stride (broadcast) operands and the empty product (`K = 0`).
//! 4. **Honest rejection.** Rank-1 operands, a strided output, a broadcast
//!    (read-only) output, a zero-stride contraction and stray attributes are
//!    refused with a message naming FN3-14 (#60), never approximated.
//! 5. **Device evidence.** The residency record asserts
//!    `proves_device_execution()`, `inputs_read_back == false`,
//!    `input_readbacks == 0` and the recorded upload/readback byte counts.
//! 6. **A measured baseline.** The decode projection is dispatched
//!    device-resident (`alloc_buffer`/`upload`/`dispatch`/`download`) with no
//!    host staging, its result is checked against the CPU reference, and its
//!    device time and GFLOP/s are printed. That is a measurement on the named
//!    workstation GPU only, not a general performance claim.
//!
//! Physical-GPU tests skip with `[fn3vk][skip]` when no non-software adapter is
//! present; `FN3_VK_REQUIRE_PHYSICAL=1` turns a skip into a failure for evidence
//! runs. Selection is by UUID (`FN3_VK_DEVICE_UUID`), never by index, because
//! the loader also enumerates a software ICD.
//!
//! The tests serialise on `GPU_LOCK`: validation counts are process-global and
//! physical-GPU runs are serialised by the project's concurrency policy.

#![cfg(feature = "vulkan")]

mod fn3_fixtures;

use std::sync::{Arc, Mutex};

use fn3_fixtures::{Actual, Suite};

use fastnn::fn3::backend::cpu::CpuBackend;
use fastnn::fn3::backend::vulkan::instance::validation_counts;
use fastnn::fn3::backend::vulkan::kernels::matmul as matmul_kernel;
use fastnn::fn3::backend::vulkan::{
    adapters, BindingArg, BindingRole, BufferDesc, KernelInvocation, VulkanBackend, VulkanConfig,
};
use fastnn::fn3::backend::{Backend, OpParams, Support};
use fastnn::fn3::device::{DeviceId, DeviceRequest};
use fastnn::fn3::dtype::DType;
use fastnn::fn3::error::Fn3Error;
use fastnn::fn3::ops::OpId;
use fastnn::fn3::shape::{Shape, Strides};
use fastnn::fn3::storage::Storage;
use fastnn::fn3::tensor::{Tensor, TensorMeta};

// ---------------------------------------------------------------------------
// Device selection (mirrors tests/fn3_vulkan_smoke.rs, which #60 may not edit)
// ---------------------------------------------------------------------------

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

fn open_physical() -> Option<Arc<VulkanBackend>> {
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
            "[fn3vk][skip] no physical Vulkan device with a compute queue is present; this test \
             produced no physical-device evidence"
        );
        return None;
    };
    match VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    ) {
        Ok(backend) => {
            let adapter = backend.adapter();
            eprintln!(
                "[fn3vk][test] device=\"{}\" uuid={} api={} driver={} queue_family={} \
                 validate={} maxPushConstantsSize={} maxComputeSharedMemorySize={}",
                adapter.name,
                adapter.uuid_string(),
                fastnn::fn3::backend::vulkan::device_info::format_version(adapter.api_version),
                adapter.driver_version_string(),
                backend.vulkan_device().queue_family(),
                backend.validation_enabled(),
                backend.vulkan_device().limits().max_push_constants_size,
                backend
                    .vulkan_device()
                    .limits()
                    .max_compute_shared_memory_size,
            );
            Some(backend)
        }
        Err(error) if require_physical() => {
            panic!("FN3_VK_REQUIRE_PHYSICAL=1 but adapter {index} could not be opened: {error}")
        }
        Err(error) => {
            eprintln!("[fn3vk][skip] could not open physical adapter {index}: {error}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Operands, cases and comparison
// ---------------------------------------------------------------------------

/// One operand: a shape, the element strides it is addressed with, an element
/// offset, and the values of its *whole* storage allocation.
struct Operand {
    dims: Vec<u64>,
    strides: Vec<i64>,
    offset: usize,
    stored: Vec<f32>,
    /// Metadata built through `TensorMeta::broadcast`, which accepts zero
    /// strides for explicitly broadcast (read-only) axes.
    broadcast: bool,
}

impl Operand {
    fn contiguous(dims: &[u64], values: Vec<f32>) -> Operand {
        Operand {
            dims: dims.to_vec(),
            strides: row_major_strides(dims),
            offset: 0,
            stored: values,
            broadcast: false,
        }
    }

    /// A view of `dims` over a differently-stored allocation.
    fn strided(dims: &[u64], strides: &[i64], stored: Vec<f32>) -> Operand {
        Operand {
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            offset: 0,
            stored,
            broadcast: false,
        }
    }

    /// A read-only broadcast view: zero strides re-read one element.
    fn broadcast(dims: &[u64], strides: &[i64], stored: Vec<f32>) -> Operand {
        Operand {
            broadcast: true,
            ..Operand::strided(dims, strides, stored)
        }
    }

    fn tensor_on(&self, device: DeviceId) -> Tensor {
        let shape = Shape::from_dims(self.dims.iter().copied());
        let strides = Strides::from_elements(self.strides.clone());
        let meta = if self.broadcast {
            TensorMeta::broadcast(shape, strides, self.offset, DType::F32, device)
        } else {
            TensorMeta::strided(shape, strides, self.offset, DType::F32, device)
        }
        .unwrap_or_else(|error| panic!("operand metadata {:?}: {error}", self.dims));
        let storage =
            Storage::from_bytes(bytemuck::cast_slice(&self.stored)).expect("operand storage");
        Tensor::from_storage(storage, meta).expect("operand tensor")
    }
}

fn row_major_strides(dims: &[u64]) -> Vec<i64> {
    let mut strides = vec![0i64; dims.len()];
    let mut step = 1i64;
    for axis in (0..dims.len()).rev() {
        strides[axis] = step;
        step *= dims[axis] as i64;
    }
    strides
}

struct Case {
    label: String,
    lhs: Operand,
    rhs: Operand,
    out_dims: Vec<u64>,
    atol: f64,
    rtol: f64,
}

struct CaseResult {
    gpu: Vec<f32>,
    agreement: Agreement,
}

/// Measured elementwise difference.
#[derive(Debug, Clone, Copy)]
struct Agreement {
    elements: usize,
    max_abs: f64,
    mean_abs: f64,
}

impl Case {
    // A table-driven helper: eight named arguments (two shapes and two value
    // slices plus the output shape and tolerances) read better than a struct
    // built at each of the 41 call sites.
    #[allow(clippy::too_many_arguments)]
    fn contiguous(
        label: impl Into<String>,
        lhs_dims: &[u64],
        lhs_values: &[f32],
        rhs_dims: &[u64],
        rhs_values: &[f32],
        out_dims: &[u64],
        atol: f64,
        rtol: f64,
    ) -> Case {
        Case {
            label: label.into(),
            lhs: Operand::contiguous(lhs_dims, lhs_values.to_vec()),
            rhs: Operand::contiguous(rhs_dims, rhs_values.to_vec()),
            out_dims: out_dims.to_vec(),
            atol,
            rtol,
        }
    }

    /// A case whose operands need explicitly-strided or broadcast metadata.
    fn with_operands(
        label: impl Into<String>,
        lhs: Operand,
        rhs: Operand,
        out_dims: &[u64],
        atol: f64,
        rtol: f64,
    ) -> Case {
        Case {
            label: label.into(),
            lhs,
            rhs,
            out_dims: out_dims.to_vec(),
            atol,
            rtol,
        }
    }
}

fn zeros_on(device: DeviceId, dims: &[u64]) -> Tensor {
    let meta = TensorMeta::contiguous(Shape::from_dims(dims.iter().copied()), DType::F32, device)
        .expect("zeros metadata");
    let storage = Storage::zeroed(meta.required_bytes().expect("zeros bytes")).expect("zeros");
    Tensor::from_storage(storage, meta).expect("zeros tensor")
}

fn execute_matmul(
    backend: &dyn Backend,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
) -> Result<(), Fn3Error> {
    let lhs_view = lhs.as_view()?;
    let rhs_view = rhs.as_view()?;
    let mut out_view = out.as_mut_view()?;
    backend.execute_op(
        OpId::MatMul,
        &[lhs_view, rhs_view],
        std::slice::from_mut(&mut out_view),
        &OpParams::default(),
    )
}

/// Run one case on the GPU and on the CPU reference, compare, and report.
fn run_case(backend: &VulkanBackend, case: &Case) -> CaseResult {
    let device = backend.device_id();
    let lhs = case.lhs.tensor_on(device);
    let rhs = case.rhs.tensor_on(device);
    let mut out = zeros_on(device, &case.out_dims);
    execute_matmul(backend, &lhs, &rhs, &mut out)
        .unwrap_or_else(|error| panic!("{}: vulkan matmul failed: {error}", case.label));
    let gpu = out
        .as_f32_slice()
        .expect("gpu output is contiguous")
        .to_vec();

    let cpu_backend = CpuBackend::new();
    let lhs = case.lhs.tensor_on(DeviceId::Cpu);
    let rhs = case.rhs.tensor_on(DeviceId::Cpu);
    let mut expected = zeros_on(DeviceId::Cpu, &case.out_dims);
    execute_matmul(&cpu_backend, &lhs, &rhs, &mut expected)
        .unwrap_or_else(|error| panic!("{}: cpu matmul failed: {error}", case.label));
    let reference = expected.as_f32_slice().expect("cpu output");

    let agreement = compare(&case.label, &gpu, reference, case.atol, case.rtol);
    CaseResult { gpu, agreement }
}

fn compare(label: &str, actual: &[f32], reference: &[f32], atol: f64, rtol: f64) -> Agreement {
    assert_eq!(
        actual.len(),
        reference.len(),
        "{label}: element count differs"
    );
    let mut max_abs = 0.0f64;
    let mut total = 0.0f64;
    for (index, (&actual, &reference)) in actual.iter().zip(reference).enumerate() {
        assert!(
            actual.is_finite() == reference.is_finite(),
            "{label}: element {index} finiteness differs: gpu {actual} cpu {reference}"
        );
        let difference = (f64::from(actual) - f64::from(reference)).abs();
        max_abs = max_abs.max(difference);
        total += difference;
        assert!(
            difference <= atol + rtol * f64::from(reference).abs(),
            "{label}: element {index} gpu {actual} vs cpu {reference}, |diff|={difference:e} \
             exceeds {atol:e}+{rtol:e}|ref|"
        );
    }
    let agreement = Agreement {
        elements: reference.len(),
        max_abs,
        mean_abs: if reference.is_empty() {
            0.0
        } else {
            total / reference.len() as f64
        },
    };
    eprintln!(
        "[fn3vk][matmul] {label:58} elements={:6} max_abs={:e} mean_abs={:e} \
         tol={atol:e}+{rtol:e}|ref|",
        agreement.elements, agreement.max_abs, agreement.mean_abs
    );
    agreement
}

/// Compare a GPU result with the stored fixture payload, printing the same
/// measurement the CPU parity suite reports.
fn check_fixture(suite: &Suite, name: &str, actual: &[f32], dims: &[u64]) -> Agreement {
    let reference = suite.tensor(name);
    let agreement = reference
        .check(Actual::F32(actual), dims)
        .unwrap_or_else(|error| panic!("fixture {name}: {error}"));
    eprintln!(
        "[fn3vk][matmul] fixture {name:32} shape {dims:?} elements={:6} max_abs={:e} \
         mean_abs={:e} tolerance {}",
        agreement.elements,
        agreement.max_abs,
        agreement.mean_abs,
        reference.tolerance()
    );
    Agreement {
        elements: agreement.elements,
        max_abs: agreement.max_abs,
        mean_abs: agreement.mean_abs,
    }
}

/// Deterministic pseudo-random `f32` values in `[low, high)`; the same
/// generator `tests/fn3_cpu_reference.rs` uses for its real-shape cases.
fn pseudo_values(count: usize, seed: u64, low: f32, high: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            low + (high - low) * unit
        })
        .collect()
}

/// `atol` scaled the way the fixture policy scales with the contraction length:
/// `2e-5 * sqrt(K / 64)`, with the `1/64` floor for short contractions.
fn matmul_atol(base: f64, k: u64) -> f64 {
    base * (k as f64 / 64.0).max(1.0).sqrt()
}

/// Assert the validation layer produced nothing since `before` and print the
/// counts, so the run's validation summary is part of the captured output.
fn assert_validation_clean(before: (u64, u64), context: &str) {
    let after = validation_counts();
    eprintln!(
        "[fn3vk][validation] {context}: errors={} warnings={} (opened with errors={} \
         warnings={})",
        after.0, after.1, before.0, before.1
    );
    assert_eq!(
        after, before,
        "{context}: the validation layer reported an error or warning; see the [fn3vk] lines above"
    );
}

fn expect_unsupported(error: Fn3Error, what: &str) {
    match error {
        Fn3Error::UnsupportedOp {
            op,
            backend,
            reason,
        } => {
            assert_eq!(op, OpId::MatMul, "{what}");
            assert_eq!(backend, "vulkan", "{what}");
            assert!(
                reason.contains("#60"),
                "{what}: the rejection must name its owner, got: {reason}"
            );
        }
        other => panic!("{what}: expected UnsupportedOp, got {other}"),
    }
}

// ---------------------------------------------------------------------------
// 1. Fixture parity against the CPU reference
// ---------------------------------------------------------------------------

#[test]
fn fixture_matmul_cases_match_the_cpu_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    let suite = Suite::load("ops");

    // matmul.2d_k3: [2,3] @ [3,4], odd K.
    {
        let a = suite.tensor("matmul.2d_k3.a").f32_values();
        let b = suite.tensor("matmul.2d_k3.b").f32_values();
        let tolerance = suite.tensor("matmul.2d_k3.out").tolerance();
        let case = Case::contiguous(
            "matmul.2d_k3 [2,3] @ [3,4]",
            &[2, 3],
            &a,
            &[3, 4],
            &b,
            &[2, 4],
            tolerance.atol,
            tolerance.rtol,
        );
        let result = run_case(&backend, &case);
        check_fixture(&suite, "matmul.2d_k3.out", &result.gpu, &[2, 4]);
    }

    // matmul.gemm_k13_bias: Gemm(transB=1) is MatMul(a, w^T), so the weight is a
    // transposed view with no copy. The bias add is a separate op and stays on
    // the host here; FN3-13 (#59) owns Vulkan `Add`.
    {
        let a = suite.tensor("matmul.gemm_k13_bias.a").f32_values();
        let w = suite.tensor("matmul.gemm_k13_bias.w").f32_values();
        let bias = suite.tensor("matmul.gemm_k13_bias.bias").f32_values();
        let tolerance = suite.tensor("matmul.gemm_k13_bias.out").tolerance();
        // Stored as [5, 13]; addressed as [13, 5] with strides [1, 13].
        let case = Case::with_operands(
            "matmul.gemm_k13_bias [7,13] @ [5,13].T",
            Operand::contiguous(&[7, 13], a),
            Operand::strided(&[13, 5], &[1, 13], w),
            &[7, 5],
            tolerance.atol,
            tolerance.rtol,
        );
        let result = run_case(&backend, &case);
        let with_bias: Vec<f32> = result
            .gpu
            .iter()
            .enumerate()
            .map(|(index, &product)| product + bias[index % bias.len()])
            .collect();
        check_fixture(&suite, "matmul.gemm_k13_bias.out", &with_bias, &[7, 5]);
    }

    // matmul.batched_4d: [2,3,5,7] @ [2,3,7,11], tail K.
    {
        let a = suite.tensor("matmul.batched_4d.a").f32_values();
        let b = suite.tensor("matmul.batched_4d.b").f32_values();
        let tolerance = suite.tensor("matmul.batched_4d.out").tolerance();
        let case = Case::contiguous(
            "matmul.batched_4d [2,3,5,7] @ [2,3,7,11]",
            &[2, 3, 5, 7],
            &a,
            &[2, 3, 7, 11],
            &b,
            &[2, 3, 5, 11],
            tolerance.atol,
            tolerance.rtol,
        );
        let result = run_case(&backend, &case);
        check_fixture(&suite, "matmul.batched_4d.out", &result.gpu, &[2, 3, 5, 11]);
    }

    // matmul.model_proj: [1,13,128] @ [128,8], a rank-3 by rank-2 projection.
    {
        let a = suite.tensor("matmul.model_proj.a").f32_values();
        let b = suite.tensor("matmul.model_proj.b").f32_values();
        let tolerance = suite.tensor("matmul.model_proj.out").tolerance();
        let case = Case::contiguous(
            "matmul.model_proj [1,13,128] @ [128,8]",
            &[1, 13, 128],
            &a,
            &[128, 8],
            &b,
            &[1, 13, 8],
            tolerance.atol,
            tolerance.rtol,
        );
        let result = run_case(&backend, &case);
        check_fixture(&suite, "matmul.model_proj.out", &result.gpu, &[1, 13, 8]);
    }

    // matmul.tail_k65: [3,65] @ [65,7], non-power-of-two K tail.
    {
        let a = suite.tensor("matmul.tail_k65.a").f32_values();
        let b = suite.tensor("matmul.tail_k65.b").f32_values();
        let tolerance = suite.tensor("matmul.tail_k65.out").tolerance();
        let case = Case::contiguous(
            "matmul.tail_k65 [3,65] @ [65,7]",
            &[3, 65],
            &a,
            &[65, 7],
            &b,
            &[3, 7],
            tolerance.atol,
            tolerance.rtol,
        );
        let result = run_case(&backend, &case);
        check_fixture(&suite, "matmul.tail_k65.out", &result.gpu, &[3, 7]);
    }

    drop(backend);
    assert_validation_clean(before, "fixture cases");
}

// ---------------------------------------------------------------------------
// 2. Real model geometry
// ---------------------------------------------------------------------------

#[test]
fn real_model_shapes_match_the_cpu_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();

    // Prefill projection: [13, 128] @ [128, 384] (c_attn for one 13-token pass).
    {
        let (sequence, hidden, projected) = (13usize, 128usize, 384usize);
        let case = Case::contiguous(
            "real prefill [13,128] @ [128,384]",
            &[sequence as u64, hidden as u64],
            &pseudo_values(sequence * hidden, 0x51_0001, -2.0, 2.0),
            &[hidden as u64, projected as u64],
            &pseudo_values(hidden * projected, 0x51_0002, -1.0, 1.0),
            &[sequence as u64, projected as u64],
            matmul_atol(2.0e-5, hidden as u64),
            1.0e-5,
        );
        let result = run_case(&backend, &case);
        assert_eq!(result.agreement.elements, sequence * projected);
    }

    // Decode `lm_head`: [1, 1, 128] @ [128, 50257]. The largest single op the
    // pinned model executes; it must not materialise a copy of either operand.
    {
        let (hidden, vocabulary) = (128usize, 50_257usize);
        let case = Case::contiguous(
            "real decode [1,1,128] @ [128,50257]",
            &[1, 1, hidden as u64],
            &pseudo_values(hidden, 0x52_0001, -1.0, 1.0),
            &[hidden as u64, vocabulary as u64],
            &pseudo_values(hidden * vocabulary, 0x52_0002, -1.0, 1.0),
            &[1, 1, vocabulary as u64],
            matmul_atol(1.0e-5, hidden as u64),
            1.0e-4,
        );
        let result = run_case(&backend, &case);
        assert_eq!(result.agreement.elements, vocabulary);
    }

    // Attention score geometry: [1, 2 heads, 17 queries, 64 head dim] @
    // [1, 2, 64, 17]^T-style product, batch axis 1 matching on both sides.
    {
        let (heads, queries, keys, head_dim) = (2u64, 17u64, 17u64, 64u64);
        let case = Case::contiguous(
            "real attention [1,2,17,64] @ [1,2,64,17]",
            &[1, heads, queries, head_dim],
            &pseudo_values((heads * queries * head_dim) as usize, 0x53_0001, -1.0, 1.0),
            &[1, heads, head_dim, keys],
            &pseudo_values((heads * head_dim * keys) as usize, 0x53_0002, -1.0, 1.0),
            &[1, heads, queries, keys],
            2.0e-5,
            1.0e-4,
        );
        let result = run_case(&backend, &case);
        assert_eq!(result.agreement.elements, (heads * queries * keys) as usize);
    }

    // Report the op-scoped device time of each model shape (one submission:
    // upload, dispatch, readback — the alpha tensor-storage contract).
    let records = backend.residency().records();
    for record in records.iter().rev().take(3).rev() {
        eprintln!(
            "[fn3vk][timing] op={} shape_workgroups={:?} submissions={} upload_bytes={} \
             readback_bytes={} gpu_ns={:?} host_ns={}",
            record.op,
            record.workgroups,
            record.submissions,
            record.upload_bytes,
            record.readback_bytes,
            record.gpu_ns,
            record.host_ns
        );
    }

    drop(backend);
    assert_validation_clean(before, "real model shapes");
}

// ---------------------------------------------------------------------------
// 3. Tails, degenerate dimensions, batches, strides
// ---------------------------------------------------------------------------

#[test]
fn contraction_length_covers_the_tile_boundary() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    let (rows, columns) = (5u64, 7u64);
    for k in [1u64, 2, 3, 13, 65, 127, 128, 129, 130] {
        let case = Case::contiguous(
            format!("K boundary M=5 K={k} N=7"),
            &[rows, k],
            &pseudo_values((rows * k) as usize, 0x60_0001 + k, -2.0, 2.0),
            &[k, columns],
            &pseudo_values((k * columns) as usize, 0x61_0002 + k, -2.0, 2.0),
            &[rows, columns],
            matmul_atol(2.0e-5, k),
            1.0e-5,
        );
        run_case(&backend, &case);
    }
    drop(backend);
    assert_validation_clean(before, "K boundary sweep");
}

#[test]
fn degenerate_rows_and_columns_are_exact() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    for (rows, k, columns) in [
        (1u64, 1u64, 1u64),
        (1, 17, 64),
        (64, 17, 1),
        (1, 129, 3),
        (5, 1, 1),
        (1, 7, 1),
    ] {
        let case = Case::contiguous(
            format!("degenerate M={rows} K={k} N={columns}"),
            &[rows, k],
            &pseudo_values((rows * k) as usize, 0x70_0001 + k, -2.0, 2.0),
            &[k, columns],
            &pseudo_values((k * columns) as usize, 0x71_0002 + rows, -2.0, 2.0),
            &[rows, columns],
            matmul_atol(2.0e-5, k),
            1.0e-5,
        );
        run_case(&backend, &case);
    }
    drop(backend);
    assert_validation_clean(before, "degenerate M/N sweep");
}

#[test]
fn batched_rank_three_and_four_broadcast_like_the_cpu_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();

    // Rank-3 by rank-2: the batch axes come from the left operand only.
    let a = pseudo_values(2 * 3 * 4, 0x56_0001, -1.0, 1.0);
    let b = pseudo_values(4 * 5, 0x56_0002, -1.0, 1.0);
    run_case(
        &backend,
        &Case::contiguous(
            "rank3 [2,3,4] @ rank2 [4,5]",
            &[2, 3, 4],
            &a,
            &[4, 5],
            &b,
            &[2, 3, 5],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Rank-3 by rank-3, matching batch.
    let b = pseudo_values(2 * 4 * 5, 0x56_0003, -1.0, 1.0);
    run_case(
        &backend,
        &Case::contiguous(
            "rank3 [2,3,4] @ [2,4,5]",
            &[2, 3, 4],
            &a,
            &[2, 4, 5],
            &b,
            &[2, 3, 5],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Left operand broadcasts over the batch: [1,3,4] @ [2,4,5].
    let a_one = pseudo_values(3 * 4, 0x56_0004, -1.0, 1.0);
    run_case(
        &backend,
        &Case::contiguous(
            "broadcast lhs [1,3,4] @ [2,4,5]",
            &[1, 3, 4],
            &a_one,
            &[2, 4, 5],
            &b,
            &[2, 3, 5],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Right operand broadcasts over the batch: [2,3,4] @ [1,4,5].
    let b_one = pseudo_values(4 * 5, 0x56_0005, -1.0, 1.0);
    run_case(
        &backend,
        &Case::contiguous(
            "broadcast rhs [2,3,4] @ [1,4,5]",
            &[2, 3, 4],
            &a,
            &[1, 4, 5],
            &b_one,
            &[2, 3, 5],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Rank-4 with two-sided broadcast: [2,1,3,4] @ [1,5,4,6] -> [2,5,3,6].
    let a4 = pseudo_values(2 * 3 * 4, 0x56_0006, -1.0, 1.0);
    let b4 = pseudo_values(5 * 4 * 6, 0x56_0007, -1.0, 1.0);
    run_case(
        &backend,
        &Case::contiguous(
            "broadcast rank4 [2,1,3,4] @ [1,5,4,6]",
            &[2, 1, 3, 4],
            &a4,
            &[1, 5, 4, 6],
            &b4,
            &[2, 5, 3, 6],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Rank-4 by rank-2 keeps every leading axis of the left operand.
    run_case(
        &backend,
        &Case::contiguous(
            "rank4 by rank2 [2,1,3,4] @ [4,6]",
            &[2, 1, 3, 4],
            &a4,
            &[4, 6],
            &pseudo_values(4 * 6, 0x56_0008, -1.0, 1.0),
            &[2, 1, 3, 6],
            2.0e-5,
            1.0e-5,
        ),
    );

    drop(backend);
    assert_validation_clean(before, "batched broadcast sweep");
}

#[test]
fn strided_offset_and_broadcast_operands_are_honoured() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    let (rows, k, columns) = (4u64, 9u64, 6u64);

    // Transposed left-hand side: stored [K, M] = [9, 4], addressed [M, K] with
    // strides [1, M].
    let stored_lhs = pseudo_values((k * rows) as usize, 0x80_0001, -1.0, 1.0);
    run_case(
        &backend,
        &Case::with_operands(
            "transposed lhs [4,9] over stored [9,4]",
            Operand::strided(&[rows, k], &[1, rows as i64], stored_lhs),
            Operand::contiguous(
                &[k, columns],
                pseudo_values((k * columns) as usize, 0x80_0002, -1.0, 1.0),
            ),
            &[rows, columns],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Transposed right-hand side with an odd K (the Gemm(transB=1) shape).
    let stored_rhs = pseudo_values((columns * k) as usize, 0x80_0003, -1.0, 1.0);
    run_case(
        &backend,
        &Case::with_operands(
            "transposed rhs [9,6] over stored [6,9]",
            Operand::contiguous(
                &[rows, k],
                pseudo_values((rows * k) as usize, 0x80_0004, -1.0, 1.0),
            ),
            Operand::strided(&[k, columns], &[1, k as i64], stored_rhs),
            &[rows, columns],
            2.0e-5,
            1.0e-5,
        ),
    );

    // A contiguous window that starts at a non-zero element offset: the kernel
    // uploads only the addressed window, so the base offset must be applied.
    let prefix = 37usize;
    let mut stored = vec![0.0f32; prefix];
    stored.extend(pseudo_values((rows * k) as usize, 0x80_0005, -1.0, 1.0));
    let mut window = Operand::contiguous(&[rows, k], stored);
    window.offset = prefix;
    run_case(
        &backend,
        &Case::with_operands(
            "offset window lhs at +37 elements",
            window,
            Operand::contiguous(
                &[k, columns],
                pseudo_values((k * columns) as usize, 0x80_0006, -1.0, 1.0),
            ),
            &[rows, columns],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Explicitly broadcast left-hand side rows: one stored row re-read for
    // every output row ([0, 1] strides on the M axis). The CPU reference
    // accepts a zero stride on a non-contraction axis for exactly this reason.
    let one_row = pseudo_values(k as usize, 0x80_0007, -1.0, 1.0);
    run_case(
        &backend,
        &Case::with_operands(
            "broadcast lhs rows [4,9] stride [0,1]",
            Operand::broadcast(&[rows, k], &[0, 1], one_row),
            Operand::contiguous(
                &[k, columns],
                pseudo_values((k * columns) as usize, 0x80_0008, -1.0, 1.0),
            ),
            &[rows, columns],
            2.0e-5,
            1.0e-5,
        ),
    );

    // Explicitly broadcast column axis: one stored column re-read for every
    // output column ([1, 0] strides).
    let one_column = pseudo_values(k as usize, 0x80_0009, -1.0, 1.0);
    run_case(
        &backend,
        &Case::with_operands(
            "broadcast rhs columns [9,6] stride [1,0]",
            Operand::contiguous(
                &[rows, k],
                pseudo_values((rows * k) as usize, 0x80_000a, -1.0, 1.0),
            ),
            Operand::broadcast(&[k, columns], &[1, 0], one_column),
            &[rows, columns],
            2.0e-5,
            1.0e-5,
        ),
    );

    drop(backend);
    assert_validation_clean(before, "strided/broadcast operands");
}

#[test]
fn empty_contraction_is_positive_zero() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    // K == 0: the empty product is +0.0 for every output element. The shader
    // loop does not execute, which is why this also covers the "no bytes to
    // upload" descriptor path.
    let case = Case::contiguous(
        "empty contraction [2,0] @ [0,3]",
        &[2, 0],
        &[],
        &[0, 3],
        &[],
        &[2, 3],
        0.0,
        0.0,
    );
    let result = run_case(&backend, &case);
    assert!(result.gpu.iter().all(|value| *value == 0.0));
    drop(backend);
    assert_validation_clean(before, "empty contraction");
}

// ---------------------------------------------------------------------------
// 4. Rejection of everything the kernel does not implement
// ---------------------------------------------------------------------------

#[test]
fn unsupported_requests_are_rejected_and_name_the_owner() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let device = backend.device_id();

    // Rank-1 operands: the contract covers rank >= 2 only.
    {
        let lhs =
            Operand::contiguous(&[128], pseudo_values(128, 0x90_0001, -1.0, 1.0)).tensor_on(device);
        let rhs = Operand::contiguous(&[128, 3], pseudo_values(384, 0x90_0002, -1.0, 1.0))
            .tensor_on(device);
        let mut out = zeros_on(device, &[1, 3]);
        let error = execute_matmul(&*backend, &lhs, &rhs, &mut out).expect_err("rank-1 lhs");
        expect_unsupported(error, "rank-1 lhs");
    }
    {
        let lhs = Operand::contiguous(&[2, 128], pseudo_values(256, 0x90_0003, -1.0, 1.0))
            .tensor_on(device);
        let rhs =
            Operand::contiguous(&[128], pseudo_values(128, 0x90_0004, -1.0, 1.0)).tensor_on(device);
        let mut out = zeros_on(device, &[2, 1]);
        let error = execute_matmul(&*backend, &lhs, &rhs, &mut out).expect_err("rank-1 rhs");
        expect_unsupported(error, "rank-1 rhs");
    }

    // A strided (transposed) output cannot be written as one contiguous window.
    {
        let lhs =
            Operand::contiguous(&[2, 3], pseudo_values(6, 0x90_0005, -1.0, 1.0)).tensor_on(device);
        let rhs =
            Operand::contiguous(&[3, 4], pseudo_values(12, 0x90_0006, -1.0, 1.0)).tensor_on(device);
        let mut out = Operand::strided(&[2, 4], &[1, 2], vec![0.0f32; 2 + 3 * 2]).tensor_on(device);
        let error = execute_matmul(&*backend, &lhs, &rhs, &mut out).expect_err("strided output");
        expect_unsupported(error, "strided output");
    }

    // A read-only broadcast output never reaches the backend: `TensorMut::new`
    // rejects it while the mutable view is being built. The kernel still checks
    // `is_broadcast` as defence in depth; the enforceable case it owns is the
    // strided output above.

    // A zero-stride contraction is not a matrix product; the CPU reference
    // rejects it with the same error kind.
    {
        let lhs =
            Operand::contiguous(&[2, 3], pseudo_values(6, 0x90_0009, -1.0, 1.0)).tensor_on(device);
        let rhs = Operand::broadcast(&[3, 4], &[0, 1], pseudo_values(4, 0x90_000a, -1.0, 1.0))
            .tensor_on(device);
        let mut out = zeros_on(device, &[2, 4]);
        let error =
            execute_matmul(&*backend, &lhs, &rhs, &mut out).expect_err("zero-stride contraction");
        assert!(
            matches!(error, Fn3Error::InvalidStride { .. }),
            "expected InvalidStride, got {error}"
        );

        let cpu_lhs = Operand::contiguous(&[2, 3], pseudo_values(6, 0x90_0009, -1.0, 1.0))
            .tensor_on(DeviceId::Cpu);
        let cpu_rhs = Operand::broadcast(&[3, 4], &[0, 1], pseudo_values(4, 0x90_000a, -1.0, 1.0))
            .tensor_on(DeviceId::Cpu);
        let mut cpu_out = zeros_on(DeviceId::Cpu, &[2, 4]);
        let cpu_error = execute_matmul(&CpuBackend::new(), &cpu_lhs, &cpu_rhs, &mut cpu_out)
            .expect_err("cpu must reject it too");
        assert!(
            matches!(cpu_error, Fn3Error::InvalidStride { .. }),
            "cpu and vulkan must agree on the rejection; cpu said {cpu_error}"
        );
    }

    // Stray attributes are rejected rather than ignored.
    {
        let lhs =
            Operand::contiguous(&[2, 3], pseudo_values(6, 0x90_000b, -1.0, 1.0)).tensor_on(device);
        let rhs =
            Operand::contiguous(&[3, 4], pseudo_values(12, 0x90_000c, -1.0, 1.0)).tensor_on(device);
        let mut out = zeros_on(device, &[2, 4]);
        let lhs_view = lhs.as_view().expect("lhs view");
        let rhs_view = rhs.as_view().expect("rhs view");
        let mut out_view = out.as_mut_view().expect("out view");
        let error = backend
            .execute_op(
                OpId::MatMul,
                &[lhs_view, rhs_view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default().with_axes([1]),
            )
            .expect_err("matmul takes no axes");
        assert!(
            matches!(error, Fn3Error::InvalidShape { .. }),
            "expected InvalidShape, got {error}"
        );
    }

    drop(backend);
}

// ---------------------------------------------------------------------------
// 5. Support verdict and device execution evidence
// ---------------------------------------------------------------------------

#[test]
fn support_is_full_and_the_record_proves_device_execution() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();

    assert_eq!(
        backend.supports(OpId::MatMul),
        Support::Full,
        "the tiled kernel covers the whole rank >= 2 contract"
    );
    // `CapabilityFeatures::broadcasting` is the *elementwise* broadcasting flag
    // and it is an aggregate over every registered kernel (FN3-13 / #59 owns the
    // kernel that sets it). Matmul's own batch broadcasting is part of this op's
    // contract, so the only stable assertion here is the implication: if the
    // flag is advertised, the elementwise kernel that backs it must be Full.
    // Asserting the aggregate is *false* would fail the moment a pointwise
    // kernel lands, which is what happened when #59 merged.
    if backend.capabilities().features.broadcasting {
        assert_eq!(
            backend.supports(OpId::Add),
            Support::Full,
            "the broadcasting flag may only be advertised when a kernel implements \
             elementwise broadcasting"
        );
    }

    let (sequence, hidden, projected) = (13u64, 128u64, 384u64);
    let case = Case::contiguous(
        "evidence [13,128] @ [128,384]",
        &[sequence, hidden],
        &pseudo_values((sequence * hidden) as usize, 0xa0_0001, -1.0, 1.0),
        &[hidden, projected],
        &pseudo_values((hidden * projected) as usize, 0xa0_0002, -1.0, 1.0),
        &[sequence, projected],
        2.0e-5,
        1.0e-5,
    );
    run_case(&backend, &case);

    let records = backend.residency().records();
    let record = records.last().expect("one execution record");
    assert_eq!(record.backend, "vulkan");
    assert_eq!(record.op, OpId::MatMul);
    assert_eq!(record.kernel, matmul_kernel::MATMUL_TILED.name);
    assert!(record.submissions >= 1, "{record:?}");
    assert!(
        record.workgroups.iter().all(|count| *count > 0),
        "a zero dispatch axis cannot prove work: {record:?}"
    );
    assert!(record.proves_device_execution(), "{record:?}");
    assert!(!record.inputs_read_back, "inputs must never be read back");
    assert_eq!(
        record.upload_bytes,
        (sequence * hidden + hidden * projected) * 4
    );
    assert_eq!(record.readback_bytes, sequence * projected * 4);
    assert!(
        record.device_local_bytes
            >= (sequence * hidden + hidden * projected + sequence * projected) * 4,
        "{record:?}"
    );
    assert!(
        record.gpu_ns.is_some(),
        "the queue family supports timestamps, so device time must be recorded"
    );
    eprintln!(
        "[fn3vk][evidence] workgroups={:?} submissions={} bindings={} upload_bytes={} \
         readback_bytes={} device_local_bytes={} gpu_ns={:?} gpu_ticks={:?} host_ns={}",
        record.workgroups,
        record.submissions,
        record.bindings.len(),
        record.upload_bytes,
        record.readback_bytes,
        record.device_local_bytes,
        record.gpu_ns,
        record.gpu_ticks,
        record.host_ns
    );

    let summary = backend.residency().summary();
    assert_eq!(summary.executions, 1);
    assert_eq!(summary.proven_device_executions, 1);
    assert_eq!(summary.input_readbacks, 0);
    assert_eq!(summary.ops, vec![(OpId::MatMul, 1)]);

    drop(backend);
    assert_validation_clean(before, "evidence [13,128]@[128,384]");
}

// ---------------------------------------------------------------------------
// 6. Device-resident decode timing (measurement on this device only)
// ---------------------------------------------------------------------------

/// Push-constant words for a batch-free 2-D matmul.
///
/// This mirrors the private block in `kernels::matmul` / `shaders/
/// matmul_tiled.comp`: `m, n, k, a_base, b_base, out_base, a_row, a_k, b_k,
/// b_col, batch_dims[6], a_batch_strides[6], b_batch_strides[6]`. The size is
/// asserted against [`matmul_kernel::MATMUL_TILED`] and the dispatch result is
/// compared with the CPU reference below, so a layout drift fails the test
/// instead of producing a number for a wrong dispatch.
fn push_words_2d(m: u32, n: u32, k: u32, a_row: u32, a_k: u32, b_k: u32, b_col: u32) -> [u32; 28] {
    let mut words = [0u32; 28];
    words[0] = m;
    words[1] = n;
    words[2] = k;
    words[6] = a_row;
    words[7] = a_k;
    words[8] = b_k;
    words[9] = b_col;
    for slot in words.iter_mut().take(16).skip(10) {
        *slot = 1; // unused batch axes have extent 1 and stride 0
    }
    words
}

#[test]
fn decode_projection_device_timing() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let before = validation_counts();
    assert_eq!(
        matmul_kernel::MATMUL_TILED.push_constant_bytes,
        (std::mem::size_of::<[u32; 28]>()) as u32,
        "the timing path builds the documented 28-word block"
    );

    let hidden = 128u32;
    let vocabulary = 50_257u32;
    let rows = 1u32;
    let lhs_values = pseudo_values(hidden as usize, 0xb0_0001, -1.0, 1.0);
    let rhs_values = pseudo_values(
        (hidden as usize) * (vocabulary as usize),
        0xb0_0002,
        -1.0,
        1.0,
    );

    let lhs_bytes = u64::from(hidden) * 4;
    let rhs_bytes = u64::from(hidden) * u64::from(vocabulary) * 4;
    let out_bytes = u64::from(vocabulary) * 4;
    let lhs_buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(lhs_bytes, "bench.lhs"))
        .expect("lhs buffer");
    let rhs_buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(rhs_bytes, "bench.rhs"))
        .expect("rhs buffer");
    let out_buffer = backend
        .alloc_buffer(&BufferDesc::device_storage(out_bytes, "bench.out"))
        .expect("out buffer");
    backend
        .upload(&lhs_buffer, bytemuck::cast_slice(&lhs_values))
        .expect("upload lhs");
    backend
        .upload(&rhs_buffer, bytemuck::cast_slice(&rhs_values))
        .expect("upload rhs");

    let kernel = backend
        .kernel(&matmul_kernel::MATMUL_TILED)
        .expect("compiled matmul kernel");
    let workgroups = [
        vocabulary.div_ceil(matmul_kernel::TILE),
        rows.div_ceil(matmul_kernel::TILE),
        1,
    ];
    let push = push_words_2d(rows, vocabulary, hidden, hidden, 1, vocabulary, 1);
    let push_bytes: &[u8] = bytemuck::cast_slice(&push);
    let bindings = [
        BindingArg::whole(0, BindingRole::Input, &lhs_buffer),
        BindingArg::whole(1, BindingRole::Input, &rhs_buffer),
        BindingArg::whole(2, BindingRole::Output, &out_buffer),
    ];
    let invocation = KernelInvocation {
        kernel: &kernel,
        bindings: &bindings,
        push_constants: push_bytes,
        workgroups,
    };

    // Warm up (pipeline binding, descriptor pool growth, GPU clocks) and then
    // collect device-time samples for single dispatches, no host staging.
    for _ in 0..3 {
        backend.dispatch(&invocation).expect("warm-up dispatch");
    }
    let mut samples: Vec<u64> = Vec::new();
    for _ in 0..20 {
        let outcome = backend.dispatch(&invocation).expect("timed dispatch");
        samples.push(
            outcome
                .gpu_ns
                .expect("the queue family reports device timestamps"),
        );
    }
    let mut readback = vec![0u8; out_bytes as usize];
    backend
        .download(&out_buffer, &mut readback)
        .expect("download logits");
    let gpu: &[f32] = bytemuck::cast_slice(&readback);

    // The timed dispatch must produce the reference result.
    let cpu_backend = CpuBackend::new();
    let lhs_cpu = Operand::contiguous(&[u64::from(rows), u64::from(hidden)], lhs_values)
        .tensor_on(DeviceId::Cpu);
    let rhs_cpu = Operand::contiguous(&[u64::from(hidden), u64::from(vocabulary)], rhs_values)
        .tensor_on(DeviceId::Cpu);
    let mut out_cpu = zeros_on(DeviceId::Cpu, &[u64::from(rows), u64::from(vocabulary)]);
    execute_matmul(&cpu_backend, &lhs_cpu, &rhs_cpu, &mut out_cpu).expect("cpu decode projection");
    let reference = out_cpu.as_f32_slice().expect("cpu logits");
    compare(
        "device-resident decode [1,128] @ [128,50257]",
        gpu,
        reference,
        matmul_atol(1.0e-5, u64::from(hidden)),
        1.0e-4,
    );

    samples.sort_unstable();
    let median = samples[samples.len() / 2];
    let fastest = samples[0];
    let flops = 2.0 * f64::from(rows) * f64::from(vocabulary) * f64::from(hidden);
    let gflops_median = flops / median as f64;
    let gflops_fastest = flops / fastest as f64;
    eprintln!(
        "[fn3vk][timing] device-resident decode [1,128] @ [128,50257]: dispatch-only, no host \
         staging, {} dispatches of one submission each on \"{}\" (uuid {})",
        samples.len(),
        backend.adapter().name,
        backend.adapter().uuid_string()
    );
    eprintln!(
        "[fn3vk][timing] device_ns median={median} min={fastest} max={} \
         flops={flops:.3e} gflops_median={gflops_median:.2} gflops_min={gflops_fastest:.2}",
        samples[samples.len() - 1]
    );
    eprintln!(
        "[fn3vk][timing] NOTE: measured on this workstation GPU only; not a general performance claim"
    );

    drop(out_buffer);
    drop(rhs_buffer);
    drop(lhs_buffer);
    assert_eq!(
        backend.memory_stats().live_device_local_bytes,
        0,
        "the benchmark frees its device buffers"
    );
    drop(backend);
    assert_validation_clean(before, "decode timing");
}
