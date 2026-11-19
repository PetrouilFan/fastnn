//! FN3-15 (#61): Vulkan normalization, reduction and attention conformance.
//!
//! Run with:
//!
//! ```bash
//! FN3_VK_DEVICE_UUID=<physical-gpu-uuid> CARGO_BUILD_JOBS=4 \
//!   cargo test --features vulkan --test fn3_vulkan_attention -- --nocapture --test-threads=1
//! ```
//!
//! # What these tests prove
//!
//! 1. **Fixture parity.** Every committed `ops` fixture this issue owns
//!    (`layernorm`, `softmax` unmasked/causal/padded/all-masked, `attention`) is
//!    executed through the real `Backend::execute_op` path and compared with the
//!    fixture's own tolerance. The fused `MaskedAttention` result is additionally
//!    compared with the CPU reference decomposition.
//! 2. **Real geometry.** LayerNorm at `[1, 13, 128]` and attention at
//!    `S = 17`, `head_dim = 64` with a causal mask, plus grouped-query
//!    broadcasting and a sequence longer than the fixture (`S = 33`).
//! 3. **Masked-row semantics.** An all-masked row is asserted **exactly**
//!    uniform `[0.25, 0.25, 0.25, 0.25]` and finite element by element.
//! 4. **Execution evidence.** Every case asserts `proves_device_execution()` and
//!    that no input was read back (`input_readbacks == 0`), and that the
//!    validation layer reported zero errors and zero warnings.
//! 5. **Honest support.** Ops register `Full`/`Partial` with a named subset and
//!    reject everything else with `UnsupportedOp` naming #61.
//!
//! Physical-GPU tests skip with `[fn3vk][skip]` when no non-software adapter is
//! present. Selection is by UUID (`FN3_VK_DEVICE_UUID`), never by index: the
//! loader also enumerates lavapipe. Validation is enabled in code; do **not**
//! pass `VK_INSTANCE_LAYERS`, which would duplicate the layer.

#![cfg(feature = "vulkan")]

mod fn3_fixtures;

use std::sync::{Arc, Mutex};

use fastnn::inference::backend::cpu::CpuBackend;
use fastnn::inference::backend::vulkan::instance::validation_counts;
use fastnn::inference::backend::vulkan::{adapters, ExecutionRecord, VulkanBackend, VulkanConfig};
use fastnn::inference::backend::{Backend, OpParams, Support};
use fastnn::inference::device::{DeviceId, DeviceRequest};
use fastnn::inference::dtype::DType;
use fastnn::inference::error::Fn3Error;
use fastnn::inference::ops::OpId;
use fastnn::inference::shape::{Shape, Strides};
use fastnn::inference::tensor::{Tensor, TensorMeta, TensorView};
use fastnn::inference::Storage;

use fn3_fixtures::{Actual, Suite};

// ---------------------------------------------------------------------------
// Device gate
// ---------------------------------------------------------------------------

/// Serialises physical-GPU use and validation-count assertions.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// True when this process must produce a real device result rather than skip.
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
            "[fn3vk][skip] no physical Vulkan device with a compute queue is present; the \
             lavapipe lane is CI-only and does not satisfy the FN3-15 gate"
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
                 validation={} shared_memory_limit={}",
                adapter.name,
                adapter.uuid_string(),
                fastnn::inference::backend::vulkan::device_info::format_version(
                    adapter.api_version
                ),
                adapter.driver_version_string(),
                backend.vulkan_device().queue_family(),
                backend.validation_enabled(),
                adapter.limits.max_compute_shared_memory_size
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

/// Assert the process-global validation counters did not move.
fn assert_validation_clean(counts_before: (u64, u64)) {
    let counts_after = validation_counts();
    assert_eq!(
        counts_after, counts_before,
        "validation layer reported errors/warnings: before {counts_before:?}, after \
         {counts_after:?}; see the [fn3vk][validation] lines above"
    );
    eprintln!(
        "[fn3vk][validation] errors={} warnings={} (delta 0/0 since the section opened)",
        counts_after.0, counts_after.1
    );
}

// ---------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------

/// Contiguous metadata that survives extent-1 axes.
///
/// `Shape::default_strides` gives an interior or trailing extent-1 axis the
/// stride of the next axis, which `Strides::check_no_overlap` rejects. Those
/// shapes are ordinary for this issue (`[2, 1, 4, 4]` is the canonical additive
/// attention mask), so each extent-1 axis gets the stride `numel`: it is never
/// advanced, keeps the row-major layout, and stays `is_contiguous`.
fn meta_for(dims: &[u64], dtype: DType, device: DeviceId) -> TensorMeta {
    let shape = Shape::from_dims(dims.iter().copied());
    let numel = shape.numel().unwrap();
    let mut strides = shape.default_strides().unwrap().as_slice().to_vec();
    let mut patched = false;
    if numel > 0 {
        for (axis, &extent) in dims.iter().enumerate() {
            if extent == 1 && strides[axis] != numel as i64 {
                strides[axis] = numel as i64;
                patched = true;
            }
        }
    }
    if patched {
        TensorMeta::strided(shape, Strides::from_elements(strides), 0, dtype, device)
            .expect("patched contiguous metadata")
    } else {
        TensorMeta::contiguous(shape, dtype, device).expect("contiguous metadata")
    }
}

fn f32_tensor_on(device: DeviceId, dims: &[u64], values: &[f32]) -> Tensor {
    let meta = meta_for(dims, DType::F32, device);
    assert_eq!(
        values.len() as u64,
        meta.numel().unwrap(),
        "test tensor element count for {dims:?}"
    );
    let storage = Storage::from_bytes(bytemuck::cast_slice(values)).expect("host storage");
    Tensor::from_storage(storage, meta).expect("tensor")
}

fn i64_tensor_on(device: DeviceId, dims: &[u64], values: &[i64]) -> Tensor {
    let meta = meta_for(dims, DType::I64, device);
    let storage = Storage::from_bytes(bytemuck::cast_slice(values)).expect("host storage");
    Tensor::from_storage(storage, meta).expect("tensor")
}

fn i32_tensor_on(device: DeviceId, dims: &[u64], values: &[i32]) -> Tensor {
    let meta = meta_for(dims, DType::I32, device);
    let storage = Storage::from_bytes(bytemuck::cast_slice(values)).expect("host storage");
    Tensor::from_storage(storage, meta).expect("tensor")
}

fn zeros_on(device: DeviceId, dims: &[u64]) -> Tensor {
    let meta = meta_for(dims, DType::F32, device);
    let bytes = meta.required_bytes().expect("zero tensor bytes");
    let storage = Storage::zeroed(bytes).expect("zeroed storage");
    Tensor::from_storage(storage, meta).expect("zero tensor")
}

fn zeros_i64_on(device: DeviceId, dims: &[u64]) -> Tensor {
    let meta = meta_for(dims, DType::I64, device);
    let bytes = meta.required_bytes().expect("zero i64 tensor bytes");
    let storage = Storage::zeroed(bytes).expect("zeroed storage");
    Tensor::from_storage(storage, meta).expect("zero i64 tensor")
}

/// Execute one op into one output through `backend`.
fn run(
    backend: &dyn Backend,
    op: OpId,
    inputs: &[&Tensor],
    out: &mut Tensor,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().expect("input view"))
        .collect();
    let mut out_view = out.as_mut_view().expect("output view");
    backend.execute_op(op, &views, std::slice::from_mut(&mut out_view), params)
}

/// Execute one op into two outputs through `backend`.
fn run_two(
    backend: &dyn Backend,
    op: OpId,
    inputs: &[&Tensor],
    first: &mut Tensor,
    second: &mut Tensor,
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().expect("input view"))
        .collect();
    let mut outputs = [
        first.as_mut_view().expect("first output view"),
        second.as_mut_view().expect("second output view"),
    ];
    backend.execute_op(op, &views, &mut outputs, params)
}

fn transpose_last_two(tensor: &Tensor) -> TensorMeta {
    let dims = tensor.shape().dims().to_vec();
    let rank = dims.len();
    let strides = tensor.meta().strides().as_slice().to_vec();
    let mut new_dims = dims.clone();
    new_dims.swap(rank - 2, rank - 1);
    let mut new_strides = strides.clone();
    new_strides.swap(rank - 2, rank - 1);
    TensorMeta::strided(
        Shape::from_dims(new_dims),
        Strides::from_elements(new_strides),
        tensor.meta().offset(),
        DType::F32,
        tensor.device(),
    )
    .expect("transposed view")
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// Compare `actual` with `reference` under `atol + rtol * |reference|`.
fn measure(label: &str, actual: &[f32], reference: &[f32], atol: f64, rtol: f64) {
    assert_eq!(actual.len(), reference.len(), "{label}: length");
    let mut max_abs = 0.0f64;
    let mut mean_abs = 0.0f64;
    for (position, (&actual, &reference)) in actual.iter().zip(reference.iter()).enumerate() {
        if actual.is_finite() != reference.is_finite() {
            panic!(
                "{label}: element {position} is {actual} but the reference is {reference} \
                 (finiteness mismatch)"
            );
        }
        let difference = (f64::from(actual) - f64::from(reference)).abs();
        max_abs = max_abs.max(difference);
        mean_abs += difference;
        let bound = atol + rtol * f64::from(reference).abs();
        assert!(
            difference <= bound || (actual.is_nan() && reference.is_nan()),
            "{label}: element {position} is {actual}, reference {reference}, |diff|={difference:e} \
             exceeds {bound:e}"
        );
    }
    let elements = actual.len();
    mean_abs /= elements.max(1) as f64;
    eprintln!(
        "[fn3vk][compare] {label:52} elements={elements:6} max_abs={max_abs:e} \
         mean_abs={mean_abs:e} bound={atol:e}+{rtol:e}|ref|"
    );
}

/// Compare with a committed fixture through the fixture suite's own predicates.
fn measure_fixture(suite: &Suite, name: &str, actual: &[f32], shape: &[u64]) {
    let reference = suite.tensor(name);
    let agreement = reference
        .check(Actual::F32(actual), shape)
        .unwrap_or_else(|error| panic!("fixture {name}: {error}"));
    reference
        .check_summary(Actual::F32(actual), shape)
        .unwrap_or_else(|error| panic!("fixture {name} summary: {error}"));
    eprintln!(
        "[fn3vk][fixture] {name:38} shape={shape:?} {agreement} tolerance={}",
        reference.tolerance()
    );
}

fn assert_all_finite(label: &str, values: &[f32]) {
    for (index, value) in values.iter().enumerate() {
        assert!(
            value.is_finite(),
            "{label}: element {index} is {value}, not finite"
        );
    }
}

fn assert_exactly_uniform(label: &str, row: &[f32], expected: f32) {
    for (index, value) in row.iter().enumerate() {
        assert_eq!(
            *value, expected,
            "{label}: element {index} is {value}, expected exactly {expected}"
        );
    }
}

/// Every record for `op` must prove device execution and no input readback.
fn assert_evidence(backend: &VulkanBackend, op: OpId) -> u64 {
    let summary = backend.residency().summary();
    assert_eq!(
        summary.input_readbacks, 0,
        "an input was read back to the host: {summary:?}"
    );
    let records: Vec<ExecutionRecord> = backend
        .residency()
        .records()
        .into_iter()
        .filter(|record| record.op == op)
        .collect();
    assert!(
        !records.is_empty(),
        "no residency record for {op}; the op did not run on the device"
    );
    let mut device_local = 0u64;
    for record in &records {
        assert_eq!(record.backend, "vulkan");
        assert_eq!(record.device, backend.device_id());
        assert!(
            record.proves_device_execution(),
            "{op}: record does not prove device execution: {record:?}"
        );
        assert!(
            !record.inputs_read_back,
            "{op}: an input was read back: {record:?}"
        );
        assert!(record.upload_bytes > 0, "{op}: nothing was uploaded");
        assert!(record.readback_bytes > 0, "{op}: nothing was read back");
        device_local = device_local.max(record.device_local_bytes);
        eprintln!(
            "[fn3vk][evidence] op={:14} kernel={:24} submissions={} workgroups={:?} \
             bindings={} upload_bytes={} readback_bytes={} device_local_bytes={} host_ns={} \
             gpu_ns={:?}",
            record.op.name(),
            record.kernel,
            record.submissions,
            record.workgroups,
            record.bindings.len(),
            record.upload_bytes,
            record.readback_bytes,
            record.device_local_bytes,
            record.host_ns,
            record.gpu_ns
        );
    }
    device_local
}

fn assert_no_live_device_memory(backend: &VulkanBackend) {
    assert_eq!(
        backend.memory_stats().live_device_local_bytes,
        0,
        "op-scoped device buffers must be freed: {:?}",
        backend.memory_stats()
    );
}

// ---------------------------------------------------------------------------
// CPU reference decomposition for attention
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct AttentionGeometry {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    queries: usize,
    keys: usize,
    head_dim: usize,
}

impl AttentionGeometry {
    fn group(&self) -> usize {
        self.heads / self.kv_heads
    }
}

/// `f32` mask fill value: the pinned GPT-2 additive mask, exactly `-FLT_MAX`.
const NEGATIVE_FLT_MAX: f32 = -3.402_823_5e38;

/// Broadcast every KV head to the query heads that read it.
fn expand_kv(values: &[f32], geometry: &AttentionGeometry) -> Vec<f32> {
    let group = geometry.group();
    let mut expanded =
        vec![0.0f32; geometry.batch * geometry.heads * geometry.keys * geometry.head_dim];
    for batch in 0..geometry.batch {
        for head in 0..geometry.heads {
            let kv_head = head / group;
            for key in 0..geometry.keys {
                for d in 0..geometry.head_dim {
                    expanded[((batch * geometry.heads + head) * geometry.keys + key)
                        * geometry.head_dim
                        + d] = values[((batch * geometry.kv_heads + kv_head) * geometry.keys
                        + key)
                        * geometry.head_dim
                        + d];
                }
            }
        }
    }
    expanded
}

/// Effective additive mask after applying the causal rule (if any).
fn effective_mask(
    mask: Option<&[f32]>,
    mask_batch: usize,
    mask_heads: usize,
    geometry: &AttentionGeometry,
    causal: bool,
) -> Option<(Vec<f32>, usize, usize)> {
    if !causal {
        return mask.map(|values| (values.to_vec(), mask_batch, mask_heads));
    }
    let mut values = mask.map(|values| values.to_vec()).unwrap_or_else(|| {
        vec![0.0f32; mask_batch * mask_heads * geometry.queries * geometry.keys]
    });
    for batch in 0..mask_batch {
        for head in 0..mask_heads {
            for query in 0..geometry.queries {
                let limit = geometry.keys - geometry.queries + query;
                for key in 0..geometry.keys {
                    if key > limit {
                        values[((batch * mask_heads + head) * geometry.queries + query)
                            * geometry.keys
                            + key] = NEGATIVE_FLT_MAX;
                    }
                }
            }
        }
    }
    Some((values, mask_batch, mask_heads))
}

/// The CPU reference decomposition: `MatMul → Div → Add(mask) → Softmax → MatMul`.
// A reference helper mirroring the attention operands one-for-one; grouping
// them into a struct would hide which tensor each shape belongs to.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    mask_batch: usize,
    mask_heads: usize,
    geometry: &AttentionGeometry,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let cpu = DeviceId::Cpu;
    let backend = CpuBackend::new();
    let g = geometry;
    let q_tensor = f32_tensor_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.queries as u64,
            g.head_dim as u64,
        ],
        q,
    );
    let k_expanded = expand_kv(k, g);
    let v_expanded = expand_kv(v, g);
    let k_tensor = f32_tensor_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.keys as u64,
            g.head_dim as u64,
        ],
        &k_expanded,
    );
    let v_tensor = f32_tensor_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.keys as u64,
            g.head_dim as u64,
        ],
        &v_expanded,
    );

    let k_transposed = TensorView::new(k_tensor.storage(), transpose_last_two(&k_tensor))
        .expect("transposed k view");
    let q_view = q_tensor.as_view().expect("q view");
    let mut scores = zeros_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.queries as u64,
            g.keys as u64,
        ],
    );
    {
        let mut out_view = scores.as_mut_view().unwrap();
        backend
            .execute_op(
                OpId::MatMul,
                &[q_view, k_transposed],
                std::slice::from_mut(&mut out_view),
                &OpParams::default(),
            )
            .expect("cpu scores");
    }

    let divisor = [(g.head_dim as f32).sqrt()];
    let divisor_tensor = f32_tensor_on(cpu, &[], &divisor);
    let mut scaled = zeros_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.queries as u64,
            g.keys as u64,
        ],
    );
    run(
        &backend,
        OpId::Div,
        &[&scores, &divisor_tensor],
        &mut scaled,
        &OpParams::default(),
    )
    .expect("cpu scale");

    let mask_values = effective_mask(mask, mask_batch, mask_heads, g, causal);
    let masked = match &mask_values {
        Some((values, batch, heads)) => {
            let mask_tensor = f32_tensor_on(
                cpu,
                &[
                    *batch as u64,
                    *heads as u64,
                    g.queries as u64,
                    g.keys as u64,
                ],
                values,
            );
            let mut masked = zeros_on(
                cpu,
                &[
                    g.batch as u64,
                    g.heads as u64,
                    g.queries as u64,
                    g.keys as u64,
                ],
            );
            run(
                &backend,
                OpId::Add,
                &[&scaled, &mask_tensor],
                &mut masked,
                &OpParams::default(),
            )
            .expect("cpu add mask");
            masked
        }
        None => scaled,
    };

    let mut probs = zeros_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.queries as u64,
            g.keys as u64,
        ],
    );
    run(
        &backend,
        OpId::Softmax,
        &[&masked],
        &mut probs,
        &OpParams::default().with_axes([-1]),
    )
    .expect("cpu softmax");

    let mut out = zeros_on(
        cpu,
        &[
            g.batch as u64,
            g.heads as u64,
            g.queries as u64,
            g.head_dim as u64,
        ],
    );
    run(
        &backend,
        OpId::MatMul,
        &[&probs, &v_tensor],
        &mut out,
        &OpParams::default(),
    )
    .expect("cpu out");

    (
        out.as_f32_slice().expect("cpu out values").to_vec(),
        probs.as_f32_slice().expect("cpu probs values").to_vec(),
    )
}

/// Scale a fixture tolerance for a sequence longer than 16, per
/// `fixtures/parity/README.md`: `atol *= sqrt(S / 16)`, `rtol` unchanged.
fn scaled_atol(atol: f64, sequence: usize) -> f64 {
    if sequence > 16 {
        atol * ((sequence as f64) / 16.0).sqrt()
    } else {
        atol
    }
}

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

// ---------------------------------------------------------------------------
// Fixture parity
// ---------------------------------------------------------------------------

#[test]
fn fixture_layernorm_matches_the_committed_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let suite = Suite::load("ops");
    let device = backend.device_id();

    for (input_name, weight_name, bias_name, out_name, dims) in [
        (
            "layernorm.input",
            "layernorm.weight",
            "layernorm.bias",
            "layernorm.out",
            [2u64, 3, 5],
        ),
        (
            "layernorm.model_shape.input",
            "layernorm.model_shape.weight",
            "layernorm.model_shape.bias",
            "layernorm.model_shape.out",
            [1u64, 4, 128],
        ),
    ] {
        let input = f32_tensor_on(device, &dims, &suite.tensor(input_name).f32_values());
        let hidden = dims[dims.len() - 1];
        let weight = f32_tensor_on(device, &[hidden], &suite.tensor(weight_name).f32_values());
        let bias = f32_tensor_on(device, &[hidden], &suite.tensor(bias_name).f32_values());
        let mut out = zeros_on(device, &dims);
        run(
            &*backend,
            OpId::LayerNorm,
            &[&input, &weight, &bias],
            &mut out,
            &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
        )
        .expect("vulkan layernorm");
        let values = out.as_f32_slice().expect("layernorm output").to_vec();

        measure_fixture(&suite, out_name, &values, &dims);

        let cpu_out = {
            let cpu = DeviceId::Cpu;
            let input_cpu = f32_tensor_on(cpu, &dims, &suite.tensor(input_name).f32_values());
            let weight_cpu = f32_tensor_on(cpu, &[hidden], &suite.tensor(weight_name).f32_values());
            let bias_cpu = f32_tensor_on(cpu, &[hidden], &suite.tensor(bias_name).f32_values());
            let mut out_cpu = zeros_on(cpu, &dims);
            run(
                &CpuBackend::new(),
                OpId::LayerNorm,
                &[&input_cpu, &weight_cpu, &bias_cpu],
                &mut out_cpu,
                &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
            )
            .expect("cpu layernorm");
            out_cpu.as_f32_slice().expect("cpu layernorm").to_vec()
        };
        measure(
            &format!("layernorm {dims:?} vs CpuBackend"),
            &values,
            &cpu_out,
            2.0e-5,
            1.0e-5,
        );
        assert_all_finite("layernorm output", &values);
    }

    assert!(backend.supports(OpId::LayerNorm).is_full());
    assert_evidence(&backend, OpId::LayerNorm);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

fn masked_softmax_case(
    backend: &Arc<VulkanBackend>,
    suite: &Suite,
    case: &str,
    dims: &[u64],
) -> Vec<f32> {
    let device = backend.device_id();
    let input = f32_tensor_on(
        device,
        dims,
        &suite.tensor(&format!("softmax.{case}.input")).f32_values(),
    );
    let mask = f32_tensor_on(
        device,
        dims,
        &suite.tensor(&format!("softmax.{case}.mask")).f32_values(),
    );
    let mut scores = zeros_on(device, dims);
    // The additive mask is an explicit `Add` before the `Softmax`, exactly as
    // the pinned graph expresses it. Same-shape contiguous Add is the FN3-11
    // smoke kernel, which is the only Vulkan `Add` in the alpha build.
    run(
        &**backend,
        OpId::Add,
        &[&input, &mask],
        &mut scores,
        &OpParams::default(),
    )
    .expect("vulkan add mask");
    let mut out = zeros_on(device, dims);
    run(
        &**backend,
        OpId::Softmax,
        &[&scores],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("vulkan softmax");
    let actual = out.as_f32_slice().expect("softmax output").to_vec();

    // The fixture is the independent reference; the CPU backend is the merged
    // numerical oracle, so compare with both.
    let cpu = DeviceId::Cpu;
    let cpu_input = f32_tensor_on(
        cpu,
        dims,
        &suite.tensor(&format!("softmax.{case}.input")).f32_values(),
    );
    let cpu_mask = f32_tensor_on(
        cpu,
        dims,
        &suite.tensor(&format!("softmax.{case}.mask")).f32_values(),
    );
    let mut cpu_scores = zeros_on(cpu, dims);
    run(
        &CpuBackend::new(),
        OpId::Add,
        &[&cpu_input, &cpu_mask],
        &mut cpu_scores,
        &OpParams::default(),
    )
    .expect("cpu add mask");
    let mut cpu_out = zeros_on(cpu, dims);
    run(
        &CpuBackend::new(),
        OpId::Softmax,
        &[&cpu_scores],
        &mut cpu_out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("cpu softmax");
    let reference = cpu_out.as_f32_slice().expect("cpu softmax").to_vec();
    measure(
        &format!("softmax.{case} vs CpuBackend"),
        &actual,
        &reference,
        2.0e-6,
        1.0e-6,
    );
    actual
}

#[test]
fn fixture_softmax_unmasked_causal_and_padded() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let suite = Suite::load("ops");

    let unmasked_dims = [2u64, 3, 4, 5];
    let unmasked_input = f32_tensor_on(
        backend.device_id(),
        &unmasked_dims,
        &suite.tensor("softmax.unmasked.input").f32_values(),
    );
    let mut unmasked_out = zeros_on(backend.device_id(), &unmasked_dims);
    run(
        &*backend,
        OpId::Softmax,
        &[&unmasked_input],
        &mut unmasked_out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("vulkan softmax unmasked");
    let unmasked = unmasked_out.as_f32_slice().expect("softmax").to_vec();
    measure_fixture(&suite, "softmax.unmasked.out", &unmasked, &unmasked_dims);
    let cpu_unmasked = {
        let input = f32_tensor_on(
            DeviceId::Cpu,
            &unmasked_dims,
            &suite.tensor("softmax.unmasked.input").f32_values(),
        );
        let mut out = zeros_on(DeviceId::Cpu, &unmasked_dims);
        run(
            &CpuBackend::new(),
            OpId::Softmax,
            &[&input],
            &mut out,
            &OpParams::default().with_axes([-1]),
        )
        .expect("cpu softmax unmasked");
        out.as_f32_slice().expect("cpu softmax").to_vec()
    };
    measure(
        "softmax.unmasked vs CpuBackend",
        &unmasked,
        &cpu_unmasked,
        2.0e-6,
        1.0e-6,
    );

    let causal = masked_softmax_case(&backend, &suite, "causal", &[1, 2, 4, 4]);
    measure_fixture(&suite, "softmax.causal.out", &causal, &[1, 2, 4, 4]);

    let padded = masked_softmax_case(&backend, &suite, "padded", &[1, 1, 3, 5]);
    measure_fixture(&suite, "softmax.padded.out", &padded, &[1, 1, 3, 5]);

    for values in [&unmasked, &causal, &padded] {
        assert_all_finite("softmax output", values);
    }
    let support = backend.supports(OpId::Softmax);
    assert!(support.is_supported() && !support.is_full(), "{support}");
    assert!(
        support
            .reason()
            .unwrap_or_default()
            .contains("trailing axis"),
        "the partial subset must name its limit: {support}"
    );
    assert_evidence(&backend, OpId::Softmax);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn fixture_softmax_all_masked_rows_are_exactly_uniform_on_device() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let suite = Suite::load("ops");
    let dims = [1u64, 1, 2, 4];
    let values = masked_softmax_case(&backend, &suite, "all_masked", &dims);
    measure_fixture(&suite, "softmax.all_masked.out", &values, &dims);

    // The row that is entirely -FLT_MAX: finite and exactly 1/4 per entry.
    let masked_row = &values[4..8];
    assert_exactly_uniform("all-masked softmax row", masked_row, 0.25);
    assert_all_finite("all-masked softmax row", masked_row);
    eprintln!("[fn3vk][contract] softmax all-masked row = {masked_row:?} (exact, finite)");

    assert_evidence(&backend, OpId::Softmax);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn fixture_attention_fused_matches_the_decomposition() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let suite = Suite::load("ops");
    let device = backend.device_id();
    let geometry = AttentionGeometry {
        batch: 2,
        heads: 2,
        kv_heads: 2,
        queries: 4,
        keys: 4,
        head_dim: 5,
    };
    let q_values = suite.tensor("attention.q").f32_values();
    let k_values = suite.tensor("attention.k").f32_values();
    let v_values = suite.tensor("attention.v").f32_values();
    let mask_values = suite.tensor("attention.mask").f32_values();

    let q = f32_tensor_on(device, &[2, 2, 4, 5], &q_values);
    let k = f32_tensor_on(device, &[2, 2, 4, 5], &k_values);
    let v = f32_tensor_on(device, &[2, 2, 4, 5], &v_values);
    let mask = f32_tensor_on(device, &[2, 1, 4, 4], &mask_values);
    let mut out = zeros_on(device, &[2, 2, 4, 5]);
    let mut probs = zeros_on(device, &[2, 2, 4, 4]);
    run_two(
        &*backend,
        OpId::MaskedAttention,
        &[&q, &k, &v, &mask],
        &mut out,
        &mut probs,
        &OpParams::default(),
    )
    .expect("vulkan fused attention");

    let out_values = out.as_f32_slice().expect("attention out").to_vec();
    let probs_values = probs.as_f32_slice().expect("attention probs").to_vec();
    measure_fixture(&suite, "attention.probs", &probs_values, &[2, 2, 4, 4]);
    measure_fixture(&suite, "attention.out", &out_values, &[2, 2, 4, 5]);

    let (cpu_out, cpu_probs) = cpu_attention(
        &q_values,
        &k_values,
        &v_values,
        Some(&mask_values),
        2,
        1,
        &geometry,
        false,
    );
    measure(
        "attention probs vs CpuBackend decomposition",
        &probs_values,
        &cpu_probs,
        3.0e-6,
        1.0e-6,
    );
    measure(
        "attention out vs CpuBackend decomposition",
        &out_values,
        &cpu_out,
        3.0e-6,
        1.0e-6,
    );

    // Batch 1, both heads, query row 2 is fully masked by the fixture: the
    // probabilities must be exactly uniform and finite, and so must the result.
    for head in 0..2usize {
        let batch = 1;
        let heads = 2;
        let base = ((batch * heads + head) * 4 + 2) * 4;
        let row = &probs_values[base..base + 4];
        assert_exactly_uniform("attention all-masked probs row", row, 0.25);
        eprintln!(
            "[fn3vk][contract] attention all-masked probs row (batch 1, head {head}) = {row:?}"
        );
    }
    assert_all_finite("attention probs", &probs_values);
    assert_all_finite("attention out", &out_values);

    let support = backend.supports(OpId::MaskedAttention);
    assert!(
        support.is_full(),
        "the fused attention contract is implemented in full: {support}"
    );
    assert!(
        backend.capabilities().features.masked_attention,
        "the fused attention feature flag must be advertised"
    );
    assert_evidence(&backend, OpId::MaskedAttention);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

// ---------------------------------------------------------------------------
// Real model geometry
// ---------------------------------------------------------------------------

#[test]
fn real_shape_layernorm_13_rows_of_128() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let dims = [1u64, 13, 128];
    let values = pseudo_values(13 * 128, 0x61_0001, -3.0, 3.0);
    let weight_values = pseudo_values(128, 0x61_0002, 0.5, 1.5);
    let bias_values = pseudo_values(128, 0x61_0003, -0.5, 0.5);

    let input = f32_tensor_on(device, &dims, &values);
    let weight = f32_tensor_on(device, &[128], &weight_values);
    let bias = f32_tensor_on(device, &[128], &bias_values);
    let mut out = zeros_on(device, &dims);
    run(
        &*backend,
        OpId::LayerNorm,
        &[&input, &weight, &bias],
        &mut out,
        &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
    )
    .expect("vulkan layernorm [1,13,128]");
    let actual = out.as_f32_slice().expect("layernorm output").to_vec();

    let cpu = DeviceId::Cpu;
    let input_cpu = f32_tensor_on(cpu, &dims, &values);
    let weight_cpu = f32_tensor_on(cpu, &[128], &weight_values);
    let bias_cpu = f32_tensor_on(cpu, &[128], &bias_values);
    let mut out_cpu = zeros_on(cpu, &dims);
    run(
        &CpuBackend::new(),
        OpId::LayerNorm,
        &[&input_cpu, &weight_cpu, &bias_cpu],
        &mut out_cpu,
        &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
    )
    .expect("cpu layernorm [1,13,128]");
    let reference = out_cpu.as_f32_slice().expect("cpu layernorm").to_vec();

    measure(
        "layernorm [1,13,128] vs CpuBackend",
        &actual,
        &reference,
        2.0e-5,
        1.0e-5,
    );
    assert_all_finite("layernorm [1,13,128]", &actual);
    assert_evidence(&backend, OpId::LayerNorm);
    assert_no_live_device_memory(&backend);

    // Device timing for this real shape (one measurement on this device only).
    let mut gpu_samples = Vec::new();
    let mut host_samples = Vec::new();
    for _ in 0..20 {
        run(
            &*backend,
            OpId::LayerNorm,
            &[&input, &weight, &bias],
            &mut out,
            &OpParams::default().with_axes([-1]).with_epsilon(1e-5),
        )
        .expect("vulkan layernorm timing");
        let record = backend
            .residency()
            .records()
            .into_iter()
            .rev()
            .find(|record| record.op == OpId::LayerNorm)
            .expect("timing record");
        host_samples.push(record.host_ns);
        if let Some(gpu_ns) = record.gpu_ns {
            gpu_samples.push(gpu_ns);
        }
    }
    report_timing("layernorm [1,13,128]", &mut gpu_samples, &mut host_samples);

    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn real_shape_attention_s17_head_dim_64_with_a_causal_mask() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let geometry = AttentionGeometry {
        batch: 1,
        heads: 2,
        kv_heads: 2,
        queries: 17,
        keys: 17,
        head_dim: 64,
    };
    let q_values = pseudo_values(2 * 17 * 64, 0x62_0001, -1.0, 1.0);
    let k_values = pseudo_values(2 * 17 * 64, 0x62_0002, -1.0, 1.0);
    let v_values = pseudo_values(2 * 17 * 64, 0x62_0003, -1.0, 1.0);
    // Causal additive mask: -FLT_MAX above the diagonal, 0 elsewhere.
    let mut mask_values = vec![0.0f32; 17 * 17];
    for row in 0..17usize {
        for column in 0..17usize {
            if column > row {
                mask_values[row * 17 + column] = NEGATIVE_FLT_MAX;
            }
        }
    }

    let q = f32_tensor_on(device, &[1, 2, 17, 64], &q_values);
    let k = f32_tensor_on(device, &[1, 2, 17, 64], &k_values);
    let v = f32_tensor_on(device, &[1, 2, 17, 64], &v_values);
    let mask = f32_tensor_on(device, &[1, 1, 17, 17], &mask_values);
    let mut out = zeros_on(device, &[1, 2, 17, 64]);
    let mut probs = zeros_on(device, &[1, 2, 17, 17]);
    run_two(
        &*backend,
        OpId::MaskedAttention,
        &[&q, &k, &v, &mask],
        &mut out,
        &mut probs,
        &OpParams::default(),
    )
    .expect("vulkan fused attention S=17");
    let out_values = out.as_f32_slice().expect("attention out").to_vec();
    let probs_values = probs.as_f32_slice().expect("attention probs").to_vec();

    let (cpu_out, cpu_probs) = cpu_attention(
        &q_values,
        &k_values,
        &v_values,
        Some(&mask_values),
        1,
        1,
        &geometry,
        false,
    );
    let atol = scaled_atol(3.0e-6, 17);
    measure(
        "attention S=17 probs vs CpuBackend (atol scaled x sqrt(17/16))",
        &probs_values,
        &cpu_probs,
        atol,
        1.0e-6,
    );
    measure(
        "attention S=17 out vs CpuBackend (atol scaled x sqrt(17/16))",
        &out_values,
        &cpu_out,
        atol,
        1.0e-6,
    );
    assert_all_finite("attention S=17 probs", &probs_values);
    assert_all_finite("attention S=17 out", &out_values);

    assert_evidence(&backend, OpId::MaskedAttention);
    assert_no_live_device_memory(&backend);

    let mut gpu_samples = Vec::new();
    let mut host_samples = Vec::new();
    for _ in 0..20 {
        run_two(
            &*backend,
            OpId::MaskedAttention,
            &[&q, &k, &v, &mask],
            &mut out,
            &mut probs,
            &OpParams::default(),
        )
        .expect("vulkan fused attention timing");
        let record = backend
            .residency()
            .records()
            .into_iter()
            .rev()
            .find(|record| record.op == OpId::MaskedAttention)
            .expect("timing record");
        host_samples.push(record.host_ns);
        if let Some(gpu_ns) = record.gpu_ns {
            gpu_samples.push(gpu_ns);
        }
    }
    report_timing(
        "attention [1,2,17,64] (S=17)",
        &mut gpu_samples,
        &mut host_samples,
    );

    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn cached_decode_rows_over_several_context_lengths() {
    // A decode step is one query row over the cached key extent. With the
    // causal flag the diagonal shifts by `keys - queries`, so every cached key
    // is visible and the row is an ordinary softmax; a kernel that ignored the
    // shift would mask all but the last key. One output: the fused path must
    // not require the optional probs tensor.
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    for keys in [5usize, 17, 33] {
        let geometry = AttentionGeometry {
            batch: 1,
            heads: 2,
            kv_heads: 2,
            queries: 1,
            keys,
            head_dim: 64,
        };
        let q_values = pseudo_values(2 * 64, 0x67_0001 + keys as u64, -1.0, 1.0);
        let k_values = pseudo_values(2 * keys * 64, 0x67_1001 + keys as u64, -1.0, 1.0);
        let v_values = pseudo_values(2 * keys * 64, 0x67_2001 + keys as u64, -1.0, 1.0);
        let q = f32_tensor_on(device, &[1, 2, 1, 64], &q_values);
        let k = f32_tensor_on(device, &[1, 2, keys as u64, 64], &k_values);
        let v = f32_tensor_on(device, &[1, 2, keys as u64, 64], &v_values);
        let mut out = zeros_on(device, &[1, 2, 1, 64]);
        run(
            &*backend,
            OpId::MaskedAttention,
            &[&q, &k, &v],
            &mut out,
            &OpParams::default().with_keep_dims(true),
        )
        .expect("vulkan cached decode attention");
        let actual = out.as_f32_slice().expect("decode out").to_vec();

        let (cpu_out, _) =
            cpu_attention(&q_values, &k_values, &v_values, None, 1, 1, &geometry, true);
        let atol = scaled_atol(3.0e-6, keys);
        measure(
            &format!("cached decode keys={keys} out vs CpuBackend"),
            &actual,
            &cpu_out,
            atol,
            1.0e-6,
        );
        assert_all_finite("cached decode out", &actual);
        assert_evidence(&backend, OpId::MaskedAttention);
    }
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

fn report_timing(label: &str, gpu_samples: &mut [u64], host_samples: &mut [u64]) {
    if gpu_samples.is_empty() {
        eprintln!("[fn3vk][timing] {label}: device timestamps unavailable");
        return;
    }
    gpu_samples.sort_unstable();
    host_samples.sort_unstable();
    let device_min = gpu_samples[0];
    let device_median = gpu_samples[gpu_samples.len() / 2];
    let host_min = host_samples[0];
    let host_median = host_samples[host_samples.len() / 2];
    eprintln!(
        "[fn3vk][timing] {label}: n={} device gpu_ns min={device_min} median={device_median}; \
         host_ns min={host_min} median={host_median} (one submission = upload + dispatch + \
         readback; measured on this device only, not a cross-device claim)",
        gpu_samples.len()
    );
}

#[test]
fn gqa_head_broadcasting_matches_the_expanded_cpu_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let geometry = AttentionGeometry {
        batch: 2,
        heads: 4,
        kv_heads: 2,
        queries: 5,
        keys: 5,
        head_dim: 8,
    };
    let q_values = pseudo_values(2 * 4 * 5 * 8, 0x63_0001, -1.5, 1.5);
    let k_values = pseudo_values(2 * 2 * 5 * 8, 0x63_0002, -1.5, 1.5);
    let v_values = pseudo_values(2 * 2 * 5 * 8, 0x63_0003, -1.5, 1.5);
    let mut mask_values = vec![0.0f32; 2 * 5 * 5];
    for batch in 0..2usize {
        for row in 0..5usize {
            for column in 0..5usize {
                if batch == 1 && column > row {
                    mask_values[(batch * 5 + row) * 5 + column] = NEGATIVE_FLT_MAX;
                }
            }
        }
    }

    let q = f32_tensor_on(device, &[2, 4, 5, 8], &q_values);
    let k = f32_tensor_on(device, &[2, 2, 5, 8], &k_values);
    let v = f32_tensor_on(device, &[2, 2, 5, 8], &v_values);
    let mask = f32_tensor_on(device, &[2, 1, 5, 5], &mask_values);
    let mut out = zeros_on(device, &[2, 4, 5, 8]);
    let mut probs = zeros_on(device, &[2, 4, 5, 5]);
    run_two(
        &*backend,
        OpId::MaskedAttention,
        &[&q, &k, &v, &mask],
        &mut out,
        &mut probs,
        &OpParams::default(),
    )
    .expect("vulkan grouped-query attention");
    let out_values = out.as_f32_slice().expect("attention out").to_vec();
    let probs_values = probs.as_f32_slice().expect("attention probs").to_vec();

    let (cpu_out, cpu_probs) = cpu_attention(
        &q_values,
        &k_values,
        &v_values,
        Some(&mask_values),
        2,
        1,
        &geometry,
        false,
    );
    measure(
        "GQA 4 query heads / 2 KV heads probs vs CpuBackend",
        &probs_values,
        &cpu_probs,
        3.0e-6,
        1.0e-6,
    );
    measure(
        "GQA 4 query heads / 2 KV heads out vs CpuBackend",
        &out_values,
        &cpu_out,
        3.0e-6,
        1.0e-6,
    );

    // The comparison above is the mapping proof: the CPU reference expands each
    // KV head `h / (heads / kv_heads)` across its query group, so any other
    // mapping (for example `h % kv_heads`) would disagree. Second check: heads
    // that share a KV head still have distinct queries, so a kernel that
    // collapsed the head axis would make these rows identical.
    for batch in 0..2usize {
        for pair in 0..2usize {
            let head = batch * 4 + pair * 2;
            let left_probs = head * 5 * 5;
            let right_probs = (head + 1) * 5 * 5;
            let left_out = head * 5 * 8;
            let right_out = (head + 1) * 5 * 8;
            assert_ne!(
                &probs_values[left_probs..left_probs + 25],
                &probs_values[right_probs..right_probs + 25],
                "query heads with distinct queries must not share probabilities"
            );
            assert_ne!(
                &out_values[left_out..left_out + 40],
                &out_values[right_out..right_out + 40],
                "query heads with distinct queries must not share outputs"
            );
        }
    }
    assert_all_finite("GQA probs", &probs_values);
    assert_all_finite("GQA out", &out_values);

    assert_evidence(&backend, OpId::MaskedAttention);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn causal_mode_and_a_sequence_longer_than_the_fixture() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();
    let geometry = AttentionGeometry {
        batch: 1,
        heads: 2,
        kv_heads: 1,
        queries: 33,
        keys: 33,
        head_dim: 16,
    };
    let q_values = pseudo_values(2 * 33 * 16, 0x64_0001, -1.0, 1.0);
    let k_values = pseudo_values(33 * 16, 0x64_0002, -1.0, 1.0);
    let v_values = pseudo_values(33 * 16, 0x64_0003, -1.0, 1.0);

    let q = f32_tensor_on(device, &[1, 2, 33, 16], &q_values);
    let k = f32_tensor_on(device, &[1, 1, 33, 16], &k_values);
    let v = f32_tensor_on(device, &[1, 1, 33, 16], &v_values);
    let mut out = zeros_on(device, &[1, 2, 33, 16]);
    let mut probs = zeros_on(device, &[1, 2, 33, 33]);
    // No mask input at all: the causal attribute is the only mask. `OpParams`
    // has exactly one boolean, so the fusion's causal flag rides in `keep_dims`
    // and the output shape is still q's shape.
    run_two(
        &*backend,
        OpId::MaskedAttention,
        &[&q, &k, &v],
        &mut out,
        &mut probs,
        &OpParams::default().with_keep_dims(true),
    )
    .expect("vulkan causal attention S=33");
    let out_values = out.as_f32_slice().expect("attention out").to_vec();
    let probs_values = probs.as_f32_slice().expect("attention probs").to_vec();

    let (cpu_out, cpu_probs) =
        cpu_attention(&q_values, &k_values, &v_values, None, 1, 1, &geometry, true);
    let atol = scaled_atol(3.0e-6, 33);
    measure(
        "causal S=33 (GQA 2/1, no mask input) probs vs CpuBackend",
        &probs_values,
        &cpu_probs,
        atol,
        1.0e-6,
    );
    measure(
        "causal S=33 (GQA 2/1, no mask input) out vs CpuBackend",
        &out_values,
        &cpu_out,
        atol,
        1.0e-6,
    );

    // Every probability above the diagonal must be exactly zero, and the row
    // sums must be 1 within a documented bound.
    for head in 0..2usize {
        for row in 0..33usize {
            let base = (head * 33 + row) * 33;
            for column in (row + 1)..33 {
                assert_eq!(
                    probs_values[base + column],
                    0.0,
                    "causal probs must be exactly zero above the diagonal (head {head}, row \
                     {row}, column {column})"
                );
            }
            let sum: f32 = probs_values[base..base + row + 1].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-5,
                "causal row {row} of head {head} sums to {sum}"
            );
        }
    }
    assert_all_finite("causal S=33 probs", &probs_values);
    assert_all_finite("causal S=33 out", &out_values);

    assert_evidence(&backend, OpId::MaskedAttention);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

#[test]
fn reductions_match_the_cpu_reference() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();

    // ReduceSum over the trailing axis: a plain ascending add, so the
    // comparison is exact (0 tolerance). This is what proves the accumulation
    // order reproduces the CPU reference.
    let dims = [2u64, 3, 5];
    let values = pseudo_values(2 * 3 * 5, 0x65_0001, -2.0, 2.0);
    let input = f32_tensor_on(device, &dims, &values);
    let mut out = zeros_on(device, &[2, 3]);
    run(
        &*backend,
        OpId::ReduceSum,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("vulkan reduce_sum [-1]");
    let actual = out.as_f32_slice().expect("reduce_sum output").to_vec();
    let input_cpu = f32_tensor_on(DeviceId::Cpu, &dims, &values);
    let mut out_cpu = zeros_on(DeviceId::Cpu, &[2, 3]);
    run(
        &CpuBackend::new(),
        OpId::ReduceSum,
        &[&input_cpu],
        &mut out_cpu,
        &OpParams::default().with_axes([-1]),
    )
    .expect("cpu reduce_sum [-1]");
    let reference = out_cpu.as_f32_slice().expect("cpu reduce_sum").to_vec();
    measure(
        "reduce_sum axes=[-1] (exact) vs CpuBackend",
        &actual,
        &reference,
        0.0,
        0.0,
    );

    // ReduceMean over the same axis set. The accumulation is the same exact
    // ascending add; only the final division can differ, because Vulkan requires
    // 2.5 ULP (not IEEE) for `OpFDiv`, so the published bound is 1e-6.
    for (axes, keep_dims, out_dims, label) in [
        (vec![-1i64], false, vec![2u64, 3], "reduce_mean axes=[-1]"),
        (
            vec![1],
            true,
            vec![2, 1, 5],
            "reduce_mean axes=[1] keep_dims",
        ),
        (vec![0, 1, 2], false, vec![], "reduce_mean all axes"),
    ] {
        let input = f32_tensor_on(device, &dims, &values);
        let mut out = zeros_on(device, &out_dims);
        let params = OpParams::default()
            .with_axes(axes.clone())
            .with_keep_dims(keep_dims);
        run(&*backend, OpId::ReduceMean, &[&input], &mut out, &params).expect("vulkan reduce_mean");
        let actual = out.as_f32_slice().expect("reduce_mean output").to_vec();

        let input_cpu = f32_tensor_on(DeviceId::Cpu, &dims, &values);
        let mut out_cpu = zeros_on(DeviceId::Cpu, &out_dims);
        run(
            &CpuBackend::new(),
            OpId::ReduceMean,
            &[&input_cpu],
            &mut out_cpu,
            &params,
        )
        .expect("cpu reduce_mean");
        let reference = out_cpu.as_f32_slice().expect("cpu reduce_mean").to_vec();
        measure(
            &format!("{label} vs CpuBackend"),
            &actual,
            &reference,
            1.0e-6,
            1.0e-6,
        );
    }

    // ReduceSum over two axes with keep_dims.
    let input = f32_tensor_on(device, &dims, &values);
    let mut out = zeros_on(device, &[2, 1, 1]);
    let params = OpParams::default().with_axes([1, 2]).with_keep_dims(true);
    run(&*backend, OpId::ReduceSum, &[&input], &mut out, &params).expect("vulkan reduce_sum");
    let actual = out.as_f32_slice().expect("reduce_sum output").to_vec();
    let input_cpu = f32_tensor_on(DeviceId::Cpu, &dims, &values);
    let mut out_cpu = zeros_on(DeviceId::Cpu, &[2, 1, 1]);
    run(
        &CpuBackend::new(),
        OpId::ReduceSum,
        &[&input_cpu],
        &mut out_cpu,
        &params,
    )
    .expect("cpu reduce_sum");
    let reference = out_cpu.as_f32_slice().expect("cpu reduce_sum").to_vec();
    measure(
        "reduce_sum axes=[1,2] keep_dims vs CpuBackend",
        &actual,
        &reference,
        0.0,
        0.0,
    );

    assert!(backend.supports(OpId::ReduceMean).is_full());
    assert!(backend.supports(OpId::ReduceSum).is_full());
    assert_evidence(&backend, OpId::ReduceMean);
    assert_evidence(&backend, OpId::ReduceSum);
    assert_no_live_device_memory(&backend);

    // ArgMax: f32 with a tie (lowest index wins) and i32.
    let argmax_dims = [2u64, 4];
    let argmax_values = [0.5f32, 3.0, 3.0, 1.0, -1.0, -1.0, -2.0, -3.0];
    let input = f32_tensor_on(device, &argmax_dims, &argmax_values);
    let mut out = zeros_i64_on(device, &[2]);
    run(
        &*backend,
        OpId::ArgMax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("vulkan argmax f32");
    let actual = out.as_i64_slice().expect("argmax output").to_vec();
    let input_cpu = f32_tensor_on(DeviceId::Cpu, &argmax_dims, &argmax_values);
    let mut out_cpu = zeros_i64_on(DeviceId::Cpu, &[2]);
    run(
        &CpuBackend::new(),
        OpId::ArgMax,
        &[&input_cpu],
        &mut out_cpu,
        &OpParams::default().with_axes([-1]),
    )
    .expect("cpu argmax");
    assert_eq!(actual, out_cpu.as_i64_slice().expect("cpu argmax").to_vec());
    assert_eq!(actual, vec![1i64, 0], "ties keep the lowest index");

    let i32_values = [7i32, 3, 9, 9];
    let input = i32_tensor_on(device, &[2, 2], &i32_values);
    let mut out = zeros_i64_on(device, &[2]);
    run(
        &*backend,
        OpId::ArgMax,
        &[&input],
        &mut out,
        &OpParams::default().with_axes([-1]),
    )
    .expect("vulkan argmax i32");
    assert_eq!(
        out.as_i64_slice().expect("argmax i32").to_vec(),
        vec![0i64, 0]
    );
    assert!(assert_unsupported_names_issue(|| {
        let i64_values = [1i64, 2, 3, 4];
        let input = i64_tensor_on(device, &[2, 2], &i64_values);
        let mut out = zeros_i64_on(device, &[2]);
        run(
            &*backend,
            OpId::ArgMax,
            &[&input],
            &mut out,
            &OpParams::default().with_axes([-1]),
        )
    }));

    let support = backend.supports(OpId::ArgMax);
    assert!(support.is_supported() && !support.is_full(), "{support}");

    assert_evidence(&backend, OpId::ArgMax);
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

/// `true` when the error is `UnsupportedOp` naming FN3-15 (#61).
fn assert_unsupported_names_issue(action: impl FnOnce() -> Result<(), Fn3Error>) -> bool {
    match action() {
        Err(Fn3Error::UnsupportedOp {
            op,
            backend,
            reason,
        }) => {
            eprintln!("[fn3vk][reject] op={op} backend={backend} reason={reason}");
            assert!(
                reason.contains("#61"),
                "an UnsupportedOp must name the owning issue: {reason}"
            );
            assert_eq!(backend, "vulkan");
            true
        }
        Err(other) => panic!("expected UnsupportedOp naming #61, got {other}"),
        Ok(()) => panic!("expected UnsupportedOp naming #61, but the op succeeded"),
    }
}

// ---------------------------------------------------------------------------
// Honest support and rejection paths
// ---------------------------------------------------------------------------

#[test]
fn unsupported_requests_are_rejected_by_name() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = backend.device_id();

    // Softmax over a non-trailing axis.
    let input = f32_tensor_on(device, &[2, 3, 4], &pseudo_values(24, 0x66_0001, -1.0, 1.0));
    let mut out = zeros_on(device, &[2, 3, 4]);
    assert!(assert_unsupported_names_issue(|| {
        run(
            &*backend,
            OpId::Softmax,
            &[&input],
            &mut out,
            &OpParams::default().with_axes([1]),
        )
    }));

    // MaskedAttention with more keys than the shared-memory row supports.
    let q = f32_tensor_on(device, &[1, 1, 1, 1], &[0.5f32]);
    let oversized = 4097u64;
    let k = f32_tensor_on(
        device,
        &[1, 1, oversized, 1],
        &vec![0.25f32; oversized as usize],
    );
    let v = f32_tensor_on(
        device,
        &[1, 1, oversized, 1],
        &vec![0.5f32; oversized as usize],
    );
    let mut out = zeros_on(device, &[1, 1, 1, 1]);
    assert!(assert_unsupported_names_issue(|| {
        run(
            &*backend,
            OpId::MaskedAttention,
            &[&q, &k, &v],
            &mut out,
            &OpParams::default(),
        )
    }));

    // MaskedAttention with a head count that is not a multiple of the KV heads.
    let q = f32_tensor_on(
        device,
        &[1, 3, 2, 4],
        &pseudo_values(24, 0x66_0002, -1.0, 1.0),
    );
    let k = f32_tensor_on(
        device,
        &[1, 2, 2, 4],
        &pseudo_values(16, 0x66_0003, -1.0, 1.0),
    );
    let v = f32_tensor_on(
        device,
        &[1, 2, 2, 4],
        &pseudo_values(16, 0x66_0004, -1.0, 1.0),
    );
    let mut out = zeros_on(device, &[1, 3, 2, 4]);
    assert!(assert_unsupported_names_issue(|| {
        run(
            &*backend,
            OpId::MaskedAttention,
            &[&q, &k, &v],
            &mut out,
            &OpParams::default(),
        )
    }));

    // A non-contiguous softmax operand (a transposed view).
    let base = f32_tensor_on(device, &[4, 3], &pseudo_values(12, 0x66_0005, -1.0, 1.0));
    let transposed = transpose_last_two(&base);
    let view = TensorView::new(base.storage(), transposed).expect("transposed view");
    let mut out = zeros_on(device, &[3, 4]);
    {
        let mut out_view = out.as_mut_view().unwrap();
        assert!(assert_unsupported_names_issue(|| {
            backend.execute_op(
                OpId::Softmax,
                &[view],
                std::slice::from_mut(&mut out_view),
                &OpParams::default().with_axes([-1]),
            )
        }));
    }

    // Nothing above may have dispatched, so nothing may be recorded.
    assert_eq!(
        backend.residency().summary().executions,
        0,
        "a rejected request must not dispatch"
    );
    assert_no_live_device_memory(&backend);
    drop(backend);
    assert_validation_clean(counts_before);
}

#[test]
fn verdicts_are_three_valued_and_honest() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let cases = [
        (
            OpId::LayerNorm,
            Support::Full,
            "the whole trailing-axis contract",
        ),
        (
            OpId::ReduceMean,
            Support::Full,
            "any axes, either keep_dims",
        ),
        (OpId::ReduceSum, Support::Full, "any axes, either keep_dims"),
        (
            OpId::Softmax,
            Support::Partial {
                reason: String::new(),
            },
            "trailing axis",
        ),
        (
            OpId::ArgMax,
            Support::Partial {
                reason: String::new(),
            },
            "f32 and i32",
        ),
        (
            OpId::MaskedAttention,
            Support::Full,
            "the whole fused attention contract",
        ),
    ];
    for (op, expected, needle) in cases {
        let support = backend.supports(op);
        match (&expected, &support) {
            (Support::Full, Support::Full) => {}
            (Support::Partial { .. }, Support::Partial { .. }) => {}
            _ => panic!("{op}: expected {expected}, got {support}"),
        }
        if !support.is_full() {
            let reason = support.reason().unwrap_or_default();
            assert!(
                reason.contains(needle),
                "{op}: the partial verdict must name its subset ({needle}): {reason}"
            );
        }
    }
    // Ops owned by other issues either are honestly unsupported with an owner
    // named, or are implemented by their owner's kernel (MatMul is #60's and is
    // Full on this branch). This asserts the invariant, not the set of the day.
    for op in [OpId::MatMul, OpId::Transpose, OpId::Where] {
        let support = backend.supports(op);
        if !support.is_supported() {
            let reason = support.reason().unwrap_or_default();
            assert!(
                reason.contains('#'),
                "{op}: an unsupported verdict must name the owning issue: {reason}"
            );
        }
    }
    assert!(backend.capabilities().features.masked_attention);
    drop(backend);
    assert_validation_clean(counts_before);
}
