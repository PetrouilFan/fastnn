//! FN3-13 (#59): Vulkan pointwise, transform and embedding kernel conformance.
//!
//! Run with:
//!
//! ```bash
//! CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_pointwise \
//!   -- --nocapture --test-threads=1
//! ```
//!
//! # What these tests prove
//!
//! For every op `pointwise::register`, `transform::register` and
//! `embedding::register` register:
//!
//! 1. the kernel produces the same result as `CpuBackend` through the same
//!    [`Backend::execute_op`] path, within a stated bound (exact for the ops
//!    whose arithmetic is a copy or a single IEEE-754 operation, and the
//!    `docs/reports/fn3-03-parity-fixtures.md` bound for the transcendental
//!    ones);
//! 2. the `ExecutionRecord` `proves_device_execution()` and reports
//!    `inputs_read_back == false`;
//! 3. the validation layer reports no error or warning across the run;
//! 4. at least one case per family has a size that is not a multiple of the
//!    workgroup (a prime-sized vector, and a `Bool` output that forces the
//!    4-byte packing tail).
//!
//! The embedding lookup and the GELU fixture are compared against the committed
//! `fixtures/parity/ops` payloads through the read-only `fn3_fixtures` loader,
//! as well as against the CPU reference.
//!
//! # Tolerances used here (and why)
//!
//! | family | bound vs CPU | source |
//! |---|---|---|
//! | `Add` `Sub` `Mul` `Div` `Neg` `Relu` `Sqrt` | exact | one IEEE-754 operation per element |
//! | `Tanh` `Sin` `Cos` | `atol 1e-6 + rtol 1e-6` | GPU transcendental vs platform libm (<= ~8 f32 ulp) |
//! | `Pow` | `atol 1e-5 + rtol 1e-5` | GPU `pow` vs libm `powf` |
//! | `Erf` | `atol 2e-7` | Abramowitz & Stegun 7.1.26 bound is 1.5e-7 |
//! | `Gelu` | `atol 2e-6 + rtol 1e-6` | the parity fixture's GELU row |
//! | `Cast` `Where` `Equal` `Less` `Greater` | exact | integer/selection/compare |
//! | `Gather` (embedding) | exact | the parity fixture's embedding row |
//! | `Transpose` `Reshape` `Flatten` `Squeeze` `Unsqueeze` `Slice` `Split` `Concat` `Expand` `Shape` `ConstantOfShape` | exact | byte/structure moves |
//! | `Range` | exact | `start + i * delta`, evaluated identically |
//!
//! Physical-GPU tests skip cleanly with `[fn3vk][skip]` when no non-software
//! adapter is present.

#![cfg(feature = "vulkan")]

mod fn3_fixtures;

use std::sync::{Arc, Mutex};

use fn3_fixtures::{Actual, Suite};

use fastnn::fn3::backend::cpu::CpuBackend;
use fastnn::fn3::backend::vulkan::instance::validation_counts;
use fastnn::fn3::backend::vulkan::residency::ExecutionRecord;
use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::backend::{Backend, OpParams, Support};
use fastnn::fn3::device::{DeviceId, DeviceRequest, PhysicalDeviceId};
use fastnn::fn3::dtype::DType;
use fastnn::fn3::error::Fn3Error;
use fastnn::fn3::ops::OpId;
use fastnn::fn3::shape::Shape;
use fastnn::fn3::storage::Storage;
use fastnn::fn3::tensor::{Tensor, TensorMeta, TensorMut, TensorView};

/// Serialises physical-GPU use and validation-count assertions.
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

/// The first real (non-software) adapter with a compute queue.
///
/// Selection is by UUID when the caller pins one; never by index, because the
/// loader also enumerates lavapipe.
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
                 present"
            );
        }
        eprintln!(
            "[fn3vk][skip] no physical Vulkan device with a compute queue is present; this test \
             file has no software-ICD lane because it is a physical-device gate"
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
                "[fn3vk][test] device=\"{}\" uuid={} driver={} queue_family={} validation={}",
                adapter.name,
                adapter.uuid_string(),
                adapter.driver_version_string(),
                backend.vulkan_device().queue_family(),
                backend.validation_enabled()
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

fn gpu_device(backend: &VulkanBackend) -> DeviceId {
    DeviceId::Vulkan(PhysicalDeviceId::new(backend.adapter().index))
}

// ---------------------------------------------------------------------------
// Tensor harness
// ---------------------------------------------------------------------------

fn tensor_on(device: DeviceId, dims: &[u64], dtype: DType, bytes: &[u8]) -> Tensor {
    let shape = Shape::from_dims(dims.iter().copied());
    let storage = Storage::from_bytes(bytes).expect("host storage");
    let meta = TensorMeta::contiguous(shape, dtype, device).expect("tensor metadata");
    Tensor::from_storage(storage, meta).expect("tensor")
}

fn zeros_on(device: DeviceId, dims: &[u64], dtype: DType) -> Tensor {
    // `Tensor::zeros` refuses a Vulkan device (the alpha storage model is
    // host-backed); the documented validation path is `from_storage` with
    // Vulkan-tagged metadata, which is what the FN3-11 smoke test uses too.
    let shape = Shape::from_dims(dims.iter().copied());
    let elements = shape.numel().expect("element count");
    let bytes = vec![0u8; elements as usize * dtype.size_bytes()];
    tensor_on(device, dims, dtype, &bytes)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    bytemuck::cast_slice(values).to_vec()
}

fn bool_bytes(values: &[bool]) -> Vec<u8> {
    values.iter().map(|&value| u8::from(value)).collect()
}

/// The contiguous window of a tensor, as bytes.
fn window(tensor: &Tensor) -> Vec<u8> {
    let start = tensor.meta().byte_offset().expect("byte offset");
    let len = tensor.meta().byte_len().expect("byte length");
    tensor.storage().as_bytes()[start..start + len].to_vec()
}

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    let bytes = window(tensor);
    bytemuck::cast_slice(&bytes).to_vec()
}

fn i64_values(tensor: &Tensor) -> Vec<i64> {
    let bytes = window(tensor);
    bytemuck::cast_slice(&bytes).to_vec()
}

fn execute<B: Backend + ?Sized>(
    backend: &B,
    op: OpId,
    inputs: &[&Tensor],
    outputs: &mut [&mut Tensor],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().expect("view"))
        .collect();
    let mut out_views: Vec<TensorMut<'_>> = outputs
        .iter_mut()
        .map(|tensor| tensor.as_mut_view().expect("mutable view"))
        .collect();
    backend.execute_op(op, &views, &mut out_views, params)
}

/// Execute an op whose outputs are already built as mutable views.
fn execute_views<B: Backend + ?Sized>(
    backend: &B,
    op: OpId,
    inputs: &[&Tensor],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Result<(), Fn3Error> {
    let views: Vec<TensorView<'_>> = inputs
        .iter()
        .map(|tensor| tensor.as_view().expect("view"))
        .collect();
    backend.execute_op(op, &views, outputs, params)
}

/// Run one op on the GPU, assert its evidence record, and return the output.
///
/// The record assertions are the "no silent host fallback" gate from the
/// FN3-11/FN3-13 contract: one submission, a non-zero dispatch, at least one
/// device-local binding and no input readback.
fn run_gpu(
    backend: &VulkanBackend,
    op: OpId,
    inputs: &[&Tensor],
    out_dims: &[u64],
    out_dtype: DType,
    params: &OpParams,
) -> Tensor {
    let mut out = zeros_on(gpu_device(backend), out_dims, out_dtype);
    backend.residency().clear();
    execute(backend, op, inputs, &mut [&mut out], params)
        .unwrap_or_else(|error| panic!("{op} on the GPU failed: {error}"));
    let records = backend.residency().records();
    assert_eq!(records.len(), 1, "{op}: one op must append one record");
    check_record(op, &records[0]);
    out
}

/// Run one op on the CPU reference and return the output bytes.
fn run_cpu(
    op: OpId,
    inputs: &[&Tensor],
    out_dims: &[u64],
    out_dtype: DType,
    params: &OpParams,
) -> Tensor {
    let mut out = zeros_on(DeviceId::Cpu, out_dims, out_dtype);
    execute(&CpuBackend::new(), op, inputs, &mut [&mut out], params)
        .unwrap_or_else(|error| panic!("{op} on the CPU reference failed: {error}"));
    out
}

fn check_record(op: OpId, record: &ExecutionRecord) {
    assert_eq!(record.op, op, "{op}: record names the wrong op");
    assert_eq!(record.backend, "vulkan");
    assert!(
        record.proves_device_execution(),
        "{op}: record must prove device execution: {record:?}"
    );
    assert_eq!(record.submissions, 1, "{op}: one submission per op");
    assert!(!record.inputs_read_back, "{op}: an input was read back");
    assert!(record.device_local_bytes > 0, "{op}: no device memory");
    assert!(
        record.readback_bytes > 0,
        "{op}: the output was never read back"
    );
}

/// Prove the validation layer produced nothing during this test.
///
/// The counters are process-global and monotonic, so a zero delta also means
/// the absolute total stayed at zero for the whole run.
fn assert_validation_unchanged(before: (u64, u64)) {
    let after = validation_counts();
    eprintln!(
        "[fn3vk][validation] delta_errors={} delta_warnings={} total_errors={} total_warnings={}",
        after.0 - before.0,
        after.1 - before.1,
        after.0,
        after.1
    );
    assert_eq!(
        after, before,
        "validation layer reported an error or warning; see the [fn3vk][validation] lines above"
    );
}

fn assert_exact(label: &str, actual: &Tensor, reference: &Tensor) {
    let got = window(actual);
    let want = window(reference);
    assert_eq!(
        got,
        want,
        "{label}: bytes differ ({} vs {} bytes)",
        got.len(),
        want.len()
    );
}

/// Measured `(max_abs, mean_abs)` between two f32 tensors.
fn f32_error(actual: &Tensor, reference: &Tensor) -> (f64, f64) {
    let got = f32_values(actual);
    let want = f32_values(reference);
    assert_eq!(got.len(), want.len(), "element count");
    let mut max_abs = 0.0f64;
    let mut sum = 0.0f64;
    for (index, (&actual, &reference)) in got.iter().zip(want.iter()).enumerate() {
        let error = if actual == reference || (actual.is_nan() && reference.is_nan()) {
            // Equal, including equal infinities and matching NaNs.
            0.0
        } else if actual.is_finite() && reference.is_finite() {
            (f64::from(actual) - f64::from(reference)).abs()
        } else {
            panic!("non-finite mismatch at {index}: gpu={actual} cpu={reference}");
        };
        max_abs = max_abs.max(error);
        sum += error;
    }
    (
        max_abs,
        if want.is_empty() {
            0.0
        } else {
            sum / want.len() as f64
        },
    )
}

fn assert_within(label: &str, actual: &Tensor, reference: &Tensor, atol: f64, rtol: f64) {
    let (max_abs, mean_abs) = f32_error(actual, reference);
    let got = f32_values(actual);
    let want = f32_values(reference);
    let mut worst = 0.0f64;
    for (&actual, &reference) in got.iter().zip(want.iter()) {
        let error = (f64::from(actual) - f64::from(reference)).abs()
            - (atol + rtol * f64::from(reference).abs());
        worst = worst.max(error);
    }
    eprintln!(
        "[fn3vk][parity] {label:<58} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} \
         bound=atol {atol:.1e} + rtol {rtol:.1e}"
    );
    assert!(
        worst <= 0.0,
        "{label}: max_abs {max_abs:.3e} / mean_abs {mean_abs:.3e} exceeds \
         atol {atol:.1e} + rtol {rtol:.1e}"
    );
}

/// A deterministic spread in `[-bound, bound)`.
fn spread(count: usize, seed: u64, bound: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32) / ((1u64 << 31) as f32);
            (unit * 2.0 - 1.0) * bound
        })
        .collect()
}

fn positive(count: usize, seed: u64) -> Vec<f32> {
    spread(count, seed, 4.0)
        .into_iter()
        .map(|value| value.abs() + 0.01)
        .collect()
}

fn pattern(count: usize, seed: u64) -> Vec<f32> {
    (0..count)
        .map(|index| (((index as u64 * 7 + seed) % 13) as f32) - 6.0)
        .collect()
}

// ---------------------------------------------------------------------------
// Pointwise arithmetic
// ---------------------------------------------------------------------------

#[test]
fn broadcast_binary_ops_replace_the_smoke_kernel_and_match_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);

    // The capability and verdict the contract requires of #59.
    assert!(
        backend.capabilities().features.broadcasting,
        "Add must advertise broadcasting"
    );
    assert!(
        backend.supports(OpId::Add).is_full(),
        "{:?}",
        backend.supports(OpId::Add)
    );
    for op in [OpId::Sub, OpId::Mul, OpId::Div, OpId::Pow] {
        assert!(backend.supports(op).is_full(), "{op}");
    }

    // [3, 1, 5] op [4, 5] -> [3, 4, 5]: both operands broadcast.
    let lhs_dims = [3u64, 1, 5];
    let rhs_dims = [4u64, 5];
    let out_dims = [3u64, 4, 5];
    let lhs = spread(15, 3, 3.0);
    let rhs = spread(20, 5, 2.0);

    for (op, atol, rtol) in [
        (OpId::Add, 0.0, 0.0),
        (OpId::Sub, 0.0, 0.0),
        (OpId::Mul, 0.0, 0.0),
        (OpId::Div, 1e-6, 1e-6),
    ] {
        let gpu_lhs = tensor_on(device, &lhs_dims, DType::F32, &f32_bytes(&lhs));
        let gpu_rhs = tensor_on(device, &rhs_dims, DType::F32, &f32_bytes(&rhs));
        let cpu_lhs = tensor_on(DeviceId::Cpu, &lhs_dims, DType::F32, &f32_bytes(&lhs));
        let cpu_rhs = tensor_on(DeviceId::Cpu, &rhs_dims, DType::F32, &f32_bytes(&rhs));
        let params = OpParams::default();
        let got = run_gpu(
            &backend,
            op,
            &[&gpu_lhs, &gpu_rhs],
            &out_dims,
            DType::F32,
            &params,
        );
        let want = run_cpu(op, &[&cpu_lhs, &cpu_rhs], &out_dims, DType::F32, &params);
        if atol == 0.0 && rtol == 0.0 {
            assert_exact(&format!("{op} broadcast [3,1,5] op [4,5]"), &got, &want);
        } else {
            assert_within(&format!("{op} broadcast"), &got, &want, atol, rtol);
        }
    }

    // Same-shape, right-aligned scalar, and a prime-sized tail that is not a
    // multiple of the 64-invocation workgroup.
    let params = OpParams::default();
    let tail = 100_003usize;
    let a = pattern(tail, 1);
    let b = pattern(tail, 2);
    let gpu_a = tensor_on(device, &[tail as u64], DType::F32, &f32_bytes(&a));
    let gpu_b = tensor_on(device, &[tail as u64], DType::F32, &f32_bytes(&b));
    let cpu_a = tensor_on(DeviceId::Cpu, &[tail as u64], DType::F32, &f32_bytes(&a));
    let cpu_b = tensor_on(DeviceId::Cpu, &[tail as u64], DType::F32, &f32_bytes(&b));
    let got = run_gpu(
        &*backend,
        OpId::Add,
        &[&gpu_a, &gpu_b],
        &[tail as u64],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Add,
        &[&cpu_a, &cpu_b],
        &[tail as u64],
        DType::F32,
        &params,
    );
    assert_exact("add prime tail", &got, &want);

    // Pow broadcast over the same prime-sized vector, with a scalar exponent;
    // `spread` includes negative bases, so this exercises the negative-base
    // branch of the shader's C-pow emulation.
    let exponents = vec![3.0f32];
    let gpu_exp = tensor_on(device, &[1], DType::F32, &f32_bytes(&exponents));
    let cpu_exp = tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&exponents));
    let got = run_gpu(
        &*backend,
        OpId::Pow,
        &[&gpu_a, &gpu_exp],
        &[tail as u64],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Pow,
        &[&cpu_a, &cpu_exp],
        &[tail as u64],
        DType::F32,
        &params,
    );
    assert_within("pow cubed prime tail", &got, &want, 1e-5, 1e-5);

    let scalar = vec![2.5f32];
    let gpu_s = tensor_on(device, &[1], DType::F32, &f32_bytes(&scalar));
    let cpu_s = tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&scalar));
    let got = run_gpu(
        &*backend,
        OpId::Mul,
        &[&gpu_a, &gpu_s],
        &[tail as u64],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Mul,
        &[&cpu_a, &cpu_s],
        &[tail as u64],
        DType::F32,
        &params,
    );
    assert_exact("mul by a scalar", &got, &want);

    // Pow with a negative base: the pinned graph's `x^3` (gelu_new) really does
    // see negative activations, so this is the branch that must not be NaN.
    let base = vec![-2.0f32, -0.5, 0.0, 0.5, 2.0, 3.0, -1.25, 4.0];
    let exponents = vec![2.0f32, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0];
    let gpu_base = tensor_on(device, &[8], DType::F32, &f32_bytes(&base));
    let gpu_exp = tensor_on(device, &[8], DType::F32, &f32_bytes(&exponents));
    let cpu_base = tensor_on(DeviceId::Cpu, &[8], DType::F32, &f32_bytes(&base));
    let cpu_exp = tensor_on(DeviceId::Cpu, &[8], DType::F32, &f32_bytes(&exponents));
    let got = run_gpu(
        &*backend,
        OpId::Pow,
        &[&gpu_base, &gpu_exp],
        &[8],
        DType::F32,
        &params,
    );
    let want = run_cpu(OpId::Pow, &[&cpu_base, &cpu_exp], &[8], DType::F32, &params);
    // The GPU `pow` differs from libm `powf` in the last bits (~2 ulp).
    assert_within("pow with negative bases", &got, &want, 1e-5, 1e-5);
    let values = f32_values(&got);
    assert!((values[0] - 4.0).abs() < 1e-5, "(-2)^2 = {}", values[0]);
    assert!(
        (values[1] + 0.125).abs() < 1e-6,
        "(-0.5)^3 must stay negative: {}",
        values[1]
    );
    assert!(
        values.iter().all(|value| value.is_finite()),
        "every pow value must be finite"
    );

    // A fractional exponent on a negative base is NaN in C `pow`; the kernel
    // must agree rather than emit a finite value.
    let fractional = vec![0.5f32];
    let neg_base = vec![-4.0f32];
    let gpu_base = tensor_on(device, &[1], DType::F32, &f32_bytes(&neg_base));
    let gpu_exp = tensor_on(device, &[1], DType::F32, &f32_bytes(&fractional));
    let cpu_base = tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&neg_base));
    let cpu_exp = tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&fractional));
    let got = run_gpu(
        &*backend,
        OpId::Pow,
        &[&gpu_base, &gpu_exp],
        &[1],
        DType::F32,
        &params,
    );
    let want = run_cpu(OpId::Pow, &[&cpu_base, &cpu_exp], &[1], DType::F32, &params);
    let got_value = f32_values(&got)[0];
    let want_value = f32_values(&want)[0];
    assert!(
        got_value.is_nan() && want_value.is_nan(),
        "(-4)^0.5 must be NaN on both backends, got {got_value} vs {want_value}"
    );

    drop(backend);
    assert_validation_unchanged(counts_before);
}

#[test]
fn unary_ops_and_activations_match_cpu_and_the_gelu_fixture() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();
    let count = 100_003usize;
    let dims = [count as u64];

    for (op, values, atol, rtol) in [
        (OpId::Neg, spread(count, 7, 5.0), 0.0, 0.0),
        (OpId::Relu, spread(count, 11, 5.0), 0.0, 0.0),
        // `sqrt` and `div` are single driver instructions but not correctly
        // rounded on this GPU: the measured difference is one f32 ulp, well
        // inside the bound.
        (OpId::Sqrt, positive(count, 13), 1e-6, 1e-6),
        (OpId::Tanh, spread(count, 17, 6.0), 1e-6, 1e-6),
        (OpId::Sin, spread(count, 19, 12.0), 1e-6, 1e-6),
        (OpId::Cos, spread(count, 23, 12.0), 1e-6, 1e-6),
        // A&S 7.1.26 evaluated in f32: measured max_abs 5.4e-7, so the
        // bound is 1e-6. The model-facing bound is GELU's 2e-6 below.
        (OpId::Erf, spread(count, 29, 5.0), 1e-6, 0.0),
        (OpId::Gelu, spread(count, 31, 6.0), 2e-6, 1e-6),
    ] {
        let gpu_in = tensor_on(device, &dims, DType::F32, &f32_bytes(&values));
        let cpu_in = tensor_on(DeviceId::Cpu, &dims, DType::F32, &f32_bytes(&values));
        let got = run_gpu(&backend, op, &[&gpu_in], &dims, DType::F32, &params);
        let want = run_cpu(op, &[&cpu_in], &dims, DType::F32, &params);
        if atol == 0.0 && rtol == 0.0 {
            assert_exact(&format!("{op} 100003"), &got, &want);
        } else {
            assert_within(&format!("{op} 100003"), &got, &want, atol, rtol);
        }
    }

    // The committed GELU (erf form) fixture, with its own tolerance table row.
    let suite = Suite::load("ops");
    let input = suite.tensor("gelu.input").f32_values();
    let reference = suite.tensor("gelu.erf.out");
    let gpu_in = tensor_on(device, &[69], DType::F32, &f32_bytes(&input));
    let got = run_gpu(&backend, OpId::Gelu, &[&gpu_in], &[69], DType::F32, &params);
    let values = f32_values(&got);
    let agreement = reference
        .check(Actual::F32(&values), &[69])
        .unwrap_or_else(|error| panic!("gelu.erf fixture: {error}"));
    eprintln!(
        "[fn3vk][fixture] gelu.erf tolerance {} agreement {agreement}",
        reference.tolerance()
    );

    // And the exact `erf` fixture-independent table points.
    let table_in = vec![0.0f32, 0.125, 0.5, 1.0, 2.0, 3.0, -0.5, -1.5, -3.0];
    let table_expected = vec![
        0.0f32,
        0.140_316_2,
        0.520_499_9,
        0.842_700_8,
        0.995_322_3,
        0.999_977_9,
        -0.520_499_9,
        -0.966_105_1,
        -0.999_977_9,
    ];
    let gpu_in = tensor_on(device, &[9], DType::F32, &f32_bytes(&table_in));
    let got = run_gpu(&backend, OpId::Erf, &[&gpu_in], &[9], DType::F32, &params);
    let values = f32_values(&got);
    for (index, (&actual, &expected)) in values.iter().zip(table_expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "erf({}) = {actual}, expected {expected}",
            table_in[index]
        );
    }

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Where / comparison
// ---------------------------------------------------------------------------

#[test]
fn where_and_comparisons_match_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // Where: cond [2, 1, 4] and y [1] both broadcast into [2, 3, 4].
    let cond = [true, false, true, false, false, true, true, false];
    let x = spread(24, 37, 4.0);
    let y = [9.5f32];
    let gpu_cond = tensor_on(device, &[2, 1, 4], DType::Bool, &bool_bytes(&cond));
    let gpu_x = tensor_on(device, &[2, 3, 4], DType::F32, &f32_bytes(&x));
    let gpu_y = tensor_on(device, &[1], DType::F32, &f32_bytes(&y));
    let cpu_cond = tensor_on(DeviceId::Cpu, &[2, 1, 4], DType::Bool, &bool_bytes(&cond));
    let cpu_x = tensor_on(DeviceId::Cpu, &[2, 3, 4], DType::F32, &f32_bytes(&x));
    let cpu_y = tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&y));
    let got = run_gpu(
        &*backend,
        OpId::Where,
        &[&gpu_cond, &gpu_x, &gpu_y],
        &[2, 3, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Where,
        &[&cpu_cond, &cpu_x, &cpu_y],
        &[2, 3, 4],
        DType::F32,
        &params,
    );
    assert_exact("where f32 broadcast", &got, &want);

    // Where over i64 values moves whole 8-byte elements.
    let xi = vec![1i64, -2, 3, -4];
    let yi = vec![100i64];
    let gpu_cond = tensor_on(
        device,
        &[4],
        DType::Bool,
        &bool_bytes(&[true, false, true, false]),
    );
    let gpu_x = tensor_on(device, &[4], DType::I64, &i64_bytes(&xi));
    let gpu_y = tensor_on(device, &[1], DType::I64, &i64_bytes(&yi));
    let cpu_cond = tensor_on(
        DeviceId::Cpu,
        &[4],
        DType::Bool,
        &bool_bytes(&[true, false, true, false]),
    );
    let cpu_x = tensor_on(DeviceId::Cpu, &[4], DType::I64, &i64_bytes(&xi));
    let cpu_y = tensor_on(DeviceId::Cpu, &[1], DType::I64, &i64_bytes(&yi));
    let got = run_gpu(
        &*backend,
        OpId::Where,
        &[&gpu_cond, &gpu_x, &gpu_y],
        &[4],
        DType::I64,
        &params,
    );
    let want = run_cpu(
        OpId::Where,
        &[&cpu_cond, &cpu_x, &cpu_y],
        &[4],
        DType::I64,
        &params,
    );
    assert_exact("where i64", &got, &want);

    // Compare: f32 broadcast, i64 shape-style compare, i32, and Bool equality.
    let lhs = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = vec![1.0f32, 3.0, 2.0];
    let gpu_lhs = tensor_on(device, &[2, 3], DType::F32, &f32_bytes(&lhs));
    let gpu_rhs = tensor_on(device, &[3], DType::F32, &f32_bytes(&rhs));
    let cpu_lhs = tensor_on(DeviceId::Cpu, &[2, 3], DType::F32, &f32_bytes(&lhs));
    let cpu_rhs = tensor_on(DeviceId::Cpu, &[3], DType::F32, &f32_bytes(&rhs));
    for op in [OpId::Equal, OpId::Less, OpId::Greater] {
        let got = run_gpu(
            &backend,
            op,
            &[&gpu_lhs, &gpu_rhs],
            &[2, 3],
            DType::Bool,
            &params,
        );
        let want = run_cpu(op, &[&cpu_lhs, &cpu_rhs], &[2, 3], DType::Bool, &params);
        assert_exact(&format!("{op} f32 broadcast"), &got, &want);
    }

    let big = vec![0i64, -1, 1 << 40, -(1 << 40), i64::MAX, i64::MIN];
    let reference = vec![0i64, 1, 1 << 40, -(1 << 41), i64::MAX - 1, i64::MIN + 1];
    let gpu_a = tensor_on(device, &[6], DType::I64, &i64_bytes(&big));
    let gpu_b = tensor_on(device, &[6], DType::I64, &i64_bytes(&reference));
    let cpu_a = tensor_on(DeviceId::Cpu, &[6], DType::I64, &i64_bytes(&big));
    let cpu_b = tensor_on(DeviceId::Cpu, &[6], DType::I64, &i64_bytes(&reference));
    for op in [OpId::Equal, OpId::Less, OpId::Greater] {
        let got = run_gpu(&backend, op, &[&gpu_a, &gpu_b], &[6], DType::Bool, &params);
        let want = run_cpu(op, &[&cpu_a, &cpu_b], &[6], DType::Bool, &params);
        assert_exact(&format!("{op} i64 extremes"), &got, &want);
    }

    let ai = vec![1i32, -2, 3, 4];
    let bi = vec![1i32, 2, -3, 5];
    let gpu_a = tensor_on(device, &[4], DType::I32, &i32_bytes(&ai));
    let gpu_b = tensor_on(device, &[4], DType::I32, &i32_bytes(&bi));
    let cpu_a = tensor_on(DeviceId::Cpu, &[4], DType::I32, &i32_bytes(&ai));
    let cpu_b = tensor_on(DeviceId::Cpu, &[4], DType::I32, &i32_bytes(&bi));
    for op in [OpId::Equal, OpId::Less, OpId::Greater] {
        let got = run_gpu(&backend, op, &[&gpu_a, &gpu_b], &[4], DType::Bool, &params);
        let want = run_cpu(op, &[&cpu_a, &cpu_b], &[4], DType::Bool, &params);
        assert_exact(&format!("{op} i32"), &got, &want);
    }

    // A prime number of Bool verdicts exercises the 4-byte packing tail.
    let count = 100_003usize;
    let values = pattern(count, 41);
    let gpu_a = tensor_on(device, &[count as u64], DType::F32, &f32_bytes(&values));
    let cpu_a = tensor_on(
        DeviceId::Cpu,
        &[count as u64],
        DType::F32,
        &f32_bytes(&values),
    );
    let gpu_b = tensor_on(
        device,
        &[count as u64],
        DType::F32,
        &f32_bytes(&pattern(count, 43)),
    );
    let cpu_b = tensor_on(
        DeviceId::Cpu,
        &[count as u64],
        DType::F32,
        &f32_bytes(&pattern(count, 43)),
    );
    let got = run_gpu(
        &*backend,
        OpId::Less,
        &[&gpu_a, &gpu_b],
        &[count as u64],
        DType::Bool,
        &params,
    );
    let want = run_cpu(
        OpId::Less,
        &[&cpu_a, &cpu_b],
        &[count as u64],
        DType::Bool,
        &params,
    );
    assert_exact("less prime tail", &got, &want);
    assert_eq!(window(&got).len(), count, "the Bool output is not padded");

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

#[test]
fn every_cast_pair_matches_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // Sources with fractional, negative, extreme and NaN values.
    let floats = vec![
        0.0f32,
        -0.0,
        2.9,
        -2.9,
        0.5,
        -1.0e30,
        1.0e30,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1_048_576.0,
    ];
    let integers = vec![
        0i64,
        1,
        -1,
        42,
        -42,
        1 << 40,
        -(1 << 40),
        i64::MAX,
        i64::MIN,
    ];
    let small = vec![0i32, 1, -1, 2147483647, -2147483648, 7];
    let flags = vec![true, false, true, true, false, false];

    let sources: [(DType, Vec<u8>, usize); 4] = [
        (DType::F32, f32_bytes(&floats), floats.len()),
        (DType::I64, i64_bytes(&integers), integers.len()),
        (DType::I32, i32_bytes(&small), small.len()),
        (DType::Bool, bool_bytes(&flags), flags.len()),
    ];
    for (source, bytes, count) in &sources {
        for target in [DType::F32, DType::I64, DType::I32, DType::Bool] {
            let dims = [*count as u64];
            let gpu_in = tensor_on(device, &dims, *source, bytes);
            let cpu_in = tensor_on(DeviceId::Cpu, &dims, *source, bytes);
            let got = run_gpu(&backend, OpId::Cast, &[&gpu_in], &dims, target, &params);
            let want = run_cpu(OpId::Cast, &[&cpu_in], &dims, target, &params);
            assert_exact(&format!("cast {source} -> {target}"), &got, &want);
        }
    }

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Embedding / Gather
// ---------------------------------------------------------------------------

#[test]
fn embedding_lookup_matches_the_fixture_exactly() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    let suite = Suite::load("ops");
    let table = suite.tensor("embedding.table").f32_values();
    let indices = suite.tensor("embedding.indices").i64_values();
    let reference = suite.tensor("embedding.out");

    let gpu_table = tensor_on(device, &[17, 7], DType::F32, &f32_bytes(&table));
    let gpu_indices = tensor_on(device, &[2, 5], DType::I64, &i64_bytes(&indices));
    let cpu_table = tensor_on(DeviceId::Cpu, &[17, 7], DType::F32, &f32_bytes(&table));
    let cpu_indices = tensor_on(DeviceId::Cpu, &[2, 5], DType::I64, &i64_bytes(&indices));

    let got = run_gpu(
        &*backend,
        OpId::Gather,
        &[&gpu_table, &gpu_indices],
        &[2, 5, 7],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Gather,
        &[&cpu_table, &cpu_indices],
        &[2, 5, 7],
        DType::F32,
        &params,
    );
    assert_exact("embedding gather vs cpu", &got, &want);

    // The fixture tolerance is atol = rtol = 0: an embedding lookup is exact.
    let values = f32_values(&got);
    let agreement = reference
        .check(Actual::F32(&values), &[2, 5, 7])
        .unwrap_or_else(|error| panic!("embedding fixture: {error}"));
    eprintln!(
        "[fn3vk][fixture] embedding tolerance {} agreement {agreement}",
        reference.tolerance()
    );
    assert_eq!(reference.tolerance().atol, 0.0);
    assert_eq!(reference.tolerance().rtol, 0.0);

    drop(backend);
    assert_validation_unchanged(counts_before);
}

#[test]
fn gather_covers_other_axes_index_dtypes_and_bad_indices() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);

    // axis 1 of [2, 3, 4] with five i64 indices -> [2, 5, 4].
    let table = pattern(24, 47);
    let indices = vec![2i64, -1, 0, 1, -3];
    let params = OpParams::default().with_axes([1]);
    let gpu_table = tensor_on(device, &[2, 3, 4], DType::F32, &f32_bytes(&table));
    let gpu_indices = tensor_on(device, &[5], DType::I64, &i64_bytes(&indices));
    let cpu_table = tensor_on(DeviceId::Cpu, &[2, 3, 4], DType::F32, &f32_bytes(&table));
    let cpu_indices = tensor_on(DeviceId::Cpu, &[5], DType::I64, &i64_bytes(&indices));
    let got = run_gpu(
        &*backend,
        OpId::Gather,
        &[&gpu_table, &gpu_indices],
        &[2, 5, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Gather,
        &[&cpu_table, &cpu_indices],
        &[2, 5, 4],
        DType::F32,
        &params,
    );
    assert_exact("gather axis 1", &got, &want);

    // A rank-1 table of i64 (the `Shape` -> `Gather` pattern) with i32 indices.
    let shape_vector = vec![128i64, 2, 64, 50257];
    let refs = vec![0i32, 3, 2];
    let params = OpParams::default();
    let gpu_table = tensor_on(device, &[4], DType::I64, &i64_bytes(&shape_vector));
    let gpu_indices = tensor_on(device, &[3], DType::I32, &i32_bytes(&refs));
    let cpu_table = tensor_on(DeviceId::Cpu, &[4], DType::I64, &i64_bytes(&shape_vector));
    let cpu_indices = tensor_on(DeviceId::Cpu, &[3], DType::I32, &i32_bytes(&refs));
    let got = run_gpu(
        &*backend,
        OpId::Gather,
        &[&gpu_table, &gpu_indices],
        &[3],
        DType::I64,
        &params,
    );
    let want = run_cpu(
        OpId::Gather,
        &[&cpu_table, &cpu_indices],
        &[3],
        DType::I64,
        &params,
    );
    assert_exact("gather i64 table with i32 indices", &got, &want);
    assert_eq!(i64_values(&got), vec![128, 50257, 64]);

    // An out-of-range index is rejected with the CPU reference's error, before
    // a dispatch, and appends no record.
    let bad = vec![17i64];
    let gpu_table = tensor_on(device, &[17, 7], DType::F32, &f32_bytes(&pattern(119, 53)));
    let gpu_indices = tensor_on(device, &[1], DType::I64, &i64_bytes(&bad));
    let mut out = zeros_on(device, &[1, 7], DType::F32);
    backend.residency().clear();
    let error = execute(
        &*backend,
        OpId::Gather,
        &[&gpu_table, &gpu_indices],
        &mut [&mut out],
        &params,
    )
    .expect_err("an index equal to the extent is out of bounds");
    eprintln!("[fn3vk][failure] gather-out-of-bounds: {error}");
    assert!(matches!(error, Fn3Error::OutOfBounds { .. }), "{error:?}");
    assert!(error.to_string().contains("gather index"), "{error}");
    assert!(
        backend.residency().is_empty(),
        "a rejected op must not record"
    );

    // A Bool table is one byte per element; the shader packs four gathered
    // bytes per invocation. A prime element count exercises the tail.
    let flags: Vec<bool> = (0..7).map(|index| index % 3 == 0).collect();
    let gpu_table = tensor_on(device, &[7], DType::Bool, &bool_bytes(&flags));
    let gpu_indices = tensor_on(device, &[5], DType::I64, &i64_bytes(&[6, 0, 2, -1, 4]));
    let cpu_table = tensor_on(DeviceId::Cpu, &[7], DType::Bool, &bool_bytes(&flags));
    let cpu_indices = tensor_on(
        DeviceId::Cpu,
        &[5],
        DType::I64,
        &i64_bytes(&[6, 0, 2, -1, 4]),
    );
    let got = run_gpu(
        &*backend,
        OpId::Gather,
        &[&gpu_table, &gpu_indices],
        &[5],
        DType::Bool,
        &params,
    );
    let want = run_cpu(
        OpId::Gather,
        &[&cpu_table, &cpu_indices],
        &[5],
        DType::Bool,
        &params,
    );
    assert_exact("gather Bool table", &got, &want);

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Layout: transpose / reshape / squeeze / unsqueeze / flatten
// ---------------------------------------------------------------------------

#[test]
fn transpose_and_shape_views_match_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // 3-D permutation [2, 0, 1].
    let values = pattern(24, 59);
    let gpu_in = tensor_on(device, &[2, 3, 4], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(DeviceId::Cpu, &[2, 3, 4], DType::F32, &f32_bytes(&values));
    let permuted = OpParams::default().with_axes([2, 0, 1]);
    let got = run_gpu(
        &*backend,
        OpId::Transpose,
        &[&gpu_in],
        &[4, 2, 3],
        DType::F32,
        &permuted,
    );
    let want = run_cpu(
        OpId::Transpose,
        &[&cpu_in],
        &[4, 2, 3],
        DType::F32,
        &permuted,
    );
    assert_exact("transpose [2,0,1]", &got, &want);

    // The ONNX default reverses every axis.
    let got = run_gpu(
        &*backend,
        OpId::Transpose,
        &[&gpu_in],
        &[4, 3, 2],
        DType::F32,
        &params,
    );
    let want = run_cpu(OpId::Transpose, &[&cpu_in], &[4, 3, 2], DType::F32, &params);
    assert_exact("transpose default (reverse)", &got, &want);

    // 4-D and an i64 tensor (the KV-cache transpose shape, and shape vectors).
    let values = i64_bytes(&(0..2 * 2 * 3 * 4).collect::<Vec<i64>>());
    let gpu_in = tensor_on(device, &[2, 2, 3, 4], DType::I64, &values);
    let cpu_in = tensor_on(DeviceId::Cpu, &[2, 2, 3, 4], DType::I64, &values);
    let permuted = OpParams::default().with_axes([0, 1, 3, 2]);
    let got = run_gpu(
        &*backend,
        OpId::Transpose,
        &[&gpu_in],
        &[2, 2, 4, 3],
        DType::I64,
        &permuted,
    );
    let want = run_cpu(
        OpId::Transpose,
        &[&cpu_in],
        &[2, 2, 4, 3],
        DType::I64,
        &permuted,
    );
    assert_exact("transpose 4-D i64", &got, &want);

    // Reshape with -1 inference and 0 copy.
    let values = pattern(24, 61);
    let gpu_in = tensor_on(device, &[2, 3, 4], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(DeviceId::Cpu, &[2, 3, 4], DType::F32, &f32_bytes(&values));
    let target = i64_bytes(&[-1, 4]);
    let gpu_target = tensor_on(device, &[2], DType::I64, &target);
    let cpu_target = tensor_on(DeviceId::Cpu, &[2], DType::I64, &target);
    let got = run_gpu(
        &*backend,
        OpId::Reshape,
        &[&gpu_in, &gpu_target],
        &[6, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Reshape,
        &[&cpu_in, &cpu_target],
        &[6, 4],
        DType::F32,
        &params,
    );
    assert_exact("reshape [-1, 4]", &got, &want);

    let target = i64_bytes(&[0, 3, -1]);
    let gpu_target = tensor_on(device, &[3], DType::I64, &target);
    let cpu_target = tensor_on(DeviceId::Cpu, &[3], DType::I64, &target);
    let got = run_gpu(
        &*backend,
        OpId::Reshape,
        &[&gpu_in, &gpu_target],
        &[2, 3, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Reshape,
        &[&cpu_in, &cpu_target],
        &[2, 3, 4],
        DType::F32,
        &params,
    );
    assert_exact("reshape [0, 3, -1]", &got, &want);

    // Flatten.
    let got = run_gpu(
        &*backend,
        OpId::Flatten,
        &[&gpu_in],
        &[6, 4],
        DType::F32,
        &OpParams::default().with_axes([2]),
    );
    let want = run_cpu(
        OpId::Flatten,
        &[&cpu_in],
        &[6, 4],
        DType::F32,
        &OpParams::default().with_axes([2]),
    );
    assert_exact("flatten axis 2", &got, &want);

    // Squeeze every extent-1 axis.
    let values = pattern(12, 67);
    let gpu_in = tensor_on(device, &[1, 3, 1, 4], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(
        DeviceId::Cpu,
        &[1, 3, 1, 4],
        DType::F32,
        &f32_bytes(&values),
    );
    let got = run_gpu(
        &*backend,
        OpId::Squeeze,
        &[&gpu_in],
        &[3, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(OpId::Squeeze, &[&cpu_in], &[3, 4], DType::F32, &params);
    assert_exact("squeeze all unit axes", &got, &want);

    // Squeeze with an axes input.
    let axes = i64_bytes(&[2]);
    let gpu_axes = tensor_on(device, &[1], DType::I64, &axes);
    let cpu_axes = tensor_on(DeviceId::Cpu, &[1], DType::I64, &axes);
    let got = run_gpu(
        &*backend,
        OpId::Squeeze,
        &[&gpu_in, &gpu_axes],
        &[1, 3, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Squeeze,
        &[&cpu_in, &cpu_axes],
        &[1, 3, 4],
        DType::F32,
        &params,
    );
    assert_exact("squeeze axis input", &got, &want);

    // Unsqueeze at two positions.
    let values = pattern(12, 71);
    let gpu_in = tensor_on(device, &[3, 4], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(DeviceId::Cpu, &[3, 4], DType::F32, &f32_bytes(&values));
    let axes = OpParams::default().with_axes([0, 3]);
    let got = run_gpu(
        &*backend,
        OpId::Unsqueeze,
        &[&gpu_in],
        &[1, 3, 4, 1],
        DType::F32,
        &axes,
    );
    let want = run_cpu(
        OpId::Unsqueeze,
        &[&cpu_in],
        &[1, 3, 4, 1],
        DType::F32,
        &axes,
    );
    assert_exact("unsqueeze [0,3]", &got, &want);

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Slice / Split / Concat
// ---------------------------------------------------------------------------

#[test]
fn slice_split_and_concat_match_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // Slice axis 1: start 1, open end, step 2 -> [4, 3].
    let values = pattern(24, 73);
    let gpu_in = tensor_on(device, &[4, 6], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(DeviceId::Cpu, &[4, 6], DType::F32, &f32_bytes(&values));
    let starts = i64_bytes(&[1]);
    let ends = i64_bytes(&[i64::MAX]);
    let axes = i64_bytes(&[1]);
    let steps = i64_bytes(&[2]);
    let gpu_scalars = [
        tensor_on(device, &[1], DType::I64, &starts),
        tensor_on(device, &[1], DType::I64, &ends),
        tensor_on(device, &[1], DType::I64, &axes),
        tensor_on(device, &[1], DType::I64, &steps),
    ];
    let cpu_scalars = [
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &starts),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &ends),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &axes),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &steps),
    ];
    let got = run_gpu(
        &*backend,
        OpId::Slice,
        &[
            &gpu_in,
            &gpu_scalars[0],
            &gpu_scalars[1],
            &gpu_scalars[2],
            &gpu_scalars[3],
        ],
        &[4, 3],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Slice,
        &[
            &cpu_in,
            &cpu_scalars[0],
            &cpu_scalars[1],
            &cpu_scalars[2],
            &cpu_scalars[3],
        ],
        &[4, 3],
        DType::F32,
        &params,
    );
    assert_exact("slice axis 1 step 2", &got, &want);

    // Reverse slice on axis 0 with an unbounded end.
    let starts = i64_bytes(&[i64::MAX]);
    let ends = i64_bytes(&[i64::MIN]);
    let axes = i64_bytes(&[0]);
    let steps = i64_bytes(&[-1]);
    let gpu_scalars = [
        tensor_on(device, &[1], DType::I64, &starts),
        tensor_on(device, &[1], DType::I64, &ends),
        tensor_on(device, &[1], DType::I64, &axes),
        tensor_on(device, &[1], DType::I64, &steps),
    ];
    let cpu_scalars = [
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &starts),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &ends),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &axes),
        tensor_on(DeviceId::Cpu, &[1], DType::I64, &steps),
    ];
    let got = run_gpu(
        &*backend,
        OpId::Slice,
        &[
            &gpu_in,
            &gpu_scalars[0],
            &gpu_scalars[1],
            &gpu_scalars[2],
            &gpu_scalars[3],
        ],
        &[4, 6],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Slice,
        &[
            &cpu_in,
            &cpu_scalars[0],
            &cpu_scalars[1],
            &cpu_scalars[2],
            &cpu_scalars[3],
        ],
        &[4, 6],
        DType::F32,
        &params,
    );
    assert_exact("slice reverse axis 0", &got, &want);

    // Split [2, 9] on axis 1 into three explicit sizes.
    let values = pattern(18, 79);
    let gpu_in = tensor_on(device, &[2, 9], DType::F32, &f32_bytes(&values));
    let cpu_in = tensor_on(DeviceId::Cpu, &[2, 9], DType::F32, &f32_bytes(&values));
    let sizes = i64_bytes(&[3, 5, 1]);
    let gpu_sizes = tensor_on(device, &[3], DType::I64, &sizes);
    let cpu_sizes = tensor_on(DeviceId::Cpu, &[3], DType::I64, &sizes);
    let pieces = [[2u64, 3], [2, 5], [2, 1]];
    let params = OpParams::default().with_axes([1]);
    let mut gpu_outs: Vec<Tensor> = pieces
        .iter()
        .map(|dims| zeros_on(device, dims, DType::F32))
        .collect();
    let mut cpu_outs: Vec<Tensor> = pieces
        .iter()
        .map(|dims| zeros_on(DeviceId::Cpu, dims, DType::F32))
        .collect();
    backend.residency().clear();
    {
        let mut out_views: Vec<TensorMut<'_>> = gpu_outs
            .iter_mut()
            .map(|tensor| tensor.as_mut_view().expect("split out view"))
            .collect();
        execute_views(
            &*backend,
            OpId::Split,
            &[&gpu_in, &gpu_sizes],
            &mut out_views,
            &params,
        )
        .expect("split on the GPU");
    }
    let records = backend.residency().records();
    assert_eq!(records.len(), 1, "split is one op with one record");
    check_record(OpId::Split, &records[0]);
    {
        let mut out_views: Vec<TensorMut<'_>> = cpu_outs
            .iter_mut()
            .map(|tensor| tensor.as_mut_view().expect("split out view"))
            .collect();
        execute_views(
            &CpuBackend::new(),
            OpId::Split,
            &[&cpu_in, &cpu_sizes],
            &mut out_views,
            &params,
        )
        .expect("split on the CPU");
    }
    for (position, (got, want)) in gpu_outs.iter().zip(cpu_outs.iter()).enumerate() {
        assert_exact(&format!("split piece {position}"), got, want);
    }

    // Concat three pieces on axis 0 and on the last axis.
    let a = pattern(6, 83);
    let b = pattern(12, 89);
    let c = pattern(6, 97);
    let gpu_parts = [
        tensor_on(device, &[1, 2, 3], DType::F32, &f32_bytes(&a)),
        tensor_on(device, &[2, 2, 3], DType::F32, &f32_bytes(&b)),
        tensor_on(device, &[1, 2, 3], DType::F32, &f32_bytes(&c)),
    ];
    let cpu_parts = [
        tensor_on(DeviceId::Cpu, &[1, 2, 3], DType::F32, &f32_bytes(&a)),
        tensor_on(DeviceId::Cpu, &[2, 2, 3], DType::F32, &f32_bytes(&b)),
        tensor_on(DeviceId::Cpu, &[1, 2, 3], DType::F32, &f32_bytes(&c)),
    ];
    let concat_axis0 = OpParams::default();
    let got = run_gpu(
        &*backend,
        OpId::Concat,
        &[&gpu_parts[0], &gpu_parts[1], &gpu_parts[2]],
        &[4, 2, 3],
        DType::F32,
        &concat_axis0,
    );
    let want = run_cpu(
        OpId::Concat,
        &[&cpu_parts[0], &cpu_parts[1], &cpu_parts[2]],
        &[4, 2, 3],
        DType::F32,
        &concat_axis0,
    );
    assert_exact("concat axis 0", &got, &want);

    let parts = [pattern(4, 101), pattern(4, 103), pattern(4, 107)];
    let gpu_parts = [
        tensor_on(device, &[2, 2], DType::F32, &f32_bytes(&parts[0])),
        tensor_on(device, &[2, 1], DType::F32, &f32_bytes(&parts[1])),
        tensor_on(device, &[2, 1], DType::F32, &f32_bytes(&parts[2])),
    ];
    let cpu_parts = [
        tensor_on(DeviceId::Cpu, &[2, 2], DType::F32, &f32_bytes(&parts[0])),
        tensor_on(DeviceId::Cpu, &[2, 1], DType::F32, &f32_bytes(&parts[1])),
        tensor_on(DeviceId::Cpu, &[2, 1], DType::F32, &f32_bytes(&parts[2])),
    ];
    let params = OpParams::default().with_axes([1]);
    let got = run_gpu(
        &*backend,
        OpId::Concat,
        &[&gpu_parts[0], &gpu_parts[1], &gpu_parts[2]],
        &[2, 4],
        DType::F32,
        &params,
    );
    let want = run_cpu(
        OpId::Concat,
        &[&cpu_parts[0], &cpu_parts[1], &cpu_parts[2]],
        &[2, 4],
        DType::F32,
        &params,
    );
    assert_exact("concat axis 1", &got, &want);

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Expand / Shape / ConstantOfShape / Range
// ---------------------------------------------------------------------------

#[test]
fn expand_shape_fill_and_range_match_cpu() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // Expand [1, 3] and [3] to [2, 3].
    for dims in [vec![1u64, 3], vec![3u64]] {
        let values = vec![1.5f32, -2.5, 3.25];
        let gpu_in = tensor_on(device, &dims, DType::F32, &f32_bytes(&values));
        let cpu_in = tensor_on(DeviceId::Cpu, &dims, DType::F32, &f32_bytes(&values));
        let shape_bytes = i64_bytes(&[2, 3]);
        let gpu_shape = tensor_on(device, &[2], DType::I64, &shape_bytes);
        let cpu_shape = tensor_on(DeviceId::Cpu, &[2], DType::I64, &shape_bytes);
        let got = run_gpu(
            &backend,
            OpId::Expand,
            &[&gpu_in, &gpu_shape],
            &[2, 3],
            DType::F32,
            &params,
        );
        let want = run_cpu(
            OpId::Expand,
            &[&cpu_in, &cpu_shape],
            &[2, 3],
            DType::F32,
            &params,
        );
        assert_exact(&format!("expand {dims:?}"), &got, &want);
    }

    // Shape of [2, 3, 4] -> i64 [3] = [2, 3, 4].
    let gpu_in = tensor_on(
        device,
        &[2, 3, 4],
        DType::F32,
        &f32_bytes(&pattern(24, 109)),
    );
    let cpu_in = tensor_on(
        DeviceId::Cpu,
        &[2, 3, 4],
        DType::F32,
        &f32_bytes(&pattern(24, 109)),
    );
    let got = run_gpu(&backend, OpId::Shape, &[&gpu_in], &[3], DType::I64, &params);
    let want = run_cpu(OpId::Shape, &[&cpu_in], &[3], DType::I64, &params);
    assert_exact("shape of [2,3,4]", &got, &want);
    assert_eq!(i64_values(&got), vec![2, 3, 4]);

    // ConstantOfShape in f32 (the additive -FLT_MAX mask), i64 and Bool.
    for (dtype, value) in [
        (DType::F32, f32_bytes(&[f32::MIN])),
        (DType::I64, i64_bytes(&[-7])),
        (DType::Bool, bool_bytes(&[true])),
    ] {
        let shape_bytes = i64_bytes(&[2, 3]);
        let gpu_shape = tensor_on(device, &[2], DType::I64, &shape_bytes);
        let cpu_shape = tensor_on(DeviceId::Cpu, &[2], DType::I64, &shape_bytes);
        let gpu_value = tensor_on(device, &[1], dtype, &value);
        let cpu_value = tensor_on(DeviceId::Cpu, &[1], dtype, &value);
        let got = run_gpu(
            &backend,
            OpId::ConstantOfShape,
            &[&gpu_shape, &gpu_value],
            &[2, 3],
            dtype,
            &params,
        );
        let want = run_cpu(
            OpId::ConstantOfShape,
            &[&cpu_shape, &cpu_value],
            &[2, 3],
            dtype,
            &params,
        );
        assert_exact(&format!("constant_of_shape {dtype}"), &got, &want);
    }

    // Range f32 and i64, including a negative step.
    for (start, limit, delta, dtype, dims) in [
        (0.0f32, 10.0, 2.5, DType::F32, [4u64]),
        (1.0f32, -1.0, -0.5, DType::F32, [4u64]),
    ] {
        let gpu_inputs = [
            tensor_on(device, &[1], DType::F32, &f32_bytes(&[start])),
            tensor_on(device, &[1], DType::F32, &f32_bytes(&[limit])),
            tensor_on(device, &[1], DType::F32, &f32_bytes(&[delta])),
        ];
        let cpu_inputs = [
            tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&[start])),
            tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&[limit])),
            tensor_on(DeviceId::Cpu, &[1], DType::F32, &f32_bytes(&[delta])),
        ];
        let got = run_gpu(
            &backend,
            OpId::Range,
            &[&gpu_inputs[0], &gpu_inputs[1], &gpu_inputs[2]],
            &dims,
            dtype,
            &params,
        );
        let want = run_cpu(
            OpId::Range,
            &[&cpu_inputs[0], &cpu_inputs[1], &cpu_inputs[2]],
            &dims,
            dtype,
            &params,
        );
        assert_exact(
            &format!("range f32 {start}..{limit} step {delta}"),
            &got,
            &want,
        );
    }

    for (start, limit, delta, dims) in [
        (5i64, 0, -2, [3u64]),
        (0i64, 8, 3, [3u64]),
        (-4i64, 4, 1, [8u64]),
    ] {
        let gpu_inputs = [
            tensor_on(device, &[1], DType::I64, &i64_bytes(&[start])),
            tensor_on(device, &[1], DType::I64, &i64_bytes(&[limit])),
            tensor_on(device, &[1], DType::I64, &i64_bytes(&[delta])),
        ];
        let cpu_inputs = [
            tensor_on(DeviceId::Cpu, &[1], DType::I64, &i64_bytes(&[start])),
            tensor_on(DeviceId::Cpu, &[1], DType::I64, &i64_bytes(&[limit])),
            tensor_on(DeviceId::Cpu, &[1], DType::I64, &i64_bytes(&[delta])),
        ];
        let got = run_gpu(
            &backend,
            OpId::Range,
            &[&gpu_inputs[0], &gpu_inputs[1], &gpu_inputs[2]],
            &dims,
            DType::I64,
            &params,
        );
        let want = run_cpu(
            OpId::Range,
            &[&cpu_inputs[0], &cpu_inputs[1], &cpu_inputs[2]],
            &dims,
            DType::I64,
            &params,
        );
        assert_exact(
            &format!("range i64 {start}..{limit} step {delta}"),
            &got,
            &want,
        );
    }

    drop(backend);
    assert_validation_unchanged(counts_before);
}

// ---------------------------------------------------------------------------
// Honest rejection
// ---------------------------------------------------------------------------

#[test]
fn unsupported_layouts_and_dtypes_are_rejected_not_approximated() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let device = gpu_device(&backend);
    let params = OpParams::default();

    // A strided (transposed) view is a layout the backend's capability table
    // says it cannot execute; the kernel must say so, naming #59.
    let values = pattern(6, 113);
    let storage = Storage::from_bytes(&f32_bytes(&values)).expect("storage");
    let meta = TensorMeta::strided(
        Shape::from_dims([3, 2]),
        fastnn::fn3::shape::Strides::from_elements(vec![1, 3]),
        0,
        DType::F32,
        device,
    )
    .expect("strided metadata");
    let strided = Tensor::from_storage(storage, meta).expect("strided tensor");
    let other = tensor_on(device, &[3, 2], DType::F32, &f32_bytes(&pattern(6, 127)));
    let mut out = zeros_on(device, &[3, 2], DType::F32);
    let error = execute(
        &*backend,
        OpId::Add,
        &[&strided, &other],
        &mut [&mut out],
        &params,
    )
    .expect_err("a strided operand is not supported");
    eprintln!("[fn3vk][failure] strided-add: {error}");
    assert!(
        matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Add,
                backend: "vulkan",
                ..
            }
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("#59"), "{error}");

    // An integer `Add` is not a compute dtype: reject, never coerce.
    let left = tensor_on(device, &[3], DType::I64, &i64_bytes(&[1, 2, 3]));
    let right = tensor_on(device, &[3], DType::I64, &i64_bytes(&[4, 5, 6]));
    let mut out = zeros_on(device, &[3], DType::I64);
    let error = execute(
        &*backend,
        OpId::Add,
        &[&left, &right],
        &mut [&mut out],
        &params,
    )
    .expect_err("i64 add is not implemented");
    eprintln!("[fn3vk][failure] i64-add: {error}");
    assert!(
        matches!(error, Fn3Error::UnsupportedDType { .. }),
        "{error:?}"
    );

    // A slice that changes two axes would need two dispatches; it must name
    // #59 rather than move half the tensor.
    let values = pattern(24, 131);
    let input = tensor_on(device, &[4, 6], DType::F32, &f32_bytes(&values));
    let starts = i64_bytes(&[1, 1]);
    let ends = i64_bytes(&[i64::MAX, i64::MAX]);
    let axes = i64_bytes(&[0, 1]);
    let scalars = [
        tensor_on(device, &[2], DType::I64, &starts),
        tensor_on(device, &[2], DType::I64, &ends),
        tensor_on(device, &[2], DType::I64, &axes),
    ];
    let mut out = zeros_on(device, &[3, 5], DType::F32);
    let error = execute(
        &*backend,
        OpId::Slice,
        &[&input, &scalars[0], &scalars[1], &scalars[2]],
        &mut [&mut out],
        &params,
    )
    .expect_err("a two-axis slice is not implemented");
    eprintln!("[fn3vk][failure] two-axis-slice: {error}");
    assert!(
        matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Slice,
                backend: "vulkan",
                ..
            }
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("#59"), "{error}");

    // A Bool transpose needs one-byte writes; name the gap.
    let flags = bool_bytes(&[true, false, true, false]);
    let input = tensor_on(device, &[2, 2], DType::Bool, &flags);
    let mut out = zeros_on(device, &[2, 2], DType::Bool);
    let error = execute(
        &*backend,
        OpId::Transpose,
        &[&input],
        &mut [&mut out],
        &params,
    )
    .expect_err("a Bool transpose is not implemented");
    eprintln!("[fn3vk][failure] bool-transpose: {error}");
    assert!(
        matches!(
            error,
            Fn3Error::UnsupportedOp {
                op: OpId::Transpose,
                backend: "vulkan",
                ..
            }
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains("#59"), "{error}");

    // Every op this family does not own keeps an honest verdict, and a
    // non-Full answer names its owner. This is deliberately an invariant
    // rather than a list of today's gaps: the FN3 ledger forbids support-gap
    // snapshots because another kernel merge (for example #61) would break
    // this test without changing a line of #59's contract.
    for op in OpId::ALL {
        let support = backend.supports(op);
        if !support.is_full() {
            assert!(
                support.reason().unwrap_or_default().contains('#'),
                "{op} is not Full and names no owner: {support}"
            );
        }
    }

    assert!(
        backend.residency().is_empty(),
        "rejections append no record"
    );
    assert_eq!(backend.memory_stats().live_bytes, 0, "no buffer leaked");

    drop(backend);
    assert_validation_unchanged(counts_before);
}

/// Every registered op reports the verdict its kernel actually implements.
///
/// `Full` is claimed only where the kernel covers the whole documented
/// contract; every `Partial` names the missing case and its owner (#59). This
/// is #59's own conformance table, not a snapshot of the whole kernel set, so
/// another family's merge cannot break it.
#[test]
fn support_table_matches_the_registrations() {
    let _guard = gpu_lock();
    let Some(backend) = open_physical() else {
        return;
    };
    let counts_before = validation_counts();
    let mut expected: Vec<(OpId, bool)> = vec![
        (OpId::Add, true),
        (OpId::Sub, true),
        (OpId::Mul, true),
        (OpId::Div, true),
        (OpId::Pow, true),
        (OpId::Neg, true),
        (OpId::Sqrt, true),
        (OpId::Tanh, true),
        (OpId::Relu, true),
        (OpId::Sin, true),
        (OpId::Cos, true),
        (OpId::Erf, true),
        (OpId::Gelu, true),
        (OpId::Equal, true),
        (OpId::Less, true),
        (OpId::Greater, true),
        (OpId::Cast, true),
        (OpId::Shape, true),
        (OpId::ConstantOfShape, true),
        (OpId::Range, true),
        // Narrower than the documented op contract: the reason names the gap.
        (OpId::Where, false),
        (OpId::Gather, true),
        (OpId::Transpose, false),
        (OpId::Reshape, false),
        (OpId::Flatten, false),
        (OpId::Squeeze, false),
        (OpId::Unsqueeze, false),
        (OpId::Slice, false),
        (OpId::Split, false),
        (OpId::Concat, false),
        (OpId::Expand, false),
    ];
    expected.sort_by_key(|(op, _)| op.code());
    for (op, full) in &expected {
        let support = backend.supports(*op);
        assert!(
            support.is_supported(),
            "{op} must be supported, got {support}"
        );
        let kernel = backend.kernels().get(*op).expect("registered kernel");
        assert!(
            !kernel.dtypes.is_empty(),
            "{op}: the kernel declares no dtype"
        );
        if *full {
            assert_eq!(support, Support::Full, "{op}");
        } else {
            assert!(
                matches!(support, Support::Partial { .. }),
                "{op} must be Partial, got {support}"
            );
            assert!(
                support.reason().unwrap_or_default().contains("#59"),
                "{op}: a Partial verdict must name its owner: {support}"
            );
        }
        eprintln!(
            "[fn3vk][support] {:<18} {:<9} kernel={} dtypes={:?}",
            op.name(),
            if *full { "full" } else { "partial" },
            kernel.kernel.name,
            kernel.dtypes
        );
    }
    // The FN3-11 smoke kernel is replaced, not appended to.
    let add = backend.kernels().get(OpId::Add).expect("add");
    assert_eq!(add.kernel.name, "pointwise.binary.f32");
    assert!(add.features.broadcasting);
    assert!(backend.capabilities().features.broadcasting);
    assert!(!backend.capabilities().supports_strided_views);
    drop(backend);
    assert_validation_unchanged(counts_before);
}
