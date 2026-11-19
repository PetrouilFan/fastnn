//! FN3-24 (#70): W8A16 quantized projections on the resident Vulkan path.
//!
//! # What this binary proves
//!
//! | test | claim |
//! |---|---|
//! | `format_parity_on_the_committed_blobs` | the kernel consumes the locked `fn3-quant-format-v1` bytes (including a short final group) and reproduces the reference `sum x*(q-z)` arithmetic |
//! | `projection_shapes_match_the_reference` | prefill (M>1) and batch-1 decode (output-channel split), every pinned projection shape, tails, and N not a multiple of 64 |
//! | `the_selection_rule_falls_back_where_the_format_says` | the admission rule declines K<64, numel<4096, dynamic RHS, shared weights, rank-3 weights, non-f32 and batched activations, and the FP32 kernel serves them |
//! | `resident_quantized_projections_upload_nothing` | with prepared packed weights, every quantized dispatch has **0 upload bytes**, **0 readback bytes**, three device-local weight bindings and no input readback; readback is exactly the logits rows |
//! | `end_to_end_accuracy_against_fp32` | per-projection relative Frobenius, prefill/decode KL, top-1/top-5 margins against the FP32 run, with the #68 thresholds |
//! | `quantized_latency_and_memory` (`FN3_QUANT_PERF=1`) | p50/p95 prefill and decode, device/host peak, traffic and submissions, quantized vs FP32 |
//!
//! # Why a decorator, not a planner change
//!
//! `KernelSet` holds one kernel per [`OpId`], and the alpha has no quantized op
//! id, so `kernels::quant` registers in its own `QuantKernelSet` and the plan
//! cannot select it yet. Mounting it needs a core change that #70 does not own
//! (a quantized projection identity plus the planner rule that emits it); this
//! file supplies the adapter that change will replace — a `Backend` that
//! intercepts `MatMul` when its right-hand side is a prepared packed weight and
//! delegates everything else to the real Vulkan backend. The GPU work is the
//! production kernel: no arithmetic happens here except the independent
//! reference used to grade it.
//!
//! ```bash
//! FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
//! FN3_VK_REQUIRE_PHYSICAL=1 \
//! FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
//!   cargo test --features vulkan --test fn3_vulkan_quant -- --nocapture --test-threads=1
//! ```

#![cfg(feature = "vulkan")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use fastnn::fn3::backend::vulkan::kernels::quant::{
    admit_projection, encode_projection, PackedProjection, ProjectionRequest, QuantKernelSet,
    QuantSections, WeightSource,
};
use fastnn::fn3::backend::vulkan::resident::ResidencyPhase;
use fastnn::fn3::backend::vulkan::{adapters, VulkanBackend, VulkanConfig};
use fastnn::fn3::backend::{Backend, Capabilities, OpParams, Support};
use fastnn::fn3::device::{DeviceId, DeviceRequest};
use fastnn::fn3::model::SessionOptions;
use fastnn::fn3::model::{Planner, SlotRole};
use fastnn::fn3::quant::QuantBlob;
use fastnn::fn3::shape::Shape;
use fastnn::fn3::{DType, Fn3Error, Fn3Result, OpId, Session, Tensor, TensorMut, TensorView};

#[path = "fn3_e2e/harness.rs"]
mod harness;

use harness::*;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// A host tensor tagged with `device` without copying bytes.
///
/// `Tensor::zeros` refuses a Vulkan device ("only the cpu device can own a
/// tensor") because its storage is host memory; the #72 harness has the same
/// workaround (`retag_weights`) and `Storage` is shared, so the identity the
/// persistent table keys on is preserved.
fn host_tensor(shape: Shape, dtype: DType, device: DeviceId) -> Fn3Result<Tensor> {
    let tensor = Tensor::zeros(shape.clone(), dtype, DeviceId::Cpu)?;
    let meta = fastnn::fn3::TensorMeta::contiguous(shape, dtype, device)?;
    Tensor::from_storage(tensor.storage().clone(), meta)
}

fn retag(tensor: &Tensor, device: DeviceId) -> Fn3Result<Tensor> {
    let meta = fastnn::fn3::TensorMeta::contiguous(tensor.shape().clone(), tensor.dtype(), device)?;
    Tensor::from_storage(tensor.storage().clone(), meta)
}

fn require_physical() -> bool {
    std::env::var("FN3_VK_REQUIRE_PHYSICAL").as_deref() == Ok("1")
}

fn physical_adapter() -> Option<u32> {
    let pinned = std::env::var("FN3_VK_DEVICE_UUID")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());
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
        eprintln!("[fn3-24][skip] no physical Vulkan device; the quantized gate needs one");
        return None;
    };
    let backend = VulkanBackend::open_with(
        &DeviceRequest::Vulkan { index: Some(index) },
        VulkanConfig::validated(),
    )
    .unwrap_or_else(|error| panic!("opening adapter {index} failed: {error}"));
    assert!(
        !backend.is_software(),
        "the quantized gate needs a real GPU"
    );
    Some(backend)
}

// ---------------------------------------------------------------------------
// Independent reference arithmetic
// ---------------------------------------------------------------------------

/// The format's normative projection arithmetic (§5), over the packed sections.
///
/// Deliberately written from the document rather than from the shader: per
/// group, one `f32` accumulator over ascending `k` of `x * (code - zero)`, then
/// one multiply by the binary16 scale widened to `f32`.
fn reference_grouped(sections: &QuantSections, activation: &[f32], m: usize) -> Vec<f32> {
    let n = sections.n_rows() as usize;
    let k = sections.k() as usize;
    let groups = sections.groups_per_row() as usize;
    let stride = sections.row_stride_bytes() as usize;
    assert_eq!(activation.len(), m * k, "reference activation size");
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut value = 0.0f32;
            for group in 0..groups {
                let start = group * 64;
                let end = (start + 64).min(k);
                let meta = column * groups + group;
                let scale = f16_at(sections.scales(), meta);
                let zero = i16_at(sections.zeros(), meta) as i32;
                let mut partial = 0.0f32;
                for step in start..end {
                    let code = sections.codes()[column * stride + step] as i8 as i32;
                    partial += activation[row * k + step] * (code - zero) as f32;
                }
                value += partial * scale;
            }
            out[row * n + column] = value;
        }
    }
    out
}

fn f16_at(bytes: &[u8], index: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])).to_f32()
}

fn i16_at(bytes: &[u8], index: usize) -> i16 {
    i16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}

/// The FP32 projection a quantized one is compared against: `x @ W` with the
/// original `[N, K]` weights, one `f32` accumulator over ascending `k`.
fn reference_fp32(weights: &[f32], activation: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(weights.len(), n * k, "reference weight size");
    assert_eq!(activation.len(), m * k, "reference activation size");
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for column in 0..n {
            let mut value = 0.0f32;
            for step in 0..k {
                value += activation[row * k + step] * weights[column * k + step];
            }
            out[row * n + column] = value;
        }
    }
    out
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Relative Frobenius error `||got - want||_F / ||want||_F`.
fn relative_frobenius(got: &[f32], want: &[f32]) -> f64 {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let difference = f64::from(*a) - f64::from(*b);
        numerator += difference * difference;
        denominator += f64::from(*b) * f64::from(*b);
    }
    if denominator == 0.0 {
        return if numerator == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (numerator / denominator).sqrt()
}

fn softmax_f64(values: &[f32]) -> Vec<f64> {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let exponentials: Vec<f64> = values
        .iter()
        .map(|value| (f64::from(*value) - maximum).exp())
        .collect();
    let total: f64 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

/// Kullback-Leibler divergence `KL(p || q)` in nats, both already normalised.
fn kl_divergence(reference: &[f32], candidate: &[f32]) -> f64 {
    let p = softmax_f64(reference);
    let q = softmax_f64(candidate);
    p.iter()
        .zip(&q)
        .map(|(p, q)| if *p > 0.0 { p * (p / q).ln() } else { 0.0 })
        .sum()
}

/// Indices of the `count` largest values, descending.
fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
    indices.truncate(count);
    indices
}

/// The gap between the largest and second-largest logit.
fn top1_margin(values: &[f32]) -> f32 {
    let top = top_indices(values, 2);
    values[top[0]] - values[top[1]]
}

// ---------------------------------------------------------------------------
// The packed model
// ---------------------------------------------------------------------------

/// Storage identity of a view: the same key `PersistentTable` uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WeightKey {
    identity: usize,
    offset: usize,
    len: usize,
}

impl WeightKey {
    fn of_view(view: &TensorView<'_>) -> Option<WeightKey> {
        Some(WeightKey {
            identity: view.storage().identity(),
            offset: view.meta().byte_offset().ok()?,
            len: view.meta().byte_len().ok()?,
        })
    }
    fn of_tensor(tensor: &Tensor) -> Option<WeightKey> {
        Some(WeightKey {
            identity: tensor.storage().identity(),
            offset: tensor.meta().byte_offset().ok()?,
            len: tensor.meta().byte_len().ok()?,
        })
    }
}

/// One packed projection plus the facts the report and the grader need.
#[derive(Debug)]
struct Projection {
    /// Source graph weight name (`transformer.h.0.attn.c_attn.weight`).
    name: String,
    /// Identity of the source f32 tensor, when the weight is a plan weight.
    key: Option<WeightKey>,
    /// Packed device payload, resident from prepare.
    packed: Arc<PackedProjection>,
    /// The original `[N, K]` f32 values, kept only for accuracy capture.
    source: Option<Vec<f32>>,
}

/// Which right-hand sides the harness routes to the quantized kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadRoute {
    /// `lm_head` stays FP32 (the plan materialises a transpose per step).
    Fp32,
    /// `lm_head` is packed and matched by shape (requires folding the
    /// per-step transpose of the tied embedding table; measured, not shipped).
    Packed,
}

/// Every packed projection prepared for one model.
#[derive(Debug)]
struct PackedSet {
    /// The projection weights, in weight-table order.
    projections: Vec<Projection>,
    /// By source storage identity, for the four projections that reach
    /// `MatMul` as a weight slot.
    by_key: BTreeMap<WeightKey, usize>,
    /// The tied embedding table packed as `[vocab, hidden]`, when routed.
    head: Option<Projection>,
    /// Source `[K, N]` shapes of every packed projection, for the report.
    shapes: Vec<(String, u64, u64)>,
}

impl PackedSet {
    /// Pack every admissible projection weight of `model`.
    ///
    /// `capture` keeps the source f32 values so the accuracy test can compute
    /// the FP32 projection the quantized one replaces.
    fn quantize(
        backend: &VulkanBackend,
        model: &fastnn::fn3::PreparedModel,
        head: HeadRoute,
        capture: bool,
    ) -> Fn3Result<PackedSet> {
        let mut projections = Vec::new();
        let mut by_key = BTreeMap::new();
        let mut head_projection = None;
        let mut shapes = Vec::new();
        let weights = model.weights();
        for index in 0..weights.len() {
            let Some(tensor) = weights.get(index) else {
                continue;
            };
            let Some(name) = weights.name(index) else {
                continue;
            };
            if tensor.dtype() != DType::F32 || tensor.shape().rank() != 2 {
                continue;
            }
            let dims = tensor.shape().dims();
            let (k, n) = (dims[0], dims[1]);
            let is_projection = name.ends_with("attn.c_attn.weight")
                || name.ends_with("attn.c_proj.weight")
                || name.ends_with("mlp.c_fc.weight")
                || name.ends_with("mlp.c_proj.weight");
            if !is_projection {
                continue;
            }
            let request =
                ProjectionRequest::constant(Shape::from_dims([1, k]), Shape::from_dims([k, n]));
            if admit_projection(&request).is_err() {
                continue;
            }
            // The graph consumes `[K, N]`; the format packs `[N, K]`.
            let source_f32 = tensor.as_f32_slice()?;
            let mut values = vec![0.0f32; (n * k) as usize];
            for row in 0..n as usize {
                for column in 0..k as usize {
                    values[row * k as usize + column] = source_f32[column * n as usize + row];
                }
            }
            let sections = encode_projection(&values, n as usize, k as usize)?;
            let packed = Arc::new(PackedProjection::prepare(backend, name, sections)?);
            shapes.push((name.to_string(), k, n));
            let key = WeightKey::of_tensor(tensor);
            if let Some(key) = key {
                by_key.insert(key, projections.len());
            }
            projections.push(Projection {
                name: name.to_string(),
                key,
                packed,
                source: capture.then_some(values),
            });
        }
        if head == HeadRoute::Packed {
            let table = weights
                .named("transformer.wte.weight")
                .or_else(|| {
                    (0..weights.len()).find_map(|index| {
                        (weights.name(index) == Some("transformer.wte.weight"))
                            .then(|| weights.get(index))
                            .flatten()
                    })
                })
                .ok_or_else(|| Fn3Error::InvalidShape {
                    context: "quantized lm_head",
                    detail: "the model has no transformer.wte.weight to pack".to_string(),
                })?;
            let dims = table.shape().dims();
            let (n, k) = (dims[0], dims[1]);
            let values = table.as_f32_slice()?.to_vec();
            let sections = encode_projection(&values, n as usize, k as usize)?;
            let packed = Arc::new(PackedProjection::prepare(backend, "lm_head", sections)?);
            shapes.push(("lm_head (packed wte)".to_string(), k, n));
            head_projection = Some(Projection {
                name: "lm_head".to_string(),
                key: None,
                packed,
                source: capture.then_some(values),
            });
        }
        Ok(PackedSet {
            projections,
            by_key,
            head: head_projection,
            shapes,
        })
    }

    /// Total packed payload bytes (codes + scales + zero-points).
    fn payload_bytes(&self) -> u64 {
        self.projections
            .iter()
            .map(|projection| projection.packed.payload_bytes())
            .sum::<u64>()
            + self
                .head
                .as_ref()
                .map(|head| head.packed.payload_bytes())
                .unwrap_or(0)
    }

    /// The smallest packed payload, in bytes.
    fn smallest_payload(&self) -> u64 {
        self.projections
            .iter()
            .map(|projection| projection.packed.payload_bytes())
            .chain(self.head.as_ref().map(|head| head.packed.payload_bytes()))
            .min()
            .unwrap_or(0)
    }

    fn total_device_bytes(&self) -> u64 {
        self.projections
            .iter()
            .map(|projection| projection.packed.device_bytes())
            .sum::<u64>()
            + self
                .head
                .as_ref()
                .map(|head| head.packed.device_bytes())
                .unwrap_or(0)
    }

    /// The projection a `MatMul` right-hand side selects, if any.
    fn resolve(&self, activation: &TensorView<'_>, rhs: &TensorView<'_>) -> Option<&Projection> {
        if let Some(key) = WeightKey::of_view(rhs) {
            if let Some(index) = self.by_key.get(&key) {
                let projection = &self.projections[*index];
                if projection.packed.k() as u64 == rhs.shape().dims()[0]
                    && projection.packed.n_rows() as u64 == rhs.shape().dims()[1]
                {
                    return Some(projection);
                }
            }
        }
        // `lm_head` reaches `MatMul` as a materialised `wte^T` slot, so it has
        // no weight identity to match. The packed head is selected when the RHS
        // is exactly `[hidden, vocab]` of a rank-2 activation.
        let head = self.head.as_ref()?;
        let dims = rhs.shape().dims();
        let activation_dims = activation.shape().dims();
        if dims.len() == 2
            && activation_dims.last().copied() == Some(u64::from(head.packed.k()))
            && u64::from(head.packed.n_rows()) == dims[1]
            && dims[0] == u64::from(head.packed.k())
        {
            return Some(head);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// The adapter that selects the quantized kernel
// ---------------------------------------------------------------------------

/// What one quantized projection did, aggregated for the report.
#[derive(Debug, Clone, Default)]
struct QuantStats {
    calls: u64,
    upload_bytes: u64,
    readback_bytes: u64,
    weight_bytes: u64,
    workgroups: BTreeMap<&'static str, u64>,
}

/// One captured projection comparison.
#[derive(Debug, Clone)]
struct Captured {
    name: String,
    m: u64,
    n: u64,
    k: u64,
    /// Kernel vs the format's reference arithmetic (kernel fidelity).
    reference_max_abs: f32,
    /// Quantized kernel vs the FP32 projection it replaces.
    fp32_relative_frobenius: f64,
    fp32_max_abs: f32,
}

/// `MatMul` goes to the packed W8A16 kernel when a packed weight matches;
/// everything else is the real Vulkan backend.
#[derive(Debug)]
struct QuantBackend<'a> {
    inner: &'a VulkanBackend,
    packed: &'a PackedSet,
    capture: Option<Mutex<Vec<Captured>>>,
    stats: Mutex<QuantStats>,
}

impl<'a> QuantBackend<'a> {
    fn new(inner: &'a VulkanBackend, packed: &'a PackedSet, capture: bool) -> QuantBackend<'a> {
        QuantBackend {
            inner,
            packed,
            capture: capture.then(|| Mutex::new(Vec::new())),
            stats: Mutex::new(QuantStats::default()),
        }
    }

    fn captured(&self) -> Vec<Captured> {
        self.capture
            .as_ref()
            .map(|capture| {
                capture
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone()
            })
            .unwrap_or_default()
    }

    fn stats(&self) -> QuantStats {
        self.stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl Backend for QuantBackend<'_> {
    fn name(&self) -> &'static str {
        // The reports say "vulkan" because the arithmetic really did run on the
        // Vulkan device; the adapter only chooses which kernel to record.
        self.inner.name()
    }

    fn device(&self) -> DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        // `MatMul` stays the FP32 answer: this adapter never widens the
        // backend's op-level verdict, it only adds a kernel to select at
        // execution time.
        self.inner.supports(op)
    }

    fn admits(&self, op: OpId, shapes: &[Shape], dtypes: &[DType], params: &OpParams) -> Support {
        self.inner.admits(op, shapes, dtypes, params)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        if op == OpId::MatMul && inputs.len() == 2 && outputs.len() == 1 {
            if let Some(projection) = self.packed.resolve(&inputs[0], &inputs[1]) {
                let dims = inputs[1].shape().dims();
                let request = ProjectionRequest::constant(
                    inputs[0].shape().clone(),
                    Shape::from_dims(dims.iter().copied()),
                );
                if admit_projection(&request).is_ok() {
                    let launch = fastnn::fn3::backend::vulkan::kernels::quant::launch_projection(
                        self.inner,
                        &projection.packed,
                        &inputs[0],
                        &mut outputs[0],
                    )?;
                    {
                        let mut stats = self
                            .stats
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        stats.calls += 1;
                        stats.upload_bytes += launch.upload_bytes;
                        stats.readback_bytes += launch.readback_bytes;
                        stats.weight_bytes += launch.weight_bytes;
                        *stats.workgroups.entry(launch.kernel).or_default() += 1;
                    }
                    if let Some(capture) = &self.capture {
                        let captured = self.grade(projection, &inputs[0], dims[0], dims[1])?;
                        capture
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(captured);
                    }
                    return Ok(());
                }
            }
        }
        self.inner.execute_op(op, inputs, outputs, params)
    }

    fn supports_resident_storage(&self) -> bool {
        self.inner.supports_resident_storage()
    }

    fn allocate_resident(&self, shape: &Shape, dtype: DType) -> Result<Tensor, Fn3Error> {
        self.inner.allocate_resident(shape, dtype)
    }

    fn prepare_persistent(&self, tensors: &[&Tensor]) -> Result<(), Fn3Error> {
        self.inner.prepare_persistent(tensors)
    }

    fn begin_step(&self, label: &'static str, batch: bool) -> Result<(), Fn3Error> {
        self.inner.begin_step(label, batch)
    }

    fn write_resident(
        &self,
        dst: &fastnn::fn3::storage::Storage,
        byte_offset: u64,
        bytes: &[u8],
    ) -> Result<(), Fn3Error> {
        self.inner.write_resident(dst, byte_offset, bytes)
    }

    fn copy_resident_regions(
        &self,
        dst: &fastnn::fn3::storage::Storage,
        src: &fastnn::fn3::storage::Storage,
        regions: &[(u64, u64, u64)],
    ) -> Result<(), Fn3Error> {
        self.inner.copy_resident_regions(dst, src, regions)
    }

    fn note_readback(&self, bytes: u64) {
        self.inner.note_readback(bytes)
    }

    fn end_step(&self) -> Result<u32, Fn3Error> {
        self.inner.end_step()
    }
}

impl QuantBackend<'_> {
    /// Grade one quantized projection against the format's reference and
    /// against the FP32 projection it replaces.
    ///
    /// The kernel is executed a second time into a *host* output so the result
    /// can be read here; that is a test-only readback and the resident proof
    /// runs with capture off, which is why it never appears in the proof's
    /// counters.
    fn grade(
        &self,
        projection: &Projection,
        activation: &TensorView<'_>,
        k: u64,
        n: u64,
    ) -> Result<Captured, Fn3Error> {
        let dims = activation.shape().dims();
        let m = dims[dims.len() - 2];
        let host_activation = activation.to_host()?;
        let values = host_activation.as_f32_slice()?;
        let shape = Shape::from_dims(
            dims[..dims.len() - 1]
                .iter()
                .copied()
                .chain([n])
                .collect::<Vec<u64>>(),
        );
        let mut probe = host_tensor(shape, DType::F32, self.inner.device_id())?;
        {
            let mut view = probe.as_mut_view()?;
            fastnn::fn3::backend::vulkan::kernels::quant::launch_projection(
                self.inner,
                &projection.packed,
                activation,
                &mut view,
            )?;
        }
        let got = probe.as_f32_slice()?.to_vec();
        let reference = reference_grouped(projection.packed.sections(), values, m as usize);
        let (fp32_relative_frobenius, fp32_max_abs) = match &projection.source {
            Some(source) => {
                let fp32 = reference_fp32(
                    source,
                    values,
                    m as usize,
                    n as usize,
                    projection.packed.k() as usize,
                );
                (relative_frobenius(&got, &fp32), max_abs(&got, &fp32))
            }
            None => (f64::NAN, f32::NAN),
        };
        Ok(Captured {
            name: projection.name.clone(),
            m,
            n,
            k,
            reference_max_abs: max_abs(&got, &reference),
            fp32_relative_frobenius,
            fp32_max_abs,
        })
    }
}

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

/// One row of the coverage table.
#[derive(Debug, Clone)]
struct CoverageRow {
    layer: u32,
    projection: String,
    activation: String,
    weight: String,
    decision: String,
}

/// Classify every `MatMul` in one plan of the pinned graph.
fn coverage(
    import: &fastnn::fn3::import::ModelImport,
    model: &fastnn::fn3::PreparedModel,
    family: fastnn::fn3::backend::vulkan::kernels::quant::QuantFamily,
    live: fastnn::fn3::LiveDims,
) -> Vec<CoverageRow> {
    let backend = fastnn::fn3::CpuBackend::new();
    let planner = Planner::new(import, &backend);
    let envelope = envelope(&planner, FIXTURE_CONTEXT);
    let plan = planner
        .plan(&envelope, live)
        .unwrap_or_else(|error| panic!("plan {live:?}: {error}"));
    let kernels = QuantKernelSet::alpha();
    let weights = model.weights();
    let mut rows = Vec::new();
    for instruction in plan.instructions() {
        if instruction.op != OpId::MatMul {
            continue;
        }
        let lhs = plan.slot(instruction.inputs[0]).expect("lhs slot");
        let rhs = plan.slot(instruction.inputs[1]).expect("rhs slot");
        let (name, request) = match rhs.role() {
            SlotRole::Weight { id } => {
                let name = weights.name(id.index()).unwrap_or("<unnamed>").to_string();
                (
                    name.clone(),
                    ProjectionRequest::constant(lhs.shape().clone(), rhs.shape().clone()),
                )
            }
            SlotRole::Constant { .. } => (
                "<folded constant>".to_string(),
                ProjectionRequest::constant(lhs.shape().clone(), rhs.shape().clone()),
            ),
            SlotRole::GraphInput => (
                "<graph input>".to_string(),
                ProjectionRequest {
                    weight_source: WeightSource::Dynamic,
                    ..ProjectionRequest::constant(lhs.shape().clone(), rhs.shape().clone())
                },
            ),
            SlotRole::InstructionOutput { instruction } => {
                let producer = plan
                    .instructions()
                    .get(*instruction as usize)
                    .map(|producer| producer.op.name())
                    .unwrap_or("?");
                (
                    format!("<{producer} output>"),
                    ProjectionRequest {
                        weight_source: WeightSource::Dynamic,
                        ..ProjectionRequest::constant(lhs.shape().clone(), rhs.shape().clone())
                    },
                )
            }
        };
        let verdict = kernels.admits(family, &request);
        let decision = match &verdict {
            Support::Full => "quantized".to_string(),
            Support::Partial { reason } => {
                let detail = reason
                    .split("concrete operands: ")
                    .nth(1)
                    .unwrap_or(reason)
                    .to_string();
                format!("fallback: {detail}")
            }
            Support::Unsupported { reason } => format!("unsupported: {reason}"),
        };
        let layer = name
            .split(".h.")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|index| index.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        rows.push(CoverageRow {
            layer,
            projection: name,
            activation: lhs.shape().to_string(),
            weight: rhs.shape().to_string(),
            decision,
        });
    }
    rows
}

fn print_coverage(label: &str, rows: &[CoverageRow]) {
    println!("[fn3-24][coverage] {label}: {} matmuls", rows.len());
    for row in rows {
        println!(
            "[fn3-24][coverage] {label} layer={} {} {} x {} -> {}",
            if row.layer == u32::MAX {
                "-".to_string()
            } else {
                row.layer.to_string()
            },
            row.projection,
            row.activation,
            row.weight,
            row.decision
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: kernel parity
// ---------------------------------------------------------------------------

/// Deterministic activation values that exercise both signs and small magnitudes.
fn activations(rows: usize, k: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..rows * k)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            unit * 2.0 + (index % 7) as f32 * 0.01
        })
        .collect()
}

/// Run one packed projection on the device and compare with the reference.
///
/// `tolerance` is a *relative* bound on the Frobenius norm and on the largest
/// element (scaled by the largest reference magnitude). The one difference the
/// kernel is allowed to have from the scalar reference is fusing the multiply
/// and add of a group's inner loop into an FMA, which is at most one rounding
/// per term: measured here at 2-4e-7 relative, i.e. a few `f32` ulps.
fn check_projection(
    backend: &VulkanBackend,
    name: &str,
    sections: QuantSections,
    m: usize,
    tolerance: f64,
) -> f64 {
    let k = sections.k() as usize;
    let n = sections.n_rows() as usize;
    let values = activations(m, k, 0x5eed_0000 + (n * 131 + k) as u64);
    let packed = PackedProjection::prepare(backend, name, sections).expect("prepare packed");
    let activation = retag(
        &Tensor::from_f32(Shape::from_dims([m as u64, k as u64]), &values).expect("activation"),
        backend.device_id(),
    )
    .expect("retag activation");
    let mut out = host_tensor(
        Shape::from_dims([m as u64, n as u64]),
        DType::F32,
        backend.device_id(),
    )
    .expect("output");
    let launch = {
        let activation_view = activation.as_view().expect("activation view");
        let mut out_view = out.as_mut_view().expect("output view");
        fastnn::fn3::backend::vulkan::kernels::quant::launch_projection(
            backend,
            &packed,
            &activation_view,
            &mut out_view,
        )
        .unwrap_or_else(|error| panic!("{name} [{m}, {k}] x [{k}, {n}]: {error}"))
    };
    assert_eq!(launch.weight_bindings, 3);
    assert_eq!(launch.kernel, "quant.matmul.w8a16");
    assert_eq!(launch.upload_bytes, (m * k * 4) as u64);
    assert_eq!(launch.readback_bytes, (m * n * 4) as u64);
    let got = out.as_f32_slice().expect("host output");
    let expected = reference_grouped(packed.sections(), &values, m);
    let error = max_abs(got, &expected);
    let magnitude = expected
        .iter()
        .fold(0.0f32, |worst, value| worst.max(value.abs()));
    let relative = relative_frobenius(got, &expected);
    assert!(
        relative <= tolerance,
        "{name} [{m}, {k}] x [{k}, {n}]: relative_frobenius {relative:e} above {tolerance:e} \
         (max_abs {error:e} against a largest value of {magnitude:e})"
    );
    assert!(
        f64::from(error) <= tolerance * f64::from(magnitude.max(1.0)),
        "{name} [{m}, {k}] x [{k}, {n}]: max_abs {error:e} above {tolerance:e} x {magnitude:e}"
    );
    println!(
        "[fn3-24][kernel] {name} [{m}, {k}] x [{k}, {n}] max_abs {error:e} \
         relative_frobenius {relative:e} workgroups {:?}",
        launch.workgroups
    );
    relative
}

#[test]
fn format_parity_on_the_committed_blobs() {
    let Some(backend) = open_physical() else {
        return;
    };
    // The committed fixtures are the locked format's own bytes, decoded by the
    // real reader. Two of them exercise a short final group (K = 70) and a
    // non-zero zero-point, which is exactly what the shader's tail path does.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/quant/v1");
    for file in [
        "basic.fn3q",
        "tail_k70.fn3q",
        "zero_point.fn3q",
        "two_tensors.fn3q",
    ] {
        let path = root.join(file);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("fixture {}: {error}", path.display()));
        let blob = QuantBlob::parse(&bytes)
            .unwrap_or_else(|error| panic!("fixture {file} failed to parse: {error}"));
        for tensor in blob.tensors() {
            let sections = QuantSections::from_tensor(tensor);
            // The format's own matrices are small; K = 8 is below the selection
            // rule, but the *kernel* must still be exact, so exercise it.
            for m in [1usize, 5] {
                let relative = check_projection(
                    &backend,
                    &format!("{file}:{}", tensor.name()),
                    sections.clone(),
                    m,
                    1e-6,
                );
                assert!(relative < 1e-6, "{file}:{m} relative {relative:e}");
            }
        }
    }
    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
}

#[test]
fn projection_shapes_match_the_reference() {
    let Some(backend) = open_physical() else {
        return;
    };
    // (N, K) pairs: the pinned tiny and trained projections, a decode-shaped
    // vocabulary projection, N and K not multiples of 64, and a K tail.
    let shapes = [
        (384usize, 128usize),
        (128, 128),
        (512, 128),
        (3072, 768),
        (2304, 768),
        (768, 768),
        (4096, 768),
        (100, 70),
        (65, 64),
    ];
    for (n, k) in shapes {
        let values: Vec<f32> = (0..n * k)
            .map(|index| {
                let row = (index / k) as f32;
                let column = (index % k) as f32;
                ((row * 0.37).sin() * 0.5 + (column * 0.11).cos() * 0.25) * 0.02
            })
            .collect();
        let sections = encode_projection(&values, n, k).expect("encode");
        // Decode (one row), prefill (13 rows) and two row tiles (20 rows).
        for m in [1usize, 13, 20] {
            let relative = check_projection(
                &backend,
                &format!("synthetic_n{n}_k{k}"),
                sections.clone(),
                m,
                4e-6,
            );
            assert!(relative < 4e-6, "N={n} K={k} M={m}: relative {relative:e}");
        }
    }
    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
}

// ---------------------------------------------------------------------------
// Tests: selection
// ---------------------------------------------------------------------------

#[test]
fn the_selection_rule_falls_back_where_the_format_says() {
    let family = fastnn::fn3::backend::vulkan::kernels::quant::QuantFamily::GroupedW8A16;
    let kernels = QuantKernelSet::alpha();
    let cases: [(&str, ProjectionRequest); 7] = [
        (
            "K below one group",
            ProjectionRequest::constant(Shape::from_dims([1, 32]), Shape::from_dims([32, 256])),
        ),
        (
            "numel below 4096",
            ProjectionRequest::constant(Shape::from_dims([1, 64]), Shape::from_dims([64, 32])),
        ),
        (
            "dynamic right-hand side",
            ProjectionRequest {
                weight_source: WeightSource::Dynamic,
                ..ProjectionRequest::constant(
                    Shape::from_dims([4, 64, 64]),
                    Shape::from_dims([4, 64, 64]),
                )
            },
        ),
        (
            "shared weight without a packed copy",
            ProjectionRequest {
                packed_copy_permitted: false,
                ..ProjectionRequest::constant(
                    Shape::from_dims([1, 128]),
                    Shape::from_dims([128, 384]),
                )
            },
        ),
        (
            "rank-3 weight",
            ProjectionRequest::constant(
                Shape::from_dims([1, 128]),
                Shape::from_dims([2, 128, 384]),
            ),
        ),
        (
            "i64 weight",
            ProjectionRequest {
                weight_dtype: DType::I64,
                ..ProjectionRequest::constant(
                    Shape::from_dims([1, 128]),
                    Shape::from_dims([128, 384]),
                )
            },
        ),
        (
            "batched activation",
            ProjectionRequest::constant(
                Shape::from_dims([2, 13, 128]),
                Shape::from_dims([128, 384]),
            ),
        ),
    ];
    for (label, request) in cases {
        let verdict = kernels.admits(family, &request);
        assert!(
            matches!(verdict, Support::Partial { .. }),
            "{label}: {verdict}"
        );
        let reason = verdict.reason().unwrap().to_string();
        assert!(reason.contains("#70"), "{label}: {reason}");
        println!("[fn3-24][fallback] {label}: {reason}");
    }
    // The pinned shapes are inside the subset and the op-level verdict is not.
    let admitted = kernels.admits(
        family,
        &ProjectionRequest::constant(Shape::from_dims([13, 768]), Shape::from_dims([768, 3072])),
    );
    assert_eq!(admitted, Support::Full);
    assert!(matches!(kernels.supports(family), Support::Partial { .. }));
}

#[test]
fn coverage_of_the_pinned_graphs() {
    let Some(import) = pinned() else {
        return;
    };
    let device = DeviceId::Cpu;
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let family = fastnn::fn3::backend::vulkan::kernels::quant::QuantFamily::GroupedW8A16;
    print_coverage(
        "tiny prefill(13,0)",
        &coverage(&import, &model, family, dims(13, 0)),
    );
    print_coverage(
        "tiny decode(1,13)",
        &coverage(&import, &model, family, dims(1, 13)),
    );
}

// ---------------------------------------------------------------------------
// Tests: resident proof
// ---------------------------------------------------------------------------

#[test]
fn resident_quantized_projections_upload_nothing() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();

    // ---- prepare ---------------------------------------------------------
    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prepare);
    let packed = PackedSet::quantize(&backend, &model, HeadRoute::Fp32, false)
        .expect("pack the projection weights");
    let prepare = backend.residency_stats();
    assert_eq!(
        prepare.staged_upload_bytes[ResidencyPhase::Prepare.index()],
        packed.payload_bytes(),
        "prepare must move exactly the packed payload"
    );
    assert!(packed.smallest_payload() > 0);
    println!(
        "[fn3-24][prepare] projections={} packed_payload_bytes={} packed_device_bytes={} \
         staged_upload_bytes={}",
        packed.projections.len(),
        packed.payload_bytes(),
        packed.total_device_bytes(),
        prepare.staged_upload_bytes[ResidencyPhase::Prepare.index()]
    );

    // ---- a resident session through the adapter --------------------------
    backend.set_phase(ResidencyPhase::Prepare);
    let decorated = QuantBackend::new(&backend, &packed, false);
    let mut session =
        Session::with_runtime_options(&model, &import, &decorated, SessionOptions::default())
            .expect("session over the adapter");
    let session_prepare = backend.residency_stats();
    assert_eq!(
        session_prepare.late_persistent_uploads, 0,
        "prepare re-uploaded a persistent tensor"
    );
    assert!(
        session_prepare.persistent_prepare_bytes > 0,
        "the FP32 weights are still resident for the fallback projections"
    );

    // ---- prefill ---------------------------------------------------------
    backend.residency().clear();
    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Prefill);
    let submissions_before = backend.queue_submissions();
    let logits = session
        .prefill(&tokens[..PROMPT])
        .expect("quantized prefill");
    let prefill_submissions = backend.queue_submissions() - submissions_before;
    let prefill = backend.residency_stats();
    assert_eq!(prefill.late_persistent_uploads, 0);
    assert!(
        prefill.staged_upload_bytes[ResidencyPhase::Prefill.index()] < packed.smallest_payload(),
        "prefill staged {} bytes, at or above the smallest packed projection ({}): a weight \
         reached the host path",
        prefill.staged_upload_bytes[ResidencyPhase::Prefill.index()],
        packed.smallest_payload()
    );
    let prefill_records = backend.residency().records();
    let quantized = |records: &[fastnn::fn3::backend::vulkan::residency::ExecutionRecord]| {
        records
            .iter()
            .filter(|record| record.kernel == "quant.matmul.w8a16")
            .cloned()
            .collect::<Vec<_>>()
    };
    let prefill_quantized = quantized(&prefill_records);
    assert_eq!(
        prefill_quantized.len(),
        packed.projections.len(),
        "every packed projection (both layers) must have run once"
    );
    for record in &prefill_quantized {
        assert_eq!(
            record.upload_bytes, 0,
            "a quantized op uploaded weight bytes"
        );
        assert_eq!(
            record.readback_bytes, 0,
            "a quantized op read a host output"
        );
        assert!(!record.inputs_read_back);
        assert!(record.proves_device_execution(), "{record:?}");
        for binding in 0..3u32 {
            let snapshot = record
                .bindings
                .iter()
                .find(|snapshot| snapshot.binding == binding)
                .unwrap_or_else(|| panic!("binding {binding} missing"));
            assert!(
                snapshot.device_local,
                "binding {binding} ({:?}) is not device-local",
                snapshot.role
            );
        }
    }
    let prefill_summary = backend.residency().summary();
    assert_eq!(prefill_summary.input_readbacks, 0);
    assert_eq!(
        prefill_summary.proven_device_executions, prefill_summary.executions,
        "every prefill record must prove device execution"
    );
    assert_eq!(logits.shape().dims(), &[1, 1, VOCAB as u64]);

    // ---- decode ----------------------------------------------------------
    backend.residency().clear();
    backend.reset_residency_stats();
    backend.set_phase(ResidencyPhase::Decode);
    let submissions_before = backend.queue_submissions();
    for token in [640u32, 11, 612] {
        session.decode(&[token]).expect("quantized decode");
    }
    let decode_submissions = backend.queue_submissions() - submissions_before;
    let decode = backend.residency_stats();
    assert_eq!(decode.late_persistent_uploads, 0);
    assert!(
        decode.staged_upload_bytes[ResidencyPhase::Decode.index()] < packed.smallest_payload(),
        "decode staged {} bytes, at or above the smallest packed projection ({})",
        decode.staged_upload_bytes[ResidencyPhase::Decode.index()],
        packed.smallest_payload()
    );
    let decode_records = backend.residency().records();
    let decode_quantized = quantized(&decode_records);
    assert_eq!(
        decode_quantized.len(),
        3 * packed.projections.len(),
        "three decode steps, every packed projection"
    );
    for record in &decode_quantized {
        assert_eq!(record.upload_bytes, 0);
        assert_eq!(record.readback_bytes, 0);
        assert!(!record.inputs_read_back);
        assert!(record.proves_device_execution());
    }
    let decode_summary = backend.residency().summary();
    assert_eq!(decode_summary.input_readbacks, 0);
    assert_eq!(
        decode_summary.proven_device_executions,
        decode_summary.executions
    );

    // ---- readback is exactly the logits ----------------------------------
    let steps = 1 + 3;
    assert_eq!(
        decode.readback_bytes + prefill.readback_bytes,
        steps * VOCAB as u64 * 4,
        "the only bytes read back are the four logits rows"
    );
    assert_eq!(
        prefill_summary.readback_bytes + decode_summary.readback_bytes,
        0,
        "no op record read a tensor back"
    );

    let stats = decorated.stats();
    assert_eq!(
        stats.calls as usize,
        (1 + 3) * packed.projections.len(),
        "the adapter must have routed every projection"
    );
    assert_eq!(stats.upload_bytes, 0);
    assert_eq!(stats.readback_bytes, 0);
    println!(
        "[fn3-24][proof] projections={} calls={} quantized_records prefill={} decode={} \
         upload_bytes=0 readback_bytes=0 readback_total={} (logits only) \
         prefill_submissions={} decode_submissions={} prefill_staged={} decode_staged={} \
         weight_bindings_device_local=3",
        packed.projections.len(),
        stats.calls,
        prefill_quantized.len(),
        decode_quantized.len(),
        prefill.readback_bytes + decode.readback_bytes,
        prefill_submissions,
        decode_submissions,
        prefill.staged_upload_bytes[ResidencyPhase::Prefill.index()],
        decode.staged_upload_bytes[ResidencyPhase::Decode.index()],
    );

    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
    println!("[fn3-24][validation] errors {errors} warnings {warnings}");
}

#[test]
fn a_declined_projection_falls_back_to_the_fp32_kernel() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();
    // Pack only the `mlp.c_fc` projections; `c_attn`, `c_proj` and the head
    // must then execute through the FP32 kernel, with the same logits as an
    // undecorated run.
    backend.set_phase(ResidencyPhase::Prepare);
    let full = PackedSet::quantize(&backend, &model, HeadRoute::Fp32, false).expect("pack");
    let only_fc = PackedSet {
        projections: full
            .projections
            .iter()
            .filter(|projection| projection.name.ends_with("mlp.c_fc.weight"))
            .map(|projection| Projection {
                name: projection.name.clone(),
                key: projection.key,
                packed: Arc::clone(&projection.packed),
                source: None,
            })
            .collect(),
        by_key: full
            .by_key
            .iter()
            .filter(|(_, index)| full.projections[**index].name.ends_with("mlp.c_fc.weight"))
            .map(|(key, index)| {
                let position = full.projections[..*index]
                    .iter()
                    .filter(|projection| projection.name.ends_with("mlp.c_fc.weight"))
                    .count();
                (*key, position)
            })
            .collect(),
        head: None,
        shapes: full.shapes.clone(),
    };
    assert_eq!(only_fc.projections.len(), 2, "one c_fc per layer");
    let decorated = QuantBackend::new(&backend, &only_fc, false);
    let mut session =
        Session::with_runtime_options(&model, &import, &decorated, SessionOptions::default())
            .expect("session");
    backend.residency().clear();
    backend.set_phase(ResidencyPhase::Prefill);
    session.prefill(&tokens[..PROMPT]).expect("prefill");
    let records = backend.residency().records();
    let quantized = records
        .iter()
        .filter(|record| record.kernel == "quant.matmul.w8a16")
        .count();
    let fp32_matmul = records
        .iter()
        .filter(|record| record.op == OpId::MatMul && record.kernel == "matmul.tiled.f32")
        .count();
    assert_eq!(
        quantized, 2,
        "only the two packed c_fc projections may run quantized"
    );
    // The pinned graph has 13 `MatMul` instructions per prefill step: four
    // projections per layer (eight), two attention products per layer (four)
    // and the tied head. Two are packed here, so eleven must be FP32.
    assert_eq!(
        fp32_matmul, 11,
        "the three other projections per layer, the four attention products and the head must \
         stay on the FP32 kernel"
    );
    println!(
        "[fn3-24][fallback] pack 2 of 8 projections: quantized records {quantized}, FP32 matmul \
         records {fp32_matmul}"
    );
    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
}

// ---------------------------------------------------------------------------
// Tests: accuracy
// ---------------------------------------------------------------------------

/// Compare one arm's logits with the FP32 arm and apply the #68 gates.
///
/// The KL bound and the per-step top-5 overlap are hard. A top-1 that changes
/// is only a failure when the FP32 margin was *decisive*: larger than twice the
/// largest logit perturbation this arm produced, which is the same "decisive
/// gap" rule `harness::compare_logits` applies to the fixtures. That keeps a
/// tie on a random-weight model from being reported as an accuracy win or loss.
fn compare_quantized_arm(label: &str, fp32: &ArmLogits, quantized: &ArmLogits) -> (f64, bool) {
    let mut worst_kl = 0.0f64;
    let mut all_top1 = true;
    println!(
        "[fn3-24][accuracy] {label} step kl top1_fp32 top1_quant top5_overlap margin_fp32 \
         margin_quant perturbation"
    );
    for (step, (reference, candidate)) in std::iter::once((&fp32.prefill, &quantized.prefill))
        .chain(fp32.decode.iter().zip(&quantized.decode))
        .enumerate()
    {
        let kl = kl_divergence(reference, candidate);
        worst_kl = worst_kl.max(kl);
        let top1_reference = top_indices(reference, 1)[0];
        let top1_candidate = top_indices(candidate, 1)[0];
        let top5_reference = top_indices(reference, 5);
        let top5_candidate = top_indices(candidate, 5);
        let overlap = top5_reference
            .iter()
            .filter(|index| top5_candidate.contains(index))
            .count();
        let perturbation = max_abs(reference, candidate);
        let margin = top1_margin(reference);
        let same = top1_reference == top1_candidate;
        all_top1 &= same;
        println!(
            "[fn3-24][accuracy] {label} {step} {kl:.6e} {top1_reference} {top1_candidate} \
             {overlap}/5 {margin:.6e} {:.6e} {perturbation:.6e}",
            top1_margin(candidate)
        );
        assert!(
            overlap >= 4,
            "{label} step {step}: top-5 overlap {overlap}/5"
        );
        if !same {
            assert!(
                margin <= 2.0 * perturbation,
                "{label} step {step}: top-1 moved from {top1_reference} to {top1_candidate} with \
                 a decisive FP32 margin of {margin:e} against a perturbation of {perturbation:e}"
            );
        }
    }
    assert!(
        worst_kl <= 0.013,
        "{label}: worst KL {worst_kl:e} above the #68 bound 0.013"
    );
    (worst_kl, all_top1)
}

/// One arm's logits for the 13-token prefill and four decode steps.
struct ArmLogits {
    prefill: Vec<f32>,
    decode: Vec<Vec<f32>>,
}

fn run_arm(
    backend: &dyn Backend,
    import: &fastnn::fn3::import::ModelImport,
    model: &fastnn::fn3::PreparedModel,
    tokens: &[u32],
    options: SessionOptions,
) -> ArmLogits {
    let mut session =
        Session::with_runtime_options(model, import, backend, options).expect("session");
    let prefill = session
        .prefill(&tokens[..PROMPT])
        .expect("prefill")
        .as_f32_slice()
        .expect("host logits")
        .to_vec();
    let mut decode = Vec::new();
    for token in [640u32, 11, 612] {
        let logits = session.decode(&[token]).expect("decode");
        decode.push(logits.as_f32_slice().expect("host logits").to_vec());
    }
    ArmLogits { prefill, decode }
}

#[test]
fn end_to_end_accuracy_against_fp32() {
    let Some(backend) = open_physical() else {
        return;
    };
    let Some(import) = pinned() else {
        return;
    };
    let device = backend.device_id();
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();

    let fp32 = run_arm(
        &*backend,
        &import,
        &model,
        &tokens,
        SessionOptions::default(),
    );
    // The FP32 arm is graded against the committed fixtures, so the comparison
    // below is known to be against the real model.
    compare_logits(
        "prefill.p13.last_logits [fp32]",
        &fp32.prefill,
        &fixture("prefill.p13.last_logits"),
    );

    backend.set_phase(ResidencyPhase::Prepare);
    let packed = PackedSet::quantize(&backend, &model, HeadRoute::Packed, true).expect("pack");
    let decorated = QuantBackend::new(&backend, &packed, true);
    // The accuracy arm grades each projection by reading its result, and a
    // readback cannot happen inside a batched step (#72's rule). It therefore
    // runs the staged session; the arithmetic, the kernel and the values are
    // identical, and `resident_quantized_projections_upload_nothing` proves the
    // resident path separately.
    let quantized = run_arm(
        &decorated,
        &import,
        &model,
        &tokens,
        SessionOptions::staged(),
    );

    // ---- per-projection ---------------------------------------------------
    let captured = decorated.captured();
    let mut worst_reference = 0.0f32;
    let mut by_projection: BTreeMap<(String, u64), Captured> = BTreeMap::new();
    for entry in &captured {
        worst_reference = worst_reference.max(entry.reference_max_abs);
        // Prefill (M=13) and decode (M=1) are separate rows; keep the worst.
        let key = (entry.name.clone(), entry.m);
        match by_projection.get(&key) {
            Some(previous) if previous.fp32_relative_frobenius >= entry.fp32_relative_frobenius => {
            }
            _ => {
                by_projection.insert(key, entry.clone());
            }
        }
    }
    assert!(
        worst_reference <= 1e-5,
        "the kernel deviates from the format's reference arithmetic by {worst_reference:e}"
    );
    let mut worst_relative = 0.0f64;
    println!("[fn3-24][accuracy] projection m n k reference_max_abs fp32_relative_frobenius fp32_max_abs");
    for ((name, m), entry) in &by_projection {
        assert!(
            entry.fp32_relative_frobenius.is_finite(),
            "{name}: no FP32 source was captured"
        );
        worst_relative = worst_relative.max(entry.fp32_relative_frobenius);
        println!(
            "[fn3-24][accuracy] {name} {m} {} {} {:.3e} {:.6e} {:.3e}",
            entry.n,
            entry.k,
            entry.reference_max_abs,
            entry.fp32_relative_frobenius,
            entry.fp32_max_abs
        );
    }
    // #68's per-projection bound.
    assert!(
        worst_relative <= 0.0064,
        "per-projection relative Frobenius {worst_relative:e} above the #68 bound 0.0064"
    );

    // ---- logits -----------------------------------------------------------
    let (worst_kl, all_top1) = compare_quantized_arm("tiny", &fp32, &quantized);

    let stats = decorated.stats();
    println!(
        "[fn3-24][accuracy] quantized_calls={} captured={} worst_reference_max_abs={:.3e} \
         worst_relative_frobenius={:.6e} worst_kl={:.6e} top1_all_equal={}",
        stats.calls,
        captured.len(),
        worst_reference,
        worst_relative,
        worst_kl,
        all_top1
    );
    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
}

// ---------------------------------------------------------------------------
// Tests: the trained model
// ---------------------------------------------------------------------------

const TRAINED_BYTES: usize = 498_126_358;
const TRAINED_SHA256: &str = "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d";

/// Load and verify the pinned trained GPT-2 graph, or print why it is skipped.
///
/// The digest is the one in `fixtures/parity/trained-gpt2/model.lock.json`
/// (`onnx/model.onnx`), checked against the file rather than quoted.
fn trained() -> Option<fastnn::fn3::import::ModelImport> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(path) = std::env::var("FN3_QUANT_TRAINED_MODEL") {
        candidates.push(std::path::PathBuf::from(path));
    }
    candidates.push(std::path::PathBuf::from(
        "/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx",
    ));
    for path in candidates {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        assert_eq!(
            bytes.len(),
            TRAINED_BYTES,
            "the file at {} is not the pinned trained graph",
            path.display()
        );
        assert_eq!(
            sha256(&bytes),
            TRAINED_SHA256,
            "the file at {} has the right length but not the pinned bytes",
            path.display()
        );
        println!(
            "[fn3-24][trained] {} bytes, sha256 {TRAINED_SHA256}",
            TRAINED_BYTES
        );
        return Some(
            fastnn::fn3::import::import_onnx_model(&bytes)
                .unwrap_or_else(|error| panic!("the trained model failed to import:\n{error}")),
        );
    }
    if std::env::var("FN3_QUANT_TRAINED_REQUIRED").as_deref() == Ok("1") {
        panic!(
            "FN3_QUANT_TRAINED_REQUIRED=1 but the pinned 498,126,358-byte trained graph is not \
             present; set FN3_QUANT_TRAINED_MODEL"
        );
    }
    println!(
        "[fn3-24][skip] the trained GPT-2 graph is not present; set FN3_QUANT_TRAINED_MODEL or \
         run `python3 tools/parity/fetch_models.py --suite trained-gpt2`"
    );
    None
}

#[test]
fn trained_accuracy_and_coverage() {
    // Opt-in: the trained graph is 498 MB and the packer runs the reference
    // encoder over 494 MB of weights, which is a release-build job. The tiny
    // arm of the same code is part of the default suite.
    if std::env::var("FN3_QUANT_TRAINED").as_deref() != Ok("1") {
        println!(
            "[fn3-24][skip] set FN3_QUANT_TRAINED=1 (and FN3_QUANT_TRAINED_MODEL) to run the \
             trained arm; it is a release-build measurement"
        );
        return;
    }
    let Some(import) = trained() else {
        return;
    };
    let Some(backend) = open_physical() else {
        return;
    };
    let device = backend.device_id();
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let tokens = token_ids();

    // ---- coverage ---------------------------------------------------------
    let family = fastnn::fn3::backend::vulkan::kernels::quant::QuantFamily::GroupedW8A16;
    let geometry = import.geometry();
    let live = |seq: u64, past: u64| {
        fastnn::fn3::LiveDims::new(
            1,
            seq,
            past,
            u64::from(geometry.query_heads),
            u64::from(geometry.kv_heads),
            u64::from(geometry.head_dim),
        )
        .expect("total fits")
    };
    let prefill_rows = coverage(&import, &model, family, live(PROMPT as u64, 0));
    let decode_rows = coverage(&import, &model, family, live(1, PROMPT as u64));
    print_coverage("trained prefill(13,0)", &prefill_rows);
    print_coverage("trained decode(1,13)", &decode_rows);
    let quantized_rows = prefill_rows
        .iter()
        .filter(|row| row.decision == "quantized")
        .count();
    assert_eq!(
        quantized_rows,
        12 * 4,
        "every one of the 12 layers' four projections must be inside the W8A16 subset"
    );
    assert_eq!(
        prefill_rows.len(),
        12 * 6 + 1,
        "12 layers x (4 projections + 2 attention products) + the head"
    );

    // ---- accuracy ---------------------------------------------------------
    let fp32 = run_arm(
        &*backend,
        &import,
        &model,
        &tokens,
        SessionOptions::default(),
    );
    backend.set_phase(ResidencyPhase::Prepare);
    let packed = PackedSet::quantize(&backend, &model, HeadRoute::Packed, true).expect("pack");
    println!(
        "[fn3-24][trained] projections={} packed_payload_bytes={} packed_device_bytes={}",
        packed.projections.len(),
        packed.payload_bytes(),
        packed.total_device_bytes()
    );
    let decorated = QuantBackend::new(&backend, &packed, true);
    let quantized = run_arm(
        &decorated,
        &import,
        &model,
        &tokens,
        SessionOptions::staged(),
    );
    let captured = decorated.captured();
    let mut worst_reference = 0.0f32;
    let mut worst_relative = 0.0f64;
    let mut per_projection: BTreeMap<(String, u64), (f64, f32)> = BTreeMap::new();
    for entry in &captured {
        worst_reference = worst_reference.max(entry.reference_max_abs);
        worst_relative = worst_relative.max(entry.fp32_relative_frobenius);
        let key = (entry.name.clone(), entry.m);
        let slot = per_projection.entry(key).or_insert((0.0, 0.0));
        if entry.fp32_relative_frobenius >= slot.0 {
            *slot = (entry.fp32_relative_frobenius, entry.fp32_max_abs);
        }
    }
    println!("[fn3-24][accuracy] trained projection m fp32_relative_frobenius fp32_max_abs");
    for ((name, m), (relative, maximum)) in &per_projection {
        println!("[fn3-24][accuracy] trained {name} {m} {relative:.6e} {maximum:.3e}");
    }
    assert!(
        worst_reference <= 1e-5,
        "the kernel deviates from the format's reference arithmetic by {worst_reference:e}"
    );
    assert!(
        worst_relative <= 0.0064,
        "per-projection relative Frobenius {worst_relative:e} above the #68 bound 0.0064"
    );
    let (worst_kl, all_top1) = compare_quantized_arm("trained", &fp32, &quantized);
    let stats = decorated.stats();
    println!(
        "[fn3-24][trained] quantized_calls={} captured={} worst_reference_max_abs={:.3e} \
         worst_relative_frobenius={:.6e} worst_kl={:.6e} top1_all_equal={}",
        stats.calls,
        captured.len(),
        worst_reference,
        worst_relative,
        worst_kl,
        all_top1
    );
    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!((errors, warnings), (0, 0), "validation counts");
}

// ---------------------------------------------------------------------------
// Tests: measurement
// ---------------------------------------------------------------------------

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("{name}=`{value}` is not a number: {error}")),
        Err(_) => default,
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn peak_rss_bytes() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(kib) = rest.split_whitespace().next() {
                return kib.parse::<u64>().unwrap_or(0) * 1024;
            }
        }
    }
    0
}

/// One measured arm.
struct Arm {
    label: &'static str,
    ttft: Vec<f64>,
    prefill: Vec<f64>,
    decode: Vec<f64>,
    upload_bytes: u64,
    readback_bytes: u64,
    executions: u64,
    submissions: u64,
    input_readbacks: u64,
    device_peak_live_bytes: u64,
    device_live_bytes: u64,
    allocation_count: u64,
    host_peak_rss_bytes: u64,
    quantized_calls: u64,
    quantized_bytes: u64,
}

fn measure_arm(
    backend: &VulkanBackend,
    import: &fastnn::fn3::import::ModelImport,
    model: &fastnn::fn3::PreparedModel,
    decorated: Option<&QuantBackend<'_>>,
    label: &'static str,
    iterations: usize,
    warmups: usize,
) -> Arm {
    let prompt: [u32; 10] = [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13];
    let decode_tokens: [u32; 3] = [640, 11, 612];
    let run =
        |collect: bool, ttft: &mut Vec<f64>, prefill: &mut Vec<f64>, decode: &mut Vec<f64>| {
            let target: &dyn Backend = match decorated {
                Some(decorated) => decorated,
                None => backend,
            };
            let started = std::time::Instant::now();
            let mut session =
                Session::with_runtime_options(model, import, target, SessionOptions::default())
                    .expect("measurement session");
            let prefill_started = std::time::Instant::now();
            let logits = session.prefill(&prompt).expect("prefill");
            let prefill_done = prefill_started.elapsed();
            let _ = logits.as_f32_slice().expect("host logits");
            let ttft_done = started.elapsed();
            let mut decoded = Vec::new();
            for token in decode_tokens {
                let start = std::time::Instant::now();
                let logits = session.decode(&[token]).expect("decode");
                let _ = logits.as_f32_slice().expect("host logits");
                decoded.push(start.elapsed());
            }
            if collect {
                ttft.push(ttft_done.as_secs_f64() * 1e3);
                prefill.push(prefill_done.as_secs_f64() * 1e3);
                for sample in decoded {
                    decode.push(sample.as_secs_f64() * 1e3);
                }
            }
        };
    for _ in 0..warmups {
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        run(false, &mut a, &mut b, &mut c);
    }
    if let Some(decorated) = decorated {
        // Count the measured iterations only: the warm-ups are not evidence.
        *decorated
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = QuantStats::default();
    }
    backend.residency().clear();
    let memory_before = backend.memory_stats();
    let mut ttft = Vec::with_capacity(iterations);
    let mut prefill = Vec::with_capacity(iterations);
    let mut decode = Vec::with_capacity(iterations * 3);
    for _ in 0..iterations {
        run(true, &mut ttft, &mut prefill, &mut decode);
    }
    let memory_after = backend.memory_stats();
    let summary = backend.residency().summary();
    let (quantized_calls, quantized_bytes) = match decorated {
        Some(decorated) => {
            let stats = decorated.stats();
            (stats.calls, stats.weight_bytes)
        }
        None => (0, 0),
    };
    Arm {
        label,
        ttft,
        prefill,
        decode,
        upload_bytes: summary.upload_bytes,
        readback_bytes: summary.readback_bytes,
        executions: summary.executions,
        submissions: summary.submissions,
        input_readbacks: summary.input_readbacks,
        device_peak_live_bytes: memory_after
            .peak_live_bytes
            .max(memory_before.peak_live_bytes),
        device_live_bytes: memory_after.live_bytes,
        allocation_count: memory_after.allocation_count,
        host_peak_rss_bytes: peak_rss_bytes(),
        quantized_calls,
        quantized_bytes,
    }
}

#[test]
fn quantized_latency_and_memory() {
    if std::env::var("FN3_QUANT_PERF").as_deref() != Ok("1") {
        println!("SKIP: set FN3_QUANT_PERF=1 to run the #70 measurement harness");
        return;
    }
    let Some(backend) = open_physical() else {
        panic!("FN3_QUANT_PERF=1 but no physical Vulkan device is present");
    };
    let Some(import) = pinned() else {
        panic!("FN3_QUANT_PERF=1 but the pinned ONNX model is not present");
    };
    let iterations = env_u64("FN3_QUANT_ITERATIONS", 20) as usize;
    let warmups = env_u64("FN3_QUANT_WARMUPS", 3) as usize;
    let device = backend.device_id();
    let (model, _) = prepared(&import, device, FIXTURE_CONTEXT);
    let mut arms = Vec::new();

    // Fingerprint the FP32 weight allocation before any packing so the memory
    // columns can be attributed.
    let fp32_weight_bytes: u64 = {
        let weights = model.weights();
        (0..weights.len())
            .filter_map(|index| weights.get(index))
            .map(|tensor| tensor.meta().byte_len().unwrap_or(0) as u64)
            .sum()
    };

    let fp32 = measure_arm(&backend, &import, &model, None, "fp32", iterations, warmups);
    arms.push(fp32);

    backend.set_phase(ResidencyPhase::Prepare);
    let projections = PackedSet::quantize(&backend, &model, HeadRoute::Fp32, false).expect("pack");
    let projection_payload = projections.payload_bytes();
    let projection_device = projections.total_device_bytes();
    let decorated = QuantBackend::new(&backend, &projections, false);
    arms.push(measure_arm(
        &backend,
        &import,
        &model,
        Some(&decorated),
        "quantized-projections",
        iterations,
        warmups,
    ));

    backend.set_phase(ResidencyPhase::Prepare);
    let with_head = PackedSet::quantize(&backend, &model, HeadRoute::Packed, false).expect("pack");
    let head_payload = with_head.payload_bytes() - projection_payload;
    let head_device = with_head.total_device_bytes() - projection_device;
    let decorated_head = QuantBackend::new(&backend, &with_head, false);
    arms.push(measure_arm(
        &backend,
        &import,
        &model,
        Some(&decorated_head),
        "quantized-with-head",
        iterations,
        warmups,
    ));

    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let load = load.split_whitespace().next().unwrap_or("?").to_string();
    println!(
        "[fn3-24][perf] loadavg={load} iterations={iterations} warmups={warmups} \
         fp32_weight_bytes={fp32_weight_bytes} projection_payload_bytes={projection_payload} \
         projection_device_bytes={projection_device} head_payload_bytes={head_payload} \
         head_device_bytes={head_device}"
    );
    let mut csv = String::from("arm,sample,kind,ms\n");
    let mut json = String::from("[");
    for (position, arm) in arms.iter_mut().enumerate() {
        let sort = |values: &mut Vec<f64>| {
            values.sort_by(|a, b| a.partial_cmp(b).expect("finite sample"));
            (
                percentile(values, 0.5),
                percentile(values, 0.95),
                values.first().copied().unwrap_or(0.0),
                values.last().copied().unwrap_or(0.0),
            )
        };
        let (ttft_p50, ttft_p95, _, _) = sort(&mut arm.ttft);
        let (prefill_p50, prefill_p95, prefill_min, prefill_max) = sort(&mut arm.prefill);
        let (decode_p50, decode_p95, decode_min, decode_max) = sort(&mut arm.decode);
        println!(
            "[fn3-24][perf] {} prefill_ms p50={prefill_p50:.3} p95={prefill_p95:.3} \
             min={prefill_min:.3} max={prefill_max:.3} | ttft_ms p50={ttft_p50:.3} \
             p95={ttft_p95:.3} | decode_ms p50={decode_p50:.3} p95={decode_p95:.3} \
             min={decode_min:.3} max={decode_max:.3}",
            arm.label
        );
        println!(
            "[fn3-24][perf] {} traffic upload_bytes={} readback_bytes={} executions={} \
             submissions={} input_readbacks={} quantized_calls={} quantized_weight_bytes={}",
            arm.label,
            arm.upload_bytes,
            arm.readback_bytes,
            arm.executions,
            arm.submissions,
            arm.input_readbacks,
            arm.quantized_calls,
            arm.quantized_bytes
        );
        println!(
            "[fn3-24][perf] {} memory device_peak_live_bytes={} device_live_bytes={} \
             allocation_count={} host_peak_rss_bytes={}",
            arm.label,
            arm.device_peak_live_bytes,
            arm.device_live_bytes,
            arm.allocation_count,
            arm.host_peak_rss_bytes
        );
        for (index, sample) in arm.ttft.iter().enumerate() {
            csv.push_str(&format!("{},{index},ttft,{sample:.4}\n", arm.label));
        }
        for (index, sample) in arm.prefill.iter().enumerate() {
            csv.push_str(&format!("{},{index},prefill,{sample:.4}\n", arm.label));
        }
        for (index, sample) in arm.decode.iter().enumerate() {
            csv.push_str(&format!("{},{index},decode,{sample:.4}\n", arm.label));
        }
        if position > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"arm\":\"{}\",\"iterations\":{iterations},\"warmups\":{warmups},\
             \"prefill_ms\":{{\"p50\":{prefill_p50:.4},\"p95\":{prefill_p95:.4},\
             \"min\":{prefill_min:.4},\"max\":{prefill_max:.4}}},\
             \"ttft_ms\":{{\"p50\":{ttft_p50:.4},\"p95\":{ttft_p95:.4}}},\
             \"decode_ms\":{{\"p50\":{decode_p50:.4},\"p95\":{decode_p95:.4},\
             \"min\":{decode_min:.4},\"max\":{decode_max:.4}}},\
             \"upload_bytes\":{},\"readback_bytes\":{},\"executions\":{},\"submissions\":{},\
             \"input_readbacks\":{},\"quantized_calls\":{},\"quantized_weight_bytes\":{},\
             \"device_peak_live_bytes\":{},\"device_live_bytes\":{},\
             \"allocation_count\":{},\"host_peak_rss_bytes\":{},\"loadavg\":\"{load}\"}}",
            arm.label,
            arm.upload_bytes,
            arm.readback_bytes,
            arm.executions,
            arm.submissions,
            arm.input_readbacks,
            arm.quantized_calls,
            arm.quantized_bytes,
            arm.device_peak_live_bytes,
            arm.device_live_bytes,
            arm.allocation_count,
            arm.host_peak_rss_bytes,
        ));
    }
    json.push(']');
    println!("[fn3-24][json] {json}");
    println!("[fn3-24][csv]\n{csv}");

    if let Ok(directory) = std::env::var("FN3_QUANT_OUT") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("measurement directory");
        std::fs::write(directory.join("tiny.json"), format!("{json}\n")).expect("json");
        std::fs::write(directory.join("tiny.csv"), &csv).expect("csv");
        println!("[fn3-24][perf] wrote {}", directory.display());
    }

    let (errors, warnings) = fastnn::fn3::backend::vulkan::instance::validation_counts();
    assert_eq!(
        (errors, warnings),
        (0, 0),
        "the measured run must be validation-clean"
    );
}
