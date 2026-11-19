//! W8A16 quantized projections on the resident Vulkan path (FN3-24 / #70).
//!
//! # What this module is
//!
//! One fused projection kernel for the locked `fn3-quant-format-v1` format
//! ([`crate::fn3::quant`]): signed INT8 weight codes in groups of 64 along the
//! contraction axis, an IEEE binary16 scale and an int16 zero-point per
//! `(row, group)`, FP32 activations, and **no dequantized weight buffer**.
//!
//! ```text
//! y[m, n] = sum_g ( sum_{k in g} x[m, k] * (code[n, k] - zero[n, g]) ) * scale[n, g]
//! ```
//!
//! The per-group sum is one FP32 accumulator over ascending `k`; the scale is
//! applied once per group with one FP32 multiply. The codes, scales and
//! zero-points never leave their packed form and are never widened except one
//! scale value per group.
//!
//! # Why it is not registered in [`KernelSet`]
//!
//! [`KernelSet`](super::KernelSet) holds **one kernel per [`OpId`]**, and the
//! alpha has no quantized op id: `OpId::MatMul` is the FP32 tiled product and a
//! second registration for the same id would *replace* it. A quantized
//! projection is selected by the packed weight a node binds, not by its op, so
//! the family registers in [`QuantKernelSet`] instead. That table mirrors the
//! `KernelSet` contract exactly —
//!
//! - `register` replaces by [`QuantFamily`], like `KernelSet::register` by op;
//! - `supports` is the op-level verdict (`Partial`, naming the subset and the
//!   fallback) and never claims `Full` for a shape the launch rejects;
//! - `admits` is the concrete-operand refinement: a rule can only confirm the
//!   documented subset, never widen it, exactly like
//!   [`crate::fn3::backend::Backend::admits`].
//!
//! [`KernelSet::alpha`](super::KernelSet::alpha) is deliberately unchanged, so
//! `supports(OpId::MatMul)` still means "the FP32 kernel" and never silently
//! becomes a promise about W8A16. Mounting this family in the plan needs one
//! core change that is **not** in this issue's ownership: a quantized
//! projection identity (`OpId`, or a packed-weight operand in
//! [`crate::fn3::backend::OpParams`]) plus the planner rule that emits it. Until
//! then a caller selects this kernel explicitly, which is what
//! `tests/fn3_vulkan_quant.rs` does through a `Backend` decorator.
//!
//! # The resident contract (#72)
//!
//! A [`PackedProjection`] owns its **three device buffers** from `prepare`.
//! There is no host copy in the launch path at all: [`launch_projection`] binds
//! `codes`, `scales` and `zeros` by handle, so a per-op weight upload is not a
//! missed branch — it is unrepresentable. The prepare upload is attributed to
//! [`ResidencyPhase::Prepare`](super::super::resident::ResidencyPhase) through
//! `check_staged_input`, so `staged_upload_bytes` in the prefill and decode
//! phases stays at zero for weight bytes, and each launch appends one
//! [`ExecutionRecord`](super::super::residency::ExecutionRecord) whose three
//! weight bindings are device-local.
//!
//! Tile-scale dequantization is *not* used: the group partial is formed from
//! the code byte, so the reported device bytes are the packed bytes and nothing
//! else. (The format permits a tile scratch buffer for a backend without a
//! fused affine; this one has the fused affine.)
//!
//! # Shapes
//!
//! One workgroup covers 64 output columns and up to 16 activation rows, 64
//! invocations, 28 push-constant bytes. Grid `x` is the output-column tile
//! (batch-1 decode's output-channel split: `ceil(50257 / 64) = 786`
//! workgroups), grid `y` the 16-row activation tile. `K` may be any positive
//! length: a non-multiple of 64 has a short final group, and the payload's row
//! padding is zero by the format's rules.
//!
//! # Fallback (format §7)
//!
//! [`admit_projection`] declines, with its own reason string, a weight that is
//! not 2-D F32, a dynamic right-hand side, a shared weight whose packed copy is
//! not permitted, `K < 64` and `N * K < 4096` (metadata would dominate the
//! weight). A decline is a *selection* decision: the FP32 [`OpId::MatMul`]
//! kernel serves the node, exactly as the format's "per-tensor fallback, not a
//! silent graph-wide one" requires.

use std::sync::Arc;

use half::f16;

use super::super::context::{BindingRole, Buffer, BufferDesc};
use super::super::error::{Error as VulkanError, Result as VulkanResult};
use super::super::resident::padded_bytes;
use super::super::VulkanBackend;
use super::io::{self, Operand, OperandMut};
use super::pointwise::Job;
use super::{BindingSpec, GridInput, KernelSpec};
use crate::fn3::backend::vulkan::VULKAN_BACKEND_NAME;
use crate::fn3::backend::{Backend, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::quant::{QuantBlob, QuantTensor, GROUP_SIZE};
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Owning issue named in this family's rejection messages.
pub const KERNEL_OWNER: &str = "FN3-24 (#70)";

/// Output columns one workgroup covers.
pub const COLUMNS_PER_WORKGROUP: u32 = 64;

/// Activation rows one workgroup covers.
pub const ROWS_PER_WORKGROUP: u32 = 16;

/// Words in the push-constant block; must match the shader and its reflection.
const PUSH_CONSTANT_WORDS: usize = 7;

/// Declared kernel: fused grouped-W8A16 projection.
pub const QUANT_MATMUL: KernelSpec = KernelSpec {
    name: "quant.matmul.w8a16",
    spirv: include_bytes!("../shaders/quant_matmul.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "codes"),
        BindingSpec::storage(1, "scales"),
        BindingSpec::storage(2, "zeros"),
        BindingSpec::storage(3, "activation"),
        BindingSpec::storage(4, "out"),
    ],
    push_constant_bytes: (PUSH_CONSTANT_WORDS * 4) as u32,
    workgroup_size: [COLUMNS_PER_WORKGROUP, 1, 1],
    dtypes: &[DType::F32],
    shape: "f32 contiguous activation [M, K] (leading extents 1) against a packed \
            fn3-quant-format-v1 weight [N, K] held device-resident as int8 codes, binary16 \
            scales and int16 zero-points; contiguous f32 output [M, N]; any K >= 1, including \
            a short final group",
    grid: grid_projection,
};

// ---------------------------------------------------------------------------
// Selection policy (format §7)
// ---------------------------------------------------------------------------

/// Where a projection's right-hand side comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSource {
    /// A model weight / folded constant: 2-D F32 bytes available at prepare.
    Constant,
    /// Computed by another instruction at run time (an attention score, a
    /// materialised transpose, a quantized activation).
    Dynamic,
}

/// The concrete operands of one prospective quantized projection.
///
/// This is the "shape/shape/dtype" question the brief asks an admission rule to
/// answer: two shapes (activation and weight), two dtypes, and the provenance
/// facts the format's fallback list names.
#[derive(Debug, Clone)]
pub struct ProjectionRequest {
    /// Activation shape as the `MatMul` sees it (`[..., M, K]`).
    pub activation: Shape,
    /// Source weight shape as the `MatMul` sees it (`[K, N]`).
    pub weight: Shape,
    /// Activation element type.
    pub activation_dtype: DType,
    /// Source weight element type before packing.
    pub weight_dtype: DType,
    /// Whether the weight is available at prepare.
    pub weight_source: WeightSource,
    /// `false` when the weight shares storage with a non-`MatMul` consumer and
    /// the prepared artifact may not carry a separate packed copy.
    pub packed_copy_permitted: bool,
}

impl ProjectionRequest {
    /// A projection of a constant 2-D F32 weight with a permitted packed copy.
    pub fn constant(activation: Shape, weight: Shape) -> ProjectionRequest {
        ProjectionRequest {
            activation,
            weight,
            activation_dtype: DType::F32,
            weight_dtype: DType::F32,
            weight_source: WeightSource::Constant,
            packed_copy_permitted: true,
        }
    }
}

/// `(N, K)` of a `[K, N]` weight, or an error naming the rank.
fn weight_extents(request: &ProjectionRequest) -> std::result::Result<(u64, u64), String> {
    let dims = request.weight.dims();
    if dims.len() != 2 {
        return Err(format!(
            "the weight is {} (rank {}); v1 packs a 2-D weight, so a rank-{} weight keeps the \
             FP32 kernel",
            request.weight,
            dims.len(),
            dims.len()
        ));
    }
    Ok((dims[1], dims[0]))
}

/// The format's selection rule, as the family's concrete admission function.
///
/// `Ok(())` means every operand is inside the documented W8A16 subset; `Err`
/// names the first operand that is not, in the vocabulary of
/// `docs/quantization/fn3-quant-format-v1.md` §7.
pub fn admit_projection(request: &ProjectionRequest) -> std::result::Result<(), String> {
    if request.activation_dtype != DType::F32 {
        return Err(format!(
            "the activation is {}, and v1 is weight-only: activations stay FP32 and are never \
             quantized ({KERNEL_OWNER})",
            request.activation_dtype
        ));
    }
    if request.weight_dtype != DType::F32 {
        return Err(format!(
            "the weight is {}, not f32; the packer reads f32 weights ({}), so this projection \
             keeps the FP32 kernel",
            request.weight_dtype, KERNEL_OWNER
        ));
    }
    if request.weight_source == WeightSource::Dynamic {
        let dims = request.weight.dims();
        return Err(format!(
            "the right-hand side is computed at run time ({}); v1 quantizes a weight known at \
             prepare, and a dynamic RHS keeps the FP32 kernel ({KERNEL_OWNER})",
            format_args!("rank {} {}", dims.len(), request.weight)
        ));
    }
    let (n, k) = weight_extents(request)?;
    if !request.packed_copy_permitted {
        return Err(format!(
            "the weight is shared with a non-MatMul consumer and a packed copy is not \
             permitted; the FP32 table stays authoritative ({KERNEL_OWNER})"
        ));
    }
    if k == 0 || n == 0 {
        return Err(format!(
            "the packed weight would be [{n}, {k}]; a zero extent has no group to quantize \
             ({KERNEL_OWNER})"
        ));
    }
    if k < GROUP_SIZE as u64 {
        return Err(format!(
            "K is {k}, below the {} elements one group needs; the per-group metadata would \
             dominate the weight, so this projection keeps the FP32 kernel ({KERNEL_OWNER})",
            GROUP_SIZE
        ));
    }
    let numel = n.saturating_mul(k);
    if numel < 4096 {
        return Err(format!(
            "the weight has {numel} elements, below the 4096 where quantized metadata pays for \
             itself; this projection keeps the FP32 kernel ({KERNEL_OWNER})"
        ));
    }
    let dims = request.activation.dims();
    if dims.len() < 2 {
        return Err(format!(
            "the activation is {} (rank {}); a projection needs the matrix axes last \
             ({KERNEL_OWNER})",
            request.activation,
            dims.len()
        ));
    }
    let inner = dims[dims.len() - 1];
    if inner != k {
        return Err(format!(
            "the activation contracts {inner} but the weight is [{n}, {k}] ({KERNEL_OWNER})"
        ));
    }
    if dims[..dims.len() - 2].iter().any(|extent| *extent != 1) {
        return Err(format!(
            "the activation {} has a batch axis above 1; the packed kernel is a rank-2 \
             projection and a batched weight is a different contract ({KERNEL_OWNER})",
            request.activation
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Registration table (the `KernelSet::alpha()` equivalent)
// ---------------------------------------------------------------------------

/// Which packed format a registration serves.
///
/// A v2 (4-bit) family would be a second variant; [`QuantKernelSet::register`]
/// replaces by this key, exactly as [`super::KernelSet`] replaces by [`OpId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantFamily {
    /// `fn3-quant-format-v1`: grouped signed INT8, binary16 scale, int16 zero.
    GroupedW8A16,
}

impl QuantFamily {
    /// Stable name used in reports and diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            QuantFamily::GroupedW8A16 => "grouped-w8a16",
        }
    }
}

/// A registration's concrete admission rule.
///
/// The [`QuantKernelSet`] analogue of [`super::AdmitFn`]: `Ok(())` when the
/// projection lies inside the subset the registration's reason names, `Err`
/// naming the operand that does not.
pub type QuantAdmitFn = fn(&ProjectionRequest) -> std::result::Result<(), String>;

/// A family's launch function: packed weights, one activation, one output.
pub type QuantLaunchFn = fn(
    &VulkanBackend,
    &PackedProjection,
    &TensorView<'_>,
    &mut TensorMut<'_>,
) -> Fn3Result<QuantLaunch>;

/// One quantized-family registration.
#[derive(Clone)]
pub struct QuantKernel {
    /// Family this kernel implements.
    pub family: QuantFamily,
    /// Honest support level: `Partial` naming the subset and its fallback.
    pub support: Support,
    /// Concrete-operand refinement of that `Partial` verdict.
    pub admits: Option<QuantAdmitFn>,
    /// The declared and compiled kernel.
    pub kernel: &'static KernelSpec,
    /// The launch function.
    pub launch: QuantLaunchFn,
}

impl std::fmt::Debug for QuantKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantKernel")
            .field("family", &self.family)
            .field("support", &self.support)
            .field("kernel", &self.kernel.name)
            .finish()
    }
}

/// The quantized family's registration table.
#[derive(Debug, Default)]
pub struct QuantKernelSet {
    entries: Vec<QuantKernel>,
}

impl QuantKernelSet {
    /// An empty table.
    pub fn new() -> QuantKernelSet {
        QuantKernelSet {
            entries: Vec::new(),
        }
    }

    /// The registrations available in this build.
    ///
    /// The analogue of [`super::KernelSet::alpha`]: one place to append a
    /// family, replacement by [`QuantFamily`].
    pub fn alpha() -> QuantKernelSet {
        let mut set = QuantKernelSet::new();
        register(&mut set);
        set
    }

    /// Register (or replace) one family's kernel.
    pub fn register(&mut self, kernel: QuantKernel) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|existing| existing.family == kernel.family)
        {
            *existing = kernel;
        } else {
            self.entries.push(kernel);
        }
    }

    /// Look a family up.
    pub fn get(&self, family: QuantFamily) -> Option<&QuantKernel> {
        self.entries.iter().find(|kernel| kernel.family == family)
    }

    /// Every registration, in registration order.
    pub fn entries(&self) -> &[QuantKernel] {
        &self.entries
    }

    /// Every declared kernel, in registration order.
    pub fn specs(&self) -> impl Iterator<Item = &'static KernelSpec> + '_ {
        self.entries.iter().map(|entry| entry.kernel)
    }

    /// Honest support answer for a family.
    pub fn supports(&self, family: QuantFamily) -> Support {
        match self.get(family) {
            Some(kernel) => kernel.support.clone(),
            None => Support::unsupported(format!(
                "no Vulkan kernel for the {} quantized family in this build (owner: {})",
                family.name(),
                KERNEL_OWNER
            )),
        }
    }

    /// Support answer for one concrete prospective projection.
    ///
    /// `Full` and `Unsupported` answer exactly as [`QuantKernelSet::supports`]
    /// does; a `Partial` registration is refined by its rule, and only then.
    pub fn admits(&self, family: QuantFamily, request: &ProjectionRequest) -> Support {
        let Some(kernel) = self.get(family) else {
            return self.supports(family);
        };
        match &kernel.support {
            Support::Full => Support::Full,
            Support::Unsupported { .. } => kernel.support.clone(),
            Support::Partial { reason } => match kernel.admits {
                Some(rule) => match rule(request) {
                    Ok(()) => Support::Full,
                    Err(detail) => {
                        Support::partial(format!("{reason}; concrete operands: {detail}"))
                    }
                },
                None => kernel.support.clone(),
            },
        }
    }
}

/// Register the fused grouped-W8A16 projection.
pub fn register(set: &mut QuantKernelSet) {
    set.register(QuantKernel {
        family: QuantFamily::GroupedW8A16,
        // Partial, never Full: the format's own fallback list is part of this
        // kernel's contract, so a projection outside it must still use the FP32
        // `MatMul`. `supports(OpId::MatMul)` is untouched by this registration.
        support: Support::partial(format!(
            "grouped W8A16 projections of a constant 2-D f32 weight: K >= {GROUP_SIZE}, \
             N*K >= 4096, a permitted packed copy, and a rank-2 activation whose batch axes are \
             1. Every other projection falls back to the FP32 MatMul kernel ({KERNEL_OWNER})"
        )),
        admits: Some(admit_projection),
        kernel: &QUANT_MATMUL,
        launch: launch_projection,
    });
}

// ---------------------------------------------------------------------------
// The packed payload (host side, then device side)
// ---------------------------------------------------------------------------

/// The three payload sections of one packed projection, in the format's layout.
#[derive(Debug, Clone)]
pub struct QuantSections {
    n_rows: u32,
    k: u32,
    groups_per_row: u32,
    row_stride_bytes: u32,
    codes: Vec<u8>,
    scales: Vec<u8>,
    zeros: Vec<u8>,
}

impl QuantSections {
    /// Packed weight rows (N).
    pub fn n_rows(&self) -> u32 {
        self.n_rows
    }
    /// Contraction length (K).
    pub fn k(&self) -> u32 {
        self.k
    }
    /// Groups per row, `ceil(K / 64)`.
    pub fn groups_per_row(&self) -> u32 {
        self.groups_per_row
    }
    /// Packed bytes per code row, `align_up(K, 16)`.
    pub fn row_stride_bytes(&self) -> u32 {
        self.row_stride_bytes
    }
    /// The signed INT8 codes, `N * row_stride_bytes` long.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }
    /// The binary16 scales, `N * groups_per_row * 2` long.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }
    /// The int16 zero-points, `N * groups_per_row * 2` long.
    pub fn zeros(&self) -> &[u8] {
        &self.zeros
    }
    /// Total payload bytes: exactly what goes to the device.
    pub fn byte_len(&self) -> u64 {
        (self.codes.len() + self.scales.len() + self.zeros.len()) as u64
    }
    /// Bytes of the code section.
    pub fn codes_bytes(&self) -> u64 {
        self.codes.len() as u64
    }
    /// Bytes of the scale section.
    pub fn scales_bytes(&self) -> u64 {
        self.scales.len() as u64
    }
    /// Bytes of the zero-point section.
    pub fn zeros_bytes(&self) -> u64 {
        self.zeros.len() as u64
    }

    /// Copy the sections of one validated artifact tensor.
    ///
    /// The blob is already in the format's layout, so this is a plain slice
    /// copy: no dequantization and no re-encoding happens between the reader and
    /// the device.
    pub fn from_tensor(tensor: &QuantTensor<'_>) -> QuantSections {
        QuantSections {
            n_rows: u32::try_from(tensor.n_rows()).expect("the format caps N at 32 bits"),
            k: u32::try_from(tensor.k()).expect("the format caps K at 32 bits"),
            groups_per_row: u32::try_from(tensor.groups_per_row())
                .expect("the format caps the group count at 32 bits"),
            row_stride_bytes: u32::try_from(tensor.row_stride_bytes())
                .expect("the format caps the row stride at 32 bits"),
            codes: tensor.codes().to_vec(),
            scales: tensor.scales().to_vec(),
            zeros: tensor.zero_points().to_vec(),
        }
    }
}

/// One packed projection held in device memory.
#[derive(Debug, Clone)]
pub struct PackedProjection {
    name: String,
    n_rows: u32,
    k: u32,
    groups_per_row: u32,
    row_stride_words: u32,
    codes: Arc<Buffer>,
    scales: Arc<Buffer>,
    zeros: Arc<Buffer>,
    sections: QuantSections,
}

impl PackedProjection {
    /// The name this projection was prepared under.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Output rows (N).
    pub fn n_rows(&self) -> u32 {
        self.n_rows
    }
    /// Contraction length (K).
    pub fn k(&self) -> u32 {
        self.k
    }
    /// Groups per row.
    pub fn groups_per_row(&self) -> u32 {
        self.groups_per_row
    }
    /// Packed bytes per code row.
    pub fn row_stride_bytes(&self) -> u32 {
        self.row_stride_words * 4
    }
    /// Bytes the three device buffers occupy (the allocation, not the payload).
    pub fn device_bytes(&self) -> u64 {
        self.codes.size() + self.scales.size() + self.zeros.size()
    }
    /// The packed payload bytes (codes + scales + zero-points).
    pub fn payload_bytes(&self) -> u64 {
        self.sections.byte_len()
    }
    /// The host sections this projection was prepared from, for auditing.
    pub fn sections(&self) -> &QuantSections {
        &self.sections
    }

    /// Upload one packed projection to the device, once.
    ///
    /// This is the whole residency story for a quantized weight: three
    /// device-local allocations and one transfer each, attributed to the current
    /// residency phase (call it in the prepare phase). After this returns the
    /// launch path has no host bytes to move, so a per-op re-upload is not a
    /// branch that can be forgotten.
    pub fn prepare(
        backend: &VulkanBackend,
        name: impl Into<String>,
        sections: QuantSections,
    ) -> Fn3Result<PackedProjection> {
        let name = name.into();
        validate_sections(&name, &sections)?;
        let codes = upload_section(backend, "quant.codes", &sections.codes)?;
        let scales = upload_section(backend, "quant.scales", &sections.scales)?;
        let zeros = upload_section(backend, "quant.zeros", &sections.zeros)?;
        Ok(PackedProjection {
            name,
            n_rows: sections.n_rows,
            k: sections.k,
            groups_per_row: sections.groups_per_row,
            row_stride_words: sections.row_stride_bytes / 4,
            codes,
            scales,
            zeros,
            sections,
        })
    }

    /// Bind one tensor of a validated artifact blob, by name.
    pub fn from_blob(
        backend: &VulkanBackend,
        blob: &QuantBlob<'_>,
        name: &str,
    ) -> Fn3Result<PackedProjection> {
        let tensor = blob.tensor(name).ok_or_else(|| Fn3Error::InvalidShape {
            context: "packed projection from artifact",
            detail: format!(
                "the artifact has no tensor named \"{name}\"; it holds {:?}",
                blob.tensors()
                    .iter()
                    .map(|tensor| tensor.name())
                    .collect::<Vec<_>>()
            ),
        })?;
        PackedProjection::prepare(backend, name, QuantSections::from_tensor(tensor))
    }
}

/// Reject a section set that does not describe the extents it claims.
fn validate_sections(name: &str, sections: &QuantSections) -> Fn3Result<()> {
    let expected_stride = u64::from(sections.k).div_ceil(16) * 16;
    let expected_groups = u64::from(sections.k).div_ceil(GROUP_SIZE as u64);
    let expected_codes = u64::from(sections.n_rows) * expected_stride;
    let expected_meta = u64::from(sections.n_rows) * expected_groups * 2;
    let checks = [
        (
            "row stride",
            u64::from(sections.row_stride_bytes),
            expected_stride,
        ),
        (
            "groups per row",
            u64::from(sections.groups_per_row),
            expected_groups,
        ),
        ("code bytes", sections.codes_bytes(), expected_codes),
        ("scale bytes", sections.scales_bytes(), expected_meta),
        ("zero-point bytes", sections.zeros_bytes(), expected_meta),
    ];
    for (what, found, expected) in checks {
        if found != expected {
            return Err(Fn3Error::InvalidShape {
                context: "packed projection sections",
                detail: format!(
                    "{name}: {what} is {found}, but N={} K={} needs {expected}",
                    sections.n_rows, sections.k
                ),
            });
        }
    }
    if sections.n_rows == 0 || sections.k == 0 {
        return Err(Fn3Error::InvalidShape {
            context: "packed projection sections",
            detail: format!(
                "{name}: extents [{}, {}] are empty",
                sections.n_rows, sections.k
            ),
        });
    }
    Ok(())
}

/// Allocate one device-local section and move its bytes once.
fn upload_section(
    backend: &VulkanBackend,
    name: &'static str,
    bytes: &[u8],
) -> Fn3Result<Arc<Buffer>> {
    let size = padded_bytes(bytes.len() as u64).max(4);
    backend.capabilities().check_buffer(size)?;
    let buffer = backend.alloc_buffer(&BufferDesc::device_storage(size, name))?;
    // Count the prepare transfer in the residency ledger's upload accounting.
    // (`VulkanBackend::upload` is the raw transfer; `check_staged_input` is the
    // one place that attributes host bytes to a phase.)
    backend.check_staged_input(None, bytes.len() as u64)?;
    let mut padded = Vec::with_capacity(size as usize);
    padded.extend_from_slice(bytes);
    padded.resize(size as usize, 0);
    backend.upload(&buffer, &padded)?;
    Ok(Arc::new(buffer))
}

// ---------------------------------------------------------------------------
// The encoder policy (format §4.5), for harnesses and future prepare paths
// ---------------------------------------------------------------------------

/// Quantize one `[N, K]` f32 weight with the format's reference encoder policy.
///
/// This is the policy of `docs/quantization/fn3-quant-format-v1.md` §4.5 and of
/// `tools/quant/fn3_quant.py::quantize_int8_grouped`, re-derived here because
/// #70 has to build packed weights from a source graph that carries none. It is
/// not part of the reader; a prepared artifact already holds packed sections
/// and is bound with [`PackedProjection::from_blob`].
///
/// Every value must be finite: a NaN or infinity would produce a scale the
/// locked reader rejects, and a silent zero would corrupt the projection.
pub fn encode_projection(values: &[f32], n_rows: usize, k: usize) -> Fn3Result<QuantSections> {
    let expected = n_rows
        .checked_mul(k)
        .ok_or_else(|| Fn3Error::ShapeOverflow {
            context: "quantize projection",
            axis: None,
            detail: format!("[{n_rows}, {k}] elements"),
        })?;
    if n_rows == 0 || k == 0 || values.len() != expected {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: format!("[{n_rows}, {k}] = {expected} f32 values"),
            actual: format!("{} values", values.len()),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Fn3Error::InvalidShape {
            context: "quantize projection",
            detail: "the weight contains a non-finite value; v1 has no encoding for NaN or \
                     infinity and will not substitute a finite one"
                .to_string(),
        });
    }
    let groups = k.div_ceil(GROUP_SIZE);
    let row_stride = k.div_ceil(16) * 16;
    let mut codes = vec![0u8; n_rows * row_stride];
    let mut scales = vec![0u8; n_rows * groups * 2];
    let mut zeros = vec![0u8; n_rows * groups * 2];

    for row in 0..n_rows {
        for group in 0..groups {
            let start = group * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(k);
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for column in start..end {
                let value = values[row * k + column];
                low = low.min(value);
                high = high.max(value);
            }
            let span = high - low;
            let constant = span == 0.0;
            let asym_scale = (span / 255.0).max(1e-30);
            let asym_zero = -128.0 - low / asym_scale;
            let use_asym = !constant && (-128.0..=127.0).contains(&asym_zero);
            let symmetric_abs = low.abs().max(high.abs());
            let symmetric_scale = if symmetric_abs == 0.0 {
                1.0
            } else {
                (symmetric_abs / 127.0).max(1e-6)
            };
            let scale = if use_asym {
                asym_scale
            } else {
                symmetric_scale
            };
            let scale16 = f16::from_f32(scale);
            let scale32 = scale16.to_f32();
            if !scale32.is_finite() || scale32 <= 0.0 {
                return Err(Fn3Error::InvalidShape {
                    context: "quantize projection",
                    detail: format!(
                        "row {row} group {group}: the scale {scale} does not round to a finite, \
                         positive binary16 value; the format has no encoding for it"
                    ),
                });
            }
            let zero = if use_asym {
                round_half_away(-128.0 - f64::from(low / scale32)).clamp(-128.0, 127.0)
            } else {
                0.0
            };
            let zero_i32 = zero as i32;
            let meta = row * groups + group;
            scales[meta * 2..meta * 2 + 2].copy_from_slice(&scale16.to_bits().to_le_bytes());
            zeros[meta * 2..meta * 2 + 2].copy_from_slice(&(zero_i32 as i16).to_le_bytes());
            for column in start..end {
                let value = values[row * k + column];
                let rounded = round_half_away(f64::from(value / scale32)) + zero;
                let code = rounded.clamp(-128.0, 127.0) as i32;
                codes[row * row_stride + column] = (code as i8) as u8;
            }
        }
    }
    Ok(QuantSections {
        n_rows: n_rows as u32,
        k: k as u32,
        groups_per_row: groups as u32,
        row_stride_bytes: row_stride as u32,
        codes,
        scales,
        zeros,
    })
}

/// Round to nearest with ties away from zero, in `f64` (the format's one rule).
///
/// `trunc(v + copysign(0.5, v))`, the rule `tools/quant/fn3_quant.py` applies.
fn round_half_away(value: f64) -> f64 {
    let bias = if value.is_sign_negative() { -0.5 } else { 0.5 };
    (value + bias).trunc()
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

/// What one quantized projection did, for the caller's evidence.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuantLaunch {
    /// Kernel that ran.
    pub kernel: &'static str,
    /// Activation rows.
    pub m: u64,
    /// Output columns.
    pub n: u64,
    /// Contraction length.
    pub k: u64,
    /// Groups per row.
    pub groups: u64,
    /// Dispatch geometry.
    pub workgroups: [u32; 3],
    /// Packed weight bytes bound (codes + scales + zero-points).
    pub weight_bytes: u64,
    /// Device-local weight bindings (`3`).
    pub weight_bindings: u32,
    /// Activation bytes read.
    pub activation_bytes: u64,
    /// Bytes this op uploaded (0 when the activation was resident).
    pub upload_bytes: u64,
    /// Bytes this op read back (0 when the output was resident).
    pub readback_bytes: u64,
}

/// The 28-byte push-constant block, in the shader's declaration order.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PushConstants {
    m: u32,
    n: u32,
    k: u32,
    groups: u32,
    row_stride_words: u32,
    act_base: u32,
    out_base: u32,
}

impl PushConstants {
    fn words(&self) -> [u32; PUSH_CONSTANT_WORDS] {
        [
            self.m,
            self.n,
            self.k,
            self.groups,
            self.row_stride_words,
            self.act_base,
            self.out_base,
        ]
    }
}

/// Dispatch geometry: 64-column tiles on `x`, 16-row tiles on `y`.
fn grid_projection(input: &GridInput<'_>) -> VulkanResult<[u32; 3]> {
    let dims = input.output_dims;
    let rank = dims.len();
    if rank < 2 {
        return Err(VulkanError::internal(format!(
            "quantized projection grid: output rank {rank} has no matrix axes"
        )));
    }
    let columns = dims[rank - 1];
    let rows = dims[rank - 2];
    let axis = |value: u64, what: &str| {
        u32::try_from(value).map_err(|_| {
            VulkanError::internal(format!(
                "quantized projection grid: {value} {what} tiles exceed the 32-bit dispatch limit"
            ))
        })
    };
    Ok([
        axis(
            columns
                .div_ceil(u64::from(input.workgroup_size[0].max(1)))
                .max(1),
            "column",
        )?,
        axis(rows.div_ceil(u64::from(ROWS_PER_WORKGROUP)).max(1), "row")?,
        1,
    ])
}

/// `true` when every leading extent of `shape` is 1.
fn leading_extents_are_one(shape: &Shape) -> bool {
    shape.dims()[..shape.rank() - 2]
        .iter()
        .all(|extent| *extent == 1)
}

fn check_u32(what: &'static str, value: u64) -> Fn3Result<u32> {
    u32::try_from(value).map_err(|_| Fn3Error::UnsupportedOp {
        op: OpId::MatMul,
        backend: VULKAN_BACKEND_NAME,
        reason: format!(
            "{what} is {value}, above the 32-bit index the quantized projection shader \
                 addresses ({KERNEL_OWNER})"
        ),
    })
}

/// A named `UnsupportedOp` that always says who owns the quantized kernel.
fn unsupported(reason: impl Into<String>) -> Fn3Error {
    Fn3Error::UnsupportedOp {
        op: OpId::MatMul,
        backend: VULKAN_BACKEND_NAME,
        reason: format!("{} ({KERNEL_OWNER})", reason.into()),
    }
}

/// Execute one packed projection into `out`.
///
/// `activation` is `[..., M, K]` with every leading extent 1 and `out` is
/// `[..., M, N]` with the same leading extents; both must be contiguous and on
/// this backend's device. The three weight buffers are bound from `projection`
/// and never staged.
pub fn launch_projection(
    backend: &VulkanBackend,
    projection: &PackedProjection,
    activation: &TensorView<'_>,
    out: &mut TensorMut<'_>,
) -> Fn3Result<QuantLaunch> {
    let device = backend.device_id();
    activation
        .meta()
        .require_device(Some(OpId::MatMul), device)?;
    out.meta().require_device(Some(OpId::MatMul), device)?;
    activation
        .meta()
        .require_dtype(Some(OpId::MatMul), DType::F32)?;
    out.meta().require_dtype(Some(OpId::MatMul), DType::F32)?;

    let dims = activation.shape().dims();
    if dims.len() < 2 {
        return Err(unsupported(format!(
            "the activation {} has rank {}; a projection needs the matrix axes last",
            activation.shape(),
            dims.len()
        )));
    }
    if !leading_extents_are_one(activation.shape()) {
        return Err(unsupported(format!(
            "the activation {} has a batch axis above 1; the packed kernel serves a rank-2 \
             projection",
            activation.shape()
        )));
    }
    let m = dims[dims.len() - 2];
    let k = dims[dims.len() - 1];
    let n = u64::from(projection.n_rows);
    if k != u64::from(projection.k) {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: format!(
                "activation contraction {k} against the packed weight K {}",
                projection.k
            ),
            actual: format!("activation {}", activation.shape()),
        });
    }
    let expected = Shape::from_dims(
        dims[..dims.len() - 2]
            .iter()
            .copied()
            .chain([m, n])
            .collect::<Vec<u64>>(),
    );
    if out.shape() != &expected {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(OpId::MatMul),
            expected: expected.to_string(),
            actual: out.shape().to_string(),
        });
    }
    if out.shape().numel()? == 0 {
        return Ok(QuantLaunch {
            kernel: QUANT_MATMUL.name,
            m,
            n,
            k,
            groups: u64::from(projection.groups_per_row),
            ..QuantLaunch::default()
        });
    }
    if !activation.is_contiguous() {
        return Err(unsupported(format!(
            "the activation {} is a strided view ({}); the packed kernel reads one contiguous \
             row-major window",
            activation.meta(),
            activation.strides()
        )));
    }
    if out.meta().is_broadcast() || !out.is_contiguous() {
        return Err(unsupported(format!(
            "the output {} is a broadcast or strided view; the packed kernel writes one \
             contiguous row-major window",
            out.meta()
        )));
    }

    let m32 = check_u32("activation rows M", m)?;
    let n32 = check_u32("output columns N", n)?;
    let k32 = check_u32("contraction length K", k)?;
    let groups = u64::from(projection.groups_per_row);
    let groups32 = check_u32("groups per row", groups)?;
    let activation_bytes = activation.meta().byte_len()? as u64;
    let out_bytes = out.meta().byte_len()? as u64;
    for bytes in [
        activation_bytes,
        out_bytes,
        projection.codes.size(),
        projection.scales.size(),
        projection.zeros.size(),
    ] {
        backend.capabilities().check_buffer(bytes)?;
    }

    let push = PushConstants {
        m: m32,
        n: n32,
        k: k32,
        groups: groups32,
        row_stride_words: projection.row_stride_words,
        act_base: 0,
        out_base: 0,
    };
    let words = push.words();
    let push_bytes: &[u8] = bytemuck::cast_slice(&words);

    let kernel = backend.kernel(&QUANT_MATMUL)?;
    let workgroups = kernel
        .spec()
        .workgroups(out.shape().numel()?, out.shape().dims())
        .map_err(|error| error.to_fn3())?;
    let limits = backend
        .vulkan_device()
        .limits()
        .max_compute_work_group_count;
    for axis in 0..3 {
        if workgroups[axis] > limits[axis] {
            return Err(unsupported(format!(
                "dispatch needs {} workgroups on axis {axis} but \"{}\" allows {}: N {n32}, \
                 M {m32} with {COLUMNS_PER_WORKGROUP}x{ROWS_PER_WORKGROUP} tiles",
                workgroups[axis],
                backend.adapter().name,
                limits[axis]
            )));
        }
    }

    // Resolve operands: three resident weight buffers and the shared
    // activation/output resolver (a resident slot binds in place, a host tensor
    // stages, and a prepared tensor that reaches staging is a hard error).
    let activation_operand = io::input(
        KERNEL_OWNER,
        backend,
        OpId::MatMul,
        "quant.activation",
        activation,
        DType::F32,
    )?;
    let upload_bytes = match &activation_operand {
        Operand::Host { bytes } => bytes.len() as u64,
        Operand::Resident { .. } => 0,
    };
    let out_operand = io::output(
        KERNEL_OWNER,
        backend,
        OpId::MatMul,
        "quant.out",
        out,
        DType::F32,
    )?;
    let readback_bytes = match &out_operand {
        OperandMut::Host { bytes } => bytes.len() as u64,
        OperandMut::Resident { .. } => 0,
    };

    let mut job = Job::new(backend, OpId::MatMul, &QUANT_MATMUL)?;
    let codes_index = job.input_operand(
        "quant.codes",
        &Operand::Resident {
            buffer: Arc::clone(&projection.codes),
        },
    )?;
    let scales_index = job.input_operand(
        "quant.scales",
        &Operand::Resident {
            buffer: Arc::clone(&projection.scales),
        },
    )?;
    let zeros_index = job.input_operand(
        "quant.zeros",
        &Operand::Resident {
            buffer: Arc::clone(&projection.zeros),
        },
    )?;
    let activation_index = job.input_operand("quant.activation", &activation_operand)?;
    let out_index = job.output_operand("quant.out", out_operand)?;
    job.step(
        push_bytes.to_vec(),
        workgroups,
        vec![
            (0, codes_index, BindingRole::Shared),
            (1, scales_index, BindingRole::Shared),
            (2, zeros_index, BindingRole::Shared),
            (3, activation_index, BindingRole::Input),
            (4, out_index, BindingRole::Output),
        ],
    );
    job.run()?;

    Ok(QuantLaunch {
        kernel: QUANT_MATMUL.name,
        m,
        n,
        k,
        groups,
        workgroups,
        weight_bytes: projection.payload_bytes(),
        weight_bindings: 3,
        activation_bytes,
        upload_bytes,
        readback_bytes,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(dims: &[u64]) -> Shape {
        Shape::from_dims(dims.iter().copied())
    }

    #[test]
    fn spec_matches_the_committed_shader_metadata() {
        assert_eq!(QUANT_MATMUL.workgroup_size, [64, 1, 1]);
        assert_eq!(
            QUANT_MATMUL.push_constant_bytes,
            (PUSH_CONSTANT_WORDS * 4) as u32
        );
        assert_eq!(QUANT_MATMUL.bindings.len(), 5);
        assert_eq!(QUANT_MATMUL.entry_point, "main");
        assert!(QUANT_MATMUL.validate().is_ok());
    }

    #[test]
    fn grid_tiles_columns_on_x_and_rows_on_y() {
        let input = GridInput {
            element_count: 0,
            output_dims: &[1, 50257],
            workgroup_size: [COLUMNS_PER_WORKGROUP, 1, 1],
        };
        assert_eq!(grid_projection(&input).unwrap(), [786, 1, 1]);
        let input = GridInput {
            output_dims: &[13, 384],
            ..input
        };
        assert_eq!(grid_projection(&input).unwrap(), [6, 1, 1]);
        let input = GridInput {
            output_dims: &[32, 128],
            ..input
        };
        assert_eq!(grid_projection(&input).unwrap(), [2, 2, 1]);
        let scalar = GridInput {
            output_dims: &[7],
            ..input
        };
        assert!(grid_projection(&scalar).is_err());
    }

    #[test]
    fn push_constants_pack_in_declaration_order() {
        let push = PushConstants {
            m: 1,
            n: 2,
            k: 3,
            groups: 4,
            row_stride_words: 5,
            act_base: 6,
            out_base: 7,
        };
        assert_eq!(push.words().to_vec(), (1u32..=7).collect::<Vec<u32>>());
    }

    #[test]
    fn the_format_encoder_round_trips_the_policy() {
        // A group with a wide positive span: asymmetric, zero-point negative.
        let mut values = Vec::new();
        for row in 0..2 {
            for column in 0..64 {
                values.push(1.0 + row as f32 + column as f32 * 0.01);
            }
        }
        let sections = encode_projection(&values, 2, 64).unwrap();
        assert_eq!(sections.n_rows(), 2);
        assert_eq!(sections.k(), 64);
        assert_eq!(sections.groups_per_row(), 1);
        assert_eq!(sections.row_stride_bytes(), 64);
        assert_eq!(sections.byte_len(), 2 * 64 + 2 * 2 + 2 * 2);
        // Dequantize with the reader's normative arithmetic and compare.
        let dequantized: Vec<f32> = (0..2)
            .flat_map(|row| {
                let scale = f16::from_bits(u16::from_le_bytes([
                    sections.scales()[row * 2],
                    sections.scales()[row * 2 + 1],
                ]))
                .to_f32();
                let zero =
                    i16::from_le_bytes([sections.zeros()[row * 2], sections.zeros()[row * 2 + 1]])
                        as i32;
                (0..64)
                    .map(|column| {
                        let code = sections.codes()[row * 64 + column] as i8 as i32;
                        (code - zero) as f32 * scale
                    })
                    .collect::<Vec<f32>>()
            })
            .collect();
        for (actual, expected) in dequantized.iter().zip(&values) {
            let error = (actual - expected).abs();
            assert!(
                error <= 1e-2 * expected.abs().max(1.0),
                "{actual} vs {expected} ({error})"
            );
        }
    }

    #[test]
    fn the_format_encoder_uses_the_constant_and_short_group_cases() {
        // A constant zero group and a constant non-zero group.
        let mut values = vec![0.0f32; 64];
        values.extend(std::iter::repeat_n(2.0f32, 64));
        let sections = encode_projection(&values, 2, 64).unwrap();
        assert_eq!(
            f16::from_bits(u16::from_le_bytes([
                sections.scales()[0],
                sections.scales()[1]
            ]))
            .to_f32(),
            1.0
        );
        assert_eq!(sections.zeros()[0], 0);
        assert_eq!(sections.codes()[0], 0);
        // A constant 2.0 group is symmetric with scale 2/127, so every code is
        // 127 and the dequantized value is 127 * (2/127) = 2.
        assert!(sections.codes()[64..].iter().all(|code| *code == 127));
        assert!(sections.zeros()[2] == 0 && sections.zeros()[3] == 0);

        // K = 70: two groups, the second six elements long.
        let values: Vec<f32> = (0..70).map(|index| index as f32 * 0.5 - 10.0).collect();
        let sections = encode_projection(&values, 1, 70).unwrap();
        assert_eq!(sections.groups_per_row(), 2);
        assert_eq!(sections.row_stride_bytes(), 80);
        assert_eq!(sections.codes().len(), 80);
        assert!(sections.codes()[70..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn the_encoder_rejects_what_the_format_cannot_represent() {
        assert!(encode_projection(&[0.0; 63], 1, 63).is_ok());
        assert!(encode_projection(&[f32::NAN; 64], 1, 64).is_err());
        assert!(encode_projection(&[f32::INFINITY; 64], 1, 64).is_err());
        assert!(encode_projection(&[0.0; 63], 1, 64).is_err());
    }

    #[test]
    fn support_is_partial_and_names_the_fallback() {
        let set = QuantKernelSet::alpha();
        let support = set.supports(QuantFamily::GroupedW8A16);
        assert!(matches!(support, Support::Partial { .. }), "{support}");
        let reason = support.reason().unwrap();
        assert!(reason.contains("#70"), "{reason}");
        assert!(reason.contains("FP32"), "{reason}");
        assert_eq!(set.specs().count(), 1);
        assert_eq!(set.entries().len(), 1);
    }

    #[test]
    fn the_family_sits_beside_the_fp32_matmul_registration() {
        // The registration seam must not turn `supports(MatMul)` into a promise
        // about W8A16: `MatMul` still answers with the FP32 tiled kernel's own
        // verdict, and the quantized family is a separate table keyed by the
        // packed weight a projection binds.
        let set = super::super::KernelSet::alpha();
        let matmul = set.get(OpId::MatMul).expect("MatMul is registered");
        assert_eq!(matmul.kernel.name, super::super::matmul::MATMUL_TILED.name);
        assert_eq!(set.supports(OpId::MatMul), matmul.support);
        let family = super::super::KernelSet::quantized();
        assert_eq!(family.entries().len(), 1);
        let support = family.supports(QuantFamily::GroupedW8A16);
        assert!(matches!(support, Support::Partial { .. }), "{support}");
        assert!(support.reason().unwrap().contains("FP32 MatMul"));
    }

    #[test]
    fn a_partial_verdict_is_refined_by_the_concrete_rule_only() {
        let set = QuantKernelSet::alpha();
        let family = QuantFamily::GroupedW8A16;
        let admitted = set.admits(
            family,
            &ProjectionRequest::constant(shape(&[13, 128]), shape(&[128, 384])),
        );
        assert_eq!(admitted, Support::Full);

        // A dynamic RHS is refused with a reason, never widened to Full.
        let mut request = ProjectionRequest::constant(shape(&[1, 128]), shape(&[128, 384]));
        request.weight_source = WeightSource::Dynamic;
        let verdict = set.admits(family, &request);
        assert!(matches!(verdict, Support::Partial { .. }), "{verdict}");
        assert!(verdict.reason().unwrap().contains("dynamic"), "{verdict}");
    }

    #[test]
    fn the_selection_rule_matches_the_format_fallback_list() {
        let family = QuantFamily::GroupedW8A16;
        let set = QuantKernelSet::alpha();
        let cases: Vec<(&str, ProjectionRequest)> = vec![
            (
                "K below one group",
                ProjectionRequest::constant(shape(&[1, 63]), shape(&[63, 384])),
            ),
            (
                "numel below 4096",
                ProjectionRequest::constant(shape(&[1, 64]), shape(&[64, 32])),
            ),
            (
                "non-f32 weight",
                ProjectionRequest {
                    weight_dtype: DType::I64,
                    ..ProjectionRequest::constant(shape(&[1, 128]), shape(&[128, 384]))
                },
            ),
            (
                "non-f32 activation",
                ProjectionRequest {
                    activation_dtype: DType::I32,
                    ..ProjectionRequest::constant(shape(&[1, 128]), shape(&[128, 384]))
                },
            ),
            (
                "shared weight without a packed copy",
                ProjectionRequest {
                    packed_copy_permitted: false,
                    ..ProjectionRequest::constant(shape(&[1, 128]), shape(&[128, 384]))
                },
            ),
            (
                "rank-3 weight",
                ProjectionRequest::constant(shape(&[1, 128]), shape(&[2, 64, 384])),
            ),
            (
                "batched activation",
                ProjectionRequest::constant(shape(&[2, 13, 128]), shape(&[128, 384])),
            ),
            (
                "rank-1 activation",
                ProjectionRequest::constant(shape(&[128]), shape(&[128, 384])),
            ),
        ];
        for (label, request) in cases {
            let verdict = set.admits(family, &request);
            assert!(
                matches!(verdict, Support::Partial { .. }),
                "{label}: {verdict}"
            );
            assert!(verdict.reason().unwrap().contains("#70"), "{label}");
        }
        // The pinned tiny and trained projection shapes are inside the subset.
        for (activation, weight) in [
            ([13u64, 128u64], [128u64, 384u64]),
            ([1, 128], [128, 50257]),
            ([13, 768], [768, 3072]),
            ([1, 768], [768, 2304]),
        ] {
            let request = ProjectionRequest::constant(shape(&activation), shape(&weight));
            assert_eq!(set.admits(family, &request), Support::Full, "{weight:?}");
        }
    }

    #[test]
    fn an_unregistered_family_is_unsupported_with_its_owner() {
        let set = QuantKernelSet::new();
        let family = QuantFamily::GroupedW8A16;
        let support = set.supports(family);
        assert!(!support.is_supported(), "{support}");
        assert!(support.reason().unwrap().contains("#70"));
    }

    #[test]
    fn later_registration_replaces_the_earlier_one() {
        let mut set = QuantKernelSet::alpha();
        let mut replacement = set
            .get(QuantFamily::GroupedW8A16)
            .expect("registered")
            .clone();
        replacement.support = Support::Full;
        set.register(replacement);
        assert_eq!(set.supports(QuantFamily::GroupedW8A16), Support::Full);
        assert_eq!(set.entries().len(), 1);
    }
}
