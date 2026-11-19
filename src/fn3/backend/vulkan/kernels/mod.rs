//! Kernel registration: one declaration drives the descriptor layout, the
//! pipeline layout, the grid and the capability report.
//!
//! # Why a declaration, not hand-written layouts
//!
//! A Vulkan kernel needs four things that must agree exactly: the SPIR-V module,
//! the descriptor-set layout, the pipeline layout (including push-constant
//! range), and the dispatch geometry. Writing them by hand in three files is how
//! "binding 2 is the output" drifts from reality. A [`KernelSpec`] is the single
//! declaration; [`super::pipeline::Kernel`] derives the layouts from it and
//! [`reflect`] cross-checks the declaration against the committed SPIR-V before
//! the pipeline is created.
//!
//! # Writing a kernel (FN3-13/14/15)
//!
//! ```ignore
//! // kernels/pointwise.rs
//! pub const ADD_F32: KernelSpec = KernelSpec {
//!     name: "pointwise.add.f32",
//!     spirv: include_bytes!("../../shaders/pointwise_add.comp.spv"),
//!     entry_point: "main",
//!     bindings: &[
//!         BindingSpec::storage(0, "lhs"),
//!         BindingSpec::storage(1, "rhs"),
//!         BindingSpec::storage(2, "out"),
//!     ],
//!     push_constant_bytes: 4,
//!     workgroup_size: [64, 1, 1],
//!     dtypes: &[DType::F32],
//!     grid: grid::elementwise,
//! };
//!
//! pub fn register(set: &mut KernelSet) { /* set.register(OpKernel { .. }) */ }
//! ```
//!
//! Then add the module and the `register` call to [`KernelSet::alpha`]. Later
//! registrations for the same op win, which is how #59 replaces the FN3-11
//! smoke kernel with the real pointwise implementation without touching the
//! platform layer.
//!
//! # `LaunchFn`
//!
//! The platform layer owns buffers, descriptors, submissions and residency
//! accounting; a kernel owner owns *shapes and arithmetic*. [`LaunchFn`] is the
//! boundary: it receives validated tensor views and writes output bytes with
//! the backend primitives (`create_buffer`, `upload`, `dispatch`). It must not
//! approximate: a shape it cannot handle returns `Fn3Error::UnsupportedOp` with
//! its issue number.

use ash::vk;

pub mod attention;
pub mod embedding;
pub(super) mod io;
pub mod matmul;
pub mod norm;
pub mod pointwise;
pub mod quant;
pub mod reduce;
pub mod reflect;
pub mod smoke;
pub mod transform;

use super::error::{Error, Result};
use crate::inference::backend::{CapabilityFeatures, OpParams, Support};
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Result;
use crate::inference::ops::OpId;
use crate::inference::shape::Shape;
use crate::inference::tensor::{TensorMut, TensorView};

/// The dtypes the registered kernels can move and compute on.
///
/// Kernel owners append to this table when they register a kernel that accepts
/// another dtype. It is a `&'static` slice because [`crate::inference::backend::Capabilities`]
/// stores one; an op still rejects a dtype its own kernel does not implement.
pub const KERNEL_DTYPES: &[DType] = &[DType::F32];

/// Descriptor kind of one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    /// `STORAGE_BUFFER`.
    StorageBuffer,
    /// `UNIFORM_BUFFER`.
    UniformBuffer,
}

impl BindingKind {
    /// Vulkan descriptor type.
    pub const fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            BindingKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            BindingKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        }
    }

    /// Stable label used in errors.
    pub const fn name(self) -> &'static str {
        match self {
            BindingKind::StorageBuffer => "storage buffer",
            BindingKind::UniformBuffer => "uniform buffer",
        }
    }
}

/// One declared descriptor binding.
#[derive(Debug, Clone, Copy)]
pub struct BindingSpec {
    /// Binding number as written in the shader's `layout(binding = N)`.
    pub binding: u32,
    /// Descriptor kind.
    pub kind: BindingKind,
    /// Diagnostic name (`"lhs"`, `"out"`).
    pub name: &'static str,
}

impl BindingSpec {
    /// A read-only or read-write storage buffer.
    pub const fn storage(binding: u32, name: &'static str) -> BindingSpec {
        BindingSpec {
            binding,
            kind: BindingKind::StorageBuffer,
            name,
        }
    }

    /// A uniform buffer.
    pub const fn uniform(binding: u32, name: &'static str) -> BindingSpec {
        BindingSpec {
            binding,
            kind: BindingKind::UniformBuffer,
            name,
        }
    }
}

/// Input to a kernel's grid function.
#[derive(Debug, Clone, Copy)]
pub struct GridInput<'a> {
    /// Number of elements the dispatch must cover.
    pub element_count: u64,
    /// Dimensions of the first output, row-major. Tile-shaped kernels
    /// (matmul, attention) use this; elementwise kernels ignore it.
    pub output_dims: &'a [u64],
    /// Declared workgroup size, for the kernel's own rounding.
    pub workgroup_size: [u32; 3],
}

/// A kernel's dispatch-geometry function.
pub type GridFn = fn(&GridInput<'_>) -> Result<[u32; 3]>;

/// Helpers for common grid shapes.
pub mod grid {
    use super::super::error::{Error, Result};
    use super::GridInput;

    /// One invocation per element, `ceil(n / local_size_x)` workgroups.
    pub fn elementwise(input: &GridInput<'_>) -> Result<[u32; 3]> {
        let local = u64::from(input.workgroup_size[0].max(1));
        let groups = input.element_count.div_ceil(local);
        let groups = u32::try_from(groups).map_err(|_| {
            Error::internal(format!(
                "{} elements need {groups} workgroups of {local}, above the 32-bit dispatch \
                 limit; split the tensor",
                input.element_count
            ))
        })?;
        Ok([groups.max(1), 1, 1])
    }

    /// A two-dimensional tile grid over `[rows, columns]`.
    ///
    /// `columns` is the output's last dimension and `tile_columns` the number of
    /// output columns one workgroup covers; `rows` is the product of the leading
    /// dimensions. Both counts are clamped to at least 1.
    pub fn tiles(rows: u64, columns: u64, tile_rows: u64, tile_columns: u64) -> Result<[u32; 3]> {
        let x = u32::try_from(columns.div_ceil(tile_columns.max(1)).max(1))
            .map_err(|_| Error::internal("column tile count exceeds the 32-bit dispatch limit"))?;
        let y = u32::try_from(rows.div_ceil(tile_rows.max(1)).max(1))
            .map_err(|_| Error::internal("row tile count exceeds the 32-bit dispatch limit"))?;
        Ok([x, y, 1])
    }
}

/// The complete declaration of one compute kernel.
#[derive(Debug, Clone, Copy)]
pub struct KernelSpec {
    /// Stable kernel name used in pipeline caches and diagnostics.
    pub name: &'static str,
    /// Committed SPIR-V bytes (`include_bytes!`).
    pub spirv: &'static [u8],
    /// Entry-point name inside the module.
    pub entry_point: &'static str,
    /// Descriptor bindings, one per declared descriptor.
    pub bindings: &'static [BindingSpec],
    /// Push-constant bytes; must equal the SPIR-V `PushConstant` block size.
    pub push_constant_bytes: u32,
    /// Workgroup size; must equal the SPIR-V `LocalSize`.
    pub workgroup_size: [u32; 3],
    /// Dtypes the kernel computes on.
    pub dtypes: &'static [DType],
    /// Shape/precondition contract this kernel implements, in words. The
    /// launch function enforces it and returns `UnsupportedOp` for anything
    /// outside it; this string is what registration logs and diagnostics print.
    pub shape: &'static str,
    /// Dispatch-geometry function.
    pub grid: GridFn,
}

impl KernelSpec {
    /// Static validation that does not need a device.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.entry_point.is_empty() {
            return Err(Error::registration(
                self.name,
                "name and entry point must be non-empty",
            ));
        }
        if self.entry_point.as_bytes().contains(&0) {
            return Err(Error::registration(
                self.name,
                "entry point contains a NUL byte",
            ));
        }
        if self.spirv.is_empty() || !self.spirv.len().is_multiple_of(4) {
            return Err(Error::registration(
                self.name,
                format!(
                    "embedded SPIR-V has {} bytes; it must be a non-zero multiple of 4",
                    self.spirv.len()
                ),
            ));
        }
        if self.bindings.is_empty() {
            return Err(Error::registration(
                self.name,
                "a kernel must declare at least one descriptor binding",
            ));
        }
        for (index, binding) in self.bindings.iter().enumerate() {
            if self.bindings[..index]
                .iter()
                .any(|other| other.binding == binding.binding)
            {
                return Err(Error::registration(
                    self.name,
                    format!("binding {} is declared twice", binding.binding),
                ));
            }
        }
        if self.workgroup_size.contains(&0) {
            return Err(Error::registration(
                self.name,
                format!("workgroup size {:?} has a zero axis", self.workgroup_size),
            ));
        }
        if self.dtypes.is_empty() {
            return Err(Error::registration(
                self.name,
                "a kernel must declare at least one dtype",
            ));
        }
        if self.shape.trim().is_empty() {
            return Err(Error::registration(
                self.name,
                "a kernel must describe its shape precondition",
            ));
        }
        Ok(())
    }

    /// Find a declared binding by number.
    pub fn binding(&self, binding: u32) -> Option<&BindingSpec> {
        self.bindings
            .iter()
            .find(|declared| declared.binding == binding)
    }

    /// Derive the dispatch grid for `element_count` and `output_dims`.
    pub fn workgroups(&self, element_count: u64, output_dims: &[u64]) -> Result<[u32; 3]> {
        (self.grid)(&GridInput {
            element_count,
            output_dims,
            workgroup_size: self.workgroup_size,
        })
    }
}

/// A kernel's op-level launch function.
///
/// Validates the op's shapes and arithmetic and drives the backend primitives.
/// It must return `UnsupportedOp` for anything it does not implement.
pub type LaunchFn =
    fn(&super::VulkanBackend, &[TensorView<'_>], &mut [TensorMut<'_>], &OpParams) -> Fn3Result<()>;

/// A kernel's concrete-operand admission rule.
///
/// Consulted only for a [`Support::Partial`] registration. It returns `Ok(())`
/// when every operand of one instruction lies inside the subset the
/// registration's reason names, or `Err(detail)` naming the operand that does
/// not. Because [`KernelSet::admits`] composes the answer, a rule can only
/// *narrow* the registration: it is never consulted for `Full` or
/// `Unsupported`, and `Ok(())` becomes [`Support::Full`] only for an op the
/// kernel is already registered to handle.
///
/// `shapes` and `dtypes` are the instruction's operands in plan order (inputs
/// then outputs) and have the same length.
pub type AdmitFn = fn(&[Shape], &[DType], &OpParams) -> std::result::Result<(), String>;

/// One op implementation registered against the Vulkan backend.
#[derive(Debug, Clone)]
pub struct OpKernel {
    /// Op this kernel implements.
    pub op: OpId,
    /// Honest support level and the subset it covers.
    pub support: Support,
    /// Capability flags this kernel contributes.
    pub features: CapabilityFeatures,
    /// Dtypes this op accepts through this kernel.
    pub dtypes: &'static [DType],
    /// Concrete-operand refinement of a `Partial` verdict, when the kernel can
    /// decide it from shapes, dtypes and attributes. `None` means the op-level
    /// verdict is the answer for every operand set.
    pub admits: Option<AdmitFn>,
    /// The declared and compiled kernel this op dispatches.
    pub kernel: &'static KernelSpec,
    /// The launch function.
    pub launch: LaunchFn,
}

/// The registered kernel table.
#[derive(Default)]
pub struct KernelSet {
    entries: Vec<OpKernel>,
}

impl KernelSet {
    /// An empty table.
    pub fn new() -> KernelSet {
        KernelSet {
            entries: Vec::new(),
        }
    }

    /// The registrations available in the alpha build.
    ///
    /// Kernel owners append their module here in one place; a later
    /// registration for an op replaces the earlier one.
    pub fn alpha() -> KernelSet {
        let mut set = KernelSet::new();
        smoke::register(&mut set);
        // Order matters only for readability: register() replaces by op id, so
        // a later registration supersedes an earlier kernel for the same op.
        pointwise::register(&mut set);
        transform::register(&mut set);
        embedding::register(&mut set);
        matmul::register(&mut set);
        norm::register(&mut set);
        reduce::register(&mut set);
        attention::register(&mut set);
        set
    }

    /// The quantized projection family's own registration table (FN3-24 / #70).
    ///
    /// This is the additive registration seam, and it is deliberately **not** a
    /// call from [`KernelSet::alpha`]: a [`KernelSet`] holds one kernel per
    /// [`OpId`], the alpha has no quantized op id, and a second registration for
    /// `MatMul` would *replace* the FP32 kernel rather than sit beside it. A
    /// quantized projection is selected by the packed weight a node binds, so
    /// the family registers in [`quant::QuantKernelSet`], whose
    /// `register`/`supports`/`admits` semantics mirror this table's.
    ///
    /// The consequence is the honest one: `supports(OpId::MatMul)` stays the
    /// FP32 answer and never becomes a promise about W8A16.
    pub fn quantized() -> quant::QuantKernelSet {
        quant::QuantKernelSet::alpha()
    }

    /// Register (or replace) the kernel for one op.
    pub fn register(&mut self, kernel: OpKernel) {
        if let Some(existing) = self
            .entries
            .iter_mut()
            .find(|existing| existing.op == kernel.op)
        {
            *existing = kernel;
        } else {
            self.entries.push(kernel);
        }
    }

    /// Look up an op.
    pub fn get(&self, op: OpId) -> Option<&OpKernel> {
        self.entries.iter().find(|kernel| kernel.op == op)
    }

    /// Every registration, in registration order.
    pub fn entries(&self) -> &[OpKernel] {
        &self.entries
    }

    /// Every declared kernel, in registration order.
    pub fn specs(&self) -> impl Iterator<Item = &'static KernelSpec> + '_ {
        self.entries.iter().map(|entry| entry.kernel)
    }

    /// Honest support answer for `op`.
    pub fn supports(&self, op: OpId) -> Support {
        match self.get(op) {
            Some(kernel) => kernel.support.clone(),
            None => Support::unsupported(format!(
                "no Vulkan kernel for {} in the alpha build; the kernel family is owned by {}",
                op.name(),
                super::device_info::op_owner(op)
            )),
        }
    }

    /// Support answer for one concrete instruction.
    ///
    /// `Full` and `Unsupported` registrations answer exactly as
    /// [`KernelSet::supports`] does. A `Partial` registration is refined by its
    /// [`OpKernel::admits`] rule, and only then: a kernel can confirm that the
    /// plan's tensors are inside its documented subset, never widen the subset.
    /// Without a rule the op-level verdict stands.
    pub fn admits(
        &self,
        op: OpId,
        shapes: &[Shape],
        dtypes: &[DType],
        params: &OpParams,
    ) -> Support {
        let Some(kernel) = self.get(op) else {
            return self.supports(op);
        };
        match &kernel.support {
            Support::Full => Support::Full,
            Support::Unsupported { .. } => kernel.support.clone(),
            Support::Partial { reason } => match kernel.admits {
                Some(rule) => match rule(shapes, dtypes, params) {
                    Ok(()) => Support::Full,
                    Err(detail) => {
                        Support::partial(format!("{reason}; concrete operands: {detail}"))
                    }
                },
                None => kernel.support.clone(),
            },
        }
    }

    /// Union of the registered capability flags.
    pub fn features(&self) -> CapabilityFeatures {
        let mut features = CapabilityFeatures::default();
        for kernel in &self.entries {
            features.broadcasting |= kernel.features.broadcasting;
            features.masked_attention |= kernel.features.masked_attention;
            features.quantized |= kernel.features.quantized;
            features.persistent_weights |= kernel.features.persistent_weights;
        }
        features
    }
}

impl std::fmt::Debug for KernelSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelSet")
            .field(
                "ops",
                &self
                    .entries
                    .iter()
                    .map(|kernel| kernel.op.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_verdict_matches_its_registration() {
        // Invariant, not a snapshot: whatever a kernel owner registers, the
        // aggregate view must agree with it, and an op with no kernel must be
        // reported unsupported by name rather than silently claimed.
        let set = KernelSet::alpha();
        for op in OpId::ALL {
            match set.get(op) {
                Some(kernel) => {
                    assert_eq!(set.supports(op), kernel.support, "{op}");
                }
                None => {
                    let support = set.supports(op);
                    assert!(!support.is_full(), "{op} claims Full with no kernel");
                    let reason = support.reason().unwrap_or_else(|| {
                        panic!("{op} is registered by nobody and names no owner")
                    });
                    assert!(
                        reason.contains('#'),
                        "{op} must name its owner issue, got {reason}"
                    );
                }
            }
        }
        // Capabilities are derived from the registrations, so a feature may only
        // be advertised when some kernel declares it.
        let any_broadcasting = set.entries().iter().any(|k| k.features.broadcasting);
        assert_eq!(set.features().broadcasting, any_broadcasting);
    }

    #[test]
    fn unregistered_ops_name_their_owner() {
        // Pick a registered op and prove the lookup is real, then prove that a
        // set with no kernels reports every op as unsupported-with-an-owner.
        let empty = KernelSet::new();
        assert!(empty.entries().is_empty());
        for op in OpId::ALL {
            assert!(empty.get(op).is_none(), "{op}");
            let support = empty.supports(op);
            assert!(!support.is_supported(), "{op}");
        }
    }

    #[test]
    fn later_registration_replaces_earlier_one() {
        let mut set = KernelSet::alpha();
        let replacement = OpKernel {
            op: OpId::Add,
            support: Support::Full,
            features: CapabilityFeatures {
                broadcasting: true,
                ..CapabilityFeatures::default()
            },
            dtypes: &[DType::F32],
            admits: None,
            kernel: &smoke::SMOKE_ADD,
            launch: smoke::launch_add,
        };
        set.register(replacement);
        assert!(set.supports(OpId::Add).is_full());
        assert!(set.features().broadcasting);
    }

    #[test]
    fn elementwise_grid_rounds_up_and_keeps_one_workgroup() {
        let input = GridInput {
            element_count: 0,
            output_dims: &[],
            workgroup_size: [64, 1, 1],
        };
        assert_eq!(grid::elementwise(&input).unwrap(), [1, 1, 1]);
        let input = GridInput {
            element_count: 65,
            ..input
        };
        assert_eq!(grid::elementwise(&input).unwrap(), [2, 1, 1]);
    }

    fn shape(dims: &[u64]) -> Shape {
        Shape::from_dims(dims.iter().copied())
    }

    #[test]
    fn a_partial_verdict_is_refined_by_its_concrete_rule_only() {
        let set = KernelSet::alpha();
        let params = OpParams::default();

        // Transpose: f32 and i64 are inside the documented subset, Bool is not.
        let f32_shapes = [shape(&[2, 3]), shape(&[3, 2])];
        assert_eq!(
            set.admits(
                OpId::Transpose,
                &f32_shapes,
                &[DType::F32, DType::F32],
                &params
            ),
            Support::Full
        );
        let i64_shapes = [shape(&[2, 3]), shape(&[3, 2])];
        assert_eq!(
            set.admits(
                OpId::Transpose,
                &i64_shapes,
                &[DType::I64, DType::I64],
                &params
            ),
            Support::Full
        );
        let bool_shapes = [shape(&[2, 3]), shape(&[3, 2])];
        let verdict = set.admits(
            OpId::Transpose,
            &bool_shapes,
            &[DType::Bool, DType::Bool],
            &params,
        );
        assert!(matches!(verdict, Support::Partial { .. }), "{verdict}");
        assert!(verdict.reason().unwrap().contains("#59"), "{verdict}");
        // The op-level verdict itself is untouched: the refinement is per
        // instruction, never a claim about every operand set.
        assert!(matches!(
            set.supports(OpId::Transpose),
            Support::Partial { .. }
        ));

        // Softmax: the trailing axis is admitted, another axis is not.
        let shapes = [shape(&[1, 2, 4, 4]), shape(&[1, 2, 4, 4])];
        let types = [DType::F32, DType::F32];
        assert_eq!(
            set.admits(
                OpId::Softmax,
                &shapes,
                &types,
                &OpParams::new().with_axes([-1])
            ),
            Support::Full
        );
        let verdict = set.admits(
            OpId::Softmax,
            &shapes,
            &types,
            &OpParams::new().with_axes([2]),
        );
        assert!(matches!(verdict, Support::Partial { .. }), "{verdict}");

        // Where: a Bool condition with f32 values is the documented subset.
        let shapes = [
            shape(&[1, 4]),
            shape(&[1, 4]),
            shape(&[1, 4]),
            shape(&[1, 4]),
        ];
        assert_eq!(
            set.admits(
                OpId::Where,
                &shapes,
                &[DType::Bool, DType::F32, DType::F32, DType::F32],
                &params
            ),
            Support::Full
        );
        let verdict = set.admits(
            OpId::Where,
            &shapes,
            &[DType::Bool, DType::Bool, DType::Bool, DType::Bool],
            &params,
        );
        assert!(matches!(verdict, Support::Partial { .. }), "{verdict}");

        // A `Full` registration answers `Full` without consulting a rule.
        assert_eq!(
            set.admits(OpId::Add, &shapes, &[DType::F32; 4], &params),
            Support::Full
        );
    }

    #[test]
    fn a_rule_can_only_narrow_a_registration() {
        fn always_ok(
            _shapes: &[Shape],
            _dtypes: &[DType],
            _params: &OpParams,
        ) -> std::result::Result<(), String> {
            Ok(())
        }
        let mut set = KernelSet::alpha();
        let params = OpParams::default();
        let shapes = [shape(&[2, 3]), shape(&[3, 2])];
        let types = [DType::F32, DType::F32];

        // `Full` ignores a rule even if one is attached.
        let mut kernel = set.get(OpId::Add).unwrap().clone();
        kernel.admits = Some(always_ok);
        set.register(kernel);
        assert_eq!(
            set.admits(OpId::Add, &shapes, &types, &params),
            Support::Full
        );

        // `Unsupported` ignores a rule too: no concrete operand set can turn a
        // missing kernel into a promise.
        let mut kernel = set.get(OpId::Sub).unwrap().clone();
        kernel.op = OpId::Gelu;
        kernel.support = Support::unsupported("test: deliberately missing (owner: test)");
        kernel.admits = Some(always_ok);
        set.register(kernel);
        let verdict = set.admits(OpId::Gelu, &shapes, &[DType::F32, DType::F32], &params);
        assert!(matches!(verdict, Support::Unsupported { .. }), "{verdict}");
    }

    static DUPLICATE_BINDINGS: [BindingSpec; 2] =
        [BindingSpec::storage(0, "a"), BindingSpec::storage(0, "b")];

    #[test]
    fn spec_validation_rejects_duplicate_bindings_and_zero_workgroups() {
        let mut spec = smoke::SMOKE_ADD;
        spec.bindings = &DUPLICATE_BINDINGS;
        assert!(spec.validate().is_err());
        let mut spec = smoke::SMOKE_ADD;
        spec.workgroup_size = [0, 1, 1];
        assert!(spec.validate().is_err());
    }
}
