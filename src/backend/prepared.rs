use crate::backend::{BufferSlice, ExecutablePlan, Instruction};
use crate::ir::node::NodeId;
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct PreparedExecutablePlan {
    pub instructions: Vec<PreparedInstruction>,
    pub arena_size: usize,
    pub scratch_size: usize,
    /// Static constant data attached to this plan (metadata only — not
    /// consulted by any runtime dispatch path yet). Populated via
    /// [`PreparedExecutablePlan::register_constant_arena`].
    constant_arena: Option<PreparedConstantArena>,
}

#[derive(Clone, Debug)]
pub enum PreparedInstruction {
    Generic { instruction_index: usize },
    Conv2d(PreparedConv2d),
    MatMul(PreparedMatMul),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedActivation {
    None,
    Relu,
    Gelu,
    Silu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedConvKernelKind {
    CurrentIm2colGemm,
    CurrentOneByOneGemm,
    FuturePackedFp32,
    FuturePackedI8,
    FuturePackedU4,
}

#[derive(Clone, Debug)]
pub struct PreparedConv2d {
    pub instruction_index: usize,
    pub node_id: Option<NodeId>,
    pub input: BufferSlice,
    pub weight: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub n: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub f: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub activation: PreparedActivation,
    pub kernel: PreparedConvKernelKind,
    pub packed_weight: Option<PackedWeightId>,
    pub scratch_offset: usize,
    pub scratch_len: usize,
}

#[derive(Clone, Debug)]
pub struct PreparedMatMul {
    pub instruction_index: usize,
    pub node_id: Option<NodeId>,
    pub a: BufferSlice,
    pub b: BufferSlice,
    pub bias: Option<BufferSlice>,
    pub output: BufferSlice,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub activation: PreparedActivation,
    pub packed_weight: Option<PackedWeightId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedWeightId {
    /// Dense slot index in the owning [`PreparedConstantArena`].
    pub index: usize,
    /// Dtype tag describing how the slot stores its bytes. Defaults to
    /// [`PackedWeightKind::Fp32`] in this skeleton — every other variant
    /// is reserved for future packed precision lanes.
    pub kind: PackedWeightKind,
    /// Optional byte offset reserved for a future arena-layout planner.
    /// Always `None` today.
    pub arena_slot: Option<u32>,
}

impl PackedWeightId {
    /// Build a default `fp32` handle pointing at `index`. Equivalent to
    /// the wave-3 newtype constructor `PackedWeightId(index)` — preserved
    /// here as a small helper so callers do not need to spell the full
    /// struct literal.
    pub const fn new(index: usize) -> Self {
        Self {
            index,
            kind: PackedWeightKind::Fp32,
            arena_slot: None,
        }
    }

    /// Build a handle with the explicit kind and arena-slot metadata.
    pub const fn with_kind(index: usize, kind: PackedWeightKind, arena_slot: Option<u32>) -> Self {
        Self {
            index,
            kind,
            arena_slot,
        }
    }
}

impl Default for PackedWeightId {
    fn default() -> Self {
        Self::new(0)
    }
}

impl From<usize> for PackedWeightId {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

/// Dtype tag describing how a constant slot is encoded on disk / in the
/// arena.
///
/// Only [`PackedWeightKind::Fp32`] is exercised by the current runtime.
/// The remaining variants reserve discriminant space for future packed
/// precision modes (`i8`, packed `u4`, NF4) so consumers can match
/// exhaustively without breaking when those lanes light up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PackedWeightKind {
    Fp32 = 0,
    /// Reserved — symmetric int8 weight-only quantisation.
    I8 = 1,
    /// Reserved — packed 4-bit unsigned weight-only quantisation.
    U4 = 2,
    /// Reserved — packed 4-bit NormalFloat weight-only quantisation.
    Nf4 = 3,
}

impl PackedWeightKind {
    /// Stable, human-readable tag used for logging and introspection.
    pub const fn name(&self) -> &'static str {
        match self {
            PackedWeightKind::Fp32 => "fp32",
            PackedWeightKind::I8 => "i8",
            PackedWeightKind::U4 => "u4",
            PackedWeightKind::Nf4 => "nf4",
        }
    }

    /// Element width in bytes for the encoded representation. The
    /// reserved kinds report their nominal element widths even though
    /// no kernel reads them yet.
    pub const fn element_bytes(&self) -> usize {
        match self {
            PackedWeightKind::Fp32 => std::mem::size_of::<f32>(),
            PackedWeightKind::I8 => 1,
            // u4/nf4 are sub-byte; report 1 byte per *pair* of elements.
            PackedWeightKind::U4 | PackedWeightKind::Nf4 => 1,
        }
    }
}

impl Default for PackedWeightKind {
    fn default() -> Self {
        PackedWeightKind::Fp32
    }
}

/// Per-slot storage payload for an entry in a [`PreparedConstantArena`].
///
/// The enum is intentionally narrow today: `Unpacked` carries the f32
/// reference payload that the current runtime can already consume, and
/// `Reserved` is a placeholder discriminant so the surface stays small
/// while still being exhaustive when matched. Future packed precision
/// lanes will add new variants here (e.g. `PackedI8`, `PackedU4`) once
/// the corresponding kernels and accuracy gates land.
#[derive(Clone, Debug)]
pub enum PackedWeightStore {
    /// Reference `fp32` payload — the layout the current CPU backend
    /// consumes directly via `WriteConst`.
    Unpacked(Vec<f32>),
    /// Reserved discriminant for future packed precision variants. Carries
    /// no payload and is never produced by the current code paths.
    Reserved,
}

impl PackedWeightStore {
    /// Build an [`PackedWeightStore::Unpacked`] entry holding `data`.
    pub fn unpacked(data: Vec<f32>) -> Self {
        PackedWeightStore::Unpacked(data)
    }

    /// View the stored data as an `f32` slice when the store is
    /// `Unpacked`; returns `None` for every other variant.
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            PackedWeightStore::Unpacked(data) => Some(data.as_slice()),
            PackedWeightStore::Reserved => None,
        }
    }

    /// Short tag identifying the variant, used for introspection.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            PackedWeightStore::Unpacked(_) => "unpacked",
            PackedWeightStore::Reserved => "reserved",
        }
    }
}

/// Metadata + payload for one constant tensor registered in a
/// [`PreparedConstantArena`].
#[derive(Clone, Debug)]
pub struct PreparedConstantEntry {
    /// Handle returned by [`PreparedConstantArena::insert`].
    pub id: PackedWeightId,
    /// Symbolic name / role (e.g. `"conv_weight"`, `"matmul_bias"`).
    pub name: String,
    /// Dtype tag for the stored payload. Always [`PackedWeightKind::Fp32`]
    /// in this skeleton.
    pub kind: PackedWeightKind,
    /// Element count of the stored tensor.
    pub numel: usize,
    /// Total byte length of the stored payload (numel * element_bytes).
    pub byte_len: usize,
    /// Backing storage variant. Always
    /// [`PackedWeightStore::Unpacked`] in this skeleton.
    pub store: PackedWeightStore,
}

/// Owns the static `Vec<f32>` payloads attached to a prepared plan.
///
/// This type is **metadata only** in the current wave: no kernel and no
/// dispatch path consults it. A later lane will populate it from
/// [`Instruction::WriteConst`] producers and route conv/matmul weight
/// reads through the arena instead of re-materialising bytes on every
/// forward pass. Until then the arena is a parallel storage surface
/// whose lifetime is tied to the [`PreparedExecutablePlan`] it is
/// registered against.
#[derive(Clone, Debug, Default)]
pub struct PreparedConstantArena {
    entries: Vec<PreparedConstantEntry>,
    name_to_id: HashMap<String, PackedWeightId>,
}

impl PreparedConstantArena {
    /// Empty arena with no registered constants.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new `fp32` constant under `name`.
    ///
    /// Returns the [`PackedWeightId`] the constant occupies. When `name`
    /// is already registered the existing id is returned and `data` is
    /// dropped — duplicate names never duplicate storage.
    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>) -> PackedWeightId {
        let name = name.into();
        if let Some(id) = self.name_to_id.get(&name) {
            return *id;
        }
        let index = self.entries.len();
        let numel = data.len();
        let byte_len = numel * PackedWeightKind::Fp32.element_bytes();
        let id = PackedWeightId::new(index);
        self.entries.push(PreparedConstantEntry {
            id,
            name: name.clone(),
            kind: PackedWeightKind::Fp32,
            numel,
            byte_len,
            store: PackedWeightStore::unpacked(data),
        });
        self.name_to_id.insert(name, id);
        id
    }

    /// Resolve `name` to its registered [`PackedWeightId`].
    pub fn id_for(&self, name: &str) -> Option<PackedWeightId> {
        self.name_to_id.get(name).copied()
    }

    /// Fetch the `f32` payload for `id`. Returns `None` when the id is
    /// out of range or the slot is not stored as
    /// [`PackedWeightStore::Unpacked`].
    pub fn get(&self, id: PackedWeightId) -> Option<&[f32]> {
        self.entries
            .get(id.index)
            .and_then(|entry| entry.store.as_f32_slice())
    }

    /// Full metadata for the constant referenced by `id`, if any.
    pub fn entry(&self, id: PackedWeightId) -> Option<&PreparedConstantEntry> {
        self.entries.get(id.index)
    }

    /// Number of registered constants.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no constants have been registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sum of [`PreparedConstantEntry::byte_len`] across all entries.
    pub fn total_bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.byte_len).sum()
    }

    /// Iterator over the registered entries in insertion order.
    pub fn entries(&self) -> impl Iterator<Item = &PreparedConstantEntry> {
        self.entries.iter()
    }
}

/// Map a kernel name suffix to the corresponding [`PreparedActivation`].
///
/// Recognized names: `"conv2d"` (None), `"conv2d_relu"`, `"conv2d_gelu"`,
/// `"conv2d_silu"`. Everything else returns `None`.
pub fn activation_from_kernel_name(kernel_name: &str) -> PreparedActivation {
    match kernel_name {
        "conv2d_relu" | "matmul_relu" => PreparedActivation::Relu,
        "conv2d_gelu" | "matmul_gelu" => PreparedActivation::Gelu,
        "conv2d_silu" | "matmul_silu" => PreparedActivation::Silu,
        _ => PreparedActivation::None,
    }
}

/// Map a kernel name to the appropriate [`PreparedConvKernelKind`].
///
/// Quantized kernels (`"conv2d_u4"`, `"conv2d_u8"`) map to their future
/// packed variants; float kernels map to the current im2col/gemm path,
/// with a 1x1 specialisation when kernel dimensions are both 1.
pub fn kernel_kind_from_kernel_name(kernel_name: &str) -> PreparedConvKernelKind {
    match kernel_name {
        "conv2d_u4" => PreparedConvKernelKind::FuturePackedU4,
        "conv2d_u8" => PreparedConvKernelKind::FuturePackedI8,
        _ => PreparedConvKernelKind::CurrentIm2colGemm,
    }
}

/// Attempt to promote an [`Instruction`] into a [`PreparedInstruction::Conv2d`].
///
/// Returns `Some(PreparedInstruction::Conv2d(...))` only when:
/// - The instruction is a `CallKernel` whose `kernel_name` starts with `"conv2d"`.
/// - `params` has exactly 9 elements: `[stride, padding, dilation, groups, c, h, w, kh, kw]`.
/// - At least two input slices are present (input tensor + weight tensor).
///
/// Returns `None` for any instruction that does not satisfy these invariants;
/// callers should fall back to `PreparedInstruction::Generic`.
///
/// # Required metadata fields from `CallKernel`
/// - `kernel_name`: must begin with `"conv2d"` to trigger promotion.
/// - `params`: `[stride, padding, dilation, groups, c, h, w, kh, kw]`
///   (all `usize`). These are baked in at compile time by the CPU backend.
/// - `input_slices[0]`: input tensor; `size / 4` gives element count (f32).
/// - `input_slices[1]`: weight tensor; `size / 4` gives element count.
/// - `input_slices[2]` (optional): bias tensor.
/// - `output_slice`: output tensor.
/// - `node_id`: optional node identifier for runtime param tightening.
pub fn try_prepare_conv2d(
    instruction: &Instruction,
    instruction_index: usize,
) -> Option<PreparedInstruction> {
    let Instruction::CallKernel {
        kernel_name,
        input_slices,
        output_slice,
        params,
        node_id,
        ..
    } = instruction
    else {
        return None;
    };

    if !kernel_name.starts_with("conv2d") {
        return None;
    }

    // Conv2d params: [stride, padding, dilation, groups, c, h, w, kh, kw]
    let slice = params.as_slice();
    if slice.len() != 9 {
        return None;
    }
    let [stride, padding, dilation, groups, c, h, w, kh, kw]: [usize; 9] = slice.try_into().ok()?;

    if input_slices.len() < 2 {
        return None;
    }

    let input = input_slices[0];
    let weight = input_slices[1];
    let bias = input_slices.get(2).copied();

    // Derive n (batch) and f (output channels) from buffer sizes.
    // Buffers store f32, so element count = size / 4.
    let input_elems = input.size / std::mem::size_of::<f32>();
    let n = input_elems / c.saturating_mul(h).saturating_mul(w).max(1);

    let weight_elems = weight.size / std::mem::size_of::<f32>();
    let c_per_group = c / groups.max(1);
    let f = weight_elems / c_per_group.saturating_mul(kh).saturating_mul(kw).max(1);

    let activation = activation_from_kernel_name(kernel_name);
    let kernel = kernel_kind_from_kernel_name(kernel_name);

    Some(PreparedInstruction::Conv2d(PreparedConv2d {
        instruction_index,
        node_id: *node_id,
        input,
        weight,
        bias,
        output: *output_slice,
        n,
        c,
        h,
        w,
        f,
        kh,
        kw,
        stride,
        padding,
        dilation,
        groups,
        activation,
        kernel,
        packed_weight: None,
        scratch_offset: 0,
        scratch_len: 0,
    }))
}

/// Build a [`PreparedExecutablePlan`] by inspecting each instruction.
///
/// Statically recognizable Conv2d `CallKernel` instructions are promoted to
/// `PreparedInstruction::Conv2d`; everything else becomes
/// `PreparedInstruction::Generic`. This preserves runtime semantics while
/// enabling future kernel-specialization paths.
pub fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan {
    PreparedExecutablePlan {
        instructions: plan
            .instructions
            .iter()
            .enumerate()
            .map(|(instruction_index, inst)| {
                try_prepare_conv2d(inst, instruction_index)
                    .unwrap_or(PreparedInstruction::Generic { instruction_index })
            })
            .collect(),
        arena_size: plan.arena_size,
        scratch_size: 0,
        constant_arena: None,
    }
}

// ── PreparedInstruction helpers ──────────────────────────────

impl PreparedInstruction {
    /// Return the original instruction index within the [`ExecutablePlan`]
    /// that produced this prepared instruction.
    pub fn instruction_index(&self) -> usize {
        match self {
            PreparedInstruction::Generic { instruction_index } => *instruction_index,
            PreparedInstruction::Conv2d(c) => c.instruction_index,
            PreparedInstruction::MatMul(m) => m.instruction_index,
        }
    }
}

// ── PreparedExecutablePlan helpers ───────────────────────────

impl PreparedExecutablePlan {
    /// Number of prepared instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Returns `true` when the plan contains no instructions.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Original instruction indices of all `Generic` fallback instructions.
    pub fn generic_instruction_indices(&self) -> Vec<usize> {
        self.instructions
            .iter()
            .filter_map(|inst| match inst {
                PreparedInstruction::Generic { instruction_index } => Some(*instruction_index),
                _ => None,
            })
            .collect()
    }

    /// The original instruction indices in plan order.
    ///
    /// Each entry is the index of the corresponding instruction in the
    /// source [`ExecutablePlan`]. The returned `Vec` has the same length
    /// and ordering as `self.instructions`.
    pub fn original_instruction_order(&self) -> Vec<usize> {
        self.instructions
            .iter()
            .map(|inst| inst.instruction_index())
            .collect()
    }

    /// Attach a [`PreparedConstantArena`] to this plan.
    ///
    /// The arena is **metadata only** in this wave: no execution path
    /// consults it. A later lane will detect [`Instruction::WriteConst`]
    /// producers feeding conv/matmul weight slots and route those reads
    /// through the arena instead of re-materialising the bytes on every
    /// forward pass.
    pub fn register_constant_arena(&mut self, arena: PreparedConstantArena) {
        self.constant_arena = Some(arena);
    }

    /// Borrow the attached constant arena, if any.
    pub fn constant_arena(&self) -> Option<&PreparedConstantArena> {
        self.constant_arena.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BufferSlice, Instruction};

    fn empty_plan() -> ExecutablePlan {
        ExecutablePlan {
            instructions: vec![],
            arena_size: 1024,
            levels: vec![],
        }
    }

    /// Helper: build a conv2d CallKernel instruction with the canonical param layout.
    #[derive(Clone, Copy)]
    struct Conv2dParams {
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        with_bias: bool,
    }

    impl Default for Conv2dParams {
        fn default() -> Self {
            Self {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: 1,
                c: 3,
                h: 8,
                w: 8,
                kh: 3,
                kw: 3,
                with_bias: false,
            }
        }
    }

    fn make_conv2d_instruction(kernel_name: &str, params: Conv2dParams) -> Instruction {
        // f32 element size = 4 bytes
        let input_size = params.c * params.h * params.w * 4; // 1 batch
        let c_per_group = params.c / params.groups.max(1);
        let f = 8; // arbitrary output channels
        let weight_size = f * c_per_group * params.kh * params.kw * 4;
        let bias_size = if params.with_bias { f * 4 } else { 0 };

        let mut input_slices = vec![
            BufferSlice::new(0, input_size),           // input
            BufferSlice::new(input_size, weight_size), // weight
        ];
        if params.with_bias {
            input_slices.push(BufferSlice::new(input_size + weight_size, bias_size));
        }

        Instruction::CallKernel {
            kernel_name: kernel_name.to_string(),
            input_slices,
            output_slice: BufferSlice::new(
                input_size + weight_size + bias_size,
                input_size, // output size doesn't matter for metadata
            ),
            secondary_output_slice: None,
            params: vec![
                params.stride,
                params.padding,
                params.dilation,
                params.groups,
                params.c,
                params.h,
                params.w,
                params.kh,
                params.kw,
            ],
            node_id: Some(42usize),
            param_dims: None,
            weight_meta: None,
        }
    }

    // ── prepare_executable_plan ────────────────────────────────

    #[test]
    fn prepare_empty_plan() {
        let plan = empty_plan();
        let prepared = prepare_executable_plan(&plan);
        assert!(prepared.instructions.is_empty());
        assert_eq!(prepared.arena_size, 1024);
        assert_eq!(prepared.scratch_size, 0);
    }

    #[test]
    fn prepare_plan_with_generic_instructions() {
        let plan = ExecutablePlan {
            instructions: vec![
                Instruction::Fill {
                    dst: BufferSlice::new(0, 4),
                    value: 0.0,
                },
                Instruction::MemCopy {
                    dst: BufferSlice::new(0, 4),
                    src: BufferSlice::new(4, 4),
                },
            ],
            arena_size: 2048,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.instructions.len(), 2);
        match &prepared.instructions[0] {
            PreparedInstruction::Generic { instruction_index } => {
                assert_eq!(*instruction_index, 0);
            }
            _ => panic!("expected Generic instruction"),
        }
        match &prepared.instructions[1] {
            PreparedInstruction::Generic { instruction_index } => {
                assert_eq!(*instruction_index, 1);
            }
            _ => panic!("expected Generic instruction"),
        }
        assert_eq!(prepared.arena_size, 2048);
    }

    #[test]
    fn conv2d_promoted_in_full_plan() {
        let conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 1.0,
        };
        let plan = ExecutablePlan {
            instructions: vec![conv, fill],
            arena_size: 4096,
            levels: vec![0, 0],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.instructions.len(), 2);
        assert!(matches!(
            &prepared.instructions[0],
            PreparedInstruction::Conv2d(_)
        ));
        assert!(matches!(
            &prepared.instructions[1],
            PreparedInstruction::Generic {
                instruction_index: 1
            }
        ));
    }

    // ── activation_from_kernel_name ────────────────────────────

    #[test]
    fn activation_from_kernel_name_none() {
        assert_eq!(
            activation_from_kernel_name("conv2d"),
            PreparedActivation::None
        );
    }

    #[test]
    fn activation_from_kernel_name_relu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_relu"),
            PreparedActivation::Relu
        );
        assert_eq!(
            activation_from_kernel_name("matmul_relu"),
            PreparedActivation::Relu
        );
    }

    #[test]
    fn activation_from_kernel_name_gelu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_gelu"),
            PreparedActivation::Gelu
        );
    }

    #[test]
    fn activation_from_kernel_name_silu() {
        assert_eq!(
            activation_from_kernel_name("conv2d_silu"),
            PreparedActivation::Silu
        );
    }

    #[test]
    fn activation_from_kernel_name_fallback() {
        assert_eq!(
            activation_from_kernel_name("conv2d_u4"),
            PreparedActivation::None
        );
        assert_eq!(
            activation_from_kernel_name("matmul"),
            PreparedActivation::None
        );
        assert_eq!(
            activation_from_kernel_name("unknown_kernel"),
            PreparedActivation::None
        );
    }

    // ── kernel_kind_from_kernel_name ───────────────────────────

    #[test]
    fn kernel_kind_im2col() {
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_relu"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
    }

    #[test]
    fn kernel_kind_quantized() {
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_u4"),
            PreparedConvKernelKind::FuturePackedU4
        );
        assert_eq!(
            kernel_kind_from_kernel_name("conv2d_u8"),
            PreparedConvKernelKind::FuturePackedI8
        );
    }

    #[test]
    fn kernel_kind_fallback() {
        assert_eq!(
            kernel_kind_from_kernel_name("matmul"),
            PreparedConvKernelKind::CurrentIm2colGemm
        );
    }

    // ── try_prepare_conv2d ─────────────────────────────────────

    #[test]
    fn try_prepare_conv2d_basic() {
        let inst = make_conv2d_instruction(
            "conv2d",
            Conv2dParams {
                stride: 2,
                padding: 1,
                c: 16,
                h: 32,
                w: 32,
                ..Default::default()
            },
        );
        let result = try_prepare_conv2d(&inst, 0).expect("should promote conv2d");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.node_id, Some(42usize));
                assert_eq!(c.stride, 2);
                assert_eq!(c.padding, 1);
                assert_eq!(c.dilation, 1);
                assert_eq!(c.groups, 1);
                assert_eq!(c.c, 16);
                assert_eq!(c.h, 32);
                assert_eq!(c.w, 32);
                assert_eq!(c.kh, 3);
                assert_eq!(c.kw, 3);
                assert_eq!(c.n, 1);
                assert_eq!(c.f, 8);
                assert_eq!(c.activation, PreparedActivation::None);
                assert_eq!(c.kernel, PreparedConvKernelKind::CurrentIm2colGemm);
                assert!(c.bias.is_none());
                assert_eq!(c.scratch_offset, 0);
                assert_eq!(c.scratch_len, 0);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_with_bias() {
        let inst = make_conv2d_instruction(
            "conv2d_relu",
            Conv2dParams {
                c: 8,
                h: 16,
                w: 16,
                kh: 1,
                kw: 1,
                with_bias: true,
                ..Default::default()
            },
        );
        let result = try_prepare_conv2d(&inst, 5).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                let bias = c.bias.expect("should have bias");
                assert_eq!(bias.offset, 8 * 16 * 16 * 4 + 8 * 8 * 4);
                assert_eq!(bias.size, 8 * 4);
                assert_eq!(c.activation, PreparedActivation::Relu);
                assert_eq!(c.kh, 1);
                assert_eq!(c.kw, 1);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_with_groups() {
        let inst = make_conv2d_instruction(
            "conv2d",
            Conv2dParams {
                groups: 4,
                c: 32,
                ..Default::default()
            },
        );
        let result = try_prepare_conv2d(&inst, 2).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.groups, 4);
                let c_per_group = c.c / c.groups;
                assert_eq!(c_per_group, 8);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_quantized_u4() {
        let inst = make_conv2d_instruction(
            "conv2d_u4",
            Conv2dParams {
                c: 16,
                ..Default::default()
            },
        );
        let result = try_prepare_conv2d(&inst, 0).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.kernel, PreparedConvKernelKind::FuturePackedU4);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_quantized_u8() {
        let inst = make_conv2d_instruction(
            "conv2d_u8",
            Conv2dParams {
                c: 16,
                ..Default::default()
            },
        );
        let result = try_prepare_conv2d(&inst, 0).expect("should promote");
        match &result {
            PreparedInstruction::Conv2d(c) => {
                assert_eq!(c.kernel, PreparedConvKernelKind::FuturePackedI8);
            }
            _ => panic!("expected Conv2d"),
        }
    }

    #[test]
    fn try_prepare_conv2d_not_conv2d_kernel() {
        let inst = Instruction::CallKernel {
            kernel_name: "matmul".to_string(),
            input_slices: vec![BufferSlice::new(0, 16), BufferSlice::new(16, 16)],
            output_slice: BufferSlice::new(32, 16),
            secondary_output_slice: None,
            params: vec![4, 4, 4],
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_not_call_kernel() {
        let inst = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 0.0,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_wrong_param_count() {
        let inst = Instruction::CallKernel {
            kernel_name: "conv2d".to_string(),
            input_slices: vec![BufferSlice::new(0, 64), BufferSlice::new(64, 64)],
            output_slice: BufferSlice::new(128, 64),
            secondary_output_slice: None,
            params: vec![1, 0, 1, 1], // only 4 params, need 9
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_conv2d_insufficient_inputs() {
        let inst = Instruction::CallKernel {
            kernel_name: "conv2d".to_string(),
            input_slices: vec![BufferSlice::new(0, 64)], // only 1 input, need >= 2
            output_slice: BufferSlice::new(64, 64),
            secondary_output_slice: None,
            params: vec![1, 0, 1, 1, 3, 8, 8, 3, 3],
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_conv2d(&inst, 0).is_none());
    }

    // ── PreparedActivation ─────────────────────────────────────

    #[test]
    fn prepared_activation_variants() {
        assert_eq!(PreparedActivation::None, PreparedActivation::None);
        assert_eq!(PreparedActivation::Relu, PreparedActivation::Relu);
        assert_ne!(PreparedActivation::Relu, PreparedActivation::Gelu);
    }

    #[test]
    fn prepared_activation_debug() {
        assert_eq!(format!("{:?}", PreparedActivation::Silu), "Silu");
    }

    // ── PackedWeightId ─────────────────────────────────────────

    #[test]
    fn packed_weight_id_equality() {
        let a = PackedWeightId::new(0);
        let b = PackedWeightId::new(0);
        let c = PackedWeightId::new(1);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn packed_weight_id_default_and_from() {
        let from: PackedWeightId = 7usize.into();
        assert_eq!(from.index, 7);
        assert_eq!(from.kind, PackedWeightKind::Fp32);
        assert!(from.arena_slot.is_none());

        let default = PackedWeightId::default();
        assert_eq!(default.index, 0);
        assert_eq!(default.kind, PackedWeightKind::Fp32);
        assert!(default.arena_slot.is_none());

        let with_kind = PackedWeightId::with_kind(3, PackedWeightKind::I8, Some(128));
        assert_eq!(with_kind.index, 3);
        assert_eq!(with_kind.kind, PackedWeightKind::I8);
        assert_eq!(with_kind.arena_slot, Some(128));
    }

    // ── PreparedConvKernelKind ─────────────────────────────────

    #[test]
    fn kernel_kind_debug() {
        assert_eq!(
            format!("{:?}", PreparedConvKernelKind::FuturePackedU4),
            "FuturePackedU4"
        );
    }

    #[test]
    fn kernel_kind_equality() {
        assert_eq!(
            PreparedConvKernelKind::CurrentIm2colGemm,
            PreparedConvKernelKind::CurrentIm2colGemm
        );
        assert_ne!(
            PreparedConvKernelKind::CurrentIm2colGemm,
            PreparedConvKernelKind::FuturePackedFp32
        );
    }

    // ── PreparedInstruction::instruction_index ─────────────────

    #[test]
    fn instruction_index_generic() {
        let inst = PreparedInstruction::Generic {
            instruction_index: 7,
        };
        assert_eq!(inst.instruction_index(), 7);
    }

    #[test]
    fn instruction_index_conv2d() {
        let inst = PreparedInstruction::Conv2d(PreparedConv2d {
            instruction_index: 3,
            node_id: None,
            input: BufferSlice::new(0, 4),
            weight: BufferSlice::new(4, 4),
            bias: None,
            output: BufferSlice::new(8, 4),
            n: 1,
            c: 1,
            h: 1,
            w: 1,
            f: 1,
            kh: 1,
            kw: 1,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            activation: PreparedActivation::None,
            kernel: PreparedConvKernelKind::CurrentIm2colGemm,
            packed_weight: None,
            scratch_offset: 0,
            scratch_len: 0,
        });
        assert_eq!(inst.instruction_index(), 3);
    }

    #[test]
    fn instruction_index_matmul() {
        let inst = PreparedInstruction::MatMul(PreparedMatMul {
            instruction_index: 5,
            node_id: None,
            a: BufferSlice::new(0, 4),
            b: BufferSlice::new(4, 4),
            bias: None,
            output: BufferSlice::new(8, 4),
            m: 1,
            k: 1,
            n: 1,
            activation: PreparedActivation::None,
            packed_weight: None,
        });
        assert_eq!(inst.instruction_index(), 5);
    }

    // ── PreparedExecutablePlan helpers ─────────────────────────

    #[test]
    fn plan_len_and_empty() {
        let empty = PreparedExecutablePlan {
            instructions: vec![],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());

        let one = PreparedExecutablePlan {
            instructions: vec![PreparedInstruction::Generic {
                instruction_index: 0,
            }],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(one.len(), 1);
        assert!(!one.is_empty());
    }

    #[test]
    fn plan_preserves_instruction_count_and_order() {
        let conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 1.0,
        };
        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(0, 4),
            src: BufferSlice::new(4, 4),
        };
        let plan = ExecutablePlan {
            instructions: vec![conv, fill, memcopy],
            arena_size: 4096,
            levels: vec![0, 0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.len(), 3);
        let order = prepared.original_instruction_order();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn conv2d_promoted_preserves_original_instruction_index() {
        let conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 1.0,
        };
        let plan = ExecutablePlan {
            instructions: vec![conv, fill],
            arena_size: 4096,
            levels: vec![0, 0],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.len(), 2);
        // Conv2d at original index 0
        assert_eq!(prepared.instructions[0].instruction_index(), 0);
        assert!(matches!(
            &prepared.instructions[0],
            PreparedInstruction::Conv2d(_)
        ));
        // Generic at original index 1
        assert_eq!(prepared.instructions[1].instruction_index(), 1);
        assert!(matches!(
            &prepared.instructions[1],
            PreparedInstruction::Generic {
                instruction_index: 1
            }
        ));
    }

    #[test]
    fn generic_fallback_preserves_instruction_index() {
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 0.0,
        };
        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(0, 4),
            src: BufferSlice::new(4, 4),
        };
        let plan = ExecutablePlan {
            instructions: vec![fill, memcopy],
            arena_size: 1024,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        let order = prepared.original_instruction_order();
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn generic_instruction_indices_only_returns_generics() {
        let conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        let fill = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 0.0,
        };
        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(0, 4),
            src: BufferSlice::new(4, 4),
        };
        let plan = ExecutablePlan {
            instructions: vec![conv, fill, memcopy],
            arena_size: 4096,
            levels: vec![0, 0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        let generic_indices = prepared.generic_instruction_indices();
        // Only fill (index 1) and memcopy (index 2) are generic
        assert_eq!(generic_indices, vec![1, 2]);
    }

    // ── PreparedConstantArena ──────────────────────────────────

    #[test]
    fn arena_insert_lookup() {
        let mut arena = PreparedConstantArena::new();
        assert!(arena.is_empty());

        let weight_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let weight_id = arena.insert("conv_weight", weight_data.clone());

        assert_eq!(weight_id.index, 0);
        assert_eq!(weight_id.kind, PackedWeightKind::Fp32);
        assert!(weight_id.arena_slot.is_none());
        assert_eq!(arena.len(), 1);
        assert!(!arena.is_empty());

        // get() round-trips the payload
        let view = arena.get(weight_id).expect("registered slot must resolve");
        assert_eq!(view, weight_data.as_slice());

        // id_for() resolves the name back to the same handle
        assert_eq!(arena.id_for("conv_weight"), Some(weight_id));
        assert_eq!(arena.id_for("not_registered"), None);

        // entry() carries the full metadata
        let entry = arena.entry(weight_id).expect("entry must exist");
        assert_eq!(entry.name, "conv_weight");
        assert_eq!(entry.numel, 4);
        assert_eq!(entry.byte_len, 4 * std::mem::size_of::<f32>());
        assert_eq!(entry.kind, PackedWeightKind::Fp32);
        assert_eq!(entry.store.kind_name(), "unpacked");

        // A second distinct name produces a fresh id at the next slot
        let bias_id = arena.insert("conv_bias", vec![0.5_f32, 0.25]);
        assert_eq!(bias_id.index, 1);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(bias_id), Some([0.5_f32, 0.25].as_slice()));

        // Iteration yields entries in insertion order
        let names: Vec<&str> = arena.entries().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["conv_weight", "conv_bias"]);
    }

    #[test]
    fn arena_duplicate_name_reuses_slot() {
        let mut arena = PreparedConstantArena::new();
        let first = arena.insert("matmul_weight", vec![1.0_f32, 2.0, 3.0]);
        let second = arena.insert("matmul_weight", vec![9.0_f32, 9.0, 9.0, 9.0]);

        // Same name returns the same id without growing storage.
        assert_eq!(first, second);
        assert_eq!(arena.len(), 1);

        // The stored payload is the original — duplicate inserts never
        // overwrite or duplicate data.
        let view = arena.get(first).expect("entry must exist");
        assert_eq!(view, [1.0_f32, 2.0, 3.0].as_slice());
        assert_eq!(arena.id_for("matmul_weight"), Some(first));
    }

    #[test]
    fn arena_len_is_empty_and_total_bytes() {
        let mut arena = PreparedConstantArena::new();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert_eq!(arena.total_bytes(), 0);

        arena.insert("a", vec![0.0_f32; 4]); // 16 bytes
        assert_eq!(arena.len(), 1);
        assert!(!arena.is_empty());
        assert_eq!(arena.total_bytes(), 4 * std::mem::size_of::<f32>());

        arena.insert("b", vec![0.0_f32; 10]); // +40 bytes
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.total_bytes(), 14 * std::mem::size_of::<f32>());

        // Duplicate name does not change byte totals.
        arena.insert("a", vec![0.0_f32; 32]);
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.total_bytes(), 14 * std::mem::size_of::<f32>());
    }

    #[test]
    fn arena_get_returns_none_for_unknown_id() {
        let arena = PreparedConstantArena::new();
        assert!(arena.get(PackedWeightId::new(42)).is_none());
        assert!(arena.entry(PackedWeightId::new(42)).is_none());
    }

    // ── PackedWeightStore ──────────────────────────────────────

    #[test]
    fn packed_weight_unpacked_view() {
        let payload = vec![1.0_f32, 2.0, 3.0];
        let store = PackedWeightStore::unpacked(payload.clone());

        let view = store.as_f32_slice().expect("unpacked must expose a view");
        assert_eq!(view, payload.as_slice());
        assert_eq!(store.kind_name(), "unpacked");

        let reserved = PackedWeightStore::Reserved;
        assert!(reserved.as_f32_slice().is_none());
        assert_eq!(reserved.kind_name(), "reserved");
    }

    // ── PackedWeightKind ───────────────────────────────────────

    #[test]
    fn packed_weight_kind_name() {
        assert_eq!(PackedWeightKind::Fp32.name(), "fp32");
        assert_eq!(PackedWeightKind::I8.name(), "i8");
        assert_eq!(PackedWeightKind::U4.name(), "u4");
        assert_eq!(PackedWeightKind::Nf4.name(), "nf4");
    }

    #[test]
    fn packed_weight_kind_default_is_fp32() {
        assert_eq!(PackedWeightKind::default(), PackedWeightKind::Fp32);
        assert_eq!(
            PackedWeightKind::Fp32.element_bytes(),
            std::mem::size_of::<f32>()
        );
        assert_eq!(PackedWeightKind::I8.element_bytes(), 1);
    }

    // ── PreparedExecutablePlan::register_constant_arena ───────

    #[test]
    fn plan_holds_constant_arena() {
        let mut prepared = prepare_executable_plan(&empty_plan());
        // No arena is attached by default.
        assert!(prepared.constant_arena().is_none());

        let mut arena = PreparedConstantArena::new();
        let id = arena.insert("conv_weight", vec![1.0_f32, 2.0, 3.0]);
        prepared.register_constant_arena(arena);

        let attached = prepared
            .constant_arena()
            .expect("arena should be attached after register_constant_arena");
        assert_eq!(attached.len(), 1);
        assert_eq!(attached.id_for("conv_weight"), Some(id));
        assert_eq!(
            attached.get(id),
            Some([1.0_f32, 2.0, 3.0].as_slice()),
            "registered payload must be readable through the plan"
        );

        // Replacing the arena swaps the attached metadata wholesale.
        let mut replacement = PreparedConstantArena::new();
        replacement.insert("matmul_weight", vec![0.0_f32, 0.0]);
        prepared.register_constant_arena(replacement);
        let after = prepared
            .constant_arena()
            .expect("replacement arena should still be attached");
        assert_eq!(after.len(), 1);
        assert!(after.id_for("conv_weight").is_none());
        assert!(after.id_for("matmul_weight").is_some());
    }

    #[test]
    fn prepare_executable_plan_starts_without_constant_arena() {
        let prepared = prepare_executable_plan(&empty_plan());
        assert!(prepared.constant_arena().is_none());
    }
}
