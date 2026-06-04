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

/// One static-weight binding produced by [`detect_static_weights`].
///
/// Each binding points a single consumer instruction's input slot at a
/// [`PackedWeightId`] inside the attached [`PreparedConstantArena`].
/// Runtime dispatch is unchanged — bindings are metadata-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticWeightBinding {
    /// Index of the consumer instruction (`Conv2d` / `MatMul`) in the
    /// source [`ExecutablePlan`].
    pub instruction_index: usize,
    /// Index of the input slot on the consumer instruction
    /// (`1` = weight, `2` = bias for conv / fused-matmul).
    pub input_index: usize,
    /// Handle for the constant bytes inside the
    /// [`PreparedConstantArena`].
    pub weight_id: PackedWeightId,
    /// Dtype tag for the slot. Always [`PackedWeightKind::Fp32`] in this
    /// skeleton.
    pub kind: PackedWeightKind,
}

/// Map of detected static-weight bindings keyed by the consumer
/// instruction index in the source plan. One consumer may have multiple
/// bindings (weight + bias); the order matches consumer input order.
pub type StaticWeightMap = Vec<StaticWeightBinding>;

// ── Internal helpers for the detection pass ──────────────────

/// Snapshot of one [`Instruction::WriteConst`] in the plan. Used as an
/// intermediate representation while we match consumer input slots
/// against static producers.
#[derive(Clone, Debug)]
struct WriteConstEntry {
    /// Plan-wide index of the WriteConst instruction.
    writer_idx: usize,
    /// Destination slot the WriteConst materialises.
    dst: BufferSlice,
    /// Raw bytes the WriteConst deposits into the arena.
    data: Vec<u8>,
}

/// Map a kernel name suffix to the corresponding [`PreparedActivation`].
///
/// Recognized names: `"conv2d"` (None), `"conv2d_relu"`, `"conv2d_gelu"`,
/// `"conv2d_silu"`; `"matmul_relu"`, `"matmul_gelu"`, `"matmul_silu"`; and
/// the `"fused_matmul_add_*"` family (`_relu` / `_gelu` / `_silu`).
/// Everything else returns `None`.
pub fn activation_from_kernel_name(kernel_name: &str) -> PreparedActivation {
    match kernel_name {
        "conv2d_relu" | "matmul_relu" | "fused_matmul_add_relu" => PreparedActivation::Relu,
        "conv2d_gelu" | "matmul_gelu" | "fused_matmul_add_gelu" => PreparedActivation::Gelu,
        "conv2d_silu" | "matmul_silu" | "fused_matmul_add_silu" => PreparedActivation::Silu,
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

/// `true` when `name` is a matmul-family kernel that the prepared plan
/// should promote to [`PreparedInstruction::MatMul`].
///
/// Covers the plain `"matmul"` kernel, the activation-fused
/// `"matmul_relu" | "matmul_gelu" | "matmul_silu"` variants, the
/// `"fused_matmul_add_*"` family (which carry a 3rd bias input), and
/// the quantized `"matmul_u4" | "matmul_u4_i8" | "matmul_u8" |
/// "matmul_u8_i8"` variants. Quantized matmuls still keep the same
/// `[m, k, n]` param layout, so detection promotion is uniform.
pub fn is_matmul_kernel_name(kernel_name: &str) -> bool {
    kernel_name == "matmul"
        || kernel_name.starts_with("matmul_")
        || kernel_name.starts_with("fused_matmul_add_")
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

/// Attempt to promote an [`Instruction`] into a [`PreparedInstruction::MatMul`].
///
/// Returns `Some(PreparedInstruction::MatMul(...))` only when:
/// - The instruction is a `CallKernel` whose `kernel_name` is recognised
///   by [`is_matmul_kernel_name`].
/// - `params` has exactly 3 elements: `[m, k, n]` (uniform across the
///   plain, activation-fused, and quantized matmul kernels).
/// - At least two input slices are present (`A` + `B`).
///
/// Returns `None` for any instruction that does not satisfy these
/// invariants; callers should fall back to `PreparedInstruction::Generic`.
///
/// # Required metadata fields from `CallKernel`
/// - `kernel_name`: must be a matmul-family name.
/// - `params`: `[m, k, n]`. Concrete element counts baked in by the CPU
///   backend; symbolic dims are resolved through `param_dims` at runtime.
/// - `input_slices[0]`: left operand `A`.
/// - `input_slices[1]`: right operand `B` / weight tensor.
/// - `input_slices[2]` (optional): bias — only present for
///   `"fused_matmul_add_*"` kernels.
/// - `output_slice`: output tensor.
/// - `node_id`: optional node identifier for runtime param tightening.
pub fn try_prepare_matmul(
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

    if !is_matmul_kernel_name(kernel_name) {
        return None;
    }

    if params.len() != 3 {
        return None;
    }
    let m = params[0];
    let k = params[1];
    let n = params[2];

    if input_slices.len() < 2 {
        return None;
    }

    let a = input_slices[0];
    let b = input_slices[1];
    let bias = input_slices.get(2).copied();

    let activation = activation_from_kernel_name(kernel_name);

    Some(PreparedInstruction::MatMul(PreparedMatMul {
        instruction_index,
        node_id: *node_id,
        a,
        b,
        bias,
        output: *output_slice,
        m,
        k,
        n,
        activation,
        packed_weight: None,
    }))
}

/// Walk the [`ExecutablePlan`] and bind every `Conv2d` / `MatMul`
/// input slot that is fed by an [`Instruction::WriteConst`] producer
/// into the supplied [`PreparedConstantArena`].
///
/// Returns a [`StaticWeightMap`] of `(instruction_index, input_index,
/// weight_id, kind)` bindings. The arena is populated as a side
/// effect: each WriteConst producer registered in the plan gets exactly
/// one slot in the arena, even when multiple consumers reference it.
///
/// # Detection rules
///
/// 1. The plan is scanned for `WriteConst { dst, data }` instructions.
///    Each non-empty `WriteConst` becomes a candidate producer.
/// 2. For every `CallKernel` whose name starts with `"conv2d"` or
///    matches a matmul-family name (see
///    [`is_matmul_kernel_name`]), the consumer's `input_slices[1]`
///    (weight) and — when present — `input_slices[2]` (bias) are
///    matched against producer ranges.
/// 3. A producer "covers" a consumer slot when the producer's
///    destination `BufferSlice` is byte-equal to the consumer slot, or
///    strictly contains it (`wc.offset <= slot.offset` and
///    `wc.offset + wc.data.len() >= slot.offset + slot.size`).
/// 4. Bindings are deduplicated at the arena level by producer
///    identity: the same `WriteConst` slot is registered once under a
///    stable name and every consumer that points at it shares the
///    resulting [`PackedWeightId`].
///
/// # Runtime impact
///
/// **None.** Bindings and arena entries are metadata only. The
/// original `WriteConst` / `CallKernel` instructions remain in the
/// [`ExecutablePlan`] untouched, so `GraphExecutor::execute` keeps its
/// byte-identical behaviour.
pub fn detect_static_weights(
    plan: &ExecutablePlan,
    arena: &mut PreparedConstantArena,
) -> StaticWeightMap {
    // First pass: collect every WriteConst in the plan.
    let mut const_slots: Vec<WriteConstEntry> = Vec::new();
    for (writer_idx, inst) in plan.instructions.iter().enumerate() {
        if let Instruction::WriteConst { dst, data } = inst {
            if !data.is_empty() {
                const_slots.push(WriteConstEntry {
                    writer_idx,
                    dst: *dst,
                    data: data.clone(),
                });
            }
        }
    }

    // Second pass: walk Conv2d / MatMul consumers and bind their
    // weight + optional bias input slots.
    let mut bindings: StaticWeightMap = Vec::new();
    for (instruction_index, inst) in plan.instructions.iter().enumerate() {
        let (input_slices, role) = match inst {
            Instruction::CallKernel {
                kernel_name,
                input_slices,
                ..
            } if kernel_name.starts_with("conv2d") => (input_slices.as_slice(), "conv"),
            Instruction::CallKernel {
                kernel_name,
                input_slices,
                ..
            } if is_matmul_kernel_name(kernel_name) => (input_slices.as_slice(), "matmul"),
            _ => continue,
        };

        // input_slices[1] is the weight tensor for both conv2d and
        // matmul-family kernels.
        if let Some(weight_slice) = input_slices.get(1) {
            if let Some(binding) = bind_consumer_input(
                &const_slots,
                arena,
                instruction_index,
                1,
                weight_slice,
                role,
            ) {
                bindings.push(binding);
            }
        }

        // input_slices[2] is the bias slot — present for conv2d (always
        // when the plan provides one) and for "fused_matmul_add_*"
        // matmul kernels. For plain "matmul" the slot is absent and
        // this branch is a no-op.
        if let Some(bias_slice) = input_slices.get(2) {
            if let Some(binding) =
                bind_consumer_input(&const_slots, arena, instruction_index, 2, bias_slice, role)
            {
                bindings.push(binding);
            }
        }
    }

    bindings
}

/// Look up a `WriteConst` covering `slot`, register it in `arena` under
/// a stable producer-based name, and return the resulting
/// [`StaticWeightBinding`]. Returns `None` when no `WriteConst` covers
/// the slot — the consumer is treated as dynamic in that case.
fn bind_consumer_input(
    const_slots: &[WriteConstEntry],
    arena: &mut PreparedConstantArena,
    instruction_index: usize,
    input_index: usize,
    slot: &BufferSlice,
    role: &str,
) -> Option<StaticWeightBinding> {
    let producer = find_covering_write_const(const_slots, slot)?;
    let name = format!("{}_{}_i{}", role, producer.writer_idx, input_index);
    let payload = bytes_to_f32_vec(&producer.data);
    let weight_id = arena.insert(&name, payload);
    Some(StaticWeightBinding {
        instruction_index,
        input_index,
        weight_id,
        kind: PackedWeightKind::Fp32,
    })
}

/// Return the first `WriteConst` whose destination either equals
/// `slot` (byte-exact) or strictly covers it. Order of producers
/// follows plan order, so a tie-break favours the earliest writer.
fn find_covering_write_const<'a>(
    const_slots: &'a [WriteConstEntry],
    slot: &BufferSlice,
) -> Option<&'a WriteConstEntry> {
    const_slots
        .iter()
        .find(|entry| covers_slice(&entry.dst, entry.data.len(), slot))
}

/// Producer range covers consumer slot when offsets line up and the
/// producer's bytes fully enclose the consumer slot.
fn covers_slice(producer: &BufferSlice, producer_bytes: usize, slot: &BufferSlice) -> bool {
    let producer_end = producer.offset.saturating_add(producer_bytes);
    let slot_end = slot.offset.saturating_add(slot.size);
    producer.offset <= slot.offset && producer_end >= slot_end && producer_bytes > 0
}

/// Interpret `bytes` as a little-endian `f32` payload. Trailing bytes
/// that do not form a complete `f32` are dropped — `WriteConst` always
/// carries whole-element payloads in practice.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            // chunks_exact guarantees exactly 4 bytes per chunk.
            let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// Apply a slice of [`StaticWeightBinding`]s to a prepared instruction,
/// setting `packed_weight` on the corresponding `Conv2d` / `MatMul`
/// from the binding whose `input_index == 1` (the weight slot). Bias
/// bindings (`input_index == 2`) are recorded in the returned map but
/// the prepared structs do not yet expose a `packed_bias` field — a
/// later lane will plumb that into the dispatch path.
fn apply_static_weight_bindings(
    instruction: PreparedInstruction,
    bindings: &[&StaticWeightBinding],
) -> PreparedInstruction {
    let weight_binding = bindings.iter().find(|b| b.input_index == 1).copied();
    match (instruction, weight_binding) {
        (PreparedInstruction::Conv2d(mut c), Some(b)) => {
            c.packed_weight = Some(b.weight_id);
            PreparedInstruction::Conv2d(c)
        }
        (PreparedInstruction::MatMul(mut m), Some(b)) => {
            m.packed_weight = Some(b.weight_id);
            PreparedInstruction::MatMul(m)
        }
        (other, _) => other,
    }
}

/// Validate that `prepared` is consistent with `plan` for use as the
/// driver of an opt-in prepared-execution fallback path.
///
/// A prepared plan is "consistent" with an [`ExecutablePlan`] when:
///
/// - the prepared plan has the same number of instructions as the
///   source plan, and
/// - [`PreparedExecutablePlan::original_instruction_order`] reports
///   the identity permutation `[0, 1, 2, ..., N-1]`, i.e. the prepared
///   instructions appear in the same plan order and carry their
///   original instruction index.
///
/// This is a cheap O(N) sanity check. It is intended to be called from
/// the opt-in prepared-fallback execution path before delegating to
/// the existing dispatch loop: a failure here means the caller handed
/// in a prepared plan that was either built from a different
/// [`ExecutablePlan`] or has been mutated in ways the fallback cannot
/// reconcile, and the safe behaviour is to refuse rather than silently
/// desynchronise prepared metadata with the live plan.
///
/// The check is metadata-only — it does **not** look at
/// [`PreparedConstantArena`] contents, weight bindings, or any
/// per-instruction payload. Future lanes that add richer prepared
/// execution semantics will layer additional validation on top of
/// this baseline.
pub fn validate_prepared_against_plan(
    prepared: &PreparedExecutablePlan,
    plan: &ExecutablePlan,
) -> Result<(), crate::backend::BackendError> {
    use crate::backend::BackendError;
    let order = prepared.original_instruction_order();
    if order.len() != plan.instructions.len() {
        return Err(BackendError::Dispatch(format!(
            "prepared fallback: prepared plan has {} instructions, plan has {}",
            order.len(),
            plan.instructions.len(),
        )));
    }
    for (i, &idx) in order.iter().enumerate() {
        if idx != i {
            return Err(BackendError::Dispatch(format!(
                "prepared fallback: order mismatch at slot {}: prepared says {}, expected {}",
                i, idx, i
            )));
        }
    }
    Ok(())
}

/// Build a [`PreparedExecutablePlan`] by inspecting each instruction.
///
/// Statically recognizable Conv2d and MatMul `CallKernel` instructions
/// are promoted to `PreparedInstruction::Conv2d` /
/// `PreparedInstruction::MatMul`; everything else becomes
/// `PreparedInstruction::Generic`. This preserves runtime semantics
/// while enabling future kernel-specialization paths.
///
/// The build also runs [`detect_static_weights`]: any Conv2d / MatMul
/// input fed by a [`Instruction::WriteConst`] producer is bound to a
/// [`PackedWeightId`] inside an attached [`PreparedConstantArena`]. The
/// arena is created internally and attached via
/// [`PreparedExecutablePlan::register_constant_arena`]; both the arena
/// and the bindings are metadata-only.
pub fn prepare_executable_plan(plan: &ExecutablePlan) -> PreparedExecutablePlan {
    let mut arena = PreparedConstantArena::new();
    let bindings = detect_static_weights(plan, &mut arena);

    // Bucket bindings by consumer instruction index so the
    // per-instruction pass below can locate its bindings in O(1).
    let mut by_instr: HashMap<usize, Vec<&StaticWeightBinding>> = HashMap::new();
    for binding in &bindings {
        by_instr
            .entry(binding.instruction_index)
            .or_default()
            .push(binding);
    }

    let instructions: Vec<PreparedInstruction> = plan
        .instructions
        .iter()
        .enumerate()
        .map(|(instruction_index, inst)| {
            let empty: Vec<&StaticWeightBinding> = Vec::new();
            let inst_bindings = by_instr
                .get(&instruction_index)
                .map(|v| v.iter().copied().collect::<Vec<_>>())
                .unwrap_or(empty);
            let prepared = try_prepare_conv2d(inst, instruction_index)
                .or_else(|| try_prepare_matmul(inst, instruction_index))
                .unwrap_or(PreparedInstruction::Generic { instruction_index });
            apply_static_weight_bindings(prepared, &inst_bindings)
        })
        .collect();

    let mut prepared = PreparedExecutablePlan {
        instructions,
        arena_size: plan.arena_size,
        scratch_size: 0,
        constant_arena: None,
    };
    prepared.register_constant_arena(arena);
    prepared
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

    /// Total number of static-weight bindings actually recorded on the
    /// prepared instructions in this plan.
    ///
    /// The current wave-4 representation stores at most one
    /// [`PackedWeightId`] per prepared consumer — the weight slot on
    /// [`PreparedConv2d::packed_weight`] /
    /// [`PreparedMatMul::packed_weight`]. Bias bindings produced by
    /// [`detect_static_weights`] are still metadata-only and not
    /// reflected in the prepared structs, so they are not counted
    /// here. The count is therefore the number of `Conv2d` / `MatMul`
    /// instructions whose `packed_weight` slot is `Some(...)`.
    ///
    /// Returns `0` when the plan carries no promoted conv/matmul
    /// instructions or when none of them were bound to a static
    /// constant.
    pub fn static_weight_binding_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|inst| match inst {
                PreparedInstruction::Conv2d(c) => c.packed_weight.is_some(),
                PreparedInstruction::MatMul(m) => m.packed_weight.is_some(),
                PreparedInstruction::Generic { .. } => false,
            })
            .count()
    }

    /// Number of entries in the attached [`PreparedConstantArena`].
    ///
    /// Returns `0` when no arena has been registered yet (e.g. a
    /// hand-built [`PreparedExecutablePlan`] that never called
    /// [`PreparedExecutablePlan::register_constant_arena`]).
    pub fn constant_arena_entry_count(&self) -> usize {
        self.constant_arena
            .as_ref()
            .map(|arena| arena.len())
            .unwrap_or(0)
    }

    /// Total payload size, in bytes, of the attached
    /// [`PreparedConstantArena`].
    ///
    /// Mirrors [`PreparedConstantArena::total_bytes`]: sums
    /// [`PreparedConstantEntry::byte_len`] across all entries. Returns
    /// `0` when no arena is attached.
    pub fn constant_arena_total_bytes(&self) -> usize {
        self.constant_arena
            .as_ref()
            .map(|arena| arena.total_bytes())
            .unwrap_or(0)
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

    /// Helper: build a matmul CallKernel instruction with the canonical
    /// `[m, k, n]` param layout. Use `with_bias=true` to mimic the
    /// `fused_matmul_add_*` family (3 inputs, kernel name prefix
    /// `fused_matmul_add_`).
    fn make_matmul_instruction(
        kernel_name: &str,
        m: usize,
        k: usize,
        n: usize,
        with_bias: bool,
    ) -> Instruction {
        let a_size = m * k * 4;
        let b_size = k * n * 4;
        let bias_size = if with_bias { n * 4 } else { 0 };

        let mut input_slices = vec![
            BufferSlice::new(0, a_size),
            BufferSlice::new(a_size, b_size),
        ];
        if with_bias {
            input_slices.push(BufferSlice::new(a_size + b_size, bias_size));
        }

        Instruction::CallKernel {
            kernel_name: kernel_name.to_string(),
            input_slices,
            output_slice: BufferSlice::new(a_size + b_size + bias_size, m * n * 4),
            secondary_output_slice: None,
            params: vec![m, k, n],
            node_id: Some(7usize),
            param_dims: None,
            weight_meta: None,
        }
    }

    /// Build a WriteConst that writes `data` (interpreted as little-endian
    /// f32) into the given `BufferSlice`.
    fn make_write_const(dst: BufferSlice, f32_payload: &[f32]) -> Instruction {
        let data: Vec<u8> = f32_payload.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(data.len(), dst.size, "write_const payload size mismatch");
        Instruction::WriteConst { dst, data }
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

    // ── try_prepare_matmul ─────────────────────────────────────

    #[test]
    fn try_prepare_matmul_basic() {
        let inst = make_matmul_instruction("matmul", 4, 8, 16, false);
        let result = try_prepare_matmul(&inst, 0).expect("should promote matmul");
        match &result {
            PreparedInstruction::MatMul(m) => {
                assert_eq!(m.node_id, Some(7usize));
                assert_eq!(m.m, 4);
                assert_eq!(m.k, 8);
                assert_eq!(m.n, 16);
                assert_eq!(m.activation, PreparedActivation::None);
                assert!(m.bias.is_none());
                assert!(m.packed_weight.is_none());
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn try_prepare_matmul_with_activation() {
        let inst = make_matmul_instruction("matmul_relu", 2, 3, 5, false);
        let result = try_prepare_matmul(&inst, 0).expect("should promote");
        match &result {
            PreparedInstruction::MatMul(m) => {
                assert_eq!(m.activation, PreparedActivation::Relu);
                assert_eq!(m.m, 2);
                assert_eq!(m.n, 5);
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn try_prepare_matmul_fused_with_bias() {
        let inst = make_matmul_instruction("fused_matmul_add_relu", 4, 8, 16, true);
        let result = try_prepare_matmul(&inst, 3).expect("should promote");
        match &result {
            PreparedInstruction::MatMul(m) => {
                let bias = m.bias.expect("fused kernel must carry a bias slot");
                assert_eq!(bias.size, 16 * 4);
                assert_eq!(m.activation, PreparedActivation::Relu);
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn try_prepare_matmul_quantized() {
        let inst = make_matmul_instruction("matmul_u4", 4, 8, 16, false);
        let result = try_prepare_matmul(&inst, 0).expect("should promote quantized");
        match &result {
            PreparedInstruction::MatMul(m) => {
                assert_eq!(m.m, 4);
                assert_eq!(m.k, 8);
                assert_eq!(m.n, 16);
            }
            _ => panic!("expected MatMul"),
        }
    }

    #[test]
    fn try_prepare_matmul_wrong_param_count() {
        let inst = Instruction::CallKernel {
            kernel_name: "matmul".to_string(),
            input_slices: vec![BufferSlice::new(0, 16), BufferSlice::new(16, 16)],
            output_slice: BufferSlice::new(32, 16),
            secondary_output_slice: None,
            params: vec![4, 4], // need 3
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_matmul(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_matmul_insufficient_inputs() {
        let inst = Instruction::CallKernel {
            kernel_name: "matmul".to_string(),
            input_slices: vec![BufferSlice::new(0, 16)],
            output_slice: BufferSlice::new(16, 16),
            secondary_output_slice: None,
            params: vec![4, 4, 4],
            node_id: None,
            param_dims: None,
            weight_meta: None,
        };
        assert!(try_prepare_matmul(&inst, 0).is_none());
    }

    #[test]
    fn try_prepare_matmul_not_call_kernel() {
        let inst = Instruction::Fill {
            dst: BufferSlice::new(0, 4),
            value: 0.0,
        };
        assert!(try_prepare_matmul(&inst, 0).is_none());
    }

    #[test]
    fn is_matmul_kernel_name_recognises_family() {
        assert!(is_matmul_kernel_name("matmul"));
        assert!(is_matmul_kernel_name("matmul_relu"));
        assert!(is_matmul_kernel_name("matmul_gelu"));
        assert!(is_matmul_kernel_name("matmul_silu"));
        assert!(is_matmul_kernel_name("matmul_u4"));
        assert!(is_matmul_kernel_name("matmul_u4_i8"));
        assert!(is_matmul_kernel_name("matmul_u8"));
        assert!(is_matmul_kernel_name("matmul_u8_i8"));
        assert!(is_matmul_kernel_name("fused_matmul_add_relu"));
        assert!(is_matmul_kernel_name("fused_matmul_add_gelu"));
        assert!(is_matmul_kernel_name("fused_matmul_add_silu"));
        assert!(!is_matmul_kernel_name("conv2d"));
        assert!(!is_matmul_kernel_name("add_f32"));
        assert!(!is_matmul_kernel_name(""));
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
        // prepare_executable_plan always attaches an (initially empty)
        // arena so callers can introspect the static-weight bindings
        // without a follow-up registration.
        assert!(prepared.constant_arena().is_some());
        assert_eq!(
            prepared
                .constant_arena()
                .map(|a| a.len())
                .unwrap_or_default(),
            0
        );

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
    fn prepare_executable_plan_attaches_empty_arena_by_default() {
        let prepared = prepare_executable_plan(&empty_plan());
        // The static-weight detection pass always populates an arena,
        // even when no Conv2d / MatMul was found. The arena is just
        // empty in that case.
        let arena = prepared
            .constant_arena()
            .expect("plan should carry an arena, even if empty");
        assert!(arena.is_empty());
    }

    // ── detect_static_weights: required tests ─────────────────

    /// Required test: a `Conv2d` whose weight input is fed by a
    /// `WriteConst` gets a `packed_weight` binding pointing at the
    /// matching arena slot.
    #[test]
    fn detects_conv_weight_from_write_const() {
        // WriteConst writes 8 f32 weights (32 bytes) at offset 1024.
        let weight_payload: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let write_const = make_write_const(BufferSlice::new(1024, 32), &weight_payload);

        // Conv2d whose weight input exactly matches the WriteConst.
        // input_size is 3*8*8*4 = 768, but we set the conv to read
        // its input from a separate region. We only need the weight
        // slice to match.
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(1024, 32);
        } else {
            panic!("expected CallKernel");
        }

        let plan = ExecutablePlan {
            instructions: vec![write_const, conv],
            arena_size: 4096,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        // The conv at plan index 1 must carry a packed_weight binding.
        match &prepared.instructions[1] {
            PreparedInstruction::Conv2d(c) => {
                let id = c
                    .packed_weight
                    .expect("conv2d fed by write_const must have packed_weight");
                assert_eq!(id.kind, PackedWeightKind::Fp32);
                assert_eq!(id.index, 0);

                // The arena should expose the same bytes under the
                // expected stable name.
                let arena = prepared
                    .constant_arena()
                    .expect("plan should have an attached arena");
                assert_eq!(arena.len(), 1);
                assert_eq!(arena.get(id), Some(weight_payload.as_slice()));
                assert_eq!(arena.id_for("conv_0_i1"), Some(id));
            }
            other => panic!("expected Conv2d at index 1, got {other:?}"),
        }
    }

    /// Required test: a `MatMul` whose weight (B) input is fed by a
    /// `WriteConst` gets a `packed_weight` binding pointing at the
    /// matching arena slot.
    #[test]
    fn detects_matmul_weight_from_write_const() {
        // WriteConst writes a 2x3 weight tensor (6 f32 = 24 bytes)
        // starting at offset 256.
        let weight_payload: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let write_const = make_write_const(BufferSlice::new(256, 24), &weight_payload);

        // Matmul: M=4, K=3, N=2 — needs B tensor of size K*N = 6 f32
        // at the same offset 256.
        let mut mm = make_matmul_instruction("matmul", 4, 3, 2, false);
        if let Instruction::CallKernel { input_slices, .. } = &mut mm {
            input_slices[0] = BufferSlice::new(0, 4 * 3 * 4);
            input_slices[1] = BufferSlice::new(256, 24);
        } else {
            panic!("expected CallKernel");
        }

        let plan = ExecutablePlan {
            instructions: vec![write_const, mm],
            arena_size: 4096,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        match &prepared.instructions[1] {
            PreparedInstruction::MatMul(m) => {
                let id = m
                    .packed_weight
                    .expect("matmul fed by write_const must have packed_weight");
                assert_eq!(id.kind, PackedWeightKind::Fp32);
                assert_eq!(id.index, 0);

                let arena = prepared
                    .constant_arena()
                    .expect("plan should have an attached arena");
                assert_eq!(arena.len(), 1);
                assert_eq!(arena.get(id), Some(weight_payload.as_slice()));
                assert_eq!(arena.id_for("matmul_0_i1"), Some(id));
            }
            other => panic!("expected MatMul at index 1, got {other:?}"),
        }
    }

    /// Required test: a Conv2d whose weight input is NOT backed by a
    /// WriteConst (e.g. it shares an arena slot with a MemCopy-fed
    /// dynamic tensor) gets no `packed_weight` binding.
    #[test]
    fn skips_dynamic_input() {
        // A MemCopy into the weight slot — no WriteConst, so the
        // weight is dynamic. The conv reads from the same slot.
        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(512, 32),
            src: BufferSlice::new(0, 32),
        };
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(512, 32);
        } else {
            panic!("expected CallKernel");
        }

        let plan = ExecutablePlan {
            instructions: vec![memcopy, conv],
            arena_size: 4096,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        match &prepared.instructions[1] {
            PreparedInstruction::Conv2d(c) => {
                assert!(
                    c.packed_weight.is_none(),
                    "conv with dynamic weight must not be bound"
                );
            }
            other => panic!("expected Conv2d at index 1, got {other:?}"),
        }

        // No arena slots should have been allocated.
        let arena = prepared
            .constant_arena()
            .expect("plan still gets an arena, just an empty one");
        assert_eq!(arena.len(), 0);
    }

    /// Required test: when two Conv2d consumers share the same
    /// `WriteConst` producer, only one arena slot is allocated and
    /// both consumers' bindings point at the same `PackedWeightId`.
    #[test]
    fn arena_reused_for_duplicate_weight() {
        let weight_payload: Vec<f32> = (0..8).map(|i| i as f32 * 0.25).collect();
        let write_const = make_write_const(BufferSlice::new(128, 32), &weight_payload);

        // Two convs, both reading their weight from the same offset.
        let mut conv_a = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv_a {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(128, 32);
        } else {
            panic!("expected CallKernel");
        }
        let mut conv_b = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv_b {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(128, 32);
        } else {
            panic!("expected CallKernel");
        }

        let plan = ExecutablePlan {
            instructions: vec![write_const, conv_a, conv_b],
            arena_size: 4096,
            levels: vec![0, 1, 2],
        };
        let prepared = prepare_executable_plan(&plan);

        let arena = prepared
            .constant_arena()
            .expect("plan should have an attached arena");
        // Single WriteConst → single arena slot.
        assert_eq!(arena.len(), 1, "shared WriteConst must dedupe to one slot");

        // Both convs share the same packed_weight id.
        let id_a = match &prepared.instructions[1] {
            PreparedInstruction::Conv2d(c) => {
                c.packed_weight.expect("first conv must have a binding")
            }
            other => panic!("expected Conv2d at index 1, got {other:?}"),
        };
        let id_b = match &prepared.instructions[2] {
            PreparedInstruction::Conv2d(c) => {
                c.packed_weight.expect("second conv must have a binding")
            }
            other => panic!("expected Conv2d at index 2, got {other:?}"),
        };
        assert_eq!(id_a, id_b, "shared weight must resolve to the same id");
        assert_eq!(arena.get(id_a), Some(weight_payload.as_slice()));
    }

    // ── extra detection coverage ──────────────────────────────

    /// Fused matmul with bias: both weight and bias feed by WriteConst
    /// producers should produce two bindings, with the weight one
    /// landing on `packed_weight`.
    #[test]
    fn detects_fused_matmul_weight_and_bias() {
        let weight_payload: Vec<f32> = (0..6).map(|i| (i + 1) as f32).collect();
        let bias_payload: Vec<f32> = vec![0.1, 0.2];
        let wc_weight = make_write_const(BufferSlice::new(0, 24), &weight_payload);
        let wc_bias = make_write_const(BufferSlice::new(64, 8), &bias_payload);

        let mut mm = make_matmul_instruction("fused_matmul_add_relu", 4, 3, 2, true);
        if let Instruction::CallKernel { input_slices, .. } = &mut mm {
            input_slices[0] = BufferSlice::new(128, 4 * 3 * 4);
            input_slices[1] = BufferSlice::new(0, 24);
            input_slices[2] = BufferSlice::new(64, 8);
        } else {
            panic!("expected CallKernel");
        }

        let plan = ExecutablePlan {
            instructions: vec![wc_weight, wc_bias, mm],
            arena_size: 4096,
            levels: vec![0, 0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        let arena = prepared
            .constant_arena()
            .expect("plan should have an attached arena");
        // Two distinct writers → two arena slots.
        assert_eq!(arena.len(), 2);
        assert!(arena.id_for("matmul_0_i1").is_some());
        assert!(arena.id_for("matmul_1_i2").is_some());

        match &prepared.instructions[2] {
            PreparedInstruction::MatMul(m) => {
                let id = m
                    .packed_weight
                    .expect("fused matmul must have a packed_weight binding");
                assert_eq!(id.index, 0);
                assert_eq!(arena.get(id), Some(weight_payload.as_slice()));
            }
            other => panic!("expected MatMul at index 2, got {other:?}"),
        }
    }

    /// Conv2d with bias: both the weight and the bias feed by
    /// `WriteConst` producers are detected.
    #[test]
    fn detects_conv_weight_and_bias() {
        let params = Conv2dParams {
            with_bias: true,
            ..Default::default()
        };
        // Use the helper's layout: input (0..input_size), weight
        // (input_size..input_size+weight_size), bias
        // (input_size+weight_size..). Build a matching WriteConst for
        // each.
        let input_size = params.c * params.h * params.w * 4;
        let c_per_group = params.c / params.groups.max(1);
        let f = 8;
        let weight_size = f * c_per_group * params.kh * params.kw * 4;
        let bias_size = f * 4;
        let weight_numel = weight_size / 4;
        let weight_payload: Vec<f32> = (0..weight_numel).map(|i| i as f32 * 0.1).collect();
        let bias_payload: Vec<f32> = vec![0.0; f];

        let wc_weight =
            make_write_const(BufferSlice::new(input_size, weight_size), &weight_payload);
        let wc_bias = make_write_const(
            BufferSlice::new(input_size + weight_size, bias_size),
            &bias_payload,
        );

        let conv = make_conv2d_instruction("conv2d", params);
        let plan = ExecutablePlan {
            instructions: vec![wc_weight, wc_bias, conv],
            arena_size: 4096,
            levels: vec![0, 0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        let arena = prepared
            .constant_arena()
            .expect("plan should have an attached arena");
        assert_eq!(arena.len(), 2);
        assert!(arena.id_for("conv_0_i1").is_some());
        assert!(arena.id_for("conv_1_i2").is_some());

        match &prepared.instructions[2] {
            PreparedInstruction::Conv2d(c) => {
                let id = c
                    .packed_weight
                    .expect("conv with const weight must have a binding");
                assert_eq!(id.index, 0);
                assert_eq!(arena.get(id), Some(weight_payload.as_slice()));
            }
            other => panic!("expected Conv2d at index 2, got {other:?}"),
        }
    }

    /// A plan with no Conv2d / MatMul and no WriteConst should still
    /// produce a (metadata-only) empty arena and no bindings.
    #[test]
    fn detect_static_weights_empty_plan() {
        let mut arena = PreparedConstantArena::new();
        let bindings = detect_static_weights(&empty_plan(), &mut arena);
        assert!(bindings.is_empty());
        assert!(arena.is_empty());
    }

    /// Helper sanity check: `bytes_to_f32_vec` reads little-endian
    /// f32 payloads and drops trailing partial elements.
    #[test]
    fn bytes_to_f32_vec_round_trip() {
        let payload: Vec<f32> = vec![1.5, -2.25, 3.125, 4.0];
        let bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
        let recovered = bytes_to_f32_vec(&bytes);
        assert_eq!(recovered, payload);

        // A trailing partial chunk (1 byte after a single f32) is
        // dropped: only the first 4 bytes survive as 1 f32.
        let mut partial = payload[0].to_le_bytes().to_vec();
        partial.push(0xff);
        let recovered_partial = bytes_to_f32_vec(&partial);
        assert_eq!(recovered_partial, vec![payload[0]]);
    }

    // ── introspection helpers: static_weight_binding_count ────

    #[test]
    fn static_weight_binding_count_empty_plan() {
        let prepared = prepare_executable_plan(&empty_plan());
        assert_eq!(prepared.static_weight_binding_count(), 0);
    }

    #[test]
    fn static_weight_binding_count_no_static_weights() {
        // A plan with a Conv2d whose weight is fed by a MemCopy (i.e.
        // dynamic) gets no `packed_weight` binding.
        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(512, 32),
            src: BufferSlice::new(0, 32),
        };
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(512, 32);
        } else {
            panic!("expected CallKernel");
        }
        let plan = ExecutablePlan {
            instructions: vec![memcopy, conv],
            arena_size: 4096,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        // The conv was promoted but never bound to a static weight.
        assert_eq!(prepared.static_weight_binding_count(), 0);
    }

    #[test]
    fn static_weight_binding_count_counts_conv_and_matmul() {
        // Build a plan with a WriteConst feeding a Conv2d weight, a
        // second WriteConst feeding a fused matmul weight, and a third
        // consumer (a plain matmul) that is dynamic.
        let weight_payload: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let mm_weight_payload: Vec<f32> = (0..6).map(|i| (i + 1) as f32).collect();

        let wc_conv = make_write_const(BufferSlice::new(0, 32), &weight_payload);
        let wc_mm = make_write_const(BufferSlice::new(64, 24), &mm_weight_payload);

        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(0, 32);
        } else {
            panic!("expected CallKernel");
        }

        let mut mm = make_matmul_instruction("fused_matmul_add_relu", 4, 3, 2, true);
        if let Instruction::CallKernel { input_slices, .. } = &mut mm {
            input_slices[0] = BufferSlice::new(128, 4 * 3 * 4);
            input_slices[1] = BufferSlice::new(64, 24);
            // bias slot fed by a MemCopy → dynamic, no binding
            input_slices[2] = BufferSlice::new(256, 8);
        } else {
            panic!("expected CallKernel");
        }

        let memcopy = Instruction::MemCopy {
            dst: BufferSlice::new(256, 8),
            src: BufferSlice::new(0, 8),
        };

        let plan = ExecutablePlan {
            instructions: vec![wc_conv, wc_mm, memcopy, conv, mm],
            arena_size: 4096,
            levels: vec![0, 0, 0, 1, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        // Two weight bindings: the conv and the fused matmul.
        // Bias bindings are still metadata-only and not counted.
        assert_eq!(prepared.static_weight_binding_count(), 2);
    }

    #[test]
    fn static_weight_binding_count_hand_construction() {
        // A hand-built plan with explicit packed_weight assignments.
        let mut conv = PreparedConv2d {
            instruction_index: 0,
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
            packed_weight: Some(PackedWeightId::new(0)),
            scratch_offset: 0,
            scratch_len: 0,
        };
        let mut mm = PreparedMatMul {
            instruction_index: 1,
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
        };
        let plan = PreparedExecutablePlan {
            instructions: vec![
                PreparedInstruction::Conv2d(conv.clone()),
                PreparedInstruction::MatMul(mm.clone()),
            ],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(plan.static_weight_binding_count(), 1);

        // Flip both to bound → count climbs to 2.
        mm.packed_weight = Some(PackedWeightId::new(1));
        let plan = PreparedExecutablePlan {
            instructions: vec![
                PreparedInstruction::Conv2d(conv.clone()),
                PreparedInstruction::MatMul(mm),
            ],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(plan.static_weight_binding_count(), 2);

        // Reset conv → only matmul is bound, count drops to 1.
        conv.packed_weight = None;
        let plan = PreparedExecutablePlan {
            instructions: vec![
                PreparedInstruction::Conv2d(conv),
                PreparedInstruction::Generic {
                    instruction_index: 2,
                },
            ],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(plan.static_weight_binding_count(), 0);
    }

    // ── introspection helpers: constant_arena_entry_count ─────

    #[test]
    fn constant_arena_entry_count_empty_plan_starts_at_zero() {
        let prepared = prepare_executable_plan(&empty_plan());
        assert_eq!(prepared.constant_arena_entry_count(), 0);
    }

    #[test]
    fn constant_arena_entry_count_no_arena_returns_zero() {
        // Hand-built plan with no arena attached.
        let plan = PreparedExecutablePlan {
            instructions: vec![],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(plan.constant_arena_entry_count(), 0);
    }

    #[test]
    fn constant_arena_entry_count_reflects_arena_size() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::Fill {
                dst: BufferSlice::new(0, 4),
                value: 0.0,
            }],
            arena_size: 0,
            levels: vec![0],
        };
        let prepared = prepare_executable_plan(&plan);
        // No static weights were detected, so the auto-attached arena
        // stays empty.
        assert_eq!(prepared.constant_arena_entry_count(), 0);

        // Build a plan that yields two arena entries (one Conv2d
        // weight + one fused matmul weight).
        let wc_a = make_write_const(BufferSlice::new(0, 32), &vec![0.0_f32; 8]);
        let wc_b = make_write_const(BufferSlice::new(64, 24), &vec![1.0_f32; 6]);
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(0, 32);
        } else {
            panic!("expected CallKernel");
        }
        let mut mm = make_matmul_instruction("fused_matmul_add_relu", 4, 3, 2, true);
        if let Instruction::CallKernel { input_slices, .. } = &mut mm {
            input_slices[0] = BufferSlice::new(128, 4 * 3 * 4);
            input_slices[1] = BufferSlice::new(64, 24);
            input_slices[2] = BufferSlice::new(256, 8);
        } else {
            panic!("expected CallKernel");
        }
        let plan = ExecutablePlan {
            instructions: vec![wc_a, wc_b, conv, mm],
            arena_size: 4096,
            levels: vec![0, 0, 1, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        // Two WriteConsts (weight + weight); the bias slot is dynamic
        // so it is NOT registered in the arena.
        assert_eq!(prepared.constant_arena_entry_count(), 2);
    }

    // ── introspection helpers: constant_arena_total_bytes ─────

    #[test]
    fn constant_arena_total_bytes_empty_plan_starts_at_zero() {
        let prepared = prepare_executable_plan(&empty_plan());
        assert_eq!(prepared.constant_arena_total_bytes(), 0);
    }

    #[test]
    fn constant_arena_total_bytes_no_arena_returns_zero() {
        let plan = PreparedExecutablePlan {
            instructions: vec![],
            arena_size: 0,
            scratch_size: 0,
            constant_arena: None,
        };
        assert_eq!(plan.constant_arena_total_bytes(), 0);
    }

    #[test]
    fn constant_arena_total_bytes_sums_entry_sizes() {
        // 8 f32 (32 bytes) + 6 f32 (24 bytes) = 56 bytes.
        let wc_a = make_write_const(BufferSlice::new(0, 32), &vec![0.0_f32; 8]);
        let wc_b = make_write_const(BufferSlice::new(64, 24), &vec![1.0_f32; 6]);
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(0, 32);
        } else {
            panic!("expected CallKernel");
        }
        let mut mm = make_matmul_instruction("matmul", 4, 3, 2, false);
        if let Instruction::CallKernel { input_slices, .. } = &mut mm {
            input_slices[0] = BufferSlice::new(128, 4 * 3 * 4);
            input_slices[1] = BufferSlice::new(64, 24);
        } else {
            panic!("expected CallKernel");
        }
        let plan = ExecutablePlan {
            instructions: vec![wc_a, wc_b, conv, mm],
            arena_size: 4096,
            levels: vec![0, 0, 1, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        assert_eq!(prepared.constant_arena_total_bytes(), 32 + 24);
    }

    #[test]
    fn constant_arena_total_bytes_matches_arena_helper() {
        // Cross-check: the helper should report the same value as
        // reading the arena directly.
        let wc = make_write_const(BufferSlice::new(0, 16), &vec![0.0_f32; 4]);
        let mut conv = make_conv2d_instruction("conv2d", Conv2dParams::default());
        if let Instruction::CallKernel { input_slices, .. } = &mut conv {
            input_slices[0] = BufferSlice::new(0, 768);
            input_slices[1] = BufferSlice::new(0, 16);
        } else {
            panic!("expected CallKernel");
        }
        let plan = ExecutablePlan {
            instructions: vec![wc, conv],
            arena_size: 4096,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        let arena = prepared
            .constant_arena()
            .expect("plan should carry an arena");
        assert_eq!(prepared.constant_arena_total_bytes(), arena.total_bytes());
        assert_eq!(prepared.constant_arena_entry_count(), arena.len());
    }

    // ── validate_prepared_against_plan ──────────────────────────

    /// `validate_prepared_against_plan` returns `Ok` for a freshly
    /// prepared plan that matches the source plan in length and
    /// original-instruction order.
    #[test]
    fn validate_prepared_against_plan_ok_on_consistent_plan() {
        let plan = ExecutablePlan {
            instructions: vec![
                Instruction::Fill {
                    dst: BufferSlice::new(0, 4),
                    value: 1.0,
                },
                Instruction::MemCopy {
                    dst: BufferSlice::new(0, 4),
                    src: BufferSlice::new(4, 4),
                },
                make_conv2d_instruction("conv2d", Conv2dParams::default()),
            ],
            arena_size: 4096,
            levels: vec![0, 0, 1],
        };
        let prepared = prepare_executable_plan(&plan);
        validate_prepared_against_plan(&prepared, &plan)
            .expect("consistent plan must validate cleanly");
    }

    /// `validate_prepared_against_plan` returns `Err` when the prepared
    /// plan has a different number of instructions than the source
    /// plan — e.g. the caller accidentally swapped a different
    /// prepared plan in.
    #[test]
    fn validate_prepared_against_plan_rejects_length_mismatch() {
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
            arena_size: 1024,
            levels: vec![0, 1],
        };
        let prepared = prepare_executable_plan(&plan);

        // Truncate the plan to a single instruction; the prepared plan
        // (still holding the full two-instruction order) must now
        // disagree on length.
        let truncated = ExecutablePlan {
            instructions: plan.instructions[..1].to_vec(),
            arena_size: plan.arena_size,
            levels: plan.levels[..1].to_vec(),
        };
        let err = validate_prepared_against_plan(&prepared, &truncated)
            .expect_err("length mismatch must be rejected");
        assert!(matches!(err, crate::backend::BackendError::Dispatch(_)));
    }

    /// `validate_prepared_against_plan` returns `Err` when the prepared
    /// plan's `original_instruction_order` is not the identity
    /// permutation. We forge a hand-built plan whose order reports
    /// `[1, 0]` for a two-instruction source plan.
    #[test]
    fn validate_prepared_against_plan_rejects_scrambled_order() {
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
            arena_size: 1024,
            levels: vec![0, 1],
        };
        // Build a hand-crafted prepared plan whose `instruction_index`
        // for the first slot is `1` (a deliberate mismatch).
        let scrambled = PreparedExecutablePlan {
            instructions: vec![
                PreparedInstruction::Generic {
                    instruction_index: 1,
                },
                PreparedInstruction::Generic {
                    instruction_index: 0,
                },
            ],
            arena_size: plan.arena_size,
            scratch_size: 0,
            constant_arena: None,
        };
        let err = validate_prepared_against_plan(&scrambled, &plan)
            .expect_err("scrambled order must be rejected");
        assert!(matches!(err, crate::backend::BackendError::Dispatch(_)));
    }

    /// `validate_prepared_against_plan` rejects a prepared plan whose
    /// `original_instruction_order` reports an out-of-bounds index.
    #[test]
    fn validate_prepared_against_plan_rejects_out_of_bounds_index() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::Fill {
                dst: BufferSlice::new(0, 4),
                value: 0.0,
            }],
            arena_size: 64,
            levels: vec![0],
        };
        let forged = PreparedExecutablePlan {
            instructions: vec![PreparedInstruction::Generic {
                instruction_index: 5,
            }],
            arena_size: plan.arena_size,
            scratch_size: 0,
            constant_arena: None,
        };
        let err = validate_prepared_against_plan(&forged, &plan)
            .expect_err("out-of-bounds index must be rejected");
        assert!(matches!(err, crate::backend::BackendError::Dispatch(_)));
    }
}
