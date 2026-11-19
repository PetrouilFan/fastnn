//! Bounded execution plans: the specialised artifact of the runtime planner.
//!
//! # What this module owns
//!
//! [`ExecutionPlan`] (in [`super`]) is the *executable* contract: an ordered
//! instruction list over flat value slots, and nothing else. It deliberately
//! does not know what each slot holds, how big it is, or where a constant comes
//! from. This module adds the missing half without changing that contract:
//!
//! ```text
//! Plan
//!   ├── ExecutionPlan        instructions + value_count (core contract)
//!   ├── Vec<SlotPlan>        dtype, shape and role of every slot
//!   ├── Vec<Constant>        integer shape constants folded at build time
//!   ├── Vec<MaskConstant>    the finite mask fill values, statically checked
//!   ├── Vec<FoldedNode>      which imported nodes were folded, and why they ran
//!   ├── LiveDims + envelope  the concrete environment this plan is valid for
//!   └── PlanId               128-bit content fingerprint (cache/share key)
//! ```
//!
//! A plan is produced by [`super::specialize::Planner`]; it is immutable,
//! `Send + Sync` and cheap to share behind an `Arc`, exactly like
//! [`super::PreparedModel`].
//!
//! # Why slots carry shapes at all
//!
//! [`ExecutionPlan::execute`] takes `&mut [Option<Tensor>]` whose entries the
//! caller allocated. Without a per-slot shape table the caller cannot know how
//! big slot 37 is, so allocation policy would leak back into execution. The
//! table makes allocation a pure function of the plan:
//!
//! ```text
//! let mut slots = plan.materialise(model.weights(), device)?;
//! slots[plan.input("input_ids")?.slot() as usize] = Some(tokens);
//! plan.execute(&backend, &mut slots)?;
//! let logits = slots[plan.output("logits")?.slot() as usize]
//!     .as_ref().expect("the plan writes its outputs");
//! ```
//!
//! # Bounded plans
//!
//! Two byte figures are computed at build time from the resolved shapes, never
//! estimated at run time:
//!
//! | Figure | Meaning |
//! |---|---|
//! | [`Plan::allocation_bytes`] | bytes [`Plan::materialise`] allocates: folded constants + instruction outputs |
//! | [`Plan::peak_bytes`] | largest simultaneous live set of the same slots, i.e. the bound a pooling executor must meet |
//! | [`Plan::weight_bytes`] | immutable, `Arc`-shared weight bytes, reported separately because they are resident for the model's lifetime |
//!
//! `peak_bytes` is the interesting one: it is computed by liveness over the
//! instruction list, with graph inputs, weights, constants and graph outputs
//! treated as resident (they must survive the whole step) and every other
//! value freed after its last read. Because both figures are functions of the
//! resolved extents, [`super::specialize::Planner::bound`] can plan the worst
//! corner of an envelope once and prove every admissible shape fits.
//!
//! # Determinism
//!
//! Nothing here iterates a hash map. Slots are in `GraphIr` value order,
//! constants in first-use order, instructions in graph order, and the
//! fingerprint is a pure function of those sequences. The same model, envelope
//! and [`LiveDims`] therefore always produce the same [`PlanId`] and an equal
//! [`Plan`] (`PartialEq` is derived on the executable half and on every table).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use crate::inference::device::DeviceId;
use crate::inference::dtype::DType;
use crate::inference::error::Fn3Error;
use crate::inference::ops::OpId;
use crate::inference::shape::{LiveDims, Shape, ShapeEnvelope, Symbol};
use crate::inference::tensor::Tensor;

use super::weights::WeightId;
use super::{ExecutionPlan, Instruction, ModelWeights};

// ---------------------------------------------------------------------------
// Plan identity
// ---------------------------------------------------------------------------

/// A 128-bit content fingerprint of a specialised plan.
///
/// The fingerprint covers the declared envelope, the concrete [`LiveDims`], the
/// backend name, every slot's dtype/shape/role, every folded constant's exact
/// bytes, and every instruction's op id, operands and attributes. It is a
/// cache/share key, not a cryptographic hash: this crate has no crypto
/// dependency, so it uses FNV-1a at 128 bits, and [`Plan`] derives `PartialEq`
/// so a caller that must be certain can compare plans exactly after a hit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PlanId([u8; 16]);

impl PlanId {
    /// Raw little-endian bytes of the digest.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Lowercase hexadecimal rendering.
    pub fn to_hex(&self) -> String {
        let mut text = String::with_capacity(32);
        for byte in self.0 {
            text.push_str(&format!("{byte:02x}"));
        }
        text
    }
}

impl fmt::Display for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for PlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanId({})", self.to_hex())
    }
}

/// FNV-1a, 128-bit parameters (the published FNV reference constants).
///
/// FNV is used because it is dependency-free, streamable and exactly specified;
/// the digest is a fingerprint over plan content, and `Plan: PartialEq` is the
/// authority when two fingerprints match.
struct Digest(u128);

impl Digest {
    const OFFSET_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    fn new() -> Digest {
        Digest(Digest::OFFSET_BASIS)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u128::from(byte);
            self.0 = self.0.wrapping_mul(Digest::PRIME);
        }
    }

    fn u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.write(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    /// Length-prefixed so `["ab"]` and `["a", "b"]` cannot collide.
    fn text(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.write(value.as_bytes());
    }

    fn shape(&mut self, shape: &Shape) {
        self.u32(shape.rank() as u32);
        for dim in shape.dims() {
            self.u64(*dim);
        }
    }

    fn dims(&mut self, dims: &LiveDims) {
        for symbol in Symbol::ALL {
            self.u64(dims.get(symbol));
        }
    }

    fn envelope(&mut self, envelope: &ShapeEnvelope) {
        for symbol in Symbol::ALL {
            let bound = envelope.get(symbol);
            self.u64(bound.min());
            self.u64(bound.max());
        }
    }

    fn finish(self) -> PlanId {
        PlanId(self.0.to_le_bytes())
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Typed payload of a constant the planner folded or lifted.
///
/// Only the integer domain is folded. Floating-point work stays on the backend
/// so there is exactly one numerical path for `f32`, and the alpha CPU
/// arithmetic kernels are documented `f32`-only, so an `I64` `Add` has no
/// executable form at all (see [`super::specialize`]).
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantData {
    /// `I64` elements, row-major.
    I64(Vec<i64>),
    /// `I32` elements, row-major.
    I32(Vec<i32>),
    /// `Bool` elements, one `0`/`1` byte each.
    Bool(Vec<u8>),
}

impl ConstantData {
    /// Element type of the payload.
    pub fn dtype(&self) -> DType {
        match self {
            ConstantData::I64(_) => DType::I64,
            ConstantData::I32(_) => DType::I32,
            ConstantData::Bool(_) => DType::Bool,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            ConstantData::I64(values) => values.len(),
            ConstantData::I32(values) => values.len(),
            ConstantData::Bool(values) => values.len(),
        }
    }

    /// `true` for an empty payload.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The elements as `i64`, widening `I32` and `Bool`.
    pub fn as_i64(&self) -> Vec<i64> {
        match self {
            ConstantData::I64(values) => values.clone(),
            ConstantData::I32(values) => values.iter().map(|&value| i64::from(value)).collect(),
            ConstantData::Bool(values) => values.iter().map(|&value| i64::from(value)).collect(),
        }
    }

    /// Exact payload bytes, little-endian, row-major.
    pub fn bytes(&self) -> Vec<u8> {
        match self {
            ConstantData::I64(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 8);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            ConstantData::I32(values) => {
                let mut bytes = Vec::with_capacity(values.len() * 4);
                for value in values {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            }
            ConstantData::Bool(values) => values.clone(),
        }
    }
}

/// One constant the planner materialises into a value slot.
#[derive(Debug, Clone, PartialEq)]
pub struct Constant {
    shape: Shape,
    data: ConstantData,
}

impl Constant {
    /// Construct from a payload and its resolved shape.
    ///
    /// The element count must agree; that is the same checked rule every
    /// tensor constructor applies, so a folded constant can never describe a
    /// shape its payload does not fill.
    pub fn new(shape: Shape, data: ConstantData) -> Result<Constant, Fn3Error> {
        let expected = shape.numel()?;
        if expected != data.len() as u64 {
            return Err(Fn3Error::ShapeMismatch {
                op: None,
                expected: format!("{expected} elements for folded constant {shape}"),
                actual: format!("{} elements of {}", data.len(), data.dtype()),
            });
        }
        Ok(Constant { shape, data })
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Resolved shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// The typed payload.
    pub fn data(&self) -> &ConstantData {
        &self.data
    }

    /// Payload size in bytes.
    pub fn byte_len(&self) -> u64 {
        (self.data.len() * self.data.dtype().size_bytes()) as u64
    }

    /// Materialise a host tensor on the CPU.
    pub fn to_tensor(&self) -> Result<Tensor, Fn3Error> {
        match &self.data {
            ConstantData::I64(values) => Tensor::from_i64(self.shape.clone(), values),
            ConstantData::I32(values) => Tensor::from_i32(self.shape.clone(), values),
            ConstantData::Bool(values) => {
                Tensor::from_slice(self.shape.clone(), DType::Bool, values.as_slice())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Slots
// ---------------------------------------------------------------------------

/// Where a slot's contents come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotRole {
    /// A `GraphIr` input; the caller supplies the tensor.
    GraphInput,
    /// A model weight; [`Plan::materialise`] clones it from `ModelWeights`.
    Weight {
        /// Stable handle into the weight table.
        id: WeightId,
    },
    /// A constant folded at build time, index into [`Plan::constants`].
    Constant {
        /// Index into [`Plan::constants`].
        index: u32,
    },
    /// Written by the instruction at this position in the plan.
    InstructionOutput {
        /// Instruction index.
        instruction: u32,
    },
}

impl SlotRole {
    /// Short stable tag used by diagnostics and the plan digest.
    fn tag(&self) -> &'static str {
        match self {
            SlotRole::GraphInput => "input",
            SlotRole::Weight { .. } => "weight",
            SlotRole::Constant { .. } => "constant",
            SlotRole::InstructionOutput { .. } => "output",
        }
    }

    /// `true` if the slot must stay live for the whole step.
    pub fn is_resident(&self) -> bool {
        !matches!(self, SlotRole::InstructionOutput { .. })
    }
}

/// The name, dtype, resolved shape and origin of one value slot.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotPlan {
    name: String,
    dtype: DType,
    shape: Shape,
    role: SlotRole,
}

impl SlotPlan {
    /// Construct one slot description.
    pub fn new(name: String, dtype: DType, shape: Shape, role: SlotRole) -> SlotPlan {
        SlotPlan {
            name,
            dtype,
            shape,
            role,
        }
    }

    /// Name of the `GraphIr` value this slot holds.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Resolved shape; every extent is a concrete `u64`.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Where the contents come from.
    pub fn role(&self) -> &SlotRole {
        &self.role
    }

    /// Payload size in bytes.
    pub fn byte_len(&self) -> Result<u64, Fn3Error> {
        Ok(self.shape.byte_len(self.dtype)? as u64)
    }
}

/// One value bound to a slot: a graph input or a graph output.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundValue {
    name: String,
    slot: u32,
    dtype: DType,
    shape: Shape,
}

impl BoundValue {
    /// Construct one bound value.
    pub(super) fn new(name: String, slot: u32, dtype: DType, shape: Shape) -> BoundValue {
        BoundValue {
            name,
            slot,
            dtype,
            shape,
        }
    }

    /// Model-spelled name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Slot index for [`super::ExecutionPlan::execute`].
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Resolved shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

// ---------------------------------------------------------------------------
// Diagnostics recorded by the planner
// ---------------------------------------------------------------------------

/// Why a node was removed from the instruction list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldKind {
    /// Evaluated at plan time; the result lives in [`Plan::constants`].
    Constant,
    /// Proven identical to one of its inputs, so its value resolves to that
    /// input's slot. No bytes are produced and no instruction is emitted.
    Identity,
}

/// An imported node the planner folded away instead of executing.
///
/// This is the evidence that shape arithmetic never reaches a kernel: a folded
/// node has no [`Instruction`], and its result lives in [`Plan::constants`]
/// ([`FoldKind::Constant`]) or in the slot of the value it aliases
/// ([`FoldKind::Identity`]).
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedNode {
    /// `GraphNode::onnx_index` of the node that was folded.
    pub onnx_index: u32,
    /// ONNX node name.
    pub name: String,
    /// The op that would have executed.
    pub op: OpId,
    /// `GraphIr` value indices it produced.
    pub outputs: Vec<u32>,
    /// Constant evaluation or proven identity.
    pub kind: FoldKind,
}

/// Why a float constant is recorded as a mask value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskRole {
    /// The fill value of `ConstantOfShape`.
    Fill,
    /// The selected branch (`x`) of a `Where`.
    Selected,
    /// The alternative branch (`y`) of a `Where`.
    Alternative,
    /// The optional additive mask operand of `MaskedAttention`.
    AttentionMask,
}

impl MaskRole {
    /// Stable tag used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            MaskRole::Fill => "constant_of_shape.fill",
            MaskRole::Selected => "where.x",
            MaskRole::Alternative => "where.y",
            MaskRole::AttentionMask => "attention.mask",
        }
    }
}

/// A finite float constant the graph uses to mask attention scores.
///
/// The parity policy pinned in `fixtures/parity/README.md` is `-FLT_MAX`, never
/// `-inf`: a fully masked softmax row must stay finite and become uniform. The
/// planner refuses a plan whose mask fill is not finite, so that failure is a
/// build error naming the value rather than a NaN at step time.
#[derive(Debug, Clone, PartialEq)]
pub struct MaskConstant {
    /// Instruction that consumes it, index into [`Plan::instructions`].
    pub instruction: u32,
    /// Op of that instruction.
    pub op: OpId,
    /// What the value is used as.
    pub role: MaskRole,
    /// The `f32` value.
    pub value: f32,
    /// Where it came from (`weight "…"`).
    pub origin: String,
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// A specialised, bounded, executable plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    execution: ExecutionPlan,
    slots: Vec<SlotPlan>,
    constants: Vec<Constant>,
    folded: Vec<FoldedNode>,
    mask: Vec<MaskConstant>,
    inputs: Vec<BoundValue>,
    outputs: Vec<BoundValue>,
    by_name: BTreeMap<String, u32>,
    dims: LiveDims,
    envelope: ShapeEnvelope,
    backend: &'static str,
    identity: PlanId,
    peak_bytes: u64,
    allocation_bytes: u64,
    weight_bytes: u64,
}

/// The pieces [`Plan::assemble`] needs. Kept as one struct so the constructor
/// stays readable and an added field is a compile error rather than a silent
/// default.
pub(super) struct PlanParts {
    pub(super) execution: ExecutionPlan,
    pub(super) slots: Vec<SlotPlan>,
    pub(super) constants: Vec<Constant>,
    pub(super) folded: Vec<FoldedNode>,
    pub(super) mask: Vec<MaskConstant>,
    pub(super) inputs: Vec<BoundValue>,
    pub(super) outputs: Vec<BoundValue>,
    pub(super) dims: LiveDims,
    pub(super) envelope: ShapeEnvelope,
    pub(super) backend: &'static str,
}

impl Plan {
    /// Validate the tables against each other, compute the byte figures and the
    /// fingerprint, and freeze the result.
    pub(super) fn assemble(parts: PlanParts) -> Result<Plan, Fn3Error> {
        let PlanParts {
            execution,
            slots,
            constants,
            folded,
            mask,
            inputs,
            outputs,
            dims,
            envelope,
            backend,
        } = parts;

        if slots.len() != execution.value_count() as usize {
            return Err(Fn3Error::InvalidShape {
                context: "Plan::assemble",
                detail: format!(
                    "the executable plan addresses {} slots but {} slot descriptions were built",
                    execution.value_count(),
                    slots.len()
                ),
            });
        }
        for (index, slot) in slots.iter().enumerate() {
            if let SlotRole::Constant { index: constant } = slot.role {
                let constant =
                    constants
                        .get(constant as usize)
                        .ok_or_else(|| Fn3Error::OutOfBounds {
                            what: "Plan::assemble (constant index)",
                            index: u64::from(constant),
                            len: constants.len() as u64,
                        })?;
                if constant.shape() != &slot.shape || constant.dtype() != slot.dtype {
                    return Err(Fn3Error::InvalidShape {
                        context: "Plan::assemble",
                        detail: format!(
                            "slot {index} describes {} {} but its constant is {} {}",
                            slot.dtype,
                            slot.shape,
                            constant.dtype(),
                            constant.shape()
                        ),
                    });
                }
            }
        }
        // A mask constant is only meaningful if it really is finite; this is the
        // static replacement for "a NaN showed up in step 3".
        for entry in &mask {
            if !entry.value.is_finite() {
                return Err(Fn3Error::InvalidShape {
                    context: "Plan::assemble (mask constant)",
                    detail: format!(
                        "{} = {} at instruction {} (op {}) is not finite; the parity contract \
                         requires the finite f32 -FLT_MAX for masked positions so a fully masked \
                         row stays uniform instead of becoming NaN",
                        entry.role.name(),
                        entry.value,
                        entry.instruction,
                        entry.op
                    ),
                });
            }
        }
        for bound in inputs.iter().chain(outputs.iter()) {
            let slot = slots
                .get(bound.slot as usize)
                .ok_or_else(|| Fn3Error::OutOfBounds {
                    what: "Plan::assemble (bound value slot)",
                    index: u64::from(bound.slot),
                    len: slots.len() as u64,
                })?;
            if slot.dtype != bound.dtype || slot.shape != bound.shape {
                return Err(Fn3Error::InvalidShape {
                    context: "Plan::assemble",
                    detail: format!(
                        "value `{}` is bound to slot {} ({} {}) but the slot is {} {}",
                        bound.name, bound.slot, bound.dtype, bound.shape, slot.dtype, slot.shape
                    ),
                });
            }
        }

        let allocation_bytes = slots.iter().try_fold(0u64, |total, slot| {
            // Weight slots share `Arc` storage with the model; they are not
            // allocated (or re-allocated) per plan materialisation.
            if matches!(slot.role, SlotRole::Weight { .. }) {
                Ok::<u64, Fn3Error>(total)
            } else {
                Ok(total + slot.byte_len()?)
            }
        })?;
        let weight_bytes = slots.iter().try_fold(0u64, |total, slot| {
            if matches!(slot.role, SlotRole::Weight { .. }) {
                Ok::<u64, Fn3Error>(total + slot.byte_len()?)
            } else {
                Ok(total)
            }
        })?;
        let peak_bytes = peak_bytes(&execution, &slots, &outputs)?;
        let by_name = slots
            .iter()
            .enumerate()
            .map(|(index, slot)| (slot.name().to_string(), index as u32))
            .collect();

        let mut plan = Plan {
            execution,
            slots,
            constants,
            folded,
            mask,
            inputs,
            outputs,
            by_name,
            dims,
            envelope,
            backend,
            identity: PlanId([0; 16]),
            peak_bytes,
            allocation_bytes,
            weight_bytes,
        };
        plan.identity = plan.compute_identity();
        Ok(plan)
    }

    /// Fingerprint over everything that determines execution.
    fn compute_identity(&self) -> PlanId {
        let mut digest = Digest::new();
        // This digest domain is independent of the public module name.
        digest.write(b"fastnn.fn3.plan.v1");
        digest.text(self.backend);
        digest.envelope(&self.envelope);
        digest.dims(&self.dims);
        digest.u32(self.slots.len() as u32);
        for slot in &self.slots {
            digest.u8(slot.dtype.code());
            digest.shape(&slot.shape);
            digest.text(slot.role.tag());
            digest.text(slot.name());
            match &slot.role {
                // The weight handle is part of the identity even when two
                // weights share a name: the slot table is per weight, not per
                // name.
                SlotRole::Weight { id } => digest.u32(id.index() as u32),
                SlotRole::Constant { index } => digest.u32(*index),
                SlotRole::InstructionOutput { instruction } => digest.u32(*instruction),
                SlotRole::GraphInput => {}
            }
        }
        digest.u32(self.constants.len() as u32);
        for constant in &self.constants {
            digest.u8(constant.dtype().code());
            digest.shape(constant.shape());
            digest.write(&constant.data().bytes());
        }
        for instruction in self.execution.instructions() {
            digest.u16(instruction.op.code());
            digest.u32(instruction.inputs.len() as u32);
            for input in &instruction.inputs {
                digest.u32(*input);
            }
            digest.u32(instruction.outputs.len() as u32);
            for output in &instruction.outputs {
                digest.u32(*output);
            }
            digest.u32(instruction.params.axes.len() as u32);
            for axis in &instruction.params.axes {
                digest.write(&axis.to_le_bytes());
            }
            digest.u8(u8::from(instruction.params.keep_dims));
            digest.u32(instruction.params.epsilon.to_bits());
        }
        for value in self.inputs.iter().chain(self.outputs.iter()) {
            digest.text(&value.name);
            digest.u32(value.slot);
            digest.u8(value.dtype.code());
            digest.shape(&value.shape);
        }
        digest.finish()
    }

    /// The executable instruction list and value-slot count.
    pub fn execution(&self) -> &ExecutionPlan {
        &self.execution
    }

    /// The instructions, in execution order.
    pub fn instructions(&self) -> &[Instruction] {
        self.execution.instructions()
    }

    /// Number of value slots the plan addresses.
    pub fn value_count(&self) -> u32 {
        self.execution.value_count()
    }

    /// Every slot, in index order.
    pub fn slots(&self) -> &[SlotPlan] {
        &self.slots
    }

    /// One slot.
    pub fn slot(&self, index: u32) -> Option<&SlotPlan> {
        self.slots.get(index as usize)
    }

    /// The constants the planner folded, in first-use order.
    pub fn constants(&self) -> &[Constant] {
        &self.constants
    }

    /// The nodes the planner folded instead of executing.
    pub fn folded(&self) -> &[FoldedNode] {
        &self.folded
    }

    /// The folded node that came from ONNX node `onnx_index`, if any.
    pub fn folded_node(&self, onnx_index: u32) -> Option<&FoldedNode> {
        self.folded
            .iter()
            .find(|node| node.onnx_index == onnx_index)
    }

    /// The finite mask constants the plan checked, with the value of each.
    pub fn mask_constants(&self) -> &[MaskConstant] {
        &self.mask
    }

    /// The concrete dynamic environment this plan was built for.
    pub fn dims(&self) -> LiveDims {
        self.dims
    }

    /// The envelope this plan is valid inside.
    pub fn envelope(&self) -> &ShapeEnvelope {
        &self.envelope
    }

    /// Backend name this plan was preflighted against.
    pub fn backend_name(&self) -> &'static str {
        self.backend
    }

    /// Content fingerprint; stable across processes and runs.
    pub fn identity(&self) -> PlanId {
        self.identity
    }

    /// Bytes of the slots [`Plan::materialise`] has to allocate: the folded
    /// constants plus every instruction output. Graph inputs are supplied by
    /// the caller and weight slots share `Arc` storage with the model, so
    /// neither is counted here.
    pub fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    /// Bytes of the immutable weight slots.
    ///
    /// These are resident for the model's whole lifetime and shared with
    /// [`super::PreparedModel`] through `Arc`, so they are reported separately
    /// rather than folded into the per-step figures.
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Largest simultaneous live set of slots, in bytes.
    ///
    /// Graph inputs, weights, constants and graph outputs are resident for the
    /// whole step; every other value is freed after its last read. This is the
    /// figure a pooling executor must stay under, and the figure
    /// [`PlanBound`] bounds across an envelope.
    pub fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Graph inputs, in graph order.
    pub fn inputs(&self) -> &[BoundValue] {
        &self.inputs
    }

    /// Graph outputs, in graph order.
    pub fn outputs(&self) -> &[BoundValue] {
        &self.outputs
    }

    /// A graph input by name.
    pub fn input(&self, name: &str) -> Option<&BoundValue> {
        self.inputs.iter().find(|value| value.name == name)
    }

    /// A graph output by name.
    pub fn output(&self, name: &str) -> Option<&BoundValue> {
        self.outputs.iter().find(|value| value.name == name)
    }

    /// Slot index of a graph input or weight by name.
    pub fn slot_of(&self, name: &str) -> Option<u32> {
        self.by_name.get(name).copied()
    }

    /// Reject a plan that contains an op the backend cannot run at all.
    pub fn check_supported(
        &self,
        backend: &dyn crate::inference::backend::Backend,
    ) -> Result<(), Fn3Error> {
        self.execution.check_supported(backend)
    }

    /// Execute the plan against pre-allocated slots.
    pub fn execute(
        &self,
        backend: &dyn crate::inference::backend::Backend,
        slots: &mut [Option<Tensor>],
    ) -> Result<(), Fn3Error> {
        self.execution.execute(backend, slots)
    }

    /// Allocate every slot this plan needs, on `device`.
    ///
    /// Graph inputs are left `None` for the caller to fill. Weight slots clone
    /// the shared `Arc` storage, so the copy is a reference-count bump, not a
    /// byte copy. Output and constant slots are freshly allocated.
    ///
    /// Alpha storage is host `AlignedBuffer` on every device (see
    /// [`Tensor::from_storage`]), so this is the CPU path; a device-resident
    /// allocator belongs to FN3-12 (#58).
    pub fn materialise(
        &self,
        weights: &ModelWeights,
        device: DeviceId,
    ) -> Result<Vec<Option<Tensor>>, Fn3Error> {
        let mut slots = Vec::with_capacity(self.slots.len());
        for (index, slot) in self.slots.iter().enumerate() {
            let tensor = match &slot.role {
                SlotRole::GraphInput => None,
                SlotRole::Weight { id } => {
                    let tensor = weights
                        .get(id.index())
                        .ok_or_else(|| Fn3Error::OutOfBounds {
                            what: "Plan::materialise (weight index)",
                            index: id.index() as u64,
                            len: weights.len() as u64,
                        })?;
                    if tensor.dtype() != slot.dtype || tensor.shape() != &slot.shape {
                        return Err(Fn3Error::InvalidShape {
                            context: "Plan::materialise",
                            detail: format!(
                                "weight `{}` is {} {} but slot {index} expects {} {}",
                                slot.name(),
                                tensor.dtype(),
                                tensor.shape(),
                                slot.dtype,
                                slot.shape
                            ),
                        });
                    }
                    Some(tensor.clone())
                }
                SlotRole::Constant { index: constant } => {
                    let constant = self.constants.get(*constant as usize).ok_or_else(|| {
                        Fn3Error::OutOfBounds {
                            what: "Plan::materialise (constant index)",
                            index: u64::from(*constant),
                            len: self.constants.len() as u64,
                        }
                    })?;
                    Some(constant.to_tensor()?)
                }
                SlotRole::InstructionOutput { .. } => {
                    Some(Tensor::zeros(slot.shape.clone(), slot.dtype, device)?)
                }
            };
            slots.push(tensor);
        }
        Ok(slots)
    }
}

/// Largest simultaneous live byte count of the plan's slots.
fn peak_bytes(
    execution: &ExecutionPlan,
    slots: &[SlotPlan],
    outputs: &[BoundValue],
) -> Result<u64, Fn3Error> {
    // Weight slots share `Arc` storage with the model and are resident for its
    // lifetime, so they contribute nothing to the per-step live set; the
    // `resident` mark below keeps them out of the release pass as well.
    let sizes: Vec<u64> = slots
        .iter()
        .map(|slot| {
            if matches!(slot.role, SlotRole::Weight { .. }) {
                Ok(0)
            } else {
                slot.byte_len()
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    // A graph output may be read by a later instruction *and* must survive the
    // step; treat it as resident so the figure is never optimistic.
    let mut resident = vec![false; slots.len()];
    for (index, slot) in slots.iter().enumerate() {
        resident[index] = slot.role.is_resident();
    }
    for output in outputs {
        resident[output.slot as usize] = true;
    }
    // Last instruction that reads each slot. `None` means "never read", which
    // only a graph output can be (it is resident and released by nobody).
    let mut last_read: Vec<Option<usize>> = vec![None; slots.len()];
    for (position, instruction) in execution.instructions().iter().enumerate() {
        for &input in &instruction.inputs {
            last_read[input as usize] = Some(position);
        }
    }
    let mut live = vec![false; slots.len()];
    let mut current = 0u64;
    for (index, is_resident) in resident.iter().enumerate() {
        if *is_resident {
            live[index] = true;
            current += sizes[index];
        }
    }
    let mut peak = current;
    for (position, instruction) in execution.instructions().iter().enumerate() {
        for &output in &instruction.outputs {
            if !live[output as usize] {
                live[output as usize] = true;
                current += sizes[output as usize];
            }
        }
        if current > peak {
            peak = current;
        }
        for (index, size) in sizes.iter().enumerate() {
            if live[index] && !resident[index] && last_read[index] == Some(position) {
                live[index] = false;
                current -= size;
            }
        }
    }
    // Every non-resident slot must have been released; anything else means the
    // liveness table and the instruction list disagree.
    let leaked: Vec<usize> = (0..slots.len())
        .filter(|index| live[*index] && !resident[*index])
        .collect();
    if !leaked.is_empty() {
        return Err(Fn3Error::InvalidShape {
            context: "Plan::assemble (liveness)",
            detail: format!(
                "slots {leaked:?} are written but never read and never released; that is a \
                 planner bug, not a model property"
            ),
        });
    }
    Ok(peak)
}

// ---------------------------------------------------------------------------
// Bounds
// ---------------------------------------------------------------------------

/// The declared bound every admissible plan of one envelope must respect.
///
/// [`super::specialize::Planner::bound`] computes it by planning the worst
/// feasible corner of the envelope. The admissible shape sets are
/// `{past in [past.min, past.max], seq in [seq.min, seq.max],
/// past + seq in [total.min, total.max]}`, and every resolved extent in a
/// decoder graph is a product of non-negative powers of those symbols, so the
/// byte figures are monotone and the maximum is attained at a maximal element
/// of that set — one or two corners, which the bound records.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanBound {
    envelope: ShapeEnvelope,
    batch: u64,
    corners: Vec<LiveDims>,
    backend: &'static str,
    max_value_slots: u32,
    max_instructions: u32,
    max_peak_bytes: u64,
    max_allocation_bytes: u64,
    max_weight_bytes: u64,
}

impl PlanBound {
    /// Construct a bound from the plans of `corners`.
    pub(super) fn from_corners(
        envelope: ShapeEnvelope,
        batch: u64,
        corners: Vec<LiveDims>,
        plans: &[Plan],
    ) -> Result<PlanBound, Fn3Error> {
        let first = plans.first().ok_or_else(|| Fn3Error::InvalidShape {
            context: "PlanBound::from_corners",
            detail: "a bound needs at least one planned corner".to_string(),
        })?;
        let backend = first.backend_name();
        if plans.iter().any(|plan| plan.backend_name() != backend) {
            return Err(Fn3Error::InvalidShape {
                context: "PlanBound::from_corners",
                detail: "the corner plans were preflighted against different backends".to_string(),
            });
        }
        Ok(PlanBound {
            envelope,
            batch,
            corners,
            backend,
            max_value_slots: plans.iter().map(Plan::value_count).max().unwrap_or(0),
            max_instructions: plans
                .iter()
                .map(|plan| plan.instructions().len() as u32)
                .max()
                .unwrap_or(0),
            max_peak_bytes: plans.iter().map(Plan::peak_bytes).max().unwrap_or(0),
            max_allocation_bytes: plans.iter().map(Plan::allocation_bytes).max().unwrap_or(0),
            max_weight_bytes: plans.iter().map(Plan::weight_bytes).max().unwrap_or(0),
        })
    }

    /// The envelope this bound was derived from.
    pub fn envelope(&self) -> &ShapeEnvelope {
        &self.envelope
    }

    /// The batch size the corners were planned at.
    pub fn batch(&self) -> u64 {
        self.batch
    }

    /// The corners that attained the bound, in plan order.
    pub fn corners(&self) -> &[LiveDims] {
        &self.corners
    }

    /// Backend the bound was proven against.
    pub fn backend_name(&self) -> &'static str {
        self.backend
    }

    /// Largest value-slot count any admissible plan may declare.
    pub fn max_value_slots(&self) -> u32 {
        self.max_value_slots
    }

    /// Largest instruction count any admissible plan may declare.
    pub fn max_instructions(&self) -> u32 {
        self.max_instructions
    }

    /// Largest [`Plan::peak_bytes`] any admissible plan may report.
    pub fn max_peak_bytes(&self) -> u64 {
        self.max_peak_bytes
    }

    /// Largest [`Plan::allocation_bytes`] any admissible plan may report.
    pub fn max_allocation_bytes(&self) -> u64 {
        self.max_allocation_bytes
    }

    /// Largest [`Plan::weight_bytes`] any admissible plan may report.
    ///
    /// Weight shapes do not depend on the dynamic symbols for a decoder, so in
    /// practice this is one figure for every shape; it is in the bound so a
    /// model that contradicts that assumption is still covered.
    pub fn max_weight_bytes(&self) -> u64 {
        self.max_weight_bytes
    }

    /// Check that `plan` stays inside this bound.
    ///
    /// Fails with [`Fn3Error::CapacityExceeded`] naming the figure that was
    /// exceeded, so a plan that somehow escaped the envelope is reported rather
    /// than executed.
    pub fn admits(&self, plan: &Plan) -> Result<(), Fn3Error> {
        self.envelope.check(&plan.dims())?;
        if plan.backend_name() != self.backend {
            return Err(Fn3Error::InvalidShape {
                context: "PlanBound::admits",
                detail: format!(
                    "plan was built for backend `{}` but the bound is for `{}`",
                    plan.backend_name(),
                    self.backend
                ),
            });
        }
        if plan.value_count() > self.max_value_slots {
            return Err(Fn3Error::CapacityExceeded {
                what: "plan value slots",
                requested: u64::from(plan.value_count()),
                capacity: u64::from(self.max_value_slots),
            });
        }
        if plan.instructions().len() as u32 > self.max_instructions {
            return Err(Fn3Error::CapacityExceeded {
                what: "plan instructions",
                requested: plan.instructions().len() as u64,
                capacity: u64::from(self.max_instructions),
            });
        }
        if plan.peak_bytes() > self.max_peak_bytes {
            return Err(Fn3Error::CapacityExceeded {
                what: "plan peak bytes",
                requested: plan.peak_bytes(),
                capacity: self.max_peak_bytes,
            });
        }
        if plan.allocation_bytes() > self.max_allocation_bytes {
            return Err(Fn3Error::CapacityExceeded {
                what: "plan allocation bytes",
                requested: plan.allocation_bytes(),
                capacity: self.max_allocation_bytes,
            });
        }
        if plan.weight_bytes() > self.max_weight_bytes {
            return Err(Fn3Error::CapacityExceeded {
                what: "plan weight bytes",
                requested: plan.weight_bytes(),
                capacity: self.max_weight_bytes,
            });
        }
        Ok(())
    }
}

impl fmt::Display for PlanBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "plan bound (batch {}, backend {}, envelope batch {} seq {} past {} total {}): \
             <= {} slots, <= {} instructions, <= {} peak bytes, <= {} allocated bytes, \
             {} weight bytes",
            self.batch,
            self.backend,
            self.envelope.get(Symbol::Batch),
            self.envelope.get(Symbol::Seq),
            self.envelope.get(Symbol::Past),
            self.envelope.get(Symbol::Total),
            self.max_value_slots,
            self.max_instructions,
            self.max_peak_bytes,
            self.max_allocation_bytes,
            self.max_weight_bytes
        )
    }
}

// ---------------------------------------------------------------------------
// Plan cache
// ---------------------------------------------------------------------------

/// A bounded cache of specialised plans for one model, envelope and backend.
///
/// The issue this module belongs to asks for "repeated calls reuse the
/// applicable plan" with an explicit key and explicit invalidation, so the key
/// is `(batch, seq, past)` — `total` is derived, and the head counts are fixed
/// by the weights — and invalidation is "the cache belongs to exactly one
/// envelope and one backend". A request outside either is an error, never a
/// silent miss that produces a plan for the wrong bounds:
///
/// ```ignore
/// let mut cache = PlanCache::new(&envelope, backend.name(), 8)?;
/// let prefill = cache.plan(&planner, &envelope, LiveDims::new(1, 4, 0, 2, 2, 64)?)?;
/// let again = cache.plan(&planner, &envelope, LiveDims::new(1, 4, 0, 2, 2, 64)?)?;
/// assert!(Arc::ptr_eq(&prefill, &again));   // one plan, two callers
/// ```
///
/// Eviction is deterministic: when the cache is full the smallest key (by
/// `batch, seq, past`) is dropped, so runs are reproducible and a report can
/// say what a full cache holds.
/// `(batch, seq, past, n_heads, n_kv_heads, head_dim)`; `total` is derived.
type PlanKey = (u64, u64, u64, u64, u64, u64);

#[derive(Debug, Clone)]
pub struct PlanCache {
    envelope: ShapeEnvelope,
    backend: &'static str,
    capacity: usize,
    plans: BTreeMap<PlanKey, Arc<Plan>>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl PlanCache {
    /// Create a cache for one envelope and one backend.
    pub fn new(
        envelope: ShapeEnvelope,
        backend: &'static str,
        capacity: usize,
    ) -> Result<PlanCache, Fn3Error> {
        if capacity == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "PlanCache::new",
                detail: "a plan cache needs capacity >= 1".to_string(),
            });
        }
        envelope.validate()?;
        Ok(PlanCache {
            envelope,
            backend,
            capacity,
            plans: BTreeMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        })
    }

    fn key(dims: &LiveDims) -> PlanKey {
        (
            dims.batch,
            dims.seq,
            dims.past,
            dims.n_heads,
            dims.n_kv_heads,
            dims.head_dim,
        )
    }

    /// The envelope every plan in this cache is valid for.
    pub fn envelope(&self) -> &ShapeEnvelope {
        &self.envelope
    }

    /// The number of plans the cache holds.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// `true` when the cache holds no plans.
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// Maximum number of plans.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Plans served from the cache.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Plans that had to be built.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Plans dropped because the cache was full.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Look a plan up without building one.
    pub fn get(&mut self, dims: LiveDims) -> Option<Arc<Plan>> {
        match self.plans.get(&PlanCache::key(&dims)) {
            Some(plan) => {
                self.hits += 1;
                Some(Arc::clone(plan))
            }
            None => None,
        }
    }

    /// Return the plan for `dims`, building it on a miss.
    ///
    /// The requested dims must be inside this cache's envelope and the planner
    /// must target the same backend and envelope, so a cache entry can never be
    /// reused for a shape it was not specialised for.
    pub fn plan(
        &mut self,
        planner: &super::specialize::Planner<'_>,
        envelope: &ShapeEnvelope,
        dims: LiveDims,
    ) -> Result<Arc<Plan>, Fn3Error> {
        if envelope != &self.envelope {
            return Err(Fn3Error::InvalidShape {
                context: "PlanCache::plan",
                detail: "the requested envelope is not the one this cache is keyed on; a cached \
                         plan is only valid inside the bounds it was built for"
                    .to_string(),
            });
        }
        if planner.backend().name() != self.backend {
            return Err(Fn3Error::InvalidShape {
                context: "PlanCache::plan",
                detail: format!(
                    "the cache holds `{}` plans but the planner targets `{}`",
                    self.backend,
                    planner.backend().name()
                ),
            });
        }
        self.envelope.check(&dims)?;
        if let Some(plan) = self.get(dims) {
            return Ok(plan);
        }
        self.misses += 1;
        let plan = Arc::new(planner.plan(envelope, dims)?);
        if self.plans.len() >= self.capacity {
            if let Some(oldest) = self.plans.keys().next().copied() {
                self.plans.remove(&oldest);
                self.evictions += 1;
            }
        }
        self.plans.insert(PlanCache::key(&dims), Arc::clone(&plan));
        Ok(plan)
    }

    /// Drop every plan. Used when the weights or the device change, which is
    /// the only thing a plan identity cannot see.
    pub fn clear(&mut self) {
        self.plans.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_stable_across_calls() {
        let mut first = Digest::new();
        first.write(b"fn3");
        first.u64(7);
        let second = {
            let mut digest = Digest::new();
            digest.write(b"fn3");
            digest.u64(7);
            digest
        };
        assert_eq!(first.finish(), second.finish());
    }

    #[test]
    fn digest_separates_length_prefixed_text() {
        let mut concatenated = Digest::new();
        concatenated.text("ab");
        let mut split = Digest::new();
        split.text("a");
        split.text("b");
        assert_ne!(concatenated.finish(), split.finish());
    }

    #[test]
    fn constant_rejects_a_shape_its_payload_does_not_fill() {
        let error =
            Constant::new(Shape::from_dims([3]), ConstantData::I64(vec![1, 2])).unwrap_err();
        assert!(matches!(error, Fn3Error::ShapeMismatch { .. }), "{error}");
        let constant = Constant::new(Shape::from_dims([2]), ConstantData::I64(vec![1, 2])).unwrap();
        assert_eq!(constant.byte_len(), 16);
        assert_eq!(
            constant.to_tensor().unwrap().shape(),
            &Shape::from_dims([2])
        );
    }

    #[test]
    fn plan_id_renders_hex_of_its_bytes() {
        let id = PlanId([
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ]);
        assert_eq!(id.to_hex(), "00112233445566778899aabbccddeeff");
        assert_eq!(id.to_string(), id.to_hex());
    }
}
