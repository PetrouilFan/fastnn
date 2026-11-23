//! The `PLANS` section: one prepared execution plan per target profile.
//!
//! # What a plan entry carries
//!
//! * the target name (`"cpu"`, `"cpu-reference"`) and the precision (`f32`);
//! * the layout policy;
//! * the arena and peak byte counts;
//! * every slot's source (`arena` or a weight id), offset and byte length;
//! * the graph interface, as slot ids in declaration order;
//! * every segment's engine choice, operands, results, full [`OpParams`] and
//!   [`SourceLoc`], including recognised fused chains;
//! * the fusion report;
//! * the layout conversions with their byte counts;
//! * the provider fallbacks and their reasons.
//!
//! A `load` deserializes this plan and hands it to a session. It does not import
//! ONNX, run the shape pass or rebuild the plan: the decisions recorded here are
//! the decisions the session executes.
//!
//! # Validation
//!
//! A decoded plan is checked against the graph and the weight table before it is
//! returned: one slot per graph value with the same type and byte count, weight
//! ids in range with matching payload lengths, arena offsets aligned and inside
//! the arena, interface names in the declared order, every operand and result in
//! range, fusion entries that point at fused segments, and a reorder byte sum
//! that matches its list. A file that disagrees with itself is
//! [`FormatErrorKind::Inconsistent`], never a plan that runs the wrong bytes.

use crate::inference::backend::{Activation, FusedOp, SegmentEngine};
use crate::inference::ir::{Graph, SourceLoc, ValueId};
use crate::inference::ops::OpId;
use crate::inference::runtime::plan::{
    EngineFallback, Fusion, LayoutStrategy, Plan, Reorder, ReorderDirection, Segment, SegmentOp,
    Slot, SlotId, SlotSource, SLOT_ALIGNMENT,
};

use super::error::{FormatError, FormatErrorKind};
use super::graph::decode_loc;
use super::graph::decode_params;
use super::graph::encode_loc;
use super::graph::encode_params;
use super::weights::ArtifactWeight;
use super::wire::{Reader, Writer};
use super::{MAX_NODES, MAX_PARAM_AXES, MAX_VALUES};

/// Largest number of target profiles one artifact may carry.
pub const MAX_TARGETS: u64 = 8;

/// Execution precision recorded by a plan.
///
/// Format version 2 executes `f32` only. The field exists because precision is
/// plan data (ADR-0001 §4), and a future W8A8 plan must not be silently executed
/// by an F32 reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// IEEE-754 binary32 activations and weights.
    #[default]
    F32,
    /// 8-bit activations and 8-bit weights. Reserved, not executable yet:
    /// `prepare` refuses it with the measured reason (#103), and the artifact
    /// encoder rejects a plan that carries it.
    W8A8,
}

impl Precision {
    /// Stable lowercase name used by the CLI and the Python facade.
    pub const fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::W8A8 => "w8a8",
        }
    }

    /// Stable on-disk code.
    pub const fn code(self) -> u8 {
        match self {
            Precision::F32 => 0,
            Precision::W8A8 => 1,
        }
    }

    /// Parse a user-facing name.
    pub fn parse(value: &str) -> Option<Precision> {
        match value {
            "f32" | "fp32" => Some(Precision::F32),
            "w8a8" | "int8" => Some(Precision::W8A8),
            _ => None,
        }
    }

    /// Decode an on-disk code.
    pub fn from_code(code: u8) -> Option<Precision> {
        match code {
            0 => Some(Precision::F32),
            1 => Some(Precision::W8A8),
            _ => None,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One decoded target profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ArtifactPlan {
    target: String,
    precision: Precision,
    plan: Plan,
}

impl ArtifactPlan {
    /// Build a target profile from its name and an already-built plan.
    pub fn new(target: impl Into<String>, precision: Precision, plan: Plan) -> ArtifactPlan {
        ArtifactPlan {
            target: target.into(),
            precision,
            plan,
        }
    }

    /// Target name (`"cpu"`).
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Execution precision.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The plan itself.
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
}

/// Encode every target profile.
pub(super) fn encode_plans(plans: &[ArtifactPlan]) -> Result<Vec<u8>, FormatError> {
    if plans.is_empty() {
        return Err(FormatError::inconsistent(
            "PLANS section",
            "an artifact needs at least one target plan",
        ));
    }
    if plans.len() as u64 > MAX_TARGETS {
        return Err(FormatError::limit(
            "target profile count",
            plans.len() as u64,
            MAX_TARGETS,
        ));
    }
    let mut section = Writer::with_capacity(256 + plans.len() * 512);
    section.u32(plans.len() as u32);
    section.u32(0);
    for (position, artifact_plan) in plans.iter().enumerate() {
        if artifact_plan.target.is_empty() {
            return Err(FormatError::inconsistent(
                "target profile",
                format!("profile {position} has an empty target name"),
            ));
        }
        if plans[..position]
            .iter()
            .any(|other| other.target == artifact_plan.target)
        {
            return Err(FormatError::inconsistent(
                "target profile",
                format!("target `{}` appears more than once", artifact_plan.target),
            ));
        }
        encode_plan(&mut section, artifact_plan)?;
    }
    Ok(section.into_bytes())
}

/// Decode the target profiles and check them against the graph and weights.
pub(super) fn decode_plans(
    section: &[u8],
    base: u64,
    graph: &Graph,
    interface_inputs: &[String],
    interface_outputs: &[String],
    weights: &[ArtifactWeight],
) -> Result<Vec<ArtifactPlan>, FormatError> {
    let mut reader = Reader::at(section, base);
    let count = reader.count("target profile table", MAX_TARGETS, 32)?;
    if count == 0 {
        return Err(FormatError::inconsistent(
            "PLANS section",
            "the file declares no target profiles",
        ));
    }
    reader.zero_u32("target profile table reserved word")?;
    let mut plans = Vec::with_capacity(count);
    for index in 0..count {
        let plan = decode_plan(
            &mut reader,
            graph,
            interface_inputs,
            interface_outputs,
            weights,
        )
        .map_err(|error| match error.kind() {
            FormatErrorKind::MalformedAt { .. } => error,
            _ => error,
        })?;
        if plans
            .iter()
            .any(|other: &ArtifactPlan| other.target == plan.target)
        {
            return Err(FormatError::inconsistent(
                "target profile",
                format!("profile {index} repeats target `{}`", plan.target),
            ));
        }
        plans.push(plan);
    }
    reader.finish("PLANS section")?;
    Ok(plans)
}

fn encode_plan(writer: &mut Writer, artifact_plan: &ArtifactPlan) -> Result<(), FormatError> {
    let plan = &artifact_plan.plan;
    writer.string(&artifact_plan.target)?;
    writer.u8(artifact_plan.precision.code());
    writer.u8(layout_code(plan.layout));
    writer.u64(plan.arena_bytes as u64);
    writer.u64(plan.peak_bytes as u64);

    let slot_count = u32::try_from(plan.slots.len()).map_err(|_| {
        FormatError::limit("plan slot count", plan.slots.len() as u64, u32::MAX as u64)
    })?;
    writer.u32(slot_count);
    for slot in &plan.slots {
        match slot.source {
            SlotSource::Arena => {
                writer.u8(0);
            }
            SlotSource::Weight(id) => {
                writer.u8(1);
                writer.u32(id.index() as u32);
            }
        }
        writer.u64(slot.offset as u64);
        writer.u64(slot.bytes as u64);
    }
    encode_slot_ids(writer, &plan.inputs, "plan inputs")?;
    encode_slot_ids(writer, &plan.outputs, "plan outputs")?;

    let segment_count = u32::try_from(plan.segments.len()).map_err(|_| {
        FormatError::limit(
            "plan segment count",
            plan.segments.len() as u64,
            u32::MAX as u64,
        )
    })?;
    writer.u32(segment_count);
    for segment in &plan.segments {
        encode_segment(writer, segment)?;
    }

    let fusion_count = u32::try_from(plan.fusions.len()).map_err(|_| {
        FormatError::limit("fusion count", plan.fusions.len() as u64, u32::MAX as u64)
    })?;
    writer.u32(fusion_count);
    for fusion in &plan.fusions {
        writer.u32(fusion.segment as u32);
        encode_fused(writer, &fusion.fused)?;
    }

    let reorder_count = u32::try_from(plan.reorders.len()).map_err(|_| {
        FormatError::limit("reorder count", plan.reorders.len() as u64, u32::MAX as u64)
    })?;
    writer.u32(reorder_count);
    for reorder in &plan.reorders {
        writer.u32(reorder.segment as u32);
        writer.u32(reorder.slot.index() as u32);
        writer.u8(reorder_direction_code(reorder.direction));
        writer.u64(reorder.bytes as u64);
    }
    writer.u64(plan.reorder_bytes as u64);

    let fallback_count = u32::try_from(plan.fallbacks.len()).map_err(|_| {
        FormatError::limit(
            "fallback count",
            plan.fallbacks.len() as u64,
            u32::MAX as u64,
        )
    })?;
    writer.u32(fallback_count);
    for fallback in &plan.fallbacks {
        writer.u32(fallback.segment as u32);
        writer.u16(fallback.op.code());
        writer.string(&fallback.reason)?;
    }
    Ok(())
}

fn decode_plan(
    reader: &mut Reader<'_>,
    graph: &Graph,
    interface_inputs: &[String],
    interface_outputs: &[String],
    weights: &[ArtifactWeight],
) -> Result<ArtifactPlan, FormatError> {
    let target = reader.string("target profile name")?;
    if target.is_empty() {
        return Err(FormatError::malformed_at(
            "PLANS",
            reader.pos_abs(),
            "target profile name is empty",
        ));
    }
    let precision_code = reader.u8("target precision")?;
    let precision = Precision::from_code(precision_code).ok_or_else(|| {
        FormatError::new(FormatErrorKind::UnsupportedPrecision {
            found: precision_code,
            supported: "f32 (code 0)",
        })
    })?;
    let layout = layout_from_code(reader.u8("target layout")?)?;
    let arena_bytes = reader.u64("target arena bytes")?;
    let peak_bytes = reader.u64("target peak bytes")?;
    if peak_bytes > arena_bytes {
        return Err(FormatError::inconsistent(
            "plan memory",
            format!(
                "target `{target}` has peak {peak_bytes} bytes above its {arena_bytes}-byte arena"
            ),
        ));
    }
    let arena_bytes = usize::try_from(arena_bytes)
        .map_err(|_| FormatError::limit("plan arena bytes", arena_bytes, usize::MAX as u64))?;

    let slot_count = reader.count("plan slot table", MAX_VALUES, 17)?;
    if slot_count != graph.value_count() {
        return Err(FormatError::inconsistent(
            "plan slot table",
            format!(
                "target `{target}` has {slot_count} slots but the graph has {} values",
                graph.value_count()
            ),
        ));
    }
    let mut slots = Vec::with_capacity(slot_count);
    for index in 0..slot_count {
        let value = &graph.values()[index];
        let here = reader.pos_abs();
        let source_kind = reader.enum_code(format!("slot {index} source"), 1)?;
        let source = match source_kind {
            0 => SlotSource::Arena,
            1 => {
                let id = crate::inference::model::WeightId::new(
                    reader.u32(format!("slot {index} weight id"))?,
                );
                if id.index() >= weights.len() {
                    return Err(FormatError::malformed_at(
                        "PLANS",
                        here,
                        format!(
                            "slot {index} names weight {}, but the weight table has {} entries",
                            id.index(),
                            weights.len()
                        ),
                    ));
                }
                SlotSource::Weight(id)
            }
            other => {
                return Err(FormatError::malformed_at(
                    "PLANS",
                    here,
                    format!("slot {index} source kind {other} is not defined"),
                ))
            }
        };
        let slot_offset = reader.u64(format!("slot {index} offset"))?;
        let slot_bytes = reader.u64(format!("slot {index} bytes"))?;
        let expected_bytes = graph_bytes(value)?;
        if slot_bytes != expected_bytes {
            return Err(FormatError::inconsistent(
                "plan slot table",
                format!(
                    "{} in target `{target}` is {slot_bytes} bytes but its type needs \
                     {expected_bytes}",
                    value.id
                ),
            ));
        }
        let slot_offset_usize = usize::try_from(slot_offset)
            .map_err(|_| FormatError::limit("slot offset", slot_offset, usize::MAX as u64))?;
        let slot_bytes_usize = usize::try_from(slot_bytes)
            .map_err(|_| FormatError::limit("slot bytes", slot_bytes, usize::MAX as u64))?;
        match source {
            SlotSource::Arena => {
                let offset = slot_offset_usize;
                let bytes = slot_bytes_usize;
                if offset % SLOT_ALIGNMENT != 0 {
                    return Err(FormatError::inconsistent(
                        "plan slot table",
                        format!(
                            "{} in target `{target}` starts at byte {offset}, which is not \
                             {SLOT_ALIGNMENT}-byte aligned",
                            value.id
                        ),
                    ));
                }
                let end = offset.checked_add(bytes).ok_or_else(|| {
                    FormatError::inconsistent(
                        "plan slot table",
                        format!("{} in target `{target}` overflows its range", value.id),
                    )
                })?;
                if end > arena_bytes {
                    return Err(FormatError::inconsistent(
                        "plan slot table",
                        format!(
                            "{} in target `{target}` ends at byte {end}, past the \
                             {arena_bytes}-byte arena",
                            value.id
                        ),
                    ));
                }
            }
            SlotSource::Weight(id) => {
                if slot_offset != 0 {
                    return Err(FormatError::inconsistent(
                        "plan slot table",
                        format!(
                            "weight slot {} in target `{target}` has arena offset {slot_offset}",
                            value.id
                        ),
                    ));
                }
                let entry = &weights[id.index()];
                let dtype = value.ty.dtype;
                if entry.dtype() != dtype || entry.len() != slot_bytes {
                    return Err(FormatError::inconsistent(
                        "plan slot table",
                        format!(
                            "weight slot {} in target `{target}` is {dtype} {slot_bytes} bytes, \
                             but weight {} is {} {} bytes",
                            value.id,
                            id.index(),
                            entry.dtype(),
                            entry.len()
                        ),
                    ));
                }
            }
        }
        slots.push(Slot {
            value: value.id,
            name: value.name.clone(),
            ty: value.ty.clone(),
            source,
            offset: slot_offset_usize,
            bytes: slot_bytes_usize,
        });
    }

    let inputs = decode_slot_ids(reader, "plan inputs", slots.len())?;
    let outputs = decode_slot_ids(reader, "plan outputs", slots.len())?;
    check_interface_slots(&slots, &inputs, interface_inputs, true, &target)?;
    check_interface_slots(&slots, &outputs, interface_outputs, false, &target)?;

    let segment_count = reader.count("plan segment table", MAX_NODES, 32)?;
    let mut segments = Vec::with_capacity(segment_count);
    for index in 0..segment_count {
        segments.push(decode_segment(reader, slots.len(), index)?);
    }

    let fusion_count = reader.count("fusion table", MAX_NODES, 8)?;
    let mut fusions = Vec::with_capacity(fusion_count);
    for index in 0..fusion_count {
        let segment = reader.u32(format!("fusion {index} segment"))? as usize;
        let fused = decode_fused(reader, index)?;
        let candidate = segments.get(segment).ok_or_else(|| {
            FormatError::inconsistent(
                "fusion table",
                format!(
                    "fusion {index} names segment {segment}, but the plan has {} segments",
                    segments.len()
                ),
            )
        })?;
        if candidate.fused() != Some(&fused) {
            return Err(FormatError::inconsistent(
                "fusion table",
                format!("fusion {index} does not match segment {segment}"),
            ));
        }
        if let Some(previous) = fusions.last() {
            let previous: &Fusion = previous;
            if previous.segment > segment {
                return Err(FormatError::inconsistent(
                    "fusion table",
                    "fusions are not in segment order",
                ));
            }
        }
        fusions.push(Fusion { segment, fused });
    }

    let reorder_count = reader.count("reorder table", MAX_NODES, 12)?;
    let mut reorders = Vec::with_capacity(reorder_count);
    let mut reorder_bytes = 0usize;
    for index in 0..reorder_count {
        let segment = reader.u32(format!("reorder {index} segment"))? as usize;
        let slot = SlotId::new(reader.u32(format!("reorder {index} slot"))?);
        let direction = reorder_direction_from_code(
            reader.enum_code(format!("reorder {index} direction"), 1)?,
        )?;
        let bytes = reader.u64(format!("reorder {index} bytes"))?;
        if segment >= segments.len() {
            return Err(FormatError::inconsistent(
                "reorder table",
                format!(
                    "reorder {index} names segment {segment}, but the plan has {} segments",
                    segments.len()
                ),
            ));
        }
        if slot.index() >= slots.len() {
            return Err(FormatError::inconsistent(
                "reorder table",
                format!(
                    "reorder {index} names slot {}, but the plan has {} slots",
                    slot.index(),
                    slots.len()
                ),
            ));
        }
        let bytes = usize::try_from(bytes)
            .map_err(|_| FormatError::limit("reorder bytes", bytes, usize::MAX as u64))?;
        reorder_bytes = reorder_bytes.checked_add(bytes).ok_or_else(|| {
            FormatError::inconsistent("reorder table", "the byte sum overflows usize")
        })?;
        reorders.push(Reorder {
            segment,
            slot,
            direction,
            bytes,
        });
    }
    let declared_reorder_bytes = reader.u64("total reorder bytes")?;
    if declared_reorder_bytes != reorder_bytes as u64 {
        return Err(FormatError::inconsistent(
            "reorder table",
            format!(
                "the header declares {declared_reorder_bytes} reorder bytes but the list sums to \
                 {reorder_bytes}"
            ),
        ));
    }

    let fallback_count = reader.count("fallback table", MAX_NODES, 9)?;
    let mut fallbacks = Vec::with_capacity(fallback_count);
    for index in 0..fallback_count {
        let segment = reader.u32(format!("fallback {index} segment"))? as usize;
        let code = reader.u16(format!("fallback {index} op"))?;
        let op = OpId::from_u16(code)
            .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
        let reason = reader.string(format!("fallback {index} reason"))?;
        let candidate = segments.get(segment).ok_or_else(|| {
            FormatError::inconsistent(
                "fallback table",
                format!(
                    "fallback {index} names segment {segment}, but the plan has {} segments",
                    segments.len()
                ),
            )
        })?;
        if candidate.op_id() != op {
            return Err(FormatError::inconsistent(
                "fallback table",
                format!(
                    "fallback {index} names {op} but segment {segment} runs {}",
                    candidate.op_id()
                ),
            ));
        }
        fallbacks.push(EngineFallback {
            segment,
            op,
            reason,
        });
    }

    Ok(ArtifactPlan {
        target,
        precision,
        plan: Plan {
            slots,
            inputs,
            outputs,
            arena_bytes,
            peak_bytes: usize::try_from(peak_bytes).map_err(|_| {
                FormatError::limit("plan peak bytes", peak_bytes, usize::MAX as u64)
            })?,
            segments,
            fallbacks,
            fusions,
            layout,
            reorders,
            reorder_bytes,
        },
    })
}

fn encode_segment(writer: &mut Writer, segment: &Segment) -> Result<(), FormatError> {
    match &segment.op {
        SegmentOp::Op(op) => {
            writer.u8(0);
            writer.u16(op.code());
        }
        SegmentOp::Fused(fused) => {
            writer.u8(1);
            writer.u16(fused.anchor.code());
        }
    }
    writer.u8(engine_code(segment.engine));
    encode_slot_ids(writer, &segment.inputs, "segment inputs")?;
    encode_slot_ids(writer, &segment.outputs, "segment outputs")?;
    encode_params(writer, &segment.params)?;
    encode_loc(writer, &segment.loc)?;
    if let SegmentOp::Fused(fused) = &segment.op {
        encode_fused(writer, fused)?;
    }
    Ok(())
}

fn decode_segment(
    reader: &mut Reader<'_>,
    slot_count: usize,
    index: usize,
) -> Result<Segment, FormatError> {
    let kind = reader.enum_code(format!("segment {index} kind"), 1)?;
    let code = reader.u16(format!("segment {index} op"))?;
    let anchor = OpId::from_u16(code)
        .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
    let engine = engine_from_code(reader.enum_code(format!("segment {index} engine"), 1)?)?;
    let inputs = decode_slot_ids(reader, &format!("segment {index} inputs"), slot_count)?;
    let outputs = decode_slot_ids(reader, &format!("segment {index} outputs"), slot_count)?;
    if outputs.is_empty() {
        return Err(FormatError::inconsistent(
            "plan segment table",
            format!("segment {index} has no outputs"),
        ));
    }
    let params = decode_params(reader, index)?;
    let loc = decode_loc(reader, index)?;
    let op = match kind {
        0 => SegmentOp::Op(anchor),
        1 => {
            let fused = decode_fused(reader, index)?;
            if let Some(position) = fused.sum {
                if position >= inputs.len() {
                    return Err(FormatError::inconsistent(
                        "plan segment table",
                        format!(
                            "segment {index} names residual operand {position} but has {} inputs",
                            inputs.len()
                        ),
                    ));
                }
            }
            if fused.anchor != anchor {
                return Err(FormatError::inconsistent(
                    "plan segment table",
                    format!(
                        "segment {index} names anchor {anchor} but its fused form anchors {}",
                        fused.anchor
                    ),
                ));
            }
            SegmentOp::Fused(fused)
        }
        other => {
            return Err(FormatError::malformed_at(
                "PLANS",
                reader.pos_abs(),
                format!("segment {index} kind {other} is not defined"),
            ))
        }
    };
    Ok(Segment {
        op,
        engine,
        inputs,
        outputs,
        params,
        loc,
    })
}

fn encode_fused(writer: &mut Writer, fused: &FusedOp) -> Result<(), FormatError> {
    writer.u16(fused.anchor.code());
    match fused.activation {
        None => writer.u8(0),
        Some(activation) => {
            writer.u8(1);
            writer.u8(activation_kind(activation));
            if let Activation::Clip { low, high } = activation {
                writer.f32(low);
                writer.f32(high);
            }
        }
    }
    match fused.sum {
        None => writer.u8(0),
        Some(position) => {
            writer.u8(1);
            writer.u32(position as u32);
        }
    }
    let chain = u32::try_from(fused.chain.len()).map_err(|_| {
        FormatError::limit(
            "fused chain length",
            fused.chain.len() as u64,
            u32::MAX as u64,
        )
    })?;
    writer.u32(chain);
    for op in &fused.chain {
        writer.u16(op.code());
    }
    Ok(())
}

fn decode_fused(reader: &mut Reader<'_>, index: usize) -> Result<FusedOp, FormatError> {
    let code = reader.u16(format!("fused {index} anchor"))?;
    let anchor = OpId::from_u16(code)
        .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
    let activation = if reader.boolean(format!("fused {index} activation flag"))? {
        let kind = reader.enum_code(format!("fused {index} activation"), 5)?;
        Some(match kind {
            0 => Activation::Relu,
            1 => {
                let low = reader.f32(format!("fused {index} clip low"))?;
                let high = reader.f32(format!("fused {index} clip high"))?;
                Activation::Clip { low, high }
            }
            2 => Activation::Sigmoid,
            3 => Activation::Tanh,
            4 => Activation::Gelu,
            5 => Activation::Swish,
            other => {
                return Err(FormatError::malformed_at(
                    "PLANS",
                    reader.pos_abs(),
                    format!("activation {other} is not defined"),
                ))
            }
        })
    } else {
        None
    };
    let sum = if reader.boolean(format!("fused {index} sum flag"))? {
        Some(reader.u32(format!("fused {index} sum operand"))? as usize)
    } else {
        None
    };
    let chain_len = reader.count(format!("fused {index} chain"), MAX_PARAM_AXES, 2)?;
    let mut chain = Vec::with_capacity(chain_len);
    for _ in 0..chain_len {
        let code = reader.u16("fused chain op")?;
        let op = OpId::from_u16(code)
            .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
        chain.push(op);
    }
    if chain.len() < 2 {
        return Err(FormatError::inconsistent(
            "fusion",
            format!(
                "fused chain {index} has {} ops; a fusion replaces at least two",
                chain.len()
            ),
        ));
    }
    if chain[0] != anchor {
        return Err(FormatError::inconsistent(
            "fusion",
            format!(
                "fused chain {index} starts with {} but anchors on {anchor}",
                chain[0]
            ),
        ));
    }
    Ok(FusedOp {
        anchor,
        activation,
        sum,
        chain,
    })
}

fn encode_slot_ids(writer: &mut Writer, ids: &[SlotId], what: &str) -> Result<(), FormatError> {
    let count = u32::try_from(ids.len())
        .map_err(|_| FormatError::limit(what, ids.len() as u64, u32::MAX as u64))?;
    writer.u32(count);
    for id in ids {
        writer.u32(id.index() as u32);
    }
    Ok(())
}

fn decode_slot_ids(
    reader: &mut Reader<'_>,
    what: &str,
    slot_count: usize,
) -> Result<Vec<SlotId>, FormatError> {
    let count = reader.count(what, MAX_PARAM_AXES, 4)?;
    let mut ids = Vec::with_capacity(count);
    for position in 0..count {
        let id = SlotId::new(reader.u32(format!("{what} {position}"))?);
        if id.index() >= slot_count {
            return Err(FormatError::inconsistent(
                "plan slot ids",
                format!(
                    "{what} entry {position} names slot {}, but the plan has {slot_count} slots",
                    id.index()
                ),
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

fn check_interface_slots(
    slots: &[Slot],
    ids: &[SlotId],
    names: &[String],
    inputs: bool,
    target: &str,
) -> Result<(), FormatError> {
    if ids.len() != names.len() {
        return Err(FormatError::inconsistent(
            "plan interface",
            format!(
                "target `{target}` declares {} {} slots but the graph interface has {}",
                ids.len(),
                if inputs { "input" } else { "output" },
                names.len()
            ),
        ));
    }
    for (position, (id, name)) in ids.iter().zip(names).enumerate() {
        match slots[id.index()].name.as_deref() {
            Some(found) if found == name => {}
            other => {
                return Err(FormatError::inconsistent(
                    "plan interface",
                    format!(
                        "target `{target}` {} {position} names `{name}` but slot {id} is {:?}",
                        if inputs { "input" } else { "output" },
                        other
                    ),
                ))
            }
        }
    }
    Ok(())
}

fn graph_bytes(value: &crate::inference::ir::Value) -> Result<u64, FormatError> {
    let shape = value.ty.static_shape().ok_or_else(|| {
        FormatError::inconsistent(
            "plan slot table",
            format!(
                "{} is {}; a serialized plan needs every extent static",
                value.id, value.ty
            ),
        )
    })?;
    let elements = shape
        .numel()
        .map_err(|error| FormatError::contract("plan slot bytes", error))?;
    elements
        .checked_mul(value.ty.dtype.size_bytes() as u64)
        .ok_or_else(|| {
            FormatError::inconsistent(
                "plan slot bytes",
                format!("{} overflows its byte count", value.id),
            )
        })
}

fn engine_code(engine: SegmentEngine) -> u8 {
    match engine {
        SegmentEngine::Scalar => 0,
        SegmentEngine::OneDnn => 1,
    }
}

fn engine_from_code(code: u8) -> Result<SegmentEngine, FormatError> {
    match code {
        0 => Ok(SegmentEngine::Scalar),
        1 => Ok(SegmentEngine::OneDnn),
        other => Err(FormatError::malformed_at(
            "PLANS",
            0,
            format!("engine code {other} is not defined"),
        )),
    }
}

fn layout_code(layout: LayoutStrategy) -> u8 {
    match layout {
        LayoutStrategy::Plain => 0,
        LayoutStrategy::Chained => 1,
    }
}

fn layout_from_code(code: u8) -> Result<LayoutStrategy, FormatError> {
    match code {
        0 => Ok(LayoutStrategy::Plain),
        1 => Ok(LayoutStrategy::Chained),
        other => Err(FormatError::malformed_at(
            "PLANS",
            0,
            format!("layout code {other} is not defined"),
        )),
    }
}

fn reorder_direction_code(direction: ReorderDirection) -> u8 {
    match direction {
        ReorderDirection::NchwToBlocked => 0,
        ReorderDirection::BlockedToNchw => 1,
    }
}

fn reorder_direction_from_code(code: u8) -> Result<ReorderDirection, FormatError> {
    match code {
        0 => Ok(ReorderDirection::NchwToBlocked),
        1 => Ok(ReorderDirection::BlockedToNchw),
        other => Err(FormatError::malformed_at(
            "PLANS",
            0,
            format!("reorder direction code {other} is not defined"),
        )),
    }
}

fn activation_kind(activation: Activation) -> u8 {
    match activation {
        Activation::Relu => 0,
        Activation::Clip { .. } => 1,
        Activation::Sigmoid => 2,
        Activation::Tanh => 3,
        Activation::Gelu => 4,
        Activation::Swish => 5,
    }
}

/// A source location, encoded so diagnostics survive a round trip.
#[allow(dead_code)]
fn _keep_source_loc_used(_: SourceLoc, _: ValueId) {}
