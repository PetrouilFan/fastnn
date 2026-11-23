//! The `GRAPH` section: the typed v3 IR, its provenance and its interface.
//!
//! # What is stored
//!
//! The graph section is the typed graph itself, not a decoder profile and not a
//! second IR. For every value it stores the id, the optional source name, the
//! dtype, every extent as a `DimExpr` constant or symbol, and the producing
//! node. For every node it stores the id, the `OpId`, operand and result ids,
//! the **complete** [`OpParams`] (all 28 fields and five enums) and the
//! [`SourceLoc`] so a runtime failure can still say
//! `node 7 (onnx "Conv", output "y")`. The graph interface is stored as names in
//! declaration order; a missing or duplicate name is an inconsistency, never a
//! guessed id.
//!
//! # Why the full `OpParams`
//!
//! Format version 1 stored only `axes`, `keep_dims` and `epsilon`. A CNN graph
//! cannot round-trip through that: convolution window geometry, pooling modes,
//! padding, resize sampling and TopK attributes would all be lost. Version 2
//! stores every field, and a decoder of a future field set must bump the
//! version rather than silently dropping an attribute.
//!
//! # Determinism
//!
//! Every table is a `Vec` in index order and every optional field has one
//! canonical absent form. Re-encoding a decoded graph is byte-identical.

use crate::inference::backend::{
    AutoPad, CoordinateTransformationMode, NearestMode, OpParams, PadMode, ResizeMode,
};
use crate::inference::dtype::DType;
use crate::inference::import::OpsetImport;
use crate::inference::ir::{
    DimExpr, Graph, Node, NodeId, SourceLoc, TensorType, Value, ValueId, MAX_OPERANDS, MAX_RANK,
};
use crate::inference::ops::OpId;
use crate::inference::shape::Symbol;

use super::error::{FormatError, FormatErrorKind};
use super::wire::{Reader, Writer};
use super::{MAX_NODES, MAX_PARAM_AXES, MAX_VALUES};

/// Provenance carried by the artifact.
///
/// The source ONNX digest lives in the fixed header (so `inspect` can print it
/// without decoding a section); everything else is here because it describes
/// the graph rather than the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactMetadata {
    /// ONNX IR version of the source model.
    pub ir_version: i64,
    /// Opset of the source model's default domain.
    pub opset: OpsetImport,
    /// Exporter name (`"pytorch"`).
    pub producer: String,
    /// Exporter version (`"2.8.0"`).
    pub producer_version: String,
    /// Source graph name.
    pub graph_name: String,
}

impl ArtifactMetadata {
    /// Read the provenance off a typed import.
    pub fn from_import(import: &crate::inference::import::ImportedGraph) -> ArtifactMetadata {
        ArtifactMetadata {
            ir_version: import.ir_version(),
            opset: import.opset().clone(),
            producer: import.producer().to_string(),
            producer_version: import.producer_version().to_string(),
            graph_name: import.graph_name().to_string(),
        }
    }
}

/// A decoded graph section: provenance, graph and interface names.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct GraphParts {
    pub(super) metadata: ArtifactMetadata,
    pub(super) graph: Graph,
    pub(super) inputs: Vec<String>,
    pub(super) outputs: Vec<String>,
}

// Smallest encoded value: id (4), name flag (1), dtype (1), rank (4), producer
// flag (1).
const MIN_VALUE_ENTRY: u64 = 11;
// Deliberately below the real node encoding (>= 100 bytes): a lower bound only
// has to be safe, and this keeps a legitimate small graph from being rejected.
const MIN_NODE_ENTRY: u64 = 32;

/// Encode the graph section.
pub(super) fn encode_graph(parts: &GraphParts) -> Result<Vec<u8>, FormatError> {
    let metadata = &parts.metadata;
    let graph = &parts.graph;
    let mut section = Writer::with_capacity(256 + graph.value_count() * 48);

    section.i64(metadata.ir_version);
    section.string(&metadata.opset.domain)?;
    section.i64(metadata.opset.version);
    section.string(&metadata.producer)?;
    section.string(&metadata.producer_version)?;
    section.string(&metadata.graph_name)?;

    let value_count = u32::try_from(graph.value_count()).map_err(|_| {
        FormatError::limit("value count", graph.value_count() as u64, u32::MAX as u64)
    })?;
    section.u32(value_count);
    for (position, value) in graph.values().iter().enumerate() {
        if value.id.index() != position {
            return Err(FormatError::inconsistent(
                "graph value table",
                format!(
                    "value table position {position} holds {}; ids must equal their position",
                    value.id
                ),
            ));
        }
        section.u32(value.id.index() as u32);
        match &value.name {
            Some(name) => {
                section.u8(1);
                section.string(name)?;
            }
            None => section.u8(0),
        }
        section.u8(value.ty.dtype.code());
        let rank = u32::try_from(value.ty.rank()).map_err(|_| {
            FormatError::limit("value rank", value.ty.rank() as u64, u32::MAX as u64)
        })?;
        section.u32(rank);
        for dim in value.ty.dims() {
            encode_dim(&mut section, *dim);
        }
        match value.producer {
            Some(producer) => {
                section.u8(1);
                section.u32(producer.index() as u32);
            }
            None => section.u8(0),
        }
    }

    let node_count = u32::try_from(graph.node_count()).map_err(|_| {
        FormatError::limit("node count", graph.node_count() as u64, u32::MAX as u64)
    })?;
    section.u32(node_count);
    for (position, node) in graph.nodes().iter().enumerate() {
        if node.id.index() != position {
            return Err(FormatError::inconsistent(
                "graph node table",
                format!(
                    "node table position {position} holds {}; ids must equal their position",
                    node.id
                ),
            ));
        }
        section.u32(node.id.index() as u32);
        section.u16(node.op.code());
        encode_handles(&mut section, &node.inputs, "node inputs")?;
        encode_handles(&mut section, &node.outputs, "node outputs")?;
        encode_params(&mut section, &node.params)?;
        encode_loc(&mut section, &node.loc)?;
    }

    encode_names(&mut section, &parts.inputs, "graph inputs")?;
    encode_names(&mut section, &parts.outputs, "graph outputs")?;
    Ok(section.into_bytes())
}

/// Decode the graph section.
pub(super) fn decode_graph(section: &[u8], base: u64) -> Result<GraphParts, FormatError> {
    let mut reader = Reader::at(section, base);

    let ir_version = reader.i64("graph ir_version")?;
    let domain = reader.string("graph opset domain")?;
    let opset_version = reader.i64("graph opset version")?;
    let producer = reader.string("graph producer")?;
    let producer_version = reader.string("graph producer_version")?;
    let graph_name = reader.string("graph name")?;

    let value_count = reader.count("graph value table", MAX_VALUES, MIN_VALUE_ENTRY)?;
    let mut values = Vec::with_capacity(value_count);
    for index in 0..value_count {
        let id = ValueId::new(reader.u32(format!("value {index} id"))?);
        if id.index() != index {
            return Err(FormatError::malformed_at(
                "GRAPH",
                reader.pos_abs().saturating_sub(4),
                format!("value table position {index} carries id {}", id.index()),
            ));
        }
        let name = if reader.boolean(format!("value {index} name flag"))? {
            Some(reader.string(format!("value {index} name"))?)
        } else {
            None
        };
        let dtype = DType::from_code(reader.u8(format!("value {index} dtype"))?, "GRAPH value")
            .map_err(|error| FormatError::new(FormatErrorKind::UnknownDType(error)))?;
        let rank = reader.count(format!("value {index} rank"), MAX_RANK as u64, 9)?;
        let mut dims = Vec::with_capacity(rank);
        for axis in 0..rank {
            dims.push(decode_dim(
                &mut reader,
                format!("value {index} axis {axis}"),
            )?);
        }
        let producer = if reader.boolean(format!("value {index} producer flag"))? {
            Some(NodeId::new(reader.u32(format!("value {index} producer"))?))
        } else {
            None
        };
        values.push(Value {
            id,
            name,
            ty: TensorType::new(dtype, dims),
            producer,
        });
    }

    let node_count = reader.count("graph node table", MAX_NODES, MIN_NODE_ENTRY)?;
    let mut nodes = Vec::with_capacity(node_count);
    for index in 0..node_count {
        let id = NodeId::new(reader.u32(format!("node {index} id"))?);
        if id.index() != index {
            return Err(FormatError::malformed_at(
                "GRAPH",
                reader.pos_abs().saturating_sub(4),
                format!("node table position {index} carries id {}", id.index()),
            ));
        }
        let code = reader.u16(format!("node {index} op code"))?;
        let op = OpId::from_u16(code)
            .ok_or_else(|| FormatError::new(FormatErrorKind::UnknownOpCode { code }))?;
        let inputs = decode_handles(&mut reader, format!("node {index} inputs"))?;
        let outputs = decode_handles(&mut reader, format!("node {index} outputs"))?;
        let params = decode_params(&mut reader, index)?;
        let loc = decode_loc(&mut reader, index)?;
        nodes.push(Node {
            id,
            op,
            inputs,
            outputs,
            params,
            loc,
        });
    }
    let inputs = decode_names(&mut reader, "graph inputs")?;
    let outputs = decode_names(&mut reader, "graph outputs")?;
    reader.finish("GRAPH section")?;

    let graph = Graph::from_parts(values, nodes)
        .map_err(|error| FormatError::contract("prepared graph", error))?;
    check_interface(&graph, &inputs, &outputs)?;
    Ok(GraphParts {
        metadata: ArtifactMetadata {
            ir_version,
            opset: OpsetImport {
                domain,
                version: opset_version,
            },
            producer,
            producer_version,
            graph_name,
        },
        graph,
        inputs,
        outputs,
    })
}

pub(super) fn check_interface(
    graph: &Graph,
    inputs: &[String],
    outputs: &[String],
) -> Result<(), FormatError> {
    for (kind, names) in [("input", inputs), ("output", outputs)] {
        for (position, name) in names.iter().enumerate() {
            if name.is_empty() {
                return Err(FormatError::inconsistent(
                    "graph interface",
                    format!("{kind} {position} has an empty name"),
                ));
            }
            let repeated = names[..position].contains(name);
            if repeated {
                return Err(FormatError::inconsistent(
                    "graph interface",
                    format!("{kind} name `{name}` appears more than once"),
                ));
            }
            let matches: Vec<&Value> = graph
                .values()
                .iter()
                .filter(|value| value.name.as_deref() == Some(name.as_str()))
                .collect();
            match matches.as_slice() {
                [] => {
                    return Err(FormatError::inconsistent(
                        "graph interface",
                        format!("{kind} `{name}` names no value in the graph"),
                    ))
                }
                [value] => {
                    if kind == "input" {
                        if let Some(producer) = value.producer {
                            return Err(FormatError::inconsistent(
                                "graph interface",
                                format!(
                                    "input `{name}` is written by {producer}, so it is not an input"
                                ),
                            ));
                        }
                    }
                }
                _ => {
                    return Err(FormatError::inconsistent(
                        "graph interface",
                        format!("{kind} name `{name}` matches more than one value"),
                    ))
                }
            }
        }
    }
    Ok(())
}

fn encode_dim(writer: &mut Writer, dim: DimExpr) {
    match dim {
        DimExpr::Const(extent) => {
            writer.u8(0);
            writer.u64(extent);
        }
        DimExpr::Symbol(symbol) => {
            writer.u8(1);
            writer.u8(symbol_code(symbol));
        }
    }
}

fn decode_dim(reader: &mut Reader<'_>, what: String) -> Result<DimExpr, FormatError> {
    match reader.enum_code(what.clone(), 1)? {
        0 => Ok(DimExpr::Const(reader.u64(format!("{what} extent"))?)),
        1 => {
            let code = reader.u8(format!("{what} symbol"))?;
            symbol_from_code(code).map(DimExpr::Symbol).ok_or_else(|| {
                FormatError::malformed_at(
                    "GRAPH",
                    reader.pos_abs().saturating_sub(1),
                    format!("{what} names symbol code {code}, which is not defined"),
                )
            })
        }
        other => Err(FormatError::malformed_at(
            "GRAPH",
            reader.pos_abs(),
            format!("{what} has dim kind {other}"),
        )),
    }
}

fn symbol_code(symbol: Symbol) -> u8 {
    match symbol {
        Symbol::Batch => 0,
        Symbol::Seq => 1,
        Symbol::Past => 2,
        Symbol::Total => 3,
        Symbol::NHeads => 4,
        Symbol::NKvHeads => 5,
        Symbol::HeadDim => 6,
    }
}

fn symbol_from_code(code: u8) -> Option<Symbol> {
    Symbol::ALL
        .into_iter()
        .find(|symbol| symbol_code(*symbol) == code)
}

fn encode_handles(writer: &mut Writer, handles: &[ValueId], what: &str) -> Result<(), FormatError> {
    let count = u32::try_from(handles.len())
        .map_err(|_| FormatError::limit(what, handles.len() as u64, u32::MAX as u64))?;
    writer.u32(count);
    for handle in handles {
        writer.u32(handle.index() as u32);
    }
    Ok(())
}

fn decode_handles(reader: &mut Reader<'_>, what: String) -> Result<Vec<ValueId>, FormatError> {
    let count = reader.count(what, MAX_OPERANDS as u64, 4)?;
    let mut handles = Vec::with_capacity(count);
    for position in 0..count {
        handles.push(ValueId::new(reader.u32(format!("operand {position}"))?));
    }
    Ok(handles)
}

fn encode_names(writer: &mut Writer, names: &[String], what: &str) -> Result<(), FormatError> {
    let count = u32::try_from(names.len())
        .map_err(|_| FormatError::limit(what, names.len() as u64, u32::MAX as u64))?;
    writer.u32(count);
    for name in names {
        writer.string(name)?;
    }
    Ok(())
}

fn decode_names(reader: &mut Reader<'_>, what: &str) -> Result<Vec<String>, FormatError> {
    let count = reader.count(what, MAX_VALUES, 5)?;
    let mut names = Vec::with_capacity(count);
    for position in 0..count {
        names.push(reader.string(format!("{what} {position}"))?);
    }
    Ok(names)
}

pub(super) fn encode_loc(writer: &mut Writer, loc: &SourceLoc) -> Result<(), FormatError> {
    let node_index = u32::try_from(loc.node_index).map_err(|_| {
        FormatError::limit("source node index", loc.node_index as u64, u32::MAX as u64)
    })?;
    writer.u32(node_index);
    writer.string(&loc.op_type)?;
    match &loc.output {
        Some(output) => {
            writer.u8(1);
            writer.string(output)?;
        }
        None => writer.u8(0),
    }
    Ok(())
}

pub(super) fn decode_loc(reader: &mut Reader<'_>, index: usize) -> Result<SourceLoc, FormatError> {
    let node_index = reader.u32(format!("node {index} source index"))? as usize;
    let op_type = reader.string(format!("node {index} source op"))?;
    let output = if reader.boolean(format!("node {index} output flag"))? {
        Some(reader.string(format!("node {index} source output"))?)
    } else {
        None
    };
    Ok(SourceLoc {
        node_index,
        op_type,
        output,
    })
}

pub(super) fn encode_params(writer: &mut Writer, params: &OpParams) -> Result<(), FormatError> {
    encode_i64s(writer, &params.axes, "axes")?;
    writer.u8(u8::from(params.keep_dims));
    writer.f32(params.epsilon);
    encode_i64s(writer, &params.strides, "strides")?;
    encode_i64s(writer, &params.pads, "pads")?;
    encode_i64s(writer, &params.dilations, "dilations")?;
    encode_i64s(writer, &params.kernel_shape, "kernel_shape")?;
    writer.i64(params.group);
    writer.u8(u8::from(params.ceil_mode));
    writer.u8(u8::from(params.count_include_pad));
    writer.u8(auto_pad_code(params.auto_pad));
    writer.u8(resize_mode_code(params.resize_mode));
    writer.u8(coordinate_mode_code(params.coordinate_transformation_mode));
    writer.u8(nearest_mode_code(params.nearest_mode));
    encode_f32s(writer, &params.resize_scales, "resize_scales")?;
    encode_i64s(writer, &params.resize_sizes, "resize_sizes")?;
    writer.u8(pad_mode_code(params.pad_mode));
    writer.f32(params.pad_value);
    encode_optional_f32(writer, params.clip_min);
    encode_optional_f32(writer, params.clip_max);
    writer.i64(params.topk_axis);
    writer.i64(params.topk_k);
    writer.u8(u8::from(params.largest));
    writer.u8(u8::from(params.sorted));
    encode_i64s(writer, &params.split_sizes, "split_sizes")?;
    encode_i64s(writer, &params.slice_starts, "slice_starts")?;
    encode_i64s(writer, &params.slice_ends, "slice_ends")?;
    encode_i64s(writer, &params.slice_steps, "slice_steps")?;
    Ok(())
}

pub(super) fn decode_params(
    reader: &mut Reader<'_>,
    index: usize,
) -> Result<OpParams, FormatError> {
    let axes = decode_i64s(reader, format!("node {index} axes"))?;
    let keep_dims = reader.boolean(format!("node {index} keep_dims"))?;
    let epsilon = reader.f32(format!("node {index} epsilon"))?;
    let strides = decode_i64s(reader, format!("node {index} strides"))?;
    let pads = decode_i64s(reader, format!("node {index} pads"))?;
    let dilations = decode_i64s(reader, format!("node {index} dilations"))?;
    let kernel_shape = decode_i64s(reader, format!("node {index} kernel_shape"))?;
    let group = reader.i64(format!("node {index} group"))?;
    let ceil_mode = reader.boolean(format!("node {index} ceil_mode"))?;
    let count_include_pad = reader.boolean(format!("node {index} count_include_pad"))?;
    let auto_pad = auto_pad_from_code(reader.enum_code(format!("node {index} auto_pad"), 3)?)?;
    let resize_mode =
        resize_mode_from_code(reader.enum_code(format!("node {index} resize_mode"), 2)?)?;
    let coordinate_transformation_mode = coordinate_mode_from_code(
        reader.enum_code(format!("node {index} coordinate_transformation_mode"), 4)?,
    )?;
    let nearest_mode =
        nearest_mode_from_code(reader.enum_code(format!("node {index} nearest_mode"), 3)?)?;
    let resize_scales = decode_f32s(reader, format!("node {index} resize_scales"))?;
    let resize_sizes = decode_i64s(reader, format!("node {index} resize_sizes"))?;
    let pad_mode = pad_mode_from_code(reader.enum_code(format!("node {index} pad_mode"), 3)?)?;
    let pad_value = reader.f32(format!("node {index} pad_value"))?;
    let clip_min = decode_optional_f32(reader, format!("node {index} clip_min"))?;
    let clip_max = decode_optional_f32(reader, format!("node {index} clip_max"))?;
    let topk_axis = reader.i64(format!("node {index} topk_axis"))?;
    let topk_k = reader.i64(format!("node {index} topk_k"))?;
    let largest = reader.boolean(format!("node {index} largest"))?;
    let sorted = reader.boolean(format!("node {index} sorted"))?;
    let split_sizes = decode_i64s(reader, format!("node {index} split_sizes"))?;
    let slice_starts = decode_i64s(reader, format!("node {index} slice_starts"))?;
    let slice_ends = decode_i64s(reader, format!("node {index} slice_ends"))?;
    let slice_steps = decode_i64s(reader, format!("node {index} slice_steps"))?;
    Ok(OpParams {
        axes,
        keep_dims,
        epsilon,
        strides,
        pads,
        dilations,
        kernel_shape,
        group,
        ceil_mode,
        count_include_pad,
        auto_pad,
        resize_mode,
        coordinate_transformation_mode,
        nearest_mode,
        resize_scales,
        resize_sizes,
        pad_mode,
        pad_value,
        clip_min,
        clip_max,
        topk_axis,
        topk_k,
        largest,
        sorted,
        split_sizes,
        slice_starts,
        slice_ends,
        slice_steps,
    })
}

fn encode_i64s(writer: &mut Writer, values: &[i64], what: &str) -> Result<(), FormatError> {
    let count = u32::try_from(values.len())
        .map_err(|_| FormatError::limit(what, values.len() as u64, u32::MAX as u64))?;
    writer.u32(count);
    for value in values {
        writer.i64(*value);
    }
    Ok(())
}

fn decode_i64s(reader: &mut Reader<'_>, what: String) -> Result<Vec<i64>, FormatError> {
    let count = reader.count(what, MAX_PARAM_AXES, 8)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.i64("i64 entry")?);
    }
    Ok(values)
}

fn encode_f32s(writer: &mut Writer, values: &[f32], what: &str) -> Result<(), FormatError> {
    let count = u32::try_from(values.len())
        .map_err(|_| FormatError::limit(what, values.len() as u64, u32::MAX as u64))?;
    writer.u32(count);
    for value in values {
        writer.f32(*value);
    }
    Ok(())
}

fn decode_f32s(reader: &mut Reader<'_>, what: String) -> Result<Vec<f32>, FormatError> {
    let count = reader.count(what, MAX_PARAM_AXES, 4)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(reader.f32("f32 entry")?);
    }
    Ok(values)
}

fn encode_optional_f32(writer: &mut Writer, value: Option<f32>) {
    match value {
        Some(value) => {
            writer.u8(1);
            writer.f32(value);
        }
        None => {
            writer.u8(0);
            writer.f32(0.0);
        }
    }
}

fn decode_optional_f32(reader: &mut Reader<'_>, what: String) -> Result<Option<f32>, FormatError> {
    let present = reader.boolean(format!("{what} flag"))?;
    let bits = reader.u32(format!("{what} value"))?;
    match (present, bits) {
        (false, 0) => Ok(None),
        (false, _) => Err(FormatError::malformed_at(
            "GRAPH",
            reader.pos_abs().saturating_sub(4),
            format!("{what} is absent but its value bits are {bits:#010x}"),
        )),
        (true, _) => Ok(Some(f32::from_bits(bits))),
    }
}

fn auto_pad_code(value: AutoPad) -> u8 {
    match value {
        AutoPad::NotSet => 0,
        AutoPad::SameUpper => 1,
        AutoPad::SameLower => 2,
        AutoPad::Valid => 3,
    }
}

fn auto_pad_from_code(code: u8) -> Result<AutoPad, FormatError> {
    match code {
        0 => Ok(AutoPad::NotSet),
        1 => Ok(AutoPad::SameUpper),
        2 => Ok(AutoPad::SameLower),
        3 => Ok(AutoPad::Valid),
        other => Err(FormatError::malformed_at(
            "GRAPH",
            0,
            format!("auto_pad code {other} is not defined"),
        )),
    }
}

fn resize_mode_code(value: ResizeMode) -> u8 {
    match value {
        ResizeMode::Nearest => 0,
        ResizeMode::Linear => 1,
        ResizeMode::Cubic => 2,
    }
}

fn resize_mode_from_code(code: u8) -> Result<ResizeMode, FormatError> {
    match code {
        0 => Ok(ResizeMode::Nearest),
        1 => Ok(ResizeMode::Linear),
        2 => Ok(ResizeMode::Cubic),
        other => Err(FormatError::malformed_at(
            "GRAPH",
            0,
            format!("resize_mode code {other} is not defined"),
        )),
    }
}

fn coordinate_mode_code(value: CoordinateTransformationMode) -> u8 {
    match value {
        CoordinateTransformationMode::HalfPixel => 0,
        CoordinateTransformationMode::PytorchHalfPixel => 1,
        CoordinateTransformationMode::AlignCorners => 2,
        CoordinateTransformationMode::Asymmetric => 3,
        CoordinateTransformationMode::TfCropAndResize => 4,
    }
}

fn coordinate_mode_from_code(code: u8) -> Result<CoordinateTransformationMode, FormatError> {
    match code {
        0 => Ok(CoordinateTransformationMode::HalfPixel),
        1 => Ok(CoordinateTransformationMode::PytorchHalfPixel),
        2 => Ok(CoordinateTransformationMode::AlignCorners),
        3 => Ok(CoordinateTransformationMode::Asymmetric),
        4 => Ok(CoordinateTransformationMode::TfCropAndResize),
        other => Err(FormatError::malformed_at(
            "GRAPH",
            0,
            format!("coordinate_transformation_mode code {other} is not defined"),
        )),
    }
}

fn nearest_mode_code(value: NearestMode) -> u8 {
    match value {
        NearestMode::RoundPreferFloor => 0,
        NearestMode::RoundPreferCeil => 1,
        NearestMode::Floor => 2,
        NearestMode::Ceil => 3,
    }
}

fn nearest_mode_from_code(code: u8) -> Result<NearestMode, FormatError> {
    match code {
        0 => Ok(NearestMode::RoundPreferFloor),
        1 => Ok(NearestMode::RoundPreferCeil),
        2 => Ok(NearestMode::Floor),
        3 => Ok(NearestMode::Ceil),
        other => Err(FormatError::malformed_at(
            "GRAPH",
            0,
            format!("nearest_mode code {other} is not defined"),
        )),
    }
}

fn pad_mode_code(value: PadMode) -> u8 {
    match value {
        PadMode::Constant => 0,
        PadMode::Reflect => 1,
        PadMode::Edge => 2,
        PadMode::Wrap => 3,
    }
}

fn pad_mode_from_code(code: u8) -> Result<PadMode, FormatError> {
    match code {
        0 => Ok(PadMode::Constant),
        1 => Ok(PadMode::Reflect),
        2 => Ok(PadMode::Edge),
        3 => Ok(PadMode::Wrap),
        other => Err(FormatError::malformed_at(
            "GRAPH",
            0,
            format!("pad_mode code {other} is not defined"),
        )),
    }
}
