//! ONNX → typed-IR bridge.
//!
//! The importer's [`GraphIr`](super::GraphIr) tracks shapes with
//! [`Dim::Unknown`](super::Dim), because it can describe a value it could not
//! prove. The v3 typed graph ([`crate::inference::ir`]) cannot: every extent is
//! either a constant or a named symbol, and every op output is resolved by
//! [`infer_shapes`](crate::inference::ir::infer_shapes). This module is the one
//! conversion, and it reports — rather than guesses — the places where the two
//! contracts differ.
//!
//! # What it does
//!
//! 1. Every importer value becomes a typed value with the same name and dtype.
//!    A fixed extent becomes [`DimExpr::Const`], a symbol keeps its name, and an
//!    unproven extent is an error that names the value and the reason.
//! 2. Every importer node becomes a typed node with the same [`OpId`], operands,
//!    [`SourceLoc`] and attributes. Intermediate node outputs start as rank-0
//!    placeholders and are overwritten by the shape pass; they are never read
//!    before their producer, because the importer's node order is topological.
//! 3. [`infer_shapes`] runs once. A failure is the typed pass's own node-named
//!    error, wrapped in an [`ImportError`] so a caller sees one failure type.
//!
//! # Why attributes can differ from the artifact's
//!
//! The 3.0 artifact format version 1 stores `axes`, `keep_dims` and `epsilon`.
//! `Reshape`, `Slice` and `Split` need their constant operands visible to the
//! shape pass, which reads [`OpParams`] and never a tensor payload. The importer
//! carries those resolved bounds in a side table (`GraphParts::typed_params`)
//! so the typed graph sees them without changing what an artifact stores.

use crate::inference::backend::OpParams;
use crate::inference::error::Fn3Error;
use crate::inference::ir::{infer_shapes, DimExpr, Graph, SourceLoc, TensorType, ValueId};
#[cfg(test)]
use crate::inference::ops::OpId;

use super::error::{ImportError, ImportProblemKind};
use super::graph::{
    Dim, GraphIr, GraphParts, ImportSummary, ImportWarning, OpsetImport, ValueOrigin, ValueShape,
};
use super::WeightSet;

/// An ONNX model imported into the v3 typed graph, with every extent resolved.
///
/// This is the CNN path: it stops before the decoder architecture profile, so a
/// graph that is not a GPT-2-shaped decoder still imports. [`import_onnx_model`]
/// remains the decoder entry point and keeps its geometry contract.
///
/// [`import_onnx_model`]: super::import_onnx_model
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedGraph {
    graph: Graph,
    weights: WeightSet,
    summary: ImportSummary,
    warnings: Vec<ImportWarning>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
    ir_version: i64,
    producer: String,
    producer_version: String,
    opset: OpsetImport,
    graph_name: String,
}

impl ImportedGraph {
    /// Build an in-memory model with no ONNX file.
    ///
    /// Crate-internal and test-only: the unit tests for the v3 runtime need a
    /// typed graph plus a weight table, and synthesising an ONNX file for a
    /// three-node graph would test the encoder instead of the runtime. The
    /// summary is filled from the graph so a caller that asserts on it sees
    /// consistent numbers.
    #[cfg(test)]
    pub(crate) fn for_tests(
        graph: Graph,
        weights: WeightSet,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<ImportedGraph, Fn3Error> {
        let mut ops = Vec::new();
        for node in graph.nodes() {
            match ops
                .iter_mut()
                .find(|(op, _): &&mut (OpId, usize)| *op == node.op)
            {
                Some((_, count)) => *count += 1,
                None => ops.push((node.op, 1usize)),
            }
        }
        ops.sort_unstable_by_key(|(op, _)| op.code());
        let summary = ImportSummary {
            onnx_nodes: graph.node_count(),
            fn3_nodes: graph.node_count(),
            values: graph.value_count(),
            initializers: weights.len(),
            weight_bytes: weights.total_bytes(),
            ops,
            ..ImportSummary::default()
        };
        Ok(ImportedGraph {
            graph,
            weights,
            summary,
            warnings: Vec::new(),
            inputs,
            outputs,
            ir_version: 7,
            producer: "fastnn-tests".to_string(),
            producer_version: "0".to_string(),
            opset: OpsetImport {
                domain: String::new(),
                version: 12,
            },
            graph_name: "test".to_string(),
        })
    }

    /// The typed graph, with every value's type resolved by shape inference.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The weight table, in file order.
    pub fn weights(&self) -> &WeightSet {
        &self.weights
    }

    /// Counts and censuses of the source graph.
    pub fn summary(&self) -> &ImportSummary {
        &self.summary
    }

    /// Declared shapes that contradicted the inferred shape.
    pub fn warnings(&self) -> &[ImportWarning] {
        &self.warnings
    }

    /// Graph inputs, in file order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// Graph outputs, in file order.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// A value by exact source name.
    pub fn lookup(&self, name: &str) -> Option<ValueId> {
        self.graph
            .values()
            .iter()
            .find(|value| value.name.as_deref() == Some(name))
            .map(|value| value.id)
    }

    /// `ModelProto.ir_version`.
    pub fn ir_version(&self) -> i64 {
        self.ir_version
    }

    /// `ModelProto.producer_name`.
    pub fn producer(&self) -> &str {
        &self.producer
    }

    /// `ModelProto.producer_version`.
    pub fn producer_version(&self) -> &str {
        &self.producer_version
    }

    /// Default-domain opset.
    pub fn opset(&self) -> &OpsetImport {
        &self.opset
    }

    /// `GraphProto.name`.
    pub fn graph_name(&self) -> &str {
        &self.graph_name
    }
}

/// Convert one decoded graph into [`ImportedGraph`].
pub(crate) fn build(parts: GraphParts) -> Result<ImportedGraph, ImportError> {
    let GraphParts {
        ir_version,
        producer,
        producer_version,
        opset,
        graph_name,
        graph,
        weights,
        summary,
        warnings,
        typed_params,
    } = parts;
    let (mut typed, expected) = convert(&graph, &typed_params)?;
    infer_shapes(&mut typed).map_err(shape_error)?;
    // The importer and the typed pass are two independent implementations of
    // the same ONNX rules. Agreeing on the final output is not enough: a rule
    // that is wrong in the same way on both sides would still pass. This
    // compares every value whose type the importer proved, so a disagreement
    // fails at the node that produced it.
    for (id, expected) in expected.iter().enumerate() {
        let Some(expected) = expected else {
            continue;
        };
        let inferred = &typed
            .value(ValueId::new(id as u32))
            .ok_or_else(|| {
                shape_error(Fn3Error::OutOfBounds {
                    what: "import::typed value id",
                    index: id as u64,
                    len: typed.value_count() as u64,
                })
            })?
            .ty;
        if inferred != expected {
            return Err(shape_error(Fn3Error::InvalidShape {
                context: "import::typed",
                detail: format!(
                    "value `{}` is {} in the importer and {} after the typed shape pass; the two                      rules disagree",
                    graph.value(super::graph::ValueId::new(id as u32)).name,
                    expected,
                    inferred
                ),
            }));
        }
    }
    let inputs = graph
        .inputs()
        .iter()
        .map(|id| ValueId::new(id.index() as u32))
        .collect();
    let outputs = graph
        .outputs()
        .iter()
        .map(|id| ValueId::new(id.index() as u32))
        .collect();
    Ok(ImportedGraph {
        graph: typed,
        weights,
        summary,
        warnings,
        inputs,
        outputs,
        ir_version,
        producer,
        producer_version,
        opset,
        graph_name,
    })
}

/// Convert the importer's tables, returning the graph and the importer's type
/// for every value it proved (`None` for a placeholder).
fn convert(
    graph: &GraphIr,
    typed_params: &[OpParams],
) -> Result<(Graph, Vec<Option<TensorType>>), ImportError> {
    let mut typed = Graph::new();
    let mut expected: Vec<Option<TensorType>> = Vec::with_capacity(graph.value_count());
    for value in graph.values() {
        let ty = match (&value.shape, value.origin) {
            (ValueShape::Dims(dims), _) => {
                let mut extents = Vec::with_capacity(dims.len());
                for dim in dims {
                    extents.push(dim_expr(graph, value.name.as_str(), value.origin, dim)?);
                }
                TensorType::new(value.dtype, extents)
            }
            // A node output is overwritten by the shape pass; a rank-0
            // placeholder keeps the value addressable in the meantime.
            (ValueShape::Unknown, ValueOrigin::Node(_)) => {
                TensorType::new(value.dtype, std::iter::empty())
            }
            (ValueShape::Unknown, origin) => {
                return Err(shape_error(Fn3Error::InvalidShape {
                    context: "import::typed",
                    detail: format!(
                        "value `{}` is {} with an unknown shape; the typed IR needs a rank for \
                         every input and weight",
                        value.name,
                        origin_name(origin)
                    ),
                }))
            }
        };
        expected.push(match &value.shape {
            ValueShape::Dims(_) => Some(ty.clone()),
            ValueShape::Unknown => None,
        });
        typed
            .add_value(Some(value.name.as_str()), ty)
            .map_err(shape_error)?;
    }
    if typed_params.len() != graph.nodes().len() {
        return Err(shape_error(Fn3Error::InvalidShape {
            context: "import::typed",
            detail: format!(
                "internal: {} imported nodes carry {} typed attribute sets",
                graph.nodes().len(),
                typed_params.len()
            ),
        }));
    }
    for (position, node) in graph.nodes().iter().enumerate() {
        let output_name = node.outputs.first().map(|id| graph.value(*id).name.clone());
        let mut loc = SourceLoc::new(node.onnx_index as usize, node.onnx_op.clone());
        if let Some(name) = output_name {
            loc = loc.with_output(name);
        }
        typed
            .add_node(
                node.op,
                node.inputs
                    .iter()
                    .map(|id| ValueId::new(id.index() as u32))
                    .collect(),
                node.outputs
                    .iter()
                    .map(|id| ValueId::new(id.index() as u32))
                    .collect(),
                typed_params[position].clone(),
                loc,
            )
            .map_err(shape_error)?;
    }
    Ok((typed, expected))
}

fn dim_expr(
    graph: &GraphIr,
    value: &str,
    origin: ValueOrigin,
    dim: &Dim,
) -> Result<DimExpr, ImportError> {
    match dim {
        Dim::Known(extent) => Ok(DimExpr::Const(*extent)),
        Dim::Symbol(symbol) => Ok(DimExpr::Symbol(*symbol)),
        Dim::Unknown(reason) => {
            let producer = match origin {
                ValueOrigin::Node(index) => graph.nodes().get(index as usize).map(|node| {
                    let mut loc = SourceLoc::new(node.onnx_index as usize, node.onnx_op.clone());
                    if let Some(output) = node.outputs.first() {
                        loc = loc.with_output(graph.value(*output).name.clone());
                    }
                    loc.describe()
                }),
                _ => None,
            };
            let where_from = match producer {
                Some(node) => format!(" produced by {node}"),
                None => String::new(),
            };
            Err(shape_error(Fn3Error::InvalidShape {
                context: "import::typed",
                detail: format!(
                    "value `{value}`{where_from} has an extent this import could not prove \
                     ({reason}); the typed IR has no unknown dimension"
                ),
            }))
        }
    }
}

fn origin_name(origin: ValueOrigin) -> &'static str {
    match origin {
        ValueOrigin::GraphInput => "a graph input",
        ValueOrigin::Weight(_) => "a weight",
        ValueOrigin::Node(_) => "a node output",
    }
}

/// Wrap a typed-IR failure as an import problem.
///
/// The inner message carries the source node (`node 7 (onnx "Conv", output
/// "y")`) because this bridge does not reconstruct one.
fn shape_error(error: Fn3Error) -> ImportError {
    ImportError::single(ImportProblemKind::ShapeMismatch {
        detail: format!("typed IR: {error}"),
    })
}
