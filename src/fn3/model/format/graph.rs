//! The graph section of a prepared-model artifact.
//!
//! # Why this type exists
//!
//! The importer ([`crate::inference::import`]) owns the graph IR: [`GraphIr`] has
//! private fields and is only ever built by decoding an ONNX file. A reader of a
//! serialized artifact must be able to *reconstruct* the same graph from bytes,
//! which `GraphIr` cannot express today. `PreparedGraph` is therefore the
//! format's constructible container over the importer's own element types
//! ([`ValueInfo`], [`GraphNode`], [`ValueId`]); it mirrors `GraphIr`'s accessors
//! one for one so a planner written against `GraphIr` reads the same way here.
//!
//! The follow-up this file looked forward to — a validated
//! `GraphIr::from_parts(values, nodes, inputs, outputs)` on the importer's side
//! — exists, so converting back is [`PreparedGraph::into_graph_ir`]. The mirror
//! is retained because it is the format's own read representation and because
//! [`PreparedGraph::check`] states the invariants the importer also enforces;
//! [`PreparedArtifact::into_model_import`] is the one call a consumer needs.
//!
//! # Invariants (enforced by [`PreparedGraph::new`])
//!
//! * every value handle, node output and graph input/output names a slot in
//!   range;
//! * a value's origin and the node that writes it agree in **both** directions —
//!   no value is written twice and no node claims a value it does not write;
//! * `ValueOrigin::GraphInput` values are exactly the declared graph inputs;
//! * value names are non-empty and unique.
//!
//! A `PreparedGraph` is immutable once built. Nothing here can panic on
//! untrusted input: the only constructor that consumes bytes
//! ([`super::PreparedArtifact::from_bytes`]) validates these invariants first.

use std::collections::BTreeSet;

use crate::inference::import::{GraphIr, GraphNode, ValueId, ValueInfo, ValueOrigin};

use super::error::FormatError;

/// A validated value table plus the ops that write it, in execution order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PreparedGraph {
    values: Vec<ValueInfo>,
    nodes: Vec<GraphNode>,
    inputs: Vec<ValueId>,
    outputs: Vec<ValueId>,
}

impl PreparedGraph {
    /// Copy a graph produced by the importer.
    ///
    /// The importer validates the same invariants while it builds, so this is a
    /// copy rather than a re-check; [`PreparedGraph::new`] is the validating
    /// constructor used on the read path.
    pub fn from_graph(graph: &GraphIr) -> PreparedGraph {
        PreparedGraph {
            values: graph.values().to_vec(),
            nodes: graph.nodes().to_vec(),
            inputs: graph.inputs().to_vec(),
            outputs: graph.outputs().to_vec(),
        }
    }

    /// Validate and construct.
    pub fn new(
        values: Vec<ValueInfo>,
        nodes: Vec<GraphNode>,
        inputs: Vec<ValueId>,
        outputs: Vec<ValueId>,
    ) -> Result<PreparedGraph, FormatError> {
        let graph = PreparedGraph {
            values,
            nodes,
            inputs,
            outputs,
        };
        graph.check()?;
        Ok(graph)
    }

    /// Every value, indexed by [`ValueId`].
    pub fn values(&self) -> &[ValueInfo] {
        &self.values
    }

    /// Number of values.
    pub fn value_count(&self) -> usize {
        self.values.len()
    }

    /// One value.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range. Every handle in a `PreparedGraph` is
    /// validated at construction, so this needs a hand-built `ValueId`.
    pub fn value(&self, id: ValueId) -> &ValueInfo {
        &self.values[id.index()]
    }

    /// Look a value up by exact name.
    pub fn lookup(&self, name: &str) -> Option<ValueId> {
        self.values
            .iter()
            .position(|value| value.name == name)
            .map(|index| ValueId::new(index as u32))
    }

    /// A value by exact name, with its info.
    pub fn find(&self, name: &str) -> Option<(ValueId, &ValueInfo)> {
        self.lookup(name).map(|id| (id, self.value(id)))
    }

    /// Operations in execution order.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Graph inputs, in file order.
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// Graph outputs, in file order.
    pub fn outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// The node that produced `id`, if a node did.
    ///
    /// Returns `None` for a handle that names no value rather than indexing, so
    /// a handle from an unrelated graph cannot panic here.
    pub fn producer(&self, id: ValueId) -> Option<&GraphNode> {
        match self.values.get(id.index())?.origin {
            ValueOrigin::Node(index) => self.nodes.get(index as usize),
            _ => None,
        }
    }

    /// Consume into the four parts a `GraphIr` is built from.
    ///
    /// [`PreparedGraph::into_graph_ir`] is the usual caller; these parts are
    /// also what a caller that wants the vectors themselves takes.
    pub fn parts(self) -> (Vec<ValueInfo>, Vec<GraphNode>, Vec<ValueId>, Vec<ValueId>) {
        (self.values, self.nodes, self.inputs, self.outputs)
    }

    /// Rebuild the importer's graph container, with no ONNX file present.
    ///
    /// The mirror is kept because it is the format's own read representation
    /// and [`PreparedGraph::check`] states the invariants the importer also
    /// enforces. A graph that reached a file and came back is therefore safe to
    /// hand to the planner and the session in the importer's own type.
    pub fn into_graph_ir(self) -> GraphIr {
        let (values, nodes, inputs, outputs) = self.parts();
        GraphIr::from_parts(values, nodes, inputs, outputs)
    }

    /// Check the invariants documented at module level.
    pub fn check(&self) -> Result<(), FormatError> {
        let value_count = self.values.len();
        if value_count == 0 {
            return Err(FormatError::inconsistent(
                "value table",
                "a graph needs at least one value",
            ));
        }
        let mut names: BTreeSet<&str> = BTreeSet::new();
        for (index, value) in self.values.iter().enumerate() {
            if value.name.is_empty() {
                return Err(FormatError::inconsistent(
                    "value table",
                    format!("value {index} has an empty name"),
                ));
            }
            if !names.insert(value.name.as_str()) {
                return Err(FormatError::inconsistent(
                    "value table",
                    format!("value name `{}` appears more than once", value.name),
                ));
            }
            match value.origin {
                ValueOrigin::GraphInput => {}
                ValueOrigin::Weight(_) => {}
                ValueOrigin::Node(node) => {
                    let node_ref = self.nodes.get(node as usize).ok_or_else(|| {
                        FormatError::inconsistent(
                            "value table",
                            format!(
                                "value {index} (`{}`) says it is written by node {node}, but the \
                                 graph has {} nodes",
                                value.name,
                                self.nodes.len()
                            ),
                        )
                    })?;
                    if !node_ref.outputs.contains(&ValueId::new(index as u32)) {
                        return Err(FormatError::inconsistent(
                            "value table",
                            format!(
                                "value {index} (`{}`) says it is written by node {node} \
                                 ({}, onnx index {}), but that node's outputs do not list it",
                                value.name, node_ref.op, node_ref.onnx_index
                            ),
                        ));
                    }
                }
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if node.outputs.is_empty() {
                return Err(FormatError::inconsistent(
                    "node table",
                    format!("node {index} ({}) has no output", node.op),
                ));
            }
            let mut written: BTreeSet<u32> = BTreeSet::new();
            for id in node.inputs.iter().chain(node.outputs.iter()) {
                if id.index() >= value_count {
                    return Err(FormatError::inconsistent(
                        "node table",
                        format!(
                            "node {index} ({}) names value {}, but the graph has {value_count} \
                             values",
                            node.op,
                            id.index()
                        ),
                    ));
                }
            }
            for id in &node.outputs {
                if !written.insert(id.index() as u32) {
                    return Err(FormatError::inconsistent(
                        "node table",
                        format!(
                            "node {index} ({}) writes value {} twice",
                            node.op,
                            id.index()
                        ),
                    ));
                }
                if self.values[id.index()].origin != ValueOrigin::Node(index as u32) {
                    return Err(FormatError::inconsistent(
                        "node table",
                        format!(
                            "node {index} ({}) writes value {} (`{}`), whose origin is {:?}",
                            node.op,
                            id.index(),
                            self.values[id.index()].name,
                            self.values[id.index()].origin
                        ),
                    ));
                }
            }
        }
        let mut declared: BTreeSet<u32> = BTreeSet::new();
        for id in &self.inputs {
            if id.index() >= value_count {
                return Err(FormatError::inconsistent(
                    "graph inputs",
                    format!(
                        "graph input value {} is out of range for {value_count} values",
                        id.index()
                    ),
                ));
            }
            if !declared.insert(id.index() as u32) {
                return Err(FormatError::inconsistent(
                    "graph inputs",
                    format!("graph input value {} is listed twice", id.index()),
                ));
            }
            if self.values[id.index()].origin != ValueOrigin::GraphInput {
                return Err(FormatError::inconsistent(
                    "graph inputs",
                    format!(
                        "graph input value {} (`{}`) has origin {:?}",
                        id.index(),
                        self.values[id.index()].name,
                        self.values[id.index()].origin
                    ),
                ));
            }
        }
        for (index, value) in self.values.iter().enumerate() {
            if value.origin == ValueOrigin::GraphInput && !declared.contains(&(index as u32)) {
                return Err(FormatError::inconsistent(
                    "graph inputs",
                    format!(
                        "value {index} (`{}`) is marked as a graph input but is not listed",
                        value.name
                    ),
                ));
            }
        }
        for id in &self.outputs {
            if id.index() >= value_count {
                return Err(FormatError::inconsistent(
                    "graph outputs",
                    format!(
                        "graph output value {} is out of range for {value_count} values",
                        id.index()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Check that every weight handle a value names exists in `weights`.
    ///
    /// Split from [`PreparedGraph::check`] because the weight table is a
    /// separate section: the graph is validated before the weights are decoded,
    /// so a graph that references weight 10,000 of a file with three weights is
    /// reported as an inconsistency instead of an out-of-range index.
    pub(super) fn check_weights(&self, weight_count: usize) -> Result<(), FormatError> {
        for (index, value) in self.values.iter().enumerate() {
            if let ValueOrigin::Weight(weight) = value.origin {
                if weight.index() >= weight_count {
                    return Err(FormatError::inconsistent(
                        "value table",
                        format!(
                            "value {index} (`{}`) names weight {}, but the weight table has \
                             {weight_count} entries",
                            value.name,
                            weight.index()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::inference::dtype::DType;
    use crate::inference::import::ValueShape;
    use crate::inference::ops::OpId;
    use crate::inference::shape::Symbol;

    fn value(name: &str, origin: ValueOrigin) -> ValueInfo {
        ValueInfo {
            name: name.to_string(),
            dtype: DType::F32,
            shape: ValueShape::Dims(vec![crate::inference::import::Dim::Known(2)]),
            origin,
        }
    }

    fn node(op: OpId, inputs: &[u32], outputs: &[u32]) -> GraphNode {
        GraphNode {
            onnx_index: 0,
            name: "n".to_string(),
            onnx_op: "Add".to_string(),
            op,
            inputs: inputs.iter().copied().map(ValueId::new).collect(),
            outputs: outputs.iter().copied().map(ValueId::new).collect(),
            params: Default::default(),
        }
    }

    #[test]
    fn a_consistent_graph_validates_and_mirrors_lookups() {
        let graph = PreparedGraph::new(
            vec![
                value("a", ValueOrigin::GraphInput),
                value(
                    "b",
                    ValueOrigin::Weight(crate::inference::model::WeightId::new(0)),
                ),
                value("c", ValueOrigin::Node(0)),
            ],
            vec![node(OpId::Add, &[0, 1], &[2])],
            vec![ValueId::new(0)],
            vec![ValueId::new(2)],
        )
        .unwrap();
        assert_eq!(graph.value_count(), 3);
        assert_eq!(graph.lookup("c"), Some(ValueId::new(2)));
        assert_eq!(graph.find("b").unwrap().1.dtype, DType::F32);
        assert!(graph.lookup("missing").is_none());
        assert_eq!(graph.producer(ValueId::new(2)).unwrap().op, OpId::Add);
        assert!(graph.producer(ValueId::new(0)).is_none());
        assert_eq!(graph.inputs(), &[ValueId::new(0)]);
        assert_eq!(graph.outputs(), &[ValueId::new(2)]);
        graph.check_weights(1).unwrap();
        assert!(graph.check_weights(0).is_err());
    }

    #[test]
    fn a_node_that_writes_a_value_it_does_not_own_is_rejected() {
        let error = PreparedGraph::new(
            vec![
                value("a", ValueOrigin::GraphInput),
                value("b", ValueOrigin::GraphInput),
            ],
            vec![node(OpId::Add, &[0], &[1])],
            vec![ValueId::new(0)],
            vec![ValueId::new(1)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("whose origin is"), "{error}");
    }

    #[test]
    fn a_value_written_by_a_missing_node_is_rejected() {
        let error = PreparedGraph::new(
            vec![
                value("a", ValueOrigin::GraphInput),
                value("b", ValueOrigin::Node(7)),
            ],
            vec![],
            vec![ValueId::new(0)],
            vec![ValueId::new(1)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("written by node 7"), "{error}");
    }

    #[test]
    fn duplicate_names_and_missing_inputs_are_rejected() {
        let duplicate = PreparedGraph::new(
            vec![
                value("a", ValueOrigin::GraphInput),
                value(
                    "a",
                    ValueOrigin::Weight(crate::inference::model::WeightId::new(0)),
                ),
            ],
            vec![],
            vec![ValueId::new(0)],
            vec![ValueId::new(0)],
        )
        .unwrap_err();
        assert!(
            duplicate.to_string().contains("appears more than once"),
            "{duplicate}"
        );

        let unlisted = PreparedGraph::new(
            vec![value("a", ValueOrigin::GraphInput)],
            vec![],
            vec![],
            vec![ValueId::new(0)],
        )
        .unwrap_err();
        assert!(
            unlisted.to_string().contains("marked as a graph input"),
            "{unlisted}"
        );
    }

    #[test]
    fn an_out_of_range_handle_is_rejected_not_indexed() {
        let error = PreparedGraph::new(
            vec![value("a", ValueOrigin::GraphInput)],
            vec![node(OpId::Add, &[0, 9], &[0])],
            vec![ValueId::new(0)],
            vec![ValueId::new(0)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("graph has 1 values"), "{error}");
    }

    #[test]
    fn a_symbolic_dimension_survives_the_round_trip_through_parts() {
        let mut info = value("a", ValueOrigin::GraphInput);
        info.shape = ValueShape::Dims(vec![
            crate::inference::import::Dim::Symbol(Symbol::Batch),
            crate::inference::import::Dim::unknown("batch * seq"),
        ]);
        let graph = PreparedGraph::new(
            vec![info],
            vec![],
            vec![ValueId::new(0)],
            vec![ValueId::new(0)],
        )
        .unwrap();
        let (values, nodes, inputs, outputs) = graph.parts();
        let rebuilt = PreparedGraph::new(values, nodes, inputs, outputs).unwrap();
        assert_eq!(
            rebuilt.value(ValueId::new(0)).shape.render(),
            "[batch, ?(batch * seq)]"
        );
    }
}
