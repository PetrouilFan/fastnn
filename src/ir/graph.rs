//! Graph storage, mutation, and cache ownership for the IR.

use super::opcode::Opcode;
use super::types::{DimExpr, IrDType, TensorType};
use crate::error::{FastnnError, FastnnResult};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Display;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub type NodeId = usize;

/// Declared semantic role of a graph. Compiler passes use this instead of
/// inferring training/backward/update intent from whichever opcodes are present.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GraphKind {
    #[default]
    Inference,
    TrainingForward,
    Backward,
    OptimizerUpdate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorValue {
    Float(f32),
    Int(i64),
    Data {
        bytes: Vec<u8>,
        tensor_type: TensorType,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IRNode {
    pub id: NodeId,
    pub opcode: Opcode,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
    pub secondary_output_type: Option<TensorType>,
    pub attrs: HashMap<String, String>,
    pub name: String,
}

impl IRNode {
    pub fn num_outputs(&self) -> usize {
        if self.secondary_output_type.is_some() {
            2
        } else {
            1
        }
    }

    pub fn required_attr<T>(&self, key: &str) -> FastnnResult<T>
    where
        T: FromStr,
        T::Err: Display,
    {
        let value = self.attrs.get(key).ok_or_else(|| {
            FastnnError::shape(format!(
                "{:?} node {} is missing required attribute {key:?}",
                self.opcode, self.id
            ))
        })?;
        value.parse::<T>().map_err(|error| {
            FastnnError::shape(format!(
                "{:?} node {} has invalid {key:?} attribute {value:?}: {error}",
                self.opcode, self.id
            ))
        })
    }

    pub fn optional_attr<T>(&self, key: &str) -> FastnnResult<Option<T>>
    where
        T: FromStr,
        T::Err: Display,
    {
        self.attrs
            .get(key)
            .map(|value| {
                value.parse::<T>().map_err(|error| {
                    FastnnError::shape(format!(
                        "{:?} node {} has invalid {key:?} attribute {value:?}: {error}",
                        self.opcode, self.id
                    ))
                })
            })
            .transpose()
    }

    pub fn optional_attr_list<T>(&self, key: &str) -> FastnnResult<Option<Vec<T>>>
    where
        T: FromStr,
        T::Err: Display,
    {
        let Some(raw) = self.attrs.get(key) else {
            return Ok(None);
        };
        if raw.is_empty() {
            return Ok(Some(Vec::new()));
        }
        raw.split(',')
            .enumerate()
            .map(|(index, value)| {
                let value = value.trim();
                value.parse::<T>().map_err(|error| {
                    FastnnError::shape(format!(
                        "{:?} node {} has invalid {key:?} item {index} {value:?}: {error}",
                        self.opcode, self.id
                    ))
                })
            })
            .collect::<FastnnResult<Vec<_>>>()
            .map(Some)
    }

    pub fn optional_bool_attr(&self, key: &str) -> FastnnResult<Option<bool>> {
        let Some(value) = self.attrs.get(key) else {
            return Ok(None);
        };
        match value.trim() {
            "1" | "true" | "True" | "TRUE" => Ok(Some(true)),
            "0" | "false" | "False" | "FALSE" => Ok(Some(false)),
            _ => Err(FastnnError::shape(format!(
                "{:?} node {} has invalid boolean {key:?} attribute {value:?}",
                self.opcode, self.id
            ))),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComputeGraph {
    #[serde(default)]
    pub kind: GraphKind,
    pub nodes: Vec<IRNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub required_nodes: HashSet<NodeId>,
    pub next_id: NodeId,
    #[serde(skip)]
    node_index: FxHashMap<NodeId, usize>,
    #[serde(skip)]
    consumers_map: Mutex<FxHashMap<NodeId, Vec<NodeId>>>,
    #[serde(skip)]
    consumers_map_dirty: AtomicBool,
    #[serde(skip)]
    sorted_nodes_cache: Mutex<Option<Vec<NodeId>>>,
    #[serde(skip)]
    sorted_nodes_gen: AtomicU64,
    #[serde(skip)]
    graph_gen: AtomicU64,
}

impl Clone for ComputeGraph {
    fn clone(&self) -> Self {
        ComputeGraph {
            kind: self.kind,
            nodes: self.nodes.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            required_nodes: self.required_nodes.clone(),
            next_id: self.next_id,
            node_index: self.node_index.clone(),
            consumers_map: Mutex::new(self.consumers_map.lock().clone()),
            consumers_map_dirty: AtomicBool::new(self.consumers_map_dirty.load(Ordering::Relaxed)),
            sorted_nodes_cache: Mutex::new(self.sorted_nodes_cache.lock().clone()),
            sorted_nodes_gen: AtomicU64::new(self.sorted_nodes_gen.load(Ordering::Relaxed)),
            graph_gen: AtomicU64::new(self.graph_gen.load(Ordering::Relaxed)),
        }
    }
}

impl ComputeGraph {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ComputeGraph {
            kind: GraphKind::Inference,
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            required_nodes: HashSet::new(),
            next_id: 1,
            node_index: FxHashMap::default(),
            consumers_map: Mutex::new(FxHashMap::default()),
            consumers_map_dirty: AtomicBool::new(false),
            sorted_nodes_cache: Mutex::new(None),
            sorted_nodes_gen: AtomicU64::new(0),
            graph_gen: AtomicU64::new(0),
        }
    }

    pub fn with_kind(kind: GraphKind) -> Self {
        let mut graph = Self::new();
        graph.kind = kind;
        graph
    }

    pub fn set_kind(&mut self, kind: GraphKind) {
        self.kind = kind;
    }

    pub(crate) fn mark_mutated(&mut self) {
        self.graph_gen.fetch_add(1, Ordering::Release);
        self.consumers_map_dirty.store(true, Ordering::Release);
    }

    pub(crate) fn rebuild_node_index(&mut self) {
        self.node_index.clear();
        for (i, node) in self.nodes.iter().enumerate() {
            self.node_index.insert(node.id, i);
        }
    }

    pub fn add_node(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs: HashMap::new(),
            name: String::new(),
        });
        self.node_index.insert(id, self.nodes.len() - 1);
        self.mark_mutated();
        id
    }

    pub fn add_node_with_name(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        name: &str,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs: HashMap::new(),
            name: name.to_string(),
        });
        self.node_index.insert(id, self.nodes.len() - 1);
        self.mark_mutated();
        id
    }

    /// Add a node with attributes set.
    pub fn add_node_with_attrs(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        output_type: TensorType,
        attrs: HashMap<String, String>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type,
            secondary_output_type: None,
            attrs,
            name: String::new(),
        });
        self.node_index.insert(id, self.nodes.len() - 1);
        self.mark_mutated();
        id
    }

    pub fn add_constant(&mut self, value: TensorValue) -> NodeId {
        let tensor_type = match &value {
            TensorValue::Float(_) => TensorType::new(Vec::new(), IrDType::F32),
            TensorValue::Int(_) => TensorType::new(Vec::new(), IrDType::I64),
            TensorValue::Data { tensor_type, .. } => tensor_type.clone(),
        };
        self.add_node(Opcode::Constant(value), Vec::new(), tensor_type)
    }

    pub fn add_node_with_secondary_output(
        &mut self,
        opcode: Opcode,
        inputs: Vec<NodeId>,
        primary_type: TensorType,
        secondary_type: TensorType,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(IRNode {
            id,
            opcode,
            inputs,
            output_type: primary_type,
            secondary_output_type: Some(secondary_type),
            attrs: HashMap::new(),
            name: String::new(),
        });
        self.node_index.insert(id, self.nodes.len() - 1);
        self.mark_mutated();
        id
    }

    pub fn get_node(&self, id: NodeId) -> Option<&IRNode> {
        if let Some(&idx) = self.node_index.get(&id) {
            if let Some(node) = self.nodes.get(idx) {
                if node.id == id {
                    return Some(node);
                }
            }
        }
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut IRNode> {
        let cache_hit = self.node_index.get(&id).copied();
        if let Some(idx) = cache_hit {
            if self.nodes.get(idx).is_some_and(|n| n.id == id) {
                return self.nodes.get_mut(idx);
            }
        }
        self.rebuild_node_index();
        let idx = self.node_index.get(&id)?;
        self.nodes.get_mut(*idx)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_inputs(&mut self, inputs: Vec<NodeId>) {
        self.inputs = inputs;
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    pub fn add_required_node(&mut self, node_id: NodeId) {
        self.required_nodes.insert(node_id);
    }

    pub fn try_topological_sort(&self) -> FastnnResult<Vec<NodeId>> {
        let gen = self.graph_gen.load(Ordering::Acquire);
        {
            let cache = self.sorted_nodes_cache.lock();
            if let Some(cached) = cache.as_ref() {
                if self.sorted_nodes_gen.load(Ordering::Acquire) == gen {
                    return Ok(cached.clone());
                }
            }
        }

        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            in_degree.insert(node.id, 0);
            adjacency.insert(node.id, Vec::new());
        }

        for node in &self.nodes {
            for &input_id in &node.inputs {
                if adjacency.contains_key(&input_id) {
                    adjacency
                        .get_mut(&input_id)
                        .expect("adjacency entry for input_id")
                        .push(node.id);
                    *in_degree
                        .get_mut(&node.id)
                        .expect("in_degree entry for node.id") += 1;
                }
            }
        }

        let mut queue: VecDeque<NodeId> = {
            let mut zero_deg: Vec<NodeId> = in_degree
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&id, _)| id)
                .collect();
            zero_deg.sort(); // deterministic order for zero-degree nodes
            VecDeque::from(zero_deg)
        };

        let mut sorted = Vec::with_capacity(self.nodes.len());
        while let Some(node_id) = queue.pop_front() {
            sorted.push(node_id);
            if let Some(children) = adjacency.get(&node_id) {
                for &child_id in children {
                    if let Some(deg) = in_degree.get_mut(&child_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(child_id);
                        }
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            return Err(FastnnError::compilation(
                "cycle detected in the computation graph",
            ));
        }

        let mut cache = self.sorted_nodes_cache.lock();
        *cache = Some(sorted.clone());
        self.sorted_nodes_gen.store(gen, Ordering::Release);
        Ok(sorted)
    }

    pub fn consumers(&self, node_id: NodeId) -> Vec<NodeId> {
        if self.consumers_map_dirty.load(Ordering::Acquire) {
            let mut map: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();
            for node in &self.nodes {
                for &input_id in &node.inputs {
                    map.entry(input_id).or_default().push(node.id);
                }
            }
            let mut consumers_map = self.consumers_map.lock();
            *consumers_map = map;
            self.consumers_map_dirty.store(false, Ordering::Release);
            consumers_map.get(&node_id).cloned().unwrap_or_default()
        } else {
            let consumers_map = self.consumers_map.lock();
            consumers_map.get(&node_id).cloned().unwrap_or_default()
        }
    }

    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        for node in &mut self.nodes {
            node.inputs.retain(|&i| i != id);
        }
        self.inputs.retain(|&i| i != id);
        self.outputs.retain(|&i| i != id);
        self.required_nodes.retain(|&i| i != id);
        self.rebuild_node_index();
        self.mark_mutated();
    }

    /// Save the ComputeGraph to a .fnn binary file.
    pub fn save_fnn(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::serialize(self)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a ComputeGraph from a .fnn binary file created by [`save_fnn`](Self::save_fnn).
    pub fn load_fnn(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let graph: ComputeGraph = bincode::deserialize(&bytes)?;
        Ok(graph)
    }

    /// Returns `true` when every node in the graph has a fully-known
    /// (i.e. containing zero symbolic [`DimExpr`] variants) output shape.
    ///
    /// When this returns `true` the per-inference shape-resolution and
    /// memory-plan tightening steps always produce the same result, so
    /// they can be cached after the first inference and skipped on
    /// subsequent calls.
    pub fn has_static_shapes(&self) -> bool {
        self.nodes.iter().all(|n| {
            n.output_type
                .shape
                .iter()
                .all(|d| matches!(d, DimExpr::Known(_)))
        })
    }
}

#[cfg(test)]
mod graph_kind_tests {
    use super::{ComputeGraph, GraphKind};
    use crate::ir::{IrDType, Opcode, TensorType};
    use std::collections::HashMap;

    #[test]
    fn typed_attribute_access_rejects_missing_and_malformed_values() {
        let mut graph = ComputeGraph::new();
        let mut attrs = HashMap::new();
        attrs.insert("bit_width".to_string(), "four".to_string());
        let id = graph.add_node_with_attrs(
            Opcode::Quantize,
            vec![],
            TensorType::new(vec![], IrDType::F32),
            attrs,
        );
        let node = graph.get_node(id).unwrap();
        assert!(node.required_attr::<usize>("missing").is_err());
        assert!(node.required_attr::<usize>("bit_width").is_err());
        assert!(node.optional_attr::<usize>("bit_width").is_err());
        assert_eq!(node.optional_attr::<usize>("missing").unwrap(), None);

        let mut node = node.clone();
        node.attrs.insert("axes".into(), "0, 2,3".into());
        assert_eq!(
            node.optional_attr_list::<usize>("axes").unwrap(),
            Some(vec![0, 2, 3])
        );
        node.attrs.insert("axes".into(), "0,bad,3".into());
        assert!(node.optional_attr_list::<usize>("axes").is_err());

        node.attrs.insert("keepdim".into(), "0".into());
        assert_eq!(node.optional_bool_attr("keepdim").unwrap(), Some(false));
        node.attrs.insert("keepdim".into(), "sometimes".into());
        assert!(node.optional_bool_attr("keepdim").is_err());
    }

    #[test]
    fn graph_kind_is_preserved_by_clone_and_serialization() {
        let graph = ComputeGraph::with_kind(GraphKind::TrainingForward);
        assert_eq!(graph.clone().kind, GraphKind::TrainingForward);

        let bytes = bincode::serialize(&graph).unwrap();
        let decoded: ComputeGraph = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.kind, GraphKind::TrainingForward);
    }
}
