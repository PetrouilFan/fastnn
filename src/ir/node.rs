#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

pub type NodeId = usize;

#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Conv2d,
    Relu,
    Gelu,
    Silu,
    Sigmoid,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Neg,
    Abs,
    Reshape,
    Transpose,
    Concat,
    Slice,
    Flatten,
    Squeeze,
    Unsqueeze,
    ReduceSum,
    ReduceMean,
    MaxPool,
    AvgPool,
    BatchNorm,
    LayerNorm,
    Softmax,
    BiasAdd,
    Pad,
    Gather,
    ScatterNd,
    Constant(TensorValue),
    /// Placeholder for graph inputs — data provided at execution time.
    /// No producer instruction; the executor writes data into the arena slot.
    Input,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    Known(u64),
    Symbol(String),
    Bounded { sym: String, max: u64 },
}

impl DimExpr {
    pub fn is_known(&self) -> bool {
        matches!(self, DimExpr::Known(_))
    }

    pub fn evaluate(&self) -> Option<u64> {
        match self {
            DimExpr::Known(v) => Some(*v),
            DimExpr::Bounded { max, .. } => Some(*max),
            DimExpr::Symbol(_) => None,
        }
    }
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Known(v) => write!(f, "{}", v),
            DimExpr::Symbol(s) => write!(f, "{}", s),
            DimExpr::Bounded { sym, max } => write!(f, "{}∈[1,{}]", sym, max),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IrDType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    Bool,
    U4 { scale: f32, zero_point: i8 },
    U8 { scale: f32, zero_point: i8 },
}

impl IrDType {
    pub fn byte_size(&self) -> usize {
        match self {
            IrDType::F32 => 4,
            IrDType::F16 => 2,
            IrDType::BF16 => 2,
            IrDType::I32 => 4,
            IrDType::I64 => 8,
            IrDType::Bool => 1,
            IrDType::U4 { .. } => 1,
            IrDType::U8 { .. } => 1,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            IrDType::F32 => "f32",
            IrDType::F16 => "f16",
            IrDType::BF16 => "bf16",
            IrDType::I32 => "i32",
            IrDType::I64 => "i64",
            IrDType::Bool => "bool",
            IrDType::U4 { .. } => "u4",
            IrDType::U8 { .. } => "u8",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    pub shape: Vec<DimExpr>,
    pub dtype: IrDType,
}

impl TensorType {
    pub fn new(shape: Vec<DimExpr>, dtype: IrDType) -> Self {
        TensorType { shape, dtype }
    }

    pub fn numel(&self) -> Option<u64> {
        let mut total = 1u64;
        for dim in &self.shape {
            match dim.evaluate() {
                Some(v) => total = total.checked_mul(v)?,
                None => return None,
            }
        }
        Some(total)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorValue {
    Float(f32),
    Int(i64),
    Data {
        bytes: Vec<u8>,
        tensor_type: TensorType,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct IRNode {
    pub id: NodeId,
    pub opcode: Opcode,
    pub inputs: Vec<NodeId>,
    pub output_type: TensorType,
    pub attrs: HashMap<String, String>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub nodes: Vec<IRNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    pub next_id: NodeId,
}

impl ComputeGraph {
    pub fn new() -> Self {
        ComputeGraph {
            nodes: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_id: 1,
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
            attrs: HashMap::new(),
            name: String::new(),
        });
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
            attrs: HashMap::new(),
            name: name.to_string(),
        });
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
            attrs,
            name: String::new(),
        });
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

    pub fn get_node(&self, id: NodeId) -> Option<&IRNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut IRNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
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

    pub fn topological_sort(&self) -> Vec<NodeId> {
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

        let mut queue: Vec<NodeId> = {
            let mut zero_deg: Vec<NodeId> = in_degree
                .iter()
                .filter(|(_, &deg)| deg == 0)
                .map(|(&id, _)| id)
                .collect();
            zero_deg.sort(); // deterministic order for zero-degree nodes
            zero_deg
        };

        let mut sorted = Vec::with_capacity(self.nodes.len());
        while let Some(node_id) = queue.pop() {
            sorted.push(node_id);
            if let Some(children) = adjacency.get(&node_id) {
                for &child_id in children {
                    if let Some(deg) = in_degree.get_mut(&child_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(child_id);
                        }
                    }
                }
            }
        }

        if sorted.len() != self.nodes.len() {
            panic!("ComputeGraph::topological_sort: cycle detected in the computation graph");
        }

        sorted
    }

    pub fn consumers(&self, node_id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.inputs.contains(&node_id))
            .map(|n| n.id)
            .collect()
    }

    pub fn remove_node(&mut self, id: NodeId) {
        self.nodes.retain(|n| n.id != id);
        for node in &mut self.nodes {
            node.inputs.retain(|&i| i != id);
        }
        self.inputs.retain(|&i| i != id);
        self.outputs.retain(|&i| i != id);
    }
}
