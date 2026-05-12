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

/// Default maximum extent assumed for a purely symbolic dimension (no Bounded bound).
/// Used by the memory planner to allocate sufficient arena space for graphs that
/// contain unbounded Symbol dims.  Callers that know tighter bounds should use
/// [`DimExpr::Bounded`] instead.
pub const SYMBOL_DIM_MAX: u64 = 4096;

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

    /// Symbolic multiplication of two DimExpr values.
    pub fn mul(&self, other: &DimExpr) -> DimExpr {
        match (self, other) {
            (DimExpr::Known(0), _) | (_, DimExpr::Known(0)) => DimExpr::Known(0),
            (DimExpr::Known(1), other) | (other, DimExpr::Known(1)) => other.clone(),
            (DimExpr::Known(va), DimExpr::Known(vb)) => DimExpr::Known(va * vb),
            (DimExpr::Known(v), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Known(v)) => DimExpr::Bounded {
                sym: sym.clone(),
                max: max * v,
            },
            (DimExpr::Known(v), DimExpr::Symbol(s))
            | (DimExpr::Symbol(s), DimExpr::Known(v)) => DimExpr::Bounded {
                sym: s.clone(),
                max: *v,
            },
            (DimExpr::Bounded { sym, max }, DimExpr::Bounded { sym: _, max: mb }) => {
                DimExpr::Bounded {
                    sym: sym.clone(),
                    max: max * mb,
                }
            }
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) => {
                DimExpr::Symbol(format!("({}*{})", s, t))
            }
            (DimExpr::Symbol(s), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Symbol(s)) => {
                DimExpr::Bounded {
                    sym: format!("({}*{})", s, sym),
                    max: *max,
                }
            }
        }
    }

    /// Symbolic addition of two DimExpr values.
    pub fn add(&self, other: &DimExpr) -> DimExpr {
        match (self, other) {
            (DimExpr::Known(0), other) | (other, DimExpr::Known(0)) => other.clone(),
            (DimExpr::Known(va), DimExpr::Known(vb)) => DimExpr::Known(va + vb),
            (DimExpr::Known(v), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Known(v)) => DimExpr::Bounded {
                sym: sym.clone(),
                max: max + v,
            },
            (DimExpr::Known(v), DimExpr::Symbol(s))
            | (DimExpr::Symbol(s), DimExpr::Known(v)) => DimExpr::Bounded {
                sym: s.clone(),
                max: *v,
            },
            (DimExpr::Bounded { sym, max }, DimExpr::Bounded { sym: _, max: mb }) => {
                DimExpr::Bounded {
                    sym: sym.clone(),
                    max: max + mb,
                }
            }
            (DimExpr::Symbol(s), DimExpr::Symbol(t)) => {
                DimExpr::Symbol(format!("({}+{})", s, t))
            }
            (DimExpr::Symbol(s), DimExpr::Bounded { sym, max })
            | (DimExpr::Bounded { sym, max }, DimExpr::Symbol(s)) => {
                DimExpr::Bounded {
                    sym: format!("({}+{})", s, sym),
                    max: *max,
                }
            }
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

// =============================================================================
// ShapeEnv — runtime shape environment that resolves Symbol → concrete value
// =============================================================================

/// Runtime mapping from symbolic dimension names to concrete integer values.
///
/// Created by the executor before dispatch by inspecting input buffer sizes
/// against the shapes declared in `GraphBuilder::input_with_dims()`.
#[derive(Debug, Clone, Default)]
pub struct ShapeEnv {
    symbols: HashMap<String, u64>,
}

impl ShapeEnv {
    pub fn new() -> Self {
        ShapeEnv {
            symbols: HashMap::new(),
        }
    }

    /// Bind a symbol name to a concrete value.
    ///
    /// # Panics
    /// Panics if the symbol is already bound to a *different* value, to prevent
    /// silently inconsistent runtime shapes.
    pub fn bind(&mut self, name: &str, value: u64) {
        if let Some(&existing) = self.symbols.get(name) {
            assert_eq!(
                existing, value,
                "ShapeEnv: symbol '{}' bound to {} then inconsistently to {}",
                name, existing, value
            );
        }
        self.symbols.insert(name.to_string(), value);
    }

    /// Resolve a symbol name to its concrete value, if bound.
    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.symbols.get(name).copied()
    }

    /// Build a ShapeEnv by matching input byte sizes against graph input node shapes.
    ///
    /// Returns an error if:
    /// - Input byte count is not evenly divisible by the known-stride element count.
    /// - Two inputs infer different values for the same symbol (via [`bind`](Self::bind)).
    ///
    /// The algorithm has two passes:
    ///
    /// 1. **Single-symbol inputs** — if an input shape has exactly one
    ///    [`DimExpr::Symbol`] or [`DimExpr::Bounded`] dimension, its value is
    ///    computed directly from `data_bytes / (known_numel * elem_size)`.
    ///
    /// 2. **Multi-symbol inputs** — after pass 1, any already-bound symbols
    ///    are treated as known.  If exactly one symbol remains unbound, it is
    ///    inferred from the remaining unknown element count.
    ///
    /// This handles the common case of `[B, T, D]` where `B` is also present
    /// in another input (e.g. a bias of shape `[B]`).
    pub fn from_graph_inputs(graph: &ComputeGraph, inputs: &[&[u8]]) -> Result<Self, String> {
        let mut env = ShapeEnv::new();
        let mut input_infos: Vec<(NodeId, usize, usize, Vec<String>)> = Vec::new();

        // Pass 1: collect shape info, bind single-symbol inputs
        for (i, &input_id) in graph.inputs.iter().enumerate() {
            let data_bytes = inputs.get(i).map(|b| b.len()).unwrap_or(0);
            if data_bytes == 0 {
                continue;
            }
            if let Some(node) = graph.get_node(input_id) {
                let elem_size = node.output_type.dtype.byte_size();
                let known_numel: usize = node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Known(v) => Some(*v as usize),
                        _ => None,
                    })
                    .product();
                if known_numel == 0 || node.output_type.shape.is_empty() {
                    continue;
                }
                let stride = elem_size * known_numel;
                if data_bytes % stride != 0 {
                    return Err(format!(
                        "ShapeEnv: input {} node {}: data size {} not divisible by known-stride {} (shape {:?}, elem_size {})",
                        i, input_id, data_bytes, stride,
                        node.output_type.shape, elem_size
                    ));
                }
                let total_numel = data_bytes / elem_size;
                let unknown_numel = total_numel / known_numel;
                if unknown_numel == 0 {
                    continue;
                }
                let symbolic: Vec<String> = node
                    .output_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Symbol(s) => Some(s.clone()),
                        DimExpr::Bounded { sym, .. } => Some(sym.clone()),
                        _ => None,
                    })
                    .collect();

                if symbolic.len() == 1 {
                    env.bind(&symbolic[0], unknown_numel as u64);
                } else if symbolic.len() > 1 {
                    input_infos.push((input_id, known_numel, total_numel, symbolic));
                }
            }
        }

        // Pass 2: multi-symbol inputs — resolve remaining symbols using
        // already-bound values.
        for (_id, known_numel, total_numel, symbolic) in &input_infos {
            let mut known_product = *known_numel;
            let mut unbound: Vec<&str> = Vec::new();
            for sym in symbolic {
                if let Some(val) = env.resolve(sym) {
                    known_product *= val as usize;
                } else {
                    unbound.push(sym);
                }
            }
            if unbound.len() == 1 && known_product > 0 {
                if total_numel % known_product != 0 {
                    return Err(format!(
                        "ShapeEnv: multi-symbol input node {}: remaining numel {} not divisible by known_product {}",
                        _id, total_numel, known_product
                    ));
                }
                let val = total_numel / known_product;
                if val > 0 {
                    env.bind(unbound[0], val as u64);
                }
            }
        }

        Ok(env)
    }
}

impl DimExpr {
    /// Evaluate with runtime shape environment.  Bounded dims try resolving
    /// their symbol name first, falling back to the compile-time max bound.
    ///
    /// # Panics
    /// If a [`DimExpr::Bounded`] symbol resolves but the value exceeds `max`,
    /// indicating the runtime shape overflows the compile-time bound contract.
    pub fn evaluate_with_env(&self, env: &ShapeEnv) -> Option<u64> {
        match self {
            DimExpr::Known(v) => Some(*v),
            DimExpr::Bounded { sym, max } => match env.resolve(sym) {
                Some(v) => {
                    assert!(
                        v <= *max,
                        "DimExpr::Bounded: symbol '{}' resolved to {}, exceeds bound {}",
                        sym, v, max
                    );
                    Some(v)
                }
                None => Some(*max),
            },
            DimExpr::Symbol(s) => env.resolve(s),
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

    /// Compute the byte size of this tensor's storage, using `SYMBOL_DIM_MAX`
    /// as the fallback extent for pure [`DimExpr::Symbol`] dimensions.
    pub fn byte_size(&self) -> usize {
        self.byte_size_with_env(None)
    }

    /// Compute the byte size of this tensor's storage, resolving symbolic
    /// dimensions against the provided [`ShapeEnv`] when available.
    /// Pure [`DimExpr::Symbol`] dimensions (not [`DimExpr::Bounded`]) still
    /// fall back to `SYMBOL_DIM_MAX`.
    pub fn byte_size_with_env(&self, env: Option<&ShapeEnv>) -> usize {
        let numel: usize = self
            .shape
            .iter()
            .map(|d| match d {
                DimExpr::Known(v) => *v as usize,
                DimExpr::Bounded { max, .. } => *max as usize,
                DimExpr::Symbol(_) => match env {
                    Some(e) => d.evaluate_with_env(e).unwrap_or(SYMBOL_DIM_MAX) as usize,
                    None => SYMBOL_DIM_MAX as usize,
                },
            })
            .product();
        numel * self.dtype.byte_size()
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
