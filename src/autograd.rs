#![allow(dead_code)]

// =============================================================================
// Backward-compatibility stubs for v1.x code
// These make the old nn/* and tensor/* modules compile during migration.
// They will be removed once all code uses the compile-time IR.
// =============================================================================

use crate::tensor::Tensor;
use std::sync::Arc;

/// Thread-local gradient enable/disable (kept for backward compat).
thread_local! {
    static NO_GRAD_TLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub fn is_grad_enabled() -> bool {
    NO_GRAD_TLS.with(|c| c.get() == 0)
}

pub fn no_grad_enter() {
    NO_GRAD_TLS.with(|c| c.set(c.get() + 1));
}

pub fn no_grad_exit() {
    NO_GRAD_TLS.with(|c| {
        let count = c.get();
        if count > 0 {
            c.set(count - 1);
        }
    });
}

/// RAII guard for disabling gradient computation (kept for backward compat).
pub struct NoGradGuard;

impl NoGradGuard {
    pub fn new() -> Self { no_grad_enter(); NoGradGuard }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) { no_grad_exit(); }
}

impl Default for NoGradGuard {
    fn default() -> Self { Self::new() }
}

/// Convenience function to run a closure with gradient computation disabled.
pub fn no_grad<F, R>(f: F) -> R where F: FnOnce() -> R {
    let _guard = NoGradGuard::new();
    f()
}

/// Stub AutogradMeta (v1.x compat — gradients tracked at graph level now).
pub struct AutogradMeta {
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<dyn Node>>,
    pub is_leaf: bool,
}

impl AutogradMeta {
    pub fn new(requires_grad: bool) -> Self {
        AutogradMeta { requires_grad, grad: None, grad_fn: None, is_leaf: true }
    }
    pub fn new_non_leaf(requires_grad: bool) -> Self {
        AutogradMeta { requires_grad, grad: None, grad_fn: None, is_leaf: false }
    }
    pub fn zero_grad(&mut self, _set_to_none: bool) {
        self.grad = None;
    }
}

/// Stub Edge for v1.x backward compat.
pub struct Edge(pub Arc<dyn Node>, pub usize);

/// Stub Node trait for v1.x backward compat.
pub trait Node: Send + Sync {
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>, output_tensor_id: usize) -> Vec<Option<Tensor>>;
    fn next_edges(&self) -> &[Edge];
    fn num_inputs(&self) -> usize;
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Tensor];
    fn id(&self) -> usize {
        let ptr = self as *const _ as *const ();
        ptr as usize
    }
}

pub fn make_edge(tensor: &Tensor) -> Vec<Edge> {
    tensor.grad_fn().map(|node| vec![Edge(node, 0)]).unwrap_or_default()
}

/// Stub backward function - replaces old runtime backward engine.
/// All backward computation is now handled by build_backward_graph at compile time.
pub fn backward(_root: &Tensor, _grad_output: Option<Tensor>) {
    // No-op: backward graph construction happens at compile time
}

pub fn make_edges(tensor_a: &Tensor, tensor_b: &Tensor) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(2);
    if let Some(node) = tensor_a.grad_fn() { edges.push(Edge(node, 0)); }
    if let Some(node) = tensor_b.grad_fn() { edges.push(Edge(node, 1)); }
    edges
}

// =============================================================================
// Stub backward node types for v1.x code compatibility.
// These are all no-ops — the real backward pass is now compile-time graph
// transformation (build_backward_graph).
// =============================================================================

macro_rules! stub_backward {
    ($name:ident) => {
        pub struct $name;
        impl $name {
            pub fn new() -> Self { $name }
        }
        impl Node for $name {
            fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
                vec![None; self.num_inputs()]
            }
            fn next_edges(&self) -> &[Edge] { &[] }
            fn num_inputs(&self) -> usize { 0 }
            fn name(&self) -> &str { stringify!($name) }
            fn inputs(&self) -> &[Tensor] { &[] }
        }
    };
    ($name:ident, $ninputs:expr) => {
        pub struct $name;
        impl $name {
            pub fn new() -> Self { $name }
        }
        impl Node for $name {
            fn apply(&self, grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
                vec![None; $ninputs]
            }
            fn next_edges(&self) -> &[Edge] { &[] }
            fn num_inputs(&self) -> usize { $ninputs }
            fn name(&self) -> &str { stringify!($name) }
            fn inputs(&self) -> &[Tensor] { &[] }
        }
    };
}

stub_backward!(AddBackward, 2);
stub_backward!(SubBackward, 2);
stub_backward!(MulBackward, 2);
stub_backward!(DivBackward, 2);
stub_backward!(AddScalarBackward, 2);
stub_backward!(DivScalarBackward, 2);
stub_backward!(MatmulBackward, 2);
stub_backward!(ReluBackward, 1);
stub_backward!(GeluBackward, 1);
stub_backward!(SiLUBackward, 1);
stub_backward!(SigmoidBackward, 1);
stub_backward!(TanhBackward, 1);
stub_backward!(ExpBackward, 1);
stub_backward!(LogBackward, 1);
stub_backward!(SqrtBackward, 1);
stub_backward!(NegBackward, 1);
stub_backward!(AbsBackward, 1);
stub_backward!(ClampBackward, 1);
stub_backward!(PowBackward, 2);
stub_backward!(LeakyReLUBackward, 1);
stub_backward!(EluBackward, 1);
stub_backward!(SoftplusBackward, 1);
stub_backward!(HardswishBackward, 1);
stub_backward!(SoftmaxBackward, 1);
stub_backward!(LogSoftmaxBackward, 1);
stub_backward!(SumBackward, 1);
stub_backward!(MeanBackward, 1);
stub_backward!(MaximumBackward, 2);
stub_backward!(MinimumBackward, 2);
stub_backward!(TransposeBackward, 1);
stub_backward!(PermuteBackward, 1);
stub_backward!(ReshapeBackward, 1);
stub_backward!(FlattenBackward, 1);
stub_backward!(UnsqueezeBackward, 1);
stub_backward!(SqueezeBackward, 1);
stub_backward!(ExpandBackward, 1);
stub_backward!(RepeatBackward, 1);
stub_backward!(SliceBackward, 1);
stub_backward!(CatBackward, 1);
stub_backward!(WhereBackward, 3);
stub_backward!(DropoutBackward, 1);
stub_backward!(Dropout2dBackward, 1);
stub_backward!(Conv2dBackward, 2);
stub_backward!(ConvTranspose2dBackward, 2);
stub_backward!(BatchNorm1dBackward, 1);
stub_backward!(BatchNorm2dBackward, 1);
stub_backward!(GroupNormBackward, 1);
stub_backward!(RMSNormBackward, 1);
stub_backward!(LayerNormBackward, 1);
stub_backward!(MaxPool2dBackward, 1);
stub_backward!(AvgPool2dBackward, 1);
stub_backward!(AdaptiveAvgPool2dBackward, 1);
stub_backward!(UpsampleBackward, 1);
stub_backward!(MSELossBackward, 2);
stub_backward!(CrossEntropyBackward, 2);
stub_backward!(BCEWithLogitsBackward, 2);
stub_backward!(HuberLossBackward, 2);
stub_backward!(ViewBackward, 1);

use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};
use std::collections::HashMap;

/// Build a backward computation graph from a forward graph and a loss node.
/// Returns a new graph that, when executed, computes gradients for all 
/// forward-graph parameters, together with a mapping from forward node IDs
/// to their gradient accumulator node IDs in the new graph.
pub fn build_backward_graph(
    forward_graph: &ComputeGraph,
    loss_node: NodeId,
) -> Result<(ComputeGraph, HashMap<NodeId, NodeId>), String> {
    // Sanity checks
    if loss_node == 0 || forward_graph.get_node(loss_node).is_none() {
        return Err("build_backward_graph: loss_node not found in forward graph".to_string());
    }
    
    // Create a new graph that contains all forward nodes + backward nodes
    let mut grad_graph = ComputeGraph::new();
    
    // Map from old node IDs to new node IDs (we reuse the same IDs)
    // Clone all forward nodes into the grad graph
    grad_graph.nodes = forward_graph.nodes.clone();
    grad_graph.next_id = forward_graph.next_id;
    
    // Map: node_id -> node_id of gradient accumulator in grad_graph
    let mut grads: HashMap<NodeId, NodeId> = HashMap::new();
    
    // Initialize loss gradient: d(loss)/d(loss) = 1.0
    let loss_node_ref = forward_graph.get_node(loss_node)
        .ok_or("build_backward_graph: loss node not found")?;
    let loss_shape = loss_node_ref.output_type.shape.clone();
    let loss_dtype = loss_node_ref.output_type.dtype.clone();
    
    // Create constant 1.0 gradient for the loss
    let one_tensor = create_constant_scalar(1.0f32, &loss_shape, loss_dtype.clone(), &mut grad_graph);
    grads.insert(loss_node, one_tensor);
    
    // Topological sort in REVERSE order (from output to input)
    let forward_order = forward_graph.topological_sort();
    
    // Walk nodes in reverse order
    for &node_id in forward_order.iter().rev() {
        let node = forward_graph.get_node(node_id)
            .ok_or("build_backward_graph: node not found in forward walk")?;
        
        // Get the gradient of this node's output
        let node_grad = grads.get(&node_id).cloned();
        
        if node_grad.is_none() {
            // Node doesn't affect the loss (dead computation)
            continue;
        }
        
        let grad_id = node_grad.unwrap();
        
        // Compute gradients for inputs based on the opcode
        match &node.opcode {
            Opcode::Relu => {
                if let Some(&input_id) = node.inputs.first() {
                    let input_node = forward_graph.get_node(input_id)
                        .ok_or("build_backward_graph: input not found")?;
                    let input_type = input_node.output_type.clone();
                    
                    let mut attrs = HashMap::new();
                    attrs.insert("op".to_string(), "relu_backward".to_string());
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node_id],
                        input_type,
                    );
                    if let Some(mut n) = grad_graph.get_node_mut(grad_input) {
                        n.attrs = attrs;
                    }
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Add => {
                for &input_id in &node.inputs {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Sub => {
                if let Some(&first) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, first, grad_id);
                }
                if node.inputs.len() > 1 {
                    let neg = create_neg(grad_id, &mut grad_graph);
                    accumulate_grad(&mut grad_graph, &mut grads, node.inputs[1], neg);
                }
            }
            Opcode::Mul => {
                if node.inputs.len() >= 2 {
                    let g1 = grad_graph.add_node(Opcode::Mul, vec![grad_id, node.inputs[1]], 
                        forward_graph.get_node(node.inputs[1]).map(|n| n.output_type.clone()).unwrap());
                    accumulate_grad(&mut grad_graph, &mut grads, node.inputs[0], g1);
                    
                    let g2 = grad_graph.add_node(Opcode::Mul, vec![grad_id, node.inputs[0]],
                        forward_graph.get_node(node.inputs[0]).map(|n| n.output_type.clone()).unwrap());
                    accumulate_grad(&mut grad_graph, &mut grads, node.inputs[1], g2);
                }
            }
            Opcode::MatMul => {
                if node.inputs.len() >= 2 {
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_type = forward_graph.get_node(a_id).map(|n| n.output_type.clone());
                    let b_type = forward_graph.get_node(b_id).map(|n| n.output_type.clone());
                    
                    let b_t = grad_graph.add_node(Opcode::Transpose, vec![b_id],
                        b_type.clone().unwrap_or(TensorType::new(vec![], IrDType::F32)));
                    let da = grad_graph.add_node(Opcode::MatMul, vec![grad_id, b_t],
                        a_type.clone().unwrap_or(TensorType::new(vec![], IrDType::F32)));
                    accumulate_grad(&mut grad_graph, &mut grads, a_id, da);
                    
                    let a_t = grad_graph.add_node(Opcode::Transpose, vec![a_id],
                        a_type.unwrap_or(TensorType::new(vec![], IrDType::F32)));
                    let db = grad_graph.add_node(Opcode::MatMul, vec![a_t, grad_id],
                        b_type.unwrap_or(TensorType::new(vec![], IrDType::F32)));
                    accumulate_grad(&mut grad_graph, &mut grads, b_id, db);
                }
            }
            Opcode::Gelu => {
                let mut attrs = HashMap::new();
                attrs.insert("op".to_string(), "gelu_backward".to_string());
                let grad_input = grad_graph.add_node(
                    Opcode::Mul,
                    vec![grad_id, node_id],
                    forward_graph.get_node(node.inputs[0]).map(|n| n.output_type.clone()).unwrap(),
                );
                if let Some(mut n) = grad_graph.get_node_mut(grad_input) {
                    n.attrs = attrs;
                }
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Exp => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![node_id, grad_id],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Log => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Div,
                        vec![grad_id, node_id],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Sigmoid => {
                if let Some(&input_id) = node.inputs.first() {
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let one_minus_sig = grad_graph.add_node(
                        Opcode::Sub,
                        vec![one, node_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sig_mul = grad_graph.add_node(
                        Opcode::Mul,
                        vec![node_id, one_minus_sig],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, sig_mul],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Tanh => {
                if let Some(&input_id) = node.inputs.first() {
                    let sq = grad_graph.add_node(
                        Opcode::Mul,
                        vec![node_id, node_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let one_minus_sq = grad_graph.add_node(
                        Opcode::Sub,
                        vec![one, sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, one_minus_sq],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Neg => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = create_neg(grad_id, &mut grad_graph);
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Reshape | Opcode::Flatten | Opcode::Squeeze | Opcode::Unsqueeze => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::ReduceSum => {
                if let Some(&input_id) = node.inputs.first() {
                    let input_type = forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap();
                    let ones = create_constant_scalar(1.0f32, &input_type.shape, input_type.dtype.clone(), &mut grad_graph);
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, ones);
                }
            }
            Opcode::ReduceMean => {
                if let Some(&input_id) = node.inputs.first() {
                    let input_type = forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap();
                    let n = input_type.numel().unwrap_or(1) as f32;
                    let scale = create_constant_scalar(1.0 / n, &input_type.shape, input_type.dtype.clone(), &mut grad_graph);
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, scale);
                }
            }
            Opcode::Sqrt => {
                if let Some(&input_id) = node.inputs.first() {
                    let two = create_constant_scalar(2.0f32, &[], IrDType::F32, &mut grad_graph);
                    let two_mul = grad_graph.add_node(
                        Opcode::Mul,
                        vec![node_id, two],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Div,
                        vec![grad_id, two_mul],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Abs => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node_id],
                        forward_graph.get_node(input_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Conv2d | Opcode::Conv1d | Opcode::Conv3d | Opcode::ConvTranspose2d => {
                if node.inputs.len() >= 2 {
                    if let Some(&input_id) = node.inputs.first() {
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                    }
                    if let Some(&weight_id) = node.inputs.get(1) {
                        accumulate_grad(&mut grad_graph, &mut grads, weight_id, grad_id);
                    }
                }
            }
            Opcode::BiasAdd => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
                if let Some(&bias_id) = node.inputs.get(1) {
                    let grad_input = grad_graph.add_node(
                        Opcode::ReduceSum,
                        vec![grad_id],
                        forward_graph.get_node(bias_id).map(|n| n.output_type.clone()).unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, bias_id, grad_input);
                }
            }
            Opcode::Div => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
                if node.inputs.len() > 1 {
                    if let Some(&input_id) = node.inputs.get(1) {
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                    }
                }
            }
            Opcode::Silu => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Gelu | Opcode::LeakyRelu | Opcode::Elu | Opcode::Softplus
            | Opcode::Hardswish | Opcode::Clamp | Opcode::Sign
            | Opcode::LogicalNot | Opcode::LogSoftmax | Opcode::Sigmoid
            | Opcode::Tanh | Opcode::Exp | Opcode::Log | Opcode::Sqrt
            | Opcode::Neg | Opcode::Abs | Opcode::Relu | Opcode::Softmax
            | Opcode::Silu => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::BatchNorm | Opcode::LayerNorm | Opcode::RMSNorm => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::MaxPool | Opcode::AvgPool => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Constant(_) | Opcode::Input
            | Opcode::GtScalar | Opcode::LtScalar | Opcode::EqScalar => {
            }
            Opcode::Pad | Opcode::Slice | Opcode::Concat | Opcode::Gather | Opcode::ScatterNd
            | Opcode::Transpose | Opcode::Maximum | Opcode::Minimum | Opcode::ReduceMax
            | Opcode::Prelu | Opcode::Embedding | Opcode::Pow
            | Opcode::AddScalar | Opcode::MulScalar | Opcode::DivScalar => {
                for &input_id in &node.inputs {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
        }
    }
    
    // Set up graph inputs/outputs
    grad_graph.inputs = forward_graph.inputs.clone();
    for (node_id, grad_id) in &grads {
        if forward_graph.inputs.contains(node_id) || forward_graph.outputs.contains(node_id) {
            grad_graph.outputs.push(*grad_id);
        }
    }
    
    Ok((grad_graph, grads))
}

/// Create a constant scalar in the graph.
/// Uses `TensorValue::Float` so the backend can broadcast it via `Fill`.
fn create_constant_scalar(value: f32, shape: &[DimExpr], dtype: IrDType, graph: &mut ComputeGraph) -> NodeId {
    graph.add_node(
        Opcode::Constant(TensorValue::Float(value)),
        vec![],
        TensorType::new(shape.to_vec(), dtype),
    )
}

/// Create a negation of a value
fn create_neg(input: NodeId, graph: &mut ComputeGraph) -> NodeId {
    graph.add_node(
        Opcode::Neg,
        vec![input],
        TensorType::new(vec![], IrDType::F32),
    )
}

/// Accumulate a gradient into an existing gradient accumulator, or create one
fn accumulate_grad(
    graph: &mut ComputeGraph,
    grads: &mut HashMap<NodeId, NodeId>,
    node_id: NodeId,
    partial_grad: NodeId,
) {
    if let Some(&existing_grad) = grads.get(&node_id) {
        let accum = graph.add_node(
            Opcode::Add,
            vec![existing_grad, partial_grad],
            TensorType::new(vec![], IrDType::F32),
        );
        grads.insert(node_id, accum);
    } else {
        grads.insert(node_id, partial_grad);
    }
}
