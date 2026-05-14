#![allow(dead_code)]

// =============================================================================
// Backward-compatibility stubs for v1.x code
// These make the old nn/* and tensor/* modules compile during migration.
// They will be removed once all code uses the compile-time IR.
// =============================================================================

use crate::tensor::Tensor;
use std::sync::Arc;

// Thread-local gradient enable/disable (kept for backward compat).
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
            fn apply(&self, _grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
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
            fn apply(&self, _grad_outputs: Vec<Option<Tensor>>, _output_tensor_id: usize) -> Vec<Option<Tensor>> {
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
stub_backward!(MishBackward, 1);
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
                    if let Some(n) = grad_graph.get_node_mut(grad_input) {
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
                if let Some(n) = grad_graph.get_node_mut(grad_input) {
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
            Opcode::LeakyRelu => {
                // d_input = d_output * (x > 0 ? 1.0 : alpha)
                if let Some(&input_id) = node.inputs.first() {
                    let alpha: f32 = node.attrs.get("negative_slope")
                        .and_then(|a| a.parse().ok()).unwrap_or(0.01);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let alpha_c = create_constant_scalar(alpha, &[], IrDType::F32, &mut grad_graph);
                    let mask = grad_graph.add_node(
                        Opcode::GtScalar, vec![input_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let neg_grad = grad_graph.add_node(
                        Opcode::Mul, vec![grad_id, alpha_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where, vec![mask, grad_id, neg_grad],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Elu => {
                // d_input = d_output * (x > 0 ? 1 : alpha * exp(x))
                if let Some(&input_id) = node.inputs.first() {
                    let alpha: f32 = node.attrs.get("alpha")
                        .and_then(|a| a.parse().ok()).unwrap_or(1.0);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let alpha_c = create_constant_scalar(alpha, &[], IrDType::F32, &mut grad_graph);
                    let exp_x = grad_graph.add_node(
                        Opcode::Exp, vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let alpha_exp = grad_graph.add_node(
                        Opcode::Mul, vec![alpha_c, exp_x],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mask = grad_graph.add_node(
                        Opcode::GtScalar, vec![input_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where, vec![mask, grad_id, alpha_exp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Softplus => {
                // d_input = d_output * sigmoid(x)
                if let Some(&input_id) = node.inputs.first() {
                    let sig = grad_graph.add_node(
                        Opcode::Sigmoid, vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul, vec![grad_id, sig],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Hardswish => {
                // hardswish(x) = x * clamp(x/3 + 0.5, 0, 1)  for x in [-3, 3]
                // derivative: 2x/3 + 1/2 for x in [-3, 3], 0 for x <= -3, 1 for x >= 3
                // Simplified: grad * (x > 3 ? 1 : (x < -3 ? 0 : 2*x/3 + 0.5))
                if let Some(&input_id) = node.inputs.first() {
                    let three = create_constant_scalar(3.0f32, &[], IrDType::F32, &mut grad_graph);
                    let neg_three = create_constant_scalar(-3.0f32, &[], IrDType::F32, &mut grad_graph);
                    let two_thirds = create_constant_scalar(2.0f32/3.0f32, &[], IrDType::F32, &mut grad_graph);
                    let half = create_constant_scalar(0.5f32, &[], IrDType::F32, &mut grad_graph);
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    // mask_high = x > 3
                    let mask_high = grad_graph.add_node(
                        Opcode::GtScalar, vec![input_id, three],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // mask_low = x < -3
                    let mask_low = grad_graph.add_node(
                        Opcode::LtScalar, vec![input_id, neg_three],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // linear part: 2*x/3 + 0.5
                    let x_23 = grad_graph.add_node(
                        Opcode::Mul, vec![input_id, two_thirds],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mid_deriv = grad_graph.add_node(
                        Opcode::Add, vec![x_23, half],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // not_low: grad * mid_deriv  (pass-through for masked region)
                    let not_low_deriv = grad_graph.add_node(
                        Opcode::Where, vec![mask_low, zero, mid_deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // deriv = 1 where x > 3, mid_deriv where -3 <= x <= 3, 0 where x < -3
                    let deriv = grad_graph.add_node(
                        Opcode::Where, vec![mask_high, one, not_low_deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul, vec![grad_id, deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Mish => {
                // mish(x) = x * tanh(softplus(x))
                // derivative: tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
                // Simplified: use autograd-friendly decomposition via output node
                // dmish/dx = mish_out * sigmoid(x) + mish_out / x ... actually
                // Simplest correct approach: delta = tanh(sp) + x * (1-tanh²(sp)) * sigmoid(x)
                if let Some(&input_id) = node.inputs.first() {
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let sp = grad_graph.add_node(
                        Opcode::Softplus, vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let tanh_sp = grad_graph.add_node(
                        Opcode::Tanh, vec![sp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sig_x = grad_graph.add_node(
                        Opcode::Sigmoid, vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // sech²(sp) = 1 - tanh²(sp)
                    let tanh_sq = grad_graph.add_node(
                        Opcode::Mul, vec![tanh_sp, tanh_sp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sech2 = grad_graph.add_node(
                        Opcode::Sub, vec![one, tanh_sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_times_sig = grad_graph.add_node(
                        Opcode::Mul, vec![input_id, sig_x],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_sech2_sig = grad_graph.add_node(
                        Opcode::Mul, vec![x_times_sig, sech2],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let deriv = grad_graph.add_node(
                        Opcode::Add, vec![tanh_sp, x_sech2_sig],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul, vec![grad_id, deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Clamp => {
                // d_input = d_output where x in [min, max], else 0
                if let Some(&input_id) = node.inputs.first() {
                    let min_val: f32 = node.attrs.get("min").and_then(|m| m.parse().ok()).unwrap_or(f32::NEG_INFINITY);
                    let max_val: f32 = node.attrs.get("max").and_then(|m| m.parse().ok()).unwrap_or(f32::INFINITY);
                    let min_c = create_constant_scalar(min_val, &[], IrDType::F32, &mut grad_graph);
                    let max_c = create_constant_scalar(max_val, &[], IrDType::F32, &mut grad_graph);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    // mask = x >= min AND x <= max
                    let ge_min = grad_graph.add_node(
                        Opcode::GtScalar, vec![input_id, min_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let le_max = grad_graph.add_node(
                        Opcode::LtScalar, vec![input_id, max_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // For inclusive bounds, handle equality:
                    let eq_min = grad_graph.add_node(
                        Opcode::EqScalar, vec![input_id, min_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let eq_max = grad_graph.add_node(
                        Opcode::EqScalar, vec![input_id, max_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ge_or_eq = grad_graph.add_node(
                        Opcode::Add, vec![ge_min, eq_min],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let le_or_eq = grad_graph.add_node(
                        Opcode::Add, vec![le_max, eq_max],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mask = grad_graph.add_node(
                        Opcode::Mul, vec![ge_or_eq, le_or_eq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where, vec![mask, grad_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Sign => {
                // subgradient: 0 almost everywhere (pass-through for x != 0)
                // d_input = d_output * 0 (stop gradients through sign)
                if let Some(&input_id) = node.inputs.first() {
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul, vec![grad_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::LogicalNot => {
                // no gradient needed (boolean op)
            }
            Opcode::LogSoftmax => {
                // log_softmax = x - log(sum(exp(x)))
                // d_input = d_output - softmax(x) * sum(d_output)
                // Simplified: grad - softmax * sum(grad)
                if let Some(&input_id) = node.inputs.first() {
                    // compute softmax(x) from log_softmax by exp
                    let sm = grad_graph.add_node(
                        Opcode::Exp, vec![node_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // sum of grad along appropriate axis
                    // For now, reduce all dims (same as softmax backward)
                    let grad_sum = grad_graph.add_node(
                        Opcode::ReduceSum, vec![grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_sm = grad_graph.add_node(
                        Opcode::Mul, vec![sm, grad_sum],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Sub, vec![grad_id, grad_sm],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Softmax => {
                // softmax backward: d_input = s * (d_output - sum(s * d_output))
                // where s = softmax(x). s is the node's output (node_id).
                if let Some(&input_id) = node.inputs.first() {
                    let s_times_grad = grad_graph.add_node(
                        Opcode::Mul, vec![node_id, grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sum_sg = grad_graph.add_node(
                        Opcode::ReduceSum, vec![s_times_grad],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_centered = grad_graph.add_node(
                        Opcode::Sub, vec![grad_id, sum_sg],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul, vec![node_id, grad_centered],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::BatchNorm => {
                // BN forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
                // Inputs: [x, gamma, beta, mean, var]
                // dx = gamma * grad / sqrt(var + eps)
                // dgamma = sum(grad * (x - mean) / sqrt(var + eps))
                // dbeta = sum(grad)
                // For simplicity: pass dx to input[0], accumulate dgamma/dbeta
                if let Some(&input_id) = node.inputs.first() {
                    let eps_val: f64 = node.attrs.get("eps").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                    let eps_c = create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    // gamma = inputs[1] (weight), var = inputs[4]
                    if node.inputs.len() >= 5 {
                        let gamma_id = node.inputs[1];
                        let var_id = node.inputs[4];
                        // sqrt(var + eps)
                        let var_eps = grad_graph.add_node(
                            Opcode::Add, vec![var_id, eps_c],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let std = grad_graph.add_node(
                            Opcode::Sqrt, vec![var_eps],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        // gamma / std
                        let gamma_div_std = grad_graph.add_node(
                            Opcode::Div, vec![gamma_id, std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        // dx = grad * gamma / std
                        let grad_input = grad_graph.add_node(
                            Opcode::Mul, vec![grad_id, gamma_div_std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        // dgamma: sum(grad * (x - mean) / std)
                        if let Some(&mean_id) = node.inputs.get(3) {
                            let x_minus_mean = grad_graph.add_node(
                                Opcode::Sub, vec![input_id, mean_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let x_minus_mean_div_std = grad_graph.add_node(
                                Opcode::Div, vec![x_minus_mean, std],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let grad_gamma_unreduced = grad_graph.add_node(
                                Opcode::Mul, vec![grad_id, x_minus_mean_div_std],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let grad_gamma = grad_graph.add_node(
                                Opcode::ReduceSum, vec![grad_gamma_unreduced],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, gamma_id, grad_gamma);
                        }
                        // dbeta: sum(grad)
                        if let Some(&beta_id) = node.inputs.get(2) {
                            let grad_beta = grad_graph.add_node(
                                Opcode::ReduceSum, vec![grad_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, beta_id, grad_beta);
                        }
                    }
                }
            }
            Opcode::LayerNorm => {
                if let Some(&input_id) = node.inputs.first() {
                    let eps_val: f64 = node.attrs.get("eps").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                    let eps_c = create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    let mean = grad_graph.add_node(
                        Opcode::ReduceMean, vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_minus_mean = grad_graph.add_node(
                        Opcode::Sub, vec![input_id, mean],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sq = grad_graph.add_node(
                        Opcode::Mul, vec![x_minus_mean, x_minus_mean],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let var = grad_graph.add_node(
                        Opcode::ReduceMean, vec![sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let var_eps = grad_graph.add_node(
                        Opcode::Add, vec![var, eps_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let std = grad_graph.add_node(
                        Opcode::Sqrt, vec![var_eps],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_hat = grad_graph.add_node(
                        Opcode::Div, vec![x_minus_mean, std],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    if node.inputs.len() >= 2 {
                        let gamma_id = node.inputs[1];
                        let dx_hat = grad_graph.add_node(
                            Opcode::Mul, vec![grad_id, gamma_id],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let grad_input = grad_graph.add_node(
                            Opcode::Div, vec![dx_hat, std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        let dy_x_hat = grad_graph.add_node(
                            Opcode::Mul, vec![grad_id, x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dgamma = grad_graph.add_node(
                            Opcode::ReduceSum, vec![dy_x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, gamma_id, dgamma);
                        if let Some(&beta_id) = node.inputs.get(2) {
                            let dbeta = grad_graph.add_node(
                                Opcode::ReduceSum, vec![grad_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, beta_id, dbeta);
                        }
                    } else {
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                    }
                }
            }
            Opcode::RMSNorm => {
                if let Some(&input_id) = node.inputs.first() {
                    let eps_val: f64 = node.attrs.get("eps").and_then(|e| e.parse().ok()).unwrap_or(1e-5);
                    let eps_c = create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    let sq = grad_graph.add_node(
                        Opcode::Mul, vec![input_id, input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ms = grad_graph.add_node(
                        Opcode::ReduceMean, vec![sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ms_eps = grad_graph.add_node(
                        Opcode::Add, vec![ms, eps_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let rms = grad_graph.add_node(
                        Opcode::Sqrt, vec![ms_eps],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    if node.inputs.len() >= 2 {
                        let gamma_id = node.inputs[1];
                        let gamma_div_rms = grad_graph.add_node(
                            Opcode::Div, vec![gamma_id, rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let grad_input = grad_graph.add_node(
                            Opcode::Mul, vec![grad_id, gamma_div_rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        let x_hat = grad_graph.add_node(
                            Opcode::Div, vec![input_id, rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dy_x_hat = grad_graph.add_node(
                            Opcode::Mul, vec![grad_id, x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dgamma = grad_graph.add_node(
                            Opcode::ReduceSum, vec![dy_x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, gamma_id, dgamma);
                    } else {
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                    }
                }
            }
            Opcode::MaxPool => {
                // d_input = scatter grad into positions that were max
                // Simplified: pass through (approximate - only correct if no overlap)
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::AvgPool => {
                if let Some(&input_id) = node.inputs.first() {
                    let kernel_size: usize = node.attrs.get("kernel_size")
                        .and_then(|s| s.parse().ok()).unwrap_or(2);
                    let pool_area = (kernel_size * kernel_size) as f32;
                    let scale = create_constant_scalar(1.0 / pool_area, &[], IrDType::F32, &mut grad_graph);
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, scale],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Constant(_) | Opcode::Input
            | Opcode::GtScalar | Opcode::LtScalar | Opcode::EqScalar
            | Opcode::SgdUpdate | Opcode::AdamUpdate | Opcode::AdamWUpdate => {
            }
            Opcode::Transpose => {
                // gradient of transpose is transpose with inverse permutation
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Transpose, vec![grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Concat => {
                // gradient of concat is split along the concat axis
                // For now, pass gradients to all inputs (correct split shape requires axis info)
                for &input_id in &node.inputs {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Slice => {
                // gradient of slice is scatter with zeros at sliced-out positions
                // For now, pass grad to first input (scatter requires attrs)
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Pad => {
                // gradient of pad is crop
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Gather => {
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::ScatterNd => {
                if let Some(&data_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, data_id, grad_id);
                }
                if node.inputs.len() >= 3 {
                    if let Some(&updates_id) = node.inputs.get(2) {
                        if let Some(&indices_id) = node.inputs.get(1) {
                            let d_updates = grad_graph.add_node(
                                Opcode::Gather,
                                vec![grad_id, indices_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, updates_id, d_updates);
                        }
                    }
                }
            }
            Opcode::Maximum | Opcode::Minimum => {
                // d_input = grad where input is the max/min, else 0
                // For now pass through to both (approximate)
                for &input_id in &node.inputs {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::ReduceMax | Opcode::ArgMax => {
                // pass grad to first input (argmax has no gradient, max needs argmax mask)
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Prelu | Opcode::Embedding | Opcode::Pow
            | Opcode::AddScalar | Opcode::MulScalar | Opcode::DivScalar
            | Opcode::UpsampleNearest2d | Opcode::UpsampleBilinear2d
            | Opcode::AdaptiveAvgPool2d | Opcode::Repeat
            | Opcode::CumSum | Opcode::Erf | Opcode::Flip | Opcode::Where
            | Opcode::TopKValues | Opcode::TopKIndices => {
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
