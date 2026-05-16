#![allow(dead_code)]

// =============================================================================
// Backward-compatibility stubs for v1.x code
// These make the old nn/* and tensor/* modules compile during migration.
// They will be removed once all code uses the compile-time IR.
// =============================================================================

use crate::ir::builder::GraphBuilder;
use crate::storage::{Device, Storage};
use crate::tensor::{dtype_to_ir, Tensor, TensorImpl};
use smallvec::SmallVec;
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
    pub fn new() -> Self {
        no_grad_enter();
        NoGradGuard
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        no_grad_exit();
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to run a closure with gradient computation disabled.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

// =============================================================================
// Lightweight tape entry — replaces per-op Arc<dyn Node> stub allocations.
// =============================================================================

/// A lightweight record of a forward operation, stored in AutogradMeta.
/// Replaces the per-op `Arc<dyn Node>` stub types, eliminating 50+ trait
/// object allocations per forward pass.
pub struct NodeInfo {
    pub op_name: &'static str,
    pub inputs: Vec<Tensor>,
}

impl NodeInfo {
    pub fn new(op_name: &'static str, inputs: Vec<Tensor>) -> Self {
        NodeInfo { op_name, inputs }
    }

    pub fn edges(&self) -> Vec<Edge> {
        self.inputs
            .iter()
            .filter_map(|t| t.grad_fn())
            .map(|node| Edge(node, 0))
            .collect()
    }

    pub fn inputs(&self) -> &[Tensor] {
        &self.inputs
    }

    pub fn id(&self) -> usize {
        let ptr = self as *const Self as *const ();
        ptr as usize
    }
}

/// Stub AutogradMeta (v1.x compat — gradients tracked at graph level now).
pub struct AutogradMeta {
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<NodeInfo>>,
    pub is_leaf: bool,
}

impl AutogradMeta {
    pub fn new(requires_grad: bool) -> Self {
        AutogradMeta {
            requires_grad,
            grad: None,
            grad_fn: None,
            is_leaf: true,
        }
    }
    pub fn new_non_leaf(requires_grad: bool) -> Self {
        AutogradMeta {
            requires_grad,
            grad: None,
            grad_fn: None,
            is_leaf: false,
        }
    }
    pub fn zero_grad(&mut self, _set_to_none: bool) {
        self.grad = None;
    }
}

/// Stub Edge for v1.x backward compat.
pub struct Edge(pub Arc<NodeInfo>, pub usize);

/// Stub Node trait for v1.x backward compat (legacy — no longer used for backward).
pub trait Node: Send + Sync {
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        output_tensor_id: usize,
    ) -> Vec<Option<Tensor>>;
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
    tensor
        .grad_fn()
        .map(|node| vec![Edge(node, 0)])
        .unwrap_or_default()
}

// =============================================================================
// Reconstruction engine — bridges v1.x autograd chain to IR build_backward_graph
// =============================================================================

use std::collections::{HashMap, HashSet};

/// BFS from root tensor to collect all backward nodes and build
/// a grad_fn → forward_tensor map.
///
/// Returns (backward_nodes, grad_fn_id → forward_output_tensor).
fn collect_backward_nodes(root: &Tensor) -> (Vec<Arc<NodeInfo>>, HashMap<usize, Tensor>) {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut nodes: Vec<Arc<NodeInfo>> = Vec::new();
    let mut grad_fn_to_tensor: HashMap<usize, Tensor> = HashMap::new();
    let mut queue: Vec<Tensor> = vec![root.clone()];

    while let Some(tensor) = queue.pop() {
        if let Some(node) = tensor.grad_fn() {
            let node_id = Arc::as_ptr(&node) as usize;
            if visited.insert(node_id) {
                grad_fn_to_tensor.insert(node_id, tensor.clone());
                nodes.push(node.clone());

                // Queue the input tensors of this backward node.
                // These correspond to the forward op's input tensors;
                // their own grad_fns continue the chain back toward the leaves.
                for input_tensor in &node.inputs {
                    queue.push(input_tensor.clone());
                }
            }
        }
    }

    (nodes, grad_fn_to_tensor)
}

/// From a set of backward nodes, find all leaf input tensors —
/// those that require gradients and have no grad_fn of their own.
fn collect_leaf_tensors(nodes: &[Arc<NodeInfo>]) -> Vec<Tensor> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut leaf_tensors: Vec<Tensor> = Vec::new();

    for node in nodes {
        for input in &node.inputs {
            let tid = input.id();
            if seen.insert(tid) && input.requires_grad() && input.grad_fn().is_none() {
                leaf_tensors.push(input.clone());
            }
        }
    }

    leaf_tensors
}

/// Topological sort of backward nodes in **forward execution order** (predecessors first).
///
/// Uses DFS: edges point to predecessor backward nodes, so
/// the DFS emits predecessors before their dependents.
fn topological_order(nodes: &[Arc<NodeInfo>]) -> Vec<Arc<NodeInfo>> {
    fn dfs(node: &Arc<NodeInfo>, visited: &mut HashSet<usize>, order: &mut Vec<Arc<NodeInfo>>) {
        let ptr = Arc::as_ptr(node) as usize;
        if !visited.insert(ptr) {
            return;
        }
        for edge in node.edges() {
            dfs(&edge.0, visited, order);
        }
        order.push(node.clone());
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut order: Vec<Arc<NodeInfo>> = Vec::new();
    for node in nodes {
        dfs(node, &mut visited, &mut order);
    }
    order
}

/// Extract the f32 scalar value from a 1-element CPU tensor.
fn scalar_from_tensor(t: &Tensor) -> f32 {
    let cpu_t = t.to_cpu();
    let data = cpu_t.as_bytes();
    if data.len() >= 4 {
        f32::from_le_bytes([data[0], data[1], data[2], data[3]])
    } else {
        0.0
    }
}

/// Infer the reduction dimension and `keepdim` flag by comparing
/// the input and output shapes of a reduction op.
fn infer_reduction_params(input_shape: &[i64], output_shape: &[i64]) -> (usize, bool) {
    if input_shape.is_empty() {
        return (0, false);
    }

    let keepdim = input_shape.len() == output_shape.len();

    if keepdim {
        // Same rank — find the dim whose size changed
        for i in 0..input_shape.len() {
            if input_shape[i] != output_shape[i] {
                return (i, true);
            }
        }
        (0, true)
    } else if output_shape.is_empty() || *output_shape == [1] {
        // Reduced to scalar — default to dim 0
        (0, false)
    } else {
        // Rank decreased by one — find the removed dim
        for i in 0..output_shape.len() {
            if input_shape[i] != output_shape[i] {
                return (i, false);
            }
        }
        // All leading dims match → last dim was reduced
        (input_shape.len() - 1, false)
    }
}

/// Full backward pass: reconstructs the forward ComputeGraph from the
/// v1.x autograd chain, then delegates to `build_backward_graph` for
/// gradient computation.
///
/// This makes `loss.backward()` work by:
/// 1. Walking the grad_fn chain from root (BFS)
/// 2. Replaying the forward ops via GraphBuilder to create a ComputeGraph
/// 3. Calling `GraphBuilder::backward()` (which calls `build_backward_graph`)
/// 4. Compiling and executing the backward graph
/// 5. Storing the resulting gradients back on the leaf tensors
pub fn backward(root: &Tensor, _grad_output: Option<Tensor>) {
    if !is_grad_enabled() {
        return;
    }

    let (all_nodes, grad_fn_to_tensor) = collect_backward_nodes(root);
    if all_nodes.is_empty() {
        return;
    }

    let leaf_inputs = collect_leaf_tensors(&all_nodes);
    if leaf_inputs.is_empty() {
        return;
    }

    // ── Build forward IR graph ──────────────────────────────────────────
    use crate::ir::builder::GraphTensor;

    let builder = GraphBuilder::new();
    let mut tensor_map: HashMap<usize, GraphTensor> = HashMap::new();

    // Collect ALL unique tensor inputs that are NOT produced by a backward node.
    // A tensor is "produced by a backward node" if it has a grad_fn whose id()
    // is in grad_fn_to_tensor.  Such tensors are intermediate forward results
    // that will be computed during forward replay.  Everything else is either
    // a leaf parameter or an external input and must be registered as a graph input.
    let mut all_input_tensors: Vec<Tensor> = Vec::new();
    let mut seen_input: HashSet<usize> = HashSet::new();
    for node in &all_nodes {
        for input in node.inputs() {
            let tid = input.id();
            if seen_input.insert(tid) {
                // Check if this tensor is produced by one of our backward nodes
                let is_intermediate = input
                    .grad_fn()
                    .as_ref()
                    .map(|gf| grad_fn_to_tensor.contains_key(&gf.id()))
                    .unwrap_or(false);
                if !is_intermediate {
                    all_input_tensors.push(input.clone());
                }
            }
        }
    }

    // Register all input tensors as GraphBuilder inputs
    for tensor in &all_input_tensors {
        let shape: Vec<DimExpr> = tensor
            .shape()
            .iter()
            .map(|&s| DimExpr::Known(s as u64))
            .collect();
        let gt = builder.input_with_dims(&shape, dtype_to_ir(tensor.dtype()));
        tensor_map.insert(tensor.id(), gt);
    }

    // Replay ops in forward topological order
    for node in topological_order(&all_nodes) {
        let inputs: Vec<Tensor> = node.inputs.clone();
        let input_gts: Vec<GraphTensor> = inputs
            .iter()
            .map(|t| {
                tensor_map
                    .get(&t.id())
                    .cloned()
                    .expect("backward: missing input tensor in tensor_map")
            })
            .collect();

        // The forward output tensor that this backward node is attached to
        let node_ptr = Arc::as_ptr(&node) as usize;
        let forward_tensor = grad_fn_to_tensor
            .get(&node_ptr)
            .cloned()
            .expect("backward: forward tensor not found for backward node");

        let output_gt = match node.op_name {
            // ── Binary arithmetic ─────────────────────────────────────────
            "AddBackward" => builder.add(&input_gts[0], &input_gts[1]),
            "SubBackward" => builder.sub(&input_gts[0], &input_gts[1]),
            "MulBackward" => builder.mul(&input_gts[0], &input_gts[1]),
            "DivBackward" => builder.div(&input_gts[0], &input_gts[1]),
            "MatmulBackward" => builder.matmul(&input_gts[0], &input_gts[1]),

            // ── Unary arithmetic ─────────────────────────────────────────
            "NegBackward" => builder.neg(&input_gts[0]),

            // ── Activations ──────────────────────────────────────────────
            "ReluBackward" => builder.relu(&input_gts[0]),
            "ExpBackward" => builder.exp(&input_gts[0]),
            "LogBackward" => builder.log(&input_gts[0]),
            "SigmoidBackward" => builder.sigmoid(&input_gts[0]),
            "TanhBackward" => builder.tanh(&input_gts[0]),

            // ── Scalar ops ───────────────────────────────────────────────
            "AddScalarBackward" => {
                let scalar_val = scalar_from_tensor(&inputs[1]);
                let scalar_bytes = scalar_val.to_le_bytes().to_vec();
                let scalar_gt = builder.constant(
                    &scalar_bytes,
                    crate::ir::node::TensorType::new(vec![], crate::ir::node::IrDType::F32),
                );
                builder.add_scalar(&input_gts[0], &scalar_gt)
            }
            "DivScalarBackward" => {
                let scalar_val = scalar_from_tensor(&inputs[1]);
                let scalar_bytes = scalar_val.to_le_bytes().to_vec();
                let scalar_gt = builder.constant(
                    &scalar_bytes,
                    crate::ir::node::TensorType::new(vec![], crate::ir::node::IrDType::F32),
                );
                builder.div_scalar(&input_gts[0], &scalar_gt)
            }

            // ── Reductions (infer dim from shape change) ────────────────
            "SumBackward" => {
                let input_shape = inputs[0].shape();
                let output_shape = forward_tensor.shape();
                let (dim, keepdim) = infer_reduction_params(&input_shape, &output_shape);
                builder.reduce_sum(&input_gts[0], dim, keepdim)
            }
            "MeanBackward" => {
                let input_shape = inputs[0].shape();
                let output_shape = forward_tensor.shape();
                let (dim, keepdim) = infer_reduction_params(&input_shape, &output_shape);
                builder.reduce_mean(&input_gts[0], dim, keepdim)
            }

            // ── Ops without a builder replay path ───────────────────────
            // These are not replayed in the reconstruction graph; their
            // backward formulas are still handled by build_backward_graph
            // when it processes the forward IR graph that the reconstruction
            // has already built.
            _ => continue,
        };

        tensor_map.insert(forward_tensor.id(), output_gt);
    }

    // Ensure the loss tensor was reached by the replay
    let loss_gt = match tensor_map.get(&root.id()) {
        Some(gt) => gt.clone(),
        None => return,
    };

    // ── Build backward graph via existing IR infrastructure ───────────
    let grad_tensors = match builder.backward(&loss_gt) {
        Ok(g) => g,
        Err(_) => return,
    };

    // ── Compile and execute ───────────────────────────────────────────
    use crate::backend::cpu::CpuBackend;

    // Input data order must match the order inputs were registered (all_input_tensors)
    let input_refs: Vec<&[u8]> = all_input_tensors.iter().map(|t| t.as_bytes()).collect();
    let grad_refs: Vec<&GraphTensor> = grad_tensors.iter().collect();

    let mut results = match builder.compile_and_execute(&grad_refs, CpuBackend, &input_refs) {
        Ok(r) => r,
        Err(_) => return,
    };

    // ── Store gradients back on leaf tensors (accumulating if exists) ──
    for tensor in leaf_inputs.iter() {
        if !tensor.requires_grad() {
            continue;
        }

        // Find this leaf tensor's position in the all_input_tensors list
        let pos = all_input_tensors.iter().position(|t| t.id() == tensor.id());
        let i = match pos {
            Some(idx) => idx,
            None => continue,
        };

        if i >= results.len() {
            continue;
        }

        let result_bytes = std::mem::take(&mut results[i]);
        let numel = tensor.shape().iter().product::<i64>() as usize;
        let dtype_size = tensor.dtype().size();
        let expected_bytes = numel * dtype_size;

        if result_bytes.len() != expected_bytes {
            continue;
        }

        // Build a fresh CPU Storage from the result bytes
        let storage = Storage::from_vec(result_bytes, tensor.dtype(), Device::Cpu);
        let shape: SmallVec<[i64; 8]> = tensor.shape().to_vec().into();
        let new_grad = Tensor::new(TensorImpl::new(Arc::new(storage), shape, tensor.dtype()));

        // Accumulate: if a gradient already exists, add the new one to it in-place
        let final_grad = if let Some(existing_grad) = tensor.grad() {
            let mut grad = existing_grad;
            grad.add_(&new_grad);
            grad
        } else {
            new_grad
        };

        TensorImpl::set_grad_for_tensor(tensor, Some(final_grad));
    }
}

// =============================================================================
// Tests for the v1.x autograd → IR bridge
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::DType;
    use crate::tensor::Tensor;

    /// Helper: create an f32 tensor of shape [n] with sequential values 1.0..=n
    fn seq_tensor(n: i64) -> Tensor {
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let t = Tensor::zeros(vec![n], DType::F32, Device::Cpu);
        // Zero-init then overwrite via internal data pointer
        let ptr = t.data_ptr_f32() as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            unsafe {
                *ptr.add(i) = v;
            }
        }
        t
    }

    #[test]
    fn test_collect_backward_nodes_simple() {
        let a = seq_tensor(4).requires_grad_(true);
        let b = seq_tensor(4).requires_grad_(true);
        let c = a.add(&b);
        let loss = c.mean(0, false);

        let (nodes, map) = collect_backward_nodes(&loss);
        assert!(!nodes.is_empty(), "should collect backward nodes");
        assert!(
            map.contains_key(&nodes[0].id()),
            "forward tensor should be mapped"
        );
    }

    #[test]
    fn test_simple_add_backward() {
        let a = seq_tensor(4).requires_grad_(true);
        let b = seq_tensor(4).requires_grad_(true);
        let c = a.add(&b);
        let loss = c.mean(0, false);

        backward(&loss, None);

        let grad_a = a.grad().expect("a should have grad");
        let grad_vals_a: Vec<f32> = unsafe {
            let ptr = grad_a.data_ptr_f32() as *const f32;
            (0..4).map(|i| *ptr.add(i)).collect()
        };
        // d(loss)/da = d(mean(a+b))/da = 1/4 = 0.25
        for &g in &grad_vals_a {
            assert!((g - 0.25).abs() < 1e-5, "grad_a should be 0.25, got {}", g);
        }
        // b should also have grad = 0.25
        let grad_b = b.grad().expect("b should have grad");
        let grad_vals_b: Vec<f32> = unsafe {
            let ptr = grad_b.data_ptr_f32() as *const f32;
            (0..4).map(|i| *ptr.add(i)).collect()
        };
        for &g in &grad_vals_b {
            assert!((g - 0.25).abs() < 1e-5, "grad_b should be 0.25, got {}", g);
        }
    }

    #[test]
    fn test_mul_backward() {
        let a = seq_tensor(4).requires_grad_(true);
        let b = seq_tensor(4).requires_grad_(true);
        // c = a * b, loss = mean(c)
        let c = a.mul(&b);
        let loss = c.mean(0, false);

        backward(&loss, None);

        let grad_a = a.grad().expect("a should have grad");
        let grad_b = b.grad().expect("b should have grad");
        let vals_a: Vec<f32> = unsafe {
            let ptr = grad_a.data_ptr_f32() as *const f32;
            (0..4).map(|i| *ptr.add(i)).collect()
        };
        // d(mean(a*b))/da_i = b_i / 4
        // b = [1, 2, 3, 4], so grad_a = [1/4, 2/4, 3/4, 4/4]
        for (i, &g) in vals_a.iter().enumerate() {
            let expected = (i as f32 + 1.0) / 4.0;
            assert!(
                (g - expected).abs() < 1e-5,
                "grad_a[{}] should be {}, got {}",
                i,
                expected,
                g
            );
        }
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::zeros(vec![2, 3], DType::F32, Device::Cpu).requires_grad_(true);
        let b = Tensor::zeros(vec![3, 2], DType::F32, Device::Cpu).requires_grad_(true);

        // Fill a with 1..6, b with 7..12
        let a_ptr = a.data_ptr_f32() as *mut f32;
        for i in 0..6 {
            unsafe {
                *a_ptr.add(i) = (i + 1) as f32;
            }
        }
        let b_ptr = b.data_ptr_f32() as *mut f32;
        for i in 0..6 {
            unsafe {
                *b_ptr.add(i) = (i + 7) as f32;
            }
        }

        let c = a.matmul(&b);
        let loss = c.mean(0, false); // shape [2, 2] → [2]

        backward(&loss, None);

        let grad_a = a.grad().expect("a should have grad");
        // Shape check: grad_a should be [2, 3] (same as a)
        assert_eq!(a.shape(), grad_a.shape(), "grad_a shape should match a");
    }

    #[test]
    fn test_relu_backward() {
        let a = Tensor::zeros(vec![4], DType::F32, Device::Cpu).requires_grad_(true);
        let a_ptr = a.data_ptr_f32() as *mut f32;
        // Mix of positive and negative values
        for (i, &v) in [-1.0, 2.0, -3.0, 4.0].iter().enumerate() {
            unsafe {
                *a_ptr.add(i) = v;
            }
        }

        let r = a.relu();
        let loss = r.mean(0, false);

        backward(&loss, None);

        let grad_a = a.grad().expect("a should have grad");
        assert_eq!(a.shape(), grad_a.shape(), "grad shape should match input");

        let vals: Vec<f32> = unsafe {
            let ptr = grad_a.data_ptr_f32() as *const f32;
            (0..4).map(|i| *ptr.add(i)).collect()
        };
        // Non-negative inputs should have non-zero gradients (relu' = 1 for x>0)
        // Negative inputs should have zero gradient (relu' = 0 for x<=0)
        assert!(
            (vals[0] - 0.0).abs() < 1e-5,
            "grad[0] should be 0 (neg input), got {}",
            vals[0]
        );
        assert!(
            vals[2].abs() < 1e-5,
            "grad[2] should be 0 (neg input), got {}",
            vals[2]
        );
        // Positive inputs should have non-zero gradient
        assert!(
            vals[1].abs() > 0.0,
            "grad[1] should be non-zero (pos input)"
        );
        assert!(
            vals[3].abs() > 0.0,
            "grad[3] should be non-zero (pos input)"
        );
        // All zero elements should be for negative inputs
        assert!(
            vals[1].abs() > 0.0,
            "positive input should have non-zero gradient"
        );
    }

    #[test]
    fn test_backward_twice() {
        let a = seq_tensor(4).requires_grad_(true);
        let b = seq_tensor(4).requires_grad_(true);
        let c = a.add(&b);
        let loss = c.sum(0, false);

        // First backward
        backward(&loss, None);
        let grad_a_1 = a.grad().expect("a should have grad after first backward");
        let val_1: f32 = unsafe { *(grad_a_1.data_ptr_f32() as *const f32) };
        // d(sum(a+b))/da = 1 for each element → sum of grads = 4
        assert!(
            (val_1 - 1.0).abs() < 1e-5,
            "first backward: grad should be 1, got {}",
            val_1
        );

        // Second backward (gradients accumulate)
        backward(&loss, None);
        let grad_a_2 = a.grad().expect("a should have grad after second backward");
        let val_2: f32 = unsafe { *(grad_a_2.data_ptr_f32() as *const f32) };
        // Should accumulate: 1 + 1 = 2
        assert!(
            (val_2 - 2.0).abs() < 1e-5,
            "second backward: grad should be 2 (accumulated), got {}",
            val_2
        );
    }

    #[test]
    fn test_no_grad() {
        let a = seq_tensor(4).requires_grad_(true);
        let c = a.add(&a);
        let loss = c.mean(0, false);

        // With no_grad guard
        {
            let _guard = NoGradGuard::new();
            backward(&loss, None);
        }

        // No gradients should have been computed
        assert!(
            a.grad().is_none(),
            "no_grad: gradients should not be computed"
        );
    }

    #[test]
    fn test_training_step() {
        // Simple 2-layer MLP equivalent:
        // x → matmul(W1) → relu → matmul(W2) → mean → loss
        let x = Tensor::zeros(vec![1, 4], DType::F32, Device::Cpu);
        let w1 = Tensor::zeros(vec![4, 8], DType::F32, Device::Cpu).requires_grad_(true);
        let w2 = Tensor::zeros(vec![8, 2], DType::F32, Device::Cpu).requires_grad_(true);

        // Fill with small random-ish values
        let x_ptr = x.data_ptr_f32() as *mut f32;
        for i in 0..4 {
            unsafe {
                *x_ptr.add(i) = (i as f32 + 1.0) / 10.0;
            }
        }
        let w1_ptr = w1.data_ptr_f32() as *mut f32;
        for i in 0..32 {
            unsafe {
                *w1_ptr.add(i) = ((i % 5) as f32) / 10.0;
            }
        }
        let w2_ptr = w2.data_ptr_f32() as *mut f32;
        for i in 0..16 {
            unsafe {
                *w2_ptr.add(i) = ((i % 3) as f32) / 10.0;
            }
        }

        // Forward pass
        let h = x.matmul(&w1);
        let h_relu = h.relu();
        let logits = h_relu.matmul(&w2);
        let loss = logits.mean(0, false);

        let w1_init = w1.grad();
        assert!(w1_init.is_none(), "w1 should not have grad before backward");

        // Backward pass
        backward(&loss, None);

        let w1_grad = w1.grad().expect("w1 should have grad after backward");
        let w2_grad = w2.grad().expect("w2 should have grad after backward");

        assert_eq!(w1.shape(), w1_grad.shape(), "w1 grad shape mismatch");
        assert_eq!(w2.shape(), w2_grad.shape(), "w2 grad shape mismatch");

        // Verify gradients are non-zero
        let w1_grad_sum: f32 = unsafe {
            let ptr = w1_grad.data_ptr_f32() as *const f32;
            (0..32).map(|i| *ptr.add(i)).sum()
        };
        assert!(
            w1_grad_sum.abs() > 0.0,
            "w1 gradient sum should be non-zero"
        );
    }

    #[test]
    fn test_backward_no_requires_grad() {
        // When no tensor requires grad, backward should be a no-op
        let a = seq_tensor(4); // no requires_grad
        let b = seq_tensor(4);
        let c = a.add(&b);
        let loss = c.mean(0, false);

        // This should not panic
        backward(&loss, None);
        // All grads should be None
        assert!(a.grad().is_none(), "a should not have grad");
        assert!(b.grad().is_none(), "b should not have grad");
    }
}

pub fn make_edges(tensor_a: &Tensor, tensor_b: &Tensor) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(2);
    if let Some(node) = tensor_a.grad_fn() {
        edges.push(Edge(node, 0));
    }
    if let Some(node) = tensor_b.grad_fn() {
        edges.push(Edge(node, 1));
    }
    edges
}

/// Helper: create a NodeInfo and wrap in Arc. Used by all forward ops.
pub fn make_node_info(op_name: &'static str, inputs: Vec<Tensor>) -> Arc<NodeInfo> {
    Arc::new(NodeInfo::new(op_name, inputs))
}

/// Helper: set a NodeInfo on an output tensor (fast-path, inline pattern).
pub fn attach_node_info(output: &mut Tensor, op_name: &'static str, inputs: Vec<Tensor>) {
    let mut meta = AutogradMeta::new_non_leaf(true);
    meta.grad_fn = Some(make_node_info(op_name, inputs));
    let inner = Arc::make_mut(&mut output.inner);
    inner.autograd_meta = Some(Arc::new(std::sync::Mutex::new(meta)));
    inner.requires_grad = true;
}

// =============================================================================
// Stub backward node types for v1.x code compatibility.
// These are all no-ops — the real backward pass is now compile-time graph
// transformation (build_backward_graph).
// =============================================================================

macro_rules! stub_backward {
    ($name:ident) => {
        stub_backward!($name, 0);
    };
    ($name:ident, $ninputs:expr) => {
        pub struct $name {
            edges: Vec<Edge>,
            inputs: Vec<Tensor>,
        }
        impl $name {
            pub fn new(edges: Vec<Edge>, inputs: Vec<Tensor>) -> Self {
                $name { edges, inputs }
            }
        }
        impl Node for $name {
            fn apply(
                &self,
                _grad_outputs: Vec<Option<Tensor>>,
                _output_tensor_id: usize,
            ) -> Vec<Option<Tensor>> {
                vec![None; self.num_inputs()]
            }
            fn next_edges(&self) -> &[Edge] {
                &self.edges
            }
            fn num_inputs(&self) -> usize {
                $ninputs
            }
            fn name(&self) -> &str {
                stringify!($name)
            }
            fn inputs(&self) -> &[Tensor] {
                &self.inputs
            }
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

stub_backward!(ErfBackward, 1);
stub_backward!(CumSumBackward, 1);
stub_backward!(GatherBackward, 2);

use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};

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
    let loss_node_ref = forward_graph
        .get_node(loss_node)
        .ok_or("build_backward_graph: loss node not found")?;
    let loss_shape = loss_node_ref.output_type.shape.clone();
    let loss_dtype = loss_node_ref.output_type.dtype.clone();

    // Create constant 1.0 gradient for the loss
    let one_tensor =
        create_constant_scalar(1.0f32, &loss_shape, loss_dtype.clone(), &mut grad_graph);
    grads.insert(loss_node, one_tensor);

    // Topological sort in REVERSE order (from output to input)
    let forward_order = forward_graph.topological_sort();

    // Walk nodes in reverse order
    for &node_id in forward_order.iter().rev() {
        let node = forward_graph
            .get_node(node_id)
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
                    let input_node = forward_graph
                        .get_node(input_id)
                        .ok_or("build_backward_graph: input not found")?;
                    let input_type = input_node.output_type.clone();

                    let mut attrs = HashMap::new();
                    attrs.insert("op".to_string(), "relu_backward".to_string());
                    let grad_input =
                        grad_graph.add_node(Opcode::Mul, vec![grad_id, node_id], input_type);
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
                    let g1 = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node.inputs[1]],
                        forward_graph
                            .get_node(node.inputs[1])
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, node.inputs[0], g1);

                    let g2 = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node.inputs[0]],
                        forward_graph
                            .get_node(node.inputs[0])
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, node.inputs[1], g2);
                }
            }
            Opcode::MatMul => {
                if node.inputs.len() >= 2 {
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_type = forward_graph.get_node(a_id).map(|n| n.output_type.clone());
                    let b_type = forward_graph.get_node(b_id).map(|n| n.output_type.clone());

                    // Handle fused activation: if fused_op is set, chain activation
                    // backward before the weight backward.
                    //   fused_op="OpRelu":  forward was MatMul(x,W)→Relu
                    //   fused_op="MatMulAddRelu":  forward was MatMul(x,W)→Add(,bias)→Relu
                    // The node_id here refers to the fused forward output (post-activation),
                    // so dRelu = Mul(grad, fwd_output) gives the gradient before activation.
                    let effective_grad = match node.attrs.get("fused_op").map(|s| s.as_str()) {
                        Some("OpRelu") | Some("MatMulAddRelu") => {
                            let mut attrs = HashMap::new();
                            attrs.insert("op".to_string(), "relu_backward".to_string());
                            let grad_input = grad_graph.add_node(
                                Opcode::Mul,
                                vec![grad_id, node_id],
                                node.output_type.clone(),
                            );
                            if let Some(n) = grad_graph.get_node_mut(grad_input) {
                                n.attrs = attrs;
                            }
                            grad_input
                        }
                        Some("OpGelu") | Some("MatMulAddGelu") => {
                            let mut attrs = HashMap::new();
                            attrs.insert("op".to_string(), "gelu_backward".to_string());
                            let grad_input = grad_graph.add_node(
                                Opcode::Mul,
                                vec![grad_id, node_id],
                                node.output_type.clone(),
                            );
                            if let Some(n) = grad_graph.get_node_mut(grad_input) {
                                n.attrs = attrs;
                            }
                            grad_input
                        }
                        Some("OpSilu") | Some("MatMulAddSilu") => {
                            let mut attrs = HashMap::new();
                            attrs.insert("op".to_string(), "silu_backward".to_string());
                            let grad_input = grad_graph.add_node(
                                Opcode::Mul,
                                vec![grad_id, node_id],
                                node.output_type.clone(),
                            );
                            if let Some(n) = grad_graph.get_node_mut(grad_input) {
                                n.attrs = attrs;
                            }
                            grad_input
                        }
                        _ => grad_id,
                    };

                    let b_t = grad_graph.add_node(
                        Opcode::Transpose,
                        vec![b_id],
                        b_type
                            .clone()
                            .unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    let da = grad_graph.add_node(
                        Opcode::MatMul,
                        vec![effective_grad, b_t],
                        a_type
                            .clone()
                            .unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, a_id, da);

                    let a_t = grad_graph.add_node(
                        Opcode::Transpose,
                        vec![a_id],
                        a_type.unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    let db = grad_graph.add_node(
                        Opcode::MatMul,
                        vec![a_t, effective_grad],
                        b_type.unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, b_id, db);

                    // For fused MatMulAdd{Relu,Gelu,Silu}: bias lives at input[2]
                    if let Some(fused) = node.attrs.get("fused_op").map(|s| s.as_str()) {
                        if fused == "MatMulAddRelu"
                            || fused == "MatMulAddGelu"
                            || fused == "MatMulAddSilu"
                        {
                            if let Some(&bias_id) = node.inputs.get(2) {
                                let dbias = grad_graph.add_node(
                                    Opcode::ReduceSum,
                                    vec![effective_grad],
                                    forward_graph
                                        .get_node(bias_id)
                                        .map(|n| n.output_type.clone())
                                        .unwrap(),
                                );
                                accumulate_grad(&mut grad_graph, &mut grads, bias_id, dbias);
                            }
                        }
                    }
                }
            }
            Opcode::Gelu => {
                let mut attrs = HashMap::new();
                attrs.insert("op".to_string(), "gelu_backward".to_string());
                let grad_input = grad_graph.add_node(
                    Opcode::Mul,
                    vec![grad_id, node_id],
                    forward_graph
                        .get_node(node.inputs[0])
                        .map(|n| n.output_type.clone())
                        .unwrap(),
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
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Log => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Div,
                        vec![grad_id, node_id],
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
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
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, sig_mul],
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
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
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
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
                // Forward: y = sum(x, dims)
                // Backward: dx = dy  (broadcast reduced dims back to input shape)
                if let Some(&input_id) = node.inputs.first() {
                    let input_type = forward_graph
                        .get_node(input_id)
                        .map(|n| n.output_type.clone())
                        .unwrap();
                    // Get the gradient's output type from grad_graph (all nodes live there)
                    let grad_type = grad_graph
                        .get_node(grad_id)
                        .map(|n| n.output_type.clone())
                        .unwrap();
                    let input_numel = input_type.numel().unwrap_or(1);
                    let grad_numel = grad_type.numel().unwrap_or(1);
                    if input_numel == grad_numel {
                        // Same element count: reshape grad to input shape (adds dims, no data change)
                        let reshaped =
                            grad_graph.add_node(Opcode::Reshape, vec![grad_id], input_type.clone());
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, reshaped);
                    } else {
                        // Scalar gradient (e.g., reduce over all dims): broadcast via MulScalar
                        let ones = create_constant_scalar(
                            1.0,
                            &input_type.shape,
                            input_type.dtype.clone(),
                            &mut grad_graph,
                        );
                        let scaled = grad_graph.add_node(
                            Opcode::MulScalar,
                            vec![ones, grad_id],
                            input_type.clone(),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, scaled);
                    }
                }
            }
            Opcode::ReduceMean => {
                // Forward: y = mean(x, dims)
                // Backward: dx = dy / n
                //
                // We must NOT use total input numel as n;  instead compute n
                // as the PRODUCT of the REDUCED dimensions only:
                //
                //     n = input_numel / output_numel
                //
                // This correctly handles cases like mean([[a,b]], dim=0) where
                // the reduced dim has size 1 (n=1, total numel=2).
                if let Some(&input_id) = node.inputs.first() {
                    let input_type = forward_graph
                        .get_node(input_id)
                        .map(|n| n.output_type.clone())
                        .unwrap();
                    // Get the gradient's output type from grad_graph (all nodes live there)
                    let grad_type = grad_graph
                        .get_node(grad_id)
                        .map(|n| n.output_type.clone())
                        .unwrap();
                    let input_numel = input_type.numel().unwrap_or(1);
                    let output_numel = node.output_type.numel().unwrap_or(1);
                    let grad_numel = grad_type.numel().unwrap_or(1);
                    // n = product of reduced dim sizes = total_input / total_output
                    let n = (input_numel as f32) / (output_numel as f32);
                    let inv_n = create_constant_scalar(
                        1.0 / n,
                        &input_type.shape,
                        input_type.dtype.clone(),
                        &mut grad_graph,
                    );
                    if input_numel == grad_numel {
                        // Same element count: reshape grad to input shape, then Mul element-wise
                        let reshaped =
                            grad_graph.add_node(Opcode::Reshape, vec![grad_id], input_type.clone());
                        let scaled_grad = grad_graph.add_node(
                            Opcode::Mul,
                            vec![reshaped, inv_n],
                            input_type.clone(),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, scaled_grad);
                    } else {
                        // Scalar gradient: MulScalar broadcasts the scalar across all elements
                        let scaled_grad = grad_graph.add_node(
                            Opcode::MulScalar,
                            vec![inv_n, grad_id],
                            input_type.clone(),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, scaled_grad);
                    }
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
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Abs => {
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node_id],
                        forward_graph
                            .get_node(input_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
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
                let effective_grad = match node.attrs.get("fused_op").map(|s| s.as_str()) {
                    Some("OpRelu") => grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, node_id],
                        node.output_type.clone(),
                    ),
                    _ => grad_id,
                };
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, effective_grad);
                }
                if let Some(&bias_id) = node.inputs.get(1) {
                    let grad_input = grad_graph.add_node(
                        Opcode::ReduceSum,
                        vec![effective_grad],
                        forward_graph
                            .get_node(bias_id)
                            .map(|n| n.output_type.clone())
                            .unwrap(),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, bias_id, grad_input);
                }
            }
            Opcode::Div => {
                // d(x/y)/dx = 1/y,  d(x/y)/dy = -x/y²
                if let Some(&x_id) = node.inputs.first() {
                    let y_id = node.inputs.get(1).copied().unwrap_or(x_id);
                    let y_type = forward_graph.get_node(y_id).map(|n| n.output_type.clone());
                    // dx = grad / y
                    let dx = grad_graph.add_node(
                        Opcode::Div,
                        vec![grad_id, y_id],
                        y_type
                            .clone()
                            .unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, x_id, dx);
                }
                if node.inputs.len() > 1 {
                    if let Some(&y_id) = node.inputs.get(1) {
                        let x_id = node.inputs[0];
                        let x_type = forward_graph.get_node(x_id).map(|n| n.output_type.clone());
                        // dy = -grad * x / y²
                        let y_sq = grad_graph.add_node(
                            Opcode::Mul,
                            vec![y_id, y_id],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let grad_x = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, x_id],
                            x_type
                                .clone()
                                .unwrap_or(TensorType::new(vec![], IrDType::F32)),
                        );
                        let temp = grad_graph.add_node(
                            Opcode::Div,
                            vec![grad_x, y_sq],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dy = create_neg(temp, &mut grad_graph);
                        accumulate_grad(&mut grad_graph, &mut grads, y_id, dy);
                    }
                }
            }
            Opcode::Silu => {
                // silu(x) = x * sigmoid(x)
                // derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                if let Some(&input_id) = node.inputs.first() {
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let sig_x = grad_graph.add_node(
                        Opcode::Sigmoid,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let one_minus_sig = grad_graph.add_node(
                        Opcode::Sub,
                        vec![one, sig_x],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_times_1ms = grad_graph.add_node(
                        Opcode::Mul,
                        vec![input_id, one_minus_sig],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let one2 = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let one_plus = grad_graph.add_node(
                        Opcode::Add,
                        vec![one2, x_times_1ms],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let deriv = grad_graph.add_node(
                        Opcode::Mul,
                        vec![sig_x, one_plus],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::LeakyRelu => {
                // d_input = d_output * (x > 0 ? 1.0 : alpha)
                if let Some(&input_id) = node.inputs.first() {
                    let alpha: f32 = node
                        .attrs
                        .get("negative_slope")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0.01);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let alpha_c = create_constant_scalar(alpha, &[], IrDType::F32, &mut grad_graph);
                    let mask = grad_graph.add_node(
                        Opcode::GtScalar,
                        vec![input_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let neg_grad = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, alpha_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask, grad_id, neg_grad],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Elu => {
                // d_input = d_output * (x > 0 ? 1 : alpha * exp(x))
                if let Some(&input_id) = node.inputs.first() {
                    let alpha: f32 = node
                        .attrs
                        .get("alpha")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(1.0);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let alpha_c = create_constant_scalar(alpha, &[], IrDType::F32, &mut grad_graph);
                    let exp_x = grad_graph.add_node(
                        Opcode::Exp,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let alpha_exp = grad_graph.add_node(
                        Opcode::Mul,
                        vec![alpha_c, exp_x],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mask = grad_graph.add_node(
                        Opcode::GtScalar,
                        vec![input_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask, grad_id, alpha_exp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Softplus => {
                // d_input = d_output * sigmoid(x)
                if let Some(&input_id) = node.inputs.first() {
                    let sig = grad_graph.add_node(
                        Opcode::Sigmoid,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, sig],
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
                    let neg_three =
                        create_constant_scalar(-3.0f32, &[], IrDType::F32, &mut grad_graph);
                    let two_thirds =
                        create_constant_scalar(2.0f32 / 3.0f32, &[], IrDType::F32, &mut grad_graph);
                    let half = create_constant_scalar(0.5f32, &[], IrDType::F32, &mut grad_graph);
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    // mask_high = x > 3
                    let mask_high = grad_graph.add_node(
                        Opcode::GtScalar,
                        vec![input_id, three],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // mask_low = x < -3
                    let mask_low = grad_graph.add_node(
                        Opcode::LtScalar,
                        vec![input_id, neg_three],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // linear part: 2*x/3 + 0.5
                    let x_23 = grad_graph.add_node(
                        Opcode::Mul,
                        vec![input_id, two_thirds],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mid_deriv = grad_graph.add_node(
                        Opcode::Add,
                        vec![x_23, half],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // not_low: grad * mid_deriv  (pass-through for masked region)
                    let not_low_deriv = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask_low, zero, mid_deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // deriv = 1 where x > 3, mid_deriv where -3 <= x <= 3, 0 where x < -3
                    let deriv = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask_high, one, not_low_deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, deriv],
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
                        Opcode::Softplus,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let tanh_sp = grad_graph.add_node(
                        Opcode::Tanh,
                        vec![sp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sig_x = grad_graph.add_node(
                        Opcode::Sigmoid,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // sech²(sp) = 1 - tanh²(sp)
                    let tanh_sq = grad_graph.add_node(
                        Opcode::Mul,
                        vec![tanh_sp, tanh_sp],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sech2 = grad_graph.add_node(
                        Opcode::Sub,
                        vec![one, tanh_sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_times_sig = grad_graph.add_node(
                        Opcode::Mul,
                        vec![input_id, sig_x],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_sech2_sig = grad_graph.add_node(
                        Opcode::Mul,
                        vec![x_times_sig, sech2],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let deriv = grad_graph.add_node(
                        Opcode::Add,
                        vec![tanh_sp, x_sech2_sig],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, deriv],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Clamp => {
                // d_input = d_output where x in [min, max], else 0
                if let Some(&input_id) = node.inputs.first() {
                    let min_val: f32 = node
                        .attrs
                        .get("min")
                        .and_then(|m| m.parse().ok())
                        .unwrap_or(f32::NEG_INFINITY);
                    let max_val: f32 = node
                        .attrs
                        .get("max")
                        .and_then(|m| m.parse().ok())
                        .unwrap_or(f32::INFINITY);
                    let min_c = create_constant_scalar(min_val, &[], IrDType::F32, &mut grad_graph);
                    let max_c = create_constant_scalar(max_val, &[], IrDType::F32, &mut grad_graph);
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    // mask = x >= min AND x <= max
                    let ge_min = grad_graph.add_node(
                        Opcode::GtScalar,
                        vec![input_id, min_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let le_max = grad_graph.add_node(
                        Opcode::LtScalar,
                        vec![input_id, max_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // For inclusive bounds, handle equality:
                    let eq_min = grad_graph.add_node(
                        Opcode::EqScalar,
                        vec![input_id, min_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let eq_max = grad_graph.add_node(
                        Opcode::EqScalar,
                        vec![input_id, max_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ge_or_eq = grad_graph.add_node(
                        Opcode::Add,
                        vec![ge_min, eq_min],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let le_or_eq = grad_graph.add_node(
                        Opcode::Add,
                        vec![le_max, eq_max],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let mask = grad_graph.add_node(
                        Opcode::Mul,
                        vec![ge_or_eq, le_or_eq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask, grad_id, zero],
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
                        Opcode::Mul,
                        vec![grad_id, zero],
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
                        Opcode::Exp,
                        vec![node_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // sum of grad along appropriate axis
                    // For now, reduce all dims (same as softmax backward)
                    let grad_sum = grad_graph.add_node(
                        Opcode::ReduceSum,
                        vec![grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_sm = grad_graph.add_node(
                        Opcode::Mul,
                        vec![sm, grad_sum],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Sub,
                        vec![grad_id, grad_sm],
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
                        Opcode::Mul,
                        vec![node_id, grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sum_sg = grad_graph.add_node(
                        Opcode::ReduceSum,
                        vec![s_times_grad],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_centered = grad_graph.add_node(
                        Opcode::Sub,
                        vec![grad_id, sum_sg],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![node_id, grad_centered],
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
                    let eps_val: f64 = node
                        .attrs
                        .get("eps")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(1e-5);
                    let eps_c =
                        create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    // gamma = inputs[1] (weight), var = inputs[4]
                    if node.inputs.len() >= 5 {
                        let gamma_id = node.inputs[1];
                        let var_id = node.inputs[4];
                        // sqrt(var + eps)
                        let var_eps = grad_graph.add_node(
                            Opcode::Add,
                            vec![var_id, eps_c],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let std = grad_graph.add_node(
                            Opcode::Sqrt,
                            vec![var_eps],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        // gamma / std
                        let gamma_div_std = grad_graph.add_node(
                            Opcode::Div,
                            vec![gamma_id, std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        // dx = grad * gamma / std
                        let grad_input = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, gamma_div_std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        // dgamma: sum(grad * (x - mean) / std)
                        if let Some(&mean_id) = node.inputs.get(3) {
                            let x_minus_mean = grad_graph.add_node(
                                Opcode::Sub,
                                vec![input_id, mean_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let x_minus_mean_div_std = grad_graph.add_node(
                                Opcode::Div,
                                vec![x_minus_mean, std],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let grad_gamma_unreduced = grad_graph.add_node(
                                Opcode::Mul,
                                vec![grad_id, x_minus_mean_div_std],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            let grad_gamma = grad_graph.add_node(
                                Opcode::ReduceSum,
                                vec![grad_gamma_unreduced],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, gamma_id, grad_gamma);
                        }
                        // dbeta: sum(grad)
                        if let Some(&beta_id) = node.inputs.get(2) {
                            let grad_beta = grad_graph.add_node(
                                Opcode::ReduceSum,
                                vec![grad_id],
                                TensorType::new(vec![], IrDType::F32),
                            );
                            accumulate_grad(&mut grad_graph, &mut grads, beta_id, grad_beta);
                        }
                    }
                }
            }
            Opcode::LayerNorm => {
                if let Some(&input_id) = node.inputs.first() {
                    let eps_val: f64 = node
                        .attrs
                        .get("eps")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(1e-5);
                    let eps_c =
                        create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    let mean = grad_graph.add_node(
                        Opcode::ReduceMean,
                        vec![input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_minus_mean = grad_graph.add_node(
                        Opcode::Sub,
                        vec![input_id, mean],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let sq = grad_graph.add_node(
                        Opcode::Mul,
                        vec![x_minus_mean, x_minus_mean],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let var = grad_graph.add_node(
                        Opcode::ReduceMean,
                        vec![sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let var_eps = grad_graph.add_node(
                        Opcode::Add,
                        vec![var, eps_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let std = grad_graph.add_node(
                        Opcode::Sqrt,
                        vec![var_eps],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let x_hat = grad_graph.add_node(
                        Opcode::Div,
                        vec![x_minus_mean, std],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    if node.inputs.len() >= 2 {
                        let gamma_id = node.inputs[1];
                        let dx_hat = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, gamma_id],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let grad_input = grad_graph.add_node(
                            Opcode::Div,
                            vec![dx_hat, std],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        let dy_x_hat = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dgamma = grad_graph.add_node(
                            Opcode::ReduceSum,
                            vec![dy_x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, gamma_id, dgamma);
                        if let Some(&beta_id) = node.inputs.get(2) {
                            let dbeta = grad_graph.add_node(
                                Opcode::ReduceSum,
                                vec![grad_id],
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
                    let eps_val: f64 = node
                        .attrs
                        .get("eps")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(1e-5);
                    let eps_c =
                        create_constant_scalar(eps_val as f32, &[], IrDType::F32, &mut grad_graph);
                    let sq = grad_graph.add_node(
                        Opcode::Mul,
                        vec![input_id, input_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ms = grad_graph.add_node(
                        Opcode::ReduceMean,
                        vec![sq],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let ms_eps = grad_graph.add_node(
                        Opcode::Add,
                        vec![ms, eps_c],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    let rms = grad_graph.add_node(
                        Opcode::Sqrt,
                        vec![ms_eps],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    if node.inputs.len() >= 2 {
                        let gamma_id = node.inputs[1];
                        let gamma_div_rms = grad_graph.add_node(
                            Opcode::Div,
                            vec![gamma_id, rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let grad_input = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, gamma_div_rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                        let x_hat = grad_graph.add_node(
                            Opcode::Div,
                            vec![input_id, rms],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dy_x_hat = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dgamma = grad_graph.add_node(
                            Opcode::ReduceSum,
                            vec![dy_x_hat],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, gamma_id, dgamma);
                    } else {
                        accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                    }
                }
            }
            Opcode::MaxPool => {
                // MaxPool now stores argmax indices as a secondary output
                // (node_id, output_index=1). A proper backward would use these
                // indices to scatter grad to the winning positions via ScatterNd.
                // For now: pass gradient through (approximately correct for
                // non-overlapping pools).
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::AvgPool => {
                if let Some(&input_id) = node.inputs.first() {
                    let kernel_size: usize = node
                        .attrs
                        .get("kernel_size")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let pool_area = (kernel_size * kernel_size) as f32;
                    let scale =
                        create_constant_scalar(1.0 / pool_area, &[], IrDType::F32, &mut grad_graph);
                    let grad_input = grad_graph.add_node(
                        Opcode::Mul,
                        vec![grad_id, scale],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_input);
                }
            }
            Opcode::Constant(_)
            | Opcode::Input
            | Opcode::GtScalar
            | Opcode::LtScalar
            | Opcode::EqScalar
            | Opcode::SgdUpdate
            | Opcode::AdamUpdate
            | Opcode::AdamWUpdate
            | Opcode::MuonUpdate
            | Opcode::LionUpdate
            | Opcode::RmspropUpdate
            | Opcode::Shape
            | Opcode::Range
            | Opcode::FusedResidualAddNorm => {}
            Opcode::Transpose => {
                // gradient of transpose is transpose with inverse permutation
                if let Some(&input_id) = node.inputs.first() {
                    let grad_input = grad_graph.add_node(
                        Opcode::Transpose,
                        vec![grad_id],
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
                // Uses GtScalar for Maximum (x > y → dx gets grad) or
                // LtScalar for Minimum (x < y → dx gets grad).
                if node.inputs.len() >= 2 {
                    let x_id = node.inputs[0];
                    let y_id = node.inputs[1];
                    let zero = create_constant_scalar(0.0f32, &[], IrDType::F32, &mut grad_graph);
                    let is_max = matches!(node.opcode, Opcode::Maximum);
                    let mask = grad_graph.add_node(
                        if is_max {
                            Opcode::GtScalar
                        } else {
                            Opcode::LtScalar
                        },
                        vec![x_id, y_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    // dx = Where(mask, grad, 0)
                    let dx = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask, grad_id, zero],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, x_id, dx);
                    // dy = Where(mask, 0, grad)
                    let dy = grad_graph.add_node(
                        Opcode::Where,
                        vec![mask, zero, grad_id],
                        TensorType::new(vec![], IrDType::F32),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, y_id, dy);
                }
            }
            Opcode::ReduceMax | Opcode::ArgMax => {
                // pass grad to first input (argmax has no gradient, max needs argmax mask)
                if let Some(&input_id) = node.inputs.first() {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::Pow => {
                // d(x^e)/dx = e * x^(e-1)
                // d(x^e)/de = ln(x) * x^e = ln(x) * output
                if let Some(&x_id) = node.inputs.first() {
                    let one = create_constant_scalar(1.0f32, &[], IrDType::F32, &mut grad_graph);
                    let x_type = forward_graph.get_node(x_id).map(|n| n.output_type.clone());
                    // dx = grad * exponent * Pow(x, exponent - 1)
                    let exponent_id = if node.inputs.len() >= 2 {
                        node.inputs.get(1).copied()
                    } else {
                        None
                    };
                    if let Some(exp_id) = exponent_id {
                        let exp_minus_1 = grad_graph.add_node(
                            Opcode::Sub,
                            vec![exp_id, one],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let x_pow = grad_graph.add_node(
                            Opcode::Pow,
                            vec![x_id, exp_minus_1],
                            x_type
                                .clone()
                                .unwrap_or(TensorType::new(vec![], IrDType::F32)),
                        );
                        let dx_unscaled = grad_graph.add_node(
                            Opcode::Mul,
                            vec![exp_id, x_pow],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dx = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, dx_unscaled],
                            x_type.unwrap_or(TensorType::new(vec![], IrDType::F32)),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, x_id, dx);
                    }
                }
                // Gradient for the exponent (less common, but mathematically correct)
                if node.inputs.len() > 1 {
                    if let Some(&exp_id) = node.inputs.get(1) {
                        let x_id = node.inputs[0];
                        let ln_x = grad_graph.add_node(
                            Opcode::Log,
                            vec![x_id],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        // d_exponent = grad * ln(x) * output
                        let ln_times_out = grad_graph.add_node(
                            Opcode::Mul,
                            vec![ln_x, node_id],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        let dexponent = grad_graph.add_node(
                            Opcode::Mul,
                            vec![grad_id, ln_times_out],
                            TensorType::new(vec![], IrDType::F32),
                        );
                        accumulate_grad(&mut grad_graph, &mut grads, exp_id, dexponent);
                    }
                }
            }
            Opcode::Prelu
            | Opcode::Embedding
            | Opcode::AddScalar
            | Opcode::DivScalar
            | Opcode::UpsampleNearest2d
            | Opcode::UpsampleBilinear2d
            | Opcode::AdaptiveAvgPool2d
            | Opcode::Repeat
            | Opcode::CumSum
            | Opcode::Erf
            | Opcode::Flip
            | Opcode::Where
            | Opcode::TopK
            | Opcode::Cast
            | Opcode::Expand
            | Opcode::Tile
            | Opcode::Quantize
            | Opcode::Dequantize
            | Opcode::ToF16
            | Opcode::ToF32
            | Opcode::QuantizeActivations
            | Opcode::DequantizeActivations => {
                for &input_id in &node.inputs {
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, grad_id);
                }
            }
            Opcode::MulScalar => {
                // Forward: y = a * b  (element-wise multiply with broadcasting)
                // Backward: da = dy * b,  db = dy * a
                if node.inputs.len() >= 2 {
                    let a_id = node.inputs[0];
                    let b_id = node.inputs[1];
                    let a_type = forward_graph.get_node(a_id).map(|n| n.output_type.clone());
                    let b_type = forward_graph.get_node(b_id).map(|n| n.output_type.clone());
                    // The gradient for each input has the same shape as the input,
                    // so use the forward input's output type (which is the TensorValue
                    // shape that determines buffer size).
                    let da = grad_graph.add_node(
                        Opcode::MulScalar,
                        vec![grad_id, b_id],
                        a_type.unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, a_id, da);
                    let db = grad_graph.add_node(
                        Opcode::MulScalar,
                        vec![grad_id, a_id],
                        b_type.unwrap_or(TensorType::new(vec![], IrDType::F32)),
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, b_id, db);
                }
            }
            Opcode::GradientScale => {
                // Forward: y = x * scale  (scale is an attribute)
                // Backward: d_input = d_output * scale
                if let Some(&input_id) = node.inputs.first() {
                    let scale_attr = node
                        .attrs
                        .get("scale")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1.0);
                    let scale_const =
                        create_constant_scalar(scale_attr, &[], IrDType::F32, &mut grad_graph);
                    let input_type = forward_graph
                        .get_node(input_id)
                        .map(|n| n.output_type.clone())
                        .unwrap_or(TensorType::new(vec![], IrDType::F32));
                    let d_input = grad_graph.add_node(
                        Opcode::MulScalar,
                        vec![grad_id, scale_const],
                        input_type, // use the input's shape so the gradient buffer is correctly sized
                    );
                    accumulate_grad(&mut grad_graph, &mut grads, input_id, d_input);
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
fn create_constant_scalar(
    value: f32,
    shape: &[DimExpr],
    dtype: IrDType,
    graph: &mut ComputeGraph,
) -> NodeId {
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
