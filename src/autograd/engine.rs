use crate::autograd::{Edge, Node};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Check for NaN or Inf in a tensor's data (debug builds only)
#[cfg(debug_assertions)]
fn check_gradient_validity(tensor: &Tensor, context: &str) {
    if let Some(data) = tensor.inner.cpu_data() {
        let f32_data: &[f32] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const f32,
                data.len() / std::mem::size_of::<f32>(),
            )
        };
        for (i, &val) in f32_data.iter().enumerate() {
            if val.is_nan() {
                eprintln!(
                    "WARNING: NaN detected in gradient at index {} during {}. \
                     This may indicate numerical instability.",
                    i, context
                );
                break;
            }
            if val.is_infinite() {
                eprintln!(
                    "WARNING: Inf detected in gradient at index {} during {}. \
                     This may indicate numerical instability.",
                    i, context
                );
                break;
            }
        }
    }
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn check_gradient_validity(_tensor: &Tensor, _context: &str) {
    // No-op in release builds
}

pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    if !root.requires_grad() {
        return;
    }

    // Warn if root has no grad_fn (no computation graph to backprop through)
    if root.grad_fn().is_none() && !root.is_leaf() {
        eprintln!(
            "WARNING: backward() called on a non-leaf tensor without grad_fn. \
             This means no computation graph was built during the forward pass. \
             Gradients will not be computed for any parameters."
        );
    }

    let grad_output = grad_output.unwrap_or_else(|| {
        // Create scalar gradient on the same device as root
        Tensor::full(vec![], 1.0, root.dtype(), root.device())
    });

    // Map from tensor_id to accumulated gradient
    let mut grads: HashMap<usize, Tensor> = HashMap::new();

    // Map from node_ptr to number of dependencies (input tensors that still need gradients)
    let mut node_dependencies: HashMap<usize, usize> = HashMap::new();

    // Map from node_ptr to the node itself and its output tensor id
    let mut node_info: HashMap<usize, (Arc<dyn Node>, usize)> = HashMap::new();

    let root_id = root.id();
    grads.insert(root_id, grad_output);

    // Build the graph and count how many gradient contributions each node needs
    // A node needs to wait for all its consumers (nodes that use its output) to send gradients
    let mut queue: VecDeque<(Arc<dyn Node>, usize)> = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();

    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_id));
    }

    // First pass: traverse graph to build dependency counts
    // node_dependencies[node] = number of output edges from this node that lead to other nodes in the graph
    while let Some((node, tensor_id)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if visited.contains(&node_ptr) {
            continue;
        }

        visited.insert(node_ptr);

        // Store node info
        node_info.insert(node_ptr, (node.clone(), tensor_id));

        // Count how many of this node's output edges lead to unvisited nodes
        // This determines how many gradient contributions this node needs to wait for
        let next_edges = node.next_edges();
        let mut unvisited_edges = 0usize;
        for edge in next_edges.iter() {
            let Edge(next_node, _) = edge;
            let next_node_ptr = (&**next_node) as *const _ as *const () as usize;
            if !visited.contains(&next_node_ptr) {
                queue.push_back((next_node.clone(), next_node.id()));
                unvisited_edges += 1;
            }
        }
        // The node needs to wait for gradients from all paths that lead to the root
        // We count this as the number of incoming edges in the backward traversal
        node_dependencies.insert(node_ptr, unvisited_edges);
    }

    // Reset visited for processing
    visited.clear();

    // Second pass: process nodes in topological order
    // Start from root and work backwards, only processing when all deps are satisfied
    queue.clear();
    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_id));
    }

    while let Some((node, tensor_id)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        if visited.contains(&node_ptr) {
            continue;
        }

        // Check if all gradient contributions have arrived
        let deps = node_dependencies.get(&node_ptr).copied().unwrap_or(0);
        if deps > 0 {
            // Not all contributions arrived yet, re-queue
            queue.push_back((node, tensor_id));
            continue;
        }

        visited.insert(node_ptr);

        // Get the accumulated gradient for this node's output
        let grad_output_for_node = grads.remove(&tensor_id);

        // Check gradient validity in debug builds
        if let Some(ref grad) = grad_output_for_node {
            check_gradient_validity(grad, &format!("backward pass for node {}", node.name()));
        }

        // Apply backward to get gradients for inputs
        let grad_inputs = node.apply(vec![grad_output_for_node]);

        // Get the input tensors for this node
        let input_tensors = node.inputs();

        // Propagate gradients to input tensors
        for (input_tensor, grad_input_opt) in input_tensors.iter().zip(grad_inputs.into_iter()) {
            if let Some(grad_input) = grad_input_opt {
                if input_tensor.is_leaf() {
                    // Accumulate gradient for leaf tensor
                    if let Some(meta) = &input_tensor.inner.autograd_meta {
                        match meta.lock() {
                            Ok(mut lock) => {
                                if let Some(existing_grad) = &mut lock.grad {
                                    existing_grad.add_(&grad_input);
                                } else {
                                    lock.grad = Some(grad_input);
                                }
                            }
                            Err(_) => {
                                eprintln!(
                                    "WARNING: AutogradMeta lock poisoned for tensor {}. \
                                     Gradient accumulation failed.",
                                    input_tensor.id()
                                );
                            }
                        }
                    }
                } else {
                    // For non-leaf tensors, accumulate in grads map
                    let input_id = input_tensor.id();
                    match grads.get_mut(&input_id) {
                        Some(existing_grad) => {
                            existing_grad.add_(&grad_input);
                        }
                        None => {
                            grads.insert(input_id, grad_input);
                        }
                    }

                    // Add this tensor's grad_fn to queue if it exists
                    if let Some(input_grad_fn) = input_tensor.grad_fn() {
                        let input_node_ptr = (&*input_grad_fn) as *const _ as *const () as usize;
                        if !visited.contains(&input_node_ptr) {
                            // Decrement dependency count - one more gradient contribution has arrived
                            if let Some(deps) = node_dependencies.get_mut(&input_node_ptr) {
                                if *deps > 0 {
                                    *deps -= 1;
                                }
                            }
                            queue.push_back((input_grad_fn, input_id));
                        }
                    }
                }
            }
        }
    }
}
