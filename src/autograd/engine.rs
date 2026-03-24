use crate::autograd::{Edge, Node};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    if !root.requires_grad() {
        return;
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

    // Build the graph and count dependencies
    let mut queue: VecDeque<(Arc<dyn Node>, usize)> = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();

    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_id));
    }

    // Traverse the graph to build dependency counts
    while let Some((node, tensor_id)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if visited.contains(&node_ptr) {
            continue;
        }

        visited.insert(node_ptr);

        // Store node info
        node_info.insert(node_ptr, (node.clone(), tensor_id));

        // Count inputs for this node
        let num_inputs = node.num_inputs();
        node_dependencies.insert(node_ptr, num_inputs);

        // Add dependencies to queue
        let next_edges = node.next_edges();
        for edge in next_edges.iter() {
            let Edge(next_node, _) = edge;
            let next_node_ptr = (&**next_node) as *const _ as *const () as usize;
            if !visited.contains(&next_node_ptr) {
                queue.push_back((next_node.clone(), next_node.id()));
            }
        }
    }

    // Reset visited for processing
    visited.clear();

    // Process nodes: start from root and work backwards
    queue.clear();
    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_id));
    }

    while let Some((node, tensor_id)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        if visited.contains(&node_ptr) {
            continue;
        }

        // Get the gradient for this node's output
        let grad_output_for_node = grads.get(&tensor_id).cloned();

        // Apply backward to get gradients for inputs
        let grad_inputs = node.apply(&[grad_output_for_node]);

        // Get the input tensors for this node
        let input_tensors = node.inputs();

        // Process each input gradient
        for (input_tensor, grad_input_opt) in input_tensors.iter().zip(grad_inputs.iter()) {
            if let Some(grad_input) = grad_input_opt {
                if input_tensor.is_leaf() {
                    // Accumulate gradient for leaf tensor
                    if let Some(meta) = &input_tensor.inner.autograd_meta {
                        if let Ok(mut lock) = meta.lock() {
                            if let Some(existing_grad) = &mut lock.grad {
                                // In-place addition for gradient accumulation
                                existing_grad.add_(grad_input);
                            } else {
                                lock.grad = Some(grad_input.clone());
                            }
                        }
                    }
                } else {
                    // For non-leaf tensors, accumulate in grads map
                    let input_id = input_tensor.id();
                    if let Some(existing_grad) = grads.get_mut(&input_id) {
                        existing_grad.add_(grad_input);
                    } else {
                        grads.insert(input_id, grad_input.clone());
                    }

                    // Add this tensor's grad_fn to queue if it hasn't been processed
                    if let Some(input_grad_fn) = input_tensor.grad_fn() {
                        let input_node_ptr = (&*input_grad_fn) as *const _ as *const () as usize;
                        if !visited.contains(&input_node_ptr) {
                            queue.push_back((input_grad_fn, input_id));
                        }
                    }
                }
            }
        }

        visited.insert(node_ptr);
    }
}
