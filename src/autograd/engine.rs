use crate::autograd::{Edge, Node};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub fn backward(root: &Tensor, grad_output: Option<Tensor>, _retain_graph: bool) {
    if !root.requires_grad() {
        return;
    }

    let grad_output = grad_output.unwrap_or_else(|| {
        // Create scalar gradient on the same device as root
        Tensor::full(vec![], 1.0, root.dtype(), root.device())
    });

    // Map from tensor_id to accumulated gradient
    let mut grads: HashMap<usize, Tensor> = HashMap::new();

    // Map from node_ptr to the node itself and its output tensor id
    let mut node_info: HashMap<usize, (Arc<dyn Node>, usize)> = HashMap::new();

    let root_id = root.id();
    grads.insert(root_id, grad_output);

    // PERF-9: Check for cached topological order
    let root_meta = root.inner.autograd_meta.clone();
    let mut cached_topo_order: Option<Vec<Arc<dyn Node>>> = None;

    if let Some(meta) = &root_meta {
        if let Ok(mut lock) = meta.lock() {
            cached_topo_order = lock.topo_order_cache.clone();
        }
    }

    let topo_order = if let Some(cached) = cached_topo_order {
        // Use cached topological order
        cached
    } else {
        // Build the graph and count dependencies (topological sort)
        let mut queue: VecDeque<(Arc<dyn Node>, usize)> = VecDeque::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut topo_order_vec: Vec<Arc<dyn Node>> = Vec::new();

        if let Some(grad_fn) = root.grad_fn() {
            queue.push_back((grad_fn, root_id));
        }

        // Traverse the graph to build topological order
        while let Some((node, tensor_id)) = queue.pop_front() {
            let node_ptr = (&*node) as *const _ as *const () as usize;
            if visited.contains(&node_ptr) {
                continue;
            }

            visited.insert(node_ptr);
            topo_order_vec.push(node.clone());

            // Store node info
            node_info.insert(node_ptr, (node.clone(), tensor_id));

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

        // Cache the topological order for future use
        if let Some(meta) = &root_meta {
            if let Ok(mut lock) = meta.lock() {
                lock.topo_order_cache = Some(topo_order_vec.clone());
            }
        }

        topo_order_vec
    };

    // Process nodes in topological order (reverse for backward pass)
    let mut visited: HashSet<usize> = HashSet::new();

    for node in topo_order.iter().rev() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        if visited.contains(&node_ptr) {
            continue;
        }

        // Get the tensor_id for this node from node_info
        let tensor_id = if let Some((_, id)) = node_info.get(&node_ptr) {
            *id
        } else {
            continue;
        };

        // Get the gradient for this node's output
        let grad_output_for_node = grads.get(&tensor_id).cloned();

        // Apply backward to get gradients for inputs
        let grad_inputs = node.apply(&[grad_output_for_node]);

        // Get the input tensors for this node
        let input_tensors = node.inputs();

        // Process each input gradient
        // PERF-10: Use into_iter() to consume grad_inputs and avoid cloning
        for (input_tensor, grad_input_opt) in input_tensors.iter().zip(grad_inputs.into_iter()) {
            if let Some(grad_input) = grad_input_opt {
                if input_tensor.is_leaf() {
                    // Accumulate gradient for leaf tensor
                    if let Some(meta) = &input_tensor.inner.autograd_meta {
                        if let Ok(mut lock) = meta.lock() {
                            if let Some(existing_grad) = &mut lock.grad {
                                // In-place addition for gradient accumulation
                                existing_grad.add_(&grad_input);
                            } else {
                                // PERF-10: Take ownership instead of cloning
                                lock.grad = Some(grad_input);
                            }
                        }
                    }
                } else {
                    // For non-leaf tensors, accumulate in grads map
                    let input_id = input_tensor.id();
                    if let Some(existing_grad) = grads.get_mut(&input_id) {
                        existing_grad.add_(&grad_input);
                    } else {
                        // PERF-10: Take ownership instead of cloning
                        grads.insert(input_id, grad_input);
                    }
                }
            }
        }

        visited.insert(node_ptr);
    }
}
