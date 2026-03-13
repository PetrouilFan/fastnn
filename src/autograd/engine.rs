use crate::autograd::{Edge, Node};
use crate::tensor::{Tensor, TensorImpl};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    if !root.requires_grad() {
        return;
    }

    let grad_output = grad_output.unwrap_or_else(|| Tensor::from_scalar(1.0));

    let mut grads: HashMap<usize, Tensor> = HashMap::new();

    let root_id = root.id();
    grads.insert(root_id, grad_output);

    let mut queue: VecDeque<(Arc<dyn Node>, usize)> = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();

    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_id));
    }

    while let Some((node, tensor_id)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if visited.contains(&node_ptr) {
            continue;
        }
        visited.insert(node_ptr);

        let grad_output_for_node = grads.get(&tensor_id).cloned();

        let grad_inputs = node.apply(&[grad_output_for_node]);

        let next_edges = node.next_edges();

        for (i, edge) in next_edges.iter().enumerate() {
            let Edge(next_node, _input_nr) = edge;

            if let Some(grad) = grad_inputs.get(i).and_then(|g| g.as_ref()) {
                let next_node_id = next_node.id();

                let new_grad = if let Some(existing) = grads.get(&next_node_id) {
                    let existing_data = existing.to_numpy();
                    let grad_data = grad.to_numpy();
                    let mut result = existing_data;
                    for (j, &v) in grad_data.iter().enumerate() {
                        result[j] += v;
                    }
                    Tensor::from_vec(result, existing.shape())
                } else {
                    grad.clone()
                };
                grads.insert(next_node_id, new_grad);

                queue.push_back((next_node.clone(), next_node_id));
            }
        }

        let input_tensors = node.inputs();
        for (i, input_tensor) in input_tensors.iter().enumerate() {
            if let Some(grad) = grad_inputs.get(i).and_then(|g| g.as_ref()) {
                if input_tensor.is_leaf() {
                    let tensor_impl_ptr = Arc::as_ptr(&input_tensor.inner);
                    unsafe {
                        let tensor_impl = &mut *(tensor_impl_ptr as *mut TensorImpl);
                        if let Some(meta) = &mut tensor_impl.autograd_meta {
                            if let Some(existing_grad) = &meta.grad {
                                let existing_data = existing_grad.to_numpy();
                                let grad_data = grad.to_numpy();
                                let mut result = existing_data;
                                for (j, &v) in grad_data.iter().enumerate() {
                                    result[j] += v;
                                }
                                meta.grad = Some(Tensor::from_vec(result, existing_grad.shape()));
                            } else {
                                meta.grad = Some(grad.clone());
                            }
                        }
                    }
                }
            }
        }
    }
}
