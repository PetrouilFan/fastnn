use crate::autograd::Node;
use crate::tensor::{Tensor, TensorImpl};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    if !root.requires_grad() {
        return;
    }

    let grad_output = grad_output.unwrap_or_else(|| Tensor::from_scalar(1.0));

    let mut queue: VecDeque<Arc<dyn Node>> = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut grads: HashMap<usize, Tensor> = HashMap::new();

    if let Some(grad_fn) = root.grad_fn() {
        let node_ptr = (&*grad_fn) as *const _ as *const () as usize;
        if !visited.contains(&node_ptr) {
            visited.insert(node_ptr);
            queue.push_back(grad_fn);
        }
    }

    let root_ptr = Arc::as_ptr(&root.inner) as usize;
    grads.insert(root_ptr, grad_output);

    while let Some(node) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        let grad_output_for_node = grads.get(&node_ptr).cloned();
        let grad_inputs = node.apply(&[grad_output_for_node]);

        let input_tensors = node.inputs();
        for (i, input_tensor) in input_tensors.iter().enumerate() {
            if let Some(grad) = grad_inputs.get(i).and_then(|g| g.as_ref()) {
                let input_ptr = Arc::as_ptr(&input_tensor.inner) as usize;
                if let Some(existing) = grads.get(&input_ptr) {
                    let new_grad = existing.clone() + grad.clone();
                    grads.insert(input_ptr, new_grad);
                } else {
                    grads.insert(input_ptr, grad.clone());
                }
            }
        }

        for edge in node.next_edges() {
            let next_node = &edge.0;
            let next_ptr = (&**next_node) as *const _ as *const () as usize;
            if !visited.contains(&next_ptr) {
                visited.insert(next_ptr);
                queue.push_back(next_node.clone());
            }
        }
    }

    let root_ptr = Arc::as_ptr(&root.inner) as usize;
    if let Some(grad) = grads.get(&root_ptr) {
        if root.is_leaf() && root.requires_grad() {
            unsafe {
                let ptr = Arc::as_ptr(&root.inner) as *mut TensorImpl;
                if let Some(meta) = (*ptr).autograd_meta.as_mut() {
                    if let Some(existing) = &meta.grad {
                        meta.grad = Some(existing.clone() + grad.clone());
                    } else {
                        meta.grad = Some(grad.clone());
                    }
                }
            }
        }
    }

    if let Some(grad_fn) = root.grad_fn() {
        let input_tensors = grad_fn.inputs();
        for input_tensor in input_tensors {
            let input_ptr = Arc::as_ptr(&input_tensor.inner) as usize;
            if let Some(grad) = grads.get(&input_ptr) {
                if input_tensor.is_leaf() && input_tensor.requires_grad() {
                    unsafe {
                        let ptr = Arc::as_ptr(&input_tensor.inner) as *mut TensorImpl;
                        if let Some(meta) = (*ptr).autograd_meta.as_mut() {
                            if let Some(existing) = &meta.grad {
                                meta.grad = Some(existing.clone() + grad.clone());
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
