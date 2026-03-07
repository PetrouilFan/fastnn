use crate::autograd::Node;
use crate::tensor::Tensor;
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

    let root_ptr = root.data_ptr() as usize;
    grads.insert(root_ptr, grad_output);

    while let Some(node) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        let grad_output_for_node = grads.get(&node_ptr).cloned();
        let grad_inputs = node.apply(&[grad_output_for_node]);

        let input_tensors = node.inputs();
        for (i, input_tensor) in input_tensors.iter().enumerate() {
            if let Some(grad) = grad_inputs.get(i).and_then(|g| g.as_ref()) {
                let input_ptr = input_tensor.data_ptr() as usize;
                let new_grad = if let Some(existing) = grads.get(&input_ptr) {
                    let existing_data = existing.to_numpy();
                    let grad_data = grad.to_numpy();
                    let mut result = existing_data;
                    for (j, v) in grad_data.iter().enumerate() {
                        result[j] += v;
                    }
                    Tensor::from_vec(result, existing.shape())
                } else {
                    grad.clone()
                };
                grads.insert(input_ptr, new_grad);
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
}
