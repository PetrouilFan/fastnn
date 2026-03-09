use crate::autograd::Node;
use crate::tensor::{Tensor, TensorImpl};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub fn backward(root: &Tensor, grad_output: Option<Tensor>) {
    if !root.requires_grad() {
        return;
    }

    // grad_output for root - default is 1.0 if not specified
    let grad_output = grad_output.unwrap_or_else(|| Tensor::from_scalar(1.0));

    // Map from tensor data pointer to gradient
    let mut grads: HashMap<usize, Tensor> = HashMap::new();
    
    // Set gradient for root
    let root_ptr = root.data_ptr() as usize;
    grads.insert(root_ptr, grad_output);

    // Queue of (node, tensor_ptr) - node produces tensor with data_ptr
    let mut queue: VecDeque<(Arc<dyn Node>, usize)> = VecDeque::new();
    let mut visited: HashSet<usize> = HashSet::new();

    // Start with root's grad_fn
    if let Some(grad_fn) = root.grad_fn() {
        queue.push_back((grad_fn, root_ptr));
    }

    while let Some((node, tensor_ptr)) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if visited.contains(&node_ptr) {
            continue;
        }
        visited.insert(node_ptr);

        // Get gradient for this tensor's output
        let grad_output_for_node = grads.get(&tensor_ptr).cloned();
        
        // Compute gradients for inputs
        let grad_inputs = node.apply(&[grad_output_for_node]);

        let input_tensors = node.inputs();
        for (i, input_tensor) in input_tensors.iter().enumerate() {
            if let Some(grad) = grad_inputs.get(i).and_then(|g| g.as_ref()) {
                let input_ptr = input_tensor.data_ptr() as usize;
                
                // Store gradient in the input tensor's autograd_meta if it's a leaf
                if input_tensor.is_leaf() {
                    // Get the TensorImpl pointer from the Arc
                    let tensor_impl_ptr = Arc::as_ptr(&input_tensor.inner);
                    unsafe {
                        let tensor_impl = &mut *(tensor_impl_ptr as *mut TensorImpl);
                        if let Some(meta) = &mut tensor_impl.autograd_meta {
                            meta.grad = Some(grad.clone());
                        }
                    }
                } else {
                    // If not a leaf, add its grad_fn to the queue
                    if let Some(input_grad_fn) = input_tensor.grad_fn() {
                        queue.push_back((input_grad_fn, input_ptr));
                    }
                }
                
                // Also store in grads map for subsequent nodes
                let new_grad = if let Some(existing) = grads.get(&input_ptr) {
                    let existing_slice = existing.as_f32_slice();
                    let grad_slice = grad.as_f32_slice();
                    let mut result: Vec<f32> = existing_slice.to_vec();
                    for (j, &v) in grad_slice.iter().enumerate() {
                        result[j] += v;
                    }
                    Tensor::from_vec(result, existing.shape())
                } else {
                    grad.clone()
                };
                grads.insert(input_ptr, new_grad);
            }
        }
    }
}
