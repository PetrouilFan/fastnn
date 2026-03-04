use crate::autograd::{AutogradMeta, Edge, Node};
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

    let mut add_node_to_queue = |node: Arc<dyn Node>| {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if !visited.contains(&node_ptr) {
            visited.insert(node_ptr);
            queue.push_back(node);
        }
    };

    if let Some(grad_fn) = root.grad_fn() {
        add_node_to_queue(grad_fn);
    }

    let root_ptr = Arc::as_ptr(&root.inner) as usize;
    grads.insert(root_ptr, grad_output);

    while let Some(node) = queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        let mut grad_inputs = node.apply(&[grads.get(&node_ptr).cloned()]);

        for (i, edge) in node.next_edges().iter().enumerate() {
            let Edge(ref next_node, _) = edge;
            let next_ptr = (&**next_node) as *const _ as *const () as usize;

            if let Some(grad) = grad_inputs.get(i) {
                if let Some(existing) = grads.get(&next_ptr) {
                    let new_grad = existing.clone() + grad.clone().unwrap();
                    grads.insert(next_ptr, new_grad);
                } else if let Some(g) = grad {
                    grads.insert(next_ptr, g.clone());
                }
            }

            add_node_to_queue(next_node.clone());
        }
    }

    fn accumulate_grad(tensor: &Tensor, grad: Tensor) {
        let ptr = Arc::as_ptr(&tensor.inner) as usize;

        let mut meta = tensor.inner.autograd_meta.clone();
        if let Some(m) = &mut meta {
            if let Some(existing) = &m.grad {
                m.grad = Some(existing.clone() + grad);
            } else {
                m.grad = Some(grad);
            }
        }
    }

    fn find_leaves_and_accumulate(tensor: &Tensor, grads: &HashMap<usize, Tensor>) {
        if tensor.is_leaf() && tensor.requires_grad() {
            let ptr = Arc::as_ptr(&tensor.inner) as usize;
            if let Some(grad) = grads.get(&ptr) {
                accumulate_grad(tensor, grad.clone());
            }
        }
    }

    let mut tensors_to_process: Vec<Tensor> = vec![root.clone()];
    let mut seen: HashSet<usize> = HashSet::new();

    while let Some(t) = tensors_to_process.pop() {
        let ptr = Arc::as_ptr(&t.inner) as usize;
        if seen.contains(&ptr) {
            continue;
        }
        seen.insert(ptr);

        if t.is_leaf() && t.requires_grad() {
            if let Some(grad) = grads.get(&ptr) {
                accumulate_grad(&t, grad.clone());
            }
        } else if let Some(grad_fn) = t.grad_fn() {
            let node_ptr = (&*grad_fn) as *const _ as *const () as usize;
            if let Some(grad) = grads.get(&node_ptr) {
                for edge in grad_fn.next_edges() {
                    let Edge(ref node, idx) = edge;
                    // This is simplified - real implementation would track input tensors
                }
            }
        }
    }
}

pub fn grad(tensor: &Tensor) -> Option<Tensor> {
    tensor.grad()
}
