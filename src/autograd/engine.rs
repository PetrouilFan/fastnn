use crate::autograd::{is_grad_enabled, no_grad_enter, no_grad_exit, Node};
use crate::tensor::Tensor;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::Arc;

struct BackwardWorkspace {
    grads: FxHashMap<usize, Tensor>,
    node_deps: FxHashMap<usize, usize>,
    queue: VecDeque<(Arc<dyn Node>, usize)>,
    visited: FxHashSet<usize>,
}

impl BackwardWorkspace {
    fn new() -> Self {
        Self {
            grads: FxHashMap::default(),
            node_deps: FxHashMap::default(),
            queue: VecDeque::new(),
            visited: FxHashSet::default(),
        }
    }

    fn clear(&mut self) {
        self.grads.clear();
        self.node_deps.clear();
        self.queue.clear();
        self.visited.clear();
    }
}

thread_local! {
    static WORKSPACE: RefCell<BackwardWorkspace> = RefCell::new(BackwardWorkspace::new());
}

/// Check for NaN or Inf in a tensor's data (debug builds only)
#[cfg(debug_assertions)]
fn check_gradient_validity(tensor: &Tensor, context: &str) {
    if let Some(data) = tensor.inner.cpu_data() {

// SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
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

    let prev_grad_enabled = is_grad_enabled();
    if prev_grad_enabled {
        no_grad_enter();
    }

    let grad_output =
        grad_output.unwrap_or_else(|| Tensor::full(vec![], 1.0, root.dtype(), root.device()));

    let mut ws = WORKSPACE.with(|w| w.replace(BackwardWorkspace::new()));
    ws.clear();

    let root_id = root.id();
    ws.grads.insert(root_id, grad_output);

    if let Some(grad_fn) = root.grad_fn() {
        ws.queue.push_back((grad_fn, root_id));
    }

    // First pass: discover all nodes and build dependency counts
    while let Some((node, _output_tensor_id)) = ws.queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;
        if ws.visited.contains(&node_ptr) {
            continue;
        }
        ws.visited.insert(node_ptr);
        ws.node_deps.entry(node_ptr).or_insert(0);

        let input_tensors = node.inputs();
        for input_tensor in input_tensors {
            if let Some(input_grad_fn) = input_tensor.grad_fn() {
                let input_node_ptr = (&*input_grad_fn) as *const _ as *const () as usize;
                *ws.node_deps.entry(input_node_ptr).or_insert(0) += 1;

                if !ws.visited.contains(&input_node_ptr) {
                    ws.queue.push_back((input_grad_fn, input_tensor.id()));
                }
            }
        }
    }

    // Second pass: process nodes in topological order
    ws.queue.clear();
    ws.visited.clear();

    if let Some(grad_fn) = root.grad_fn() {
        ws.queue.push_back((grad_fn, root_id));
    }

    while let Some((node, tensor_id)) = ws.queue.pop_front() {
        let node_ptr = (&*node) as *const _ as *const () as usize;

        if ws.visited.contains(&node_ptr) {
            continue;
        }

        let deps = ws.node_deps.get(&node_ptr).copied().unwrap_or(0);
        if deps > 0 {
            ws.queue.push_back((node, tensor_id));
            continue;
        }

        ws.visited.insert(node_ptr);

        let grad_output_for_node = ws.grads.remove(&tensor_id);

        if let Some(ref grad) = grad_output_for_node {
            check_gradient_validity(grad, &format!("backward pass for node {}", node.name()));
        }

        let grad_inputs = node.apply(vec![grad_output_for_node], tensor_id);
        let input_tensors = node.inputs();

        for (input_tensor, grad_input_opt) in input_tensors.iter().zip(grad_inputs) {
            if let Some(grad_input) = grad_input_opt {
                if input_tensor.is_leaf() {
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
                    let input_id = input_tensor.id();
                    match ws.grads.get_mut(&input_id) {
                        Some(existing_grad) => {
                            existing_grad.add_(&grad_input);
                        }
                        None => {
                            ws.grads.insert(input_id, grad_input);
                        }
                    }

                    if let Some(input_grad_fn) = input_tensor.grad_fn() {
                        let input_node_ptr = (&*input_grad_fn) as *const _ as *const () as usize;
                        if !ws.visited.contains(&input_node_ptr) {
                            if let Some(deps) = ws.node_deps.get_mut(&input_node_ptr) {
                                if *deps > 0 {
                                    *deps -= 1;
                                }
                            }
                            ws.queue.push_back((input_grad_fn, input_id));
                        }
                    }
                }
            }
        }
    }

    // Restore gradient tracking
    if prev_grad_enabled {
        no_grad_exit();
    }

    // Return workspace to thread-local storage
    WORKSPACE.with(|w| {
        w.replace(ws);
    });
}
