#![allow(dead_code)]
mod engine;
pub use engine::backward;

use crate::dispatcher::dispatch;
use crate::tensor::Tensor;
use std::cell::Cell;
use std::sync::Arc;

thread_local! {
    static NO_GRAD_TLS: Cell<usize> = const { Cell::new(0) };
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

/// RAII guard for disabling gradient computation.
/// When this guard is dropped, the previous gradient state is restored.
/// This is useful for validation/inference to save memory and computation.
///
/// # Example
/// ```ignore
/// let _guard = NoGradGuard::new();
/// // All tensor operations here won't build computation graph
/// let output = model.forward(&input);
/// ```
pub struct NoGradGuard {
    prev_state: bool,
}

impl NoGradGuard {
    /// Create a new NoGradGuard that disables gradient computation.
    /// The previous state is saved and will be restored when the guard is dropped.
    pub fn new() -> Self {
        let prev_state = is_grad_enabled();
        no_grad_enter();
        NoGradGuard { prev_state }
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        if self.prev_state {
            no_grad_exit();
        }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to run a closure with gradient computation disabled.
///
/// # Example
/// ```ignore
/// let output = no_grad(|| {
///     model.forward(&input)
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

pub struct AutogradMeta {
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub grad_fn: Option<Arc<dyn Node>>,
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

    /// Zero the gradient.
    /// If set_to_none is true, drop the grad buffer (old behavior).
    /// If false, zero-fill the existing grad tensor's data (TODO: implement).
    pub fn zero_grad(&mut self, set_to_none: bool) {
        if set_to_none {
            self.grad = None;
        } else {
            // TODO: zero-fill grad tensor's data to avoid reallocation
            // For now, just drop it (same as set_to_none=true)
            self.grad = None;
        }
    }
}

#[derive(Clone)]
pub struct Edge(pub Arc<dyn Node>, pub usize);

pub fn make_edge(tensor: &Tensor) -> Vec<Edge> {
    tensor
        .grad_fn()
        .map(|node| Edge(node, 0))
        .map(|e| vec![e])
        .unwrap_or_default()
}

pub fn make_edges(tensor_a: &Tensor, tensor_b: &Tensor) -> Vec<Edge> {
    let mut edges = Vec::new();
    if let Some(node) = tensor_a.grad_fn() {
        edges.push(Edge(node, 0));
    }
    if let Some(node) = tensor_b.grad_fn() {
        edges.push(Edge(node, 1));
    }
    edges
}

#[allow(clippy::len_zero)]
pub trait Node: Send + Sync {
    /// Apply backward pass. Takes ownership of grad_outputs to avoid cloning.
    /// Implementations should consume gradients instead of cloning them.
    fn apply(&self, grad_outputs: Vec<Option<Tensor>>) -> Vec<Option<Tensor>>;
    fn next_edges(&self) -> &[Edge];
    fn num_inputs(&self) -> usize;
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Tensor];
    fn id(&self) -> usize {
        let ptr = self as *const _ as *const ();
        ptr as usize
    }
}

include!("elementwise.rs");
include!("matmul.rs");
include!("reductions.rs");
include!("conv.rs");
include!("losses.rs");
include!("views.rs");
