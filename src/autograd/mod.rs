#![allow(dead_code)]
mod engine;
pub use engine::backward;

use crate::dispatcher::dispatch;
use crate::storage::Storage;
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
pub struct NoGradGuard;

impl NoGradGuard {
    /// Create a new NoGradGuard that disables gradient computation.
    /// The previous state is restored when the guard is dropped.
    pub fn new() -> Self {
        no_grad_enter();
        NoGradGuard
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        no_grad_exit();
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
    /// If false, zero-fill the existing grad tensor's data in-place.
    pub fn zero_grad(&mut self, set_to_none: bool) {
        if set_to_none {
            self.grad = None;
        } else if let Some(ref mut grad_tensor) = self.grad {
            let inner = Arc::make_mut(&mut grad_tensor.inner);
            let numel = inner.numel() as usize;
            let elem_size = inner.dtype.size();
            let offset = inner.storage_offset as usize;
            let storage = Arc::make_mut(&mut inner.storage);
            match storage {
                Storage::Cpu(cpu) => {
                    let data = Arc::make_mut(&mut cpu.data);
                    let start = offset * elem_size;
                    data[start..start + numel * elem_size].fill(0);
                }
                Storage::Wgpu(_) => {
                    self.grad = None;
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Edge(pub Arc<dyn Node>, pub usize);

pub fn make_edge(tensor: &Tensor) -> Vec<Edge> {
    tensor
        .grad_fn()
        .map(|node| vec![Edge(node, 0)])
        .unwrap_or_default()
}

pub fn make_edges(tensor_a: &Tensor, tensor_b: &Tensor) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(2);
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
    /// `output_tensor_id` is the ID of the output tensor that triggered backward.
    fn apply(
        &self,
        grad_outputs: Vec<Option<Tensor>>,
        output_tensor_id: usize,
    ) -> Vec<Option<Tensor>>;
    fn next_edges(&self) -> &[Edge];
    fn num_inputs(&self) -> usize;
    fn name(&self) -> &str;
    fn inputs(&self) -> &[Tensor];
    fn id(&self) -> usize {
        let ptr = self as *const _ as *const ();
        ptr as usize
    }
}

/// Helper to sum a gradient tensor to match a target shape (handle broadcasting).
/// This efficiently computes which dimensions need to be summed and does it in one pass.
pub fn sum_to_shape(mut grad: Tensor, target_shape: &[i64]) -> Tensor {
    let grad_shape = grad.shape_ref();
    if grad_shape == target_shape {
        return grad;
    }

    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();
    let diff = grad_ndim as i32 - target_ndim as i32;

    // Collect dims to sum
    let mut dims_to_sum: Vec<i32> = Vec::with_capacity(grad_ndim);
    for i in 0..grad_ndim {
        let target_dim = if i as i32 >= diff {
            target_shape[(i as i32 - diff) as usize]
        } else {
            1
        };
        if target_dim != grad_shape[i] {
            dims_to_sum.push(i as i32);
        }
    }

    // Sum all dims at once from outermost to innermost
    for &dim in dims_to_sum.iter().rev() {
        grad = grad.sum(dim, false);
    }

    grad
}

/// Helper to extract the first gradient from grad_outputs.
/// Returns None if the gradient is not present.
/// This is a common pattern used in many backward implementations.
pub fn extract_first_grad(grad_outputs: Vec<Option<Tensor>>) -> Option<Tensor> {
    grad_outputs.into_iter().next().flatten()
}

/// Helper to ensure a tensor is on CPU, returning a CPU tensor.
/// If the tensor is already on CPU, returns a clone. Otherwise, converts to CPU.
///
/// **Important**: This also materializes broadcast views (e.g., from `expand()`) into
/// contiguous storage. Without this, a gradient created via `expand()` has `numel=N` but
/// shares the scalar's underlying storage (`storage_len=1`), causing `as_f32_slice()` to
/// panic with `offset + numel exceeds storage bounds`.
pub fn ensure_cpu(tensor: &Tensor) -> Tensor {
    let t = if tensor.inner.is_cpu() {
        tensor.clone()
    } else {
        tensor.to_cpu()
    };
    // Materialize broadcast views (stride=0) so that as_f32_slice() can return
    // a valid flat slice covering all logical elements.
    if !t.is_contiguous() {
        t.contiguous()
    } else {
        t
    }
}

include!("elementwise.rs");
include!("matmul.rs");
include!("reductions.rs");
include!("conv.rs");
include!("losses.rs");
include!("views.rs");
include!("pooling.rs");
include!("norm_backward.rs");
