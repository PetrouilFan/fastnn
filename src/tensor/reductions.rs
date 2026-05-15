// Tensor reduction methods

use crate::autograd;
use crate::storage::{DType, Device};
use std::sync::Arc;

use super::Tensor;

impl Tensor {
    pub fn sum(&self, dim: i32, keepdim: bool) -> Tensor {
        // Fast path: contiguous CPU F32, last-dim sum, no keepdim, 2D tensors only
        if self.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && self.is_contiguous()
            && !keepdim
            && self.inner.ndim() == 2
        {
            let ndim = self.inner.ndim() as i32;
            let dim_normalized = if dim < 0 { ndim + dim } else { dim } as usize;
            if dim_normalized == ndim as usize - 1 {
                let shape = self.shape();
                let dim_size = shape[dim_normalized] as usize;
                let num_rows = shape[0] as usize;
                let output = crate::backend::cpu::reductions_fast::sum_last_dim_contiguous(
                    self, dim_size, num_rows,
                );
                if autograd::is_grad_enabled() && self.requires_grad() {
                    let mut output = output;
                    let edges = autograd::make_edge(self);
                    let inputs = vec![self.clone()];
                    let backward = autograd::SumBackward::new(edges, inputs);
                    let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                    meta.grad_fn = Some(std::sync::Arc::new(backward));
                    Arc::make_mut(&mut output.inner).autograd_meta =
                        Some(Arc::new(std::sync::Mutex::new(meta)));
                    return output;
                }
                return output;
            }
        }

        let ndim = self.inner.ndim() as i32;
        let norm_dim = if dim < 0 { ndim + dim } else { dim } as usize;
        let output = Tensor::exec_aot(&[self], |g, ins| {
            vec![g.reduce_sum(&ins[0], norm_dim, keepdim)]
        })
        .expect("Tensor::sum: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::SumBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn max(&self, dim: i32, keepdim: bool) -> Tensor {
        let ndim = self.inner.ndim() as i32;
        let norm_dim = if dim < 0 { ndim + dim } else { dim } as usize;
        let output = Tensor::exec_aot(&[self], |g, ins| {
            vec![g.reduce_max(&ins[0], norm_dim, keepdim)]
        })
        .expect("Tensor::max: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = std::sync::Arc::new(autograd::MaximumBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn mean(&self, dim: i32, keepdim: bool) -> Tensor {
        let ndim = self.inner.ndim() as i32;
        let norm_dim = if dim < 0 { ndim + dim } else { dim } as usize;
        let output = Tensor::exec_aot(&[self], |g, ins| {
            vec![g.reduce_mean(&ins[0], norm_dim, keepdim)]
        })
        .expect("Tensor::mean: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _dim_size = self.shape()[dim as usize];
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::MeanBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn cumsum(&self, dim: i64, exclusive: bool, reverse: bool) -> Tensor {
        let norm_dim = if dim < 0 {
            (self.ndim() as i64 + dim) as usize
        } else {
            dim as usize
        };
        let result = Tensor::exec_aot(&[self], |g, ins| {
            let c = g.cumsum(&ins[0], norm_dim, exclusive, reverse);
            vec![c]
        })
        .expect("Tensor::cumsum: AOT execution failed");
        let output = result.into_iter().next().unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = std::sync::Arc::new(autograd::CumSumBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }
}
