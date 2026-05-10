// Tensor reduction methods

use crate::autograd;
use crate::dispatcher::{device_to_dispatch_key, dispatch};
use crate::storage::{DType, Device};
use std::sync::Arc;

use super::factories::dim_scalar;
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
                let output = crate::kernels::cpu::sum_last_dim_contiguous(self, dim_size, num_rows);
                if autograd::is_grad_enabled() && self.requires_grad() {
                    let mut output = output;
                    let edges = autograd::make_edge(self);
                    let backward =
                        autograd::SumBackward::new(self.clone(), dim_normalized, keepdim, edges);
                    let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                    meta.grad_fn = Some(std::sync::Arc::new(backward));
                    Arc::make_mut(&mut output.inner).autograd_meta =
                        Some(Arc::new(std::sync::Mutex::new(meta)));
                    return output;
                }
                return output;
            }
        }

        // Fallback: dispatch
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "sum",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        )
        .expect("Tensor::sum: dispatch failed");
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SumBackward::new(
                self.clone(),
                dim as usize,
                keepdim,
                edges,
            ));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn max(&self, dim: i32, keepdim: bool) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "max",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        )
        .expect("Tensor::max: dispatch failed");
        result[0].clone()
    }

    pub fn mean(&self, dim: i32, keepdim: bool) -> Tensor {
        let dispatch_key = device_to_dispatch_key(self.device());
        let result = dispatch(
            "mean",
            dispatch_key,
            &[
                self,
                &dim_scalar(dim),
                &Tensor::from_scalar(if keepdim { 1.0 } else { 0.0 }),
            ],
        )
        .expect("Tensor::mean: dispatch failed");
        let output = result[0].clone();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let dim_size = self.shape()[dim as usize];
            let edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::MeanBackward::new(
                self.clone(),
                dim as usize,
                keepdim,
                dim_size,
                edges,
            ));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn cumsum(&self, dim: i64, exclusive: bool, reverse: bool) -> Tensor {
        let data = self.as_f32_slice();
        let shape = self.shape_ref();
        let rank = shape.len() as i64;
        let axis = if dim < 0 { (dim + rank) as usize } else { dim as usize };
        let mut result_data = data.to_vec();
        
        // Calculate strides
        let mut stride = 1i64;
        for i in (axis + 1)..shape.len() {
            stride *= shape[i];
        }
        let outer = data.len() as i64 / (shape[axis] * stride);
        
        if reverse {
            for o in 0..outer {
                for s in 0..stride {
                    let _base = (o * shape[axis] + 0) * stride + s;
                    let mut running = 0.0f32;
                    for d in (0..shape[axis]).rev() {
                        let idx = (o * shape[axis] + d) * stride + s;
                        running += data[idx as usize];
                        result_data[idx as usize] = if exclusive { running - data[idx as usize] } else { running };
                    }
                }
            }
        } else {
            for o in 0..outer {
                for s in 0..stride {
                    let mut running = 0.0f32;
                    for d in 0..shape[axis] {
                        let idx = (o * shape[axis] + d) * stride + s;
                        running += data[idx as usize];
                        result_data[idx as usize] = if exclusive { running - data[idx as usize] } else { running };
                    }
                }
            }
        }
        
        Tensor::from_vec(result_data, shape.to_vec())
    }
}
