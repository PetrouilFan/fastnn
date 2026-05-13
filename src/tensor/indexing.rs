use crate::autograd::{self, Edge};
use crate::ir::node::DimExpr;
use smallvec::SmallVec;
use std::sync::Arc;

use super::{Tensor, TensorImpl};

impl TensorImpl {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        if dim >= self.ndim() {
            panic!(
                "slice: dimension {} is out of range for {}-dimensional tensor",
                dim,
                self.ndim()
            );
        }

        let size = self.sizes[dim];
        let start = if start < 0 { size + start } else { start };
        let end = if end < 0 { size + end } else { end };
        let start = start.max(0) as usize;
        let end = (end.min(size)) as usize;

        if start > end {
            panic!(
                "slice: invalid range [{}, {}) for dimension {} of size {}",
                start, end, dim, size
            );
        }

        let mut sizes = self.sizes.clone();
        let numel = ((end as i64 - start as i64 + step - 1) / step) as usize;
        sizes[dim] = numel as i64;

        let storage_offset = self.storage_offset + (start as i64) * self.strides[dim];

        let mut strides = self.strides.clone();
        strides[dim] *= step;

        let mut t = self.new_view_from(sizes, strides, storage_offset);
        t.autograd_meta = None;
        t.into()
    }
}

impl Tensor {
    pub fn slice(&self, dim: usize, start: i64, end: i64, step: i64) -> Tensor {
        let output = self.inner.slice(dim, start, end, step);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SliceBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn maximum(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.maximum(&ins[0], &ins[1])])
            .expect("Tensor::maximum: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::MaximumBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn gt_scalar(&self, threshold: f32) -> Tensor {
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.gt_scalar(&ins[0], &ins[1])])
            .expect("Tensor::gt_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn sign(&self) -> Tensor {
        Tensor::exec_aot(&[self], |g, ins| vec![g.sign(&ins[0])])
            .expect("Tensor::sign: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn minimum(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.minimum(&ins[0], &ins[1])])
            .expect("Tensor::minimum: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::MinimumBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn ge_tensor(&self, other: &Tensor) -> Tensor {
        // ge(a,b) = not(lt(a,b))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let lt = g.lt_scalar(&ins[0], &ins[1]);
            vec![g.logical_not(&lt)]
        }).expect("Tensor::ge_tensor: AOT execution failed")
        .into_iter().next().unwrap()
    }

    pub fn le_tensor(&self, other: &Tensor) -> Tensor {
        // le(a,b) = not(gt(a,b))
        Tensor::exec_aot(&[self, other], |g, ins| {
            let gt = g.gt_scalar(&ins[0], &ins[1]);
            vec![g.logical_not(&gt)]
        }).expect("Tensor::le_tensor: AOT execution failed")
        .into_iter().next().unwrap()
    }

    pub fn lt_scalar(&self, threshold: f32) -> Tensor {
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.lt_scalar(&ins[0], &ins[1])])
            .expect("Tensor::lt_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn eq_scalar(&self, threshold: f32) -> Tensor {
        let scalar = Tensor::from_scalar(threshold);
        Tensor::exec_aot(&[self, &scalar], |g, ins| vec![g.eq_scalar(&ins[0], &ins[1])])
            .expect("Tensor::eq_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let s = Tensor::from_scalar(scalar);
        let output = Tensor::exec_aot(&[self, &s], |g, ins| vec![g.add_scalar(&ins[0], &ins[1])])
            .expect("Tensor::add_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::AddScalarBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let s = Tensor::from_scalar(scalar);
        let output = Tensor::exec_aot(&[self, &s], |g, ins| vec![g.div_scalar(&ins[0], &ins[1])])
            .expect("Tensor::div_scalar: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::DivScalarBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn logical_not(&self) -> Tensor {
        Tensor::exec_aot(&[self], |g, ins| vec![g.logical_not(&ins[0])])
            .expect("Tensor::logical_not: AOT execution failed")
            .into_iter()
            .next()
            .unwrap()
    }

    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        let result = Tensor::exec_aot(&[self], |g, ins| {
            let axis = dim.map(|d| DimExpr::Known(d as u64));
            let out = g.argmax(&ins[0], axis);
            vec![out]
        })
        .expect("Tensor::argmax: AOT execution failed");
        result.into_iter().next().unwrap()
    }

    pub fn cat(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("Tensor::cat: empty input");
        }
        let ndim = tensors[0].ndim();
        let dim = if dim < 0 { ndim as i32 + dim } else { dim } as usize;
        if dim >= ndim {
            panic!("Tensor::cat: dimension out of range");
        }
        for t in tensors {
            if t.ndim() != ndim {
                panic!("Tensor::cat: tensors must have same number of dimensions");
            }
            for d in 0..ndim {
                if d != dim && tensors[0].inner.sizes[d] != t.inner.sizes[d] {
                    panic!("Tensor::cat: tensor sizes mismatch at dimension {}", d);
                }
            }
        }

        let inputs: Vec<&Tensor> = tensors.iter().collect();
        let output = Tensor::exec_aot(&inputs, |g, ins| {
            let refs: Vec<_> = ins.iter().collect();
            vec![g.concat(&refs, dim)]
        })
        .expect("Tensor::cat: AOT execution failed")
        .into_iter()
        .next()
        .unwrap();

        if autograd::is_grad_enabled() && tensors.iter().any(|t| t.requires_grad()) {
            let _split_sizes: Vec<usize> = tensors
                .iter()
                .map(|t| t.inner.sizes[dim] as usize)
                .collect();
            let mut _edges = Vec::new();
            for (i, t) in tensors.iter().enumerate() {
                if let Some(node) = t.grad_fn() {
                    _edges.push(Edge(node, i));
                }
            }
            let backward = Arc::new(autograd::CatBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn stack(tensors: &[Tensor], dim: i32) -> Tensor {
        if tensors.is_empty() {
            panic!("stack: need at least one tensor");
        }

        let first_shape = tensors[0].shape();
        let ndim = first_shape.len();

        let dim = if dim < 0 {
            (ndim as i32 + dim + 1) as usize
        } else {
            dim as usize
        };
        if dim > ndim {
            panic!("stack: dimension out of range");
        }

        for t in tensors {
            if t.shape() != first_shape {
                panic!("stack: tensors must have same shape");
            }
        }

        let unsqueezed: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(dim)).collect();

        Tensor::cat(&unsqueezed, dim as i32)
    }

    pub fn repeat(&self, repeats: &[i64]) -> Tensor {
        if repeats.len() < self.ndim() {
            panic!("repeat: number of repeats must be >= number of dimensions");
        }
        let reps: Vec<usize> = repeats.iter().map(|&r| r as usize).collect();
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.repeat(&ins[0], &reps)])
            .expect("Tensor::repeat: AOT failed")
            .into_iter()
            .next()
            .unwrap();

        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::RepeatBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn where_tensor(&self, condition: &Tensor, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[condition, self, other], |g, ins| {
            vec![g.where_tensor(&ins[0], &ins[1], &ins[2])]
        })
        .expect("Tensor::where_tensor: AOT failed")
        .into_iter()
        .next()
        .unwrap();

        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = autograd::make_edges(self, other);
            let backward = Arc::new(autograd::WhereBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn gather(&self, axis: i64, indices: &Tensor) -> Tensor {
        let norm_axis = if axis < 0 {
            (self.ndim() as i64 + axis) as usize
        } else {
            axis as usize
        };
        Tensor::exec_aot(&[self, indices], |g, ins| {
            vec![g.gather(&ins[0], &ins[1], norm_axis)]
        })
        .expect("Tensor::gather: AOT failed")
        .into_iter()
        .next()
        .unwrap()
    }

    pub fn nonzero(&self) -> Vec<Vec<i64>> {
        let data = self.as_f32_slice();
        let shape = self.shape_ref();
        let mut result = Vec::new();

        if shape.is_empty() {
            return result;
        }

        let total = data.len();
        for i in 0..total {
            if data[i] != 0.0 {
                // Convert flat index to multi-dimensional index
                let mut idx = i;
                let mut coords = Vec::with_capacity(shape.len());
                for &dim in shape.iter().rev() {
                    coords.push((idx % dim as usize) as i64);
                    idx /= dim as usize;
                }
                coords.reverse();
                result.push(coords);
            }
        }
        result
    }
}
