use crate::autograd;
use crate::autograd::AutogradMeta;
use crate::error::{FastnnError, FastnnResult};
use smallvec::smallvec;
use smallvec::SmallVec;
use std::sync::Arc;

use super::{Tensor, TensorImpl};

pub(crate) fn compute_strides(sizes: &[i64]) -> SmallVec<[i64; 8]> {
    let mut strides: SmallVec<[i64; 8]> = smallvec![0; sizes.len()];
    if sizes.is_empty() {
        return strides;
    }

    let mut stride = 1i64;
    for i in (0..sizes.len()).rev() {
        strides[i] = stride;
        stride *= sizes[i];
    }
    strides
}

impl TensorImpl {
    pub fn ndim(&self) -> usize {
        self.sizes.len()
    }

    pub fn numel(&self) -> i64 {
        self.sizes.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1i64;
        for (size, &stride) in self.sizes.iter().rev().zip(self.strides.iter().rev()) {
            if *size != 1 {
                if stride != expected_stride {
                    return false;
                }
                expected_stride *= *size;
            }
        }
        true
    }

    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return Tensor::new(self.clone());
        }
        let numel = self.numel() as usize;
        let ndim = self.ndim();
        let mut data = vec![0.0f32; numel];
        // SAFETY: We verified the storage is `Storage::Cpu`. The pointer is derived
        // from the backing `Vec<u8>` allocation, which is valid for the duration of
        // this function. The length `len` is within the bounds of the allocation.
        let src = unsafe {
            let crate::storage::Storage::Cpu(cpu) = &self.storage.as_ref() else {
                panic!("contiguous(): only supported for CPU tensors");
            };
            let ptr = cpu.data.as_ref().as_ptr() as *const f32;
            let len = cpu.data.len() / 4;
            std::slice::from_raw_parts(ptr, len)
        };
        let strides = &self.strides;
        let sizes = &self.sizes;
        let offset = self.storage_offset;
        let mut indices: SmallVec<[i64; 6]> = smallvec![0i64; ndim];
        for i in 0..numel {
            let mut src_idx = offset;
            for d in 0..ndim {
                src_idx += indices[d] * strides[d];
            }
            data[i] = src[src_idx as usize];
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < sizes[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        let sizes = self.sizes.clone();
        let mut new_tensor = Tensor::from_vec(data, sizes.to_vec());
        if let Some(meta) = &self.autograd_meta {
            let Ok(meta_lock) = meta.lock() else {
                return new_tensor;
            };
            if meta_lock.requires_grad {
                let new_meta = AutogradMeta::new_non_leaf(true);
                Arc::make_mut(&mut new_tensor.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(new_meta)));
            }
        }
        new_tensor
    }

    pub fn view(&self, sizes: SmallVec<[i64; 8]>) -> TensorImpl {
        if !self.is_contiguous() {
            panic!("view() requires contiguous tensor, use reshape() instead");
        }
        let mut new_sizes = sizes.clone();
        let mut product: i64 = 1;
        let mut minus_one_idx = None;

        for (i, s) in new_sizes.iter().enumerate() {
            if *s == -1 {
                if minus_one_idx.is_some() {
                    panic!("view: can only specify one unknown dimension (-1), got multiple");
                }
                minus_one_idx = Some(i);
            } else if *s < 0 {
                panic!("view: negative dimension size not allowed (got {})", s);
            } else {
                product *= s;
            }
        }

        if let Some(idx) = minus_one_idx {
            if product == 0 {
                panic!("view: cannot infer dimension with zero-sized dimensions");
            }
            if self.numel() % product != 0 {
                panic!(
                    "view: size mismatch, cannot reshape tensor of {} elements into shape {:?} ({} is not divisible by {})",
                    self.numel(),
                    new_sizes,
                    self.numel(),
                    product
                );
            }
            new_sizes[idx] = self.numel() / product;
        }

        let numel: i64 = new_sizes.iter().product();
        if numel != self.numel() {
            panic!(
                "view: size mismatch, cannot reshape tensor of {} elements into shape {:?} (target has {} elements)",
                self.numel(),
                new_sizes,
                numel
            );
        }

        self.new_view_from(
            new_sizes.clone(),
            compute_strides(&new_sizes),
            self.storage_offset,
        )
    }

    pub fn reshape(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        if !self.is_contiguous() {
            let t = self.contiguous();
            return t.inner.reshape(sizes);
        }
        let mut new_sizes = sizes.clone();
        let mut product: i64 = 1;
        let mut minus_one_idx = None;

        for (i, s) in new_sizes.iter().enumerate() {
            if *s == -1 {
                if minus_one_idx.is_some() {
                    panic!("reshape: can only specify one unknown dimension (-1), got multiple");
                }
                minus_one_idx = Some(i);
            } else if *s < 0 {
                panic!("reshape: negative dimension size not allowed (got {})", s);
            } else {
                product *= s;
            }
        }

        if let Some(idx) = minus_one_idx {
            if product == 0 {
                panic!("reshape: cannot infer dimension with zero-sized dimensions");
            }
            if self.numel() % product != 0 {
                panic!(
                    "reshape: size mismatch, cannot reshape tensor of {} elements into shape {:?} ({} is not divisible by {})",
                    self.numel(),
                    new_sizes,
                    self.numel(),
                    product
                );
            }
            let known: i64 = self.numel() / product;
            new_sizes[idx] = known;
        } else {
            let numel: i64 = sizes.iter().product();
            if numel != self.numel() {
                panic!(
                    "reshape: size mismatch, cannot reshape tensor of {} elements into shape {:?} (target has {} elements)",
                    self.numel(),
                    sizes,
                    numel
                );
            }
        }

        self.view(new_sizes).into()
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            panic!(
                "transpose: dimension {} or {} is out of range for {}-dimensional tensor",
                dim0, dim1, ndim
            );
        }

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();

        sizes.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        self.new_view_from(sizes, strides, self.storage_offset)
            .into()
    }

    pub fn permute(&self, dims: SmallVec<[i64; 8]>) -> Tensor {
        let ndim = self.ndim();
        if dims.len() != ndim {
            panic!(
                "permute: number of dimensions mismatch (tensor has {} dims, got {} dims)",
                ndim,
                dims.len()
            );
        }

        let mut seen = vec![false; ndim];
        for &d in &dims {
            if d < 0 || (d as usize) >= ndim || seen[d as usize] {
                panic!(
                    "permute: invalid permutation {:?} for {}-dimensional tensor",
                    dims, ndim
                );
            }
            seen[d as usize] = true;
        }

        let mut sizes = SmallVec::new();
        let mut strides = SmallVec::new();

        for &d in &dims {
            sizes.push(self.sizes[d as usize]);
            strides.push(self.strides[d as usize]);
        }

        self.new_view_from(sizes, strides, self.storage_offset)
            .into()
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let ndim = self.ndim();
        let dim = if dim > ndim { ndim } else { dim };

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();
        sizes.insert(dim, 1);
        strides.insert(dim, if dim == ndim { 1 } else { self.strides[dim] });

        let _input_tensor = Tensor::new(self.clone());

        let mut tensor = self.new_view_from(sizes, strides, self.storage_offset);

        if self.requires_grad() {
            let edges = autograd::make_edge(&_input_tensor);
            let inputs = vec![_input_tensor.clone()];
            let backward = Arc::new(autograd::UnsqueezeBackward::new(edges, inputs));
            let meta = AutogradMeta {
                grad: None,
                grad_fn: Some(backward.clone()),
                requires_grad: true,
                is_leaf: false,
            };
            tensor.autograd_meta = Some(Arc::new(std::sync::Mutex::new(meta)));
        }

        tensor.into()
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        match dim {
            Some(d) => {
                if d >= self.ndim() || self.sizes[d] != 1 {
                    return self.clone().into();
                }
                let mut sizes = self.sizes.clone();
                let mut strides = self.strides.clone();
                sizes.remove(d);
                strides.remove(d);

                self.new_view_from(sizes, strides, self.storage_offset)
                    .into()
            }
            None => {
                let mut sizes = SmallVec::new();
                let mut strides = SmallVec::new();
                for (s, &st) in self.sizes.iter().zip(self.strides.iter()) {
                    if *s != 1 {
                        sizes.push(*s);
                        strides.push(st);
                    }
                }

                if sizes.is_empty() {
                    sizes.push(1);
                    strides.push(1);
                }

                self.new_view_from(sizes, strides, self.storage_offset)
                    .into()
            }
        }
    }

    pub fn expand(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        if sizes.len() < self.ndim() {
            panic!(
                "expand: target shape has {} dimensions but tensor has {} dimensions",
                sizes.len(),
                self.ndim()
            );
        }

        let new_sizes = sizes.clone();
        let offset = sizes.len() - self.ndim();

        for i in 0..self.ndim() {
            let target = new_sizes[offset + i];
            let source = self.sizes[i];
            if target != source && source != 1 {
                panic!(
                    "expand: cannot expand dimension {} from {} to {} (only size-1 dimensions can be expanded)",
                    i, source, target
                );
            }
        }

        let mut new_strides: SmallVec<[i64; 8]> = smallvec![0; sizes.len()];
        for i in 0..self.ndim() {
            new_strides[offset + i] = if self.sizes[i] == 1 {
                0
            } else {
                self.strides[i]
            };
        }

        self.new_view_from(sizes, new_strides, self.storage_offset)
            .into()
    }
}

impl Tensor {
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn shape(&self) -> Vec<i64> {
        self.inner.sizes.to_vec()
    }

    pub fn shape_ref(&self) -> &[i64] {
        &self.inner.sizes
    }

    pub fn strides(&self) -> Vec<i64> {
        self.inner.strides.to_vec()
    }

    pub fn numel(&self) -> i64 {
        self.inner.numel()
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    pub fn contiguous(&self) -> Tensor {
        self.inner.contiguous()
    }

    pub fn view(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output: Tensor = self.inner.view(sizes).into();

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::ViewBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn reshape(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output = self.inner.reshape(sizes);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::ViewBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn reshape_permute(&self, shape: Vec<i64>, perm: Vec<i64>) -> Tensor {
        let reshaped = self.reshape(shape);
        let output = reshaped.permute(perm);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::ViewBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let output = self.inner.transpose(dim0, dim1);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::TransposeBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn permute(&self, dims: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = dims.clone().into();
        let output = self.inner.permute(sizes);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::PermuteBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let output = self.inner.squeeze(dim);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::ViewBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        self.inner.unsqueeze(dim)
    }

    pub fn expand(&self, shape: Vec<i64>) -> Tensor {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output = self.inner.expand(sizes);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let edges = autograd::make_edge(self);
            let inputs = vec![self.clone()];
            let backward = Arc::new(autograd::ExpandBackward::new(edges, inputs));
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn flip(&self, dims: &[usize]) -> Tensor {
        let result = Tensor::exec_aot(&[self], |g, ins| vec![g.flip(&ins[0], dims)])
            .expect("Tensor::flip: AOT execution failed");
        result.into_iter().next().unwrap()
    }

    pub(crate) fn broadcast_shapes(a: &[i64], b: &[i64]) -> FastnnResult<Vec<i64>> {
        let ndim = a.len().max(b.len());
        let mut out = vec![1i64; ndim];
        for i in 0..ndim {
            let a_dim = if i < ndim - a.len() {
                1
            } else {
                a[i - (ndim - a.len())]
            };
            let b_dim = if i < ndim - b.len() {
                1
            } else {
                b[i - (ndim - b.len())]
            };
            if a_dim == b_dim {
                out[i] = a_dim;
            } else if a_dim == 1 {
                out[i] = b_dim;
            } else if b_dim == 1 {
                out[i] = a_dim;
            } else {
                return Err(FastnnError::Shape(format!(
                    "shapes {:?} and {:?} are not broadcast-compatible",
                    a, b
                )));
            }
        }
        Ok(out)
    }
}
