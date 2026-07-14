use crate::autograd;
use crate::autograd::AutogradMeta;
use crate::error::{FastnnError, FastnnResult};
use crate::storage::{CpuStorage, Storage};
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
        stride = stride.saturating_mul(sizes[i]);
    }
    strides
}

fn resolve_reshape_sizes(
    source_numel: i64,
    mut sizes: SmallVec<[i64; 8]>,
    operation: &str,
) -> FastnnResult<SmallVec<[i64; 8]>> {
    let mut known_product = 1_i64;
    let mut inferred = None;

    for (index, &size) in sizes.iter().enumerate() {
        match size {
            -1 if inferred.is_some() => {
                return Err(FastnnError::shape(format!(
                    "{operation} can only infer one dimension"
                )))
            }
            -1 => inferred = Some(index),
            value if value < 0 => {
                return Err(FastnnError::shape(format!(
                    "{operation} dimension {index} has invalid size {value}"
                )))
            }
            value => {
                known_product = known_product.checked_mul(value).ok_or_else(|| {
                    FastnnError::Overflow(format!("{operation} shape product overflow"))
                })?;
            }
        }
    }

    if let Some(index) = inferred {
        if known_product == 0 {
            return Err(FastnnError::shape(format!(
                "{operation} cannot infer a dimension when known dimensions have zero elements"
            )));
        }
        if source_numel % known_product != 0 {
            return Err(FastnnError::shape(format!(
                "{operation} cannot map {source_numel} elements to shape {sizes:?}"
            )));
        }
        sizes[index] = source_numel / known_product;
    } else if known_product != source_numel {
        return Err(FastnnError::shape(format!(
            "{operation} cannot map {source_numel} elements to shape {sizes:?} ({known_product} elements)"
        )));
    }

    Ok(sizes)
}

impl TensorImpl {
    pub fn ndim(&self) -> usize {
        self.sizes.len()
    }

    pub fn numel(&self) -> i64 {
        self.sizes.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        let cached = self
            .contiguous_cache
            .load(std::sync::atomic::Ordering::Relaxed);
        if cached != -1 {
            return cached == 1;
        }
        let mut expected_stride = 1i64;
        for (size, &stride) in self.sizes.iter().rev().zip(self.strides.iter().rev()) {
            if *size != 1 {
                if stride != expected_stride {
                    self.contiguous_cache
                        .store(0, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
                expected_stride *= *size;
            }
        }
        self.contiguous_cache
            .store(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    pub fn contiguous(&self) -> Tensor {
        self.try_contiguous().expect("Tensor::contiguous failed")
    }

    pub fn try_contiguous(&self) -> FastnnResult<Tensor> {
        if self.is_contiguous() {
            return Ok(Tensor::new(self.clone()));
        }
        let numel = usize::try_from(self.numel())
            .map_err(|_| FastnnError::Overflow("contiguous element count overflow".into()))?;
        let ndim = self.ndim();
        let elem_size = self.dtype.scalar_byte_width().ok_or_else(|| {
            FastnnError::dtype("contiguous conversion requires plain scalar storage")
        })?;
        let nbytes = numel
            .checked_mul(elem_size)
            .ok_or_else(|| FastnnError::Overflow("contiguous byte size overflow".into()))?;
        let mut data = Vec::new();
        data.try_reserve_exact(nbytes)
            .map_err(|error| FastnnError::Allocation(error.to_string()))?;
        data.resize(nbytes, 0u8);
        let (src_ptr, source_len) = match self.storage.as_ref() {
            Storage::Cpu(cpu) => (cpu.data.as_ref().as_ptr(), cpu.data.len()),
            #[cfg(feature = "gpu")]
            Storage::Wgpu(_) => {
                return Err(FastnnError::device(
                    "contiguous conversion requires CPU storage",
                ));
            }
        };
        let strides = &self.strides;
        let sizes = &self.sizes;
        let offset = self.storage_offset;
        let mut indices: SmallVec<[i64; 6]> = smallvec![0i64; ndim];
        let dst = data.as_mut_ptr();
        for i in 0..numel {
            let mut src_idx = offset;
            for d in 0..ndim {
                let contribution = indices[d].checked_mul(strides[d]).ok_or_else(|| {
                    FastnnError::Overflow("contiguous source index overflow".into())
                })?;
                src_idx = src_idx.checked_add(contribution).ok_or_else(|| {
                    FastnnError::Overflow("contiguous source index overflow".into())
                })?;
            }
            let src_idx = usize::try_from(src_idx)
                .map_err(|_| FastnnError::shape("contiguous source index is negative"))?;
            let src_byte = src_idx
                .checked_mul(elem_size)
                .ok_or_else(|| FastnnError::Overflow("contiguous source byte overflow".into()))?;
            let src_end = src_byte
                .checked_add(elem_size)
                .ok_or_else(|| FastnnError::Overflow("contiguous source range overflow".into()))?;
            if src_end > source_len {
                return Err(FastnnError::shape(format!(
                    "contiguous source range {src_byte}..{src_end} exceeds storage length {source_len}"
                )));
            }
            // SAFETY: source and destination ranges were checked above, each range is
            // exactly one scalar wide, and the newly allocated destination is disjoint.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(src_byte),
                    dst.add(i * elem_size),
                    elem_size,
                );
            }
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < sizes[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        let sizes = self.sizes.clone();
        let storage = Arc::new(Storage::Cpu(CpuStorage::from_vec(data, nbytes)));
        let mut new_tensor = Tensor::new(TensorImpl::try_new(storage, sizes, self.dtype)?);
        if let Some(meta) = &self.autograd_meta {
            let meta_lock = meta.lock();
            if meta_lock.requires_grad {
                let new_meta = AutogradMeta::new_non_leaf(true);
                Arc::make_mut(&mut new_tensor.inner).set_autograd_meta(new_meta);
            }
        }
        Ok(new_tensor)
    }

    pub fn view(&self, sizes: SmallVec<[i64; 8]>) -> TensorImpl {
        self.try_view(sizes).expect("Tensor::view failed")
    }

    pub fn try_view(&self, sizes: SmallVec<[i64; 8]>) -> FastnnResult<TensorImpl> {
        if !self.is_contiguous() {
            return Err(FastnnError::shape(
                "view requires a contiguous tensor; use reshape instead",
            ));
        }
        let new_sizes = resolve_reshape_sizes(self.numel(), sizes, "view")?;
        Ok(self.new_view_from(
            new_sizes.clone(),
            compute_strides(&new_sizes),
            self.storage_offset,
        ))
    }

    pub fn reshape(&self, sizes: SmallVec<[i64; 8]>) -> Tensor {
        self.try_reshape(sizes).expect("Tensor::reshape failed")
    }

    pub fn try_reshape(&self, sizes: SmallVec<[i64; 8]>) -> FastnnResult<Tensor> {
        let new_sizes = resolve_reshape_sizes(self.numel(), sizes, "reshape")?;
        if !self.is_contiguous() {
            let tensor = self.contiguous();
            return tensor.inner.try_reshape(new_sizes);
        }
        Ok(self.try_view(new_sizes)?.into())
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        self.try_transpose(dim0, dim1)
            .expect("Tensor::transpose failed")
    }

    pub fn try_transpose(&self, dim0: usize, dim1: usize) -> FastnnResult<Tensor> {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(FastnnError::shape(format!(
                "transpose dimensions {dim0} and {dim1} are out of range for a {ndim}-dimensional tensor"
            )));
        }

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();
        sizes.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        Ok(self
            .new_view_from(sizes, strides, self.storage_offset)
            .into())
    }

    pub fn permute(&self, dims: SmallVec<[i64; 8]>) -> Tensor {
        self.try_permute(dims).expect("Tensor::permute failed")
    }

    pub fn try_permute(&self, dims: SmallVec<[i64; 8]>) -> FastnnResult<Tensor> {
        let ndim = self.ndim();
        if dims.len() != ndim {
            return Err(FastnnError::shape(format!(
                "permute requires {ndim} dimensions, got {}",
                dims.len()
            )));
        }

        let mut seen = vec![false; ndim];
        for &dimension in &dims {
            if dimension < 0 || (dimension as usize) >= ndim || seen[dimension as usize] {
                return Err(FastnnError::shape(format!(
                    "{dims:?} is not a permutation of dimensions 0..{ndim}"
                )));
            }
            seen[dimension as usize] = true;
        }

        let mut sizes = SmallVec::new();
        let mut strides = SmallVec::new();
        for &dimension in &dims {
            sizes.push(self.sizes[dimension as usize]);
            strides.push(self.strides[dimension as usize]);
        }

        Ok(self
            .new_view_from(sizes, strides, self.storage_offset)
            .into())
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        self.try_unsqueeze(dim).expect("Tensor::unsqueeze failed")
    }

    pub fn try_unsqueeze(&self, dim: usize) -> FastnnResult<Tensor> {
        let ndim = self.ndim();
        if dim > ndim {
            return Err(FastnnError::shape(format!(
                "unsqueeze dimension {dim} is out of range for a {ndim}-dimensional tensor"
            )));
        }

        let mut sizes = self.sizes.clone();
        let mut strides = self.strides.clone();
        sizes.insert(dim, 1);
        strides.insert(dim, if dim == ndim { 1 } else { self.strides[dim] });

        let _input_tensor = Tensor::new(self.clone());

        let mut tensor = self.new_view_from(sizes, strides, self.storage_offset);

        if self.requires_grad() {
            let inputs = vec![_input_tensor.clone()];
            let meta = AutogradMeta {
                grad: None,
                grad_fn: Some(autograd::make_node_info("UnsqueezeBackward", inputs)),
                requires_grad: true,
                is_leaf: false,
            };
            tensor.set_autograd_meta(meta);
        }

        Ok(tensor.into())
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
        self.try_expand(sizes).expect("Tensor::expand failed")
    }

    pub fn try_expand(&self, sizes: SmallVec<[i64; 8]>) -> FastnnResult<Tensor> {
        if sizes.len() < self.ndim() {
            return Err(FastnnError::shape(format!(
                "expand target has {} dimensions but the tensor has {}",
                sizes.len(),
                self.ndim()
            )));
        }
        if sizes.iter().any(|&size| size < 0) {
            return Err(FastnnError::shape(format!(
                "expand target dimensions must be non-negative, got {sizes:?}"
            )));
        }

        let new_sizes = sizes.clone();
        let offset = sizes.len() - self.ndim();

        for i in 0..self.ndim() {
            let target = new_sizes[offset + i];
            let source = self.sizes[i];
            if target != source && source != 1 {
                return Err(FastnnError::shape(format!(
                    "expand cannot change dimension {i} from {source} to {target}; only size-1 dimensions are expandable"
                )));
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

        Ok(self
            .new_view_from(sizes, new_strides, self.storage_offset)
            .into())
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
        self.try_contiguous().expect("Tensor::contiguous failed")
    }

    pub fn try_contiguous(&self) -> FastnnResult<Tensor> {
        self.inner.try_contiguous()
    }

    pub fn view(&self, shape: Vec<i64>) -> Tensor {
        self.try_view(shape).expect("Tensor::view failed")
    }

    pub fn try_view(&self, shape: Vec<i64>) -> FastnnResult<Tensor> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output: Tensor = self.inner.try_view(sizes)?.into();

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("ViewBackward", inputs))
        } else {
            output
        })
    }

    pub fn reshape(&self, shape: Vec<i64>) -> Tensor {
        self.try_reshape(shape).expect("Tensor::reshape failed")
    }

    pub fn try_reshape(&self, shape: Vec<i64>) -> FastnnResult<Tensor> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output = self.inner.try_reshape(sizes)?;

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("ViewBackward", inputs))
        } else {
            output
        })
    }

    pub fn reshape_permute(&self, shape: Vec<i64>, perm: Vec<i64>) -> Tensor {
        let reshaped = self.reshape(shape);
        let output = reshaped.permute(perm);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("ViewBackward", inputs))
        } else {
            output
        }
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        self.try_transpose(dim0, dim1)
            .expect("Tensor::transpose failed")
    }

    pub fn try_transpose(&self, dim0: usize, dim1: usize) -> FastnnResult<Tensor> {
        let output = self.inner.try_transpose(dim0, dim1)?;

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(
                output,
                autograd::make_node_info("TransposeBackward", inputs),
            )
        } else {
            output
        })
    }

    pub fn permute(&self, dims: Vec<i64>) -> Tensor {
        self.try_permute(dims).expect("Tensor::permute failed")
    }

    pub fn try_permute(&self, dims: Vec<i64>) -> FastnnResult<Tensor> {
        let sizes: SmallVec<[i64; 8]> = dims.into();
        let output = self.inner.try_permute(sizes)?;

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("PermuteBackward", inputs))
        } else {
            output
        })
    }

    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let output = self.inner.squeeze(dim);

        if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("ViewBackward", inputs))
        } else {
            output
        }
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        self.try_unsqueeze(dim).expect("Tensor::unsqueeze failed")
    }

    pub fn try_unsqueeze(&self, dim: usize) -> FastnnResult<Tensor> {
        self.inner.try_unsqueeze(dim)
    }

    pub fn expand(&self, shape: Vec<i64>) -> Tensor {
        self.try_expand(shape).expect("Tensor::expand failed")
    }

    pub fn try_expand(&self, shape: Vec<i64>) -> FastnnResult<Tensor> {
        let sizes: SmallVec<[i64; 8]> = shape.into();
        let output = self.inner.try_expand(sizes)?;

        Ok(if autograd::is_grad_enabled() && self.requires_grad() {
            let inputs = vec![self.clone()];
            Self::attach_grad_fn(output, autograd::make_node_info("ExpandBackward", inputs))
        } else {
            output
        })
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
