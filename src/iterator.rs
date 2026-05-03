use crate::tensor::Tensor;
use smallvec::smallvec;
use smallvec::SmallVec;

#[derive(Clone)]
#[allow(dead_code)]
pub struct TensorIteratorInput {
    pub tensor: Tensor,
    pub sizes: SmallVec<[i64; 8]>,
    pub strides: SmallVec<[i64; 8]>,
    pub storage_offset: i64,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct TensorIterator {
    pub output_shape: SmallVec<[i64; 8]>,
    pub inner_strides: SmallVec<[i64; 8]>,
    pub inputs: Vec<TensorIteratorInput>,
    pub inner_dim: usize,
    pub numel: usize,
}

impl TensorIterator {
    pub fn build_for_binary(a: &Tensor, b: &Tensor) -> Self {
        let output_shape = broadcast_shapes(&a.inner.sizes, &b.inner.sizes);
        let numel: i64 = output_shape.iter().product();

        let a_strides = compute_broadcast_strides(&a.inner.sizes, &output_shape);
        let b_strides = compute_broadcast_strides(&b.inner.sizes, &output_shape);

        let inputs = vec![
            TensorIteratorInput {
                tensor: a.clone(),
                sizes: a.inner.sizes.clone(),
                strides: a_strides,
                storage_offset: a.inner.storage_offset,
            },
            TensorIteratorInput {
                tensor: b.clone(),
                sizes: b.inner.sizes.clone(),
                strides: b_strides,
                storage_offset: b.inner.storage_offset,
            },
        ];

        let inner_dim = find_contiguous_inner_dim(&output_shape);
        let inner_strides: SmallVec<[i64; 8]> = (0..output_shape.len())
            .map(|i| if i >= inner_dim { output_shape[i] } else { 1 })
            .collect();

        TensorIterator {
            output_shape,
            inner_strides,
            inputs,
            inner_dim,
            numel: numel as usize,
        }
    }

    pub fn build_for_unary(a: &Tensor) -> Self {
        let output_shape = a.inner.sizes.clone();
        let numel: i64 = output_shape.iter().product();

        let a_strides = a.inner.strides.clone();

        let inputs = vec![TensorIteratorInput {
            tensor: a.clone(),
            sizes: a.inner.sizes.clone(),
            strides: a_strides,
            storage_offset: a.inner.storage_offset,
        }];

        let inner_dim = find_contiguous_inner_dim(&output_shape);
        let inner_strides: SmallVec<[i64; 8]> = (0..output_shape.len())
            .map(|i| if i >= inner_dim { output_shape[i] } else { 1 })
            .collect();

        TensorIterator {
            output_shape,
            inner_strides,
            inputs,
            inner_dim,
            numel: numel as usize,
        }
    }

    #[allow(dead_code)]
    pub fn build_for_reduction(a: &Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        let input_shape = &a.inner.sizes;
        let mut output_shape: SmallVec<[i64; 8]> = smallvec![];

        match dim {
            Some(d) => {
                for (i, &s) in input_shape.iter().enumerate() {
                    if i == d {
                        if keepdim {
                            output_shape.push(1);
                        }
                    } else {
                        output_shape.push(s);
                    }
                }
            }
            None => {
                if keepdim {
                    output_shape = smallvec![1; input_shape.len()];
                }
            }
        }

        let numel: i64 = if output_shape.is_empty() {
            1
        } else {
            output_shape.iter().product()
        };

        let inputs = vec![TensorIteratorInput {
            tensor: a.clone(),
            sizes: a.inner.sizes.clone(),
            strides: a.inner.strides.clone(),
            storage_offset: a.inner.storage_offset,
        }];

        let inner_dim = 0;
        let inner_strides: SmallVec<[i64; 8]> = smallvec![];

        TensorIterator {
            output_shape,
            inner_strides,
            inputs,
            inner_dim,
            numel: numel as usize,
        }
    }

    #[allow(dead_code)]
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&[&[u8]]) + Send + Sync,
    {
        if self.numel == 0 {
            return;
        }

        let ndim = self.output_shape.len();
        let output_numel = self.numel;

        // Pre-compute strides for carry-based offset tracking
        let mut input_offsets: SmallVec<[i64; 4]> =
            self.inputs.iter().map(|inp| inp.storage_offset).collect();
        let mut indices = vec![0usize; ndim];

        for _ in 0..output_numel {
            let mut ptrs: SmallVec<[&[u8]; 4]> = SmallVec::with_capacity(self.inputs.len());

            for (inp_idx, input) in self.inputs.iter().enumerate() {
                let ptr = match input.tensor.inner.storage.as_ref() {
                    crate::storage::Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr(),
                    crate::storage::Storage::Wgpu(_) => {
                        panic!("Iterator doesn't support GPU storage. Use .to_cpu() first.");
                    }
                };
                let nbytes = input.tensor.inner.dtype.size();
                let offset = input_offsets[inp_idx] as usize * nbytes;
                let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset), nbytes) };
                ptrs.push(slice);
            }

            f(&ptrs);

            // Increment indices and update offsets (carry-based)
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.output_shape[d] as usize {
                    for (inp_idx, input) in self.inputs.iter().enumerate() {
                        if d < input.strides.len() && d < input.sizes.len() && input.sizes[d] != 1 {
                            input_offsets[inp_idx] += input.strides[d];
                        }
                    }
                    break;
                }
                for (inp_idx, input) in self.inputs.iter().enumerate() {
                    if d < input.strides.len() && d < input.sizes.len() && input.sizes[d] != 1 {
                        input_offsets[inp_idx] -= (self.output_shape[d] - 1) * input.strides[d];
                    }
                }
                indices[d] = 0;
            }
        }
    }

    #[allow(dead_code)]
    pub fn for_each_with_index<F>(&self, mut f: F)
    where
        F: FnMut(usize, &[&[u8]]) + Send + Sync,
    {
        if self.numel == 0 {
            return;
        }

        let ndim = self.output_shape.len();
        let output_numel = self.numel;

        let mut input_offsets: SmallVec<[i64; 4]> =
            self.inputs.iter().map(|inp| inp.storage_offset).collect();
        let mut indices = vec![0usize; ndim];

        for idx in 0..output_numel {
            let mut ptrs: SmallVec<[&[u8]; 4]> = SmallVec::with_capacity(self.inputs.len());

            for (inp_idx, input) in self.inputs.iter().enumerate() {
                let ptr = match input.tensor.inner.storage.as_ref() {
                    crate::storage::Storage::Cpu(cpu) => cpu.data.as_ref().as_ptr(),
                    crate::storage::Storage::Wgpu(_) => {
                        panic!("Iterator doesn't support GPU storage. Use .to_cpu() first.");
                    }
                };
                let nbytes = input.tensor.inner.dtype.size();
                let offset = input_offsets[inp_idx] as usize * nbytes;
                let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset), nbytes) };
                ptrs.push(slice);
            }

            f(idx, &ptrs);

            // Increment indices and update offsets (carry-based)
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.output_shape[d] as usize {
                    for (inp_idx, input) in self.inputs.iter().enumerate() {
                        if d < input.strides.len() && d < input.sizes.len() && input.sizes[d] != 1 {
                            input_offsets[inp_idx] += input.strides[d];
                        }
                    }
                    break;
                }
                for (inp_idx, input) in self.inputs.iter().enumerate() {
                    if d < input.strides.len() && d < input.sizes.len() && input.sizes[d] != 1 {
                        input_offsets[inp_idx] -= (self.output_shape[d] - 1) * input.strides[d];
                    }
                }
                indices[d] = 0;
            }
        }
    }
}

fn broadcast_shapes(a: &[i64], b: &[i64]) -> SmallVec<[i64; 8]> {
    let ndim = std::cmp::max(a.len(), b.len());
    let mut result: SmallVec<[i64; 8]> = SmallVec::new();
    result.resize(ndim, 1);

    let offset = ndim - a.len();
    for (i, &s) in a.iter().enumerate() {
        result[offset + i] = s;
    }

    let offset = ndim - b.len();
    for (i, &s) in b.iter().enumerate() {
        result[offset + i] = std::cmp::max(result[offset + i], s);
    }

    let offset = ndim - a.len();
    for (i, &s) in a.iter().enumerate() {
        let idx = offset + i;
        let target = result[idx];
        if s != 1 && s != target {
            panic!("broadcast: cannot broadcast shape {:?} and {:?}", a, b);
        }
    }

    let offset = ndim - b.len();
    for (i, &s) in b.iter().enumerate() {
        let idx = offset + i;
        let target = result[idx];
        if s != 1 && s != target {
            panic!("broadcast: cannot broadcast shape {:?} and {:?}", a, b);
        }
    }

    result
}

fn compute_broadcast_strides(input_shape: &[i64], output_shape: &[i64]) -> SmallVec<[i64; 8]> {
    let ndim = output_shape.len();
    let input_offset = ndim - input_shape.len();
    let mut strides: SmallVec<[i64; 8]> = SmallVec::new();
    strides.resize(ndim, 0);

    let mut stride = 1i64;
    for i in (0..input_shape.len()).rev() {
        let output_idx = input_offset + i;
        if input_shape[i] == output_shape[output_idx] {
            strides[output_idx] = stride;
        } else if input_shape[i] == 1 {
            strides[output_idx] = 0;
        }
        stride *= input_shape[i];
    }

    strides
}

fn find_contiguous_inner_dim(shape: &[i64]) -> usize {
    if shape.is_empty() {
        return 0;
    }

    let mut inner_dim = shape.len() - 1;
    let mut expected_stride = 1i64;

    while inner_dim > 0 {
        if shape[inner_dim] == expected_stride {
            expected_stride *= shape[inner_dim - 1];
            inner_dim -= 1;
        } else {
            break;
        }
    }

    inner_dim
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    #[test]
    fn test_broadcast_shapes_same() {
        let a = [3, 4];
        let b = [3, 4];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[3i64, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_extra_dim() {
        let a = [2, 3, 4];
        let b = [3, 4];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[2i64, 3, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        let a = [];
        let b = [2, 3, 4];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[2i64, 3, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_one_dim() {
        let a = [4];
        let b = [2, 3, 4];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[2i64, 3, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_one_in_shape() {
        let a = [1, 4];
        let b = [2, 3, 4];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[2i64, 3, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_both_ones() {
        let a = [1, 4];
        let b = [3, 1];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[3i64, 4][..]);
    }

    #[test]
    fn test_broadcast_shapes_empty() {
        let a = [];
        let b = [];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[] as &[i64]);
    }

    #[test]
    #[should_panic(expected = "broadcast: cannot broadcast shape")]
    fn test_broadcast_shapes_incompatible() {
        let a = [3, 4];
        let b = [2, 3];
        broadcast_shapes(&a, &b);
    }

    #[test]
    fn test_broadcast_shapes_large_dims() {
        let a = [1, 1, 5, 1, 7];
        let b = [2, 3, 1, 1, 7];
        let result = broadcast_shapes(&a, &b);
        assert_eq!(result.as_slice(), &[2i64, 3, 5, 1, 7][..]);
    }
}
