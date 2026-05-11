use crate::dtypes::PackedWord;

fn zeroed_vec<T: bytemuck::Pod>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(len);

// SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
    unsafe {
        std::ptr::write_bytes(v.as_mut_ptr() as *mut u8, 0, len * std::mem::size_of::<T>());
        v.set_len(len);
    }
    v
}

pub struct PackedTensor<T: PackedWord> {
    pub(crate) data: Vec<T>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scales: Vec<f32>,
    pub(crate) zeros: Vec<f32>,
}

impl<T: PackedWord> PackedTensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(T::ITEMS);
        let data = zeroed_vec(packed_len);
        PackedTensor {
            data,
            shape: shape.to_vec(),
            scales: vec![1.0],
            zeros: vec![0.0],
        }
    }

    pub fn from_raw(data: Vec<T>, shape: Vec<usize>, scales: Vec<f32>, zeros: Vec<f32>) -> Self {
        PackedTensor { data, shape, scales, zeros }
    }

    pub fn from_f32_slice(data: &[f32], shape: &[usize], scale: f32, zero: f32) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        let packed_len = numel.div_ceil(T::ITEMS);
        let mut packed = vec![T::default(); packed_len];

        if scale != 1.0 || zero != 0.0 {
            let inv_scale = 1.0 / scale;
            for chunk_idx in 0..packed_len {
                let mut arr = T::Array::default();
                let arr_ref = arr.as_mut();
                for i in 0..T::ITEMS {
                    let elem_idx = chunk_idx * T::ITEMS + i;
                    if elem_idx < numel {
                        arr_ref[i] = (data[elem_idx] - zero) * inv_scale;
                    }
                }
                packed[chunk_idx] = T::pack_from_f32(arr);
            }
        } else {
            for chunk_idx in 0..packed_len {
                let mut arr = T::Array::default();
                let arr_ref = arr.as_mut();
                for i in 0..T::ITEMS {
                    let elem_idx = chunk_idx * T::ITEMS + i;
                    if elem_idx < numel {
                        arr_ref[i] = data[elem_idx];
                    }
                }
                packed[chunk_idx] = T::pack_from_f32(arr);
            }
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scales: vec![scale],
            zeros: vec![zero],
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let numel = self.numel();
        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            numel
        };
        let items = T::ITEMS;
        let mut result = Vec::with_capacity(numel);
        let is_per_row = self.scales.len() > 1;

        for (i, word) in self.data.iter().enumerate() {
            let unpacked = word.unpack_to_f32();
            let arr = unpacked.as_ref();
            let elem_idx_base = i * items;
            let row = elem_idx_base / inner_stride;
            let s = if is_per_row {
                self.scales[row]
            } else {
                self.scales[0]
            };
            let z = if is_per_row {
                self.zeros[row]
            } else {
                self.zeros[0]
            };
            if s != 1.0 || z != 0.0 {
                for j in 0..items {
                    let elem_idx = elem_idx_base + j;
                    if elem_idx < numel {
                        result.push(arr[j] * s + z);
                    }
                }
            } else {
                for j in 0..items {
                    let elem_idx = elem_idx_base + j;
                    if elem_idx < numel {
                        result.push(arr[j]);
                    }
                }
            }
        }
        result
    }

    #[inline]
    pub fn packed_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scales[0]
    }

    #[inline]
    pub fn zero(&self) -> f32 {
        self.zeros[0]
    }

    #[inline]
    pub fn scale_for_row(&self, row: usize) -> f32 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[row]
        }
    }

    #[inline]
    pub fn zero_for_row(&self, row: usize) -> f32 {
        if self.zeros.len() == 1 {
            self.zeros[0]
        } else {
            self.zeros[row]
        }
    }

    #[inline]
    pub fn is_per_channel(&self) -> bool {
        self.scales.len() > 1
    }

    pub fn get(&self, idx: usize) -> f32 {
        assert!(idx < self.numel(), "Index out of bounds");
        let word_idx = idx / T::ITEMS;
        let elem_idx = idx % T::ITEMS;
        let unpacked = self.data[word_idx].unpack_to_f32();
        let val = unpacked.as_ref()[elem_idx];
        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            self.numel()
        };
        let row = idx / inner_stride;
        let s = self.scale_for_row(row);
        let z = self.zero_for_row(row);
        if s != 1.0 || z != 0.0 {
            val * s + z
        } else {
            val
        }
    }

    pub fn set(&mut self, idx: usize, val: f32) {
        assert!(idx < self.numel(), "Index out of bounds");
        let word_idx = idx / T::ITEMS;
        let elem_idx = idx % T::ITEMS;
        let mut arr = self.data[word_idx].unpack_to_f32();
        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            self.numel()
        };
        let row = idx / inner_stride;
        let s = self.scale_for_row(row);
        let z = self.zero_for_row(row);
        let quantized = if s != 1.0 || z != 0.0 {
            (val - z) / s
        } else {
            val
        };
        arr.as_mut()[elem_idx] = quantized;
        self.data[word_idx] = T::pack_from_f32(arr);
    }

    #[inline]
    pub fn as_packed(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_packed_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    #[inline]
    pub fn as_u32(&self) -> &[u32] {

// SAFETY: The pointer is valid, properly aligned, and points to `len` initialized elements derived from a valid Tensor allocation.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u32, self.data.len()) }
    }

    pub fn add(&self, other: &Self) -> Self {
        crate::backends::cpu::add_cpu(self, other)
    }

    pub fn sub(&self, other: &Self) -> Self {
        crate::backends::cpu::sub_cpu(self, other)
    }

    pub fn max(&self, other: &Self) -> Self {
        crate::backends::cpu::max_cpu(self, other)
    }

    pub fn min(&self, other: &Self) -> Self {
        crate::backends::cpu::min_cpu(self, other)
    }

    pub fn relu_backward_inplace(&mut self, pre_relu: &Self) {
        crate::backends::cpu::relu_backward_cpu(self, pre_relu);
    }

    pub fn compute_scale(data: &[f32]) -> f32 {
        if T::IS_FLOAT {
            return 1.0;
        }
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_val = ((1u32 << (T::BIT_WIDTH - 1)) - 1) as f32;
        if max_abs == 0.0 {
            1.0
        } else {
            max_abs / max_val
        }
    }

    pub fn from_f32_auto(data: &[f32], shape: &[usize]) -> Self {
        let scale = Self::compute_scale(data);
        Self::from_f32_slice(data, shape, scale, 0.0)
    }

    pub fn compute_scales_per_channel(data: &[f32], shape: &[usize]) -> Vec<f32> {
        if T::IS_FLOAT {
            return vec![1.0];
        }
        assert!(shape.len() >= 2, "Per-channel requires 2D+ shape");
        let m = shape[0];
        let inner_stride: usize = shape[1..].iter().product();
        let max_val = ((1u32 << (T::BIT_WIDTH - 1)) - 1) as f32;
        let mut scales = Vec::with_capacity(m);
        for row in 0..m {
            let start = row * inner_stride;
            let end = start + inner_stride;
            let max_abs = data[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            scales.push(if max_abs == 0.0 {
                1.0
            } else {
                max_abs / max_val
            });
        }
        scales
    }

    pub fn from_f32_per_channel(data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);

        let scales = Self::compute_scales_per_channel(data, shape);
        let m = if shape.len() >= 2 { shape[0] } else { 1 };
        let inner_stride = if shape.len() >= 2 { shape[1..].iter().product() } else { numel };

        let packed_len = numel.div_ceil(T::ITEMS);
        let mut packed = vec![T::default(); packed_len];

        for chunk_idx in 0..packed_len {
            let mut arr = T::Array::default();
            let arr_ref = arr.as_mut();
            let elem_idx_base = chunk_idx * T::ITEMS;
            let row = elem_idx_base / inner_stride;
            let s = if scales.len() == 1 {
                scales[0]
            } else {
                scales[row]
            };
            if s != 1.0 {
                let inv_s = 1.0 / s;
                for i in 0..T::ITEMS {
                    let elem_idx = elem_idx_base + i;
                    if elem_idx < numel {
                        arr_ref[i] = data[elem_idx] * inv_s;
                    }
                }
            } else {
                for i in 0..T::ITEMS {
                    let elem_idx = elem_idx_base + i;
                    if elem_idx < numel {
                        arr_ref[i] = data[elem_idx];
                    }
                }
            }
            packed[chunk_idx] = T::pack_from_f32(arr);
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scales,
            zeros: vec![0.0; m],
        }
    }
}

impl<T: PackedWord> Clone for PackedTensor<T> {
    fn clone(&self) -> Self {
        PackedTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            scales: self.scales.clone(),
            zeros: self.zeros.clone(),
        }
    }
}

impl<T: PackedWord> std::fmt::Debug for PackedTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTensor")
            .field("shape", &self.shape)
            .field("packed_len", &self.data.len())
            .field("scales", &self.scales)
            .field("zeros", &self.zeros)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::F16x2;
    use crate::dtypes::F32x1;
    use crate::dtypes::U4x8;
    use crate::dtypes::U8x4;

    #[test]
    fn test_packed_tensor_zeros() {
        let t = PackedTensor::<U4x8>::zeros(&[16]);
        assert_eq!(t.numel(), 16);
        assert_eq!(t.packed_len(), 2);
        let vals = t.to_f32_vec();
        assert!(vals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_packed_tensor_roundtrip_u4x8() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
        let t = PackedTensor::<U4x8>::from_f32_auto(&data, &[16]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let tolerance = t.scale() + 0.01;
            assert!(
                (orig - rec).abs() <= tolerance,
                "Mismatch at index {}: orig={}, rec={}, scale={}",
                i, orig, rec, t.scale()
            );
        }
    }

    #[test]
    fn test_packed_tensor_roundtrip_u8x4() {
        let data: Vec<f32> = vec![0.0, 50.0, -50.0, 100.0];
        let t = PackedTensor::<U8x4>::from_f32_auto(&data, &[4]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= t.scale() + 0.01,
                "Mismatch at index {}: orig={}, rec={}", i, orig, rec
            );
        }
    }

    #[test]
    fn test_packed_tensor_f16x2() {
        let data: Vec<f32> = vec![1.5, -2.5];
        let t = PackedTensor::<F16x2>::from_f32_auto(&data, &[2]);
        let recovered = t.to_f32_vec();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.01, "Mismatch: orig={}, rec={}", orig, rec);
        }
    }

    #[test]
    fn test_packed_tensor_f32x1() {
        let data: Vec<f32> = vec![3.14, -2.71, 1.41];
        let t = PackedTensor::<F32x1>::from_f32_auto(&data, &[3]);
        let recovered = t.to_f32_vec();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_eq!(orig, rec);
        }
    }

    #[test]
    fn test_get_set() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut t = PackedTensor::<U8x4>::from_f32_auto(&data, &[4]);
        let val = t.get(2);
        assert!((val - 3.0).abs() <= t.scale() + 0.01);
        t.set(2, 3.5);
        let val2 = t.get(2);
        assert!((val2 - 3.5).abs() <= t.scale() + 0.01);
    }
}
