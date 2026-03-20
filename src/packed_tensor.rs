use crate::dtypes::PackedWord;

/// Create a zero-initialized Vec<T>.
fn zeroed_vec<T: bytemuck::Pod>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }
    let mut v = Vec::with_capacity(len);
    unsafe {
        std::ptr::write_bytes(v.as_mut_ptr() as *mut u8, 0, len * std::mem::size_of::<T>());
        v.set_len(len);
    }
    v
}

/// A tensor whose values are packed into u32 words using a PackedWord type.
///
/// For U4x8: 8 values per u32 (8x memory savings vs f32)
/// For U8x4: 4 values per u32 (4x memory savings vs f32)
/// For F16x2: 2 values per u32 (2x memory savings vs f32)
/// For F32x1: 1 value per u32 (baseline, no packing)
pub struct PackedTensor<T: PackedWord> {
    /// Packed storage
    data: Vec<T>,
    /// Logical shape in element counts (not word counts)
    shape: Vec<usize>,
    /// Global scale factor for dequantization: real = packed * scale + zero
    scale: f32,
    /// Zero point for dequantization
    zero: f32,
}

impl<T: PackedWord> PackedTensor<T> {
    /// Create a zero-initialized packed tensor with the given logical shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = numel.div_ceil(T::ITEMS);
        let mut data = zeroed_vec(packed_len);
        for d in data.iter_mut() {
            *d = T::zeroed();
        }
        PackedTensor {
            data,
            shape: shape.to_vec(),
            scale: 1.0,
            zero: 0.0,
        }
    }

    /// Create a packed tensor from an f32 slice with the given scale and zero point.
    /// Values are quantized and packed.
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
        let mut packed = zeroed_vec(packed_len);

        for chunk_idx in 0..packed_len {
            let mut arr = T::Array::default();
            let arr_ref = arr.as_mut();
            for i in 0..T::ITEMS {
                let elem_idx = chunk_idx * T::ITEMS + i;
                if elem_idx < numel {
                    arr_ref[i] = if scale != 1.0 || zero != 0.0 {
                        (data[elem_idx] - zero) / scale
                    } else {
                        data[elem_idx]
                    };
                }
            }
            packed[chunk_idx] = T::pack_from_f32(arr);
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scale,
            zero,
        }
    }

    /// Unpack all values to a Vec<f32>, applying scale and zero correction.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        for (i, word) in self.data.iter().enumerate() {
            let unpacked = word.unpack_to_f32();
            let arr = unpacked.as_ref();
            for j in 0..T::ITEMS {
                let elem_idx = i * T::ITEMS + j;
                if elem_idx < numel {
                    let val = if self.scale != 1.0 || self.zero != 0.0 {
                        arr[j] * self.scale + self.zero
                    } else {
                        arr[j]
                    };
                    result.push(val);
                }
            }
        }
        result
    }

    /// Number of u32 words in the packed representation.
    #[inline]
    pub fn packed_len(&self) -> usize {
        self.data.len()
    }

    /// Total number of logical elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the logical shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the scale factor.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get the zero point.
    #[inline]
    pub fn zero(&self) -> f32 {
        self.zero
    }

    /// Get a single element as f32.
    pub fn get(&self, idx: usize) -> f32 {
        assert!(idx < self.numel(), "Index out of bounds");
        let word_idx = idx / T::ITEMS;
        let elem_idx = idx % T::ITEMS;
        let unpacked = self.data[word_idx].unpack_to_f32();
        let val = unpacked.as_ref()[elem_idx];
        if self.scale != 1.0 || self.zero != 0.0 {
            val * self.scale + self.zero
        } else {
            val
        }
    }

    /// Set a single element from f32.
    pub fn set(&mut self, idx: usize, val: f32) {
        assert!(idx < self.numel(), "Index out of bounds");
        let word_idx = idx / T::ITEMS;
        let elem_idx = idx % T::ITEMS;
        let mut arr = self.data[word_idx].unpack_to_f32();
        let quantized = if self.scale != 1.0 || self.zero != 0.0 {
            (val - self.zero) / self.scale
        } else {
            val
        };
        arr.as_mut()[elem_idx] = quantized;
        self.data[word_idx] = T::pack_from_f32(arr);
    }

    /// Get the raw packed data as a slice of T.
    #[inline]
    pub fn as_packed(&self) -> &[T] {
        &self.data
    }

    /// Get the raw packed data as a mutable slice of T.
    #[inline]
    pub fn as_packed_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get the raw packed data as a byte slice (for wgpu buffer upload).
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }

    /// Get the raw packed data as a slice of u32.
    #[inline]
    pub fn as_u32(&self) -> &[u32] {
        // SAFETY: T is repr(transparent) over u32 for all our types
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u32, self.data.len()) }
    }

    /// Compute optimal scale for packing from an f32 slice.
    /// For INT4: scale = max_abs / 7.0
    /// For INT8: scale = max_abs / 127.0
    /// For FP types: scale = 1.0
    pub fn compute_scale(data: &[f32]) -> f32 {
        if T::IS_FLOAT {
            return 1.0;
        }
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_val = ((1u32 << (T::BIT_WIDTH - 1)) - 1) as f32; // 7 for 4-bit, 127 for 8-bit
        if max_abs == 0.0 {
            1.0
        } else {
            max_abs / max_val
        }
    }

    /// Create from f32 slice with automatic scale computation.
    pub fn from_f32_auto(data: &[f32], shape: &[usize]) -> Self {
        let scale = Self::compute_scale(data);
        Self::from_f32_slice(data, shape, scale, 0.0)
    }
}

impl<T: PackedWord> Clone for PackedTensor<T> {
    fn clone(&self) -> Self {
        PackedTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            scale: self.scale,
            zero: self.zero,
        }
    }
}

impl<T: PackedWord> std::fmt::Debug for PackedTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedTensor")
            .field("shape", &self.shape)
            .field("packed_len", &self.data.len())
            .field("scale", &self.scale)
            .field("zero", &self.zero)
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
        assert_eq!(t.packed_len(), 2); // 16 elements / 8 per word
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
                i,
                orig,
                rec,
                t.scale()
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
                "Mismatch at index {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_packed_tensor_f16x2() {
        let data: Vec<f32> = vec![1.5, -2.5];
        let t = PackedTensor::<F16x2>::from_f32_auto(&data, &[2]);
        let recovered = t.to_f32_vec();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 0.01,
                "Mismatch: orig={}, rec={}",
                orig,
                rec
            );
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
        // Value should be approximately 3.0 (with quantization error)
        assert!((val - 3.0).abs() <= t.scale() + 0.01);
        // Set to a value within the representable range (scale is ~0.03 for this data)
        // Quantized: 3.5 / scale ≈ 111, fits in i8 range
        t.set(2, 3.5);
        let val2 = t.get(2);
        assert!((val2 - 3.5).abs() <= t.scale() + 0.01);
    }
}
