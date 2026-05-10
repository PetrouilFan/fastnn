//! Quantized tensor storage with block-wise scales and lazy loading.
//!
//! This enables:
//! - 75% memory reduction with INT4 KV cache
//! - Per-block quantization for better accuracy

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;
use std::marker::PhantomData;

/// A quantizable tensor that stores data in blocks with independent scale/zero.
/// Each block has independent scale/zero for better accuracy.
pub struct QuantizedTensor<T: PackedWord> {
    /// Block-wise scales and zeros: [(scale, zero), ...]
    pub scale_zp: Vec<(f32, f32)>,
    /// Raw packed data
    data: Vec<T>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Block size (in elements) used for quantization
    pub block_size: usize,
    _marker: PhantomData<T>,
}

impl<T: PackedWord> QuantizedTensor<T> {
    /// Create a new quantized tensor from f32 data with block-wise quantization.
    pub fn from_f32_blockwise(data: &[f32], shape: &[usize], block_size: usize) -> Self {
        assert!(
            block_size % T::ITEMS == 0,
            "block_size ({}) must be a multiple of T::ITEMS ({}) for QuantizedTensor",
            block_size, T::ITEMS
        );
        let numel: usize = shape.iter().product();
        let n_blocks = numel.div_ceil(block_size);

        let mut scale_zp = Vec::with_capacity(n_blocks);
        let mut packed_data = Vec::with_capacity(numel.div_ceil(T::ITEMS));

        for block_idx in 0..n_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(numel);

            let max_abs = data[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            let max_val = ((1u32 << (T::BIT_WIDTH - 1)) - 1) as f32;
            let scale = if max_abs == 0.0 {
                1.0
            } else {
                max_abs / max_val
            };

            pack_block::<T>(&data[start..end], scale, 0.0, &mut packed_data);

            scale_zp.push((scale, 0.0));
        }

        QuantizedTensor {
            scale_zp,
            data: packed_data,
            shape: shape.to_vec(),
            block_size,
            _marker: PhantomData,
        }
    }

    /// Convert to PackedTensor for computation.
    pub fn to_packed(&self) -> PackedTensor<T> {
        let numel: usize = self.shape.iter().product();
        let packed_len = numel.div_ceil(T::ITEMS);
        let max_val = ((1u32 << (T::BIT_WIDTH - 1)) - 1) as f32;

        // Pass 1: compute global max_abs across all dequantized values (no full buffer)
        let mut global_max_abs = 0.0f32;
        for block_idx in 0..self.scale_zp.len() {
            let (scale, _zero) = self.scale_zp[block_idx];
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(numel);
            let packed_start = start / T::ITEMS;
            let packed_end = end.div_ceil(T::ITEMS);

            for p in packed_start..packed_end {
                let word = self.data[p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx >= start && idx < end {
                        let val = unpacked.as_ref()[j] * scale;
                        let abs_val = val.abs();
                        if abs_val > global_max_abs {
                            global_max_abs = abs_val;
                        }
                    }
                }
            }
        }

        let global_scale = if global_max_abs == 0.0 {
            1.0
        } else {
            global_max_abs / max_val
        };
        let inv_global = 1.0 / global_scale;

        // Pass 2: write directly to pre-allocated packed buffer
        let mut packed = vec![T::default(); packed_len];

        for block_idx in 0..self.scale_zp.len() {
            let (block_scale, _zero) = self.scale_zp[block_idx];
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(numel);
            let packed_start = start / T::ITEMS;
            let packed_end = end.div_ceil(T::ITEMS);

            let ratio = block_scale * inv_global;

            for p in packed_start..packed_end {
                let word = self.data[p];
                let unpacked = word.unpack_to_f32();
                let mut arr = <T as PackedWord>::Array::default();
                let arr_ref = arr.as_mut();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx >= start && idx < end {
                        arr_ref[j] = unpacked.as_ref()[j] * ratio;
                    }
                }
                packed[p] = T::pack_from_f32(arr);
            }
        }

        PackedTensor {
            data: packed,
            shape: self.shape.clone(),
            scales: vec![global_scale],
            zeros: vec![0.0],
        }
    }

    /// Convert to f32 vec (dequantized).
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let numel: usize = self.shape.iter().product();
        let mut result = Vec::with_capacity(numel);

        for block_idx in 0..self.scale_zp.len() {
            let (scale, zero) = self.scale_zp[block_idx];
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(numel);

            let packed_start = start / T::ITEMS;
            let packed_end = end.div_ceil(T::ITEMS);

            for p in packed_start..packed_end {
                let word = self.data[p];
                let unpacked = word.unpack_to_f32();
                let base = p * T::ITEMS;
                for j in 0..T::ITEMS {
                    let idx = base + j;
                    if idx >= start && idx < end {
                        let val = unpacked.as_ref()[j];
                        result.push(val * scale + zero);
                    }
                }
            }
        }
        result
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * 4 + self.scale_zp.len() * 8
    }
}

/// Pack a block of f32 values into packed representation.
fn pack_block<T: PackedWord>(block_data: &[f32], scale: f32, zero: f32, packed: &mut Vec<T>) {
    let items = T::ITEMS;
    let packed_len = block_data.len().div_ceil(items);

    if scale != 1.0 || zero != 0.0 {
        let inv_scale = 1.0 / scale;
        for chunk_idx in 0..packed_len {
            let mut arr: <T as PackedWord>::Array = <T as PackedWord>::Array::default();
            let arr_ref = arr.as_mut();
            for i in 0..items {
                let elem_idx = chunk_idx * items + i;
                if elem_idx < block_data.len() {
                    arr_ref[i] = (block_data[elem_idx] - zero) * inv_scale;
                }
            }
            packed.push(T::pack_from_f32(arr));
        }
    } else {
        for chunk_idx in 0..packed_len {
            let mut arr: <T as PackedWord>::Array = <T as PackedWord>::Array::default();
            let arr_ref = arr.as_mut();
            for i in 0..items {
                let elem_idx = chunk_idx * items + i;
                if elem_idx < block_data.len() {
                    arr_ref[i] = block_data[elem_idx];
                }
            }
            packed.push(T::pack_from_f32(arr));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::U4x8;
    use crate::dtypes::U8x4;

    fn max_absolute_error(orig: &[f32], recovered: &[f32]) -> f32 {
        orig.iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_quantized_tensor_creation() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) - 64.0).collect();
        let qt = QuantizedTensor::<U4x8>::from_f32_blockwise(&data, &[128], 32);
        assert_eq!(qt.scale_zp.len(), 4);
        assert!(qt.memory_bytes() < data.len());
    }

    #[test]
    fn test_quantized_tensor_roundtrip_u4x8() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.5).collect();
        let qt = QuantizedTensor::<U4x8>::from_f32_blockwise(&data, &[64], 16);
        let recovered = qt.to_f32_vec();
        let max_err = max_absolute_error(&data, &recovered);
        let max_scale = qt.scale_zp.iter().map(|&(s, _)| s).fold(0.0f32, f32::max);
        assert!(
            max_err <= max_scale + 1e-4,
            "U4x8 roundtrip max error {} exceeds scale {}",
            max_err,
            max_scale
        );
    }

    #[test]
    fn test_quantized_tensor_roundtrip_u8x4() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 2.0).collect();
        let qt = QuantizedTensor::<U8x4>::from_f32_blockwise(&data, &[64], 16);
        let recovered = qt.to_f32_vec();
        let max_err = max_absolute_error(&data, &recovered);
        let max_scale = qt.scale_zp.iter().map(|&(s, _)| s).fold(0.0f32, f32::max);
        assert!(
            max_err <= max_scale + 1e-4,
            "U8x4 roundtrip max error {} exceeds scale {}",
            max_err,
            max_scale
        );
    }

    #[test]
    fn test_quantized_tensor_to_packed_u4x8() {
        let data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.5).collect();
        let qt = QuantizedTensor::<U4x8>::from_f32_blockwise(&data, &[64], 16);
        let packed = qt.to_packed();
        let recovered = packed.to_f32_vec();
        let max_err = max_absolute_error(&data, &recovered);
        let max_scale = qt.scale_zp.iter().map(|&(s, _)| s).fold(0.0f32, f32::max);
        assert!(
            max_err <= max_scale + 1e-4,
            "to_packed roundtrip max error {} exceeds scale {}",
            max_err,
            max_scale
        );
    }
}
