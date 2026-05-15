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
    /// Block size for block-major layout (1 = standard row-major).
    /// When > 1, packed data is interleaved in blocks of `block_size` rows
    /// for SIMD-friendly access patterns: within each block, all rows' words
    /// at the same K-position are stored contiguously.
    pub(crate) block_size: usize,
}

impl<T: PackedWord> PackedTensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let packed_len = if shape.len() >= 2 {
            let m = shape[0];
            let inner: usize = shape[1..].iter().product();
            m * inner.div_ceil(T::ITEMS)
        } else {
            numel.div_ceil(T::ITEMS)
        };
        let data = zeroed_vec(packed_len);
        PackedTensor {
            data,
            shape: shape.to_vec(),
            scales: vec![1.0],
            zeros: vec![0.0],
            block_size: 1,
        }
    }

    pub fn from_raw(data: Vec<T>, shape: Vec<usize>, scales: Vec<f32>, zeros: Vec<f32>) -> Self {
        PackedTensor {
            data,
            shape,
            scales,
            zeros,
            block_size: 1,
        }
    }

    pub fn from_raw_blocked(
        data: Vec<T>,
        shape: Vec<usize>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
        block_size: usize,
    ) -> Self {
        PackedTensor {
            data,
            shape,
            scales,
            zeros,
            block_size,
        }
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

        // Use per-row packing for 2D+ tensors so each row is independently packed.
        // This matches the layout expected by GEMV kernels (row * k_packed offset).
        let (m, inner_stride) = if shape.len() >= 2 {
            (shape[0], shape[1..].iter().product())
        } else {
            (1, numel)
        };
        let k_packed = inner_stride.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        const SIMD_MARGIN: usize = 16;
        let mut packed = vec![T::default(); packed_len + SIMD_MARGIN];

        if scale != 1.0 || zero != 0.0 {
            let inv_scale = 1.0 / scale;
            for row in 0..m {
                for word in 0..k_packed {
                    let chunk_idx = row * k_packed + word;
                    let mut arr = T::Array::default();
                    let arr_ref = arr.as_mut();
                    for i in 0..T::ITEMS {
                        let elem_idx = row * inner_stride + word * T::ITEMS + i;
                        if elem_idx < (row + 1) * inner_stride && elem_idx < numel {
                            arr_ref[i] = (data[elem_idx] - zero) * inv_scale;
                        }
                    }
                    packed[chunk_idx] = T::pack_from_f32(arr);
                }
            }
        } else {
            for row in 0..m {
                for word in 0..k_packed {
                    let chunk_idx = row * k_packed + word;
                    let mut arr = T::Array::default();
                    let arr_ref = arr.as_mut();
                    for i in 0..T::ITEMS {
                        let elem_idx = row * inner_stride + word * T::ITEMS + i;
                        if elem_idx < (row + 1) * inner_stride && elem_idx < numel {
                            arr_ref[i] = data[elem_idx];
                        }
                    }
                    packed[chunk_idx] = T::pack_from_f32(arr);
                }
            }
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scales: vec![scale],
            zeros: vec![zero],
            block_size: 1,
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let numel = self.numel();

        // Block-major: slow path via get() — only used in debugging/tests
        if self.is_block_major() {
            let mut result = Vec::with_capacity(numel);
            for i in 0..numel {
                result.push(self.get(i));
            }
            return result;
        }

        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            numel
        };
        let items = T::ITEMS;
        let mut result = Vec::with_capacity(numel);
        let is_per_row = self.scales.len() > 1;

        // Per-row packing: word `i` = (row * k_packed + word_in_row)
        let k_packed = inner_stride.div_ceil(items);
        let m = if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            1
        };
        let actual_words = m * k_packed;

        for (i, word) in self.data[..actual_words].iter().enumerate() {
            let unpacked = word.unpack_to_f32();
            let arr = unpacked.as_ref();
            let row = i.checked_div(k_packed).unwrap_or(0);
            let word_in_row = i.checked_rem(k_packed).unwrap_or(0);
            let elem_idx_base = row * inner_stride + word_in_row * items;
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
            let row_end = (row + 1) * inner_stride;
            if s != 1.0 || z != 0.0 {
                for j in 0..items {
                    let elem_idx = elem_idx_base + j;
                    if elem_idx < numel && elem_idx < row_end {
                        result.push(arr[j] * s + z);
                    }
                }
            } else {
                for j in 0..items {
                    let elem_idx = elem_idx_base + j;
                    if elem_idx < numel && elem_idx < row_end {
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

    /// Compute the packed word index for a given logical element index.
    /// Handles both row-major (block_size=1) and block-major layouts.
    ///
    /// For a tensor with shape [M, K], element `idx` is at:
    ///   row = idx / K,  elem_in_row = idx % K
    ///   word_in_row = elem_in_row / T::ITEMS
    ///   elem_in_word = elem_in_row % T::ITEMS
    ///
    /// The packed word index depends on layout:
    /// - Row-major:  word = row * k_packed + word_in_row
    /// - Block-major (zone 1): word = block * block_size * k_packed + word_in_row * block_size + local_row
    /// - Block-major (zone 2): word = tail_base + local_row * k_packed + word_in_row
    fn word_index(&self, idx: usize) -> (usize, usize) {
        let items = T::ITEMS;
        let inner: usize = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            self.numel()
        };
        let k_packed = inner.div_ceil(items);
        let row = idx / inner;
        let elem_in_row = idx % inner;
        let word_in_row = elem_in_row / items;
        let elem_in_word = elem_in_row % items;

        if self.is_block_major() {
            let m = self.shape[0];
            let m_aligned = (m / self.block_size) * self.block_size;

            if row < m_aligned {
                // Zone 1: block-major interleaved
                let block_size = self.block_size;
                let block = row / block_size;
                let local_row = row % block_size;
                let word_idx = block * block_size * k_packed + word_in_row * block_size + local_row;
                (word_idx, elem_in_word)
            } else {
                // Zone 2: row-major tail (appended after block-major zone)
                let tail_local = row - m_aligned;
                let tail_base = m_aligned * k_packed;
                let word_idx = tail_base + tail_local * k_packed + word_in_row;
                (word_idx, elem_in_word)
            }
        } else {
            let word_idx = row * k_packed + word_in_row;
            (word_idx, elem_in_word)
        }
    }

    pub fn get(&self, idx: usize) -> f32 {
        assert!(idx < self.numel(), "Index out of bounds");
        let (word_idx, elem_idx) = self.word_index(idx);
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
        let (word_idx, elem_idx) = self.word_index(idx);
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
        bytemuck::cast_slice(&self.data)
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
        let inner_stride = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            numel
        };

        // Use per-row packing (same layout as from_f32_slice) for GEMV compatibility.
        // Each row has its own scale and occupies k_packed packed words.
        let k_packed = inner_stride.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        const SIMD_MARGIN: usize = 16;
        let mut packed = vec![T::default(); packed_len + SIMD_MARGIN];

        for row in 0..m {
            let row_scale = if scales.len() == 1 {
                scales[0]
            } else {
                scales[row]
            };
            let inv_s = if row_scale != 1.0 {
                1.0 / row_scale
            } else {
                1.0
            };
            for word in 0..k_packed {
                let chunk_idx = row * k_packed + word;
                let mut arr = T::Array::default();
                let arr_ref = arr.as_mut();
                for i in 0..T::ITEMS {
                    let elem_idx = row * inner_stride + word * T::ITEMS + i;
                    if elem_idx < (row + 1) * inner_stride && elem_idx < numel {
                        arr_ref[i] = data[elem_idx] * inv_s;
                    }
                }
                packed[chunk_idx] = T::pack_from_f32(arr);
            }
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scales,
            zeros: vec![0.0; m],
            block_size: 1,
        }
    }

    pub fn from_f32_per_channel_asymmetric(data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);

        let m = if shape.len() >= 2 { shape[0] } else { 1 };
        let inner_stride = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            numel
        };

        let mut mins = vec![f32::MAX; m];
        let mut maxs = vec![f32::MIN; m];
        for row in 0..m {
            for i in 0..inner_stride {
                let idx = row * inner_stride + i;
                if idx < numel {
                    let v = data[idx];
                    mins[row] = mins[row].min(v);
                    maxs[row] = maxs[row].max(v);
                }
            }
        }

        // Asymmetric quantization for signed types:
        //   scale = (max - min) / (2^n - 1)
        //   zero_point = min + 2^(n-1) * scale
        // This maps q = -2^(n-1) → x = min  and  q = 2^(n-1)-1 → x = max.
        let unsigned_max = ((1u32 << T::BIT_WIDTH) - 1) as f32;
        let signed_bias = (1u32 << (T::BIT_WIDTH - 1)) as f32;

        let mut scales = Vec::with_capacity(m);
        let mut zeros = Vec::with_capacity(m);
        for row in 0..m {
            let range = maxs[row] - mins[row];
            if range == 0.0 || unsigned_max == 0.0 {
                scales.push(1.0);
                zeros.push(0.0);
            } else {
                let scale = range / unsigned_max;
                let zp = mins[row] + signed_bias * scale;
                scales.push(scale);
                zeros.push(zp);
            }
        }

        let k_packed = inner_stride.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        const SIMD_MARGIN: usize = 16;
        let mut packed = vec![T::default(); packed_len + SIMD_MARGIN];

        for row in 0..m {
            let inv_scale = if scales[row] != 0.0 {
                1.0 / scales[row]
            } else {
                1.0
            };
            let zp = zeros[row];
            for word in 0..k_packed {
                let chunk_idx = row * k_packed + word;
                let mut arr = T::Array::default();
                let arr_ref = arr.as_mut();
                for i in 0..T::ITEMS {
                    let elem_idx = row * inner_stride + word * T::ITEMS + i;
                    if elem_idx < (row + 1) * inner_stride && elem_idx < numel {
                        arr_ref[i] = (data[elem_idx] - zp) * inv_scale;
                    }
                }
                packed[chunk_idx] = T::pack_from_f32(arr);
            }
        }

        PackedTensor {
            data: packed,
            shape: shape.to_vec(),
            scales,
            zeros,
            block_size: 1,
        }
    }

    /// Convert to block-major layout for SIMD-friendly access patterns.
    ///
    /// Block-major interleaves `block_size` consecutive rows' packed words so that
    /// all rows' words at the same K-position are stored contiguously. This enables
    /// SIMD kernels to process multiple output channels simultaneously, reusing
    /// activation data across the block.
    ///
    /// Layout (two zones):
    ///   1. Block-major zone for fully-aligned blocks (rows 0 .. m_aligned):
    ///      Within each block of `block_size` rows, words are interleaved as
    ///      `[row0_w0, row1_w0, ..., rowB_w0, row0_w1, row1_w1, ..., rowB_w1, ...]`
    ///   2. Row-major zone for remaining tail rows (m_aligned .. m):
    ///      Rows that don't fill a full block stay in standard per-row order.
    ///
    /// `block_size` must be a power of 2 and > 0.
    pub fn to_block_major(&self, block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        assert!(
            block_size.is_power_of_two(),
            "block_size must be a power of 2"
        );
        assert!(self.shape.len() >= 2, "Block-major requires 2D+ tensor");
        let m = self.shape[0];
        let inner: usize = self.shape[1..].iter().product();
        let k_packed = inner.div_ceil(T::ITEMS);
        let m_aligned = (m / block_size) * block_size;
        let tail_rows = m - m_aligned;
        let block_words = m_aligned * k_packed; // block-major zone size (in packed words)
        let tail_words = tail_rows * k_packed;
        let total_words = block_words + tail_words;

        // Allocate new buffer (preserve SIMD margin)
        let margin = self.data.len().saturating_sub(m * k_packed);
        let mut reordered = vec![T::default(); total_words + margin];

        // --- Zone 1: Block-major (full blocks) ---
        let full_blocks = m_aligned / block_size;
        for block in 0..full_blocks {
            let dst_base = block * block_size * k_packed;
            for local_row in 0..block_size {
                let src_row = block * block_size + local_row;
                let src_base = src_row * k_packed;
                for k in 0..k_packed {
                    let src_idx = src_base + k;
                    let dst_idx = dst_base + k * block_size + local_row;
                    reordered[dst_idx] = self.data[src_idx];
                }
            }
        }

        // --- Zone 2: Row-major tail (rows m_aligned .. m) ---
        for local_row in 0..tail_rows {
            let src_row = m_aligned + local_row;
            let src_base = src_row * k_packed;
            let dst_base = block_words + local_row * k_packed;
            reordered[dst_base..(dst_base + k_packed)]
                .copy_from_slice(&self.data[src_base..(src_base + k_packed)]);
        }

        // Copy remaining SIMD margin
        let old_margin_start = m * k_packed;
        for i in old_margin_start..self.data.len() {
            let dst = total_words + (i - old_margin_start);
            if dst < reordered.len() {
                reordered[dst] = self.data[i];
            }
        }

        PackedTensor {
            data: reordered,
            shape: self.shape.clone(),
            scales: self.scales.clone(),
            zeros: self.zeros.clone(),
            block_size,
        }
    }

    /// Returns the block size for block-major layout (1 = standard row-major).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns true if this tensor uses block-major layout.
    pub fn is_block_major(&self) -> bool {
        self.block_size > 1
    }
}

impl<T: PackedWord> Clone for PackedTensor<T> {
    fn clone(&self) -> Self {
        PackedTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            scales: self.scales.clone(),
            zeros: self.zeros.clone(),
            block_size: self.block_size,
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
    fn test_block_major_roundtrip_u8x4() {
        // 6 rows × 8 cols, U8x4 (4 values per word → k_packed = 2)
        let data: Vec<f32> = (0..48).map(|i| (i as f32) - 24.0).collect();
        let t_row = PackedTensor::<U8x4>::from_f32_auto(&data, &[6, 8]);
        let t_block = t_row.to_block_major(4);

        assert!(t_block.is_block_major());
        assert_eq!(t_block.block_size(), 4);
        assert_eq!(t_block.numel(), 48);
        assert_eq!(t_block.packed_len(), t_row.packed_len());

        // Values should match via get()
        for i in 0..48 {
            let orig = t_row.get(i);
            let got = t_block.get(i);
            let tol = (t_row.scale().abs().max(t_block.scale().abs())) + 0.01;
            assert!(
                (orig - got).abs() <= tol,
                "Mismatch at {}: row={} vs block={} (tol={})",
                i,
                orig,
                got,
                tol
            );
        }

        // to_f32_vec should match via slow path
        let vals = t_block.to_f32_vec();
        for (i, v) in vals.iter().enumerate() {
            let orig = t_row.get(i);
            let tol = (t_row.scale().abs().max(t_block.scale().abs())) + 0.01;
            assert!(
                (orig - v).abs() <= tol,
                "to_f32_vec mismatch at {}: {} vs {} (tol={})",
                i,
                orig,
                v,
                tol
            );
        }
    }

    #[test]
    fn test_block_major_f16x2() {
        // 8 rows × 4 cols, F16x2 (2 values per word → k_packed = 2)
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5).collect();
        let t_row = PackedTensor::<F16x2>::from_f32_auto(&data, &[8, 4]);
        let t_block = t_row.to_block_major(4);

        assert!(t_block.is_block_major());
        for i in 0..32 {
            let orig = t_row.get(i);
            let got = t_block.get(i);
            assert!(
                (orig - got).abs() < 0.05,
                "Mismatch at {}: {} vs {}",
                i,
                orig,
                got
            );
        }
    }

    #[test]
    fn test_block_major_non_multiple_rows() {
        // 5 rows × 8 cols (not a multiple of block_size=4)
        let data: Vec<f32> = (0..40).map(|i| (i as f32) - 20.0).collect();
        let t_row = PackedTensor::<U8x4>::from_f32_auto(&data, &[5, 8]);
        let t_block = t_row.to_block_major(4);

        assert_eq!(t_block.block_size(), 4);
        for i in 0..40 {
            let orig = t_row.get(i);
            let got = t_block.get(i);
            let tol = t_row.scale().abs() + 0.01;
            assert!(
                (orig - got).abs() <= tol,
                "Mismatch at {}: {} vs {} (tol={})",
                i,
                orig,
                got,
                tol
            );
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

    #[test]
    fn test_u4x8_small_k_roundtrip() {
        // U4x8 packs 8 × 4-bit values per u32 word. With K=4 (inner_dim < ITEMS=8),
        // each row fits in a single word with 4 valid elements + 4 padding positions.
        // This tests the edge case where word_index() must use elem_in_row % T::ITEMS
        // (not idx % T::ITEMS) to find the correct sub-word element position.
        let data: Vec<f32> = vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 128.0, 64.0];
        let t = PackedTensor::<U4x8>::from_f32_auto(&data, &[2, 4]);
        let tol = t.scale() + 0.01;

        // Verify get() for every element
        for i in 0..8 {
            let val = t.get(i);
            let expected = data[i];
            assert!(
                (val - expected).abs() <= tol,
                "get({}) mismatch: got={}, expected={}, tol={}",
                i,
                val,
                expected,
                tol
            );
        }

        // Verify set() roundtrip
        let mut t2 = PackedTensor::<U4x8>::from_f32_auto(&data, &[2, 4]);
        t2.set(1, 25.0);
        t2.set(5, 225.0);
        let expected_1 = 25.0f32;
        let expected_5 = 225.0f32;
        assert!(
            (t2.get(1) - expected_1).abs() <= tol,
            "set/get(1) mismatch: got={}, expected={}",
            t2.get(1),
            expected_1
        );
        assert!(
            (t2.get(5) - expected_5).abs() <= tol,
            "set/get(5) mismatch: got={}, expected={}",
            t2.get(5),
            expected_5
        );

        // Verify to_f32_vec matches (uses separate iteration logic)
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= tol,
                "to_f32_vec mismatch at {}: orig={}, rec={}",
                i,
                orig,
                rec
            );
        }
    }
}
