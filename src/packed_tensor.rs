use crate::dtypes::PackedWord;
use crate::types::{
    QuantizationGranularity, RepresentationTransform, ScalarType, StorageEncoding,
    ValueRepresentation, PACKED_SIMD_MARGIN_WORDS,
};
use std::sync::{Arc, OnceLock};

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

fn kmeans_16(values: &[f32], max_iter: usize) -> [f32; 16] {
    let mut centroids = [0.0f32; 16];
    if values.is_empty() {
        return centroids;
    }
    let n = values.len();
    if n == 1 {
        for c in centroids.iter_mut() {
            *c = values[0];
        }
        return centroids;
    }

    // Sort data for percentile initialization.
    // Percentile init places centroids where data is dense (unlike uniform [min, max]
    // which wastes centroids on empty tails). For bell-curve NN weights, this
    // naturally puts several centroids near zero where ~60% of values live.
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for i in 0..16 {
        let percentile = (i as f32 + 0.5) / 16.0;
        let idx = (percentile * n as f32) as usize;
        centroids[i] = sorted[idx.min(n - 1)];
    }

    // Lloyd-Max iterations (1D k-means)
    let mut assignments = vec![0usize; n];
    let mut counts = [0usize; 16];
    let mut sums = [0.0f32; 16];
    for _iter in 0..max_iter {
        for (vi, &v) in values.iter().enumerate() {
            let mut best_d = (v - centroids[0]).abs();
            let mut best_c = 0usize;
            for ci in 1..16 {
                let d = (v - centroids[ci]).abs();
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            assignments[vi] = best_c;
        }
        counts.fill(0);
        sums.fill(0.0);
        for (vi, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            sums[c] += values[vi];
        }
        for ci in 0..16 {
            if counts[ci] > 0 {
                centroids[ci] = sums[ci] / counts[ci] as f32;
            }
        }
    }
    centroids.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    centroids
}

fn nearest_codebook_index(v: f32, codebook: &[f32; 16]) -> usize {
    let mut best_d = (v - codebook[0]).abs();
    let mut best_i = 0usize;
    for (ci, &c) in codebook.iter().enumerate().skip(1) {
        let d = (v - c).abs();
        if d < best_d {
            best_d = d;
            best_i = ci;
        }
    }
    best_i
}

fn extract_nibble<T: PackedWord>(word: &T, elem: usize) -> usize {
    let word_u32: u32 = unsafe { std::mem::transmute_copy(word) };
    ((word_u32 >> (elem * 4)) & 0xF) as usize
}

pub struct PackedTensor<T: PackedWord> {
    pub(crate) data: Arc<Vec<T>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) scales: Vec<f32>,
    pub(crate) zeros: Vec<f32>,
    /// Block size for block-major layout (1 = standard row-major).
    /// When > 1, packed data is interleaved in blocks of `block_size` rows
    /// for SIMD-friendly access patterns: within each block, all rows' words
    /// at the same K-position are stored contiguously.
    pub(crate) block_size: usize,
    /// Per-group quantization granularity. 0 = per-tensor, 1 = per-channel,
    /// N > 1 = per-N-rows. When set, `scale_for_row(r)` maps to
    /// `scales[r / group_size]`.
    pub(crate) group_size: usize,
    /// Per-block quantization granularity within each row. 0 = not per-block
    /// (use per-channel/grouped lookup). When > 1, each row's elements are
    /// divided into blocks of this size, each with its own (scale, zero).
    /// Scale/zero layout: `scales[row * blocks_per_row + col / quant_block_size]`
    /// where `blocks_per_row = inner_dim.div_ceil(quant_block_size)`.
    pub(crate) quant_block_size: usize,
    /// Per-block codebook for INT4 codebook quantization.  Each block has 16
    /// f32 entries (normalized K-means centroids in [-1, 1]).
    /// Per-block scales stored in `self.scales` (one per block).
    /// Dequant: `scales[blk] * codebooks[0][nibble]` (global codebook, per-block scale).
    /// Replaces scales/zeros when present (they are ignored).
    /// Indexed as `codebooks[row * blocks_per_row + col / quant_block_size]`.
    pub(crate) codebooks: Vec<[f32; 16]>,
    /// Lazily-computed dequantized f32 weights. Populated on first access,
    /// then reused for all subsequent forward passes. Eliminates per-call
    /// unpack_weight_f32 overhead for packed float types (F8x4, F8x4R, F4x8).
    pub(crate) cached_f32_weights: OnceLock<Vec<f32>>,
}

impl<T: PackedWord> PackedTensor<T> {
    /// Canonical quantization granularity owned by this runtime tensor descriptor.
    pub fn quantization_granularity(&self) -> QuantizationGranularity {
        if self.quant_block_size > 0 {
            QuantizationGranularity::PerGroup {
                axis: self.shape.len().saturating_sub(1),
                group_size: self.quant_block_size,
            }
        } else if self.group_size > 1 {
            QuantizationGranularity::PerGroup {
                axis: 0,
                group_size: self.group_size,
            }
        } else if self.scales.len() > 1 {
            QuantizationGranularity::PerAxis { axis: 0 }
        } else {
            QuantizationGranularity::PerTensor
        }
    }

    /// Resolve the runtime-owned scale, offset, codebook, and granularity metadata.
    pub fn representation_transform(&self) -> RepresentationTransform {
        let granularity = self.quantization_granularity();
        if !self.codebooks.is_empty() {
            return RepresentationTransform::Codebook {
                granularity,
                entries: self
                    .codebooks
                    .iter()
                    .map(|codebook| codebook.to_vec())
                    .collect(),
                scales: self.scales.clone(),
                offsets: self.zeros.clone(),
            };
        }
        if T::IS_FLOAT {
            RepresentationTransform::ScaledAffine {
                granularity,
                scales: self.scales.clone(),
                offsets: self.zeros.clone(),
            }
        } else {
            RepresentationTransform::AffineDequantization {
                granularity,
                scales: self.scales.clone(),
                offsets: self.zeros.clone(),
            }
        }
    }

    /// Full canonical representation resolved from the word format and runtime metadata.
    pub fn value_representation(&self) -> ValueRepresentation {
        ValueRepresentation {
            logical: ScalarType::F32,
            storage: T::SCALAR_TYPE,
            encoding: StorageEncoding::Packed {
                word_bits: (std::mem::size_of::<T>() * 8) as u8,
                lanes: T::ITEMS as u8,
            },
            transform: self.representation_transform(),
        }
    }

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
            data: Arc::new(data),
            shape: shape.to_vec(),
            scales: vec![1.0],
            zeros: vec![0.0],
            block_size: 1,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    pub fn from_raw(data: Vec<T>, shape: Vec<usize>, scales: Vec<f32>, zeros: Vec<f32>) -> Self {
        PackedTensor {
            data: Arc::new(data),
            shape,
            scales,
            zeros,
            block_size: 1,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    pub fn from_raw_arc(
        data: Arc<Vec<T>>,
        shape: Vec<usize>,
        scales: Vec<f32>,
        zeros: Vec<f32>,
    ) -> Self {
        PackedTensor {
            data,
            shape,
            scales,
            zeros,
            block_size: 1,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
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
            data: Arc::new(data),
            shape,
            scales,
            zeros,
            block_size,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
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
        // Safety margin for SIMD kernels that may read beyond the logical word boundary
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

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
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales: vec![scale],
            zeros: vec![zero],
            block_size: 1,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
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

        if !self.codebooks.is_empty() {
            return self.to_f32_vec_codebook();
        }

        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            numel
        };
        let items = T::ITEMS;
        let mut result = Vec::with_capacity(numel);
        let is_per_row = self.scales.len() > 1;
        let per_block = self.quant_block_size > 0;

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
            let row_end = (row + 1) * inner_stride;

            if per_block {
                for j in 0..items {
                    let elem_idx = elem_idx_base + j;
                    if elem_idx < numel && elem_idx < row_end {
                        let col = elem_idx % inner_stride;
                        let s = self.scale_for_elem(row, col);
                        let z = self.zero_for_elem(row, col);
                        result.push(arr[j] * s + z);
                    }
                }
            } else {
                let s = if is_per_row {
                    if self.group_size > 1 {
                        self.scales[row / self.group_size]
                    } else {
                        self.scales[row]
                    }
                } else {
                    self.scales[0]
                };
                let z = if is_per_row {
                    if self.group_size > 1 {
                        self.zeros[row / self.group_size]
                    } else {
                        self.zeros[row]
                    }
                } else {
                    self.zeros[0]
                };
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
        }
        result
    }

    fn to_f32_vec_codebook(&self) -> Vec<f32> {
        let numel = self.numel();
        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            numel
        };
        let items = T::ITEMS;
        let k_packed = inner_stride.div_ceil(items);
        let m = if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            1
        };
        let actual_words = m * k_packed;
        let bpr = self.blocks_per_row();
        let qb = self.quant_block_size;
        let mut result = Vec::with_capacity(numel);

        for (i, word) in self.data[..actual_words].iter().enumerate() {
            let row = i / k_packed;
            let word_in_row = i % k_packed;
            let elem_idx_base = row * inner_stride + word_in_row * items;
            let row_end = (row + 1) * inner_stride;

            for j in 0..items {
                let elem_idx = elem_idx_base + j;
                if elem_idx < numel && elem_idx < row_end {
                    let col = elem_idx % inner_stride;
                    let blk = row * bpr + col / qb;
                    let nibble = extract_nibble(word, j);
                    result.push(self.scales[blk] * self.codebooks[0][nibble]);
                }
            }
        }
        result
    }

    #[inline]
    pub fn packed_len(&self) -> usize {
        self.data.len()
    }

    /// Validate the storage and row-wise affine metadata required by packed
    /// matrix kernels. This intentionally does not infer or repair malformed
    /// caller-provided tensors.
    pub(crate) fn validate_matrix_storage(&self) -> Result<(), String> {
        if self.shape.len() != 2 {
            return Err(format!(
                "packed matrix must have rank 2, got rank {}",
                self.shape.len()
            ));
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let words_per_row = columns.div_ceil(T::ITEMS);
        let expected_words = rows
            .checked_mul(words_per_row)
            .ok_or_else(|| "packed matrix storage size overflows usize".to_string())?;
        if self.data.len() < expected_words {
            return Err(format!(
                "packed matrix storage has {} words, requires at least {expected_words}",
                self.data.len()
            ));
        }
        let required_metadata = if self.group_size > 1 {
            rows.div_ceil(self.group_size)
        } else {
            rows
        };
        for (name, values) in [("scales", &self.scales), ("zero points", &self.zeros)] {
            if values.len() > 1 && values.len() < required_metadata {
                return Err(format!(
                    "packed matrix has {} {name}, requires at least {required_metadata}",
                    values.len()
                ));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!("packed matrix {name} must be finite"));
            }
        }
        Ok(())
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
        if self.scales.is_empty() {
            1.0
        } else {
            self.scales[0]
        }
    }

    #[inline]
    pub fn zero(&self) -> f32 {
        if self.zeros.is_empty() {
            0.0
        } else {
            self.zeros[0]
        }
    }

    #[inline]
    pub fn scale_for_row(&self, row: usize) -> f32 {
        if self.scales.is_empty() {
            1.0
        } else if self.scales.len() == 1 {
            self.scales[0]
        } else if self.group_size > 1 {
            self.scales[row / self.group_size]
        } else {
            self.scales[row]
        }
    }

    #[inline]
    pub fn zero_for_row(&self, row: usize) -> f32 {
        if self.zeros.is_empty() {
            0.0
        } else if self.zeros.len() == 1 {
            self.zeros[0]
        } else if self.group_size > 1 {
            self.zeros[row / self.group_size]
        } else {
            self.zeros[row]
        }
    }

    #[inline]
    pub fn is_per_channel(&self) -> bool {
        self.scales.len() > 1
    }

    pub fn get_or_init_f32_weights(&self) -> &[f32] {
        self.cached_f32_weights.get_or_init(|| {
            let rows = self.shape[0];
            let inner: usize = self.shape[1..].iter().product();
            let k_packed = inner.div_ceil(T::ITEMS);

            // Codebook dequant path
            if !self.codebooks.is_empty() {
                let bpr = self.blocks_per_row();
                let qb = self.quant_block_size;
                let mut buf = vec![0.0f32; rows * inner];
                for row in 0..rows {
                    let w_row = &self.as_packed()[row * k_packed..(row + 1) * k_packed];
                    let out_row = &mut buf[row * inner..(row + 1) * inner];
                    let mut idx = 0;
                    for word in w_row {
                        for i in 0..T::ITEMS {
                            if idx < inner {
                                let blk = row * bpr + idx / qb;
                                let nibble = extract_nibble(word, i);
                                out_row[idx] = self.scales[blk] * self.codebooks[0][nibble];
                                idx += 1;
                            }
                        }
                    }
                }
                return buf;
            }

            let per_elem = self.quant_block_size > 0;
            let mut buf = vec![0.0f32; rows * inner];
            for row in 0..rows {
                let w_row = &self.as_packed()[row * k_packed..(row + 1) * k_packed];
                let out_row = &mut buf[row * inner..(row + 1) * inner];
                let mut idx = 0;
                for kk in 0..k_packed {
                    let unpacked = w_row[kk].unpack_to_f32();
                    for i in 0..T::ITEMS {
                        if idx < inner {
                            let (w_scale, w_zero) = if per_elem {
                                (self.scale_for_elem(row, idx), self.zero_for_elem(row, idx))
                            } else {
                                (self.scale_for_row(row), self.zero_for_row(row))
                            };
                            out_row[idx] = unpacked.as_ref()[i] * w_scale + w_zero;
                            idx += 1;
                        }
                    }
                }
            }
            buf
        })
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
        let inner_stride = if self.shape.len() >= 2 {
            self.shape[1..].iter().product()
        } else {
            self.numel()
        };
        let row = idx / inner_stride;
        let col = idx % inner_stride;

        if !self.codebooks.is_empty() {
            let blk = row * self.blocks_per_row() + col / self.quant_block_size;
            let nibble = extract_nibble(&self.data[word_idx], elem_idx);
            return self.scales[blk] * self.codebooks[0][nibble];
        }

        let word = self.data[word_idx].unpack_to_f32();
        let val = word.as_ref()[elem_idx];
        let s = self.scale_for_elem(row, col);
        let z = self.zero_for_elem(row, col);
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
        let col = idx % inner_stride;
        let s = self.scale_for_elem(row, col);
        let z = self.zero_for_elem(row, col);
        let quantized = if s != 1.0 || z != 0.0 {
            (val - z) / s
        } else {
            val
        };
        arr.as_mut()[elem_idx] = quantized;
        let data_mut = Arc::make_mut(&mut self.data);
        data_mut[word_idx] = T::pack_from_f32(arr);
    }

    #[inline]
    pub fn as_packed(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn as_packed_mut(&mut self) -> &mut [T] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.data.as_slice())
    }

    #[inline]
    pub fn as_u32(&self) -> &[u32] {
        bytemuck::cast_slice(self.data.as_slice())
    }

    pub fn compute_scale(data: &[f32]) -> f32 {
        if T::IS_FLOAT && T::BIT_WIDTH >= 16 {
            return 1.0;
        }
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_val = T::MAX_REPRESENTABLE;
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
        if T::IS_FLOAT && T::BIT_WIDTH >= 16 {
            return vec![1.0];
        }
        assert!(shape.len() >= 2, "Per-channel requires 2D+ shape");
        let m = shape[0];
        let inner_stride: usize = shape[1..].iter().product();
        let max_val = T::MAX_REPRESENTABLE;
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
        // Safety margin for SIMD kernels that may read beyond the logical word boundary
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

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
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales,
            zeros: vec![0.0; m],
            block_size: 1,
            group_size: 0,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    /// Same as `from_f32_per_channel` but with per-group quantization.
    /// `group_size` divides the output channels (rows) into groups of this size.
    /// Each group shares one scale. Use 0 or 1 for per-channel.
    pub fn from_f32_per_channel_grouped(data: &[f32], shape: &[usize], group_size: usize) -> Self {
        if group_size <= 1 {
            return Self::from_f32_per_channel(data, shape);
        }
        Self::from_f32_per_channel_grouped_impl(data, shape, group_size)
    }

    fn from_f32_per_channel_grouped_impl(data: &[f32], shape: &[usize], group_size: usize) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);

        let m = if shape.len() >= 2 { shape[0] } else { 1 };
        let inner_stride = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            numel
        };

        let mut row_max_abs = vec![0.0f32; m];
        for row in 0..m {
            let start = row * inner_stride;
            let end = (start + inner_stride).min(numel);
            let max_abs = data[start..end]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            row_max_abs[row] = max_abs;
        }

        let num_groups = m.div_ceil(group_size);
        let max_val = T::MAX_REPRESENTABLE;
        let mut scales = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let start_row = g * group_size;
            let end_row = (start_row + group_size).min(m);
            let grp_max_abs = row_max_abs[start_row..end_row]
                .iter()
                .copied()
                .fold(0.0f32, f32::max);
            scales.push(if grp_max_abs == 0.0 {
                1.0
            } else {
                grp_max_abs / max_val
            });
        }

        let k_packed = inner_stride.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        // Safety margin for SIMD kernels that may read beyond the logical word boundary
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

        for row in 0..m {
            let gi = row / group_size;
            let row_scale = scales[gi];
            let inv_s = if row_scale != 0.0 {
                1.0 / row_scale
            } else {
                0.0
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
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales,
            zeros: vec![0.0; m],
            block_size: 1,
            group_size,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    pub fn from_f32_per_channel_asymmetric(data: &[f32], shape: &[usize]) -> Self {
        Self::from_f32_per_channel_asymmetric_impl(data, shape, 0)
    }

    /// Same as `from_f32_per_channel_asymmetric` but with per-group quantization.
    /// `group_size` divides the output channels (rows) into groups of this size.
    /// Each group shares one (scale, zero_point). Use 0 for per-channel.
    pub fn from_f32_per_channel_asymmetric_grouped(
        data: &[f32],
        shape: &[usize],
        group_size: usize,
    ) -> Self {
        if group_size == 0 || group_size == 1 {
            return Self::from_f32_per_channel_asymmetric(data, shape);
        }
        Self::from_f32_per_channel_asymmetric_impl(data, shape, group_size)
    }

    fn from_f32_per_channel_asymmetric_impl(
        data: &[f32],
        shape: &[usize],
        group_size: usize,
    ) -> Self {
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

        let unsigned_max = ((1u32 << T::BIT_WIDTH) - 1) as f32;
        let signed_bias = (1u32 << (T::BIT_WIDTH - 1)) as f32;
        let is_fp4 = T::IS_FLOAT && T::MAX_REPRESENTABLE < signed_bias;

        let num_groups = if group_size > 1 {
            m.div_ceil(group_size)
        } else {
            m
        };
        let mut scales = Vec::with_capacity(num_groups);
        let mut zeros = Vec::with_capacity(num_groups);
        if group_size > 1 {
            for g in 0..num_groups {
                let start_row = g * group_size;
                let end_row = (start_row + group_size).min(m);
                let mut grp_min = f32::MAX;
                let mut grp_max = f32::MIN;
                for row in start_row..end_row {
                    grp_min = grp_min.min(mins[row]);
                    grp_max = grp_max.max(maxs[row]);
                }
                let range = grp_max - grp_min;
                if range == 0.0 || unsigned_max == 0.0 {
                    scales.push(1.0);
                    zeros.push(0.0);
                } else if is_fp4 {
                    let effective_magnitude = 4.0;
                    // Mean over all rows in the group — see per-block rationale above.
                    let mut sum = 0.0f32;
                    let mut cnt: usize = 0;
                    for row in start_row..end_row {
                        for i in 0..inner_stride {
                            let idx = row * inner_stride + i;
                            if idx < numel {
                                sum += data[idx];
                                cnt += 1;
                            }
                        }
                    }
                    let zp = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                    // Use max(|data - zp|) as spread, not range/2 — for skewed
                    // distributions the mean (zero-point) is not centered in [bmin,bmax],
                    // so range/2 overestimates how much code-space headroom is needed
                    // on the distant side.  max_dev guarantees all values fit within
                    // FP4's active [-4, +4] code range after quantization.
                    let dev_from_zp = (zp - grp_min).abs().max((grp_max - zp).abs());
                    let max_dev = if dev_from_zp == 0.0 {
                        1e-10
                    } else {
                        dev_from_zp
                    };
                    let scale = max_dev / effective_magnitude;
                    scales.push(scale);
                    zeros.push(zp);
                } else {
                    let init_scale = range / unsigned_max;
                    let lo = grp_min - init_scale;
                    let hi = grp_max + init_scale;
                    let scale = (hi - lo) / unsigned_max;
                    let zp = lo + signed_bias * scale;
                    scales.push(scale);
                    zeros.push(zp);
                }
            }
        } else {
            for row in 0..m {
                let range = maxs[row] - mins[row];
                if range == 0.0 || unsigned_max == 0.0 {
                    scales.push(1.0);
                    zeros.push(0.0);
                } else if is_fp4 {
                    let effective_magnitude = 4.0;
                    let row_start = row * inner_stride;
                    let row_end = row_start + inner_stride;
                    let mean = data[row_start..row_end.min(numel)].iter().sum::<f32>()
                        / inner_stride as f32;
                    // max_dev guarantees all values fit within FP4 [-4,+4] after quant
                    let dev_from_zp = (mean - mins[row]).abs().max((maxs[row] - mean).abs());
                    let max_dev = if dev_from_zp == 0.0 {
                        1e-10
                    } else {
                        dev_from_zp
                    };
                    let scale = max_dev / effective_magnitude;
                    scales.push(scale);
                    zeros.push(mean);
                } else {
                    let init_scale = range / unsigned_max;
                    let lo = mins[row] - init_scale;
                    let hi = maxs[row] + init_scale;
                    let scale = (hi - lo) / unsigned_max;
                    let zp = lo + signed_bias * scale;
                    scales.push(scale);
                    zeros.push(zp);
                }
            }
        }

        let k_packed = inner_stride.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        // Safety margin for SIMD kernels that may read beyond the logical word boundary
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

        for row in 0..m {
            let (_scale_idx, inv_scale, zp) = if group_size > 1 {
                let gi = row / group_size;
                let s = scales[gi];
                (gi, if s != 0.0 { 1.0 / s } else { 0.0 }, zeros[gi])
            } else {
                let s = scales[row];
                (row, if s != 0.0 { 1.0 / s } else { 0.0 }, zeros[row])
            };
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

        // Direct absolute error threshold on the block: verify that every
        // value dequantizes with error ≤ its group's quantization step.
        #[cfg(debug_assertions)]
        {
            for row in 0..m {
                let (s, zp) = if group_size > 1 {
                    let gi = row / group_size;
                    (scales[gi], zeros[gi])
                } else {
                    (scales[row], zeros[row])
                };
                let k_packed_deq = inner_stride.div_ceil(T::ITEMS);
                for word in 0..k_packed_deq {
                    let unpacked = packed[row * k_packed_deq + word].unpack_to_f32();
                    let arr = unpacked.as_ref();
                    for i in 0..T::ITEMS {
                        let elem_idx = row * inner_stride + word * T::ITEMS + i;
                        if elem_idx < (row + 1) * inner_stride && elem_idx < numel {
                            let orig = data[elem_idx];
                            let deq = arr[i] * s + zp;
                            let err = (orig - deq).abs();
                            assert!(
                                err <= s,
                                "quantization error {err} exceeds step size {s} for row {row} (1-bit guard failed)"
                            );
                        }
                    }
                }
            }
        }

        PackedTensor {
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales,
            zeros,
            block_size: 1,
            group_size,
            quant_block_size: 0,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    /// Per-block asymmetric quantization: each contiguous block of `qblock`
    /// elements within every row gets its own (scale, zero_point).
    /// Scales/zeros layout: `M * ceil(K / qblock)` entries, indexed as
    /// `scales[row * blocks_per_row + col / qblock]`.
    pub fn from_f32_per_block_asymmetric(data: &[f32], shape: &[usize], qblock: usize) -> Self {
        assert!(qblock > 0, "quantization block size must be > 0");
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);

        let m = if shape.len() >= 2 { shape[0] } else { 1 };
        let inner = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            numel
        };
        assert!(
            inner % qblock == 0,
            "inner dimension {inner} must be divisible by quantization block size {qblock}"
        );

        let blocks_per_row = inner / qblock;
        let unsigned_max = ((1u32 << T::BIT_WIDTH) - 1) as f32;
        let signed_bias = (1u32 << (T::BIT_WIDTH - 1)) as f32;
        let num_blocks = m * blocks_per_row;

        let mut scales = Vec::with_capacity(num_blocks);
        let mut zeros = Vec::with_capacity(num_blocks);

        for row in 0..m {
            for blk in 0..blocks_per_row {
                let start = row * inner + blk * qblock;
                let end = start + qblock;
                let block_data = &data[start..end];
                let mut bmin = f32::MAX;
                let mut bmax = f32::MIN;
                for &v in block_data {
                    bmin = bmin.min(v);
                    bmax = bmax.max(v);
                }
                let range = bmax - bmin;
                if range == 0.0 || unsigned_max == 0.0 {
                    scales.push(1.0);
                    zeros.push(0.0);
                } else if T::IS_FLOAT && T::MAX_REPRESENTABLE < signed_bias {
                    // FP4 non-uniform types: use non-linear clipping to avoid the
                    // worst FP4 magnitude gap (4.0→6.0 = 2.0). We clamp the effective
                    // code range to ±4.0 instead of ±6.0, sacrificing the two most
                    // extreme magnitude entries (±6.0) for uniformly smaller gaps.
                    //   Full FP4 magnitudes: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
                    //   Active after clip:  [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
                    //   Max gap: 1.0 (3→4) vs 2.0 (4→6) without clipping
                    //   Max abs error: range/16 vs range/12 (25% improvement)
                    //   Exact reconstruction for bmin/bmap at ±4.0 codes.
                    //
                    // Use mean as zero-point instead of midpoint — for skewed
                    // weight distributions the mean better centers the FP4
                    // non-uniform code space where most values actually lie,
                    // reducing average quantization error.
                    let effective_magnitude = 4.0; // clip to ±4, drop ±6
                    let zp = block_data.iter().sum::<f32>() / qblock as f32;
                    let dev_from_zp = (zp - bmin).abs().max((bmax - zp).abs());
                    let max_dev = if dev_from_zp == 0.0 {
                        1e-10
                    } else {
                        dev_from_zp
                    };
                    let scale = max_dev / effective_magnitude;
                    scales.push(scale);
                    zeros.push(zp);
                } else {
                    // Integer types (I4, I8): uniform code space [0, 2^bits - 1]
                    // with guard padding to prevent clipping at boundaries.
                    let init_scale = range / unsigned_max;
                    let lo = bmin - init_scale;
                    let hi = bmax + init_scale;
                    let scale = (hi - lo) / unsigned_max;
                    let zp = lo + signed_bias * scale;
                    scales.push(scale);
                    zeros.push(zp);
                }
            }
        }

        let k_packed = inner.div_ceil(T::ITEMS);
        let packed_len = m * k_packed;
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

        for row in 0..m {
            for blk in 0..blocks_per_row {
                let si = row * blocks_per_row + blk;
                let s = scales[si];
                let z = zeros[si];
                let inv_s = if s != 0.0 { 1.0 / s } else { 0.0 };
                let blk_start = blk * qblock;
                let blk_end = blk_start + qblock;

                for word_off in 0..(qblock.div_ceil(T::ITEMS)) {
                    let global_word = blk_start / T::ITEMS + word_off;
                    let mut arr = T::Array::default();
                    let arr_ref = arr.as_mut();
                    for i in 0..T::ITEMS {
                        let elem_in_blk = word_off * T::ITEMS + i;
                        let elem_in_row = blk_start + elem_in_blk;
                        if elem_in_row < blk_end && elem_in_row < inner {
                            let flat_idx = row * inner + elem_in_row;
                            if flat_idx < numel {
                                arr_ref[i] = (data[flat_idx] - z) * inv_s;
                            }
                        }
                    }
                    let chunk_idx = row * k_packed + global_word;
                    if chunk_idx < packed.len() {
                        packed[chunk_idx] = T::pack_from_f32(arr);
                    }
                }
            }
        }

        #[cfg(debug_assertions)]
        {
            for row in 0..m {
                for blk in 0..blocks_per_row {
                    let si = row * blocks_per_row + blk;
                    let (s, z) = (scales[si], zeros[si]);
                    let blk_start = blk * qblock;
                    let blk_end = blk_start + qblock;
                    for elem_in_row in blk_start..blk_end {
                        let flat_idx = row * inner + elem_in_row;
                        if flat_idx < numel {
                            let global_word = elem_in_row / T::ITEMS;
                            let sub_idx = elem_in_row % T::ITEMS;
                            let unpacked = packed[row * k_packed + global_word].unpack_to_f32();
                            let deq = unpacked.as_ref()[sub_idx] * s + z;
                            let err = (data[flat_idx] - deq).abs();
                            assert!(
                                err <= s,
                                "per-block quant error {err} exceeds step {s} at row={row} blk={blk} elem={elem_in_row}"
                            );
                        }
                    }
                }
            }
        }

        PackedTensor {
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales,
            zeros,
            block_size: 1,
            group_size: 0,
            quant_block_size: qblock,
            codebooks: vec![],
            cached_f32_weights: OnceLock::new(),
        }
    }

    pub fn from_f32_per_block_codebook(data: &[f32], shape: &[usize], qblock: usize) -> Self {
        assert!(qblock > 0, "quantization block size must be > 0");
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel);

        let m = if shape.len() >= 2 { shape[0] } else { 1 };
        let inner = if shape.len() >= 2 {
            shape[1..].iter().product()
        } else {
            numel
        };

        // Handle non-divisible inner dims: last block is partial (fewer than qblock elements).
        let blocks_per_row = inner.div_ceil(qblock);
        let num_blocks = m * blocks_per_row;
        let items = T::ITEMS;

        // Step 1: Compute per-block scales and collect normalized values.
        // Normalizing each block by its own max_abs separates the problem:
        // codebook captures distribution SHAPE (where values fall relative to block peak),
        // scale captures MAGNITUDE (the block's absolute range).
        let mut block_scales = Vec::with_capacity(num_blocks);
        let mut normalized_values = Vec::with_capacity(numel);
        for blk in 0..num_blocks {
            let row = blk / blocks_per_row;
            let blk_in_row = blk % blocks_per_row;
            let base = row * inner + blk_in_row * qblock;
            let blk_end = (base + qblock).min((row + 1) * inner);
            let block_len = blk_end - base;
            let max_abs_block = if block_len > 0 {
                data[base..blk_end]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
            } else {
                0.0
            };
            block_scales.push(max_abs_block);
            if max_abs_block > 0.0 {
                let inv = 1.0 / max_abs_block;
                for &v in data[base..blk_end].iter() {
                    normalized_values.push(v * inv);
                }
            } else {
                normalized_values.extend(std::iter::repeat_n(0.0, block_len));
            }
        }

        // Step 2: Compute codebook on normalized values.
        // The centroids are in [-1, 1] range but are data-adaptive (learned from actual
        // normalized weight distribution), not a fixed Gaussian.
        let codebook = kmeans_16(&normalized_values, 30);

        // Step 3: Packing — nibble = nearest(v / blk_scale, codebook).
        let k_packed = inner.div_ceil(items);
        let packed_len = m * k_packed;
        let mut packed = vec![T::default(); packed_len + PACKED_SIMD_MARGIN_WORDS];

        for row in 0..m {
            for blk in 0..blocks_per_row {
                let blk_idx = row * blocks_per_row + blk;
                let blk_scale = block_scales[blk_idx];
                let inv_blk_scale = if blk_scale > 0.0 {
                    1.0 / blk_scale
                } else {
                    0.0
                };
                let base = row * inner + blk * qblock;
                let blk_end = (base + qblock).min((row + 1) * inner);
                let block_len = blk_end - base;
                for elem_in_blk in (0..block_len).step_by(items) {
                    let word_in_row = (blk * qblock + elem_in_blk) / items;
                    let chunk_idx = row * k_packed + word_in_row;
                    let mut w = 0u32;
                    for i in 0..items {
                        let elem_idx = base + elem_in_blk + i;
                        if elem_idx >= blk_end || elem_idx >= numel {
                            w |= (0u32) << (i * 4);
                        } else {
                            let v_normalized = data[elem_idx] * inv_blk_scale;
                            let idx = nearest_codebook_index(v_normalized, &codebook);
                            w |= (idx as u32) << (i * 4);
                        }
                    }
                    packed[chunk_idx] = T::default();
                    let word_ref: &mut T = &mut packed[chunk_idx];
                    let word_u32: &mut u32 = unsafe { std::mem::transmute(word_ref) };
                    *word_u32 = w;
                }
            }
        }

        PackedTensor {
            data: Arc::new(packed),
            shape: shape.to_vec(),
            scales: block_scales,
            zeros: vec![],
            block_size: 1,
            group_size: 0,
            quant_block_size: qblock,
            codebooks: vec![codebook],
            cached_f32_weights: OnceLock::new(),
        }
    }

    #[inline]
    pub fn blocks_per_row(&self) -> usize {
        if self.quant_block_size == 0 {
            0
        } else {
            let inner: usize = if self.shape.len() >= 2 {
                self.shape[1..].iter().product()
            } else {
                self.numel()
            };
            inner.div_ceil(self.quant_block_size)
        }
    }

    #[inline]
    pub fn scale_for_elem(&self, row: usize, col: usize) -> f32 {
        if let Some(block_idx) = col.checked_div(self.quant_block_size) {
            let bpr = self.blocks_per_row();
            self.scales[row * bpr + block_idx]
        } else {
            self.scale_for_row(row)
        }
    }

    #[inline]
    pub fn zero_for_elem(&self, row: usize, col: usize) -> f32 {
        if let Some(block_idx) = col.checked_div(self.quant_block_size) {
            let bpr = self.blocks_per_row();
            self.zeros[row * bpr + block_idx]
        } else {
            self.zero_for_row(row)
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
            data: Arc::new(reordered),
            shape: self.shape.clone(),
            scales: self.scales.clone(),
            zeros: self.zeros.clone(),
            codebooks: self.codebooks.clone(),
            block_size,
            group_size: self.group_size,
            quant_block_size: self.quant_block_size,
            cached_f32_weights: OnceLock::new(),
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
            codebooks: self.codebooks.clone(),
            block_size: self.block_size,
            group_size: self.group_size,
            quant_block_size: self.quant_block_size,
            cached_f32_weights: OnceLock::new(),
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
    use crate::dtypes::{F4x8, F8x4, F8x4R, I4x8, I8x4};
    use crate::types::{ScalarType, StorageEncoding, TensorStorageLayout};

    #[test]
    fn packed_tensor_capacity_matches_canonical_layout() {
        let shape = [2, 9];
        let tensor = PackedTensor::<I4x8>::from_f32_slice(&[0.0; 18], &shape, 1.0, 0.0);
        let layout = TensorStorageLayout {
            encoding: StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
        };
        assert_eq!(
            tensor.data.len() * std::mem::size_of::<I4x8>(),
            layout.allocation_bytes(ScalarType::I4, &shape).unwrap()
        );
    }

    #[test]
    fn packed_tensor_resolves_runtime_quantization_metadata() {
        let data = [0.0; 32];
        let per_tensor = PackedTensor::<I4x8>::from_f32_slice(&data, &[4, 8], 1.0, 0.0);
        assert_eq!(
            per_tensor.quantization_granularity(),
            QuantizationGranularity::PerTensor
        );
        assert!(matches!(
            per_tensor.representation_transform(),
            RepresentationTransform::AffineDequantization {
                granularity: QuantizationGranularity::PerTensor,
                ..
            }
        ));
        let representation = per_tensor.value_representation();
        assert_eq!(representation.storage, ScalarType::I4);
        assert_eq!(
            representation.encoding,
            StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            }
        );
        representation.validate().unwrap();

        let per_channel = PackedTensor::<I4x8>::from_f32_per_channel(&data, &[4, 8]);
        assert_eq!(
            per_channel.quantization_granularity(),
            QuantizationGranularity::PerAxis { axis: 0 }
        );

        let per_block = PackedTensor::<I4x8>::from_f32_per_block_asymmetric(&data, &[4, 8], 4);
        assert_eq!(
            per_block.quantization_granularity(),
            QuantizationGranularity::PerGroup {
                axis: 1,
                group_size: 4,
            }
        );
    }

    #[test]
    fn test_packed_tensor_zeros() {
        let t = PackedTensor::<I4x8>::zeros(&[16]);
        assert_eq!(t.numel(), 16);
        assert_eq!(t.packed_len(), 2);
        let vals = t.to_f32_vec();
        assert!(vals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_packed_tensor_roundtrip_i4x8() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32) - 8.0).collect();
        let t = PackedTensor::<I4x8>::from_f32_auto(&data, &[16]);
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
    fn test_packed_tensor_roundtrip_i8x4() {
        let data: Vec<f32> = vec![0.0, 50.0, -50.0, 100.0];
        let t = PackedTensor::<I8x4>::from_f32_auto(&data, &[4]);
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
        let data: Vec<f32> = vec![std::f32::consts::PI, -2.71, 1.41];
        let t = PackedTensor::<F32x1>::from_f32_auto(&data, &[3]);
        let recovered = t.to_f32_vec();
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert_eq!(orig, rec);
        }
    }

    #[test]
    fn test_block_major_roundtrip_i8x4() {
        // 6 rows × 8 cols, I8x4 (4 values per word → k_packed = 2)
        let data: Vec<f32> = (0..48).map(|i| (i as f32) - 24.0).collect();
        let t_row = PackedTensor::<I8x4>::from_f32_auto(&data, &[6, 8]);
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
        let t_row = PackedTensor::<I8x4>::from_f32_auto(&data, &[5, 8]);
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
        let mut t = PackedTensor::<I8x4>::from_f32_auto(&data, &[4]);
        let val = t.get(2);
        assert!((val - 3.0).abs() <= t.scale() + 0.01);
        t.set(2, 3.5);
        let val2 = t.get(2);
        assert!((val2 - 3.5).abs() <= t.scale() + 0.01);
    }

    #[test]
    fn test_i4x8_small_k_roundtrip() {
        // I4x8 packs 8 × 4-bit values per u32 word. With K=4 (inner_dim < ITEMS=8),
        // each row fits in a single word with 4 valid elements + 4 padding positions.
        // This tests the edge case where word_index() must use elem_in_row % T::ITEMS
        // (not idx % T::ITEMS) to find the correct sub-word element position.
        let data: Vec<f32> = vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 128.0, 64.0];
        let t = PackedTensor::<I4x8>::from_f32_auto(&data, &[2, 4]);
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
        let mut t2 = PackedTensor::<I4x8>::from_f32_auto(&data, &[2, 4]);
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

    #[test]
    fn test_packed_tensor_f8x4_roundtrip() {
        let data: Vec<f32> = vec![1.0, -1.0, 128.0, -128.0];
        let t = PackedTensor::<F8x4>::from_f32_auto(&data, &[4]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 0.5 * orig.abs().max(1.0),
                "F8x4 mismatch at {}: orig={}, rec={}, err={}",
                i,
                orig,
                rec,
                err
            );
        }
    }

    #[test]
    fn test_packed_tensor_f8x4r_roundtrip() {
        let data: Vec<f32> = vec![1.0, -1.0, 256.0, -256.0];
        let t = PackedTensor::<F8x4R>::from_f32_auto(&data, &[4]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 0.5 * orig.abs().max(1.0),
                "F8x4R mismatch at {}: orig={}, rec={}, err={}",
                i,
                orig,
                rec,
                err
            );
        }
    }

    #[test]
    fn test_packed_tensor_f4x8_roundtrip() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 6.0];
        let t = PackedTensor::<F4x8>::from_f32_auto(&data, &[8]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            assert!(
                err < 0.5 * orig.abs().max(1.0),
                "F4x8 mismatch at {}: orig={}, rec={}, err={}",
                i,
                orig,
                rec,
                err
            );
        }
    }

    #[test]
    fn test_packed_tensor_f4x8_single_word() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 4.0, -0.0, -1.0, -2.0, -4.0];
        let t = PackedTensor::<F4x8>::from_f32_auto(&data, &[8]);
        assert_eq!(
            t.packed_len(),
            17,
            "F4x8 packs 8 values into 1 u32 + 16 SIMD margin"
        );
        let recovered = t.to_f32_vec();
        for i in 0..8 {
            let err = (data[i] - recovered[i]).abs();
            assert!(
                err < 0.5 * data[i].abs().max(1.0),
                "F4x8 single_word mismatch at {}: orig={}, rec={}",
                i,
                data[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_packed_tensor_f8x4_zeros() {
        let t = PackedTensor::<F8x4>::zeros(&[4]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.packed_len(), 1);
        let vals = t.to_f32_vec();
        assert!(vals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_packed_tensor_f8x4r_zeros() {
        let t = PackedTensor::<F8x4R>::zeros(&[4]);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.packed_len(), 1);
        let vals = t.to_f32_vec();
        assert!(vals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_packed_tensor_f4x8_zeros() {
        let t = PackedTensor::<F4x8>::zeros(&[8]);
        assert_eq!(t.numel(), 8);
        assert_eq!(t.packed_len(), 1);
        let vals = t.to_f32_vec();
        assert!(vals.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_packed_tensor_f4x8_get_set() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 2.0, 4.0, -4.0, 6.0, -6.0];
        let mut t = PackedTensor::<F4x8>::from_f32_auto(&data, &[8]);
        for i in 0..8 {
            let v = t.get(i);
            let err = (v - data[i]).abs();
            assert!(
                err < 0.5 * data[i].abs().max(1.0),
                "F4x8 get({}) mismatch: got={}, expected={}",
                i,
                v,
                data[i]
            );
        }
        t.set(0, 3.0);
        let v = t.get(0);
        let err = (v - 3.0).abs();
        assert!(err < 0.5 * 3.0f32.max(1.0), "F4x8 set(0, 3.0) got={}", v);
    }

    #[test]
    fn test_f4x8_asymmetric_per_block_roundtrip() {
        let mut data: Vec<f32> = Vec::with_capacity(128);
        for i in 0..128 {
            data.push(-6.0 + (i as f32) * 12.0 / 127.0);
        }
        let t = PackedTensor::<F4x8>::from_f32_per_block_asymmetric(&data, &[128], 64);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            let s = t.scale_for_elem(0, i);
            assert!(
                err <= s + 1e-6,
                "F4x8 per-block asymmetric err {} exceeds scale {} at i={} orig={} rec={}",
                err,
                s,
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_f4x8_asymmetric_per_block_endpoints() {
        let mut data: Vec<f32> = Vec::with_capacity(128);
        for i in 0..128 {
            data.push(-6.0 + (i as f32) * 12.0 / 127.0);
        }
        let t = PackedTensor::<F4x8>::from_f32_per_block_asymmetric(&data, &[128], 64);
        let recovered = t.to_f32_vec();
        assert!(
            (recovered[0] + 6.0).abs() < 0.01,
            "bmin not exact: got {}",
            recovered[0]
        );
        assert!(
            (recovered[127] - 6.0).abs() < 0.01,
            "bmax not exact: got {}",
            recovered[127]
        );
    }

    #[test]
    fn test_f4x8_asymmetric_per_channel_roundtrip() {
        let mut data: Vec<f32> = Vec::with_capacity(64);
        for i in 0..64 {
            data.push(-6.0 + (i as f32) * 12.0 / 63.0);
        }
        let t = PackedTensor::<F4x8>::from_f32_per_channel_asymmetric(&data, &[64]);
        let recovered = t.to_f32_vec();
        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            let s = t.scale_for_elem(0, i);
            assert!(
                err <= s + 1e-6,
                "F4x8 per-channel asymmetric err {} exceeds scale {} at i={} orig={} rec={}",
                err,
                s,
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_f4x8_asymmetric_per_block_zero_range() {
        let data = vec![1.0f32; 128];
        let t = PackedTensor::<F4x8>::from_f32_per_block_asymmetric(&data, &[128], 64);
        let recovered = t.to_f32_vec();
        for (i, rec) in recovered.iter().enumerate() {
            assert_eq!(*rec, 1.0, "zero-range block at i={} got {}", i, rec);
        }
    }

    #[test]
    fn test_f4x8_asymmetric_per_block_scale_zp_values() {
        let mut data: Vec<f32> = Vec::with_capacity(128);
        for i in 0..64 {
            data.push(-3.0 + (i as f32) * 6.0 / 63.0);
        }
        for i in 0..64 {
            data.push(0.0 + (i as f32) * 2.0 / 63.0);
        }
        let t = PackedTensor::<F4x8>::from_f32_per_block_asymmetric(&data, &[128], 64);

        assert_eq!(t.scales.len(), 2, "should have 2 blocks");
        assert_eq!(t.zeros.len(), 2);

        let blk0_min = -3.0f32;
        let blk0_max = 3.0f32;
        let blk0_range = blk0_max - blk0_min;
        // FP4 clipped range = 8.0 (±4.0), not 12.0 (±6.0)
        let expected_scale0 = blk0_range / 8.0;
        // Mean used as zero-point; for uniform symmetric data mean = midpoint = 0.0
        let expected_zp0 = (blk0_max + blk0_min) / 2.0;
        assert!(
            (t.scales[0] - expected_scale0).abs() < 1e-5,
            "block 0 scale: got {}, expected {}",
            t.scales[0],
            expected_scale0
        );
        assert!(
            (t.zeros[0] - expected_zp0).abs() < 1e-5,
            "block 0 zero: got {}, expected {}",
            t.zeros[0],
            expected_zp0
        );

        let blk1_min = 0.0f32;
        let blk1_max = 2.0f32;
        let blk1_range = blk1_max - blk1_min;
        let expected_scale1 = blk1_range / 8.0;
        // Mean = 1.0 for uniform [0, 2]; midpoint = 1.0 — same for symmetric data
        let expected_zp1 = (blk1_max + blk1_min) / 2.0;
        assert!(
            (t.scales[1] - expected_scale1).abs() < 1e-5,
            "block 1 scale: got {}, expected {}",
            t.scales[1],
            expected_scale1
        );
        assert!(
            (t.zeros[1] - expected_zp1).abs() < 1e-5,
            "block 1 zero: got {}, expected {}",
            t.zeros[1],
            expected_zp1
        );

        let recovered = t.to_f32_vec();
        let s0 = t.scales[0];
        let z0 = t.zeros[0];
        for i in 0..64 {
            let err = (data[i] - recovered[i]).abs();
            let target = (data[i] - z0) / s0;
            let code = crate::dtypes::f4x8::f32_to_fp4(target);
            let decoded = crate::dtypes::f4x8::fp4_to_f32(code);
            let expected = decoded * s0 + z0;
            let expected_err = (data[i] - expected).abs();
            assert!(
                (err - expected_err).abs() < 1e-4,
                "block 0 elem {}: orig={}, recovered={}, err={}, expected_err={}",
                i,
                data[i],
                recovered[i],
                err,
                expected_err
            );
        }
    }

    #[test]
    fn test_f4x8_asymmetric_per_channel_scale_zp_values() {
        let mut data: Vec<f32> = Vec::with_capacity(128);
        for i in 0..64 {
            data.push(-5.0 + (i as f32) * 10.0 / 63.0);
        }
        for i in 0..64 {
            data.push(1.0 + (i as f32) * 4.0 / 63.0);
        }
        let t = PackedTensor::<F4x8>::from_f32_per_channel_asymmetric(&data, &[2, 64]);

        assert_eq!(t.scales.len(), 2, "per-channel should have 2 scales");

        let expected_scale0 = 10.0 / 8.0; // FP4 clipped range = 8.0
        let expected_zp0 = (-5.0 + 5.0) / 2.0; // uniform → mean == 0
        assert!(
            (t.scales[0] - expected_scale0).abs() < 1e-5,
            "row 0 scale: got {}, expected {}",
            t.scales[0],
            expected_scale0
        );
        assert!(
            (t.zeros[0] - expected_zp0).abs() < 1e-5,
            "row 0 zero: got {}, expected {}",
            t.zeros[0],
            expected_zp0
        );

        let expected_scale1 = 4.0 / 8.0; // FP4 clipped range = 8.0
        let _expected_zp1 = (1.0 + 5.0) / 2.0; // uniform → mean == 3.0
        assert!(
            (t.scales[1] - expected_scale1).abs() < 1e-5,
            "row 1 scale: got {}, expected {}",
            t.scales[1],
            expected_scale1
        );
        assert!(
            (t.zeros[0] - expected_zp0).abs() < 1e-5,
            "row 0 zero: got {}, expected {}",
            t.zeros[0],
            expected_zp0
        );
    }

    #[test]
    fn test_i4x8_codebook_debug() {
        // Two rows with different magnitude ranges
        let data: Vec<f32> = vec![
            // Row 0: range [-15, 15]
            -15.0, -10.0, -7.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, -5.0, 8.0,
            3.0, 6.0, -12.0, 0.5, -0.5, 11.0, -8.0, 2.5, -3.0, 9.0, -14.0, 13.0, -6.0, 4.5, -1.5,
            7.5, -9.0, // Row 1: range [-12, 20]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0,
            -11.0, -12.0,
        ];

        let qblock = 8; // 8 elements per block
        let t = PackedTensor::<I4x8>::from_f32_per_block_codebook(&data, &[2, 32], qblock);

        eprintln!("=== CODEBOOK DEBUG ===");
        eprintln!("Codebooks count: {}", t.codebooks.len());
        eprintln!("Codebook[0]: {:?}", t.codebooks[0]);
        eprintln!("Scales count: {}", t.scales.len());
        eprintln!("Scales: {:?}", t.scales);
        eprintln!("Shape: {:?}", t.shape);
        eprintln!("quant_block_size: {}", t.quant_block_size);

        // Row 0 has 4 blocks of 8, Row 1 has 4 blocks of 8
        // blocks_per_row = 32 / 8 = 4
        // scales = [row0_blk0, row0_blk1, row0_blk2, row0_blk3,
        //          row1_blk0, row1_blk1, row1_blk2, row1_blk3]

        let recovered = t.to_f32_vec();

        eprintln!("\n=== ORIGINAL vs RECOVERED (row 0) ===");
        let mut max_err = 0.0f32;
        for i in 0..32 {
            let err = (data[i] - recovered[i]).abs();
            if err > max_err {
                max_err = err;
            }
            if i < 16 {
                eprintln!(
                    "  [{:2}] orig={:7.3}  rec={:7.3}  err={:.4}",
                    i, data[i], recovered[i], err
                );
            }
        }
        eprintln!("\n=== ORIGINAL vs RECOVERED (row 1) ===");
        for i in 32..64 {
            let err = (data[i] - recovered[i]).abs();
            if err > max_err {
                max_err = err;
            }
            if i < 48 {
                eprintln!(
                    "  [{:2}] orig={:7.3}  rec={:7.3}  err={:.4}",
                    i, data[i], recovered[i], err
                );
            }
        }
        eprintln!("\nMax error: {:.4}", max_err);

        // Also check get_or_init_f32_weights (same path as backend)
        let f32_weights = t.get_or_init_f32_weights();
        eprintln!(
            "\nget_or_init_f32_weights matches to_f32_vec: {}",
            f32_weights == recovered.as_slice()
        );

        // Check per-block: each block scale should map the codebook to the right range
        eprintln!("\n=== BLOCK ANALYSIS ===");
        let blocks_per_row = 4;
        for blk in 0..8 {
            let row = blk / blocks_per_row;
            let blk_in_row = blk % blocks_per_row;
            let scale = t.scales[blk];
            let base = row * 32 + blk_in_row * qblock;
            let block_orig = &data[base..base + qblock];
            let block_rec = &recovered[base..base + qblock];
            let orig_abs_max = block_orig.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let rec_abs_max = block_rec.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let block_err: f32 = block_orig
                .iter()
                .zip(block_rec.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                / qblock as f32;
            eprintln!(
                "  Block {} (row {}): scale={:.4} orig_max={:.4} rec_max={:.4} avg_err={:.4}",
                blk, row, scale, orig_abs_max, rec_abs_max, block_err
            );
            eprintln!("    orig: {:?}", block_orig);
            eprintln!("    rec:  {:?}", block_rec);
        }

        // Assert reasonable quality
        assert!(
            max_err < 2.0,
            "Max error {} is too high for codebook quantization",
            max_err
        );
    }
}
