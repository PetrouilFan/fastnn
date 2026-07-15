#![allow(clippy::shadow_unrelated)]
#![allow(clippy::let_and_return)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::redundant_locals)]
#![allow(clippy::get_first)]
#![allow(clippy::if_same_then_else)]

use crate::backend::cpu::blas::matmul_blas_into;
use crate::backend::cpu::microkernels::{
    blocked_row_matmul, tls_alloc_f32, tls_alloc_u8, tls_alloc_zeroed_f32, tls_alloc_zeroed_i8,
    tls_alloc_zeroed_i8x4,
};
use crate::backend::prepared::PreparedActivation;
use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::plan::MemoryPlan;
use crate::dtypes::{F4x8, F8x4, F8x4R, I4x8, I8x4, PackedWord, U4x8, U8x4};
use crate::ir::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, ShapeEnv, TensorValue};
use crate::packed_tensor::PackedTensor;
use bytemuck;
use smallvec::{smallvec, SmallVec};
use std::any::TypeId;
use std::borrow::Cow;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

mod arena;
pub mod blas;
pub mod flash_attn;
pub mod im2col;
pub mod microkernels;
pub mod packed_conv;
pub mod packed_gemm;
pub mod reductions_fast;
pub mod sgemm;
pub mod swar;
pub mod telemetry;

mod dispatch_helpers;
use dispatch_helpers::*;

mod elementwise;
use elementwise::fused_binary_activation_dispatch;
mod scalar;
use scalar::{scalar_kernel_instruction, scalar_op_dispatch, unary_op_dispatch};
#[cfg(feature = "parallel")]
pub mod affinity;
mod params;
pub mod topology;
use params::resolve_params;
mod matmul;
use matmul::{
    matmul_activation_dispatch, quantized_matmul_dispatch, quantized_matmul_dispatch_i8_u4,
    quantized_matmul_dispatch_i8_u8,
};

/// Global cache for aligned packed weight data, keyed by
/// Per-plan packed weight cache keyed by `(raw_ptr, write_offset, raw_len, TypeId)`.
/// The raw data pointer acts as a unique compilation-session identifier so that
/// distinct weight values (even at the same arena offset) never collide.
fn get_or_cache_packed<T: PackedWord + 'static>(
    write_offset: usize,
    write_size: usize,
    raw: &[u8],
) -> Arc<Vec<T>> {
    let cache_key = (
        raw.as_ptr() as usize,
        write_offset,
        write_size,
        TypeId::of::<T>(),
    );
    type PackedCache = OnceLock<
        Mutex<HashMap<(usize, usize, usize, TypeId), Arc<dyn std::any::Any + Send + Sync>>>,
    >;
    static CACHE: PackedCache = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(arc_any) = cache.get(&cache_key) {
        if let Ok(arc_vec) = arc_any.clone().downcast::<Vec<T>>() {
            return arc_vec;
        }
    }
    let v: Arc<Vec<T>> = Arc::new(aligned_packed_slice::<T>(raw));
    cache.insert(cache_key, v.clone());
    v
}

type F32WeightCacheKey = (usize, usize, usize, TypeId);

/// Global cache for unpacked f32 weights, keyed by `(raw_ptr, write_offset, write_size, TypeId)`.
/// The raw data pointer acts as a unique compilation-session identifier so that
/// distinct weight values (even at the same arena offset) never collide.
/// This cache is scoped to the process lifetime but can be cleared between
/// executor instances (different quantization dtypes) via
/// [`clear_f32_weight_cache()`] to prevent memory accumulation.
static F32_WEIGHT_CACHE: OnceLock<
    Mutex<HashMap<F32WeightCacheKey, Arc<dyn std::any::Any + Send + Sync>>>,
> = OnceLock::new();

/// Clear the global f32 weight cache, freeing dequantized weight data.
/// Call this between executor instances (e.g., when switching quantization dtypes)
/// to prevent unbounded memory growth.
pub fn clear_f32_weight_cache() {
    if let Some(cache) = F32_WEIGHT_CACHE.get() {
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Retrieve or create a [`PackedTensor<T>`] with cached f32 dequantized weights.
/// The packed data is obtained from the per-plan packed-data cache, and the
/// dequantized f32 copy is stored in the module-level [`F32_WEIGHT_CACHE`] so
/// that repeated dispatch of the same conv layer reuses the f32 weights without
/// re-dequantization.
fn get_or_cache_f32_weights<T: PackedWord + 'static>(
    write_offset: usize,
    write_size: usize,
    raw: &[u8],
    shape: &[usize],
    scales: &[f32],
    zeros: &[f32],
    quant_block_size: usize,
    codebooks: &[[f32; 16]],
) -> Arc<PackedTensor<T>> {
    let cache_key = (
        raw.as_ptr() as usize,
        write_offset,
        write_size,
        TypeId::of::<T>(),
    );
    let mut cache = F32_WEIGHT_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(arc_any) = cache.get(&cache_key) {
        if let Ok(arc_pt) = arc_any.clone().downcast::<PackedTensor<T>>() {
            return arc_pt;
        }
    }
    let packed_data = get_or_cache_packed::<T>(write_offset, write_size, raw);
    let mut pt =
        PackedTensor::from_raw_arc(packed_data, shape.to_vec(), scales.to_vec(), zeros.to_vec());
    pt.quant_block_size = quant_block_size;
    if !codebooks.is_empty() {
        pt.codebooks = codebooks.to_vec();
    }
    let pt = Arc::new(pt);
    pt.get_or_init_f32_weights();
    cache.insert(cache_key, pt.clone());
    pt
}

#[allow(clippy::cognitive_complexity)]
/// Align-packed-weight helper: cast raw bytes to &[PackedType] when the
/// pointer is 4-byte aligned (the common case for arena-backed data),
/// otherwise fall back to a u32-anchored copy.
fn aligned_packed_slice<T: PackedWord>(raw: &[u8]) -> Vec<T> {
    if (raw.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        // SAFETY: bytemuck::cast_slice panics if misaligned; we checked alignment.
        bytemuck::cast_slice::<u8, T>(raw).to_vec()
    } else {
        let n_words = raw.len().div_ceil(4);
        let mut aligned: Vec<u32> = vec![0u32; n_words];
        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
        byte_slice[..raw.len()].copy_from_slice(raw);
        // Zero-copy cast: Vec<u32> -> Vec<T>.
        // SAFETY: T is repr(transparent) over u32, so size_of::<T>() == size_of::<u32>()
        // and align_of::<T>() == align_of::<u32>(). The element counts are identical.
        let mut v = std::mem::ManuallyDrop::new(aligned);
        unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity()) }
    }
}

/// CPU memory arena with interior mutability for zero-allocation dispatch.
///
/// # Soundness
///
/// `CpuBuffer` wraps [`Vec<u8>`] in an [`UnsafeCell`] so that the
/// [`dispatch`](Backend::dispatch) method can mutate the arena through
/// a shared `&CpuBuffer` reference.  Dispatch is single-threaded and
/// processes instructions sequentially, so the `&mut [u8]` slices
/// returned by [`data_mut`](CpuBuffer::data_mut) are never aliased.
///
/// `Send + Sync` are safe because the inner `Vec<u8>` is itself
/// `Send + Sync` and the arena is never accessed concurrently.
pub struct CpuBuffer(UnsafeCell<Vec<u8>>);

impl CpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        CpuBuffer(UnsafeCell::new(data))
    }

    /// Get a mutable slice to the arena data.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other `&mut [u8]` reference
    /// derived from this arena is live when this method is called.
    /// This is satisfied by the sequential dispatch loop â€” each
    /// `data_mut` call's borrow ends before the next one begins.
    #[allow(clippy::mut_from_ref)]
    pub fn data_mut(&self) -> &mut [u8] {
        // SAFETY: The `UnsafeCell` gives `&mut` to the inner `Vec<u8>`. This is safe because
        // `data_mut()` returns a borrow that is never aliased â€” dispatch processes instructions
        // sequentially and each borrow ends before the next begins.
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }

    /// Create a zero-copy `&[f32]` view over `[offset, offset + size)` bytes.
    ///
    /// # Safety
    ///
    /// - `offset + size` must be within the buffer bounds.
    /// - `offset` must be 4-byte aligned (f32 alignment).
    /// - No mutable access to this byte range may exist for the lifetime of the view.
    #[inline]
    pub unsafe fn view_f32(&self, offset: usize, size: usize) -> &[f32] {
        let ptr = (*self.0.get()).as_ptr().add(offset) as *const f32;
        std::slice::from_raw_parts(ptr, size / 4)
    }

    /// Create a zero-copy `&mut [f32]` view over `[offset, offset + size)` bytes.
    ///
    /// # Safety
    ///
    /// - `offset + size` must be within the buffer bounds.
    /// - `offset` must be 4-byte aligned (f32 alignment).
    /// - No other access (shared or mutable) to this byte range may exist for the
    ///   lifetime of the view.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn view_f32_mut(&self, offset: usize, size: usize) -> &mut [f32] {
        let ptr = (*self.0.get()).as_mut_ptr().add(offset) as *mut f32;
        std::slice::from_raw_parts_mut(ptr, size / 4)
    }

    /// Create a zero-copy `&[u8]` view over `[offset, offset + size)` bytes.
    ///
    /// # Safety
    ///
    /// - `offset + size` must be within the buffer bounds.
    /// - No mutable access to this byte range may exist for the lifetime of the view.
    #[inline]
    pub unsafe fn view_u8(&self, offset: usize, size: usize) -> &[u8] {
        let ptr = (*self.0.get()).as_ptr().add(offset);
        std::slice::from_raw_parts(ptr, size)
    }

    /// Create a zero-copy `&mut [u8]` view over `[offset, offset + size)` bytes.
    ///
    /// # Safety
    ///
    /// - `offset + size` must be within the buffer bounds.
    /// - No other access (shared or mutable) to this byte range may exist for the
    ///   lifetime of the view.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn view_u8_mut(&self, offset: usize, size: usize) -> &mut [u8] {
        // SAFETY: The caller guarantees bounds and exclusive access to this byte range.
        let ptr = (*self.0.get()).as_mut_ptr().add(offset);
        std::slice::from_raw_parts_mut(ptr, size)
    }
}

// SAFETY: `Vec<u8>` is `Send + Sync`.  The arena is never accessed
// concurrently â€” dispatch is single-threaded â€” so interior mutability
// via `UnsafeCell` does not introduce data races.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

#[cfg(test)]
mod cpu_buffer_tests {
    use super::CpuBuffer;

    #[test]
    fn mutable_byte_view_preserves_non_word_aligned_lengths() {
        for size in [1, 2, 3, 5, 6, 7] {
            let buffer = CpuBuffer::new(vec![0; 16]);
            {
                // SAFETY: The range is in bounds and this is the only live view.
                let view = unsafe { buffer.view_u8_mut(2, size) };
                assert_eq!(view.len(), size);
                view.fill(0xA5);
            }
            let data = buffer.data_mut();
            assert!(data[..2].iter().all(|byte| *byte == 0));
            assert!(data[2..2 + size].iter().all(|byte| *byte == 0xA5));
            assert!(data[2 + size..].iter().all(|byte| *byte == 0));
        }
    }
}

/// CPU execution context. Zero allocation during dispatch.
#[derive(Clone)]
pub struct CpuBackend;

fn read_adam_runtime_step(
    data: &[u8],
    slice: BufferSlice,
    kernel_name: &str,
) -> Result<f32, BackendError> {
    let bytes = data.get(slice.offset..slice.offset + 8).ok_or_else(|| {
        BackendError::Dispatch(format!("{kernel_name}: runtime step range is invalid"))
    })?;
    let step = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    if step == 0 || step > i32::MAX as u64 {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: optimizer step must be in 1..={}",
            i32::MAX
        )));
    }
    Ok(step as f32)
}

fn validate_adam_dispatch(
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    uses_f16_state: bool,
    decoupled_weight_decay: bool,
) -> Result<(), BackendError> {
    let valid_input_count = if decoupled_weight_decay {
        matches!(input_slices.len(), 4 | 5)
    } else {
        input_slices.len() == 4
    };
    if !valid_input_count {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: invalid input count {}",
            input_slices.len()
        )));
    }
    let required_params = if decoupled_weight_decay && input_slices.len() == 4 {
        6
    } else {
        5
    };
    if params.len() < required_params {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected at least {required_params} parameters"
        )));
    }
    let weight_size = input_slices[0].size;
    if weight_size == 0
        || !weight_size.is_multiple_of(4)
        || !input_slices[0].offset.is_multiple_of(4)
        || input_slices[1].size != weight_size
        || !input_slices[1].offset.is_multiple_of(4)
        || output_slice.size != weight_size
        || !output_slice.offset.is_multiple_of(4)
    {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: weight, gradient, and output slices must be matching f32 tensors"
        )));
    }
    let state_size = if uses_f16_state {
        weight_size / 2
    } else {
        weight_size
    };
    let state_alignment = if uses_f16_state { 2 } else { 4 };
    if input_slices[2].size != state_size
        || input_slices[3].size != state_size
        || !input_slices[2].offset.is_multiple_of(state_alignment)
        || !input_slices[3].offset.is_multiple_of(state_alignment)
    {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: optimizer state slices do not match the weight tensor"
        )));
    }
    if input_slices.len() == 5 && input_slices[4].size != 8 {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: runtime step tensor must contain exactly 8 bytes"
        )));
    }
    if input_slices.len() == 4 && (params[4] == 0 || params[4] > i32::MAX as usize) {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: optimizer step must be in 1..={}",
            i32::MAX
        )));
    }
    if decoupled_weight_decay {
        let weight_decay_index = if input_slices.len() == 5 { 4 } else { 5 };
        let weight_decay = f32::from_bits(params[weight_decay_index] as u32);
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: weight decay must be finite and non-negative"
            )));
        }
    }
    let lr = f32::from_bits(params[0] as u32);
    let beta1 = f32::from_bits(params[1] as u32);
    let beta2 = f32::from_bits(params[2] as u32);
    let eps = f32::from_bits(params[3] as u32);
    if !lr.is_finite()
        || lr < 0.0
        || !beta1.is_finite()
        || !(0.0..1.0).contains(&beta1)
        || !beta2.is_finite()
        || !(0.0..1.0).contains(&beta2)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: invalid optimizer hyperparameters"
        )));
    }
    Ok(())
}

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;

    fn name(&self) -> &str {
        "cpu"
    }

    fn allocate_arena(&self, total_bytes: usize) -> CpuBuffer {
        // Zero-fill the arena to guarantee deterministic behavior across
        // platform allocators.  On Linux, mmap-backed pages are typically
        // zero-filled by the kernel, but macOS and Windows allocators may
        // return reused memory with non-zero contents.  Zero-initialization
        // ensures that any kernel that fails to write its output slot
        // (e.g. due to a missed dispatch path) produces zeros instead of
        // platform-dependent garbage.
        let buf = vec![0u8; total_bytes];
        CpuBuffer::new(buf)
    }

    fn try_allocate_arena(&self, total_bytes: usize) -> Result<CpuBuffer, BackendError> {
        let mut buf = Vec::new();
        buf.try_reserve_exact(total_bytes).map_err(|error| {
            BackendError::Dispatch(format!(
                "failed to allocate {total_bytes}-byte CPU arena: {error}"
            ))
        })?;
        buf.resize(total_bytes, 0);
        Ok(CpuBuffer::new(buf))
    }

    fn compile(
        &self,
        graph: &ComputeGraph,
        memory_plan: &MemoryPlan,
    ) -> Result<ExecutablePlan, BackendError> {
        let mut instructions = Vec::with_capacity(graph.nodes.len());
        let order = graph
            .try_topological_sort()
            .map_err(|error| BackendError::Compilation(error.to_string()))?;

        // â”€â”€ Preâ€‘compute topological levels for parallel dispatch â”€â”€â”€â”€â”€â”€
        // level[node] = max(level[input]) + 1   (with input level = 0 for graph inputs)
        let mut node_level: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();
        for &node_id in &order {
            let node = graph.get_node(node_id).ok_or_else(|| {
                BackendError::Compilation(format!(
                    "topological order references missing node {node_id}"
                ))
            })?;
            let level = node
                .inputs
                .iter()
                .filter_map(|id| node_level.get(id))
                .max()
                .copied()
                .map_or(Ok(0), |level| {
                    level.checked_add(1).ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "topological level for node {node_id} overflows"
                        ))
                    })
                })?;
            node_level.insert(node_id, level);
        }
        let mut instruction_levels: Vec<usize> = Vec::with_capacity(order.len());

        for &node_id in &order {
            let node = graph
                .get_node(node_id)
                .ok_or_else(|| BackendError::Compilation(format!("Node {} not found", node_id)))?;
            let level = node_level.get(&node_id).copied().ok_or_else(|| {
                BackendError::Compilation(format!(
                    "topological level for node {node_id} is missing"
                ))
            })?;

            let input_slices: Vec<BufferSlice> = node
                .inputs
                .iter()
                .filter_map(|&input_id| {
                    memory_plan
                        .slots
                        .get(&input_id)
                        .map(|slot| BufferSlice::new(slot.offset, slot.size))
                })
                .collect();

            // Collect input shapes for dimension-dependent kernels.
            // Symbolic dims that can't be resolved at compile time are
            // replaced with SYMBOL_DIM_MAX to preserve shape rank.
            let symbol_max = crate::ir::SYMBOL_DIM_MAX.load(Ordering::Relaxed);
            let input_shapes: Vec<Vec<u64>> = node
                .inputs
                .iter()
                .filter_map(|&input_id| graph.get_node(input_id))
                .map(|n| {
                    n.output_type
                        .shape
                        .iter()
                        .map(|d| d.evaluate().unwrap_or(symbol_max))
                        .collect()
                })
                .collect();
            // Also collect raw DimExpr shapes for symbolic dispatch resolution
            let input_shape_dims: Vec<Vec<DimExpr>> = node
                .inputs
                .iter()
                .filter_map(|&input_id| graph.get_node(input_id))
                .map(|n| n.output_type.shape.clone())
                .collect();

            let output_slice = memory_plan
                .slots
                .get(&node_id)
                .map(|slot| BufferSlice::new(slot.offset, slot.size))
                .ok_or_else(|| {
                    BackendError::Compilation(format!(
                        "node {} ({:?}) has no memory slot",
                        node_id, node.opcode
                    ))
                })?;
            let secondary_output_slice = memory_plan
                .secondary_slots
                .get(&(node_id, 1))
                .map(|slot| BufferSlice::new(slot.offset, slot.size));

            match &node.opcode {
                Opcode::Constant(val) => match val {
                    TensorValue::Float(v) => {
                        instructions.push(Instruction::Fill {
                            dst: output_slice,
                            value: *v,
                        });
                    }
                    TensorValue::Int(v) => {
                        instructions.push(Instruction::Fill {
                            dst: output_slice,
                            value: *v as f32,
                        });
                    }
                    TensorValue::Data { bytes, .. } => {
                        instructions.push(Instruction::WriteConst {
                            dst: output_slice,
                            data: bytes.clone(),
                        });
                    }
                },
                Opcode::MatMul => {
                    // Detect quantized dtypes from input nodes to select the right kernel
                    let input_dtypes: Vec<_> = node
                        .inputs
                        .iter()
                        .filter_map(|&input_id| graph.get_node(input_id))
                        .map(|n| n.output_type.dtype.clone())
                        .collect();
                    let is_quantized = input_dtypes.iter().any(|d| {
                        matches!(
                            d,
                            IrDType::I4 { .. }
                                | IrDType::I8Scaled { .. }
                                | IrDType::U4Scaled { .. }
                                | IrDType::U8Scaled { .. }
                                | IrDType::F4 { .. }
                                | IrDType::F8 { .. }
                                | IrDType::F8R { .. }
                        )
                    });

                    let fused_type = node.attrs.get("fused_op").map(|s| s.as_str());
                    // Determine fusion params for the unified "matmul" kernel
                    let has_bias = input_slices.len() >= 3;
                    let activation_type = match fused_type {
                        Some("MatMulAddRelu") | Some("OpRelu") => 1, // ReLU
                        Some("MatMulAddGelu") | Some("OpGelu") => 2, // GELU
                        Some("MatMulAddSilu") | Some("OpSilu") => 3, // SiLU
                        _ => 0,                                      // No activation
                    };
                    let kernel_name = match (fused_type, is_quantized) {
                        // Non-quantized fused patterns: use specialized kernels for now
                        // (they have mature, tested implementations)
                        (Some("MatMulAddRelu"), false) => "fused_matmul_add_relu",
                        (Some("MatMulAddGelu"), false) => "fused_matmul_add_gelu",
                        (Some("MatMulAddSilu"), false) => "fused_matmul_add_silu",
                        (Some("OpRelu"), false) => "matmul_relu",
                        (Some("OpGelu"), false) => "matmul_gelu",
                        (Some("OpSilu"), false) => "matmul_silu",
                        // Non-quantized non-fused: use unified "matmul" kernel
                        (_, false) => "matmul",
                        // Quantized: I8 activation + U4 weight
                        (_, true)
                            if input_dtypes
                                .first()
                                .is_some_and(|d| matches!(d, IrDType::I8))
                                && input_dtypes.iter().any(|d| {
                                    matches!(d, IrDType::I4 { .. } | IrDType::U4Scaled { .. })
                                }) =>
                        {
                            "matmul_i4_i8"
                        }
                        // Quantized: I8 activation + U8 weight (no U4)
                        (_, true)
                            if input_dtypes
                                .first()
                                .is_some_and(|d| matches!(d, IrDType::I8))
                                && input_dtypes.iter().any(|d| {
                                    matches!(d, IrDType::I8Scaled { .. } | IrDType::U8Scaled { .. })
                                })
                                && !input_dtypes.iter().any(|d| {
                                    matches!(d, IrDType::I4 { .. } | IrDType::U4Scaled { .. })
                                }) =>
                        {
                            "matmul_i8_i8"
                        }
                        // Quantized: F32 activation + Signed I4 weight
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::I4 { .. })) =>
                        {
                            "matmul_i4"
                        }
                        // Quantized: F32 activation + Unsigned U4 weight
                        (_, true)
                            if input_dtypes
                                .iter()
                                .any(|d| matches!(d, IrDType::U4Scaled { .. })) =>
                        {
                            "matmul_u4"
                        }
                        // Quantized: F32 activation + FP4 weight
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::F4 { .. })) =>
                        {
                            "matmul_f4"
                        }
                        // Quantized: F32 activation + FP8 E4M3 weight
                        (_, true)
                            if input_dtypes.iter().any(|d| matches!(d, IrDType::F8 { .. })) =>
                        {
                            "matmul_f8"
                        }
                        // Quantized: F32 activation + FP8 E5M2 weight
                        (_, true)
                            if input_dtypes
                                .iter()
                                .any(|d| matches!(d, IrDType::F8R { .. })) =>
                        {
                            "matmul_f8r"
                        }
                        // Quantized: F32 activation + Unsigned U8 weight
                        (_, true)
                            if input_dtypes
                                .iter()
                                .any(|d| matches!(d, IrDType::U8Scaled { .. })) =>
                        {
                            "matmul_u8"
                        }
                        // Quantized: F32 activation + Signed I8 weight
                        (_, true) => "matmul_i8",
                    };
                    // Extract M, K, N from input shapes
                    let m = input_shapes
                        .first()
                        .and_then(|s| s.get(s.len().saturating_sub(2)).copied())
                        .unwrap_or(1) as usize;
                    let k = input_shapes
                        .first()
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1) as usize;
                    let n = input_shapes
                        .get(1)
                        .and_then(|s| s.last().copied())
                        .unwrap_or(1) as usize;
                    // Capture symbolic dims for runtime resolution
                    let m_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(s.len().saturating_sub(2)).cloned())
                        .unwrap_or(DimExpr::Known(m as u64));
                    let k_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.last().cloned())
                        .unwrap_or(DimExpr::Known(k as u64));
                    let n_dim = input_shape_dims
                        .get(1)
                        .and_then(|s| s.last().cloned())
                        .unwrap_or(DimExpr::Known(n as u64));
                    // Extract weight metadata for quantized matmul kernels.
                    // For 2D weights, the quantized data is stored transposed
                    // ([N, K] instead of [K, N]) so the shape must match.
                    let weight_meta = if is_quantized {
                        node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bit_width, scales, dequant_offsets, codebooks) = match &wn
                                    .output_type
                                    .dtype
                                {
                                    IrDType::I4 {
                                        scales,
                                        dequant_offsets,
                                        codebooks,
                                    } => (
                                        4usize,
                                        scales.clone(),
                                        dequant_offsets.clone(),
                                        codebooks.clone(),
                                    ),
                                    IrDType::U8Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (8usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    IrDType::U4Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (4usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    IrDType::I8Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (8usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    IrDType::F4 {
                                        scales,
                                        dequant_offsets: zeros,
                                        ..
                                    } => (4usize, scales.clone(), zeros.clone(), vec![]),
                                    IrDType::F8 { scales } => {
                                        (8usize, scales.clone(), vec![0.0; scales.len()], vec![])
                                    }
                                    IrDType::F8R { scales } => {
                                        (8usize, scales.clone(), vec![0.0; scales.len()], vec![])
                                    }
                                    _ => (0usize, vec![], vec![], vec![]),
                                };
                                let mut w_shape: Vec<usize> = wn
                                    .output_type
                                    .shape
                                    .iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                // The quantization pass transposes 2D weight data from
                                // [K, N] to [N, K] and records the logical shape ([K, N])
                                // on the node; reverse to [N, K] for the packed-gemm meta.
                                if w_shape.len() == 2 {
                                    w_shape.reverse();
                                }
                                let quant_block_size =
                                    if let Some(qbs) = wn.attrs.get("quant_block_size") {
                                        qbs.parse::<usize>().unwrap_or(0)
                                    } else if scales.len() > w_shape[0] && w_shape.len() >= 2 {
                                        let inner: usize = w_shape[1..].iter().product();
                                        let blocks_per_row = scales.len() / w_shape[0];
                                        inner / blocks_per_row
                                    } else {
                                        0
                                    };
                                std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                                    bit_width,
                                    scales,
                                    dequant_offsets,
                                    shape: w_shape,
                                    quant_block_size,
                                    codebooks,
                                })
                            })
                        })
                    } else {
                        None
                    };
                    // For non-quantized matmul, pass fusion params (has_bias, activation_type)
                    // Quantized kernels keep the original [M, K, N] params
                    let (params, param_dims) = if kernel_name == "matmul" {
                        (
                            vec![m, k, n, has_bias as usize, activation_type],
                            Some(vec![
                                m_dim,
                                k_dim,
                                n_dim,
                                DimExpr::Known(has_bias as u64),
                                DimExpr::Known(activation_type as u64),
                            ]),
                        )
                    } else {
                        (vec![m, k, n], Some(vec![m_dim, k_dim, n_dim]))
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims,
                        weight_meta,
                    });
                }
                Opcode::Add
                | Opcode::Sub
                | Opcode::Mul
                | Opcode::Div
                | Opcode::Maximum
                | Opcode::Minimum => {
                    let mut kernel = match node.opcode {
                        Opcode::Add => "add_f32",
                        Opcode::Sub => "sub_f32",
                        Opcode::Mul => "mul_f32",
                        Opcode::Div => "div_f32",
                        Opcode::Maximum => "max_f32",
                        Opcode::Minimum => "min_f32",
                        _ => unreachable!(),
                    };
                    // Op+Activation fusion: if fused_op is set, use the fused kernel name
                    match node.attrs.get("fused_op").map(|s| s.as_str()) {
                        Some("OpRelu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_relu_f32",
                                Opcode::Sub => "sub_relu_f32",
                                Opcode::Mul => "mul_relu_f32",
                                Opcode::Div => "div_relu_f32",
                                _ => unreachable!(),
                            };
                        }
                        Some("OpGelu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_gelu_f32",
                                Opcode::Sub => "sub_gelu_f32",
                                Opcode::Mul => "mul_gelu_f32",
                                Opcode::Div => "div_gelu_f32",
                                _ => unreachable!(),
                            };
                        }
                        Some("OpSilu") => {
                            kernel = match node.opcode {
                                Opcode::Add => "add_silu_f32",
                                Opcode::Sub => "sub_silu_f32",
                                Opcode::Mul => "mul_silu_f32",
                                Opcode::Div => "div_silu_f32",
                                _ => unreachable!(),
                            };
                        }
                        _ => {}
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Relu
                | Opcode::Gelu
                | Opcode::Silu
                | Opcode::Sigmoid
                | Opcode::Tanh
                | Opcode::Exp
                | Opcode::Log
                | Opcode::Sqrt
                | Opcode::Neg
                | Opcode::Abs
                | Opcode::LeakyRelu
                | Opcode::Elu
                | Opcode::Softplus
                | Opcode::Hardswish
                | Opcode::Clamp
                | Opcode::Sign
                | Opcode::Round
                | Opcode::LogicalNot
                | Opcode::LogSoftmax
                | Opcode::Mish => {
                    let kernel = match node.opcode {
                        Opcode::Relu => "relu_f32",
                        Opcode::Gelu => "gelu_f32",
                        Opcode::Silu => "silu_f32",
                        Opcode::Sigmoid => "sigmoid_f32",
                        Opcode::Tanh => "tanh_f32",
                        Opcode::Exp => "exp_f32",
                        Opcode::Log => "log_f32",
                        Opcode::Sqrt => "sqrt_f32",
                        Opcode::Neg => "neg_f32",
                        Opcode::Abs => "abs_f32",
                        Opcode::LeakyRelu => "leaky_relu_f32",
                        Opcode::Elu => "elu_f32",
                        Opcode::Softplus => "softplus_f32",
                        Opcode::Hardswish => "hardswish_f32",
                        Opcode::Clamp => "clamp_f32",
                        Opcode::Sign => "sign_f32",
                        Opcode::Round => "round_f32",
                        Opcode::LogicalNot => "logical_not_f32",
                        Opcode::LogSoftmax => "log_softmax_f32",
                        Opcode::Mish => "mish_f32",
                        _ => unreachable!(),
                    };
                    let mut extra_params: Vec<usize> = Vec::with_capacity(graph.nodes.len());
                    if let Opcode::LeakyRelu = node.opcode {
                        let slope: f32 = node
                            .attrs
                            .get("negative_slope")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.01);
                        extra_params.push(slope.to_bits() as usize);
                    }
                    if let Opcode::Clamp = node.opcode {
                        let min: f32 = node
                            .attrs
                            .get("min")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0);
                        let max: f32 = node
                            .attrs
                            .get("max")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1.0);
                        extra_params.push(min.to_bits() as usize);
                        extra_params.push(max.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: extra_params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Reshape | Opcode::Flatten | Opcode::Squeeze | Opcode::Unsqueeze => {
                    if let Some(&input_id) = node.inputs.first() {
                        if let (Some(in_slot), Some(out_slot)) = (
                            memory_plan.slots.get(&input_id),
                            memory_plan.slots.get(&node_id),
                        ) {
                            if in_slot.offset != out_slot.offset || in_slot.size != out_slot.size {
                                instructions.push(Instruction::MemCopy {
                                    dst: output_slice,
                                    src: BufferSlice::new(in_slot.offset, in_slot.size),
                                });
                            }
                        }
                    }
                }
                Opcode::Conv2d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let dilation: usize = node
                        .attrs
                        .get("dilation")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(1);
                    let groups: usize = node
                        .attrs
                        .get("groups")
                        .and_then(|g| g.parse().ok())
                        .unwrap_or(1);
                    // Detect quantized weights for packed conv2d dispatch
                    let weight_dtype = node
                        .inputs
                        .get(1)
                        .and_then(|&w_id| graph.get_node(w_id))
                        .map(|wn| wn.output_type.dtype.clone());
                    let is_quantized = weight_dtype.as_ref().is_some_and(|d| {
                        matches!(
                            d,
                            IrDType::I4 { .. }
                                | IrDType::I8Scaled { .. }
                                | IrDType::U4Scaled { .. }
                                | IrDType::U8Scaled { .. }
                                | IrDType::F4 { .. }
                                | IrDType::F8 { .. }
                                | IrDType::F8R { .. }
                        )
                    });
                    let (kernel_name, weight_meta) = if is_quantized {
                        let dtype = weight_dtype.as_ref().ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "quantized convolution node {node_id} has no weight dtype"
                            ))
                        })?;
                        let mut kernel = if matches!(dtype, IrDType::F4 { .. }) {
                            "conv2d_f4".to_string()
                        } else if matches!(dtype, IrDType::F8 { .. }) {
                            "conv2d_f8".to_string()
                        } else if matches!(dtype, IrDType::F8R { .. }) {
                            "conv2d_f8r".to_string()
                        } else if matches!(dtype, IrDType::U4Scaled { .. }) {
                            "conv2d_u4".to_string()
                        } else if matches!(dtype, IrDType::U8Scaled { .. }) {
                            "conv2d_u8".to_string()
                        } else if matches!(dtype, IrDType::I4 { .. }) {
                            "conv2d_i4".to_string()
                        } else {
                            "conv2d_i8".to_string()
                        };
                        let bit_width = if matches!(
                            dtype,
                            IrDType::I4 { .. } | IrDType::U4Scaled { .. } | IrDType::F4 { .. }
                        ) {
                            4usize
                        } else {
                            8usize
                        };

                        // Detect signed INT8 activations from QuantizeActivations and append suffix.
                        // Treat both IrDType::I8 and IrDType::I8Scaled as i8-activation sources, because the
                        // activation-quantization compiler pass currently emits U8 payloads.
                        let act_dtype = node
                            .inputs
                            .first()
                            .and_then(|&a_id| graph.get_node(a_id))
                            .map(|an| an.output_type.dtype.clone());
                        if act_dtype
                            .as_ref()
                            .is_some_and(|d| matches!(d, IrDType::I8 | IrDType::I8Scaled { .. }))
                        {
                            kernel = format!("{}_i8", kernel);
                        }

                        // Detect fused activation (SiLU/ReLU/GELU) from operator fusion
                        // and append to kernel name so dispatch can apply it.
                        if let Some(fused_op) = node.attrs.get("fused_op").map(|s| s.as_str()) {
                            match fused_op {
                                "OpRelu" => kernel = format!("{}_relu", kernel),
                                "OpGelu" => kernel = format!("{}_gelu", kernel),
                                "OpSilu" => kernel = format!("{}_silu", kernel),
                                _ => {}
                            }
                        }

                        let meta = node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bw, scales, dequant_offsets, codebooks) = match &wn
                                    .output_type
                                    .dtype
                                {
                                    IrDType::I4 {
                                        scales,
                                        dequant_offsets,
                                        codebooks,
                                    } => (
                                        4usize,
                                        scales.clone(),
                                        dequant_offsets.clone(),
                                        codebooks.clone(),
                                    ),
                                    IrDType::U8Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (8usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    IrDType::U4Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (4usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    IrDType::F4 {
                                        scales,
                                        dequant_offsets: zeros,
                                        ..
                                    } => (4usize, scales.clone(), zeros.clone(), vec![]),
                                    IrDType::F8 { scales } => {
                                        (8usize, scales.clone(), vec![0.0; scales.len()], vec![])
                                    }
                                    IrDType::F8R { scales } => {
                                        (8usize, scales.clone(), vec![0.0; scales.len()], vec![])
                                    }
                                    IrDType::I8Scaled {
                                        scales,
                                        dequant_offsets,
                                    } => (8usize, scales.clone(), dequant_offsets.clone(), vec![]),
                                    _ => (bit_width, vec![], vec![], vec![]),
                                };
                                let w_shape: Vec<usize> = wn
                                    .output_type
                                    .shape
                                    .iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                let quant_block_size =
                                    if let Some(qbs) = wn.attrs.get("quant_block_size") {
                                        qbs.parse::<usize>().unwrap_or(0)
                                    } else if scales.len() > w_shape[0] && w_shape.len() >= 2 {
                                        let inner: usize = w_shape[1..].iter().product();
                                        let blocks_per_row = scales.len() / w_shape[0];
                                        inner / blocks_per_row
                                    } else {
                                        0
                                    };
                                std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                                    bit_width: bw,
                                    scales,
                                    dequant_offsets,
                                    shape: w_shape,
                                    quant_block_size,
                                    codebooks,
                                })
                            })
                        });
                        (kernel.to_string(), meta)
                    } else {
                        let fused_type = node.attrs.get("fused_op").map(|s| s.as_str());
                        let base_name = match fused_type {
                            Some("OpRelu") => "conv2d_relu",
                            Some("OpGelu") => "conv2d_gelu",
                            Some("OpSilu") => "conv2d_silu",
                            _ => "conv2d",
                        };
                        (base_name.to_string(), None)
                    };
                    // Extract spatial dims from input shapes to avoid
                    // ambiguous dim inference at dispatch time.
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name,
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        // params: [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]
                        params: vec![
                            stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h,
                            kernel_w,
                        ],
                        param_dims: None,
                        weight_meta,
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    let eps = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    let is_batch_norm = if matches!(node.opcode, Opcode::BatchNorm) {
                        1
                    } else {
                        0
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "norm_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize, is_batch_norm],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Softmax => {
                    // Read and normalize the axis attribute.
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let rank = input_shapes.first().map(|s| s.len()).unwrap_or(1);
                    let normalized_axis = if axis < 0 {
                        (rank as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    // Capture the axis-dimension size for the dispatch handler,
                    // which needs to know how many elements comprise a single
                    // softmax "row" (all elements along the reduction axis).
                    let axis_dim = input_shapes
                        .first()
                        .and_then(|s| s.get(normalized_axis).copied())
                        .unwrap_or(1) as usize;
                    let axis_dim_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(normalized_axis).cloned())
                        .unwrap_or(DimExpr::Known(1));
                    // Compute stride: product of dims after the axis.
                    // This is needed because softmax rows are strided for
                    // non-last dimensions (e.g. axis=2 on [N,C,H,W]).
                    let stride = input_shapes
                        .first()
                        .map(|s| {
                            s[normalized_axis + 1..]
                                .iter()
                                .copied()
                                .map(|x| x as usize)
                                .product::<usize>()
                                .max(1)
                        })
                        .unwrap_or(1);

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "softmax".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis_dim, stride],
                        param_dims: Some(vec![axis_dim_dim, DimExpr::Known(stride as u64)]),
                        weight_meta: None,
                    });
                }
                Opcode::BiasAdd => {
                    // Channel stride = product of spatial dims (H*W for 4D NCHW).
                    // This is needed for correct NCHW channel-wise broadcast:
                    //   bias_idx = (flat_idx / channel_stride) % num_channels
                    let channel_stride = input_shapes
                        .first()
                        .map(|s| s.iter().skip(2).product::<u64>())
                        .unwrap_or(1) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "biasadd".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![channel_stride],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Concat => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    // Compute inner_stride (product of dims after axis) and
                    // outer_count (product of dims before axis) from output shape.
                    let output_shape: Vec<u64> = node
                        .output_type
                        .shape
                        .iter()
                        .map(|d| d.evaluate().unwrap_or(symbol_max))
                        .collect();
                    let rank = output_shape.len();
                    let inner_stride: u64 = if axis + 1 < rank {
                        output_shape[axis + 1..].iter().product()
                    } else {
                        1
                    };
                    let outer_count: u64 = if axis > 0 {
                        output_shape[..axis].iter().product()
                    } else {
                        1
                    };
                    let _input_ids_str = node
                        .inputs
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    #[cfg(feature = "debug_canary")]
                    eprintln!(
                        "[FNN_DBG_CONCAT_COMPILE] nid={} op=Concat inputs=[{}] axis={} inner_stride={} outer_count={} output_shape={:?}",
                        node_id,
                        _input_ids_str,
                        axis,
                        inner_stride,
                        outer_count,
                        output_shape,
                    );
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "concat".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis, inner_stride as usize, outer_count as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MaxPool | Opcode::AvgPool => {
                    let kernel_size: usize = node
                        .attrs
                        .get("kernel_size")
                        .and_then(|k| k.parse().ok())
                        .unwrap_or(2);
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let is_max = if matches!(node.opcode, Opcode::MaxPool) {
                        1
                    } else {
                        0
                    };
                    let secondary_output_slice = if is_max == 1 {
                        memory_plan
                            .secondary_slots
                            .get(&(node_id, 1))
                            .map(|slot| BufferSlice::new(slot.offset, slot.size))
                    } else {
                        None
                    };
                    // Pass explicit input dims so pool_f32 doesn't need to
                    // infer N,C,H,W from flat element counts (ambiguous).
                    let input_n = input_shapes
                        .first()
                        .and_then(|s| s.get(0).copied())
                        .unwrap_or(1) as usize;
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pool_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        // params: [kernel_size, stride, padding, is_max, N, C, H, W]
                        params: vec![
                            kernel_size,
                            stride,
                            padding,
                            is_max,
                            input_n,
                            input_c,
                            input_h,
                            input_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pad => {
                    let pads_str = node.attrs.get("pads").cloned().unwrap_or_default();
                    let pads: Vec<usize> = pads_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pad_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: pads,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Gather => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "gather".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![axis],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Slice => {
                    let dim: usize = node
                        .attrs
                        .get("dim")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(0);
                    let start: usize = node
                        .attrs
                        .get("start")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let end: usize = node
                        .attrs
                        .get("end")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(1);
                    // Compute the stride (product of dim sizes after `dim`)
                    // so the kernel can correctly compute the offset for non-batch dims.
                    let stride = input_shapes
                        .first()
                        .filter(|s| dim < s.len())
                        .map(|s| s[dim + 1..].iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "slice_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![dim, start, end, stride],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ScatterNd => {
                    let data_shape: Vec<u64> = input_shapes.first().cloned().unwrap_or_default();
                    let indices_shape: Vec<u64> = input_shapes.get(1).cloned().unwrap_or_default();
                    let index_depth = match indices_shape.len() {
                        0 | 1 => 1,
                        _ => indices_shape[indices_shape.len() - 1] as usize,
                    };
                    let mut params: Vec<usize> = vec![index_depth];
                    params.extend(data_shape.iter().map(|&d| d as usize));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "scatter_nd".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax => {
                    // Group size = product of dims being reduced over.
                    // For a single-axis reduce this is just input_shape[axis],
                    // which is typically Known (e.g. reduce over dim 1 of [N,4]
                    // has group_size=Known(4)).
                    let axis_i64: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let axis = if axis_i64 < 0 {
                        (rank + axis_i64).max(0)
                    } else {
                        axis_i64.min((rank - 1).max(0))
                    } as usize;
                    let group_size_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(axis).cloned())
                        .unwrap_or(DimExpr::Known(1));
                    let (is_mean, is_max) = match node.opcode {
                        Opcode::ReduceMean => (1, 0),
                        Opcode::ReduceMax => (0, 1),
                        _ => (0, 0), // ReduceSum
                    };
                    let group_size = group_size_dim.evaluate().unwrap_or_else(|| {
                        // Symbolic dim â€” use SYMBOL_DIM_MAX as compile-time
                        // estimate; runtime resolves via param_dims.
                        crate::ir::SYMBOL_DIM_MAX.load(Ordering::Relaxed)
                    }) as usize;

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "reduce_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![group_size, is_mean, is_max],
                        // Pass the group_size as a symbolic DimExpr so dispatch
                        // can re-evaluate it when shape_env is available (e.g.
                        // reduce over symbolic batch dim N).
                        param_dims: Some(vec![group_size_dim]),
                        weight_meta: None,
                    });
                }
                Opcode::Transpose => {
                    let input_shape: Vec<usize> = input_shapes
                        .first()
                        .map(|s| s.iter().map(|&d| d as usize).collect())
                        .unwrap_or_default();
                    let rank = input_shape.len();

                    // Read perm from node attrs (e.g. "0,3,1,2")
                    let perm_str: String = node.attrs.get("perm").cloned().unwrap_or_default();
                    let perm: Vec<usize> = if perm_str.is_empty() {
                        (0..rank).rev().collect()
                    } else {
                        perm_str.split(',').filter_map(|s| s.parse().ok()).collect()
                    };

                    // Simple 2D transpose [1,0] on a rank-2 tensor â†’ use fast kernel
                    if rank == 2 && perm.len() >= 2 && perm[0] == 1 && perm[1] == 0 {
                        let m = input_shape[0];
                        let n = input_shape[1];
                        let m_dim = input_shape_dims
                            .first()
                            .and_then(|s| s.first().cloned())
                            .unwrap_or(DimExpr::Known(m as u64));
                        let n_dim = input_shape_dims
                            .first()
                            .and_then(|s| s.get(1).cloned())
                            .unwrap_or(DimExpr::Known(n as u64));
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "transpose_f32".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![m, n],
                            param_dims: Some(vec![m_dim, n_dim]),
                            weight_meta: None,
                        });
                    } else {
                        // N-D permute transpose: params = [rank, d0..dN, p0..pN]
                        let mut nd_params: Vec<usize> = Vec::with_capacity(1 + 2 * rank);
                        nd_params.push(rank);
                        nd_params.extend_from_slice(&input_shape);
                        for i in 0..rank {
                            nd_params.push(perm.get(i).copied().unwrap_or(i));
                        }
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "transpose_perm_f32".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: nd_params,
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Conv1d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv1d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![stride, padding, input_c, input_w, kernel_w],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Conv3d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let dilation: usize = node
                        .attrs
                        .get("dilation")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(1);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_d = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(4).copied())
                        .unwrap_or(0) as usize;
                    let kernel_d = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(4).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv3d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            stride, padding, dilation, input_c, input_d, input_h, input_w,
                            kernel_d, kernel_h, kernel_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ConvTranspose2d => {
                    let stride: usize = node
                        .attrs
                        .get("stride")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let padding: usize = node
                        .attrs
                        .get("padding")
                        .and_then(|p| p.parse().ok())
                        .unwrap_or(0);
                    let input_c = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let input_h = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let input_w = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    let kernel_h = input_shapes
                        .get(1)
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(0) as usize;
                    let kernel_w = input_shapes
                        .get(1)
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "conv_transpose2d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            stride, padding, input_c, input_h, input_w, kernel_h, kernel_w,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Prelu => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "prelu".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::RMSNorm => {
                    let eps = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse::<f32>().ok())
                        .unwrap_or(1e-5);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "rms_norm".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Embedding => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "embedding".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pow => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pow_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::GtScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "gt_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::LtScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "lt_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::EqScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "eq_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::AddScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "add_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::MulScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "mul_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                Opcode::DivScalar => {
                    instructions.push(scalar_kernel_instruction(
                        node_id,
                        "div_scalar_f32",
                        input_slices,
                        output_slice,
                    ));
                }
                // Input nodes have no producer instruction â€” data is written
                // by the executor before dispatch.
                Opcode::Input => {
                    // No instruction needed.
                }
                Opcode::ArgMax => {
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(-1);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let normalized = if axis < 0 {
                        (rank + axis).max(0)
                    } else {
                        axis.min((rank - 1).max(0))
                    };
                    let (dim_size, inner) = input_shapes
                        .first()
                        .and_then(|s| {
                            let idx = normalized as usize;
                            if idx < s.len() {
                                let ds = s[idx] as usize;
                                let inn: usize =
                                    s[idx + 1..].iter().copied().map(|x| x as usize).product();
                                Some((ds, inn))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0, 1));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "argmax".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![normalized as usize, dim_size, inner],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::UpsampleNearest2d | Opcode::UpsampleBilinear2d => {
                    let scale_h: usize = node
                        .attrs
                        .get("scale_h")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    let scale_w: usize = node
                        .attrs
                        .get("scale_w")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2);
                    // Pass input spatial dims so the kernel doesn't need to guess H,W
                    // from flat buffer size (which is ambiguous for NCHW layouts).
                    let h_in = input_shapes
                        .first()
                        .and_then(|s| s.get(2).copied())
                        .unwrap_or(1) as usize;
                    let w_in = input_shapes
                        .first()
                        .and_then(|s| s.get(3).copied())
                        .unwrap_or(1) as usize;
                    let kernel_name = match node.opcode {
                        Opcode::UpsampleNearest2d => "upsample_nearest2d",
                        _ => "upsample_bilinear2d",
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![scale_h, scale_w, h_in, w_in],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdaptiveAvgPool2d => {
                    let out_h: usize = node
                        .attrs
                        .get("output_h")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let out_w: usize = node
                        .attrs
                        .get("output_w")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "adaptive_avg_pool2d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![out_h, out_w],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Repeat => {
                    let repeats_str = node.attrs.get("repeats").cloned().unwrap_or_default();
                    let repeats: Vec<usize> = repeats_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "repeat".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: repeats,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::CumSum => {
                    let dim: usize = node
                        .attrs
                        .get("dim")
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(0);
                    let exclusive: usize = node
                        .attrs
                        .get("exclusive")
                        .and_then(|e| e.parse().ok())
                        .unwrap_or(0);
                    let rev: usize = node
                        .attrs
                        .get("reverse")
                        .and_then(|r| r.parse().ok())
                        .unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "cumsum".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![dim, exclusive, rev],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Erf => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "erf_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Flip => {
                    let dims_str = node.attrs.get("dims").cloned().unwrap_or_default();
                    let dims: Vec<usize> = dims_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    let input_shape: Vec<u64> = input_shapes.first().cloned().unwrap_or_default();
                    let mut params = vec![dims.len()];
                    params.extend_from_slice(&dims);
                    params.extend(input_shape.iter().map(|&s| s as usize));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "flip".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Where => {
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "where_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::TopK => {
                    let k: usize = node
                        .attrs
                        .get("k")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(-1);
                    let rank = input_shapes.first().map(|s| s.len() as i64).unwrap_or(0);
                    let normalized = if axis < 0 {
                        (rank + axis).max(0)
                    } else {
                        axis.min((rank - 1).max(0))
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "topk_fused".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        params: vec![k, normalized as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                // â”€â”€ Optimizer ops â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                Opcode::SgdUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "sgd_update_f32".to_string(),
                        input_slices, // [weight, grad] â€” weight must be same slot as output
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![lr.to_bits() as usize, wd.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    let t: u64 = node
                        .attrs
                        .get("t")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    // Detect F16 state tensors (m and v at inputs[2] and inputs[3]).
                    let has_f16_state = node.inputs.len() >= 4
                        && graph
                            .get_node(node.inputs[2])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false)
                        && graph
                            .get_node(node.inputs[3])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false);
                    let kernel_name = if has_f16_state {
                        "adam_update_f16_state"
                    } else {
                        "adam_update_f32"
                    };
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices, // [weight, grad, m, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            beta2.to_bits() as usize,
                            eps.to_bits() as usize,
                            t as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamWUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let has_f16_state = node.inputs.len() >= 4
                        && graph
                            .get_node(node.inputs[2])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false)
                        && graph
                            .get_node(node.inputs[3])
                            .map(|n| n.output_type.dtype == IrDType::F16)
                            .unwrap_or(false);
                    let kernel_name = if has_f16_state {
                        "adamw_update_f16_state"
                    } else {
                        "adamw_update_f32"
                    };
                    let t_attr: u64 = node
                        .attrs
                        .get("t")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let has_t_input = node.inputs.len() >= 5;
                    let mut adamw_params = vec![
                        lr.to_bits() as usize,
                        beta1.to_bits() as usize,
                        beta2.to_bits() as usize,
                        eps.to_bits() as usize,
                    ];
                    if has_t_input {
                        // New path: t is a runtime tensor (5th input slice)
                        adamw_params.push(wd.to_bits() as usize);
                    } else {
                        // Old path: t is stored in params
                        adamw_params.push(t_attr as usize);
                        adamw_params.push(wd.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices, // [weight, grad, m, v, t] (t only for training pass path)
                        output_slice,
                        secondary_output_slice: None,
                        params: adamw_params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MuonUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.01);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let weight_decay: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "muon_update_f32".to_string(),
                        input_slices, // [weight, grad, m]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            weight_decay.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::LionUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta1: f32 = node
                        .attrs
                        .get("beta1")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.9);
                    let beta2: f32 = node
                        .attrs
                        .get("beta2")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.999);
                    let wd: f32 = node
                        .attrs
                        .get("weight_decay")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.0);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "lion_update_f32".to_string(),
                        input_slices, // [weight, grad, m]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta1.to_bits() as usize,
                            beta2.to_bits() as usize,
                            wd.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::RmspropUpdate => {
                    let lr: f32 = node
                        .attrs
                        .get("lr")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.001);
                    let beta: f32 = node
                        .attrs
                        .get("beta")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0.99);
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-8);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "rmsprop_update_f32".to_string(),
                        input_slices, // [weight, grad, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![
                            lr.to_bits() as usize,
                            beta.to_bits() as usize,
                            eps.to_bits() as usize,
                        ],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Shape => {
                    // Write the shape of the input tensor as F32 values.
                    // The arena stores all data as f32 (4 bytes/element), so we
                    // write f32-le bytes.  Downstream ops (Gather, Concat, etc.)
                    // read from the arena as f32 slices and get correct values.
                    // Resolve input shape at compile time (known dims directly,
                    // symbolic dims use SYMBOL_DIM_MAX â€” they'll be resolved
                    // at dispatch by param_dims).
                    let in_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "shape node {node_id} has no resolved input shape"
                        ))
                    })?;
                    let byte_len = in_shape.len().checked_mul(4).ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "shape node {node_id} output size overflows"
                        ))
                    })?;
                    if output_slice.size != byte_len {
                        return Err(BackendError::Compilation(format!(
                            "shape node {node_id} output slot has {} bytes, expected {byte_len}",
                            output_slice.size
                        )));
                    }
                    let mut shape_bytes = Vec::new();
                    shape_bytes.try_reserve_exact(byte_len).map_err(|_| {
                        BackendError::Compilation(format!(
                            "shape node {node_id} output allocation failed"
                        ))
                    })?;
                    for &dimension in in_shape {
                        // Shape values currently use the legacy f32 tensor contract. Reject
                        // dimensions that cannot round-trip instead of silently changing them.
                        let encoded = dimension as f32;
                        if encoded as u64 != dimension {
                            return Err(BackendError::Compilation(format!(
                                "shape node {node_id} dimension {dimension} is not exactly representable as f32"
                            )));
                        }
                        shape_bytes.extend_from_slice(&encoded.to_le_bytes());
                    }
                    instructions.push(Instruction::WriteConst {
                        dst: output_slice,
                        data: shape_bytes,
                    });
                }
                Opcode::Cast => {
                    // Cast: same-shape, potentially different dtype.
                    // For same-byte-size casts, MemCopy is sufficient.
                    // For different byte sizes, do element conversion.
                    let in_type = node
                        .inputs
                        .first()
                        .and_then(|id| graph.get_node(*id))
                        .map(|n| n.output_type.dtype.clone());
                    let out_type = node.output_type.dtype.clone();
                    let in_byte_size = in_type.as_ref().map(|d| d.byte_size()).unwrap_or(4);
                    let out_byte_size = out_type.byte_size();
                    if in_byte_size == out_byte_size {
                        // Same byte size: just copy.
                        if let Some(&input_id) = node.inputs.first() {
                            if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                                instructions.push(Instruction::MemCopy {
                                    dst: output_slice,
                                    src: BufferSlice::new(in_slot.offset, in_slot.size),
                                });
                            }
                        }
                    } else {
                        // Different byte size: use a kernel call.
                        let in_slot = node.inputs.first().and_then(|id| memory_plan.slots.get(id));
                        let input_slices = in_slot
                            .map(|s| vec![BufferSlice::new(s.offset, s.size)])
                            .unwrap_or_default();
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "cast".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![in_byte_size, out_byte_size],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Quantize => {
                    let bit_width: usize = node
                        .attrs
                        .get("bit_width")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4);
                    let kernel_name = match bit_width {
                        4 => "quantize_f32_u4",
                        8 => "quantize_f32_u8",
                        _ => {
                            return Err(BackendError::Compilation(format!(
                                "Quantize: unsupported bit_width={bit_width}"
                            )))
                        }
                    };
                    // num_channels = dim 0 (rows), num_elems_per_channel = product of rest
                    let num_channels = input_shapes
                        .first()
                        .and_then(|s| s.first().copied())
                        .unwrap_or(1) as usize;
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    let num_elems_per_channel = if num_channels > 0 {
                        numel / num_channels
                    } else {
                        numel
                    };

                    // If the predecessor node already carries calibrated scales/zeros
                    // (e.g. from wrap_quantized_optimizer re-quant path), forward
                    // them through params so the kernel can skip the O(NÃ—K) scan.
                    let (cached_scales, cached_zeros) = node
                        .inputs
                        .first()
                        .and_then(|&input_id| graph.get_node(input_id))
                        .map(|n| match &n.output_type.dtype {
                            IrDType::I4 {
                                scales,
                                dequant_offsets,
                                ..
                            }
                            | IrDType::U4Scaled {
                                scales,
                                dequant_offsets,
                                ..
                            }
                            | IrDType::I8Scaled {
                                scales,
                                dequant_offsets,
                                ..
                            }
                            | IrDType::U8Scaled {
                                scales,
                                dequant_offsets,
                                ..
                            } => (scales.clone(), dequant_offsets.clone()),
                            _ => (vec![], vec![]),
                        })
                        .unwrap_or_default();

                    let mut params = vec![num_channels, num_elems_per_channel, numel];
                    if !cached_scales.is_empty() && cached_scales.len() == num_channels {
                        params.push(1); // flag: cached
                        for &s in &cached_scales {
                            params.push(s.to_bits() as usize);
                        }
                        for &zp in &cached_zeros {
                            params.push(zp.to_bits() as usize);
                        }
                    }

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params, // includes cached scales/zeros when available
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Dequantize => {
                    let input_node = node
                        .inputs
                        .first()
                        .and_then(|&input_id| graph.get_node(input_id))
                        .ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "dequantize node {node_id} has no input metadata"
                            ))
                        })?;
                    let input_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "dequantize node {node_id} has no resolved input shape"
                        ))
                    })?;
                    let numel_u64 = input_shape.iter().try_fold(1u64, |acc, value| {
                        acc.checked_mul(*value).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "dequantize node {node_id} element count overflows"
                            ))
                        })
                    })?;
                    let numel = usize::try_from(numel_u64).map_err(|_| {
                        BackendError::Compilation(format!(
                            "dequantize node {node_id} element count exceeds this platform"
                        ))
                    })?;
                    // Preserve the storage width explicitly; dispatch must not infer
                    // representation from payload length.
                    let (scales, mut dequant_offsets, bit_width) =
                        match &input_node.output_type.dtype {
                            IrDType::I4 {
                                scales,
                                dequant_offsets,
                                ..
                            }
                            | IrDType::U4Scaled {
                                scales,
                                dequant_offsets,
                                ..
                            } => (scales.clone(), dequant_offsets.clone(), 4usize),
                            IrDType::I8Scaled {
                                scales,
                                dequant_offsets,
                            }
                            | IrDType::U8Scaled {
                                scales,
                                dequant_offsets,
                            } => (scales.clone(), dequant_offsets.clone(), 8usize),
                            dtype => {
                                return Err(BackendError::Compilation(format!(
                                "dequantize node {node_id} has unsupported input dtype {dtype:?}"
                            )))
                            }
                        };
                    if dequant_offsets.is_empty() && !scales.is_empty() {
                        dequant_offsets.resize(scales.len(), 0.0);
                    }
                    if scales.len() != dequant_offsets.len() {
                        return Err(BackendError::Compilation(format!(
                            "dequantize node {node_id} scale and offset counts differ"
                        )));
                    }
                    let has_metadata = !scales.is_empty();
                    let format_flag: usize = usize::from(has_metadata);
                    // [numel, format, bit_width, channels, scales..., offsets...]
                    let mut params = vec![numel, format_flag, bit_width];
                    let num_channels = scales.len();
                    params.push(num_channels);
                    for &scale in &scales {
                        params.push(scale.to_bits() as usize);
                    }
                    for &offset in &dequant_offsets {
                        params.push(offset.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "dequantize_kernel".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ToF16 => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "to_f16".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ToF32 => {
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "to_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::QuantizeActivations => {
                    let input_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "activation quantization node {node_id} has no input shape"
                        ))
                    })?;
                    let numel_u64 = input_shape.iter().try_fold(1u64, |acc, value| {
                        acc.checked_mul(*value).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "activation quantization node {node_id} element count overflows"
                            ))
                        })
                    })?;
                    let numel = usize::try_from(numel_u64).map_err(|_| {
                        BackendError::Compilation(format!(
                            "activation quantization node {node_id} exceeds this platform"
                        ))
                    })?;
                    let mode = node.attrs.get("mode").map(String::as_str);
                    let is_per_channel = mode == Some("per_channel");
                    if !matches!(mode, None | Some("per_tensor") | Some("per_channel")) {
                        return Err(BackendError::Compilation(format!(
                            "activation quantization node {node_id} has invalid mode"
                        )));
                    }
                    let num_channels = if is_per_channel {
                        let value = node
                            .attrs
                            .get("num_channels")
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "activation quantization node {node_id} is missing num_channels"
                                ))
                            })?
                            .parse::<usize>()
                            .map_err(|_| {
                                BackendError::Compilation(format!(
                                    "activation quantization node {node_id} has invalid num_channels"
                                ))
                            })?;
                        if value == 0
                            || value > u32::MAX as usize
                            || numel % value != 0
                            || numel / value > u32::MAX as usize
                        {
                            return Err(BackendError::Compilation(format!(
                                "activation quantization node {node_id} has incompatible channels"
                            )));
                        }
                        value
                    } else {
                        0
                    };
                    let parse_affine = |name: &str| -> Result<Vec<f32>, BackendError> {
                        node.attrs
                            .get(name)
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "activation quantization node {node_id} is missing {name}"
                                ))
                            })?
                            .split(',')
                            .map(|value| {
                                value.parse::<f32>().map_err(|_| {
                                    BackendError::Compilation(format!(
                                        "activation quantization node {node_id} has invalid {name}"
                                    ))
                                })
                            })
                            .collect()
                    };
                    let mut params = vec![numel, usize::from(is_per_channel), num_channels];
                    if is_per_channel {
                        let scales = parse_affine("scales")?;
                        let offsets = parse_affine("zero_points")?;
                        if scales.len() != num_channels
                            || offsets.len() != num_channels
                            || scales
                                .iter()
                                .any(|scale| !scale.is_finite() || *scale <= 0.0)
                            || offsets.iter().any(|offset| !offset.is_finite())
                        {
                            return Err(BackendError::Compilation(format!(
                                "activation quantization node {node_id} has invalid affine metadata"
                            )));
                        }
                        params.extend(scales.into_iter().map(|value| value.to_bits() as usize));
                        params.extend(offsets.into_iter().map(|value| value.to_bits() as usize));
                    } else {
                        match (node.attrs.get("scale"), node.attrs.get("zero_point")) {
                            (None, None) => params.push(0),
                            (Some(scale), Some(offset)) => {
                                let scale = scale.parse::<f32>().map_err(|_| {
                                    BackendError::Compilation(format!(
                                        "activation quantization node {node_id} has invalid scale"
                                    ))
                                })?;
                                let offset = offset.parse::<f32>().map_err(|_| {
                                    BackendError::Compilation(format!(
                                        "activation quantization node {node_id} has invalid zero_point"
                                    ))
                                })?;
                                if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                                    return Err(BackendError::Compilation(format!(
                                        "activation quantization node {node_id} has invalid affine metadata"
                                    )));
                                }
                                params.push(1);
                                params.push(scale.to_bits() as usize);
                                params.push(offset.to_bits() as usize);
                            }
                            _ => {
                                return Err(BackendError::Compilation(format!(
                                    "activation quantization node {node_id} has incomplete affine metadata"
                                )))
                            }
                        }
                    }
                    // Output buffer:
                    //   per-tensor: [scale(f32)][zp(f32)][i8_data(numel bytes)]
                    //   per-channel: [num_channels(u32)][chunk_size(u32)][scale_1]...[scale_n][zp_1]...[zp_n][channel_data...]
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "quantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::DequantizeActivations => {
                    let input_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "activation dequantization node {node_id} has no input shape"
                        ))
                    })?;
                    let numel_u64 = input_shape.iter().try_fold(1u64, |acc, value| {
                        acc.checked_mul(*value).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "activation dequantization node {node_id} element count overflows"
                            ))
                        })
                    })?;
                    let numel = usize::try_from(numel_u64).map_err(|_| {
                        BackendError::Compilation(format!(
                            "activation dequantization node {node_id} exceeds this platform"
                        ))
                    })?;
                    let predecessor = node
                        .inputs
                        .first()
                        .and_then(|predecessor| graph.get_node(*predecessor))
                        .ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "activation dequantization node {node_id} has no predecessor"
                            ))
                        })?;
                    if predecessor.opcode != Opcode::QuantizeActivations {
                        return Err(BackendError::Compilation(format!(
                            "activation dequantization node {node_id} must consume QuantizeActivations"
                        )));
                    }
                    let mode = predecessor.attrs.get("mode").map(String::as_str);
                    let is_per_channel = mode == Some("per_channel");
                    if !matches!(mode, None | Some("per_tensor") | Some("per_channel")) {
                        return Err(BackendError::Compilation(format!(
                            "activation dequantization node {node_id} has invalid mode"
                        )));
                    }
                    let num_channels = if is_per_channel {
                        let channels = predecessor
                            .attrs
                            .get("num_channels")
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "activation dequantization node {node_id} is missing num_channels"
                                ))
                            })?
                            .parse::<usize>()
                            .map_err(|_| {
                                BackendError::Compilation(format!(
                                    "activation dequantization node {node_id} has invalid num_channels"
                                ))
                            })?;
                        if channels == 0 || numel % channels != 0 {
                            return Err(BackendError::Compilation(format!(
                                "activation dequantization node {node_id} has incompatible channels"
                            )));
                        }
                        channels
                    } else {
                        0
                    };
                    let params = vec![numel, usize::from(is_per_channel), num_channels];
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "dequantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::FusedResidualAddNorm => {
                    let eps: f32 = node
                        .attrs
                        .get("eps")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1e-5);
                    let norm_type = node
                        .attrs
                        .get("norm_type")
                        .map(|s| s.as_str())
                        .unwrap_or("layer_norm");
                    let kernel_name = format!("fused_residual_add_{}", norm_type);

                    let output_numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    let row_size = node
                        .attrs
                        .get("normalized_ndims")
                        .and_then(|s| s.parse::<usize>().ok())
                        .and_then(|ndims| {
                            input_shapes.first().map(|shape| {
                                let start = shape.len().saturating_sub(ndims);
                                shape[start..]
                                    .iter()
                                    .copied()
                                    .map(|d| d as usize)
                                    .product::<usize>()
                                    .max(1)
                            })
                        })
                        .or_else(|| {
                            input_shapes
                                .get(2)
                                .map(|s| s.iter().product::<u64>() as usize)
                        })
                        .unwrap_or(output_numel.max(1));
                    let row_size_dim = node
                        .attrs
                        .get("normalized_ndims")
                        .and_then(|s| s.parse::<usize>().ok())
                        .and_then(|ndims| {
                            input_shape_dims.first().map(|shape| {
                                let start = shape.len().saturating_sub(ndims);
                                shape[start..]
                                    .iter()
                                    .cloned()
                                    .fold(DimExpr::Known(1), |acc, dim| acc.mul(&dim))
                            })
                        })
                        .unwrap_or(DimExpr::Known(row_size as u64));
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name,
                        input_slices, // [residual, main_output, weight, optional_bias]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![eps.to_bits() as usize, row_size],
                        param_dims: Some(vec![row_size_dim]),
                        weight_meta: None,
                    });
                }
                Opcode::GradientScale => {
                    let scale: f32 = node
                        .attrs
                        .get("scale")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                    let numel = input_shapes
                        .first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    if !input_slices.is_empty() {
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "gradient_scale".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![numel, scale.to_bits() as usize],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::QuantizeGradient => {
                    if let Some(&_input_slice) = input_slices.first() {
                        let numel = input_shapes
                            .first()
                            .map(|s| s.iter().product::<u64>() as usize)
                            .unwrap_or(1);
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "quantize_gradient_f32_to_f8x4r".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![numel],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::DequantizeGradient => {
                    if let Some(&_input_slice) = input_slices.first() {
                        let numel = input_shapes
                            .first()
                            .map(|s| s.iter().product::<u64>() as usize)
                            .unwrap_or(1);
                        instructions.push(Instruction::CallKernel {
                            node_id: Some(node_id),
                            kernel_name: "dequantize_gradient_f8x4r_to_f32".to_string(),
                            input_slices,
                            output_slice,
                            secondary_output_slice: None,
                            params: vec![numel],
                            param_dims: None,
                            weight_meta: None,
                        });
                    }
                }
                Opcode::Expand => {
                    // Expand broadcasts input[0] to the shape specified by input[1].
                    // Resolve input and output shapes at compile time and pack them
                    // into params: [max_rank, in_d0, in_d1, ..., out_d0, out_d1, ...].
                    // The kernel uses these to compute broadcast strides.
                    let data_shape_dims = input_shape_dims.first().cloned().unwrap_or_default();
                    let out_shape_dims = node.output_type.shape.clone();
                    let out_rank = out_shape_dims.len();
                    let data_rank = data_shape_dims.len();
                    let max_rank = out_rank.max(data_rank);

                    // Resolve to concrete values (Known dims, safe in YOLO pipeline)
                    let resolve = |d: &DimExpr| d.evaluate().unwrap_or(1) as usize;

                    // Build params: [max_rank, padded_in_dims..., padded_out_dims...]
                    let mut params = vec![max_rank];
                    for i in 0..max_rank {
                        if i < max_rank - data_rank {
                            params.push(1);
                        } else {
                            params.push(resolve(&data_shape_dims[i - (max_rank - data_rank)]));
                        }
                    }
                    for i in 0..max_rank {
                        if i < max_rank - out_rank {
                            params.push(1);
                        } else {
                            params.push(resolve(&out_shape_dims[i - (max_rank - out_rank)]));
                        }
                    }

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "expand_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Tile => {
                    // Tile copies first input (no broadcast yet).
                    if let Some(&input_id) = node.inputs.first() {
                        if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                            instructions.push(Instruction::MemCopy {
                                dst: output_slice,
                                src: BufferSlice::new(in_slot.offset, in_slot.size),
                            });
                        }
                    }
                }
                Opcode::Range => {
                    // Range(start, limit, step) â€” produce 1D F32 tensor.
                    // All 3 inputs are scalars (4 bytes each).
                    let input_slices: Vec<BufferSlice> = node
                        .inputs
                        .iter()
                        .filter_map(|id| memory_plan.slots.get(id))
                        .map(|slot| BufferSlice::new(slot.offset, slot.size))
                        .collect();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "range_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                #[allow(unreachable_patterns)]
                _ => {
                    if let Some(&input_id) = node.inputs.first() {
                        if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                            instructions.push(Instruction::MemCopy {
                                dst: output_slice,
                                src: BufferSlice::new(in_slot.offset, in_slot.size),
                            });
                        } else {
                            // no input slot â€” userâ€‘error / unexpected
                        }
                    }
                }
            }
            instruction_levels.push(level);
        }

        Ok(ExecutablePlan {
            instructions,
            arena_size: memory_plan.total_size,
            levels: instruction_levels,
        })
    }

    fn dispatch(
        &self,
        plan: &ExecutablePlan,
        arena: &CpuBuffer,
        shape_env: &ShapeEnv,
    ) -> Result<(), BackendError> {
        plan.validate()?;
        let arena_size = arena.data_mut().len();
        if arena_size < plan.arena_size {
            return Err(BackendError::Dispatch(format!(
                "CPU arena has {arena_size} bytes but plan requires {}",
                plan.arena_size
            )));
        }
        // Ensure the global thread pool is initialized with pinned physical cores.
        #[cfg(feature = "parallel")]
        crate::backend::cpu::affinity::ensure_global_pool_initialized();

        // â”€â”€ Debug: collect MaxPool primary output ranges â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        //     (only active with `debug_canary` feature â€” expensive)
        // ANCHOR: debug-canary-start
        #[cfg(feature = "debug_canary")]
        let maxpool_ranges: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            for instr in &plan.instructions {
                if let Instruction::CallKernel {
                    kernel_name,
                    params,
                    output_slice,
                    ..
                } = instr
                {
                    if kernel_name == "pool_f32" && params.len() >= 4 && params[3] == 1 {
                        v.push((output_slice.offset, output_slice.size));
                    }
                }
            }
            v
        };
        // Track: after each MaxPool kernel, snapshot its first f32 value;
        // after every other instruction, check if any snapshot changed.
        #[cfg(feature = "debug_canary")]
        let mut maxpool_snapshot: Vec<Option<f32>> = vec![None; maxpool_ranges.len()];
        #[cfg(feature = "debug_canary")]
        let mut maxpool_seen: Vec<bool> = vec![false; maxpool_ranges.len()];
        // ANCHOR: debug-canary-end

        for instr in &plan.instructions {
            match instr {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    secondary_output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    ..
                } => {
                    let out_start = output_slice.offset;
                    let out_end = output_slice.offset + output_slice.size;

                    match kernel_name.as_str() {
                        "add_f32" => {
                            fused_binary_activation_dispatch(
                                "add_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x,
                            );
                        }
                        "sub_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x,
                            );
                        }
                        "mul_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x,
                            );
                        }
                        "div_f32" => {
                            fused_binary_activation_dispatch(
                                "div_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x,
                            );
                        }
                        "relu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, relu_f32);
                        }
                        "gelu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, gelu_f32);
                        }
                        "silu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, silu_f32);
                        }
                        "exp_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, exp_f32);
                        }
                        "log_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, log_f32);
                        }
                        "sqrt_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sqrt_f32);
                        }
                        "neg_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, neg_f32);
                        }
                        "abs_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, abs_f32);
                        }
                        "sigmoid_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sigmoid_f32);
                        }
                        "tanh_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, tanh_f32);
                        }
                        "leaky_relu_f32" => {
                            let slope = if !params.is_empty() {
                                f32::from_bits(params[0] as u32)
                            } else {
                                0.01
                            };
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |input, out_f32| {
                                    leaky_relu_f32(input, out_f32, slope);
                                },
                            );
                        }
                        "elu_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, elu_f32);
                        }
                        "softplus_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                softplus_f32,
                            );
                        }
                        "hardswish_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                hardswish_f32,
                            );
                        }
                        "clamp_f32" => {
                            let min_val = if !params.is_empty() {
                                f32::from_bits(params[0] as u32)
                            } else {
                                0.0
                            };
                            let max_val = if params.len() > 1 {
                                f32::from_bits(params[1] as u32)
                            } else {
                                1.0
                            };
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |input, out_f32| {
                                    clamp_f32(input, out_f32, min_val, max_val);
                                },
                            );
                        }
                        "sign_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, sign_f32);
                        }
                        "round_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, round_f32);
                        }
                        "logical_not_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                logical_not_f32,
                            );
                        }
                        "log_softmax_f32" => {
                            unary_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                log_softmax_f32,
                            );
                        }
                        "mish_f32" => {
                            unary_op_dispatch(input_slices, arena, out_start, out_end, mish_f32);
                        }
                        "max_f32" => {
                            fused_binary_activation_dispatch(
                                "max_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a: f32, b: f32| a.max(b),
                                |x| x,
                            );
                        }
                        "min_f32" => {
                            fused_binary_activation_dispatch(
                                "min_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a: f32, b: f32| a.min(b),
                                |x| x,
                            );
                        }
                        "matmul" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                // params: [M, K, N, has_bias(0/1), activation(0=none, 1=relu, 2=gelu, 3=silu)]
                                let matmul_params =
                                    resolve_params(params, param_dims, shape_env, 5)?;
                                let &[m, _k, n, has_bias, activation] = &matmul_params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "matmul: expected params [M,K,N,has_bias,activation]"
                                            .into(),
                                    ));
                                };
                                let has_bias = has_bias != 0;
                                let bias_slice = if has_bias {
                                    Some(input_slices[2])
                                } else {
                                    None
                                };
                                let out_size = (out_end - out_start) / 4;
                                let a_stride = m * _k;
                                let b_stride = _k * n;
                                let out_stride = m * n;
                                let batch_count = out_size / out_stride;
                                let use_blas = m * _k * n >= blas::MIN_BLAS_SIZE * 64;
                                let apply_fusion = has_bias || activation != 0;

                                // Get mutable output region from arena
                                let out_slice = {
                                    let d = arena.data_mut();
                                    &mut d[out_start..out_end]
                                };

                                // Compute into arena output directly — no intermediate Vec needed.
                                // SAFETY: arena guarantees input and output ranges are disjoint.
                                unsafe {
                                    let (a, b, bias) = {
                                        let d = arena.data_mut().as_mut_ptr();
                                        let a = std::slice::from_raw_parts(
                                            d.add(a_slice.offset) as *const f32,
                                            a_slice.size / 4,
                                        );
                                        let b = std::slice::from_raw_parts(
                                            d.add(b_slice.offset) as *const f32,
                                            b_slice.size / 4,
                                        );
                                        let bias = bias_slice.map(|bs| {
                                            std::slice::from_raw_parts(
                                                d.add(bs.offset) as *const f32,
                                                bs.size / 4,
                                            )
                                        });
                                        (a, b, bias)
                                    };

                                    // Cast out_slice to f32 slice for direct computation
                                    let out_f32: &mut [f32] = bytemuck::cast_slice_mut(out_slice);

                                    // Zero the output first
                                    out_f32.fill(0.0);

                                    let b_batched = b.len() > b_stride;

                                    if use_blas {
                                        for batch in 0..batch_count {
                                            let a_s = batch * a_stride;
                                            let b_s = if b_batched { batch * b_stride } else { 0 };
                                            let out_s = batch * out_stride;
                                            let ob = &mut out_f32[out_s..out_s + out_stride];

                                            if has_bias {
                                                if let Some(bs) = bias {
                                                    for j in 0..m {
                                                        let row_start = j * n;
                                                        ob[row_start..row_start + n]
                                                            .copy_from_slice(&bs[..n]);
                                                    }
                                                }
                                            }

                                            matmul_blas_into(
                                                &a[a_s..a_s + a_stride],
                                                &b[b_s..b_s + b_stride],
                                                ob,
                                                m,
                                                _k,
                                                n,
                                            );

                                            if activation != 0 {
                                                for i in 0..ob.len() {
                                                    ob[i] = match activation {
                                                        1 => ob[i].max(0.0),
                                                        2 => {
                                                            let x = ob[i];
                                                            let x3 = x * x * x;
                                                            let tanh_arg =
                                                                0.797_884_6 * (x + 0.044_715 * x3);
                                                            0.5 * x * (1.0 + tanh_arg.tanh())
                                                        }
                                                        3 => ob[i] / (1.0 + (-ob[i]).exp()),
                                                        _ => ob[i],
                                                    };
                                                }
                                            }
                                        }
                                    } else {
                                        let total_rows = batch_count * m;
                                        let b_batch_stride = if b_batched { b_stride } else { 0 };
                                        let use_parallel = {
                                            #[cfg(feature = "parallel")]
                                            {
                                                total_rows
                                                    >= crate::backend::cpu::topology::physical_core_count()
                                            }
                                            #[cfg(not(feature = "parallel"))]
                                            {
                                                false
                                            }
                                        };

                                        if use_parallel {
                                            #[cfg(feature = "parallel")]
                                            {
                                                use rayon::prelude::*;
                                                let out_ptr = out_f32.as_mut_ptr() as usize;
                                                (0..total_rows).into_par_iter().for_each(
                                                    move |row| {
                                                        let op = out_ptr as *mut f32;
                                                        blocked_row_matmul(
                                                            a.as_ptr(),
                                                            b.as_ptr(),
                                                            op,
                                                            row,
                                                            m,
                                                            n,
                                                            _k,
                                                            a_stride,
                                                            _k,
                                                            1,
                                                            b_batch_stride,
                                                            n,
                                                            1,
                                                        );
                                                        if apply_fusion {
                                                            let rs = row * n;
                                                            for i in 0..n {
                                                                let p = rs + i;
                                                                let x = *op.add(p)
                                                                    + if has_bias {
                                                                        if let Some(bs) = bias {
                                                                            bs[i]
                                                                        } else {
                                                                            0.0
                                                                        }
                                                                    } else {
                                                                        0.0
                                                                    };
                                                                *op.add(p) = match activation {
                                                                    1 => x.max(0.0),
                                                                    2 => {
                                                                        let x3 = x * x * x;
                                                                        let tanh_arg = 0.797_884_6
                                                                            * (x + 0.044_715 * x3);
                                                                        0.5 * x
                                                                            * (1.0
                                                                                + tanh_arg.tanh())
                                                                    }
                                                                    3 => x / (1.0 + (-x).exp()),
                                                                    _ => x,
                                                                };
                                                            }
                                                        }
                                                    },
                                                );
                                            }
                                        } else {
                                            for row in 0..total_rows {
                                                blocked_row_matmul(
                                                    a.as_ptr(),
                                                    b.as_ptr(),
                                                    out_f32.as_mut_ptr(),
                                                    row,
                                                    m,
                                                    n,
                                                    _k,
                                                    a_stride,
                                                    _k,
                                                    1,
                                                    b_batch_stride,
                                                    n,
                                                    1,
                                                );
                                                if apply_fusion {
                                                    let rs = row * n;
                                                    for i in 0..n {
                                                        let p = rs + i;
                                                        let x = out_f32[p]
                                                            + if has_bias {
                                                                if let Some(bs) = bias {
                                                                    bs[i]
                                                                } else {
                                                                    0.0
                                                                }
                                                            } else {
                                                                0.0
                                                            };
                                                        out_f32[p] = match activation {
                                                            1 => x.max(0.0),
                                                            2 => {
                                                                let x3 = x * x * x;
                                                                let tanh_arg = 0.797_884_6
                                                                    * (x + 0.044_715 * x3);
                                                                0.5 * x * (1.0 + tanh_arg.tanh())
                                                            }
                                                            3 => x / (1.0 + (-x).exp()),
                                                            _ => x,
                                                        };
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "fused_matmul_add_relu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_relu",
                                |x| x.max(0.0),
                            )?;
                        }
                        "fused_matmul_add_gelu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_gelu",
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            )?;
                        }
                        "fused_matmul_add_silu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "fused_matmul_add_silu",
                                |x| x / (1.0 + (-x).exp()),
                            )?;
                        }
                        "matmul_relu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_relu",
                                |x| x.max(0.0),
                            )?;
                        }
                        "matmul_gelu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_gelu",
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            )?;
                        }
                        "matmul_silu" => {
                            matmul_activation_dispatch(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                out_start,
                                out_end,
                                "matmul_silu",
                                |x| x / (1.0 + (-x).exp()),
                            )?;
                        }
                        "matmul_i4" => {
                            quantized_matmul_dispatch::<I4x8>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                4,
                                "matmul_i4",
                            )?;
                        }
                        "matmul_u4" => {
                            quantized_matmul_dispatch::<U4x8>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                4,
                                "matmul_u4",
                            )?;
                        }
                        "matmul_i4_i8" => {
                            quantized_matmul_dispatch_i8_u4(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                "matmul_i4_i8",
                            )?;
                        }
                        "matmul_i8_i8" => {
                            quantized_matmul_dispatch_i8_u8(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                "matmul_i8_i8",
                            )?;
                        }
                        "matmul_i8" => {
                            quantized_matmul_dispatch::<I8x4>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                8,
                                "matmul_i8",
                            )?;
                        }
                        "matmul_u8" => {
                            quantized_matmul_dispatch::<U8x4>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                8,
                                "matmul_u8",
                            )?;
                        }
                        "matmul_f4" => {
                            quantized_matmul_dispatch::<F4x8>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                4,
                                "matmul_f4",
                            )?;
                        }
                        "matmul_f8" => {
                            quantized_matmul_dispatch::<F8x4>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                8,
                                "matmul_f8",
                            )?;
                        }
                        "matmul_f8r" => {
                            quantized_matmul_dispatch::<F8x4R>(
                                input_slices,
                                arena,
                                params,
                                param_dims,
                                shape_env,
                                weight_meta,
                                out_start,
                                out_end,
                                8,
                                "matmul_f8r",
                            )?;
                        }
                        "reduce_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let &[group_size, is_mean, is_max] = &params[..3] else {
                                    return Err(BackendError::Dispatch(
                                        "reduce_f32: expected params [group_size, is_mean, is_max]"
                                            .into(),
                                    ));
                                };
                                let effective_group_size = match param_dims {
                                    Some(dims) if !dims.is_empty() => {
                                        dims[0].evaluate_with_env(shape_env).map_err(|e| {
                                            BackendError::Dispatch(format!("reduce_f32: {e}"))
                                        })? as usize
                                    }
                                    _ => group_size,
                                };
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        reduce_f32(
                                            input,
                                            out_f32,
                                            effective_group_size,
                                            is_mean == 1,
                                            is_max == 1,
                                        );
                                    },
                                );
                            }
                        }
                        "transpose_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let transpose_params =
                                    resolve_params(params, param_dims, shape_env, 2)?;
                                let &[m, n] = &transpose_params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "transpose_f32: expected params [M,N]".into(),
                                    ));
                                };
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() && m >= 8 && n >= 8 {
                                            unsafe {
                                                microkernels::transpose_f32_avx2(
                                                    input, out_f32, m, n,
                                                );
                                            }
                                            return;
                                        }
                                        #[cfg(not(feature = "parallel"))]
                                        {
                                            for i in 0..m {
                                                for j in 0..n {
                                                    out_f32[j * m + i] = input[i * n + j];
                                                }
                                            }
                                        }
                                        #[cfg(feature = "parallel")]
                                        {
                                            use rayon::prelude::*;
                                            out_f32.par_chunks_mut(m).enumerate().for_each(
                                                |(j, col)| {
                                                    for i in 0..m {
                                                        col[i] = input[i * n + j];
                                                    }
                                                },
                                            );
                                        }
                                    },
                                );
                            }
                        }
                        "transpose_perm_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let rank = params.first().copied().unwrap_or(2);
                                let nd_params =
                                    resolve_params(params, param_dims, shape_env, 1 + 2 * rank)?;
                                let dims: Vec<usize> = nd_params[1..1 + rank].to_vec();
                                let perm: Vec<usize> = nd_params[1 + rank..1 + 2 * rank].to_vec();
                                let mut in_strides = vec![1usize; rank];
                                let mut out_strides = vec![1usize; rank];
                                for i in (0..rank - 1).rev() {
                                    in_strides[i] = in_strides[i + 1] * dims[i + 1];
                                }
                                for i in (0..rank - 1).rev() {
                                    out_strides[perm[i]] =
                                        out_strides[perm[i + 1]] * dims[perm[i + 1]];
                                }
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let total = out_f32.len();
                                        let mut lookup = Vec::with_capacity(total);
                                        for out_idx in 0..total {
                                            let mut in_idx = 0usize;
                                            let mut remaining = out_idx;
                                            for k in 0..rank {
                                                let coord = remaining / out_strides[perm[k]];
                                                remaining %= out_strides[perm[k]];
                                                in_idx += coord * in_strides[perm[k]];
                                            }
                                            lookup.push(in_idx);
                                        }
                                        #[cfg(not(feature = "parallel"))]
                                        {
                                            for (i, &in_idx) in lookup.iter().enumerate() {
                                                out_f32[i] = input[in_idx];
                                            }
                                        }
                                        #[cfg(feature = "parallel")]
                                        {
                                            use rayon::prelude::*;
                                            if total >= 4096 {
                                                out_f32.par_iter_mut().enumerate().for_each(
                                                    |(i, v)| {
                                                        *v = input[lookup[i]];
                                                    },
                                                );
                                            } else {
                                                for (i, &in_idx) in lookup.iter().enumerate() {
                                                    out_f32[i] = input[in_idx];
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "add_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x.max(0.0),
                            );
                        }
                        "sub_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x.max(0.0),
                            );
                        }
                        "mul_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x.max(0.0),
                            );
                        }
                        "softmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let softmax_params =
                                            resolve_params(params, param_dims, shape_env, 2)
                                                .unwrap_or_else(|_| {
                                                    Cow::Owned(vec![input.len(), 1])
                                                });
                                        let axis_dim_size = softmax_params[0].max(1);
                                        let stride =
                                            softmax_params.get(1).copied().unwrap_or(1).max(1);
                                        let num_rows = input.len() / axis_dim_size.max(1);
                                        softmax_f32(
                                            input,
                                            out_f32,
                                            axis_dim_size,
                                            stride,
                                            num_rows,
                                        );
                                    },
                                );
                            }
                        }
                        "biasadd" => {
                            if let [data_slice, bias_slice] = &input_slices[..] {
                                let (data, bias): (&[f32], &[f32]) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        ),
                                        bytemuck::cast_slice::<_, f32>(
                                            &d[bias_slice.offset
                                                ..bias_slice.offset + bias_slice.size],
                                        ),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let &[channel_stride] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "biasadd: expected params [channel_stride]".into(),
                                    ));
                                };
                                biasadd_f32(data, bias, out_f32, channel_stride);
                            }
                        }
                        "norm_f32" => {
                            let &[eps_bits, is_batch_norm] = &params[..] else {
                                return Err(BackendError::Dispatch(
                                    "norm_f32: expected params [eps_bits, is_batch_norm]".into(),
                                ));
                            };
                            let eps = f32::from_bits(eps_bits as u32);
                            if is_batch_norm == 1 {
                                // Batch norm (evaluation mode): use running_mean and running_var
                                if let [data_slice, weight_slice, bias_slice, mean_slice, var_slice] =
                                    &input_slices[..]
                                {
                                    type F32Slices<'a> =
                                        (&'a [f32], &'a [f32], &'a [f32], &'a [f32], &'a [f32]);
                                    let (data, weight, bias, running_mean, running_var): F32Slices<
                                        '_,
                                    > = {
                                        let d = arena.data_mut();
                                        (
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[data_slice.offset
                                                    ..data_slice.offset + data_slice.size],
                                            ),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[weight_slice.offset
                                                    ..weight_slice.offset + weight_slice.size],
                                            ),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[bias_slice.offset
                                                    ..bias_slice.offset + bias_slice.size],
                                            ),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[mean_slice.offset
                                                    ..mean_slice.offset + mean_slice.size],
                                            ),
                                            bytemuck::cast_slice::<_, f32>(
                                                &d[var_slice.offset
                                                    ..var_slice.offset + var_slice.size],
                                            ),
                                        )
                                    };
                                    let out_f32 = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, f32>(
                                            &mut d[out_start..out_end],
                                        )
                                    };
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    {
                                        use crate::backend::cpu::microkernels::has_avx2;
                                        if has_avx2() {
                                            // SAFETY: AVX2 feature checked by has_avx2()
                                            unsafe {
                                                crate::backend::cpu::microkernels::batch_norm_inference_f32_avx2(
                                                    data, weight, bias, running_mean, running_var,
                                                    out_f32, eps,
                                                );
                                            }
                                        } else {
                                            crate::backend::cpu::microkernels::batch_norm_inference_f32(
                                                data, weight, bias, running_mean, running_var,
                                                out_f32, eps,
                                            );
                                        }
                                    }
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    {
                                        crate::backend::cpu::microkernels::batch_norm_inference_f32(
                                            data,
                                            weight,
                                            bias,
                                            running_mean,
                                            running_var,
                                            out_f32,
                                            eps,
                                        );
                                    }
                                }
                            } else if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let row_size = input.len() / out_f32.len().max(1);
                                        norm_layernorm_f32(input, out_f32, row_size, eps);
                                    },
                                );
                            }
                        }
                        "div_relu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_relu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x.max(0.0),
                            );
                        }
                        // â”€â”€ Fused elementwise + GELU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        "add_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "sub_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "mul_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        "div_gelu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_gelu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| {
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    0.5 * x * (1.0 + t)
                                },
                            );
                        }
                        // â”€â”€ Fused elementwise + SiLU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        "add_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "add_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a + b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "sub_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "sub_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a - b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "mul_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "mul_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a * b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "div_silu_f32" => {
                            fused_binary_activation_dispatch(
                                "div_silu_f32",
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                |a, b| a / b,
                                |x| x / (1.0 + (-x).exp()),
                            );
                        }
                        "conv2d" | "conv2d_relu" | "conv2d_gelu" | "conv2d_silu" => {
                            let fused_act = match kernel_name.as_str() {
                                "conv2d_relu" => Some("relu"),
                                "conv2d_gelu" => Some("gelu"),
                                "conv2d_silu" => Some("silu"),
                                _ => None,
                            };
                            if let [input_slice, weight_slice] = &input_slices[..2] {
                                if params.len() < 9 {
                                    return Err(BackendError::Dispatch(
                                        "conv2d: expected at least 9 params".into(),
                                    ));
                                }
                                let stride = params[0];
                                let padding = params[1];
                                let dilation = params[2];
                                let groups = params[3];
                                let c = params[4];
                                let h = params[5];
                                let w = params[6];
                                let kh = params[7];
                                let kw = params[8];
                                let c_per_group = c / groups.max(1);
                                let f32_size = std::mem::size_of::<f32>();
                                let n_in_derived =
                                    (input_slice.size / f32_size) / (c * h * w).max(1);
                                let f_out_derived =
                                    (weight_slice.size / f32_size) / (c_per_group * kh * kw).max(1);
                                let (n_in, f_out) = if params.len() >= 15 {
                                    (params[9], params[10])
                                } else {
                                    (n_in_derived, f_out_derived)
                                };
                                let _h_out = (h + 2 * padding)
                                    .saturating_sub(dilation * (kh - 1) + 1)
                                    / stride
                                    + 1;
                                let _w_out = (w + 2 * padding)
                                    .saturating_sub(dilation * (kw - 1) + 1)
                                    / stride
                                    + 1;
                                let out_slice = BufferSlice::new(out_start, out_end - out_start);
                                // Build optional activation closure for fused conv+activation.
                                // The activation is applied inside the scatter loop, avoiding a
                                // separate memory round-trip over the output tensor.
                                let conv_act = fused_act.map(|act| match act {
                                    "relu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Relu
                                    }
                                    "gelu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Gelu
                                    }
                                    "silu" => {
                                        crate::backend::cpu::microkernels::ConvActivation::Silu
                                    }
                                    _ => unreachable!(),
                                });
                                // Borrow the kernel call with arena slices directly: input, weight,
                                // and bias (if present) are read-only, output is the only mut
                                // region. with_nary_f32_slices handles disjoint/overlap correctly
                                // and avoids the previous per-call `.to_vec()` copies of the input
                                // and weight tensors.
                                if let [input_s, weight_s, bias_s @ ..] = &input_slices[..] {
                                    let inputs_for_kernel: SmallVec<[BufferSlice; 4]> =
                                        if bias_s.is_empty() {
                                            smallvec![*input_s, *weight_s]
                                        } else {
                                            smallvec![*input_s, *weight_s, bias_s[0]]
                                        };
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs_for_kernel,
                                        out_slice,
                                        |inputs, out_f32| {
                                            let input = inputs[0];
                                            let weight = inputs[1];
                                            let bias = if inputs.len() >= 3 {
                                                inputs[2]
                                            } else {
                                                &[][..]
                                            };
                                            // Delegate to the im2col + GEMM microkernel
                                            // (falls back to tiled for small tensors).
                                            crate::backend::cpu::microkernels::conv2d_f32_im2col_gemm(
                                                input, weight, bias, out_f32, n_in, c, h, w, f_out,
                                                kh, kw, stride, padding, dilation, groups, conv_act,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        // â”€â”€ Conv2d Quantized (u4/u8) â€” i8 activation path (arena) â”€â”€
                        "conv2d_i4_i8" | "conv2d_i4_i8_relu" | "conv2d_i4_i8_gelu"
                        | "conv2d_i4_i8_silu" | "conv2d_i8_i8" | "conv2d_i8_i8_relu"
                        | "conv2d_i8_i8_gelu" | "conv2d_i8_i8_silu" => {
                            if input_slices.len() >= 2 {
                                let a_slice = &input_slices[0];
                                let w_slice = &input_slices[1];
                                let activation_payload: &[u8] =
                                    unsafe { arena.view_u8(a_slice.offset, a_slice.size) };
                                let raw: &[u8] =
                                    unsafe { arena.view_u8(w_slice.offset, w_slice.size) };
                                let bias_data: Option<&[f32]> = if input_slices.len() >= 3 {
                                    let b = &input_slices[2];
                                    Some(unsafe { arena.view_f32(b.offset, b.size) })
                                } else {
                                    None
                                };
                                if params.len() < 9 {
                                    return Err(BackendError::Dispatch("conv2d_i4_i8/u8_i8: expected at least 9 params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into()));
                                }
                                let stride = params[0];
                                let padding = params[1];
                                let dilation = params[2];
                                let groups = params[3];
                                let input_c = params[4];
                                let input_h = params[5];
                                let input_w = params[6];
                                let kernel_h = params[7];
                                let kernel_w = params[8];
                                if stride == 0
                                    || dilation == 0
                                    || groups == 0
                                    || kernel_h == 0
                                    || kernel_w == 0
                                    || !input_c.is_multiple_of(groups)
                                {
                                    return Err(BackendError::Dispatch(
                                        "conv2d_i4_i8/u8_i8: invalid stride, dilation, groups, channels, or kernel dimensions".into(),
                                    ));
                                }
                                let meta = weight_meta.clone().ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv2d_i4_i8/u8_i8: missing weight_meta".into(),
                                    )
                                })?;
                                if output_slice.offset % std::mem::align_of::<f32>() != 0
                                    || output_slice.size % std::mem::size_of::<f32>() != 0
                                {
                                    return Err(BackendError::Dispatch(
                                        "conv2d_i4_i8/u8_i8: output slice is not f32-aligned"
                                            .into(),
                                    ));
                                }
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };

                                if activation_payload.len() < 8 {
                                    return Err(BackendError::Dispatch(format!(
                                        "conv2d_i4_i8/u8_i8: activation payload has {} bytes, expected an 8-byte affine header",
                                        activation_payload.len()
                                    )));
                                }
                                let scale = f32::from_le_bytes([
                                    activation_payload[0],
                                    activation_payload[1],
                                    activation_payload[2],
                                    activation_payload[3],
                                ]);
                                let zp = f32::from_le_bytes([
                                    activation_payload[4],
                                    activation_payload[5],
                                    activation_payload[6],
                                    activation_payload[7],
                                ]);
                                if !scale.is_finite() || scale <= 0.0 || !zp.is_finite() {
                                    return Err(BackendError::Dispatch(
                                        "conv2d_i4_i8/u8_i8: activation affine metadata must be finite with a positive scale".into(),
                                    ));
                                }
                                let affine = microkernels::I8ActivationAffine { scale, zero: zp };
                                // Zero-copy view of i8 activation data (payload starts at byte 8)
                                let payload = &activation_payload[8..];
                                let image_size = input_c
                                    .checked_mul(input_h)
                                    .and_then(|value| value.checked_mul(input_w))
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "conv2d_i4_i8/u8_i8: activation shape overflows".into(),
                                        )
                                    })?;
                                if image_size == 0 || !payload.len().is_multiple_of(image_size) {
                                    return Err(BackendError::Dispatch(format!(
                                        "conv2d_i4_i8/u8_i8: activation data length {} is not a whole number of {image_size}-element images",
                                        payload.len()
                                    )));
                                }
                                let act_i8: &[i8] = bytemuck::cast_slice(payload);

                                let oc = meta.shape[0];
                                let c_per_g = input_c / groups;
                                let oc_per_g = oc / groups;
                                let h_out = packed_conv::conv_out_size(
                                    input_h, kernel_h, stride, padding, dilation,
                                );
                                let w_out = packed_conv::conv_out_size(
                                    input_w, kernel_w, stride, padding, dilation,
                                );
                                let num_pixels = h_out * w_out;
                                let k = c_per_g * kernel_h * kernel_w;
                                let n = act_i8.len() / (input_c * input_h * input_w).max(1);

                                let fused_act = if kernel_name.contains("_relu") {
                                    PreparedActivation::Relu
                                } else if kernel_name.contains("_gelu") {
                                    PreparedActivation::Gelu
                                } else if kernel_name.contains("_silu") {
                                    PreparedActivation::Silu
                                } else {
                                    PreparedActivation::None
                                };

                                let bit_width = meta.bit_width;
                                let inner: usize = meta.shape[1..].iter().product();
                                let mut col_buf = tls_alloc_zeroed_i8(num_pixels * k);
                                let mut payload = tls_alloc_u8(8 + num_pixels * k);
                                payload.clear();
                                let mut packed_act =
                                    tls_alloc_zeroed_i8x4(num_pixels * inner.div_ceil(4));
                                let mut temp = tls_alloc_zeroed_f32(num_pixels * oc_per_g);

                                for nn in 0..n {
                                    let act_base = nn * input_c * input_h * input_w;
                                    let out_base = nn * oc * num_pixels;
                                    for g in 0..groups {
                                        let g_c_off = g * c_per_g;
                                        let g_oc_off = g * oc_per_g;
                                        let act_group =
                                            &act_i8[act_base + g_c_off * input_h * input_w..];
                                        unsafe {
                                            packed_conv::im2col_i8(
                                                act_group,
                                                c_per_g,
                                                input_h,
                                                input_w,
                                                kernel_h,
                                                kernel_w,
                                                stride,
                                                padding,
                                                dilation,
                                                col_buf.as_mut_slice(),
                                            );
                                        }
                                        if bit_width == 4 {
                                            // U4 path: build flat-i8 payload into reusable buffer
                                            payload.clear();
                                            payload.extend_from_slice(&affine.scale.to_le_bytes());
                                            payload.extend_from_slice(&affine.zero.to_le_bytes());
                                            let col_u8: &[u8] = bytemuck::cast_slice(&col_buf);
                                            payload.extend_from_slice(col_u8);
                                            let kp = inner.div_ceil(I4x8::ITEMS);
                                            let w_sl = get_or_cache_packed::<I4x8>(
                                                w_slice.offset,
                                                w_slice.size,
                                                raw,
                                            );
                                            let w_slice =
                                                &w_sl[g_oc_off * kp..(g_oc_off + oc_per_g) * kp];
                                            let local_scales = if meta.scales.len() > 1 {
                                                meta.scales[g_oc_off..g_oc_off + oc_per_g].to_vec()
                                            } else {
                                                meta.scales.clone()
                                            };
                                            let local_zps = if meta.dequant_offsets.len() > 1 {
                                                meta.dequant_offsets[g_oc_off..g_oc_off + oc_per_g]
                                                    .to_vec()
                                            } else {
                                                meta.dequant_offsets.clone()
                                            };
                                            let pt = PackedTensor::from_raw(
                                                w_slice.to_vec(),
                                                vec![oc_per_g, inner],
                                                local_scales,
                                                local_zps,
                                            );
                                            microkernels::gemm_cpu_flat_i8_i4x8(
                                                &pt,
                                                &payload,
                                                temp.as_mut_slice(),
                                                num_pixels,
                                                k,
                                                oc_per_g,
                                            );
                                            for pixel in 0..num_pixels {
                                                for f in 0..oc_per_g {
                                                    let mut val = temp[pixel * oc_per_g + f];
                                                    if let Some(bias) = bias_data {
                                                        if g_oc_off + f < bias.len() {
                                                            val += bias[g_oc_off + f];
                                                        }
                                                    }
                                                    match fused_act {
                                                        PreparedActivation::Relu => {
                                                            val = val.max(0.0)
                                                        }
                                                        PreparedActivation::Silu => {
                                                            val = val / (1.0 + (-val).exp())
                                                        }
                                                        PreparedActivation::Gelu => {
                                                            val = val
                                                                * 0.5
                                                                * (1.0
                                                                    + (val
                                                                        * 0.797_884_6
                                                                        * (1.0
                                                                            + 0.044715
                                                                                * val
                                                                                * val))
                                                                        .tanh())
                                                        }
                                                        PreparedActivation::None => {}
                                                    }
                                                    out_f32[out_base
                                                        + (g_oc_off + f) * num_pixels
                                                        + pixel] = val;
                                                }
                                            }
                                        } else {
                                            // U8 path: pack i8 â†’ I8x4 (lossless), use fast packed SWAR GEMM
                                            let kp = inner.div_ceil(I8x4::ITEMS);
                                            let w_sl = get_or_cache_packed::<I8x4>(
                                                w_slice.offset,
                                                w_slice.size,
                                                raw,
                                            );
                                            let w_slice =
                                                &w_sl[g_oc_off * kp..(g_oc_off + oc_per_g) * kp];
                                            let per_channel_w = meta.scales.len() > 1;
                                            let per_channel_zp = meta.dequant_offsets.len() > 1;
                                            let w_scales = if per_channel_w {
                                                &meta.scales[g_oc_off..g_oc_off + oc_per_g]
                                            } else {
                                                &meta.scales
                                            };
                                            let w_zps = if per_channel_zp {
                                                &meta.dequant_offsets[g_oc_off..g_oc_off + oc_per_g]
                                            } else {
                                                &meta.dequant_offsets
                                            };
                                            // Pack i8 â†’ I8x4 into reusable buffer
                                            packed_conv::pack_i8_col_to_i8x4(
                                                &col_buf,
                                                num_pixels,
                                                k,
                                                packed_act.as_mut_slice(),
                                            );
                                            let bias_group = bias_data.and_then(|bias| {
                                                if g_oc_off < bias.len() {
                                                    Some(&bias[g_oc_off..g_oc_off + oc_per_g])
                                                } else {
                                                    None
                                                }
                                            });
                                            // Raw packed GEMM â€” no PackedTensor wrapping, no allocations
                                            packed_conv::gemm_packed_i8x4_fused_raw(
                                                &packed_act,
                                                num_pixels,
                                                k,
                                                affine.scale,
                                                affine.zero,
                                                w_slice,
                                                oc_per_g,
                                                w_scales,
                                                w_zps,
                                                bias_group,
                                                fused_act,
                                                temp.as_mut_slice(),
                                            );
                                            for pixel in 0..num_pixels {
                                                for f in 0..oc_per_g {
                                                    out_f32[out_base
                                                        + (g_oc_off + f) * num_pixels
                                                        + pixel] = temp[pixel * oc_per_g + f];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // â”€â”€ Conv2d Quantized (u4/u8) â€” FP32 activation path (arena) â”€â”€
                        "conv2d_f4" | "conv2d_f4_relu" | "conv2d_f4_gelu" | "conv2d_f4_silu"
                        | "conv2d_u4" | "conv2d_u4_relu" | "conv2d_u4_gelu" | "conv2d_u4_silu"
                        | "conv2d_u8" | "conv2d_u8_relu" | "conv2d_u8_gelu" | "conv2d_u8_silu"
                        | "conv2d_f8" | "conv2d_f8_relu" | "conv2d_f8_gelu" | "conv2d_f8_silu"
                        | "conv2d_f8r" | "conv2d_f8r_relu" | "conv2d_f8r_gelu"
                        | "conv2d_f8r_silu" | "conv2d_i4" | "conv2d_i4_relu" | "conv2d_i4_gelu"
                        | "conv2d_i4_silu" | "conv2d_i8" | "conv2d_i8_relu" | "conv2d_i8_gelu"
                        | "conv2d_i8_silu" => {
                            // Quantized conv2d using SWAR packed kernels.
                            // input_slices: [activation (f32), weight (packed)]  optional: [bias (f32)]
                            // weight_meta carries bit_width, shape=[OC, IC_per_group*KH*KW], scales[], zero_points[]
                            if input_slices.len() >= 2 {
                                let a_slice = &input_slices[0];
                                let w_slice = &input_slices[1];
                                // Zero-copy views into arena (no to_vec() allocations)
                                let input_data: &[f32] =
                                    unsafe { arena.view_f32(a_slice.offset, a_slice.size) };
                                let raw: &[u8] =
                                    unsafe { arena.view_u8(w_slice.offset, w_slice.size) };
                                let bias_data: Option<&[f32]> = if input_slices.len() >= 3 {
                                    let b = &input_slices[2];
                                    Some(unsafe { arena.view_f32(b.offset, b.size) })
                                } else {
                                    None
                                };
                                if params.len() < 9 {
                                    return Err(BackendError::Dispatch("conv2d_i4/u8: expected at least 9 params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into()));
                                }
                                let stride = params[0];
                                let padding = params[1];
                                let dilation = params[2];
                                let groups = params[3];
                                let input_c = params[4];
                                let input_h = params[5];
                                let input_w = params[6];
                                let kernel_h = params[7];
                                let kernel_w = params[8];
                                if groups == 0 {
                                    return Err(BackendError::Dispatch(
                                        "conv2d_i4/u8: groups=0".into(),
                                    ));
                                }
                                let meta = weight_meta.clone().ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv2d_i4/u8: missing weight_meta".into(),
                                    )
                                })?;
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let (oc, inner) = if meta.shape.len() >= 4 {
                                    (meta.shape[0], meta.shape[1..].iter().product::<usize>())
                                } else {
                                    (meta.shape[0], meta.shape[1..].iter().product::<usize>())
                                };
                                let n = input_data.len() / (input_c * input_h * input_w).max(1);
                                let bias_opt = bias_data.filter(|b| !b.is_empty());
                                let fused_act = if kernel_name.contains("_relu") {
                                    PreparedActivation::Relu
                                } else if kernel_name.contains("_gelu") {
                                    PreparedActivation::Gelu
                                } else if kernel_name.contains("_silu") {
                                    PreparedActivation::Silu
                                } else {
                                    PreparedActivation::None
                                };
                                macro_rules! dispatch_packed_conv {
                                    ($PackedType:ty, $fn:path) => {{
                                        let packed_data = get_or_cache_packed::<$PackedType>(
                                            w_slice.offset,
                                            w_slice.size,
                                            &raw,
                                        );
                                        let mut pt = PackedTensor::from_raw_arc(
                                            packed_data,
                                            vec![oc, inner],
                                            meta.scales.clone(),
                                            meta.dequant_offsets.clone(),
                                        );
                                        pt.quant_block_size = meta.quant_block_size;
                                        unsafe {
                                            $fn(
                                                &input_data,
                                                n,
                                                input_c,
                                                input_h,
                                                input_w,
                                                &pt,
                                                bias_opt,
                                                stride,
                                                padding,
                                                dilation,
                                                groups,
                                                kernel_h,
                                                kernel_w,
                                                fused_act,
                                                out_f32,
                                            );
                                        }
                                    }};
                                }
                                macro_rules! dispatch_packed_conv_cached {
                                    ($PackedType:ty) => {{
                                        let pt = get_or_cache_f32_weights::<$PackedType>(
                                            w_slice.offset,
                                            w_slice.size,
                                            &raw,
                                            &[oc, inner],
                                            &meta.scales,
                                            &meta.dequant_offsets,
                                            meta.quant_block_size,
                                            &meta.codebooks,
                                        );
                                        let f32_weights = pt.get_or_init_f32_weights();
                                        let conv_act = match fused_act {
                                            PreparedActivation::None => None,
                                            PreparedActivation::Relu => {
                                                Some(microkernels::ConvActivation::Relu)
                                            }
                                            PreparedActivation::Gelu => {
                                                Some(microkernels::ConvActivation::Gelu)
                                            }
                                            PreparedActivation::Silu => {
                                                Some(microkernels::ConvActivation::Silu)
                                            }
                                        };
                                        let bias_slice = bias_opt.unwrap_or(&[]);
                                        microkernels::conv::conv2d_f32_im2col_gemm(
                                            &input_data,
                                            f32_weights,
                                            bias_slice,
                                            out_f32,
                                            n,
                                            input_c,
                                            input_h,
                                            input_w,
                                            oc,
                                            kernel_h,
                                            kernel_w,
                                            stride,
                                            padding,
                                            dilation,
                                            groups,
                                            conv_act,
                                        );
                                    }};
                                }
                                if kernel_name.starts_with("conv2d_u4") {
                                    dispatch_packed_conv_cached!(U4x8);
                                } else if kernel_name.starts_with("conv2d_u8") {
                                    dispatch_packed_conv_cached!(U8x4);
                                } else if kernel_name.starts_with("conv2d_f4") {
                                    dispatch_packed_conv_cached!(F4x8);
                                } else if kernel_name.starts_with("conv2d_f8r") {
                                    dispatch_packed_conv_cached!(F8x4R);
                                } else if kernel_name.starts_with("conv2d_f8") {
                                    dispatch_packed_conv_cached!(F8x4);
                                } else if meta.bit_width == 4 && !meta.codebooks.is_empty() {
                                    dispatch_packed_conv_cached!(I4x8);
                                } else if meta.bit_width == 4 {
                                    dispatch_packed_conv!(I4x8, packed_conv::conv2d_packed_i4x8);
                                } else {
                                    dispatch_packed_conv!(I8x4, packed_conv::conv2d_packed_i8x4);
                                }
                            }
                        }
                        "concat" => {
                            if !input_slices.is_empty() && params.len() >= 3 {
                                let _axis = params[0];
                                let _inner_stride = params[1];
                                let outer_count = params[2];
                                // For each input, compute block_size = elements per outer position.
                                let num_inputs = input_slices.len();
                                let mut block_sizes: Vec<usize> = Vec::with_capacity(num_inputs);
                                for slice in input_slices {
                                    let elems = slice.size / std::mem::size_of::<f32>();
                                    block_sizes.push(elems / outer_count.max(1));
                                }
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let mut output_offset = 0;
                                let d = arena.data_mut();
                                for outer_pos in 0..outer_count {
                                    for (si, slice) in input_slices.iter().enumerate() {
                                        let input_data = bytemuck::cast_slice::<_, f32>(
                                            &d[slice.offset..slice.offset + slice.size],
                                        );
                                        let bs = block_sizes[si];
                                        let src_start = outer_pos * bs;
                                        let src_end = (src_start + bs).min(input_data.len());
                                        let copy_len = src_end - src_start;
                                        let dst_end = (output_offset + copy_len).min(out_f32.len());
                                        let actual_copy = dst_end - output_offset;
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::concat_f32_avx2(
                                                    &input_data[src_start..src_start + actual_copy],
                                                    out_f32,
                                                    output_offset,
                                                );
                                            }
                                        } else {
                                            out_f32[output_offset..dst_end].copy_from_slice(
                                                &input_data[src_start..src_start + actual_copy],
                                            );
                                        }
                                        #[cfg(not(all(
                                            feature = "simd",
                                            target_arch = "x86_64"
                                        )))]
                                        out_f32[output_offset..dst_end].copy_from_slice(
                                            &input_data[src_start..src_start + actual_copy],
                                        );
                                        #[cfg(feature = "debug_canary")]
                                        eprintln!(
                                            "[FNN_DBG_CONCAT] out=[{},{}) outer={} input[{}]: off={} sz={} block={} copy={}",
                                            out_start, out_end, outer_pos, si, slice.offset, slice.size, bs, copy_len
                                        );
                                        output_offset += copy_len;
                                    }
                                }
                            } else if !input_slices.is_empty() {
                                // Fallback: flat concat (legacy, no axis info)
                                let mut output_offset = 0;
                                for slice in input_slices {
                                    let input_data =
                                        unsafe { arena.view_f32(slice.offset, slice.size) };
                                    let out_f32 = unsafe {
                                        arena.view_f32_mut(out_start, out_end - out_start)
                                    };
                                    let end = (output_offset + input_data.len()).min(out_f32.len());
                                    let actual_copy = end - output_offset;
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    if microkernels::simd_avx2_available() {
                                        unsafe {
                                            microkernels::concat_f32_avx2(
                                                &input_data[..actual_copy],
                                                out_f32,
                                                output_offset,
                                            );
                                        }
                                    } else {
                                        out_f32[output_offset..end]
                                            .copy_from_slice(&input_data[..actual_copy]);
                                    }
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    out_f32[output_offset..end]
                                        .copy_from_slice(&input_data[..actual_copy]);
                                    #[cfg(feature = "debug_canary")]
                                    eprintln!(
                                        "[FNN_DBG_CONCAT] out=[{},{}) input: off={} sz={} numel={} (flat fallback)",
                                        out_start, out_end, slice.offset, slice.size, input_data.len()
                                    );
                                    output_offset += input_data.len();
                                }
                            }
                        }

                        "pool_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input_end = input_slice.offset + input_slice.size;
                                let overlaps = arena::ranges_overlap(
                                    input_slice.offset,
                                    input_end,
                                    out_start,
                                    out_end,
                                );
                                let mut input_copy;
                                let input: &[f32] = if overlaps {
                                    let d = arena.data_mut();
                                    let src = bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_end],
                                    );
                                    input_copy = tls_alloc_f32(src.len());
                                    input_copy.copy_from_slice(src);
                                    &input_copy
                                } else {
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) }
                                };
                                // params: [kernel, stride, padding, is_max, N, C, H, W]
                                let &[kernel, stride_val, padding_val, is_max, n, c, h, w] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch("pool_f32: expected params [kernel, stride, padding, is_max, N, C, H, W]".into()));
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let h_out =
                                    (h + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                let w_out =
                                    (w + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                #[cfg(feature = "parallel")]
                                let hw_out = h_out * w_out;
                                // â”€â”€ Sequential path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                                #[cfg(not(feature = "parallel"))]
                                {
                                    if is_max == 1 {
                                        let indices_out: Option<&mut [i64]> =
                                            secondary_output_slice.as_ref().map(|sec_slice| {
                                                let d = arena.data_mut();
                                                bytemuck::cast_slice_mut::<_, i64>(
                                                    &mut d[sec_slice.offset
                                                        ..sec_slice.offset + sec_slice.size],
                                                )
                                            });
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::pool_max_f32_avx2(
                                                    &input,
                                                    out_f32,
                                                    n,
                                                    c,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                    indices_out,
                                                );
                                            }
                                        } else {
                                            microkernels::pool_max_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                                indices_out,
                                            );
                                        }
                                        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                        {
                                            microkernels::pool_max_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                                indices_out,
                                            );
                                        }
                                    } else {
                                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                        if microkernels::simd_avx2_available() {
                                            unsafe {
                                                microkernels::pool_avg_f32_avx2(
                                                    &input,
                                                    out_f32,
                                                    n,
                                                    c,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                );
                                            }
                                        } else {
                                            microkernels::pool_avg_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                            );
                                        }
                                        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                        {
                                            microkernels::pool_avg_f32_scalar(
                                                &input,
                                                out_f32,
                                                n,
                                                c,
                                                h,
                                                w,
                                                kernel,
                                                stride_val,
                                                padding_val,
                                                h_out,
                                                w_out,
                                            );
                                        }
                                    }
                                }
                                // â”€â”€ Parallel path (rayon) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let nc = n * c;
                                    let input_ptr_val = input.as_ptr() as usize;
                                    let out_ptr_val = out_f32.as_mut_ptr() as usize;
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    let use_simd = microkernels::simd_avx2_available();
                                    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                                    let use_simd = false;
                                    if is_max == 1 {
                                        let (idx_ptr_val, has_indices) =
                                            if let Some(sec_slice) = secondary_output_slice {
                                                let d = arena.data_mut();
                                                let idx_end = sec_slice.offset + sec_slice.size;
                                                let idx_slice = bytemuck::cast_slice_mut::<_, i64>(
                                                    &mut d[sec_slice.offset..idx_end],
                                                );
                                                (idx_slice.as_mut_ptr() as usize, true)
                                            } else {
                                                (0usize, false)
                                            };
                                        (0..nc).into_par_iter().for_each(|nc_idx| {
                                            let nn = nc_idx / c;
                                            let cc = nc_idx % c;
                                            let inp = unsafe {
                                                std::slice::from_raw_parts(
                                                    (input_ptr_val
                                                        + (nn * (c * h * w) + cc * (h * w))
                                                            * std::mem::size_of::<f32>())
                                                        as *const f32,
                                                    h * w,
                                                )
                                            };
                                            let out = unsafe {
                                                std::slice::from_raw_parts_mut(
                                                    (out_ptr_val
                                                        + (nn * (c * hw_out) + cc * hw_out)
                                                            * std::mem::size_of::<f32>())
                                                        as *mut f32,
                                                    hw_out,
                                                )
                                            };
                                            let idx: Option<&mut [i64]> = if has_indices {
                                                Some(unsafe {
                                                    std::slice::from_raw_parts_mut(
                                                        (idx_ptr_val
                                                            + (nn * (c * hw_out) + cc * hw_out)
                                                                * core::mem::size_of::<i64>())
                                                            as *mut i64,
                                                        hw_out,
                                                    )
                                                })
                                            } else {
                                                None
                                            };
                                            if use_simd {
                                                #[cfg(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                ))]
                                                unsafe {
                                                    microkernels::pool_max_f32_avx2(
                                                        inp,
                                                        out,
                                                        1,
                                                        1,
                                                        h,
                                                        w,
                                                        kernel,
                                                        stride_val,
                                                        padding_val,
                                                        h_out,
                                                        w_out,
                                                        idx,
                                                    );
                                                }
                                                #[cfg(not(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                )))]
                                                {
                                                    unreachable!();
                                                }
                                            } else {
                                                microkernels::pool_max_f32_scalar(
                                                    inp,
                                                    out,
                                                    1,
                                                    1,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                    idx,
                                                );
                                            }
                                        });
                                    } else {
                                        (0..nc).into_par_iter().for_each(|nc_idx| {
                                            let nn = nc_idx / c;
                                            let cc = nc_idx % c;
                                            let inp = unsafe {
                                                std::slice::from_raw_parts(
                                                    (input_ptr_val
                                                        + (nn * (c * h * w) + cc * (h * w))
                                                            * std::mem::size_of::<f32>())
                                                        as *const f32,
                                                    h * w,
                                                )
                                            };
                                            let out = unsafe {
                                                std::slice::from_raw_parts_mut(
                                                    (out_ptr_val
                                                        + (nn * (c * hw_out) + cc * hw_out)
                                                            * std::mem::size_of::<f32>())
                                                        as *mut f32,
                                                    hw_out,
                                                )
                                            };
                                            if use_simd {
                                                #[cfg(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                ))]
                                                unsafe {
                                                    microkernels::pool_avg_f32_avx2(
                                                        inp,
                                                        out,
                                                        1,
                                                        1,
                                                        h,
                                                        w,
                                                        kernel,
                                                        stride_val,
                                                        padding_val,
                                                        h_out,
                                                        w_out,
                                                    );
                                                }
                                                #[cfg(not(all(
                                                    feature = "simd",
                                                    target_arch = "x86_64"
                                                )))]
                                                {
                                                    unreachable!();
                                                }
                                            } else {
                                                microkernels::pool_avg_f32_scalar(
                                                    inp,
                                                    out,
                                                    1,
                                                    1,
                                                    h,
                                                    w,
                                                    kernel,
                                                    stride_val,
                                                    padding_val,
                                                    h_out,
                                                    w_out,
                                                );
                                            }
                                        });
                                    }
                                }
                            }
                        }
                        "pad_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                // Zero-copy view of input
                                let input: &[f32] =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                // Simple flat copy: the caller right-sizes the output buffer.
                                // Output includes padding zeros already allocated.
                                let end = input.len().min(out_f32.len());
                                out_f32[..end].copy_from_slice(&input[..end]);
                            }
                        }
                        "gather" => {
                            if let [data_slice, indices_slice] = &input_slices[..] {
                                // Zero-copy views of input and indices
                                let input: &[f32] =
                                    unsafe { arena.view_f32(data_slice.offset, data_slice.size) };
                                let indices: &[f32] = unsafe {
                                    arena.view_f32(indices_slice.offset, indices_slice.size)
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let axis = if !params.is_empty() { params[0] } else { 0 };
                                let inner = if axis == 0 {
                                    input.len() / out_f32.len().max(1)
                                } else {
                                    1
                                };
                                for i in 0..out_f32.len() {
                                    let idx_idx = i.checked_div(inner).unwrap_or(i);
                                    let idx = indices[idx_idx.min(indices.len().saturating_sub(1))]
                                        as usize;
                                    let src = idx * inner + (i % inner);
                                    out_f32[i] = if src < input.len() { input[src] } else { 0.0 };
                                }
                            }
                        }
                        "slice_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input: &[f32] =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let &[_dim, start, end, stride] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "slice_f32: expected params [dim, start, end, stride]"
                                            .into(),
                                    ));
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                // General strided slice along dimension `dim`.
                                // input layout (row-major): outer * dim_size * stride
                                // output layout:             outer * (end-start) * stride
                                let in_len = input.len();
                                let out_len = out_f32.len();
                                let range_len = (end - start).max(1);
                                // dim_size = in_len * range_len / out_len
                                let dim_size = if out_len > 0 {
                                    (in_len * range_len) / out_len
                                } else {
                                    0
                                };
                                let outer = if dim_size > 0 && stride > 0 {
                                    in_len / dim_size / stride
                                } else {
                                    1
                                };
                                let slice_elems = range_len * stride;
                                for i in 0..outer {
                                    let src_off = i * dim_size * stride + start * stride;
                                    let dst_off = i * slice_elems;
                                    let copy_len = slice_elems.min(in_len.saturating_sub(src_off));
                                    if copy_len > 0 {
                                        out_f32[dst_off..(dst_off + copy_len)]
                                            .copy_from_slice(&input[src_off..(src_off + copy_len)]);
                                    }
                                }
                            }
                        }
                        "scatter_nd" => {
                            if let [data_slice, indices_slice, updates_slice] = &input_slices[..] {
                                let data: &[f32] =
                                    unsafe { arena.view_f32(data_slice.offset, data_slice.size) };
                                let indices_f32: &[f32] = unsafe {
                                    arena.view_f32(indices_slice.offset, indices_slice.size)
                                };
                                let updates: &[f32] = unsafe {
                                    arena.view_f32(updates_slice.offset, updates_slice.size)
                                };
                                let out_f32: &mut [f32] =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                out_f32.copy_from_slice(data);
                                if let Some((&index_depth, data_dims)) = params.split_first() {
                                    let data_rank = data_dims.len();
                                    if index_depth > 0
                                        && index_depth <= data_rank
                                        && indices_f32.len() >= index_depth
                                    {
                                        let num_indices = indices_f32.len() / index_depth;
                                        let inner_size: usize =
                                            data_dims[index_depth..].iter().product();
                                        for i in 0..num_indices {
                                            let mut linear_offset = 0usize;
                                            for j in 0..index_depth {
                                                let idx = indices_f32[i * index_depth + j] as usize;
                                                let mut stride = 1usize;
                                                for k in (j + 1)..data_rank {
                                                    stride *= data_dims[k];
                                                }
                                                linear_offset += idx * stride;
                                            }
                                            let update_start = i * inner_size;
                                            let update_end = update_start + inner_size;
                                            if linear_offset + inner_size <= out_f32.len()
                                                && update_end <= updates.len()
                                            {
                                                out_f32[linear_offset..linear_offset + inner_size]
                                                    .copy_from_slice(
                                                        &updates[update_start..update_end],
                                                    );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "conv1d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let input: &[f32] =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let weight: &[f32] = unsafe {
                                    arena.view_f32(weight_slice.offset, weight_slice.size)
                                };
                                let &[stride, padding, c, w, kw] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "conv1d: expected params [stride, padding, input_c, input_w, kernel_w]".into(),
                                    ));
                                };
                                let n = input.len() / (c * w).max(1);
                                let f = weight.len() / (c * kw).max(1);
                                let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
                                let out_f32: &mut [f32] =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                for nn in 0..n {
                                    for ff in 0..f {
                                        for ww in 0..w_out {
                                            let mut sum = 0.0f32;
                                            for cc in 0..c {
                                                for kkw in 0..kw {
                                                    let w_in = ww * stride + kkw;
                                                    if w_in >= padding {
                                                        let w_in_s = w_in - padding;
                                                        if w_in_s < w {
                                                            sum += input
                                                                [nn * (c * w) + cc * w + w_in_s]
                                                                * weight
                                                                    [ff * (c * kw) + cc * kw + kkw];
                                                        }
                                                    }
                                                }
                                            }
                                            out_f32[nn * (f * w_out) + ff * w_out + ww] = sum;
                                        }
                                    }
                                }
                            }
                        }
                        "conv3d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = unsafe {
                                    (
                                        arena.view_f32(input_slice.offset, input_slice.size),
                                        arena.view_f32(weight_slice.offset, weight_slice.size),
                                    )
                                };
                                let &[stride, padding, dilation, c, d, h, w, kd, kh, kw] =
                                    &params[..]
                                else {
                                    return Err(BackendError::Dispatch(
                                        "conv3d: expected params [stride, padding, dilation, input_c, input_d, input_h, input_w, kernel_d, kernel_h, kernel_w]".into(),
                                    ));
                                };
                                let n = input.len() / (c * d * h * w).max(1);
                                let f = weight.len() / (c * kd * kh * kw).max(1);
                                let d_out = (d + 2 * padding)
                                    .saturating_sub(dilation * (kd - 1) + 1)
                                    / stride
                                    + 1;
                                let h_out = (h + 2 * padding)
                                    .saturating_sub(dilation * (kh - 1) + 1)
                                    / stride
                                    + 1;
                                let w_out = (w + 2 * padding)
                                    .saturating_sub(dilation * (kw - 1) + 1)
                                    / stride
                                    + 1;
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                for nn in 0..n {
                                    for ff in 0..f {
                                        for dd in 0..d_out {
                                            for hh in 0..h_out {
                                                for ww in 0..w_out {
                                                    let mut sum = 0.0f32;
                                                    for cc in 0..c {
                                                        for kkd in 0..kd {
                                                            for kkh in 0..kh {
                                                                for kkw in 0..kw {
                                                                    let d_in = dd * stride
                                                                        + kkd * dilation;
                                                                    let h_in = hh * stride
                                                                        + kkh * dilation;
                                                                    let w_in = ww * stride
                                                                        + kkw * dilation;
                                                                    if d_in >= padding
                                                                        && h_in >= padding
                                                                        && w_in >= padding
                                                                    {
                                                                        let d_in_s = d_in - padding;
                                                                        let h_in_s = h_in - padding;
                                                                        let w_in_s = w_in - padding;
                                                                        if d_in_s < d
                                                                            && h_in_s < h
                                                                            && w_in_s < w
                                                                        {
                                                                            let input_idx = nn
                                                                                * (c * d * h * w)
                                                                                + cc * (d * h * w)
                                                                                + d_in_s * (h * w)
                                                                                + h_in_s * w
                                                                                + w_in_s;
                                                                            let weight_idx = ff
                                                                                * (c * kd
                                                                                    * kh
                                                                                    * kw)
                                                                                + cc * (kd
                                                                                    * kh
                                                                                    * kw)
                                                                                + kkd * (kh * kw)
                                                                                + kkh * kw
                                                                                + kkw;
                                                                            sum += input[input_idx]
                                                                                * weight
                                                                                    [weight_idx];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    let out_idx = nn * (f * d_out * h_out * w_out)
                                                        + ff * (d_out * h_out * w_out)
                                                        + dd * (h_out * w_out)
                                                        + hh * w_out
                                                        + ww;
                                                    out_f32[out_idx] = sum;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "conv_transpose2d" => {
                            if input_slices.len() != 2 || params.len() != 7 {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: expected two inputs and seven parameters"
                                        .into(),
                                ));
                            }
                            let [stride, padding, c, hin, win, kh, kw] = params.as_slice() else {
                                unreachable!();
                            };
                            if *stride == 0
                                || *c == 0
                                || *hin == 0
                                || *win == 0
                                || *kh == 0
                                || *kw == 0
                            {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: stride, channels, and dimensions must be positive".into(),
                                ));
                            }
                            let checked_elements = |dims: &[usize], label: &str| {
                                dims.iter().try_fold(1usize, |acc, dim| {
                                    acc.checked_mul(*dim).ok_or_else(|| {
                                        BackendError::Dispatch(format!(
                                            "conv_transpose2d: {label} size overflows"
                                        ))
                                    })
                                })
                            };
                            let image_elements =
                                checked_elements(&[*c, *hin, *win], "input image")?;
                            let kernel_elements = checked_elements(&[*c, *kh, *kw], "kernel")?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if input_slices.iter().any(|slice| {
                                !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                    || !slice.size.is_multiple_of(scalar_bytes)
                            }) || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !(out_end - out_start).is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: slices must contain aligned f32 scalars"
                                        .into(),
                                ));
                            }
                            let input_elements = input_slices[0].size / scalar_bytes;
                            let weight_elements = input_slices[1].size / scalar_bytes;
                            if !input_elements.is_multiple_of(image_elements)
                                || !weight_elements.is_multiple_of(kernel_elements)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: input or weight storage is incomplete"
                                        .into(),
                                ));
                            }
                            let n = input_elements / image_elements;
                            let f = weight_elements / kernel_elements;
                            if n == 0 || f == 0 {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: batch and output channels must be positive"
                                        .into(),
                                ));
                            }
                            let padded_h = (hin - 1)
                                .checked_mul(*stride)
                                .and_then(|value| value.checked_add(*kh))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv_transpose2d: output height overflows".into(),
                                    )
                                })?;
                            let padded_w = (win - 1)
                                .checked_mul(*stride)
                                .and_then(|value| value.checked_add(*kw))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv_transpose2d: output width overflows".into(),
                                    )
                                })?;
                            let double_padding = padding.checked_mul(2).ok_or_else(|| {
                                BackendError::Dispatch("conv_transpose2d: padding overflows".into())
                            })?;
                            let h_out = padded_h.checked_sub(double_padding).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "conv_transpose2d: padding exceeds output height".into(),
                                )
                            })?;
                            let w_out = padded_w.checked_sub(double_padding).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "conv_transpose2d: padding exceeds output width".into(),
                                )
                            })?;
                            if h_out == 0 || w_out == 0 {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: output dimensions must be positive".into(),
                                ));
                            }
                            let expected_output =
                                checked_elements(&[n, f, h_out, w_out], "output")?
                                    .checked_mul(scalar_bytes)
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "conv_transpose2d: output bytes overflow".into(),
                                        )
                                    })?;
                            if out_end - out_start != expected_output {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: output storage does not match geometry"
                                        .into(),
                                ));
                            }
                            let output_overlaps = input_slices.iter().any(|slice| {
                                let end = slice.offset + slice.size;
                                slice.offset < out_end && out_start < end
                            });
                            if output_overlaps {
                                return Err(BackendError::Dispatch(
                                    "conv_transpose2d: input and output slices overlap".into(),
                                ));
                            }
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = unsafe {
                                    (
                                        arena.view_f32(input_slice.offset, input_slice.size),
                                        arena.view_f32(weight_slice.offset, weight_slice.size),
                                    )
                                };
                                let &[stride, padding, c, hin, win, kh, kw] = &params[..] else {
                                    return Err(BackendError::Dispatch(
                                        "conv_transpose2d: expected params [stride, padding, input_c, input_h, input_w, kernel_h, kernel_w]"
                                            .into(),
                                    ));
                                };
                                let n = input.len() / (c * hin * win).max(1);
                                let f = weight.len() / (c * kh * kw).max(1);
                                let h_out = ((hin - 1) * stride + kh).saturating_sub(2 * padding);
                                let w_out = ((win - 1) * stride + kw).saturating_sub(2 * padding);
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                out_f32.fill(0.0f32);
                                for nn in 0..n {
                                    for cc in 0..c {
                                        for hh in 0..hin {
                                            for ww in 0..win {
                                                for ff in 0..f {
                                                    for kkh in 0..kh {
                                                        for kkw in 0..kw {
                                                            let h_out_idx = hh * stride + kkh;
                                                            let w_out_idx = ww * stride + kkw;
                                                            if h_out_idx >= padding
                                                                && w_out_idx >= padding
                                                            {
                                                                let h_out_s = h_out_idx - padding;
                                                                let w_out_s = w_out_idx - padding;
                                                                if h_out_s < h_out
                                                                    && w_out_s < w_out
                                                                {
                                                                    let out_idx = nn
                                                                        * (f * h_out * w_out)
                                                                        + ff * (h_out * w_out)
                                                                        + h_out_s * w_out
                                                                        + w_out_s;
                                                                    let input_idx = nn
                                                                        * (c * hin * win)
                                                                        + cc * (hin * win)
                                                                        + hh * win
                                                                        + ww;
                                                                    let weight_idx = cc
                                                                        * (f * kh * kw)
                                                                        + ff * (kh * kw)
                                                                        + kkh * kw
                                                                        + kkw;
                                                                    if out_idx < out_f32.len()
                                                                        && input_idx < input.len()
                                                                        && weight_idx < weight.len()
                                                                    {
                                                                        out_f32[out_idx] += input
                                                                            [input_idx]
                                                                            * weight[weight_idx];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "prelu" => {
                            if input_slices.len() != 2 || !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "prelu: expected data and weight inputs without parameters"
                                        .into(),
                                ));
                            }
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let output_size = out_end - out_start;
                            let data_slice = input_slices[0];
                            let weight_slice = input_slices[1];
                            if data_slice.size == 0
                                || weight_slice.size == 0
                                || data_slice.size != output_size
                                || !data_slice.size.is_multiple_of(weight_slice.size)
                                || !data_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !weight_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !data_slice.size.is_multiple_of(scalar_bytes)
                                || !weight_slice.size.is_multiple_of(scalar_bytes)
                                || !output_size.is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "prelu: storage must contain compatible nonempty f32 tensors"
                                        .into(),
                                ));
                            }
                            let output_overlaps = input_slices.iter().any(|slice| {
                                let end = slice.offset + slice.size;
                                slice.offset < out_end && out_start < end
                            });
                            if output_overlaps {
                                return Err(BackendError::Dispatch(
                                    "prelu: input and output slices overlap".into(),
                                ));
                            }
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = unsafe {
                                    (
                                        arena.view_f32(data_slice.offset, data_slice.size),
                                        arena.view_f32(weight_slice.offset, weight_slice.size),
                                    )
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let channel_stride =
                                    if !weight.is_empty() && input.len() > weight.len() {
                                        input.len() / weight.len()
                                    } else {
                                        1
                                    };
                                for i in 0..out_f32.len() {
                                    let w_idx = if weight.len() == 1 {
                                        0
                                    } else {
                                        (i / channel_stride) % weight.len()
                                    };
                                    let slope = weight[w_idx];
                                    out_f32[i] = if input[i] > 0.0 {
                                        input[i]
                                    } else {
                                        input[i] * slope
                                    };
                                }
                            }
                        }
                        "rms_norm" => {
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_nary_f32_slices(
                                    arena,
                                    &[*data_slice, *weight_slice],
                                    output_slice,
                                    |inputs, out_f32| {
                                        let input = inputs[0];
                                        let weight = inputs[1];
                                        let row_size = if !weight.is_empty() {
                                            input.len() / weight.len()
                                        } else {
                                            input.len()
                                        };
                                        rms_norm_f32(input, weight, out_f32, row_size, eps);
                                    },
                                );
                            }
                        }
                        "fused_residual_add_layer_norm" => {
                            if !(2..=4).contains(&input_slices.len()) || params.len() != 2 {
                                return Err(BackendError::Dispatch(
                                    "fused residual layer norm requires 2-4 inputs and eps/row-size parameters".into(),
                                ));
                            }
                            let eps = f32::from_bits(params[0] as u32);
                            let row_size = params[1];
                            let row_bytes = row_size
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "fused residual layer norm row size overflows".into(),
                                    )
                                })?;
                            let output_size = out_end - out_start;
                            if !eps.is_finite()
                                || eps <= 0.0
                                || row_size == 0
                                || output_size == 0
                                || !output_size.is_multiple_of(row_bytes)
                                || input_slices[0].size != output_size
                                || input_slices[1].size != output_size
                                || input_slices
                                    .get(2)
                                    .is_some_and(|slice| slice.size != row_bytes)
                                || input_slices
                                    .get(3)
                                    .is_some_and(|slice| slice.size != row_bytes)
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                })
                            {
                                return Err(BackendError::Dispatch(
                                    "fused residual layer norm has invalid storage or numerical metadata".into(),
                                ));
                            }
                            if let [residual_slice, main_slice, rest @ ..] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                let bias_slice = rest.get(1).copied();
                                let fallback_row_size = weight_slice
                                    .or(bias_slice)
                                    .map(|slice| slice.size / std::mem::size_of::<f32>())
                                    .unwrap_or(output_slice.size / std::mem::size_of::<f32>());
                                let row_size = params
                                    .get(1)
                                    .copied()
                                    .filter(|&value| value > 0)
                                    .unwrap_or(fallback_row_size);

                                if residual_slice.size == main_slice.size
                                    && main_slice.size == output_slice.size
                                    && row_size > 0
                                    && (output_slice.size / std::mem::size_of::<f32>())
                                        .is_multiple_of(row_size)
                                    && weight_slice.is_none_or(|w| {
                                        w.size / std::mem::size_of::<f32>() == row_size
                                    })
                                    && bias_slice.is_none_or(|b| {
                                        b.size / std::mem::size_of::<f32>() == row_size
                                    })
                                {
                                    let mut inputs = vec![*residual_slice, *main_slice];
                                    if let Some(w) = weight_slice {
                                        inputs.push(w);
                                    }
                                    if let Some(b) = bias_slice {
                                        inputs.push(b);
                                    }
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs,
                                        output_slice,
                                        |inputs, out_f32| {
                                            let weight = if weight_slice.is_some() {
                                                inputs[2]
                                            } else {
                                                &[]
                                            };
                                            let bias = match (
                                                weight_slice.is_some(),
                                                bias_slice.is_some(),
                                            ) {
                                                (true, true) => inputs[3],
                                                (false, true) => inputs[2],
                                                _ => &[],
                                            };
                                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                            if crate::backend::cpu::microkernels::simd_avx2_available() {
                                                unsafe {
                                                    microkernels::fused_residual_add_layer_norm_f32_avx2(
                                                        inputs[0], inputs[1], weight, bias, out_f32,
                                                        row_size, eps,
                                                    );
                                                }
                                            } else {
                                                microkernels::fused_residual_add_layer_norm_f32_scalar(
                                                    inputs[0], inputs[1], weight, bias, out_f32,
                                                    row_size, eps,
                                                );
                                            }
                                            #[cfg(not(all(
                                                feature = "simd",
                                                target_arch = "x86_64"
                                            )))]
                                            microkernels::fused_residual_add_layer_norm_f32_scalar(
                                                inputs[0], inputs[1], weight, bias, out_f32,
                                                row_size, eps,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        "fused_residual_add_rms_norm" => {
                            if !(2..=3).contains(&input_slices.len()) || params.len() != 2 {
                                return Err(BackendError::Dispatch(
                                    "fused residual RMS norm requires 2-3 inputs and eps/row-size parameters".into(),
                                ));
                            }
                            let eps = f32::from_bits(params[0] as u32);
                            let row_size = params[1];
                            let row_bytes = row_size
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "fused residual RMS norm row size overflows".into(),
                                    )
                                })?;
                            let output_size = out_end - out_start;
                            if !eps.is_finite()
                                || eps <= 0.0
                                || row_size == 0
                                || output_size == 0
                                || !output_size.is_multiple_of(row_bytes)
                                || input_slices[0].size != output_size
                                || input_slices[1].size != output_size
                                || input_slices
                                    .get(2)
                                    .is_some_and(|slice| slice.size != row_bytes)
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                })
                            {
                                return Err(BackendError::Dispatch(
                                    "fused residual RMS norm has invalid storage or numerical metadata".into(),
                                ));
                            }
                            if let [residual_slice, main_slice, rest @ ..] = &input_slices[..] {
                                let eps =
                                    f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                let fallback_row_size = weight_slice
                                    .map(|slice| slice.size / std::mem::size_of::<f32>())
                                    .unwrap_or(output_slice.size / std::mem::size_of::<f32>());
                                let row_size = params
                                    .get(1)
                                    .copied()
                                    .filter(|&value| value > 0)
                                    .unwrap_or(fallback_row_size);

                                if residual_slice.size == main_slice.size
                                    && main_slice.size == output_slice.size
                                    && row_size > 0
                                    && (output_slice.size / std::mem::size_of::<f32>())
                                        .is_multiple_of(row_size)
                                    && weight_slice.is_none_or(|w| {
                                        w.size / std::mem::size_of::<f32>() == row_size
                                    })
                                {
                                    let mut inputs = vec![*residual_slice, *main_slice];
                                    if let Some(w) = weight_slice {
                                        inputs.push(w);
                                    }
                                    arena::with_nary_f32_slices(
                                        arena,
                                        &inputs,
                                        output_slice,
                                        |inputs, out_f32| {
                                            let weight = if weight_slice.is_some() {
                                                inputs[2]
                                            } else {
                                                &[]
                                            };
                                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                            if crate::backend::cpu::microkernels::simd_avx2_available() {
                                                unsafe {
                                                    microkernels::fused_residual_add_rms_norm_f32_avx2(
                                                        inputs[0], inputs[1], weight, out_f32, row_size,
                                                        eps,
                                                    );
                                                }
                                            } else {
                                                microkernels::fused_residual_add_rms_norm_f32_scalar(
                                                    inputs[0], inputs[1], weight, out_f32, row_size,
                                                    eps,
                                                );
                                            }
                                            #[cfg(not(all(
                                                feature = "simd",
                                                target_arch = "x86_64"
                                            )))]
                                            microkernels::fused_residual_add_rms_norm_f32_scalar(
                                                inputs[0], inputs[1], weight, out_f32, row_size,
                                                eps,
                                            );
                                        },
                                    );
                                }
                            }
                        }
                        "embedding" => {
                            if let [weight_slice, indices_slice] = &input_slices[..] {
                                let (weight, indices) = unsafe {
                                    (
                                        arena.view_f32(weight_slice.offset, weight_slice.size),
                                        arena.view_f32(indices_slice.offset, indices_slice.size),
                                    )
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let dim = if !weight.is_empty() && !indices.is_empty() {
                                    out_f32.len() / indices.len()
                                } else {
                                    1
                                };
                                for i in 0..indices.len() {
                                    let idx = indices[i] as usize;
                                    let src_start = idx * dim;
                                    let dst_start = i * dim;
                                    let len = dim
                                        .min(weight.len().saturating_sub(src_start))
                                        .min(out_f32.len().saturating_sub(dst_start));
                                    if len > 0 {
                                        out_f32[dst_start..dst_start + len]
                                            .copy_from_slice(&weight[src_start..src_start + len]);
                                    }
                                }
                            }
                        }
                        "pow_f32" => {
                            if let [data_slice, exp_slice] = &input_slices[..] {
                                let (data, exponent) = unsafe {
                                    (
                                        arena.view_f32(data_slice.offset, data_slice.size),
                                        arena.view_f32(exp_slice.offset, exp_slice.size),
                                    )
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                {
                                    for i in 0..len {
                                        let e = if i < exponent.len() {
                                            exponent[i]
                                        } else {
                                            exponent[exponent.len().saturating_sub(1)]
                                        };
                                        out_f32[i] = data[i].powf(e);
                                    }
                                }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let exponent = &exponent;
                                    if len >= 4096 {
                                        out_f32[..len].par_iter_mut().enumerate().for_each(
                                            |(i, o)| {
                                                let e = if i < exponent.len() {
                                                    exponent[i]
                                                } else {
                                                    exponent[exponent.len().saturating_sub(1)]
                                                };
                                                *o = data[i].powf(e);
                                            },
                                        );
                                    } else {
                                        for i in 0..len {
                                            let e = if i < exponent.len() {
                                                exponent[i]
                                            } else {
                                                exponent[exponent.len().saturating_sub(1)]
                                            };
                                            out_f32[i] = data[i].powf(e);
                                        }
                                    }
                                }
                            }
                        }
                        "gt_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                gt_scalar_f32,
                            );
                        }
                        "lt_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                lt_scalar_f32,
                            );
                        }
                        "eq_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                eq_scalar_f32,
                            );
                        }
                        "add_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                add_scalar_f32,
                            );
                        }
                        "mul_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                mul_scalar_f32,
                            );
                        }
                        "div_scalar_f32" => {
                            scalar_op_dispatch(
                                input_slices,
                                arena,
                                out_start,
                                out_end,
                                div_scalar_f32,
                            );
                        }
                        "argmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let axis = params.first().copied().unwrap_or(usize::MAX);
                                let dim_size = params.get(1).copied().unwrap_or(0);
                                let inner = params.get(2).copied().unwrap_or(1);
                                let input_end = input_slice.offset + input_slice.size;
                                let output_bytes = BufferSlice::new(out_start, out_end - out_start);

                                if !arena::ranges_overlap(
                                    input_slice.offset,
                                    input_end,
                                    output_bytes.offset,
                                    output_bytes.offset + output_bytes.size,
                                ) {
                                    let d = arena.data_mut();
                                    assert!(input_end <= d.len());
                                    assert!(out_end <= d.len());
                                    // SAFETY: bounds were checked above and the input/output byte
                                    // ranges are disjoint, so a shared f32 input slice cannot alias
                                    // the mutable u64 output slice.
                                    unsafe {
                                        let input = std::slice::from_raw_parts(
                                            d.as_ptr().add(input_slice.offset).cast::<f32>(),
                                            input_slice.size / std::mem::size_of::<f32>(),
                                        );
                                        let out_u64 = std::slice::from_raw_parts_mut(
                                            d.as_mut_ptr().add(out_start).cast::<u64>(),
                                            (out_end - out_start) / std::mem::size_of::<u64>(),
                                        );
                                        argmax_f32(input, out_u64, axis, dim_size, inner);
                                    }
                                } else {
                                    let input = {
                                        let d = arena.data_mut();
                                        let src = bytemuck::cast_slice::<_, f32>(
                                            &d[input_slice.offset..input_end],
                                        );
                                        let mut copy =
                                            crate::backend::cpu::microkernels::TlsVecPool::alloc(
                                                src.len(),
                                            );
                                        copy.copy_from_slice(src);
                                        crate::backend::cpu::telemetry::record_arena_temp_copy(
                                            input_slice.size,
                                        );
                                        copy
                                    };
                                    let d = arena.data_mut();
                                    let out_u64 = bytemuck::cast_slice_mut::<_, u64>(
                                        &mut d[out_start..out_end],
                                    );
                                    argmax_f32(&input, out_u64, axis, dim_size, inner);
                                }
                            }
                        }
                        "topk_fused" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let k = params.first().copied().unwrap_or(1);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_slice| {
                                        let mut indexed: Vec<(usize, f32)> =
                                            input.iter().copied().enumerate().collect();
                                        if input.len() > k {
                                            indexed.select_nth_unstable_by(
                                                input.len().saturating_sub(k),
                                                |a, b| {
                                                    a.1.partial_cmp(&b.1)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                },
                                            );
                                        }

                                        // Write values (f32) to primary output
                                        for i in 0..k.min(out_slice.len()) {
                                            out_slice[i] =
                                                indexed[input.len().saturating_sub(k) + i].1;
                                        }

                                        // Write indices (i64) to secondary output
                                        if let Some(sec_slice) = secondary_output_slice {
                                            let d = arena.data_mut();
                                            let sec_start = sec_slice.offset;
                                            let sec_end = sec_slice.offset + sec_slice.size;
                                            let idx_slice = bytemuck::cast_slice_mut::<_, u64>(
                                                &mut d[sec_start..sec_end],
                                            );
                                            for i in 0..k.min(idx_slice.len()) {
                                                idx_slice[i] =
                                                    indexed[input.len().saturating_sub(k) + i].0
                                                        as u64;
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "topk_values" | "topk_indices" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let k = params.first().copied().unwrap_or(1);
                                let _axis = params.get(1).copied().unwrap_or(usize::MAX);
                                let is_values = kernel_name == "topk_values";
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_slice| {
                                        let mut indexed: Vec<(usize, f32)> =
                                            input.iter().copied().enumerate().collect();
                                        if input.len() > k {
                                            indexed.select_nth_unstable_by(
                                                input.len().saturating_sub(k),
                                                |a, b| {
                                                    a.1.partial_cmp(&b.1)
                                                        .unwrap_or(std::cmp::Ordering::Equal)
                                                },
                                            );
                                        }
                                        if is_values {
                                            let out_f32 =
                                                bytemuck::cast_slice_mut::<_, f32>(out_slice);
                                            for i in 0..k.min(out_f32.len()) {
                                                out_f32[i] =
                                                    indexed[input.len().saturating_sub(k) + i].1;
                                            }
                                        } else {
                                            let out_u64 =
                                                bytemuck::cast_slice_mut::<_, u64>(out_slice);
                                            for i in 0..k.min(out_u64.len()) {
                                                out_u64[i] =
                                                    indexed[input.len().saturating_sub(k) + i].0
                                                        as u64;
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "upsample_nearest2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let h_in = params.get(2).copied().unwrap_or(1);
                                let w_in = params.get(3).copied().unwrap_or(1);
                                let hw = h_in * w_in;
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let in_len = input.len();
                                        if scale_h > 0
                                            && scale_w > 0
                                            && hw > 0
                                            && in_len > 0
                                            && in_len % hw == 0
                                        {
                                            let nc = in_len / hw;
                                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                            {
                                                use crate::backend::cpu::microkernels::has_avx2;
                                                if has_avx2() {
                                                    unsafe {
                                                        crate::backend::cpu::microkernels::upsample_nearest2d_f32_avx2(
                                                        input, out_f32, nc, h_in, w_in, scale_h, scale_w,
                                                    );
                                                    }
                                                    return;
                                                }
                                            }
                                            crate::backend::cpu::microkernels::upsample_nearest2d_f32(
                                            input, out_f32, nc, h_in, w_in, scale_h, scale_w,
                                        );
                                        }
                                    },
                                );
                            }
                        }
                        "upsample_bilinear2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let h_in = params.get(2).copied().unwrap_or(1);
                                let w_in = params.get(3).copied().unwrap_or(1);
                                let hw = h_in * w_in;
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let out_len = out_f32.len();
                                        let in_len = input.len();
                                        if scale_h > 0
                                            && scale_w > 0
                                            && hw > 0
                                            && out_len == in_len * scale_h * scale_w
                                            && in_len > 0
                                            && in_len % hw == 0
                                        {
                                            let nc = in_len / hw;
                                            for nci in 0..nc {
                                                for hi in 0..h_in * scale_h {
                                                    for wi in 0..w_in * scale_w {
                                                        let src_h = (hi as f64 / scale_h as f64)
                                                            .min((h_in - 1) as f64);
                                                        let src_w = (wi as f64 / scale_w as f64)
                                                            .min((w_in - 1) as f64);
                                                        let h0 = src_h.floor() as usize;
                                                        let w0 = src_w.floor() as usize;
                                                        let h1 = (h0 + 1).min(h_in - 1);
                                                        let w1 = (w0 + 1).min(w_in - 1);
                                                        let dh = src_h - h0 as f64;
                                                        let dw = src_w - w0 as f64;
                                                        let v00 = input[nci * hw + h0 * w_in + w0];
                                                        let v01 = input[nci * hw + h0 * w_in + w1];
                                                        let v10 = input[nci * hw + h1 * w_in + w0];
                                                        let v11 = input[nci * hw + h1 * w_in + w1];
                                                        let v0 = v00 * (1.0 - dw as f32)
                                                            + v01 * dw as f32;
                                                        let v1 = v10 * (1.0 - dw as f32)
                                                            + v11 * dw as f32;
                                                        let val =
                                                            v0 * (1.0 - dh as f32) + v1 * dh as f32;
                                                        let out_idx =
                                                            nci * h_in * scale_h * w_in * scale_w
                                                                + hi * w_in * scale_w
                                                                + wi;
                                                        if out_idx < out_len {
                                                            out_f32[out_idx] = val;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "adaptive_avg_pool2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let out_h = params.first().copied().unwrap_or(1);
                                let out_w = params.get(1).copied().unwrap_or(1);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let out_len = out_f32.len();
                                        if out_len > 0 {
                                            let in_len = input.len();
                                            let nc = if out_h > 0 && out_w > 0 {
                                                out_len / (out_h * out_w)
                                            } else {
                                                0
                                            };
                                            if nc > 0 && in_len > 0 && in_len % nc == 0 {
                                                let hw = in_len / nc;
                                                let mut h = (hw as f64).sqrt() as usize;
                                                while h > 0 && hw % h != 0 {
                                                    h -= 1;
                                                }
                                                let w = hw / h;
                                                if h >= out_h && w >= out_w && h > 0 && w > 0 {
                                                    microkernels::adaptive_avg_pool2d_f32_scalar(
                                                        input, out_f32, nc, h, w, out_h, out_w,
                                                    );
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "repeat" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let in_len = input.len();
                                        let out_len = out_f32.len();
                                        if in_len > 0 && out_len >= in_len && out_len % in_len == 0
                                        {
                                            let factor = out_len / in_len;
                                            for i in 0..in_len {
                                                let val = input[i];
                                                for f in 0..factor {
                                                    out_f32[i * factor + f] = val;
                                                }
                                            }
                                        } else if in_len > 0 {
                                            for i in 0..out_len {
                                                out_f32[i] = input[i % in_len];
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "cumsum" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let exclusive = params.get(1).copied().unwrap_or(0);
                                let rev = params.get(2).copied().unwrap_or(0);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let len = out_f32.len().min(input.len());
                                        if rev == 0 {
                                            let mut s = 0.0f32;
                                            for i in 0..len {
                                                s += input[i];
                                                out_f32[i] =
                                                    if exclusive != 0 { s - input[i] } else { s };
                                            }
                                        } else {
                                            let mut s = 0.0f32;
                                            for i in (0..len).rev() {
                                                s += input[i];
                                                out_f32[i] =
                                                    if exclusive != 0 { s - input[i] } else { s };
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "erf_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        for i in 0..out_f32.len().min(input.len()) {
                                            let x = input[i];
                                            let t = 1.0 / (1.0 + 0.3275911 * x.abs());
                                            #[allow(clippy::excessive_precision)]
                                            let y = 1.0
                                                - (((((1.061405429 * t - 1.453152027) * t)
                                                    + 1.421413741)
                                                    * t
                                                    - 0.284496736)
                                                    * t
                                                    + 0.254829592)
                                                    * t
                                                    * (-x * x).exp();
                                            out_f32[i] = x.signum() * y;
                                        }
                                    },
                                );
                            }
                        }
                        "flip" => {
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let num_dims = params.first().copied().unwrap_or(0);
                                let flip_dims: Vec<usize> = if num_dims > 0 {
                                    params[1..1 + num_dims].to_vec()
                                } else {
                                    vec![]
                                };
                                let shape: Vec<usize> = if num_dims > 0 {
                                    params[1 + num_dims..].to_vec()
                                } else {
                                    vec![]
                                };
                                let ndim = shape.len();
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        let len = out_f32.len().min(input.len());
                                        if params.is_empty() {
                                            for i in 0..len {
                                                out_f32[i] = input[len - 1 - i];
                                            }
                                        } else {
                                            let mut indices = vec![0i64; ndim];
                                            let mut strides = vec![0i64; ndim];
                                            let mut stride = 1i64;
                                            for d in (0..ndim).rev() {
                                                strides[d] = stride;
                                                stride *= shape[d] as i64;
                                            }
                                            for out_idx in 0..len {
                                                let mut src_idx = 0i64;
                                                for d in 0..ndim {
                                                    let idx = if flip_dims.contains(&d) {
                                                        shape[d] as i64 - 1 - indices[d]
                                                    } else {
                                                        indices[d]
                                                    };
                                                    src_idx += idx * strides[d];
                                                }
                                                out_f32[out_idx] = input[src_idx as usize];
                                                for d in (0..ndim).rev() {
                                                    indices[d] += 1;
                                                    if indices[d] < shape[d] as i64 {
                                                        break;
                                                    }
                                                    indices[d] = 0;
                                                }
                                            }
                                        }
                                    },
                                );
                            }
                        }
                        "where_f32" => {
                            if input_slices.len() >= 3 {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let cond_slice = input_slices[0];
                                let x_slice = input_slices[1];
                                let y_slice = input_slices[2];
                                arena::with_nary_f32_slices(
                                    arena,
                                    &[cond_slice, x_slice, y_slice],
                                    output_slice,
                                    |inputs, out_f32| {
                                        let cond = inputs[0];
                                        let x = inputs[1];
                                        let y = inputs[2];
                                        let len = out_f32.len();
                                        for i in 0..len {
                                            let c =
                                                cond.get(i % cond.len()).copied().unwrap_or(0.0);
                                            out_f32[i] = if c != 0.0 {
                                                x.get(i % x.len()).copied().unwrap_or(0.0)
                                            } else {
                                                y.get(i % y.len()).copied().unwrap_or(0.0)
                                            };
                                        }
                                    },
                                );
                            }
                        }
                        // â”€â”€ Optimizer kernels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        "sgd_update_f32" => {
                            let w_new = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let wd = if params.len() > 1 {
                                    f32::from_bits(params[1] as u32)
                                } else {
                                    0.0
                                };
                                sgd_update_f32(w_init, g_slice, lr, wd)
                            };
                            let d = arena.data_mut();
                            let w_off = input_slices[0].offset;
                            let w_sz = input_slices[0].size;
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[w_off..w_off + w_sz])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                        }
                        "quantize_gradient_f32_to_f8x4r" => {
                            if let Some(input_slice) = input_slices.first() {
                                let in_f32 =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let out = unsafe {
                                    bytemuck::cast_slice_mut(
                                        arena.view_f32_mut(out_start, out_end - out_start),
                                    )
                                };
                                let num_words = out.len() / 4;
                                for w in 0..num_words {
                                    let base = w * 4;
                                    let mut vals = [0.0f32; 4];
                                    for j in 0..4 {
                                        let idx = base + j;
                                        vals[j] = in_f32.get(idx).copied().unwrap_or(0.0);
                                    }
                                    let packed = F8x4R::pack_from_f32(vals).0;
                                    out[w * 4..w * 4 + 4].copy_from_slice(&packed.to_le_bytes());
                                }
                            }
                        }
                        "dequantize_gradient_f8x4r_to_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let in_u32 = unsafe {
                                    std::slice::from_raw_parts(
                                        arena
                                            .view_f32(input_slice.offset, input_slice.size)
                                            .as_ptr()
                                            as *const u32,
                                        input_slice.size / 4,
                                    )
                                };
                                let out_bytes = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        arena
                                            .view_f32_mut(out_start, out_end - out_start)
                                            .as_mut_ptr()
                                            as *mut u8,
                                        out_end - out_start,
                                    )
                                };
                                let num_words = numel.div_ceil(4);
                                for w in 0..num_words {
                                    let word = in_u32.get(w).copied().unwrap_or(0);
                                    let word = F8x4R(word);
                                    let vals = word.unpack_to_f32();
                                    let base = w * 4;
                                    for j in 0..4 {
                                        let idx = base + j;
                                        if idx < numel {
                                            let bytes = vals[j].to_le_bytes();
                                            let off = idx * 4;
                                            if off + 4 <= out_bytes.len() {
                                                out_bytes[off..off + 4].copy_from_slice(&bytes);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "gradient_scale" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let scale = f32::from_bits(*params.get(1).unwrap_or(&0) as u32);
                                let in_f32 =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let len = out_f32.len().min(in_f32.len()).min(numel);
                                #[cfg(not(feature = "parallel"))]
                                {
                                    for i in 0..len {
                                        out_f32[i] = in_f32[i] * scale;
                                    }
                                }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    if len >= 4096 {
                                        out_f32[..len]
                                            .par_iter_mut()
                                            .enumerate()
                                            .for_each(|(i, o)| *o = in_f32[i] * scale);
                                    } else {
                                        for i in 0..len {
                                            out_f32[i] = in_f32[i] * scale;
                                        }
                                    }
                                }
                            }
                        }
                        "adam_update_f32" => {
                            validate_adam_dispatch(
                                "adam_update_f32",
                                input_slices,
                                *output_slice,
                                params,
                                false,
                                false,
                            )?;
                            let d = arena.data_mut();
                            let d_ptr = d.as_mut_ptr();
                            // SAFETY: Slices are non-overlapping (except w which is
                            // read-before-write in adam_update_f32_scalar_into).
                            unsafe {
                                let w_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[0].offset) as *const f32,
                                    input_slices[0].size / 4,
                                );
                                let g_slice = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[1].offset) as *const f32,
                                    input_slices[1].size / 4,
                                );
                                let m_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[2].offset) as *const f32,
                                    input_slices[2].size / 4,
                                );
                                let v_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[3].offset) as *const f32,
                                    input_slices[3].size / 4,
                                );
                                let w_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[0].offset) as *mut f32,
                                    input_slices[0].size / 4,
                                );
                                let m_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[2].offset) as *mut f32,
                                    input_slices[2].size / 4,
                                );
                                let v_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[3].offset) as *mut f32,
                                    input_slices[3].size / 4,
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                let bias_corr1 = 1.0 - beta1.powi(t as i32);
                                let bias_corr2 = 1.0 - beta2.powi(t as i32);
                                adam_update_f32_into(
                                    w_init, g_slice, m_init, v_init, lr, beta1, beta2, eps,
                                    bias_corr1, bias_corr2, w_out, m_out, v_out,
                                );
                                // Copy updated w to output slot
                                std::ptr::copy_nonoverlapping(
                                    w_out.as_ptr(),
                                    d_ptr.add(out_start) as *mut f32,
                                    w_out.len(),
                                );
                            }
                        }
                        "adamw_update_f32" => {
                            validate_adam_dispatch(
                                "adamw_update_f32",
                                input_slices,
                                *output_slice,
                                params,
                                false,
                                true,
                            )?;
                            let d = arena.data_mut();
                            let d_ptr = d.as_mut_ptr();
                            // SAFETY: Slices are non-overlapping (except w which is
                            // read-before-write in adamw_update_f32_scalar_into).
                            unsafe {
                                let w_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[0].offset) as *const f32,
                                    input_slices[0].size / 4,
                                );
                                let g_slice = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[1].offset) as *const f32,
                                    input_slices[1].size / 4,
                                );
                                let m_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[2].offset) as *const f32,
                                    input_slices[2].size / 4,
                                );
                                let v_init = std::slice::from_raw_parts(
                                    d_ptr.add(input_slices[3].offset) as *const f32,
                                    input_slices[3].size / 4,
                                );
                                let w_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[0].offset) as *mut f32,
                                    input_slices[0].size / 4,
                                );
                                let m_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[2].offset) as *mut f32,
                                    input_slices[2].size / 4,
                                );
                                let v_out = std::slice::from_raw_parts_mut(
                                    d_ptr.add(input_slices[3].offset) as *mut f32,
                                    input_slices[3].size / 4,
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let (t, wd) = if input_slices.len() >= 5 {
                                    let d_ref: &[u8] = &*d;
                                    let t = read_adam_runtime_step(
                                        d_ref,
                                        input_slices[4],
                                        "adamw_update_f32",
                                    )?;
                                    let wd = f32::from_bits(params[4] as u32);
                                    (t, wd)
                                } else {
                                    let t = params[4] as f32;
                                    let wd = f32::from_bits(params[5] as u32);
                                    (t, wd)
                                };
                                let bias_corr1 = 1.0 - beta1.powi(t as i32);
                                let bias_corr2 = 1.0 - beta2.powi(t as i32);
                                adamw_update_f32_into(
                                    w_init, g_slice, m_init, v_init, lr, beta1, beta2, eps,
                                    bias_corr1, bias_corr2, wd, w_out, m_out, v_out,
                                );
                                // Copy updated w to output slot
                                std::ptr::copy_nonoverlapping(
                                    w_out.as_ptr(),
                                    d_ptr.add(out_start) as *mut f32,
                                    w_out.len(),
                                );
                            }
                        }
                        "muon_update_f32" => {
                            let (w_new, m_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta = f32::from_bits(params[1] as u32);
                                let wd = f32::from_bits(params[2] as u32);
                                muon_update_f32(w_init, g_slice, m_init, lr, beta, wd)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                        }
                        "lion_update_f32" => {
                            let (w_new, m_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let m_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let wd = if params.len() > 3 {
                                    f32::from_bits(params[3] as u32)
                                } else {
                                    0.0
                                };
                                lion_update_f32(w_init, g_slice, m_init, lr, beta1, beta2, wd)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&m_new);
                        }
                        "rmsprop_update_f32" => {
                            let (w_new, v_new) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                );
                                let g_slice = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                );
                                let v_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                );
                                let lr = f32::from_bits(params[0] as u32);
                                let beta = f32::from_bits(params[1] as u32);
                                let eps = f32::from_bits(params[2] as u32);
                                rmsprop_update_f32(w_init, g_slice, v_init, lr, beta, eps)
                            };
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size],
                            )
                            .copy_from_slice(&v_new);
                        }
                        // â”€â”€ F16 state optimizer kernels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        // m and v are stored as F16 (2 bytes/elem), w and grad are F32 (4 bytes/elem).
                        // Read F16 state, convert to f32 internally, apply update, write back as F16.
                        "adam_update_f16_state" => {
                            validate_adam_dispatch(
                                "adam_update_f16_state",
                                input_slices,
                                *output_slice,
                                params,
                                true,
                                false,
                            )?;
                            let n = input_slices[0].size / 4;
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                )
                                .to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let m_init: Vec<f32> = m_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let v_init: Vec<f32> = v_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let grad = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                )
                                .to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n.min(w_init.len()).min(m_init.len()).min(v_init.len());
                            let mut w_new = w_init.clone();
                            let mut m_new_f32 = vec![0.0f32; len];
                            let mut v_new_f32 = vec![0.0f32; len];
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                    m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                    v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                    let m_hat = m_new_f32[i] / bias_corr1;
                                    let v_hat = v_new_f32[i] / bias_corr2;
                                    w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                if len >= 4096 {
                                    w_new
                                        .par_iter_mut()
                                        .zip(m_new_f32.par_iter_mut())
                                        .zip(v_new_f32.par_iter_mut())
                                        .enumerate()
                                        .for_each(|(i, ((w, m), v))| {
                                            let g =
                                                grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                            *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                            *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                            let m_hat = *m / bias_corr1;
                                            let v_hat = *v / bias_corr2;
                                            *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                        });
                                } else {
                                    for i in 0..len {
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = m_new_f32[i] / bias_corr1;
                                        let v_hat = v_new_f32[i] / bias_corr2;
                                        w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                    }
                                }
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let v_bytes: Vec<u8> = v_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end]
                                .copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end]
                                .copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "adamw_update_f16_state" => {
                            validate_adam_dispatch(
                                "adamw_update_f16_state",
                                input_slices,
                                *output_slice,
                                params,
                                true,
                                true,
                            )?;
                            let n = input_slices[0].size / 4;
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[0].offset
                                        ..input_slices[0].offset + input_slices[0].size],
                                )
                                .to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset
                                    ..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset
                                    ..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2)
                                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                                    .collect();
                                let m_init: Vec<f32> = m_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let v_init: Vec<f32> = v_raw
                                    .iter()
                                    .map(|&bits| half::f16::from_bits(bits).to_f32())
                                    .collect();
                                let grad = bytemuck::cast_slice::<_, f32>(
                                    &d_ref[input_slices[1].offset
                                        ..input_slices[1].offset + input_slices[1].size],
                                )
                                .to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let (t, wd) = if input_slices.len() >= 5 {
                                    // New path: t is a runtime tensor (5th input slice)
                                    let t = read_adam_runtime_step(
                                        d_ref,
                                        input_slices[4],
                                        "adamw_update_f16_state",
                                    )?;
                                    let wd = f32::from_bits(params[4] as u32);
                                    (t, wd)
                                } else {
                                    // Old path: t is in params[4], wd in params[5]
                                    let t = params[4] as f32;
                                    let wd = f32::from_bits(params[5] as u32);
                                    (t, wd)
                                };
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n.min(w_init.len()).min(m_init.len()).min(v_init.len());
                            let mut w_new = w_init.clone();
                            let mut m_new_f32 = vec![0.0f32; len];
                            let mut v_new_f32 = vec![0.0f32; len];
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    w_new[i] -= lr * wd * w_init[i];
                                    let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                    m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                    v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                    let m_hat = m_new_f32[i] / bias_corr1;
                                    let v_hat = v_new_f32[i] / bias_corr2;
                                    w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                if len >= 4096 {
                                    w_new
                                        .par_iter_mut()
                                        .zip(m_new_f32.par_iter_mut())
                                        .zip(v_new_f32.par_iter_mut())
                                        .enumerate()
                                        .for_each(|(i, ((w, m), v))| {
                                            *w -= lr * wd * w_init[i];
                                            let g =
                                                grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                            *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                            *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                            let m_hat = *m / bias_corr1;
                                            let v_hat = *v / bias_corr2;
                                            *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                        });
                                } else {
                                    for i in 0..len {
                                        w_new[i] -= lr * wd * w_init[i];
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = m_new_f32[i] / bias_corr1;
                                        let v_hat = v_new_f32[i] / bias_corr2;
                                        w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                                    }
                                }
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[input_slices[0].offset
                                    ..input_slices[0].offset + input_slices[0].size],
                            )
                            .copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                .copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let v_bytes: Vec<u8> = v_new_f32
                                .iter()
                                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                .collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end]
                                .copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end]
                                .copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "cast" => {
                            let in_byte_size = *params.first().unwrap_or(&4);
                            let out_byte_size = *params.get(1).unwrap_or(&4);
                            if let Some(input_slice) = input_slices.first() {
                                if in_byte_size == 4 && out_byte_size == 8 {
                                    let in_f32 = unsafe {
                                        arena.view_f32(input_slice.offset, input_slice.size)
                                    };
                                    let out = unsafe {
                                        bytemuck::cast_slice_mut(
                                            arena.view_f32_mut(out_start, out_end - out_start),
                                        )
                                    };
                                    let num = in_f32.len();
                                    for i in 0..num {
                                        let byte_off = i * 8;
                                        if byte_off + 8 <= out.len() {
                                            out[byte_off..byte_off + 8]
                                                .copy_from_slice(&(in_f32[i] as i64).to_le_bytes());
                                        }
                                    }
                                } else if in_byte_size == 8 && out_byte_size == 4 {
                                    let in_f32_src = unsafe {
                                        arena.view_f32(input_slice.offset, input_slice.size)
                                    };
                                    let in_i64: &[i64] = bytemuck::cast_slice(in_f32_src);
                                    let out_f32 = unsafe {
                                        arena.view_f32_mut(out_start, out_end - out_start)
                                    };
                                    let len = in_i64.len().min(out_f32.len());
                                    for i in 0..len {
                                        out_f32[i] = in_i64[i] as f32;
                                    }
                                }
                                // Same-size casts handled by MemCopy at compile time
                            }
                        }

                        "expand_f32" => {
                            // Expand broadcasts input[0] (f32 data) using target
                            // shape input[1] (i64 dims).  params layout:
                            //   [max_rank, in_d0..in_dN, out_d0..out_dN]
                            if input_slices.len() < 2 {
                                return Err(BackendError::Dispatch(
                                    "expand_f32 needs 2 inputs (data + shape)".into(),
                                ));
                            }
                            let max_rank = *params.first().ok_or_else(|| {
                                BackendError::Dispatch("expand_f32: missing max_rank".into())
                            })?;
                            if max_rank > 8 {
                                return Err(BackendError::Dispatch(format!(
                                    "expand_f32: rank {max_rank} exceeds the supported rank 8"
                                )));
                            }
                            let expected_params = max_rank
                                .checked_mul(2)
                                .and_then(|value| value.checked_add(1))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "expand_f32: parameter count overflows".into(),
                                    )
                                })?;
                            if params.len() < expected_params {
                                return Err(BackendError::Dispatch(format!(
                                    "expand_f32: expected {} params, got {}",
                                    expected_params,
                                    params.len()
                                )));
                            }
                            // Extract padded input dims and output dims
                            let in_dims: Vec<usize> = params[1..1 + max_rank].to_vec();
                            let out_dims: Vec<usize> =
                                params[1 + max_rank..1 + max_rank * 2].to_vec();

                            let data_slice = &input_slices[0];
                            let _shape_slice = &input_slices[1];
                            for (input_dim, output_dim) in in_dims.iter().zip(&out_dims) {
                                if input_dim != output_dim && *input_dim != 1 {
                                    return Err(BackendError::Dispatch(format!(
                                        "expand_f32: input dimension {input_dim} cannot expand to {output_dim}"
                                    )));
                                }
                            }
                            let expected_input_numel =
                                in_dims.iter().try_fold(1usize, |acc, dim| {
                                    acc.checked_mul(*dim).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "expand_f32: input element count overflows".into(),
                                        )
                                    })
                                })?;
                            let expected_output_numel =
                                out_dims.iter().try_fold(1usize, |acc, dim| {
                                    acc.checked_mul(*dim).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "expand_f32: output element count overflows".into(),
                                        )
                                    })
                                })?;
                            let expected_input_size =
                                expected_input_numel.checked_mul(4).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "expand_f32: input size overflows".into(),
                                    )
                                })?;
                            let expected_output_size =
                                expected_output_numel.checked_mul(4).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "expand_f32: output size overflows".into(),
                                    )
                                })?;
                            if data_slice.size != expected_input_size
                                || output_slice.size != expected_output_size
                                || !data_slice.offset.is_multiple_of(4)
                                || !output_slice.offset.is_multiple_of(4)
                            {
                                return Err(BackendError::Dispatch(
                                    "expand_f32: tensor slices do not match declared dimensions"
                                        .into(),
                                ));
                            }
                            let input_end = data_slice.offset + data_slice.size;
                            let output_end = output_slice.offset + output_slice.size;
                            if data_slice.offset < output_end && output_slice.offset < input_end {
                                return Err(BackendError::Dispatch(
                                    "expand_f32: input and output slices overlap".into(),
                                ));
                            }
                            let data_numel = data_slice.size / 4; // f32 = 4 bytes
                            let out_numel = output_slice.size / 4;

                            let in_f32 =
                                unsafe { arena.view_f32(data_slice.offset, data_slice.size) };

                            let out_f32 =
                                unsafe { arena.view_f32_mut(out_start, output_slice.size) };

                            // Broadcast: for each output element, map back to input coords
                            #[cfg(not(feature = "parallel"))]
                            {
                                for out_linear in 0..out_numel {
                                    let mut out_coord = vec![0usize; max_rank];
                                    let mut remaining = out_linear;
                                    for i in (0..max_rank).rev() {
                                        out_coord[i] = remaining % out_dims[i];
                                        remaining /= out_dims[i];
                                    }
                                    let mut in_linear: usize = 0;
                                    let mut in_stride = 1usize;
                                    for i in (0..max_rank).rev() {
                                        let in_dim = in_dims[i];
                                        let out_dim = out_dims[i];
                                        let in_coord = if in_dim == out_dim {
                                            out_coord[i]
                                        } else if in_dim == 1 {
                                            0
                                        } else {
                                            0
                                        };
                                        in_linear += in_coord * in_stride;
                                        in_stride *= in_dim;
                                    }
                                    if in_linear < data_numel {
                                        out_f32[out_linear] = in_f32[in_linear];
                                    }
                                }
                            }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                let max_rank = max_rank;
                                let out_addr = out_f32.as_mut_ptr() as usize;
                                if out_numel >= 4096 {
                                    (0..out_numel).into_par_iter().for_each(|out_linear| {
                                        let mut out_coord = vec![0usize; max_rank];
                                        let mut remaining = out_linear;
                                        for i in (0..max_rank).rev() {
                                            out_coord[i] = remaining % out_dims[i];
                                            remaining /= out_dims[i];
                                        }
                                        let mut in_linear: usize = 0;
                                        let mut in_stride = 1usize;
                                        for i in (0..max_rank).rev() {
                                            let in_dim = in_dims[i];
                                            let out_dim = out_dims[i];
                                            let in_coord = if in_dim == out_dim {
                                                out_coord[i]
                                            } else if in_dim == 1 {
                                                0
                                            } else {
                                                0
                                            };
                                            in_linear += in_coord * in_stride;
                                            in_stride *= in_dim;
                                        }
                                        if in_linear < data_numel {
                                            unsafe {
                                                *(out_addr as *mut f32).add(out_linear) =
                                                    in_f32[in_linear];
                                            }
                                        }
                                    });
                                } else {
                                    for out_linear in 0..out_numel {
                                        let mut out_coord = [0usize; 8];
                                        let mut remaining = out_linear;
                                        for i in (0..max_rank).rev() {
                                            out_coord[i] = remaining % out_dims[i];
                                            remaining /= out_dims[i];
                                        }
                                        let mut in_linear: usize = 0;
                                        let mut in_stride = 1usize;
                                        for i in (0..max_rank).rev() {
                                            let in_dim = in_dims[i];
                                            let out_dim = out_dims[i];
                                            let in_coord = if in_dim == out_dim {
                                                out_coord[i]
                                            } else if in_dim == 1 {
                                                0
                                            } else {
                                                0
                                            };
                                            in_linear += in_coord * in_stride;
                                            in_stride *= in_dim;
                                        }
                                        if in_linear < data_numel {
                                            out_f32[out_linear] = in_f32[in_linear];
                                        }
                                    }
                                }
                            }
                        }
                        "range_f32" => {
                            // Range(start, limit, step): produce 1D F32 tensor.
                            let d = arena.data_mut();
                            let start_val = if let Some(s) = input_slices.first() {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4]),
                                )
                            } else {
                                0.0
                            };
                            let limit_val = if let Some(s) = input_slices.get(1) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4]),
                                )
                            } else {
                                0.0
                            };
                            let step_val = if let Some(s) = input_slices.get(2) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([1u8; 4]),
                                )
                            } else {
                                1.0
                            };
                            let n = if step_val > 0.0 {
                                ((limit_val - start_val) / step_val).ceil().max(0.0) as usize
                            } else if step_val < 0.0 {
                                ((start_val - limit_val) / (-step_val)).ceil().max(0.0) as usize
                            } else {
                                0
                            };
                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[out_start..out_start + output_slice.size],
                            );
                            let actual_len = out_f32.len().min(n);
                            for i in 0..actual_len {
                                out_f32[i] = start_val + i as f32 * step_val;
                            }
                        }
                        "quantize_f32_u4" | "quantize_f32_u8" => {
                            let num_channels = *params.first().unwrap_or(&1);
                            let num_elems_per_channel = *params.get(1).unwrap_or(&1);
                            let numel = *params.get(2).unwrap_or(&1);
                            let bit_width = if kernel_name == "quantize_f32_u4" {
                                4
                            } else {
                                8
                            };
                            let max_q = (1i32 << (bit_width - 1)) - 1; // 7 for U4, 127 for U8
                            let items_per_word = 32 / bit_width; // 8 for U4, 4 for U8

                            // Check for cached scales from wrap_quantized_optimizer
                            let has_cached = params.get(3).copied().unwrap_or(0) == 1;
                            let mut cached_scales = vec![];
                            let mut cached_zeros = vec![];
                            if has_cached {
                                let sc_start = 4;
                                let sc_end = sc_start + num_channels;
                                let zp_start = sc_end;
                                let zp_end = zp_start + num_channels;
                                for i in sc_start..sc_end {
                                    let bits = *params.get(i).unwrap_or(&0);
                                    cached_scales.push(f32::from_bits(bits as u32));
                                }
                                for i in zp_start..zp_end {
                                    let bits = *params.get(i).unwrap_or(&0);
                                    cached_zeros.push(f32::from_bits(bits as u32));
                                }
                            }

                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let input_f32 = bytemuck::cast_slice::<_, f32>(
                                    &d[input_slice.offset..input_slice.offset + input_slice.size],
                                );
                                let mut f32_data = tls_alloc_f32(input_f32.len());
                                f32_data.copy_from_slice(input_f32);

                                let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                let mut zero_points: Vec<f32> = vec![0.0; num_channels];
                                let packed_words = numel.div_ceil(items_per_word);
                                let mut packed: Vec<u32> = vec![0u32; packed_words];

                                if has_cached
                                    && cached_scales.len() == num_channels
                                    && cached_zeros.len() == num_channels
                                {
                                    scales = cached_scales;
                                    zero_points = cached_zeros;
                                    // Pack using cached scales
                                    for ch in 0..num_channels {
                                        let start = ch * num_elems_per_channel;
                                        let end =
                                            (start + num_elems_per_channel).min(f32_data.len());
                                        let scale = scales[ch];
                                        let inv_s = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                                        let zp = zero_points[ch];
                                        for j in start..end {
                                            let q = ((f32_data[j] - zp) * inv_s)
                                                .round()
                                                .clamp(-(max_q as f32), max_q as f32)
                                                as i32;
                                            let word_idx = j / items_per_word;
                                            let shift = (j % items_per_word) * bit_width;
                                            packed[word_idx] |=
                                                ((q as u32) & ((1 << bit_width) - 1)) << shift;
                                        }
                                    }
                                } else {
                                    // Original path: recompute per-channel scales
                                    for ch in 0..num_channels {
                                        let start = ch * num_elems_per_channel;
                                        let end =
                                            (start + num_elems_per_channel).min(f32_data.len());
                                        let max_abs = f32_data[start..end]
                                            .iter()
                                            .map(|v| v.abs())
                                            .fold(0.0f32, f32::max);
                                        let scale = if max_abs == 0.0 {
                                            1.0
                                        } else {
                                            max_abs / max_q as f32
                                        };
                                        scales.push(scale);

                                        // Quantize and pack
                                        for j in start..end {
                                            let q = (f32_data[j] / scale)
                                                .round()
                                                .clamp(-(max_q as f32), max_q as f32)
                                                as i32;
                                            let word_idx = j / items_per_word;
                                            let shift = (j % items_per_word) * bit_width;
                                            packed[word_idx] |=
                                                ((q as u32) & ((1 << bit_width) - 1)) << shift;
                                        }
                                    }
                                }

                                // Write output: [num_channels(u32)][num_elems_per_channel(u32)]
                                //             [scales(f32 x N)][zero_points(f32 x N)][packed_data]
                                let header_size = 8 + 8 * num_channels; // 2 u32 + N f32 + N f32
                                let total_size = header_size + packed.len() * 4;
                                let out_end = (out_start + total_size).min(d.len());
                                let out = &mut d[out_start..out_end];

                                let mut offset = 0;
                                out[offset..offset + 4]
                                    .copy_from_slice(&(num_channels as u32).to_le_bytes());
                                offset += 4;
                                out[offset..offset + 4]
                                    .copy_from_slice(&(num_elems_per_channel as u32).to_le_bytes());
                                offset += 4;
                                for &s in &scales {
                                    out[offset..offset + 4].copy_from_slice(&s.to_le_bytes());
                                    offset += 4;
                                }
                                for &z in &zero_points {
                                    out[offset..offset + 4].copy_from_slice(&z.to_le_bytes());
                                    offset += 4;
                                }
                                for &w in &packed {
                                    out[offset..offset + 4].copy_from_slice(&w.to_le_bytes());
                                    offset += 4;
                                }
                            }
                        }
                        "dequantize_kernel" => {
                            let input_slice = input_slices.first().ok_or_else(|| {
                                BackendError::Dispatch(
                                    "dequantize_kernel: missing input slice".into(),
                                )
                            })?;
                            if params.len() < 4 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_kernel: expected numel, format, bit width, and channel count".into(),
                                ));
                            }
                            let numel = params[0];
                            let format_flag = params[1];
                            let bit_width = params[2];
                            if format_flag > 1 || !matches!(bit_width, 4 | 8) {
                                return Err(BackendError::Dispatch(
                                    "dequantize_kernel: invalid format or bit width".into(),
                                ));
                            }
                            let expected_output = numel.checked_mul(4).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "dequantize_kernel: output size overflows".into(),
                                )
                            })?;
                            if output_slice.size != expected_output
                                || !output_slice.offset.is_multiple_of(4)
                            {
                                return Err(BackendError::Dispatch(
                                    "dequantize_kernel: output slice does not match numel".into(),
                                ));
                            }
                            {
                                let in_data = {
                                    let d = arena.data_mut();
                                    let mut buf = tls_alloc_u8(input_slice.size);
                                    buf.copy_from_slice(
                                        &d[input_slice.offset
                                            ..input_slice.offset + input_slice.size],
                                    );
                                    buf
                                };

                                let (
                                    num_channels,
                                    num_elems_per_channel,
                                    scales,
                                    zero_points,
                                    data_offset,
                                    bit_width,
                                ) = if format_flag == 1 {
                                    // Metadata-based: scales/zero_points passed as params
                                    let num_channels = params[3];
                                    let expected_params = num_channels
                                        .checked_mul(2)
                                        .and_then(|value| value.checked_add(4))
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "dequantize_kernel: metadata parameter count overflows"
                                                    .into(),
                                            )
                                        })?;
                                    if params.len() != expected_params || num_channels == 0 {
                                        return Err(BackendError::Dispatch(
                                            "dequantize_kernel: metadata parameter count mismatch"
                                                .into(),
                                        ));
                                    }
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = params[4 + j];
                                        scales.push(f32::from_bits(bits as u32));
                                    }
                                    let mut zero_points: Vec<f32> =
                                        Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = params[4 + num_channels + j];
                                        zero_points.push(f32::from_bits(bits as u32));
                                    }
                                    // The packed data starts at offset 0 (no header)
                                    let data_offset = 0;
                                    // Infer num_elems_per_channel from numel
                                    let num_elems_per_channel = if num_channels > 0 {
                                        numel / num_channels
                                    } else {
                                        numel
                                    };

                                    (
                                        num_channels,
                                        num_elems_per_channel,
                                        scales,
                                        zero_points,
                                        data_offset,
                                        bit_width,
                                    )
                                } else {
                                    // Header-based: parse [num_channels][num_elems][scales...][zps...][packed_data]
                                    if params.len() != 4 || in_data.len() < 8 {
                                        return Err(BackendError::Dispatch(
                                            "dequantize_kernel: truncated header payload".into(),
                                        ));
                                    }
                                    let num_channels = u32::from_le_bytes([
                                        in_data[0], in_data[1], in_data[2], in_data[3],
                                    ])
                                        as usize;
                                    let num_elems_per_channel = u32::from_le_bytes([
                                        in_data[4], in_data[5], in_data[6], in_data[7],
                                    ])
                                        as usize;
                                    let metadata_bytes = num_channels
                                        .checked_mul(8)
                                        .and_then(|value| value.checked_add(8))
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "dequantize_kernel: header size overflows".into(),
                                            )
                                        })?;
                                    if num_channels == 0
                                        || num_elems_per_channel == 0
                                        || metadata_bytes > in_data.len()
                                    {
                                        return Err(BackendError::Dispatch(
                                            "dequantize_kernel: invalid channel header".into(),
                                        ));
                                    }
                                    let mut hdr_offset = 8usize;
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for _ in 0..num_channels {
                                        let bytes = &in_data[hdr_offset..hdr_offset + 4];
                                        scales.push(f32::from_le_bytes([
                                            bytes[0], bytes[1], bytes[2], bytes[3],
                                        ]));
                                        hdr_offset += 4;
                                    }
                                    let mut zero_points: Vec<f32> =
                                        Vec::with_capacity(num_channels);
                                    for _ in 0..num_channels {
                                        let bytes = &in_data[hdr_offset..hdr_offset + 4];
                                        zero_points.push(f32::from_le_bytes([
                                            bytes[0], bytes[1], bytes[2], bytes[3],
                                        ]));
                                        hdr_offset += 4;
                                    }
                                    let data_offset = hdr_offset;

                                    (
                                        num_channels,
                                        num_elems_per_channel,
                                        scales,
                                        zero_points,
                                        data_offset,
                                        bit_width,
                                    )
                                };

                                let declared_values = num_channels
                                    .checked_mul(num_elems_per_channel)
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_kernel: channel element count overflows"
                                                .into(),
                                        )
                                    })?;
                                if num_channels == 0
                                    || num_elems_per_channel == 0
                                    || declared_values != numel
                                    || scales.len() != num_channels
                                    || zero_points.len() != num_channels
                                    || scales
                                        .iter()
                                        .any(|scale| !scale.is_finite() || *scale <= 0.0)
                                    || zero_points.iter().any(|offset| !offset.is_finite())
                                {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_kernel: invalid channel or affine metadata"
                                            .into(),
                                    ));
                                }
                                let items_per_word = 32 / bit_width;
                                let packed_words_required = numel
                                    .checked_add(items_per_word - 1)
                                    .map(|value| value / items_per_word)
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_kernel: packed word count overflows".into(),
                                        )
                                    })?;
                                let packed_bytes_required =
                                    packed_words_required.checked_mul(4).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_kernel: packed byte count overflows".into(),
                                        )
                                    })?;
                                let packed_end = data_offset
                                    .checked_add(packed_bytes_required)
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_kernel: packed payload range overflows"
                                                .into(),
                                        )
                                    })?;
                                if packed_end > in_data.len() {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_kernel: truncated packed payload".into(),
                                    ));
                                }

                                let total_packed_bytes = in_data.len() - data_offset;
                                let packed_words = total_packed_bytes / 4;

                                // Write output
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let max_out = out_f32.len().min(numel);
                                for i in 0..max_out {
                                    let ch = if num_channels > 0 {
                                        i / num_elems_per_channel % num_channels
                                    } else {
                                        0
                                    };
                                    let word_idx = i / items_per_word;
                                    let shift = (i % items_per_word) * bit_width;
                                    if word_idx < packed_words {
                                        let word_start = data_offset + word_idx * 4;
                                        let word = if word_start + 4 <= in_data.len() {
                                            u32::from_le_bytes(
                                                in_data[word_start..word_start + 4]
                                                    .try_into()
                                                    .unwrap(),
                                            )
                                        } else {
                                            0
                                        };
                                        let q = ((word >> shift) & ((1 << bit_width) - 1)) as i32;
                                        // Sign-extend for signed types
                                        let sign_bit = 1 << (bit_width - 1);
                                        let q_signed = if (q & sign_bit) != 0 {
                                            q | (!((1 << bit_width) - 1))
                                        } else {
                                            q
                                        };
                                        let scale = scales.get(ch).copied().unwrap_or(1.0);
                                        let zp = zero_points.get(ch).copied().unwrap_or(0.0);
                                        out_f32[i] = q_signed as f32 * scale + zp;
                                    }
                                }
                            }
                        }
                        "to_f16" => {
                            if let Some(input_slice) = input_slices.first() {
                                let f32_data =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let out_bytes =
                                    unsafe { arena.view_u8_mut(out_start, out_end - out_start) };
                                for (i, &v) in f32_data.iter().enumerate() {
                                    let f16_val = half::f16::from_f32(v);
                                    let bytes = f16_val.to_le_bytes();
                                    let start = i * 2;
                                    let end = (start + 2).min(out_bytes.len());
                                    if start < out_bytes.len() {
                                        out_bytes[start..end]
                                            .copy_from_slice(&bytes[..end - start]);
                                    }
                                }
                            }
                        }
                        "to_f32" => {
                            if input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "to_f32: expected exactly one f16 input".into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            if input_slice.size % 2 != 0
                                || input_slice.offset % 2 != 0
                                || !output_slice.offset.is_multiple_of(4)
                            {
                                return Err(BackendError::Dispatch(
                                    "to_f32: input or output slice has invalid scalar alignment"
                                        .into(),
                                ));
                            }
                            let numel = input_slice.size / 2;
                            let expected_output = numel.checked_mul(4).ok_or_else(|| {
                                BackendError::Dispatch("to_f32: output size overflows".into())
                            })?;
                            if output_slice.size != expected_output {
                                return Err(BackendError::Dispatch(format!(
                                    "to_f32: output has {} bytes, expected {expected_output}",
                                    output_slice.size
                                )));
                            }
                            let input_end = input_slice.offset + input_slice.size;
                            if input_slice.offset < out_end && out_start < input_end {
                                return Err(BackendError::Dispatch(
                                    "to_f32: input and output slices overlap".into(),
                                ));
                            }
                            let in_data =
                                unsafe { arena.view_u8(input_slice.offset, input_slice.size) };
                            let out_f32 =
                                unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                            for (index, output) in out_f32.iter_mut().enumerate() {
                                let start = index * 2;
                                *output =
                                    half::f16::from_le_bytes([in_data[start], in_data[start + 1]])
                                        .to_f32();
                            }
                        }
                        "quantize_activations" => {
                            if params.len() < 3 || input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "quantize_activations: incomplete parameters or inputs".into(),
                                ));
                            }
                            let numel = params[0];
                            let mode = params[1];
                            let num_channels = params[2];
                            if mode > 1 {
                                return Err(BackendError::Dispatch(
                                    "quantize_activations: invalid quantization mode".into(),
                                ));
                            }
                            let is_per_channel = mode == 1;
                            let expected_params = if is_per_channel {
                                if num_channels == 0
                                    || num_channels > u32::MAX as usize
                                    || numel % num_channels != 0
                                    || numel / num_channels > u32::MAX as usize
                                {
                                    return Err(BackendError::Dispatch(
                                        "quantize_activations: incompatible channel metadata"
                                            .into(),
                                    ));
                                }
                                num_channels
                                    .checked_mul(2)
                                    .and_then(|value| value.checked_add(3))
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "quantize_activations: parameter count overflows"
                                                .into(),
                                        )
                                    })?
                            } else {
                                match params.get(3) {
                                    Some(0) => 4,
                                    Some(1) => 6,
                                    _ => {
                                        return Err(BackendError::Dispatch(
                                            "quantize_activations: invalid calibration mode".into(),
                                        ))
                                    }
                                }
                            };
                            if params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "quantize_activations: affine parameter count mismatch".into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let input_bytes = numel.checked_mul(4).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "quantize_activations: input size overflows".into(),
                                )
                            })?;
                            let metadata_bytes = if is_per_channel {
                                num_channels
                                    .checked_mul(8)
                                    .and_then(|value| value.checked_add(8))
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "quantize_activations: metadata size overflows".into(),
                                        )
                                    })?
                            } else {
                                8
                            };
                            let output_bytes =
                                metadata_bytes.checked_add(numel).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "quantize_activations: output size overflows".into(),
                                    )
                                })?;
                            if !input_slice.offset.is_multiple_of(4)
                                || input_slice.size != input_bytes
                                || output_slice.size != output_bytes
                            {
                                return Err(BackendError::Dispatch(
                                    "quantize_activations: storage does not match metadata".into(),
                                ));
                            }
                            let input_end = input_slice.offset + input_slice.size;
                            if input_slice.offset < out_end && out_start < input_end {
                                return Err(BackendError::Dispatch(
                                    "quantize_activations: input and output slices overlap".into(),
                                ));
                            }
                            let f32_data =
                                unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                            let out_bytes =
                                unsafe { arena.view_u8_mut(out_start, out_end - out_start) };

                            if is_per_channel {
                                let scale_index = 3;
                                let offset_index = 3 + num_channels;
                                let chunk_size = numel / num_channels;
                                out_bytes[0..4]
                                    .copy_from_slice(&(num_channels as u32).to_le_bytes());
                                out_bytes[4..8].copy_from_slice(&(chunk_size as u32).to_le_bytes());
                                let scale_start = 8;
                                let offset_start = 8 + num_channels * 4;
                                let data_start = metadata_bytes;
                                for channel in 0..num_channels {
                                    let scale =
                                        f32::from_bits(params[scale_index + channel] as u32);
                                    let offset =
                                        f32::from_bits(params[offset_index + channel] as u32);
                                    if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                                        return Err(BackendError::Dispatch(
                                            "quantize_activations: invalid affine metadata".into(),
                                        ));
                                    }
                                    out_bytes[scale_start + channel * 4
                                        ..scale_start + (channel + 1) * 4]
                                        .copy_from_slice(&scale.to_le_bytes());
                                    out_bytes[offset_start + channel * 4
                                        ..offset_start + (channel + 1) * 4]
                                        .copy_from_slice(&offset.to_le_bytes());
                                    let channel_start = channel * chunk_size;
                                    let channel_end = channel_start + chunk_size;
                                    for index in channel_start..channel_end {
                                        let quantized = ((f32_data[index] - offset) / scale)
                                            .round()
                                            .clamp(-128.0, 127.0)
                                            as i8;
                                        out_bytes[data_start + index] = quantized as u8;
                                    }
                                }
                            } else {
                                let (scale, offset) = if params[3] == 0 {
                                    if f32_data.iter().any(|value| !value.is_finite()) {
                                        return Err(BackendError::Dispatch(
                                            "quantize_activations: dynamic calibration requires finite input"
                                                .into(),
                                        ));
                                    }
                                    let max_abs = f32_data
                                        .iter()
                                        .map(|value| value.abs())
                                        .fold(0.0f32, f32::max);
                                    (if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 }, 0.0)
                                } else {
                                    (
                                        f32::from_bits(params[4] as u32),
                                        f32::from_bits(params[5] as u32),
                                    )
                                };
                                if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                                    return Err(BackendError::Dispatch(
                                        "quantize_activations: invalid affine metadata".into(),
                                    ));
                                }
                                out_bytes[0..4].copy_from_slice(&scale.to_le_bytes());
                                out_bytes[4..8].copy_from_slice(&offset.to_le_bytes());
                                for index in 0..numel {
                                    let quantized = ((f32_data[index] - offset) / scale)
                                        .round()
                                        .clamp(-128.0, 127.0)
                                        as i8;
                                    out_bytes[8 + index] = quantized as u8;
                                }
                            }
                        }
                        "dequantize_activations" => {
                            if params.len() != 3 || input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_activations: expected numel, mode, channel count, and one input"
                                        .into(),
                                ));
                            }
                            let numel = params[0];
                            if params[1] > 1 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_activations: invalid quantization mode".into(),
                                ));
                            }
                            let is_per_channel = params[1] == 1;
                            let num_channels = params[2];
                            if !is_per_channel && num_channels != 0 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_activations: per-tensor mode cannot declare channels"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            if output_slice.offset % std::mem::align_of::<f32>() != 0
                                || output_slice.size % std::mem::size_of::<f32>() != 0
                            {
                                return Err(BackendError::Dispatch(
                                    "dequantize_activations: output slice is not f32-aligned"
                                        .into(),
                                ));
                            }
                            let input_end = input_slice
                                .offset
                                .checked_add(input_slice.size)
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "dequantize_activations: input range overflows".into(),
                                    )
                                })?;
                            if input_slice.offset < out_end && out_start < input_end {
                                return Err(BackendError::Dispatch(
                                    "dequantize_activations: input and output slices overlap"
                                        .into(),
                                ));
                            }
                            let expected_output = numel.checked_mul(4).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "dequantize_activations: output size overflows".into(),
                                )
                            })?;
                            if output_slice.size != expected_output {
                                return Err(BackendError::Dispatch(format!(
                                    "dequantize_activations: output has {} bytes, expected {expected_output}",
                                    output_slice.size
                                )));
                            }
                            let in_data: &[u8] =
                                unsafe { arena.view_u8(input_slice.offset, input_slice.size) };
                            let read_f32 = |offset: usize| -> Result<f32, BackendError> {
                                let bytes = in_data.get(offset..offset + 4).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "dequantize_activations: truncated affine metadata".into(),
                                    )
                                })?;
                                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                            };

                            if is_per_channel {
                                if num_channels == 0 || in_data.len() < 8 {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_activations: invalid per-channel header".into(),
                                    ));
                                }
                                let nc = u32::from_le_bytes([
                                    in_data[0], in_data[1], in_data[2], in_data[3],
                                ]) as usize;
                                let chunk_size = u32::from_le_bytes([
                                    in_data[4], in_data[5], in_data[6], in_data[7],
                                ]) as usize;
                                if nc != num_channels || chunk_size == 0 {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_activations: per-channel header disagrees with parameters".into(),
                                    ));
                                }
                                let pair_bytes = nc.checked_mul(8).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "dequantize_activations: channel metadata size overflows"
                                            .into(),
                                    )
                                })?;
                                let data_start =
                                    8usize.checked_add(pair_bytes).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_activations: channel data offset overflows"
                                                .into(),
                                        )
                                    })?;
                                let encoded_values = nc.checked_mul(chunk_size).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "dequantize_activations: channel payload size overflows".into(),
                                    )
                                })?;
                                let expected_input =
                                    data_start.checked_add(encoded_values).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_activations: total input size overflows"
                                                .into(),
                                        )
                                    })?;
                                if encoded_values != numel || in_data.len() != expected_input {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_activations: per-channel payload length mismatch".into(),
                                    ));
                                }
                                let mut scales = Vec::with_capacity(nc);
                                let mut offsets = Vec::with_capacity(nc);
                                for channel in 0..nc {
                                    let scale = read_f32(8 + channel * 4)?;
                                    let offset = read_f32(8 + nc * 4 + channel * 4)?;
                                    if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                                        return Err(BackendError::Dispatch(
                                            "dequantize_activations: affine metadata must be finite with positive scales".into(),
                                        ));
                                    }
                                    scales.push(scale);
                                    offsets.push(offset);
                                }
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                for channel in 0..nc {
                                    let start = channel * chunk_size;
                                    for local in 0..chunk_size {
                                        let q = in_data[data_start + start + local] as i8;
                                        out_f32[start + local] =
                                            (q as f32) * scales[channel] + offsets[channel];
                                    }
                                }
                            } else {
                                let expected_input = numel.checked_add(8).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "dequantize_activations: input size overflows".into(),
                                    )
                                })?;
                                if in_data.len() != expected_input {
                                    return Err(BackendError::Dispatch(format!(
                                        "dequantize_activations: input has {} bytes, expected {expected_input}",
                                        in_data.len()
                                    )));
                                }
                                let scale = read_f32(0)?;
                                let offset = read_f32(4)?;
                                if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_activations: affine metadata must be finite with a positive scale".into(),
                                    ));
                                }
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                for (index, output) in out_f32.iter_mut().enumerate() {
                                    *output = (in_data[8 + index] as i8 as f32) * scale + offset;
                                }
                            }
                        }
                        _ => {
                            return Err(BackendError::UnsupportedOp(kernel_name.clone()));
                        }
                    }
                }
                Instruction::MemCopy { dst, src } => {
                    let data = arena.data_mut();
                    let src_start = src.offset;
                    let dst_start = dst.offset;
                    let len = dst.size.min(src.size);
                    let src_range = src_start..src_start + len;
                    data.copy_within(src_range, dst_start);
                }
                Instruction::Fill { dst, value } => {
                    let data = arena.data_mut();
                    let start = dst.offset;
                    let end = dst.offset + dst.size;
                    let bytes = &mut data[start..end];
                    let f32_slice = bytemuck::cast_slice_mut::<_, f32>(bytes);
                    f32_slice.fill(*value);
                }
                Instruction::WriteConst { dst, data } => {
                    let arena_data = arena.data_mut();
                    let end = (dst.offset + data.len()).min(arena_data.len());
                    arena_data[dst.offset..end].copy_from_slice(&data[..end - dst.offset]);
                }
            }

            // â”€â”€ Debug: per-instruction MaxPool canary check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            // After the current instruction has executed, check whether
            // any MaxPool primary slot has been overwritten.
            // Only active with `debug_canary` feature (expensive).
            #[cfg(feature = "debug_canary")]
            {
                let d = arena.data_mut();
                // Determine if this instruction IS a MaxPool kernel, and
                // if so, which index in maxpool_ranges it corresponds to.
                let is_mp_and_idx: Option<usize> = match instr {
                    Instruction::CallKernel {
                        kernel_name,
                        params,
                        output_slice,
                        ..
                    } if kernel_name == "pool_f32" && params.len() >= 4 && params[3] == 1 => {
                        maxpool_ranges
                            .iter()
                            .position(|&(off, _)| off == output_slice.offset)
                    }
                    _ => None,
                };

                if let Some(mp_idx) = is_mp_and_idx {
                    // This instruction just wrote MaxPool data â†’ snapshot
                    let (mp_off, mp_sz) = maxpool_ranges[mp_idx];
                    if mp_sz >= 4 && mp_off + 4 <= d.len() {
                        let bytes: [u8; 4] = d[mp_off..mp_off + 4].try_into().unwrap_or([0; 4]);
                        maxpool_snapshot[mp_idx] = Some(f32::from_le_bytes(bytes));
                        maxpool_seen[mp_idx] = true;
                        // Also log the kernel's output_slice for reference
                        if let Instruction::CallKernel {
                            kernel_name,
                            output_slice,
                            ..
                        } = instr
                        {
                            eprintln!(
                                "[FNN_DBG_CANARY] MaxPool nid={}: off={} sz={} first_f32={} (AFTER kernel={} out=[{},{})",
                                mp_idx, mp_off, mp_sz,
                                maxpool_snapshot[mp_idx].unwrap(),
                                kernel_name,
                                output_slice.offset,
                                output_slice.offset + output_slice.size,
                            );
                        }
                    }
                } else {
                    // Not a MaxPool kernel â€” check if any MaxPool was corrupted
                    for (mp_idx, &(mp_off, mp_sz)) in maxpool_ranges.iter().enumerate() {
                        if !maxpool_seen[mp_idx] {
                            continue; // MaxPool hasn't executed yet
                        }
                        if let Some(expected) = maxpool_snapshot[mp_idx] {
                            if mp_sz >= 4 && mp_off + 4 <= d.len() {
                                let bytes: [u8; 4] =
                                    d[mp_off..mp_off + 4].try_into().unwrap_or([0; 4]);
                                let actual = f32::from_le_bytes(bytes);
                                if actual.to_bits() != expected.to_bits() {
                                    let desc = match instr {
                                        Instruction::CallKernel {
                                            kernel_name,
                                            output_slice,
                                            ..
                                        } => format!(
                                            "kernel={} out=[{},{})",
                                            kernel_name,
                                            output_slice.offset,
                                            output_slice.offset + output_slice.size
                                        ),
                                        Instruction::MemCopy { dst, src } => {
                                            format!(
                                                "MemCopy dst=[{},{}) src=[{},{})",
                                                dst.offset,
                                                dst.offset + dst.size,
                                                src.offset,
                                                src.offset + src.size
                                            )
                                        }
                                        Instruction::Fill { dst, value } => {
                                            format!(
                                                "Fill dst=[{},{}) value={}",
                                                dst.offset,
                                                dst.offset + dst.size,
                                                value
                                            )
                                        }
                                        Instruction::WriteConst { dst, .. } => {
                                            format!(
                                                "WriteConst dst=[{},{})",
                                                dst.offset,
                                                dst.offset + dst.size
                                            )
                                        }
                                    };
                                    eprintln!(
                                        "[FNN_DBG_CORRUPT] MaxPool mp_off={} expected={} actual={} AFTER {}",
                                        mp_off, expected, actual, desc
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn write_arena(&self, arena: &CpuBuffer, offset: usize, data: &[u8]) {
        let _ = self.try_write_arena(arena, offset, data);
    }

    fn try_write_arena(
        &self,
        arena: &CpuBuffer,
        offset: usize,
        data: &[u8],
    ) -> Result<(), BackendError> {
        let end = offset
            .checked_add(data.len())
            .ok_or_else(|| BackendError::Dispatch("CPU arena write range overflows".into()))?;
        let buf = arena.data_mut();
        let capacity = buf.len();
        let destination = buf.get_mut(offset..end).ok_or_else(|| {
            BackendError::Dispatch(format!(
                "CPU arena write range {offset}..{end} exceeds {capacity} bytes"
            ))
        })?;
        destination.copy_from_slice(data);
        Ok(())
    }

    fn read_arena(&self, arena: &CpuBuffer, offset: usize, size: usize) -> Vec<u8> {
        self.try_read_arena(arena, offset, size).unwrap_or_default()
    }

    fn try_read_arena(
        &self,
        arena: &CpuBuffer,
        offset: usize,
        size: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| BackendError::Dispatch("CPU arena read range overflows".into()))?;
        let buf = arena.data_mut();
        let source = buf.get(offset..end).ok_or_else(|| {
            BackendError::Dispatch(format!(
                "CPU arena read range {offset}..{end} exceeds {} bytes",
                buf.len()
            ))
        })?;
        Ok(source.to_vec())
    }

    #[cfg(feature = "prepared-plan")]
    fn dispatch_with_persistent_view(
        &self,
        plan: &ExecutablePlan,
        arena: &CpuBuffer,
        shape_env: &ShapeEnv,
        persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
    ) -> Result<(), BackendError> {
        use crate::backend::prepared::PersistentPreparedWeights;

        plan.validate()?;
        let arena_size = arena.data_mut().len();
        if arena_size < plan.arena_size {
            return Err(BackendError::Dispatch(format!(
                "CPU persistent dispatch arena has {arena_size} bytes, plan requires {}",
                plan.arena_size
            )));
        }

        // Fast path: empty / missing view degrades to the standard
        // dispatch without any per-instruction overhead.
        let view: &PersistentPreparedWeights = match persistent_view {
            Some(v) if !v.is_empty() => v,
            _ => return self.dispatch(plan, arena, shape_env),
        };

        for instruction in &plan.instructions {
            match instruction {
                // Skip WriteConst for fp32 weight slots the persistent
                // view will satisfy directly.  Quantized weight slots are
                // NOT skipped — the kernel dispatch path for quantized
                // Conv2d/MatMul reads weights from the arena, not from
                // the persistent view, so the WriteConst must execute to
                // populate the arena.
                Instruction::WriteConst { dst, .. }
                    if view.get(&(dst.offset, dst.size)).is_some() =>
                {
                    continue;
                }

                // Conv2d fp32 + (optional) bias.  Resolved directly
                // from the persistent view when available; otherwise
                // falls through to the standard per-instruction
                // dispatch path which already handles the
                // input/weight/bias reads from the mutable arena.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "conv2d" | "conv2d_relu" | "conv2d_gelu" | "conv2d_silu"
                ) =>
                {
                    let has_override = input_slices
                        .get(1)
                        .map(|w| view.get(&(w.offset, w.size)).is_some())
                        .unwrap_or(false)
                        || input_slices
                            .get(2)
                            .map(|b| view.get(&(b.offset, b.size)).is_some())
                            .unwrap_or(false);
                    if has_override {
                        dispatch_conv2d_fp32_with_view(
                            arena,
                            shape_env,
                            kernel_name,
                            input_slices,
                            *output_slice,
                            params,
                            node_id.unwrap_or(0),
                            view,
                        )?;
                    } else {
                        let single = ExecutablePlan {
                            instructions: vec![instruction.clone()],
                            arena_size: plan.arena_size,
                            levels: vec![0],
                        };
                        self.dispatch(&single, arena, shape_env)?;
                    }
                }

                // Fp32 MatMul family (matmul, matmul_relu/gelu/silu,
                // fused_matmul_add_*).  Resolved from the persistent
                // view for both the B slot and the optional bias slot.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "matmul"
                        | "matmul_relu"
                        | "matmul_gelu"
                        | "matmul_silu"
                        | "fused_matmul_add_relu"
                        | "fused_matmul_add_gelu"
                        | "fused_matmul_add_silu"
                ) =>
                {
                    let has_override = input_slices
                        .get(1)
                        .map(|b| view.get(&(b.offset, b.size)).is_some())
                        .unwrap_or(false)
                        || input_slices
                            .get(2)
                            .map(|b| view.get(&(b.offset, b.size)).is_some())
                            .unwrap_or(false);
                    if has_override {
                        dispatch_matmul_fp32_with_view(
                            arena,
                            shape_env,
                            kernel_name,
                            input_slices,
                            *output_slice,
                            params,
                            param_dims,
                            view,
                        )?;
                    } else {
                        let single = ExecutablePlan {
                            instructions: vec![instruction.clone()],
                            arena_size: plan.arena_size,
                            levels: vec![0],
                        };
                        self.dispatch(&single, arena, shape_env)?;
                    }
                }

                // Quantized MatMul family (u4, u8, and i8-activation
                // variants).  Fall back to the standard dispatch path
                // â€” PersistentPreparedWeights currently stores only f32
                // payloads, so a zero-copy quantized path isn't wired yet.
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                    node_id,
                    ..
                } if matches!(
                    kernel_name.as_str(),
                    "matmul_i4" | "matmul_i4_i8" | "matmul_i8" | "matmul_i8_i8"
                ) =>
                {
                    let single = ExecutablePlan {
                        instructions: vec![instruction.clone()],
                        arena_size: plan.arena_size,
                        levels: vec![0],
                    };
                    self.dispatch(&single, arena, shape_env)?;
                }

                // Everything else: defer to the standard per-instruction
                // dispatch so we keep the existing single-threaded,
                // sequential behaviour for the rest of the plan.
                _ => {
                    let single = ExecutablePlan {
                        instructions: vec![instruction.clone()],
                        arena_size: plan.arena_size,
                        levels: vec![0],
                    };
                    self.dispatch(&single, arena, shape_env)?;
                }
            }
        }
        Ok(())
    }
}

// â”€â”€ Persistent-view dispatch helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// These helpers are private to the CpuBackend. They are intentionally
// near-clones of the corresponding branches in `dispatch()` so the
// fp32 Conv2d / MatMul kernels can borrow the weight/bias bytes
// directly from the [`PersistentPreparedWeights`] view instead of
// pulling them out of the mutable arena.  The input tensor is still
// read from the mutable arena; only the static weight / bias slots
// are routed through the view.

#[cfg(feature = "prepared-plan")]
fn dispatch_conv2d_fp32_with_view(
    arena: &CpuBuffer,
    _shape_env: &ShapeEnv,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    _node_id: usize,
    view: &crate::backend::prepared::PersistentPreparedWeights,
) -> Result<(), BackendError> {
    use crate::backend::cpu::microkernels::{conv2d_f32_im2col_gemm, ConvActivation};

    let fused_act: Option<ConvActivation> = match kernel_name {
        "conv2d_relu" => Some(ConvActivation::Relu),
        "conv2d_gelu" => Some(ConvActivation::Gelu),
        "conv2d_silu" => Some(ConvActivation::Silu),
        _ => None,
    };

    if !matches!(input_slices.len(), 2 | 3) {
        return Err(BackendError::Dispatch(format!(
            "conv2d_persistent: expected 2 or 3 input slices, got {}",
            input_slices.len()
        )));
    }
    if params.len() != 15 {
        return Err(BackendError::Dispatch(
            "conv2d_persistent: expected the resolved 15-parameter contract".into(),
        ));
    }
    let input_slice = input_slices[0];
    let weight_slice = input_slices[1];
    let bias_slice = input_slices.get(2).copied();
    let stride = params[0];
    let padding = params[1];
    let dilation = params[2];
    let groups = params[3];
    let c = params[4];
    let h = params[5];
    let w = params[6];
    let kh = params[7];
    let kw = params[8];
    let n_in = params[9];
    let f_out = params[10];
    if stride == 0
        || dilation == 0
        || groups == 0
        || c == 0
        || h == 0
        || w == 0
        || kh == 0
        || kw == 0
        || n_in == 0
        || f_out == 0
        || !c.is_multiple_of(groups)
    {
        return Err(BackendError::Dispatch(
            "conv2d_persistent: invalid convolution geometry".into(),
        ));
    }
    let checked_product = |values: &[usize], label: &str| -> Result<usize, BackendError> {
        values
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| {
                BackendError::Dispatch(format!("conv2d_persistent: {label} size overflows"))
            })
    };
    let padded_h = h
        .checked_add(padding.checked_mul(2).ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: vertical padding overflows".into())
        })?)
        .ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: padded height overflows".into())
        })?;
    let padded_w = w
        .checked_add(padding.checked_mul(2).ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: horizontal padding overflows".into())
        })?)
        .ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: padded width overflows".into())
        })?;
    let kernel_h = dilation
        .checked_mul(kh - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: kernel height overflows".into())
        })?;
    let kernel_w = dilation
        .checked_mul(kw - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| {
            BackendError::Dispatch("conv2d_persistent: kernel width overflows".into())
        })?;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(BackendError::Dispatch(
            "conv2d_persistent: kernel exceeds padded input".into(),
        ));
    }
    let h_out = (padded_h - kernel_h) / stride + 1;
    let w_out = (padded_w - kernel_w) / stride + 1;
    let expected_input = checked_product(&[n_in, c, h, w], "input")?;
    let expected_weight = checked_product(&[f_out, c / groups, kh, kw], "weight")?;
    let expected_output = checked_product(&[n_in, f_out, h_out, w_out], "output")?;
    let expected_bias = checked_product(&[f_out], "bias")?;
    if !input_slice.offset.is_multiple_of(4)
        || !weight_slice.offset.is_multiple_of(4)
        || !output_slice.offset.is_multiple_of(4)
        || input_slice.size != expected_input
        || weight_slice.size != expected_weight
        || output_slice.size != expected_output
        || bias_slice.is_some_and(|bias| bias.offset % 4 != 0 || bias.size != expected_bias)
    {
        return Err(BackendError::Dispatch(
            "conv2d_persistent: storage does not match convolution geometry".into(),
        ));
    }
    // Resolve the weight / bias f32 slices.  Persistent-view entries
    // are borrowed directly (no copy); non-overridden slots fall
    // back to a Vec copy of the arena bytes (these are rare in
    // practice — the no-copy plan only filters WriteConst for
    // overridden slots, so any non-overridden slot still has its
    // WriteConst running and the arena bytes are valid).
    let weight_f32: Vec<f32>;
    let weight_slice_ref: &[f32] = match view.get(&(weight_slice.offset, weight_slice.size)) {
        Some(payload) => payload, // borrow directly, no copy
        None => {
            let d = arena.data_mut();
            weight_f32 = bytemuck::cast_slice::<_, f32>(
                &d[weight_slice.offset..weight_slice.offset + weight_slice.size],
            )
            .to_vec();
            &weight_f32
        }
    };
    let bias_f32: Vec<f32>;
    let bias_slice_ref: &[f32] = if let Some(b) = bias_slice {
        match view.get(&(b.offset, b.size)) {
            Some(payload) => payload,
            None => {
                let d = arena.data_mut();
                bias_f32 = bytemuck::cast_slice::<_, f32>(&d[b.offset..b.offset + b.size]).to_vec();
                &bias_f32
            }
        }
    } else {
        &[]
    };

    // Borrow the input tensor from the arena.  When the input and output
    // overlap in the arena — which happens with optimized memory plans
    // that reuse dead regions — we must copy the input to a local Vec
    // to avoid UB from creating simultaneous &[f32] and &mut [f32]
    // references to the same arena memory.
    let input_overlaps = arena::ranges_overlap(
        input_slice.offset,
        input_slice.offset + input_slice.size,
        output_slice.offset,
        output_slice.offset + output_slice.size,
    );
    let input_owned: Vec<f32>;
    let input_f32: &[f32] = if input_overlaps {
        let d = arena.data_mut();
        input_owned = bytemuck::cast_slice::<_, f32>(
            &d[input_slice.offset..input_slice.offset + input_slice.size],
        )
        .to_vec();
        &input_owned
    } else {
        let d = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(
            &d[input_slice.offset..input_slice.offset + input_slice.size],
        )
    };
    let out_f32: &mut [f32] = {
        let d = arena.data_mut();
        bytemuck::cast_slice_mut::<_, f32>(
            &mut d[output_slice.offset..output_slice.offset + output_slice.size],
        )
    };

    conv2d_f32_im2col_gemm(
        input_f32,
        weight_slice_ref,
        bias_slice_ref,
        out_f32,
        n_in,
        c,
        h,
        w,
        f_out,
        kh,
        kw,
        stride,
        padding,
        dilation,
        groups,
        fused_act,
    );
    Ok(())
}

#[cfg(feature = "prepared-plan")]
fn dispatch_matmul_fp32_with_view(
    arena: &CpuBuffer,
    shape_env: &ShapeEnv,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    view: &crate::backend::prepared::PersistentPreparedWeights,
) -> Result<(), BackendError> {
    use crate::backend::cpu::blas::matmul_blas_into;

    if !matches!(input_slices.len(), 2 | 3) {
        return Err(BackendError::Dispatch(format!(
            "matmul_persistent: expected 2 or 3 input slices, got {}",
            input_slices.len()
        )));
    }
    let a_slice = input_slices[0];
    let b_slice = input_slices[1];
    let bias_slice = input_slices.get(2).copied();
    let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
    let &[m, k, n] = &matmul_params[..] else {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: expected params [M,K,N]"
        )));
    };
    if m == 0 || k == 0 || n == 0 {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: matrix dimensions must be positive"
        )));
    }
    let checked_f32_size = |dimensions: &[usize], label: &str| {
        dimensions
            .iter()
            .try_fold(1usize, |acc, value| acc.checked_mul(*value))
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| BackendError::Dispatch(format!("{kernel_name}: {label} size overflows")))
    };
    let expected_a = checked_f32_size(&[m, k], "left operand")?;
    let expected_b = checked_f32_size(&[k, n], "right operand")?;
    let expected_output = checked_f32_size(&[m, n], "output")?;
    let expected_bias = checked_f32_size(&[n], "bias")?;
    if !a_slice.offset.is_multiple_of(4)
        || !b_slice.offset.is_multiple_of(4)
        || !output_slice.offset.is_multiple_of(4)
        || a_slice.size != expected_a
        || b_slice.size != expected_b
        || output_slice.size != expected_output
        || bias_slice.is_some_and(|bias| bias.offset % 4 != 0 || bias.size != expected_bias)
    {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: storage does not match matrix dimensions"
        )));
    }
    let output_end = output_slice.offset + output_slice.size;
    let b_end = b_slice.offset + b_slice.size;
    if b_slice.offset < output_end && output_slice.offset < b_end {
        return Err(BackendError::Dispatch(format!(
            "{kernel_name}: right operand and output slices overlap"
        )));
    }
    if let Some(bias) = bias_slice {
        let bias_end = bias.offset + bias.size;
        if bias.offset < output_end && output_slice.offset < bias_end {
            return Err(BackendError::Dispatch(format!(
                "{kernel_name}: bias and output slices overlap"
            )));
        }
    }

    // Resolve the B (weight) / bias f32 slices.
    let b_f32: Vec<f32>;
    let b_slice_ref: &[f32] = match view.get(&(b_slice.offset, b_slice.size)) {
        Some(payload) => payload, // borrow directly, no copy
        None => {
            let d = arena.data_mut();
            b_f32 =
                bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size])
                    .to_vec();
            &b_f32
        }
    };
    let bias_f32: Vec<f32>;
    let bias_slice_ref: &[f32] = if let Some(b) = bias_slice {
        match view.get(&(b.offset, b.size)) {
            Some(payload) => payload,
            None => {
                let d = arena.data_mut();
                bias_f32 = bytemuck::cast_slice::<_, f32>(&d[b.offset..b.offset + b.size]).to_vec();
                &bias_f32
            }
        }
    } else {
        &[]
    };

    // Borrow the activation tensor when disjoint. Optimized plans may reuse
    // its dead slot for output, in which case materialize it before mutation.
    let a_end = a_slice.offset + a_slice.size;
    let a_overlaps_output = a_slice.offset < output_end && output_slice.offset < a_end;
    let a_owned: Vec<f32>;
    let a_f32: &[f32] = if a_overlaps_output {
        let data = arena.data_mut();
        a_owned = bytemuck::cast_slice::<_, f32>(&data[a_slice.offset..a_end]).to_vec();
        &a_owned
    } else {
        let data = arena.data_mut();
        bytemuck::cast_slice::<_, f32>(&data[a_slice.offset..a_end])
    };

    let out_f32: &mut [f32] = {
        let d = arena.data_mut();
        bytemuck::cast_slice_mut::<_, f32>(
            &mut d[output_slice.offset..output_slice.offset + output_slice.size],
        )
    };

    matmul_blas_into(a_f32, b_slice_ref, out_f32, m, k, n);

    // Apply fused activation / bias on top of the GEMM output, mirroring
    // the `matmul_activation_dispatch` semantics.
    let has_bias = !bias_slice_ref.is_empty();
    let act: fn(f32) -> f32 = match kernel_name {
        "matmul_relu" | "fused_matmul_add_relu" => |x| x.max(0.0),
        "matmul_gelu" | "fused_matmul_add_gelu" => |x: f32| {
            let x3 = x * x * x;
            let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
            let t = tanh_arg.tanh();
            0.5 * x * (1.0 + t)
        },
        "matmul_silu" | "fused_matmul_add_silu" => |x: f32| x / (1.0 + (-x).exp()),
        _ => |x| x,
    };
    for i in 0..out_f32.len() {
        let x = out_f32[i]
            + if has_bias && i % n < bias_slice_ref.len() {
                bias_slice_ref[i % n]
            } else {
                0.0
            };
        out_f32[i] = act(x);
    }
    Ok(())
}
