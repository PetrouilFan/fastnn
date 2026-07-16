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
/// `CpuBuffer` is `Send` but deliberately not `Sync`: moving an arena between
/// threads is valid, while sharing one across concurrent dispatches is not.
/// Mutable byte access is crate-private so safe external code cannot manufacture
/// aliased mutable slices.
///
/// ```compile_fail
/// use fastnn::backend::cpu::CpuBuffer;
/// let buffer = CpuBuffer::new(vec![0; 16]);
/// let _ = buffer.data_mut();
/// ```
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
    pub(crate) fn data_mut(&self) -> &mut [u8] {
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

// SAFETY: ownership of the arena may move between threads. `CpuBuffer` is
// intentionally not `Sync`, so safe code cannot access one arena concurrently.
unsafe impl Send for CpuBuffer {}

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

fn try_filled_vec<T: Clone>(len: usize, value: T, context: &str) -> Result<Vec<T>, BackendError> {
    let mut vector = Vec::new();
    vector.try_reserve_exact(len).map_err(|error| {
        BackendError::Dispatch(format!("{context}: metadata allocation failed: {error}"))
    })?;
    vector.resize(len, value);
    Ok(vector)
}

fn try_copy_slice<T: Copy>(source: &[T], context: &str) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values.try_reserve_exact(source.len()).map_err(|error| {
        BackendError::Dispatch(format!("{context}: allocation failed: {error}"))
    })?;
    values.extend_from_slice(source);
    Ok(values)
}

fn try_with_unary_f32_slices<R>(
    arena: &CpuBuffer,
    input: BufferSlice,
    output: BufferSlice,
    context: &str,
    f: impl FnOnce(&[f32], &mut [f32]) -> R,
) -> Result<R, BackendError> {
    let input_end = input.offset + input.size;
    let output_end = output.offset + output.size;
    if arena::ranges_overlap(input.offset, input_end, output.offset, output_end) {
        let input_copy = {
            let arena_data = arena.data_mut();
            try_copy_slice(
                bytemuck::cast_slice::<_, f32>(&arena_data[input.offset..input_end]),
                context,
            )?
        };
        telemetry::record_arena_temp_copy(input.size);
        let output_f32 = unsafe { arena.view_f32_mut(output.offset, output.size) };
        Ok(f(&input_copy, output_f32))
    } else {
        let arena_data = arena.data_mut();
        // SAFETY: callers validate complete aligned f32 ranges before this helper;
        // the ranges were proven disjoint, so shared input and mutable output do not alias.
        unsafe {
            let input_f32 = std::slice::from_raw_parts(
                arena_data.as_ptr().add(input.offset).cast::<f32>(),
                input.size / std::mem::size_of::<f32>(),
            );
            let output_f32 = std::slice::from_raw_parts_mut(
                arena_data.as_mut_ptr().add(output.offset).cast::<f32>(),
                output.size / std::mem::size_of::<f32>(),
            );
            Ok(f(input_f32, output_f32))
        }
    }
}

fn try_decode_f16(source: &[u8], context: &str) -> Result<Vec<f32>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len() / 2)
        .map_err(|error| {
            BackendError::Dispatch(format!("{context}: allocation failed: {error}"))
        })?;
    for chunk in source.chunks_exact(2) {
        values.push(half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
    }
    Ok(values)
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
                    memory_plan.slots.get(&input_id).map(|slot| {
                        let size = graph
                            .get_node(input_id)
                            .and_then(|input| match &input.opcode {
                                Opcode::Constant(TensorValue::Data { bytes, .. }) => {
                                    Some(bytes.len())
                                }
                                _ => None,
                            })
                            .unwrap_or(slot.size);
                        BufferSlice::new(slot.offset, size)
                    })
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
                    if is_quantized {
                        let weight_id = node.inputs.get(1).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "quantized matmul node {node_id} is missing its weight input"
                            ))
                        })?;
                        let weight_node = graph.get_node(*weight_id).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "quantized matmul node {node_id} references missing weight node {weight_id}"
                            ))
                        })?;
                        if weight_node.output_type.shape.is_empty() {
                            return Err(BackendError::Compilation(format!(
                                "quantized matmul node {node_id} weight shape must not be empty"
                            )));
                        }
                    }
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
                                    } else if w_shape.len() >= 2 && scales.len() > w_shape[0] {
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
                        let raw = node.attrs.get("negative_slope").ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "leaky relu node {node_id} is missing negative_slope"
                            ))
                        })?;
                        let slope: f32 = raw.parse().map_err(|_| {
                            BackendError::Compilation(format!(
                                "leaky relu node {node_id} has invalid negative_slope"
                            ))
                        })?;
                        if !slope.is_finite() {
                            return Err(BackendError::Compilation(format!(
                                "leaky relu node {node_id} requires a finite negative_slope"
                            )));
                        }
                        extra_params.push(slope.to_bits() as usize);
                    }
                    if let Opcode::Clamp = node.opcode {
                        let parse_bound = |name: &str| -> Result<f32, BackendError> {
                            let raw = node.attrs.get(name).ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "clamp node {node_id} is missing {name}"
                                ))
                            })?;
                            let value: f32 = raw.parse().map_err(|_| {
                                BackendError::Compilation(format!(
                                    "clamp node {node_id} has invalid {name}"
                                ))
                            })?;
                            if !value.is_finite() {
                                return Err(BackendError::Compilation(format!(
                                    "clamp node {node_id} requires a finite {name}"
                                )));
                            }
                            Ok(value)
                        };
                        let min = parse_bound("min")?;
                        let max = parse_bound("max")?;
                        if min > max {
                            return Err(BackendError::Compilation(format!(
                                "clamp node {node_id} requires min <= max"
                            )));
                        }
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
                    let parse_attr = |name: &str| -> Result<usize, BackendError> {
                        node.attrs
                            .get(name)
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "conv2d node {node_id} is missing {name}"
                                ))
                            })?
                            .parse::<usize>()
                            .map_err(|_| {
                                BackendError::Compilation(format!(
                                    "conv2d node {node_id} has invalid {name}"
                                ))
                            })
                    };
                    let stride = parse_attr("stride")?;
                    let padding = parse_attr("padding")?;
                    let dilation = parse_attr("dilation")?;
                    let groups = parse_attr("groups")?;
                    if stride == 0 || dilation == 0 || groups == 0 {
                        return Err(BackendError::Compilation(format!(
                            "conv2d node {node_id} requires positive stride, dilation, and groups"
                        )));
                    }
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
                    if is_quantized {
                        let weight_id = node.inputs.get(1).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "quantized convolution node {node_id} is missing its weight input"
                            ))
                        })?;
                        let weight_node = graph.get_node(*weight_id).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "quantized convolution node {node_id} references missing weight node {weight_id}"
                            ))
                        })?;
                        if weight_node.output_type.shape.is_empty() {
                            return Err(BackendError::Compilation(format!(
                                "quantized convolution node {node_id} weight shape must not be empty"
                            )));
                        }
                    }
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
                                    } else if w_shape.len() >= 2 && scales.len() > w_shape[0] {
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
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "softmax node {node_id} is missing axis"
                            ))
                        })?
                        .parse()
                        .map_err(|_| {
                            BackendError::Compilation(format!(
                                "softmax node {node_id} has invalid axis"
                            ))
                        })?;
                    let input_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "softmax node {node_id} is missing its input shape"
                        ))
                    })?;
                    let rank = i64::try_from(input_shape.len()).map_err(|_| {
                        BackendError::Compilation(format!(
                            "softmax node {node_id} rank does not fit i64"
                        ))
                    })?;
                    let normalized_axis = if axis < 0 { rank + axis } else { axis };
                    if normalized_axis < 0 || normalized_axis >= rank {
                        return Err(BackendError::Compilation(format!(
                            "softmax node {node_id} axis {axis} is out of range for rank {rank}"
                        )));
                    }
                    let normalized_axis = usize::try_from(normalized_axis).map_err(|_| {
                        BackendError::Compilation(format!(
                            "softmax node {node_id} axis conversion failed"
                        ))
                    })?;
                    let axis_dim = usize::try_from(input_shape[normalized_axis]).map_err(|_| {
                        BackendError::Compilation(format!(
                            "softmax node {node_id} axis size does not fit usize"
                        ))
                    })?;
                    let axis_dim_dim = input_shape_dims
                        .first()
                        .and_then(|shape| shape.get(normalized_axis))
                        .cloned()
                        .ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "softmax node {node_id} is missing symbolic axis metadata"
                            ))
                        })?;
                    let stride = input_shape[normalized_axis + 1..].iter().try_fold(
                        1usize,
                        |product, &dimension| {
                            let dimension = usize::try_from(dimension).map_err(|_| {
                                BackendError::Compilation(format!(
                                    "softmax node {node_id} stride dimension does not fit usize"
                                ))
                            })?;
                            product.checked_mul(dimension).ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "softmax node {node_id} stride overflows"
                                ))
                            })
                        },
                    )?;

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
                        .ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "concat node {node_id} is missing axis"
                            ))
                        })?
                        .parse()
                        .map_err(|_| {
                            BackendError::Compilation(format!(
                                "concat node {node_id} has invalid axis"
                            ))
                        })?;
                    let output_shape: Vec<u64> = node
                        .output_type
                        .shape
                        .iter()
                        .map(|dimension| dimension.evaluate().unwrap_or(symbol_max))
                        .collect();
                    let rank = output_shape.len();
                    if axis >= rank {
                        return Err(BackendError::Compilation(format!(
                            "concat node {node_id} axis {axis} is out of range for rank {rank}"
                        )));
                    }
                    let checked_product = |dimensions: &[u64], label: &str| {
                        dimensions.iter().try_fold(1u64, |product, &dimension| {
                            product.checked_mul(dimension).ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "concat node {node_id} {label} overflows"
                                ))
                            })
                        })
                    };
                    let inner_stride = checked_product(&output_shape[axis + 1..], "inner stride")?;
                    let outer_count = checked_product(&output_shape[..axis], "outer count")?;
                    let inner_stride = usize::try_from(inner_stride).map_err(|_| {
                        BackendError::Compilation(format!(
                            "concat node {node_id} inner stride does not fit usize"
                        ))
                    })?;
                    let outer_count = usize::try_from(outer_count).map_err(|_| {
                        BackendError::Compilation(format!(
                            "concat node {node_id} outer count does not fit usize"
                        ))
                    })?;
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
                        params: vec![axis, inner_stride, outer_count],
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
                    let parsed_pads: Vec<usize> = pads_str
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    let input_shape = input_shapes.first().cloned().unwrap_or_default();
                    let rank = input_shape.len();
                    let mut pad_params = Vec::with_capacity(1 + rank * 3);
                    pad_params.push(rank);
                    pad_params.extend(input_shape.into_iter().map(|size| size as usize));
                    for dimension in 0..rank {
                        pad_params.push(parsed_pads.get(dimension * 2).copied().unwrap_or(0));
                        pad_params.push(parsed_pads.get(dimension * 2 + 1).copied().unwrap_or(0));
                    }
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "pad_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: pad_params,
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
                    let data_shape = input_shapes.first().cloned().unwrap_or_default();
                    let indices_numel = input_shapes
                        .get(1)
                        .and_then(|shape| {
                            shape
                                .iter()
                                .try_fold(1u64, |count, size| count.checked_mul(*size))
                        })
                        .unwrap_or(0) as usize;
                    let mut gather_params = Vec::with_capacity(data_shape.len() + 3);
                    gather_params.push(data_shape.len());
                    gather_params.extend(data_shape.into_iter().map(|size| size as usize));
                    gather_params.extend([indices_numel, axis]);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "gather".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: gather_params,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Slice => {
                    let parse_attr = |name: &str| -> Result<i64, BackendError> {
                        node.attrs
                            .get(name)
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "slice node {node_id} is missing {name}"
                                ))
                            })?
                            .parse::<i64>()
                            .map_err(|_| {
                                BackendError::Compilation(format!(
                                    "slice node {node_id} has invalid {name}"
                                ))
                            })
                    };
                    let dim_value = parse_attr("dim")?;
                    let start_value = parse_attr("start")?;
                    let end_value = parse_attr("end")?;
                    if dim_value < 0 || start_value < 0 {
                        return Err(BackendError::Compilation(format!(
                            "slice node {node_id} requires non-negative dim and start"
                        )));
                    }
                    let dim = usize::try_from(dim_value).map_err(|_| {
                        BackendError::Compilation(format!(
                            "slice node {node_id} dimension does not fit usize"
                        ))
                    })?;
                    let input_shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "slice node {node_id} is missing its input shape"
                        ))
                    })?;
                    if dim >= input_shape.len() {
                        return Err(BackendError::Compilation(format!(
                            "slice node {node_id} dimension {dim} is out of range for rank {}",
                            input_shape.len()
                        )));
                    }
                    let dimension_size = i64::try_from(input_shape[dim]).map_err(|_| {
                        BackendError::Compilation(format!(
                            "slice node {node_id} dimension size does not fit i64"
                        ))
                    })?;
                    let normalized_end = if end_value < 0 {
                        dimension_size
                            .checked_add(end_value)
                            .and_then(|value| value.checked_add(1))
                            .ok_or_else(|| {
                                BackendError::Compilation(format!(
                                    "slice node {node_id} negative end overflows"
                                ))
                            })?
                    } else {
                        end_value
                    };
                    if start_value >= dimension_size
                        || normalized_end > dimension_size
                        || start_value >= normalized_end
                    {
                        return Err(BackendError::Compilation(format!(
                            "slice node {node_id} has invalid range [{start_value}, {normalized_end}) for dimension size {dimension_size}"
                        )));
                    }
                    let start = usize::try_from(start_value).map_err(|_| {
                        BackendError::Compilation(format!(
                            "slice node {node_id} start does not fit usize"
                        ))
                    })?;
                    let end = usize::try_from(normalized_end).map_err(|_| {
                        BackendError::Compilation(format!(
                            "slice node {node_id} end does not fit usize"
                        ))
                    })?;
                    let mut slice_params = Vec::with_capacity(input_shape.len() + 4);
                    slice_params.push(input_shape.len());
                    for &size in input_shape {
                        slice_params.push(usize::try_from(size).map_err(|_| {
                            BackendError::Compilation(format!(
                                "slice node {node_id} input dimension does not fit usize"
                            ))
                        })?);
                    }
                    slice_params.extend([dim, start, end]);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "slice_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: slice_params,
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
                    let (is_mean, is_max) = match node.opcode {
                        Opcode::ReduceMean => (1, 0),
                        Opcode::ReduceMax => (0, 1),
                        _ => (0, 0), // ReduceSum
                    };
                    let shape_dims = input_shape_dims
                        .first()
                        .filter(|shape| !shape.is_empty())
                        .cloned()
                        .unwrap_or_else(|| {
                            input_shapes
                                .first()
                                .map(|shape| {
                                    shape
                                        .iter()
                                        .copied()
                                        .map(DimExpr::Known)
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default()
                        });
                    let mut reduce_params = Vec::with_capacity(shape_dims.len() + 4);
                    reduce_params.push(shape_dims.len());
                    reduce_params.extend(shape_dims.iter().map(|dimension| {
                        dimension
                            .evaluate()
                            .unwrap_or_else(|| crate::ir::SYMBOL_DIM_MAX.load(Ordering::Relaxed))
                            as usize
                    }));
                    reduce_params.extend([axis, is_mean, is_max]);

                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "reduce_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: reduce_params,
                        param_dims: (!shape_dims.is_empty()).then_some(shape_dims),
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
                    let shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "PReLU node {node_id} is missing input geometry"
                        ))
                    })?;
                    if shape.is_empty() {
                        return Err(BackendError::Compilation(format!(
                            "PReLU node {node_id} requires positive input rank"
                        )));
                    }
                    let mut params = Vec::new();
                    params.try_reserve_exact(shape.len() + 1).map_err(|error| {
                        BackendError::Compilation(format!(
                            "PReLU node {node_id} metadata allocation failed: {error}"
                        ))
                    })?;
                    params.push(shape.len());
                    for dimension in shape {
                        params.push(usize::try_from(*dimension).map_err(|_| {
                            BackendError::Compilation(format!(
                                "PReLU node {node_id} dimension does not fit usize"
                            ))
                        })?);
                    }
                    let dynamic_shape = input_shape_dims
                        .first()
                        .filter(|dims| dims.len() == shape.len())
                        .cloned();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "prelu".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params,
                        param_dims: dynamic_shape,
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
                    let h_in = input_shapes
                        .first()
                        .and_then(|shape| shape.get(2).copied())
                        .unwrap_or(1) as usize;
                    let w_in = input_shapes
                        .first()
                        .and_then(|shape| shape.get(3).copied())
                        .unwrap_or(1) as usize;
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "adaptive_avg_pool2d".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![out_h, out_w, h_in, w_in],
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
                    let input_shape = input_shapes.first().cloned().unwrap_or_default();
                    let mut repeat_params =
                        Vec::with_capacity(1 + input_shape.len() + repeats.len());
                    repeat_params.push(input_shape.len());
                    repeat_params.extend(input_shape.into_iter().map(|dim| dim as usize));
                    repeat_params.extend(repeats);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "repeat".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: repeat_params,
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
                    let input_shape = input_shapes.first().cloned().unwrap_or_default();
                    let mut cumsum_params = Vec::with_capacity(input_shape.len() + 4);
                    cumsum_params.push(input_shape.len());
                    cumsum_params.extend(input_shape.into_iter().map(|size| size as usize));
                    cumsum_params.extend([dim, exclusive, rev]);
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "cumsum".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: cumsum_params,
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
                    let shape = input_shapes.first().ok_or_else(|| {
                        BackendError::Compilation(format!(
                            "TopK node {node_id} is missing input geometry"
                        ))
                    })?;
                    if shape.is_empty() {
                        return Err(BackendError::Compilation(format!(
                            "TopK node {node_id} requires positive input rank"
                        )));
                    }
                    let rank = shape.len();
                    let normalized = if axis < 0 {
                        axis.checked_add(rank as i64).ok_or_else(|| {
                            BackendError::Compilation(format!(
                                "TopK node {node_id} axis normalization overflows"
                            ))
                        })?
                    } else {
                        axis
                    };
                    if normalized < 0 || normalized as usize >= rank {
                        return Err(BackendError::Compilation(format!(
                            "TopK node {node_id} axis {axis} is out of range for rank {rank}"
                        )));
                    }
                    let mut params = Vec::with_capacity(rank + 3);
                    params.extend([k, rank]);
                    for dimension in shape {
                        params.push(usize::try_from(*dimension).map_err(|_| {
                            BackendError::Compilation(format!(
                                "TopK node {node_id} dimension does not fit usize"
                            ))
                        })?);
                    }
                    params.push(normalized as usize);
                    let dynamic_shape = input_shape_dims
                        .first()
                        .filter(|dims| dims.len() == rank)
                        .cloned();
                    instructions.push(Instruction::CallKernel {
                        node_id: Some(node_id),
                        kernel_name: "topk_fused".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        params,
                        param_dims: dynamic_shape,
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
                    } else {
                        params.push(0); // flag: derive affine metadata from the runtime input
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
                    // Runtime Quantize produces a self-describing payload with geometry
                    // and affine metadata in its header. Static packed values carry
                    // their affine metadata in the IR and contain packed words only.
                    let has_metadata = input_node.opcode != Opcode::Quantize && !scales.is_empty();
                    // 0 = self-describing runtime payload, 2 = static
                    // PackedTensor words followed by its explicit 64-byte SIMD margin.
                    let format_flag: usize = if has_metadata { 2 } else { 0 };
                    // [numel, format, bit_width, channels, scales..., offsets...]
                    let mut params = vec![numel, format_flag, bit_width];
                    let num_channels = scales.len();
                    params.push(num_channels);
                    if has_metadata {
                        for &scale in &scales {
                            params.push(scale.to_bits() as usize);
                        }
                        for &offset in &dequant_offsets {
                            params.push(offset.to_bits() as usize);
                        }
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

                    let is_unary_f32 = matches!(
                        kernel_name.as_str(),
                        "relu_f32"
                            | "gelu_f32"
                            | "silu_f32"
                            | "exp_f32"
                            | "log_f32"
                            | "sqrt_f32"
                            | "neg_f32"
                            | "abs_f32"
                            | "sigmoid_f32"
                            | "tanh_f32"
                            | "leaky_relu_f32"
                            | "elu_f32"
                            | "softplus_f32"
                            | "hardswish_f32"
                            | "clamp_f32"
                            | "sign_f32"
                            | "round_f32"
                            | "logical_not_f32"
                            | "log_softmax_f32"
                            | "mish_f32"
                            | "erf_f32"
                    );
                    if is_unary_f32 {
                        let input = input_slices.first().copied();
                        if input_slices.len() != 1
                            || input.is_none_or(|slice| {
                                slice.size != output_slice.size
                                    || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                    || !slice.size.is_multiple_of(std::mem::size_of::<f32>())
                            })
                            || !output_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                            || !output_slice.size.is_multiple_of(std::mem::size_of::<f32>())
                        {
                            return Err(BackendError::Dispatch(format!(
                                "{kernel_name}: unary f32 input and output storage must match"
                            )));
                        }
                    }

                    let is_binary_f32 = matches!(
                        kernel_name.as_str(),
                        "add_f32"
                            | "sub_f32"
                            | "mul_f32"
                            | "div_f32"
                            | "max_f32"
                            | "min_f32"
                            | "add_relu_f32"
                            | "sub_relu_f32"
                            | "mul_relu_f32"
                            | "div_relu_f32"
                            | "add_gelu_f32"
                            | "sub_gelu_f32"
                            | "mul_gelu_f32"
                            | "div_gelu_f32"
                            | "add_silu_f32"
                            | "sub_silu_f32"
                            | "mul_silu_f32"
                            | "div_silu_f32"
                    );
                    if is_binary_f32 {
                        let output_size = output_slice.size;
                        let invalid_input = |slice: &BufferSlice| {
                            !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                || !slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                || (output_size > 0
                                    && (slice.size == 0 || !output_size.is_multiple_of(slice.size)))
                                || (output_size == 0 && slice.size != 0)
                        };
                        if input_slices.len() != 2
                            || !params.is_empty()
                            || input_slices.iter().any(invalid_input)
                            || !output_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                            || !output_size.is_multiple_of(std::mem::size_of::<f32>())
                        {
                            return Err(BackendError::Dispatch(format!(
                                "{kernel_name}: binary f32 storage or parameter contract is invalid"
                            )));
                        }
                    }

                    let is_scalar_f32 = matches!(
                        kernel_name.as_str(),
                        "gt_scalar_f32"
                            | "lt_scalar_f32"
                            | "eq_scalar_f32"
                            | "add_scalar_f32"
                            | "mul_scalar_f32"
                            | "div_scalar_f32"
                    );
                    if is_scalar_f32 {
                        let scalar_bytes = std::mem::size_of::<f32>();
                        if input_slices.len() != 2
                            || !params.is_empty()
                            || input_slices[0].size != output_slice.size
                            || input_slices[1].size < scalar_bytes
                            || input_slices.iter().any(|slice| {
                                !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                    || !slice.size.is_multiple_of(scalar_bytes)
                            })
                            || !output_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                            || !output_slice.size.is_multiple_of(scalar_bytes)
                        {
                            return Err(BackendError::Dispatch(format!(
                                "{kernel_name}: scalar f32 storage or parameter contract is invalid"
                            )));
                        }
                    }

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
                            if params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "leaky_relu_f32: expected one slope parameter".into(),
                                ));
                            }
                            let slope = f32::from_bits(params[0] as u32);
                            if !slope.is_finite() {
                                return Err(BackendError::Dispatch(
                                    "leaky_relu_f32: slope must be finite".into(),
                                ));
                            }
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
                            if params.len() != 2 {
                                return Err(BackendError::Dispatch(
                                    "clamp_f32: expected minimum and maximum parameters".into(),
                                ));
                            }
                            let min_val = f32::from_bits(params[0] as u32);
                            let max_val = f32::from_bits(params[1] as u32);
                            if !min_val.is_finite() || !max_val.is_finite() || min_val > max_val {
                                return Err(BackendError::Dispatch(
                                    "clamp_f32: bounds must be finite and ordered".into(),
                                ));
                            }
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
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "reduce_f32: expected one input and encoded geometry".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank.checked_add(4).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "reduce_f32: parameter count overflows".into(),
                                )
                            })?;
                            if rank == 0 || params.len() != expected_params {
                                return Err(BackendError::Dispatch(format!(
                                    "reduce_f32: malformed shape metadata (rank {rank}, {} parameters)",
                                    params.len()
                                )));
                            }
                            let mut shape = params[1..1 + rank].to_vec();
                            if let Some(dimensions) = param_dims {
                                if dimensions.len() != rank {
                                    return Err(BackendError::Dispatch(
                                        "reduce_f32: symbolic shape rank mismatch".into(),
                                    ));
                                }
                                for (size, dimension) in shape.iter_mut().zip(dimensions) {
                                    *size =
                                        dimension.evaluate_with_env(shape_env).map_err(|error| {
                                            BackendError::Dispatch(format!("reduce_f32: {error}"))
                                        })? as usize;
                                }
                            }
                            let axis = params[1 + rank];
                            let is_mean = params[2 + rank];
                            let is_max = params[3 + rank];
                            if axis >= rank || is_mean > 1 || is_max > 1 || is_mean + is_max > 1 {
                                return Err(BackendError::Dispatch(
                                    "reduce_f32: invalid axis or reduction mode".into(),
                                ));
                            }
                            let axis_size = shape[axis];
                            if axis_size == 0 {
                                return Err(BackendError::Dispatch(
                                    "reduce_f32: cannot reduce an empty axis".into(),
                                ));
                            }
                            let input_elements =
                                shape.iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "reduce_f32: input shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let outer =
                                shape[..axis].iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "reduce_f32: outer geometry overflows".into(),
                                        )
                                    })
                                })?;
                            let inner =
                                shape[axis + 1..]
                                    .iter()
                                    .try_fold(1usize, |count, dimension| {
                                        count.checked_mul(*dimension).ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "reduce_f32: inner geometry overflows".into(),
                                            )
                                        })
                                    })?;
                            let output_elements = outer.checked_mul(inner).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "reduce_f32: output geometry overflows".into(),
                                )
                            })?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_bytes =
                                input_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "reduce_f32: input size overflows".into(),
                                    )
                                })?;
                            let output_bytes =
                                output_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "reduce_f32: output size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != input_bytes
                                || output_slice.size != output_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "reduce_f32: geometry and f32 storage disagree".into(),
                                ));
                            }
                            try_with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                "reduce_f32 input",
                                |input, output| {
                                    for outer_index in 0..outer {
                                        for inner_index in 0..inner {
                                            let mut value =
                                                if is_max == 1 { f32::NEG_INFINITY } else { 0.0 };
                                            for axis_index in 0..axis_size {
                                                let input_index =
                                                    (outer_index * axis_size + axis_index) * inner
                                                        + inner_index;
                                                if is_max == 1 {
                                                    value = value.max(input[input_index]);
                                                } else {
                                                    value += input[input_index];
                                                }
                                            }
                                            if is_mean == 1 {
                                                value /= axis_size as f32;
                                            }
                                            output[outer_index * inner + inner_index] = value;
                                        }
                                    }
                                },
                            )?;
                        }
                        "transpose_f32" => {
                            if input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "transpose_f32: expected exactly one input".into(),
                                ));
                            }
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let transpose_params =
                                resolve_params(params, param_dims, shape_env, 2)?;
                            let &[m, n] = &transpose_params[..] else {
                                return Err(BackendError::Dispatch(
                                    "transpose_f32: expected params [M,N]".into(),
                                ));
                            };
                            let expected_bytes = m
                                .checked_mul(n)
                                .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "transpose_f32: matrix storage size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            if input_slice.size != expected_bytes
                                || output_slice.size != expected_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "transpose_f32: geometry and f32 storage disagree".into(),
                                ));
                            }
                            try_with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                "transpose_f32 input",
                                |input, out_f32| {
                                    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                                    if microkernels::simd_avx2_available() && m >= 8 && n >= 8 {
                                        unsafe {
                                            microkernels::transpose_f32_avx2(input, out_f32, m, n);
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
                            )?;
                        }
                        "transpose_perm_f32" => {
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "transpose_perm_f32: expected one input and encoded permutation"
                                        .into(),
                                ));
                            }
                            let rank = params[0];
                            if rank == 0 {
                                return Err(BackendError::Dispatch(
                                    "transpose_perm_f32: rank must be positive".into(),
                                ));
                            }
                            let expected_params = rank
                                .checked_mul(2)
                                .and_then(|count| count.checked_add(1))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "transpose_perm_f32: parameter count overflows".into(),
                                    )
                                })?;
                            let nd_params =
                                resolve_params(params, param_dims, shape_env, expected_params)?;
                            let dims = &nd_params[1..1 + rank];
                            let perm = &nd_params[1 + rank..];
                            let mut seen =
                                try_filled_vec(rank, false, "transpose_perm_f32: rank metadata")?;
                            for axis in perm {
                                if *axis >= rank || seen[*axis] {
                                    return Err(BackendError::Dispatch(
                                        "transpose_perm_f32: permutation must contain each axis once"
                                            .into(),
                                    ));
                                }
                                seen[*axis] = true;
                            }
                            let total = dims.iter().try_fold(1usize, |count, dimension| {
                                count.checked_mul(*dimension).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "transpose_perm_f32: shape product overflows".into(),
                                    )
                                })
                            })?;
                            let expected_bytes = total
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "transpose_perm_f32: storage size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != expected_bytes
                                || output_slice.size != expected_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "transpose_perm_f32: shape and f32 storage disagree".into(),
                                ));
                            }
                            let mut input_strides =
                                try_filled_vec(rank, 1usize, "transpose_perm_f32: input strides")?;
                            for axis in (0..rank - 1).rev() {
                                input_strides[axis] = input_strides[axis + 1]
                                    .checked_mul(dims[axis + 1])
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "transpose_perm_f32: input stride overflows".into(),
                                        )
                                    })?;
                            }
                            let mut output_strides =
                                try_filled_vec(rank, 1usize, "transpose_perm_f32: output strides")?;
                            for axis in (0..rank - 1).rev() {
                                output_strides[axis] = output_strides[axis + 1]
                                    .checked_mul(dims[perm[axis + 1]])
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "transpose_perm_f32: output stride overflows".into(),
                                        )
                                    })?;
                            }
                            try_with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                "transpose_perm_f32 input",
                                |input, output| {
                                    for (output_index, value) in output.iter_mut().enumerate() {
                                        let mut remaining = output_index;
                                        let mut input_index = 0usize;
                                        for output_axis in 0..rank {
                                            let coordinate =
                                                remaining / output_strides[output_axis];
                                            remaining %= output_strides[output_axis];
                                            input_index +=
                                                coordinate * input_strides[perm[output_axis]];
                                        }
                                        *value = input[input_index];
                                    }
                                },
                            )?;
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
                            if input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "softmax: expected exactly one input".into(),
                                ));
                            }
                            let softmax_params = resolve_params(params, param_dims, shape_env, 2)?;
                            let [axis_dim_size, stride] = softmax_params.as_ref() else {
                                unreachable!();
                            };
                            let input_slice = input_slices[0];
                            let output_size = out_end - out_start;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_elements = input_slice.size / scalar_bytes;
                            let row_span = axis_dim_size.checked_mul(*stride).ok_or_else(|| {
                                BackendError::Dispatch("softmax: row span overflows".into())
                            })?;
                            if *axis_dim_size == 0
                                || *stride == 0
                                || input_slice.size == 0
                                || input_slice.size != output_size
                                || !input_elements.is_multiple_of(row_span)
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(scalar_bytes)
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "softmax: invalid geometry or f32 storage contract".into(),
                                ));
                            }
                            let num_rows = input_elements / axis_dim_size;
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                try_with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    "softmax input",
                                    |input, out_f32| {
                                        softmax_f32(
                                            input,
                                            out_f32,
                                            *axis_dim_size,
                                            *stride,
                                            num_rows,
                                        );
                                    },
                                )?;
                            }
                        }
                        "biasadd" => {
                            if input_slices.len() != 2 || params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "biasadd: expected data/bias inputs and channel stride".into(),
                                ));
                            }
                            let data_slice = input_slices[0];
                            let bias_slice = input_slices[1];
                            let channel_stride = params[0];
                            let output_size = out_end - out_start;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let data_elements = data_slice.size / scalar_bytes;
                            let bias_elements = bias_slice.size / scalar_bytes;
                            let channel_block =
                                bias_elements.checked_mul(channel_stride).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "biasadd: channel geometry overflows".into(),
                                    )
                                })?;
                            if channel_stride == 0
                                || data_slice.size != output_size
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !output_size.is_multiple_of(scalar_bytes)
                                || (data_elements > 0
                                    && (bias_elements == 0
                                        || !data_elements.is_multiple_of(channel_block)))
                            {
                                return Err(BackendError::Dispatch(
                                    "biasadd: invalid geometry or f32 storage contract".into(),
                                ));
                            }
                            let output_overlaps = input_slices.iter().any(|slice| {
                                let end = slice.offset + slice.size;
                                slice.offset < out_end && out_start < end
                            });
                            if output_overlaps {
                                return Err(BackendError::Dispatch(
                                    "biasadd: input and output slices overlap".into(),
                                ));
                            }
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
                            if !eps.is_finite() || eps <= 0.0 || is_batch_norm > 1 {
                                return Err(BackendError::Dispatch(
                                    "norm_f32: epsilon must be finite and positive and mode must be 0 or 1"
                                        .into(),
                                ));
                            }
                            if is_batch_norm == 1 {
                                if input_slices.len() != 5 {
                                    return Err(BackendError::Dispatch(
                                        "norm_f32 batch mode expects data, weight, bias, mean, and variance"
                                            .into(),
                                    ));
                                }
                                let data_slice = input_slices[0];
                                let channel_slices = &input_slices[1..];
                                let output_size = out_end - out_start;
                                let scalar_bytes = std::mem::size_of::<f32>();
                                let channel_bytes = channel_slices[0].size;
                                let channel_count = channel_bytes / scalar_bytes;
                                if channel_count == 0
                                    || data_slice.size != output_size
                                    || channel_slices
                                        .iter()
                                        .any(|slice| slice.size != channel_bytes)
                                    || input_slices.iter().any(|slice| {
                                        !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                            || !slice.size.is_multiple_of(scalar_bytes)
                                    })
                                    || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                    || !output_size.is_multiple_of(scalar_bytes)
                                    || !(data_slice.size / scalar_bytes)
                                        .is_multiple_of(channel_count)
                                {
                                    return Err(BackendError::Dispatch(
                                        "norm_f32 batch mode has invalid channel or f32 storage"
                                            .into(),
                                    ));
                                }
                                let output_overlaps = input_slices.iter().any(|slice| {
                                    let end = slice.offset + slice.size;
                                    slice.offset < out_end && out_start < end
                                });
                                {
                                    let data = arena.data_mut();
                                    let metadata_is_invalid = channel_slices.iter().any(|slice| {
                                        bytemuck::cast_slice::<_, f32>(
                                            &data[slice.offset..slice.offset + slice.size],
                                        )
                                        .iter()
                                        .any(|value| !value.is_finite())
                                    });
                                    let variance = bytemuck::cast_slice::<_, f32>(
                                        &data[channel_slices[3].offset
                                            ..channel_slices[3].offset + channel_slices[3].size],
                                    );
                                    if metadata_is_invalid
                                        || variance.iter().any(|value| *value < 0.0)
                                    {
                                        return Err(BackendError::Dispatch(
                                            "norm_f32 batch metadata must be finite with nonnegative variance"
                                                .into(),
                                        ));
                                    }
                                }
                                if output_overlaps {
                                    let (data, weight, bias, running_mean, running_var) = {
                                        let arena_data = arena.data_mut();
                                        let copy = |slice: BufferSlice, label: &str| {
                                            try_copy_slice(
                                                bytemuck::cast_slice::<_, f32>(
                                                    &arena_data
                                                        [slice.offset..slice.offset + slice.size],
                                                ),
                                                label,
                                            )
                                        };
                                        (
                                            copy(input_slices[0], "batch norm data")?,
                                            copy(input_slices[1], "batch norm weight")?,
                                            copy(input_slices[2], "batch norm bias")?,
                                            copy(input_slices[3], "batch norm running mean")?,
                                            copy(input_slices[4], "batch norm running variance")?,
                                        )
                                    };
                                    let output = unsafe {
                                        arena.view_f32_mut(out_start, out_end - out_start)
                                    };
                                    dispatch_helpers::batch_norm_inference_f32(
                                        &data,
                                        &weight,
                                        &bias,
                                        &running_mean,
                                        &running_var,
                                        output,
                                        eps,
                                    );
                                // Batch norm (evaluation mode): use running_mean and running_var
                                } else if let [data_slice, weight_slice, bias_slice, mean_slice, var_slice] =
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
                                    dispatch_helpers::batch_norm_inference_f32(
                                        data,
                                        weight,
                                        bias,
                                        running_mean,
                                        running_var,
                                        out_f32,
                                        eps,
                                    );
                                }
                            } else {
                                if input_slices.len() != 3 {
                                    return Err(BackendError::Dispatch(
                                        "norm_f32 layer mode expects data, weight, and bias".into(),
                                    ));
                                }
                                let data_slice = input_slices[0];
                                let weight_slice = input_slices[1];
                                let bias_slice = input_slices[2];
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let scalar_bytes = std::mem::size_of::<f32>();
                                let row_size = weight_slice.size / scalar_bytes;
                                let data_elements = data_slice.size / scalar_bytes;
                                if row_size == 0
                                    || data_slice.size != output_slice.size
                                    || bias_slice.size != weight_slice.size
                                    || !data_elements.is_multiple_of(row_size)
                                    || input_slices.iter().any(|slice| {
                                        !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                            || !slice.size.is_multiple_of(scalar_bytes)
                                    })
                                    || !output_slice
                                        .offset
                                        .is_multiple_of(std::mem::align_of::<f32>())
                                    || !output_slice.size.is_multiple_of(scalar_bytes)
                                {
                                    return Err(BackendError::Dispatch(
                                        "norm_f32 layer mode has invalid row or f32 storage".into(),
                                    ));
                                }
                                {
                                    let data = arena.data_mut();
                                    if [weight_slice, bias_slice].iter().any(|slice| {
                                        bytemuck::cast_slice::<_, f32>(
                                            &data[slice.offset..slice.offset + slice.size],
                                        )
                                        .iter()
                                        .any(|value| !value.is_finite())
                                    }) {
                                        return Err(BackendError::Dispatch(
                                            "norm_f32 layer affine metadata must be finite".into(),
                                        ));
                                    }
                                }
                                let output_overlaps = input_slices.iter().any(|slice| {
                                    let end = slice.offset + slice.size;
                                    slice.offset < out_end && out_start < end
                                });
                                if output_overlaps {
                                    let (data, weight, bias) = {
                                        let arena_data = arena.data_mut();
                                        let copy = |slice: BufferSlice, label: &str| {
                                            try_copy_slice(
                                                bytemuck::cast_slice::<_, f32>(
                                                    &arena_data
                                                        [slice.offset..slice.offset + slice.size],
                                                ),
                                                label,
                                            )
                                        };
                                        (
                                            copy(data_slice, "layer norm data")?,
                                            copy(weight_slice, "layer norm weight")?,
                                            copy(bias_slice, "layer norm bias")?,
                                        )
                                    };
                                    let output = unsafe {
                                        arena.view_f32_mut(out_start, out_end - out_start)
                                    };
                                    norm_layernorm_f32(&data, output, row_size, eps);
                                    for (index, value) in output.iter_mut().enumerate() {
                                        let column = index % row_size;
                                        *value = *value * weight[column] + bias[column];
                                    }
                                } else {
                                    arena::with_nary_f32_slices(
                                        arena,
                                        input_slices,
                                        output_slice,
                                        |inputs, out_f32| {
                                            norm_layernorm_f32(inputs[0], out_f32, row_size, eps);
                                            for (index, output) in out_f32.iter_mut().enumerate() {
                                                let column = index % row_size;
                                                *output =
                                                    *output * inputs[1][column] + inputs[2][column];
                                            }
                                        },
                                    );
                                }
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
                            // input_slices: [activation (f32), weight (packed)] optional: [bias (f32)]
                            // weight_meta carries bit_width, shape=[OC, IC_per_group*KH*KW], scales[], zero_points[]
                            if !(2..=3).contains(&input_slices.len()) {
                                return Err(BackendError::Dispatch(format!(
                                    "conv2d_i4/u8: expected 2 or 3 inputs, received {}",
                                    input_slices.len()
                                )));
                            }
                            if params.len() < 9 {
                                return Err(BackendError::Dispatch("conv2d_i4/u8: expected at least 9 params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into()));
                            }
                            if params[0] == 0
                                || params[2] == 0
                                || params[3] == 0
                                || params[4..=8].contains(&0)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv2d_i4/u8: stride, dilation, groups, channels, spatial dimensions, and kernel dimensions must be positive".into(),
                                ));
                            }
                            let activation = &input_slices[0];
                            let bias = input_slices.get(2);
                            if !activation
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                                || !activation.size.is_multiple_of(std::mem::size_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                || bias.is_some_and(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                })
                            {
                                return Err(BackendError::Dispatch(
                                    "conv2d_i4/u8: activation, bias, and output storage must be f32-aligned and scalar-exact".into(),
                                ));
                            }
                            let output_end = output_slice.offset + output_slice.size;
                            if input_slices.iter().any(|slice| {
                                arena::ranges_overlap(
                                    output_slice.offset,
                                    output_end,
                                    slice.offset,
                                    slice.offset + slice.size,
                                )
                            }) {
                                return Err(BackendError::Dispatch(
                                    "conv2d_i4/u8: output storage overlaps an input".into(),
                                ));
                            }
                            let image_elements = params[4]
                                .checked_mul(params[5])
                                .and_then(|value| value.checked_mul(params[6]))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv2d_i4/u8: activation geometry overflows".into(),
                                    )
                                })?;
                            let activation_elements = activation.size / std::mem::size_of::<f32>();
                            if image_elements == 0
                                || !activation_elements.is_multiple_of(image_elements)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv2d_i4/u8: activation storage does not contain complete images".into(),
                                ));
                            }
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
                            if input_slices.is_empty() || params.len() != 3 {
                                return Err(BackendError::Dispatch(
                                    "concat: expected inputs and [axis, inner_stride, outer_count]"
                                        .into(),
                                ));
                            }
                            let inner_stride = params[1];
                            let outer_count = params[2];
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if inner_stride == 0
                                || input_slices.iter().any(|slice| {
                                    !slice.size.is_multiple_of(scalar_bytes)
                                        || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                })
                                || !output_slice.size.is_multiple_of(scalar_bytes)
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "concat: invalid stride or incomplete f32 storage".into(),
                                ));
                            }
                            if outer_count == 0 {
                                if output_slice.size != 0
                                    || input_slices.iter().any(|slice| slice.size != 0)
                                {
                                    return Err(BackendError::Dispatch(
                                        "concat: zero outer count requires zero-sized tensors"
                                            .into(),
                                    ));
                                }
                                continue;
                            }
                            let mut block_sizes = Vec::new();
                            block_sizes
                                .try_reserve_exact(input_slices.len())
                                .map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "concat: block metadata allocation failed: {error}"
                                    ))
                                })?;
                            let mut expected_output_elements = 0usize;
                            for slice in input_slices {
                                let elements = slice.size / scalar_bytes;
                                if !elements.is_multiple_of(outer_count) {
                                    return Err(BackendError::Dispatch(
                                        "concat: input does not contain complete outer blocks"
                                            .into(),
                                    ));
                                }
                                let block_size = elements / outer_count;
                                if !block_size.is_multiple_of(inner_stride) {
                                    return Err(BackendError::Dispatch(
                                        "concat: input block disagrees with inner stride".into(),
                                    ));
                                }
                                expected_output_elements = expected_output_elements
                                    .checked_add(elements)
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "concat: output element count overflows".into(),
                                        )
                                    })?;
                                block_sizes.push(block_size);
                            }
                            let expected_output_bytes = expected_output_elements
                                .checked_mul(scalar_bytes)
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "concat: output storage size overflows".into(),
                                    )
                                })?;
                            if output_slice.size != expected_output_bytes {
                                return Err(BackendError::Dispatch(
                                    "concat: output storage does not equal all input elements"
                                        .into(),
                                ));
                            }
                            arena::with_nary_f32_slices(
                                arena,
                                input_slices,
                                output_slice,
                                |inputs, output| {
                                    let mut output_offset = 0usize;
                                    for outer_position in 0..outer_count {
                                        for (input, block_size) in
                                            inputs.iter().zip(block_sizes.iter().copied())
                                        {
                                            let source = outer_position * block_size;
                                            output[output_offset..output_offset + block_size]
                                                .copy_from_slice(
                                                    &input[source..source + block_size],
                                                );
                                            output_offset += block_size;
                                        }
                                    }
                                },
                            );
                        }

                        "pool_f32" => {
                            if input_slices.len() != 1 || params.len() != 8 {
                                return Err(BackendError::Dispatch(
                                    "pool_f32: expected one input and eight geometry parameters"
                                        .into(),
                                ));
                            }
                            let [kernel, stride, padding, is_max, n, c, h, w] = [
                                params[0], params[1], params[2], params[3], params[4], params[5],
                                params[6], params[7],
                            ];
                            if kernel == 0
                                || stride == 0
                                || n == 0
                                || c == 0
                                || h == 0
                                || w == 0
                                || is_max > 1
                            {
                                return Err(BackendError::Dispatch(
                                    "pool_f32: invalid mode, stride, or dimensions".into(),
                                ));
                            }
                            let padded_height = padding
                                .checked_mul(2)
                                .and_then(|value| h.checked_add(value))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: padded height overflows".into(),
                                    )
                                })?;
                            let padded_width = padding
                                .checked_mul(2)
                                .and_then(|value| w.checked_add(value))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: padded width overflows".into(),
                                    )
                                })?;
                            if kernel > padded_height || kernel > padded_width {
                                return Err(BackendError::Dispatch(
                                    "pool_f32: kernel exceeds padded input".into(),
                                ));
                            }
                            let h_out = (padded_height - kernel) / stride + 1;
                            let w_out = (padded_width - kernel) / stride + 1;
                            let input_elements = n
                                .checked_mul(c)
                                .and_then(|value| value.checked_mul(h))
                                .and_then(|value| value.checked_mul(w))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: input geometry overflows".into(),
                                    )
                                })?;
                            let output_elements = n
                                .checked_mul(c)
                                .and_then(|value| value.checked_mul(h_out))
                                .and_then(|value| value.checked_mul(w_out))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: output geometry overflows".into(),
                                    )
                                })?;
                            let input_bytes = input_elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: input storage size overflows".into(),
                                    )
                                })?;
                            let output_bytes = output_elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pool_f32: output storage size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            if input_slice.size != input_bytes
                                || out_end - out_start != output_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "pool_f32: geometry and f32 storage disagree".into(),
                                ));
                            }
                            if let Some(indices) = secondary_output_slice {
                                let indices_bytes = output_elements
                                    .checked_mul(std::mem::size_of::<i64>())
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "pool_f32: index storage size overflows".into(),
                                        )
                                    })?;
                                if is_max == 0
                                    || indices.size != indices_bytes
                                    || !indices.offset.is_multiple_of(std::mem::align_of::<i64>())
                                    || arena::ranges_overlap(
                                        indices.offset,
                                        indices.offset + indices.size,
                                        input_slice.offset,
                                        input_slice.offset + input_slice.size,
                                    )
                                    || arena::ranges_overlap(
                                        indices.offset,
                                        indices.offset + indices.size,
                                        out_start,
                                        out_end,
                                    )
                                {
                                    return Err(BackendError::Dispatch(
                                        "pool_f32: invalid or overlapping indices output".into(),
                                    ));
                                }
                            }
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
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "pad_f32: expected one input and encoded shape/padding".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank
                                .checked_mul(3)
                                .and_then(|count| count.checked_add(1))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pad_f32: parameter count overflows".into(),
                                    )
                                })?;
                            if params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "pad_f32: malformed shape/padding metadata".into(),
                                ));
                            }
                            let input_shape = &params[1..1 + rank];
                            let pads = &params[1 + rank..];
                            let input_elements =
                                input_shape.iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "pad_f32: input shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let output_elements = input_shape.iter().enumerate().try_fold(
                                1usize,
                                |count, (dimension, size)| {
                                    let output_size = size
                                        .checked_add(pads[dimension * 2])
                                        .and_then(|value| {
                                            value.checked_add(pads[dimension * 2 + 1])
                                        })
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "pad_f32: padded dimension overflows".into(),
                                            )
                                        })?;
                                    count.checked_mul(output_size).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "pad_f32: output shape product overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let input_bytes = input_elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pad_f32: input storage size overflows".into(),
                                    )
                                })?;
                            let output_bytes = output_elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "pad_f32: output storage size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != input_bytes
                                || output_slice.size != output_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "pad_f32: shape and f32 storage disagree".into(),
                                ));
                            }
                            try_with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                "pad_f32 input",
                                |input, output| {
                                    for (output_index, value) in output.iter_mut().enumerate() {
                                        let mut remaining = output_index;
                                        let mut source_index = 0usize;
                                        let mut source_stride = 1usize;
                                        let mut inside = true;
                                        for dimension in (0..rank).rev() {
                                            let output_dimension = input_shape[dimension]
                                                + pads[dimension * 2]
                                                + pads[dimension * 2 + 1];
                                            let coordinate = remaining % output_dimension;
                                            remaining /= output_dimension;
                                            let low_pad = pads[dimension * 2];
                                            if coordinate < low_pad
                                                || coordinate >= low_pad + input_shape[dimension]
                                            {
                                                inside = false;
                                            } else {
                                                source_index +=
                                                    (coordinate - low_pad) * source_stride;
                                            }
                                            source_stride *= input_shape[dimension];
                                        }
                                        *value = if inside { input[source_index] } else { 0.0 };
                                    }
                                },
                            )?;
                        }
                        "gather" => {
                            if input_slices.len() != 2 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "gather: expected data, indices, and encoded geometry".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank.checked_add(3).ok_or_else(|| {
                                BackendError::Dispatch("gather: parameter count overflows".into())
                            })?;
                            if rank == 0 || params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "gather: malformed data shape metadata".into(),
                                ));
                            }
                            let data_shape = &params[1..1 + rank];
                            let indices_numel = params[1 + rank];
                            let axis = params[2 + rank];
                            if axis >= rank {
                                return Err(BackendError::Dispatch(
                                    "gather: axis is outside data rank".into(),
                                ));
                            }
                            let data_elements =
                                data_shape.iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "gather: data shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let outer = data_shape[..axis].iter().try_fold(
                                1usize,
                                |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "gather: outer geometry overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let inner = data_shape[axis + 1..].iter().try_fold(
                                1usize,
                                |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "gather: inner geometry overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let axis_size = data_shape[axis];
                            let output_elements = outer
                                .checked_mul(indices_numel)
                                .and_then(|count| count.checked_mul(inner))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "gather: output geometry overflows".into(),
                                    )
                                })?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let expected_data =
                                data_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("gather: data size overflows".into())
                                })?;
                            let expected_indices =
                                indices_numel.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("gather: index size overflows".into())
                                })?;
                            let expected_output =
                                output_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("gather: output size overflows".into())
                                })?;
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slices[0].size != expected_data
                                || input_slices[1].size != expected_indices
                                || output_slice.size != expected_output
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                })
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "gather: geometry and f32 storage disagree".into(),
                                ));
                            }
                            let indices = unsafe {
                                arena.view_f32(input_slices[1].offset, input_slices[1].size)
                            };
                            if indices.iter().any(|index| {
                                !index.is_finite()
                                    || *index < 0.0
                                    || index.fract() != 0.0
                                    || *index >= axis_size as f32
                            }) {
                                return Err(BackendError::Dispatch(
                                    "gather: indices must be integral and within the axis".into(),
                                ));
                            }
                            arena::with_nary_f32_slices(
                                arena,
                                input_slices,
                                output_slice,
                                |inputs, output| {
                                    let data = inputs[0];
                                    let indices = inputs[1];
                                    for outer_index in 0..outer {
                                        for (index_position, index) in indices.iter().enumerate() {
                                            let source =
                                                (outer_index * axis_size + *index as usize) * inner;
                                            let destination = (outer_index * indices_numel
                                                + index_position)
                                                * inner;
                                            output[destination..destination + inner]
                                                .copy_from_slice(&data[source..source + inner]);
                                        }
                                    }
                                },
                            );
                        }
                        "slice_f32" => {
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "slice_f32: expected one input and encoded geometry".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank.checked_add(4).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "slice_f32: parameter count overflows".into(),
                                )
                            })?;
                            if rank == 0 || params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "slice_f32: malformed input shape metadata".into(),
                                ));
                            }
                            let input_shape = &params[1..1 + rank];
                            let dim = params[1 + rank];
                            let start = params[2 + rank];
                            let end = params[3 + rank];
                            if dim >= rank || start > end || end > input_shape[dim] {
                                return Err(BackendError::Dispatch(
                                    "slice_f32: invalid dimension or range".into(),
                                ));
                            }
                            let input_elements =
                                input_shape.iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "slice_f32: input shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let inner = input_shape[dim + 1..].iter().try_fold(
                                1usize,
                                |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "slice_f32: inner geometry overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let outer = input_shape[..dim].iter().try_fold(
                                1usize,
                                |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "slice_f32: outer geometry overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let range = end - start;
                            let output_elements = outer
                                .checked_mul(range)
                                .and_then(|count| count.checked_mul(inner))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "slice_f32: output geometry overflows".into(),
                                    )
                                })?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_bytes =
                                input_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("slice_f32: input size overflows".into())
                                })?;
                            let output_bytes =
                                output_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "slice_f32: output size overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != input_bytes
                                || output_slice.size != output_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "slice_f32: shape and f32 storage disagree".into(),
                                ));
                            }
                            let axis_size = input_shape[dim];
                            let copy_elements = range * inner;
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, output| {
                                    for outer_index in 0..outer {
                                        let source = (outer_index * axis_size + start) * inner;
                                        let destination = outer_index * copy_elements;
                                        output[destination..destination + copy_elements]
                                            .copy_from_slice(
                                                &input[source..source + copy_elements],
                                            );
                                    }
                                },
                            );
                        }
                        "scatter_nd" => {
                            if input_slices.len() != 3 || params.len() < 2 {
                                return Err(BackendError::Dispatch(
                                    "scatter_nd: expected data, indices, updates, and geometry"
                                        .into(),
                                ));
                            }
                            let index_depth = params[0];
                            let data_dims = &params[1..];
                            if index_depth == 0 || index_depth > data_dims.len() {
                                return Err(BackendError::Dispatch(
                                    "scatter_nd: invalid index depth".into(),
                                ));
                            }
                            let data_elements =
                                data_dims.iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "scatter_nd: data shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let inner_size = data_dims[index_depth..].iter().try_fold(
                                1usize,
                                |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "scatter_nd: update slice size overflows".into(),
                                        )
                                    })
                                },
                            )?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let data_bytes =
                                data_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("scatter_nd: data size overflows".into())
                                })?;
                            let indices_slice = input_slices[1];
                            if !indices_slice.size.is_multiple_of(scalar_bytes)
                                || !indices_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "scatter_nd: indices are not complete aligned f32 scalars"
                                        .into(),
                                ));
                            }
                            let index_elements = indices_slice.size / scalar_bytes;
                            if !index_elements.is_multiple_of(index_depth) {
                                return Err(BackendError::Dispatch(
                                    "scatter_nd: index storage is not a whole number of tuples"
                                        .into(),
                                ));
                            }
                            let tuple_count = index_elements / index_depth;
                            let update_elements =
                                tuple_count.checked_mul(inner_size).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "scatter_nd: update element count overflows".into(),
                                    )
                                })?;
                            let update_bytes =
                                update_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "scatter_nd: update size overflows".into(),
                                    )
                                })?;
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slices[0].size != data_bytes
                                || input_slices[2].size != update_bytes
                                || output_slice.size != data_bytes
                                || [input_slices[0], input_slices[2], output_slice].iter().any(
                                    |slice| {
                                        !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                    },
                                )
                            {
                                return Err(BackendError::Dispatch(
                                    "scatter_nd: geometry and f32 storage disagree".into(),
                                ));
                            }
                            let indices =
                                unsafe { arena.view_f32(indices_slice.offset, indices_slice.size) };
                            for tuple in indices.chunks_exact(index_depth) {
                                for (dimension, index) in tuple.iter().enumerate() {
                                    if !index.is_finite()
                                        || *index < 0.0
                                        || index.fract() != 0.0
                                        || *index >= data_dims[dimension] as f32
                                    {
                                        return Err(BackendError::Dispatch(
                                            "scatter_nd: indices must be integral and in bounds"
                                                .into(),
                                        ));
                                    }
                                }
                            }
                            arena::with_nary_f32_slices(
                                arena,
                                input_slices,
                                output_slice,
                                |inputs, output| {
                                    output.copy_from_slice(inputs[0]);
                                    let indices = inputs[1];
                                    let updates = inputs[2];
                                    for (tuple_index, tuple) in
                                        indices.chunks_exact(index_depth).enumerate()
                                    {
                                        let mut linear_offset = 0usize;
                                        let mut stride = data_elements;
                                        for (dimension, index) in tuple.iter().enumerate() {
                                            stride /= data_dims[dimension];
                                            linear_offset += *index as usize * stride;
                                        }
                                        let update_start = tuple_index * inner_size;
                                        output[linear_offset..linear_offset + inner_size]
                                            .copy_from_slice(
                                                &updates[update_start..update_start + inner_size],
                                            );
                                    }
                                },
                            );
                        }
                        "conv1d" => {
                            if input_slices.len() != 2 || params.len() != 5 {
                                return Err(BackendError::Dispatch(
                                    "conv1d: expected input, weight, and five geometry parameters"
                                        .into(),
                                ));
                            }
                            let [stride, padding, c, w, kw] =
                                [params[0], params[1], params[2], params[3], params[4]];
                            if stride == 0 || c == 0 || w == 0 || kw == 0 {
                                return Err(BackendError::Dispatch(
                                    "conv1d: stride and tensor dimensions must be positive".into(),
                                ));
                            }
                            let input_plane = c.checked_mul(w).ok_or_else(|| {
                                BackendError::Dispatch("conv1d: input geometry overflows".into())
                            })?;
                            let kernel_plane = c.checked_mul(kw).ok_or_else(|| {
                                BackendError::Dispatch("conv1d: kernel geometry overflows".into())
                            })?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if input_slices.iter().any(|slice| {
                                !slice.size.is_multiple_of(scalar_bytes)
                                    || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                            }) || !(out_end - out_start).is_multiple_of(scalar_bytes)
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "conv1d: storage is not complete aligned f32 data".into(),
                                ));
                            }
                            let input_elements = input_slices[0].size / scalar_bytes;
                            let weight_elements = input_slices[1].size / scalar_bytes;
                            if !input_elements.is_multiple_of(input_plane)
                                || !weight_elements.is_multiple_of(kernel_plane)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv1d: storage does not contain complete tensors".into(),
                                ));
                            }
                            let n = input_elements / input_plane;
                            let f = weight_elements / kernel_plane;
                            let padded_width = padding
                                .checked_mul(2)
                                .and_then(|value| w.checked_add(value))
                                .ok_or_else(|| {
                                    BackendError::Dispatch("conv1d: padded width overflows".into())
                                })?;
                            if kw > padded_width {
                                return Err(BackendError::Dispatch(
                                    "conv1d: kernel exceeds padded input width".into(),
                                ));
                            }
                            let w_out = (padded_width - kw) / stride + 1;
                            let output_elements = n
                                .checked_mul(f)
                                .and_then(|count| count.checked_mul(w_out))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv1d: output geometry overflows".into(),
                                    )
                                })?;
                            let output_bytes =
                                output_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv1d: output storage size overflows".into(),
                                    )
                                })?;
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if output_slice.size != output_bytes {
                                return Err(BackendError::Dispatch(
                                    "conv1d: output storage does not match geometry".into(),
                                ));
                            }
                            arena::with_nary_f32_slices(
                                arena,
                                input_slices,
                                output_slice,
                                |inputs, output| {
                                    let input = inputs[0];
                                    let weight = inputs[1];
                                    for nn in 0..n {
                                        for ff in 0..f {
                                            for ww in 0..w_out {
                                                let mut sum = 0.0f32;
                                                for cc in 0..c {
                                                    for kkw in 0..kw {
                                                        let padded_index = ww * stride + kkw;
                                                        if padded_index >= padding {
                                                            let input_index =
                                                                padded_index - padding;
                                                            if input_index < w {
                                                                sum += input[nn * input_plane
                                                                    + cc * w
                                                                    + input_index]
                                                                    * weight[ff * kernel_plane
                                                                        + cc * kw
                                                                        + kkw];
                                                            }
                                                        }
                                                    }
                                                }
                                                output[nn * (f * w_out) + ff * w_out + ww] = sum;
                                            }
                                        }
                                    }
                                },
                            );
                        }
                        "conv3d" => {
                            if input_slices.len() != 2 || params.len() != 10 {
                                return Err(BackendError::Dispatch(
                                    "conv3d: expected input, weight, and ten geometry parameters"
                                        .into(),
                                ));
                            }
                            let [stride, padding, dilation, c, d, h, w, kd, kh, kw] = [
                                params[0], params[1], params[2], params[3], params[4], params[5],
                                params[6], params[7], params[8], params[9],
                            ];
                            if stride == 0
                                || dilation == 0
                                || c == 0
                                || d == 0
                                || h == 0
                                || w == 0
                                || kd == 0
                                || kh == 0
                                || kw == 0
                            {
                                return Err(BackendError::Dispatch(
                                    "conv3d: stride, dilation, and dimensions must be positive"
                                        .into(),
                                ));
                            }
                            let input_volume = c
                                .checked_mul(d)
                                .and_then(|value| value.checked_mul(h))
                                .and_then(|value| value.checked_mul(w))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv3d: input geometry overflows".into(),
                                    )
                                })?;
                            let kernel_volume = c
                                .checked_mul(kd)
                                .and_then(|value| value.checked_mul(kh))
                                .and_then(|value| value.checked_mul(kw))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv3d: kernel geometry overflows".into(),
                                    )
                                })?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if input_slices.iter().any(|slice| {
                                !slice.size.is_multiple_of(scalar_bytes)
                                    || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                            }) || !(out_end - out_start).is_multiple_of(scalar_bytes)
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "conv3d: storage is not complete aligned f32 data".into(),
                                ));
                            }
                            let input_elements = input_slices[0].size / scalar_bytes;
                            let weight_elements = input_slices[1].size / scalar_bytes;
                            if !input_elements.is_multiple_of(input_volume)
                                || !weight_elements.is_multiple_of(kernel_volume)
                            {
                                return Err(BackendError::Dispatch(
                                    "conv3d: storage does not contain complete tensors".into(),
                                ));
                            }
                            let padded = |size: usize, kernel: usize| {
                                let padded_size = padding
                                    .checked_mul(2)
                                    .and_then(|value| size.checked_add(value))?;
                                let effective_kernel = dilation
                                    .checked_mul(kernel.checked_sub(1)?)?
                                    .checked_add(1)?;
                                (effective_kernel <= padded_size)
                                    .then_some((padded_size - effective_kernel) / stride + 1)
                            };
                            let d_out = padded(d, kd).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "conv3d: invalid or overflowing depth geometry".into(),
                                )
                            })?;
                            let h_out = padded(h, kh).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "conv3d: invalid or overflowing height geometry".into(),
                                )
                            })?;
                            let w_out = padded(w, kw).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "conv3d: invalid or overflowing width geometry".into(),
                                )
                            })?;
                            let n = input_elements / input_volume;
                            let f = weight_elements / kernel_volume;
                            let expected_output = n
                                .checked_mul(f)
                                .and_then(|value| value.checked_mul(d_out))
                                .and_then(|value| value.checked_mul(h_out))
                                .and_then(|value| value.checked_mul(w_out))
                                .and_then(|value| value.checked_mul(scalar_bytes))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "conv3d: output storage size overflows".into(),
                                    )
                                })?;
                            if out_end - out_start != expected_output
                                || input_slices.iter().any(|slice| {
                                    arena::ranges_overlap(
                                        slice.offset,
                                        slice.offset + slice.size,
                                        out_start,
                                        out_end,
                                    )
                                })
                            {
                                return Err(BackendError::Dispatch(
                                    "conv3d: invalid output storage or overlapping tensors".into(),
                                ));
                            }
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
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let arena_data = arena.data_mut();
                                    let input = try_copy_slice(
                                        bytemuck::cast_slice::<_, f32>(
                                            &arena_data[input_slice.offset
                                                ..input_slice.offset + input_slice.size],
                                        ),
                                        "conv_transpose2d input materialization",
                                    )?;
                                    let weight = try_copy_slice(
                                        bytemuck::cast_slice::<_, f32>(
                                            &arena_data[weight_slice.offset
                                                ..weight_slice.offset + weight_slice.size],
                                        ),
                                        "conv_transpose2d weight materialization",
                                    )?;
                                    (input, weight)
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                out_f32.fill(0.0f32);
                                for nn in 0..n {
                                    for cc in 0..*c {
                                        for hh in 0..*hin {
                                            for ww in 0..*win {
                                                for ff in 0..f {
                                                    for kkh in 0..*kh {
                                                        for kkw in 0..*kw {
                                                            let h_out_idx = hh * *stride + kkh;
                                                            let w_out_idx = ww * *stride + kkw;
                                                            if h_out_idx >= *padding
                                                                && w_out_idx >= *padding
                                                            {
                                                                let h_out_s = h_out_idx - *padding;
                                                                let w_out_s = w_out_idx - *padding;
                                                                if h_out_s < h_out
                                                                    && w_out_s < w_out
                                                                {
                                                                    let out_idx = nn
                                                                        * (f * h_out * w_out)
                                                                        + ff * (h_out * w_out)
                                                                        + h_out_s * w_out
                                                                        + w_out_s;
                                                                    let input_idx = nn
                                                                        * (*c * *hin * *win)
                                                                        + cc * (*hin * *win)
                                                                        + hh * *win
                                                                        + ww;
                                                                    let weight_idx = cc
                                                                        * (f * *kh * *kw)
                                                                        + ff * (*kh * *kw)
                                                                        + kkh * *kw
                                                                        + kkw;
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
                        "prelu" => {
                            if input_slices.len() != 2 || params.len() < 2 {
                                return Err(BackendError::Dispatch(
                                    "prelu: expected data, weight, and complete shape metadata"
                                        .into(),
                                ));
                            }
                            let rank = params[0];
                            if rank == 0 || params.len() != rank + 1 {
                                return Err(BackendError::Dispatch(
                                    "prelu: malformed shape metadata".into(),
                                ));
                            }
                            let mut dims = Vec::new();
                            dims.try_reserve_exact(rank).map_err(|error| {
                                BackendError::Dispatch(format!(
                                    "prelu: shape allocation failed: {error}"
                                ))
                            })?;
                            dims.extend_from_slice(&params[1..]);
                            if let Some(dynamic_dims) = param_dims {
                                if dynamic_dims.len() != rank {
                                    return Err(BackendError::Dispatch(
                                        "prelu: dynamic shape rank mismatch".into(),
                                    ));
                                }
                                for (dimension, expression) in
                                    dims.iter_mut().zip(dynamic_dims.iter())
                                {
                                    let resolved = expression
                                        .evaluate_with_env(shape_env)
                                        .map_err(|error| {
                                            BackendError::Dispatch(format!(
                                                "prelu: dynamic dimension failed: {error}"
                                            ))
                                        })?;
                                    *dimension = usize::try_from(resolved).map_err(|_| {
                                        BackendError::Dispatch(
                                            "prelu: dynamic dimension does not fit usize".into(),
                                        )
                                    })?;
                                }
                            }
                            let numel = dims.iter().try_fold(1usize, |count, dimension| {
                                count.checked_mul(*dimension).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "prelu: tensor geometry overflows".into(),
                                    )
                                })
                            })?;
                            let channels = if rank == 1 { dims[0] } else { dims[1] };
                            let spatial = if rank <= 2 {
                                1
                            } else {
                                dims[2..].iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "prelu: spatial geometry overflows".into(),
                                        )
                                    })
                                })?
                            };
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let expected_data =
                                numel.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("prelu: data size overflows".into())
                                })?;
                            let data_slice = input_slices[0];
                            let weight_slice = input_slices[1];
                            let output_size = out_end - out_start;
                            let weight_elements = weight_slice.size / scalar_bytes;
                            if numel == 0
                                || channels == 0
                                || spatial == 0
                                || data_slice.size != expected_data
                                || output_size != expected_data
                                || !matches!(weight_elements, 1) && weight_elements != channels
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "prelu: invalid geometry or f32 storage".into(),
                                ));
                            }
                            let (input, weight) = {
                                let arena_data = arena.data_mut();
                                let input = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &arena_data[data_slice.offset
                                            ..data_slice.offset + data_slice.size],
                                    ),
                                    "prelu data materialization",
                                )?;
                                let weight = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &arena_data[weight_slice.offset
                                            ..weight_slice.offset + weight_slice.size],
                                    ),
                                    "prelu weight materialization",
                                )?;
                                (input, weight)
                            };
                            let output =
                                unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                            for (index, value) in output.iter_mut().enumerate() {
                                let channel = if weight_elements == 1 {
                                    0
                                } else {
                                    (index / spatial) % channels
                                };
                                *value = if input[index] > 0.0 {
                                    input[index]
                                } else {
                                    input[index] * weight[channel]
                                };
                            }
                        }
                        "rms_norm" => {
                            if input_slices.len() != 2 || params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "rms_norm: expected data, weight, and epsilon".into(),
                                ));
                            }
                            let eps = f32::from_bits(params[0] as u32);
                            let data_slice = input_slices[0];
                            let weight_slice = input_slices[1];
                            let output_size = out_end - out_start;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if !eps.is_finite()
                                || eps <= 0.0
                                || data_slice.size == 0
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
                                    "rms_norm: invalid numerical or f32 storage contract".into(),
                                ));
                            }
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let row_size = weight_slice.size / scalar_bytes;
                                arena::with_nary_f32_slices(
                                    arena,
                                    &[*data_slice, *weight_slice],
                                    output_slice,
                                    |inputs, out_f32| {
                                        rms_norm_f32(inputs[0], inputs[1], out_f32, row_size, eps);
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
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                let bias_slice = rest.get(1).copied();
                                {
                                    let mut inputs: SmallVec<[BufferSlice; 4]> =
                                        smallvec![*residual_slice, *main_slice];
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
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                let weight_slice = rest.first().copied();
                                {
                                    let mut inputs: SmallVec<[BufferSlice; 4]> =
                                        smallvec![*residual_slice, *main_slice];
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
                            if input_slices.len() != 2 || !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "embedding: expected weight and indices inputs without parameters"
                                        .into(),
                                ));
                            }
                            let weight_slice = input_slices[0];
                            let indices_slice = input_slices[1];
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let output_size = out_end - out_start;
                            if weight_slice.size == 0
                                || indices_slice.size == 0
                                || output_size == 0
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !output_size.is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "embedding: slices must contain nonempty aligned f32 storage"
                                        .into(),
                                ));
                            }
                            let index_count = indices_slice.size / scalar_bytes;
                            let output_elements = output_size / scalar_bytes;
                            if !output_elements.is_multiple_of(index_count) {
                                return Err(BackendError::Dispatch(
                                    "embedding: output does not contain complete embedding rows"
                                        .into(),
                                ));
                            }
                            let dim = output_elements / index_count;
                            let weight_elements = weight_slice.size / scalar_bytes;
                            if dim == 0 || !weight_elements.is_multiple_of(dim) {
                                return Err(BackendError::Dispatch(
                                    "embedding: weight storage does not contain complete rows"
                                        .into(),
                                ));
                            }
                            let vocabulary_size = weight_elements / dim;
                            let output_overlaps = input_slices.iter().any(|slice| {
                                let end = slice.offset + slice.size;
                                slice.offset < out_end && out_start < end
                            });
                            if output_overlaps {
                                return Err(BackendError::Dispatch(
                                    "embedding: input and output slices overlap".into(),
                                ));
                            }
                            {
                                let data = arena.data_mut();
                                let indices = bytemuck::cast_slice::<_, f32>(
                                    &data[indices_slice.offset
                                        ..indices_slice.offset + indices_slice.size],
                                );
                                if indices.iter().any(|index| {
                                    !index.is_finite()
                                        || *index < 0.0
                                        || index.fract() != 0.0
                                        || *index >= vocabulary_size as f32
                                }) {
                                    return Err(BackendError::Dispatch(
                                        "embedding: index is non-integral or out of range".into(),
                                    ));
                                }
                            }
                            if let [weight_slice, indices_slice] = &input_slices[..] {
                                let (weight, indices) = unsafe {
                                    (
                                        arena.view_f32(weight_slice.offset, weight_slice.size),
                                        arena.view_f32(indices_slice.offset, indices_slice.size),
                                    )
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                for (i, index) in indices.iter().enumerate() {
                                    let idx = *index as usize;
                                    let src_start = idx * dim;
                                    let dst_start = i * dim;
                                    out_f32[dst_start..dst_start + dim]
                                        .copy_from_slice(&weight[src_start..src_start + dim]);
                                }
                            }
                        }
                        "pow_f32" => {
                            if input_slices.len() != 2 || !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "pow_f32: expected data and exponent inputs without parameters"
                                        .into(),
                                ));
                            }
                            let data_slice = input_slices[0];
                            let exponent_slice = input_slices[1];
                            let output_size = out_end - out_start;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if data_slice.size == 0
                                || exponent_slice.size == 0
                                || data_slice.size != output_size
                                || !matches!(exponent_slice.size, 4)
                                    && exponent_slice.size != data_slice.size
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !output_size.is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "pow_f32: invalid f32 storage or exponent broadcasting".into(),
                                ));
                            }
                            if let [data_slice, exp_slice] = &input_slices[..] {
                                let (data, exponent) = {
                                    let arena_data = arena.data_mut();
                                    let data = try_copy_slice(
                                        bytemuck::cast_slice::<_, f32>(
                                            &arena_data[data_slice.offset
                                                ..data_slice.offset + data_slice.size],
                                        ),
                                        "pow_f32 data materialization",
                                    )?;
                                    let exponent = try_copy_slice(
                                        bytemuck::cast_slice::<_, f32>(
                                            &arena_data[exp_slice.offset
                                                ..exp_slice.offset + exp_slice.size],
                                        ),
                                        "pow_f32 exponent materialization",
                                    )?;
                                    (data, exponent)
                                };
                                let out_f32 =
                                    unsafe { arena.view_f32_mut(out_start, out_end - out_start) };
                                let len = data.len();
                                #[cfg(not(feature = "parallel"))]
                                {
                                    for i in 0..len {
                                        let e = if exponent.len() == 1 {
                                            exponent[0]
                                        } else {
                                            exponent[i]
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
                                                let e = if exponent.len() == 1 {
                                                    exponent[0]
                                                } else {
                                                    exponent[i]
                                                };
                                                *o = data[i].powf(e);
                                            },
                                        );
                                    } else {
                                        for i in 0..len {
                                            let e = if exponent.len() == 1 {
                                                exponent[0]
                                            } else {
                                                exponent[i]
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
                            if input_slices.len() != 1 || params.len() != 3 {
                                return Err(BackendError::Dispatch(
                                    "argmax: expected one input and axis/dimension/inner parameters"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let axis = params[0];
                            let dim_size = params[1];
                            let inner = params[2];
                            let input_elements = input_slice.size / std::mem::size_of::<f32>();
                            let reduction_span = dim_size.checked_mul(inner).ok_or_else(|| {
                                BackendError::Dispatch("argmax: reduction span overflows".into())
                            })?;
                            if dim_size == 0
                                || inner == 0
                                || input_slice.size == 0
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<u64>())
                                || !(out_end - out_start).is_multiple_of(std::mem::size_of::<u64>())
                                || !input_elements.is_multiple_of(reduction_span)
                            {
                                return Err(BackendError::Dispatch(
                                    "argmax: invalid reduction geometry or scalar storage".into(),
                                ));
                            }
                            let outer = input_elements / reduction_span;
                            let expected_output = outer
                                .checked_mul(inner)
                                .and_then(|values| values.checked_mul(std::mem::size_of::<u64>()))
                                .ok_or_else(|| {
                                    BackendError::Dispatch("argmax: output size overflows".into())
                                })?;
                            if out_end - out_start != expected_output {
                                return Err(BackendError::Dispatch(
                                    "argmax: output storage does not match reduction geometry"
                                        .into(),
                                ));
                            }
                            if let Some(input_slice) = input_slices.first() {
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
                                        let copy =
                                            try_copy_slice(src, "argmax input materialization")?;
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
                            if input_slices.len() != 1 || params.len() < 4 {
                                return Err(BackendError::Dispatch(
                                    "topk_fused: expected one input and complete shape metadata"
                                        .into(),
                                ));
                            }
                            let k = params[0];
                            let rank = params[1];
                            let expected_params = rank.checked_add(3).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "topk_fused: parameter count overflows".into(),
                                )
                            })?;
                            if rank == 0 || params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "topk_fused: malformed shape metadata".into(),
                                ));
                            }
                            let mut dims = Vec::new();
                            dims.try_reserve_exact(rank).map_err(|error| {
                                BackendError::Dispatch(format!(
                                    "topk_fused: shape allocation failed: {error}"
                                ))
                            })?;
                            dims.extend_from_slice(&params[2..2 + rank]);
                            if let Some(dynamic_dims) = param_dims {
                                if dynamic_dims.len() != rank {
                                    return Err(BackendError::Dispatch(
                                        "topk_fused: dynamic shape rank mismatch".into(),
                                    ));
                                }
                                for (dimension, expression) in
                                    dims.iter_mut().zip(dynamic_dims.iter())
                                {
                                    let resolved = expression
                                        .evaluate_with_env(shape_env)
                                        .map_err(|error| {
                                            BackendError::Dispatch(format!(
                                                "topk_fused: dynamic dimension failed: {error}"
                                            ))
                                        })?;
                                    *dimension = usize::try_from(resolved).map_err(|_| {
                                        BackendError::Dispatch(
                                            "topk_fused: dynamic dimension does not fit usize"
                                                .into(),
                                        )
                                    })?;
                                }
                            }
                            let axis = params[2 + rank];
                            if axis >= rank {
                                return Err(BackendError::Dispatch(
                                    "topk_fused: axis is out of range".into(),
                                ));
                            }
                            let axis_size = dims[axis];
                            let outer =
                                dims[..axis].iter().try_fold(1usize, |count, dimension| {
                                    count.checked_mul(*dimension).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "topk_fused: outer geometry overflows".into(),
                                        )
                                    })
                                })?;
                            let inner =
                                dims[axis + 1..]
                                    .iter()
                                    .try_fold(1usize, |count, dimension| {
                                        count.checked_mul(*dimension).ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "topk_fused: inner geometry overflows".into(),
                                            )
                                        })
                                    })?;
                            let input_elements = outer
                                .checked_mul(axis_size)
                                .and_then(|count| count.checked_mul(inner))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "topk_fused: input geometry overflows".into(),
                                    )
                                })?;
                            let output_elements = outer
                                .checked_mul(k)
                                .and_then(|count| count.checked_mul(inner))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "topk_fused: output geometry overflows".into(),
                                    )
                                })?;
                            let input_slice = input_slices[0];
                            let secondary = secondary_output_slice.ok_or_else(|| {
                                BackendError::Dispatch(
                                    "topk_fused: missing secondary indices output".into(),
                                )
                            })?;
                            let expected_input =
                                input_elements.checked_mul(4).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "topk_fused: input size overflows".into(),
                                    )
                                })?;
                            let expected_values =
                                output_elements.checked_mul(4).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "topk_fused: values output size overflows".into(),
                                    )
                                })?;
                            let expected_indices =
                                output_elements.checked_mul(8).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "topk_fused: indices output size overflows".into(),
                                    )
                                })?;
                            let ranges_overlap = |a: BufferSlice, b: BufferSlice| {
                                a.offset < b.offset + b.size && b.offset < a.offset + a.size
                            };
                            if k == 0
                                || k > axis_size
                                || input_slice.size != expected_input
                                || !input_slice.offset.is_multiple_of(4)
                                || !output_slice.offset.is_multiple_of(4)
                                || output_slice.size != expected_values
                                || !secondary.offset.is_multiple_of(8)
                                || secondary.size != expected_indices
                                || ranges_overlap(input_slice, *output_slice)
                                || ranges_overlap(input_slice, secondary)
                                || ranges_overlap(*output_slice, secondary)
                            {
                                return Err(BackendError::Dispatch(
                                    "topk_fused: invalid geometry, storage, or overlapping outputs"
                                        .into(),
                                ));
                            }
                            let mut indexed = Vec::new();
                            indexed.try_reserve_exact(axis_size).map_err(|error| {
                                BackendError::Dispatch(format!(
                                    "topk_fused: candidate allocation failed: {error}"
                                ))
                            })?;
                            let mut selected_indices = Vec::new();
                            selected_indices
                                .try_reserve_exact(output_elements)
                                .map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "topk_fused: index allocation failed: {error}"
                                    ))
                                })?;
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, output| {
                                    for outer_index in 0..outer {
                                        for inner_index in 0..inner {
                                            indexed.clear();
                                            for axis_index in 0..axis_size {
                                                let input_index =
                                                    (outer_index * axis_size + axis_index) * inner
                                                        + inner_index;
                                                indexed.push((axis_index, input[input_index]));
                                            }
                                            if axis_size > k {
                                                indexed.select_nth_unstable_by(
                                                    axis_size - k,
                                                    |a, b| a.1.total_cmp(&b.1),
                                                );
                                            }
                                            for (selected_index, candidate) in
                                                indexed[axis_size - k..].iter().enumerate()
                                            {
                                                let output_index =
                                                    (outer_index * k + selected_index) * inner
                                                        + inner_index;
                                                output[output_index] = candidate.1;
                                                selected_indices.push(candidate.0 as u64);
                                            }
                                        }
                                    }
                                },
                            );
                            let data = arena.data_mut();
                            let indices = bytemuck::cast_slice_mut::<_, u64>(
                                &mut data[secondary.offset..secondary.offset + secondary.size],
                            );
                            for outer_index in 0..outer {
                                for inner_index in 0..inner {
                                    for selected_index in 0..k {
                                        let logical_index = (outer_index * inner + inner_index) * k
                                            + selected_index;
                                        let output_index = (outer_index * k + selected_index)
                                            * inner
                                            + inner_index;
                                        indices[output_index] = selected_indices[logical_index];
                                    }
                                }
                            }
                        }

                        "upsample_nearest2d" => {
                            if input_slices.len() != 1 || params.len() != 4 {
                                return Err(BackendError::Dispatch(
                                    "upsample_nearest2d: expected one input and scale/input geometry"
                                        .into(),
                                ));
                            }
                            let [scale_h, scale_w, h_in, w_in] = params[..] else {
                                unreachable!();
                            };
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_elements = input_slice.size / scalar_bytes;
                            let hw = h_in.checked_mul(w_in).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_nearest2d: input geometry overflows".into(),
                                )
                            })?;
                            let scale = scale_h.checked_mul(scale_w).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_nearest2d: scale geometry overflows".into(),
                                )
                            })?;
                            let expected_output = input_elements
                                .checked_mul(scale)
                                .and_then(|elements| elements.checked_mul(scalar_bytes))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "upsample_nearest2d: output size overflows".into(),
                                    )
                                })?;
                            if scale_h == 0
                                || scale_w == 0
                                || h_in == 0
                                || w_in == 0
                                || !input_elements.is_multiple_of(hw)
                                || output_slice.size != expected_output
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(scalar_bytes)
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "upsample_nearest2d: invalid geometry or f32 storage".into(),
                                ));
                            }
                            let nc = input_elements / hw;
                            try_with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                "upsample_nearest2d input",
                                |input, out_f32| {
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
                                },
                            )?;
                        }
                        "upsample_bilinear2d" => {
                            if input_slices.len() != 1 || params.len() != 4 {
                                return Err(BackendError::Dispatch(
                                    "upsample_bilinear2d: expected one input and scale/input geometry"
                                        .into(),
                                ));
                            }
                            let [scale_h, scale_w, h_in, w_in] = params[..] else {
                                unreachable!();
                            };
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_elements = input_slice.size / scalar_bytes;
                            let hw = h_in.checked_mul(w_in).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_bilinear2d: input geometry overflows".into(),
                                )
                            })?;
                            let scale = scale_h.checked_mul(scale_w).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_bilinear2d: scale geometry overflows".into(),
                                )
                            })?;
                            let expected_output = input_elements
                                .checked_mul(scale)
                                .and_then(|elements| elements.checked_mul(scalar_bytes))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "upsample_bilinear2d: output size overflows".into(),
                                    )
                                })?;
                            if scale_h == 0
                                || scale_w == 0
                                || h_in == 0
                                || w_in == 0
                                || !input_elements.is_multiple_of(hw)
                                || output_slice.size != expected_output
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(scalar_bytes)
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "upsample_bilinear2d: invalid geometry or f32 storage".into(),
                                ));
                            }
                            let nc = input_elements / hw;
                            let h_out = h_in.checked_mul(scale_h).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_bilinear2d: output height overflows".into(),
                                )
                            })?;
                            let w_out = w_in.checked_mul(scale_w).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "upsample_bilinear2d: output width overflows".into(),
                                )
                            })?;
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, out_f32| {
                                    for nci in 0..nc {
                                        for hi in 0..h_out {
                                            for wi in 0..w_out {
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
                                                let v0 = v00 * (1.0 - dw as f32) + v01 * dw as f32;
                                                let v1 = v10 * (1.0 - dw as f32) + v11 * dw as f32;
                                                let val = v0 * (1.0 - dh as f32) + v1 * dh as f32;
                                                let out_idx = nci * h_out * w_out + hi * w_out + wi;
                                                out_f32[out_idx] = val;
                                            }
                                        }
                                    }
                                },
                            );
                        }
                        "adaptive_avg_pool2d" => {
                            if input_slices.len() != 1 || params.len() != 4 {
                                return Err(BackendError::Dispatch(
                                    "adaptive_avg_pool2d: expected one input and output/input geometry"
                                        .into(),
                                ));
                            }
                            let [out_h, out_w, h_in, w_in] = params[..] else {
                                unreachable!();
                            };
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_elements = input_slice.size / scalar_bytes;
                            let input_hw = h_in.checked_mul(w_in).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "adaptive_avg_pool2d: input geometry overflows".into(),
                                )
                            })?;
                            let output_hw = out_h.checked_mul(out_w).ok_or_else(|| {
                                BackendError::Dispatch(
                                    "adaptive_avg_pool2d: output geometry overflows".into(),
                                )
                            })?;
                            if out_h == 0 || out_w == 0 || h_in == 0 || w_in == 0 {
                                return Err(BackendError::Dispatch(
                                    "adaptive_avg_pool2d: dimensions must be positive".into(),
                                ));
                            }
                            let nc = input_elements / input_hw;
                            let expected_output = nc
                                .checked_mul(output_hw)
                                .and_then(|elements| elements.checked_mul(scalar_bytes))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "adaptive_avg_pool2d: output size overflows".into(),
                                    )
                                })?;
                            if out_h == 0
                                || out_w == 0
                                || h_in == 0
                                || w_in == 0
                                || !input_elements.is_multiple_of(input_hw)
                                || output_slice.size != expected_output
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(scalar_bytes)
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "adaptive_avg_pool2d: invalid geometry or f32 storage".into(),
                                ));
                            }
                            let input_end = input_slice.offset + input_slice.size;
                            let output_end = output_slice.offset + output_slice.size;
                            if arena::ranges_overlap(
                                input_slice.offset,
                                input_end,
                                output_slice.offset,
                                output_end,
                            ) {
                                let input = {
                                    let arena_data = arena.data_mut();
                                    try_copy_slice(
                                        bytemuck::cast_slice::<_, f32>(
                                            &arena_data[input_slice.offset..input_end],
                                        ),
                                        "adaptive_avg_pool2d input",
                                    )?
                                };
                                telemetry::record_arena_temp_copy(input_slice.size);
                                let output = unsafe {
                                    arena.view_f32_mut(output_slice.offset, output_slice.size)
                                };
                                microkernels::adaptive_avg_pool2d_f32_scalar(
                                    &input, output, nc, h_in, w_in, out_h, out_w,
                                );
                            } else {
                                let arena_data = arena.data_mut();
                                // SAFETY: dispatch validated aligned complete f32 storage,
                                // both ranges are in the arena, and they are disjoint.
                                unsafe {
                                    let input = std::slice::from_raw_parts(
                                        arena_data.as_ptr().add(input_slice.offset).cast::<f32>(),
                                        input_slice.size / scalar_bytes,
                                    );
                                    let output = std::slice::from_raw_parts_mut(
                                        arena_data
                                            .as_mut_ptr()
                                            .add(output_slice.offset)
                                            .cast::<f32>(),
                                        output_slice.size / scalar_bytes,
                                    );
                                    microkernels::adaptive_avg_pool2d_f32_scalar(
                                        input, output, nc, h_in, w_in, out_h, out_w,
                                    );
                                }
                            }
                        }
                        "repeat" => {
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "repeat: expected one input and encoded shape/repeats".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank
                                .checked_mul(2)
                                .and_then(|count| count.checked_add(1))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "repeat: parameter count overflows".into(),
                                    )
                                })?;
                            if params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "repeat: shape and repeat ranks must match".into(),
                                ));
                            }
                            let input_shape = &params[1..1 + rank];
                            let repeats = &params[1 + rank..];
                            let input_elements =
                                input_shape.iter().try_fold(1usize, |count, dim| {
                                    count.checked_mul(*dim).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "repeat: input shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let output_elements = input_shape.iter().zip(repeats).try_fold(
                                1usize,
                                |count, (dim, repeat)| {
                                    if *repeat == 0 {
                                        return Err(BackendError::Dispatch(
                                            "repeat: repeat factors must be positive".into(),
                                        ));
                                    }
                                    dim.checked_mul(*repeat)
                                        .and_then(|output_dim| count.checked_mul(output_dim))
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "repeat: output shape product overflows".into(),
                                            )
                                        })
                                },
                            )?;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let expected_input =
                                input_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("repeat: input size overflows".into())
                                })?;
                            let expected_output =
                                output_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("repeat: output size overflows".into())
                                })?;
                            if input_slice.size != expected_input
                                || output_slice.size != expected_output
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "repeat: invalid shape or f32 storage".into(),
                                ));
                            }
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, out_f32| {
                                    for (output_index, output) in out_f32.iter_mut().enumerate() {
                                        let mut remaining = output_index;
                                        let mut input_index = 0usize;
                                        let mut input_stride = 1usize;
                                        for dimension in (0..rank).rev() {
                                            let output_dim =
                                                input_shape[dimension] * repeats[dimension];
                                            let output_coordinate = remaining % output_dim;
                                            remaining /= output_dim;
                                            input_index += (output_coordinate
                                                % input_shape[dimension])
                                                * input_stride;
                                            input_stride *= input_shape[dimension];
                                        }
                                        *output = input[input_index];
                                    }
                                },
                            );
                        }
                        "cumsum" => {
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "cumsum: expected one input and encoded shape/options".into(),
                                ));
                            }
                            let rank = params[0];
                            let expected_params = rank.checked_add(4).ok_or_else(|| {
                                BackendError::Dispatch("cumsum: parameter count overflows".into())
                            })?;
                            if rank == 0 || params.len() != expected_params {
                                return Err(BackendError::Dispatch(
                                    "cumsum: malformed shape/options metadata".into(),
                                ));
                            }
                            let input_shape = &params[1..1 + rank];
                            let dim = params[1 + rank];
                            let exclusive = params[2 + rank];
                            let reverse = params[3 + rank];
                            if dim >= rank || exclusive > 1 || reverse > 1 {
                                return Err(BackendError::Dispatch(
                                    "cumsum: invalid dimension or boolean option".into(),
                                ));
                            }
                            let input_elements =
                                input_shape.iter().try_fold(1usize, |count, size| {
                                    count.checked_mul(*size).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "cumsum: input shape product overflows".into(),
                                        )
                                    })
                                })?;
                            let inner =
                                input_shape[dim + 1..]
                                    .iter()
                                    .try_fold(1usize, |count, size| {
                                        count.checked_mul(*size).ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "cumsum: inner geometry overflows".into(),
                                            )
                                        })
                                    })?;
                            let outer =
                                input_shape[..dim].iter().try_fold(1usize, |count, size| {
                                    count.checked_mul(*size).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "cumsum: outer geometry overflows".into(),
                                        )
                                    })
                                })?;
                            let axis_size = input_shape[dim];
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let expected_bytes =
                                input_elements.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch("cumsum: storage size overflows".into())
                                })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != expected_bytes
                                || output_slice.size != expected_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "cumsum: shape and f32 storage disagree".into(),
                                ));
                            }
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, out_f32| {
                                    for outer_index in 0..outer {
                                        for inner_index in 0..inner {
                                            let mut sum = 0.0f32;
                                            if reverse == 0 {
                                                for axis_index in 0..axis_size {
                                                    let index = outer_index * axis_size * inner
                                                        + axis_index * inner
                                                        + inner_index;
                                                    if exclusive == 0 {
                                                        sum += input[index];
                                                        out_f32[index] = sum;
                                                    } else {
                                                        out_f32[index] = sum;
                                                        sum += input[index];
                                                    }
                                                }
                                            } else {
                                                for axis_index in (0..axis_size).rev() {
                                                    let index = outer_index * axis_size * inner
                                                        + axis_index * inner
                                                        + inner_index;
                                                    if exclusive == 0 {
                                                        sum += input[index];
                                                        out_f32[index] = sum;
                                                    } else {
                                                        out_f32[index] = sum;
                                                        sum += input[index];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                            );
                        }
                        "erf_f32" => {
                            if !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "erf_f32: expected no parameters".into(),
                                ));
                            }
                            if let Some(input_slice) = input_slices.first() {
                                let output_slice = BufferSlice::new(out_start, out_end - out_start);
                                arena::with_unary_f32_slices(
                                    arena,
                                    *input_slice,
                                    output_slice,
                                    |input, out_f32| {
                                        for i in 0..out_f32.len() {
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
                            if input_slices.len() != 1 || params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "flip: expected one input and encoded dimensions/shape".into(),
                                ));
                            }
                            let num_dims = params[0];
                            let shape_start = num_dims.checked_add(1).ok_or_else(|| {
                                BackendError::Dispatch("flip: dimension count overflows".into())
                            })?;
                            if shape_start > params.len() {
                                return Err(BackendError::Dispatch(
                                    "flip: truncated dimension metadata".into(),
                                ));
                            }
                            let flip_dims = &params[1..shape_start];
                            let shape = &params[shape_start..];
                            if flip_dims.iter().any(|dim| *dim >= shape.len())
                                || flip_dims
                                    .iter()
                                    .enumerate()
                                    .any(|(index, dim)| flip_dims[..index].contains(dim))
                            {
                                return Err(BackendError::Dispatch(
                                    "flip: dimensions must be unique and within rank".into(),
                                ));
                            }
                            let elements = shape.iter().try_fold(1usize, |count, size| {
                                count.checked_mul(*size).ok_or_else(|| {
                                    BackendError::Dispatch("flip: shape product overflows".into())
                                })
                            })?;
                            let expected_bytes = elements
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                BackendError::Dispatch("flip: storage size overflows".into())
                            })?;
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            if input_slice.size != expected_bytes
                                || output_slice.size != expected_bytes
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "flip: shape and f32 storage disagree".into(),
                                ));
                            }
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |input, out_f32| {
                                    for (output_index, output) in out_f32.iter_mut().enumerate() {
                                        let mut remaining = output_index;
                                        let mut source_index = 0usize;
                                        let mut source_stride = 1usize;
                                        for dimension in (0..shape.len()).rev() {
                                            let coordinate = remaining % shape[dimension];
                                            remaining /= shape[dimension];
                                            let source_coordinate =
                                                if flip_dims.contains(&dimension) {
                                                    shape[dimension] - 1 - coordinate
                                                } else {
                                                    coordinate
                                                };
                                            source_index += source_coordinate * source_stride;
                                            source_stride *= shape[dimension];
                                        }
                                        *output = input[source_index];
                                    }
                                },
                            );
                        }
                        "where_f32" => {
                            if input_slices.len() != 3 || !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "where_f32: expected condition, x, and y without parameters"
                                        .into(),
                                ));
                            }
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let output_elements = output_slice.size / scalar_bytes;
                            let malformed_input = input_slices.iter().any(|slice| {
                                !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                    || !slice.size.is_multiple_of(scalar_bytes)
                                    || (output_elements > 0 && slice.size == 0)
                                    || (output_elements == 0 && slice.size != 0)
                                    || (slice.size > 0
                                        && !output_slice.size.is_multiple_of(slice.size))
                            });
                            if malformed_input
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice.size.is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "where_f32: invalid broadcasting or f32 storage".into(),
                                ));
                            }
                            arena::with_nary_f32_slices(
                                arena,
                                input_slices,
                                output_slice,
                                |inputs, out_f32| {
                                    let cond = inputs[0];
                                    let x = inputs[1];
                                    let y = inputs[2];
                                    for (index, output) in out_f32.iter_mut().enumerate() {
                                        *output = if cond[index % cond.len()] != 0.0 {
                                            x[index % x.len()]
                                        } else {
                                            y[index % y.len()]
                                        };
                                    }
                                },
                            );
                        }
                        // â”€â”€ Optimizer kernels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                        "sgd_update_f32" => {
                            if input_slices.len() != 2 || !(1..=2).contains(&params.len()) {
                                return Err(BackendError::Dispatch(
                                    "sgd_update_f32: expected weights/gradients and lr[/weight_decay]"
                                        .into(),
                                ));
                            }
                            let weight_slice = input_slices[0];
                            let gradient_slice = input_slices[1];
                            let output_size = out_end - out_start;
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let lr = f32::from_bits(params[0] as u32);
                            let wd = params
                                .get(1)
                                .map_or(0.0, |bits| f32::from_bits(*bits as u32));
                            if !lr.is_finite()
                                || lr < 0.0
                                || !wd.is_finite()
                                || wd < 0.0
                                || weight_slice.size != gradient_slice.size
                                || weight_slice.size != output_size
                                || input_slices.iter().any(|slice| {
                                    !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "sgd_update_f32: invalid numerical or f32 storage contract"
                                        .into(),
                                ));
                            }
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
                            if input_slices.len() != 1 || params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "gradient f8 quantization expects one f32 input and element count"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let numel = params[0];
                            let expected_input = numel
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "gradient f8 quantization input size overflows".into(),
                                    )
                                })?;

                            let output_size = out_end - out_start;
                            let effective_numel = input_slice.size / std::mem::size_of::<f32>();
                            let required_packed_bytes = effective_numel
                                .div_ceil(4)
                                .checked_mul(std::mem::size_of::<u32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "gradient f8 packed size overflows".into(),
                                    )
                                })?;
                            if !input_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                                || input_slice.size > expected_input
                                || !input_slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<u32>())
                                || output_size < required_packed_bytes
                                || !output_size.is_multiple_of(std::mem::size_of::<u32>())
                            {
                                return Err(BackendError::Dispatch(format!(
                                    "gradient f8 quantization has invalid storage: input={}, output={}, declared={expected_input}",
                                    input_slice.size, output_size
                                )));
                            }
                            if let Some(input_slice) = input_slices.first() {
                                let source =
                                    unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                                let mut in_f32 = Vec::new();
                                in_f32.try_reserve_exact(source.len()).map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "gradient f8 quantization input materialization failed: {error}"
                                    ))
                                })?;
                                in_f32.extend_from_slice(source);
                                let out = unsafe {
                                    bytemuck::cast_slice_mut(
                                        arena.view_f32_mut(out_start, out_end - out_start),
                                    )
                                };
                                let num_words = effective_numel.div_ceil(4);
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
                            if input_slices.len() != 1 || params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "gradient f8 dequantization expects one input and element count"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let numel = params[0];
                            // Quantized-gradient slots currently retain logical F32 byte sizing;
                            // packed words occupy the leading ceil(numel / 4) u32 values.
                            let _declared_bytes = numel
                                .checked_mul(std::mem::size_of::<f32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "gradient f8 declared size overflows".into(),
                                    )
                                })?;
                            let output_size = out_end - out_start;
                            let effective_numel = output_size / std::mem::size_of::<f32>();
                            let required_packed_bytes = effective_numel
                                .div_ceil(4)
                                .checked_mul(std::mem::size_of::<u32>())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "gradient f8 packed input size overflows".into(),
                                    )
                                })?;
                            if !input_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<u32>())
                                || input_slice.size < required_packed_bytes
                                || !input_slice.size.is_multiple_of(std::mem::size_of::<u32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<f32>())
                                || !output_size.is_multiple_of(std::mem::size_of::<f32>())
                            {
                                return Err(BackendError::Dispatch(
                                    "gradient f8 dequantization has invalid storage".into(),
                                ));
                            }
                            if let Some(input_slice) = input_slices.first() {
                                let source = unsafe {
                                    std::slice::from_raw_parts(
                                        arena
                                            .view_f32(input_slice.offset, input_slice.size)
                                            .as_ptr()
                                            as *const u32,
                                        input_slice.size / 4,
                                    )
                                };
                                let mut in_u32 = Vec::new();
                                in_u32.try_reserve_exact(source.len()).map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "gradient f8 dequantization input materialization failed: {error}"
                                    ))
                                })?;
                                in_u32.extend_from_slice(source);
                                let out_bytes = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        arena
                                            .view_f32_mut(out_start, out_end - out_start)
                                            .as_mut_ptr()
                                            as *mut u8,
                                        out_end - out_start,
                                    )
                                };
                                let num_words = effective_numel.div_ceil(4);
                                for w in 0..num_words {
                                    let word = F8x4R(in_u32[w]);
                                    let vals = word.unpack_to_f32();
                                    let base = w * 4;
                                    for j in 0..4 {
                                        let idx = base + j;
                                        if idx < effective_numel {
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
                            if input_slices.len() != 1 || params.len() != 2 {
                                return Err(BackendError::Dispatch(
                                    "gradient_scale: expected one input, element count, and scale"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let declared_numel = params[0];
                            let scale = f32::from_bits(params[1] as u32);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if !scale.is_finite()
                                || !input_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(scalar_bytes)
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice.size.is_multiple_of(scalar_bytes)
                                || (declared_numel == 0
                                    && (input_slice.size != 0 || output_slice.size != 0))
                            {
                                return Err(BackendError::Dispatch(
                                    "gradient_scale: invalid numerical or f32 storage contract"
                                        .into(),
                                ));
                            }
                            let effective_numel = declared_numel
                                .min(input_slice.size / scalar_bytes)
                                .min(output_slice.size / scalar_bytes);
                            arena::with_unary_f32_slices(
                                arena,
                                input_slice,
                                output_slice,
                                |in_f32, out_f32| {
                                    let input = &in_f32[..effective_numel];
                                    let output = &mut out_f32[..effective_numel];
                                    #[cfg(not(feature = "parallel"))]
                                    {
                                        for (output, input) in output.iter_mut().zip(input) {
                                            *output = *input * scale;
                                        }
                                    }
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        if effective_numel >= 4096 {
                                            output.par_iter_mut().zip(input.par_iter()).for_each(
                                                |(output, input)| {
                                                    *output = *input * scale;
                                                },
                                            );
                                        } else {
                                            for (output, input) in output.iter_mut().zip(input) {
                                                *output = *input * scale;
                                            }
                                        }
                                    }
                                },
                            );
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
                            if input_slices.len() != 3 || params.len() != 3 {
                                return Err(BackendError::Dispatch(
                                    "muon_update_f32: expected weights, gradients, momentum, and three parameters"
                                        .into(),
                                ));
                            }
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let lr = f32::from_bits(params[0] as u32);
                            let beta = f32::from_bits(params[1] as u32);
                            let wd = f32::from_bits(params[2] as u32);
                            let reference_size = input_slices[0].size;
                            let input_overlap =
                                input_slices.iter().enumerate().any(|(index, left)| {
                                    input_slices[index + 1..].iter().any(|right| {
                                        left.offset < right.offset + right.size
                                            && right.offset < left.offset + left.size
                                    })
                                });
                            let output_is_weight = output_slice == input_slices[0];
                            let output_overlap = !output_is_weight
                                && input_slices.iter().any(|input| {
                                    input.offset < output_slice.offset + output_slice.size
                                        && output_slice.offset < input.offset + input.size
                                });
                            if !lr.is_finite()
                                || lr < 0.0
                                || !beta.is_finite()
                                || !(0.0..1.0).contains(&beta)
                                || !wd.is_finite()
                                || wd < 0.0
                                || input_slices.iter().any(|slice| {
                                    slice.size != reference_size
                                        || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || output_slice.size != reference_size
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || input_overlap
                                || output_overlap
                            {
                                return Err(BackendError::Dispatch(
                                    "muon_update_f32: invalid numerical, storage, or overlap contract"
                                        .into(),
                                ));
                            }
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
                            if input_slices.len() != 3 || !(3..=4).contains(&params.len()) {
                                return Err(BackendError::Dispatch(
                                    "lion_update_f32: expected weights, gradients, momentum, and three/four parameters"
                                        .into(),
                                ));
                            }
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let lr = f32::from_bits(params[0] as u32);
                            let beta1 = f32::from_bits(params[1] as u32);
                            let beta2 = f32::from_bits(params[2] as u32);
                            let wd = params
                                .get(3)
                                .map_or(0.0, |bits| f32::from_bits(*bits as u32));
                            let reference_size = input_slices[0].size;
                            let input_overlap =
                                input_slices.iter().enumerate().any(|(index, left)| {
                                    input_slices[index + 1..].iter().any(|right| {
                                        left.offset < right.offset + right.size
                                            && right.offset < left.offset + left.size
                                    })
                                });
                            let output_is_weight = output_slice == input_slices[0];
                            let output_overlap = !output_is_weight
                                && input_slices.iter().any(|input| {
                                    input.offset < output_slice.offset + output_slice.size
                                        && output_slice.offset < input.offset + input.size
                                });
                            if !lr.is_finite()
                                || lr < 0.0
                                || !beta1.is_finite()
                                || !(0.0..1.0).contains(&beta1)
                                || !beta2.is_finite()
                                || !(0.0..1.0).contains(&beta2)
                                || !wd.is_finite()
                                || wd < 0.0
                                || input_slices.iter().any(|slice| {
                                    slice.size != reference_size
                                        || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || output_slice.size != reference_size
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || input_overlap
                                || output_overlap
                            {
                                return Err(BackendError::Dispatch(
                                    "lion_update_f32: invalid numerical, storage, or overlap contract"
                                        .into(),
                                ));
                            }
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
                            if input_slices.len() != 3 || params.len() != 3 {
                                return Err(BackendError::Dispatch(
                                    "rmsprop_update_f32: expected weights, gradients, variance, and three parameters"
                                        .into(),
                                ));
                            }
                            let output_slice = BufferSlice::new(out_start, out_end - out_start);
                            let scalar_bytes = std::mem::size_of::<f32>();
                            let lr = f32::from_bits(params[0] as u32);
                            let beta = f32::from_bits(params[1] as u32);
                            let eps = f32::from_bits(params[2] as u32);
                            let reference_size = input_slices[0].size;
                            let input_overlap =
                                input_slices.iter().enumerate().any(|(index, left)| {
                                    input_slices[index + 1..].iter().any(|right| {
                                        left.offset < right.offset + right.size
                                            && right.offset < left.offset + left.size
                                    })
                                });
                            let output_is_weight = output_slice == input_slices[0];
                            let output_overlap = !output_is_weight
                                && input_slices.iter().any(|input| {
                                    input.offset < output_slice.offset + output_slice.size
                                        && output_slice.offset < input.offset + input.size
                                });
                            if !lr.is_finite()
                                || lr < 0.0
                                || !beta.is_finite()
                                || !(0.0..1.0).contains(&beta)
                                || !eps.is_finite()
                                || eps <= 0.0
                                || input_slices.iter().any(|slice| {
                                    slice.size != reference_size
                                        || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                                        || !slice.size.is_multiple_of(scalar_bytes)
                                })
                                || output_slice.size != reference_size
                                || !output_slice
                                    .offset
                                    .is_multiple_of(std::mem::align_of::<f32>())
                                || input_overlap
                                || output_overlap
                            {
                                return Err(BackendError::Dispatch(
                                    "rmsprop_update_f32: invalid numerical, storage, or overlap contract"
                                        .into(),
                                ));
                            }
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
                                let w_init = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slices[0].offset
                                            ..input_slices[0].offset + input_slices[0].size],
                                    ),
                                    "f16 Adam weight snapshot",
                                )?;
                                let m_init = try_decode_f16(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                    "f16 Adam first-moment snapshot",
                                )?;
                                let v_init = try_decode_f16(
                                    &d_ref[input_slices[3].offset
                                        ..input_slices[3].offset + input_slices[3].size],
                                    "f16 Adam second-moment snapshot",
                                )?;
                                let grad = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slices[1].offset
                                            ..input_slices[1].offset + input_slices[1].size],
                                    ),
                                    "f16 Adam gradient snapshot",
                                )?;
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n;
                            let mut w_new = w_init;
                            let mut m_new_f32 =
                                try_filled_vec(len, 0.0f32, "f16 Adam first moment")?;
                            let mut v_new_f32 =
                                try_filled_vec(len, 0.0f32, "f16 Adam second moment")?;
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    let g = grad[i];
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
                                            let g = grad[i];
                                            *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                            *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                            let m_hat = *m / bias_corr1;
                                            let v_hat = *v / bias_corr2;
                                            *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                        });
                                } else {
                                    for i in 0..len {
                                        let g = grad[i];
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
                            for (chunk, value) in d[input_slices[2].offset
                                ..input_slices[2].offset + input_slices[2].size]
                                .chunks_exact_mut(2)
                                .zip(m_new_f32.iter())
                            {
                                chunk.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
                            }
                            for (chunk, value) in d[input_slices[3].offset
                                ..input_slices[3].offset + input_slices[3].size]
                                .chunks_exact_mut(2)
                                .zip(v_new_f32.iter())
                            {
                                chunk.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
                            }
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
                                let w_init = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slices[0].offset
                                            ..input_slices[0].offset + input_slices[0].size],
                                    ),
                                    "f16 Adam weight snapshot",
                                )?;
                                let m_init = try_decode_f16(
                                    &d_ref[input_slices[2].offset
                                        ..input_slices[2].offset + input_slices[2].size],
                                    "f16 Adam first-moment snapshot",
                                )?;
                                let v_init = try_decode_f16(
                                    &d_ref[input_slices[3].offset
                                        ..input_slices[3].offset + input_slices[3].size],
                                    "f16 Adam second-moment snapshot",
                                )?;
                                let grad = try_copy_slice(
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slices[1].offset
                                            ..input_slices[1].offset + input_slices[1].size],
                                    ),
                                    "f16 Adam gradient snapshot",
                                )?;
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
                            let len = n;
                            let mut w_new = w_init;
                            let mut m_new_f32 =
                                try_filled_vec(len, 0.0f32, "f16 Adam first moment")?;
                            let mut v_new_f32 =
                                try_filled_vec(len, 0.0f32, "f16 Adam second moment")?;
                            #[cfg(not(feature = "parallel"))]
                            {
                                for i in 0..len {
                                    w_new[i] -= lr * wd * w_new[i];
                                    let g = grad[i];
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
                                            *w -= lr * wd * *w;
                                            let g = grad[i];
                                            *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                            *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                            let m_hat = *m / bias_corr1;
                                            let v_hat = *v / bias_corr2;
                                            *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                        });
                                } else {
                                    for i in 0..len {
                                        w_new[i] -= lr * wd * w_new[i];
                                        let g = grad[i];
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
                            for (chunk, value) in d[input_slices[2].offset
                                ..input_slices[2].offset + input_slices[2].size]
                                .chunks_exact_mut(2)
                                .zip(m_new_f32.iter())
                            {
                                chunk.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
                            }
                            for (chunk, value) in d[input_slices[3].offset
                                ..input_slices[3].offset + input_slices[3].size]
                                .chunks_exact_mut(2)
                                .zip(v_new_f32.iter())
                            {
                                chunk.copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
                            }
                        }
                        "cast" => {
                            if input_slices.len() != 1 || params.len() != 2 {
                                return Err(BackendError::Dispatch(
                                    "cast: expected one input and input/output byte widths".into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let output_size = out_end - out_start;
                            let in_byte_size = params[0];
                            let out_byte_size = params[1];
                            match (in_byte_size, out_byte_size) {
                                (4, 8) => {
                                    let numel = input_slice.size / 4;
                                    let expected_output =
                                        numel.checked_mul(8).ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "cast: f32-to-i64 output size overflows".into(),
                                            )
                                        })?;
                                    if !input_slice.offset.is_multiple_of(4)
                                        || !input_slice.size.is_multiple_of(4)
                                        || !out_start.is_multiple_of(8)
                                        || output_size != expected_output
                                    {
                                        return Err(BackendError::Dispatch(
                                            "cast: invalid f32-to-i64 storage contract".into(),
                                        ));
                                    }
                                    let source = unsafe {
                                        arena.view_f32(input_slice.offset, input_slice.size)
                                    };
                                    let mut input = Vec::new();
                                    input.try_reserve_exact(numel).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "cast: input materialization failed: {error}"
                                        ))
                                    })?;
                                    input.extend_from_slice(source);
                                    let output = unsafe {
                                        arena.view_u8_mut(out_start, out_end - out_start)
                                    };
                                    for (index, value) in input.iter().enumerate() {
                                        let start = index * 8;
                                        output[start..start + 8]
                                            .copy_from_slice(&(*value as i64).to_le_bytes());
                                    }
                                }
                                (8, 4) => {
                                    let numel = input_slice.size / 8;
                                    let expected_output =
                                        numel.checked_mul(4).ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "cast: i64-to-f32 output size overflows".into(),
                                            )
                                        })?;
                                    if !input_slice.offset.is_multiple_of(8)
                                        || !input_slice.size.is_multiple_of(8)
                                        || !out_start.is_multiple_of(4)
                                        || output_size != expected_output
                                    {
                                        return Err(BackendError::Dispatch(
                                            "cast: invalid i64-to-f32 storage contract".into(),
                                        ));
                                    }
                                    let data = arena.data_mut();
                                    let source_bytes = &data
                                        [input_slice.offset..input_slice.offset + input_slice.size];
                                    let mut input = Vec::new();
                                    input.try_reserve_exact(numel).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "cast: input materialization failed: {error}"
                                        ))
                                    })?;
                                    input.extend_from_slice(bytemuck::cast_slice::<_, i64>(
                                        source_bytes,
                                    ));
                                    let output = bytemuck::cast_slice_mut::<_, f32>(
                                        &mut data[out_start..out_end],
                                    );
                                    for (output, input) in output.iter_mut().zip(input) {
                                        *output = input as f32;
                                    }
                                }
                                _ => {
                                    return Err(BackendError::Dispatch(format!(
                                        "cast: unsupported byte-width conversion {in_byte_size}->{out_byte_size}"
                                    )));
                                }
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
                            if input_slices.len() != 3 || !params.is_empty() {
                                return Err(BackendError::Dispatch(
                                    "range_f32: expected start, limit, step and no parameters"
                                        .into(),
                                ));
                            }
                            let scalar_bytes = std::mem::size_of::<f32>();
                            if input_slices.iter().any(|slice| {
                                slice.size < scalar_bytes
                                    || !slice.size.is_multiple_of(scalar_bytes)
                                    || !slice.offset.is_multiple_of(std::mem::align_of::<f32>())
                            }) || !output_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                                || !output_slice.size.is_multiple_of(scalar_bytes)
                            {
                                return Err(BackendError::Dispatch(
                                    "range_f32: invalid f32 scalar or output storage".into(),
                                ));
                            }
                            let read_scalar = |slice: BufferSlice| {
                                let data = arena.data_mut();
                                bytemuck::try_cast_slice::<_, f32>(
                                    &data[slice.offset..slice.offset + scalar_bytes],
                                )
                                .ok()
                                .and_then(|values| values.first().copied())
                                .ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "range_f32: scalar storage cannot be read as f32".into(),
                                    )
                                })
                            };
                            let start_value = read_scalar(input_slices[0])?;
                            let limit_value = read_scalar(input_slices[1])?;
                            let step_value = read_scalar(input_slices[2])?;
                            if !start_value.is_finite()
                                || !limit_value.is_finite()
                                || !step_value.is_finite()
                                || step_value == 0.0
                            {
                                return Err(BackendError::Dispatch(
                                    "range_f32: start, limit, and nonzero step must be finite"
                                        .into(),
                                ));
                            }
                            let span = if step_value > 0.0 {
                                (limit_value - start_value) / step_value
                            } else {
                                (start_value - limit_value) / -step_value
                            };
                            if !span.is_finite() {
                                return Err(BackendError::Dispatch(
                                    "range_f32: element count is not finite".into(),
                                ));
                            }
                            let element_count = span.ceil().max(0.0) as usize;
                            let expected_bytes =
                                element_count.checked_mul(scalar_bytes).ok_or_else(|| {
                                    BackendError::Dispatch(
                                        "range_f32: output size overflows".into(),
                                    )
                                })?;
                            if output_slice.size != expected_bytes {
                                return Err(BackendError::Dispatch(
                                    "range_f32: output storage does not match range geometry"
                                        .into(),
                                ));
                            }
                            let data = arena.data_mut();
                            let output =
                                bytemuck::cast_slice_mut::<_, f32>(&mut data[out_start..out_end]);
                            for (index, value) in output.iter_mut().enumerate() {
                                *value = start_value + index as f32 * step_value;
                            }
                        }
                        "quantize_f32_u4" | "quantize_f32_u8" => {
                            if input_slices.len() != 1 || params.len() < 4 {
                                return Err(BackendError::Dispatch(format!(
                                    "{kernel_name}: expected one input and complete quantization metadata"
                                )));
                            }
                            let num_channels = params[0];
                            let num_elems_per_channel = params[1];
                            let numel = params[2];
                            let has_cached = params[3];
                            if num_channels == 0
                                || num_elems_per_channel == 0
                                || has_cached > 1
                                || num_channels > u32::MAX as usize
                                || num_elems_per_channel > u32::MAX as usize
                            {
                                return Err(BackendError::Dispatch(format!(
                                    "{kernel_name}: invalid channel geometry or cache mode"
                                )));
                            }
                            let expected_numel = num_channels
                                .checked_mul(num_elems_per_channel)
                                .ok_or_else(|| {
                                    BackendError::Dispatch(format!(
                                        "{kernel_name}: channel geometry overflows"
                                    ))
                                })?;
                            if numel != expected_numel {
                                return Err(BackendError::Dispatch(format!(
                                    "{kernel_name}: channel geometry does not match numel"
                                )));
                            }
                            let expected_params = if has_cached == 1 {
                                num_channels
                                    .checked_mul(2)
                                    .and_then(|count| count.checked_add(4))
                                    .ok_or_else(|| {
                                        BackendError::Dispatch(format!(
                                            "{kernel_name}: cached metadata count overflows"
                                        ))
                                    })?
                            } else {
                                4
                            };
                            if params.len() != expected_params {
                                return Err(BackendError::Dispatch(format!(
                                    "{kernel_name}: incomplete cached quantization metadata"
                                )));
                            }
                            let bit_width = if kernel_name == "quantize_f32_u4" {
                                4
                            } else {
                                8
                            };
                            let max_q = (1i32 << (bit_width - 1)) - 1; // 7 for U4, 127 for U8
                            let items_per_word = 32 / bit_width; // 8 for U4, 4 for U8
                            let input_slice = input_slices[0];
                            let expected_input = numel.checked_mul(4).ok_or_else(|| {
                                BackendError::Dispatch(format!(
                                    "{kernel_name}: input size overflows"
                                ))
                            })?;
                            let packed_words = numel.div_ceil(items_per_word);
                            let header_size = num_channels
                                .checked_mul(8)
                                .and_then(|bytes| bytes.checked_add(8))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(format!(
                                        "{kernel_name}: header size overflows"
                                    ))
                                })?;
                            let expected_output = packed_words
                                .checked_mul(4)
                                .and_then(|bytes| bytes.checked_add(header_size))
                                .ok_or_else(|| {
                                    BackendError::Dispatch(format!(
                                        "{kernel_name}: output size overflows"
                                    ))
                                })?;
                            if input_slice.size != expected_input
                                || !input_slice.offset.is_multiple_of(4)
                                || output_slice.size != expected_output
                            {
                                return Err(BackendError::Dispatch(format!(
                                    "{kernel_name}: tensor geometry and storage disagree"
                                )));
                            }

                            // Check for cached scales from wrap_quantized_optimizer
                            let has_cached = has_cached == 1;
                            let mut cached_scales = Vec::new();
                            let mut cached_zeros = Vec::new();
                            if has_cached {
                                cached_scales
                                    .try_reserve_exact(num_channels)
                                    .map_err(|error| {
                                        BackendError::Dispatch(format!(
                                        "{kernel_name}: cached scale allocation failed: {error}"
                                    ))
                                    })?;
                                cached_zeros
                                    .try_reserve_exact(num_channels)
                                    .map_err(|error| {
                                        BackendError::Dispatch(format!(
                                        "{kernel_name}: cached offset allocation failed: {error}"
                                    ))
                                    })?;
                                let sc_start = 4;
                                let sc_end = sc_start + num_channels;
                                let zp_start = sc_end;
                                let zp_end = zp_start + num_channels;
                                for i in sc_start..sc_end {
                                    let bits = params[i];
                                    cached_scales.push(f32::from_bits(bits as u32));
                                }
                                for i in zp_start..zp_end {
                                    let bits = params[i];
                                    cached_zeros.push(f32::from_bits(bits as u32));
                                }
                                if cached_scales
                                    .iter()
                                    .any(|scale| !scale.is_finite() || *scale <= 0.0)
                                    || cached_zeros.iter().any(|zero| !zero.is_finite())
                                {
                                    return Err(BackendError::Dispatch(format!(
                                        "{kernel_name}: cached affine metadata must be finite with positive scales"
                                    )));
                                }
                            }

                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let input_f32 = bytemuck::cast_slice::<_, f32>(
                                    &d[input_slice.offset..input_slice.offset + input_slice.size],
                                );
                                if input_f32.iter().any(|value| !value.is_finite()) {
                                    return Err(BackendError::Dispatch(format!(
                                        "{kernel_name}: input values must be finite"
                                    )));
                                }
                                let mut f32_data = Vec::new();
                                f32_data
                                    .try_reserve_exact(input_f32.len())
                                    .map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "{kernel_name}: input materialization failed: {error}"
                                        ))
                                    })?;
                                f32_data.extend_from_slice(input_f32);

                                let mut scales = Vec::new();
                                scales.try_reserve_exact(num_channels).map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "{kernel_name}: scale allocation failed: {error}"
                                    ))
                                })?;
                                let mut zero_points =
                                    try_filled_vec(num_channels, 0.0f32, kernel_name)?;
                                let mut packed = try_filled_vec(packed_words, 0u32, kernel_name)?;

                                if has_cached
                                    && cached_scales.len() == num_channels
                                    && cached_zeros.len() == num_channels
                                {
                                    scales = cached_scales;
                                    zero_points = cached_zeros;
                                    // Pack using cached scales
                                    for ch in 0..num_channels {
                                        let start = ch * num_elems_per_channel;
                                        let end = start + num_elems_per_channel;
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
                                        let end = start + num_elems_per_channel;
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
                                let total_size = header_size + packed.len() * 4;
                                let out_end = out_start + total_size;
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
                            if input_slices.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_kernel: expected exactly one input slice".into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            if params.len() < 4 {
                                return Err(BackendError::Dispatch(
                                    "dequantize_kernel: expected numel, format, bit width, and channel count".into(),
                                ));
                            }
                            let numel = params[0];
                            let format_flag = params[1];
                            let bit_width = params[2];
                            if format_flag > 2 || !matches!(bit_width, 4 | 8) {
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
                                    let mut buf = Vec::new();
                                    buf.try_reserve_exact(input_slice.size).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "dequantize_kernel: input materialization failed: {error}"
                                        ))
                                    })?;
                                    buf.extend_from_slice(
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
                                ) = if matches!(format_flag, 1 | 2) {
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
                                    let mut scales = Vec::new();
                                    scales.try_reserve_exact(num_channels).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "dequantize_kernel: scale metadata allocation failed: {error}"
                                        ))
                                    })?;
                                    for j in 0..num_channels {
                                        let bits = params[4 + j];
                                        scales.push(f32::from_bits(bits as u32));
                                    }
                                    let mut zero_points = Vec::new();
                                    zero_points.try_reserve_exact(num_channels).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "dequantize_kernel: offset metadata allocation failed: {error}"
                                        ))
                                    })?;
                                    for j in 0..num_channels {
                                        let bits = params[4 + num_channels + j];
                                        zero_points.push(f32::from_bits(bits as u32));
                                    }
                                    // The packed data starts at offset 0 (no header)
                                    let data_offset = 0;
                                    // Infer num_elems_per_channel from numel
                                    if !numel.is_multiple_of(num_channels) {
                                        return Err(BackendError::Dispatch(
                                            "dequantize_kernel: numel is not divisible by channel count"
                                                .into(),
                                        ));
                                    }
                                    let num_elems_per_channel = numel / num_channels;

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
                                    let mut scales = Vec::new();
                                    scales.try_reserve_exact(num_channels).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "dequantize_kernel: scale metadata allocation failed: {error}"
                                        ))
                                    })?;
                                    for _ in 0..num_channels {
                                        let bytes = &in_data[hdr_offset..hdr_offset + 4];
                                        scales.push(f32::from_le_bytes([
                                            bytes[0], bytes[1], bytes[2], bytes[3],
                                        ]));
                                        hdr_offset += 4;
                                    }
                                    let mut zero_points = Vec::new();
                                    zero_points.try_reserve_exact(num_channels).map_err(|error| {
                                        BackendError::Dispatch(format!(
                                            "dequantize_kernel: offset metadata allocation failed: {error}"
                                        ))
                                    })?;
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
                                let packed_words_required = if format_flag == 2 {
                                    num_elems_per_channel
                                        .div_ceil(items_per_word)
                                        .checked_mul(num_channels)
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "dequantize_kernel: row-packed word count overflows"
                                                    .into(),
                                            )
                                        })?
                                } else {
                                    numel
                                        .checked_add(items_per_word - 1)
                                        .map(|value| value / items_per_word)
                                        .ok_or_else(|| {
                                            BackendError::Dispatch(
                                                "dequantize_kernel: packed word count overflows"
                                                    .into(),
                                            )
                                        })?
                                };
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
                                let expected_input_end = if format_flag == 2 {
                                    packed_end.checked_add(64).ok_or_else(|| {
                                        BackendError::Dispatch(
                                            "dequantize_kernel: SIMD margin range overflows".into(),
                                        )
                                    })?
                                } else {
                                    packed_end
                                };
                                if expected_input_end != in_data.len() {
                                    return Err(BackendError::Dispatch(
                                        "dequantize_kernel: packed payload size is not exact"
                                            .into(),
                                    ));
                                }

                                // Write output
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..numel {
                                    let ch = i / num_elems_per_channel;
                                    let within_channel = i % num_elems_per_channel;
                                    let word_idx = if format_flag == 2 {
                                        ch * num_elems_per_channel.div_ceil(items_per_word)
                                            + within_channel / items_per_word
                                    } else {
                                        i / items_per_word
                                    };
                                    let shift = if format_flag == 2 {
                                        (within_channel % items_per_word) * bit_width
                                    } else {
                                        (i % items_per_word) * bit_width
                                    };
                                    let word_start = data_offset + word_idx * 4;
                                    let word = u32::from_le_bytes([
                                        in_data[word_start],
                                        in_data[word_start + 1],
                                        in_data[word_start + 2],
                                        in_data[word_start + 3],
                                    ]);
                                    let q = ((word >> shift) & ((1 << bit_width) - 1)) as i32;
                                    let sign_bit = 1 << (bit_width - 1);
                                    let q_signed = if (q & sign_bit) != 0 {
                                        q | (!((1 << bit_width) - 1))
                                    } else {
                                        q
                                    };
                                    out_f32[i] = q_signed as f32 * scales[ch] + zero_points[ch];
                                }
                            }
                        }
                        "to_f16" => {
                            if input_slices.len() != 1 || params.len() != 1 {
                                return Err(BackendError::Dispatch(
                                    "to_f16: expected one f32 input and declared element count"
                                        .into(),
                                ));
                            }
                            let input_slice = input_slices[0];
                            let declared_numel = params[0];
                            let output_size = out_end - out_start;
                            if !input_slice
                                .offset
                                .is_multiple_of(std::mem::align_of::<f32>())
                                || !input_slice.size.is_multiple_of(std::mem::size_of::<f32>())
                                || !out_start.is_multiple_of(std::mem::align_of::<u16>())
                                || !output_size.is_multiple_of(std::mem::size_of::<u16>())
                                || (declared_numel == 0
                                    && (input_slice.size != 0 || output_size != 0))
                            {
                                return Err(BackendError::Dispatch(
                                    "to_f16: invalid scalar alignment or storage contract".into(),
                                ));
                            }
                            let effective_numel = declared_numel
                                .min(input_slice.size / std::mem::size_of::<f32>())
                                .min(output_size / std::mem::size_of::<u16>());
                            let source =
                                unsafe { arena.view_f32(input_slice.offset, input_slice.size) };
                            let mut f32_data = Vec::new();
                            f32_data
                                .try_reserve_exact(effective_numel)
                                .map_err(|error| {
                                    BackendError::Dispatch(format!(
                                        "to_f16: input materialization failed: {error}"
                                    ))
                                })?;
                            f32_data.extend_from_slice(&source[..effective_numel]);
                            let out_bytes =
                                unsafe { arena.view_u8_mut(out_start, out_end - out_start) };
                            for (index, value) in f32_data.iter().enumerate() {
                                let start = index * std::mem::size_of::<u16>();
                                out_bytes[start..start + 2]
                                    .copy_from_slice(&half::f16::from_f32(*value).to_le_bytes());
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
                    let src_end = src.offset.checked_add(src.size).ok_or_else(|| {
                        BackendError::Dispatch("memcopy source range overflows".into())
                    })?;
                    data.copy_within(src.offset..src_end, dst.offset);
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
                    let end = dst.offset.checked_add(data.len()).ok_or_else(|| {
                        BackendError::Dispatch("constant destination range overflows".into())
                    })?;
                    arena_data[dst.offset..end].copy_from_slice(data);
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
