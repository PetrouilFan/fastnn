#![allow(dead_code)]

use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::passes::memory_planning::MemoryPlan;
use crate::dtypes::{U4x8, U8x4};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, ShapeEnv, TensorValue};
use crate::packed_tensor::PackedTensor;
use bytemuck;
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

pub mod microkernels;

/// Resolve kernel dimension params at dispatch time using the runtime shape
/// environment. Returns `params` unchanged if no symbolic dims are present.
///
/// Resolve symbolic parameters to concrete values using the runtime ShapeEnv.
///
/// Returns `Err(BackendError::Dispatch)` when `param_dims` is malformed
/// (backend lowering bug) or a symbolic dimension cannot be resolved.
fn resolve_params(
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    expected: usize,
) -> Result<Vec<usize>, BackendError> {
    if let Some(dims) = param_dims {
        if dims.len() < expected {
            return Err(BackendError::Dispatch(format!(
                "resolve_params: param_dims has {} elements but expected {} (params={:?})",
                dims.len(),
                expected,
                params
            )));
        }
        dims[..expected]
            .iter()
            .map(|d| {
                d.evaluate_with_env(shape_env)
                    .map(|v| v as usize)
                    .map_err(|e| BackendError::Dispatch(format!("resolve_params: {e}")))
            })
            .collect()
    } else {
        Ok(params.to_vec())
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
    /// This is satisfied by the sequential dispatch loop — each
    /// `data_mut` call's borrow ends before the next one begins.
    pub fn data_mut(&self) -> &mut [u8] {
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }
}

// SAFETY: `Vec<u8>` is `Send + Sync`.  The arena is never accessed
// concurrently — dispatch is single-threaded — so interior mutability
// via `UnsafeCell` does not introduce data races.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

/// CPU execution context. Zero allocation during dispatch.
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buffer = CpuBuffer;

    fn name(&self) -> &str {
        "cpu"
    }

    fn allocate_arena(&self, total_bytes: usize) -> CpuBuffer {
        CpuBuffer::new(vec![0u8; total_bytes])
    }

    fn compile(
        &self,
        graph: &ComputeGraph,
        memory_plan: &MemoryPlan,
    ) -> Result<ExecutablePlan, BackendError> {
        let mut instructions = Vec::new();
        let order = graph.topological_sort();

        for &node_id in &order {
            let node = graph
                .get_node(node_id)
                .ok_or_else(|| BackendError::Compilation(format!("Node {} not found", node_id)))?;

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
            let symbol_max = crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed);
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

            match &node.opcode {
                Opcode::Constant(val) => {
                    match val {
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
                    }
                }
                Opcode::MatMul => {
                    // Detect quantized dtypes from input nodes to select the right kernel
                    let input_dtypes: Vec<_> = node
                        .inputs
                        .iter()
                        .filter_map(|&input_id| graph.get_node(input_id))
                        .map(|n| n.output_type.dtype.clone())
                        .collect();
                    let is_quantized = input_dtypes.iter().any(|d| matches!(d, IrDType::U4 { .. } | IrDType::U8 { .. }));

                    let fused_type = node.attrs.get("fused_op").map(|s| s.as_str());
                    let kernel_name = match (fused_type, is_quantized) {
                        (Some("MatMulAddRelu"), _) => "fused_matmul_add_relu",
                        (Some("OpRelu"), false) => "matmul_relu",
                        (_, true) if input_dtypes.iter().any(|d| matches!(d, IrDType::U4 { .. })) => "matmul_u4",
                        (_, true) => "matmul_u8",
                        _ => "matmul",
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
                    // Extract weight metadata for quantized matmul kernels
                    let weight_meta = if is_quantized {
                        node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (scale, zp) = match &wn.output_type.dtype {
                                    IrDType::U4 { scale, zero_point } => (*scale, *zero_point as f32),
                                    IrDType::U8 { scale, zero_point } => (*scale, *zero_point as f32),
                                    _ => (1.0, 0.0),
                                };
                                let w_shape: Vec<usize> = wn.output_type.shape.iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                (scale, zp, w_shape)
                            })
                        })
                    } else {
                        None
                    };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        params: vec![m, k, n],
                        param_dims: Some(vec![m_dim, k_dim, n_dim]),
                        weight_meta,
                    });
                }
                Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                    let mut kernel = match node.opcode {
                        Opcode::Add => "add_f32",
                        Opcode::Sub => "sub_f32",
                        Opcode::Mul => "mul_f32",
                        Opcode::Div => "div_f32",
                        _ => unreachable!(),
                    };
                    // Op+Relu fusion: if fused_op is "OpRelu", use the fused kernel name
                    if node.attrs.get("fused_op").map(|s| s.as_str()) == Some("OpRelu") {
                        kernel = match node.opcode {
                            Opcode::Add => "add_relu_f32",
                            Opcode::Sub => "sub_relu_f32",
                            Opcode::Mul => "mul_relu_f32",
                            Opcode::Div => "div_relu_f32",
                            _ => unreachable!(),
                        };
                    }
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
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
                | Opcode::Abs => {
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
                        _ => unreachable!(),
                    };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
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
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "conv2d".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "norm_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Softmax => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "softmax".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::BiasAdd => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "biasadd".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ReduceSum | Opcode::ReduceMean => {
                    // Group size = product of dims being reduced over.
                    // For a single-axis reduce this is just input_shape[axis],
                    // which is typically Known (e.g. reduce over dim 1 of [N,4]
                    // has group_size=Known(4)).
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let group_size_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(axis).cloned())
                        .unwrap_or(DimExpr::Known(1));
                    let is_mean = if matches!(node.opcode, Opcode::ReduceMean) { 1 } else { 0 };
                    let group_size = group_size_dim.evaluate().unwrap_or_else(|| {
                        // Symbolic dim — use SYMBOL_DIM_MAX as compile-time
                        // estimate; runtime resolves via param_dims.
                        crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed)
                    }) as usize;
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "reduce_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![group_size, is_mean],
                        // Pass the group_size as a symbolic DimExpr so dispatch
                        // can re-evaluate it when shape_env is available (e.g.
                        // reduce over symbolic batch dim N).
                        param_dims: Some(vec![group_size_dim]),
                        weight_meta: None,
                    });
                }
                Opcode::Transpose => {
                    // Extract M, N from the input shape (assume 2D)
                    let m = input_shapes
                        .first()
                        .and_then(|s| s.first().copied())
                        .unwrap_or(1) as usize;
                    let n = input_shapes
                        .first()
                        .and_then(|s| s.get(1).copied())
                        .unwrap_or(1) as usize;
                    let m_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.first().cloned())
                        .unwrap_or(DimExpr::Known(m as u64));
                    let n_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(1).cloned())
                        .unwrap_or(DimExpr::Known(n as u64));
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "transpose_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![m, n],
                        param_dims: Some(vec![m_dim, n_dim]),
                        weight_meta: None,
                    });
                }
                // Input nodes have no producer instruction — data is written
                // by the executor before dispatch.
                Opcode::Input => {
                    // No instruction needed.
                }
                _ => {
                    if let Some(&input_id) = node.inputs.first() {
                        if let Some(in_slot) = memory_plan.slots.get(&input_id) {
                            instructions.push(Instruction::MemCopy {
                                dst: output_slice,
                                src: BufferSlice::new(in_slot.offset, in_slot.size),
                            });
                        }
                    }
                }
            }
        }

        Ok(ExecutablePlan {
            instructions,
            arena_size: memory_plan.total_size,
        })
    }

    fn dispatch(
        &self,
        plan: &ExecutablePlan,
        arena: &CpuBuffer,
        shape_env: &ShapeEnv,
    ) -> Result<(), BackendError> {
        for instr in &plan.instructions {
            match instr {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
                    param_dims,
                    weight_meta,
                } => {
                    let out_start = output_slice.offset;
                    let out_end = output_slice.offset + output_slice.size;

                    match kernel_name.as_str() {
                        "add_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32)
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let len = out_f32.len().min(a.len()).min(b.len());
                                for i in 0..len { out_f32[i] = a[i] + b[i]; }
                            }
                        }
                        "sub_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32)
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let len = out_f32.len().min(a.len()).min(b.len());
                                for i in 0..len { out_f32[i] = a[i] - b[i]; }
                            }
                        }
                        "mul_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32)
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(a.len()).min(b.len()) {
                                    out_f32[i] = a[i] * b[i];
                                }
                            }
                        }
                        "div_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32)
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(a.len()).min(b.len()) {
                                    out_f32[i] = a[i] / b[i];
                                }
                            }
                        }
                        "relu_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].max(0.0);
                                }
                            }
                        }
                        "gelu_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let x = input[i];
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    out_f32[i] = 0.5 * x * (1.0 + t);
                                }
                            }
                        }
                        "silu_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let x = input[i];
                                    out_f32[i] = x / (1.0 + (-x).exp());
                                }
                            }
                        }
                        "exp_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].exp();
                                }
                            }
                        }
                        "log_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].ln();
                                }
                            }
                        }
                        "sqrt_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].sqrt();
                                }
                            }
                        }
                        "neg_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = -input[i];
                                }
                            }
                        }
                        "abs_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].abs();
                                }
                            }
                        }
                        "sigmoid_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = 1.0 / (1.0 + (-input[i]).exp());
                                }
                            }
                        }
                        "tanh_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].tanh();
                                }
                            }
                        }
                        "matmul" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32)
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, _k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul: expected params [M,K,N]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                // Simple triple-loop matmul: C[M,N] = A[M,K] @ B[K,N]
                                for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for kk in 0.._k {
                                            sum += a[i * _k + kk] * b[kk * n + j];
                                        }
                                        out_f32[i * n + j] = sum;
                                    }
                                }
                            }
                        }
                        "fused_matmul_add_relu" => {
                            if let [a_slice, b_slice, bias_slice] = &input_slices[..] {
                                let (a, b, bias) = {
                                    let d = arena.data_mut();
                                    let a_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[a_slice.offset..a_slice.offset + a_slice.size]
                                    ).to_vec();
                                    let b_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[b_slice.offset..b_slice.offset + b_slice.size]
                                    ).to_vec();
                                    let bias_f32 = bytemuck::cast_slice::<_, f32>(
                                        &d[bias_slice.offset..bias_slice.offset + bias_slice.size]
                                    ).to_vec();
                                    (a_f32, b_f32, bias_f32)
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, _k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("fused_matmul_add_relu: expected params [M,K,N]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for kk in 0.._k {
                                            sum += a[i * _k + kk] * b[kk * n + j];
                                        }
                                        sum += if j < bias.len() { bias[j] } else { 0.0 };
                                        out_f32[i * n + j] = sum.max(0.0); // relu
                                    }
                                }
                            }
                        }
                        "matmul_relu" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec())
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, _k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul_relu: expected params [M,K,N]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for kk in 0.._k {
                                            sum += a[i * _k + kk] * b[kk * n + j];
                                        }
                                        out_f32[i * n + j] = sum.max(0.0); // matmul + relu
                                    }
                                }
                            }
                        }
                        "matmul_u4" => {
                            if let [w_slice, a_slice] = &input_slices[..] {
                                let (weights, activations) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, u32>(&d[w_slice.offset..w_slice.offset + w_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec())
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul_u4: expected params [M,K,N]".into())); };
                                let (scale, zp, w_shape) = weight_meta.clone().unwrap_or((1.0, 0.0, vec![m, k]));
                                // Construct PackedTensor from arena data and call SIMD gemm
                                let num_words = weights.len();
                                let u4x8_data: Vec<U4x8> = bytemuck::cast_slice(&weights).to_vec();
                                let pt = PackedTensor::from_raw(u4x8_data, w_shape.clone(), vec![scale], vec![zp]);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Slice activations and outputs per batch item
                                let mut batch_inputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| activations[i * k..(i + 1) * k].to_vec())
                                    .collect();
                                let mut batch_outputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| vec![0.0f32; n])
                                    .collect();
                                crate::backend::cpu::microkernels::gemm_cpu::<U4x8>(&pt, &batch_inputs, &mut batch_outputs);
                                for i in 0..m {
                                    out_f32[i * n..(i + 1) * n].copy_from_slice(&batch_outputs[i]);
                                }
                            }
                        }
                        "matmul_u8" => {
                            if let [w_slice, a_slice] = &input_slices[..] {
                                let (weights, activations) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, u32>(&d[w_slice.offset..w_slice.offset + w_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec())
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul_u8: expected params [M,K,N]".into())); };
                                let (scale, zp, w_shape) = weight_meta.clone().unwrap_or((1.0, 0.0, vec![m, k]));
                                // Construct PackedTensor from arena data and call SIMD gemm
                                let u8x4_data: Vec<U8x4> = bytemuck::cast_slice(&weights).to_vec();
                                let pt = PackedTensor::from_raw(u8x4_data, w_shape.clone(), vec![scale], vec![zp]);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let mut batch_inputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| activations[i * k..(i + 1) * k].to_vec())
                                    .collect();
                                let mut batch_outputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| vec![0.0f32; n])
                                    .collect();
                                crate::backend::cpu::microkernels::gemm_cpu::<U8x4>(&pt, &batch_inputs, &mut batch_outputs);
                                for i in 0..m {
                                    out_f32[i * n..(i + 1) * n].copy_from_slice(&batch_outputs[i]);
                                }
                            }
                        }
                        "reduce_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let &[group_size, is_mean] = &params[..] else { return Err(BackendError::Dispatch("reduce_f32: expected params [group_size, is_mean]".into())); };
                                // Resolve actual group size from param_dims using shape_env.
                                // param_dims[0] is the reduced-axis dim (e.g., Symbol("N") when
                                // reducing over the batch dimension).
                                let effective_group_size = match param_dims {
                                    Some(dims) if dims.len() >= 1 => {
                                        dims[0]
                                            .evaluate_with_env(shape_env)
                                            .map_err(|e| BackendError::Dispatch(format!("reduce_f32: {e}")))?
                                            as usize
                                    }
                                    _ => group_size,
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let num_groups = out_f32.len();
                                for g in 0..num_groups {
                                    let mut sum = 0.0f32;
                                    let start = g * effective_group_size;
                                    let end = (start + effective_group_size).min(input.len());
                                    for i in start..end {
                                        sum += input[i];
                                    }
                                    if is_mean == 1 {
                                        sum /= effective_group_size as f32;
                                    }
                                    out_f32[g] = sum;
                                }
                            }
                        }
                        "transpose_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let transpose_params = resolve_params(params, param_dims, shape_env, 2)?;
                                let &[m, n] = &transpose_params[..] else { return Err(BackendError::Dispatch("transpose_f32: expected params [M,N]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                // 2D transpose: output[j * m + i] = input[i * n + j]
                                for i in 0..m {
                                    for j in 0..n {
                                        out_f32[j * m + i] = input[i * n + j];
                                    }
                                }
                            }
                        }
                        // ── Fused Op+Relu kernels ─────────────────────────────
                        "add_relu_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec())
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let len = out_f32.len().min(a.len()).min(b.len());
                                for i in 0..len { out_f32[i] = (a[i] + b[i]).max(0.0); }
                            }
                        }
                        "sub_relu_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec())
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let len = out_f32.len().min(a.len()).min(b.len());
                                for i in 0..len { out_f32[i] = (a[i] - b[i]).max(0.0); }
                            }
                        }
                        "mul_relu_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec())
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(a.len()).min(b.len()) {
                                    out_f32[i] = (a[i] * b[i]).max(0.0);
                                }
                            }
                        }
                        "softmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Softmax over the last dimension: for each row,
                                // compute exp(x_i - max) / sum(exp(x_j - max))
                                let row_size = input.len() / out_f32.len().max(1);
                                let num_rows = if row_size > 0 { input.len() / row_size } else { 1 };
                                for r in 0..num_rows {
                                    let start = r * row_size;
                                    let end = (start + row_size).min(input.len());
                                    let max_val = input[start..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let mut sum = 0.0f32;
                                    for i in start..end {
                                        let e = (input[i] - max_val).exp();
                                        out_f32[i] = e;
                                        sum += e;
                                    }
                                    if sum > 0.0 {
                                        for i in start..end {
                                            out_f32[i] /= sum;
                                        }
                                    }
                                }
                            }
                        }
                        "biasadd" => {
                            if let [data_slice, bias_slice] = &input_slices[..] {
                                let (data, bias) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[bias_slice.offset..bias_slice.offset + bias_slice.size]).to_vec())
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = data[i] + bias[i % bias.len()];
                                }
                            }
                        }
                        "norm_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Simple layer norm over the last dimension
                                let row_size = input.len() / out_f32.len().max(1);
                                let num_rows = if row_size > 0 { input.len() / row_size } else { 1 };
                                for r in 0..num_rows {
                                    let start = r * row_size;
                                    let end = (start + row_size).min(input.len());
                                    let n = (end - start) as f32;
                                    let mean: f32 = input[start..end].iter().sum::<f32>() / n;
                                    let var: f32 = input[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                                    let inv_std = 1.0 / (var + 1e-5).sqrt();
                                    for i in start..end {
                                        out_f32[i] = (input[i] - mean) * inv_std;
                                    }
                                }
                            }
                        }
                        "div_relu_f32" => {
                            if let [a_slice, b_slice] = &input_slices[..] {
                                let (a, b) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec())
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(a.len()).min(b.len()) {
                                    out_f32[i] = (a[i] / b[i]).max(0.0);
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
        }

        Ok(())
    }

    fn write_arena(&self, arena: &CpuBuffer, offset: usize, data: &[u8]) {
        let buf = arena.data_mut();
        let end = (offset + data.len()).min(buf.len());
        buf[offset..end].copy_from_slice(&data[..end - offset]);
    }

    fn read_arena(&self, arena: &CpuBuffer, offset: usize, size: usize) -> Vec<u8> {
        let buf = arena.data_mut();
        let end = (offset + size).min(buf.len());
        buf[offset..end].to_vec()
    }
}
