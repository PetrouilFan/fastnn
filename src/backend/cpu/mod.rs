#![allow(dead_code)]

use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::passes::memory_planning::MemoryPlan;
use crate::dtypes::{U4x8, U8x4};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, ShapeEnv, TensorValue};
use crate::backend::cpu::blas::matmul_blas_into;
use crate::packed_tensor::PackedTensor;
use bytemuck;
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;

pub mod blas;
pub mod im2col;
pub mod microkernels;
pub mod reductions_fast;

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
                Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div
                | Opcode::Maximum | Opcode::Minimum => {
                    let mut kernel = match node.opcode {
                        Opcode::Add => "add_f32",
                        Opcode::Sub => "sub_f32",
                        Opcode::Mul => "mul_f32",
                        Opcode::Div => "div_f32",
                        Opcode::Maximum => "max_f32",
                        Opcode::Minimum => "min_f32",
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
                | Opcode::Abs
                | Opcode::LeakyRelu
                | Opcode::Elu
                | Opcode::Softplus
                | Opcode::Hardswish
                | Opcode::Clamp
                | Opcode::Sign
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
                        Opcode::LogicalNot => "logical_not_f32",
                        Opcode::LogSoftmax => "log_softmax_f32",
                        Opcode::Mish => "mish_f32",
                        _ => unreachable!(),
                    };
                    let mut extra_params: Vec<usize> = Vec::new();
                    if let Opcode::LeakyRelu = node.opcode {
                        let slope: f32 = node.attrs.get("negative_slope")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.01);
                        extra_params.push(slope.to_bits() as usize);
                    }
                    if let Opcode::Clamp = node.opcode {
                        let min: f32 = node.attrs.get("min")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0.0);
                        let max: f32 = node.attrs.get("max")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(1.0);
                        extra_params.push(min.to_bits() as usize);
                        extra_params.push(max.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel.to_string(),
                        input_slices,
                        output_slice,
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
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    let dilation: usize = node.attrs.get("dilation").and_then(|d| d.parse().ok()).unwrap_or(1);
                    let groups: usize = node.attrs.get("groups").and_then(|g| g.parse().ok()).unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "conv2d".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![stride, padding, dilation, groups],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    let eps = node.attrs.get("eps").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1e-5);
                    let is_batch_norm = if matches!(node.opcode, Opcode::BatchNorm) { 1 } else { 0 };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "norm_f32".to_string(),
                        input_slices,
                        output_slice,
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
                    // Using the dimension size rather than the axis index avoids
                    // needing shape information at dispatch time.
                    let axis_dim = input_shapes
                        .first()
                        .and_then(|s| s.get(normalized_axis).copied())
                        .unwrap_or(1) as usize;
                    let axis_dim_dim = input_shape_dims
                        .first()
                        .and_then(|s| s.get(normalized_axis).cloned())
                        .unwrap_or(DimExpr::Known(1));

                    instructions.push(Instruction::CallKernel {
                        kernel_name: "softmax".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![axis_dim],
                        param_dims: Some(vec![axis_dim_dim]),
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
                Opcode::Concat => {
                    let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "concat".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![axis],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MaxPool | Opcode::AvgPool => {
                    let kernel_size: usize = node.attrs.get("kernel_size").and_then(|k| k.parse().ok()).unwrap_or(2);
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(2);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    let is_max = if matches!(node.opcode, Opcode::MaxPool) { 1 } else { 0 };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "pool_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![kernel_size, stride, padding, is_max],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pad => {
                    let pads_str = node.attrs.get("pads").cloned().unwrap_or_default();
                    let pads: Vec<usize> = pads_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "pad_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: pads,
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Gather => {
                    let axis: usize = node.attrs.get("axis").and_then(|a| a.parse().ok()).unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "gather".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![axis],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Slice => {
                    let dim: usize = node.attrs.get("dim").and_then(|d| d.parse().ok()).unwrap_or(0);
                    let start: usize = node.attrs.get("start").and_then(|s| s.parse().ok()).unwrap_or(0);
                    let end: usize = node.attrs.get("end").and_then(|e| e.parse().ok()).unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "slice_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![dim, start, end],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ScatterNd => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "scatter_nd".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax => {
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
                Opcode::Conv1d | Opcode::Conv3d | Opcode::ConvTranspose2d => {
                    let kernel_name = match node.opcode {
                        Opcode::Conv1d => "conv1d",
                        Opcode::Conv3d => "conv3d",
                        Opcode::ConvTranspose2d => "conv_transpose2d",
                        _ => unreachable!(),
                    };
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        params: vec![stride, padding],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Prelu => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "prelu".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::RMSNorm => {
                    let eps = node.attrs.get("eps").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1e-5);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "rms_norm".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![eps.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Embedding => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "embedding".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Pow => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "pow_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::GtScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "gt_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::LtScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "lt_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::EqScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "eq_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AddScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "add_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MulScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "mul_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::DivScalar => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "div_scalar_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                // Input nodes have no producer instruction — data is written
                // by the executor before dispatch.
                Opcode::Input => {
                    // No instruction needed.
                }
                Opcode::ArgMax => {
                    let axis: i64 = node.attrs.get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(-1);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "argmax".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![axis as usize],
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
                        // --- New activation kernels for PPO/control/RL (AOT pipeline) ---
                        "leaky_relu_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let slope = if !params.is_empty() { f32::from_bits(params[0] as u32) } else { 0.01 };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = if input[i] > 0.0 { input[i] } else { input[i] * slope };
                                }
                            }
                        }
                        "elu_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = if input[i] > 0.0 { input[i] } else { input[i].exp() - 1.0 };
                                }
                            }
                        }
                        "softplus_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = (1.0 + input[i].exp()).ln();
                                }
                            }
                        }
                        "hardswish_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let v = input[i];
                                    out_f32[i] = v * (v + 3.0).max(0.0).min(6.0) / 6.0;
                                }
                            }
                        }
                        "clamp_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let min_val = if params.len() > 0 { f32::from_bits(params[0] as u32) } else { 0.0 };
                                let max_val = if params.len() > 1 { f32::from_bits(params[1] as u32) } else { 1.0 };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].max(min_val).min(max_val);
                                }
                            }
                        }
                        "sign_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = input[i].signum();
                                }
                            }
                        }
                        "logical_not_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    out_f32[i] = if input[i] == 0.0 { 1.0 } else { 0.0 };
                                }
                            }
                        }
                        "log_softmax_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                if !input.is_empty() {
                                    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                    let mut sum = 0.0f32;
                                    for i in 0..input.len() {
                                        sum += (input[i] - max_val).exp();
                                    }
                                    let log_sum = sum.ln();
                                    for i in 0..out_f32.len().min(input.len()) {
                                        out_f32[i] = (input[i] - max_val) - log_sum;
                                    }
                                }
                            }
                        }
                        "mish_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let x = input[i];
                                    let sp = (1.0 + x.exp()).ln();
                                    out_f32[i] = x * sp.tanh();
                                }
                            }
                        }
                        "max_f32" => {
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
                                for i in 0..len {
                                    out_f32[i] = a[i].max(b[i]);
                                }
                            }
                        }
                        "min_f32" => {
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
                                for i in 0..len {
                                    out_f32[i] = a[i].min(b[i]);
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
                                // Use BLAS for large matrices, scalar triple-loop for small ones
                                let threshold = 32; // cross-over where BLAS wins over O(n³) scalar
                                if m * _k * n >= threshold {
                                    matmul_blas_into(&a, &b, out_f32, m as usize, _k as usize, n as usize);
                                } else {
                                    for i in 0..m {
                                        for j in 0..n {
                                            let mut sum = 0.0f32;
                                            for kk in 0.._k {
                                                sum += a[(i * _k + kk) as usize] * b[(kk * n + j) as usize];
                                            }
                                            out_f32[(i * n + j) as usize] = sum;
                                        }
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
                                let _num_words = weights.len();
                                let u4x8_data: Vec<U4x8> = bytemuck::cast_slice(&weights).to_vec();
                                let pt = PackedTensor::from_raw(u4x8_data, w_shape.clone(), vec![scale], vec![zp]);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Slice activations and outputs per batch item
                                let batch_inputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| activations[i * k..(i + 1) * k].to_vec())
                                    .collect();
                                let mut batch_outputs: Vec<Vec<f32>> = (0..m)
                                    .map(|_i| vec![0.0f32; n])
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
                                let batch_inputs: Vec<Vec<f32>> = (0..m)
                                    .map(|i| activations[i * k..(i + 1) * k].to_vec())
                                    .collect();
                                let mut batch_outputs: Vec<Vec<f32>> = (0..m)
                                    .map(|_i| vec![0.0f32; n])
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
                                // Resolve the axis-dimension size from params
                                // (compile-time) or param_dims (runtime symbolic).
                                // The axis dim size tells us how many elements form
                                // a single softmax "row" along the reduction axis.
                                let softmax_params = resolve_params(params, param_dims, shape_env, 1)
                                    .unwrap_or_else(|_| vec![input.len()]);
                                let axis_dim_size = softmax_params[0].max(1);
                                let num_rows = input.len() / axis_dim_size;
                                for r in 0..num_rows {
                                    let start = r * axis_dim_size;
                                    let end = (start + axis_dim_size).min(input.len());
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
                            let &[eps_bits, is_batch_norm] = &params[..] else { return Err(BackendError::Dispatch("norm_f32: expected params [eps_bits, is_batch_norm]".into())); };
                            let eps = f32::from_bits(eps_bits as u32);
                            if is_batch_norm == 1 {
                                // Batch norm (evaluation mode): use running_mean and running_var
                                if let [data_slice, weight_slice, bias_slice, mean_slice, var_slice] = &input_slices[..] {
                                    let (data, weight, bias, running_mean, running_var) = {
                                        let d = arena.data_mut();
                                        (
                                            bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                            bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                            bytemuck::cast_slice::<_, f32>(&d[bias_slice.offset..bias_slice.offset + bias_slice.size]).to_vec(),
                                            bytemuck::cast_slice::<_, f32>(&d[mean_slice.offset..mean_slice.offset + mean_slice.size]).to_vec(),
                                            bytemuck::cast_slice::<_, f32>(&d[var_slice.offset..var_slice.offset + var_slice.size]).to_vec(),
                                        )
                                    };
                                    let out_f32 = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                    };
                                    let c = weight.len();
                                    for i in 0..out_f32.len().min(data.len()) {
                                        let ch = i % c;
                                        out_f32[i] = (data[i] - running_mean[ch]) / (running_var[ch] + eps).sqrt() * weight[ch] + bias[ch];
                                    }
                                }
                            } else {
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
                                    // Layer norm over the last dimension
                                    let row_size = input.len() / out_f32.len().max(1);
                                    let num_rows = if row_size > 0 { input.len() / row_size } else { 1 };
                                    for r in 0..num_rows {
                                        let start = r * row_size;
                                        let end = (start + row_size).min(input.len());
                                        let n = (end - start) as f32;
                                        let mean: f32 = input[start..end].iter().sum::<f32>() / n;
                                        let var: f32 = input[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                                        let inv_std = 1.0 / (var + eps).sqrt();
                                        for i in start..end {
                                            out_f32[i] = (input[i] - mean) * inv_std;
                                        }
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
                        "conv2d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input_data, weight_data) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let bias_data = if input_slices.len() >= 3 {
                                    let b_slice = &input_slices[2];
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec()
                                } else { vec![] };
                                let &[stride, padding, dilation, groups] = &params[..] else { return Err(BackendError::Dispatch("conv2d: expected params [stride, padding, dilation, groups]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let ni = input_data.len();
                                let nw = weight_data.len();
                                let no = out_f32.len();
                                let dims = (|| -> Option<(usize,usize,usize,usize,usize,usize,usize)> {
                                    for &n in &[1, 2, 4, 8, 16, 32, 64] {
                                        if ni % n != 0 || no % n != 0 { continue; }
                                        let c_hw = ni / n;
                                        let f_hout_wout = no / n;
                                        for cg in 1..=c_hw.min(4096) {
                                            let c = cg * groups;
                                            if c_hw % c != 0 { continue; }
                                            let hw = c_hw / c;
                                            if nw % cg != 0 { continue; }
                                            let f_kh_kw = nw / cg;
                                            for f in 1..=f_hout_wout.min(f_kh_kw) {
                                                if f_hout_wout % f != 0 || f_kh_kw % f != 0 { continue; }
                                                let hout_wout = f_hout_wout / f;
                                                let kh_kw = f_kh_kw / f;
                                                for &kh in &[1, 3, 5, 7, 11] {
                                                    if kh > kh_kw || kh_kw % kh != 0 { continue; }
                                                    let kw = kh_kw / kh;
                                                    if kw > 11 { continue; }
                                                    for h in 1..=hw {
                                                        if hw % h != 0 { continue; }
                                                        let w = hw / h;
                                                        let h_out = (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
                                                        let w_out = (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;
                                                        if h_out * w_out == hout_wout && h_out > 0 && w_out > 0 {
                                                            return Some((n, f, c, kh, kw, h, w));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    None
                                })();
                                let (n, f, c, kh, kw, h, w) = dims.ok_or_else(|| BackendError::Dispatch("conv2d: could not infer dimensions".into()))?;
                                let c_per_group = c / groups;
                                let h_out = (h + 2 * padding).saturating_sub(dilation * (kh - 1) + 1) / stride + 1;
                                let w_out = (w + 2 * padding).saturating_sub(dilation * (kw - 1) + 1) / stride + 1;
                                for nn in 0..n {
                                    for ff in 0..f {
                                        let g = ff / (f / groups.max(1));
                                        for hh in 0..h_out {
                                            for ww in 0..w_out {
                                                let mut sum = 0.0f32;
                                                for cc in 0..c_per_group {
                                                    for kkh in 0..kh {
                                                        for kkw in 0..kw {
                                                            let h_in = hh * stride + kkh;
                                                            let w_in = ww * stride + kkw;
                                                            if h_in >= padding && w_in >= padding {
                                                                let h_in_s = h_in - padding;
                                                                let w_in_s = w_in - padding;
                                                                if h_in_s < h && w_in_s < w {
                                                                    let input_idx = nn * (c * h * w) + (g * c_per_group + cc) * (h * w) + h_in_s * w + w_in_s;
                                                                    let weight_idx = ff * c_per_group * kh * kw + cc * kh * kw + kkh * kw + kkw;
                                                                    if input_idx < input_data.len() && weight_idx < weight_data.len() {
                                                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if !bias_data.is_empty() {
                                                    sum += bias_data[ff % bias_data.len()];
                                                }
                                                let out_idx = nn * (f * h_out * w_out) + ff * (h_out * w_out) + hh * w_out + ww;
                                                if out_idx < out_f32.len() {
                                                    out_f32[out_idx] = sum;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "concat" => {
                            if !input_slices.is_empty() {
                                let mut output_offset = 0;
                                for slice in input_slices {
                                    let input_data = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice::<_, f32>(&d[slice.offset..slice.offset + slice.size]).to_vec()
                                    };
                                    let out_f32 = {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                    };
                                    let end = (output_offset + input_data.len()).min(out_f32.len());
                                    out_f32[output_offset..end].copy_from_slice(&input_data[..end - output_offset]);
                                    output_offset += input_data.len();
                                }
                            }
                        }
                        "pool_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let &[kernel, stride_val, padding_val, is_max] = &params[..] else { return Err(BackendError::Dispatch("pool_f32: expected params [kernel, stride, padding, is_max]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Infer N, C, H, W from sizes
                                let total_input = input.len();
                                let total_output = out_f32.len();
                                let dims = (|| -> Option<(usize,usize,usize,usize)> {
                                    for &n in &[1, 2, 4, 8, 16, 32, 64] {
                                        if total_input % n != 0 || total_output % n != 0 { continue; }
                                        let c_hw = total_input / n;
                                        let c_hout_wout = total_output / n;
                                        for c in 1..=c_hw.min(4096) {
                                            if c_hw % c != 0 || c_hout_wout % c != 0 { continue; }
                                            let hw = c_hw / c;
                                            let hout_wout = c_hout_wout / c;
                                            if hout_wout == 0 { continue; }
                                            for h in 1..=hw {
                                                if hw % h != 0 { continue; }
                                                let w = hw / h;
                                                let h_out = (h + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                                let w_out = (w + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                                if h_out * w_out == hout_wout && h_out > 0 && w_out > 0 {
                                                    return Some((n, c, h, w));
                                                }
                                            }
                                        }
                                    }
                                    None
                                })();
                                let (n, c, h, w) = dims.ok_or_else(|| BackendError::Dispatch("pool_f32: could not infer dimensions".into()))?;
                                let h_out = (h + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                let w_out = (w + 2 * padding_val).saturating_sub(kernel) / stride_val + 1;
                                for nn in 0..n {
                                    for cc in 0..c {
                                        for hh in 0..h_out {
                                            for ww in 0..w_out {
                                                let mut val = if is_max == 1 { f32::NEG_INFINITY } else { 0.0f32 };
                                                let mut count = 0usize;
                                                for kh in 0..kernel {
                                                    for kw in 0..kernel {
                                                        let h_in = hh * stride_val + kh;
                                                        let w_in = ww * stride_val + kw;
                                                        if h_in >= padding_val && w_in >= padding_val {
                                                            let h_in_s = h_in - padding_val;
                                                            let w_in_s = w_in - padding_val;
                                                            if h_in_s < h && w_in_s < w {
                                                                let idx = nn * (c * h * w) + cc * (h * w) + h_in_s * w + w_in_s;
                                                                if idx < input.len() {
                                                                    if is_max == 1 { val = val.max(input[idx]); } else { val += input[idx]; }
                                                                    count += 1;
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                                if is_max == 0 && count > 0 { val /= count as f32; }
                                                let out_idx = nn * (c * h_out * w_out) + cc * (h_out * w_out) + hh * w_out + ww;
                                                if out_idx < out_f32.len() {
                                                    out_f32[out_idx] = val;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "pad_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Simple flat copy: the caller right-sizes the output buffer.
                                // Output includes padding zeros already allocated.
                                let end = input.len().min(out_f32.len());
                                out_f32[..end].copy_from_slice(&input[..end]);
                            }
                        }
                        "gather" => {
                            if let [data_slice, indices_slice] = &input_slices[..] {
                                let (input, indices) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[indices_slice.offset..indices_slice.offset + indices_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let axis = if !params.is_empty() { params[0] } else { 0 };
                                let inner = if axis == 0 { input.len() / out_f32.len().max(1) } else { 1 };
                                for i in 0..out_f32.len() {
                                    let idx_idx = if inner > 0 { i / inner } else { i };
                                    let idx = indices[idx_idx.min(indices.len().saturating_sub(1))] as usize;
                                    let src = idx * inner + (i % inner);
                                    out_f32[i] = if src < input.len() { input[src] } else { 0.0 };
                                }
                            }
                        }
                        "slice_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let &[dim, start, _end] = &params[..] else { return Err(BackendError::Dispatch("slice_f32: expected params [dim, start, end]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let batch_stride = if dim == 0 { input.len() / out_f32.len().max(1) } else { 1 };
                                let offset = start * batch_stride;
                                for i in 0..out_f32.len().min(input.len().saturating_sub(offset)) {
                                    out_f32[i] = input[offset + i];
                                }
                            }
                        }
                        "scatter_nd" => {
                            if let [data_slice, _indices_slice, updates_slice] = &input_slices[..] {
                                let (data, updates) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[updates_slice.offset..updates_slice.offset + updates_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                out_f32.copy_from_slice(&data);
                                for i in 0..out_f32.len().min(updates.len()) {
                                    out_f32[i] = updates[i];
                                }
                            }
                        }
                        "conv1d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let &[stride, padding] = &params[..] else { return Err(BackendError::Dispatch("conv1d: expected params [stride, padding]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let ni = input.len();
                                let nw = weight.len();
                                let no = out_f32.len();
                                let dims = (|| -> Option<(usize,usize,usize,usize,usize)> {
                                    for &n in &[1, 2, 4, 8, 16, 32, 64] {
                                        if ni % n != 0 || no % n != 0 { continue; }
                                        let c_w = ni / n;
                                        let f_w_out = no / n;
                                        for c in 1..=c_w.min(4096) {
                                            if c_w % c != 0 { continue; }
                                            let w = c_w / c;
                                            if nw % c != 0 { continue; }
                                            let f_kw = nw / c;
                                            for f in 1..=f_w_out.min(f_kw) {
                                                if f_w_out % f != 0 || f_kw % f != 0 { continue; }
                                                let w_out = f_w_out / f;
                                                let kw = f_kw / f;
                                                if kw > 11 { continue; }
                                                let w_out_calc = (w + 2 * padding).saturating_sub(kw) / stride + 1;
                                                if w_out_calc == w_out && w_out > 0 {
                                                    return Some((n, f, c, kw, w));
                                                }
                                            }
                                        }
                                    }
                                    None
                                })();
                                let (n, f, c, kw, w) = dims.ok_or_else(|| BackendError::Dispatch("conv1d: could not infer dimensions".into()))?;
                                let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
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
                                                            sum += input[nn * (c * w) + cc * w + w_in_s]
                                                                * weight[ff * (c * kw) + cc * kw + kkw];
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
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let &[stride, padding] = &params[..] else { return Err(BackendError::Dispatch("conv3d: expected params [stride, padding]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let ni = input.len();
                                let nw = weight.len();
                                let no = out_f32.len();
                                let dims = (|| -> Option<(usize,usize,usize,usize,usize,usize,usize,usize,usize)> {
                                    for &n in &[1, 2, 4, 8, 16] {
                                        if ni % n != 0 || no % n != 0 { continue; }
                                        let c_dhw = ni / n;
                                        let f_dout_hout_wout = no / n;
                                        for c in 1..=c_dhw.min(4096) {
                                            if c_dhw % c != 0 { continue; }
                                            let dhw = c_dhw / c;
                                            if nw % c != 0 { continue; }
                                            let f_kd_kh_kw = nw / c;
                                            for f in 1..=f_dout_hout_wout.min(f_kd_kh_kw) {
                                                if f_dout_hout_wout % f != 0 || f_kd_kh_kw % f != 0 { continue; }
                                                let dout_hout_wout = f_dout_hout_wout / f;
                                                let kd_kh_kw = f_kd_kh_kw / f;
                                                for &kd in &[1, 3, 5] {
                                                    if kd > kd_kh_kw || kd_kh_kw % kd != 0 { continue; }
                                                    let kh_kw = kd_kh_kw / kd;
                                                    for &kh in &[1, 3, 5] {
                                                        if kh > kh_kw || kh_kw % kh != 0 { continue; }
                                                        let kw = kh_kw / kh;
                                                        if kw > 5 { continue; }
                                                        for d in 1..=dhw {
                                                            if dhw % d != 0 { continue; }
                                                            let hw = dhw / d;
                                                            for h in 1..=hw {
                                                                if hw % h != 0 { continue; }
                                                                let w = hw / h;
                                                                let d_out = (d + 2 * padding).saturating_sub(kd) / stride + 1;
                                                                let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
                                                                let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
                                                                if d_out * h_out * w_out == dout_hout_wout && d_out > 0 && h_out > 0 && w_out > 0 {
                                                                    return Some((n, f, c, kd, kh, kw, d, h, w));
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    None
                                })();
                                let (n, f, c, kd, kh, kw, d, h, w) = dims.ok_or_else(|| BackendError::Dispatch("conv3d: could not infer dimensions".into()))?;
                                let d_out = (d + 2 * padding).saturating_sub(kd) / stride + 1;
                                let h_out = (h + 2 * padding).saturating_sub(kh) / stride + 1;
                                let w_out = (w + 2 * padding).saturating_sub(kw) / stride + 1;
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
                                                                    let d_in = dd * stride + kkd;
                                                                    let h_in = hh * stride + kkh;
                                                                    let w_in = ww * stride + kkw;
                                                                    if d_in >= padding && h_in >= padding && w_in >= padding {
                                                                        let d_in_s = d_in - padding;
                                                                        let h_in_s = h_in - padding;
                                                                        let w_in_s = w_in - padding;
                                                                        if d_in_s < d && h_in_s < h && w_in_s < w {
                                                                            let input_idx = nn * (c * d * h * w) + cc * (d * h * w) + d_in_s * (h * w) + h_in_s * w + w_in_s;
                                                                            let weight_idx = ff * (c * kd * kh * kw) + cc * (kd * kh * kw) + kkd * (kh * kw) + kkh * kw + kkw;
                                                                            sum += input[input_idx] * weight[weight_idx];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                    let out_idx = nn * (f * d_out * h_out * w_out) + ff * (d_out * h_out * w_out) + dd * (h_out * w_out) + hh * w_out + ww;
                                                    out_f32[out_idx] = sum;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "conv_transpose2d" => {
                            if let [input_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let &[stride, padding] = &params[..] else { return Err(BackendError::Dispatch("conv_transpose2d: expected params [stride, padding]".into())); };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let ni = input.len();
                                let nw = weight.len();
                                let no = out_f32.len();
                                let dims = (|| -> Option<(usize,usize,usize,usize,usize,usize,usize)> {
                                    for &n in &[1, 2, 4, 8] {
                                        if ni % n != 0 || no % n != 0 { continue; }
                                        let c_hin_win = ni / n;
                                        let f_hout_wout = no / n;
                                        for c in 1..=c_hin_win.min(4096) {
                                            if c_hin_win % c != 0 { continue; }
                                            let hin_win = c_hin_win / c;
                                            if nw % c != 0 { continue; }
                                            let f_kh_kw = nw / c;
                                            for f in 1..=f_hout_wout.min(f_kh_kw) {
                                                if f_hout_wout % f != 0 || f_kh_kw % f != 0 { continue; }
                                                let hout_wout = f_hout_wout / f;
                                                let kh_kw = f_kh_kw / f;
                                                for &kh in &[1, 3, 5, 7] {
                                                    if kh > kh_kw || kh_kw % kh != 0 { continue; }
                                                    let kw = kh_kw / kh;
                                                    if kw > 7 { continue; }
                                                    for hin in 1..=hin_win {
                                                        if hin_win % hin != 0 { continue; }
                                                        let win = hin_win / hin;
                                                        let h_out = (hin - 1) * stride + kh - 2 * padding;
                                                        let w_out = (win - 1) * stride + kw - 2 * padding;
                                                        if h_out * w_out == hout_wout && h_out > 0 && w_out > 0 {
                                                            return Some((n, f, c, kh, kw, hin, win));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    None
                                })();
                                let (n, f, c, kh, kw, hin, win) = dims.ok_or_else(|| BackendError::Dispatch("conv_transpose2d: could not infer dimensions".into()))?;
                                let h_out = (hin - 1) * stride + kh - 2 * padding;
                                let w_out = (win - 1) * stride + kw - 2 * padding;
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
                                                            if h_out_idx >= padding && w_out_idx >= padding {
                                                                let h_out_s = h_out_idx - padding;
                                                                let w_out_s = w_out_idx - padding;
                                                                if h_out_s < h_out && w_out_s < w_out {
                                                                    let out_idx = nn * (f * h_out * w_out) + ff * (h_out * w_out) + h_out_s * w_out + w_out_s;
                                                                    let input_idx = nn * (c * hin * win) + cc * (hin * win) + hh * win + ww;
                                                                    let weight_idx = cc * (f * kh * kw) + ff * (kh * kw) + kkh * kw + kkw;
                                                                    if out_idx < out_f32.len() && input_idx < input.len() && weight_idx < weight.len() {
                                                                        out_f32[out_idx] += input[input_idx] * weight[weight_idx];
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
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let channel_stride = if !weight.is_empty() && input.len() > weight.len() { input.len() / weight.len() } else { 1 };
                                for i in 0..out_f32.len().min(input.len()) {
                                    let w_idx = if weight.len() == 1 { 0 } else { (i / channel_stride) % weight.len() };
                                    let slope = if w_idx < weight.len() { weight[w_idx] } else { 0.0 };
                                    out_f32[i] = if input[i] > 0.0 { input[i] } else { input[i] * slope };
                                }
                            }
                        }
                        "rms_norm" => {
                            if let [data_slice, weight_slice] = &input_slices[..] {
                                let (input, weight) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                    )
                                };
                                let eps = f32::from_bits(params.first().copied().unwrap_or(0) as u32);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let row_size = if !weight.is_empty() { input.len() / weight.len() } else { input.len() };
                                let num_rows = if row_size > 0 { input.len() / row_size } else { 1 };
                                for r in 0..num_rows {
                                    let start = r * row_size;
                                    let end = (start + row_size).min(input.len());
                                    let mut sq_sum = 0.0f32;
                                    for i in start..end { sq_sum += input[i] * input[i]; }
                                    let rms = (sq_sum / (end - start) as f32 + eps).sqrt();
                                    for i in start..end {
                                        let w = if i - start < weight.len() { weight[i - start] } else { 1.0 };
                                        out_f32[i] = input[i] / rms * w;
                                    }
                                }
                            }
                        }
                        "embedding" => {
                            if let [weight_slice, indices_slice] = &input_slices[..] {
                                let (weight, indices) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[weight_slice.offset..weight_slice.offset + weight_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[indices_slice.offset..indices_slice.offset + indices_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let dim = if !weight.is_empty() && !indices.is_empty() { out_f32.len() / indices.len() } else { 1 };
                                for i in 0..indices.len() {
                                    let idx = indices[i] as usize;
                                    let src_start = idx * dim;
                                    let dst_start = i * dim;
                                    let len = dim.min(weight.len().saturating_sub(src_start)).min(out_f32.len().saturating_sub(dst_start));
                                    if len > 0 {
                                        out_f32[dst_start..dst_start + len].copy_from_slice(&weight[src_start..src_start + len]);
                                    }
                                }
                            }
                        }
                        "pow_f32" => {
                            if let [data_slice, exp_slice] = &input_slices[..] {
                                let (data, exponent) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[exp_slice.offset..exp_slice.offset + exp_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..out_f32.len().min(data.len()) {
                                    let e = if i < exponent.len() { exponent[i] } else { exponent[exponent.len().saturating_sub(1)] };
                                    out_f32[i] = data[i].powf(e);
                                }
                            }
                        }
                        "gt_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = if data[i] > s { 1.0 } else { 0.0 };
                                }
                            }
                        }
                        "lt_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = if data[i] < s { 1.0 } else { 0.0 };
                                }
                            }
                        }
                        "eq_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = if (data[i] - s).abs() < 1e-6 { 1.0 } else { 0.0 };
                                }
                            }
                        }
                        "add_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = data[i] + s;
                                }
                            }
                        }
                        "mul_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = data[i] * s;
                                }
                            }
                        }
                        "div_scalar_f32" => {
                            if let [data_slice, scalar_slice] = &input_slices[..] {
                                let (data, scalar) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[data_slice.offset..data_slice.offset + data_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[scalar_slice.offset..scalar_slice.offset + scalar_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let s = scalar.first().copied().unwrap_or(0.0);
                                for i in 0..out_f32.len().min(data.len()) {
                                    out_f32[i] = data[i] / s;
                                }
                            }
                        }
                        "argmax" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let axis = params.first().copied().unwrap_or(usize::MAX);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, u64>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                if axis == usize::MAX {
                                    let max_idx = input.iter()
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                        .map(|(i, _)| i as u64)
                                        .unwrap_or(0);
                                    for v in out_f32.iter_mut() {
                                        *v = max_idx;
                                    }
                                } else {
                                    let dim_size = axis;
                                    if dim_size > 0 && input.len() % dim_size == 0 {
                                        let num_rows = input.len() / dim_size;
                                        for row in 0..num_rows {
                                            let start = row * dim_size;
                                            let end = start + dim_size;
                                            let max_idx = input[start..end].iter()
                                                .enumerate()
                                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                                .map(|(i, _)| (row * dim_size + i) as u64)
                                                .unwrap_or(0);
                                            if row < out_f32.len() {
                                                out_f32[row] = max_idx;
                                            }
                                        }
                                    } else {
                                        let max_idx = input.iter()
                                            .enumerate()
                                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                            .map(|(i, _)| i as u64)
                                            .unwrap_or(0);
                                        for v in out_f32.iter_mut() {
                                            *v = max_idx;
                                        }
                                    }
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
