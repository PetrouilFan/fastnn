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

/// Minimum number of elements for a parallel dispatch loop to be beneficial.
/// Below this threshold, sequential execution avoids rayon's task-spawning
/// overhead without measurable throughput loss.
#[cfg(feature = "parallel")]
const PARALLEL_MIN_ELEMS: usize = 1024;

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
    #[allow(clippy::mut_from_ref)]
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
            let secondary_output_slice = memory_plan
                .secondary_slots
                .get(&(node_id, 1))
                .map(|slot| BufferSlice::new(slot.offset, slot.size));

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
                    // Extract weight metadata for quantized matmul kernels.
                    // For 2D weights, the quantized data is stored transposed
                    // ([N, K] instead of [K, N]) so the shape must match.
                    let weight_meta = if is_quantized {
                        node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bit_width, scales, zero_points) = match &wn.output_type.dtype {
                                    IrDType::U4 { scales, zero_points } => (4usize, scales.clone(), zero_points.clone()),
                                    IrDType::U8 { scales, zero_points } => (8usize, scales.clone(), zero_points.clone()),
                                    _ => (0usize, vec![], vec![]),
                                };
                                let mut w_shape: Vec<usize> = wn.output_type.shape.iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                // The quantization pass transposes 2D weights from
                                // [K, N] to [N, K] for gemm_packed_batched convention.
                                if w_shape.len() == 2 {
                                    w_shape.reverse();
                                }
                                crate::backend::QuantizedWeightMeta { bit_width, scales, zero_points, shape: w_shape }
                            })
                        })
                    } else {
                        None
                    };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
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
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    let dilation: usize = node.attrs.get("dilation").and_then(|d| d.parse().ok()).unwrap_or(1);
                    let groups: usize = node.attrs.get("groups").and_then(|g| g.parse().ok()).unwrap_or(1);
                    // Detect quantized weights for packed conv2d dispatch
                    let weight_dtype = node.inputs.get(1)
                        .and_then(|&w_id| graph.get_node(w_id))
                        .map(|wn| wn.output_type.dtype.clone());
                    let is_quantized = weight_dtype.as_ref().is_some_and(|d| matches!(d, IrDType::U4 { .. } | IrDType::U8 { .. }));
                    let (kernel_name, weight_meta) = if is_quantized {
                        let dtype = weight_dtype.as_ref().unwrap();
                        let (kernel, bit_width) = if matches!(dtype, IrDType::U4 { .. }) {
                            ("conv2d_u4", 4usize)
                        } else {
                            ("conv2d_u8", 8usize)
                        };
                        let meta = node.inputs.get(1).and_then(|&w_id| {
                            graph.get_node(w_id).map(|wn| {
                                let (bw, scales, zero_points) = match &wn.output_type.dtype {
                                    IrDType::U4 { scales, zero_points } => (4usize, scales.clone(), zero_points.clone()),
                                    IrDType::U8 { scales, zero_points } => (8usize, scales.clone(), zero_points.clone()),
                                    _ => (bit_width, vec![], vec![]),
                                };
                                let w_shape: Vec<usize> = wn.output_type.shape.iter()
                                    .map(|d| d.evaluate().unwrap_or(symbol_max) as usize)
                                    .collect();
                                crate::backend::QuantizedWeightMeta { bit_width: bw, scales, zero_points, shape: w_shape }
                            })
                        });
                        (kernel.to_string(), meta)
                    } else {
                        ("conv2d".to_string(), None)
                    };
                    // Extract spatial dims from input shapes to avoid
                    // ambiguous dim inference at dispatch time.
                    let input_c = input_shapes.first().and_then(|s| s.get(1).copied()).unwrap_or(1) as usize;
                    let input_h = input_shapes.first().and_then(|s| s.get(2).copied()).unwrap_or(0) as usize;
                    let input_w = input_shapes.first().and_then(|s| s.get(3).copied()).unwrap_or(0) as usize;
                    let kernel_h = input_shapes.get(1).and_then(|s| s.get(2).copied()).unwrap_or(0) as usize;
                    let kernel_w = input_shapes.get(1).and_then(|s| s.get(3).copied()).unwrap_or(0) as usize;
                    instructions.push(Instruction::CallKernel {
                        kernel_name,
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        // params: [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]
                        params: vec![stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w],
                        param_dims: None,
                        weight_meta,
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    let eps = node.attrs.get("eps").and_then(|s| s.parse::<f32>().ok()).unwrap_or(1e-5);
                    let is_batch_norm = if matches!(node.opcode, Opcode::BatchNorm) { 1 } else { 0 };
                    instructions.push(Instruction::CallKernel {
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
                        .map(|s| s[normalized_axis + 1..].iter().copied().map(|x| x as usize).product::<usize>().max(1))
                        .unwrap_or(1);

                    instructions.push(Instruction::CallKernel {
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
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "biasadd".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
                        params: vec![axis],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::MaxPool => {
                    let kernel_size: usize = node.attrs.get("kernel_size").and_then(|k| k.parse().ok()).unwrap_or(2);
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(2);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    let secondary_output_slice = memory_plan.secondary_slots.get(&(node_id, 1)).map(|slot| {
                        BufferSlice::new(slot.offset, slot.size)
                    });
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "pool_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        params: vec![kernel_size, stride, padding, 1],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AvgPool => {
                    let kernel_size: usize = node.attrs.get("kernel_size").and_then(|k| k.parse().ok()).unwrap_or(2);
                    let stride: usize = node.attrs.get("stride").and_then(|s| s.parse().ok()).unwrap_or(2);
                    let padding: usize = node.attrs.get("padding").and_then(|p| p.parse().ok()).unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "pool_f32".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![kernel_size, stride, padding, 0],
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
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
                        secondary_output_slice: None,
                        params: vec![axis as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::UpsampleNearest2d | Opcode::UpsampleBilinear2d => {
                    let scale_h: usize = node.attrs.get("scale_h").and_then(|s| s.parse().ok()).unwrap_or(2);
                    let scale_w: usize = node.attrs.get("scale_w").and_then(|s| s.parse().ok()).unwrap_or(2);
                    let kernel_name = match node.opcode {
                        Opcode::UpsampleNearest2d => "upsample_nearest2d",
                        _ => "upsample_bilinear2d",
                    };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![scale_h, scale_w],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdaptiveAvgPool2d => {
                    let out_h: usize = node.attrs.get("output_h").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let out_w: usize = node.attrs.get("output_w").and_then(|s| s.parse().ok()).unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
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
                    let repeats: Vec<usize> = repeats_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    instructions.push(Instruction::CallKernel {
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
                    let dim: usize = node.attrs.get("dim").and_then(|d| d.parse().ok()).unwrap_or(0);
                    let exclusive: usize = node.attrs.get("exclusive").and_then(|e| e.parse().ok()).unwrap_or(0);
                    let rev: usize = node.attrs.get("reverse").and_then(|r| r.parse().ok()).unwrap_or(0);
                    instructions.push(Instruction::CallKernel {
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
                    let dims: Vec<usize> = dims_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                    let input_shape: Vec<u64> = input_shapes.first().cloned().unwrap_or_default();
                    let mut params = vec![dims.len()];
                    params.extend_from_slice(&dims);
                    params.extend(input_shape.iter().map(|&s| s as usize));
                    instructions.push(Instruction::CallKernel {
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
                    let k: usize = node.attrs.get("k").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let axis: i64 = node.attrs.get("axis").and_then(|s| s.parse().ok()).unwrap_or(-1);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "topk_fused".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice,
                        params: vec![k, axis as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                // ── Optimizer ops ──────────────────────────────────
                Opcode::SgdUpdate => {
                    let lr: f32 = node.attrs.get("lr").and_then(|s| s.parse().ok()).unwrap_or(0.01);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "sgd_update_f32".to_string(),
                        input_slices,  // [weight, grad] — weight must be same slot as output
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![lr.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamUpdate => {
                    let lr: f32 = node.attrs.get("lr").and_then(|s| s.parse().ok()).unwrap_or(0.001);
                    let beta1: f32 = node.attrs.get("beta1").and_then(|s| s.parse().ok()).unwrap_or(0.9);
                    let beta2: f32 = node.attrs.get("beta2").and_then(|s| s.parse().ok()).unwrap_or(0.999);
                    let eps: f32 = node.attrs.get("eps").and_then(|s| s.parse().ok()).unwrap_or(1e-8);
                    let t: u64 = node.attrs.get("t").and_then(|s| s.parse().ok()).unwrap_or(1);
                    // Detect F16 state tensors (m and v at inputs[2] and inputs[3]).
                    let has_f16_state = node.inputs.len() >= 4 &&
                        graph.get_node(node.inputs[2]).map(|n| n.output_type.dtype == IrDType::F16).unwrap_or(false) &&
                        graph.get_node(node.inputs[3]).map(|n| n.output_type.dtype == IrDType::F16).unwrap_or(false);
                    let kernel_name = if has_f16_state { "adam_update_f16_state" } else { "adam_update_f32" };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,  // [weight, grad, m, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![lr.to_bits() as usize, beta1.to_bits() as usize, beta2.to_bits() as usize, eps.to_bits() as usize, t as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::AdamWUpdate => {
                    let lr: f32 = node.attrs.get("lr").and_then(|s| s.parse().ok()).unwrap_or(0.001);
                    let beta1: f32 = node.attrs.get("beta1").and_then(|s| s.parse().ok()).unwrap_or(0.9);
                    let beta2: f32 = node.attrs.get("beta2").and_then(|s| s.parse().ok()).unwrap_or(0.999);
                    let eps: f32 = node.attrs.get("eps").and_then(|s| s.parse().ok()).unwrap_or(1e-8);
                    let t: u64 = node.attrs.get("t").and_then(|s| s.parse().ok()).unwrap_or(1);
                    let wd: f32 = node.attrs.get("weight_decay").and_then(|s| s.parse().ok()).unwrap_or(0.01);
                    let has_f16_state = node.inputs.len() >= 4 &&
                        graph.get_node(node.inputs[2]).map(|n| n.output_type.dtype == IrDType::F16).unwrap_or(false) &&
                        graph.get_node(node.inputs[3]).map(|n| n.output_type.dtype == IrDType::F16).unwrap_or(false);
                    let kernel_name = if has_f16_state { "adamw_update_f16_state" } else { "adamw_update_f32" };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,  // [weight, grad, m, v]
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![lr.to_bits() as usize, beta1.to_bits() as usize, beta2.to_bits() as usize, eps.to_bits() as usize, t as usize, wd.to_bits() as usize],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Shape => {
                    // Write the shape of the input tensor as I64 values.
                    // Resolve input shape at compile time (known dims directly,
                    // symbolic dims use SYMBOL_DIM_MAX — they'll be resolved
                    // at dispatch by param_dims).
                    use std::io::Write;
                    let in_shape = input_shapes.first().cloned().unwrap_or_default();
                    let mut shape_bytes = Vec::with_capacity(in_shape.len() * 8);
                    for &d in &in_shape {
                        shape_bytes.write_all(&(d as i64).to_le_bytes()).unwrap();
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
                    let in_type = node.inputs.first()
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
                        let in_slot = node.inputs.first()
                            .and_then(|id| memory_plan.slots.get(id));
                        let input_slices = in_slot
                            .map(|s| vec![BufferSlice::new(s.offset, s.size)])
                            .unwrap_or_default();
                        instructions.push(Instruction::CallKernel {
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
                    let bit_width: usize = node.attrs.get("bit_width")
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(4);
                    let kernel_name = match bit_width {
                        4 => "quantize_f32_u4",
                        8 => "quantize_f32_u8",
                        _ => return Err(BackendError::Compilation(
                            format!("Quantize: unsupported bit_width={bit_width}")
                        )),
                    };
                    // num_channels = dim 0 (rows), num_elems_per_channel = product of rest
                    let num_channels = input_shapes.first()
                        .and_then(|s| s.first().copied())
                        .unwrap_or(1) as usize;
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    let num_elems_per_channel = if num_channels > 0 { numel / num_channels } else { numel };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![num_channels, num_elems_per_channel, numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::Dequantize => {
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    // Check if the input has per-channel scale metadata (from
                    // quantized Constants or Quantize ops). These are passed as
                    // additional params so the dequantize kernel can reconstruct
                    // f32 values without relying on an inline header.
                    let (scales, zero_points) = node.inputs.first()
                        .and_then(|&input_id| graph.get_node(input_id))
                        .map(|n| match &n.output_type.dtype {
                            IrDType::U4 { scales, zero_points } => {
                                (scales.clone(), zero_points.clone())
                            }
                            IrDType::U8 { scales, zero_points } => {
                                (scales.clone(), zero_points.clone())
                            }
                            _ => (vec![], vec![]),
                        })
                        .unwrap_or_default();
                    let has_metadata = !scales.is_empty() && !zero_points.is_empty();
                    let format_flag: usize = if has_metadata { 1 } else { 0 }; // 0=header, 1=metadata
                    // Flatten scales and zero_points into params (f32 bits as usize)
                    let mut params = vec![numel, format_flag];
                    let num_channels = scales.len();
                    params.push(num_channels);
                    for &s in &scales {
                        params.push(s.to_bits() as usize);
                    }
                    for &zp in &zero_points {
                        params.push(zp.to_bits() as usize);
                    }
                    instructions.push(Instruction::CallKernel {
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
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
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
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
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
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    // Output buffer: [scale(f32)][zp(f32)][i8_data(numel bytes)]
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "quantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::DequantizeActivations => {
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "dequantize_activations".to_string(),
                        input_slices,
                        output_slice,
                        secondary_output_slice: None,
                        params: vec![numel],
                        param_dims: None,
                        weight_meta: None,
                    });
                }
                Opcode::GradientScale => {
                    let scale: f32 = node.attrs.get("scale")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1.0);
                    let numel = input_shapes.first()
                        .map(|s| s.iter().product::<u64>() as usize)
                        .unwrap_or(1);
                    if input_slices.first().is_some() {
                        instructions.push(Instruction::CallKernel {
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
                    // Range(start, limit, step) — produce 1D F32 tensor.
                    // All 3 inputs are scalars (4 bytes each).
                    let input_slices: Vec<BufferSlice> = node.inputs.iter()
                        .filter_map(|id| memory_plan.slots.get(id))
                        .map(|slot| BufferSlice::new(slot.offset, slot.size))
                        .collect();
                    instructions.push(Instruction::CallKernel {
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
                    secondary_output_slice,
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i] + b[i]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] + b[i]);
                                }
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i] - b[i]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] - b[i]);
                                }
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
                                let len = out_f32.len().min(a.len()).min(b.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i] * b[i]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] * b[i]);
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
                                let len = out_f32.len().min(a.len()).min(b.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i] / b[i]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i] / b[i]);
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].max(0.0); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].max(0.0));
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len {
                                    let x = input[i];
                                    let x3 = x * x * x;
                                    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                    let t = tanh_arg.tanh();
                                    out_f32[i] = 0.5 * x * (1.0 + t);
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                        let x = input[i];
                                        let x3 = x * x * x;
                                        let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
                                        let t = tanh_arg.tanh();
                                        *o = 0.5 * x * (1.0 + t);
                                    });
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len {
                                    let x = input[i];
                                    out_f32[i] = x / (1.0 + (-x).exp());
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                        let x = input[i];
                                        *o = x / (1.0 + (-x).exp());
                                    });
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].exp(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].exp());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].ln(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].ln());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].sqrt(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].sqrt());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = -input[i]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = -input[i]);
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].abs(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].abs());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = 1.0 / (1.0 + (-input[i]).exp()); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = 1.0 / (1.0 + (-input[i]).exp()));
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].tanh(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].tanh());
                                }
                            }
                        }
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if input[i] > 0.0 { input[i] } else { input[i] * slope }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if input[i] > 0.0 { input[i] } else { input[i] * slope });
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if input[i] > 0.0 { input[i] } else { input[i].exp() - 1.0 }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if input[i] > 0.0 { input[i] } else { input[i].exp() - 1.0 });
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = (1.0 + input[i].exp()).ln(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (1.0 + input[i].exp()).ln());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len {
                                    let v = input[i];
                                    out_f32[i] = v * (v + 3.0).clamp(0.0, 6.0) / 6.0;
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                        let v = input[i];
                                        *o = v * (v + 3.0).clamp(0.0, 6.0) / 6.0;
                                    });
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
                                let min_val = if !params.is_empty() { f32::from_bits(params[0] as u32) } else { 0.0 };
                                let max_val = if params.len() > 1 { f32::from_bits(params[1] as u32) } else { 1.0 };
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].max(min_val).min(max_val); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].max(min_val).min(max_val));
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = input[i].signum(); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = input[i].signum());
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if input[i] == 0.0 { 1.0 } else { 0.0 }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if input[i] == 0.0 { 1.0 } else { 0.0 });
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
                                    let len = out_f32.len().min(input.len());
                                    #[cfg(not(feature = "parallel"))]
                                    { for i in 0..len { out_f32[i] = (input[i] - max_val) - log_sum; } }
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (input[i] - max_val) - log_sum);
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
                                let len = out_f32.len().min(input.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len {
                                    let x = input[i];
                                    let sp = (1.0 + x.exp()).ln();
                                    out_f32[i] = x * sp.tanh();
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                        let x = input[i];
                                        let sp = (1.0 + x.exp()).ln();
                                        *o = x * sp.tanh();
                                    });
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i].max(b[i]); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i].max(b[i]));
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = a[i].min(b[i]); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = a[i].min(b[i]));
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
                                // Use BLAS for large matrices, parallel scalar for medium, sequential scalar for tiny.
                                // The parallel scalar loop (rayon par_chunks_mut) competes with BLAS for medium
                                // sizes; sequential scalar is only used for the tiniest matmuls.
                                #[cfg(not(feature = "parallel"))]
                                let threshold = 32;
                                #[cfg(feature = "parallel")]
                                let threshold = 16384;
                                if m * _k * n >= threshold {
                                    matmul_blas_into(&a, &b, out_f32, m, _k, n);
                                } else {
                                    #[cfg(not(feature = "parallel"))]
                                    {
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
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        out_f32.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                                            for j in 0..n {
                                                let mut sum = 0.0f32;
                                                for kk in 0.._k {
                                                    sum += a[i * _k + kk] * b[kk * n + j];
                                                }
                                                row[j] = sum;
                                            }
                                        });
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for kk in 0.._k {
                                            sum += a[i * _k + kk] * b[kk * n + j];
                                        }
                                        sum += if j < bias.len() { bias[j] } else { 0.0 };
                                        out_f32[i * n + j] = sum.max(0.0);
                                    }
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                                        for j in 0..n {
                                            let mut sum = 0.0f32;
                                            for kk in 0.._k {
                                                sum += a[i * _k + kk] * b[kk * n + j];
                                            }
                                            sum += if j < bias.len() { bias[j] } else { 0.0 };
                                            row[j] = sum.max(0.0);
                                        }
                                    });
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for kk in 0.._k {
                                            sum += a[i * _k + kk] * b[kk * n + j];
                                        }
                                        out_f32[i * n + j] = sum.max(0.0);
                                    }
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                                        for j in 0..n {
                                            let mut sum = 0.0f32;
                                            for kk in 0.._k {
                                                sum += a[i * _k + kk] * b[kk * n + j];
                                            }
                                            row[j] = sum.max(0.0);
                                        }
                                    });
                                }
                            }
                        }
                        "matmul_u4" => {
                            // input_slices: [activation (f32), weight (packed U4)]
                            if let [a_slice, w_slice] = &input_slices[..] {
                                let (activations, packed_bytes) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     {
                                        let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                                        // Copy to u32-aligned buffer (arena may not be u32-aligned)
                                        let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                                        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                                        byte_slice[..raw.len()].copy_from_slice(raw);
                                        aligned
                                    })
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul_u4: expected params [M,K,N]".into())); };
                                let meta = weight_meta.clone().unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
                                    bit_width: 4, scales: vec![1.0], zero_points: vec![0.0], shape: vec![m, k]
                                });
                                // Construct PackedTensor from arena data and call SIMD gemm
                                let u4x8_data: Vec<U4x8> = bytemuck::cast_slice(&packed_bytes).to_vec();
                                let pt = PackedTensor::from_raw(u4x8_data, meta.shape.clone(), meta.scales, meta.zero_points);
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
                            // input_slices: [activation (f32), weight (packed U8)]
                            if let [a_slice, w_slice] = &input_slices[..] {
                                let (activations, packed_bytes) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     {
                                        let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                                        // Copy to u32-aligned buffer (arena may not be u32-aligned)
                                        let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                                        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                                        byte_slice[..raw.len()].copy_from_slice(raw);
                                        aligned
                                    })
                                };
                                let matmul_params = resolve_params(params, param_dims, shape_env, 3)?;
                                let &[m, k, n] = &matmul_params[..] else { return Err(BackendError::Dispatch("matmul_u8: expected params [M,K,N]".into())); };
                                let meta = weight_meta.clone().unwrap_or_else(|| crate::backend::QuantizedWeightMeta {
                                    bit_width: 8, scales: vec![1.0], zero_points: vec![0.0], shape: vec![m, k]
                                });
                                // Construct PackedTensor from arena data and call SIMD gemm
                                let u8x4_data: Vec<U8x4> = bytemuck::cast_slice(&packed_bytes).to_vec();
                                let pt = PackedTensor::from_raw(u8x4_data, meta.shape.clone(), meta.scales, meta.zero_points);
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
                                let effective_group_size = match param_dims {
                                    Some(dims) if !dims.is_empty() => {
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
                                #[cfg(not(feature = "parallel"))]
                                { for g in 0..num_groups {
                                    let mut sum = 0.0f32;
                                    let start = g * effective_group_size;
                                    let end = (start + effective_group_size).min(input.len());
                                    for i in start..end {
                                        sum += input[i];
                                    }
                                    if is_mean == 1 { sum /= effective_group_size as f32; }
                                    out_f32[g] = sum;
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32.par_iter_mut().enumerate().for_each(|(g, out)| {
                                        let mut sum = 0.0f32;
                                        let start = g * effective_group_size;
                                        let end = (start + effective_group_size).min(input.len());
                                        for i in start..end { sum += input[i]; }
                                        if is_mean == 1 { sum /= effective_group_size as f32; }
                                        *out = sum;
                                    });
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..m {
                                    for j in 0..n {
                                        out_f32[j * m + i] = input[i * n + j];
                                    }
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32.par_chunks_mut(m).enumerate().for_each(|(j, col)| {
                                        for i in 0..m {
                                            col[i] = input[i * n + j];
                                        }
                                    });
                                }
                            }
                        }
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = (a[i] + b[i]).max(0.0); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (a[i] + b[i]).max(0.0));
                                }
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
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = (a[i] - b[i]).max(0.0); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (a[i] - b[i]).max(0.0));
                                }
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
                                let len = out_f32.len().min(a.len()).min(b.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = (a[i] * b[i]).max(0.0); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (a[i] * b[i]).max(0.0));
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
                                let softmax_params = resolve_params(params, param_dims, shape_env, 2)
                                    .unwrap_or_else(|_| vec![input.len(), 1]);
                                let axis_dim_size = softmax_params[0].max(1);
                                let stride = softmax_params.get(1).copied().unwrap_or(1).max(1);
                                let num_rows = input.len() / axis_dim_size.max(1);
                                #[cfg(not(feature = "parallel"))]
                                { for r in 0..num_rows {
                                    let outer = r / stride;
                                    let inner = r % stride;
                                    let base = (outer * axis_dim_size * stride) + inner;
                                    let mut max_val = f32::NEG_INFINITY;
                                    for i in 0..axis_dim_size {
                                        let idx = base + i * stride;
                                        if input[idx] > max_val { max_val = input[idx]; }
                                    }
                                    let mut sum = 0.0f32;
                                    for i in 0..axis_dim_size {
                                        let idx = base + i * stride;
                                        let e = (input[idx] - max_val).exp();
                                        out_f32[idx] = e;
                                        sum += e;
                                    }
                                    if sum > 0.0 {
                                        for i in 0..axis_dim_size {
                                            let idx = base + i * stride;
                                            out_f32[idx] /= sum;
                                        }
                                    }
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let stride = stride;
                                    let axis_dim_size = axis_dim_size;
                                    // SAFETY: each row writes to a disjoint set of output indices
                                    // (different `idx` values per row), so writing through a raw
                                    // pointer from multiple threads is safe.  Cast through `usize`
                                    // to satisfy `Send` required by rayon's `for_each`.
                                    let out_addr = out_f32.as_mut_ptr() as usize;
                                    (0..num_rows).into_par_iter().for_each(|r| {
                                        let out_ptr = out_addr as *mut f32;
                                        let outer = r / stride;
                                        let inner = r % stride;
                                        let base = (outer * axis_dim_size * stride) + inner;
                                        let mut max_val = f32::NEG_INFINITY;
                                        for i in 0..axis_dim_size {
                                            let idx = base + i * stride;
                                            let v = input[idx];
                                            if v > max_val { max_val = v; }
                                        }
                                        let mut sum = 0.0f32;
                                        for i in 0..axis_dim_size {
                                            let idx = base + i * stride;
                                            let e = (input[idx] - max_val).exp();
                                            unsafe { *out_ptr.add(idx) = e; }
                                            sum += e;
                                        }
                                        if sum > 0.0 {
                                            for i in 0..axis_dim_size {
                                                let idx = base + i * stride;
                                                unsafe { *out_ptr.add(idx) /= sum; }
                                            }
                                        }
                                    });
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = data[i] + bias[i % bias.len()]; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let bias = &bias;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = data[i] + bias[i % bias.len()]);
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
                                    let len = out_f32.len().min(data.len());
                                    #[cfg(not(feature = "parallel"))]
                                    { for i in 0..len {
                                        let ch = i % c;
                                        out_f32[i] = (data[i] - running_mean[ch]) / (running_var[ch] + eps).sqrt() * weight[ch] + bias[ch];
                                    } }
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                            let ch = i % c;
                                            *o = (data[i] - running_mean[ch]) / (running_var[ch] + eps).sqrt() * weight[ch] + bias[ch];
                                        });
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
                                    let num_rows = input.len().checked_div(row_size).unwrap_or(1);
                                    #[cfg(not(feature = "parallel"))]
                                    { for r in 0..num_rows {
                                        let start = r * row_size;
                                        let end = (start + row_size).min(input.len());
                                        let n = (end - start) as f32;
                                        let mean: f32 = input[start..end].iter().sum::<f32>() / n;
                                        let var: f32 = input[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                                        let inv_std = 1.0 / (var + eps).sqrt();
                                        for i in start..end {
                                            out_f32[i] = (input[i] - mean) * inv_std;
                                        }
                                    } }
                                    #[cfg(feature = "parallel")]
                                    {
                                        use rayon::prelude::*;
                                        let row_size = row_size;
                                        let out_addr = out_f32.as_mut_ptr() as usize;
                                        (0..num_rows).into_par_iter().for_each(|r| {
                                            let start = r * row_size;
                                            let end = start + row_size;
                                            let n = row_size as f32;
                                            let mean: f32 = input[start..end].iter().sum::<f32>() / n;
                                            let var: f32 = input[start..end].iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                                            let inv_std = 1.0 / (var + eps).sqrt();
                                            let out_ptr = out_addr as *mut f32;
                                            for i in start..end {
                                                unsafe { *out_ptr.add(i) = (input[i] - mean) * inv_std; }
                                            }
                                        });
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
                                let len = out_f32.len().min(a.len()).min(b.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = (a[i] / b[i]).max(0.0); } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = (a[i] / b[i]).max(0.0));
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
                                // params: [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]
                                let &[stride, padding, dilation, groups, c, h, w, kh, kw] = &params[..] else {
                                    return Err(BackendError::Dispatch("conv2d: expected params [stride, padding, dilation, groups, c, h, w, kh, kw]".into()));
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let c_per_group = c / groups.max(1);
                                let n = input_data.len() / (c * h * w).max(1);
                                let f = weight_data.len() / (c_per_group * kh * kw).max(1);
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
                        "conv2d_u4" | "conv2d_u8" => {
                            // Quantized conv2d using im2col + packed GEMM
                            // input_slices: [activation (f32), weight (packed)]  optional: [bias (f32)]
                            // weight_meta carries bit_width, shape=[OC, IC_per_group*KH*KW], scales[], zero_points[]
                            // Approach: build im2col matrix [N*OH*OW, col_w], then call gemm_cpu per batch
                            // where PackedTensor weights have shape [OC, col_w] and each GEMV produces OC outputs.
                            if let [a_slice, w_slice] = &input_slices[..] {
                                let (input_data, packed_bytes) = {
                                    let d = arena.data_mut();
                                    (bytemuck::cast_slice::<_, f32>(&d[a_slice.offset..a_slice.offset + a_slice.size]).to_vec(),
                                     {
                                        let raw = &d[w_slice.offset..w_slice.offset + w_slice.size];
                                        // Copy to u32-aligned buffer (arena may not be u32-aligned)
                                        let mut aligned: Vec<u32> = vec![0u32; raw.len().div_ceil(4)];
                                        let byte_slice = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
                                        byte_slice[..raw.len()].copy_from_slice(raw);
                                        aligned
                                    })
                                };
                                let bias_data: Vec<f32> = if input_slices.len() >= 3 {
                                    let b_slice = &input_slices[2];
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[b_slice.offset..b_slice.offset + b_slice.size]).to_vec()
                                } else {
                                    vec![]
                                };
                                let &[stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, _kernel_w] = &params[..] else {
                                    return Err(BackendError::Dispatch("conv2d_u4/u8: expected params [stride, padding, dilation, groups, input_c, input_h, input_w, kernel_h, kernel_w]".into()));
                                };
                                let meta = weight_meta.clone().ok_or_else(|| {
                                    BackendError::Dispatch("conv2d_u4/u8: missing weight_meta".into())
                                })?;
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                // Flatten conv2d weight shape [OC, IC, KH, KW] → [OC, IC*KH*KW]
                                // so the GEMM dispatch sees 2D weight with inner = IC*KH*KW.
                                let flat_shape = if meta.shape.len() >= 4 {
                                    let oc = meta.shape[0];
                                    let inner: usize = meta.shape[1..].iter().product();
                                    vec![oc, inner]
                                } else {
                                    meta.shape.clone()
                                };
                                let oc = flat_shape.first().copied().unwrap_or(1);
                    let c = input_c;
                    let h = input_h;
                    let w = input_w;
                    let kernel_size = kernel_h;
                                if groups == 0 { return Err(BackendError::Dispatch("conv2d_u4/u8: groups=0".into())); }
                                let c_per_g = c / groups;
                                let dk = (kernel_size - 1) * dilation + 1;
                                let n = input_data.len() / (c * h * w).max(1);
                                let h_out = if h + 2 * padding >= dk { (h + 2 * padding - dk) / stride + 1 } else { 0 };
                                let w_out = if w + 2 * padding >= dk { (w + 2 * padding - dk) / stride + 1 } else { 0 };
                                // Dispatch based on bit_width to handle different PackedTensor types
                                // (PackedTensor<U4x8> and PackedTensor<U8x4> are different types)
                                macro_rules! dispatch_conv2d_quant {
                                    ($PackedType:ty, $byte_cast:expr) => {{
                                        let packed_data: Vec<$PackedType> = bytemuck::cast_slice(&packed_bytes).to_vec();
                                        let pt = PackedTensor::from_raw(packed_data, flat_shape.clone(), meta.scales, meta.zero_points);
                                        let col_w = c_per_g * kernel_size * kernel_size;
                                        for nn in 0..n {
                                            let num_pixels = h_out * w_out;
                                            let mut col_matrix = vec![0.0f32; num_pixels * col_w];
                                            for hh in 0..h_out {
                                                for ww in 0..w_out {
                                                    let row = hh * w_out + ww;
                                                    for g in 0..groups {
                                                        for cc in 0..c_per_g {
                                                            for kkh in 0..kernel_size {
                                                                for kkw in 0..kernel_size {
                                                                    let h_in = (hh * stride + kkh * dilation) as i64 - padding as i64;
                                                                    let w_in = (ww * stride + kkw * dilation) as i64 - padding as i64;
                                                                    if h_in >= 0 && h_in < h as i64 && w_in >= 0 && w_in < w as i64 {
                                                                        let src = nn * (c * h * w)
                                                                            + (g * c_per_g + cc) * (h * w)
                                                                            + h_in as usize * w + w_in as usize;
                                                                        let dst = row * col_w
                                                                            + (g * c_per_g + cc) * (kernel_size * kernel_size)
                                                                            + kkh * kernel_size + kkw;
                                                                        if src < input_data.len() && dst < col_matrix.len() {
                                                                            col_matrix[dst] = input_data[src];
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            let batch_inputs: Vec<Vec<f32>> = (0..num_pixels)
                                                .map(|p| col_matrix[p * col_w..(p + 1) * col_w].to_vec())
                                                .collect();
                                            let mut batch_outputs: Vec<Vec<f32>> = (0..num_pixels)
                                                .map(|_| vec![0.0f32; oc])
                                                .collect();
                                            crate::backend::cpu::microkernels::gemm_cpu(&pt, &batch_inputs, &mut batch_outputs);
                                            for pixel in 0..num_pixels {
                                                for ff in 0..oc {
                                                    let mut val = batch_outputs[pixel][ff];
                                                    if !bias_data.is_empty() && ff < bias_data.len() {
                                                        val += bias_data[ff];
                                                    }
                                                    let out_idx = nn * (oc * h_out * w_out) + ff * (h_out * w_out) + pixel;
                                                    if out_idx < out_f32.len() {
                                                        out_f32[out_idx] = val;
                                                    }
                                                }
                                            }
                                        }
                                    }}
                                }
                                if meta.bit_width == 4 {
                                    dispatch_conv2d_quant!(U4x8, 4);
                                } else {
                                    dispatch_conv2d_quant!(U8x4, 8);
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
                                let mut indices_out: Option<&mut [i64]> = if is_max == 1 {
                                    secondary_output_slice.as_ref().map(|sec_slice| {
                                        let d = arena.data_mut();
                                        bytemuck::cast_slice_mut::<_, i64>(
                                            &mut d[sec_slice.offset..sec_slice.offset + sec_slice.size]
                                        )
                                    })
                                } else {
                                    None
                                };
                                for nn in 0..n {
                                    for cc in 0..c {
                                        for hh in 0..h_out {
                                            for ww in 0..w_out {
                                                let mut val = if is_max == 1 { f32::NEG_INFINITY } else { 0.0f32 };
                                                let mut best_kh = 0usize;
                                                let mut best_kw = 0usize;
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
                                                                    if is_max == 1 {
                                                                        if input[idx] > val {
                                                                            val = input[idx];
                                                                            best_kh = kh;
                                                                            best_kw = kw;
                                                                        }
                                                                    } else {
                                                                        val += input[idx];
                                                                    }
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
                                                // Write argmax index if this is max pooling
                                                if let Some(ref mut idx_out) = indices_out {
                                                    if out_idx < idx_out.len() {
                                                        idx_out[out_idx] = (best_kh * kernel + best_kw) as i64;
                                                    }
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
                                    let idx_idx = i.checked_div(inner).unwrap_or(i);
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
                                let copy_len = out_f32.len().min(input.len().saturating_sub(offset));
                                out_f32[..copy_len].copy_from_slice(&input[offset..(offset + copy_len)]);
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
                                let copy_len = out_f32.len().min(updates.len());
                                out_f32[..copy_len].copy_from_slice(&updates[..copy_len]);
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
                                #[allow(clippy::type_complexity)]
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
                                let num_rows = input.len().checked_div(row_size).unwrap_or(1);
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len {
                                    let e = if i < exponent.len() { exponent[i] } else { exponent[exponent.len().saturating_sub(1)] };
                                    out_f32[i] = data[i].powf(e);
                                } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    let exponent = &exponent;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| {
                                        let e = if i < exponent.len() { exponent[i] } else { exponent[exponent.len().saturating_sub(1)] };
                                        *o = data[i].powf(e);
                                    });
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if data[i] > s { 1.0 } else { 0.0 }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if data[i] > s { 1.0 } else { 0.0 });
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if data[i] < s { 1.0 } else { 0.0 }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if data[i] < s { 1.0 } else { 0.0 });
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = if (data[i] - s).abs() < 1e-6 { 1.0 } else { 0.0 }; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = if (data[i] - s).abs() < 1e-6 { 1.0 } else { 0.0 });
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = data[i] + s; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = data[i] + s);
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = data[i] * s; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = data[i] * s);
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
                                let len = out_f32.len().min(data.len());
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = data[i] / s; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = data[i] / s);
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
                         "topk_fused" => {
                             if let Some(input_slice) = input_slices.first() {
                                 let input = {
                                     let d = arena.data_mut();
                                     bytemuck::cast_slice::<_, f32>(
                                         &d[input_slice.offset..input_slice.offset + input_slice.size]
                                     ).to_vec()
                                 };
                                 let k = params.first().copied().unwrap_or(1);
                                 let mut indexed: Vec<(usize, f32)> = input.iter().copied().enumerate().collect();
                                 if input.len() > k {
                                     indexed.select_nth_unstable_by(input.len().saturating_sub(k), |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                                 }

                                 // Write values (f32) to primary output
                                 let d = arena.data_mut();
                                 let out_slice = bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]);
                                 for i in 0..k.min(out_slice.len()) {
                                     out_slice[i] = indexed[input.len().saturating_sub(k) + i].1;
                                 }

                                 // Write indices (i64) to secondary output
                                 if let Some(sec_slice) = secondary_output_slice {
                                     let sec_start = sec_slice.offset;
                                     let sec_end = sec_slice.offset + sec_slice.size;
                                     let idx_slice = bytemuck::cast_slice_mut::<_, u64>(&mut d[sec_start..sec_end]);
                                     for i in 0..k.min(idx_slice.len()) {
                                         idx_slice[i] = indexed[input.len().saturating_sub(k) + i].0 as u64;
                                     }
                                 }
                             }
                         }
                         "topk_values" | "topk_indices" => {
                             if let Some(input_slice) = input_slices.first() {
                                 let input = {
                                     let d = arena.data_mut();
                                     bytemuck::cast_slice::<_, f32>(
                                         &d[input_slice.offset..input_slice.offset + input_slice.size]
                                     ).to_vec()
                                 };
                                 let k = params.first().copied().unwrap_or(1);
                                 let _axis = params.get(1).copied().unwrap_or(usize::MAX);
                                 let is_values = kernel_name == "topk_values";

                                 let d = arena.data_mut();
                                 if is_values {
                                     let out_slice = bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]);
                                     let mut indexed: Vec<(usize, f32)> = input.iter().copied().enumerate().collect();
                                     if input.len() > k {
                                         indexed.select_nth_unstable_by(input.len().saturating_sub(k), |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                                     }
                                     for i in 0..k.min(out_slice.len()) {
                                         out_slice[i] = indexed[input.len().saturating_sub(k) + i].1;
                                     }
                                 } else {
                                     let out_slice = bytemuck::cast_slice_mut::<_, u64>(&mut d[out_start..out_end]);
                                     let mut indexed: Vec<(usize, f32)> = input.iter().copied().enumerate().collect();
                                     if input.len() > k {
                                         indexed.select_nth_unstable_by(input.len().saturating_sub(k), |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                                     }
                                     for i in 0..k.min(out_slice.len()) {
                                         out_slice[i] = indexed[input.len().saturating_sub(k) + i].0 as u64;
                                     }
                                 }
                             }
                         }
                         "upsample_nearest2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let out_len = out_f32.len();
                                let in_len = input.len();
                                if scale_h > 0 && scale_w > 0 && out_len == in_len * scale_h * scale_w && in_len > 0 {
                                    let hw = in_len / (out_len / (scale_h * scale_w));
                                    let mut h = (hw as f64).sqrt() as usize;
                                    while h > 0 && hw % h != 0 { h -= 1; }
                                    let w = hw / h;
                                    if h > 0 && w > 0 {
                                        let nc = in_len / hw;
                                        for nci in 0..nc {
                                            for hi in 0..h {
                                                for wi in 0..w {
                                                    let val = input[nci * hw + hi * w + wi];
                                                    for sh in 0..scale_h {
                                                        for sw in 0..scale_w {
                                                            let out_idx = nci * hw * scale_h * scale_w
                                                                + (hi * scale_h + sh) * (w * scale_w)
                                                                + (wi * scale_w + sw);
                                                            if out_idx < out_len {
                                                                out_f32[out_idx] = val;
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
                        "upsample_bilinear2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let scale_h = params.first().copied().unwrap_or(2);
                                let scale_w = params.get(1).copied().unwrap_or(2);
                                let out_len = out_f32.len();
                                let in_len = input.len();
                                if scale_h > 0 && scale_w > 0 && out_len == in_len * scale_h * scale_w {
                                    let hw = in_len / (out_len / (scale_h * scale_w));
                                    let mut h = (hw as f64).sqrt() as usize;
                                    while h > 0 && hw % h != 0 { h -= 1; }
                                    let w = hw / h;
                                    if h > 0 && w > 0 {
                                        let nc = in_len / hw;
                                        for nci in 0..nc {
                                            for hi in 0..h * scale_h {
                                                for wi in 0..w * scale_w {
                                                    let src_h = (hi as f64 / scale_h as f64).min((h - 1) as f64);
                                                    let src_w = (wi as f64 / scale_w as f64).min((w - 1) as f64);
                                                    let h0 = src_h.floor() as usize;
                                                    let w0 = src_w.floor() as usize;
                                                    let h1 = (h0 + 1).min(h - 1);
                                                    let w1 = (w0 + 1).min(w - 1);
                                                    let dh = src_h - h0 as f64;
                                                    let dw = src_w - w0 as f64;
                                                    let v00 = input[nci * hw + h0 * w + w0];
                                                    let v01 = input[nci * hw + h0 * w + w1];
                                                    let v10 = input[nci * hw + h1 * w + w0];
                                                    let v11 = input[nci * hw + h1 * w + w1];
                                                    let v0 = v00 * (1.0 - dw as f32) + v01 * dw as f32;
                                                    let v1 = v10 * (1.0 - dw as f32) + v11 * dw as f32;
                                                    let val = v0 * (1.0 - dh as f32) + v1 * dh as f32;
                                                    let out_idx = nci * h * scale_h * w * scale_w + hi * w * scale_w + wi;
                                                    if out_idx < out_len { out_f32[out_idx] = val; }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "adaptive_avg_pool2d" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let out_h = params.first().copied().unwrap_or(1);
                                let out_w = params.get(1).copied().unwrap_or(1);
                                let out_len = out_f32.len();
                                if out_len > 0 {
                                    let in_len = input.len();
                                    let nc = if out_h > 0 && out_w > 0 { out_len / (out_h * out_w) } else { 0 };
                                    if nc > 0 && in_len > 0 && in_len % nc == 0 {
                                        let hw = in_len / nc;
                                        let mut h = (hw as f64).sqrt() as usize;
                                        while h > 0 && hw % h != 0 { h -= 1; }
                                        let w = hw / h;
                                        if h >= out_h && w >= out_w && h > 0 && w > 0 {
                                            for nci in 0..nc {
                                                for ohi in 0..out_h {
                                                    for owi in 0..out_w {
                                                        let h_start = ohi * h / out_h;
                                                        let h_end = (ohi + 1) * h / out_h;
                                                        let w_start = owi * w / out_w;
                                                        let w_end = (owi + 1) * w / out_w;
                                                        let mut sum = 0.0f32;
                                                        let mut count = 0;
                                                        for hi in h_start..h_end {
                                                            for wi in w_start..w_end {
                                                                sum += input[nci * hw + hi * w + wi];
                                                                count += 1;
                                                            }
                                                        }
                                                        let out_idx = nci * out_h * out_w + ohi * out_w + owi;
                                                        if out_idx < out_len {
                                                            out_f32[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "repeat" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let in_len = input.len();
                                let out_len = out_f32.len();
                                if in_len > 0 && out_len >= in_len && out_len % in_len == 0 {
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
                            }
                        }
                        "cumsum" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let _dim = params.first().copied().unwrap_or(0);
                                let exclusive = params.get(1).copied().unwrap_or(0);
                                let rev = params.get(2).copied().unwrap_or(0);
                                let len = out_f32.len().min(input.len());
                                if rev == 0 {
                                    let mut s = 0.0f32;
                                    for i in 0..len {
                                        s += input[i];
                                        out_f32[i] = if exclusive != 0 { s - input[i] } else { s };
                                    }
                                } else {
                                    let mut s = 0.0f32;
                                    for i in (0..len).rev() {
                                        s += input[i];
                                        out_f32[i] = if exclusive != 0 { s - input[i] } else { s };
                                    }
                                }
                            }
                        }
                        "erf_f32" => {
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
                                    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
                                    #[allow(clippy::excessive_precision)]
                                    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
                                    out_f32[i] = x.signum() * y;
                                }
                            }
                        }
                        "flip" => {
                            if let Some(input_slice) = input_slices.first() {
                                let input = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(&d[input_slice.offset..input_slice.offset + input_slice.size]).to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                if params.is_empty() {
                                    let len = out_f32.len().min(input.len());
                                    for i in 0..len {
                                        out_f32[i] = input[len - 1 - i];
                                    }
                                } else {
                                    let num_dims = params[0];
                                    let flip_dims: Vec<usize> = params[1..1 + num_dims].to_vec();
                                    let shape: Vec<usize> = params[1 + num_dims..].to_vec();
                                    let ndim = shape.len();
                                    let len = out_f32.len().min(input.len());
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
                            }
                        }
                        "where_f32" => {
                            if input_slices.len() >= 3 {
                                let cond_slice = &input_slices[0];
                                let x_slice = &input_slices[1];
                                let y_slice = &input_slices[2];
                                let (cond, x, y) = {
                                    let d = arena.data_mut();
                                    (
                                        bytemuck::cast_slice::<_, f32>(&d[cond_slice.offset..cond_slice.offset + cond_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[x_slice.offset..x_slice.offset + x_slice.size]).to_vec(),
                                        bytemuck::cast_slice::<_, f32>(&d[y_slice.offset..y_slice.offset + y_slice.size]).to_vec(),
                                    )
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                let len = out_f32.len();
                                for i in 0..len {
                                    let c = cond.get(i % cond.len()).copied().unwrap_or(0.0);
                                    out_f32[i] = if c != 0.0 {
                                        x.get(i % x.len()).copied().unwrap_or(0.0)
                                    } else {
                                        y.get(i % y.len()).copied().unwrap_or(0.0)
                                    };
                                }
                            }
                        }
                        // ── Optimizer kernels ───────────────────────
                        "sgd_update_f32" => {
                            let (w_init, grad, lr) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[0].offset..input_slices[0].offset + input_slices[0].size]).to_vec();
                                let grad = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[1].offset..input_slices[1].offset + input_slices[1].size]).to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                (w_init, grad, lr)
                            };
                            #[cfg(not(feature = "parallel"))]
                            { let w_new: Vec<f32> = w_init.iter().enumerate().map(|(i, &wi)| {
                                wi - lr * grad.get(i % grad.len()).copied().unwrap_or(0.0)
                            }).collect();
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new); }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                let mut w_new = w_init.clone();
                                w_new.par_iter_mut().enumerate().for_each(|(i, w)| {
                                    *w = w_init[i] - lr * grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                });
                                let d = arena.data_mut();
                                bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new);
                            }
                        }
                        "gradient_scale" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let scale = f32::from_bits(*params.get(1).unwrap_or(&0) as u32);
                                let in_f32 = {
                                    let d = arena.data_mut();
                                    let d_ref: &[u8] = &*d;
                                    bytemuck::cast_slice::<_, f32>(
                                        &d_ref[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let d = arena.data_mut();
                                let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]);
                                let len = out_f32.len().min(in_f32.len()).min(numel);
                                #[cfg(not(feature = "parallel"))]
                                { for i in 0..len { out_f32[i] = in_f32[i] * scale; } }
                                #[cfg(feature = "parallel")]
                                {
                                    use rayon::prelude::*;
                                    out_f32[..len].par_iter_mut().enumerate().for_each(|(i, o)| *o = in_f32[i] * scale);
                                }
                            }
                        }
                        "adam_update_f32" => {
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[0].offset..input_slices[0].offset + input_slices[0].size]).to_vec();
                                let m_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]).to_vec();
                                let v_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]).to_vec();
                                let grad = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[1].offset..input_slices[1].offset + input_slices[1].size]).to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = w_init.len();
                            let mut w_new = w_init.clone();
                            let mut m_new = m_init.clone();
                            let mut v_new = v_init.clone();
                            #[cfg(not(feature = "parallel"))]
                            { for i in 0..len {
                                let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                m_new[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                v_new[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                let m_hat = m_new[i] / bias_corr1;
                                let v_hat = v_new[i] / bias_corr2;
                                w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                            } }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new.par_iter_mut()
                                    .zip(m_new.par_iter_mut())
                                    .zip(v_new.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]).copy_from_slice(&m_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]).copy_from_slice(&v_new);
                        }
                        "adamw_update_f32" => {
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[0].offset..input_slices[0].offset + input_slices[0].size]).to_vec();
                                let m_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]).to_vec();
                                let v_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]).to_vec();
                                let grad = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[1].offset..input_slices[1].offset + input_slices[1].size]).to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                let wd = f32::from_bits(params[5] as u32);
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = w_init.len();
                            let mut w_new = w_init.clone();
                            let mut m_new = m_init.clone();
                            let mut v_new = v_init.clone();
                            #[cfg(not(feature = "parallel"))]
                            { for i in 0..len {
                                w_new[i] -= lr * wd * w_init[i];
                                let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                m_new[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                v_new[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                let m_hat = m_new[i] / bias_corr1;
                                let v_hat = v_new[i] / bias_corr2;
                                w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                            } }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new.par_iter_mut()
                                    .zip(m_new.par_iter_mut())
                                    .zip(v_new.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        *w -= lr * wd * w_init[i];
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]).copy_from_slice(&m_new);
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]).copy_from_slice(&v_new);
                        }
                        // ── F16 state optimizer kernels ──────────────
                        // m and v are stored as F16 (2 bytes/elem), w and grad are F32 (4 bytes/elem).
                        // Read F16 state, convert to f32 internally, apply update, write back as F16.
                        "adam_update_f16_state" => {
                            let n = {
                                let w_bytes = out_end - out_start;
                                w_bytes / 4
                            };
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[0].offset..input_slices[0].offset + input_slices[0].size]).to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                                let m_init: Vec<f32> = m_raw.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect();
                                let v_init: Vec<f32> = v_raw.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect();
                                let grad = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[1].offset..input_slices[1].offset + input_slices[1].size]).to_vec();
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
                            { for i in 0..len {
                                let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                let m_hat = m_new_f32[i] / bias_corr1;
                                let v_hat = v_new_f32[i] / bias_corr2;
                                w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                            } }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new.par_iter_mut()
                                    .zip(m_new_f32.par_iter_mut())
                                    .zip(v_new_f32.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                            let v_bytes: Vec<u8> = v_new_f32.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end].copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end].copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "adamw_update_f16_state" => {
                            let n = {
                                let w_bytes = out_end - out_start;
                                w_bytes / 4
                            };
                            let (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd) = {
                                let d = arena.data_mut();
                                let d_ref: &[u8] = &*d;
                                let w_init = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[0].offset..input_slices[0].offset + input_slices[0].size]).to_vec();
                                let m_raw: Vec<u16> = d_ref[input_slices[2].offset..input_slices[2].offset + input_slices[2].size]
                                    .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                                let v_raw: Vec<u16> = d_ref[input_slices[3].offset..input_slices[3].offset + input_slices[3].size]
                                    .chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                                let m_init: Vec<f32> = m_raw.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect();
                                let v_init: Vec<f32> = v_raw.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect();
                                let grad = bytemuck::cast_slice::<_, f32>(&d_ref[input_slices[1].offset..input_slices[1].offset + input_slices[1].size]).to_vec();
                                let lr = f32::from_bits(params[0] as u32);
                                let beta1 = f32::from_bits(params[1] as u32);
                                let beta2 = f32::from_bits(params[2] as u32);
                                let eps = f32::from_bits(params[3] as u32);
                                let t = params[4] as f32;
                                let wd = f32::from_bits(params[5] as u32);
                                (w_init, m_init, v_init, grad, lr, beta1, beta2, eps, t, wd)
                            };
                            let bias_corr1 = 1.0 - beta1.powi(t as i32);
                            let bias_corr2 = 1.0 - beta2.powi(t as i32);
                            let len = n.min(w_init.len()).min(m_init.len()).min(v_init.len());
                            let mut w_new = w_init.clone();
                            let mut m_new_f32 = vec![0.0f32; len];
                            let mut v_new_f32 = vec![0.0f32; len];
                            #[cfg(not(feature = "parallel"))]
                            { for i in 0..len {
                                w_new[i] -= lr * wd * w_init[i];
                                let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                m_new_f32[i] = beta1 * m_init[i] + (1.0 - beta1) * g;
                                v_new_f32[i] = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                let m_hat = m_new_f32[i] / bias_corr1;
                                let v_hat = v_new_f32[i] / bias_corr2;
                                w_new[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                            } }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                w_new.par_iter_mut()
                                    .zip(m_new_f32.par_iter_mut())
                                    .zip(v_new_f32.par_iter_mut())
                                    .enumerate()
                                    .for_each(|(i, ((w, m), v))| {
                                        *w -= lr * wd * w_init[i];
                                        let g = grad.get(i % grad.len()).copied().unwrap_or(0.0);
                                        *m = beta1 * m_init[i] + (1.0 - beta1) * g;
                                        *v = beta2 * v_init[i] + (1.0 - beta2) * g * g;
                                        let m_hat = *m / bias_corr1;
                                        let v_hat = *v / bias_corr2;
                                        *w -= lr * m_hat / (v_hat.sqrt() + eps);
                                    });
                            }
                            let d = arena.data_mut();
                            bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end]).copy_from_slice(&w_new);
                            let m_bytes: Vec<u8> = m_new_f32.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                            let v_bytes: Vec<u8> = v_new_f32.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                            let m_end = (input_slices[2].offset + m_bytes.len()).min(d.len());
                            let v_end = (input_slices[3].offset + v_bytes.len()).min(d.len());
                            d[input_slices[2].offset..m_end].copy_from_slice(&m_bytes[..m_end - input_slices[2].offset]);
                            d[input_slices[3].offset..v_end].copy_from_slice(&v_bytes[..v_end - input_slices[3].offset]);
                        }
                        "cast" => {
                            let in_byte_size = *params.first().unwrap_or(&4);
                            let out_byte_size = *params.get(1).unwrap_or(&4);
                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let in_data = &d[input_slice.offset..input_slice.offset + input_slice.size];
                                if in_byte_size == 4 && out_byte_size == 8 {
                                    // F32/I32 → I64: widen
                                    let in_f32 = bytemuck::cast_slice::<_, f32>(in_data).to_vec();
                                    let mut out_bytes = Vec::with_capacity(in_f32.len() * 8);
                                    for &v in &in_f32 {
                                        out_bytes.extend_from_slice(&(v as i64).to_le_bytes());
                                    }
                                    let end = (out_start + out_bytes.len()).min(d.len());
                                    d[out_start..end].copy_from_slice(&out_bytes[..end - out_start]);
                                } else if in_byte_size == 8 && out_byte_size == 4 {
                                    // I64 → F32/I32: narrow
                                    let in_i64 = bytemuck::cast_slice::<_, i64>(in_data).to_vec();
                                    let mut out_bytes = Vec::with_capacity(in_i64.len() * 4);
                                    for &v in &in_i64 {
                                        out_bytes.extend_from_slice(&(v as f32).to_le_bytes());
                                    }
                                    let end = (out_start + out_bytes.len()).min(d.len());
                                    d[out_start..end].copy_from_slice(&out_bytes[..end - out_start]);
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
                                    "expand_f32 needs 2 inputs (data + shape)".into()
                                ));
                            }
                            let max_rank = *params.first().ok_or_else(||
                                BackendError::Dispatch("expand_f32: missing max_rank".into())
                            )?;
                            if params.len() < 1 + max_rank * 2 {
                                return Err(BackendError::Dispatch(format!(
                                    "expand_f32: expected {} params, got {}",
                                    1 + max_rank * 2, params.len()
                                )));
                            }
                            // Extract padded input dims and output dims
                            let in_dims: Vec<usize> = params[1..1 + max_rank].to_vec();
                            let out_dims: Vec<usize> = params[1 + max_rank..1 + max_rank * 2].to_vec();

                            let data_slice = &input_slices[0];
                            let shape_slice = &input_slices[1];
                            let data_numel = data_slice.size / 4; // f32 = 4 bytes
                            let out_numel = output_slice.size / 4;

                            let d = arena.data_mut();

                            // Read target shape tensor (I64 values giving output dims)
                            // This is the runtime version of the shape; we use compile-time
                            // dims from params as the source of truth.
                            let _shape_data: Vec<i64> = bytemuck::cast_slice::<_, i64>(
                                &d[shape_slice.offset..shape_slice.offset + shape_slice.size]
                            ).to_vec();

                            // Read input data
                            let in_f32: Vec<f32> = bytemuck::cast_slice::<_, f32>(
                                &d[data_slice.offset..data_slice.offset + data_slice.size]
                            ).to_vec();

                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[out_start..out_start + output_slice.size]
                            );

                            // Broadcast: for each output element, map back to input coords
                            #[cfg(not(feature = "parallel"))]
                            { for out_linear in 0..out_numel {
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
                            } }
                            #[cfg(feature = "parallel")]
                            {
                                use rayon::prelude::*;
                                let max_rank = max_rank;
                                let out_addr = out_f32.as_mut_ptr() as usize;
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
                                        unsafe { *(out_addr as *mut f32).add(out_linear) = in_f32[in_linear]; }
                                    }
                                });
                            }
                        }
                        "range_f32" => {
                            // Range(start, limit, step): produce 1D F32 tensor.
                            let d = arena.data_mut();
                            let start_val = if let Some(s) = input_slices.first() {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4])
                                )
                            } else {
                                0.0
                            };
                            let limit_val = if let Some(s) = input_slices.get(1) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([0u8; 4])
                                )
                            } else {
                                0.0
                            };
                            let step_val = if let Some(s) = input_slices.get(2) {
                                f32::from_le_bytes(
                                    d[s.offset..s.offset + 4].try_into().unwrap_or([1u8; 4])
                                )
                            } else {
                                1.0
                            };
                            let n = ((limit_val - start_val) / step_val).ceil().max(0.0) as usize;
                            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(
                                &mut d[out_start..out_start + output_slice.size]
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
                            let bit_width = if kernel_name == "quantize_f32_u4" { 4 } else { 8 };
                            let max_q = (1i32 << (bit_width - 1)) - 1; // 7 for U4, 127 for U8
                            let items_per_word = 32 / bit_width; // 8 for U4, 4 for U8

                            if let Some(input_slice) = input_slices.first() {
                                let d = arena.data_mut();
                                let f32_data: Vec<f32> = bytemuck::cast_slice::<_, f32>(
                                    &d[input_slice.offset..input_slice.offset + input_slice.size]
                                ).to_vec();

                                // Compute per-channel scales and pack data
                                let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                let zero_points: Vec<f32> = vec![0.0; num_channels];
                                let packed_words = numel.div_ceil(items_per_word);
                                let mut packed: Vec<u32> = vec![0u32; packed_words];

                                for ch in 0..num_channels {
                                    let start = ch * num_elems_per_channel;
                                    let end = (start + num_elems_per_channel).min(f32_data.len());
                                    // Compute scale as max absolute value
                                    let max_abs = f32_data[start..end].iter()
                                        .map(|v| v.abs())
                                        .fold(0.0f32, f32::max);
                                    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / max_q as f32 };
                                    scales.push(scale);

                                    // Quantize and pack
                                    for j in start..end {
                                        let q = (f32_data[j] / scale).round().clamp(
                                            -(max_q as f32), max_q as f32
                                        ) as i32;
                                        let word_idx = j / items_per_word;
                                        let shift = (j % items_per_word) * bit_width;
                                        packed[word_idx] |= ((q as u32) & ((1 << bit_width) - 1)) << shift;
                                    }
                                }

                                // Write output: [num_channels(u32)][num_elems_per_channel(u32)]
                                //             [scales(f32 x N)][zero_points(f32 x N)][packed_data]
                                let header_size = 8 + 8 * num_channels; // 2 u32 + N f32 + N f32
                                let total_size = header_size + packed.len() * 4;
                                let out_end = (out_start + total_size).min(d.len());
                                let out = &mut d[out_start..out_end];

                                let mut offset = 0;
                                out[offset..offset+4].copy_from_slice(&(num_channels as u32).to_le_bytes());
                                offset += 4;
                                out[offset..offset+4].copy_from_slice(&(num_elems_per_channel as u32).to_le_bytes());
                                offset += 4;
                                for &s in &scales {
                                    out[offset..offset+4].copy_from_slice(&s.to_le_bytes());
                                    offset += 4;
                                }
                                for &z in &zero_points {
                                    out[offset..offset+4].copy_from_slice(&z.to_le_bytes());
                                    offset += 4;
                                }
                                for &w in &packed {
                                    out[offset..offset+4].copy_from_slice(&w.to_le_bytes());
                                    offset += 4;
                                }
                            }
                        }
                        "dequantize_kernel" => {
                            if let Some(input_slice) = input_slices.first() {
                                let numel = *params.first().unwrap_or(&0);
                                let format_flag = *params.get(1).unwrap_or(&0); // 0=header, 1=metadata
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size].to_vec()
                                };

                                let (num_channels, num_elems_per_channel, scales, zero_points, data_offset, bit_width) =
                                if format_flag == 1 {
                                    // Metadata-based: scales/zero_points passed as params
                                    let num_channels = *params.get(2).unwrap_or(&0);
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = *params.get(3 + j).unwrap_or(&0);
                                        scales.push(f32::from_bits(bits as u32));
                                    }
                                    let mut zero_points: Vec<f32> = Vec::with_capacity(num_channels);
                                    for j in 0..num_channels {
                                        let bits = *params.get(3 + num_channels + j).unwrap_or(&0);
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
                                    // Infer bit_width from packed byte ratio
                                    let total_packed_bytes = in_data.len().saturating_sub(data_offset);
                                    let packed_words = total_packed_bytes / 4;
                                    let bit_width = if packed_words > 0 && numel > 0 {
                                        let ratio = (packed_words * 4) as f64 / numel as f64;
                                        if ratio < 0.6 { 4 } else { 8 }
                                    } else {
                                        8
                                    };
                                    (num_channels, num_elems_per_channel, scales, zero_points, data_offset, bit_width)
                                } else {
                                    // Header-based: parse [num_channels][num_elems][scales...][zps...][packed_data]
                                    let num_channels = u32::from_le_bytes(
                                        in_data[0..4].try_into().unwrap_or([0u8; 4])
                                    ) as usize;
                                    let num_elems_per_channel = u32::from_le_bytes(
                                        in_data[4..8].try_into().unwrap_or([0u8; 4])
                                    ) as usize;
                                    let mut hdr_offset = 8usize;
                                    let mut scales: Vec<f32> = Vec::with_capacity(num_channels);
                                    for _ in 0..num_channels {
                                        if hdr_offset + 4 <= in_data.len() {
                                            let s = f32::from_le_bytes(in_data[hdr_offset..hdr_offset+4].try_into().unwrap());
                                            scales.push(s);
                                            hdr_offset += 4;
                                        }
                                    }
                                    // zero_points (currently all zero for symmetric quant)
                                    for _ in 0..num_channels {
                                        if hdr_offset + 4 <= in_data.len() {
                                            hdr_offset += 4;
                                        }
                                    }
                                    let zero_points: Vec<f32> = vec![0.0; num_channels];
                                    let data_offset = hdr_offset;
                                    let total_packed_bytes = in_data.len().saturating_sub(data_offset);
                                    let packed_words = total_packed_bytes / 4;
                                    let bit_width = if packed_words > 0 && numel > 0 {
                                        let ratio = (packed_words * 4) as f64 / numel as f64;
                                        if ratio < 0.6 { 4 } else { 8 }
                                    } else {
                                        8
                                    };
                                    (num_channels, num_elems_per_channel, scales, zero_points, data_offset, bit_width)
                                };

                                let items_per_word = 32 / bit_width;
                                let total_packed_bytes = in_data.len().saturating_sub(data_offset);
                                let packed_words = total_packed_bytes / 4;

                                // Write output
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
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
                                            u32::from_le_bytes(in_data[word_start..word_start+4].try_into().unwrap())
                                        } else { 0 };
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
                                let f32_data = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                let out_bytes = {
                                    let d = arena.data_mut();
                                    &mut d[out_start..out_end]
                                };
                                for (i, &v) in f32_data.iter().enumerate() {
                                    let f16_val = half::f16::from_f32(v);
                                    let bytes = f16_val.to_le_bytes();
                                    let start = i * 2;
                                    let end = (start + 2).min(out_bytes.len());
                                    if start < out_bytes.len() {
                                        out_bytes[start..end].copy_from_slice(&bytes[..end - start]);
                                    }
                                }
                            }
                        }
                        "to_f32" => {
                            if let Some(input_slice) = input_slices.first() {
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size].to_vec()
                                };
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let max_out = out_f32.len().min(in_data.len() / 2);
                                for i in 0..max_out {
                                    let start = i * 2;
                                    if start + 2 <= in_data.len() {
                                        let f16_val = half::f16::from_le_bytes(
                                            in_data[start..start+2].try_into().unwrap()
                                        );
                                        out_f32[i] = f16_val.to_f32();
                                    }
                                }
                            }
                        }
                        "quantize_activations" => {
                            let numel = *params.first().unwrap_or(&0);
                            if let Some(input_slice) = input_slices.first() {
                                let f32_data = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice::<_, f32>(
                                        &d[input_slice.offset..input_slice.offset + input_slice.size]
                                    ).to_vec()
                                };
                                // Symmetric INT8 quantization: scale = max_abs / 127
                                let max_abs = f32_data.iter()
                                    .map(|v| v.abs())
                                    .fold(0.0f32, f32::max);
                                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                                let zp = 0.0f32;

                                let out_bytes = {
                                    let d = arena.data_mut();
                                    &mut d[out_start..out_end]
                                };
                                // Write scale, zp, then quantized i8 data
                                let header_size = 8; // scale + zp
                                out_bytes[0..4].copy_from_slice(&scale.to_le_bytes());
                                out_bytes[4..8].copy_from_slice(&zp.to_le_bytes());
                                let max_out = (out_bytes.len() - header_size).min(numel);
                                for i in 0..max_out {
                                    let q = (f32_data[i] / scale).round().clamp(-128.0, 127.0) as i8;
                                    out_bytes[header_size + i] = q as u8;
                                }
                            }
                        }
                        "dequantize_activations" => {
                            let numel = *params.first().unwrap_or(&0);
                            if let Some(input_slice) = input_slices.first() {
                                let in_data = {
                                    let d = arena.data_mut();
                                    d[input_slice.offset..input_slice.offset + input_slice.size].to_vec()
                                };
                                let header_size = 8;
                                let scale = if in_data.len() >= 4 {
                                    f32::from_le_bytes(in_data[0..4].try_into().unwrap())
                                } else { 1.0 };
                                let zp = if in_data.len() >= 8 {
                                    f32::from_le_bytes(in_data[4..8].try_into().unwrap())
                                } else { 0.0 };

                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let max_out = out_f32.len().min(numel);
                                for i in 0..max_out {
                                    let idx = header_size + i;
                                    let q = if idx < in_data.len() { in_data[idx] as i8 } else { 0i8 };
                                    out_f32[i] = (q as f32) * scale + zp;
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
