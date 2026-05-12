#![allow(dead_code)]

use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::passes::memory_planning::MemoryPlan;
use crate::ir::node::{ComputeGraph, IrDType, Opcode, TensorValue};
use bytemuck;
use std::cell::UnsafeCell;

pub mod microkernels;

/// CPU memory arena with interior mutability for zero-allocation dispatch.
pub struct CpuBuffer(UnsafeCell<Vec<u8>>);

impl CpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        CpuBuffer(UnsafeCell::new(data))
    }

    /// Get a mutable slice to the arena data. SAFETY: caller must ensure
    /// no other references exist when this is called.
    pub fn data_mut(&self) -> &mut [u8] {
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }
}

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

            // Collect input shapes for dimension-dependent kernels
            let input_shapes: Vec<Vec<u64>> = node
                .inputs
                .iter()
                .filter_map(|&input_id| graph.get_node(input_id))
                .map(|n| {
                    n.output_type
                        .shape
                        .iter()
                        .filter_map(|d| d.evaluate())
                        .collect()
                })
                .collect();

            let output_slice = memory_plan
                .slots
                .get(&node_id)
                .map(|slot| BufferSlice::new(slot.offset, slot.size))
                .unwrap_or(BufferSlice::new(0, 0));

            match &node.opcode {
                Opcode::Constant(val) => {
                    let fill_value = match val {
                        TensorValue::Float(v) => *v,
                        TensorValue::Int(v) => *v as f32,
                        TensorValue::Data { .. } => 0.0, // will be zero-initialized
                    };
                    instructions.push(Instruction::Fill {
                        dst: output_slice,
                        value: fill_value,
                    });
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
                    instructions.push(Instruction::CallKernel {
                        kernel_name: kernel_name.to_string(),
                        input_slices,
                        output_slice,
                        params: vec![m, k, n],
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
                    });
                }
                Opcode::BatchNorm | Opcode::LayerNorm => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "norm_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                    });
                }
                Opcode::Softmax => {
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "softmax_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![],
                    });
                }
                Opcode::ReduceSum | Opcode::ReduceMean => {
                    // For reduce, compute group size from input/output element counts
                    let input_elems = input_slices
                        .first()
                        .map(|s| s.size / 4)
                        .unwrap_or(1);
                    let output_elems = output_slice.size / 4;
                    let group_size = if output_elems > 0 {
                        input_elems / output_elems
                    } else {
                        1
                    };
                    let is_mean = if matches!(node.opcode, Opcode::ReduceMean) { 1 } else { 0 };
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "reduce_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![group_size, is_mean],
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
                    instructions.push(Instruction::CallKernel {
                        kernel_name: "transpose_f32".to_string(),
                        input_slices,
                        output_slice,
                        params: vec![m, n],
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
    ) -> Result<(), BackendError> {
        for instr in &plan.instructions {
            match instr {
                Instruction::CallKernel {
                    kernel_name,
                    input_slices,
                    output_slice,
                    params,
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
                                let &[m, _k, n] = &params[..] else { return Err(BackendError::Dispatch("matmul: expected params [M,K,N]".into())); };
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
                                let &[m, _k, n] = &params[..] else { return Err(BackendError::Dispatch("fused_matmul_add_relu: expected params [M,K,N]".into())); };
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
                                let &[m, _k, n] = &params[..] else { return Err(BackendError::Dispatch("matmul_relu: expected params [M,K,N]".into())); };
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
                                let &[m, k, n] = &params[..] else { return Err(BackendError::Dispatch("matmul_u4: expected params [M,K,N]".into())); };
                                let k_packed = k.div_ceil(8);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for p in 0..k_packed {
                                            let w = weights[i * k_packed + p];
                                            for nib in 0..8 {
                                                let idx = p * 8 + nib;
                                                if idx < k {
                                                    let nibble = (w >> (nib * 4)) & 0xF;
                                                    let signed = if nibble & 0x8 != 0 {
                                                        (nibble | 0xFFFFFFF0) as i32
                                                    } else {
                                                        nibble as i32
                                                    };
                                                    sum += (signed as f32) * activations[j * k + idx];
                                                }
                                            }
                                        }
                                        out_f32[i * n + j] = sum;
                                    }
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
                                let &[m, k, n] = &params[..] else { return Err(BackendError::Dispatch("matmul_u8: expected params [M,K,N]".into())); };
                                let k_packed = k.div_ceil(4);
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(&mut d[out_start..out_end])
                                };
                                for i in 0..m {
                                    for j in 0..n {
                                        let mut sum = 0.0f32;
                                        for p in 0..k_packed {
                                            let w = weights[i * k_packed + p];
                                            let bytes = w.to_le_bytes();
                                            for b in 0..4 {
                                                let idx = p * 4 + b;
                                                if idx < k {
                                                    sum += (bytes[b] as i8 as f32) * activations[j * k + idx];
                                                }
                                            }
                                        }
                                        out_f32[i * n + j] = sum;
                                    }
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
                                let out_f32 = {
                                    let d = arena.data_mut();
                                    bytemuck::cast_slice_mut::<_, f32>(
                                        &mut d[out_start..out_end]
                                    )
                                };
                                let num_groups = out_f32.len();
                                for g in 0..num_groups {
                                    let mut sum = 0.0f32;
                                    let start = g * group_size;
                                    let end = (start + group_size).min(input.len());
                                    for i in start..end {
                                        sum += input[i];
                                    }
                                    if is_mean == 1 {
                                        sum /= (end - start) as f32;
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
                                let &[m, n] = &params[..] else { return Err(BackendError::Dispatch("transpose_f32: expected params [M,N]".into())); };
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
