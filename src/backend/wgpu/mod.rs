//! Wgpu (GPU) backend — implements [`Backend`] for GPU compute via wgpu.
//!
//! # Design
//!
//! - **compile()** reuses [`CpuBackend::compile`] (the lowering is backend-agnostic).
//! - **Buffer** is `CpuBuffer` (same interior-mutability `Vec<u8>` arena) because wgpu
//!   device memory is managed separately — the arena acts as a host-side staging/accounting
//!   buffer so that [`GraphExecutor`](crate::backend::executor::GraphExecutor) can work
//!   uniformly with both backends.
//! - **dispatch()** uploads input slices to GPU, runs a compute shader, and downloads
//!   the output back into the arena.  `MemCopy`, `Fill`, and `WriteConst` are executed
//!   on the host side (same as CPU).
//! - For ops without a dedicated GPU shader, a CPU fallback path is used (copy → dispatch
//!   on CpuBackend → copy back).
//!
//! # Supported GPU ops
//!
//! | Category | Shader |
//! |----------|--------|
//! | All element-wise (add, sub, mul, div, ReLU, GELU, SiLU, sigmoid, tanh, exp, log, sqrt, neg, abs, leaky_relu, ELU, softplus, hardswish, clamp, sign, logical_not, max, min, log_softmax, pow, scalar-cmp, scalar-arithmetic, fused-Op+ReLU) | `element_wise` unified WGSL shader (opcode-selector) |
//! | MatMul (f32) | Tiled WGSL shader (16×16 workgroups) |
//! | Conv2d | Direct convolution WGSL shader |
//! | Softmax | Dedicated row-wise WGSL shader |
//! | Reduce (sum, mean, max) | Simple parallel-reduction WGSL shader |
//! | Transpose | 2D-transpose WGSL shader |
//! | LayerNorm / RMSNorm | Dedicated (mean+var+normalize) WGSL shader |
//! | MaxPool / AvgPool | Dedicated pooling WGSL shader |
//! | Embedding | Gather WGSL shader |
//! | Concat, Pad, Gather, Slice, BiasAdd, BatchNorm | CPU fallback |

#![allow(dead_code)]

pub mod context;
mod argmax;
mod conv;
mod elementwise;
mod embed;
mod matmul;
mod norm;
mod pipeline;
mod pool;
mod reduce;
mod softmax;
mod transpose;

use crate::backend::cpu::{CpuBackend, CpuBuffer};
use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction, MemoryPlan};
use crate::ir::node::{ComputeGraph, DimExpr, ShapeEnv};
use bytemuck;
use std::cell::UnsafeCell;
use std::sync::atomic::Ordering;

// ─============================================================================
// Buffer type
// ─============================================================================

/// Host-side memory arena used by [`WgpuBackend`].
///
/// Same interior-mutability pattern as [`CpuBuffer`]: the executor writes input
/// data via `write_arena`, dispatch reads/writes via interior mutability, and
/// `read_arena` extracts outputs.
pub struct WgpuBuffer(UnsafeCell<Vec<u8>>);

impl WgpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        WgpuBuffer(UnsafeCell::new(data))
    }

    /// Get a mutable slice to the arena data.
    pub fn data_mut(&self) -> &mut [u8] {
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }
}

unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}

// ─============================================================================
// WgpuBackend
// ─============================================================================

/// GPU backend that dispatches compute shaders via wgpu.
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    type Buffer = WgpuBuffer;

    fn name(&self) -> &str {
        "wgpu"
    }

    fn allocate_arena(&self, total_bytes: usize) -> WgpuBuffer {
        WgpuBuffer::new(vec![0u8; total_bytes])
    }

    // ── Compile ──────────────────────────────────────────────────────────
    //
    // Reuse CpuBackend::compile — the instruction-lowering logic is identical
    // because Instructions are backend-agnostic.
    fn compile(
        &self,
        graph: &ComputeGraph,
        memory_plan: &MemoryPlan,
    ) -> Result<ExecutablePlan, BackendError> {
        let cpu = CpuBackend;
        cpu.compile(graph, memory_plan)
    }

    // ── Dispatch ─────────────────────────────────────────────────────────
    fn dispatch(
        &self,
        plan: &ExecutablePlan,
        arena: &WgpuBuffer,
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

                    // Try GPU dispatch first; fall back to CPU if unsupported.
                    if let Err(err) = try_gpu_dispatch(
                        arena,
                        kernel_name,
                        input_slices,
                        *output_slice,
                        params,
                        param_dims,
                        shape_env,
                        weight_meta,
                    ) {
                        // CPU fallback: copy affected slices to a temporary
                        // CpuBuffer, run CpuBackend dispatch on this single
                        // instruction, copy output back.
                        let mut tmp = vec![0u8; plan.arena_size];
                        for s in input_slices {
                            let end = (s.offset + s.size).min(arena.data_mut().len());
                            tmp[s.offset..end]
                                .copy_from_slice(&arena.data_mut()[s.offset..end]);
                        }
                        // Output slot may overlap with input; copy it too.
                        let o_end =
                            (out_start + output_slice.size).min(arena.data_mut().len());
                        tmp[out_start..o_end]
                            .copy_from_slice(&arena.data_mut()[out_start..o_end]);

                        let cpu_buf = crate::backend::cpu::CpuBuffer::new(tmp);
                        let cpu = CpuBackend;
                        let single_plan = ExecutablePlan {
                            instructions: vec![instr.clone()],
                            arena_size: plan.arena_size,
                        };
                        cpu.dispatch(&single_plan, &cpu_buf, shape_env)?;

                        // Copy output back to real arena.
                        let cpu_data = cpu_buf.data_mut();
                        let wgpu_data = arena.data_mut();
                        wgpu_data[out_start..o_end]
                            .copy_from_slice(&cpu_data[out_start..o_end]);
                    }
                }
                Instruction::MemCopy { dst, src } => {
                    let data = arena.data_mut();
                    let len = dst.size.min(src.size);
                    let src_start = src.offset;
                    let dst_start = dst.offset;
                    data.copy_within(src_start..src_start + len, dst_start);
                }
                Instruction::Fill { dst, value } => {
                    let data = arena.data_mut();
                    let f32_slice = bytemuck::cast_slice_mut::<_, f32>(
                        &mut data[dst.offset..dst.offset + dst.size],
                    );
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

    fn write_arena(&self, arena: &WgpuBuffer, offset: usize, data: &[u8]) {
        let buf = arena.data_mut();
        let end = (offset + data.len()).min(buf.len());
        buf[offset..end].copy_from_slice(&data[..end - offset]);
    }

    fn read_arena(&self, arena: &WgpuBuffer, offset: usize, size: usize) -> Vec<u8> {
        let buf = arena.data_mut();
        let end = (offset + size).min(buf.len());
        buf[offset..end].to_vec()
    }
}

// ─============================================================================
// GPU dispatch helpers
// ─============================================================================

/// Try to dispatch a kernel on GPU.  Returns `Err` if the kernel is not
/// supported (caller will fall back to CPU).
fn try_gpu_dispatch(
    arena: &WgpuBuffer,
    kernel_name: &str,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<(f32, f32, Vec<usize>)>,
) -> Result<(), BackendError> {
    let out_start = output_slice.offset;
    let out_end = output_slice.offset + output_slice.size;

    // Resolve symbolic params if present.
    let resolved_params = if let Some(dims) = param_dims {
        let n = params.len().min(dims.len());
        let mut p = params.to_vec();
        for i in 0..n {
            if let Ok(v) = dims[i].evaluate_with_env(shape_env) {
                p[i] = v as usize;
            }
        }
        p
    } else {
        params.to_vec()
    };

    // Read input data from arena.
    let read_input = |idx: usize| -> Vec<f32> {
        if idx < input_slices.len() {
            let s = &input_slices[idx];
            let d = arena.data_mut();
            bytemuck::cast_slice::<_, f32>(&d[s.offset..s.offset + s.size]).to_vec()
        } else {
            Vec::new()
        }
    };

    let input0 = read_input(0);
    let input1 = read_input(1);
    let _input2 = read_input(2); // for bias, etc.
    let numel = input0.len();

    // Dispatch based on kernel category.
    if let Some(opcode) = elementwise::elementwise_opcode(kernel_name) {
        let extra0 = if kernel_name == "leaky_relu_f32" {
            params.first().copied().unwrap_or(f32::to_bits(0.01) as usize)
        } else if kernel_name == "clamp_f32" {
            params.first().copied().unwrap_or(f32::to_bits(0.0) as usize)
        } else {
            0
        };
        let extra1 = if kernel_name == "clamp_f32" {
            params.get(1).copied().unwrap_or(f32::to_bits(1.0) as usize)
        } else {
            0
        };
        let result = elementwise::dispatch_elementwise_gpu(
            &input0,
            &input1,
            numel,
            opcode,
            extra0 as u32,
            extra1 as u32,
        )?;
        let out = arena.data_mut();
        let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
        let len = out_f32.len().min(result.len());
        out_f32[..len].copy_from_slice(&result[..len]);
        return Ok(());
    }

    match kernel_name {
        "softmax" => {
            let axis = params.first().copied().unwrap_or(0);
            let result = softmax::dispatch_softmax_gpu(&input0, numel, axis)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        "reduce_f32" => {
            let group_size = resolved_params.first().copied().unwrap_or(1);
            let is_mean = params.get(1).copied().unwrap_or(0);
            let result = reduce::dispatch_reduce_gpu(&input0, group_size, is_mean)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        "matmul" | "matmul_relu" | "fused_matmul_add_relu" => {
            matmul::dispatch_matmul_gpu(arena, input_slices, output_slice, &resolved_params, shape_env)
        }
        "transpose_f32" => {
            let m = resolved_params.first().copied().unwrap_or(1);
            let n = resolved_params.get(1).copied().unwrap_or(1);
            let result = transpose::dispatch_transpose_gpu(&input0, m, n)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        "conv2d" => {
            conv::dispatch_conv_gpu(arena, input_slices, output_slice, &resolved_params, shape_env)
        }
        "norm_f32" | "rms_norm" => {
            norm::dispatch_norm_gpu(arena, kernel_name, input_slices, output_slice, &resolved_params)
        }
        "pool_f32" => {
            pool::dispatch_pool_gpu(arena, input_slices, output_slice, &resolved_params)
        }
        "embedding" => {
            embed::dispatch_embed_gpu(arena, input_slices, output_slice, &resolved_params)
        }
        "argmax" => {
            argmax::dispatch_argmax_gpu(arena, input_slices, output_slice)
        }
        _ => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
    }
}
