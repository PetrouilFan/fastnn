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
//! | MatMul (f32) | CPU fallback (GPU tiling TBD) |
//! | Softmax | Dedicated row-wise WGSL shader |
//! | Reduce (sum, mean, max) | Simple parallel-reduction WGSL shader |
//! | Transpose | 2D-transpose WGSL shader |
//! | Concat, Pad, Gather, Slice, BiasAdd, Norm, Pooling, Conv | CPU fallback |

#![allow(dead_code)]

use crate::backend::cpu::{CpuBackend, CpuBuffer};
use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction, MemoryPlan};
use crate::backends::wgpu::with_wgpu_context;
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
    if let Some(opcode) = elementwise_opcode(kernel_name) {
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
        let result = dispatch_elementwise_gpu(
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
            let result = dispatch_softmax_gpu(&input0, numel, axis)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        "reduce_f32" => {
            let group_size = resolved_params.first().copied().unwrap_or(1);
            let is_mean = params.get(1).copied().unwrap_or(0);
            let result = dispatch_reduce_gpu(&input0, group_size, is_mean)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        "matmul" | "matmul_relu" | "fused_matmul_add_relu" => {
            // GPU matmul TBD — fall back to CPU.
            Err(BackendError::UnsupportedOp(kernel_name.to_string()))
        }
        "transpose_f32" => {
            let m = resolved_params.first().copied().unwrap_or(1);
            let n = resolved_params.get(1).copied().unwrap_or(1);
            let result = dispatch_transpose_gpu(&input0, m, n)?;
            let out = arena.data_mut();
            let out_f32 = bytemuck::cast_slice_mut::<_, f32>(&mut out[out_start..out_end]);
            let len = out_f32.len().min(result.len());
            out_f32[..len].copy_from_slice(&result[..len]);
            Ok(())
        }
        _ => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
    }
}

// ─============================================================================
// Opcode mapping
// ─============================================================================

/// Map an element-wise kernel name to a numeric opcode for the unified WGSL shader.
fn elementwise_opcode(kernel_name: &str) -> Option<u32> {
    match kernel_name {
        "add_f32" => Some(0),
        "sub_f32" => Some(1),
        "mul_f32" => Some(2),
        "div_f32" => Some(3),
        "relu_f32" => Some(4),
        "gelu_f32" => Some(5),
        "silu_f32" => Some(6),
        "sigmoid_f32" => Some(7),
        "tanh_f32" => Some(8),
        "exp_f32" => Some(9),
        "log_f32" => Some(10),
        "sqrt_f32" => Some(11),
        "neg_f32" => Some(12),
        "abs_f32" => Some(13),
        "leaky_relu_f32" => Some(14),
        "elu_f32" => Some(15),
        "softplus_f32" => Some(16),
        "hardswish_f32" => Some(17),
        "clamp_f32" => Some(18),
        "sign_f32" => Some(19),
        "logical_not_f32" => Some(20),
        "log_softmax_f32" => Some(21),
        "max_f32" => Some(22),
        "min_f32" => Some(23),
        "pow_f32" => Some(24),
        "gt_scalar_f32" => Some(25),
        "lt_scalar_f32" => Some(26),
        "eq_scalar_f32" => Some(27),
        "add_scalar_f32" => Some(28),
        "mul_scalar_f32" => Some(29),
        "div_scalar_f32" => Some(30),
        "add_relu_f32" => Some(31),
        "sub_relu_f32" => Some(32),
        "mul_relu_f32" => Some(33),
        "div_relu_f32" => Some(34),
        "mish_f32" => Some(35),
        _ => None,
    }
}

// ─============================================================================
// GPU dispatch implementations (WGSL compute shaders)
// ─============================================================================

/// Dispatch an element-wise operation on GPU.
/// Returns the output as a `Vec<f32>` (read back from GPU).
fn dispatch_elementwise_gpu(
    input0: &[f32],
    input1: &[f32],
    numel: usize,
    opcode: u32,
    extra0: u32,
    extra1: u32,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader_src = build_elementwise_shader();
        let pipeline_key = format!("wgpu_backend_element_wise_{}", opcode);
        ensure_compute_pipeline(ctx, &format!("element_wise_{}", opcode), &shader_src)
            .map_err(|e| BackendError::Dispatch(e))?;

        // Upload inputs.
        let buf0 = ctx.create_buffer(bytemuck::cast_slice(input0), "ew_input0");
        let buf1 = ctx.create_buffer(bytemuck::cast_slice(input1), "ew_input1");

        // Output buffer.
        let output_size = (numel * 4) as u64;
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct EwParams {
            numel: u32,
            opcode: u32,
            extra0: u32,
            extra1: u32,
        }
        let params = EwParams {
            numel: numel as u32,
            opcode,
            extra0,
            extra1,
        };
        let params_buf = ctx.create_uniform_buffer(&params, "ew_params");

        // Look up pipeline after ensuring it exists above.
        let pipeline = &ctx.pipelines[&pipeline_key];

        // Bind group.
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ew_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf0.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch.
        let wg_count = (numel as u32).div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ew_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ew_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read back.
        let raw = ctx.read_buffer(&output_buf, output_size as usize);
        let result: &[f32] = bytemuck::cast_slice(&raw);
        Ok(result.to_vec())
    })
}


/// Dispatch softmax on GPU.
fn dispatch_softmax_gpu(
    input: &[f32],
    numel: usize,
    _axis: usize,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader = build_softmax_shader();
        let pipeline_key = "wgpu_backend_softmax";
        ensure_compute_pipeline(ctx, "softmax", &shader)
            .map_err(|e| BackendError::Dispatch(e))?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "sf_input");
        let output_size = (numel * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sf_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Row size uniform.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SfParams {
            numel: u32,
            row_size: u32,
        }
        let row_size = if numel > 0 { input.len() / numel } else { 1 };
        let params = SfParams {
            numel: numel as u32,
            row_size: row_size as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "sf_params");

        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sf_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc = (numel as u32).div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sf_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sf_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })
}

/// Dispatch parallel reduction (sum, mean) on GPU.
fn dispatch_reduce_gpu(
    input: &[f32],
    group_size: usize,
    is_mean: usize,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader = build_reduce_shader();
        let pipeline_key = "wgpu_backend_reduce";
        ensure_compute_pipeline(ctx, "reduce", &shader)
            .map_err(|e| BackendError::Dispatch(e))?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "rd_input");
        let num_groups = if group_size > 0 { input.len() / group_size } else { 1 };
        let output_size = (num_groups * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rd_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct RdParams {
            num_groups: u32,
            group_size: u32,
            is_mean: u32,
            _pad: u32,
        }
        let params = RdParams {
            num_groups: num_groups as u32,
            group_size: group_size as u32,
            is_mean: is_mean as u32,
            _pad: 0,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "rd_params");

        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc = (num_groups as u32).div_ceil(256);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rd_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rd_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc, 1, 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })
}

/// Dispatch 2D transpose on GPU.
fn dispatch_transpose_gpu(
    input: &[f32],
    m: usize,
    n: usize,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader = build_transpose_shader();
        let pipeline_key = "wgpu_backend_transpose";
        ensure_compute_pipeline(ctx, "transpose", &shader)
            .map_err(|e| BackendError::Dispatch(e))?;

        let buf_in = ctx.create_buffer(bytemuck::cast_slice(input), "tp_input");
        let output_size = (m * n * 4) as u64;
        let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tp_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TpParams {
            m: u32,
            n: u32,
        }
        let params = TpParams {
            m: m as u32,
            n: n as u32,
        };
        let buf_params = ctx.create_uniform_buffer(&params, "tp_params");

        let pipeline = &ctx.pipelines[pipeline_key];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tp_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            ],
        });

        let wgc_x = (m as u32).div_ceil(16);
        let wgc_y = (n as u32).div_ceil(16);
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tp_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tp_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wgc_x.max(1), wgc_y.max(1), 1);
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));

        let raw = ctx.read_buffer(&buf_out, output_size as usize);
        Ok(bytemuck::cast_slice(&raw).to_vec())
    })
}

// ─============================================================================
// WGSL shader source builders
// ─============================================================================

fn build_elementwise_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input0: array<f32>;
@group(0) @binding(1) var<storage, read>     input1: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct EwParams {
    numel: u32,
    opcode: u32,
    extra0: u32,
    extra1: u32,
}
@group(0) @binding(3) var<uniform> params: EwParams;

fn gelu_impl(x: f32) -> f32 {
    let x3 = x * x * x;
    let tanh_arg = 0.7978846 * (x + 0.044715 * x3);
    return 0.5 * x * (1.0 + tanh(tanh_arg));
}

fn hardswish_impl(x: f32) -> f32 {
    return x * max(min(x + 3.0, 6.0), 0.0) / 6.0;
}

fn log_softmax_impl(x: f32, row_max: f32, log_sum: f32) -> f32 {
    return (x - row_max) - log_sum;
}

fn softplus_impl(x: f32) -> f32 {
    return log(1.0 + exp(x));
}

fn apply_op(a: f32, b: f32, op: u32, e0: f32, e1: f32) -> f32 {
    switch op {
        case 0u: { return a + b; }       // add
        case 1u: { return a - b; }       // sub
        case 2u: { return a * b; }       // mul
        case 3u: { return a / b; }       // div
        case 4u: { return max(a, 0.0); } // relu
        case 5u: { return gelu_impl(a); }
        case 6u: { return a / (1.0 + exp(-a)); } // silu
        case 7u: { return 1.0 / (1.0 + exp(-a)); } // sigmoid
        case 8u: { return tanh(a); }
        case 9u: { return exp(a); }
        case 10u: { return log(a); }
        case 11u: { return sqrt(a); }
        case 12u: { return -a; }
        case 13u: { return abs(a); }
        case 14u: { return select(a * e0, a, a > 0.0); } // leaky_relu
        case 15u: { return select(exp(a) - 1.0, a, a > 0.0); } // elu
        case 16u: { return softplus_impl(a); }
        case 17u: { return hardswish_impl(a); }
        case 18u: { return clamp(a, e0, e1); }
        case 19u: { return sign(a); }
        case 20u: { return select(1.0, 0.0, a != 0.0); } // logical_not
        case 21u: { return a; } // log_softmax (handled separately)
        case 22u: { return max(a, b); }
        case 23u: { return min(a, b); }
        case 24u: { return pow(a, b); }
        case 25u: { return select(0.0, 1.0, a > b); } // gt
        case 26u: { return select(0.0, 1.0, a < b); } // lt
        case 27u: { return select(0.0, 1.0, abs(a - b) < 1.0e-6); } // eq
        case 28u: { return a + b; } // add_scalar (same as add)
        case 29u: { return a * b; } // mul_scalar (same as mul)
        case 30u: { return a / b; } // div_scalar (same as div)
        case 31u: { return max(a + b, 0.0); } // add_relu
        case 32u: { return max(a - b, 0.0); } // sub_relu
        case 33u: { return max(a * b, 0.0); } // mul_relu
        case 34u: { return max(a / b, 0.0); } // div_relu
        case 35u: { let sp = log(1.0 + exp(a)); return a * tanh(sp); } // mish
        default: { return a; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.numel) { return; }
    let a = input0[i];
    let b = select(a, input1[i], i < arrayLength(&input1));
    let e0 = bitcast<f32>(params.extra0);
    let e1 = bitcast<f32>(params.extra1);
    output[i] = apply_op(a, b, params.opcode, e0, e1);
}
"#
    .to_string()
}

fn build_softmax_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SfParams {
    numel: u32,
    row_size: u32,
}
@group(0) @binding(2) var<uniform> params: SfParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row * params.row_size >= params.numel) { return; }
    let start = row * params.row_size;
    let end = min(start + params.row_size, params.numel);

    // Find max
    var max_val: f32 = -3.402823e+38;
    for (var j: u32 = start; j < end; j = j + 1u) {
        max_val = max(max_val, input[j]);
    }

    // Compute sum of exp
    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + exp(input[j] - max_val);
    }

    // Normalize
    for (var j: u32 = start; j < end; j = j + 1u) {
        output[j] = exp(input[j] - max_val) / sum;
    }
}
"#
    .to_string()
}

fn build_reduce_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct RdParams {
    num_groups: u32,
    group_size: u32,
    is_mean: u32,
    _pad: u32,
}
@group(0) @binding(2) var<uniform> params: RdParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= params.num_groups) { return; }
    let start = g * params.group_size;
    let end = min(start + params.group_size, arrayLength(&input));

    var sum: f32 = 0.0;
    for (var j: u32 = start; j < end; j = j + 1u) {
        sum = sum + input[j];
    }
    if (params.is_mean == 1u) {
        sum = sum / f32(end - start);
    }
    output[g] = sum;
}
"#
    .to_string()
}

fn build_transpose_shader() -> String {
    r#"
@group(0) @binding(0) var<storage, read>     input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct TpParams {
    m: u32,
    n: u32,
}
@group(0) @binding(2) var<uniform> params: TpParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= params.m || j >= params.n) { return; }
    output[j * params.m + i] = input[i * params.n + j];
}
"#
    .to_string()
}

// ─============================================================================
// Pipeline cache helper
// ─============================================================================

/// Ensure a compute pipeline exists for the given key and WGSL source.
/// Uses the global WgpuContext's device and bind-group layout.
fn ensure_compute_pipeline(
    ctx: &mut crate::backends::wgpu::WgpuContext,
    key: &str,
    wgsl_source: &str,
) -> Result<(), String> {
    let pipeline_key = format!("wgpu_backend_{}", key);
    if !ctx.pipelines.contains_key(&pipeline_key) {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&pipeline_key),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });

        // 4-bind-group layout: 2 storage read, 1 storage write, 1 uniform.
        let layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_layout", pipeline_key)),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{}_pl_layout", pipeline_key)),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&pipeline_key),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        ctx.pipelines.insert(pipeline_key, pipeline);
    }
    Ok(())
}
