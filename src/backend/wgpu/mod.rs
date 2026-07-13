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
//! | Quantized MatMul (matmul_u4/u8) | GPU quantized GEMM (per-channel unpack + dot) |
//! | Quantized Conv2d (conv2d_u4/u8) | CPU im2col + GPU quantized GEMM |

mod argmax;
pub mod context;
mod conv;
mod elementwise;
mod embed;
mod matmul;
mod norm;
mod pipeline;
mod pool;
mod quantized;
mod reduce;
mod softmax;
mod transpose;

use crate::backend::cpu::CpuBackend;
use crate::backend::{Backend, BackendError, BufferSlice, ExecutablePlan, Instruction};
use crate::compiler::MemoryPlan;
use crate::ir::{ComputeGraph, DimExpr, NodeId, ShapeEnv};
use bytemuck;
use context::{with_wgpu_context, WgpuContext};
use std::cell::UnsafeCell;

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
    #[allow(clippy::mut_from_ref)]
    pub fn data_mut(&self) -> &mut [u8] {
        // SAFETY: The `UnsafeCell` gives `&mut` to the inner `Vec<u8>`. This is safe because
        // `data_mut()` returns a borrow that is never aliased — dispatch processes instructions
        // sequentially and each borrow ends before the next begins.
        unsafe { &mut *self.0.get() }.as_mut_slice()
    }
}

// SAFETY: `WgpuBuffer` uses interior mutability via `UnsafeCell` but all
// access is properly synchronized (GPU command queues / CPU thread isolation).
// The `data` field is read-only from shared references; mutation requires `&mut self`.
unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}

// ─============================================================================
// Batched readback
// ─============================================================================

/// A deferred GPU-buffer readback request.
/// After all compute passes are recorded into the shared encoder and submitted,
/// the runtime batch-reads all `PendingRead` entries and copies the results
/// into the host arena at `cpu_offset`.
pub(super) struct PendingRead {
    pub buffer: wgpu::Buffer,
    pub cpu_offset: usize,
    pub size: usize,
}

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
        with_wgpu_context(|ctx| {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("dispatch"),
                });
            let mut pending_reads: Vec<PendingRead> = Vec::new();

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
                        node_id,
                        ..
                    } => {
                        let out_start = output_slice.offset;
                        let _out_end = output_slice.offset + output_slice.size;

                        // Try GPU dispatch first; fall back to CPU if unsupported.
                        if let Err(_err) = try_gpu_dispatch(
                            ctx,
                            &mut encoder,
                            &mut pending_reads,
                            arena,
                            kernel_name,
                            node_id.as_ref(),
                            input_slices,
                            *output_slice,
                            params,
                            param_dims,
                            shape_env,
                            weight_meta,
                        ) {
                            // CPU fallback: copy only the byte range touched
                            // by this instruction to a temporary buffer, run
                            // CPU dispatch, then copy output back.
                            let arena_len = arena.data_mut().len();
                            let o_end = (out_start + output_slice.size).min(arena_len);

                            let mut min_offset = out_start;
                            let mut max_end = o_end;
                            for s in input_slices {
                                let end = (s.offset + s.size).min(arena_len);
                                min_offset = min_offset.min(s.offset);
                                max_end = max_end.max(end);
                            }

                            let range_len = max_end - min_offset;
                            let mut tmp = vec![0u8; range_len];
                            tmp.copy_from_slice(&arena.data_mut()[min_offset..max_end]);

                            let cpu_buf = crate::backend::cpu::CpuBuffer::new(tmp);
                            let cpu = CpuBackend;

                            // Adjust offsets into the temp buffer range.
                            let adjusted_inputs: Vec<BufferSlice> = input_slices
                                .iter()
                                .map(|s| BufferSlice {
                                    offset: s.offset - min_offset,
                                    size: s.size,
                                })
                                .collect();
                            let adjusted_output = BufferSlice {
                                offset: output_slice.offset - min_offset,
                                size: output_slice.size,
                            };
                            let adjusted_secondary =
                                secondary_output_slice.as_ref().map(|s| BufferSlice {
                                    offset: s.offset - min_offset,
                                    size: s.size,
                                });

                            let single_plan = ExecutablePlan {
                                instructions: vec![Instruction::CallKernel {
                                    kernel_name: kernel_name.clone(),
                                    input_slices: adjusted_inputs,
                                    output_slice: adjusted_output,
                                    secondary_output_slice: adjusted_secondary,
                                    params: params.clone(),
                                    param_dims: param_dims.clone(),
                                    weight_meta: weight_meta.clone(),
                                    node_id: node_id.clone(),
                                }],
                                arena_size: range_len,
                                levels: vec![0],
                            };
                            cpu.dispatch(&single_plan, &cpu_buf, shape_env)?;

                            // Copy output back to real arena.
                            let cpu_data = cpu_buf.data_mut();
                            let wgpu_data = arena.data_mut();
                            let adj_out_start = output_slice.offset - min_offset;
                            let adj_out_end = o_end - min_offset;
                            wgpu_data[out_start..o_end]
                                .copy_from_slice(&cpu_data[adj_out_start..adj_out_end]);
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

            // Submit all recorded compute work.
            ctx.queue.submit(std::iter::once(encoder.finish()));

            // Batch-read all deferred readbacks and write results into the arena.
            if !pending_reads.is_empty() {
                let buf_refs: Vec<(&wgpu::Buffer, usize)> = pending_reads
                    .iter()
                    .map(|pr| (&pr.buffer, pr.size))
                    .collect();
                let results = ctx.read_buffers(&buf_refs);
                let data = arena.data_mut();
                for (pr, result) in pending_reads.iter().zip(results.iter()) {
                    let end = (pr.cpu_offset + result.len()).min(data.len());
                    data[pr.cpu_offset..end].copy_from_slice(&result[..end - pr.cpu_offset]);
                }
                // Release output buffers back to pool for reuse.
                for pr in pending_reads.drain(..) {
                    ctx.release_buffer_to_pool(pr.buffer, pr.size);
                }
            }

            Ok(())
        })
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
#[allow(unused_variables, clippy::too_many_arguments)]
fn try_gpu_dispatch(
    ctx: &mut WgpuContext,
    encoder: &mut wgpu::CommandEncoder,
    pending_reads: &mut Vec<PendingRead>,
    arena: &WgpuBuffer,
    kernel_name: &str,
    node_id: Option<&NodeId>,
    input_slices: &[BufferSlice],
    output_slice: BufferSlice,
    params: &[usize],
    param_dims: &Option<Vec<DimExpr>>,
    shape_env: &ShapeEnv,
    weight_meta: &Option<std::sync::Arc<crate::backend::QuantizedWeightMeta>>,
) -> Result<(), BackendError> {
    let out_start = output_slice.offset;

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
    let _input2 = read_input(2);
    let numel = input0.len();

    if let Some(opcode) = elementwise::elementwise_opcode(kernel_name) {
        let extra0 = if kernel_name == "leaky_relu_f32" {
            params
                .first()
                .copied()
                .unwrap_or(f32::to_bits(0.01) as usize)
        } else if kernel_name == "clamp_f32" {
            params
                .first()
                .copied()
                .unwrap_or(f32::to_bits(0.0) as usize)
        } else {
            0
        };
        let extra1 = if kernel_name == "clamp_f32" {
            params.get(1).copied().unwrap_or(f32::to_bits(1.0) as usize)
        } else {
            0
        };
        return elementwise::dispatch_elementwise_gpu(
            ctx,
            encoder,
            pending_reads,
            &input0,
            &input1,
            numel,
            opcode,
            extra0 as u32,
            extra1 as u32,
            out_start,
        );
    }

    match kernel_name {
        "softmax" => {
            let axis_dim = params.first().copied().unwrap_or(1);
            softmax::dispatch_softmax_gpu(
                ctx,
                encoder,
                pending_reads,
                &input0,
                numel,
                axis_dim,
                out_start,
            )
        }
        "reduce_f32" => {
            let group_size = resolved_params.first().copied().unwrap_or(1);
            let is_mean = params.get(1).copied().unwrap_or(0);
            reduce::dispatch_reduce_gpu(
                ctx,
                encoder,
                pending_reads,
                &input0,
                group_size,
                is_mean,
                out_start,
            )
        }
        "matmul" => matmul::dispatch_matmul_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
        ),
        "matmul_relu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "relu",
            false,
        ),
        "matmul_gelu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "gelu",
            false,
        ),
        "matmul_silu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "silu",
            false,
        ),
        "fused_matmul_add_relu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "relu",
            true,
        ),
        "fused_matmul_add_gelu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "gelu",
            true,
        ),
        "fused_matmul_add_silu" => matmul::dispatch_matmul_activation_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
            "silu",
            true,
        ),
        "transpose_f32" => {
            let m = resolved_params.first().copied().unwrap_or(1);
            let n = resolved_params.get(1).copied().unwrap_or(1);
            transpose::dispatch_transpose_gpu(ctx, encoder, pending_reads, &input0, m, n, out_start)
        }
        "conv2d" => conv::dispatch_conv_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
            shape_env,
        ),
        "norm_f32" | "rms_norm" => norm::dispatch_norm_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            kernel_name,
            input_slices,
            output_slice,
            &resolved_params,
        ),
        "pool_f32" => pool::dispatch_pool_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
        ),
        "embedding" => embed::dispatch_embed_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
            &resolved_params,
        ),
        "argmax" => argmax::dispatch_argmax_gpu(
            ctx,
            encoder,
            pending_reads,
            arena,
            input_slices,
            output_slice,
        ),
        "matmul_i4" | "matmul_i8" | "matmul_f4" | "matmul_f8" | "matmul_f8r" => {
            let dtype_tag = match kernel_name {
                "matmul_i4" => "i4",
                "matmul_i8" => "i8",
                "matmul_f4" => "f4",
                "matmul_f8" => "f8",
                "matmul_f8r" => "f8r",
                _ => unreachable!(),
            };
            let scales = weight_meta
                .as_ref()
                .map(|m| m.scales.clone())
                .unwrap_or_default();
            let zero_points = weight_meta
                .as_ref()
                .map(|m| m.dequant_offsets.clone())
                .unwrap_or_default();
            quantized::dispatch_quantized_matmul_gpu(
                ctx,
                encoder,
                pending_reads,
                arena,
                input_slices,
                output_slice,
                &resolved_params,
                dtype_tag,
                &scales,
                &zero_points,
            )
        }
        "conv2d_i4" | "conv2d_i8" | "conv2d_f4" | "conv2d_f8" | "conv2d_f8r" => {
            let dtype_tag = match kernel_name {
                "conv2d_i4" => "i4",
                "conv2d_i8" => "i8",
                "conv2d_f4" => "f4",
                "conv2d_f8" => "f8",
                "conv2d_f8r" => "f8r",
                _ => unreachable!(),
            };
            let scales = weight_meta
                .as_ref()
                .map(|m| m.scales.clone())
                .unwrap_or_default();
            let zero_points = weight_meta
                .as_ref()
                .map(|m| m.dequant_offsets.clone())
                .unwrap_or_default();
            quantized::dispatch_quantized_conv_gpu(
                ctx,
                encoder,
                pending_reads,
                arena,
                input_slices,
                output_slice,
                &resolved_params,
                dtype_tag,
                &scales,
                &zero_points,
            )
        }
        "quantize_gradient_f32_to_f8x4r" => {
            let numel = resolved_params.first().copied().unwrap_or(0);
            if let Some(input_slice) = input_slices.first() {
                quantized::dispatch_quantize_gradient_gpu(
                    ctx,
                    encoder,
                    pending_reads,
                    arena,
                    *input_slice,
                    output_slice,
                    numel,
                )
            } else {
                Err(BackendError::Dispatch(
                    "quantize_gradient: missing input".into(),
                ))
            }
        }
        "dequantize_gradient_f8x4r_to_f32" => {
            let numel = resolved_params.first().copied().unwrap_or(0);
            if let Some(input_slice) = input_slices.first() {
                quantized::dispatch_dequantize_gradient_gpu(
                    ctx,
                    encoder,
                    pending_reads,
                    arena,
                    *input_slice,
                    output_slice,
                    numel,
                )
            } else {
                Err(BackendError::Dispatch(
                    "dequantize_gradient: missing input".into(),
                ))
            }
        }
        "upsample_nearest2d" | "upsample_bilinear2d" => {
            Err(BackendError::UnsupportedOp(kernel_name.to_string()))
        }
        "adaptive_avg_pool2d" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        "repeat" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        "cumsum" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        "erf_f32" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        "flip" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        "where_f32" => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
        _ => Err(BackendError::UnsupportedOp(kernel_name.to_string())),
    }
}
