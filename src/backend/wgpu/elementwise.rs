use crate::backend::wgpu::context::with_wgpu_context;
use crate::backend::BackendError;

pub(super) fn elementwise_opcode(kernel_name: &str) -> Option<u32> {
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
        "erf_f32" => Some(36),
        _ => None,
    }
}

pub(super) fn dispatch_elementwise_gpu(
    input0: &[f32],
    input1: &[f32],
    numel: usize,
    opcode: u32,
    extra0: u32,
    extra1: u32,
) -> Result<Vec<f32>, BackendError> {
    with_wgpu_context(|ctx| -> Result<Vec<f32>, BackendError> {
        let shader_src = build_elementwise_shader();
        super::pipeline::ensure_compute_pipeline(
            ctx,
            &format!("element_wise_{}", opcode),
            &shader_src,
        )
        .map_err(BackendError::Dispatch)?;

        let buf0 = ctx.create_buffer(bytemuck::cast_slice(input0), "ew_input0");
        let buf1 = ctx.create_buffer(bytemuck::cast_slice(input1), "ew_input1");

        let output_size = (numel * 4) as u64;
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ew_output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

        let pipeline_key = format!("wgpu_backend_element_wise_{}", opcode);
        let pipeline = &ctx.pipelines[&pipeline_key];

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

        let raw = ctx.read_buffer(&output_buf, output_size as usize);
        let result: &[f32] = bytemuck::cast_slice(&raw);
        Ok(result.to_vec())
    })
}

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

fn erf_impl(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.3275911 * abs(x));
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return select(y, -y, x < 0.0);
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
        case 36u: { return erf_impl(a); }
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
