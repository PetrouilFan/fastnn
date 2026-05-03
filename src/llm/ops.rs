use crate::quants::quantized_tensor::GgmlQuantizedTensor;

#[derive(Clone)]
pub struct NormParams {
    pub eps: f32,
    pub weight: Vec<f32>,
}

#[derive(Clone)]
pub struct FFNParams {
    pub gate_weight: GgmlQuantizedTensor,
    pub up_weight: GgmlQuantizedTensor,
    pub down_weight: GgmlQuantizedTensor,
    pub intermediate_size: usize,
    pub hidden_size: usize,
}

pub struct Kernels;

impl Kernels {
    const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
    const GELU_COEF_A: f32 = 0.044715;

    pub fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (Self::SQRT_2_OVER_PI * x * (1.0 + Self::GELU_COEF_A * x * x)).tanh())
    }
    pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let mut sum_squares = 0.0f32;

        for val in x.iter() {
            sum_squares += val * val;
        }

        let scale = 1.0 / (sum_squares / n as f32 + eps).sqrt();

        for (i, val) in x.iter_mut().enumerate() {
            *val = *val * scale * weight[i];
        }
    }

    pub fn rmsnorm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let mut sum_squares = 0.0f32;

        for val in x.iter() {
            sum_squares += val * val;
        }

        let scale = 1.0 / (sum_squares / n as f32 + eps).sqrt();

        for (i, val) in x.iter_mut(). enumerate() {
            *val = *val * scale * weight[i];
        }
    }

    pub fn rope(x: &mut [f32], theta: f32, pos: usize, head_dim: usize, freq_factors: Option<&[f32]>) {
        let inv_theta = 1.0 / theta;

        for i in (0..head_dim).step_by(2) {
            if i + 1 >= head_dim {
                break;
            }

            let freq = inv_theta.powf(2.0 * (i as f32) / head_dim as f32);
            let ff = freq_factors.map_or(1.0f32, |f| f[i / 2]);
            let angle = (freq / ff) * pos as f32;
            let cos = angle.cos();
            let sin = angle.sin();

            let x0 = x[i];
            let x1 = x[i + 1];

            x[i] = x0 * cos - x1 * sin;
            x[i + 1] = x0 * sin + x1 * cos;
        }
    }

    pub fn softmax(x: &mut [f32]) {
        let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum = 0.0f32;
        for val in x.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        for val in x.iter_mut() {
            *val /= sum;
        }
    }

    pub fn silu(x: &mut [f32]) {
        for val in x.iter_mut() {
            *val = *val / (1.0 + (-*val).exp());
        }
    }

    pub fn ffn_gate_up(_x: &mut [f32], _gate: &[f32], _up: &[f32], _intermediate_size: usize, _hidden_size: usize) {
    }

    pub fn gemv_fused_gate_up_silu(
        x: &mut [f32],
        gate_weight: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
        intermediate_size: usize,
        hidden_size: usize,
        _scratch: &mut [f32],
    ) {
        let mut gate_out = vec![0.0; hidden_size];
        let mut up_out = vec![0.0; hidden_size];

        for i in 0..hidden_size {
            let mut g_sum = 0.0f32;
            let mut u_sum = 0.0f32;
            for k in 0..intermediate_size {
                g_sum += x[k] * gate_weight[k * hidden_size + i];
                u_sum += x[k] * up_weight[k * hidden_size + i];
            }
            gate_out[i] = g_sum;
            up_out[i] = u_sum;
        }

        for i in 0..hidden_size {
            let gate_val = gate_out[i];
            let up_val = up_out[i];
            x[i] = gate_val / (1.0 + (-up_val).exp());
        }

        let mut down_out = vec![0.0; intermediate_size];
        for i in 0..intermediate_size {
            let mut sum = 0.0f32;
            for k in 0..hidden_size {
                sum += x[k] * down_weight[k * intermediate_size + i];
            }
            down_out[i] = sum;
        }

        x[..intermediate_size].copy_from_slice(&down_out[..intermediate_size]);
    }

    pub fn apply_q_norm(x: &mut [f32], q_norm: &[f32]) {
        let head_dim = q_norm.len();
        if head_dim == 0 {
            return;
        }
        let num_heads = x.len() / head_dim;
        for h in 0..num_heads {
            let offset = h * head_dim;
            let slice = &mut x[offset..offset + head_dim];
            let mut sum_sq = 0.0f32;
            for v in slice.iter() {
                sum_sq += v * v;
            }
            let scale = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
            for i in 0..head_dim {
                slice[i] = slice[i] * scale * q_norm[i];
            }
        }
    }

    pub fn apply_v_norm(x: &mut [f32], head_dim: usize) {
        if head_dim == 0 {
            return;
        }
        let num_heads = x.len() / head_dim;
        for h in 0..num_heads {
            let offset = h * head_dim;
            let slice = &mut x[offset..offset + head_dim];
            let mut sum_sq = 0.0f32;
            for v in slice.iter() {
                sum_sq += v * v;
            }
            let scale = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }
    }
}