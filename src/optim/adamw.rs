use crate::optim::{Optimizer, OptimizerState, ParamGroup, ParamState};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

pub struct AdamW {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub v_hat: Vec<Tensor>,
    pub step: Vec<u64>,
    // Track which parameters should skip weight decay (e.g., biases, LayerNorm)
    pub no_decay: Vec<bool>,
    // Pre-allocated buffers to avoid clones
    pub temp_grad_scaled: Vec<Tensor>,
    pub temp_grad_sq: Vec<Tensor>,
    pub temp_m_hat: Vec<Tensor>,
    pub temp_v_hat: Vec<Tensor>,
    pub temp_update: Vec<Tensor>,
    // Fused bias corrections
    pub bias_correction1: Vec<f64>,
    pub bias_correction2: Vec<f64>,
}

impl AdamW {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Self {
        let n = params.len();
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let step: Vec<u64> = vec![0; n];
        // By default, all parameters get weight decay
        let no_decay = vec![false; n];
        // Pre-allocate buffers
        let temp_grad_scaled: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_sq: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_m_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let bias_correction1 = vec![1.0; n];
        let bias_correction2 = vec![1.0; n];

        AdamW {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m,
            v,
            v_hat,
            step,
            no_decay,
            temp_grad_scaled,
            temp_grad_sq,
            temp_m_hat,
            temp_v_hat,
            temp_update,
            bias_correction1,
            bias_correction2,
        }
    }

    /// Add parameters that should skip weight decay (e.g., biases, LayerNorm weights)
    #[allow(dead_code)]
    pub fn add_no_decay(&mut self, indices: &[usize]) {
        for &idx in indices {
            if idx < self.no_decay.len() {
                self.no_decay[idx] = true;
            }
        }
    }

    /// Mark all 1D parameters (biases) to skip weight decay
    #[allow(dead_code)]
    pub fn mark_biases_no_decay(&mut self) {
        for (i, param) in self.params.iter().enumerate() {
            if param.ndim() == 1 {
                self.no_decay[i] = true;
            }
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            self.step[i] += 1;

            self.bias_correction1[i] = 1.0 - beta1 as f64 * (1.0 - self.bias_correction1[i]);
            self.bias_correction2[i] = 1.0 - beta2 as f64 * (1.0 - self.bias_correction2[i]);

            let beta1_c = 1.0 - beta1;
            let beta2_c = 1.0 - beta2;

            let numel = param.inner.numel() as usize;
            let param_ptr = param.data_ptr_f32_mut();
            let grad_ptr = grad.data_ptr_f32();
            let m_ptr = self.m[i].data_ptr_f32_mut();
            let v_ptr = self.v[i].data_ptr_f32_mut();

            if self.amsgrad {
                let v_hat_ptr = self.v_hat[i].data_ptr_f32_mut();
                for j in 0..numel {
                    unsafe {
                        let g = *grad_ptr.add(j);
                        let mut m_val = *m_ptr.add(j);
                        let mut v_val = *v_ptr.add(j);

                        m_val = beta1 * m_val + beta1_c * g;
                        *m_ptr.add(j) = m_val;

                        let g_sq = g * g;
                        v_val = beta2 * v_val + beta2_c * g_sq;
                        *v_ptr.add(j) = v_val;

                        let m_hat = m_val / self.bias_correction1[i] as f32;

                        let mut v_hat_val = v_val / self.bias_correction2[i] as f32;
                        let existing_v_hat = *v_hat_ptr.add(j);
                        if v_hat_val > existing_v_hat {
                            *v_hat_ptr.add(j) = v_hat_val;
                        } else {
                            v_hat_val = existing_v_hat;
                        }

                        let update = m_hat / (v_hat_val.sqrt() + eps);

                        let mut p_val = *param_ptr.add(j);
                        if weight_decay != 0.0 && !self.no_decay.get(i).copied().unwrap_or(false) {
                            p_val *= 1.0 - lr * weight_decay;
                        }

                        p_val -= lr * update;
                        *param_ptr.add(j) = p_val;
                    }
                }
            } else {
                for j in 0..numel {
                    unsafe {
                        let g = *grad_ptr.add(j);
                        let mut m_val = *m_ptr.add(j);
                        let mut v_val = *v_ptr.add(j);

                        m_val = beta1 * m_val + beta1_c * g;
                        *m_ptr.add(j) = m_val;

                        let g_sq = g * g;
                        v_val = beta2 * v_val + beta2_c * g_sq;
                        *v_ptr.add(j) = v_val;

                        let m_hat = m_val / self.bias_correction1[i] as f32;
                        let v_hat_val = v_val / self.bias_correction2[i] as f32;
                        let update = m_hat / (v_hat_val.sqrt() + eps);

                        let mut p_val = *param_ptr.add(j);
                        if weight_decay != 0.0 && !self.no_decay.get(i).copied().unwrap_or(false) {
                            p_val *= 1.0 - lr * weight_decay;
                        }

                        p_val -= lr * update;
                        *param_ptr.add(j) = p_val;
                    }
                }
            }

            param.set_grad(None);
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params.iter_mut() {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        let temp_grad_scaled: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_sq: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_m_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_v_hat: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let bias_correction1 = vec![1.0; params.len()];
        let bias_correction2 = vec![1.0; params.len()];

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.step.extend(vec![0u64; params.len()]);
        self.no_decay.extend(vec![false; params.len()]);
        self.temp_grad_scaled.extend(temp_grad_scaled);
        self.temp_grad_sq.extend(temp_grad_sq);
        self.temp_m_hat.extend(temp_m_hat);
        self.temp_v_hat.extend(temp_v_hat);
        self.temp_update.extend(temp_update);
        self.bias_correction1.extend(bias_correction1);
        self.bias_correction2.extend(bias_correction2);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: self.step[i],
                    m: Some(self.m[i].clone()),
                    v: Some(self.v[i].clone()),
                    v_hat: Some(self.v_hat[i].clone()),
                },
            );
        }
        OptimizerState {
            param_groups: vec![ParamGroup {
                params: self.params.clone(),
            }],
            state,
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) {
        if let Some(group) = state.param_groups.first() {
            self.params = group.params.clone();
        }
        for (i, param_state) in state.state {
            if i < self.m.len() {
                if let Some(m) = param_state.m {
                    self.m[i] = m;
                }
                if let Some(v) = param_state.v {
                    self.v[i] = v;
                }
                if let Some(v_hat) = param_state.v_hat {
                    self.v_hat[i] = v_hat;
                }
                if i < self.step.len() {
                    self.step[i] = param_state.step;
                }
            }
        }
    }
}
