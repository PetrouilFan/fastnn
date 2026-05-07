use crate::optim::{Optimizer, OptimizerState, ParamGroup, ParamState, WeightDecayOptimizer, zeros_like};
use crate::tensor::Tensor;
use std::collections::HashMap;

pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
    pub v_hat: Vec<Tensor>,
    pub step: u64,
    // Track which parameters should skip weight decay (e.g., biases, LayerNorm)
    pub no_decay: Vec<bool>,
    // Pre-allocated buffers to avoid clones
    pub temp_grad_scaled: Vec<Tensor>,
    pub temp_grad_sq: Vec<Tensor>,
    pub temp_m_hat: Vec<Tensor>,
    pub temp_v_hat: Vec<Tensor>,
    pub temp_update: Vec<Tensor>,
}

impl Adam {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Self {
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let no_decay = vec![false; params.len()];
        let temp_grad_scaled = zeros_like(&params);
        let temp_grad_sq = zeros_like(&params);
        let temp_m_hat = zeros_like(&params);
        let temp_v_hat = zeros_like(&params);
        let temp_update = zeros_like(&params);

        Adam {
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m,
            v,
            v_hat,
            step: 0,
            no_decay,
            temp_grad_scaled,
            temp_grad_sq,
            temp_m_hat,
            temp_v_hat,
            temp_update,
        }
    }
}

impl WeightDecayOptimizer for Adam {
    fn params(&self) -> &Vec<Tensor> {
        &self.params
    }
    fn no_decay(&self) -> &Vec<bool> {
        &self.no_decay
    }
    fn no_decay_mut(&mut self) -> &mut Vec<bool> {
        &mut self.no_decay
    }
}

impl Optimizer for Adam {
    fn params_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.params
    }

    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        self.step += 1;
        let bias_correction1 = 1.0 - self.betas.0.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.betas.1.powi(self.step as i32);

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            // m = beta1 * m + (1 - beta1) * grad
            let beta1_c = 1.0 - beta1;
            self.m[i].mul_scalar_(beta1);
            self.temp_grad_scaled[i] = grad.clone();
            self.temp_grad_scaled[i].mul_scalar_(beta1_c);
            self.m[i].add_(&self.temp_grad_scaled[i]);

            // v = beta2 * v + (1 - beta2) * grad^2
            let beta2_c = 1.0 - beta2;
            self.v[i].mul_scalar_(beta2);
            self.temp_grad_sq[i] = grad.clone();
            {
                let numel = self.temp_grad_sq[i].inner.numel() as usize;
                let ptr = self.temp_grad_sq[i].data_ptr_f32_mut();
                for j in 0..numel {
                    unsafe {
                        let val = *ptr.add(j);
                        *ptr.add(j) = val * val;
                    }
                }
            }
            self.temp_grad_sq[i].mul_scalar_(beta2_c);
            self.v[i].add_(&self.temp_grad_sq[i]);

            // m_hat = m / bias_correction1
            self.temp_m_hat[i] = self.m[i].clone();
            self.temp_m_hat[i].mul_scalar_((1.0 / bias_correction1) as f32);

            // v_hat = v / bias_correction2 (with optional amsgrad)
            if self.amsgrad {
                self.temp_v_hat[i] = self.v_hat[i].maximum(&self.v[i]);
                self.v_hat[i] = self.temp_v_hat[i].clone();
            } else {
                self.temp_v_hat[i] = self.v[i].clone();
            }
            self.temp_v_hat[i].mul_scalar_((1.0 / bias_correction2) as f32);

            // update = m_hat / (sqrt(v_hat) + eps)
            self.temp_update[i] = self.temp_m_hat[i].clone();
            let denom = self.temp_v_hat[i].sqrt().add_scalar(eps);
            self.temp_update[i].div_(&denom);

            // param = param - lr * update
            self.temp_update[i].mul_scalar_(lr);
            param.sub_(&self.temp_update[i]);

            // Weight decay: param = param - lr * weight_decay * param
            // Skip weight decay for parameters marked as no_decay (e.g., biases)
            if weight_decay != 0.0 && !self.no_decay.get(i).copied().unwrap_or(false) {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let temp_grad_scaled = zeros_like(&params);
        let temp_grad_sq = zeros_like(&params);
        let temp_m_hat = zeros_like(&params);
        let temp_v_hat = zeros_like(&params);
        let temp_update = zeros_like(&params);

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.no_decay.extend(vec![false; params.len()]);
        self.temp_grad_scaled.extend(temp_grad_scaled);
        self.temp_grad_sq.extend(temp_grad_sq);
        self.temp_m_hat.extend(temp_m_hat);
        self.temp_v_hat.extend(temp_v_hat);
        self.temp_update.extend(temp_update);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: self.step,
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
                self.step = param_state.step;
            }
        }
    }
}
