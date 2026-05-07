use crate::optim::{
    apply_weight_decay, get_grad, Optimizer, OptimizerState, ParamGroup,
    ParamState, WeightDecayOptimizer, WeightDecayType, zeros_like,
};
use crate::tensor::Tensor;
use std::collections::HashMap;

use crate::impl_params_mut;

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
    pub step: Vec<u64>,
    // Track which parameters should skip weight decay (e.g., biases, LayerNorm)
    pub no_decay: Vec<bool>,
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
        let params_len = params.len();
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let no_decay = vec![false; params_len];

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
            step: vec![0u64; params_len],
            no_decay,
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
    impl_params_mut!();

    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = get_grad(param) {
                g
            } else {
                continue;
            };

            self.step[i] += 1;
            let bias_correction1 = 1.0 - self.betas.0.powi(self.step[i] as i32);
            let bias_correction2 = 1.0 - self.betas.1.powi(self.step[i] as i32);

            // m = beta1 * m + (1 - beta1) * grad
            let beta1_c = 1.0 - beta1;
            let m_update = grad.mul_scalar(beta1_c);
            self.m[i].mul_scalar_(beta1).add_(&m_update);

            // v = beta2 * v + (1 - beta2) * grad^2 (fixed inefficient squaring)
            let grad_sq = grad.pow(2.0);
            let beta2_c = 1.0 - beta2;
            let v_update = grad_sq.mul_scalar(beta2_c);
            self.v[i].mul_scalar_(beta2).add_(&v_update);

            // m_hat = m / bias_correction1
            let mut m_hat = self.m[i].div_scalar(bias_correction1 as f32);

            // v_hat = v / bias_correction2 (with optional amsgrad)
            let v_hat = if self.amsgrad {
                let max_v = self.v_hat[i].maximum(&self.v[i]);
                self.v_hat[i] = max_v.clone();
                max_v.div_scalar(bias_correction2 as f32)
            } else {
                self.v[i].clone().div_scalar(bias_correction2 as f32)
            };

            // update = m_hat / (sqrt(v_hat) + eps) * lr
            let denom = v_hat.sqrt().add_scalar(eps);
            let update = m_hat.div_(&denom).mul_scalar(lr);
            param.sub_(&update);

            // Apply decoupled weight decay consistently
            if weight_decay != 0.0 && !self.no_decay.get(i).copied().unwrap_or(false) {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.no_decay.extend(vec![false; params.len()]);
        self.step.extend(vec![0u64; params.len()]);
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
                self.step[i] = param_state.step;
            }
        }
    }
}
