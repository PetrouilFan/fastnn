use crate::optim::{
    zeros_like, Optimizer, OptimizerState, ParamGroup, ParamState, WeightDecayOptimizer,
};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_params_mut};
use std::collections::HashMap;

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
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let no_decay = vec![false; params.len()];
        let bias_correction1 = vec![1.0; params.len()];
        let bias_correction2 = vec![1.0; params.len()];

        AdamW {
            step: vec![0u64; params.len()],
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m,
            v,
            v_hat,
            no_decay,
            bias_correction1,
            bias_correction2,
        }
    }
}

impl WeightDecayOptimizer for AdamW {
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

impl Optimizer for AdamW {
    impl_params_mut!();

    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = get_grad_or_skip!(param);

            self.step[i] += 1;

            self.bias_correction1[i] = 1.0 - beta1 as f64 * (1.0 - self.bias_correction1[i]);
            self.bias_correction2[i] = 1.0 - beta2 as f64 * (1.0 - self.bias_correction2[i]);

            let beta1_c = 1.0 - beta1;
            let beta2_c = 1.0 - beta2;

            let m = &mut self.m[i];
            let v = &mut self.v[i];

            // m = beta1 * m + (1 - beta1) * grad
            let m_update = m.mul_scalar(beta1).add(&grad.mul_scalar(beta1_c));
            *m = m_update;

            // v = beta2 * v + (1 - beta2) * grad^2
            let grad_sq = grad.pow(2.0);
            let v_update = v.mul_scalar(beta2).add(&grad_sq.mul_scalar(beta2_c));
            *v = v_update;

            let bias_correction1 = self.bias_correction1[i] as f32;
            let bias_correction2 = self.bias_correction2[i] as f32;

            let m_hat = m.div_scalar(bias_correction1);
            let v_hat = v.div_scalar(bias_correction2);

            let update = m_hat.div(&v_hat.add_scalar(eps).sqrt());

            let no_decay = self.no_decay.get(i).copied().unwrap_or(false);
            if weight_decay != 0.0 && !no_decay {
                // Decoupled weight decay
                param.mul_scalar_(1.0 - lr * weight_decay);
                param.sub_(&update.mul_scalar(lr));
            } else {
                param.sub_(&update.mul_scalar(lr));
            }

            param.set_grad(None);
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);
        let bias_correction1 = vec![1.0; params.len()];
        let bias_correction2 = vec![1.0; params.len()];

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.step.extend(vec![0u64; params.len()]);
        self.no_decay.extend(vec![false; params.len()]);
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
