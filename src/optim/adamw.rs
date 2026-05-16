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

        for i in 0..self.params.len() {
            let grad = get_grad_or_skip!(&self.params[i]);

            self.step[i] += 1;
            let t = self.step[i];

            let no_decay = self.no_decay.get(i).copied().unwrap_or(false);
            let wd = if weight_decay != 0.0 && !no_decay { weight_decay } else { 0.0 };

            let mut results = self.params[i].adamw_update(
                &grad, &self.m[i], &self.v[i],
                lr, beta1, beta2, eps, t, wd,
            );
            self.v[i] = results.pop().unwrap();
            self.m[i] = results.pop().unwrap();
            self.params[i] = results.pop().unwrap();

            self.params[i].set_grad(None);
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);
        let v = zeros_like(&params);
        let v_hat = zeros_like(&params);

        self.m.extend(m);
        self.v.extend(v);
        self.v_hat.extend(v_hat);
        self.step.extend(vec![0u64; params.len()]);
        self.no_decay.extend(vec![false; params.len()]);
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
