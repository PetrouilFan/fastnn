use crate::optim::{zeros_like, Optimizer, OptimizerState, ParamGroup, ParamState};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_params_mut};
use std::collections::HashMap;

pub struct Lion {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
    pub m: Vec<Tensor>,
    pub step: Vec<u64>,
}

impl Lion {
    pub fn new(params: Vec<Tensor>, lr: f64, betas: (f64, f64), weight_decay: f64) -> Self {
        let m = zeros_like(&params);
        let step = vec![0u64; params.len()];

        Lion {
            params,
            lr,
            betas,
            weight_decay,
            m,
            step,
        }
    }
}

impl Optimizer for Lion {
    impl_params_mut!();

    fn step(&mut self) {
        let beta1 = self.betas.0 as f32;
        let beta2 = self.betas.1 as f32;
        let lr = self.lr as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = get_grad_or_skip!(param);

            self.step[i] += 1;

            // Weight decay (decoupled)
            if weight_decay != 0.0 {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }

            // Update momentum: m = beta1 * m + (1 - beta1) * grad
            let beta1_c = 1.0 - beta1;
            let m_update = self.m[i]
                .clone()
                .mul_scalar(beta1)
                .add(&grad.mul_scalar(beta1_c));
            self.m[i] = m_update;

            // Compute sign of (beta2 * m + (1 - beta2) * grad)
            let beta2_c = 1.0 - beta2;
            let update_term = self.m[i]
                .clone()
                .mul_scalar(beta2)
                .add(&grad.mul_scalar(beta2_c));
            let signed = update_term.sign();

            // param = param - lr * sign(update)
            param.sub_(&signed.mul_scalar(lr));
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);

        self.m.extend(m);
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
                    v: None,
                    v_hat: None,
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
                if i < self.step.len() {
                    self.step[i] = param_state.step;
                }
            }
        }
    }
}
