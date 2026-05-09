use crate::optim::{zeros_like, Optimizer, OptimizerState, ParamGroup, ParamState};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_params_mut};
use std::collections::HashMap;

pub struct RMSprop {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub centered: bool,
    pub square_avg: Vec<Tensor>,
    pub grad_avg: Vec<Tensor>,
    pub momentum_buf: Vec<Tensor>,
}

impl RMSprop {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> Self {
        let _n = params.len();
        let square_avg = zeros_like(&params);
        let grad_avg = if centered {
            zeros_like(&params)
        } else {
            vec![]
        };
        let momentum_buf = if momentum != 0.0 {
            zeros_like(&params)
        } else {
            vec![]
        };

        RMSprop {
            params,
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            square_avg,
            grad_avg,
            momentum_buf,
        }
    }
}

impl Optimizer for RMSprop {
    impl_params_mut!();

    fn step(&mut self) {
        let lr = self.lr as f32;
        let alpha = self.alpha as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;
        let momentum = self.momentum as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = get_grad_or_skip!(param);

            // Apply decoupled weight decay
            if weight_decay != 0.0 {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }

            // Update square average: avg = alpha * avg + (1 - alpha) * grad^2
            let grad_sq = grad.pow(2.0).mul_scalar(1.0 - alpha);
            self.square_avg[i].mul_scalar_(alpha).add_(&grad_sq);

            // Compute denominator
            let denom = if self.centered {
                // Update grad average
                let grad_avg_update = grad.mul_scalar(1.0 - alpha);
                self.grad_avg[i].mul_scalar_(alpha).add_(&grad_avg_update);
                // denom = sqrt(avg - avg_grad^2) + eps
                let grad_avg_sq = self.grad_avg[i].pow(2.0);
                self.square_avg[i]
                    .clone()
                    .sub(&grad_avg_sq)
                    .sqrt()
                    .add_scalar(eps)
            } else {
                self.square_avg[i].clone().sqrt().add_scalar(eps)
            };

            if momentum != 0.0 {
                let momentum_update = grad.div(&denom);
                self.momentum_buf[i]
                    .mul_scalar_(momentum)
                    .add_(&momentum_update);
                param.sub_(&self.momentum_buf[i].clone().mul_scalar(lr));
            } else {
                let update = grad.div(&denom).mul_scalar(lr);
                param.sub_(&update);
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let square_avg = zeros_like(&params);
        let grad_avg = if self.centered {
            zeros_like(&params)
        } else {
            vec![]
        };
        let momentum_buf = if self.momentum != 0.0 {
            zeros_like(&params)
        } else {
            vec![]
        };

        self.square_avg.extend(square_avg);
        self.grad_avg.extend(grad_avg);
        self.momentum_buf.extend(momentum_buf);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: 0,
                    m: Some(self.square_avg[i].clone()),
                    v: if self.centered {
                        Some(self.grad_avg[i].clone())
                    } else {
                        None
                    },
                    v_hat: if self.momentum != 0.0 {
                        Some(self.momentum_buf[i].clone())
                    } else {
                        None
                    },
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
            if i < self.square_avg.len() {
                if let Some(m) = param_state.m {
                    self.square_avg[i] = m;
                }
                if let Some(v) = param_state.v {
                    if i < self.grad_avg.len() {
                        self.grad_avg[i] = v;
                    }
                }
                if let Some(v_hat) = param_state.v_hat {
                    if i < self.momentum_buf.len() {
                        self.momentum_buf[i] = v_hat;
                    }
                }
            }
        }
    }
}
