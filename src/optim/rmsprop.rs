use crate::optim::{Optimizer, OptimizerState, ParamGroup, ParamState};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;

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
    // Pre-allocated buffers
    pub temp_grad_sq: Vec<Tensor>,
    pub temp_grad_avg_sq: Vec<Tensor>,
    pub temp_denom: Vec<Tensor>,
    pub temp_update: Vec<Tensor>,
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
        let square_avg: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let grad_avg: Vec<Tensor> = if centered {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let momentum_buf: Vec<Tensor> = if momentum != 0.0 {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let temp_grad_sq: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_avg_sq: Vec<Tensor> = if centered {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let temp_denom: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

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
            temp_grad_sq,
            temp_grad_avg_sq,
            temp_denom,
            temp_update,
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) {
        let lr = self.lr as f32;
        let alpha = self.alpha as f32;
        let eps = self.eps as f32;
        let weight_decay = self.weight_decay as f32;
        let momentum = self.momentum as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            // Weight decay
            if weight_decay != 0.0 {
                param.mul_scalar_(1.0 - lr * weight_decay);
            }

            // Update square average: avg = alpha * avg + (1 - alpha) * grad^2
            self.temp_grad_sq[i].copy_from_(grad);
            self.temp_grad_sq[i].pow_(2.0);
            self.square_avg[i].mul_scalar_(alpha);
            self.temp_grad_sq[i].mul_scalar_(1.0 - alpha);
            self.square_avg[i].add_(&self.temp_grad_sq[i]);

            // Compute denominator
            if self.centered {
                // Update grad average
                self.grad_avg[i].mul_scalar_(alpha);
                self.temp_update[i].copy_from_(grad);
                self.temp_update[i].mul_scalar_(1.0 - alpha);
                self.grad_avg[i].add_(&self.temp_update[i]);
                // denom = sqrt(avg - avg_grad^2) + eps
                self.temp_grad_avg_sq[i].copy_from_(&self.grad_avg[i]);
                self.temp_grad_avg_sq[i].pow_(2.0);
                self.temp_denom[i].copy_from_(&self.square_avg[i]);
                self.temp_denom[i].sub_(&self.temp_grad_avg_sq[i]);
                self.temp_denom[i].sqrt_();
                self.temp_denom[i].add_scalar_(eps);
            } else {
                self.temp_denom[i].copy_from_(&self.square_avg[i]);
                self.temp_denom[i].sqrt_();
                self.temp_denom[i].add_scalar_(eps);
            };

            // Update with momentum
            if momentum != 0.0 {
                self.momentum_buf[i].mul_scalar_(momentum);
                self.temp_update[i].copy_from_(grad);
                self.temp_update[i].div_(&self.temp_denom[i]);
                self.momentum_buf[i].add_(&self.temp_update[i]);
                self.momentum_buf[i].mul_scalar_(lr);
                param.sub_(&self.momentum_buf[i]);
            } else {
                self.temp_update[i].copy_from_(grad);
                self.temp_update[i].div_(&self.temp_denom[i]);
                self.temp_update[i].mul_scalar_(lr);
                param.sub_(&self.temp_update[i]);
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            let inner = Arc::make_mut(&mut param.inner);
            if let Some(meta) = &mut inner.autograd_meta {
                if let Ok(mut lock) = meta.lock() {
                    lock.grad = None;
                }
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let square_avg: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let grad_avg: Vec<Tensor> = if self.centered {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let momentum_buf: Vec<Tensor> = if self.momentum != 0.0 {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let temp_grad_sq: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_grad_avg_sq: Vec<Tensor> = if self.centered {
            params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
                .collect()
        } else {
            vec![]
        };
        let temp_denom: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();
        let temp_update: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        self.square_avg.extend(square_avg);
        self.grad_avg.extend(grad_avg);
        self.momentum_buf.extend(momentum_buf);
        self.temp_grad_sq.extend(temp_grad_sq);
        self.temp_grad_avg_sq.extend(temp_grad_avg_sq);
        self.temp_denom.extend(temp_denom);
        self.temp_update.extend(temp_update);
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
