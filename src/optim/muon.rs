use crate::optim::{
    apply_weight_decay, zeros_like, Optimizer, OptimizerState, ParamGroup, ParamState,
    WeightDecayType,
};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_params_mut};
use std::collections::HashMap;

pub struct Muon {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
    pub m: Vec<Tensor>, // momentum buffers
}

impl Muon {
    pub fn new(
        params: Vec<Tensor>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> Self {
        let m = zeros_like(&params);

        Muon {
            params,
            lr,
            momentum,
            weight_decay,
            nesterov,
            m,
        }
    }

    /// Newton-Schulz iteration for orthogonalization in-place
    fn newton_schulz_iteration_inplace(a: &Tensor, num_iterations: usize, out: &mut Tensor) {
        // Compute Frobenius norm: sqrt(sum(x * x))
        let norm_squared = a.mul(a).sum(-1, false).sum(-1, false);
        let norm = norm_squared.sqrt();
        let norm_val = norm.item();

        const EPSILON: f32 = 1e-8;

        if norm_val < EPSILON {
            *out = Tensor::zeros(a.shape(), a.dtype(), a.device());
            return;
        }

        // Normalize by Frobenius norm to ensure numerical stability
        // Add epsilon to prevent division by zero
        let norm_safe = Tensor::from_scalar(norm_val + EPSILON);
        *out = a.clone();
        out.div_(&norm_safe);

        for _ in 0..num_iterations {
            // X = 1.5 * X - 0.5 * X * X^T * X
            let x_t = out.transpose(0, 1);
            let x_x_t = out.matmul(&x_t);
            let mut x_x_t_x = x_x_t.matmul(out);

            // Fused: out = 1.5*out - 0.5*x_x_t_x without allocating scalar tensors
            x_x_t_x.mul_scalar_(0.5);
            out.mul_scalar_(1.5);
            out.sub_(&x_x_t_x);

            // Re-normalize after each iteration for stability
            let x_norm_squared = out.mul(out).sum(-1, false).sum(-1, false);
            let x_norm = x_norm_squared.sqrt();
            let x_norm_val = x_norm.item();
            if x_norm_val < EPSILON {
                *out = Tensor::zeros(out.shape(), out.dtype(), out.device());
                return;
            }
            let x_norm_safe = Tensor::from_scalar(x_norm_val + EPSILON);
            out.div_(&x_norm_safe);
        }
    }
}

impl Optimizer for Muon {
    impl_params_mut!();

    fn step(&mut self) {
        let lr = self.lr as f32;
        let momentum = self.momentum as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = get_grad_or_skip!(param);

            let shape = param.shape();
            let is_2d = shape.len() == 2;

            // Apply L2 weight decay
            let effective_grad = if weight_decay != 0.0 {
                apply_weight_decay(param, &grad, weight_decay, lr, WeightDecayType::L2)
            } else {
                grad
            };

            if is_2d {
                // Standard momentum: m = momentum * m + grad
                self.m[i].mul_scalar_(momentum).add_(&effective_grad);

                // Orthogonalize the momentum
                let mut ortho_m =
                    Tensor::zeros(self.m[i].shape(), self.m[i].dtype(), self.m[i].device());
                Self::newton_schulz_iteration_inplace(&self.m[i], 5, &mut ortho_m);

                // For Nesterov: update_dir = effective_grad + momentum * ortho_momentum
                let update_dir = if self.nesterov {
                    effective_grad.add(&ortho_m.mul_scalar(momentum))
                } else {
                    ortho_m
                };

                // param = param - lr * update_dir
                param.sub_(&update_dir.mul_scalar(lr));
            } else {
                // Fallback SGD with momentum for 1D params
                self.m[i].mul_scalar_(momentum).add_(&effective_grad);
                param.sub_(&self.m[i].clone().mul_scalar(lr));
            }
        }
    }

    fn add_param_group(&mut self, params: Vec<Tensor>) {
        let m = zeros_like(&params);
        self.m.extend(m);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = HashMap::new();
        for (i, _) in self.params.iter().enumerate() {
            state.insert(
                i,
                ParamState {
                    step: 0,
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
            }
        }
    }
}
