use crate::optim::{Optimizer, OptimizerState, ParamGroup};
use crate::tensor::Tensor;
use std::sync::Arc;

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
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        Muon {
            params,
            lr,
            momentum,
            weight_decay,
            nesterov,
            m,
        }
    }

    /// Newton-Schulz iteration for orthogonalization
    /// Used for 2D weight matrices
    /// For a matrix X with shape [m, k], we orthogonalize rows: X = 1.5 * X - 0.5 * X * X^T * X
    fn newton_schulz_iteration(a: &Tensor, num_iterations: usize) -> Tensor {
        // Compute Frobenius norm: sqrt(sum(x * x))
        let norm_squared = a.mul(a).sum(-1, false).sum(-1, false);
        let norm = norm_squared.sqrt();
        let norm_val = norm.item();

        const EPSILON: f32 = 1e-8;

        if norm_val < EPSILON {
            return Tensor::zeros(a.shape(), a.dtype(), a.device());
        }

        // Normalize by Frobenius norm to ensure numerical stability
        // Add epsilon to prevent division by zero
        let norm_safe = Tensor::from_scalar(norm_val + EPSILON);
        let mut x = a.div(&norm_safe);

        for _ in 0..num_iterations {
            // X = 1.5 * X - 0.5 * X * X^T * X
            let x_t = x.transpose(0, 1);
            let x_x_t = x.matmul(&x_t);
            let mut x_x_t_x = x_x_t.matmul(&x);

            // Fused: x = 1.5*x - 0.5*x_x_t_x without allocating scalar tensors
            x_x_t_x.mul_scalar_(0.5);
            x.mul_scalar_(1.5);
            x.sub_(&x_x_t_x);

            // Re-normalize after each iteration for stability
            let x_norm_squared = x.mul(&x).sum(-1, false).sum(-1, false);
            let x_norm = x_norm_squared.sqrt();
            let x_norm_val = x_norm.item();
            if x_norm_val < EPSILON {
                return Tensor::zeros(a.shape(), a.dtype(), a.device());
            }
            let x_norm_safe = Tensor::from_scalar(x_norm_val + EPSILON);
            x = x.div(&x_norm_safe);
        }

        x
    }
}

impl Optimizer for Muon {
    fn step(&mut self) {
        let lr = self.lr as f32;
        let momentum = self.momentum as f32;
        let weight_decay = self.weight_decay as f32;

        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            let shape = param.shape();
            let is_2d = shape.len() == 2;

            // Apply weight decay: effective_grad = grad + weight_decay * param
            let effective_grad = if weight_decay != 0.0 {
                let mut wd = param.clone();
                wd.mul_scalar_(weight_decay);
                let mut eg = grad.clone();
                eg.add_(&wd);
                eg
            } else {
                grad
            };

            if is_2d {
                // Standard momentum: m = momentum * m + grad
                self.m[i].mul_scalar_(momentum);
                self.m[i].add_(&effective_grad);

                // Orthogonalize the momentum
                let ortho_momentum = Self::newton_schulz_iteration(&self.m[i], 5);

                // For Nesterov: update_dir = effective_grad + momentum * ortho_momentum
                let mut update_dir = if self.nesterov {
                    let mut mom_ortho = ortho_momentum;
                    mom_ortho.mul_scalar_(momentum);
                    let mut eg = effective_grad;
                    eg.add_(&mom_ortho);
                    eg
                } else {
                    ortho_momentum
                };

                // param = param - lr * update_dir
                update_dir.mul_scalar_(lr);
                param.sub_(&update_dir);
            } else {
                // Fallback SGD with momentum for 1D params
                self.m[i].mul_scalar_(momentum);
                self.m[i].add_(&effective_grad);

                let mut update = self.m[i].clone();
                update.mul_scalar_(lr);
                param.sub_(&update);
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
        let m: Vec<Tensor> = params
            .iter()
            .map(|p| Tensor::zeros(p.shape(), p.dtype(), p.device()))
            .collect();

        self.m.extend(m);
        self.params.extend(params);
    }

    fn state_dict(&self) -> OptimizerState {
        OptimizerState {
            param_groups: vec![ParamGroup {
                params: self.params.clone(),
            }],
        }
    }

    fn load_state_dict(&mut self, state: OptimizerState) {
        if let Some(group) = state.param_groups.first() {
            self.params = group.params.clone();
        }
    }
}
