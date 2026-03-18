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

        // Check if the input is essentially zero (e.g., initial momentum buffer)
        // If so, return zero to avoid division by zero
        if norm_val < 1e-8 {
            return Tensor::zeros(a.shape(), a.dtype(), a.device());
        }

        // Normalize by Frobenius norm to ensure numerical stability
        let mut x = a.div(&norm);

        for _ in 0..num_iterations {
            // X = 1.5 * X - 0.5 * X * X^T * X
            // X has shape [m, k]
            // X^T has shape [k, m]
            // X * X^T has shape [m, m]
            // (X * X^T) * X has shape [m, k]
            let x_t = x.transpose(0, 1);
            let x_x_t = x.matmul(&x_t); // [m, m]
            let x_x_t_x = x_x_t.matmul(&x); // [m, k]

            let three_half = Tensor::from_scalar(1.5f32);
            let half = Tensor::from_scalar(0.5f32);

            x = three_half.mul(&x).sub(&half.mul(&x_x_t_x));

            // Re-normalize after each iteration for stability
            let x_norm_squared = x.mul(&x).sum(-1, false).sum(-1, false);
            let x_norm = x_norm_squared.sqrt();
            x = x.div(&x_norm);
        }

        x
    }
}

impl Optimizer for Muon {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            let grad = if let Some(g) = param.grad() {
                g
            } else {
                continue;
            };

            let shape = param.shape();
            let is_2d = shape.len() == 2;

            // Apply weight decay
            let effective_grad = if self.weight_decay != 0.0 {
                grad.add(&param.mul(&Tensor::from_scalar(self.weight_decay as f32)))
            } else {
                grad
            };

            if is_2d {
                // Muon update for 2D matrices using Newton-Schulz iteration
                // Standard momentum: m = momentum * m + grad
                self.m[i] = self.m[i]
                    .clone()
                    .mul(&Tensor::from_scalar(self.momentum as f32))
                    .add(&effective_grad);

                // Orthogonalize the momentum
                let ortho_momentum = Self::newton_schulz_iteration(&self.m[i], 5);

                // For Nesterov, we use a lookahead approach
                // Standard: update_dir = ortho_momentum
                // Nesterov: update_dir = effective_grad + momentum * ortho_momentum
                let update_dir = if self.nesterov {
                    effective_grad
                        .add(&ortho_momentum.mul(&Tensor::from_scalar(self.momentum as f32)))
                } else {
                    ortho_momentum
                };

                let lr = Tensor::from_scalar(self.lr as f32);
                let step_size = lr.mul(&update_dir);

                // Apply update (subtract for gradient descent)
                let update = step_size.neg();

                param.add_(&update);
                param.increment_version();
            } else {
                // Fallback to AdamW-style update for 1D parameters (biases, LayerNorm)
                // Simple SGD with momentum for 1D params
                self.m[i] = self.m[i]
                    .clone()
                    .mul(&Tensor::from_scalar(self.momentum as f32))
                    .add(&effective_grad);

                let lr = Tensor::from_scalar(self.lr as f32);
                let step_size = lr.mul(&self.m[i]);

                // Apply update
                param.add_(&step_size.neg());
                param.increment_version();
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
