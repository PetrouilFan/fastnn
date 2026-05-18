use crate::optim::{zeros_like, Optimizer, WeightDecayOptimizer};
use crate::tensor::Tensor;
use crate::{get_grad_or_skip, impl_optim_boilerplate, impl_params_mut, impl_weight_decay};

pub struct Lion {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub betas: (f64, f64),
    pub weight_decay: f64,
    pub m: Vec<Tensor>,
    pub step: Vec<u64>,
    pub no_decay: Vec<bool>,
}

impl Lion {
    pub fn new(params: Vec<Tensor>, lr: f64, betas: (f64, f64), weight_decay: f64) -> Self {
        let m = zeros_like(&params);
        let step = vec![0u64; params.len()];
        let no_decay = vec![false; params.len()];

        Lion {
            params,
            lr,
            betas,
            weight_decay,
            m,
            step,
            no_decay,
        }
    }
}

impl WeightDecayOptimizer for Lion {
    impl_weight_decay!();
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

    impl_optim_boilerplate!(true, m);
}
