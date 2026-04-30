use std::marker::PhantomData;
use wide::f32x4;

use crate::dtypes::PackedWord;
use crate::packed_tensor::PackedTensor;

/// Adam optimizer state for master weights.
pub struct MasterWeightOptimizer<T: PackedWord> {
    pub lr: f32,
    /// Full precision master copy
    pub master: Vec<f32>,
    /// Adam first moment
    pub m: Vec<f32>,
    /// Adam second moment
    pub v: Vec<f32>,
    /// Step counter
    pub step: u64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// How often to recalibrate scale (in optimizer steps)
    pub scale_update_freq: usize,
    /// Cached quantization scale (recomputed periodically)
    current_scale: f32,
    _dtype: PhantomData<T>,
}

impl<T: PackedWord> MasterWeightOptimizer<T> {
    /// Create a new optimizer for the given master weights.
    pub fn new(master: Vec<f32>, lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        let n = master.len();
        let current_scale = PackedTensor::<T>::compute_scale(&master);
        MasterWeightOptimizer {
            lr,
            master: master.clone(),
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            weight_decay,
            scale_update_freq: 100,
            current_scale,
            _dtype: PhantomData,
        }
    }

    /// Perform an Adam update step and return the repacked tensor.
    pub fn step(&mut self, gradients: &[f32]) -> PackedTensor<T> {
        assert_eq!(gradients.len(), self.master.len());
        self.step += 1;
        let t = self.step as f32;
        let beta1_corr = 1.0 - self.beta1.powf(t);
        let beta2_corr = 1.0 - self.beta2.powf(t);
        let lr_wd = if self.weight_decay != 0.0 {
            self.lr * self.weight_decay
        } else {
            0.0
        };
        let one_minus_beta1 = 1.0 - self.beta1;
        let one_minus_beta2 = 1.0 - self.beta2;
        let lr = self.lr;
        let eps = self.eps;

        let len = self.master.len();
        let chunk_size = 4;
        let mut i = 0;
        while i + chunk_size <= len {
            let g_arr = [
                gradients[i],
                gradients[i + 1],
                gradients[i + 2],
                gradients[i + 3],
            ];
            let g = f32x4::from(g_arr);
            let master_arr = [
                self.master[i],
                self.master[i + 1],
                self.master[i + 2],
                self.master[i + 3],
            ];
            let mut master = f32x4::from(master_arr);
            let m_arr = [self.m[i], self.m[i + 1], self.m[i + 2], self.m[i + 3]];
            let mut m = f32x4::from(m_arr);
            let v_arr = [self.v[i], self.v[i + 1], self.v[i + 2], self.v[i + 3]];
            let mut v = f32x4::from(v_arr);

            // Apply weight decay
            master = master - f32x4::splat(lr_wd) * master;

            // Update first moment
            m = f32x4::mul_add(
                f32x4::splat(self.beta1),
                m,
                f32x4::splat(one_minus_beta1) * g,
            );

            // Update second moment
            let g_sq = g * g;
            v = f32x4::mul_add(
                f32x4::splat(self.beta2),
                v,
                f32x4::splat(one_minus_beta2) * g_sq,
            );

            // Bias correction
            let m_hat = m / f32x4::splat(beta1_corr);
            let v_hat = v / f32x4::splat(beta2_corr);

            // Update master weight
            let denom = v_hat.sqrt() + f32x4::splat(eps);
            master -= f32x4::splat(lr) * m_hat / denom;

            // Store back
            let master_arr_out: [f32; 4] = master.into();
            self.master[i..i + 4].copy_from_slice(&master_arr_out);
            let m_arr_out: [f32; 4] = m.into();
            self.m[i..i + 4].copy_from_slice(&m_arr_out);
            let v_arr_out: [f32; 4] = v.into();
            self.v[i..i + 4].copy_from_slice(&v_arr_out);

            i += chunk_size;
        }

        // Handle remainder
        for j in i..len {
            let g = gradients[j];

            // Apply weight decay
            self.master[j] -= lr_wd * self.master[j];

            // Update first moment
            self.m[j] = self.beta1 * self.m[j] + one_minus_beta1 * g;

            // Update second moment
            self.v[j] = self.beta2 * self.v[j] + one_minus_beta2 * g * g;

            // Bias correction
            let m_hat = self.m[j] / beta1_corr;
            let v_hat = self.v[j] / beta2_corr;

            // Update master weight
            self.master[j] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        // Recalibrate scale periodically
        let should_recalibrate = (self.step as usize).is_multiple_of(self.scale_update_freq);
        let scale = if should_recalibrate || T::IS_FLOAT {
            let s = PackedTensor::<T>::compute_scale(&self.master);
            self.current_scale = s;
            s
        } else {
            self.current_scale
        };

        // Repack master weights into packed tensor
        let shape = vec![self.master.len()];
        PackedTensor::from_f32_slice(&self.master, &shape, scale, 0.0)
    }

    /// Repack master weights into a 2D packed tensor.
    pub fn repack_2d(&self, rows: usize, cols: usize) -> PackedTensor<T> {
        assert_eq!(rows * cols, self.master.len());
        let scale = PackedTensor::<T>::compute_scale(&self.master);
        PackedTensor::from_f32_slice(&self.master, &[rows, cols], scale, 0.0)
    }

    /// Get the current learning rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Set the learning rate.
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Set how often to update the quantization scale (in optimizer steps).
    pub fn set_scale_update_freq(&mut self, freq: usize) {
        self.scale_update_freq = freq.max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtypes::{F32x1, U8x4};

    #[test]
    fn test_master_weight_optimizer_creation() {
        let master = vec![1.0, 2.0, 3.0, 4.0];
        let opt = MasterWeightOptimizer::<U8x4>::new(master, 0.001, (0.9, 0.999), 1e-8, 0.0);
        assert_eq!(opt.step, 0);
    }

    #[test]
    fn test_optimizer_step() {
        let master = vec![1.0, 2.0, 3.0, 4.0];
        let mut opt = MasterWeightOptimizer::<F32x1>::new(master, 0.01, (0.9, 0.999), 1e-8, 0.0);
        let grad = vec![0.1, 0.2, 0.3, 0.4];
        let _packed = opt.step(&grad);
        assert_eq!(opt.step, 1);
        // Master weights should have changed
        assert_ne!(opt.master[0], 1.0);
    }

    #[test]
    fn test_optimizer_with_weight_decay() {
        let master = vec![1.0, 2.0, 3.0, 4.0];
        let mut opt =
            MasterWeightOptimizer::<F32x1>::new(master.clone(), 0.01, (0.9, 0.999), 1e-8, 0.01);
        let grad = vec![0.0, 0.0, 0.0, 0.0]; // zero grad, only weight decay
        opt.step(&grad);
        // Weight decay should reduce the weights
        for (m, orig) in opt.master.iter().zip(master.iter()) {
            assert!(*m < *orig, "Weight decay should reduce weights");
        }
    }
}
