use std::marker::PhantomData;

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
    _dtype: PhantomData<T>,
}

impl<T: PackedWord> MasterWeightOptimizer<T> {
    /// Create a new optimizer for the given master weights.
    pub fn new(master: Vec<f32>, lr: f32, betas: (f32, f32), eps: f32, weight_decay: f32) -> Self {
        let n = master.len();
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
            _dtype: PhantomData,
        }
    }

    /// Perform an Adam update step and return the repacked tensor.
    pub fn step(&mut self, gradients: &[f32]) -> PackedTensor<T> {
        assert_eq!(gradients.len(), self.master.len());
        self.step += 1;
        let t = self.step as f32;

        for i in 0..self.master.len() {
            let g = gradients[i];

            // Apply weight decay
            if self.weight_decay != 0.0 {
                self.master[i] -= self.lr * self.weight_decay * self.master[i];
            }

            // Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update second moment
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias correction
            let m_hat = self.m[i] / (1.0 - self.beta1.powf(t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powf(t));

            // Update master weight
            self.master[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }

        // Recalibrate scale periodically
        let should_recalibrate = (self.step as usize).is_multiple_of(self.scale_update_freq);
        let scale = if should_recalibrate || T::IS_FLOAT {
            PackedTensor::<T>::compute_scale(&self.master)
        } else {
            // Use existing scale to avoid frequent repacking overhead
            PackedTensor::<T>::compute_scale(&self.master)
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
