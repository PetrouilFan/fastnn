use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::atomic::AtomicBool;

pub struct Dropout {
    pub p: f64,
    pub training: AtomicBool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: AtomicBool::new(true),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.load(std::sync::atomic::Ordering::Relaxed) {
            let x_data = x.as_f32_slice();
            let scale = 1.0 / (1.0 - self.p) as f32;
            let mask_data: Vec<f32> = x_data
                .iter()
                .map(|&v| {
                    if rand::random::<f64>() < (1.0 - self.p) {
                        v * scale
                    } else {
                        0.0
                    }
                })
                .collect();

            let shape = x.shape();
            // mask_data already contains the final dropout result (x * scale or 0),
            // no need for redundant x.mul(&mask)
            Tensor::from_vec(mask_data, shape)
        } else {
            x.clone()
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {
        self.training
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn eval_mode(&self) {
        self.training
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(std::sync::atomic::Ordering::Relaxed)
    }
}
