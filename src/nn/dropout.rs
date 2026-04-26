use crate::nn::Module;
use crate::tensor::Tensor;
use rand::Rng;
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
            let keep_prob = 1.0 - self.p;
            // Handle edge case: keep_prob == 0 (p == 1.0) -> all zeros
            if keep_prob <= 0.0 {
                return Tensor::zeros(x.shape(), x.dtype(), x.device());
            }
            let scale = 1.0 / keep_prob as f32;

            // Generate mask with values 0.0 or scale
            let x_data = x.as_f32_slice();
            let mut rng = rand::thread_rng();
            let mask_data: Vec<f32> = x_data
                .iter()
                .map(|_| {
                    if rng.gen::<f64>() < keep_prob {
                        scale
                    } else {
                        0.0
                    }
                })
                .collect();

            let mask = Tensor::from_vec(mask_data, x.shape());
            // Use elementwise multiplication to preserve autograd
            x.mul(&mask)
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

pub struct Dropout2d {
    pub p: f64,
    pub training: std::sync::atomic::AtomicBool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Dropout2d {
            p,
            training: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.load(std::sync::atomic::Ordering::Relaxed) {
            let keep_prob = 1.0 - self.p;
            if keep_prob <= 0.0 {
                return Tensor::zeros(x.shape(), x.dtype(), x.device());
            }
            let scale = 1.0 / keep_prob as f32;

            let x_shape = x.shape();
            let batch = x_shape[0] as usize;
            let channels = x_shape[1] as usize;
            // spatial dims: product of remaining dims
            let spatial: usize = x_shape[2..].iter().map(|&x| x as usize).product();

            // Generate mask per channel per batch (shape [B, C, 1, 1])
            let mut rng = rand::thread_rng();
            let channel_mask_data: Vec<f32> = (0..batch * channels)
                .map(|_| if rng.gen::<f64>() < keep_prob { scale } else { 0.0 })
                .collect();
            let mask = Tensor::from_vec(channel_mask_data, vec![batch as i64, channels as i64, 1, 1]);

            // Broadcast multiplication
            x.mul(&mask)
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
