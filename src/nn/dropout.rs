use crate::tensor::Tensor;
use crate::{
    impl_training_state,
    nn::{Module, TrainingState},
};
use rand::Rng;

pub struct Dropout {
    pub p: f64,
    training: TrainingState,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Dropout {
            p,
            training: TrainingState::new(),
        }
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.is_training() {
            let x_data = x.as_f32_slice();
            let scale = 1.0 / (1.0 - self.p) as f32;
            let keep_prob = 1.0 - self.p;
            let numel = x_data.len();

            let mut rng = rand::thread_rng();
            let mut mask_data = vec![0.0f32; numel];

            let chunk_size = 4096;
            for chunk_start in (0..numel).step_by(chunk_size) {
                let chunk_end = std::cmp::min(chunk_start + chunk_size, numel);
                let chunk_len = chunk_end - chunk_start;

                let mut rand_vals = vec![0.0f32; chunk_len];
                rng.fill(&mut rand_vals[..]);

                for i in 0..chunk_len {
                    if rand_vals[i] < keep_prob as f32 {
                        mask_data[chunk_start + i] = scale;
                    }
                }
            }

            let shape = x.shape_ref();
            let mask = Tensor::from_vec(mask_data, shape.to_vec());
            let mut out = x.mul(&mask);

            if x.requires_grad() {
                let backward = crate::autograd::DropoutBackward::new();
                let mut meta = crate::autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                std::sync::Arc::make_mut(&mut out.inner).autograd_meta =
                    Some(std::sync::Arc::new(std::sync::Mutex::new(meta)));
            }
            out
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

    impl_training_state!(self, self.training);
}

pub struct Dropout2d {
    pub p: f64,
    training: TrainingState,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Dropout2d {
            p,
            training: TrainingState::new(),
        }
    }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if self.training.is_training() {
            let x_shape = x.shape_ref();
            let batch = x_shape[0] as usize;
            let channels = x_shape[1] as usize;
            let _spatial: usize = x_shape[2..].iter().map(|&x| x as usize).product();
            let scale = 1.0 / (1.0 - self.p) as f32;
            let keep_prob = 1.0 - self.p;

            // Generate mask: [N, C, 1, 1] that will broadcast
            let mut rng = rand::thread_rng();
            let num_masks = batch * channels;
            let mut channel_mask_data = vec![0.0f32; num_masks];
            let mut rand_vals = vec![0.0f32; num_masks];
            rng.fill(&mut rand_vals[..]);
            for i in 0..num_masks {
                if rand_vals[i] < keep_prob as f32 {
                    channel_mask_data[i] = scale;
                }
            }

            let channel_mask =
                Tensor::from_vec(channel_mask_data, vec![batch as i64, channels as i64, 1, 1]);
            let mut out = x.mul(&channel_mask);

            if x.requires_grad() {
                let backward = crate::autograd::Dropout2dBackward::new();
                let mut meta = crate::autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                std::sync::Arc::make_mut(&mut out.inner).autograd_meta =
                    Some(std::sync::Arc::new(std::sync::Mutex::new(meta)));
            }
            out
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

    impl_training_state!(self, self.training);
}
