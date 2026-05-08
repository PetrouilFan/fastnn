#![allow(dead_code)]
use crate::nn::Module;
use crate::tensor::Tensor;
use std::sync::Arc;

#[allow(dead_code)]
pub struct Sequential {
    pub layers: Vec<Arc<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Arc<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut output = x.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![];
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![];
        for (i, layer) in self.layers.iter().enumerate() {
            let mut layer_params = layer.named_parameters();
            for (name, param) in layer_params.iter_mut() {
                params.push((format!("{}.{}", i, name), param.clone()));
            }
        }
        params
    }

    fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }

    fn train_mode(&self) {
        for layer in &self.layers {
            layer.train_mode();
        }
    }

    fn eval_mode(&self) {
        for layer in &self.layers {
            layer.eval_mode();
        }
    }

    fn is_training(&self) -> bool {
        self.layers
            .first()
            .map(|l| l.is_training())
            .unwrap_or(false)
    }
}

/// ModuleList is an alias for Sequential - they are identical in functionality.
pub type ModuleList = Sequential;
