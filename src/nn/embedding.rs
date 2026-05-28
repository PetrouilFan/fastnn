use crate::autograd;
use crate::tensor::Tensor;
use crate::{
    impl_training_state,
    nn::{clear_grad, Module, TrainingState},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
    #[allow(dead_code)]
    pub num_embeddings: i64,
    #[allow(dead_code)]
    pub embedding_dim: i64,
    training: TrainingState,
}

impl Embedding {
    pub fn new(num_embeddings: i64, embedding_dim: i64) -> Self {
        let scale = (1.0 / (embedding_dim as f32).sqrt()).min(0.05);
        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| (crate::random_f32() - 0.5) * 2.0 * scale)
            .collect();
        let weight = Tensor::from_vec(weight_data, vec![num_embeddings, embedding_dim]);
        let weight = weight.requires_grad_(true);

        Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            training: TrainingState::new(),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, indices: &Tensor) -> Tensor {
        let mut output = Tensor::exec_aot(&[&self.weight, indices], |g, ins| {
            vec![g.embedding(&ins[0], &ins[1])]
        })
        .expect("Embedding::forward: AOT failed")
        .into_iter()
        .next()
        .unwrap();

        // Attach autograd so gradients flow to the weight
        if self.weight.requires_grad() {
            let inputs = vec![self.weight.clone(), indices.clone()];
            let mut meta = autograd::AutogradMeta::new_non_leaf(true);
            meta.grad_fn = Some(autograd::make_node_info("EmbeddingBackward", inputs));
            Arc::make_mut(&mut output.inner).autograd_meta =
                Some(Arc::new(parking_lot::Mutex::new(meta)));
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }

    fn zero_grad(&self) {
        clear_grad(&self.weight);
    }

    impl_training_state!(self, self.training);
}
