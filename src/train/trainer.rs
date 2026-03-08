use crate::train::loss::LossFn;

#[allow(dead_code)]
pub struct Trainer {
    pub model: Option<Box<dyn crate::nn::Module>>,
    pub loss_fn: Option<Box<dyn LossFn>>,
}

impl Trainer {
    pub fn new() -> Self {
        Trainer {
            model: None,
            loss_fn: None,
        }
    }
}
