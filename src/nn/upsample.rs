use crate::nn::Module;
use crate::tensor::Tensor;

pub struct Upsample {
    pub scale_h: usize,
    pub scale_w: usize,
    pub mode: String,
}

impl Upsample {
    pub fn new(scale_factor: f64, mode: String) -> Self {
        let s = scale_factor as usize;
        Upsample { scale_h: s, scale_w: s, mode }
    }

    pub fn new_with_scales(scale_h: usize, scale_w: usize, mode: String) -> Self {
        Upsample { scale_h, scale_w, mode }
    }
}

impl Module for Upsample {
    fn forward(&self, x: &Tensor) -> Tensor {
        let result = Tensor::exec_aot(&[x], |g, ins| {
            let out = match self.mode.as_str() {
                "nearest" | "nearest_neighbor" => {
                    g.upsample_nearest2d(&ins[0], self.scale_h, self.scale_w)
                }
                "bilinear" | "linear" => {
                    g.upsample_bilinear2d(&ins[0], self.scale_h, self.scale_w)
                }
                _ => {
                    g.upsample_nearest2d(&ins[0], self.scale_h, self.scale_w)
                }
            };
            vec![out]
        }).expect("Upsample::forward: AOT execution failed");
        result.into_iter().next().unwrap()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn zero_grad(&self) {}

    fn train_mode(&self) {}

    fn eval_mode(&self) {}

    fn is_training(&self) -> bool {
        false
    }
}
