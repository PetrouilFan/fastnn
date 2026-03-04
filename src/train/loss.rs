use crate::tensor::Tensor;

pub trait LossFn: Send + Sync {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Tensor;
}

pub struct MseLoss {
    pub reduction: String,
}

impl MseLoss {
    pub fn new(reduction: String) -> Self {
        MseLoss { reduction }
    }
}

impl LossFn for MseLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        let diff = pred.sub(target);
        let loss = diff.mul(&diff);

        match self.reduction.as_str() {
            "none" => loss,
            "mean" => loss
                .sum(0, false)
                .div(&Tensor::from_scalar(loss.numel() as f32)),
            "sum" => loss.sum(0, false),
            _ => loss.sum(0, false),
        }
    }
}

pub struct CrossEntropyLoss {
    pub reduction: String,
}

impl CrossEntropyLoss {
    pub fn new(reduction: String) -> Self {
        CrossEntropyLoss { reduction }
    }
}

impl LossFn for CrossEntropyLoss {
    fn compute(&self, pred: &Tensor, target: &Tensor) -> Tensor {
        match self.reduction.as_str() {
            "none" => Tensor::zeros(pred.shape(), pred.dtype(), pred.device()),
            "mean" | "sum" | _ => {
                let max_logits = pred.max(1, true);
                let log_sum_exp = pred.sub(&max_logits.clone()).exp().sum(1, true).ln();

                let target_idx = target.item() as usize;
                let class_logit = pred
                    .reshape(vec![pred.numel()])
                    .slice(0, target_idx as i64, (target_idx + 1) as i64, 1)
                    .item();

                let loss = log_sum_exp.item() - class_logit;
                Tensor::from_scalar(loss)
            }
        }
    }
}
