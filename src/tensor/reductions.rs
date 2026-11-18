// Tensor reduction methods

use super::Tensor;
use crate::error::{FastnnError, FastnnResult};
use crate::storage::{DType, Device};

fn normalize_reduction_dim(dim: i64, rank: usize, operation: &str) -> FastnnResult<usize> {
    let rank = rank as i64;
    let normalized = if dim < 0 { rank + dim } else { dim };
    if normalized < 0 || normalized >= rank {
        return Err(FastnnError::shape(format!(
            "{operation} dimension {dim} is out of range for {rank} dimensions"
        )));
    }
    Ok(normalized as usize)
}

impl Tensor {
    pub fn sum(&self, dim: i32, keepdim: bool) -> Tensor {
        self.try_sum(dim, keepdim).expect("Tensor::sum failed")
    }

    pub fn try_sum(&self, dim: i32, keepdim: bool) -> FastnnResult<Tensor> {
        let normalized_dim = normalize_reduction_dim(i64::from(dim), self.ndim(), "sum")?;

        if self.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && self.is_contiguous()
            && !keepdim
            && self.inner.ndim() == 2
            && normalized_dim == self.ndim() - 1
        {
            let shape = self.shape();
            let dim_size = shape[normalized_dim] as usize;
            let num_rows = shape[0] as usize;
            let output = crate::backend::cpu::reductions_fast::sum_last_dim_contiguous(
                self, dim_size, num_rows,
            );
            return Ok(output);
        }

        let output = Tensor::exec_aot(&[self], |graph, inputs| {
            vec![graph.reduce_sum(&inputs[0], normalized_dim, keepdim)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("sum execution returned no output".into()))?;
        Ok(output)
    }

    pub fn max(&self, dim: i32, keepdim: bool) -> Tensor {
        self.try_max(dim, keepdim).expect("Tensor::max failed")
    }

    pub fn try_max(&self, dim: i32, keepdim: bool) -> FastnnResult<Tensor> {
        let normalized_dim = normalize_reduction_dim(i64::from(dim), self.ndim(), "max")?;
        let output = Tensor::exec_aot(&[self], |graph, inputs| {
            vec![graph.reduce_max(&inputs[0], normalized_dim, keepdim)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("max execution returned no output".into()))?;
        Ok(output)
    }

    pub fn mean(&self, dim: i32, keepdim: bool) -> Tensor {
        self.try_mean(dim, keepdim).expect("Tensor::mean failed")
    }

    pub fn try_mean(&self, dim: i32, keepdim: bool) -> FastnnResult<Tensor> {
        let normalized_dim = normalize_reduction_dim(i64::from(dim), self.ndim(), "mean")?;
        let output = Tensor::exec_aot(&[self], |graph, inputs| {
            vec![graph.reduce_mean(&inputs[0], normalized_dim, keepdim)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("mean execution returned no output".into()))?;
        Ok(output)
    }

    pub fn cumsum(&self, dim: i64, exclusive: bool, reverse: bool) -> Tensor {
        self.try_cumsum(dim, exclusive, reverse)
            .expect("Tensor::cumsum failed")
    }

    pub fn try_cumsum(&self, dim: i64, exclusive: bool, reverse: bool) -> FastnnResult<Tensor> {
        let normalized_dim = normalize_reduction_dim(dim, self.ndim(), "cumsum")?;
        let output = Tensor::exec_aot(&[self], |graph, inputs| {
            vec![graph.cumsum(&inputs[0], normalized_dim, exclusive, reverse)]
        })
        .map_err(|error| FastnnError::Computation(error.to_string()))?
        .into_iter()
        .next()
        .ok_or_else(|| FastnnError::Internal("cumsum execution returned no output".into()))?;
        Ok(output)
    }
}
