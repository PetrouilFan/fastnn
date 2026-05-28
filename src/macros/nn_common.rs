//! Macros for neural network layer boilerplate.
//! Used by src/nn/norm.rs and src/nn/pooling.rs to reduce duplication.

/// Generate `parameters()`, `named_parameters()`, `named_parameters_ref()`, `zero_grad()`,
/// and (optionally) training-state methods for normalization layers.
///
/// Call *inside* an `impl Module for $name { ... }` block.
///
/// # Variants
/// - `impl_norm!(weight, bias, training)` — required weight + bias + training state
/// - `impl_norm!(weight, bias)` — required weight + bias, no training state
/// - `impl_norm!(weight, training)` — weight only + training state
/// - `impl_norm!(weight)` — weight only, no training state
#[macro_export]
macro_rules! impl_norm {
    ($weight:ident, $bias:ident, $training:ident) => {
        fn parameters(&self) -> Vec<$crate::tensor::Tensor> {
            vec![self.$weight.clone(), self.$bias.clone()]
        }

        fn named_parameters(&self) -> Vec<(String, $crate::tensor::Tensor)> {
            vec![
                ("weight".to_string(), self.$weight.clone()),
                ("bias".to_string(), self.$bias.clone()),
            ]
        }

        fn named_parameters_ref(&self) -> Vec<(&str, &$crate::tensor::Tensor)> {
            vec![("weight", &self.$weight), ("bias", &self.$bias)]
        }

        fn zero_grad(&self) {
            $crate::nn::clear_grad(&self.$weight);
            $crate::nn::clear_grad(&self.$bias);
        }

        $crate::impl_training_state!(self, self.$training);
    };
    ($weight:ident, $bias:ident) => {
        fn parameters(&self) -> Vec<$crate::tensor::Tensor> {
            vec![self.$weight.clone(), self.$bias.clone()]
        }

        fn named_parameters(&self) -> Vec<(String, $crate::tensor::Tensor)> {
            vec![
                ("weight".to_string(), self.$weight.clone()),
                ("bias".to_string(), self.$bias.clone()),
            ]
        }

        fn named_parameters_ref(&self) -> Vec<(&str, &$crate::tensor::Tensor)> {
            vec![("weight", &self.$weight), ("bias", &self.$bias)]
        }

        fn zero_grad(&self) {
            $crate::nn::clear_grad(&self.$weight);
            $crate::nn::clear_grad(&self.$bias);
        }
    };
    ($weight:ident, $training:ident) => {
        fn parameters(&self) -> Vec<$crate::tensor::Tensor> {
            vec![self.$weight.clone()]
        }

        fn named_parameters(&self) -> Vec<(String, $crate::tensor::Tensor)> {
            vec![("weight".to_string(), self.$weight.clone())]
        }

        fn named_parameters_ref(&self) -> Vec<(&str, &$crate::tensor::Tensor)> {
            vec![("weight", &self.$weight)]
        }

        fn zero_grad(&self) {
            $crate::nn::clear_grad(&self.$weight);
        }

        $crate::impl_training_state!(self, self.$training);
    };
    ($weight:ident) => {
        fn parameters(&self) -> Vec<$crate::tensor::Tensor> {
            vec![self.$weight.clone()]
        }

        fn named_parameters(&self) -> Vec<(String, $crate::tensor::Tensor)> {
            vec![("weight".to_string(), self.$weight.clone())]
        }

        fn named_parameters_ref(&self) -> Vec<(&str, &$crate::tensor::Tensor)> {
            vec![("weight", &self.$weight)]
        }

        fn zero_grad(&self) {
            $crate::nn::clear_grad(&self.$weight);
        }
    };
}

/// Implement `Module` for a pooling layer with AOT execution and autograd.
///
/// `$exec_body` is the closure body passed to `Tensor::exec_aot`.
/// Use `g` for the `GraphBuilder` and `ins` for the input tensors.
///
/// # Example
/// ```ignore
/// impl_pooling!(AvgPool2d, "AvgPool2dBackward", {
///     vec![g.avg_pool2d(&ins[0], self.kernel_size as usize, self.stride as usize, self.padding as usize)]
/// });
/// ```
#[macro_export]
macro_rules! impl_pooling {
    ($name:ident, $backward_name:literal, $exec_body:expr) => {
        impl $crate::nn::Module for $name {
            fn forward(&self, x: &$crate::tensor::Tensor) -> $crate::tensor::Tensor {
                let result = $crate::tensor::Tensor::exec_aot(&[x], |g, ins| $exec_body).expect(
                    concat!(stringify!($name), "::forward: AOT execution failed"),
                );
                let mut output = result.into_iter().next().unwrap();
                $crate::utils::attach_grad(&mut output, $backward_name, &[x]);
                output
            }
        }
    };
}
