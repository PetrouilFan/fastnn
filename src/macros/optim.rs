//! Macros for optimizer implementations.
//! These will be used by `src/optim/*.rs` to eliminate the
//! ~400 lines of duplicated code across adam, adamw, sgd,
//! lion, rmsprop, and muon.

/// Implement the `WeightDecayOptimizer` trait (params, no_decay, no_decay_mut).
///
/// The optimizer struct must have `params: Vec<Tensor>` and `no_decay: Vec<bool>` fields.
///
/// # Usage
/// ```ignore
/// impl WeightDecayOptimizer for MyOptim {
///     impl_weight_decay!();
/// }
/// ```
#[macro_export]
macro_rules! impl_weight_decay {
    () => {
        fn params(&self) -> &Vec<$crate::tensor::Tensor> {
            &self.params
        }
        fn no_decay(&self) -> &Vec<bool> {
            &self.no_decay
        }
        fn no_decay_mut(&mut self) -> &mut Vec<bool> {
            &mut self.no_decay
        }
    };
}

/// Generate the `add_param_group`, `state_dict`, and `load_state_dict` methods
/// for an `Optimizer` impl.
///
/// The optimizer struct must have these fields:
/// - `params: Vec<Tensor>`
/// - `step: Vec<u64>`
/// - `no_decay: Vec<bool>`
/// - One or more state fields listed in `$($state_field:ident),*`
///
/// Each state field should be `Vec<Tensor>` and its name must match a corresponding
/// field in `ParamState` (m, v, or v_hat).
///
/// # Parameters
/// - `$has_amsgrad: expr` — unused placeholder; kept for API compatibility.
/// - `$($state_field:ident),*` — names of the optimizer's state vectors that map 1:1
///   to `ParamState` fields (e.g., `m`, `v`, `v_hat`).
///
/// # Usage
/// ```ignore
/// impl Optimizer for Adam {
///     impl_params_mut!();
///     impl_optim_boilerplate!(true, m, v, v_hat);
///
///     fn step(&mut self) { ... }
/// }
/// ```
#[macro_export]
macro_rules! impl_optim_boilerplate {
    ($has_amsgrad:expr $(,)?) => {
        impl_optim_boilerplate!($has_amsgrad,);
    };
    ($has_amsgrad:expr, $($state_field:ident),* $(,)?) => {
        fn add_param_group(&mut self, params: Vec<$crate::tensor::Tensor>) {
            $(
                let new_field = $crate::optim::zeros_like(&params);
                self.$state_field.extend(new_field);
            )*
            self.no_decay.extend(vec![false; params.len()]);
            self.step.extend(vec![0u64; params.len()]);
            self.params.extend(params);
        }

        fn state_dict(&self) -> $crate::optim::OptimizerState {
            let mut state = ::std::collections::HashMap::new();
            for (i, _) in self.params.iter().enumerate() {
                state.insert(
                    i,
                    $crate::optim::ParamState {
                        step: self.step[i],
                        $(
                            $state_field: Some(self.$state_field[i].clone()),
                        )*
                        ..Default::default()
                    },
                );
            }
            $crate::optim::OptimizerState {
                param_groups: vec![$crate::optim::ParamGroup {
                    params: self.params.clone(),
                }],
                state,
            }
        }

        fn load_state_dict(&mut self, state: $crate::optim::OptimizerState) {
            if let Some(group) = state.param_groups.first() {
                self.params = group.params.clone();
            }
            for (i, param_state) in state.state {
                if i < self.params.len() {
                    $(
                        if let Some(val) = param_state.$state_field {
                            if i < self.$state_field.len() {
                                self.$state_field[i] = val;
                            }
                        }
                    )*
                    if i < self.step.len() {
                        self.step[i] = param_state.step;
                    }
                }
            }
        }
    };
}
