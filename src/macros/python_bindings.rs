//! Macros for Python binding boilerplate (src/python/optim.rs).
//! Eliminates the ~340 lines of duplicated #[pymethods] impl blocks
//! across 6 optimizer bindings.

/// Generate just the shared `step` and `zero_grad` methods.
///
/// Every PyO3 optimizer binding needs these two and they are identical
/// across all six optimizers.  Use inside a manual `#[pymethods] impl` block.
#[macro_export]
macro_rules! impl_py_optim_boilerplate {
    () => {
        fn step(&mut self, py: Python<'_>) {
            py.detach(|| self.inner.step());
        }

        fn zero_grad(&mut self) {
            self.inner.zero_grad();
        }
    };
}

/// Generate the full `#[pymethods] impl` block for a Python optimizer.
///
/// Injects `step` and `zero_grad` automatically, then splices in the
/// custom items (`#[new]`, `state_dict`, `load_state_dict`, etc.) passed
/// as the third argument.
#[macro_export]
macro_rules! impl_py_optim {
    ($py_name:ident, $core_type:ty, $($rest:tt)*) => {
        #[pymethods]
        impl $py_name {
            fn step(&mut self, py: Python<'_>) {
                py.detach(|| self.inner.step());
            }

            fn zero_grad(&mut self) {
                self.inner.zero_grad();
            }

            $($rest)*
        }
    };
}
