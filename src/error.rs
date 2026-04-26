use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use std::fmt;
use thiserror::Error;

/// Main error type for fastnn operations
#[derive(Error, Debug)]
pub enum FastnnError {
    /// Shape mismatch or invalid shape for operation
    #[error("Shape error: {0}")]
    Shape(String),
    
    /// Data type mismatch or unsupported dtype
    #[error("Dtype error: {0}")]
    Dtype(String),
    
    /// Device-related error (e.g., GPU unavailable)
    #[error("Device error: {0}")]
    Device(String),
    
    /// Error during serialization/deserialization
    #[error("IO error: {0}")]
    Io(String),
    
    /// Integer overflow or numeric limit exceeded
    #[error("Numeric overflow: {0}")]
    Overflow(String),
    
    /// Invalid argument or parameter
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    /// Out of bounds access
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),
    
    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    Allocation(String),
    
    /// CUDA/GPU computation error
    #[error("CUDA error: {0}")]
    Cuda(String),
    
    /// Autograd/backward pass error
    #[error("Autograd error: {0}")]
    Autograd(String),
    
    /// Computation error (e.g., kernel execution failed)
    #[error("Computation error: {0}")]
    Computation(String),
    
    /// General error with custom message
    #[error("{0}")]
    Other(String),
}

impl FastnnError {
    /// Create a shape error with context
    pub fn shape(msg: impl Into<String>) -> Self {
        FastnnError::Shape(msg.into())
    }
    
    /// Create a dtype error with context
    pub fn dtype(msg: impl Into<String>) -> Self {
        FastnnError::Dtype(msg.into())
    }
    
    /// Create a device error with context
    pub fn device(msg: impl Into<String>) -> Self {
        FastnnError::Device(msg.into())
    }
    
    /// Create a tensor operation error with context
    pub fn tensor_op(msg: impl Into<String>) -> Self {
        FastnnError::TensorOp(msg.into())
    }
}

/// Result type alias for fastnn operations
pub type FastnnResult<T> = Result<T, FastnnError>;

impl From<FastnnError> for PyErr {
    fn from(err: FastnnError) -> Self {
        let msg = err.to_string();
        match err {
            FastnnError::Shape(_) => PyRuntimeError::new_err(msg),
            FastnnError::Dtype(_) => PyRuntimeError::new_err(msg),
            FastnnError::Device(_) => PyRuntimeError::new_err(msg),
            FastnnError::Autograd(_) => PyRuntimeError::new_err(msg),
            FastnnError::Optimizer(_) => PyRuntimeError::new_err(msg),
            FastnnError::Io(_) => PyRuntimeError::new_err(msg),
            FastnnError::Cuda(_) => PyRuntimeError::new_err(msg),
            FastnnError::TensorOp(_) => PyRuntimeError::new_err(msg),
            FastnnError::DataLoader(_) => PyRuntimeError::new_err(msg),
            FastnnError::Computation(_) => PyRuntimeError::new_err(msg),
            FastnnError::Internal(_) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Helper trait for adding context to errors
pub trait Context<T> {
    fn with_context<F>(self, f: F) -> Result<T, FastnnError>
    where
        F: FnOnce() -> String;
    
    fn with_context_str(self, context: &str) -> Result<T, FastnnError>;
}

impl<T> Context<T> for Result<T, FastnnError> {
    fn with_context<F>(self, f: F) -> Result<T, FastnnError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| match e {
            FastnnError::Shape(msg) => FastnnError::shape(format!("{}: {}", f(), msg)),
            FastnnError::Dtype(msg) => FastnnError::dtype(format!("{}: {}", f(), msg)),
            FastnnError::Device(msg) => FastnnError::device(format!("{}: {}", f(), msg)),
            FastnnError::TensorOp(msg) => FastnnError::tensor_op(format!("{}: {}", f(), msg)),
            FastnnError::Autograd(msg) => FastnnError::Autograd(format!("{}: {}", f(), msg)),
            FastnnError::Optimizer(msg) => FastnnError::Optimizer(format!("{}: {}", f(), msg)),
            FastnnError::Io(msg) => FastnnError::Io(format!("{}: {}", f(), msg)),
            FastnnError::Cuda(msg) => FastnnError::Cuda(format!("{}: {}", f(), msg)),
            FastnnError::DataLoader(msg) => FastnnError::DataLoader(format!("{}: {}", f(), msg)),
            FastnnError::Computation(msg) => FastnnError::Computation(format!("{}: {}", f(), msg)),
            FastnnError::Internal(msg) => FastnnError::Internal(format!("{}: {}", f(), msg)),
        })
    }
    
    fn with_context_str(self, context: &str) -> Result<T, FastnnError> {
        self.with_context(|| context.to_string())
    }
}

/// Macro for creating shape errors with tensor information
#[macro_export]
macro_rules! shape_error {
    ($($arg:tt)*) => {
        FastnnError::shape(format!($($arg)*))
    };
}

/// Macro for creating dtype errors with tensor information
#[macro_export]
macro_rules! dtype_error {
    ($($arg:tt)*) => {
        FastnnError::dtype(format!($($arg)*))
    };
}

/// Macro for creating device errors with tensor information
#[macro_export]
macro_rules! device_error {
    ($($arg:tt)*) => {
        FastnnError::device(format!($($arg)*))
    };
}

/// Macro for creating tensor operation errors
#[macro_export]
macro_rules! tensor_op_error {
    ($($arg:tt)*) => {
        FastnnError::tensor_op(format!($($arg)*))
    };
}
