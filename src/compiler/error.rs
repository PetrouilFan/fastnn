use std::fmt;

#[derive(Debug)]
pub enum CompilerError {
    InvalidTarget(String),
    Pass { pass: &'static str, message: String },
    InvalidRepresentation { node_id: usize, message: String },
}

impl CompilerError {
    pub fn pass(pass: &'static str, error: impl fmt::Display) -> Self {
        Self::Pass {
            pass,
            message: error.to_string(),
        }
    }
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTarget(message) => write!(formatter, "invalid compile target: {message}"),
            Self::Pass { pass, message } => write!(formatter, "{pass}: {message}"),
            Self::InvalidRepresentation { node_id, message } => {
                write!(
                    formatter,
                    "invalid representation on node {node_id}: {message}"
                )
            }
        }
    }
}

impl std::error::Error for CompilerError {}

pub type CompilerResult<T> = Result<T, CompilerError>;
