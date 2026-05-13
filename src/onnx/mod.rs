//! ONNX model converter — translates parsed ONNX graphs into the AOT IR
//! [`ComputeGraph`](crate::ir::node::ComputeGraph) for compilation and
//! execution through the v2.0 compiler pipeline.
//!
//! # Design
//!
//! This module takes the same JSON-like input format (nodes + params + input/output names)
//! that the Python [`onnx`](https://pypi.org/project/onnx/) library produces,
//! and builds a [`ComputeGraph`] that the AOT pipeline can compile and execute.
//!
//! # Usage
//!
//! ```ignore
//! use crate::onnx::converter::OnnxConverter;
//!
//! let converter = OnnxConverter::new(nodes, params, input_names, output_names);
//! let graph = converter.to_compute_graph()?;
//! // Now compile & execute with the AOT pipeline
//! ```

pub mod converter;
