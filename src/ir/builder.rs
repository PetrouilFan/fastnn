//! High-level graph builder API for constructing and executing IR computation graphs.
//!
//! Provides [`GraphBuilder`] and [`GraphTensor`] as a user-friendly way to build
//! [`ComputeGraph`]s, compile them through the v2.0 compiler pipeline, and execute
//! them on a [`Backend`].
//!
//! # Example
//! ```ignore
//! use fastnn::ir::builder::GraphBuilder;
//! use fastnn::ir::node::IrDType;
//! use fastnn::backend::cpu::CpuBackend;
//!
//! let mut g = GraphBuilder::new();
//! let a = g.input(&[2, 3], IrDType::F32);
//! let b = g.input(&[2, 3], IrDType::F32);
//! let c = g.add(&a, &b);
//! let d = g.relu(&c);
//!
//! let input_a = vec![1.,2.,3.,4.,5.,6.];
//! let input_b = vec![6.,5.,4.,3.,2.,1.];
//! let input_bytes_a: Vec<u8> = bytemuck::cast_slice(&input_a).to_vec();
//! let input_bytes_b: Vec<u8> = bytemuck::cast_slice(&input_b).to_vec();
//!
//! let result = g.compile_and_execute(&[&d], CpuBackend, &[&input_bytes_a, &input_bytes_b]);
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::Ordering;

use crate::backend::{Backend, BackendError, ExecutablePlan};
use crate::backend::executor::GraphExecutor;
use crate::compiler::passes::memory_planning::MemoryPlan;
use crate::ir::node::*;

// =============================================================================
// GraphBuilder — inner state (shared via Rc<RefCell<>>)
// =============================================================================

/// Inner state shared between [`GraphBuilder`] and all [`GraphTensor`]s
/// created from it.
#[derive(Debug)]
struct BuilderInner {
    graph: ComputeGraph,
    recorded_inputs: Vec<NodeId>,
}

// =============================================================================
// GraphTensor
// =============================================================================

/// A symbolic tensor representing a node in a computation graph under construction.
///
/// `GraphTensor` values are lightweight handles — they hold a node id, shape, and
/// dtype together with a shared reference to the owning [`GraphBuilder`].  Ops on
/// `GraphTensor` (e.g. `add`, `relu`, `matmul`) append new nodes to the same graph.
///
/// `Clone` is cheap (just bumps the `Rc` refcount on the builder).
#[derive(Clone, Debug)]
pub struct GraphTensor {
    pub(crate) builder: GraphBuilder,
    pub(crate) node_id: NodeId,
    pub(crate) tensor_type: TensorType,
}

impl GraphTensor {
    /// Create a new graph tensor handle (internal).
    pub fn new(builder: GraphBuilder, node_id: NodeId, tensor_type: TensorType) -> Self {
        Self { builder, node_id, tensor_type }
    }

    /// The underlying `NodeId` in the `ComputeGraph`.
    pub fn node_id(&self) -> NodeId { self.node_id }

    /// The symbolic shape of this tensor.
    pub fn shape(&self) -> &[DimExpr] { &self.tensor_type.shape }

    /// The data type of this tensor.
    pub fn dtype(&self) -> IrDType { self.tensor_type.dtype.clone() }

    /// Full type information (shape + dtype).
    pub fn tensor_type(&self) -> &TensorType { &self.tensor_type }

    /// Return a reference to the builder that owns this tensor.
    pub fn builder(&self) -> &GraphBuilder { &self.builder }
}

// =============================================================================
// Operator overloads on GraphTensor
// =============================================================================

macro_rules! impl_binary_op {
    ($trait:ident, $fn:ident, $opcode:ident) => {
        impl std::ops::$trait for GraphTensor {
            type Output = GraphTensor;
            fn $fn(self, rhs: Self) -> Self {
                let output_type = self.tensor_type.clone();
                let mut inner = self.builder.inner.borrow_mut();
                let node_id = inner.graph.add_node(
                    Opcode::$opcode,
                    vec![self.node_id, rhs.node_id],
                    output_type.clone(),
                );
                GraphTensor::new(self.builder.clone(), node_id, output_type)
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Sub, sub, Sub);
impl_binary_op!(Mul, mul, Mul);
impl_binary_op!(Div, div, Div);

impl std::ops::Neg for GraphTensor {
    type Output = GraphTensor;
    fn neg(self) -> Self {
        let output_type = self.tensor_type.clone();
        let mut inner = self.builder.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Neg,
            vec![self.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.builder.clone(), node_id, output_type)
    }
}

// =============================================================================
// GraphBuilder — public API
// =============================================================================

/// High-level builder for constructing, compiling, and executing IR computation graphs.
///
/// `GraphBuilder` wraps a [`ComputeGraph`] with interior mutability
/// (`Rc<RefCell<>>`) so that [`GraphTensor`] handles can add nodes to the graph
/// through operator overloads without borrowing the builder.
///
/// # Compilation Pipeline
/// When `compile()` or `compile_and_execute()` is called, the builder:
/// 1. Sets `graph.inputs` and `graph.outputs` from the recorded inputs and user-specified outputs
/// 2. Runs shape inference
/// 3. Runs operator fusion
/// 4. Runs memory planning
/// 5. Calls `backend.compile()` to produce an `ExecutablePlan`
///
/// # Backward / Autograd
/// `backward()` calls `build_backward_graph()` to derive gradient computation graphs,
/// replacing the internal graph with the combined forward+backward graph and returning
/// gradient tensors for each registered input/parameter.
#[derive(Clone, Debug)]
pub struct GraphBuilder {
    inner: Rc<RefCell<BuilderInner>>,
}

impl GraphBuilder {
    /// Create a new empty graph builder.
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(BuilderInner {
                graph: ComputeGraph::new(),
                recorded_inputs: Vec::new(),
            })),
        }
    }

    // ── Input & Parameter registration ────────────────────────────────────

    /// Register a graph input with a concrete shape and dtype.
    /// Returns a [`GraphTensor`] handle that can be used in further graph ops.
    pub fn input(&self, shape: &[u64], dtype: IrDType) -> GraphTensor {
        let dims: Vec<DimExpr> = shape.iter().map(|&d| DimExpr::Known(d)).collect();
        let tt = TensorType::new(dims, dtype);
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(Opcode::Input, vec![], tt.clone());
        inner.recorded_inputs.push(node_id);
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Register a graph input with symbolic [`DimExpr`] dimensions and dtype.
    ///
    /// Use this when the exact shape is not known at graph-build time.
    /// The dims may be [`DimExpr::Known`], [`DimExpr::Symbol`], or
    /// [`DimExpr::Bounded`].
    ///
    /// # Example
    /// ```ignore
    /// let batch_size = DimExpr::Symbol("N".into());
    /// let a = g.input_with_dims(&[batch_size, DimExpr::Known(64)], IrDType::F32);
    /// ```
    pub fn input_with_dims(&self, shape: &[DimExpr], dtype: IrDType) -> GraphTensor {
        let tt = TensorType::new(shape.to_vec(), dtype);
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(Opcode::Input, vec![], tt.clone());
        inner.recorded_inputs.push(node_id);
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Register a learnable parameter (same as an input but semantically a weight).
    /// Parameters are tracked alongside inputs for gradient computation.
    pub fn parameter(&self, shape: &[u64], dtype: IrDType) -> GraphTensor {
        self.input(shape, dtype)
    }

    /// Register a learnable parameter with symbolic [`DimExpr`] dimensions.
    /// Like [`parameter`](Self::parameter) but accepts `&[DimExpr]` for dynamic shapes.
    pub fn parameter_with_dims(&self, shape: &[DimExpr], dtype: IrDType) -> GraphTensor {
        self.input_with_dims(shape, dtype)
    }

    /// Create a constant tensor node with raw byte data.
    pub fn constant(&self, data: &[u8], tensor_type: TensorType) -> GraphTensor {
        let value = TensorValue::Data {
            bytes: data.to_vec(),
            tensor_type: tensor_type.clone(),
        };
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_constant(value);
        GraphTensor::new(self.clone(), node_id, tensor_type)
    }

    // ── Graph construction ops ────────────────────────────────────────────

    /// Element-wise addition.
    pub fn add(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Add,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Sub,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Mul,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Element-wise division.
    pub fn div(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Div,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Matrix multiplication: `a @ b`.
    pub fn matmul(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let a_shape = a.shape();
        let b_shape = b.shape();
        // m = second-to-last dim of a (the "M" in MxK @ KxN).
        // a_shape.first() would be the batch dim, not M.
                    let m = a_shape.get(a_shape.len().saturating_sub(2))
            .cloned().unwrap_or(DimExpr::Known(0));
        let _k_a = a_shape.get(a_shape.len().saturating_sub(1))
            .cloned().unwrap_or(DimExpr::Known(0));
        let _k_b = b_shape.get(b_shape.len().saturating_sub(2))
            .cloned().unwrap_or(DimExpr::Known(0));
        let n = b_shape.last().cloned().unwrap_or(DimExpr::Known(0));

        // Broadcast batch dimensions
        let batch_dims: Vec<DimExpr> = {
            let ba = if a_shape.len() > 2 { &a_shape[..a_shape.len() - 2] } else { &[] };
            let bb = if b_shape.len() > 2 { &b_shape[..b_shape.len() - 2] } else { &[] };
            let max = ba.len().max(bb.len());
            let mut result = Vec::with_capacity(max);
            for i in 0..max {
                let da = if i < ba.len() { &ba[ba.len() - 1 - i] } else { &DimExpr::Known(1) };
                let db = if i < bb.len() { &bb[bb.len() - 1 - i] } else { &DimExpr::Known(1) };
                result.push(match (da, db) {
                    (DimExpr::Known(1), other) => other.clone(),
                    (other, DimExpr::Known(1)) => other.clone(),
                    _ => da.clone(), // assume compatible
                });
            }
            result.reverse();
            result
        };

        let mut output_shape = batch_dims;
        output_shape.push(m);
        output_shape.push(n);

        let output_type = TensorType::new(output_shape, a.dtype());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::MatMul,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Convolution 2D.
    pub fn conv2d(&self, input: &GraphTensor, weight: &GraphTensor,
                  stride: usize, padding: usize) -> GraphTensor {
        self.conv2d_with_params(input, weight, stride, padding, 1, 1)
    }

    /// Convolution 2D with full parameter control.
    pub fn conv2d_with_params(&self, input: &GraphTensor, weight: &GraphTensor,
                              stride: usize, padding: usize, dilation: usize, groups: usize) -> GraphTensor {
        let a_shape = input.shape();
        let w_shape = weight.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let out_channels = w_shape.first().cloned().unwrap_or(DimExpr::Known(1));

        let h_out = if a_shape.len() > 2 && w_shape.len() > 2 {
            conv_spatial_dim(&a_shape[2], &w_shape[2], stride, padding)
        } else {
            DimExpr::Known(1)
        };

        let w_out = if a_shape.len() > 3 && w_shape.len() > 3 {
            conv_spatial_dim(&a_shape[3], &w_shape[3], stride, padding)
        } else {
            DimExpr::Known(1)
        };

        let output_type = TensorType::new(
            vec![batch, out_channels, h_out, w_out],
            input.dtype(),
        );

        let mut attrs = HashMap::new();
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        attrs.insert("dilation".to_string(), dilation.to_string());
        attrs.insert("groups".to_string(), groups.to_string());

        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Activation: ReLU.
    pub fn relu(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Relu,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Activation: GELU.
    pub fn gelu(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Gelu,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Activation: SiLU (Swish).
    pub fn silu(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Silu,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Activation: Sigmoid.
    pub fn sigmoid(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Sigmoid,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Activation: Tanh.
    pub fn tanh(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Tanh,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Exponential.
    pub fn exp(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Exp,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Natural logarithm.
    pub fn log(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Log,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Square root.
    pub fn sqrt(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Sqrt,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Negation.
    pub fn neg(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Neg,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Absolute value.
    pub fn abs(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Abs,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Activations added for PPO / control / RL pipelines
    // =========================================================================

    /// Leaky ReLU activation.
    pub fn leaky_relu(&self, input: &GraphTensor, negative_slope: f32) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("negative_slope".to_string(), negative_slope.to_string());
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::LeakyRelu,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// ELU activation.
    pub fn elu(&self, input: &GraphTensor, alpha: f32) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("alpha".to_string(), alpha.to_string());
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Elu,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Softplus activation.
    pub fn softplus(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Softplus,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Hardswish activation.
    pub fn hardswish(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Hardswish,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Clamp (clip) tensor values to [min, max].
    pub fn clamp(&self, input: &GraphTensor, min: f32, max: f32) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("min".to_string(), min.to_string());
        attrs.insert("max".to_string(), max.to_string());
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Clamp,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Sign function: -1 for negative, 0 for zero, 1 for positive.
    pub fn sign(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Sign,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Logical NOT (binary mask inversion).
    pub fn logical_not(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::LogicalNot,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Log softmax activation.
    pub fn log_softmax(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::LogSoftmax,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Mish activation: x * tanh(softplus(x)).
    pub fn mish(&self, input: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Mish,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Binary element-wise ops for RL / control
    // =========================================================================

    /// Element-wise maximum.
    pub fn maximum(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone(); // broadcast shapes are validated at compile/runtime
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Maximum,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Element-wise minimum.
    pub fn minimum(&self, a: &GraphTensor, b: &GraphTensor) -> GraphTensor {
        let output_type = a.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Minimum,
            vec![a.node_id, b.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Reductions
    // =========================================================================

    /// Max reduction along a dimension.
    pub fn reduce_max(&self, input: &GraphTensor, axis: usize, keepdim: bool) -> GraphTensor {
        let mut output_type = input.tensor_type.clone();
        if axis < output_type.shape.len() {
            output_type.shape.remove(axis);
        }
        let mut inner = self.inner.borrow_mut();
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("axis".to_string(), axis.to_string());
        attrs.insert("keepdim".to_string(), if keepdim { "1" } else { "0" }.to_string());
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::ReduceMax,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// ArgMax along an optional axis. Returns indices as I64.
    pub fn argmax(&self, input: &GraphTensor, axis: Option<DimExpr>) -> GraphTensor {
        let output_type = TensorType::new(input.shape().to_vec(), IrDType::I64);
        let mut attrs = std::collections::HashMap::new();
        if let Some(ax) = &axis {
            attrs.insert("axis".to_string(), ax.to_string());
        }
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::ArgMax,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Reshape tensor.
    pub fn reshape(&self, input: &GraphTensor, shape: &[DimExpr]) -> GraphTensor {
        let output_type = TensorType::new(shape.to_vec(), input.dtype());
        let mut attrs = HashMap::new();
        let shape_str: Vec<String> = shape.iter().map(|d| format!("{}", d)).collect();
        attrs.insert("shape".to_string(), shape_str.join(","));
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Reshape,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Transpose (reverses all dimensions).
    pub fn transpose(&self, input: &GraphTensor) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        output_shape.reverse();
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Transpose,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Flatten to 2D (keep first dim, product of rest).
    pub fn flatten(&self, input: &GraphTensor) -> GraphTensor {
        let shape = input.shape();
        let first = shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let rest: DimExpr = if shape.len() > 1 {
            dim_product(shape[1..].iter())
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![first, rest], input.dtype());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Flatten,
            vec![input.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// ReduceSum along a dimension.
    pub fn reduce_sum(&self, input: &GraphTensor, dim: usize, keepdim: bool) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        if dim < output_shape.len() {
            if keepdim {
                output_shape[dim] = DimExpr::Known(1);
            } else {
                output_shape.remove(dim);
            }
        }
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::ReduceSum,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// ReduceMean along a dimension.
    pub fn reduce_mean(&self, input: &GraphTensor, dim: usize, keepdim: bool) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        if dim < output_shape.len() {
            if keepdim {
                output_shape[dim] = DimExpr::Known(1);
            } else {
                output_shape.remove(dim);
            }
        }
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::ReduceMean,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Softmax along a dimension.
    pub fn softmax(&self, input: &GraphTensor, dim: usize) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Softmax,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Bias addition (element-wise broadcast add).
    pub fn bias_add(&self, input: &GraphTensor, bias: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::BiasAdd,
            vec![input.node_id, bias.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Concatenate tensors along a dimension.
    pub fn concat(&self, tensors: &[&GraphTensor], dim: usize) -> GraphTensor {
        let mut output_shape = tensors[0].shape().to_vec();
        if dim < output_shape.len() && !tensors.is_empty() {
            let mut total = DimExpr::Known(0);
            for t in tensors {
                let d = t.shape().get(dim).cloned().unwrap_or(DimExpr::Known(1));
                total = dim_add(&total, &d);
            }
            output_shape[dim] = total;
        }
        let output_type = TensorType::new(output_shape, tensors[0].dtype());
        let input_ids: Vec<NodeId> = tensors.iter().map(|t| t.node_id).collect();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Concat,
            input_ids,
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Shape ops
    // =========================================================================

    /// Slice a tensor along a dimension.
    pub fn slice(&self, input: &GraphTensor, dim: usize, start: usize, end: usize) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        if dim < output_shape.len() {
            output_shape[dim] = DimExpr::Known((end - start) as u64);
        }
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("dim".to_string(), dim.to_string());
        attrs.insert("start".to_string(), start.to_string());
        attrs.insert("end".to_string(), end.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Slice,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Squeeze (remove) a dimension.
    pub fn squeeze(&self, input: &GraphTensor, dim: usize) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        if dim < output_shape.len() {
            output_shape.remove(dim);
        }
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("dim".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Squeeze,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Unsqueeze (insert) a dimension of size 1.
    pub fn unsqueeze(&self, input: &GraphTensor, dim: usize) -> GraphTensor {
        let mut output_shape = input.shape().to_vec();
        output_shape.insert(dim, DimExpr::Known(1));
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("dim".to_string(), dim.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Unsqueeze,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Pooling
    // =========================================================================

    /// Max pooling 2D.
    pub fn max_pool2d(&self, input: &GraphTensor, kernel_size: usize, stride: usize, padding: usize) -> GraphTensor {
        let a_shape = input.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let channels = a_shape.get(1).cloned().unwrap_or(DimExpr::Known(1));
        let h_out = if a_shape.len() > 2 {
            conv_spatial_dim(&a_shape[2], &DimExpr::Known(kernel_size as u64), stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let w_out = if a_shape.len() > 3 {
            conv_spatial_dim(&a_shape[3], &DimExpr::Known(kernel_size as u64), stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![batch, channels, h_out, w_out], input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("kernel_size".to_string(), kernel_size.to_string());
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::MaxPool,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Average pooling 2D.
    pub fn avg_pool2d(&self, input: &GraphTensor, kernel_size: usize, stride: usize, padding: usize) -> GraphTensor {
        let a_shape = input.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let channels = a_shape.get(1).cloned().unwrap_or(DimExpr::Known(1));
        let h_out = if a_shape.len() > 2 {
            conv_spatial_dim(&a_shape[2], &DimExpr::Known(kernel_size as u64), stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let w_out = if a_shape.len() > 3 {
            conv_spatial_dim(&a_shape[3], &DimExpr::Known(kernel_size as u64), stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![batch, channels, h_out, w_out], input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("kernel_size".to_string(), kernel_size.to_string());
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::AvgPool,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Normalization
    // =========================================================================

    /// Batch normalization.
    pub fn batch_norm(&self, input: &GraphTensor, weight: &GraphTensor, bias: &GraphTensor, running_mean: &GraphTensor, running_var: &GraphTensor, eps: f64) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut attrs = HashMap::new();
        attrs.insert("eps".to_string(), eps.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::BatchNorm,
            vec![input.node_id, weight.node_id, bias.node_id, running_mean.node_id, running_var.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Layer normalization.
    pub fn layer_norm(&self, input: &GraphTensor, weight: &GraphTensor, bias: &GraphTensor, eps: f64) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut attrs = HashMap::new();
        attrs.insert("eps".to_string(), eps.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::LayerNorm,
            vec![input.node_id, weight.node_id, bias.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Tensor manipulation
    // =========================================================================

    /// Pad a tensor.
    pub fn pad(&self, input: &GraphTensor, pads: &[(usize, usize)]) -> GraphTensor {
        let input_shape = input.shape();
        let mut output_shape = input_shape.to_vec();
        for (i, &(lo, hi)) in pads.iter().enumerate() {
            if i < output_shape.len() {
                let lo_d = DimExpr::Known(lo as u64);
                let hi_d = DimExpr::Known(hi as u64);
                output_shape[i] = dim_add(&dim_add(&output_shape[i], &lo_d), &hi_d);
            }
        }
        let output_type = TensorType::new(output_shape, input.dtype());
        let mut attrs = HashMap::new();
        let pads_str: Vec<String> = pads.iter().map(|(lo, hi)| format!("{},{}", lo, hi)).collect();
        attrs.insert("pads".to_string(), pads_str.join(","));
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Pad,
            vec![input.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Gather elements along an axis.
    pub fn gather(&self, input: &GraphTensor, indices: &GraphTensor, axis: usize) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), axis.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Gather,
            vec![input.node_id, indices.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Scatter ND updates into a tensor.
    pub fn scatter_nd(&self, input: &GraphTensor, indices: &GraphTensor, updates: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::ScatterNd,
            vec![input.node_id, indices.node_id, updates.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Convolutions
    // =========================================================================

    /// Convolution 1D.
    pub fn conv1d(&self, input: &GraphTensor, weight: &GraphTensor, stride: usize, padding: usize) -> GraphTensor {
        let a_shape = input.shape();
        let w_shape = weight.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let out_channels = w_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let w_out = if a_shape.len() > 2 && w_shape.len() > 2 {
            conv_spatial_dim(&a_shape[2], &w_shape[2], stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![batch, out_channels, w_out], input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Conv1d,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Convolution 3D.
    pub fn conv3d(&self, input: &GraphTensor, weight: &GraphTensor, stride: usize, padding: usize) -> GraphTensor {
        let a_shape = input.shape();
        let w_shape = weight.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let out_channels = w_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let d_out = if a_shape.len() > 2 && w_shape.len() > 2 {
            conv_spatial_dim(&a_shape[2], &w_shape[2], stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let h_out = if a_shape.len() > 3 && w_shape.len() > 3 {
            conv_spatial_dim(&a_shape[3], &w_shape[3], stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let w_out = if a_shape.len() > 4 && w_shape.len() > 4 {
            conv_spatial_dim(&a_shape[4], &w_shape[4], stride, padding)
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![batch, out_channels, d_out, h_out, w_out], input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::Conv3d,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Transposed convolution 2D.
    pub fn conv_transpose2d(&self, input: &GraphTensor, weight: &GraphTensor, stride: usize, padding: usize) -> GraphTensor {
        let a_shape = input.shape();
        let w_shape = weight.shape();
        let batch = a_shape.first().cloned().unwrap_or(DimExpr::Known(1));
        let out_channels = w_shape.get(1).cloned().unwrap_or(DimExpr::Known(1));
        let h_out = if a_shape.len() > 2 && w_shape.len() > 2 {
            match (&a_shape[2], &w_shape[2]) {
                (DimExpr::Known(h), DimExpr::Known(kh)) => {
                    DimExpr::Known((h - 1) * stride as u64 + kh - 2 * padding as u64)
                }
                _ => {
                    let h_val = a_shape[2].evaluate().unwrap_or(0);
                    let kh_val = w_shape[2].evaluate().unwrap_or(1);
                    let estimated = (h_val.saturating_sub(1)) * stride as u64 + kh_val - 2 * padding as u64;
                    DimExpr::Bounded {
                        sym: format!("conv_trans_spatial({})", estimated),
                        max: estimated,
                    }
                }
            }
        } else {
            DimExpr::Known(1)
        };
        let w_out = if a_shape.len() > 3 && w_shape.len() > 3 {
            match (&a_shape[3], &w_shape[3]) {
                (DimExpr::Known(w), DimExpr::Known(kw)) => {
                    DimExpr::Known((w - 1) * stride as u64 + kw - 2 * padding as u64)
                }
                _ => {
                    let w_val = a_shape[3].evaluate().unwrap_or(0);
                    let kw_val = w_shape[3].evaluate().unwrap_or(1);
                    let estimated = (w_val.saturating_sub(1)) * stride as u64 + kw_val - 2 * padding as u64;
                    DimExpr::Bounded {
                        sym: format!("conv_trans_spatial({})", estimated),
                        max: estimated,
                    }
                }
            }
        } else {
            DimExpr::Known(1)
        };
        let output_type = TensorType::new(vec![batch, out_channels, h_out, w_out], input.dtype());
        let mut attrs = HashMap::new();
        attrs.insert("stride".to_string(), stride.to_string());
        attrs.insert("padding".to_string(), padding.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::ConvTranspose2d,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Activations
    // =========================================================================

    /// PReLU activation.
    pub fn prelu(&self, input: &GraphTensor, weight: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Prelu,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// RMS normalization.
    pub fn rms_norm(&self, input: &GraphTensor, weight: &GraphTensor, eps: f64) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut attrs = HashMap::new();
        attrs.insert("eps".to_string(), eps.to_string());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node_with_attrs(
            Opcode::RMSNorm,
            vec![input.node_id, weight.node_id],
            output_type.clone(),
            attrs,
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Embedding
    // =========================================================================

    /// Embedding lookup.
    pub fn embedding(&self, weight: &GraphTensor, indices: &GraphTensor) -> GraphTensor {
        let w_shape = weight.shape();
        let embed_dim = w_shape.get(1).cloned().unwrap_or(DimExpr::Known(1));
        let mut output_shape = indices.shape().to_vec();
        output_shape.push(embed_dim);
        let output_type = TensorType::new(output_shape, weight.dtype());
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Embedding,
            vec![weight.node_id, indices.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Element-wise ops
    // =========================================================================

    /// Element-wise power.
    pub fn pow(&self, input: &GraphTensor, exponent: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::Pow,
            vec![input.node_id, exponent.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Greater-than scalar comparison (output as 0.0/1.0 f32).
    pub fn gt_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = TensorType::new(input.shape().to_vec(), IrDType::F32);
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::GtScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Less-than scalar comparison (output as 0.0/1.0 f32).
    pub fn lt_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = TensorType::new(input.shape().to_vec(), IrDType::F32);
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::LtScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Equal scalar comparison (output as 0.0/1.0 f32).
    pub fn eq_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = TensorType::new(input.shape().to_vec(), IrDType::F32);
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::EqScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Add scalar element-wise.
    pub fn add_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::AddScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Multiply scalar element-wise.
    pub fn mul_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::MulScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    /// Divide scalar element-wise.
    pub fn div_scalar(&self, input: &GraphTensor, scalar: &GraphTensor) -> GraphTensor {
        let output_type = input.tensor_type.clone();
        let mut inner = self.inner.borrow_mut();
        let node_id = inner.graph.add_node(
            Opcode::DivScalar,
            vec![input.node_id, scalar.node_id],
            output_type.clone(),
        );
        GraphTensor::new(self.clone(), node_id, output_type)
    }

    // =========================================================================
    // Vision / Image ops
    // =========================================================================

    /// Upsample using nearest-neighbor interpolation.
    pub fn upsample_nearest2d(
        &self, input: &GraphTensor, scale_h: usize, scale_w: usize,
    ) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("scale_h".to_string(), scale_h.to_string());
        attrs.insert("scale_w".to_string(), scale_w.to_string());
        let input_shape = input.shape();
        let out_shape = if input_shape.len() >= 4 {
            vec![
                input_shape[0].clone(),
                input_shape[1].clone(),
                input_shape[2].mul(&DimExpr::Known(scale_h as u64)),
                input_shape[3].mul(&DimExpr::Known(scale_w as u64)),
            ]
        } else {
            input_shape.to_vec()
        };
        let tt = TensorType::new(out_shape, input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::UpsampleNearest2d,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Upsample using bilinear interpolation.
    pub fn upsample_bilinear2d(
        &self, input: &GraphTensor, scale_h: usize, scale_w: usize,
    ) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("scale_h".to_string(), scale_h.to_string());
        attrs.insert("scale_w".to_string(), scale_w.to_string());
        let input_shape = input.shape();
        let out_shape = if input_shape.len() >= 4 {
            vec![
                input_shape[0].clone(),
                input_shape[1].clone(),
                input_shape[2].mul(&DimExpr::Known(scale_h as u64)),
                input_shape[3].mul(&DimExpr::Known(scale_w as u64)),
            ]
        } else {
            input_shape.to_vec()
        };
        let tt = TensorType::new(out_shape, input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::UpsampleBilinear2d,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Adaptive average pooling to a fixed output size.
    pub fn adaptive_avg_pool2d(
        &self, input: &GraphTensor, output_h: usize, output_w: usize,
    ) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("output_h".to_string(), output_h.to_string());
        attrs.insert("output_w".to_string(), output_w.to_string());
        let input_shape = input.shape();
        let out_shape = if input_shape.len() >= 4 {
            vec![
                input_shape[0].clone(),
                input_shape[1].clone(),
                DimExpr::Known(output_h as u64),
                DimExpr::Known(output_w as u64),
            ]
        } else {
            input_shape.to_vec()
        };
        let tt = TensorType::new(out_shape, input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::AdaptiveAvgPool2d,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Repeat tensor along specified dimensions.
    pub fn repeat(&self, input: &GraphTensor, repeats: &[usize]) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        let repeats_str: Vec<String> = repeats.iter().map(|r| r.to_string()).collect();
        attrs.insert("repeats".to_string(), repeats_str.join(","));
        let input_shape = input.shape();
        let out_shape: Vec<DimExpr> = input_shape.iter().zip(repeats.iter())
            .map(|(d, &r)| d.mul(&DimExpr::Known(r as u64)))
            .collect();
        let tt = TensorType::new(out_shape, input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::Repeat,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Cumulative sum along a dimension.
    pub fn cumsum(
        &self, input: &GraphTensor, dim: usize, exclusive: bool, reverse: bool,
    ) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("dim".to_string(), dim.to_string());
        attrs.insert("exclusive".to_string(), (if exclusive { 1 } else { 0 }).to_string());
        attrs.insert("reverse".to_string(), (if reverse { 1 } else { 0 }).to_string());
        let tt = TensorType::new(input.shape().to_vec(), input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::CumSum,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Error function (Gauss error function).
    pub fn erf(&self, input: &GraphTensor) -> GraphTensor {
        let tt = TensorType::new(input.shape().to_vec(), input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node(Opcode::Erf, vec![input.node_id], tt.clone())
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Flip (reverse) tensor along specified dimensions.
    pub fn flip(&self, input: &GraphTensor, dims: &[usize]) -> GraphTensor {
        let mut attrs = std::collections::HashMap::new();
        let dims_str: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
        attrs.insert("dims".to_string(), dims_str.join(","));
        let tt = TensorType::new(input.shape().to_vec(), input.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node_with_attrs(
                Opcode::Flip,
                vec![input.node_id],
                tt.clone(),
                attrs,
            )
        };
        GraphTensor::new(self.clone(), node_id, tt)
    }

    /// Element-wise where: selects elements from x where condition is true, else from y.
    pub fn where_tensor(
        &self, condition: &GraphTensor, x: &GraphTensor, y: &GraphTensor,
    ) -> GraphTensor {
        let x_shape = x.shape();
        let out_tt = TensorType::new(x_shape.to_vec(), x.dtype());
        let node_id = {
            let mut inner = self.inner.borrow_mut();
            inner.graph.add_node(
                Opcode::Where,
                vec![condition.node_id, x.node_id, y.node_id],
                out_tt.clone(),
            )
        };
        GraphTensor::new(self.clone(), node_id, out_tt)
    }

    // ── Autograd / Backward ───────────────────────────────────────────────

    /// Build the backward graph for a given loss tensor.
    ///
    /// Calls [`build_backward_graph`](crate::autograd::build_backward_graph) to
    /// derive the gradient computation graph from the forward graph, then replaces
    /// the internal graph with the combined forward+backward graph.
    ///
    /// Returns a gradient [`GraphTensor`] for each registered input/parameter
    /// (in the same order they were registered via `input()` / `parameter()`).
    pub fn backward(&self, loss: &GraphTensor) -> Result<Vec<GraphTensor>, String> {
        // First, set the outputs so the forward graph knows what to compute
        {
            let mut inner = self.inner.borrow_mut();
            inner.graph.set_outputs(vec![loss.node_id]);
        }

        // Build the backward graph — this returns both the new graph and a
        // mapping from forward node IDs to their gradient accumulator node IDs.
        let (grad_graph, grads) = {
            let inner = self.inner.borrow();
            crate::autograd::build_backward_graph(&inner.graph, loss.node_id)?
        };

        // Update the shared inner state with the combined forward+backward graph
        {
            let mut inner = self.inner.borrow_mut();
            inner.graph = grad_graph;
        }

        // Build gradient GraphTensors for each recorded input/parameter
        let inner = self.inner.borrow();
        let mut grad_tensors = Vec::new();
        for &input_id in &inner.recorded_inputs {
            if let Some(&grad_id) = grads.get(&input_id) {
                if let Some(grad_node) = inner.graph.get_node(grad_id) {
                    grad_tensors.push(GraphTensor::new(
                        self.clone(),
                        grad_id,
                        grad_node.output_type.clone(),
                    ));
                }
            }
        }
        Ok(grad_tensors)
    }

    // ── Compilation and Execution ─────────────────────────────────────────

    /// Compile the graph for a given backend, returning the executable plan,
    /// memory plan, and the finalized compute graph.
    ///
    /// The output is a fully compiled plan ready for execution.
    pub fn compile<B: Backend>(
        &self,
        outputs: &[&GraphTensor],
        backend: B,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let mut graph: ComputeGraph;
        let recorded_inputs: Vec<NodeId>;
        {
            let inner = self.inner.borrow();
            graph = inner.graph.clone();
            recorded_inputs = inner.recorded_inputs.clone();
        }

        // Set up inputs and outputs
        graph.inputs = recorded_inputs;
        graph.outputs = outputs.iter().map(|t| t.node_id).collect();

        let executor = GraphExecutor::new(backend);
        let (plan, memory_plan, compiled_graph) = executor.compile_with_plan(&graph)?;

        Ok((plan, memory_plan, compiled_graph))
    }

    /// Convenience: compile and execute the graph in one call.
    ///
    /// `inputs` must correspond 1:1 with the order [`input()`](Self::input) /
    /// [`parameter()`](Self::parameter) was called.
    pub fn compile_and_execute<B: Backend>(
        &self,
        outputs: &[&GraphTensor],
        backend: B,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut graph: ComputeGraph;
        let recorded_inputs: Vec<NodeId>;
        {
            let inner = self.inner.borrow();
            graph = inner.graph.clone();
            recorded_inputs = inner.recorded_inputs.clone();
        }
        graph.inputs = recorded_inputs;
        graph.outputs = outputs.iter().map(|t| t.node_id).collect();

        let executor = GraphExecutor::new(backend);
        let (plan, memory_plan, compiled_graph) = executor.compile_with_plan(&graph)?;
        executor.execute(&compiled_graph, &plan, &memory_plan, inputs)
    }

    /// Return the current number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.inner.borrow().graph.node_count()
    }

    /// Return the number of recorded inputs.
    pub fn num_inputs(&self) -> usize {
        self.inner.borrow().recorded_inputs.len()
    }

    /// Clone the underlying `ComputeGraph`.
    pub fn to_graph(&self) -> ComputeGraph {
        self.inner.borrow().graph.clone()
    }
}

// =============================================================================
// DimExpr arithmetic helpers
// =============================================================================

/// Multiply two `DimExpr` values symbolically.
fn dim_mul(a: &DimExpr, b: &DimExpr) -> DimExpr {
    a.mul(b)
}

/// Add two `DimExpr` values symbolically.
fn dim_add(a: &DimExpr, b: &DimExpr) -> DimExpr {
    a.add(b)
}

/// Product of an iterator of `DimExpr` values.
fn dim_product<'a, I>(dims: I) -> DimExpr
where
    I: IntoIterator<Item = &'a DimExpr>,
{
    let mut iter = dims.into_iter();
    let first = iter.next().cloned().unwrap_or(DimExpr::Known(1));
    iter.fold(first, |acc, d| dim_mul(&acc, d))
}

/// Compute the spatial output dimension for a convolution:
/// `out = (input + 2*padding - kernel) / stride + 1`.
fn conv_spatial_dim(input_dim: &DimExpr, kernel_dim: &DimExpr, stride: usize, padding: usize) -> DimExpr {
    let s = stride as u64;
    let p = padding as u64;
    match (input_dim, kernel_dim) {
        (DimExpr::Known(h), DimExpr::Known(k)) => {
            DimExpr::Known(((h + 2 * p).saturating_sub(*k)) / s + 1)
        }
        _ => {
            // Attempt partial evaluation
            let h_val = input_dim.evaluate();
            let k_val = kernel_dim.evaluate();
            match (h_val, k_val) {
                (Some(h), Some(k)) => {
                    DimExpr::Known(((h + 2 * p).saturating_sub(k)) / s + 1)
                }
                _ => {
                    // Cannot fully evaluate — produce a Bounded expression with provenance
                    let h_eval = input_dim.evaluate().unwrap_or(SYMBOL_DIM_MAX.load(Ordering::Relaxed));
                    let k_eval = kernel_dim.evaluate().unwrap_or(1);
                    let estimated = ((h_eval + 2 * p).saturating_sub(k_eval)) / s + 1;
                    let sym_name = match (input_dim, kernel_dim) {
                        (DimExpr::Symbol(s), DimExpr::Symbol(t)) => {
                            format!("conv_spatial({},{})", s, t)
                        }
                        (DimExpr::Symbol(s), _) => format!("conv_spatial({})", s),
                        (DimExpr::Bounded { sym, .. }, DimExpr::Symbol(t)) => {
                            format!("conv_spatial({},{})", sym, t)
                        }
                        (DimExpr::Bounded { sym, .. }, _) => format!("conv_spatial({})", sym),
                        (_, DimExpr::Symbol(t)) => format!("conv_spatial({})", t),
                        _ => format!("conv_spatial({})", estimated),
                    };
                    DimExpr::Bounded {
                        sym: sym_name,
                        max: estimated,
                    }
                }
            }
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self { Self::new() }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use bytemuck;

    /// Create an f32 input tensor of f32 bytes.
    fn f32_data(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    /// Read f32 data from bytes.
    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(bytes).to_vec()
    }

    #[test]
    fn test_simple_add() {
        let g = GraphBuilder::new();
        let a = g.input(&[4], IrDType::F32);
        let b = g.input(&[4], IrDType::F32);
        let c = g.add(&a, &b);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let b_data = f32_data(&[5.0, 6.0, 7.0, 8.0]);

        let result = g.compile_and_execute(&[&c], CpuBackend, &[&a_data, &b_data]).unwrap();
        let out = read_f32(&result[0]);

        assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_input_with_dims() {
        let g = GraphBuilder::new();
        // Create an input with a symbolic batch dimension and a known feature dim
        let batch = DimExpr::Symbol("N".into());
        let feat = DimExpr::Known(64);
        let x = g.input_with_dims(&[batch, feat], IrDType::F32);

        // Verify symbolic shapes propagate through ops
        let w = g.input(&[64, 10], IrDType::F32);
        let logits = g.matmul(&x, &w);
        let loss = g.reduce_mean(&logits, 1, false);

        // The matmul output should preserve the symbolic batch dim
        let logits_shape = logits.shape();
        assert_eq!(logits_shape.len(), 2, "matmul shape should be 2D");
        assert_eq!(logits_shape[0], DimExpr::Symbol("N".into()),
            "batch dim should remain symbolic through matmul");
        assert_eq!(logits_shape[1], DimExpr::Known(10),
            "feature dim should be 10");

        // After reduce_mean on axis 1 (no keepdim): shape should be [N]
        let loss_shape = loss.shape();
        assert_eq!(loss_shape.len(), 1, "loss shape should be 1D after reduce_mean");
        assert_eq!(loss_shape[0], DimExpr::Symbol("N".into()),
            "batch dim should remain symbolic through reduce_mean");
    }

    #[test]
    fn test_add_via_operator() {
        let g = GraphBuilder::new();
        let a = g.input(&[3], IrDType::F32);
        let b = g.input(&[3], IrDType::F32);
        let c = a.clone() + b.clone();

        let a_data = f32_data(&[1.0, 2.0, 3.0]);
        let b_data = f32_data(&[4.0, 5.0, 6.0]);

        let result = g.compile_and_execute(&[&c], CpuBackend, &[&a_data, &b_data]).unwrap();
        let out = read_f32(&result[0]);

        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_mul_div_neg() {
        let g = GraphBuilder::new();
        let a = g.input(&[2], IrDType::F32);
        let b = g.input(&[2], IrDType::F32);

        let s = g.sub(&a, &b);
        let m = g.mul(&a, &b);
        let d = g.div(&a, &b);
        let n = g.neg(&a);

        let a_data = f32_data(&[10.0, 20.0]);
        let b_data = f32_data(&[2.0, 5.0]);

        let result = g.compile_and_execute(
            &[&s, &m, &d, &n], CpuBackend, &[&a_data, &b_data],
        ).unwrap();

        assert_eq!(read_f32(&result[0]), vec![8.0, 15.0]);   // sub
        assert_eq!(read_f32(&result[1]), vec![20.0, 100.0]); // mul
        assert_eq!(read_f32(&result[2]), vec![5.0, 4.0]);    // div
        assert_eq!(read_f32(&result[3]), vec![-10.0, -20.0]); // neg
    }

    #[test]
    fn test_relu() {
        let g = GraphBuilder::new();
        let a = g.input(&[5], IrDType::F32);
        let r = g.relu(&a);

        let a_data = f32_data(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = g.compile_and_execute(&[&r], CpuBackend, &[&a_data]).unwrap();
        assert_eq!(read_f32(&result[0]), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_symbolic_flatten_conv_concat() {
        let g = GraphBuilder::new();
        let n = DimExpr::Symbol("N".into());
        let c = DimExpr::Known(3);
        let h = DimExpr::Known(32);
        let w = DimExpr::Known(32);
        let x = g.input_with_dims(&[n.clone(), c.clone(), h.clone(), w.clone()], IrDType::F32);

        // Flatten from [N, 3, 32, 32] → [N, 3072]
        let flat = g.flatten(&x);
        let flat_shape = flat.shape();
        assert_eq!(flat_shape.len(), 2, "flatten should produce 2D shape");
        assert_eq!(flat_shape[0], DimExpr::Symbol("N".into()),
            "flatten should preserve symbolic batch dim");
        assert_eq!(flat_shape[1], DimExpr::Known(3072),
            "flatten should compute product of remaining dims: 3*32*32=3072");

        // Concat along batch dim
        let y = g.input_with_dims(&[n, c, h, w], IrDType::F32);
        let cat = g.concat(&[&x, &y], 0);
        let cat_shape = cat.shape();
        assert_eq!(cat_shape[0], DimExpr::Bounded { sym: "2*N".into(), max: 8192 },
            "concat along symbolic dim should produce canonical 2*N");

        // Concat along known dim
        let cat_c = g.concat(&[&x, &y], 1);
        let cat_c_shape = cat_c.shape();
        assert_eq!(cat_c_shape[1], DimExpr::Known(6),
            "concat along known dim should produce sum: 3+3=6");
    }

    #[test]
    fn test_dynamic_batch_matmul() {
        let g = GraphBuilder::new();
        // Input with symbolic batch dim N, feature dim 64
        let n = DimExpr::Symbol("N".into());
        let x = g.input_with_dims(&[n, DimExpr::Known(64)], IrDType::F32);
        // Weight: [64, 10]
        let w = g.input(&[64, 10], IrDType::F32);
        // MatMul: [N, 64] @ [64, 10] → [N, 10]
        let logits = g.matmul(&x, &w);

        // Compile ONCE, execute with multiple batch sizes.
        let (plan, memory_plan, compiled_graph) = g.compile(&[&logits], CpuBackend).unwrap();
        let executor = GraphExecutor::new(CpuBackend);

        // Two different batch sizes — both should work at runtime
        // Batch = 3: input data is 3*64*4 = 768 bytes
        let x_data_3 = f32_data(&(0..192).map(|i| i as f32).collect::<Vec<_>>());
        let w_data = f32_data(&(0..640).map(|i| (i % 10) as f32).collect::<Vec<_>>());

        // Execute with batch=3
        let result_3 = executor.execute(
            &compiled_graph, &plan, &memory_plan, &[&x_data_3, &w_data],
        ).unwrap();
        let out_3 = read_f32(&result_3[0]);
        // Output should be [3, 10] = 30 elements
        assert_eq!(out_3.len(), 30, "batch=3 should produce 30 elements, got {}", out_3.len());

        // Batch = 1: input data is 1*64*4 = 256 bytes — same compiled plan
        let x_data_1 = f32_data(&(0..64).map(|i| i as f32).collect::<Vec<_>>());
        let result_1 = executor.execute(
            &compiled_graph, &plan, &memory_plan, &[&x_data_1, &w_data],
        ).unwrap();
        let out_1 = read_f32(&result_1[0]);
        // Output should be [1, 10] = 10 elements
        assert_eq!(out_1.len(), 10, "batch=1 should produce 10 elements, got {}", out_1.len());

        // Verify the actual values are correct by computing manually for batch=1
        // x = [0,1,2,...,63], w column j = [j%10, j%10, ..., j%10] (64 times)
        // logits[0,j] = sum_{k=0}^{63} k * (j%10)
        let expected_0 = (0..64).map(|k| k as f32 * (0 % 10) as f32).sum::<f32>();
        assert!((out_1[0] - expected_0).abs() < 1e-3,
            "expected logit[0,0]={}, got {}", expected_0, out_1[0]);

        // Batch = 7 (another shape) — verify the pattern generalizes
        let x_data_7 = f32_data(&(0..448).map(|i| i as f32).collect::<Vec<_>>()); // 7*64 = 448 f32s
        let result_7 = executor.execute(
            &compiled_graph, &plan, &memory_plan, &[&x_data_7, &w_data],
        ).unwrap();
        let out_7 = read_f32(&result_7[0]);
        assert_eq!(out_7.len(), 70, "batch=7 should produce 70 elements, got {}", out_7.len());
    }

    #[test]
    fn test_dynamic_batch_reduce_mean() {
        let g = GraphBuilder::new();
        // Input: [N, 4] with symbolic batch dim
        let n = DimExpr::Symbol("N".into());
        let x = g.input_with_dims(&[n, DimExpr::Known(4)], IrDType::F32);

        // Reduce mean over dim 1 (no keepdim): [N, 4] → [N]
        let mean = g.reduce_mean(&x, 1, false);
        // Reduce sum over dim 1 (no keepdim): [N, 4] → [N]
        let sum = g.reduce_sum(&x, 1, false);

        // Test with batch=2: input data is 2*4*4 = 32 bytes
        let x_data_2 = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result_2 = g.compile_and_execute(
            &[&mean, &sum], CpuBackend, &[&x_data_2],
        ).unwrap();
        // mean = [(1+2+3+4)/4=2.5, (5+6+7+8)/4=6.5]
        assert_eq!(read_f32(&result_2[0]), vec![2.5, 6.5],
            "reduce_mean with batch=2 failed: got {:?}", read_f32(&result_2[0]));
        // sum = [1+2+3+4=10, 5+6+7+8=26]
        assert_eq!(read_f32(&result_2[1]), vec![10.0, 26.0],
            "reduce_sum with batch=2 failed: got {:?}", read_f32(&result_2[1]));

        // Test with batch=1: input data is 1*4*4 = 16 bytes
        let x_data_1 = f32_data(&[10.0, 20.0, 30.0, 40.0]);
        let result_1 = g.compile_and_execute(
            &[&mean, &sum], CpuBackend, &[&x_data_1],
        ).unwrap();
        // mean = 25, sum = 100
        assert_eq!(read_f32(&result_1[0]), vec![25.0],
            "reduce_mean with batch=1 failed: got {:?}", read_f32(&result_1[0]));
        assert_eq!(read_f32(&result_1[1]), vec![100.0],
            "reduce_sum with batch=1 failed: got {:?}", read_f32(&result_1[1]));
    }

    #[test]
    fn test_shape_assertion_matmul_mismatch() {
        let g = GraphBuilder::new();
        // A: [2, 64], B: [32, 10] — inner dim mismatch (64 vs 32).
        // Compile catches this via Known dim comparison.
        let a = g.input(&[2, 64], IrDType::F32);
        let b = g.input(&[32, 10], IrDType::F32);
        let c = g.matmul(&a, &b);

        let a_data = f32_data(&(0..128).map(|i| i as f32).collect::<Vec<_>>());
        let b_data = f32_data(&(0..320).map(|i| i as f32).collect::<Vec<_>>());

        let result = g.compile_and_execute(&[&c], CpuBackend, &[&a_data, &b_data]);
        assert!(result.is_err(), "expected shape mismatch error, got {:?}", result);
        let err = result.err().unwrap();
        let msg = format!("{}", err);
        // Compile catches this, so error should mention MatMul or inner dim
        assert!(msg.contains("MatMul") || msg.contains("inner dim"),
            "expected matmul shape error, got: {}", msg);
    }

    #[test]
    fn test_shape_assertion_broadcast_mismatch() {
        let g = GraphBuilder::new();
        // A: [3, 4], B: [2, 4] — can't broadcast dim 0 (3 vs 2, neither is 1)
        let a = g.input(&[3, 4], IrDType::F32);
        let b = g.input(&[2, 4], IrDType::F32);
        let c = g.add(&a, &b);

        let a_data = f32_data(&(0..12).map(|i| i as f32).collect::<Vec<_>>());
        let b_data = f32_data(&(0..8).map(|i| i as f32).collect::<Vec<_>>());

        let result = g.compile_and_execute(&[&c], CpuBackend, &[&a_data, &b_data]);
        assert!(result.is_err(), "expected broadcast error, got {:?}", result);
        let err = result.err().unwrap();
        let msg = format!("{}", err);
        assert!(msg.contains("broadcast") || msg.contains("Broadcast") || msg.contains("shape validation"),
            "expected broadcast shape error, got: {}", msg);
    }

    #[test]
    fn test_compiled_plan_reuse_across_shapes() {
        use crate::backend::executor::GraphExecutor;

        let g = GraphBuilder::new();
        // Input: [N, 4] with symbolic batch dim
        let n = DimExpr::Symbol("N".into());
        let x = g.input_with_dims(&[n, DimExpr::Known(4)], IrDType::F32);
        // MatMul weight: [4, 3]
        let w = g.input(&[4, 3], IrDType::F32);
        // Output: [N, 3]
        let out = g.matmul(&x, &w);

        // Compile ONCE
        let (plan, memory_plan, compiled_graph) = g.compile(&[&out], CpuBackend).unwrap();
        let executor = GraphExecutor::new(CpuBackend);

        // Execute with batch=2
        let x_data_2 = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let w_data = f32_data(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let result_2 = executor.execute(
            &compiled_graph, &plan, &memory_plan, &[&x_data_2, &w_data],
        ).unwrap();
        let out_2 = read_f32(&result_2[0]);
        // batch=2, feat=3 → 6 elements
        // Row 0: [1,2,3,4] @ W = [1*1+2*0+3*0+4*1=5, 1*0+2*1+3*0+4*1=6, 1*0+2*0+3*1+4*1=7]
        // Row 1: [5,6,7,8] @ W = [5+8=13, 6+8=14, 7+8=15]
        assert_eq!(out_2.len(), 6, "batch=2 should produce 6 elements");
        assert!((out_2[0] - 5.0).abs() < 1e-4);
        assert!((out_2[4] - 14.0).abs() < 1e-4);

        // Execute SAME plan with batch=1 — no recompilation
        let x_data_1 = f32_data(&[10.0, 20.0, 30.0, 40.0]);
        let result_1 = executor.execute(
            &compiled_graph, &plan, &memory_plan, &[&x_data_1, &w_data],
        ).unwrap();
        let out_1 = read_f32(&result_1[0]);
        // batch=1, feat=3 → 3 elements
        // [10,20,30,40] @ W = [10+40=50, 20+40=60, 30+40=70]
        assert_eq!(out_1.len(), 3, "batch=1 should produce 3 elements, got {}", out_1.len());
        assert!((out_1[0] - 50.0).abs() < 1e-4);
        assert!((out_1[2] - 70.0).abs() < 1e-4);
    }

    #[test]
    fn test_dynamic_multi_symbol() {
        let g = GraphBuilder::new();
        // Input A: [B, T, 64] — two symbolic dims
        // Bias: [B, 1, 128] — shares B, lets us infer B from its data
        // Weight: [64, 128]
        let b = DimExpr::Symbol("B".into());
        let t = DimExpr::Symbol("T".into());
        let a = g.input_with_dims(&[b.clone(), t, DimExpr::Known(64)], IrDType::F32);
        let bias = g.input_with_dims(&[b, DimExpr::Known(1), DimExpr::Known(128)], IrDType::F32);
        let w = g.input(&[64, 128], IrDType::F32);

        // MatMul: [B, T, 64] @ [64, 128] → [B, T, 128]
        let mm = g.matmul(&a, &w);
        // Add bias (broadcasts): [B, T, 128] + [B, 1, 128] → [B, T, 128]
        let out = g.add(&mm, &bias);

        // Batch=2, seq_len=3: A data = 2*3*64*4 = 1536 bytes (384 f32s)
        let a_data = f32_data(&(0..384).map(|i| i as f32).collect::<Vec<_>>());
        // Bias data = 2*1*128*4 = 1024 bytes (256 f32s)
        let bias_data = f32_data(&(0..256).map(|i| i as f32).collect::<Vec<_>>());
        // Weight data = 64*128*4 = 32768 bytes (8192 f32s)
        let w_data = f32_data(&(0..8192).map(|i| (i % 128) as f32).collect::<Vec<_>>());

        // Batch=2, seq_len=3: A data = 2*3*64*4 = 1536 bytes
        let a_data = f32_data(&(0..384).map(|i| i as f32).collect::<Vec<_>>());
        // Bias data = 2*128*4 = 1024 bytes
        let bias_data = f32_data(&(0..256).map(|i| i as f32).collect::<Vec<_>>());
        // Weight data = 64*128*4 = 32768 bytes
        let w_data = f32_data(&(0..8192).map(|i| (i % 128) as f32).collect::<Vec<_>>());

        let result = g.compile_and_execute(
            &[&out], CpuBackend, &[&a_data, &bias_data, &w_data],
        ).unwrap();
        let out_f32 = read_f32(&result[0]);
        // Output should be [2, 3, 128] = 768 elements
        assert_eq!(out_f32.len(), 768,
            "multi-symbol output should be 2*3*128=768 elements, got {}", out_f32.len());
    }

    #[test]
    fn test_activations() {
        let g = GraphBuilder::new();
        let a = g.input(&[4], IrDType::F32);

        let sig = g.sigmoid(&a);
        let tan = g.tanh(&a);
        let exp = g.exp(&a);
        let sq = g.sqrt(&a);
        let ab = g.abs(&a);

        let a_data = f32_data(&[0.0, 1.0, -1.0, 2.0]);
        let result = g.compile_and_execute(
            &[&sig, &tan, &exp, &sq, &ab], CpuBackend, &[&a_data],
        ).unwrap();

        let sig_out = read_f32(&result[0]);
        let tan_out = read_f32(&result[1]);
        let exp_out = read_f32(&result[2]);
        let sqrt_out = read_f32(&result[3]);
        let abs_out = read_f32(&result[4]);

        // sigmoid
        assert!((sig_out[0] - 0.5).abs() < 1e-5);
        assert!((sig_out[1] - 0.7310586).abs() < 1e-5);
        // tanh
        assert!((tan_out[0] - 0.0).abs() < 1e-5);
        assert!((tan_out[1] - 0.76159416).abs() < 1e-5);
        // exp
        assert!((exp_out[0] - 1.0).abs() < 1e-5);
        assert!((exp_out[1] - 2.7182818).abs() < 1e-5);
        // sqrt
        assert!((sqrt_out[3] - 1.4142135).abs() < 1e-5);
        // abs
        assert!((abs_out[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul() {
        let g = GraphBuilder::new();
        // (2,3) @ (3,2) -> (2,2)
        let a = g.input(&[2, 3], IrDType::F32);
        let b = g.input(&[3, 2], IrDType::F32);
        let c = g.matmul(&a, &b);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b_data = f32_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let result = g.compile_and_execute(&[&c], CpuBackend, &[&a_data, &b_data]).unwrap();
        let out = read_f32(&result[0]);

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        // = [[58, 64], [139, 154]]
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_chain_add_relu() {
        let g = GraphBuilder::new();
        let a = g.input(&[3], IrDType::F32);
        let b = g.input(&[3], IrDType::F32);
        let c = g.add(&a, &b);
        let d = g.relu(&c);

        let a_data = f32_data(&[-1.0, 2.0, -3.0]);
        let b_data = f32_data(&[4.0, -5.0, 6.0]);

        let result = g.compile_and_execute(&[&d], CpuBackend, &[&a_data, &b_data]).unwrap();
        // -1+4=3->relu=3, 2-5=-3->relu=0, -3+6=3->relu=3
        assert_eq!(read_f32(&result[0]), vec![3.0, 0.0, 3.0]);
    }

    #[test]
    fn test_reshape_transpose_flatten() {
        let g = GraphBuilder::new();
        let a = g.input(&[2, 3], IrDType::F32);
        let t = g.transpose(&a);
        let f = g.flatten(&a);
        let r = g.reshape(&a, &[DimExpr::Known(3), DimExpr::Known(2)]);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = g.compile_and_execute(
            &[&t, &f, &r], CpuBackend, &[&a_data],
        ).unwrap();

        // Transpose: (2,3) -> (3,2) = [1,4,2,5,3,6]
        assert_eq!(read_f32(&result[0]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Flatten: (2,3) -> (2, 6)?
        // Actually flatten is (first_dim, product_of_rest) so for (2,3): (2, 3)
        assert_eq!(read_f32(&result[1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Reshape to (3,2)
        assert_eq!(read_f32(&result[2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reduce_sum_mean() {
        let g = GraphBuilder::new();
        let a = g.input(&[2, 3], IrDType::F32);
        let sum_keep = g.reduce_sum(&a, 1, true);   // keepdim
        let sum_no = g.reduce_sum(&a, 1, false);    // no keepdim
        let mean = g.reduce_mean(&a, 1, false);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = g.compile_and_execute(
            &[&sum_keep, &sum_no, &mean], CpuBackend, &[&a_data],
        ).unwrap();

        // sum over dim 1 (rows): [1+2+3=6, 4+5+6=15]
        assert_eq!(read_f32(&result[0]), vec![6.0, 15.0]);
        assert_eq!(read_f32(&result[1]), vec![6.0, 15.0]);
        // mean: [6/3=2, 15/3=5]
        assert_eq!(read_f32(&result[2]), vec![2.0, 5.0]);
    }

    #[test]
    fn test_matmul_add_relu_fusion() {
        let g = GraphBuilder::new();
        // (2,3) @ (3,2) -> (2,2) + bias(2) -> relu
        let a = g.input(&[2, 3], IrDType::F32);
        let w = g.input(&[3, 2], IrDType::F32);
        let b = g.input(&[2], IrDType::F32);

        let mm = g.matmul(&a, &w);
        let biased = g.bias_add(&mm, &b);
        let out = g.relu(&biased);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w_data = f32_data(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let b_data = f32_data(&[-50.0, 0.0]); // bias to test relu

        let result = g.compile_and_execute(&[&out], CpuBackend, &[&a_data, &w_data, &b_data]).unwrap();
        let out_f32 = read_f32(&result[0]);

        // MM = [[1*1+2*0+3*1=4, 1*0+2*1+3*0=2], [4*1+5*0+6*1=10, 4*0+5*1+6*0=5]]
        // + bias [-50, 0] = [[4-50=-46, 2+0=2], [10-50=-40, 5+0=5]]
        // relu = [[0, 2], [0, 5]]
        assert_eq!(out_f32, vec![0.0, 2.0, 0.0, 5.0]);
    }

    #[test]
    fn test_multiple_outputs() {
        let g = GraphBuilder::new();
        let a = g.input(&[2], IrDType::F32);
        let b = g.input(&[2], IrDType::F32);

        let sum = g.add(&a, &b);
        let prod = g.mul(&a, &b);

        let a_data = f32_data(&[3.0, 4.0]);
        let b_data = f32_data(&[7.0, 8.0]);

        let result = g.compile_and_execute(
            &[&sum, &prod], CpuBackend, &[&a_data, &b_data],
        ).unwrap();

        assert_eq!(read_f32(&result[0]), vec![10.0, 12.0]); // sum
        assert_eq!(read_f32(&result[1]), vec![21.0, 32.0]); // prod
    }

    #[test]
    fn test_op_relu_fusion() {
        let g = GraphBuilder::new();
        let a = g.input(&[4], IrDType::F32);
        let b = g.input(&[4], IrDType::F32);

        // Add + Relu should be fused by fuse_op_relu
        let c = g.add(&a, &b);
        let d = g.relu(&c);

        let a_data = f32_data(&[-5.0, -2.0, 1.0, 4.0]);
        let b_data = f32_data(&[3.0, -1.0, 2.0, -3.0]);

        let result = g.compile_and_execute(&[&d], CpuBackend, &[&a_data, &b_data]).unwrap();
        // c = [-2, -3, 3, 1], relu = [0, 0, 3, 1]
        assert_eq!(read_f32(&result[0]), vec![0.0, 0.0, 3.0, 1.0]);
    }

    #[test]
    fn test_matmul_relu_fusion() {
        let g = GraphBuilder::new();
        let a = g.input(&[2, 3], IrDType::F32);
        let w = g.input(&[3, 2], IrDType::F32);

        // MatMul + Relu should be fused by fuse_op_relu
        let mm = g.matmul(&a, &w);
        let out = g.relu(&mm);

        let a_data = f32_data(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
        let w_data = f32_data(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);

        let result = g.compile_and_execute(&[&out], CpuBackend, &[&a_data, &w_data]).unwrap();
        let out_f32 = read_f32(&result[0]);

        // MM = [[1*1+(-2)*0+3*1=4, 1*0+(-2)*1+3*0=-2], [-4*1+5*0+(-6)*1=-10, -4*0+5*1+(-6)*0=5]]
        // relu = [[4, 0], [0, 5]]
        assert_eq!(out_f32, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_backward_basic() {
        let g = GraphBuilder::new();
        let a = g.input(&[4], IrDType::F32);
        let b = g.input(&[4], IrDType::F32);
        let c = g.add(&a, &b);
        let loss = g.reduce_mean(&c, 0, false);

        // After backward, the graph should contain forward + backward nodes
        let grads = g.backward(&loss).unwrap();
        assert_eq!(grads.len(), 2, "should return gradients for both inputs");

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let b_data = f32_data(&[5.0, 6.0, 7.0, 8.0]);

        // Compile and execute GRADIENT outputs from the combined backward graph.
        // The backward graph contains both forward and backward nodes; requesting
        // gradient outputs ensures the forward pass feeds into the correct accumulators.
        let grad_result = g.compile_and_execute(
            &[&grads[0], &grads[1]],
            CpuBackend,
            &[&a_data, &b_data],
        ).unwrap();

        // d(loss)/da = d(mean(a+b))/da = [1/4, 1/4, 1/4, 1/4]
        let grad_a = read_f32(&grad_result[0]);
        assert_eq!(grad_a.len(), 4, "grad_a len wrong, got {:?}", grad_a);
        for &g_val in &grad_a {
            assert!((g_val - 0.25).abs() < 1e-5,
                "grad_a value {} not close to 0.25; grad_result[0]={:?} grad_result[1]={:?}",
                g_val, &grad_a, &read_f32(&grad_result[1]));
        }

        // d(loss)/db = d(mean(a+b))/db = [1/4, 1/4, 1/4, 1/4]
        let grad_b = read_f32(&grad_result[1]);
        assert_eq!(grad_b.len(), 4, "grad_b len wrong, got {:?}", grad_b);
        for &g_val in &grad_b {
            assert!((g_val - 0.25).abs() < 1e-5,
                "grad_b value {} not close to 0.25; grad_result[0]={:?} grad_result[1]={:?}",
                g_val, &read_f32(&grad_result[0]), &grad_b);
        }
    }
}
