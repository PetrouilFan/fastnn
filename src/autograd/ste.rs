//! Straight-Through Estimator (STE) gradients for quantization ops.
//!
//! These gradients allow QuantizeActivations and DequantizeActivations
//! to be differentiable during QAT/fine-tuning by passing gradients
//! through unchanged (STE).
//!
//! Note: The actual STE implementation for QuantizeActivations/DequantizeActivations
//! is in `build_backward_graph` (autograd.rs) which pattern-matches on opcodes.
//! This file provides reference STE logic for external use (e.g., Python bindings).

/// STE for fake quantization: Quantize → Dequantize
/// Forward: x → quantize → dequantize (fake quantize)
/// Backward: gradient passes through unchanged (STE)
pub fn ste_quantize_forward(x: f32, scale: f32, zero_point: f32) -> f32 {
    // Fake quantize: quantize to int8 then dequantize back to float
    // q = round(x / scale) + zero_point
    // dq = (q - zero_point) * scale
    let q = ((x / scale).round() + zero_point).clamp(-128.0, 127.0);
    let dq = (q - zero_point) * scale;
    dq
}

/// STE backward: gradient passes through unchanged
pub fn ste_backward(grad_output: f32) -> f32 {
    grad_output
}

/// STE for fake dequantization
/// Forward: identity (already dequantized in inference)
/// Backward: gradient passes through unchanged
pub fn ste_dequantize_forward(x: f32) -> f32 {
    x
}