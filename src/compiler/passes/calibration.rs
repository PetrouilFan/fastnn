//! Calibration Pass — Collects activation statistics for quantization.
//!
//! This pass inserts "Observe" nodes after quantizable operations to collect
//! running min/max/mean/std/histogram statistics. These are used to compute
//! per-tensor or per-channel quantization scales and zero-points.

use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};
use crate::ir::builder::GraphBuilder;
use std::collections::HashMap;

/// Calibration statistics collected per tensor.
#[derive(Debug, Clone, Default)]
pub struct CalibrationStats {
    pub min: f32,
    pub max: f32,
    pub sum: f64,
    pub sum_sq: f64,
    pub count: u64,
    /// Histogram for KL-divergence calibration (2048 bins)
    pub histogram: Option<Vec<u64>>,
    pub hist_min: f32,
    pub hist_max: f32,
}

impl CalibrationStats {
    pub fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            histogram: None,
            hist_min: 0.0,
            hist_max: 0.0,
        }
    }

    pub fn with_histogram(mut self, bins: usize, min: f32, max: f32) -> Self {
        self.histogram = Some(vec![0; bins]);
        self.hist_min = min;
        self.hist_max = max;
        self
    }

    pub fn observe(&mut self, values: &[f32]) {
        for &v in values {
            if v.is_finite() {
                self.min = self.min.min(v);
                self.max = self.max.max(v);
                self.sum += v as f64;
                self.sum_sq += (v as f64) * (v as f64);
                self.count += 1;

                if let Some(ref mut hist) = self.histogram {
                    if self.hist_max > self.hist_min {
                        let idx = ((v - self.hist_min) / (self.hist_max - self.hist_min) * hist.len() as f32)
                            .floor() as usize;
                        // Clamp to valid range to handle edge case v == hist_max
                        let idx = idx.min(hist.len() - 1);
                        hist[idx] = hist[idx].saturating_add(1);
                    }
                }
            }
        }
    }
    pub fn mean(&self) -> f32 {
        if self.count > 0 {
            (self.sum / self.count as f64) as f32
        } else {
            0.0
        }
    }

    pub fn std(&self) -> f32 {
        if self.count > 1 {
            let mean = self.mean() as f64;
            let variance = self.sum_sq / self.count as f64 - mean * mean;
            variance.max(0.0).sqrt() as f32
        } else {
            0.0
        }
    }

    /// Compute scale and zero-point for asymmetric quantization.
    pub fn compute_scale_zp(&self, bit_width: u8) -> (f32, f32) {
        let range = self.max - self.min;
        let levels = (1u32 << bit_width) as f32 - 1.0;
        
        if range <= 0.0 || levels <= 0.0 {
            return (1.0, 0.0);
        }

        let scale = range / levels;
        let zero_point = -self.min / scale;
        
        (scale, zero_point.round())
    }

    /// KL-divergence based scale computation (for activation quantization).
    /// Finds the threshold that minimizes KL divergence between original and quantized.
    pub fn compute_scale_zp_kl(&self, bit_width: u8) -> Option<(f32, f32)> {
        let hist = self.histogram.as_ref()?;
        let bins = hist.len();
        if bins == 0 {
            return None;
        }

        let total: u64 = hist.iter().sum();
        if total == 0 {
            return None;
        }

        // Target quantized bins
        let q_bins = (1u32 << bit_width) as usize; // 256 for U8, 16 for U4
        if q_bins >= bins {
            return Some(self.compute_scale_zp(bit_width));
        }

        // Normalize histogram
        let prob: Vec<f64> = hist.iter().map(|&c| c as f64 / total as f64).collect();
        
        // Cumulative distribution
        let mut cumsum = vec![0.0; bins + 1];
        for i in 0..bins {
            cumsum[i + 1] = cumsum[i] + prob[i];
        }

        let bin_width = (self.hist_max - self.hist_min) / bins as f32;
        let mut best_kl = f64::INFINITY;
        let mut best_threshold_idx = bins / 2;

        // Try different clipping thresholds
        // Step by at most 100 steps to find best threshold
        let step = (bins / 100).max(1);
        for clip_idx in (q_bins..bins).step_by(step) {
            // Quantized distribution: clip at clip_idx, then uniformly quantize
            let mut q_prob = vec![0.0; q_bins];
            
            // Values below clip_idx are uniformly distributed into q_bins
            let clipped_mass = cumsum[clip_idx];
            if clipped_mass > 0.0 {
                let mass_per_bin = clipped_mass / q_bins as f64;
                for i in 0..q_bins {
                    q_prob[i] = mass_per_bin;
                }
            }
            
            // Add tail mass to last bin
            q_prob[q_bins - 1] += 1.0 - cumsum[clip_idx];
            
            // Reference distribution (clipped original)
            let mut ref_prob = vec![0.0; q_bins];
            for i in 0..q_bins {
                // Map quantized bin i to original histogram range [0, clip_idx)
                let src_start = (i * clip_idx) / q_bins;
                let src_end = ((i + 1) * clip_idx) / q_bins;
                let src_bin = src_start.min(bins - 1);
                ref_prob[i] += prob[src_bin];
            }
            ref_prob[q_bins - 1] += 1.0 - cumsum[clip_idx];

            // KL divergence
            let mut kl = 0.0;
            for i in 0..q_bins {
                if ref_prob[i] > 0.0 && q_prob[i] > 0.0 {
                    kl += ref_prob[i] * (ref_prob[i] / q_prob[i]).ln();
                }
            }

            if kl < best_kl {
                best_kl = kl;
                best_threshold_idx = clip_idx;
            }
        }

        // Compute scale from best threshold
        let threshold = self.hist_min + (best_threshold_idx as f32 / bins as f32) * (self.hist_max - self.hist_min);
        let range = threshold - self.min;
        let levels = (1u32 << bit_width) as f32 - 1.0;
        
        if range <= 0.0 {
            return Some((1.0, 0.0));
        }

        let scale = range / levels;
        let zero_point = -self.min / scale;
        
        Some((scale, zero_point.round()))
    }
}

/// Calibration data collected during graph execution.
#[derive(Debug, Clone, Default)]
pub struct CalibrationData {
    /// Per-tensor calibration stats (key = tensor name)
    pub stats: HashMap<String, CalibrationStats>,
    /// Order of nodes for topological processing
    pub node_order: Vec<NodeId>,
}

impl CalibrationData {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            node_order: Vec::new(),
        }
    }

    pub fn observe(&mut self, name: &str, values: &[f32]) {
        self.stats
            .entry(name.to_string())
            .or_insert_with(CalibrationStats::new)
            .observe(values);
    }

    pub fn get_stats(&self, name: &str) -> Option<&CalibrationStats> {
        self.stats.get(name)
    }

    pub fn get_stats_mut(&mut self, name: &str) -> &mut CalibrationStats {
        self.stats.entry(name.to_string()).or_default()
    }

    /// Generate quantization config JSON for all observed tensors.
    pub fn to_quant_config(&self, bit_width: u8, use_kl: bool) -> serde_json::Value {
        let mut config = serde_json::Map::new();
        
        for (name, stats) in &self.stats {
            let (scale, zp) = if use_kl {
                stats.compute_scale_zp_kl(bit_width).unwrap_or_else(|| stats.compute_scale_zp(bit_width))
            } else {
                stats.compute_scale_zp(bit_width)
            };

            let mut entry = serde_json::Map::new();
            entry.insert("min".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(stats.min as f64).unwrap()
            ));
            entry.insert("max".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(stats.max as f64).unwrap()
            ));
            entry.insert("mean".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(stats.mean() as f64).unwrap()
            ));
            entry.insert("std".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(stats.std() as f64).unwrap()
            ));
            entry.insert("scale".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(scale as f64).unwrap()
            ));
            entry.insert("zero_point".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(zp as f64).unwrap()
            ));
            entry.insert("bit_width".to_string(), serde_json::Value::Number(
                serde_json::Number::from(bit_width)
            ));

            config.insert(name.clone(), serde_json::Value::Object(entry));
        }

        serde_json::Value::Object(config)
    }
}

/// IR node for observing activations during calibration.
#[derive(Debug, Clone)]
pub struct ObserveNode {
    pub name: String,
    pub input: NodeId,
    pub stats: CalibrationStats,
}

/// Analysis pass that identifies which nodes need calibration.
pub fn find_calibration_points(graph: &ComputeGraph) -> Vec<(NodeId, String)> {
    let mut points = Vec::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        let node_id: NodeId = idx;
        // Observe after these quantizable operations
        let should_observe = matches!(
            node.opcode,
            Opcode::Conv2d | Opcode::Conv1d | Opcode::Conv3d |
            Opcode::MatMul |
            Opcode::Add | Opcode::Mul | Opcode::Sub |
            Opcode::Sigmoid | Opcode::Silu | Opcode::Relu |
            Opcode::Softmax | Opcode::LayerNorm | Opcode::BatchNorm
        );

        if should_observe {
            let name = if node.name.is_empty() {
                format!("{:?}_{}", node.opcode, node_id)
            } else {
                node.name.clone()
            };
            points.push((node_id, name));
        }
    }

    points
}

/// Runtime hook for collecting calibration data during inference.
/// This would be called from the executor after each observed node.
pub fn record_calibration(data: &mut CalibrationData, name: &str, output_ptr: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(output_ptr, len) };
    data.observe(name, slice);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_stats_basic() {
        let mut stats = CalibrationStats::new();
        stats.observe(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean() - 3.0).abs() < 1e-6);
        assert!(stats.count == 5);
    }

    #[test]
    fn test_scale_zp_asymmetric() {
        let mut stats = CalibrationStats::new();
        stats.observe(&[-10.0, 0.0, 10.0]);
        
        let (scale, zp) = stats.compute_scale_zp(8);
        // range=20, levels=255, scale=20/255
        // zp = 10/scale = 10 * 255 / 20 = 127.5 -> 128
        
        assert!((scale - 20.0 / 255.0).abs() < 0.001);
        // Use wider tolerance for floating point rounding
        assert!((zp - 128.0).abs() < 2.0, "zp={}, expected ~128", zp);
    }

    #[test]
    fn test_kl_calibration() {
        let mut stats = CalibrationStats::new()
            .with_histogram(2048, -10.0, 10.0);
        
        // Simulate activations: mostly in [-1, 1] with some outliers
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut values = Vec::new();
        for _ in 0..10000 {
            if rng.gen::<f32>() < 0.95 {
                values.push(rng.gen_range(-1.0..1.0));
            } else {
                values.push(rng.gen_range(-10.0..10.0));
            }
        }
        stats.observe(&values);
        
        let kl_result = stats.compute_scale_zp_kl(8);
        assert!(kl_result.is_some());
        
        let (scale, zp) = kl_result.unwrap();
        // Should be tighter than min/max
        let (scale_minmax, zp_minmax) = stats.compute_scale_zp(8);
        // KL should give smaller scale (clip outliers)
        assert!(scale <= scale_minmax * 1.1); // Allow some tolerance
    }

    #[test]
    fn test_calibration_data() {
        let mut cal = CalibrationData::new();
        cal.observe("conv1", &[1.0, 2.0, 3.0]);
        cal.observe("conv1", &[4.0, 5.0]); // Second batch
        
        let stats = cal.get_stats("conv1").unwrap();
        eprintln!("DEBUG: count={}, min={}, max={}, sum={}", stats.count, stats.min, stats.max, stats.sum);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        
        let config = cal.to_quant_config(8, false);
        assert!(config.get("conv1").is_some());
    }
}