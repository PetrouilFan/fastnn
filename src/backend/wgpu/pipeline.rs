use crate::build_pipeline;
use std::sync::atomic::{AtomicU64, Ordering};

// ─══════════════════════════════════════════════════════════════════════════
// Cache hit / miss counters
// ─══════════════════════════════════════════════════════════════════════════

static SHADER_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static SHADER_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Record a shader source cache hit (string reused from OnceLock).
pub(crate) fn record_shader_hit() {
    SHADER_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Record a shader source cache miss (first call, string built).
pub(crate) fn record_shader_miss() {
    SHADER_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
fn shader_cache_stats() -> (u64, u64) {
    (
        SHADER_CACHE_HITS.load(Ordering::Relaxed),
        SHADER_CACHE_MISSES.load(Ordering::Relaxed),
    )
}

build_pipeline!(
    ensure_compute_pipeline,
    "wgpu_backend_",
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
);

// 3-binding pipeline for simple ops (softmax, reduce, transpose) that only need:
//   0: input  (storage, read, f32)
//   1: output (storage, read_write, f32)
//   2: params (uniform)
build_pipeline!(
    ensure_simple_compute_pipeline,
    "wgpu_simple_",
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
);

// Same as `ensure_compute_pipeline` but for fused matmul+activation ops that need 5 bindings:
//   0: a (storage, read, f32)
//   1: b (storage, read, f32)
//   2: output (storage, read_write, f32)
//   3: params (uniform)
//   4: bias (storage, read, f32)
build_pipeline!(
    ensure_matmul_activation_pipeline,
    "wgpu_backend_",
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
);

// Same as `ensure_compute_pipeline` but for quantized ops that need 6 bindings:
//   0: packed weights (storage, read, u32)
//   1: activations   (storage, read, f32)
//   2: scales        (storage, read, f32)
//   3: zero_points   (storage, read, f32)
//   4: output        (storage, read_write, f32)
//   5: params        (uniform)
build_pipeline!(
    ensure_quantized_compute_pipeline,
    "wgpu_backend_quantized_",
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_source_caching() {
        let before = shader_cache_stats();

        // First call should build the shader string (miss recorded inside OnceLock)
        let s1 = crate::backend::wgpu::reduce::cached_reduce_shader();
        // Second call should return the same static reference (hit recorded, no rebuild)
        let s2 = crate::backend::wgpu::reduce::cached_reduce_shader();
        assert!(
            std::ptr::eq(s1, s2),
            "cached_reduce_shader should return the same &str reference on repeated calls"
        );

        // Same pattern for matmul
        let m1 = crate::backend::wgpu::matmul::cached_matmul_shader();
        let m2 = crate::backend::wgpu::matmul::cached_matmul_shader();
        assert!(
            std::ptr::eq(m1, m2),
            "cached_matmul_shader should return the same &str reference"
        );

        // And activation variant
        let a1 = crate::backend::wgpu::matmul::cached_matmul_activation_shader();
        let a2 = crate::backend::wgpu::matmul::cached_matmul_activation_shader();
        assert!(
            std::ptr::eq(a1, a2),
            "cached_matmul_activation_shader should return the same &str reference"
        );

        let after = shader_cache_stats();
        assert_eq!(after.1 - before.1, 3);
        assert_eq!(after.0 - before.0, 3);

        // All shaders should be non-empty
        assert!(!s1.is_empty());
        assert!(!m1.is_empty());
        assert!(!a1.is_empty());
    }
}
