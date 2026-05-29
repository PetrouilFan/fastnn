use crate::build_pipeline;
use std::sync::atomic::{AtomicU64, Ordering};

// ─══════════════════════════════════════════════════════════════════════════
// Structured cache keys
// ─══════════════════════════════════════════════════════════════════════════

/// Classification of GPU operation kinds for cache keying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum OpKind {
    MatMul,
    MatMulActivation { activation: u32, has_bias: bool },
    Reduce,
    Softmax,
    Transpose,
    ElementWise { opcode: u32 },
    Conv2d,
    LayerNorm,
    RmsNorm,
    Embed,
    ArgMax,
    Pool,
    QuantizedMatMul { bit_width: usize },
}

/// Shape class used for cache keying — avoids exact-shape explosion.
/// Shapes are classified by rank and a size bucket to keep cache small.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ShapeClass {
    Scalar,
    Vector {
        len_bucket: u32,
    },
    Matrix {
        rows_bucket: u32,
        cols_bucket: u32,
    },
    Tensor3D {
        d0_bucket: u32,
        d1_bucket: u32,
        d2_bucket: u32,
    },
    Tensor4D {
        d0_bucket: u32,
        d1_bucket: u32,
        d2_bucket: u32,
        d3_bucket: u32,
    },
    Generic {
        rank: u8,
    },
}

impl ShapeClass {
    /// Classify a shape into a cache-friendly bucket.
    /// Uses power-of-two buckets to limit cache entries.
    pub fn from_shape(shape: &[usize]) -> Self {
        match shape.len() {
            0 => ShapeClass::Scalar,
            1 => ShapeClass::Vector {
                len_bucket: Self::bucket(shape[0]),
            },
            2 => ShapeClass::Matrix {
                rows_bucket: Self::bucket(shape[0]),
                cols_bucket: Self::bucket(shape[1]),
            },
            3 => ShapeClass::Tensor3D {
                d0_bucket: Self::bucket(shape[0]),
                d1_bucket: Self::bucket(shape[1]),
                d2_bucket: Self::bucket(shape[2]),
            },
            4 => ShapeClass::Tensor4D {
                d0_bucket: Self::bucket(shape[0]),
                d1_bucket: Self::bucket(shape[1]),
                d2_bucket: Self::bucket(shape[2]),
                d3_bucket: Self::bucket(shape[3]),
            },
            r => ShapeClass::Generic { rank: r as u8 },
        }
    }

    fn bucket(size: usize) -> u32 {
        if size == 0 {
            0
        } else {
            (size as f64).log2().ceil() as u32
        }
    }
}

/// Layout class for cache keying — encodes the bind group structure variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum LayoutClass {
    /// 2 storage + 1 uniform (simple ops: softmax, reduce, transpose)
    Simple3Binding,
    /// 2 storage + 1 storage + 1 uniform (matmul: a, b, out, params)
    MatMul4Binding,
    /// 2 storage + 1 storage + 1 uniform + 1 storage (fused matmul+activation+bias)
    MatMulActivation5Binding,
    /// 6-binding quantized layout
    Quantized6Binding,
}

/// Composite cache key for shader/pipeline lookup.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ShaderCacheKey {
    pub op: OpKind,
    pub dtype_key: u32,
    pub shape_class: ShapeClass,
    pub layout: LayoutClass,
    pub feature_flags: u32,
}

impl ShaderCacheKey {
    /// Produce a stable string key for use with `WgpuContext.pipelines`.
    pub fn pipeline_key(&self) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        format!("wgpu_shader_{:016x}", hasher.finish())
    }
}

// ─══════════════════════════════════════════════════════════════════════════
// Cache hit / miss counters
// ─══════════════════════════════════════════════════════════════════════════

static PIPELINE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static PIPELINE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static SHADER_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static SHADER_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Record a pipeline cache hit (pipeline already existed in HashMap).
pub(crate) fn record_pipeline_hit() {
    PIPELINE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Record a pipeline cache miss (new pipeline was created).
pub(crate) fn record_pipeline_miss() {
    PIPELINE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

/// Record a shader source cache hit (string reused from OnceLock).
pub(crate) fn record_shader_hit() {
    SHADER_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
}

/// Record a shader source cache miss (first call, string built).
pub(crate) fn record_shader_miss() {
    SHADER_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
}

/// Snapshot of cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub pipeline_hits: u64,
    pub pipeline_misses: u64,
    pub shader_hits: u64,
    pub shader_misses: u64,
}

/// Return current cache hit/miss counters (monotonic since process start).
pub fn shader_cache_stats() -> CacheStats {
    CacheStats {
        pipeline_hits: PIPELINE_CACHE_HITS.load(Ordering::Relaxed),
        pipeline_misses: PIPELINE_CACHE_MISSES.load(Ordering::Relaxed),
        shader_hits: SHADER_CACHE_HITS.load(Ordering::Relaxed),
        shader_misses: SHADER_CACHE_MISSES.load(Ordering::Relaxed),
    }
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

/// 3-binding pipeline for simple ops (softmax, reduce, transpose) that only need:
///   0: input  (storage, read, f32)
///   1: output (storage, read_write, f32)
///   2: params (uniform)
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

/// Same as `ensure_compute_pipeline` but for fused matmul+activation ops that need 5 bindings:
///   0: a (storage, read, f32)
///   1: b (storage, read, f32)
///   2: output (storage, read_write, f32)
///   3: params (uniform)
///   4: bias (storage, read, f32)
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

/// Same as `ensure_compute_pipeline` but for quantized ops that need 6 bindings:
///   0: packed weights (storage, read, u32)
///   1: activations   (storage, read, f32)
///   2: scales        (storage, read, f32)
///   3: zero_points   (storage, read, f32)
///   4: output        (storage, read_write, f32)
///   5: params        (uniform)
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
        assert_eq!(after.shader_misses - before.shader_misses, 3);
        assert_eq!(after.shader_hits - before.shader_hits, 3);

        // All shaders should be non-empty
        assert!(!s1.is_empty());
        assert!(!m1.is_empty());
        assert!(!a1.is_empty());
    }

    #[test]
    fn test_shape_class_bucketing() {
        assert_eq!(ShapeClass::from_shape(&[]), ShapeClass::Scalar);
        assert!(matches!(
            ShapeClass::from_shape(&[128]),
            ShapeClass::Vector { len_bucket: 7 }
        ));
        assert!(matches!(
            ShapeClass::from_shape(&[32, 64]),
            ShapeClass::Matrix { .. }
        ));
    }

    #[test]
    fn test_pipeline_key_deterministic() {
        let key1 = ShaderCacheKey {
            op: OpKind::MatMul,
            dtype_key: 0,
            shape_class: ShapeClass::Matrix {
                rows_bucket: 5,
                cols_bucket: 6,
            },
            layout: LayoutClass::MatMul4Binding,
            feature_flags: 0,
        };
        let key2 = key1.clone();
        assert_eq!(key1.pipeline_key(), key2.pipeline_key());
    }
}
