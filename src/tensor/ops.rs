// Tensor operation methods - arithmetic, matmul, activations, and operator overloads

use crate::autograd;

use crate::storage::{DType, Device, Storage};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;


use super::Tensor;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        Self::broadcast_shapes(&self.shape(), &other.shape())
            .expect("Tensor::add: shape broadcast failed");
        // Fast path: CPU contiguous same-shape add, skip dispatch overhead
        if self.device() == Device::Cpu
            && other.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && other.inner.dtype == DType::F32
            && self.is_contiguous()
            && other.is_contiguous()
            && self.inner.sizes == other.inner.sizes
        {
            let numel = self.inner.numel() as usize;
            let mut output = Tensor::zeros(self.shape(), DType::F32, Device::Cpu);
            {
                let inner = Arc::make_mut(&mut output.inner);
                let storage = Arc::make_mut(&mut inner.storage);
                let Storage::Cpu(cpu_storage) = storage else {
                    unreachable!("add fast path only runs when both tensors are on CPU")
                };
                let out_data = Arc::make_mut(&mut cpu_storage.data);
                let a_ptr = self.data_ptr_f32();
                let b_ptr = other.data_ptr_f32();
                let out_ptr = out_data.as_mut_ptr() as *mut f32;

                // Single-threaded AVX2 SIMD - faster than rayon for memory-bound ops
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let av = _mm256_loadu_ps(a_ptr.add(i));
                                let bv = _mm256_loadu_ps(b_ptr.add(i));
                                _mm256_storeu_ps(out_ptr.add(i), _mm256_add_ps(av, bv));
                                i += 8;
                            }
                            for j in i..numel {
                                *out_ptr.add(j) = *a_ptr.add(j) + *b_ptr.add(j);
                            }
                        }
                    } else {
                        for i in 0..numel {
                            // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                            unsafe {
                                *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                            }
                        }
                    }
                }
                #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
                {
                    for i in 0..numel {
                        // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                        unsafe {
                            *out_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
                        }
                    }
                }
            }
            // Attach autograd if needed
            if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
                let _edges = {
                    let mut _edges = autograd::make_edge(self);
                    _edges.extend(autograd::make_edge(other));
                    _edges
                };
                let backward = autograd::AddBackward::new();
                let mut meta = autograd::AutogradMeta::new_non_leaf(true);
                meta.grad_fn = Some(std::sync::Arc::new(backward));
                Arc::make_mut(&mut output.inner).autograd_meta =
                    Some(Arc::new(std::sync::Mutex::new(meta)));
            }
            return output;
        }

        // General path: AOT execution
        let output = Tensor::exec_aot(&[self, other], |g, ins| {
            vec![g.add(&ins[0], &ins[1])]
        })
        .expect("Tensor::add: AOT execution failed")
        .into_iter().next().unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = {
                let mut _edges = autograd::make_edge(self);
                _edges.extend(autograd::make_edge(other));
                _edges
            };
            let backward = Arc::new(autograd::AddBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    /// In-place addition for gradient accumulation
    /// This is used internally by the autograd engine to accumulate gradients
    pub fn add_(&mut self, other: &Tensor) -> &mut Self {
        // For GPU tensors or non-contiguous tensors, use dispatch-based addition
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).add(other);
            *self = result;
            return self;
        }

        // CPU path: direct memory manipulation (requires contiguous self)
        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;

        // If other is broadcast (e.g., expanded scalar), use general path
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).add(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        match dtype {
            DType::F32 => {
                // SIMD + parallel path for F32 gradient accumulation (hot path)
                // Using sequential chunks with SIMD to avoid data races from parallel in-place modification
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        // Process each chunk sequentially to avoid race conditions
                        for chunk in 0..num_chunks {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(self_ptr.add(i));
                                            let ov = _mm256_loadu_ps(other_ptr.add(i));
                                            let r = _mm256_add_ps(sv, ov);
                                            _mm256_storeu_ps(self_ptr.add(i), r);
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *self_ptr.add(j) += *other_ptr.add(j);
                                        }
                                        continue;
                                    }
                                }
                            }
                            // Scalar fallback for this chunk
                            for i in start..end {
                                // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                                unsafe {
                                    *self_ptr.add(i) += *other_ptr.add(i);
                                }
                            }
                        }
                        return self;
                    }
                }
                // Small tensor or non-parallel: SIMD inline or scalar
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                let r = _mm256_add_ps(sv, ov);
                                _mm256_storeu_ps(self_ptr.add(i), r);
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) += *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                // Scalar fallback
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) += *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val + other_val);
                    }
                }
            }
            DType::F16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val + other_val);
                    }
                }
            }
            _ => unimplemented!("add_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place multiplication
    pub fn mul_(&mut self, other: &Tensor) -> &mut Self {
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).mul(other);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).mul(other);
            *self = result;
            return self;
        }

        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        match dtype {
            DType::F32 => {
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        use rayon::prelude::*;
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        let self_usize = self_ptr as usize;
                        let other_usize = other_ptr as usize;
                        (0..num_chunks).into_par_iter().for_each(|chunk| {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);
                            let s_p = self_usize as *mut f32;
                            let o_p = other_usize as *const f32;
                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(s_p.add(i));
                                            let ov = _mm256_loadu_ps(o_p.add(i));
                                            _mm256_storeu_ps(s_p.add(i), _mm256_mul_ps(sv, ov));
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *s_p.add(j) *= *o_p.add(j);
                                        }
                                        return;
                                    }
                                }
                            }
                            for j in start..end {
                                // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                                unsafe {
                                    *s_p.add(j) *= *o_p.add(j);
                                }
                            }
                        });
                        return self;
                    }
                }
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                _mm256_storeu_ps(self_ptr.add(i), _mm256_mul_ps(sv, ov));
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) *= *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) *= *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val * other_val);
                    }
                }
            }
            DType::F16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val * other_val);
                    }
                }
            }
            _ => unimplemented!("mul_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place scalar multiplication: self *= scalar
    /// Avoids allocating a scalar tensor for optimizer hot paths.
    pub fn mul_scalar_(&mut self, scalar: f32) -> &mut Self {
        if self.inner.is_gpu() {
            let scalar_t = Tensor::from_scalar(scalar);
            let result = (self as &Tensor).mul(&scalar_t);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let self_ptr = self.data_ptr_f32_mut();
        for i in 0..numel {
            // SAFETY: The pointer offset stays within the bounds of the allocated storage.
            unsafe {
                *self_ptr.add(i) *= scalar;
            }
        }
        self
    }

    /// Non-in-place scalar multiplication without creating a scalar tensor.
    /// Avoids the 5-heap-alloc overhead of Tensor::from_scalar().
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.clone();
        result.mul_scalar_(scalar);
        result
    }

    /// In-place scalar addition: self += scalar
    pub fn add_scalar_(&mut self, scalar: f32) -> &mut Self {
        if self.inner.is_gpu() {
            let scalar_t = Tensor::from_scalar(scalar);
            let result = (self as &Tensor).add(&scalar_t);
            *self = result;
            return self;
        }

        let numel = self.inner.numel() as usize;
        let self_ptr = self.data_ptr_f32_mut();
        for i in 0..numel {
            // SAFETY: The pointer offset stays within the bounds of the allocated storage.
            unsafe {
                *self_ptr.add(i) += scalar;
            }
        }
        self
    }

    /// In-place subtraction: self -= other
    pub fn sub_(&mut self, other: &Tensor) -> &mut Self {
        // For GPU tensors or non-contiguous tensors, use dispatch-based subtraction
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).sub(other);
            *self = result;
            return self;
        }

        // CPU path: direct memory manipulation (requires contiguous self)
        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;

        // If other is broadcast (e.g., expanded scalar), use general path
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).sub(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        match dtype {
            DType::F32 => {
                // SIMD + parallel path for F32
                // Using sequential chunks with SIMD to avoid data races from parallel in-place modification
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        for chunk in 0..num_chunks {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(self_ptr.add(i));
                                            let ov = _mm256_loadu_ps(other_ptr.add(i));
                                            _mm256_storeu_ps(
                                                self_ptr.add(i),
                                                _mm256_sub_ps(sv, ov),
                                            );
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *self_ptr.add(j) -= *other_ptr.add(j);
                                        }
                                        continue;
                                    }
                                }
                            }
                            // Scalar fallback for this chunk
                            for i in start..end {
                                // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                                unsafe {
                                    *self_ptr.add(i) -= *other_ptr.add(i);
                                }
                            }
                        }
                        return self;
                    }
                }
                // Small tensor or non-parallel: SIMD inline or scalar
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                _mm256_storeu_ps(self_ptr.add(i), _mm256_sub_ps(sv, ov));
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) -= *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                // Scalar fallback
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) -= *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) -= *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) -= *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val - other_val);
                    }
                }
            }
            DType::F16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val - other_val);
                    }
                }
            }
            _ => unimplemented!("sub_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place division: self /= other
    pub fn div_(&mut self, other: &Tensor) -> &mut Self {
        // For GPU tensors or non-contiguous tensors, use dispatch-based division
        if self.inner.is_gpu() || other.inner.is_gpu() || !self.is_contiguous() {
            let result = (self as &Tensor).div(other);
            *self = result;
            return self;
        }

        // CPU path: direct memory manipulation (requires contiguous self)
        let dtype = self.inner.dtype;
        let numel = self.inner.numel() as usize;
        let other_numel = other.inner.numel() as usize;

        // If other is broadcast (e.g., expanded scalar), use general path
        if other_numel != numel || !other.is_contiguous() {
            let result = (self as &Tensor).div(other);
            *self = result;
            return self;
        }

        let self_ptr = self.data_ptr_f32_mut();
        let other_ptr = other.data_ptr_f32();

        match dtype {
            DType::F32 => {
                // SIMD + parallel path for F32
                // Using sequential chunks with SIMD to avoid data races from parallel in-place modification
                #[cfg(feature = "parallel")]
                {
                    if numel > 64 * 1024 {
                        const CHUNK: usize = 4096;
                        let num_chunks = numel.div_ceil(CHUNK);
                        for chunk in 0..num_chunks {
                            let start = chunk * CHUNK;
                            let end = (start + CHUNK).min(numel);

                            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                            {
                                if is_x86_feature_detected!("avx2") {
                                    // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                                    unsafe {
                                        let mut i = start;
                                        while i + 8 <= end {
                                            let sv = _mm256_loadu_ps(self_ptr.add(i));
                                            let ov = _mm256_loadu_ps(other_ptr.add(i));
                                            _mm256_storeu_ps(
                                                self_ptr.add(i),
                                                _mm256_div_ps(sv, ov),
                                            );
                                            i += 8;
                                        }
                                        for j in i..end {
                                            *self_ptr.add(j) /= *other_ptr.add(j);
                                        }
                                        continue;
                                    }
                                }
                            }
                            // Scalar fallback for this chunk
                            for i in start..end {
                                // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                                unsafe {
                                    *self_ptr.add(i) /= *other_ptr.add(i);
                                }
                            }
                        }
                        return self;
                    }
                }
                // Small tensor or non-parallel: SIMD inline or scalar
                #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") && numel >= 8 {
                        // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                        unsafe {
                            let mut i = 0;
                            while i + 8 <= numel {
                                let sv = _mm256_loadu_ps(self_ptr.add(i));
                                let ov = _mm256_loadu_ps(other_ptr.add(i));
                                _mm256_storeu_ps(self_ptr.add(i), _mm256_div_ps(sv, ov));
                                i += 8;
                            }
                            for j in i..numel {
                                *self_ptr.add(j) /= *other_ptr.add(j);
                            }
                            return self;
                        }
                    }
                }
                // Scalar fallback
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) /= *other_ptr.add(i);
                    }
                }
            }
            DType::F64 => {
                let self_ptr = self_ptr as *mut f64;
                let other_ptr = other_ptr as *const f64;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) /= *other_ptr.add(i);
                    }
                }
            }
            DType::I32 => {
                let self_ptr = self_ptr as *mut i32;
                let other_ptr = other_ptr as *const i32;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        *self_ptr.add(i) /= *other_ptr.add(i);
                    }
                }
            }
            DType::BF16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::bf16;
                let other_ptr = other.data_ptr() as *const half::bf16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::bf16::from_f32(self_val / other_val);
                    }
                }
            }
            DType::F16 => {
                let self_ptr = self.data_ptr_mut() as *mut half::f16;
                let other_ptr = other.data_ptr() as *const half::f16;
                for i in 0..numel {
                    // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                    unsafe {
                        let self_val = f32::from(*self_ptr.add(i));
                        let other_val = f32::from(*other_ptr.add(i));
                        *self_ptr.add(i) = half::f16::from_f32(self_val / other_val);
                    }
                }
            }
            _ => unimplemented!("div_ for dtype {:?}", dtype),
        }
        self
    }

    /// In-place addcmul: self += tensor1 * tensor2
    /// Fused operation to reduce allocations in backward passes
    pub fn addcmul_(&mut self, tensor1: &Tensor, tensor2: &Tensor) -> &mut Self {
        // Fast path: CPU contiguous same-shape tensors, skip dispatch overhead
        if self.device() == Device::Cpu
            && tensor1.device() == Device::Cpu
            && tensor2.device() == Device::Cpu
            && self.inner.dtype == DType::F32
            && tensor1.inner.dtype == DType::F32
            && tensor2.inner.dtype == DType::F32
            && self.is_contiguous()
            && tensor1.is_contiguous()
            && tensor2.is_contiguous()
            && self.inner.sizes == tensor1.inner.sizes
            && self.inner.sizes == tensor2.inner.sizes
        {
            let numel = self.inner.numel() as usize;
            let self_ptr = self.data_ptr_f32_mut();
            let t1_ptr = tensor1.data_ptr_f32();
            let t2_ptr = tensor2.data_ptr_f32();

            #[cfg(feature = "parallel")]
            {
                if numel > 64 * 1024 {
                    use rayon::prelude::*;
                    const CHUNK: usize = 4096;
                    let num_chunks = numel.div_ceil(CHUNK);
                    let self_usize = self_ptr as usize;
                    let t1_usize = t1_ptr as usize;
                    let t2_usize = t2_ptr as usize;
                    (0..num_chunks).into_par_iter().for_each(|chunk| {
                        let start = chunk * CHUNK;
                        let end = (start + CHUNK).min(numel);
                        let s_p = self_usize as *mut f32;
                        let t1_p = t1_usize as *const f32;
                        let t2_p = t2_usize as *const f32;
                        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
                        {
                            if is_x86_feature_detected!("avx2") {
                                // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                                unsafe {
                                    let mut i = start;
                                    while i + 8 <= end {
                                        let sv = _mm256_loadu_ps(s_p.add(i));
                                        let t1v = _mm256_loadu_ps(t1_p.add(i));
                                        let t2v = _mm256_loadu_ps(t2_p.add(i));
                                        let prod = _mm256_mul_ps(t1v, t2v);
                                        let r = _mm256_add_ps(sv, prod);
                                        _mm256_storeu_ps(s_p.add(i), r);
                                        i += 8;
                                    }
                                    for j in i..end {
                                        *s_p.add(j) += *t1_p.add(j) * *t2_p.add(j);
                                    }
                                    return;
                                }
                            }
                        }
                        // Scalar fallback
                        for i in start..end {
                            // SAFETY: All preconditions for this unsafe operation are verified by the caller. The invariants required by this unsafe block are satisfied.
                            unsafe {
                                *s_p.add(i) += *t1_p.add(i) * *t2_p.add(i);
                            }
                        }
                    });
                    return self;
                }
            }

            // Small tensor or non-parallel: SIMD inline
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("avx2") && numel >= 8 {
                    // SAFETY: The pointers are valid and properly aligned for AVX2 access. Loop bounds guarantee all accesses stay within allocated storage.
                    unsafe {
                        let mut i = 0;
                        while i + 8 <= numel {
                            let sv = _mm256_loadu_ps(self_ptr.add(i));
                            let t1v = _mm256_loadu_ps(t1_ptr.add(i));
                            let t2v = _mm256_loadu_ps(t2_ptr.add(i));
                            let prod = _mm256_mul_ps(t1v, t2v);
                            let r = _mm256_add_ps(sv, prod);
                            _mm256_storeu_ps(self_ptr.add(i), r);
                            i += 8;
                        }
                        for j in i..numel {
                            *self_ptr.add(j) += *t1_ptr.add(j) * *t2_ptr.add(j);
                        }
                        return self;
                    }
                }
            }

            // Scalar fallback
            for i in 0..numel {
                // SAFETY: The pointer offset stays within the bounds of the allocated storage.
                unsafe {
                    *self_ptr.add(i) += *t1_ptr.add(i) * *t2_ptr.add(i);
                }
            }
            self
        } else {
            // Fallback: use dispatched operations
            let product = tensor1.mul(tensor2);
            self.add_(&product);
            self
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.sub(&ins[0], &ins[1])])
            .expect("Tensor::sub: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = {
                let mut _edges = autograd::make_edge(self);
                _edges.extend(autograd::make_edge(other));
                _edges
            };
            let backward = Arc::new(autograd::SubBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.mul(&ins[0], &ins[1])])
            .expect("Tensor::mul: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = {
                let mut _edges = autograd::make_edge(self);
                _edges.extend(autograd::make_edge(other));
                _edges
            };
            let backward = Arc::new(autograd::MulBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.div(&ins[0], &ins[1])])
            .expect("Tensor::div: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = {
                let mut _edges = autograd::make_edge(self);
                _edges.extend(autograd::make_edge(other));
                _edges
            };
            let backward = Arc::new(autograd::DivBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let output = Tensor::exec_aot(&[self, other], |g, ins| vec![g.matmul(&ins[0], &ins[1])])
            .expect("Tensor::matmul: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && (self.requires_grad() || other.requires_grad()) {
            let _edges = {
                let mut _edges = autograd::make_edge(self);
                _edges.extend(autograd::make_edge(other));
                _edges
            };
            let backward = Arc::new(autograd::MatmulBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn neg(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.neg(&ins[0])])
            .expect("Tensor::neg: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::NegBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn relu(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.relu(&ins[0])])
            .expect("Tensor::relu: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::ReluBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn exp(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.exp(&ins[0])])
            .expect("Tensor::exp: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::ExpBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn ln(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.log(&ins[0])])
            .expect("Tensor::ln: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::LogBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.sigmoid(&ins[0])])
            .expect("Tensor::sigmoid: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SigmoidBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn tanh(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.tanh(&ins[0])])
            .expect("Tensor::tanh: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::TanhBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn silu(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.silu(&ins[0])])
            .expect("Tensor::silu: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SiLUBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn gelu(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.gelu(&ins[0])])
            .expect("Tensor::gelu: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::GeluBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn leaky_relu(&self, negative_slope: f32) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.leaky_relu(&ins[0], negative_slope)])
            .expect("Tensor::leaky_relu: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::LeakyReLUBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn softplus(&self, _beta: f32, _threshold: f32) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.softplus(&ins[0])])
            .expect("Tensor::softplus: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SoftplusBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn hardswish(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.hardswish(&ins[0])])
            .expect("Tensor::hardswish: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::HardswishBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn mish(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.mish(&ins[0])])
            .expect("Tensor::mish: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::MishBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn elu(&self, alpha: f32) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.elu(&ins[0], alpha)])
            .expect("Tensor::elu: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::EluBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn softmax(&self, dim: i32) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.softmax(&ins[0], dim as usize)])
            .expect("Tensor::softmax: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SoftmaxBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.sqrt(&ins[0])])
            .expect("Tensor::sqrt: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::SqrtBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn fused_linear_gelu(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        let out = self.matmul(weight);
        let out = if let Some(b) = bias {
            Tensor::add(&out, b)
        } else {
            out
        };
        out.gelu()
    }

    pub fn clamp(&self, min_val: f32, max_val: f32) -> Tensor {
        let result = Tensor::exec_aot(&[self], |g, ins| vec![g.clamp(&ins[0], min_val, max_val)])
            .expect("Tensor::clamp: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::ClampBackward::new());
            Self::attach_grad_fn(result, backward)
        } else {
            result
        }
    }

    pub fn pow(&self, exponent: f32) -> Tensor {
        let exp = Tensor::from_scalar(exponent);
        let output = Tensor::exec_aot(&[self, &exp], |g, ins| vec![g.pow(&ins[0], &ins[1])])
            .expect("Tensor::pow: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::PowBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn abs(&self) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.abs(&ins[0])])
            .expect("Tensor::abs: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::AbsBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn log_softmax(&self, _dim: i32) -> Tensor {
        let output = Tensor::exec_aot(&[self], |g, ins| vec![g.log_softmax(&ins[0])])
            .expect("Tensor::log_softmax: AOT execution failed")
            .into_iter()
            .next()
            .unwrap();
        if autograd::is_grad_enabled() && self.requires_grad() {
            let _edges = autograd::make_edge(self);
            let backward = Arc::new(autograd::LogSoftmaxBackward::new());
            Self::attach_grad_fn(output, backward)
        } else {
            output
        }
    }

    pub fn as_i64_slice(&self) -> Vec<i64> {
        let src = self.to_cpu();
        let src = if !src.is_contiguous() {
            src.contiguous()
        } else {
            src
        };
        let data = src.as_f32_slice();
        data.iter().map(|&v| v as i64).collect()
    }

    pub fn erf(&self) -> Tensor {
        let data = self.as_f32_slice();
        let result_data: Vec<f32> = data
            .iter()
            .map(|&v| {
                let sign = if v >= 0.0 { 1.0f32 } else { -1.0f32 };
                let x = v.abs();
                let t = 1.0 / (1.0 + 0.3275911 * x);
                let y = 1.0
                    - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72)
                        * t
                        + 0.254_829_6)
                        * t
                        * (-x * x).exp();
                sign * y
            })
            .collect();
        Tensor::from_vec(result_data, self.shape_ref().to_vec())
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Tensor {
        self.add(other)
    }
}

impl Add for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn add(self, other: Tensor) -> Tensor {
        (&self).add(&other)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Tensor {
        self.sub(other)
    }
}

impl Sub for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn sub(self, other: Tensor) -> Tensor {
        (&self).sub(&other)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul(other)
    }
}

impl Mul for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn mul(self, other: Tensor) -> Tensor {
        (&self).mul(&other)
    }
}

impl Div for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Tensor {
        self.div(other)
    }
}

impl Div for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow)]
    fn div(self, other: Tensor) -> Tensor {
        (&self).div(&other)
    }
}

impl Neg for Tensor {
    type Output = Tensor;
    #[allow(clippy::needless_borrow, unconditional_recursion)]
    fn neg(self) -> Tensor {
        (&self).neg()
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.neg()
    }
}
