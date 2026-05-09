use crate::error::{FastnnError, FastnnResult};
use crate::storage::DType;
use crate::tensor::Tensor;
use std::sync::Arc;

#[repr(C)]
#[allow(dead_code)]
pub struct DLDevice {
    pub device_type: u32,
    pub device_id: i32,
}

#[repr(C)]
#[allow(dead_code)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Copy, Clone)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[allow(dead_code)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

// Device types from DLPack spec
#[allow(dead_code)]
const DLDEVICE_CPU: u32 = 1;
#[allow(dead_code)]
const DLDEVICE_CUDA: u32 = 2;

// Data type codes from DLPack spec
#[allow(dead_code)]
const DLDTYPE_FLOAT: u8 = 2;
#[allow(dead_code)]
const DLDTYPE_UINT: u8 = 0;
#[allow(dead_code)]
const DLDTYPE_INT: u8 = 1;

/// Context stored in the DLPack managed tensor to keep the tensor's storage alive.
/// This prevents the data pointer from becoming dangling while the DLPack capsule is in use.
#[repr(C)]
struct DLPackContext {
    /// Strong reference to the tensor's storage - keeps memory alive
    _storage: Arc<crate::storage::Storage>,
    /// Shape array (owned by this context)
    shape: Box<[i64]>,
    /// Strides array (owned by this context)
    strides: Box<[i64]>,
}

/// Convert a fastnn DType to a DLPack DLDataType.
/// Returns an error for unsupported dtypes instead of silently falling back.
#[allow(dead_code)]
fn dtype_to_dlpack(dtype: DType) -> FastnnResult<DLDataType> {
    match dtype {
        DType::F32 => Ok(DLDataType {
            code: DLDTYPE_FLOAT,
            bits: 32,
            lanes: 1,
        }),
        DType::F64 => Ok(DLDataType {
            code: DLDTYPE_FLOAT,
            bits: 64,
            lanes: 1,
        }),
        DType::I32 => Ok(DLDataType {
            code: DLDTYPE_INT,
            bits: 32,
            lanes: 1,
        }),
        DType::I64 => Ok(DLDataType {
            code: DLDTYPE_INT,
            bits: 64,
            lanes: 1,
        }),
        _ => Err(FastnnError::Dtype(format!(
            "Unsupported dtype for DLPack export: {:?}",
            dtype
        ))),
    }
}

/// Convert a DLPack DLDataType to a fastnn DType.
#[allow(dead_code)]
fn dlpack_to_dtype(dl_dtype: DLDataType) -> FastnnResult<DType> {
    match (dl_dtype.code, dl_dtype.bits) {
        (DLDTYPE_FLOAT, 32) => Ok(DType::F32),
        (DLDTYPE_FLOAT, 64) => Ok(DType::F64),
        (DLDTYPE_INT, 32) => Ok(DType::I32),
        (DLDTYPE_INT, 64) => Ok(DType::I64),
        _ => Err(FastnnError::Serialization(format!(
            "Unsupported DLPack dtype: code={}, bits={}",
            dl_dtype.code, dl_dtype.bits
        ))),
    }
}

/// Compute contiguous row-major strides for a given shape.
fn compute_strides(shape: &[i64]) -> Vec<i64> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1i64; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a Tensor to a DLPack managed tensor (zero-copy for CPU tensors).
/// Returns a raw pointer that can be passed to other frameworks.
/// The tensor's storage is kept alive via Arc until the deleter is called.
#[allow(dead_code)]
pub fn to_dlpack(tensor: &Tensor) -> *mut DLManagedTensor {
    let shape = tensor.shape();
    let ndim = shape.len() as i32;

    // Get data pointer - this is zero-copy for CPU tensors
    let data_ptr = if tensor.inner.is_cpu() {
        tensor.data_ptr() as *mut std::ffi::c_void
    } else {
        std::ptr::null_mut()
    };

    // Allocate shape array
    let shape_box = shape.clone().into_boxed_slice();

    // Create strides using helper function
    let strides_box = compute_strides(&shape).into_boxed_slice();

    // Get dtype, returning null on error
    let dtype = match dtype_to_dlpack(tensor.dtype()) {
        Ok(d) => d,
        Err(_) => return std::ptr::null_mut(),
    };

    // Create context that holds a strong reference to the tensor's storage
    // This prevents the data pointer from becoming dangling
    let ctx = Box::new(DLPackContext {
        _storage: tensor.inner.storage.clone(),
        shape: shape_box,
        strides: strides_box,
    });
    let ctx_ptr = Box::into_raw(ctx);

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr,
            device: DLDevice {
                device_type: DLDEVICE_CPU,
                device_id: 0,
            },
            ndim,
            dtype,
            shape: unsafe { (*ctx_ptr).shape.as_mut_ptr() },
            strides: unsafe { (*ctx_ptr).strides.as_mut_ptr() },
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut std::ffi::c_void,
        deleter: Some(dlpack_deleter),
    });

    Box::into_raw(managed)
}

/// Deleter function called by DLPack consumers when done with the tensor.
/// Frees the context (including the Arc<Storage> reference) and the managed tensor.
extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    unsafe {
        let managed = Box::from_raw(managed);

        // Drop the context (this drops the Arc<Storage> reference)
        if !managed.manager_ctx.is_null() {
            let _ = Box::from_raw(managed.manager_ctx as *mut DLPackContext);
        }
        // The managed tensor itself is dropped here
    }
}

/// Create a Tensor from a DLPack managed tensor.
/// Takes ownership of the DLPack capsule.
/// Note: Currently copies data; zero-copy support is planned for future versions.
///
/// # Safety
/// The capsule must be valid and properly managed by a DLManagedTensor sentinel.
#[allow(dead_code)]
pub unsafe fn from_dlpack(capsule: *mut DLManagedTensor) -> FastnnResult<Tensor> {
    if capsule.is_null() {
        return Err(FastnnError::Serialization(
            "DLPack capsule is null".to_string(),
        ));
    }

    unsafe {
        let managed = &*capsule;
        let dl_tensor = &managed.dl_tensor;

        if dl_tensor.data.is_null() {
            return Err(FastnnError::Serialization(
                "DLPack tensor data is null".to_string(),
            ));
        }

        // Extract shape
        let ndim = dl_tensor.ndim as usize;
        let shape: Vec<i64> = if ndim > 0 {
            std::slice::from_raw_parts(dl_tensor.shape, ndim).to_vec()
        } else {
            vec![]
        };

        // Determine dtype
        let dtype = dlpack_to_dtype(dl_tensor.dtype)?;

        // F64 is not supported - return an error instead of silent conversion
        if dtype == DType::F64 {
            return Err(FastnnError::Dtype(
                "F64 tensors are not supported. Use F32 instead.".to_string(),
            ));
        }

        // Calculate total elements
        let numel: usize = shape.iter().map(|&x| x as usize).product();

        // Copy data from DLPack tensor to our tensor
        // Note: For true zero-copy, we'd need to share the memory, but that
        // requires careful lifetime management
        let data = match dtype {
            DType::F32 => {
                let src = std::slice::from_raw_parts(
                    (dl_tensor.data as *const f32).add(dl_tensor.byte_offset as usize),
                    numel,
                );
                src.to_vec()
            }
            _ => {
                return Err(FastnnError::Dtype(format!(
                    "Unsupported dtype for from_dlpack: {:?}",
                    dtype
                )))
            }
        };

        // Call the deleter if provided
        if let Some(deleter) = managed.deleter {
            deleter(capsule);
        }

        Ok(Tensor::from_vec(data, shape))
    }
}
