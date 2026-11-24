//! Hand-written FFI for the oneDNN 3.14 C API.
//!
//! # Why no bindgen
//!
//! oneDNN is an optional, static C++ library whose C API is small and stable
//! for the six primitives this engine uses (convolution, pooling, matmul,
//! reorder, memory and primitive attributes). A generated binding would add a
//! build-time dependency, a bindgen version and a header path to every build;
//! these declarations are the subset the engine calls, written once, with the
//! status of every call checked.
//!
//! The numeric constants below are the values in oneDNN 3.14.1's
//! `dnnl_types.h` (the enums are part of the C ABI, so their values are
//! stable). [`runtime_version`] reports the library actually linked, and the
//! engine refuses to run on a version whose ABI it does not know.
//!
//! # Ownership
//!
//! Every handle oneDNN returns is owned by the [`std::ops::Drop`] wrapper that
//! produced it ([`MemoryDesc`], [`Memory`], [`PrimitiveDesc`], [`Primitive`],
//! [`Stream`], [`PrimitiveAttr`], [`PostOps`]) and is destroyed exactly once,
//! in the reverse order of creation. A primitive borrows nothing from Rust:
//! the descriptors it was created from may be destroyed immediately after
//! [`PrimitiveDesc`] construction, which is why the engine keeps only the
//! primitive and its memories.
//!
//! # Status handling
//!
//! Every fallible entry point returns `dnnl_status_t` (`0` is success), and
//! every call goes through [`check`], which maps a non-success status to
//! [`Fn3Error::ProviderFailure`]. There is no `unwrap` on a status anywhere in
//! this module. Destructors are the one exception to "return the error": they
//! run inside `Drop`, where there is no caller to return to; they assert
//! success in debug builds and document why destruction of a valid handle
//! cannot fail (a non-success there would be a double free, which Rust's
//! ownership already makes impossible).

// The dependency exists for its build script: it builds and links
// `libdnnl.a`. Referencing the crate keeps Cargo from warning that the
// dependency is unused and guarantees the native library reaches the link line
// even if the linker would otherwise drop an un-referenced `--extern`.
use onednn_src as _;

use std::ffi::{c_char, c_int, c_uint, c_void};

use crate::inference::error::Fn3Error;

/// Provider name used in [`Fn3Error::ProviderFailure`].
pub(crate) const PROVIDER: &str = "oneDNN";

// ---------------------------------------------------------------------------
// Status codes (`dnnl_status_t`)
// ---------------------------------------------------------------------------

/// `dnnl_success`.
pub(crate) const STATUS_SUCCESS: i32 = 0;
/// `dnnl_out_of_memory`.
pub(crate) const STATUS_OUT_OF_MEMORY: i32 = 1;
/// `dnnl_invalid_arguments`.
pub(crate) const STATUS_INVALID_ARGUMENTS: i32 = 2;
/// `dnnl_unimplemented`.
pub(crate) const STATUS_UNIMPLEMENTED: i32 = 3;

// ---------------------------------------------------------------------------
// Enumerations the engine reads or passes
// ---------------------------------------------------------------------------

/// `dnnl_f32`.
pub(crate) const DATA_TYPE_F32: i32 = 3;
/// `dnnl_s32` (the int8 convolution accumulator).
#[cfg(test)]
pub(crate) const DATA_TYPE_S32: i32 = 4;
/// `dnnl_s8` (signed weights).
#[cfg(test)]
pub(crate) const DATA_TYPE_S8: i32 = 5;
/// `dnnl_u8` (unsigned activations).
#[cfg(test)]
pub(crate) const DATA_TYPE_U8: i32 = 6;

/// `dnnl_cpu_isa_avx2`.
pub(crate) const CPU_ISA_AVX2: i32 = 0x7;
/// `dnnl_cpu_isa_avx2_vnni`.
pub(crate) const CPU_ISA_AVX2_VNNI: i32 = 0xf;
/// `dnnl_cpu_isa_avx512_core_vnni`.
pub(crate) const CPU_ISA_AVX512_CORE_VNNI: i32 = 0x67;

/// `dnnl_format_tag_any`: let the primitive choose the physical layout.
pub(crate) const FORMAT_TAG_ANY: i32 = 1;
/// `dnnl_a`: plain 1-D.
pub(crate) const FORMAT_TAG_A: i32 = 2;
/// `dnnl_ab`: plain row-major 2-D.
pub(crate) const FORMAT_TAG_AB: i32 = 3;
/// `dnnl_abcd` (aliases `dnnl_nchw`, `dnnl_oihw`): plain row-major 4-D.
pub(crate) const FORMAT_TAG_ABCD: i32 = 5;
/// `dnnl_abcde` (aliases `dnnl_ncdhw`, `dnnl_goihw`): plain row-major 5-D.
pub(crate) const FORMAT_TAG_ABCDE: i32 = 6;
/// `dnnl_acdb` (alias `dnnl_nhwc`): 4-D channels-last. The physical order for
/// logical `[N, C, H, W]` is `N, H, W, C`, which is always dense, so a
/// channels-last tensor never needs padding and a row range of it is one
/// contiguous byte run.
pub(crate) const FORMAT_TAG_NHWC: i32 = 22;

/// `dnnl_cpu`.
pub(crate) const ENGINE_KIND_CPU: i32 = 1;
/// `dnnl_stream_default_flags`.
pub(crate) const STREAM_DEFAULT_FLAGS: c_uint = 1;

/// `dnnl_forward_inference`.
pub(crate) const PROP_KIND_FORWARD_INFERENCE: i32 = 96;

/// `dnnl_convolution_direct`.
///
/// `auto` (which may choose Winograd) was checked twice on the pinned graph,
/// four interleaved 50-forward runs each, one worker: auto 44.15 / 41.30 /
/// 44.05 / 41.34 ms against direct 44.23 / 43.19 / 41.25 / 43.84 ms. There is
/// no reproducible difference (the ordering flips between runs), so the engine
/// keeps `direct` and does not hand the accumulation order to a heuristic.
/// Winograd itself is unavailable here: oneDNN 3.14.1 registers CPU Winograd
/// only for AArch64/ACL and RISC-V/RVV, so on x64 its primitive descriptor
/// creation fails regardless of the ISA. Recorded in `docs/v3/backends.md`.
pub(crate) const ALG_CONVOLUTION_DIRECT: i32 = 1;
/// `dnnl_pooling_max`.
pub(crate) const ALG_POOLING_MAX: i32 = 511;
/// `dnnl_eltwise_relu`.
pub(crate) const ALG_ELTWISE_RELU: i32 = 0x20;
/// `dnnl_eltwise_tanh`.
pub(crate) const ALG_ELTWISE_TANH: i32 = 0x21;
/// `dnnl_eltwise_logistic`.
pub(crate) const ALG_ELTWISE_LOGISTIC: i32 = 0x29;
/// `dnnl_eltwise_swish` (`x / (1 + exp(-alpha * x))`).
pub(crate) const ALG_ELTWISE_SWISH: i32 = 0x2c;
/// `dnnl_eltwise_clip` (`min(max(x, alpha), beta)`).
pub(crate) const ALG_ELTWISE_CLIP: i32 = 0x2e;
/// `dnnl_eltwise_gelu_erf` (the exact `erf` GELU).
pub(crate) const ALG_ELTWISE_GELU_ERF: i32 = 0x31;
/// `dnnl_pooling_avg_include_padding`.
pub(crate) const ALG_POOLING_AVG_INCLUDE_PADDING: i32 = 767;
/// `dnnl_pooling_avg_exclude_padding`.
pub(crate) const ALG_POOLING_AVG_EXCLUDE_PADDING: i32 = 1023;

/// `dnnl_arg_src`.
pub(crate) const ARG_SRC: c_int = 1;
/// `dnnl_arg_dst`.
pub(crate) const ARG_DST: c_int = 17;
/// `dnnl_arg_weights`.
pub(crate) const ARG_WEIGHTS: c_int = 33;
/// `dnnl_arg_bias`.
pub(crate) const ARG_BIAS: c_int = 41;
/// `dnnl_arg_scratchpad`.
pub(crate) const ARG_SCRATCHPAD: c_int = 80;

/// `dnnl_query_weights_md`.
pub(crate) const QUERY_WEIGHTS_MD: i32 = 131;
/// `dnnl_query_scratchpad_md`.
pub(crate) const QUERY_SCRATCHPAD_MD: i32 = 136;
/// `dnnl_scratchpad_mode_user`.
pub(crate) const SCRATCHPAD_MODE_USER: i32 = 1;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

macro_rules! opaque {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[repr(C)]
        pub(crate) struct $name {
            _private: [u8; 0],
        }
    };
}

opaque!(EngineOpaque, "The opaque `dnnl_engine`.");
opaque!(StreamOpaque, "The opaque `dnnl_stream`.");
opaque!(MemoryDescOpaque, "The opaque `dnnl_memory_desc`.");
opaque!(MemoryOpaque, "The opaque `dnnl_memory`.");
opaque!(PrimitiveDescOpaque, "The opaque `dnnl_primitive_desc`.");
opaque!(PrimitiveOpaque, "The opaque `dnnl_primitive`.");
opaque!(PrimitiveAttrOpaque, "The opaque `dnnl_primitive_attr`.");
opaque!(PostOpsOpaque, "The opaque `dnnl_post_ops`.");

/// `dnnl_engine_t`.
pub(crate) type Engine = *mut EngineOpaque;
/// `dnnl_stream_t`.
pub(crate) type Stream = *mut StreamOpaque;
/// `dnnl_memory_desc_t` (`const_dnnl_memory_desc_t` is the same pointer type).
pub(crate) type MemoryDesc = *mut MemoryDescOpaque;
/// `dnnl_memory_t`.
pub(crate) type Memory = *mut MemoryOpaque;
/// `dnnl_primitive_desc_t`.
pub(crate) type PrimitiveDesc = *mut PrimitiveDescOpaque;
/// `dnnl_primitive_t`.
pub(crate) type Primitive = *mut PrimitiveOpaque;
/// `dnnl_primitive_attr_t`.
pub(crate) type PrimitiveAttr = *mut PrimitiveAttrOpaque;
/// `dnnl_post_ops_t`.
pub(crate) type PostOps = *mut PostOpsOpaque;

/// `dnnl_dim_t`, which is `int64_t`.
pub(crate) type Dim = i64;

/// `dnnl_exec_arg_t`: one `(argument, memory)` pair.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExecArg {
    /// `DNNL_ARG_*`.
    pub arg: c_int,
    /// The memory object for that argument.
    pub memory: Memory,
}

/// `dnnl_version_t`, as returned by [`dnnl_version`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VersionRaw {
    /// Major version.
    pub major: c_int,
    /// Minor version.
    pub minor: c_int,
    /// Patch version.
    pub patch: c_int,
    /// Git hash, or null when the build did not record one.
    pub hash: *const c_char,
    /// `DNNL_RUNTIME_*` the CPU engine was built with.
    pub cpu_runtime: c_uint,
    /// `DNNL_RUNTIME_*` the GPU engine was built with.
    pub gpu_runtime: c_uint,
}

extern "C" {
    // Service / engine / stream.
    fn dnnl_version() -> *const VersionRaw;
    fn dnnl_engine_create(engine: *mut Engine, kind: c_int, index: usize) -> c_int;
    fn dnnl_engine_destroy(engine: Engine) -> c_int;
    fn dnnl_stream_create(stream: *mut Stream, engine: Engine, flags: c_uint) -> c_int;
    fn dnnl_stream_wait(stream: Stream) -> c_int;
    fn dnnl_stream_destroy(stream: Stream) -> c_int;

    // Memory descriptors.
    fn dnnl_memory_desc_create_with_tag(
        memory_desc: *mut MemoryDesc,
        ndims: c_int,
        dims: *const Dim,
        data_type: i32,
        tag: i32,
    ) -> c_int;
    fn dnnl_memory_desc_create_submemory(
        memory_desc: *mut MemoryDesc,
        parent_memory_desc: MemoryDesc,
        dims: *const Dim,
        offsets: *const Dim,
    ) -> c_int;
    fn dnnl_memory_desc_clone(memory_desc: *mut MemoryDesc, existing: MemoryDesc) -> c_int;
    fn dnnl_memory_desc_destroy(memory_desc: MemoryDesc) -> c_int;
    fn dnnl_memory_desc_get_size(memory_desc: MemoryDesc) -> usize;
    fn dnnl_memory_desc_equal(lhs: MemoryDesc, rhs: MemoryDesc) -> c_int;

    // Memory.
    fn dnnl_memory_create(
        memory: *mut Memory,
        memory_desc: MemoryDesc,
        engine: Engine,
        handle: *mut c_void,
    ) -> c_int;
    fn dnnl_memory_destroy(memory: Memory) -> c_int;
    fn dnnl_memory_get_data_handle(memory: Memory, handle: *mut *mut c_void) -> c_int;
    fn dnnl_memory_set_data_handle(memory: Memory, handle: *mut c_void) -> c_int;

    // Primitive descriptors: the typed creators the engine uses.
    fn dnnl_convolution_forward_primitive_desc_create(
        primitive_desc: *mut PrimitiveDesc,
        engine: Engine,
        prop_kind: i32,
        alg_kind: i32,
        src_desc: MemoryDesc,
        weights_desc: MemoryDesc,
        bias_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        strides: *const Dim,
        dilates: *const Dim,
        padding_l: *const Dim,
        padding_r: *const Dim,
        attr: PrimitiveAttr,
    ) -> c_int;
    fn dnnl_pooling_forward_primitive_desc_create(
        primitive_desc: *mut PrimitiveDesc,
        engine: Engine,
        prop_kind: i32,
        alg_kind: i32,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        strides: *const Dim,
        kernel: *const Dim,
        dilation: *const Dim,
        padding_l: *const Dim,
        padding_r: *const Dim,
        attr: PrimitiveAttr,
    ) -> c_int;
    fn dnnl_matmul_primitive_desc_create(
        primitive_desc: *mut PrimitiveDesc,
        engine: Engine,
        src_desc: MemoryDesc,
        weights_desc: MemoryDesc,
        bias_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        attr: PrimitiveAttr,
    ) -> c_int;
    fn dnnl_reorder_primitive_desc_create(
        primitive_desc: *mut PrimitiveDesc,
        src_desc: MemoryDesc,
        src_engine: Engine,
        dst_desc: MemoryDesc,
        dst_engine: Engine,
        attr: PrimitiveAttr,
    ) -> c_int;
    fn dnnl_primitive_desc_query_md(
        primitive_desc: PrimitiveDesc,
        what: i32,
        index: c_int,
    ) -> MemoryDesc;
    fn dnnl_primitive_desc_destroy(primitive_desc: PrimitiveDesc) -> c_int;

    // Primitives.
    fn dnnl_primitive_create(primitive: *mut Primitive, primitive_desc: PrimitiveDesc) -> c_int;
    fn dnnl_primitive_destroy(primitive: Primitive) -> c_int;
    fn dnnl_primitive_execute(
        primitive: Primitive,
        stream: Stream,
        nargs: c_int,
        args: *const ExecArg,
    ) -> c_int;

    // Attributes and post-ops.
    fn dnnl_primitive_attr_create(attr: *mut PrimitiveAttr) -> c_int;
    fn dnnl_primitive_attr_destroy(attr: PrimitiveAttr) -> c_int;
    fn dnnl_primitive_attr_set_scratchpad_mode(attr: PrimitiveAttr, mode: i32) -> c_int;
    fn dnnl_primitive_attr_set_post_ops(attr: PrimitiveAttr, post_ops: PostOps) -> c_int;
    fn dnnl_post_ops_create(post_ops: *mut PostOps) -> c_int;
    fn dnnl_post_ops_destroy(post_ops: PostOps) -> c_int;
    fn dnnl_post_ops_append_eltwise(
        post_ops: PostOps,
        alg_kind: i32,
        alpha: f32,
        beta: f32,
    ) -> c_int;
    fn dnnl_post_ops_append_sum(
        post_ops: PostOps,
        scale: f32,
        zero_point: i32,
        data_type: i32,
    ) -> c_int;

    // CPU information. `dnnl_get_effective_cpu_isa` reports the ISA the
    // library will actually use, which is what decides whether an int8
    // primitive has a JIT kernel or falls back to the reference one.
    fn dnnl_get_effective_cpu_isa() -> i32;
}

/// The ISA oneDNN will execute on.
pub(crate) fn effective_cpu_isa() -> i32 {
    // SAFETY: no arguments, no state; oneDNN documents it as a pure query.
    unsafe { dnnl_get_effective_cpu_isa() }
}

/// A stable name for the ISA code, for a diagnostic.
pub(crate) fn cpu_isa_name(isa: i32) -> &'static str {
    match isa {
        0x1 => "sse41",
        0x3 => "avx",
        CPU_ISA_AVX2 => "avx2",
        CPU_ISA_AVX2_VNNI => "avx2_vnni",
        0x1f => "avx2_vnni_2",
        0x27 => "avx512_core",
        CPU_ISA_AVX512_CORE_VNNI => "avx512_core_vnni",
        0xe7 => "avx512_core_bf16",
        0xfef => "avx512_core_amx",
        _ => "an ISA this table does not name",
    }
}

/// Whether oneDNN has a JIT int8 convolution for this ISA.
///
/// The check is an explicit list rather than a numeric comparison: the codes
/// are a capability encoding, not an ordering (avx512_core is 0x27 but has no
/// VNNI, avx2_vnni is 0xf and has one).
pub(crate) fn int8_kernels_available(isa: i32) -> bool {
    matches!(
        isa,
        CPU_ISA_AVX2_VNNI | 0x1f | CPU_ISA_AVX512_CORE_VNNI | 0xe7 | 0xfef | 0x1fef | 0x22fff
    )
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

/// Require `status` to be `dnnl_success`.
///
/// The returned error carries the entry point and the library's code; a caller
/// that knows the segment and block appends that with [`context`].
pub(crate) fn check(status: i32, call: &'static str) -> Result<(), Fn3Error> {
    if status == STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Fn3Error::ProviderFailure {
            provider: PROVIDER,
            call,
            status,
            detail: status_name(status).to_string(),
        })
    }
}

/// Append what the engine was doing to a provider error.
///
/// Non-provider errors pass through unchanged, so a caller can wrap a whole
/// preparation step without hiding an unrelated failure.
pub(crate) fn context(error: Fn3Error, detail: impl FnOnce() -> String) -> Fn3Error {
    match error {
        Fn3Error::ProviderFailure {
            provider,
            call,
            status,
            detail: previous,
        } => Fn3Error::ProviderFailure {
            provider,
            call,
            status,
            detail: if previous.is_empty() {
                detail()
            } else {
                format!("{previous}: {}", detail())
            },
        },
        other => other,
    }
}

/// Destroy a handle from `Drop`, where no error can be returned.
///
/// `destroy` of a valid oneDNN handle is documented to succeed; a non-success
/// here can only mean the handle was already freed, which Rust's ownership
/// rules make impossible. The debug assertion turns a future ABI change into a
/// test failure instead of silent state loss.
fn destroy(status: i32, call: &'static str) {
    debug_assert_eq!(
        status, STATUS_SUCCESS,
        "{call} failed with status {status} while dropping a oneDNN handle"
    );
}

// ---------------------------------------------------------------------------
// RAII wrappers
// ---------------------------------------------------------------------------

/// An owned `dnnl_engine`.
#[derive(Debug)]
pub(crate) struct EngineHandle(Engine);

// SAFETY: a oneDNN engine is an immutable, internally synchronised handle to a
// CPU implementation registry. oneDNN documents engines as usable from any
// thread; the engine is never mutated through this wrapper, and it outlives
// every primitive and stream created from it because those are owned by the
// same `OneDnnEngine` and dropped first.
unsafe impl Send for EngineHandle {}

impl EngineHandle {
    /// Create the CPU engine with index zero.
    pub(crate) fn cpu() -> Result<EngineHandle, Fn3Error> {
        let mut raw: Engine = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer to a null handle; the engine kind
        // and index are constants from the header (`dnnl_cpu`, 0), and oneDNN
        // initialises the handle on success. On failure it leaves it null,
        // which is never dereferenced because we return the error.
        check(
            unsafe { dnnl_engine_create(&mut raw, ENGINE_KIND_CPU, 0) },
            "dnnl_engine_create",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_engine_create",
                status: STATUS_SUCCESS,
                detail: "the CPU engine handle is null after a success status".to_string(),
            });
        }
        Ok(EngineHandle(raw))
    }

    /// The raw handle, for the duration of a call.
    pub(crate) fn raw(&self) -> Engine {
        self.0
    }
}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a handle created by `dnnl_engine_create`, owned by
        // this wrapper and not used after `drop` returns.
        destroy(
            unsafe { dnnl_engine_destroy(self.0) },
            "dnnl_engine_destroy",
        );
    }
}

/// An owned `dnnl_stream` on one engine.
#[derive(Debug)]
pub(crate) struct StreamHandle(Stream);

// SAFETY: a CPU stream in the sequential runtime has no thread affinity and no
// mutable shared state beyond its engine; oneDNN permits a stream to be used
// from any thread. Each block owns its own stream, so two worker threads never
// share one.
unsafe impl Send for StreamHandle {}
unsafe impl Sync for StreamHandle {}

impl StreamHandle {
    /// Create a default stream on `engine`.
    pub(crate) fn new(engine: &EngineHandle) -> Result<StreamHandle, Fn3Error> {
        let mut raw: Stream = std::ptr::null_mut();
        // SAFETY: `raw` is an out-pointer; `engine.raw()` is a live engine that
        // outlives this stream by ownership order; the flags are the documented
        // default (`dnnl_stream_default_flags`).
        check(
            unsafe { dnnl_stream_create(&mut raw, engine.raw(), STREAM_DEFAULT_FLAGS) },
            "dnnl_stream_create",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_stream_create",
                status: STATUS_SUCCESS,
                detail: "the stream handle is null after a success status".to_string(),
            });
        }
        Ok(StreamHandle(raw))
    }

    /// The raw handle.
    pub(crate) fn raw(&self) -> Stream {
        self.0
    }

    /// Wait for every primitive submitted to this stream.
    pub(crate) fn wait(&self) -> Result<(), Fn3Error> {
        // SAFETY: `self.0` is a live stream owned by this wrapper.
        check(unsafe { dnnl_stream_wait(self.0) }, "dnnl_stream_wait")
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `dnnl_stream_create` and is owned by
        // this wrapper; no other stream object aliases it.
        destroy(
            unsafe { dnnl_stream_destroy(self.0) },
            "dnnl_stream_destroy",
        );
    }
}

/// An owned `dnnl_memory_desc`.
#[derive(Debug)]
pub(crate) struct Desc(MemoryDesc);

impl Desc {
    /// A plain descriptor with `tag`.
    pub(crate) fn with_tag(dims: &[Dim], tag: i32) -> Result<Desc, Fn3Error> {
        Desc::with_tag_dtype(dims, tag, DATA_TYPE_F32)
    }

    /// A plain descriptor with `tag` and an explicit element type.
    pub(crate) fn with_tag_dtype(dims: &[Dim], tag: i32, dtype: i32) -> Result<Desc, Fn3Error> {
        let mut raw: MemoryDesc = std::ptr::null_mut();
        // SAFETY: `dims` is a valid slice of `dims.len()` extents; oneDNN reads
        // exactly `ndims` elements for the duration of the call and copies them
        // into the descriptor it allocates. `raw` is a valid out-pointer.
        check(
            unsafe {
                dnnl_memory_desc_create_with_tag(
                    &mut raw,
                    dims.len() as c_int,
                    dims.as_ptr(),
                    dtype,
                    tag,
                )
            },
            "dnnl_memory_desc_create_with_tag",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_memory_desc_create_with_tag",
                status: STATUS_SUCCESS,
                detail: format!("the descriptor for dims {dims:?} is null after success"),
            });
        }
        Ok(Desc(raw))
    }

    /// A sub-region of a parent descriptor.
    ///
    /// The child keeps the parent's strides, so `offsets` shifts its origin
    /// within the parent's allocation; the memory handle passed to
    /// [`MemoryHandle::new`] is the parent's base, not the region's.
    pub(crate) fn submemory(
        parent: &Desc,
        dims: &[Dim],
        offsets: &[Dim],
    ) -> Result<Desc, Fn3Error> {
        if dims.len() != offsets.len() {
            return Err(Fn3Error::InvalidShape {
                context: "oneDNN submemory descriptor",
                detail: format!(
                    "{} dims and {} offsets; a submemory needs one offset per dimension",
                    dims.len(),
                    offsets.len()
                ),
            });
        }
        let mut raw: MemoryDesc = std::ptr::null_mut();
        // SAFETY: both slices are valid for the call, `parent` is a live
        // descriptor outliving the child (the engine keeps the parent until
        // every child is dropped), and `raw` is a valid out-pointer.
        check(
            unsafe {
                dnnl_memory_desc_create_submemory(
                    &mut raw,
                    parent.raw(),
                    dims.as_ptr(),
                    offsets.as_ptr(),
                )
            },
            "dnnl_memory_desc_create_submemory",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_memory_desc_create_submemory",
                status: STATUS_SUCCESS,
                detail: "the submemory descriptor is null after success".to_string(),
            });
        }
        Ok(Desc(raw))
    }

    /// The raw handle.
    pub(crate) fn raw(&self) -> MemoryDesc {
        self.0
    }

    /// Bytes oneDNN needs for a memory with this descriptor.
    ///
    /// Does not include `offset0` (the library's own rule); a submemory's
    /// storage is part of its parent.
    pub(crate) fn size(&self) -> usize {
        // SAFETY: `self.0` is a live descriptor owned by this wrapper.
        unsafe { dnnl_memory_desc_get_size(self.0) }
    }

    /// An owned copy of this descriptor.
    ///
    /// Used when a prepared block must remember the layout oneDNN chose after
    /// the primitive descriptor that owns the original is dropped.
    pub(crate) fn duplicate(&self) -> Result<Desc, Fn3Error> {
        let mut owned: MemoryDesc = std::ptr::null_mut();
        // SAFETY: `self.0` is a live descriptor owned by this wrapper; the
        // clone API reads it only for the duration of the call and writes a new
        // owned descriptor to `owned`.
        check(
            unsafe { dnnl_memory_desc_clone(&mut owned, self.0) },
            "dnnl_memory_desc_clone",
        )?;
        if owned.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_memory_desc_clone",
                status: STATUS_SUCCESS,
                detail: "the cloned descriptor is null after success".to_string(),
            });
        }
        Ok(Desc(owned))
    }

    /// Whether two descriptors describe the same logical and physical layout.
    pub(crate) fn equal(&self, other: &Desc) -> bool {
        // SAFETY: both handles are live and owned by their wrappers.
        unsafe { dnnl_memory_desc_equal(self.0, other.0) != 0 }
    }
}

impl Drop for Desc {
    fn drop(&mut self) {
        // SAFETY: created by `dnnl_memory_desc_create_with_tag` /
        // `_create_submemory`, owned here, destroyed once. A child descriptor
        // does not own its parent (oneDNN copies the parent's layout), and the
        // engine drops children before parents anyway.
        destroy(
            unsafe { dnnl_memory_desc_destroy(self.0) },
            "dnnl_memory_desc_destroy",
        );
    }
}

/// An owned `dnnl_memory` over a host buffer.
#[derive(Debug)]
pub(crate) struct MemoryHandle(Memory);

// SAFETY: a oneDNN memory object is a descriptor plus a borrowed host pointer.
// It performs no synchronisation of its own; the engine creates exactly one
// memory per block argument and only reads the handle during that block's
// execution, so moving it between threads is sound. The host buffers it
// borrows (the arena, prepacked weights, scratchpads) outlive it by ownership
// order in `OneDnnEngine`.
unsafe impl Send for MemoryHandle {}
unsafe impl Sync for MemoryHandle {}

impl MemoryHandle {
    /// Create a memory object with **no** underlying buffer.
    ///
    /// `DNNL_MEMORY_NONE` is the null handle: oneDNN allocates the memory
    /// object but no storage. The engine installs the real pointer with
    /// [`MemoryHandle::set_data_handle`] immediately before each execute, so a
    /// block never caches a pointer into the arena (whose `&mut` provenance
    /// would otherwise have to outlive several reborrows).
    pub(crate) fn new(desc: &Desc, engine: &EngineHandle) -> Result<MemoryHandle, Fn3Error> {
        let mut raw: Memory = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; `desc` and `engine` are live
        // handles owned by the caller; a null handle is `DNNL_MEMORY_NONE`,
        // which oneDNN documents as "create the memory object without storage".
        check(
            unsafe { dnnl_memory_create(&mut raw, desc.raw(), engine.raw(), std::ptr::null_mut()) },
            "dnnl_memory_create",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_memory_create",
                status: STATUS_SUCCESS,
                detail: "the memory handle is null after success".to_string(),
            });
        }
        Ok(MemoryHandle(raw))
    }

    /// Install the buffer this memory reads or writes.
    ///
    /// # Safety
    ///
    /// `handle` must be non-null, aligned for the descriptor's format (64 bytes
    /// for every layout oneDNN may choose), and must address at least
    /// `desc.size()` bytes plus the submemory offset for as long as the next
    /// `dnnl_primitive_execute` runs. For a submemory descriptor the handle is
    /// the **parent's** base, because oneDNN applies `offset0` itself.
    pub(crate) unsafe fn set_data_handle(&self, handle: *mut u8) -> Result<(), Fn3Error> {
        // SAFETY: the caller's promise above; `self.0` is a live memory object
        // that oneDNN will read the handle from.
        check(
            unsafe { dnnl_memory_set_data_handle(self.0, handle.cast()) },
            "dnnl_memory_set_data_handle",
        )
    }

    /// The raw handle.
    pub(crate) fn raw(&self) -> Memory {
        self.0
    }

    /// The data handle oneDNN currently holds.
    ///
    /// Used by the tests to prove a block starts with no buffer and that one is
    /// installed for the call, i.e. that no arena pointer is cached between
    /// forwards.
    pub(crate) fn data_handle(&self) -> Result<*mut c_void, Fn3Error> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: `dnnl_memory_get_data_handle` writes the pointer into
        // `handle`; `self.0` is a live memory object.
        check(
            unsafe { dnnl_memory_get_data_handle(self.0, &mut handle) },
            "dnnl_memory_get_data_handle",
        )?;
        Ok(handle)
    }
}

impl Drop for MemoryHandle {
    fn drop(&mut self) {
        // SAFETY: created by `dnnl_memory_create` with a borrowed handle (oneDNN
        // does not own or free it), owned here, destroyed once.
        destroy(
            unsafe { dnnl_memory_destroy(self.0) },
            "dnnl_memory_destroy",
        );
    }
}

/// An owned `dnnl_primitive_desc`.
#[derive(Debug)]
pub(crate) struct PrimitiveDescHandle(PrimitiveDesc);

impl PrimitiveDescHandle {
    fn from_raw(raw: PrimitiveDesc, call: &'static str) -> Result<Self, Fn3Error> {
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call,
                status: STATUS_SUCCESS,
                detail: "the primitive descriptor is null after success".to_string(),
            });
        }
        Ok(PrimitiveDescHandle(raw))
    }

    /// Convolution forward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn convolution(
        engine: &EngineHandle,
        alg: i32,
        src: &Desc,
        weights: &Desc,
        bias: Option<&Desc>,
        dst: &Desc,
        strides: &[Dim],
        dilates: &[Dim],
        padding_l: &[Dim],
        padding_r: &[Dim],
        attr: &PrimitiveAttrHandle,
    ) -> Result<PrimitiveDescHandle, Fn3Error> {
        let bias_ptr = bias.map_or(std::ptr::null_mut(), Desc::raw);
        let mut raw: PrimitiveDesc = std::ptr::null_mut();
        // SAFETY: every slice has one entry per spatial axis (checked by the
        // caller against the rank), every descriptor is a live sibling
        // descriptor, `engine` outlives the pd, and `raw` is an out-pointer.
        // oneDNN copies the descriptors and arrays into the pd.
        check(
            unsafe {
                dnnl_convolution_forward_primitive_desc_create(
                    &mut raw,
                    engine.raw(),
                    PROP_KIND_FORWARD_INFERENCE,
                    alg,
                    src.raw(),
                    weights.raw(),
                    bias_ptr,
                    dst.raw(),
                    strides.as_ptr(),
                    dilates.as_ptr(),
                    padding_l.as_ptr(),
                    padding_r.as_ptr(),
                    attr.raw(),
                )
            },
            "dnnl_convolution_forward_primitive_desc_create",
        )?;
        Self::from_raw(raw, "dnnl_convolution_forward_primitive_desc_create")
    }

    /// Pooling forward.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pooling(
        engine: &EngineHandle,
        alg: i32,
        src: &Desc,
        dst: &Desc,
        strides: &[Dim],
        kernel: &[Dim],
        dilates: &[Dim],
        padding_l: &[Dim],
        padding_r: &[Dim],
        attr: &PrimitiveAttrHandle,
    ) -> Result<PrimitiveDescHandle, Fn3Error> {
        let mut raw: PrimitiveDesc = std::ptr::null_mut();
        // SAFETY: as for convolution: all slices are per-spatial-axis, all
        // descriptors and the engine are live, and oneDNN copies the inputs.
        check(
            unsafe {
                dnnl_pooling_forward_primitive_desc_create(
                    &mut raw,
                    engine.raw(),
                    PROP_KIND_FORWARD_INFERENCE,
                    alg,
                    src.raw(),
                    dst.raw(),
                    strides.as_ptr(),
                    kernel.as_ptr(),
                    dilates.as_ptr(),
                    padding_l.as_ptr(),
                    padding_r.as_ptr(),
                    attr.raw(),
                )
            },
            "dnnl_pooling_forward_primitive_desc_create",
        )?;
        Self::from_raw(raw, "dnnl_pooling_forward_primitive_desc_create")
    }

    /// Matmul (`dst = src * weights`).
    pub(crate) fn matmul(
        engine: &EngineHandle,
        src: &Desc,
        weights: &Desc,
        bias: Option<&Desc>,
        dst: &Desc,
        attr: &PrimitiveAttrHandle,
    ) -> Result<PrimitiveDescHandle, Fn3Error> {
        let bias_ptr = bias.map_or(std::ptr::null_mut(), Desc::raw);
        let mut raw: PrimitiveDesc = std::ptr::null_mut();
        // SAFETY: all descriptors and the engine are live; the out-pointer is
        // valid; oneDNN copies the descriptors.
        check(
            unsafe {
                dnnl_matmul_primitive_desc_create(
                    &mut raw,
                    engine.raw(),
                    src.raw(),
                    weights.raw(),
                    bias_ptr,
                    dst.raw(),
                    attr.raw(),
                )
            },
            "dnnl_matmul_primitive_desc_create",
        )?;
        Self::from_raw(raw, "dnnl_matmul_primitive_desc_create")
    }

    /// Reorder from `src` to `dst`, used to prepack a weight once.
    pub(crate) fn reorder(
        engine: &EngineHandle,
        src: &Desc,
        dst: &Desc,
        attr: &PrimitiveAttrHandle,
    ) -> Result<PrimitiveDescHandle, Fn3Error> {
        let mut raw: PrimitiveDesc = std::ptr::null_mut();
        // SAFETY: both descriptors and the engine are live; a reorder on one
        // engine has the same engine on both sides. oneDNN copies the
        // descriptors.
        check(
            unsafe {
                dnnl_reorder_primitive_desc_create(
                    &mut raw,
                    src.raw(),
                    engine.raw(),
                    dst.raw(),
                    engine.raw(),
                    attr.raw(),
                )
            },
            "dnnl_reorder_primitive_desc_create",
        )?;
        Self::from_raw(raw, "dnnl_reorder_primitive_desc_create")
    }

    /// Query one memory descriptor the primitive expects and take an owned copy.
    ///
    /// `dnnl_primitive_desc_query_md` lends a descriptor owned by the primitive
    /// descriptor; the clone is independent, so the pd may be dropped before the
    /// copy is used. The copy preserves the *physical* layout (strides, inner
    /// blocks and offset), which a plain tag cannot express, so this is what
    /// makes "prepack into exactly what oneDNN asked for" possible.
    pub(crate) fn query_md(&self, what: i32, index: c_int) -> Result<Desc, Fn3Error> {
        // SAFETY: `self.0` is a live primitive descriptor. The returned pointer
        // is borrowed from it and is only read by `dnnl_memory_desc_clone`
        // before this call returns.
        let borrowed = unsafe { dnnl_primitive_desc_query_md(self.0, what, index) };
        if borrowed.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_primitive_desc_query_md",
                status: STATUS_SUCCESS,
                detail: format!("query {what} (index {index}) returned a null descriptor"),
            });
        }
        let mut owned: MemoryDesc = std::ptr::null_mut();
        // SAFETY: `borrowed` is live for the duration of the call and `owned` is
        // a valid out-pointer; oneDNN allocates an independent descriptor.
        check(
            unsafe { dnnl_memory_desc_clone(&mut owned, borrowed) },
            "dnnl_memory_desc_clone",
        )?;
        if owned.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_memory_desc_clone",
                status: STATUS_SUCCESS,
                detail: "the cloned descriptor is null after success".to_string(),
            });
        }
        Ok(Desc(owned))
    }

    /// The scratchpad descriptor, or `None` when the primitive needs none.
    ///
    /// oneDNN describes "no scratchpad" as a zero-size descriptor; the engine
    /// then passes no `DNNL_ARG_SCRATCHPAD` argument.
    pub(crate) fn scratchpad(&self) -> Result<Option<Desc>, Fn3Error> {
        let desc = self.query_md(QUERY_SCRATCHPAD_MD, 0)?;
        if desc.size() == 0 {
            Ok(None)
        } else {
            Ok(Some(desc))
        }
    }

    /// The raw handle.
    pub(crate) fn raw(&self) -> PrimitiveDesc {
        self.0
    }
}

impl Drop for PrimitiveDescHandle {
    fn drop(&mut self) {
        // SAFETY: owned, created by one of the typed creators, destroyed once.
        destroy(
            unsafe { dnnl_primitive_desc_destroy(self.0) },
            "dnnl_primitive_desc_destroy",
        );
    }
}

/// An owned `dnnl_primitive`.
#[derive(Debug)]
pub(crate) struct PrimitiveHandle(Primitive);

// SAFETY: a oneDNN primitive is an immutable compiled kernel plus its resource
// map. The sequential runtime means `execute` runs entirely on the calling
// thread; each block owns one primitive and executes it at most once per
// forward, so no two threads execute the same primitive at the same time.
unsafe impl Send for PrimitiveHandle {}
unsafe impl Sync for PrimitiveHandle {}

impl PrimitiveHandle {
    /// Compile `pd` into a primitive.
    pub(crate) fn new(pd: &PrimitiveDescHandle) -> Result<PrimitiveHandle, Fn3Error> {
        let mut raw: Primitive = std::ptr::null_mut();
        // SAFETY: `pd` is live and outlives this call; oneDNN copies what it
        // needs. `raw` is a valid out-pointer.
        check(
            unsafe { dnnl_primitive_create(&mut raw, pd.raw()) },
            "dnnl_primitive_create",
        )?;
        if raw.is_null() {
            return Err(Fn3Error::ProviderFailure {
                provider: PROVIDER,
                call: "dnnl_primitive_create",
                status: STATUS_SUCCESS,
                detail: "the primitive handle is null after success".to_string(),
            });
        }
        Ok(PrimitiveHandle(raw))
    }

    /// Execute with `args` on `stream`.
    ///
    /// The caller keeps every memory in `args` alive for the call. In the
    /// sequential runtime execution completes before the call returns.
    pub(crate) fn execute(&self, stream: &StreamHandle, args: &[ExecArg]) -> Result<(), Fn3Error> {
        // SAFETY: `self.0` and `stream` are live; `args` is a valid slice of
        // (int, memory) pairs whose memories outlive the call. oneDNN borrows
        // the host buffers behind those memories only for the duration of this
        // call; the arena and the prepacked weights outlive the stream call.
        check(
            unsafe {
                dnnl_primitive_execute(self.0, stream.raw(), args.len() as c_int, args.as_ptr())
            },
            "dnnl_primitive_execute",
        )
    }
}

impl Drop for PrimitiveHandle {
    fn drop(&mut self) {
        // SAFETY: owned, created by `dnnl_primitive_create`, destroyed once.
        destroy(
            unsafe { dnnl_primitive_destroy(self.0) },
            "dnnl_primitive_destroy",
        );
    }
}

/// An owned `dnnl_primitive_attr` with user-managed scratchpad.
#[derive(Debug)]
pub(crate) struct PrimitiveAttrHandle(PrimitiveAttr);

impl PrimitiveAttrHandle {
    /// Empty attributes, i.e. oneDNN's library-managed scratchpad.
    ///
    /// Used only for the session-open reorder, which runs synchronously on one
    /// thread and never needs a user scratchpad.
    pub(crate) fn library_default() -> Result<PrimitiveAttrHandle, Fn3Error> {
        let mut raw: PrimitiveAttr = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer.
        check(
            unsafe { dnnl_primitive_attr_create(&mut raw) },
            "dnnl_primitive_attr_create",
        )?;
        Ok(PrimitiveAttrHandle(raw))
    }

    /// Create attributes with `dnnl_scratchpad_mode_user`.
    ///
    /// The library scratchpad is a single per-primitive buffer; with
    /// `DNNL_ENABLE_CONCURRENT_EXEC=OFF` (oneDNN's default) that buffer is
    /// thread-local to the thread that created the primitive. A block executes
    /// on whichever session worker the pool assigns, so the engine owns the
    /// scratchpad instead: one per block, passed as `DNNL_ARG_SCRATCHPAD`.
    pub(crate) fn user_scratchpad() -> Result<PrimitiveAttrHandle, Fn3Error> {
        let mut raw: PrimitiveAttr = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer.
        check(
            unsafe { dnnl_primitive_attr_create(&mut raw) },
            "dnnl_primitive_attr_create",
        )?;
        let attr = PrimitiveAttrHandle(raw);
        // SAFETY: `raw` is live and owned by `attr`.
        check(
            unsafe { dnnl_primitive_attr_set_scratchpad_mode(raw, SCRATCHPAD_MODE_USER) },
            "dnnl_primitive_attr_set_scratchpad_mode",
        )?;
        Ok(attr)
    }

    /// Replace the post-op sequence.
    ///
    /// Used for the fused convolution epilogue: a `sum` followed by at most one
    /// eltwise, which is the order oneDNN's direct convolution accepts (`sum`
    /// must be the first post-op).
    pub(crate) fn set_post_ops(&self, post_ops: &PostOpsHandle) -> Result<(), Fn3Error> {
        // SAFETY: both handles are live; oneDNN copies the sequence.
        check(
            unsafe { dnnl_primitive_attr_set_post_ops(self.0, post_ops.0) },
            "dnnl_primitive_attr_set_post_ops",
        )
    }

    /// The raw handle.
    pub(crate) fn raw(&self) -> PrimitiveAttr {
        self.0
    }
}

impl Drop for PrimitiveAttrHandle {
    fn drop(&mut self) {
        // SAFETY: created by `dnnl_primitive_attr_create`, owned here.
        destroy(
            unsafe { dnnl_primitive_attr_destroy(self.0) },
            "dnnl_primitive_attr_destroy",
        );
    }
}

/// An owned `dnnl_post_ops` sequence.
///
/// Empty for an unfused segment; a fused convolution appends a `sum` (which
/// reads the destination buffer back) and at most one eltwise.
#[derive(Debug)]
pub(crate) struct PostOpsHandle(PostOps);

impl PostOpsHandle {
    /// An empty sequence.
    pub(crate) fn new() -> Result<PostOpsHandle, Fn3Error> {
        let mut raw: PostOps = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer.
        check(
            unsafe { dnnl_post_ops_create(&mut raw) },
            "dnnl_post_ops_create",
        )?;
        Ok(PostOpsHandle(raw))
    }

    /// Append `dst = scale * dst + accumulator`, i.e. add the destination's
    /// previous contents (oneDNN reads the destination back; the engine
    /// pre-fills it with the residual operand).
    pub(crate) fn append_sum(
        &mut self,
        scale: f32,
        zero_point: i32,
        data_type: i32,
    ) -> Result<(), Fn3Error> {
        // SAFETY: `self.0` is a live sequence owned by this wrapper.
        check(
            unsafe { dnnl_post_ops_append_sum(self.0, scale, zero_point, data_type) },
            "dnnl_post_ops_append_sum",
        )
    }

    /// Append an elementwise op (`relu`, `clip`, ...).
    pub(crate) fn append_eltwise(
        &mut self,
        alg: i32,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Fn3Error> {
        // SAFETY: `self.0` is a live sequence owned by this wrapper.
        check(
            unsafe { dnnl_post_ops_append_eltwise(self.0, alg, alpha, beta) },
            "dnnl_post_ops_append_eltwise",
        )
    }
}

impl Drop for PostOpsHandle {
    fn drop(&mut self) {
        // SAFETY: created by `dnnl_post_ops_create`, owned here.
        destroy(
            unsafe { dnnl_post_ops_destroy(self.0) },
            "dnnl_post_ops_destroy",
        );
    }
}

// ---------------------------------------------------------------------------
// Library version
// ---------------------------------------------------------------------------

/// The oneDNN build this process linked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Version {
    /// Major version.
    pub major: i32,
    /// Minor version.
    pub minor: i32,
    /// Patch version.
    pub patch: i32,
    /// Git hash, empty when the build did not record one.
    pub hash: String,
    /// `DNNL_RUNTIME_*` the CPU engine was built with (`1` is sequential).
    pub cpu_runtime: u32,
}

/// `DNNL_RUNTIME_SEQ`.
pub(crate) const RUNTIME_SEQ: u32 = 1;

impl Version {
    /// `true` when oneDNN's primitives do not spawn threads of their own.
    pub(crate) fn is_sequential(&self) -> bool {
        self.cpu_runtime == RUNTIME_SEQ
    }

    /// `"3.14.1"`-style string.
    pub(crate) fn number(&self) -> String {
        format!("{}.{}.{}", self.major, self.minor, self.patch)
    }

    /// The CPU threading runtime's name.
    pub(crate) fn cpu_runtime_name(&self) -> &'static str {
        match self.cpu_runtime {
            0 => "none",
            1 => "seq",
            2 => "omp",
            4 => "tbb",
            8 => "threadpool",
            256 => "ocl",
            512 => "sycl",
            1024 => "ze",
            _ => "unknown",
        }
    }
}

/// Query the linked library's version.
pub(crate) fn runtime_version() -> Result<Version, Fn3Error> {
    // SAFETY: `dnnl_version` returns a pointer to a static, immutable version
    // structure owned by the library; it is valid for the process lifetime and
    // is only read here.
    let raw = unsafe { dnnl_version() };
    if raw.is_null() {
        return Err(Fn3Error::ProviderFailure {
            provider: PROVIDER,
            call: "dnnl_version",
            status: STATUS_SUCCESS,
            detail: "null version pointer".to_string(),
        });
    }
    // SAFETY: `raw` is the static version structure described above.
    let raw = unsafe { &*raw };
    let hash = if raw.hash.is_null() {
        String::new()
    } else {
        // SAFETY: a non-null `hash` is a NUL-terminated C string owned by the
        // static version structure, valid for the process lifetime.
        unsafe { std::ffi::CStr::from_ptr(raw.hash) }
            .to_string_lossy()
            .into_owned()
    };
    Ok(Version {
        major: raw.major,
        minor: raw.minor,
        patch: raw.patch,
        hash,
        cpu_runtime: raw.cpu_runtime,
    })
}

/// A diagnostic name for a status, for an error that has to explain itself.
pub(crate) fn status_name(status: i32) -> &'static str {
    match status {
        STATUS_SUCCESS => "success",
        STATUS_OUT_OF_MEMORY => "out_of_memory",
        STATUS_INVALID_ARGUMENTS => "invalid_arguments",
        STATUS_UNIMPLEMENTED => "unimplemented",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::AlignedBuffer;

    /// The measurement behind `Precision::W8A8`'s refusal on non-VNNI hosts.
    ///
    /// Ignored because it is a benchmark, not a gate: on AVX2 the int8
    /// primitive is oneDNN's reference kernel and takes ~1.3 s for a
    /// 32x80x80x3x3 convolution, against 1.6 ms for f32. Run it on a VNNI host
    /// before enabling int8 execution (issue #103).
    #[test]
    #[ignore = "benchmark; measures the int8 fallback"]
    fn probe_int8_speed() {
        let engine = EngineHandle::cpu().expect("cpu engine");
        let stream = StreamHandle::new(&engine).expect("stream");
        let attr = PrimitiveAttrHandle::library_default().expect("attr");
        let (n, c, h, w, o, k) = (1i64, 32i64, 80i64, 80i64, 32i64, 3i64);
        let src_dims = [n, c, h, w];
        let w_dims = [o, c, k, k];
        let dst_dims = [n, o, h, w];
        for (name, sdt, wdt, ddt, sb, wb, db) in [
            (
                "f32",
                DATA_TYPE_F32,
                DATA_TYPE_F32,
                DATA_TYPE_F32,
                4usize,
                4usize,
                4usize,
            ),
            ("int8", DATA_TYPE_U8, DATA_TYPE_S8, DATA_TYPE_S32, 1, 1, 4),
        ] {
            let src = Desc::with_tag_dtype(&src_dims, 5, sdt).unwrap();
            let weights = Desc::with_tag_dtype(&w_dims, 5, wdt).unwrap();
            let dst = Desc::with_tag_dtype(&dst_dims, 5, ddt).unwrap();
            let pd = PrimitiveDescHandle::convolution(
                &engine,
                ALG_CONVOLUTION_DIRECT,
                &src,
                &weights,
                None,
                &dst,
                &[1, 1],
                &[0, 0],
                &[1, 1],
                &[1, 1],
                &attr,
            )
            .unwrap();
            let primitive = PrimitiveHandle::new(&pd).unwrap();
            let src_buffer = AlignedBuffer::zeroed((n * c * h * w) as usize * sb).unwrap();
            let w_buffer = AlignedBuffer::zeroed((o * c * k * k) as usize * wb).unwrap();
            let dst_buffer = AlignedBuffer::zeroed((n * o * h * w) as usize * db).unwrap();
            let src_memory = MemoryHandle::new(&src, &engine).unwrap();
            let w_memory = MemoryHandle::new(&weights, &engine).unwrap();
            let d_memory = MemoryHandle::new(&dst, &engine).unwrap();
            unsafe {
                src_memory
                    .set_data_handle(src_buffer.as_ptr() as *mut u8)
                    .unwrap();
                w_memory
                    .set_data_handle(w_buffer.as_ptr() as *mut u8)
                    .unwrap();
                d_memory
                    .set_data_handle(dst_buffer.as_ptr() as *mut u8)
                    .unwrap();
            }
            let args = [
                ExecArg {
                    arg: ARG_SRC,
                    memory: src_memory.raw(),
                },
                ExecArg {
                    arg: ARG_WEIGHTS,
                    memory: w_memory.raw(),
                },
                ExecArg {
                    arg: ARG_DST,
                    memory: d_memory.raw(),
                },
            ];
            for _ in 0..3 {
                primitive.execute(&stream, &args).unwrap();
                stream.wait().unwrap();
            }
            let mut times = Vec::new();
            for _ in 0..5 {
                let start = std::time::Instant::now();
                primitive.execute(&stream, &args).unwrap();
                stream.wait().unwrap();
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "probe speed {name}: median {:.3} ms, min {:.3}, max {:.3}",
                times[times.len() / 2],
                times[0],
                times[times.len() - 1]
            );
        }
    }

    /// The smoke test that decides whether this build can execute a true int8
    /// convolution: u8 activations, s8 weights, s32 accumulator. It is the
    /// narrowest shape (1x1x3x3, 2x2 kernel) so a failure names the primitive,
    /// not the data.
    #[test]
    fn an_int8_convolution_accumulates_in_s32() {
        let engine = EngineHandle::cpu().expect("cpu engine");
        let stream = StreamHandle::new(&engine).expect("stream");
        let attr = PrimitiveAttrHandle::library_default().expect("attr");
        let src =
            Desc::with_tag_dtype(&[1, 1, 3, 3], FORMAT_TAG_ABCD, DATA_TYPE_U8).expect("src desc");
        let weights = Desc::with_tag_dtype(&[1, 1, 2, 2], FORMAT_TAG_ABCD, DATA_TYPE_S8)
            .expect("weights desc");
        let dst =
            Desc::with_tag_dtype(&[1, 1, 2, 2], FORMAT_TAG_ABCD, DATA_TYPE_S32).expect("dst desc");
        let pd = PrimitiveDescHandle::convolution(
            &engine,
            ALG_CONVOLUTION_DIRECT,
            &src,
            &weights,
            None,
            &dst,
            &[1, 1],
            &[0, 0],
            &[0, 0],
            &[0, 0],
            &attr,
        )
        .expect("int8 convolution primitive descriptor");
        let primitive = PrimitiveHandle::new(&pd).expect("primitive");

        let input: Vec<u8> = (1..=9u8).collect();
        let filter: Vec<i8> = vec![1, 2, 3, 4];
        let mut output = vec![0i32; 4];
        let src_memory = MemoryHandle::new(&src, &engine).expect("src memory");
        let weights_memory = MemoryHandle::new(&weights, &engine).expect("weights memory");
        let dst_memory = MemoryHandle::new(&dst, &engine).expect("dst memory");
        unsafe {
            src_memory
                .set_data_handle(input.as_ptr() as *mut u8)
                .expect("src handle");
            weights_memory
                .set_data_handle(filter.as_ptr() as *mut u8)
                .expect("weights handle");
            dst_memory
                .set_data_handle(output.as_mut_ptr() as *mut u8)
                .expect("dst handle");
        }
        primitive
            .execute(
                &stream,
                &[
                    ExecArg {
                        arg: ARG_SRC,
                        memory: src_memory.raw(),
                    },
                    ExecArg {
                        arg: ARG_WEIGHTS,
                        memory: weights_memory.raw(),
                    },
                    ExecArg {
                        arg: ARG_DST,
                        memory: dst_memory.raw(),
                    },
                ],
            )
            .expect("execute");
        stream.wait().expect("wait");
        assert_eq!(output, vec![37, 47, 67, 77]);
    }
}
