//! The v3 segment runtime: one plan, one arena, one pool.
//!
//! # Why this module exists
//!
//! The previous CPU path executed **one op at a time** through a runtime `match`
//! on [`OpId`](crate::inference::ops::OpId) over freshly allocated tensors. That is
//! the architecture ADR-0001 replaces: the plan's atom is a segment, memory is a
//! plan decision, and the threads belong to the session. This module is that
//! structure, with scalar kernels that are obviously correct so phase 3 can put a
//! provider behind the same seam without re-validating numerics.
//!
//! ```text
//! import ──► Graph ──► Plan { segments, slots, arena_bytes } ──► Session
//!                                                                  │
//!                          ┌───────────────────────────────────────┤
//!                          ▼                                       ▼
//!                    Arena (one 64-byte                          SessionThreads
//!                    allocation, slot views)                     (own rayon pool)
//! ```
//!
//! # The pieces
//!
//! | Module | Owns |
//! |---|---|
//! | [`plan`] | [`Plan`], [`Segment`], [`Slot`], liveness-based offsets |
//! | [`arena`] | [`Arena`]: one allocation, disjoint windows, allocation counter |
//! | [`threads`] | [`Threading`], [`SessionThreads`]: a session's own pool |
//! | [`exec`] | [`Session`]: validate inputs, walk segments, read outputs |
//! | `kernels` | scalar NCHW kernels for the twelve CNN op ids |
//!
//! # What is deliberately not here
//!
//! - **No fusion.** [`SegmentOp`] has one variant, `Op(OpId)`. A `Fused(..)`
//!   variant without a provider that claims it would be a contract nothing can
//!   satisfy; phase 3 adds it together with the provider.
//! - **No library.** Nothing in [`Plan`], [`Arena`], [`Session`] or the kernels
//!   names a compute library. oneDNN arrives as a provider behind
//!   [`Session`], and the plan stays the same plan.
//! - **No global state.** No `static` thread pool, no `lazy_static`, no import-time
//!   pinning. A pool exists only while a session exists.
//! - **No `HashMap`.** Every table is a `Vec` and every walk is in index order, so
//!   two plans over the same graph are equal and two forwards produce the same
//!   bytes.
//!
//! # The forty legacy ops and the twelve CNN ops
//!
//! The forty ops the decoder profile needs keep their kernels in
//! `crate::inference::backend::cpu::kernels`; the executor calls
//! `crate::inference::backend::cpu::dispatch_kernels`, the *same* table
//! `CpuBackend::execute_op` uses, with a thread budget of one. The twelve CNN ids
//! are implemented in `kernels` and reachable from both callers, which is why
//! `backend::cpu::support` can report them `Support::Full` honestly.
//!
//! That last sentence describes the one upward edge in the module graph:
//! `backend::cpu::dispatch_kernels` calls `runtime::kernels::dispatch` for the
//! CNN ids. It exists so a support verdict and its kernel cannot drift apart.
//! Every other dependency points down: `runtime` uses `backend`, `arena`,
//! `plan`, `storage` and `tensor`; nothing below `runtime` uses it except that
//! one call.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), fastnn::inference::Fn3Error> {
//! use std::sync::Arc;
//! use fastnn::inference::import::{import_onnx_graph, ImportedGraph};
//! use fastnn::inference::runtime::{Session, Threading};
//!
//! let bytes = std::fs::read("yolov8n.onnx").unwrap();
//! let imported: Arc<ImportedGraph> =
//!     Arc::new(import_onnx_graph(&bytes).map_err(|error| {
//!         fastnn::inference::Fn3Error::InvalidFormat {
//!             format: "onnx",
//!             detail: error.to_string(),
//!         }
//!     })?);
//! let mut session = Session::new(Arc::clone(&imported), Threading::Auto)?;
//! let input = vec![0.0f32; 3 * 320 * 320];
//! let outputs = session.run(&[("images", &input)])?;
//! assert_eq!(outputs[0].0, "output0");
//! # Ok(())
//! # }
//! ```

pub mod arena;
pub mod exec;
pub mod kernels;
pub mod plan;
pub mod threads;

pub use arena::Arena;
pub use exec::{HostTensor, Session};
pub use plan::{Plan, Segment, SegmentOp, Slot, SlotId, SlotSource, SLOT_ALIGNMENT};
pub use threads::{physical_core_count, SessionThreads, Threading};
