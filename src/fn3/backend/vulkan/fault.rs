//! Deterministic fault injection for the FN3-27 (#73) failure matrix.
//!
//! # Why this is a cargo feature, not `#[cfg(test)]`
//!
//! The failure matrix is an integration test over the public backend and
//! session surface. A `#[cfg(test)]` module would be invisible to
//! `tests/fn3_vulkan_faults.rs`, and an environment variable would ship a
//! failure path in the release binary. The `vulkan-fault-injection` feature is
//! therefore non-default, additive and compiled out of every shipped
//! configuration; the only code it changes is a `take()` check at the eight
//! seams the matrix (and the quarantine regression) exercises.
//!
//! # Semantics
//!
//! [`FaultInjector::arm`] selects one point and a skip count. The *skip-th*
//! matching event is the one that fails (`skip = 0` fails the next event); the
//! injector then disarms itself, so the retry after a failure runs the real
//! path. [`FaultInjector::fired`] and [`FaultInjector::calls`] let a test prove
//! the failure was the injected one and not a device accident.
//!
//! Injection is per backend, not process-global: two tests can run in one
//! process without seeing each other's events. A `FaultInjector` is `Send +
//! Sync` (a mutex around plain counters) so it can live behind `&Device`.

use std::sync::Mutex;

/// A boundary the failure matrix can fail deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultPoint {
    /// Device memory allocation (`vkAllocateMemory` path, before submission).
    Allocation,
    /// `vkQueueSubmit`, before the driver sees the submission.
    Submit,
    /// The fence wait, **after** the real wait completed (the work is not in
    /// flight when the failure is reported). This is the post-completion
    /// matrix case; [`FaultPoint::FenceUnknown`] is the pre-wait counterpart.
    Fence,
    /// The fence wait, **before** the real wait: completion is unknown and the
    /// submission may still be running. Every in-flight resource must move to
    /// the device quarantine, never be destroyed or recycled.
    FenceUnknown,
    /// Device → host readback (the logits download).
    Readback,
    /// Recording one dispatch into the open step batch. The failure happens
    /// inside the batch's `record` closure, so the batch is marked partial and
    /// `end_step` must discard it instead of submitting a prefix of the plan.
    Record,
    /// A device-to-device region copy (the KV past-prefix refresh and append).
    ResidentCopy,
    /// `vkDeviceWaitIdle` at a teardown boundary. The failure means the device
    /// cannot prove quiescence, so teardown must leak, not free.
    Idle,
}

impl FaultPoint {
    /// Every point, for exhaustive tests.
    pub const ALL: [FaultPoint; 8] = [
        FaultPoint::Allocation,
        FaultPoint::Submit,
        FaultPoint::Fence,
        FaultPoint::FenceUnknown,
        FaultPoint::Readback,
        FaultPoint::Record,
        FaultPoint::ResidentCopy,
        FaultPoint::Idle,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    /// Stable name used in the test output and the report.
    pub const fn name(self) -> &'static str {
        match self {
            FaultPoint::Allocation => "allocation",
            FaultPoint::Submit => "submit",
            FaultPoint::Fence => "fence",
            FaultPoint::FenceUnknown => "fence-unknown",
            FaultPoint::Readback => "readback",
            FaultPoint::Record => "record",
            FaultPoint::ResidentCopy => "resident-copy",
            FaultPoint::Idle => "idle",
        }
    }
}

#[derive(Debug, Default)]
struct State {
    /// `(point, events still to skip)` for the armed fault.
    armed: Option<(FaultPoint, u32)>,
    /// How many injections fired over the injector's lifetime.
    fired: u64,
    /// The event each point has seen.
    calls: [u64; FaultPoint::ALL.len()],
    /// The point that last fired.
    last: Option<FaultPoint>,
}

/// The per-backend injection state described in the module docs.
#[derive(Debug, Default)]
pub struct FaultInjector {
    state: Mutex<State>,
}

impl FaultInjector {
    /// An injector with nothing armed.
    pub fn new() -> FaultInjector {
        FaultInjector::default()
    }

    /// Fail the `skip`-th matching event (`0` = the next one), once.
    ///
    /// Arming replaces any previous arm; the skip count applies to events
    /// *after* the call.
    pub fn arm(&self, point: FaultPoint, skip: u32) {
        let mut state = self.lock();
        state.armed = Some((point, skip));
    }

    /// Cancel the armed fault.
    pub fn disarm(&self) {
        self.lock().armed = None;
    }

    /// `true` while a fault is armed.
    pub fn is_armed(&self) -> bool {
        self.lock().armed.is_some()
    }

    /// Number of injections that have fired.
    pub fn fired(&self) -> u64 {
        self.lock().fired
    }

    /// Events this point has seen, armed or not.
    pub fn calls(&self, point: FaultPoint) -> u64 {
        self.lock().calls[point.index()]
    }

    /// The point that last fired, if any.
    pub fn fired_at(&self) -> Option<FaultPoint> {
        self.lock().last
    }

    /// Record one event at `point`; `true` means the seam must fail it.
    ///
    /// This is the only function the production seams call. It never panics
    /// and never blocks for longer than the mutex critical section.
    pub fn take(&self, point: FaultPoint) -> bool {
        let mut state = self.lock();
        state.calls[point.index()] += 1;
        match state.armed {
            Some((armed, 0)) if armed == point => {
                state.armed = None;
                state.fired += 1;
                state.last = Some(point);
                true
            }
            Some((armed, remaining)) if armed == point => {
                state.armed = Some((armed, remaining - 1));
                false
            }
            _ => false,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fault_fires_once_at_the_skip_th_event() {
        let injector = FaultInjector::new();
        injector.arm(FaultPoint::Submit, 2);
        assert!(!injector.take(FaultPoint::Submit));
        assert!(!injector.take(FaultPoint::Submit));
        assert!(injector.take(FaultPoint::Submit), "the third event fires");
        assert_eq!(injector.fired(), 1);
        assert_eq!(injector.fired_at(), Some(FaultPoint::Submit));
        assert!(!injector.is_armed(), "a fired fault disarms itself");
        assert!(
            !injector.take(FaultPoint::Submit),
            "the retry runs the real path"
        );
        assert_eq!(injector.calls(FaultPoint::Submit), 4);
    }

    #[test]
    fn events_at_other_points_do_not_advance_the_skip_count() {
        let injector = FaultInjector::new();
        injector.arm(FaultPoint::Fence, 1);
        assert!(!injector.take(FaultPoint::Submit));
        assert!(!injector.take(FaultPoint::Fence));
        assert!(injector.take(FaultPoint::Fence));
        assert_eq!(injector.calls(FaultPoint::Submit), 1);
        assert_eq!(injector.calls(FaultPoint::Fence), 2);
    }

    #[test]
    fn disarm_cancels_and_rearm_replaces() {
        let injector = FaultInjector::new();
        injector.arm(FaultPoint::Allocation, 0);
        injector.disarm();
        assert!(!injector.take(FaultPoint::Allocation));
        injector.arm(FaultPoint::Allocation, 5);
        injector.arm(FaultPoint::Readback, 0);
        assert!(injector.take(FaultPoint::Readback));
        assert!(
            !injector.is_armed(),
            "the replaced arm was the readback one"
        );
        assert_eq!(injector.fired(), 1);
    }
}
