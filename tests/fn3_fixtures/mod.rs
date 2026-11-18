//! Read-only loader for the FastNN parity fixtures (`fixtures/parity/**`).
//!
//! This module is test infrastructure, deliberately independent of any kernel:
//! it parses `manifest.json`, decodes the committed little-endian payloads,
//! verifies their SHA-256 digests, and implements the comparison predicates
//! defined in `fixtures/parity/README.md`. Later issues (#62 end-to-end) reuse
//! it unchanged, so the API is intentionally small:
//!
//! ```text
//! Suite::load("ops")                  // fixtures/parity/ops
//! suite.tensor("matmul.2d_k3.out")    // RefTensor
//! tensor.dtype() / shape() / stored()
//! tensor.f32_values() / i64_values()  // decodes and verifies the payload
//! tensor.tolerance()                  // { atol, rtol, mean_abs_bound }
//! tensor.check(Actual::F32(..), shape)// elementwise + non-finite predicates
//! tensor.check_summary(Actual::F32(..), shape)
//! tensor.check_slices(Actual::F32(..), shape)
//! ```
//!
//! # What is a fixture failure here
//!
//! `fixtures/parity/README.md` defines the predicates; this module implements
//! them once so every suite applies the same rule:
//!
//! 1. shape or dtype mismatch;
//! 2. NaN or Inf where the reference is finite (and finiteness mismatches in
//!    the other direction);
//! 3. `abs(actual - ref) <= atol + rtol * abs(ref)` for every element;
//! 4. `mean_abs` above a stated bound, when the manifest states one;
//! 5. a different argmax, but only when the reference's top-two gap exceeds
//!    `2 * atol`;
//! 6. a stored payload whose SHA-256 does not match.
//!
//! Every `check_*` method returns `Result<Agreement, String>`: `Agreement`
//! carries the measured `max_abs`/`mean_abs` so a test can *report* what it
//! measured instead of only asserting that it passed.
//!
//! # Exact digests vs. independent implementations
//!
//! The manifest's `sha256` (and `slices[].sha256`) are *regeneration*
//! identities: they match a byte-identical payload produced by
//! `tools/parity/generate_fixtures.py`. They are verified here for **stored**
//! payloads (fixture integrity), and they cannot gate an independent
//! implementation, which is why summary-only tensors are compared through their
//! `summary` blocks (which carry a tolerance) instead.
//!
//! # Panics
//!
//! The loader panics only on a corrupt or missing fixture: that is a test
//! infrastructure failure, not a kernel result, so a precise panic message is
//! more useful than a `Result` threaded through every test. Kernel results are
//! never panics: they are `Result`s.

// The module is a small, documented API reused by later suites (#62), so parts
// of it are intentionally not exercised by the FN3-07 test binary yet.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Element type of a reference tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefDType {
    F32,
    I64,
}

impl RefDType {
    pub fn name(self) -> &'static str {
        match self {
            RefDType::F32 => "f32",
            RefDType::I64 => "i64",
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            RefDType::F32 => 4,
            RefDType::I64 => 8,
        }
    }
}

impl fmt::Display for RefDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A caller-provided tensor, in the two dtypes the fixtures store.
#[derive(Clone, Copy, Debug)]
pub enum Actual<'a> {
    F32(&'a [f32]),
    I64(&'a [i64]),
}

impl Actual<'_> {
    pub fn len(&self) -> usize {
        match self {
            Actual::F32(values) => values.len(),
            Actual::I64(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> RefDType {
        match self {
            Actual::F32(_) => RefDType::F32,
            Actual::I64(_) => RefDType::I64,
        }
    }

    fn as_f32(&self) -> &[f32] {
        match self {
            Actual::F32(values) => values,
            Actual::I64(_) => panic!("fixture check: expected f32 actual values"),
        }
    }
}

/// Tolerance block of one tensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
    /// Optional systematic-bias bound (`mean_abs`), when the manifest states
    /// one. The operator suite does not; the whole-model policy block does
    /// (exposed through [`Suite::mean_abs_policy`]).
    pub mean_abs_bound: Option<f64>,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance {
            atol: 0.0,
            rtol: 0.0,
            mean_abs_bound: None,
        }
    }
}

impl Tolerance {
    /// `true` when `abs(actual - reference) <= atol + rtol * abs(reference)`.
    ///
    /// This is the elementwise rule, applied to payload bytes decoded exactly.
    pub fn permits(&self, actual: f64, reference: f64) -> bool {
        (actual - reference).abs() <= self.atol + self.rtol * reference.abs()
    }

    /// The same rule with a decimal round-trip allowance, for `summary` and
    /// `slices[].summary` statistics.
    ///
    /// Summary statistics are stored as JSON decimals. They are printed with 17
    /// significant digits, but `serde_json`'s default float parser can land one
    /// `f64` ulp away from the correctly-rounded value (a one-ulp case is
    /// pinned by `json_statistics_carry_one_ulp_of_round_trip_noise`), so a
    /// statistics comparison needs a `1e-15` relative allowance. Elementwise
    /// checks read raw little-endian payload bytes and stay exact.
    pub fn permits_statistic(&self, actual: f64, reference: f64) -> bool {
        (actual - reference).abs() <= self.atol + (self.rtol + 1.0e-15) * reference.abs() + 1.0e-300
    }
}

/// Arithmetic mean of `|actual - reference|`.
pub fn mean_abs_error(actual: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(actual.len(), reference.len(), "mean_abs_error length");
    if actual.is_empty() {
        return 0.0;
    }
    actual
        .iter()
        .zip(reference.iter())
        .map(|(&actual, &reference)| (f64::from(actual) - f64::from(reference)).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Predicate 4: `true` when the mean absolute error exceeds `bound`.
pub fn mean_abs_bound_violated(actual: &[f32], reference: &[f32], bound: f64) -> bool {
    mean_abs_error(actual, reference) > bound
}

impl fmt::Display for Tolerance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "atol={:e} rtol={:e}", self.atol, self.rtol)?;
        if let Some(bound) = self.mean_abs_bound {
            write!(f, " mean_abs<={bound:e}")?;
        }
        Ok(())
    }
}

/// Measured agreement between an implementation and the reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Agreement {
    pub elements: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    /// Reference value at the worst element, for reporting.
    pub worst_reference: f64,
}

impl fmt::Display for Agreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} elements: max_abs={:e} mean_abs={:e} (worst ref {:e})",
            self.elements, self.max_abs, self.mean_abs, self.worst_reference
        )
    }
}

/// One parity suite: a directory with `manifest.json` and `tensors/`.
pub struct Suite {
    root: PathBuf,
    manifest: Value,
    index: BTreeMap<String, usize>,
}

impl Suite {
    /// Load the committed suite named `name` (`"ops"`, `"tiny-gpt2"`, ...).
    pub fn load(name: &str) -> Suite {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("parity")
            .join(name);
        Suite::load_at(&root)
    }

    /// Load a suite from an explicit directory.
    pub fn load_at(root: &Path) -> Suite {
        let manifest_path = root.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).unwrap_or_else(|error| {
            panic!("fixture manifest {}: {error}", manifest_path.display())
        });
        let manifest: Value = serde_json::from_str(&text).unwrap_or_else(|error| {
            panic!("fixture manifest {}: {error}", manifest_path.display())
        });
        let format = manifest["format"].as_str().unwrap_or("");
        assert_eq!(
            format,
            "fastnn-parity-tensors",
            "{}: unexpected fixture format {format:?}",
            manifest_path.display()
        );
        let mut index = BTreeMap::new();
        let tensors = manifest["tensors"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: tensors must be an array", manifest_path.display()));
        for (position, tensor) in tensors.iter().enumerate() {
            let name = tensor["name"].as_str().unwrap_or_else(|| {
                panic!("{}: tensor {position} has no name", manifest_path.display())
            });
            assert!(
                index.insert(name.to_string(), position).is_none(),
                "{}: duplicate tensor name {name}",
                manifest_path.display()
            );
        }
        Suite {
            root: root.to_path_buf(),
            manifest,
            index,
        }
    }

    /// Directory the suite was loaded from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Suite name from the manifest.
    pub fn name(&self) -> &str {
        self.manifest["suite"].as_str().unwrap_or("")
    }

    /// Manifest value, for cases that need provenance this API does not model.
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }

    /// Every tensor name, in manifest order.
    pub fn names(&self) -> Vec<&str> {
        self.manifest["tensors"]
            .as_array()
            .map(|tensors| {
                tensors
                    .iter()
                    .filter_map(|tensor| tensor["name"].as_str())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Every tensor entry.
    pub fn tensors(&self) -> impl Iterator<Item = RefTensor<'_>> {
        (0..self.index.len()).map(move |position| RefTensor {
            suite: self,
            entry: &self.manifest["tensors"][position],
        })
    }

    /// The tensor named `name`.
    pub fn tensor(&self, name: &str) -> RefTensor<'_> {
        let position = self
            .index
            .get(name)
            .unwrap_or_else(|| panic!("{}: no tensor named {name}", self.root.display()));
        RefTensor {
            suite: self,
            entry: &self.manifest["tensors"][*position],
        }
    }

    /// The `mean_abs` bound from the suite's `tolerance_policy`, if stated.
    pub fn mean_abs_policy(&self) -> Option<f64> {
        self.manifest["provenance"]["tolerance_policy"]["mean_abs"].as_f64()
    }

    /// Verify every stored payload's length and SHA-256.
    ///
    /// Returns the number of bytes verified.
    pub fn verify_stored_payloads(&self) -> usize {
        let mut total = 0usize;
        for tensor in self.tensors() {
            if tensor.stored() {
                total += tensor.bytes().len();
            }
        }
        total
    }
}

/// One tensor entry in a suite manifest.
#[derive(Clone, Copy)]
pub struct RefTensor<'a> {
    suite: &'a Suite,
    entry: &'a Value,
}

impl<'a> RefTensor<'a> {
    /// Tensor name.
    pub fn name(&self) -> &'a str {
        self.entry["name"].as_str().unwrap_or("")
    }

    /// Declared dtype.
    pub fn dtype(&self) -> RefDType {
        match self.entry["dtype"].as_str() {
            Some("f32") => RefDType::F32,
            Some("i64") => RefDType::I64,
            other => panic!("{}: unsupported dtype {other:?}", self.name()),
        }
    }

    /// Declared shape, row-major.
    pub fn shape(&self) -> Vec<u64> {
        self.entry["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("{}: shape must be an array", self.name()))
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .unwrap_or_else(|| panic!("{}: shape dimension is not unsigned", self.name()))
            })
            .collect()
    }

    /// Number of elements implied by the shape.
    pub fn element_count(&self) -> usize {
        self.shape()
            .iter()
            .try_fold(1u64, |product, &dim| product.checked_mul(dim))
            .and_then(|elements| usize::try_from(elements).ok())
            .unwrap_or_else(|| panic!("{}: element count does not fit in usize", self.name()))
    }

    /// `true` when the payload is present in this checkout.
    pub fn stored(&self) -> bool {
        self.entry["stored"].as_bool().unwrap_or(false)
    }

    /// Tolerance for this tensor (`provenance.tolerance`).
    pub fn tolerance(&self) -> Tolerance {
        let block = &self.entry["provenance"]["tolerance"];
        Tolerance {
            atol: block["atol"].as_f64().unwrap_or(0.0),
            rtol: block["rtol"].as_f64().unwrap_or(0.0),
            mean_abs_bound: block["mean_abs_bound"].as_f64(),
        }
    }

    /// The manifest's `summary` block (counts, min/max, first/last, argmax).
    pub fn summary(&self) -> &'a Value {
        &self.entry["summary"]
    }

    /// The manifest's `slices` block, if present.
    pub fn slices(&self) -> Option<&'a Value> {
        self.entry.get("slices").filter(|value| !value.is_null())
    }

    /// The declared digest of the complete payload.
    pub fn digest(&self) -> &'a str {
        self.entry["sha256"].as_str().unwrap_or("")
    }

    /// The verified payload bytes of a stored tensor.
    ///
    /// Panics if the tensor is summary-only, the length differs, or the digest
    /// does not match: all three mean the checkout is corrupt.
    pub fn bytes(&self) -> Vec<u8> {
        assert!(
            self.stored(),
            "{}: summary-only tensor has no payload; regenerate it with \
             tools/parity/generate_fixtures.py",
            self.name()
        );
        let relative = self.entry["path"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: stored tensor has no path", self.name()));
        let path = self.suite.root.join(relative);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("{}: reading {}: {error}", self.name(), path.display()));
        let declared = self.entry["bytes"]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: missing byte length", self.name()))
            as usize;
        assert_eq!(
            bytes.len(),
            declared,
            "{}: {} has {} bytes, expected {declared}",
            self.name(),
            path.display(),
            bytes.len()
        );
        let digest = sha256_hex(&bytes);
        assert_eq!(
            digest,
            self.digest(),
            "{}: payload digest mismatch (fixture corrupt)",
            self.name()
        );
        bytes
    }

    /// Decode a stored `f32` payload.
    pub fn f32_values(&self) -> Vec<f32> {
        assert_eq!(
            self.dtype(),
            RefDType::F32,
            "{}: not an f32 tensor",
            self.name()
        );
        let bytes = self.bytes();
        assert_eq!(
            bytes.len() % 4,
            0,
            "{}: f32 payload is not a multiple of 4 bytes",
            self.name()
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Decode a stored `i64` payload.
    pub fn i64_values(&self) -> Vec<i64> {
        assert_eq!(
            self.dtype(),
            RefDType::I64,
            "{}: not an i64 tensor",
            self.name()
        );
        let bytes = self.bytes();
        assert_eq!(
            bytes.len() % 8,
            0,
            "{}: i64 payload is not a multiple of 8 bytes",
            self.name()
        );
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect()
    }

    /// The reference values as `f32`, whether the tensor is stored or not.
    ///
    /// Summary-only tensors have no payload: this panics for them, which keeps
    /// an elementwise check from silently passing on missing data.
    pub fn reference_f32(&self) -> Vec<f32> {
        self.f32_values()
    }

    /// Elementwise comparison against a stored reference.
    ///
    /// Implements predicates 1-3 of the README (shape/dtype, non-finite
    /// mismatch, elementwise bound) and returns the measured agreement.
    pub fn check(&self, actual: Actual<'_>, shape: &[u64]) -> Result<Agreement, String> {
        if actual.dtype() != self.dtype() {
            return Err(format!(
                "{}: dtype mismatch: actual {} vs reference {}",
                self.name(),
                actual.dtype(),
                self.dtype()
            ));
        }
        let expected_shape = self.shape();
        if shape != expected_shape.as_slice() {
            return Err(format!(
                "{}: shape mismatch: actual {shape:?} vs reference {expected_shape:?}",
                self.name()
            ));
        }
        if actual.len() != self.element_count() {
            return Err(format!(
                "{}: element count mismatch: actual {} vs reference {}",
                self.name(),
                actual.len(),
                self.element_count()
            ));
        }
        match actual {
            Actual::I64(values) => {
                let reference = self.i64_values();
                for (position, (&actual, &reference)) in
                    values.iter().zip(reference.iter()).enumerate()
                {
                    if actual != reference {
                        return Err(format!(
                            "{}: i64 element {position} is {actual}, reference is {reference} \
                             (exact match required)",
                            self.name()
                        ));
                    }
                }
                Ok(Agreement {
                    elements: values.len(),
                    max_abs: 0.0,
                    mean_abs: 0.0,
                    worst_reference: 0.0,
                })
            }
            Actual::F32(values) => {
                let reference = self.f32_values();
                let tolerance = self.tolerance();
                let mut agreement = Agreement {
                    elements: values.len(),
                    max_abs: 0.0,
                    mean_abs: 0.0,
                    worst_reference: 0.0,
                };
                let mut total = 0.0f64;
                for (position, (&actual, &reference)) in
                    values.iter().zip(reference.iter()).enumerate()
                {
                    // Predicate 2: finiteness must agree.
                    if actual.is_finite() != reference.is_finite() {
                        return Err(format!(
                            "{}: element {position} finiteness mismatch: actual {actual}, \
                             reference {reference}",
                            self.name()
                        ));
                    }
                    if !reference.is_finite() {
                        let same = (actual.is_nan() && reference.is_nan())
                            || (actual == reference && reference.is_infinite());
                        if !same {
                            return Err(format!(
                                "{}: element {position} is {actual}, reference {reference}",
                                self.name()
                            ));
                        }
                        continue;
                    }
                    let difference = (f64::from(actual) - f64::from(reference)).abs();
                    total += difference;
                    if difference > agreement.max_abs {
                        agreement.max_abs = difference;
                        agreement.worst_reference = f64::from(reference);
                    }
                    // Predicate 3.
                    if !tolerance.permits(f64::from(actual), f64::from(reference)) {
                        return Err(format!(
                            "{}: element {position} is {actual}, reference {reference}, \
                             |diff|={difference:e} exceeds {tolerance}",
                            self.name()
                        ));
                    }
                }
                if !values.is_empty() {
                    agreement.mean_abs = total / values.len() as f64;
                }
                // Predicate 4, only when the manifest states a bound.
                if let Some(bound) = tolerance.mean_abs_bound {
                    if agreement.mean_abs > bound {
                        return Err(format!(
                            "{}: mean_abs {:e} exceeds the stated bound {bound:e}",
                            self.name(),
                            agreement.mean_abs
                        ));
                    }
                }
                Ok(agreement)
            }
        }
    }

    /// Compare the summary statistics of `actual` against the manifest's.
    ///
    /// This is the check a summary-only tensor supports: element counts and
    /// NaN/Inf counts exactly, `min`/`max`/`max_abs`/`mean_abs` and the
    /// `first`/`last` values within the tensor's tolerance, and the `argmax`
    /// rule (a different argmax fails only when the reference top-two gap
    /// exceeds `2 * atol`).
    pub fn check_summary(&self, actual: Actual<'_>, shape: &[u64]) -> Result<(), String> {
        if actual.dtype() != self.dtype() {
            return Err(format!(
                "{}: dtype mismatch: actual {} vs reference {}",
                self.name(),
                actual.dtype(),
                self.dtype()
            ));
        }
        let expected_shape = self.shape();
        if shape != expected_shape.as_slice() {
            return Err(format!(
                "{}: shape mismatch: actual {shape:?} vs reference {expected_shape:?}",
                self.name()
            ));
        }
        if actual.len() != self.element_count() {
            return Err(format!(
                "{}: element count mismatch: actual {} vs reference {}",
                self.name(),
                actual.len(),
                self.element_count()
            ));
        }
        let summary = self.summary();
        let values = actual.as_f32();
        let tolerance = self.tolerance();

        if let Some(count) = summary["count"].as_u64() {
            if count as usize != values.len() {
                return Err(format!(
                    "{}: summary count {count} vs actual {} elements",
                    self.name(),
                    values.len()
                ));
            }
        }
        let nan_count = values.iter().filter(|value| value.is_nan()).count();
        let inf_count = values.iter().filter(|value| value.is_infinite()).count();
        if let Some(expected) = summary["nan_count"].as_u64() {
            if expected as usize != nan_count {
                return Err(format!(
                    "{}: summary nan_count {expected} vs actual {nan_count}",
                    self.name()
                ));
            }
        }
        if let Some(expected) = summary["inf_count"].as_u64() {
            if expected as usize != inf_count {
                return Err(format!(
                    "{}: summary inf_count {expected} vs actual {inf_count}",
                    self.name()
                ));
            }
        }
        let finite: Vec<f64> = values
            .iter()
            .filter(|value| value.is_finite())
            .map(|&value| f64::from(value))
            .collect();
        let statistic = |name: &str| summary[name].as_f64();
        if let Some(expected) = statistic("min") {
            if let Some(actual) = finite.iter().copied().reduce(f64::min) {
                if !tolerance.permits_statistic(actual, expected) {
                    return Err(format!(
                        "{}: min {actual:e} vs reference {expected:e} ({tolerance})",
                        self.name()
                    ));
                }
            }
        }
        if let Some(expected) = statistic("max") {
            if let Some(actual) = finite.iter().copied().reduce(f64::max) {
                if !tolerance.permits_statistic(actual, expected) {
                    return Err(format!(
                        "{}: max {actual:e} vs reference {expected:e} ({tolerance})",
                        self.name()
                    ));
                }
            }
        }
        if let Some(expected) = statistic("max_abs") {
            if let Some(actual) = finite.iter().map(|value| value.abs()).reduce(f64::max) {
                if !tolerance.permits_statistic(actual, expected) {
                    return Err(format!(
                        "{}: max_abs {actual:e} vs reference {expected:e} ({tolerance})",
                        self.name()
                    ));
                }
            }
        }
        if let Some(expected) = statistic("mean_abs") {
            if !finite.is_empty() {
                let actual =
                    finite.iter().map(|value| value.abs()).sum::<f64>() / finite.len() as f64;
                if !tolerance.permits_statistic(actual, expected) {
                    return Err(format!(
                        "{}: mean_abs {actual:e} vs reference {expected:e} ({tolerance})",
                        self.name()
                    ));
                }
            }
        }
        for (label, key) in [("first", "first"), ("last", "last")] {
            let Some(expected) = summary[key].as_array() else {
                continue;
            };
            let actual_slice: Vec<f32> = if key == "first" {
                values.iter().take(expected.len()).copied().collect()
            } else {
                values
                    .iter()
                    .skip(values.len().saturating_sub(expected.len()))
                    .copied()
                    .collect()
            };
            for (position, expected) in expected.iter().enumerate() {
                let Some(expected) = expected.as_f64() else {
                    continue;
                };
                let Some(&actual) = actual_slice.get(position) else {
                    continue;
                };
                if !tolerance.permits_statistic(f64::from(actual), expected) {
                    return Err(format!(
                        "{}: {label}[{position}] is {actual:e}, reference {expected:e} \
                         ({tolerance})",
                        self.name()
                    ));
                }
            }
        }
        if let Some(expected) = summary["argmax_flat"].as_i64() {
            let actual_argmax = argmax_flat(values);
            if actual_argmax != expected {
                let gap = top_two_gap(values);
                if gap > 2.0 * tolerance.atol {
                    return Err(format!(
                        "{}: argmax_flat {actual_argmax} vs reference {expected}, and the \
                         reference top-two gap {gap:e} exceeds 2*atol",
                        self.name()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Compare per-slice summaries along `slices.axis`.
    ///
    /// Stored tensors also verify each slice's exact digest (fixture
    /// integrity). Summary-only tensors compare the per-slice `summary` block
    /// within tolerance, because exact digests are a regeneration identity, not
    /// an independent-implementation gate.
    pub fn check_slices(&self, actual: Actual<'_>, shape: &[u64]) -> Result<(), String> {
        let Some(slices) = self.slices() else {
            return Ok(());
        };
        let axis = slices["axis"]
            .as_u64()
            .ok_or_else(|| format!("{}: slices block has no axis", self.name()))?
            as usize;
        if axis >= shape.len() {
            return Err(format!(
                "{}: slices axis {axis} is outside {shape:?}",
                self.name()
            ));
        }
        let reference_values: Option<Vec<f32>> = if self.stored() && self.dtype() == RefDType::F32 {
            Some(self.f32_values())
        } else {
            None
        };
        let items = slices["items"]
            .as_array()
            .ok_or_else(|| format!("{}: slices has no items", self.name()))?;
        for item in items {
            let index = item["index"]
                .as_u64()
                .ok_or_else(|| format!("{}: slice item has no index", self.name()))?;
            let item_shape: Vec<u64> = item["shape"]
                .as_array()
                .ok_or_else(|| format!("{}: slice item has no shape", self.name()))?
                .iter()
                .map(|dim| dim.as_u64().unwrap_or(0))
                .collect();
            let row = slice_along(actual, shape, axis, index)?;
            if item_shape.iter().product::<u64>() as usize != row.len() {
                return Err(format!(
                    "{}: slice {index} has {} elements, manifest says {item_shape:?}",
                    self.name(),
                    row.len()
                ));
            }
            if let Some(reference) = reference_values.as_ref() {
                let reference_row = slice_along(Actual::F32(reference), shape, axis, index)?;
                let digest = sha256_hex(&by_f32_bytes(&reference_row));
                if let Some(expected) = item["sha256"].as_str() {
                    if digest != expected {
                        return Err(format!(
                            "{}: slice {index} digest mismatch (fixture corrupt)",
                            self.name()
                        ));
                    }
                }
            }
            if let Some(summary) = item.get("summary").filter(|value| !value.is_null()) {
                let tolerance = self.tolerance();
                let finite: Vec<f64> = row
                    .iter()
                    .filter(|value| value.is_finite())
                    .map(|&value| f64::from(value))
                    .collect();
                if let Some(expected) = summary["min"].as_f64() {
                    if let Some(actual) = finite.iter().copied().reduce(f64::min) {
                        if !tolerance.permits_statistic(actual, expected) {
                            return Err(format!(
                                "{}: slice {index} min {actual:e} vs {expected:e}",
                                self.name()
                            ));
                        }
                    }
                }
                if let Some(expected) = summary["max"].as_f64() {
                    if let Some(actual) = finite.iter().copied().reduce(f64::max) {
                        if !tolerance.permits_statistic(actual, expected) {
                            return Err(format!(
                                "{}: slice {index} max {actual:e} vs {expected:e}",
                                self.name()
                            ));
                        }
                    }
                }
                if let Some(expected) = summary["mean_abs"].as_f64() {
                    if !finite.is_empty() {
                        let actual =
                            finite.iter().map(|v| v.abs()).sum::<f64>() / finite.len() as f64;
                        if !tolerance.permits_statistic(actual, expected) {
                            return Err(format!(
                                "{}: slice {index} mean_abs {actual:e} vs {expected:e}",
                                self.name()
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl fmt::Debug for RefTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefTensor")
            .field("name", &self.name())
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .field("stored", &self.stored())
            .finish()
    }
}

/// The values of `actual` where `shape[axis] == index`, in row-major order.
pub fn slice_along(
    actual: Actual<'_>,
    shape: &[u64],
    axis: usize,
    index: u64,
) -> Result<Vec<f32>, String> {
    let axis_extent = *shape
        .get(axis)
        .ok_or_else(|| format!("slices axis {axis} is outside {shape:?}"))?;
    if index >= axis_extent {
        return Err(format!(
            "slice index {index} is outside axis {axis} extent {axis_extent}"
        ));
    }
    let inner: usize = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| "slice inner element count overflows".to_string())?;
    let outer: usize = shape[..axis]
        .iter()
        .try_fold(1usize, |product, &dim| product.checked_mul(dim as usize))
        .ok_or_else(|| "slice outer element count overflows".to_string())?;
    let values = actual.as_f32();
    let mut row = Vec::with_capacity(outer * inner);
    for block in 0..outer {
        let start = (block * axis_extent as usize + index as usize) * inner;
        let end = start + inner;
        let Some(slice) = values.get(start..end) else {
            return Err(format!(
                "slice {index} addresses {start}..{end} outside {} elements",
                values.len()
            ));
        };
        row.extend_from_slice(slice);
    }
    Ok(row)
}

/// Flat index of the maximum of `values`, first occurrence on ties.
pub fn argmax_flat(values: &[f32]) -> i64 {
    let mut best = f32::NEG_INFINITY;
    let mut best_index = 0i64;
    for (position, &value) in values.iter().enumerate() {
        if value > best {
            best = value;
            best_index = position as i64;
        }
    }
    best_index
}

/// Gap between the largest and second-largest finite value.
pub fn top_two_gap(values: &[f32]) -> f64 {
    let mut largest = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &value in values {
        let value = f64::from(value);
        if !value.is_finite() {
            continue;
        }
        if value > largest {
            second = largest;
            largest = value;
        } else if value > second {
            second = value;
        }
    }
    if second == f64::NEG_INFINITY {
        0.0
    } else {
        largest - second
    }
}

fn by_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

// ---------------------------------------------------------------------------
// SHA-256 (no external crates are allowed in this tree)
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 of `bytes` (FIPS 180-4).
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    for block in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (position, chunk) in block.chunks_exact(4).enumerate() {
            words[position] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for position in 16..64 {
            let s0 = words[position - 15].rotate_right(7)
                ^ words[position - 15].rotate_right(18)
                ^ (words[position - 15] >> 3);
            let s1 = words[position - 2].rotate_right(17)
                ^ words[position - 2].rotate_right(19)
                ^ (words[position - 2] >> 10);
            words[position] = words[position - 16]
                .wrapping_add(s0)
                .wrapping_add(words[position - 7])
                .wrapping_add(s1);
        }
        let mut working = state;
        for position in 0..64 {
            let s1 = working[4].rotate_right(6)
                ^ working[4].rotate_right(11)
                ^ working[4].rotate_right(25);
            let choice = (working[4] & working[5]) ^ (!working[4] & working[6]);
            let temp1 = working[7]
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(SHA256_K[position])
                .wrapping_add(words[position]);
            let s0 = working[0].rotate_right(2)
                ^ working[0].rotate_right(13)
                ^ working[0].rotate_right(22);
            let majority =
                (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
            let temp2 = s0.wrapping_add(majority);
            working[7] = working[6];
            working[6] = working[5];
            working[5] = working[4];
            working[4] = working[3].wrapping_add(temp1);
            working[3] = working[2];
            working[2] = working[1];
            working[1] = working[0];
            working[0] = temp1.wrapping_add(temp2);
        }
        for position in 0..8 {
            state[position] = state[position].wrapping_add(working[position]);
        }
    }

    let mut digest = [0u8; 32];
    for (position, word) in state.iter().enumerate() {
        digest[position * 4..position * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Lowercase hexadecimal SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in sha256(bytes) {
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_fips_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn suite_integrity_and_summary_predicates_hold_for_the_reference_itself() {
        let suite = Suite::load("ops");
        let bytes = suite.verify_stored_payloads();
        assert_eq!(bytes, 29_096, "ops suite stored payload bytes");
        assert_eq!(suite.names().len(), 47);
        for tensor in suite.tensors() {
            assert!(tensor.stored());
            let shape = tensor.shape();
            let actual = match tensor.dtype() {
                RefDType::F32 => Actual::F32(&tensor.f32_values()),
                RefDType::I64 => Actual::I64(&tensor.i64_values()),
            };
            if matches!(actual, Actual::F32(_)) {
                tensor
                    .check_summary(actual, &shape)
                    .unwrap_or_else(|error| panic!("{error}"));
                tensor
                    .check_slices(actual, &shape)
                    .unwrap_or_else(|error| panic!("{error}"));
            }
            tensor
                .check(actual, &shape)
                .unwrap_or_else(|error| panic!("{error}"));
        }
    }

    #[test]
    fn predicates_reject_shape_dtype_and_bounds_violations() {
        let suite = Suite::load("ops");
        let tensor = suite.tensor("matmul.2d_k3.out");
        let shape = tensor.shape();
        let values = tensor.f32_values();
        assert!(tensor.check(Actual::F32(&values), &shape).is_ok());
        // Shape mismatch.
        assert!(tensor
            .check(Actual::F32(&values), &[4, 2])
            .unwrap_err()
            .contains("shape mismatch"));
        // Dtype mismatch.
        assert!(tensor
            .check(Actual::I64(&[0i64; 8]), &shape)
            .unwrap_err()
            .contains("dtype mismatch"));
        // Elementwise bound.
        let mut perturbed = values.clone();
        perturbed[0] += 1.0;
        let error = tensor.check(Actual::F32(&perturbed), &shape).unwrap_err();
        assert!(error.contains("exceeds"), "{error}");
        // Non-finite mismatch.
        let mut poisoned = values.clone();
        poisoned[1] = f32::NAN;
        let error = tensor.check(Actual::F32(&poisoned), &shape).unwrap_err();
        assert!(error.contains("finiteness"), "{error}");
        // The same value at full precision passes.
        let exact = suite.tensor("gelu.erf.out");
        let exact_values = exact.f32_values();
        assert!(exact
            .check(Actual::F32(&exact_values), &exact.shape())
            .is_ok());
    }

    #[test]
    fn json_statistics_carry_one_ulp_of_round_trip_noise() {
        // Pinned because it is the reason summary statistics use
        // `Tolerance::permits_statistic` rather than `permits`: `serde_json`'s
        // default float path can land one f64 ulp above the correctly rounded
        // value of a 17-digit decimal.
        let parsed: serde_json::Value = serde_json::from_str("1.9864081144332886").unwrap();
        let parsed = parsed.as_f64().unwrap();
        let exact: f64 = 1.9864081144332886;
        assert!(parsed >= exact);
        assert!((parsed - exact).abs() <= f64::EPSILON * exact.abs());
        let tolerance = Tolerance::default();
        assert!(!tolerance.permits(parsed, exact), "exact rule rejects it");
        assert!(
            tolerance.permits_statistic(parsed, exact),
            "the statistics rule absorbs it"
        );
    }

    #[test]
    fn mean_abs_predicate_is_the_mean_error_not_the_mean_magnitude() {
        let reference = [1.0f32, -2.0, 3.0];
        assert_eq!(mean_abs_error(&reference, &reference), 0.0);
        let biased = [1.0f32 + 1e-3, -2.0 + 1e-3, 3.0 + 1e-3];
        assert!((mean_abs_error(&biased, &reference) - 1e-3).abs() < 1e-6);
        assert!(mean_abs_bound_violated(&biased, &reference, 5e-4));
        assert!(!mean_abs_bound_violated(&biased, &reference, 2e-3));
    }

    #[test]
    fn argmax_rule_only_gates_outside_the_noise_floor() {
        let suite = Suite::load("ops");
        let tensor = suite.tensor("softmax.unmasked.out");
        let values = tensor.f32_values();
        let shape = tensor.shape();
        assert!(tensor.check_summary(Actual::F32(&values), &shape).is_ok());
        // A tiny perturbation of a value that is not near the maximum must not
        // turn a summary check into a failure.
        let mut perturbed = values.clone();
        perturbed[0] += 1.0e-7;
        assert!(tensor
            .check_summary(Actual::F32(&perturbed), &shape)
            .is_ok());
    }

    #[test]
    fn slice_rows_are_extracted_in_row_major_order() {
        let suite = Suite::load("tiny-gpt2");
        let tensor = suite.tensor("prefill.p4.logits");
        assert!(!tensor.stored());
        assert_eq!(tensor.shape(), vec![1, 4, 50257]);
        let shape = tensor.shape();
        let synthetic: Vec<f32> = (0..4 * 50257).map(|value| value as f32).collect();
        let row = slice_along(Actual::F32(&synthetic), &shape, 1, 2).unwrap();
        assert_eq!(row.first().copied(), Some(2.0 * 50257.0));
        let _ = top_two_gap(&row);
    }
}
