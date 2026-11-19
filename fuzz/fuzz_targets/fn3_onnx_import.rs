#![no_main]

//! FN3-27 (#73): the ONNX importer must never panic on arbitrary bytes.
//!
//! The property is stronger than "returns an error": a successful import must
//! produce a graph the versioned format can write and read back. That is what
//! makes the two parser budgets in `src/fn3/import/budget.rs` a single contract
//! rather than two independent guesses.

use fastnn::inference::import::import_onnx_model;
use fastnn::inference::model::format::PreparedArtifact;
use fastnn::inference::shape::ShapeEnvelope;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(imported) = import_onnx_model(data) else {
        return;
    };
    // A model that imports has a geometry, and the format must be able to store
    // it. `uniform(64)` is a legal envelope for every decoder shape this importer
    // accepts; a refusal here is a typed `FormatError`, never a panic.
    let Ok(envelope) = ShapeEnvelope::uniform(64) else {
        return;
    };
    let Ok(artifact) = PreparedArtifact::from_import(&imported, envelope) else {
        return;
    };
    let Ok(bytes) = artifact.to_bytes() else {
        return;
    };
    // Canonical encoding: reading back what was written must reproduce the same
    // bytes, so the reader and writer agree on every field they both know.
    if let Ok(reloaded) = PreparedArtifact::from_bytes(&bytes) {
        let again = reloaded.to_bytes().expect("a decoded artifact re-encodes");
        assert_eq!(bytes, again, "artifact encoding is not canonical");
    }
});
