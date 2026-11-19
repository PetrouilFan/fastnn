#![no_main]

//! FN3-27 (#73): the prepared-model artifact reader must never panic, and what
//! it accepts must be canonical.
//!
//! The second half is the interesting one. The format documents "one model, one
//! byte string": `from_bytes` succeeds only for a canonical encoding, so a
//! decoded artifact must re-encode to the *same* bytes. A file that decodes but
//! re-encodes differently is a reader that accepted something the writer would
//! not produce, which is exactly the bug class this target exists to find.

use fastnn::fn3::model::format::PreparedArtifact;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(artifact) = PreparedArtifact::from_bytes(data) else {
        return;
    };
    let bytes = artifact
        .to_bytes()
        .expect("a decoded artifact must re-encode");
    assert_eq!(
        bytes, data,
        "the reader accepted a non-canonical artifact encoding"
    );
});
