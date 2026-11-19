# Quantization format fixtures

These files are hand-checkable `fn3-quant-format-v1` objects. They contain no
model weights: every blob is tiny and every expected value is listed in
`manifest.json`.

The authoritative prose is `docs/quantization/fn3-quant-format-v1.md`. The
independent Python encoder/decoder is `tools/quant/fn3_quant.py`; the Rust
reader is `src/fn3/quant/`. `tests/fn3_quant_format.rs` requires the two
implementations to agree with these bytes.

## Layout

```text
fixtures/quant/
  README.md
  manifest.json
  v1/*.fn3q                 valid blobs
  malformed/*.fn3q          one blob per typed failure
```

`manifest.json` records, for every valid fixture, the byte length, SHA-256,
descriptor fields, scales, zero-points, a hex dump of the packed code bytes, and
the complete expected dequantized tensor. For every malformed fixture it records
the expected `QuantError::kind()` string, which is shared with the Python
reference.

## Hand checks

`v1/basic.fn3q` has one tensor `basic`, shape `[2, 64]`, one group per row,
scale `[0.5, 1.0]`, zero-point `[0, -4]`.

* Row 0 codes are `-8, -7, ..., 7` at scale `0.5`, so byte 0 is `0xF8` and the
  first values are `-4.0, -3.5, -3.0, ...`; the last is `3.5`.
* Row 1 codes are all `-4`, zero-point `-4`, scale `1.0`, so every value is
  `(-4 - -4) * 1.0 = 0.0`.

`v1/tail_k70.fn3q` has `tail_k70`, shape `[1, 70]`, so the groups are
`(0, 64)` and `(64, 6)` and each row is padded to 80 bytes.

* Group 0: codes `-8..7`, scale `0.25`, zero-point `0`.
* Group 1: codes `10, 20, 30, 40, 50, 60`, scale `0.125`, zero-point `30`, so
  the six values are `-2.5, -1.25, 0.0, 1.25, 2.5, 3.75`.
* Bytes 70..79 of the row are zero padding.

`v1/zero_point.fn3q` has `zero_point`, shape `[1, 64]`, codes `-32..31`, scale
`0.5`, zero-point `-10`, so the first value is `(-32 - -10) * 0.5 = -11.0` and
the last is `(31 - -10) * 0.5 = 20.5`.

`v1/two_tensors.fn3q` has two tensors in one container:

* `short_k8`, shape `[1, 8]`, one group shorter than 64, codes `1..8`,
  zero-point `4`, scale `1.0`, values `-3.0..4.0`.
* `square`, shape `[1, 64]`, codes `-8..7`, zero-point `-8`, scale `0.25`,
  values `0.0, 0.25, ..., 3.75`.

## Malformed matrix

Every file under `malformed/` is rejected with the `expected_error` recorded in
the manifest. The set covers header framing, offsets, descriptor semantics,
group directory overlap/gap/length, truncation and trailing bytes, both
checksums, scale zero/NaN/negative, out-of-range zero-point, and non-zero
padding. The Rust test asserts the exact typed error for each file; the Python
verifier asserts the same string from the independent decoder.

## Verification

```bash
python3 tools/quant/verify_fixtures.py --root fixtures/quant --regenerate
cargo test --test fn3_quant_format
```

`--regenerate` rebuilds every blob from `tools/quant/make_fixtures.py` into a
temporary directory and requires a byte-for-byte match. The committed bytes are
generator output; do not edit them by hand.
