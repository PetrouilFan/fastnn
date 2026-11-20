//! SHA-256, for reporting the identity of a model file.
//!
//! Both the CLI (`prepare`, `inspect`, `run`) and the Python binding report a
//! digest a caller can compare against `sha256sum`, a download page or a lock
//! file. The prepared-model format already carries a CRC-32, but that is a
//! *payload* checksum with a different coverage and a different purpose (it
//! tells the reader that the bytes it decoded are the bytes that were written).
//!
//! The crate has no hashing dependency and this module adds none, so the hash
//! lives here: it is the standard FIPS 180-4 construction, streamable so a
//! 28 MB (or 500 MB) file is never read twice, and it is pinned by the
//! published check values in the unit tests below. It is a fingerprint, not a
//! security control: nothing in FastNN authenticates a model.
//!
//! This module is deliberately dependency-free and pure so it can be read and
//! checked in one sitting; it is not a second implementation of any FastNN
//! semantics.

/// Round constants: the first 32 bits of the fractional parts of the cube
/// roots of the first 64 primes (FIPS 180-4 §4.2.2).
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// Streaming SHA-256 state.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffered: usize,
    bytes: u64,
}

impl Sha256 {
    /// Start a new digest.
    pub fn new() -> Sha256 {
        Sha256 {
            // FIPS 180-4 §5.3.3 initial hash value.
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0u8; 64],
            buffered: 0,
            bytes: 0,
        }
    }

    /// Absorb more bytes.
    pub fn update(&mut self, mut bytes: &[u8]) {
        self.bytes = self.bytes.wrapping_add(bytes.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(bytes.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&bytes[..take]);
            self.buffered += take;
            bytes = &bytes[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
        while bytes.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&bytes[..64]);
            self.compress(&block);
            bytes = &bytes[64..];
        }
        if !bytes.is_empty() {
            self.buffer[..bytes.len()].copy_from_slice(bytes);
            self.buffered = bytes.len();
        }
    }

    /// Finish the digest and return the 32 digest bytes.
    pub fn finish(mut self) -> [u8; 32] {
        let bit_len = self.bytes.wrapping_mul(8);
        // Padding: one 0x80 byte, zeros, then the 64-bit big-endian bit length.
        let mut tail = [0u8; 72];
        tail[0] = 0x80;
        let pad = if self.buffered < 56 {
            56 - self.buffered
        } else {
            120 - self.buffered
        };
        tail[pad..pad + 8].copy_from_slice(&bit_len.to_be_bytes());
        let length = pad + 8;
        self.update_exact(&tail[..length]);

        let mut digest = [0u8; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state.iter()) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }

    /// Absorb padding without counting it towards the message length.
    fn update_exact(&mut self, bytes: &[u8]) {
        let counted = self.bytes;
        self.update(bytes);
        self.bytes = counted;
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        for (slot, value) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
}

impl Default for Sha256 {
    fn default() -> Sha256 {
        Sha256::new()
    }
}

/// Raw SHA-256 digest of `bytes`.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finish()
}

/// Digest of a whole byte slice.
pub fn digest(bytes: &[u8]) -> [u8; 32] {
    sha256(bytes)
}

/// Lowercase hexadecimal rendering of a digest.
pub fn render(digest: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push_str(&format!("{byte:02x}"));
    }
    text
}

/// Lowercase hexadecimal digest of a whole byte slice.
pub fn hex(bytes: &[u8]) -> String {
    render(&digest(bytes))
}

/// Lowercase hexadecimal SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS 180-4 / NIST CAVP published check values.
    #[test]
    fn published_check_values() {
        assert_eq!(
            hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// The CLI's one-shot entry point agrees with the published `abc` value.
    #[test]
    fn one_shot_hex_matches_the_abc_check_value() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn streaming_matches_one_shot_at_every_split() {
        let message: Vec<u8> = (0..=255u8).chain((0..=255u8).rev()).collect();
        let expected = digest(&message);
        for split in [0usize, 1, 55, 56, 57, 63, 64, 65, 127, 128, 500] {
            let mut hasher = Sha256::new();
            hasher.update(&message[..split.min(message.len())]);
            hasher.update(&message[split.min(message.len())..]);
            assert_eq!(hasher.finish(), expected, "split at {split}");
        }
    }

    #[test]
    fn a_million_a_bytes() {
        let mut hasher = Sha256::new();
        let block = [b'a'; 1000];
        for _ in 0..1000 {
            hasher.update(&block);
        }
        assert_eq!(
            render(&hasher.finish()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn block_boundaries_are_handled() {
        for length in [55usize, 56, 63, 64, 65, 119, 120, 128] {
            let bytes = vec![0x61; length];
            assert_eq!(sha256_hex(&bytes).len(), 64);
        }
        // A 64-byte message exercises a full block plus the padding block.
        assert_eq!(
            sha256_hex(&[0u8; 64]),
            "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"
        );
    }
}
