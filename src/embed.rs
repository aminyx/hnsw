//! A toy text embedding: hashed bag of character n-grams.
//!
//! # These are not semantic embeddings
//!
//! Nothing here understands language. A word is chopped into overlapping
//! character n-grams, each n-gram is hashed to a signed bucket, the buckets
//! are summed and the result is normalised. Two strings end up close when
//! they *share letter sequences* — so `night`, `nights`, `knight` and
//! `mighty` cluster, and `night` and `evening` do not.
//!
//! It is used by the browser demo for one reason: it needs zero model
//! weights, zero training and zero network, so the whole demo is a single
//! Wasm file of about 350 KB. It is a stand-in for the output of a real encoder, and it is
//! genuinely representative of one in the ways that matter to an ANN index —
//! the vectors are dense, unit-length, moderately high-dimensional, and
//! clustered rather than uniform. Swap in real embeddings and nothing about
//! the index changes.
//!
//! The technique itself is not a toy: hashing features into a fixed-width
//! vector (the "hashing trick") is standard, and character n-grams are what
//! fastText uses for out-of-vocabulary words. What is missing is the learned
//! part.

/// Width the browser demo uses.
pub const DEMO_DIMS: usize = 64;

/// Shortest and longest character n-gram contributed to the vector.
const NGRAM_RANGE: core::ops::RangeInclusive<usize> = 1..=3;

/// FNV-1a over bytes. Small, fast, no state, identical everywhere.
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325 ^ seed;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

/// Write the embedding of `text` into `out`, which must be non-empty.
///
/// The text is lowercased, non-alphanumeric characters are dropped, and the
/// result is wrapped in `^`/`$` boundary markers so that prefixes and
/// suffixes are distinguishable from the same letters mid-word — without
/// them, `ate` in `late` and `ate` standing alone would be identical
/// features.
///
/// Each n-gram lands in bucket `h % dims` with sign taken from a separate bit
/// of the same hash. The signs matter: with all-positive buckets, unrelated
/// n-grams that collide always reinforce each other, so collisions bias every
/// similarity upward. With random signs, collisions cancel in expectation and
/// the dot product stays an unbiased estimate of the true overlap.
///
/// The vector is L2-normalised, so cosine distance against it is a bare dot
/// product.
pub fn embed_into(text: &str, out: &mut [f32]) {
    let dims = out.len();
    out.fill(0.0);
    if dims == 0 {
        return;
    }

    let mut cleaned = String::with_capacity(text.len() + 2);
    cleaned.push('^');
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            cleaned.push(ch.to_ascii_lowercase());
        } else if ch.is_alphanumeric() {
            cleaned.extend(ch.to_lowercase());
        }
    }
    cleaned.push('$');

    let chars: Vec<char> = cleaned.chars().collect();
    let mut buffer = [0u8; 16];
    let mut features = 0u32;

    for n in NGRAM_RANGE {
        if chars.len() < n {
            continue;
        }
        for window in chars.windows(n) {
            let mut len = 0usize;
            for &ch in window {
                if len + 4 > buffer.len() {
                    break;
                }
                len += ch.encode_utf8(&mut buffer[len..]).len();
            }
            let hash = fnv1a(&buffer[..len], n as u64);
            let bucket = (hash % dims as u64) as usize;
            let sign = if hash & (1 << 33) == 0 { 1.0 } else { -1.0 };
            // Longer n-grams are rarer and more informative, so weight by n.
            out[bucket] += sign * n as f32;
            features += 1;
        }
    }

    if features == 0 {
        return;
    }
    crate::distance::normalize(out);
}

/// Allocate and return the embedding of `text`.
#[must_use]
pub fn embed(text: &str, dims: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dims];
    embed_into(text, &mut out);
    out
}

/// Embed every string into one flat `items.len() * dims` buffer.
#[must_use]
pub fn embed_all<S: AsRef<str>>(items: &[S], dims: usize) -> Vec<f32> {
    let mut flat = vec![0.0f32; items.len() * dims];
    for (i, item) in items.iter().enumerate() {
        embed_into(item.as_ref(), &mut flat[i * dims..(i + 1) * dims]);
    }
    flat
}

/// The word list the browser demo indexes, baked into the binary.
///
/// Every lowercase, three-to-eight-letter entry in the SCOWL "small"
/// American English list — about 25000 words. The whole list rather than a
/// sample, so that a word someone actually types is almost always in the
/// index and the demo answers the question they asked.
pub const DEMO_WORDS_RAW: &str = include_str!("../assets/words.txt");

/// The demo word list, split and trimmed.
#[must_use]
pub fn demo_words() -> Vec<&'static str> {
    DEMO_WORDS_RAW
        .lines()
        .map(str::trim)
        .filter(|w| !w.is_empty())
        .collect()
}
