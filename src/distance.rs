//! Distance kernels and the [`Distance`] trait the index is generic over.
//!
//! # Why the loops look like this
//!
//! There is no `std::arch` in this crate and no `unsafe` SIMD. Instead every
//! kernel accumulates into [`LANES`] independent accumulators and only folds
//! them together at the end.
//!
//! That shape matters because IEEE-754 addition is *not* associative:
//! `(a + b) + c` and `a + (b + c)` can give different answers. A plain
//! sequential sum therefore forms a serial dependency chain that LLVM is not
//! allowed to reorder — it cannot vectorise it without `-ffast-math`-style
//! permission, which Rust does not grant. Splitting the sum into eight
//! independent chains makes the vectorisation *legal* rather than merely
//! desirable: each lane is its own reduction, so the compiler can keep eight
//! partial sums in one SIMD register and issue one `mulps`/`addps` per
//! iteration. The final fold is the only place where lane order is fixed, and
//! it is fixed deterministically, so results are bit-reproducible.
//!
//! The cost of this is one extra fold per call and a slightly longer tail
//! loop. The benefit is roughly a 3-4x speedup over the naive loop on x86-64
//! with SSE2, measured separately against a naive reference kernel rather than
//! by any target in this repository: `hnswctl bench` sweeps whole-index QPS,
//! and nothing here times the two kernels against each other.
//!
//! No architecture-specific intrinsic is involved, so the same source compiles
//! unchanged to `wasm32-unknown-unknown`. Built with
//! `-C target-feature=+simd128` it emits Wasm SIMD instructions; the module
//! published in `docs/` is built without that flag and contains none of them.
//! Wasm SIMD is a compile-time target feature, not something a runtime
//! negotiates.

use core::fmt;

/// Number of independent accumulators the kernels fan out into.
///
/// Eight `f32` lanes is one 256-bit AVX register, or two 128-bit SSE/NEON
/// registers, and divides evenly into every common embedding width
/// (64, 128, 384, 768, 1536).
pub const LANES: usize = 8;

/// Inner product of two vectors.
///
/// Only the first `min(a.len(), b.len())` elements participate; callers in
/// this crate always pass equal lengths, checked one level up.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);

    let mut acc = [0.0f32; LANES];
    let mut ac = a.chunks_exact(LANES);
    let mut bc = b.chunks_exact(LANES);
    for (x, y) in ac.by_ref().zip(bc.by_ref()) {
        for ((slot, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            *slot += xi * yi;
        }
    }

    let mut total = fold(&acc);
    for (xi, yi) in ac.remainder().iter().zip(bc.remainder()) {
        total += xi * yi;
    }
    total
}

/// Squared Euclidean distance, `sum((a_i - b_i)^2)`.
///
/// The square root is deliberately omitted: it is monotone, so it changes no
/// ranking, and skipping it removes a `sqrtps` from the hottest loop in the
/// index. FAISS makes the same choice for the same reason.
#[must_use]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);

    let mut acc = [0.0f32; LANES];
    let mut ac = a.chunks_exact(LANES);
    let mut bc = b.chunks_exact(LANES);
    for (x, y) in ac.by_ref().zip(bc.by_ref()) {
        for ((slot, xi), yi) in acc.iter_mut().zip(x).zip(y) {
            let d = xi - yi;
            *slot += d * d;
        }
    }

    let mut total = fold(&acc);
    for (xi, yi) in ac.remainder().iter().zip(bc.remainder()) {
        let d = xi - yi;
        total += d * d;
    }
    total
}

/// Euclidean length of a vector.
///
/// Computed the scale-safe way: the largest magnitude is factored out first,
/// so the sum of squares is formed over values in `[0, 1]` and cannot
/// overflow or underflow. Squaring `1e20` directly gives `+inf` in `f32` and
/// squaring `1e-25` gives zero, and either way the caller is handed a norm
/// that is useless — [`normalize`] then refuses, [`Cosine`] stores the vector
/// unscaled, and its distances leave the documented `[0, 2]` range. One
/// division per component buys the whole f32 range instead.
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    let mut max = 0.0f32;
    for &x in v {
        let a = x.abs();
        if a > max {
            max = a;
        }
    }
    if max == 0.0 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for &x in v {
        let scaled = x / max;
        acc += scaled * scaled;
    }
    max * acc.sqrt()
}

/// Scale `v` to unit length in place.
///
/// Returns `false` and leaves `v` untouched when its norm is zero or
/// non-finite — the zero vector has no direction, so there is nothing
/// sensible to normalise it to. Cosine distance against it is defined as
/// `1.0` (maximally far) rather than `NaN`.
pub fn normalize(v: &mut [f32]) -> bool {
    let n = norm(v);
    if n <= 0.0 || !n.is_finite() {
        return false;
    }
    let inv = 1.0 / n;
    for x in v.iter_mut() {
        *x *= inv;
    }
    true
}

/// Deterministic left-to-right fold of the lane accumulators.
#[inline]
fn fold(acc: &[f32; LANES]) -> f32 {
    let mut total = 0.0;
    for &x in acc {
        total += x;
    }
    total
}

/// A dissimilarity function: smaller is nearer.
///
/// The index is generic over this trait, so a caller can plug in a domain
/// metric (Hamming over binary codes, Manhattan, learned distances) without
/// touching the graph code. [`Metric`] is the built-in implementation and the
/// only one that can be persisted, because the file format stores a metric
/// tag rather than arbitrary user code.
pub trait Distance: Clone + Send + Sync {
    /// Dissimilarity between two raw vectors.
    ///
    /// The contract is that `distance(a, b)` equals `distance(b, a)`, and
    /// that `distance(a, a)` is the smallest value the implementation ever
    /// returns for `a` — `0.0` for [`Cosine`] and [`Euclidean`].
    ///
    /// Two implementations in this crate deviate, and both say so where the
    /// deviation is: [`Cosine`] returns `1.0` for the zero vector, the
    /// fallback [`Distance::preprocess`] describes, because a vector with no
    /// direction has no cosine; and [`Metric::Dot`] is not a metric at all,
    /// so a longer vector in the same direction beats `a` against itself. An
    /// implementation may do the same, as long as it documents the fallback.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Stable short name, used by the CLI and in log output.
    fn name(&self) -> &'static str;

    /// Optional per-vector preparation applied once at insert time and once
    /// per query. Cosine uses it to normalise; the others do nothing.
    fn preprocess(&self, v: &mut [f32]) {
        let _ = v;
    }

    /// Distance between two vectors that have *both* already been through
    /// [`Distance::preprocess`].
    ///
    /// Defaults to [`Distance::distance`]. Cosine overrides it to skip the two
    /// norm computations, which is the entire point of normalising at insert
    /// time: it turns a cosine into a bare dot product.
    fn distance_preprocessed(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance(a, b)
    }
}

/// The three built-in metrics.
///
/// This enum dispatches to the unit-struct implementations below. It exists
/// because the on-disk format needs a small integer tag for the metric, and
/// because the CLI needs to pick one from a string at runtime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// `1 - cos(a, b)`, in `[0, 2]`. Scale-invariant; the usual choice for
    /// text embeddings, which encode meaning in direction rather than length.
    #[default]
    Cosine,
    /// `-dot(a, b)`. Maximum inner product search, negated so that smaller is
    /// still better.
    ///
    /// Note this is **not** a metric: it violates the triangle inequality and
    /// a vector is not even closest to itself (a longer vector in the same
    /// direction beats it). HNSW's proofs assume a metric space, so recall
    /// under `Dot` is empirically good but not guaranteed the way it is for
    /// `Cosine` and `L2`. Normalise first if you can.
    Dot,
    /// Squared Euclidean distance. Monotone with true Euclidean distance, so
    /// rankings match; take `sqrt` yourself if you need the real magnitude.
    L2,
}

/// Cosine distance as a standalone unit struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cosine;

/// Negated inner product as a standalone unit struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerProduct;

/// Squared Euclidean distance as a standalone unit struct.
#[derive(Clone, Copy, Debug, Default)]
pub struct Euclidean;

impl Distance for Cosine {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let denom = norm(a) * norm(b);
        if denom <= 0.0 || !denom.is_finite() {
            return 1.0;
        }
        1.0 - dot(a, b) / denom
    }

    fn name(&self) -> &'static str {
        "cosine"
    }

    fn preprocess(&self, v: &mut [f32]) {
        // The return value is deliberately dropped. Now that `norm` is
        // scale-safe the only vector `normalize` refuses is one with no
        // direction to scale, and `distance` already defines that case as
        // 1.0 — so there is nothing to repair and nothing to report.
        normalize(v);
    }

    fn distance_preprocessed(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - dot(a, b)
    }
}

impl Distance for InnerProduct {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        -dot(a, b)
    }

    fn name(&self) -> &'static str {
        "dot"
    }
}

impl Distance for Euclidean {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        squared_l2(a, b)
    }

    fn name(&self) -> &'static str {
        "l2"
    }
}

impl Metric {
    /// Every metric, in tag order. Handy for tests and for CLI help text.
    pub const ALL: [Metric; 3] = [Metric::Cosine, Metric::Dot, Metric::L2];

    /// Stable one-byte tag written into persisted index files.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            Metric::Cosine => 0,
            Metric::Dot => 1,
            Metric::L2 => 2,
        }
    }

    /// Inverse of [`Metric::tag`].
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Metric> {
        match tag {
            0 => Some(Metric::Cosine),
            1 => Some(Metric::Dot),
            2 => Some(Metric::L2),
            _ => None,
        }
    }

    /// Parse a metric from its CLI spelling (`cosine`, `dot`, `l2`).
    #[must_use]
    pub fn parse(text: &str) -> Option<Metric> {
        match text.trim().to_ascii_lowercase().as_str() {
            "cosine" | "cos" | "angular" => Some(Metric::Cosine),
            "dot" | "ip" | "inner" => Some(Metric::Dot),
            "l2" | "euclidean" | "euclid" => Some(Metric::L2),
            _ => None,
        }
    }
}

impl Distance for Metric {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => Cosine.distance(a, b),
            Metric::Dot => InnerProduct.distance(a, b),
            Metric::L2 => Euclidean.distance(a, b),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Metric::Cosine => Cosine.name(),
            Metric::Dot => InnerProduct.name(),
            Metric::L2 => Euclidean.name(),
        }
    }

    fn preprocess(&self, v: &mut [f32]) {
        if matches!(self, Metric::Cosine) {
            Cosine.preprocess(v);
        }
    }

    fn distance_preprocessed(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::Cosine => Cosine.distance_preprocessed(a, b),
            Metric::Dot => InnerProduct.distance(a, b),
            Metric::L2 => Euclidean.distance(a, b),
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
