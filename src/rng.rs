//! A small deterministic PRNG, written here so that an index built from a
//! given seed is byte-for-byte reproducible on every platform.
//!
//! HNSW is a *randomised* data structure: which layer a node lands on is a
//! coin flip, and that single choice cascades into the entire graph topology.
//! Pulling in `rand` would make reproducibility hostage to a dependency's
//! version bumps and to `getrandom`'s platform quirks (which do not exist on
//! `wasm32-unknown-unknown` without extra features). Sixty lines of
//! xoshiro256++ removes the problem entirely.
//!
//! - [`Rng::new`] expands a `u64` seed with SplitMix64, the standard way to
//!   avoid the poor early output xoshiro gives for near-zero state.
//! - [`Rng::next_u64`] is xoshiro256++, which passes BigCrush and is roughly
//!   one nanosecond per draw.
//! - The full state is saved into persisted indexes, so inserts continue on
//!   exactly the stream they would have had without the save/load round trip.

/// SplitMix64: the seeder. Also useful standalone as a fast integer hash.
#[must_use]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Seed the generator. Every seed, including `0`, yields a full-quality
    /// stream because SplitMix64 expands it first.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let state = [
            splitmix64(&mut s),
            splitmix64(&mut s),
            splitmix64(&mut s),
            splitmix64(&mut s),
        ];
        Rng { state }
    }

    /// Rebuild a generator from a previously captured state.
    ///
    /// An all-zero state is invalid for xoshiro (it is a fixed point), so it
    /// is replaced by the state for seed `0`.
    #[must_use]
    pub fn from_state(state: [u64; 4]) -> Self {
        if state == [0; 4] {
            Rng::new(0)
        } else {
            Rng { state }
        }
    }

    /// The current internal state, for persistence.
    #[must_use]
    pub fn state(&self) -> [u64; 4] {
        self.state
    }

    /// Next raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform double in `[0, 1)`, using the top 53 bits (the full `f64`
    /// mantissa) so every representable value in the range is reachable.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform double in `(0, 1]`.
    ///
    /// The level formula takes `ln(u)`, which is `-inf` at `u == 0`, so the
    /// open end has to be at zero rather than at one.
    pub fn open_unit(&mut self) -> f64 {
        let u = self.next_f64();
        if u <= 0.0 {
            1.0
        } else {
            u
        }
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f64() as f32
    }

    /// Uniform integer in `[0, n)`. Returns `0` when `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal draw via Box-Muller.
    ///
    /// The transform produces two independent normals per call and this
    /// deliberately discards the second one. Caching it would put a value
    /// outside the xoshiro state, which would then have to be persisted too
    /// or reproducibility across save/load would silently break. One extra
    /// `sin`/`ln` per draw is a cheap price for that guarantee.
    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.open_unit();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Rng::new(0)
    }
}

/// Highest layer index this implementation will ever hand out.
///
/// With the default `mL` the probability of drawing level 32 is about
/// `1e-39`, so the cap never binds in practice. It exists so that a
/// pathological `mL` (or a hostile one from a corrupt file) cannot make the
/// per-node layer vector allocate unboundedly.
pub const MAX_LEVEL: usize = 32;

/// Draw a node's top layer from the geometric distribution HNSW prescribes.
///
/// The paper's formula is
///
/// ```text
/// l = floor(-ln(unif(0, 1]) * mL)
/// ```
///
/// which gives `P(l >= i) = exp(-i / mL)`: each layer up holds a constant
/// fraction of the one below it, so the layer populations decay geometrically
/// and the expected number of layers is `O(log n)`. Setting `mL = 1 / ln(M)`
/// makes that fraction `1/M`, which is the value the paper shows minimises
/// the expected number of distance computations — at that ratio the overlap
/// between consecutive layers is just enough to keep the descent from getting
/// lost, and no more.
#[must_use]
pub fn assign_level(rng: &mut Rng, ml: f64) -> usize {
    if !ml.is_finite() || ml <= 0.0 {
        return 0;
    }
    let level = -rng.open_unit().ln() * ml;
    if !level.is_finite() || level <= 0.0 {
        return 0;
    }
    (level as usize).min(MAX_LEVEL)
}

/// The `mL` the paper recommends for a given `M`: `1 / ln(M)`.
#[must_use]
pub fn default_ml(m: usize) -> f64 {
    let m = m.max(2) as f64;
    1.0 / m.ln()
}
