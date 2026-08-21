//! Deterministic synthetic datasets for benchmarking and testing.
//!
//! Uniform random vectors are the worst possible benchmark for an ANN index.
//! In high dimensions they are all nearly equidistant from each other — the
//! concentration of measure that makes nearest-neighbour search hard in the
//! first place — so recall numbers on uniform data say almost nothing about
//! behaviour on real embeddings, which are strongly clustered and live on a
//! low-dimensional manifold inside the ambient space.
//!
//! [`Dataset::clustered`] is therefore the default here: isotropic Gaussian
//! blobs around random centroids. It is still a toy, but it has the property
//! that matters — dense regions separated by sparse ones — which is exactly
//! the structure the neighbour-selection heuristic exists to navigate.

use crate::rng::Rng;

/// A flat block of `len * dims` floats.
#[derive(Clone, Debug)]
pub struct Dataset {
    dims: usize,
    vectors: Vec<f32>,
}

impl Dataset {
    /// Wrap an existing flat buffer.
    ///
    /// # Panics
    ///
    /// If `vectors.len()` is not a multiple of `dims`, or `dims` is zero.
    #[must_use]
    pub fn from_flat(dims: usize, vectors: Vec<f32>) -> Self {
        assert!(dims > 0, "dims must be positive");
        assert!(
            vectors.len() % dims == 0,
            "buffer length {} is not a multiple of dims {dims}",
            vectors.len()
        );
        Dataset { dims, vectors }
    }

    /// `n` points drawn from `clusters` isotropic Gaussian blobs.
    ///
    /// Centroids are uniform in `[-1, 1]^dims`; each point is assigned to a
    /// centroid **at random** and displaced by `N(0, spread)` per component.
    ///
    /// The random assignment is deliberate, and was a bug here before it was
    /// a feature. Assigning round-robin gives perfectly balanced clusters,
    /// but it also makes `index % clusters` *equal* the cluster id — so any
    /// test or benchmark that touches "every fourth vector" silently
    /// operates on whole clusters at a time instead of a random sample.
    /// Random assignment costs balanced sizes, which nothing needs, and buys
    /// independence between a vector's position in the file and its position
    /// in space, which everything downstream assumes.
    #[must_use]
    pub fn clustered(n: usize, dims: usize, clusters: usize, spread: f32, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let clusters = clusters.max(1);

        let mut centroids = vec![0.0f32; clusters * dims];
        for value in &mut centroids {
            *value = rng.range(-1.0, 1.0);
        }

        let mut vectors = vec![0.0f32; n * dims];
        for i in 0..n {
            let c = rng.below(clusters) * dims;
            for d in 0..dims {
                vectors[i * dims + d] = centroids[c + d] + spread * rng.gaussian() as f32;
            }
        }
        Dataset { dims, vectors }
    }

    /// `n` points uniform in `[-1, 1]^dims`. The hard case, kept for contrast.
    #[must_use]
    pub fn uniform(n: usize, dims: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut vectors = vec![0.0f32; n * dims];
        for value in &mut vectors {
            *value = rng.range(-1.0, 1.0);
        }
        Dataset { dims, vectors }
    }

    /// Vector width.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len() / self.dims
    }

    /// Whether there are no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The whole buffer.
    #[must_use]
    pub fn flat(&self) -> &[f32] {
        &self.vectors
    }

    /// One vector.
    ///
    /// # Panics
    ///
    /// If `i` is out of range.
    #[must_use]
    pub fn get(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dims..(i + 1) * self.dims]
    }

    /// Iterate over the vectors.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> {
        self.vectors.chunks_exact(self.dims)
    }

    /// Split off the last `n` vectors as a held-out query set.
    ///
    /// Queries must not be members of the indexed set, or recall is measured
    /// against a task where the answer is trivially reachable: a point is its
    /// own nearest neighbour at distance zero and the greedy walk finds it
    /// almost by accident. Held-out queries measure what actually matters.
    #[must_use]
    pub fn split_off_queries(&mut self, n: usize) -> Dataset {
        let n = n.min(self.len());
        let at = (self.len() - n) * self.dims;
        let tail = self.vectors.split_off(at);
        Dataset {
            dims: self.dims,
            vectors: tail,
        }
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a [f32];
    type IntoIter = std::slice::ChunksExact<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.vectors.chunks_exact(self.dims)
    }
}
