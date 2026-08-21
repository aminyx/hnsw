//! Exact k-nearest-neighbour search by linear scan.
//!
//! This is the yardstick. An approximate index without an exact baseline to
//! measure against is a claim, not a result: it will happily return ten
//! plausible-looking vectors that are not the ten nearest, and nothing in the
//! output distinguishes that from success. Every recall number in this
//! repository is `|approximate ∩ exact| / k` against this scan.
//!
//! It is also a perfectly good index below a few thousand vectors, where a
//! linear scan over contiguous memory beats a graph walk that has to chase
//! pointers.

use crate::distance::{Distance, Metric};
use crate::error::{Error, Result};
use crate::index::Neighbor;

/// A flat, exhaustively searched vector store.
#[derive(Clone, Debug)]
pub struct BruteForce<D: Distance = Metric> {
    dims: usize,
    dist: D,
    vectors: Vec<f32>,
    ids: Vec<u64>,
}

impl BruteForce<Metric> {
    /// A new empty store over `dims`-dimensional vectors.
    pub fn new(dims: usize, metric: Metric) -> Result<Self> {
        BruteForce::with_distance(dims, metric)
    }
}

impl<D: Distance> BruteForce<D> {
    /// A new empty store with a custom distance function.
    pub fn with_distance(dims: usize, dist: D) -> Result<Self> {
        if dims == 0 {
            return Err(Error::InvalidParams("dims must be at least 1"));
        }
        Ok(BruteForce {
            dims,
            dist,
            vectors: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Append a vector. Ids are not checked for uniqueness — this is a
    /// measuring stick, not a database.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: vector.len(),
            });
        }
        let start = self.vectors.len();
        self.vectors.extend_from_slice(vector);
        self.dist.preprocess(&mut self.vectors[start..]);
        self.ids.push(id);
        Ok(())
    }

    /// Append `n * dims` floats as `n` vectors with consecutive ids.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if the buffer is not a whole number of
    /// vectors, and [`Error::InvalidParams`] if the id range runs past
    /// `u64::MAX`. As in [`Hnsw::insert_batch`], a failure part of the way
    /// through leaves the vectors before it appended.
    ///
    /// [`Hnsw::insert_batch`]: crate::Hnsw::insert_batch
    pub fn insert_batch(&mut self, first_id: u64, vectors: &[f32]) -> Result<usize> {
        if vectors.len() % self.dims != 0 {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: vectors.len() % self.dims,
            });
        }
        let n = vectors.len() / self.dims;
        for i in 0..n {
            let id = first_id
                .checked_add(i as u64)
                .ok_or(Error::InvalidParams("batch ids run past u64::MAX"))?;
            self.insert(id, &vectors[i * self.dims..(i + 1) * self.dims])?;
        }
        Ok(n)
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Vector width.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The exact `k` nearest neighbours, ascending by distance.
    ///
    /// Ties break on ascending id so the result is deterministic; without
    /// that, recall against a dataset with duplicate points would fluctuate
    /// for reasons that have nothing to do with the index.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: query.len(),
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut q = query.to_vec();
        self.dist.preprocess(&mut q);

        let mut scored: Vec<Neighbor> = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, &id)| Neighbor {
                id,
                distance: self
                    .dist
                    .distance_preprocessed(&q, &self.vectors[i * self.dims..(i + 1) * self.dims]),
            })
            .collect();

        // Select the k best, then order only that prefix. Sorting all n
        // candidates and throwing away the tail is what a strawman baseline
        // does; an exact scan that a real system would ship costs
        // O(n) selection plus O(k log k), and dividing the index's throughput
        // by anything weaker overstates the speedup.
        let cmp =
            |a: &Neighbor, b: &Neighbor| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id));
        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k, cmp);
            scored.truncate(k);
        }
        scored.sort_unstable_by(cmp);
        Ok(scored)
    }
}

/// Ground-truth id lists for a batch of queries.
///
/// Returns one `Vec<u64>` per query, each the exact top `k` ids in order.
pub fn ground_truth<D: Distance>(
    base: &BruteForce<D>,
    queries: &[f32],
    k: usize,
) -> Result<Vec<Vec<u64>>> {
    let dims = base.dims();
    if queries.len() % dims != 0 {
        return Err(Error::DimensionMismatch {
            expected: dims,
            found: queries.len() % dims,
        });
    }
    (0..queries.len() / dims)
        .map(|i| {
            base.search(&queries[i * dims..(i + 1) * dims], k)
                .map(|hits| hits.into_iter().map(|n| n.id).collect())
        })
        .collect()
}

/// Fraction of the exact top-`k` that `found` recovered.
///
/// This is recall@k for one query: `|found ∩ truth| / |truth|`. Order within
/// the result is not scored, which is the standard definition — a vector
/// database that returns the right ten in the wrong order has still done its
/// job, and the caller re-ranks anyway.
#[must_use]
pub fn recall_at_k(found: &[Neighbor], truth: &[u64]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let hits = truth
        .iter()
        .filter(|id| found.iter().any(|n| n.id == **id))
        .count();
    hits as f64 / truth.len() as f64
}
