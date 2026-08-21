//! Shared fixtures. Every test in this suite is offline and deterministic:
//! datasets come from a seeded generator, indexes from a seeded PRNG.
#![allow(dead_code)]

use hnsw::{BruteForce, Dataset, Hnsw, Metric, Params};

/// Seed used for every generated dataset, so failures reproduce exactly.
pub const DATA_SEED: u64 = 20_260_821;

/// Build an index over a flat buffer with the given parameters.
pub fn build(dims: usize, metric: Metric, params: Params, vectors: &[f32]) -> Hnsw<Metric> {
    let mut index = Hnsw::new(dims, metric, params).expect("valid parameters");
    index.insert_batch(0, vectors).expect("insertable vectors");
    index
}

/// An exact baseline over the same buffer.
pub fn exact(dims: usize, metric: Metric, vectors: &[f32]) -> BruteForce<Metric> {
    let mut base = BruteForce::new(dims, metric).expect("valid dims");
    base.insert_batch(0, vectors).expect("insertable vectors");
    base
}

/// A clustered dataset plus a held-out query set.
pub fn clustered_with_queries(
    n: usize,
    dims: usize,
    clusters: usize,
    queries: usize,
    seed: u64,
) -> (Dataset, Dataset) {
    let mut data = Dataset::clustered(n + queries, dims, clusters, 0.22, seed);
    let held_out = data.split_off_queries(queries);
    (data, held_out)
}

/// Mean recall@k of `index` over `queries` against `base`.
pub fn mean_recall(
    index: &Hnsw<Metric>,
    base: &BruteForce<Metric>,
    queries: &Dataset,
    k: usize,
    ef: usize,
) -> f64 {
    let mut scratch = hnsw::Searcher::new();
    let mut total = 0.0;
    let mut count = 0usize;
    for query in queries {
        let hits = index
            .search_with(&mut scratch, query, k, ef)
            .expect("query succeeds");
        let truth: Vec<u64> = base
            .search(query, k)
            .expect("exact search succeeds")
            .into_iter()
            .map(|n| n.id)
            .collect();
        total += hnsw::recall_at_k(&hits, &truth);
        count += 1;
    }
    total / count.max(1) as f64
}
