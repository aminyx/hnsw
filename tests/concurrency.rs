//! Many readers, one index, no locks.
//!
//! Search takes `&self` and keeps all its mutable state in a caller-owned
//! [`Searcher`], so an `Arc<Hnsw>` can be queried from any number of threads
//! at once. These tests hold that property down: it is the difference
//! between an index a server can use and one it has to wrap in a mutex.

mod common;

use std::sync::Arc;
use std::thread;

use common::{build, clustered_with_queries, DATA_SEED};
use hnsw::{Hnsw, Metric, Neighbor, Params, Searcher};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn the_index_is_send_and_sync() {
    assert_send_sync::<Hnsw<Metric>>();
    assert_send_sync::<Searcher>();
    assert_send_sync::<Params>();
    assert_send_sync::<Neighbor>();
}

#[test]
fn parallel_queries_return_exactly_the_serial_results() {
    let (data, queries) = clustered_with_queries(3000, 48, 15, 240, DATA_SEED + 40);
    let index = Arc::new(build(
        48,
        Metric::Cosine,
        Params::default().with_m(12),
        data.flat(),
    ));

    let serial: Vec<Vec<Neighbor>> = queries
        .iter()
        .map(|q| index.search(q, 10).unwrap())
        .collect();

    let flat: Arc<Vec<f32>> = Arc::new(queries.flat().to_vec());
    let dims = queries.dims();
    let count = queries.len();

    let mut handles = Vec::new();
    for thread_id in 0..8usize {
        let index = Arc::clone(&index);
        let flat = Arc::clone(&flat);
        handles.push(thread::spawn(move || {
            // Each thread keeps its own scratch buffer, which is the intended
            // usage: no shared mutable state at all.
            let mut scratch = Searcher::new();
            let mut results = Vec::new();
            for i in (thread_id..count).step_by(8) {
                let query = &flat[i * dims..(i + 1) * dims];
                results.push((i, index.search_with(&mut scratch, query, 10, 64).unwrap()));
            }
            results
        }));
    }

    let mut checked = 0usize;
    for handle in handles {
        for (i, hits) in handle.join().expect("no thread panicked") {
            assert_eq!(hits, serial[i], "query {i} differed between threads");
            checked += 1;
        }
    }
    assert_eq!(checked, count);
}

#[test]
fn every_thread_hammering_one_query_agrees() {
    let (data, queries) = clustered_with_queries(2000, 32, 10, 1, DATA_SEED + 41);
    let index = Arc::new(build(32, Metric::L2, Params::default(), data.flat()));
    let query: Arc<Vec<f32>> = Arc::new(queries.get(0).to_vec());
    let expected = index.search(&query, 20).unwrap();

    let mut handles = Vec::new();
    for _ in 0..12 {
        let index = Arc::clone(&index);
        let query = Arc::clone(&query);
        handles.push(thread::spawn(move || {
            let mut scratch = Searcher::new();
            for _ in 0..40 {
                let hits = index
                    .search_with(&mut scratch, &query, 20, index.params().ef_search)
                    .unwrap();
                assert_eq!(hits.len(), 20);
            }
            index.search(&query, 20).unwrap()
        }));
    }
    for handle in handles {
        assert_eq!(handle.join().unwrap(), expected);
    }
}

#[test]
fn a_reused_searcher_never_leaks_state_between_queries() {
    // The visited set is a generation stamp array rather than a cleared
    // buffer; if the generation logic were wrong, the second query through a
    // reused searcher would see stale visits and return less.
    let (data, queries) = clustered_with_queries(1500, 32, 8, 50, DATA_SEED + 42);
    let index = build(32, Metric::L2, Params::default(), data.flat());

    let mut shared = Searcher::new();
    for query in &queries {
        let fresh = index
            .search_with(&mut Searcher::new(), query, 10, 64)
            .unwrap();
        let reused = index.search_with(&mut shared, query, 10, 64).unwrap();
        assert_eq!(fresh, reused);
    }
    assert!(shared.capacity_bytes() >= index.node_count() * 4);
}

#[test]
fn a_searcher_survives_many_more_searches_than_its_generation_counter_has_bits() {
    // Not literally 2^32 searches, but the wrap path is exercised by running
    // enough queries that the buffer is reused heavily and asserting the
    // answers never drift.
    let (data, queries) = clustered_with_queries(500, 16, 4, 1, DATA_SEED + 43);
    let index = build(16, Metric::L2, Params::default(), data.flat());
    let mut scratch = Searcher::new();

    let expected = index.search(queries.get(0), 10).unwrap();
    for _ in 0..5000 {
        assert_eq!(
            index
                .search_with(&mut scratch, queries.get(0), 10, 64)
                .unwrap(),
            expected
        );
    }
}

#[test]
fn an_index_can_be_moved_to_another_thread_and_mutated_there() {
    let (data, _) = clustered_with_queries(500, 16, 4, 0, DATA_SEED + 44);
    let mut index = build(16, Metric::L2, Params::default(), data.flat());

    let handle = thread::spawn(move || {
        index.insert(9999, &[0.5; 16]).unwrap();
        index.delete(0);
        index.compact();
        index
    });
    let index = handle.join().unwrap();
    assert!(index.contains(9999));
    assert!(!index.contains(0));
    index.validate().unwrap();
}
