//! Search correctness, measured against an exhaustive scan.
//!
//! This is the file that decides whether the index is real. Everything else
//! checks that the data structure is well formed; these tests check that it
//! returns the right answers.

mod common;

use common::{build, clustered_with_queries, exact, mean_recall, DATA_SEED};
use hnsw::{BruteForce, Hnsw, Metric, Params, Searcher};

#[test]
fn recall_at_10_is_at_least_095_on_5000_vectors_in_64_dimensions() {
    // The key test. 5000 clustered Gaussians in 64 dimensions, 200 held-out
    // queries, stock parameters, k = 10. If this regresses, the index is
    // returning plausible neighbours that are not the nearest ones.
    let (data, queries) = clustered_with_queries(5000, 64, 25, 200, DATA_SEED);
    let params = Params::default()
        .with_m(16)
        .with_ef_construction(200)
        .with_ef_search(64);

    let index = build(64, Metric::Cosine, params, data.flat());
    let base = exact(64, Metric::Cosine, data.flat());

    index.validate().expect("the graph is well formed");
    assert!(
        index.is_connected(),
        "layer 0 fragmented: {:?}",
        index.components()
    );
    assert_eq!(
        index.reachable_from_entry(0),
        index.node_count(),
        "some vectors are in the index but unreachable by search"
    );

    let recall = mean_recall(&index, &base, &queries, 10, 64);
    assert!(
        recall >= 0.95,
        "recall@10 was {recall:.4}, below the 0.95 floor"
    );
}

#[test]
fn recall_climbs_monotonically_with_ef_search() {
    let (data, queries) = clustered_with_queries(3000, 48, 15, 100, DATA_SEED + 1);
    let index = build(48, Metric::L2, Params::default().with_m(12), data.flat());
    let base = exact(48, Metric::L2, data.flat());

    let mut previous = 0.0;
    let mut measured = Vec::new();
    for ef in [4usize, 8, 16, 32, 64, 128] {
        let recall = mean_recall(&index, &base, &queries, 10, ef);
        measured.push((ef, recall));
        assert!(
            recall >= previous - 0.02,
            "recall fell from {previous:.4} to {recall:.4} when ef rose to {ef}: {measured:?}"
        );
        previous = recall;
    }
    let (_, smallest) = measured[0];
    let (_, largest) = measured[measured.len() - 1];
    assert!(
        largest > smallest,
        "a 32x wider beam bought nothing: {measured:?}"
    );
    assert!(largest >= 0.98, "recall topped out at {largest:.4}");
}

#[test]
fn a_wider_beam_costs_more_distance_computations() {
    let (data, queries) = clustered_with_queries(2000, 32, 10, 20, DATA_SEED + 2);
    let index = build(32, Metric::L2, Params::default().with_m(12), data.flat());
    let mut scratch = Searcher::new();

    let mut previous = 0usize;
    for ef in [8usize, 32, 128] {
        let (_, stats) = index
            .search_stats(&mut scratch, queries.get(0), 10, ef)
            .unwrap();
        assert!(
            stats.visited > previous,
            "ef {ef} visited {} nodes, no more than the narrower beam's {previous}",
            stats.visited
        );
        assert_eq!(stats.distance_computations, stats.visited);
        assert!(stats.hops > 0);
        previous = stats.visited;
    }
}

#[test]
fn a_query_visits_a_small_fraction_of_the_index() {
    // The entire point of the data structure.
    let (data, queries) = clustered_with_queries(5000, 64, 25, 20, DATA_SEED + 3);
    let index = build(64, Metric::Cosine, Params::default(), data.flat());
    let mut scratch = Searcher::new();

    let mut visited = 0usize;
    for query in &queries {
        let (_, stats) = index.search_stats(&mut scratch, query, 10, 64).unwrap();
        visited += stats.visited;
    }
    let fraction = visited as f64 / (queries.len() * index.node_count()) as f64;
    assert!(
        fraction < 0.15,
        "queries touched {:.1}% of the index; that is barely better than a scan",
        fraction * 100.0
    );
}

#[test]
fn per_layer_statistics_describe_the_descent() {
    let (data, queries) = clustered_with_queries(4000, 32, 20, 5, DATA_SEED + 4);
    let index = build(32, Metric::L2, Params::default(), data.flat());
    let mut scratch = Searcher::new();

    let (_, stats) = index
        .search_stats(&mut scratch, queries.get(0), 10, 64)
        .unwrap();
    assert_eq!(stats.layers.len(), index.max_level() + 1);
    // Descent order: top layer first, layer 0 last.
    for pair in stats.layers.windows(2) {
        assert!(
            pair[0].level > pair[1].level,
            "layers were not in descent order"
        );
    }
    let last = stats.layers.last().unwrap();
    assert_eq!(last.level, 0);
    assert!(
        last.visited > stats.layers[0].visited,
        "layer 0 should dominate the visit count"
    );
    assert_eq!(
        stats.visited,
        stats.layers.iter().map(|l| l.visited).sum::<usize>()
    );
    assert_eq!(
        stats.hops,
        stats.layers.iter().map(|l| l.hops).sum::<usize>()
    );
}

#[test]
fn the_exact_baseline_matches_a_hand_checked_case() {
    let mut base = BruteForce::new(2, Metric::L2).unwrap();
    base.insert(100, &[0.0, 0.0]).unwrap();
    base.insert(200, &[3.0, 4.0]).unwrap(); // squared distance 25
    base.insert(300, &[1.0, 0.0]).unwrap(); // squared distance 1
    base.insert(400, &[0.0, 2.0]).unwrap(); // squared distance 4

    let hits = base.search(&[0.0, 0.0], 3).unwrap();
    assert_eq!(
        hits.iter().map(|h| h.id).collect::<Vec<_>>(),
        vec![100, 300, 400]
    );
    assert_eq!(hits[0].distance, 0.0);
    assert_eq!(hits[1].distance, 1.0);
    assert_eq!(hits[2].distance, 4.0);
}

#[test]
fn an_indexed_vector_is_its_own_nearest_neighbour() {
    let (data, _) = clustered_with_queries(1500, 32, 8, 0, DATA_SEED + 5);
    let index = build(32, Metric::L2, Params::default(), data.flat());

    let mut found = 0usize;
    for i in (0..data.len()).step_by(37) {
        let hits = index.search(data.get(i), 1).unwrap();
        assert_eq!(hits.len(), 1);
        if hits[0].id == i as u64 {
            found += 1;
        }
    }
    let checked = (0..data.len()).step_by(37).count();
    assert_eq!(
        found,
        checked,
        "{}/{checked} points failed to retrieve themselves",
        checked - found
    );
}

#[test]
fn recall_holds_up_for_every_metric() {
    for metric in Metric::ALL {
        let (data, queries) = clustered_with_queries(2500, 48, 12, 100, DATA_SEED + 6);
        let index = build(48, metric, Params::default().with_m(16), data.flat());
        let base = exact(48, metric, data.flat());

        let recall = mean_recall(&index, &base, &queries, 10, 96);
        // The inner product is not a metric, so HNSW's guarantees do not
        // apply to it and its floor is deliberately lower. See `Metric::Dot`.
        let floor = if metric == Metric::Dot { 0.85 } else { 0.95 };
        assert!(
            recall >= floor,
            "{metric}: recall@10 was {recall:.4}, floor is {floor}"
        );
    }
}

#[test]
fn search_results_are_sorted_by_distance_and_free_of_duplicates() {
    let (data, queries) = clustered_with_queries(2000, 32, 10, 25, DATA_SEED + 7);
    let index = build(32, Metric::Cosine, Params::default(), data.flat());

    for query in &queries {
        let hits = index.search(query, 20).unwrap();
        assert_eq!(hits.len(), 20);
        for pair in hits.windows(2) {
            assert!(
                pair[0].distance <= pair[1].distance,
                "results came back unsorted: {:?}",
                hits.iter().map(|h| h.distance).collect::<Vec<_>>()
            );
        }
        let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 20, "a result id appeared twice");
    }
}

#[test]
fn repeated_searches_return_identical_results() {
    let (data, queries) = clustered_with_queries(1500, 32, 8, 10, DATA_SEED + 8);
    let index = build(32, Metric::L2, Params::default(), data.flat());
    let mut scratch = Searcher::new();

    for query in &queries {
        let first = index.search_with(&mut scratch, query, 10, 50).unwrap();
        for _ in 0..5 {
            // A fresh scratch buffer must give the same answer as a reused
            // one; if it does not, the visited stamps are leaking state.
            let mut fresh = Searcher::new();
            assert_eq!(first, index.search_with(&mut fresh, query, 10, 50).unwrap());
            assert_eq!(
                first,
                index.search_with(&mut scratch, query, 10, 50).unwrap()
            );
        }
    }
}

#[test]
fn ef_search_is_raised_to_at_least_k() {
    // Asking for 50 results with a beam of 5 must still return 50, not 5.
    let (data, queries) = clustered_with_queries(1000, 16, 5, 5, DATA_SEED + 9);
    let index = build(16, Metric::L2, Params::default(), data.flat());
    let hits = index
        .search_with(&mut Searcher::new(), queries.get(0), 50, 5)
        .unwrap();
    assert_eq!(hits.len(), 50);
}

#[test]
fn a_default_ef_search_index_uses_its_configured_beam() {
    let (data, queries) = clustered_with_queries(1200, 24, 6, 5, DATA_SEED + 10);
    let mut index = build(
        24,
        Metric::L2,
        Params::default().with_ef_search(10),
        data.flat(),
    );

    let narrow = index.search(queries.get(0), 10).unwrap();
    index.set_ef_search(300);
    let wide = index.search(queries.get(0), 10).unwrap();

    let base = exact(24, Metric::L2, data.flat());
    let truth: Vec<u64> = base
        .search(queries.get(0), 10)
        .unwrap()
        .into_iter()
        .map(|n| n.id)
        .collect();
    assert!(
        hnsw::recall_at_k(&wide, &truth) >= hnsw::recall_at_k(&narrow, &truth),
        "widening ef_search made the answer worse"
    );
    assert_eq!(index.params().ef_search, 300);
}

#[test]
fn a_5000_vector_index_is_deterministic_end_to_end() {
    let (data, queries) = clustered_with_queries(2000, 32, 10, 30, DATA_SEED + 11);
    let params = Params::default().with_seed(0xC0FFEE);

    let first: Hnsw<Metric> = build(32, Metric::Cosine, params, data.flat());
    let second: Hnsw<Metric> = build(32, Metric::Cosine, params, data.flat());

    for query in &queries {
        assert_eq!(
            first.search(query, 10).unwrap(),
            second.search(query, 10).unwrap()
        );
    }
}
