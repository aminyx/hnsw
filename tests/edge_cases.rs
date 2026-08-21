//! Degenerate inputs. Every one of these has produced a panic in some
//! published ANN implementation at some point.

mod common;

use common::{build, clustered_with_queries, DATA_SEED};
use hnsw::{BruteForce, Distance, Error, Hnsw, Metric, Params, Searcher};

#[test]
fn an_empty_index_returns_nothing_rather_than_panicking() {
    let index = Hnsw::new(8, Metric::Cosine, Params::default()).unwrap();
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.entry_point(), None);
    assert_eq!(index.max_level(), 0);
    assert!(index.layer_sizes().is_empty());
    assert_eq!(index.components(), Vec::<usize>::new());
    assert!(index.is_connected());
    assert!(index.search(&[0.1; 8], 10).unwrap().is_empty());
    index.validate().expect("an empty index is well formed");
}

#[test]
fn a_single_vector_index_returns_that_vector() {
    let mut index = Hnsw::new(4, Metric::L2, Params::default()).unwrap();
    index.insert(42, &[1.0, 2.0, 3.0, 4.0]).unwrap();

    let hits = index.search(&[1.0, 2.0, 3.0, 4.0], 10).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, 42);
    assert_eq!(hits[0].distance, 0.0);
    assert_eq!(index.entry_point(), Some(0));
    index.validate().unwrap();
}

#[test]
fn asking_for_more_neighbours_than_exist_returns_all_of_them() {
    let (data, queries) = clustered_with_queries(50, 8, 3, 5, DATA_SEED + 50);
    let index = build(8, Metric::L2, Params::default(), data.flat());
    let hits = index.search(queries.get(0), 500).unwrap();
    assert_eq!(hits.len(), 50);

    let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..50u64).collect::<Vec<_>>());
}

#[test]
fn asking_for_zero_neighbours_returns_an_empty_list() {
    let (data, queries) = clustered_with_queries(100, 8, 3, 1, DATA_SEED + 51);
    let index = build(8, Metric::L2, Params::default(), data.flat());
    assert!(index.search(queries.get(0), 0).unwrap().is_empty());
}

#[test]
fn duplicate_vectors_under_different_ids_are_all_retrievable() {
    let mut index = Hnsw::new(4, Metric::L2, Params::default().with_m(4)).unwrap();
    let point = [0.25f32, -0.5, 0.75, 1.0];
    for id in 0..40u64 {
        index.insert(id, &point).unwrap();
    }
    // Add some noise around them so the graph is not entirely degenerate.
    let mut rng = hnsw::Rng::new(5);
    for id in 40..140u64 {
        let v: Vec<f32> = (0..4).map(|_| rng.range(-2.0, 2.0)).collect();
        index.insert(id, &v).unwrap();
    }

    index.validate().unwrap();
    assert!(index.is_connected(), "duplicates fragmented the graph");
    assert_eq!(index.reachable_from_entry(0), index.node_count());

    let hits = index.search(&point, 40).unwrap();
    assert_eq!(hits.len(), 40);
    let mut ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..40u64).collect::<Vec<_>>(),
        "not every duplicate came back"
    );
    assert!(hits.iter().all(|h| h.distance == 0.0));
}

#[test]
fn every_vector_being_identical_still_produces_a_valid_graph() {
    let mut index = Hnsw::new(3, Metric::L2, Params::default().with_m(4)).unwrap();
    for id in 0..200u64 {
        index.insert(id, &[1.0, 1.0, 1.0]).unwrap();
    }
    // Every distance is exactly zero, so the diversity condition
    // `d(e, q) < d(e, r)` is never satisfied and selection degenerates to
    // keeping the M nearest. The graph still has to hold together.
    index.validate().unwrap();
    assert!(index.is_connected());
    assert_eq!(index.reachable_from_entry(0), index.node_count());
    assert_eq!(index.search(&[1.0, 1.0, 1.0], 10).unwrap().len(), 10);
}

#[test]
fn the_zero_vector_is_survivable_under_cosine() {
    // Cosine of the zero vector is undefined; the kernel defines it as
    // maximally distant rather than returning NaN, which would poison every
    // heap comparison in the search.
    let mut index = Hnsw::new(4, Metric::Cosine, Params::default().with_m(4)).unwrap();
    index.insert(0, &[0.0, 0.0, 0.0, 0.0]).unwrap();
    for id in 1..60u64 {
        let angle = id as f32 * 0.1;
        index
            .insert(id, &[angle.cos(), angle.sin(), 0.2, -0.3])
            .unwrap();
    }
    index.validate().unwrap();

    // Querying with a zero vector must not panic and must not return NaN.
    let hits = index.search(&[0.0, 0.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(hits.len(), 5);
    assert!(hits.iter().all(|h| h.distance.is_finite()));

    // The stored zero vector is reachable and sits at distance 1 from
    // everything, so it is never a false top hit for a real direction.
    let real = index.search(&[1.0, 0.0, 0.0, 0.0], 60).unwrap();
    assert_eq!(real.len(), 60);
    let zero_rank = real
        .iter()
        .position(|h| h.id == 0)
        .expect("zero is findable");
    // Everything within 90 degrees of the query beats it; everything beyond
    // 90 degrees is further away than it. It must never crowd the top.
    assert!(
        zero_rank > 10,
        "the zero vector ranked {zero_rank}, too high"
    );
}

#[test]
fn a_zero_vector_survives_the_brute_force_baseline_too() {
    let mut base = BruteForce::new(3, Metric::Cosine).unwrap();
    base.insert(1, &[0.0, 0.0, 0.0]).unwrap();
    base.insert(2, &[1.0, 0.0, 0.0]).unwrap();
    let hits = base.search(&[0.0, 0.0, 0.0], 2).unwrap();
    assert!(hits.iter().all(|h| h.distance.is_finite()));
}

#[test]
fn a_wrong_width_vector_is_rejected_on_insert_and_on_query() {
    let mut index = Hnsw::new(4, Metric::L2, Params::default()).unwrap();
    index.insert(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();

    assert!(matches!(
        index.insert(1, &[1.0, 2.0]),
        Err(Error::DimensionMismatch {
            expected: 4,
            found: 2
        })
    ));
    assert!(matches!(
        index.search(&[1.0; 9], 5),
        Err(Error::DimensionMismatch {
            expected: 4,
            found: 9
        })
    ));
    assert!(matches!(
        index.insert_batch(10, &[1.0; 7]),
        Err(Error::DimensionMismatch { .. })
    ));
    assert_eq!(index.len(), 1, "a rejected insert must not land");
}

#[test]
fn a_duplicate_id_is_rejected_without_disturbing_the_index() {
    let mut index = Hnsw::new(2, Metric::L2, Params::default()).unwrap();
    index.insert(7, &[1.0, 1.0]).unwrap();
    let before = index.node_count();

    assert!(matches!(
        index.insert(7, &[9.0, 9.0]),
        Err(Error::DuplicateId(7))
    ));
    assert_eq!(index.node_count(), before);
    assert_eq!(index.vector_of(7).unwrap(), &[1.0, 1.0]);
}

#[test]
fn zero_dimensions_and_broken_parameters_are_refused_at_construction() {
    assert!(matches!(
        Hnsw::new(0, Metric::L2, Params::default()),
        Err(Error::InvalidParams(_))
    ));
    for broken in [
        Params {
            m: 0,
            ..Params::default()
        },
        Params {
            m: 8,
            m_max0: 4,
            ..Params::default()
        },
        Params {
            ef_construction: 0,
            ..Params::default()
        },
        Params {
            ef_search: 0,
            ..Params::default()
        },
        Params {
            ml: 0.0,
            ..Params::default()
        },
        Params {
            ml: f64::NAN,
            ..Params::default()
        },
        Params {
            ml: f64::INFINITY,
            ..Params::default()
        },
    ] {
        assert!(
            Hnsw::new(4, Metric::L2, broken).is_err(),
            "these parameters should have been refused: {broken:?}"
        );
    }
}

#[test]
fn a_one_dimensional_index_works() {
    // dims = 1 means the chunked kernels run entirely in their remainder
    // loop, which is a separate code path from the eight-lane body.
    let mut index = Hnsw::new(1, Metric::L2, Params::default().with_m(4)).unwrap();
    for id in 0..200u64 {
        index.insert(id, &[id as f32 * 0.5]).unwrap();
    }
    index.validate().unwrap();
    assert!(index.is_connected());

    let hits = index.search(&[50.0], 3).unwrap();
    assert_eq!(hits[0].id, 100);
    assert_eq!(hits[0].distance, 0.0);
}

#[test]
fn m_equal_to_one_fragments_the_graph_and_larger_m_does_not() {
    // M = 1 caps layer-0 degree at 2, and a graph with maximum degree 2 is a
    // disjoint union of paths and cycles — it *cannot* be connected except by
    // an accident of ordering. This is not a bug to fix, it is the reason M
    // has a sensible floor, and the index still has to stay well formed and
    // answer queries at that setting.
    let (data, _) = clustered_with_queries(300, 8, 4, 0, DATA_SEED + 52);

    let mut counts = Vec::new();
    for m in [1usize, 2, 4, 8] {
        let params = Params {
            m,
            m_max0: m * 2,
            ef_construction: 48,
            ef_search: 48,
            ..Params::default()
        };
        let index = build(8, Metric::L2, params, data.flat());
        index.validate().expect("well formed at every M");
        assert_eq!(index.search(data.get(0), 5).unwrap().len(), 5);
        counts.push((m, index.components().len()));
    }

    assert!(
        counts[0].1 > 1,
        "M = 1 was expected to fragment: {counts:?}"
    );
    assert_eq!(counts[3].1, 1, "M = 8 must give one component: {counts:?}");
    assert!(
        counts[0].1 > counts[3].1,
        "fragmentation did not fall as M rose: {counts:?}"
    );
}

#[test]
fn extreme_magnitudes_do_not_produce_infinities() {
    // 1e20 squared overflows f32 and 1e-25 squared underflows it, so a norm
    // computed as sqrt(dot(v, v)) is +inf for the first vector and zero for
    // the second. Normalisation then gives up on both and cosine is left
    // taking the dot product of unscaled vectors, which is neither in [0, 2]
    // nor ordered the way a distance has to be.
    let huge = [1e20f32, 1e20, 1e20, 1e20];
    let mut index = Hnsw::new(4, Metric::Cosine, Params::default().with_m(4)).unwrap();
    index.insert(0, &huge).unwrap();
    index.insert(1, &[1e-25, 1e-25, 0.0, 0.0]).unwrap();
    index.insert(2, &[1.0, 0.0, 0.0, 0.0]).unwrap();

    let hits = index.search(&huge, 3).unwrap();
    assert_eq!(hits.len(), 3);
    let distances: Vec<f32> = hits.iter().map(|h| h.distance).collect();
    assert!(
        hits.iter().all(|h| (0.0..=2.0).contains(&h.distance)),
        "cosine distance left its documented [0, 2] range: {distances:?}"
    );
    assert_eq!(
        hits[0].id, 0,
        "an exact copy of the query is not the nearest neighbour: {hits:?}"
    );
}

#[test]
fn a_batch_whose_ids_run_past_u64_max_reports_it() {
    // Two vectors fit at the very top of the id space, the third does not.
    let vectors = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];

    let mut index = Hnsw::new(2, Metric::L2, Params::default()).unwrap();
    match index.insert_batch(u64::MAX - 1, &vectors) {
        Err(Error::InvalidParams(_)) => {}
        other => panic!("expected an id-range error, got {other:?}"),
    }
    // Documented on insert_batch: a batch is not a transaction.
    assert_eq!(index.len(), 2);
    assert!(index.contains(u64::MAX - 1));
    assert!(index.contains(u64::MAX));

    let mut exact = BruteForce::new(2, Metric::L2).unwrap();
    match exact.insert_batch(u64::MAX - 1, &vectors) {
        Err(Error::InvalidParams(_)) => {}
        other => panic!("expected an id-range error, got {other:?}"),
    }
    assert_eq!(exact.len(), 2);
}

#[test]
fn a_query_against_a_one_node_index_needs_no_beam() {
    let mut index = Hnsw::new(2, Metric::L2, Params::default()).unwrap();
    index.insert(1, &[0.0, 0.0]).unwrap();
    let (hits, stats) = index
        .search_stats(&mut Searcher::new(), &[1.0, 1.0], 5, 1)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(stats.visited, 1);
}

#[test]
fn dataset_helpers_reject_impossible_shapes() {
    let mut data = hnsw::Dataset::clustered(10, 4, 2, 0.1, 1);
    assert_eq!(data.len(), 10);
    assert_eq!(data.dims(), 4);
    // Asking for more queries than there are points takes all of them.
    let queries = data.split_off_queries(100);
    assert_eq!(queries.len(), 10);
    assert_eq!(data.len(), 0);
    assert!(data.is_empty());
}

#[test]
fn the_toy_embedding_is_deterministic_and_normalised() {
    let a = hnsw::embed::embed("night", 64);
    let b = hnsw::embed::embed("night", 64);
    assert_eq!(a, b);
    assert!((hnsw::distance::norm(&a) - 1.0).abs() < 1e-5);

    // Shared character n-grams should beat unrelated spelling.
    let near = Metric::Cosine.distance_preprocessed(&a, &hnsw::embed::embed("nights", 64));
    let far = Metric::Cosine.distance_preprocessed(&a, &hnsw::embed::embed("zebra", 64));
    assert!(
        near < far,
        "'nights' ({near:.3}) was not closer than 'zebra' ({far:.3})"
    );

    // Empty and punctuation-only input must not divide by zero.
    let empty = hnsw::embed::embed("", 64);
    assert!(empty.iter().all(|x| x.is_finite()));
    assert!(hnsw::embed::embed("!!!", 64).iter().all(|x| x.is_finite()));
}

#[test]
fn the_demo_word_list_is_present_and_usable() {
    let words = hnsw::embed::demo_words();
    assert!(
        (2000..=40000).contains(&words.len()),
        "the demo ships {} words",
        words.len()
    );
    assert!(words.iter().all(|w| !w.is_empty()));
    assert!(words
        .iter()
        .all(|w| w.chars().all(|c| c.is_ascii_lowercase())));

    let flat = hnsw::embed::embed_all(&words, hnsw::embed::DEMO_DIMS);
    assert_eq!(flat.len(), words.len() * hnsw::embed::DEMO_DIMS);
    assert!(flat.iter().all(|x| x.is_finite()));
}
