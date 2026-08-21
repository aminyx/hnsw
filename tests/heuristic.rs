//! The neighbour-selection heuristic — the part that makes HNSW work.
//!
//! These tests exist to show, not assert, that "keep the M closest" is wrong.

use hnsw::{Hnsw, Metric, Params, Selection};

/// Six points on a line: a tight trio near the origin and a tight trio ten
/// units away. Node `C` at position 2 is the one asked to pick neighbours.
///
/// ```text
///   A   B   C                             D   E   F
///   0   1   2                            10  11  12
/// ```
///
/// Every same-side point is nearer to `C` than every far-side point, so
/// naive top-2 selects `{B, A}` and spends both links walking left. The
/// heuristic rejects `A` — `A` is closer to the already-kept `B` than to `C`,
/// making the link redundant — and spends the freed slot on `D`, the only
/// candidate that buys reachability across the gap.
fn line_index(selection: Selection) -> (Hnsw<Metric>, Vec<u32>) {
    let positions = [0.0f32, 1.0, 2.0, 10.0, 11.0, 12.0];
    let params = Params {
        m: 2,
        m_max0: 4,
        ef_construction: 16,
        ef_search: 16,
        selection,
        ..Params::default()
    };
    let mut index = Hnsw::new(1, Metric::L2, params).unwrap();
    let nodes: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(i, &p)| index.insert(i as u64, &[p]).unwrap())
        .collect();
    (index, nodes)
}

#[test]
fn naive_top_m_spends_both_links_inside_the_near_cluster() {
    let (index, nodes) = line_index(Selection::NaiveTopM);
    let candidates = [nodes[0], nodes[1], nodes[3], nodes[4], nodes[5]];
    let picked = index
        .select_from_candidates(&[2.0], &candidates, 2)
        .unwrap();

    assert_eq!(picked.len(), 2);
    assert!(
        picked.iter().all(|n| *n <= nodes[2]),
        "naive top-M unexpectedly crossed the gap: {picked:?}"
    );
    assert_eq!(picked, vec![nodes[1], nodes[0]], "expected B then A");
}

#[test]
fn the_heuristic_trades_a_redundant_link_for_a_bridge() {
    let (index, nodes) = line_index(Selection::Heuristic);
    let candidates = [nodes[0], nodes[1], nodes[3], nodes[4], nodes[5]];
    let picked = index
        .select_from_candidates(&[2.0], &candidates, 2)
        .unwrap();

    assert_eq!(picked.len(), 2);
    assert!(
        picked.contains(&nodes[1]),
        "B is the nearest, it must be kept"
    );
    assert!(
        picked.contains(&nodes[3]),
        "the heuristic dropped the bridge to the far cluster: {picked:?}"
    );
    assert!(
        !picked.contains(&nodes[0]),
        "A is redundant behind B and should have been pruned: {picked:?}"
    );
}

#[test]
fn both_strategies_respect_the_requested_limit() {
    for selection in [Selection::Heuristic, Selection::NaiveTopM] {
        let (index, nodes) = line_index(selection);
        for m in 1..=5usize {
            let picked = index.select_from_candidates(&[2.0], &nodes, m).unwrap();
            assert!(
                picked.len() <= m,
                "{selection:?} returned {} for m={m}",
                picked.len()
            );
        }
    }
}

#[test]
fn selection_returns_candidates_ordered_by_distance() {
    let (index, nodes) = line_index(Selection::Heuristic);
    let picked = index.select_from_candidates(&[2.0], &nodes, 4).unwrap();
    let distances: Vec<f32> = picked
        .iter()
        .map(|&n| index.vector(n)[0] - 2.0)
        .map(f32::abs)
        .collect();
    for pair in distances.windows(2) {
        assert!(pair[0] <= pair[1], "results were not sorted: {distances:?}");
    }
}

#[test]
fn fewer_candidates_than_m_are_all_kept() {
    for selection in [Selection::Heuristic, Selection::NaiveTopM] {
        let (index, nodes) = line_index(selection);
        let candidates = [nodes[0], nodes[3]];
        let picked = index
            .select_from_candidates(&[2.0], &candidates, 4)
            .unwrap();
        assert_eq!(
            picked.len(),
            2,
            "{selection:?} dropped a candidate it had room for"
        );
    }
}

#[test]
fn keep_pruned_connections_tops_the_set_back_up_to_m() {
    // Nine points crowded on one side. Diversity alone would keep very few,
    // so the paper's keepPrunedConnections rule has to refill the list or
    // degree collapses in dense regions.
    let params = Params {
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        ef_search: 32,
        ..Params::default()
    };
    let mut index = Hnsw::new(1, Metric::L2, params).unwrap();
    let nodes: Vec<u32> = (0..9)
        .map(|i| index.insert(i, &[i as f32 * 0.01]).unwrap())
        .collect();

    let candidates: Vec<u32> = nodes[1..].to_vec();
    let picked = index
        .select_from_candidates(&[0.0], &candidates, 4)
        .unwrap();
    assert_eq!(
        picked.len(),
        4,
        "the set was not topped back up: {picked:?}"
    );
}

#[test]
fn naive_top_m_leaves_nodes_unreachable_and_the_heuristic_does_not() {
    // Same data, same parameters, one flag. A k-nearest graph strands
    // vectors: they point inward at their own dense neighbourhood, and
    // nothing on a path from the entry point ever points back at them.
    let (naive, smart) = build_pair(5000, 64, 25, 0.22, 16);

    let stranded_naive = naive.node_count() - naive.reachable_from_entry(0);
    let stranded_smart = smart.node_count() - smart.reachable_from_entry(0);

    assert!(
        stranded_naive > 0,
        "naive top-M reached everything, so this comparison proves nothing"
    );
    assert_eq!(
        stranded_smart, 0,
        "the heuristic stranded {stranded_smart} nodes"
    );
}

#[test]
fn the_heuristic_beats_naive_selection_on_recall_by_a_wide_margin() {
    let (naive, smart) = build_pair(5000, 64, 25, 0.22, 16);
    let (queries, truth) = query_set(5000, 64, 25, 0.22, 200, 10);

    let naive_recall = recall_of(&naive, &queries, &truth, 10, 32);
    let smart_recall = recall_of(&smart, &queries, &truth, 10, 32);

    assert!(
        smart_recall >= 0.99,
        "the heuristic managed only {smart_recall:.4} at ef 32"
    );
    assert!(
        naive_recall < 0.95,
        "naive top-M was expected to fall short but scored {naive_recall:.4}"
    );
    assert!(
        smart_recall - naive_recall > 0.05,
        "the gap was only {:.4} ({naive_recall:.4} vs {smart_recall:.4})",
        smart_recall - naive_recall
    );
}

#[test]
fn a_wider_beam_cannot_rescue_a_naively_built_graph() {
    // The point that matters in production: `ef_search` is tunable after the
    // fact and neighbour selection is not. A graph built the wrong way has a
    // recall ceiling, and paying sixteen times the query cost does not lift
    // it past what the right graph reaches almost immediately.
    let (naive, smart) = build_pair(5000, 64, 25, 0.22, 16);
    let (queries, truth) = query_set(5000, 64, 25, 0.22, 200, 10);

    let naive_wide = recall_of(&naive, &queries, &truth, 10, 128);
    let smart_narrow = recall_of(&smart, &queries, &truth, 10, 16);

    assert!(
        smart_narrow > naive_wide,
        "heuristic at ef 16 ({smart_narrow:.4}) did not beat naive at ef 128 ({naive_wide:.4})"
    );
    assert!(
        naive_wide < 0.97,
        "naive top-M reached {naive_wide:.4} at ef 128, so it has no visible ceiling"
    );
}

// -- helpers ---------------------------------------------------------------

fn dataset(n: usize, dims: usize, clusters: usize, spread: f32) -> hnsw::Dataset {
    let mut data = hnsw::Dataset::clustered(n + 200, dims, clusters, spread, 7);
    let _ = data.split_off_queries(200);
    data
}

fn query_set(
    n: usize,
    dims: usize,
    clusters: usize,
    spread: f32,
    count: usize,
    k: usize,
) -> (hnsw::Dataset, Vec<Vec<u64>>) {
    let mut data = hnsw::Dataset::clustered(n + count, dims, clusters, spread, 7);
    let queries = data.split_off_queries(count);
    let mut exact = hnsw::BruteForce::new(dims, Metric::L2).unwrap();
    exact.insert_batch(0, data.flat()).unwrap();
    let truth = hnsw::ground_truth(&exact, queries.flat(), k).unwrap();
    (queries, truth)
}

/// The same vectors indexed both ways: `(naive, heuristic)`.
fn build_pair(
    n: usize,
    dims: usize,
    clusters: usize,
    spread: f32,
    m: usize,
) -> (Hnsw<Metric>, Hnsw<Metric>) {
    let data = dataset(n, dims, clusters, spread);
    let base = Params::default().with_m(m).with_ef_construction(200);
    let mut built = Vec::new();
    for selection in [Selection::NaiveTopM, Selection::Heuristic] {
        let mut index = Hnsw::new(dims, Metric::L2, base.with_selection(selection)).unwrap();
        index.insert_batch(0, data.flat()).unwrap();
        built.push(index);
    }
    let smart = built.pop().unwrap();
    let naive = built.pop().unwrap();
    (naive, smart)
}

fn recall_of(
    index: &Hnsw<Metric>,
    queries: &hnsw::Dataset,
    truth: &[Vec<u64>],
    k: usize,
    ef: usize,
) -> f64 {
    let mut scratch = hnsw::Searcher::new();
    let mut total = 0.0;
    for (query, want) in queries.iter().zip(truth) {
        let hits = index.search_with(&mut scratch, query, k, ef).unwrap();
        total += hnsw::recall_at_k(&hits, want);
    }
    total / queries.len().max(1) as f64
}
