//! Structural invariants of the graph after construction.

mod common;

use common::{build, clustered_with_queries, DATA_SEED};
use hnsw::{Hnsw, Metric, Params};

fn small_index() -> Hnsw<Metric> {
    let (data, _) = clustered_with_queries(1200, 24, 10, 0, DATA_SEED);
    build(24, Metric::L2, Params::default().with_m(8), data.flat())
}

#[test]
fn every_invariant_holds_after_a_full_build() {
    let index = small_index();
    index.validate().expect("invariants hold");
}

#[test]
fn links_are_bidirectional_while_no_degree_cap_binds() {
    // Every link is created in both directions. In an index small enough
    // that no node ever exceeds its degree cap, nothing is ever pruned, so
    // that symmetry survives intact and can be checked exactly.
    let mut index = Hnsw::new(4, Metric::L2, Params::default().with_m(16)).unwrap();
    let mut rng = hnsw::Rng::new(21);
    for id in 0..12u64 {
        let v: Vec<f32> = (0..4).map(|_| rng.range(-1.0, 1.0)).collect();
        index.insert(id, &v).unwrap();
    }

    let mut checked = 0usize;
    for node in 0..index.node_count() as u32 {
        assert!(index.neighbors(node, 0).len() < index.params().m_max0);
        for level in 0..=index.level_of(node) {
            for &other in index.neighbors(node, level) {
                assert!(
                    index.neighbors(other, level).contains(&node),
                    "layer {level}: {node} -> {other} has no return edge"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 100, "only {checked} edges were checked");
}

#[test]
fn pruning_makes_the_graph_directed_but_most_links_stay_two_way() {
    // Once the caps bind, a node prunes only its own half of an edge, so some
    // links become one-way. That is deliberate — see `Hnsw::shrink` — and the
    // resulting asymmetry should be a minority of edges, not the norm.
    let index = small_index();
    let (mut total, mut mutual) = (0usize, 0usize);
    let mut one_way_exists = false;
    for node in 0..index.node_count() as u32 {
        for level in 0..=index.level_of(node) {
            for &other in index.neighbors(node, level) {
                total += 1;
                if index.neighbors(other, level).contains(&node) {
                    mutual += 1;
                } else {
                    one_way_exists = true;
                }
            }
        }
    }
    assert!(total > 10_000, "only {total} edges were checked");
    assert!(
        one_way_exists,
        "no pruning happened, so this proves nothing"
    );
    let share = mutual as f64 / total as f64;
    assert!(
        share > 0.75,
        "only {:.1}% of edges are two-way; pruning is destroying the graph",
        share * 100.0
    );
}

#[test]
fn no_node_is_ever_left_without_outgoing_edges() {
    // The guarantee one-sided pruning buys, and the reason a node can never
    // be stranded: its adjacency list is written by nobody but itself.
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        assert!(
            !index.neighbors(node, 0).is_empty(),
            "node {node} has no outgoing edges at layer 0"
        );
    }
}

#[test]
fn every_node_is_reachable_from_the_entry_point() {
    // The connectivity that actually decides recall: a directed walk from the
    // entry point has to be able to arrive anywhere.
    let index = small_index();
    assert_eq!(
        index.reachable_from_entry(0),
        index.node_count(),
        "{} of {} nodes are unreachable by search",
        index.node_count() - index.reachable_from_entry(0),
        index.node_count()
    );
    for level in 1..=index.max_level() {
        assert_eq!(
            index.reachable_from_entry(level),
            index.layer_sizes()[level],
            "layer {level} is not fully reachable from the entry point"
        );
    }
}

#[test]
fn degree_bounds_are_respected_at_layer_zero() {
    let params = Params::default().with_m(8);
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        let degree = index.neighbors(node, 0).len();
        assert!(
            degree <= params.m_max0,
            "node {node} has layer-0 degree {degree}, cap is {}",
            params.m_max0
        );
    }
}

#[test]
fn degree_bounds_are_respected_on_the_upper_layers() {
    let params = Params::default().with_m(8);
    let index = small_index();
    let mut upper_edges = 0usize;
    for node in 0..index.node_count() as u32 {
        for level in 1..=index.level_of(node) {
            let degree = index.neighbors(node, level).len();
            assert!(
                degree <= params.m,
                "node {node} has degree {degree} at layer {level}, cap is {}",
                params.m
            );
            upper_edges += degree;
        }
    }
    assert!(upper_edges > 0, "the index built no upper layers at all");
}

#[test]
fn layer_zero_contains_every_node() {
    let index = small_index();
    let sizes = index.layer_sizes();
    assert_eq!(sizes[0], index.node_count());
    assert_eq!(sizes.len(), index.max_level() + 1);
}

#[test]
fn layer_populations_shrink_as_they_go_up() {
    let index = small_index();
    let sizes = index.layer_sizes();
    for pair in sizes.windows(2) {
        assert!(pair[0] >= pair[1], "layer populations went up: {:?}", sizes);
    }
    assert!(
        index.max_level() >= 2,
        "expected several layers, got {sizes:?}"
    );
}

#[test]
fn a_node_on_an_upper_layer_is_present_on_every_layer_below() {
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        let level = index.level_of(node);
        for lc in 0..=level {
            // Presence is having a link list at that layer at all.
            let _ = index.neighbors(node, lc);
        }
        assert!(
            index.neighbors(node, level + 1).is_empty(),
            "node {node} has links above its own top layer"
        );
    }
}

#[test]
fn layer_zero_is_one_connected_component() {
    let index = small_index();
    let components = index.components();
    assert_eq!(
        components.len(),
        1,
        "graph split into {} components: {components:?}",
        components.len()
    );
    assert_eq!(components[0], index.node_count());
    assert!(index.is_connected());
}

#[test]
fn no_node_links_to_itself() {
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        for level in 0..=index.level_of(node) {
            assert!(
                !index.neighbors(node, level).contains(&node),
                "node {node} links to itself at layer {level}"
            );
        }
    }
}

#[test]
fn no_adjacency_list_holds_a_duplicate() {
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        for level in 0..=index.level_of(node) {
            let mut sorted = index.neighbors(node, level).to_vec();
            let before = sorted.len();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                sorted.len(),
                before,
                "node {node} layer {level} has duplicates"
            );
        }
    }
}

#[test]
fn a_neighbour_on_a_layer_actually_reaches_that_layer() {
    let index = small_index();
    for node in 0..index.node_count() as u32 {
        for level in 0..=index.level_of(node) {
            for &other in index.neighbors(node, level) {
                assert!(
                    index.level_of(other) >= level,
                    "node {other} appears on layer {level} but tops out at {}",
                    index.level_of(other)
                );
            }
        }
    }
}

#[test]
fn the_entry_point_sits_on_the_top_layer() {
    let index = small_index();
    let entry = index.entry_point().expect("non-empty index has an entry");
    assert_eq!(index.level_of(entry), index.max_level());
}

#[test]
fn mean_layer_zero_degree_is_healthy() {
    let index = small_index();
    let mean = index.mean_degree();
    let params = Params::default().with_m(8);
    assert!(
        mean >= params.m as f64,
        "mean layer-0 degree {mean:.1} is below M = {}, the graph is starved",
        params.m
    );
    assert!(mean <= params.m_max0 as f64);
}

#[test]
fn building_twice_from_the_same_seed_gives_an_identical_graph() {
    let (data, _) = clustered_with_queries(600, 16, 6, 0, DATA_SEED);
    let params = Params::default().with_m(8).with_seed(0xFEED);
    let first = build(16, Metric::L2, params, data.flat());
    let second = build(16, Metric::L2, params, data.flat());

    assert_eq!(first.max_level(), second.max_level());
    assert_eq!(first.layer_sizes(), second.layer_sizes());
    for node in 0..first.node_count() as u32 {
        assert_eq!(first.level_of(node), second.level_of(node));
        for level in 0..=first.level_of(node) {
            assert_eq!(
                first.neighbors(node, level),
                second.neighbors(node, level),
                "node {node} layer {level} differs between builds"
            );
        }
    }
}

#[test]
fn a_different_seed_gives_a_different_but_still_valid_graph() {
    let (data, _) = clustered_with_queries(600, 16, 6, 0, DATA_SEED);
    let a = build(
        16,
        Metric::L2,
        Params::default().with_m(8).with_seed(1),
        data.flat(),
    );
    let b = build(
        16,
        Metric::L2,
        Params::default().with_m(8).with_seed(2),
        data.flat(),
    );

    let levels_a: Vec<usize> = (0..a.node_count() as u32).map(|n| a.level_of(n)).collect();
    let levels_b: Vec<usize> = (0..b.node_count() as u32).map(|n| b.level_of(n)).collect();
    assert_ne!(levels_a, levels_b, "two seeds produced identical levels");

    a.validate().expect("seed 1 is valid");
    b.validate().expect("seed 2 is valid");
    assert!(a.is_connected() && b.is_connected());
}

#[test]
fn edge_count_matches_a_direct_tally() {
    let index = small_index();
    let mut tally = 0usize;
    for node in 0..index.node_count() as u32 {
        for level in 0..=index.level_of(node) {
            tally += index.neighbors(node, level).len();
        }
    }
    assert_eq!(tally, index.edge_count());
}
