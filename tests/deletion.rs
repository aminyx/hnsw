//! Tombstones and compaction.

mod common;

use common::{build, clustered_with_queries, mean_recall, DATA_SEED};
use hnsw::{Hnsw, Metric, Params};

fn index_of_2000() -> (Hnsw<Metric>, hnsw::Dataset, hnsw::Dataset) {
    let (data, queries) = clustered_with_queries(2000, 32, 10, 60, DATA_SEED + 20);
    let index = build(32, Metric::L2, Params::default().with_m(12), data.flat());
    (index, data, queries)
}

#[test]
fn a_deleted_id_is_never_returned() {
    let (mut index, data, queries) = index_of_2000();

    // Tombstone every fourth vector.
    let removed: Vec<u64> = (0..data.len() as u64).filter(|i| i % 4 == 0).collect();
    for &id in &removed {
        assert!(
            index.delete(id),
            "id {id} was live and should have been deleted"
        );
    }
    assert_eq!(index.deleted_count(), removed.len());
    assert_eq!(index.len(), data.len() - removed.len());

    for query in &queries {
        for hit in index.search(query, 25).unwrap() {
            assert!(
                !removed.contains(&hit.id),
                "tombstoned id {} came back in the results",
                hit.id
            );
        }
    }
}

#[test]
fn deleting_the_same_id_twice_reports_it_only_once() {
    let (mut index, _, _) = index_of_2000();
    assert!(index.delete(7));
    assert!(!index.delete(7), "a second delete must report no change");
    assert_eq!(index.deleted_count(), 1);
}

#[test]
fn deleting_an_unknown_id_is_a_no_op() {
    let (mut index, _, _) = index_of_2000();
    let before = index.len();
    assert!(!index.delete(999_999));
    assert_eq!(index.len(), before);
    assert_eq!(index.deleted_count(), 0);
}

#[test]
fn contains_and_is_deleted_agree_with_each_other() {
    let (mut index, _, _) = index_of_2000();
    assert!(index.contains(11));
    assert!(!index.is_deleted(11));

    index.delete(11);
    assert!(!index.contains(11), "a tombstoned id is not 'contained'");
    assert!(index.is_deleted(11));

    assert!(!index.contains(999_999));
    assert!(!index.is_deleted(999_999), "an absent id is not 'deleted'");
}

#[test]
fn deleted_nodes_stay_in_the_graph_and_keep_carrying_traffic() {
    // The whole justification for tombstoning: the topology is untouched.
    let (mut index, data, _) = index_of_2000();
    let edges_before = index.edge_count();
    let layers_before = index.layer_sizes();

    for id in 0..data.len() as u64 / 2 {
        index.delete(id);
    }
    assert_eq!(
        index.edge_count(),
        edges_before,
        "deletion rewired the graph"
    );
    assert_eq!(index.layer_sizes(), layers_before);
    assert_eq!(index.node_count(), data.len());
    index.validate().expect("the graph is still well formed");
}

#[test]
fn a_full_k_still_comes_back_when_most_of_the_index_is_tombstoned() {
    // The beam has to widen to compensate, or the filter starves the result.
    let (mut index, data, queries) = index_of_2000();
    for id in 0..data.len() as u64 {
        if id % 5 != 0 {
            index.delete(id);
        }
    }
    let live = index.len();
    assert!(live >= 10);

    for query in &queries {
        let hits = index.search(query, 10).unwrap();
        assert_eq!(
            hits.len(),
            10,
            "only {} results survived the filter",
            hits.len()
        );
    }
}

#[test]
fn tombstoning_a_whole_region_starves_queries_that_land_in_it() {
    // An honest limitation, asserted rather than hidden. Widening the beam
    // compensates for tombstones that are *spread through* the index, because
    // the live neighbours are right there next to the dead ones. It cannot
    // compensate when an entire neighbourhood is dead: greedy descent lands
    // in that neighbourhood, the beam explores it, everything it finds is
    // filtered out, and the search terminates having never had a reason to
    // walk somewhere else. Compaction is the fix, not a bigger `ef`.
    let dims = 8;
    let mut flat = Vec::new();
    for cluster in 0..2 {
        let offset = if cluster == 0 { 0.0f32 } else { 30.0 };
        let mut rng = hnsw::Rng::new(cluster + 1);
        for _ in 0..500 {
            for d in 0..dims {
                flat.push(if d == 0 { offset } else { 0.0 } + 0.3 * rng.gaussian() as f32);
            }
        }
    }
    let mut index = build(dims, Metric::L2, Params::default().with_m(8), &flat);

    // Kill the second cluster outright.
    for id in 500..1000u64 {
        index.delete(id);
    }
    assert_eq!(index.len(), 500);

    let mut query = vec![0.0f32; dims];
    query[0] = 30.0;
    let hits = index.search(&query, 10).unwrap();
    assert!(
        hits.len() < 10,
        "the starvation this test documents did not happen: {} hits",
        hits.len()
    );

    // Compaction restores it: the dead region is gone from the graph, so the
    // descent has nowhere wrong to land.
    index.compact();
    let hits = index.search(&query, 10).unwrap();
    assert_eq!(
        hits.len(),
        10,
        "compaction did not repair the starved region"
    );
    assert!(hits.iter().all(|h| h.id < 500));
}

#[test]
fn deleting_everything_leaves_an_index_that_finds_nothing() {
    let (mut index, data, queries) = index_of_2000();
    for id in 0..data.len() as u64 {
        index.delete(id);
    }
    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert!(index.search(queries.get(0), 10).unwrap().is_empty());
}

#[test]
fn compact_reclaims_the_tombstones_and_keeps_the_live_ids() {
    let (mut index, data, _) = index_of_2000();
    let removed: Vec<u64> = (0..data.len() as u64).filter(|i| i % 3 == 0).collect();
    for &id in &removed {
        index.delete(id);
    }

    let live_before = index.len();
    let reclaimed = index.compact();

    assert_eq!(reclaimed, removed.len());
    assert_eq!(index.deleted_count(), 0);
    assert_eq!(index.len(), live_before);
    assert_eq!(index.node_count(), live_before);
    index.validate().expect("the rebuilt graph is well formed");
    assert!(index.is_connected());

    for id in 0..data.len() as u64 {
        assert_eq!(index.contains(id), !removed.contains(&id), "id {id}");
    }
}

#[test]
fn compact_on_a_clean_index_does_nothing() {
    let (mut index, _, queries) = index_of_2000();
    let before = index.search(queries.get(0), 10).unwrap();
    assert_eq!(index.compact(), 0);
    assert_eq!(index.search(queries.get(0), 10).unwrap(), before);
}

#[test]
fn compact_preserves_recall_against_the_surviving_vectors() {
    let (data, queries) = clustered_with_queries(2500, 48, 12, 80, DATA_SEED + 21);
    let mut index = build(48, Metric::L2, Params::default().with_m(12), data.flat());

    // Drop a quarter of the vectors, then measure against a baseline that
    // never had them.
    let mut survivors = Vec::new();
    for id in 0..data.len() as u64 {
        if id % 4 == 1 {
            index.delete(id);
        } else {
            survivors.extend_from_slice(data.get(id as usize));
        }
    }
    let mut base = hnsw::BruteForce::new(48, Metric::L2).unwrap();
    for (i, id) in (0..data.len() as u64).filter(|i| i % 4 != 1).enumerate() {
        base.insert(id, &survivors[i * 48..(i + 1) * 48]).unwrap();
    }

    let before = mean_recall(&index, &base, &queries, 10, 96);
    index.compact();
    let after = mean_recall(&index, &base, &queries, 10, 96);

    assert!(before >= 0.95, "recall before compaction was {before:.4}");
    assert!(after >= 0.95, "recall after compaction was {after:.4}");
    assert!(
        (after - before).abs() < 0.05,
        "compaction moved recall from {before:.4} to {after:.4}"
    );
}

#[test]
fn compact_is_reproducible() {
    let (data, queries) = clustered_with_queries(1200, 24, 8, 20, DATA_SEED + 22);
    let params = Params::default().with_m(10).with_seed(0xBEEF);

    let mut a = build(24, Metric::L2, params, data.flat());
    let mut b = build(24, Metric::L2, params, data.flat());
    for id in (0..data.len() as u64).step_by(7) {
        a.delete(id);
        b.delete(id);
    }
    a.compact();
    b.compact();

    for query in &queries {
        assert_eq!(a.search(query, 10).unwrap(), b.search(query, 10).unwrap());
    }
}

#[test]
fn an_id_can_be_reinserted_after_a_compaction() {
    let (mut index, _, _) = index_of_2000();
    let vector = index.vector_of(5).expect("id 5 exists").to_vec();

    assert!(index.delete(5));
    // The tombstone still occupies the id, so a straight reinsert is refused.
    assert!(matches!(
        index.insert(5, &vector),
        Err(hnsw::Error::DuplicateId(5))
    ));

    index.compact();
    assert!(!index.contains(5));
    index
        .insert(5, &vector)
        .expect("the id is free after compaction");
    assert!(index.contains(5));

    let hits = index.search(&vector, 1).unwrap();
    assert_eq!(hits[0].id, 5);
}

#[test]
fn deleting_the_entry_point_does_not_break_search() {
    let (mut index, _, queries) = index_of_2000();
    let entry = index.entry_point().expect("there is an entry point");
    let entry_id = index.id_at(entry);

    assert!(index.delete(entry_id));
    assert_eq!(
        index.entry_point(),
        Some(entry),
        "the entry point must stay in the graph even when tombstoned"
    );

    for query in &queries {
        let hits = index.search(query, 10).unwrap();
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.id != entry_id));
    }
}
