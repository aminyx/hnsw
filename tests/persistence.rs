//! The on-disk format: round trips, and every way a file can be wrong.

mod common;

use common::{build, clustered_with_queries, DATA_SEED};
use hnsw::{crc32, Error, Hnsw, Metric, Params, Selection, FORMAT_VERSION, MAGIC};

fn fixture() -> (Hnsw<Metric>, hnsw::Dataset, hnsw::Dataset) {
    let (data, queries) = clustered_with_queries(1500, 32, 8, 40, DATA_SEED + 30);
    let params = Params::default()
        .with_m(10)
        .with_ef_construction(120)
        .with_ef_search(48)
        .with_seed(0x1234_5678);
    let index = build(32, Metric::Cosine, params, data.flat());
    (index, data, queries)
}

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("hnsw-test-{}-{name}.hnsw", std::process::id()));
    path
}

#[test]
fn a_round_trip_returns_identical_search_results() {
    let (index, _, queries) = fixture();
    let restored = Hnsw::from_bytes(&index.to_bytes()).expect("the file parses");

    for query in &queries {
        assert_eq!(
            index.search(query, 15).unwrap(),
            restored.search(query, 15).unwrap()
        );
    }
}

#[test]
fn a_round_trip_preserves_the_graph_byte_for_byte() {
    let (index, _, _) = fixture();
    let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();

    assert_eq!(restored.node_count(), index.node_count());
    assert_eq!(restored.max_level(), index.max_level());
    assert_eq!(restored.entry_point(), index.entry_point());
    assert_eq!(restored.layer_sizes(), index.layer_sizes());
    assert_eq!(restored.edge_count(), index.edge_count());
    for node in 0..index.node_count() as u32 {
        assert_eq!(restored.id_at(node), index.id_at(node));
        assert_eq!(restored.level_of(node), index.level_of(node));
        assert_eq!(restored.vector(node), index.vector(node));
        for level in 0..=index.level_of(node) {
            assert_eq!(
                restored.neighbors(node, level),
                index.neighbors(node, level)
            );
        }
    }
    restored
        .validate()
        .expect("the restored graph is well formed");
}

#[test]
fn a_round_trip_preserves_parameters_and_the_metric() {
    let (index, _, _) = fixture();
    let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();

    assert_eq!(restored.metric(), index.metric());
    assert_eq!(restored.dims(), index.dims());
    assert_eq!(restored.params(), index.params());
    assert_eq!(restored.rng_state(), index.rng_state());
}

#[test]
fn every_metric_and_selection_combination_round_trips() {
    let (data, _) = clustered_with_queries(400, 16, 5, 0, DATA_SEED + 31);
    for metric in Metric::ALL {
        for selection in [Selection::Heuristic, Selection::NaiveTopM] {
            let params = Params::default().with_m(6).with_selection(selection);
            let index = build(16, metric, params, data.flat());
            let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();
            assert_eq!(restored.metric(), metric);
            assert_eq!(restored.params().selection, selection);
        }
    }
}

#[test]
fn a_round_trip_preserves_tombstones() {
    let (mut index, _, queries) = fixture();
    for id in (0..300u64).step_by(3) {
        index.delete(id);
    }
    let deleted = index.deleted_count();
    let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();

    assert_eq!(restored.deleted_count(), deleted);
    assert_eq!(restored.len(), index.len());
    for id in (0..300u64).step_by(3) {
        assert!(restored.is_deleted(id), "id {id} lost its tombstone");
    }
    for query in &queries {
        assert_eq!(
            index.search(query, 10).unwrap(),
            restored.search(query, 10).unwrap()
        );
    }
}

#[test]
fn an_index_reloaded_from_disk_continues_the_same_random_stream() {
    // If the PRNG state were not persisted, inserts after a reload would draw
    // different levels and silently produce a different graph.
    let (data, _) = clustered_with_queries(600, 16, 5, 0, DATA_SEED + 32);
    let mut original = build(16, Metric::L2, Params::default().with_m(8), data.flat());
    let mut reloaded = Hnsw::from_bytes(&original.to_bytes()).unwrap();

    for id in 1000..1100u64 {
        let vector = data.get((id as usize - 1000) % data.len()).to_vec();
        original.insert(id, &vector).unwrap();
        reloaded.insert(id, &vector).unwrap();
    }
    assert_eq!(original.layer_sizes(), reloaded.layer_sizes());
    for node in 0..original.node_count() as u32 {
        assert_eq!(original.level_of(node), reloaded.level_of(node));
        assert_eq!(original.neighbors(node, 0), reloaded.neighbors(node, 0));
    }
}

#[test]
fn an_empty_index_round_trips() {
    let index = Hnsw::new(8, Metric::L2, Params::default()).unwrap();
    let restored = Hnsw::from_bytes(&index.to_bytes()).unwrap();
    assert_eq!(restored.node_count(), 0);
    assert_eq!(restored.entry_point(), None);
    assert!(restored.search(&[0.0; 8], 5).unwrap().is_empty());
}

#[test]
fn save_and_load_through_the_filesystem() {
    let (index, _, queries) = fixture();
    let path = temp_path("roundtrip");
    index.save(&path).expect("the file is written");

    let restored = Hnsw::load(&path).expect("the file is read back");
    for query in &queries {
        assert_eq!(
            index.search(query, 10).unwrap(),
            restored.search(query, 10).unwrap()
        );
    }
    std::fs::remove_file(&path).ok();
}

#[test]
fn loading_a_missing_file_reports_io_rather_than_corruption() {
    let mut path = std::env::temp_dir();
    path.push("hnsw-test-definitely-not-here.hnsw");
    assert!(matches!(Hnsw::load(&path), Err(Error::Io(_))));
}

#[test]
fn a_flipped_byte_anywhere_in_the_payload_is_caught_by_the_crc() {
    let (index, _, _) = fixture();
    let good = index.to_bytes();

    // Sample positions across the header, the ids, the vectors and the
    // adjacency table.
    for offset in [12usize, 20, 60, 104, 400, good.len() / 2, good.len() - 8] {
        let mut corrupt = good.clone();
        corrupt[offset] ^= 0x40;
        match Hnsw::from_bytes(&corrupt) {
            Err(Error::ChecksumMismatch { .. }) => {}
            Err(other) => panic!("byte {offset} gave {other} rather than a checksum failure"),
            Ok(_) => panic!("byte {offset} was corrupted and the file still loaded"),
        }
    }
}

#[test]
fn a_tampered_file_with_a_recomputed_crc_is_still_rejected() {
    // The realistic attack on a checksum: change the data and fix the check.
    // Structural validation has to catch what the CRC no longer can.
    let (index, _, _) = fixture();
    let mut bytes = index.to_bytes();

    // Point the very first adjacency entry at a node that does not exist.
    let n = index.node_count();
    let adjacency_start = 100 + n * 8 + n + n + n * index.dims() * 4;
    let target = adjacency_start + 4;
    bytes[target..target + 4].copy_from_slice(&(n as u32 + 500).to_le_bytes());

    let split = bytes.len() - 4;
    let crc = crc32(&bytes[..split]);
    bytes[split..].copy_from_slice(&crc.to_le_bytes());

    match Hnsw::from_bytes(&bytes) {
        Err(Error::Corrupt(_)) => {}
        other => panic!("expected a structural rejection, got {other:?}"),
    }
}

#[test]
fn a_bad_magic_is_rejected_before_anything_else() {
    let (index, _, _) = fixture();
    let mut bytes = index.to_bytes();
    bytes[0] = b'X';
    assert!(matches!(Hnsw::from_bytes(&bytes), Err(Error::BadMagic)));

    assert!(matches!(
        Hnsw::from_bytes(b"this is a text file, not an index"),
        Err(Error::BadMagic)
    ));
}

#[test]
fn a_future_format_version_is_refused_rather_than_guessed_at() {
    let (index, _, _) = fixture();
    let mut bytes = index.to_bytes();
    bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 7).to_le_bytes());

    match Hnsw::from_bytes(&bytes) {
        Err(Error::UnsupportedVersion { found, supported }) => {
            assert_eq!(found, FORMAT_VERSION + 7);
            assert_eq!(supported, FORMAT_VERSION);
        }
        other => panic!("expected a version error, got {other:?}"),
    }
}

#[test]
fn a_truncated_file_is_rejected() {
    let (index, _, _) = fixture();
    let good = index.to_bytes();
    for cut in [0usize, 4, 11, 50, 103, good.len() / 3, good.len() - 1] {
        let result = Hnsw::from_bytes(&good[..cut]);
        assert!(
            result.is_err(),
            "a file cut to {cut} of {} bytes still loaded",
            good.len()
        );
    }
}

#[test]
fn an_impossible_node_count_cannot_trigger_a_huge_allocation() {
    let (index, _, _) = fixture();
    let mut bytes = index.to_bytes();
    // Field at offset 84 is the node count.
    bytes[84..88].copy_from_slice(&u32::MAX.to_le_bytes());
    let split = bytes.len() - 4;
    let crc = crc32(&bytes[..split]);
    bytes[split..].copy_from_slice(&crc.to_le_bytes());

    match Hnsw::from_bytes(&bytes) {
        Err(Error::Corrupt(_)) => {}
        other => panic!("expected a structural rejection, got {other:?}"),
    }
}

/// Hand-build the smallest legal file — one node, one dimension, no links —
/// with three fields left hostile. Nothing real is involved, so a single
/// field can be made absurd without a valid index getting in the way.
fn crafted_one_node_file(m: u32, m_max0: u32, adjacency_count: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.push(Metric::L2.tag());
    bytes.push(Selection::Heuristic.tag());
    bytes.extend_from_slice(&[0, 0]); // reserved
    bytes.extend_from_slice(&1u32.to_le_bytes()); // dims
    bytes.extend_from_slice(&m.to_le_bytes());
    bytes.extend_from_slice(&m_max0.to_le_bytes());
    bytes.extend_from_slice(&16u32.to_le_bytes()); // ef_construction
    bytes.extend_from_slice(&16u32.to_le_bytes()); // ef_search
    bytes.extend_from_slice(&1.0f64.to_bits().to_le_bytes()); // ml
    bytes.extend_from_slice(&7u64.to_le_bytes()); // seed
    for _ in 0..4 {
        bytes.extend_from_slice(&1u64.to_le_bytes()); // PRNG state
    }
    bytes.extend_from_slice(&1u32.to_le_bytes()); // node count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // entry point
    bytes.extend_from_slice(&0u32.to_le_bytes()); // max level
    bytes.extend_from_slice(&42u64.to_le_bytes()); // the one external id
    bytes.push(0); // its top level
    bytes.push(0); // not tombstoned
    bytes.extend_from_slice(&1.0f32.to_le_bytes()); // its one component
    bytes.extend_from_slice(&adjacency_count.to_le_bytes()); // layer 0 links
    let crc = crc32(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    bytes
}

#[test]
fn an_adjacency_count_larger_than_the_file_is_rejected_before_allocating() {
    // The count is legal against the degree cap and impossible against the
    // file: there are no bytes left for even one of the links it promises.
    let bytes = crafted_one_node_file(16, 4096, 4000);
    assert!(
        bytes.len() < 200,
        "the hostile file should be tiny, it is {} bytes",
        bytes.len()
    );
    match Hnsw::from_bytes(&bytes) {
        Err(Error::Corrupt(_)) => {}
        other => panic!("expected a structural rejection, got {other:?}"),
    }
}

#[test]
fn an_absurd_degree_cap_cannot_be_smuggled_in_through_a_file() {
    // The degree cap is what bounds an adjacency allocation, and it is read
    // from the same file, so it needs a bound of its own.
    let bytes = crafted_one_node_file(u32::MAX, u32::MAX, 0);
    match Hnsw::from_bytes(&bytes) {
        Err(Error::InvalidParams(_)) => {}
        other => panic!("expected the parameters to be rejected, got {other:?}"),
    }

    let bytes = crafted_one_node_file(16, u32::MAX, 0);
    match Hnsw::from_bytes(&bytes) {
        Err(Error::InvalidParams(_)) => {}
        other => panic!("expected the parameters to be rejected, got {other:?}"),
    }
}

#[test]
fn the_magic_and_version_are_where_the_documentation_says() {
    let (index, _, _) = fixture();
    let bytes = index.to_bytes();
    assert_eq!(&bytes[..8], MAGIC);
    assert_eq!(
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        FORMAT_VERSION
    );
    assert_eq!(bytes[12], Metric::Cosine.tag());
    assert_eq!(bytes[13], Selection::Heuristic.tag());
}

#[test]
fn crc32_matches_the_published_check_value() {
    // The standard IEEE CRC-32 of "123456789" is 0xCBF43926.
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
    assert_ne!(crc32(b"a"), crc32(b"b"));
}
