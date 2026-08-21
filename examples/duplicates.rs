//! Reproduces the tie-break table documented on `index::tie_break_from`.
//!
//! Exact duplicates make every distance tie, which turns the *order ties
//! resolve in* into the thing that decides the graph's topology. Run with:
//!
//! ```text
//! cargo run --release --example duplicates
//! ```

use hnsw::{Hnsw, Metric, Params};

fn main() {
    let point = [0.25f32, -0.5, 0.75, 1.0];
    let mut index = Hnsw::new(4, Metric::L2, Params::default().with_m(4)).unwrap();

    // 40 exact duplicates ...
    for id in 0..40u64 {
        index.insert(id, &point).unwrap();
    }
    // ... hidden among 100 ordinary vectors.
    let mut rng = hnsw::Rng::new(5);
    for id in 40..140u64 {
        let v: Vec<f32> = (0..4).map(|_| rng.range(-2.0, 2.0)).collect();
        index.insert(id, &v).unwrap();
    }

    let found = index.search(&point, 40).unwrap();
    let duplicates = found.iter().filter(|h| h.id < 40).count();

    println!("nodes                 {}", index.node_count());
    println!("undirected components {:?}", index.components());
    println!(
        "reachable from entry  {} / {}",
        index.reachable_from_entry(0),
        index.node_count()
    );
    println!("duplicates recovered  {duplicates} / 40");
    println!();
    println!("Every invariant holds either way; only the last two lines tell you");
    println!("whether the index can actually find what it stores.");

    assert_eq!(duplicates, 40, "the duplicates are not all retrievable");
    index.validate().unwrap();
}
