//! Hierarchical Navigable Small World approximate nearest-neighbour search,
//! implemented from scratch with no dependencies.
//!
//! HNSW is the index inside essentially every vector database shipping today
//! — Qdrant, Weaviate, Milvus, `pgvector`'s `hnsw` access method, FAISS's
//! `IndexHNSW`, Lucene's `knn` field. This crate is that algorithm written
//! out rather than called: the layered graph, the exponential level
//! assignment, the beam search, the neighbour-selection heuristic, the
//! distance kernels, the PRNG, the file format and the CRC are all here, and
//! all explained where they are implemented.
//!
//! # Quick start
//!
//! ```
//! use hnsw::{BruteForce, Dataset, Hnsw, Metric, Params};
//!
//! // 2000 points in 32 dimensions, in eight Gaussian clusters.
//! let mut data = Dataset::clustered(2000, 32, 8, 0.25, 42);
//! let queries = data.split_off_queries(20);
//!
//! let mut index = Hnsw::new(32, Metric::L2, Params::default().with_m(16))?;
//! index.insert_batch(0, data.flat())?;
//!
//! // Measure against an exact scan rather than trusting the output.
//! let mut exact = BruteForce::new(32, Metric::L2)?;
//! exact.insert_batch(0, data.flat())?;
//!
//! let mut total = 0.0;
//! for q in &queries {
//!     let approx = index.search(q, 10)?;
//!     let truth: Vec<u64> = exact.search(q, 10)?.into_iter().map(|n| n.id).collect();
//!     total += hnsw::recall_at_k(&approx, &truth);
//! }
//! assert!(total / 20.0 > 0.95, "recall was {}", total / 20.0);
//! # Ok::<(), hnsw::Error>(())
//! ```
//!
//! # Map of the crate
//!
//! | module | what is in it |
//! |---|---|
//! | [`index`] | the graph: insert, beam search, the selection heuristic, tombstones, compaction |
//! | [`distance`] | cosine / dot / L2 kernels, written to auto-vectorise |
//! | [`rng`] | xoshiro256++ and the exponential level draw |
//! | [`brute`] | the exact baseline every recall number is measured against |
//! | [`persist`] | the versioned, CRC-checked single-file format |
//! | [`dataset`] | reproducible clustered-Gaussian data |
//! | [`eval`] | the recall/QPS sweep |
//! | [`embed`] | the toy character-n-gram embedding the browser demo uses |
//!
//! # What this is not
//!
//! Single-threaded construction, no SIMD intrinsics, in-memory only, soft
//! deletes. See the README for the full list and why each one is where the
//! line was drawn.

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod brute;
pub mod dataset;
pub mod distance;
pub mod embed;
pub mod error;
pub mod eval;
pub mod index;
pub mod persist;
pub mod rng;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use brute::{ground_truth, recall_at_k, BruteForce};
pub use dataset::Dataset;
pub use distance::{Cosine, Distance, Euclidean, InnerProduct, Metric};
pub use error::{Error, Result};
pub use index::{Hnsw, LayerStats, Neighbor, NodeId, Params, SearchStats, Searcher, Selection};
pub use persist::{crc32, FORMAT_VERSION, MAGIC};
pub use rng::Rng;

/// Crate version, from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
