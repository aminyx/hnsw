# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-21

First release. A complete Hierarchical Navigable Small World index written
from scratch with no dependencies.

### Added

- **Index** (`hnsw::index`) — multi-layer proximal graph with exponential
  level assignment, greedy descent through the sparse upper layers, and an
  `ef`-bounded beam search at layer 0. Algorithms 1-5 of Malkov & Yashunin.
- **Neighbour-selection heuristic** — the paper's diversity-preserving
  pruning rule, with `Selection::NaiveTopM` kept alongside it so the
  difference can be measured rather than asserted.
- **Distances** (`hnsw::distance`) — cosine, inner product and squared L2
  behind a `Distance` trait, written with eight independent accumulator lanes
  so LLVM is permitted to vectorise them without fast-math.
- **PRNG** (`hnsw::rng`) — SplitMix64-seeded xoshiro256++ and the exponential
  level draw. An index is bit-identical given a seed and an insertion order.
- **Deletion** — tombstones that search filters, an `ef` widening rule that
  compensates for them, and `compact()` to rebuild.
- **Persistence** (`hnsw::persist`) — single-file format with a magic header,
  a version, the PRNG state and a trailing CRC-32, validated structurally on
  load.
- **Exact baseline and harness** (`hnsw::brute`, `hnsw::eval`) — linear-scan
  ground truth plus a recall@k / QPS sweep reported as the fastest of nine
  timed passes.
- **`hnswctl`** — `gen`, `build`, `query`, `info`, `bench` and `recall`.
- **Browser demo** — the index compiled to `wasm32-unknown-unknown` behind a
  plain `extern "C"` ABI (no `wasm-bindgen`), building a live graph over ~25 000
  word vectors and reporting hops, visits and live recall per query.
- 135 tests plus doc-tests, all offline and deterministic.

### Notes

Three bugs were found by the tests during development and are documented at
the code that fixes them, because each one left every structural invariant
intact while quietly breaking retrieval:

- `Dataset::clustered` assigned cluster membership round-robin, which made
  `index % clusters` equal the cluster id and turned "delete every fourth
  vector" into "delete three whole clusters".
- Tied distances were broken by node id. Ids are handed out in insertion
  order, so pruning systematically stripped incoming edges from newer nodes.
- Hashing the id fixed the direction of that bias but not its shape: a global
  order still collapsed every node's neighbour list onto the same few hubs.
  The tie-break has to be salted with the node doing the pruning.

On 40 duplicate vectors among 140, those last two took retrieval from 10 of
40 duplicates found to 40 of 40. `cargo run --release --example duplicates`
reproduces the fixed result — 140 of 140 nodes reachable, 40 of 40 duplicates
recovered. The earlier figures were measured with the tie-break reverted to
rules the repository no longer implements, so the example cannot print them.

[0.1.0]: https://github.com/aminyx/hnsw/releases/tag/v0.1.0
