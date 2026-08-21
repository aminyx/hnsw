//! The index: a multi-layer proximal graph, built and searched from scratch.
//!
//! # The shape of the thing
//!
//! HNSW is two ideas stacked on top of each other.
//!
//! The first is a **navigable small world graph**: connect every point to a
//! handful of near neighbours *plus* keep a few long-range links, and greedy
//! hill-climbing — repeatedly step to whichever neighbour is closer to the
//! query — lands on the true nearest neighbour surprisingly often. Greedy
//! search on such a graph costs polylogarithmic hops instead of the linear
//! scan a flat index needs.
//!
//! The second is a **skip list over that graph**. A plain small-world graph
//! still has a problem: the greedy walk starts somewhere arbitrary and takes
//! many small steps to cross the space. So each node is also assigned a
//! random top layer from a geometric distribution, and layer `l` contains
//! only the nodes whose top layer is at least `l`. Layer 0 holds everything;
//! each layer up holds roughly `1/M` of the layer below. Searching descends:
//! a cheap greedy walk in the sparse upper layers covers huge distances in
//! very few hops (the links up there are long because the points are sparse),
//! and by the time it reaches layer 0 it is already in the right
//! neighbourhood and only needs local refinement.
//!
//! ```text
//!  layer 2      (o)- - - - - - - - - - - - - -(o)          ~n/M^2 nodes
//!                 \                            |
//!  layer 1      (o)-----(o)- - - - -(o)-------(o)          ~n/M   nodes
//!                 \     / \         / \        |
//!  layer 0      (o)-(o)-(o)-(o)-(o)-(o)-(o)---(o)          n      nodes
//!                       ^                  ^
//!                   entry point         query lands here
//! ```
//!
//! # What is actually implemented here
//!
//! Algorithms 1-5 of Malkov & Yashunin, *Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World graphs*
//! (2016/2018). Deletion is the one operation the paper does not specify;
//! this implementation tombstones rather than editing the graph, and
//! [`Hnsw::delete`] explains why there is no cheap alternative.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use crate::distance::{Distance, Metric};
use crate::error::{Error, Result};
use crate::rng::{assign_level, default_ml, Rng, MAX_LEVEL};

/// Internal, dense node index. External ids are `u64` and arbitrary.
pub type NodeId = u32;

/// One node's adjacency, indexed by layer: `links[level]`.
type NodeLinks = Vec<Vec<NodeId>>;

// ---------------------------------------------------------------------------
// Deterministic id map
// ---------------------------------------------------------------------------

/// Hasher for `u64` keys that does not depend on process-random state.
///
/// `std`'s default `RandomState` is seeded per process, which would make
/// iteration order — and therefore anything derived from it — vary between
/// runs. Everything in this crate has to be reproducible from a seed, so the
/// id map uses a fixed SplitMix64 finaliser instead.
#[derive(Default, Clone, Copy)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    fn write_u64(&mut self, n: u64) {
        let mut s = n;
        self.0 = crate::rng::splitmix64(&mut s);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// External id to internal node index.
pub type IdMap = HashMap<u64, NodeId, BuildHasherDefault<IdHasher>>;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// How neighbours are chosen from a candidate set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Selection {
    /// The pruning heuristic from Algorithm 4 of the paper. The default, and
    /// the reason this index works. See [`Hnsw::select_neighbors`].
    #[default]
    Heuristic,
    /// Keep the `M` nearest candidates and nothing else.
    ///
    /// Provided so the heuristic's value can be demonstrated rather than
    /// asserted: build the same data both ways and compare connectivity and
    /// recall. Do not use it in production.
    NaiveTopM,
}

impl Selection {
    /// Stable tag for persistence.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            Selection::Heuristic => 0,
            Selection::NaiveTopM => 1,
        }
    }

    /// Inverse of [`Selection::tag`].
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Selection> {
        match tag {
            0 => Some(Selection::Heuristic),
            1 => Some(Selection::NaiveTopM),
            _ => None,
        }
    }
}

/// Largest `m` this build accepts.
///
/// Far above anything useful — published work stops caring somewhere around
/// 64 — but a bound has to exist, because `m` and `m_max0` are read straight
/// out of an index file and then used to size adjacency allocations. Without
/// a ceiling a corrupt or hostile file can name a degree cap of four billion
/// and have the reader believe it.
pub const MAX_M: usize = 4096;

/// Largest `m_max0` this build accepts, for the reason [`MAX_M`] gives.
pub const MAX_M_MAX0: usize = 2 * MAX_M;

/// Construction and search parameters.
///
/// | field | meaning | raising it |
/// |---|---|---|
/// | `m` | links a new node makes per layer | better recall, more memory, slower build |
/// | `m_max0` | hard degree cap at layer 0 | better recall, more memory |
/// | `ef_construction` | beam width while building | better graph, slower build, no query cost |
/// | `ef_search` | beam width while querying | better recall, slower queries |
/// | `ml` | level-assignment scale | more layers, longer descent |
/// | `seed` | PRNG seed | a different but equally good graph |
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Number of bidirectional links a node establishes on each of its
    /// layers. The single most important knob: memory is `O(n * m)` and
    /// recall climbs steeply up to about 16, then flattens.
    pub m: usize,
    /// Degree cap at layer 0, where every node lives and where the final
    /// refinement happens. Conventionally `2 * m`.
    pub m_max0: usize,
    /// Beam width used during insertion. Bigger means each new node sees a
    /// better candidate set, so the graph is better connected. It costs build
    /// time only — queries never pay for it.
    pub ef_construction: usize,
    /// Default beam width for [`Hnsw::search`]. This is the recall/latency
    /// dial, and the only one that can be changed after the index is built.
    pub ef_search: usize,
    /// Scale of the exponential level distribution. `1 / ln(m)` is optimal
    /// per the paper; see [`crate::rng::assign_level`].
    pub ml: f64,
    /// Seed for the level assignments. Same seed and same insertion order
    /// give a bit-identical graph.
    pub seed: u64,
    /// Neighbour selection strategy.
    pub selection: Selection,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 64,
            ml: default_ml(16),
            seed: 0x2026_0A11_C0FF_EE01,
            selection: Selection::Heuristic,
        }
    }
}

impl Params {
    /// Set `m`, and follow the conventions `m_max0 = 2m` and `ml = 1/ln(m)`.
    ///
    /// Use the struct fields directly to decouple them.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max0 = m.saturating_mul(2);
        self.ml = default_ml(m);
        self
    }

    /// Set the build-time beam width.
    #[must_use]
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set the default query-time beam width.
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set the PRNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the neighbour-selection strategy.
    #[must_use]
    pub fn with_selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Reject parameter sets the index cannot honour.
    pub fn validate(&self) -> Result<()> {
        if self.m == 0 {
            return Err(Error::InvalidParams("m must be at least 1"));
        }
        if self.m > MAX_M {
            return Err(Error::InvalidParams("m is above the supported maximum"));
        }
        if self.m_max0 < self.m {
            return Err(Error::InvalidParams("m_max0 must be at least m"));
        }
        if self.m_max0 > MAX_M_MAX0 {
            return Err(Error::InvalidParams(
                "m_max0 is above the supported maximum",
            ));
        }
        if self.ef_construction == 0 {
            return Err(Error::InvalidParams("ef_construction must be at least 1"));
        }
        if self.ef_search == 0 {
            return Err(Error::InvalidParams("ef_search must be at least 1"));
        }
        if !self.ml.is_finite() || self.ml <= 0.0 {
            return Err(Error::InvalidParams("ml must be finite and positive"));
        }
        Ok(())
    }

    /// Degree cap for a given layer.
    #[must_use]
    pub fn max_degree(&self, level: usize) -> usize {
        if level == 0 {
            self.m_max0
        } else {
            self.m
        }
    }
}

// ---------------------------------------------------------------------------
// Results and scratch space
// ---------------------------------------------------------------------------

/// One search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbor {
    /// External id supplied at insert time.
    pub id: u64,
    /// Distance under the index's metric. Smaller is nearer.
    pub distance: f32,
}

/// Per-layer breakdown of a traversal, in descent order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerStats {
    /// Layer this row describes.
    pub level: usize,
    /// Nodes whose adjacency was expanded on this layer.
    pub hops: usize,
    /// Distinct nodes evaluated on this layer.
    pub visited: usize,
}

/// A single node visit, recorded when tracing is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    /// The node that was evaluated.
    pub node: NodeId,
    /// The layer it was evaluated on.
    pub level: u8,
}

/// What the traversal actually did. This is the observable that turns HNSW
/// from a black box into something you can reason about: `visited` divided by
/// the index size is exactly the fraction of the dataset the query touched.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SearchStats {
    /// Nodes whose adjacency lists were expanded, across all layers.
    pub hops: usize,
    /// Distinct nodes evaluated, across all layers.
    pub visited: usize,
    /// Distance function calls. Equal to `visited` — every node is scored
    /// exactly once per layer thanks to the visited stamps — and reported
    /// separately because that equality is the property worth watching.
    pub distance_computations: usize,
    /// Per-layer breakdown. Empty unless a stats-collecting entry point was
    /// used.
    pub layers: Vec<LayerStats>,
    /// Every visit in order. `None` unless [`Hnsw::search_traced`] was used.
    pub trace: Option<Vec<Visit>>,
    /// Whether `layers` should be filled. Off by default so that the
    /// throughput path performs no allocation at all.
    pub(crate) collect_layers: bool,
}

/// Reusable scratch space for a search.
///
/// The visited set is the one allocation a naive implementation repeats per
/// query. Here it is a stamp array: a `Vec<u32>` of generation counters and a
/// current generation, so "clear the visited set" is `generation += 1`
/// instead of an `O(n)` memset. Hand one of these to
/// [`Hnsw::search_with`] from a thread-local and queries allocate nothing at
/// all beyond the two heaps.
#[derive(Clone, Debug, Default)]
pub struct Searcher {
    stamps: Vec<u32>,
    generation: u32,
}

impl Searcher {
    /// A fresh, empty scratch buffer.
    #[must_use]
    pub fn new() -> Self {
        Searcher::default()
    }

    /// Bytes of scratch currently held.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.stamps.capacity() * core::mem::size_of::<u32>()
    }

    fn ensure(&mut self, nodes: usize) {
        if self.stamps.len() < nodes {
            self.stamps.resize(nodes, 0);
        }
    }

    /// Start a new visited set. Wraps safely: on overflow the stamps are
    /// zeroed once and counting restarts, so a long-lived `Searcher` cannot
    /// alias a stale generation.
    fn begin(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.stamps.iter_mut().for_each(|s| *s = 0);
            self.generation = 1;
        }
    }

    /// Mark `node` visited; returns `true` if it had not been visited yet.
    #[inline]
    fn visit(&mut self, node: NodeId) -> bool {
        let slot = &mut self.stamps[node as usize];
        if *slot == self.generation {
            false
        } else {
            *slot = self.generation;
            true
        }
    }
}

/// A node paired with its distance to whatever is being searched for.
///
/// `Ord` is by distance ascending, so a `BinaryHeap<Scored>` is a max-heap on
/// distance (the result set, where the worst is popped) and
/// `BinaryHeap<Reverse<Scored>>` is a min-heap (the candidate queue, where
/// the best is popped).
///
/// Ties are broken by [`tie_break`] rather than by the raw node id, for the
/// reasons set out there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Scored {
    dist: f32,
    node: NodeId,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| tie_break(self.node).cmp(&tie_break(other.node)))
    }
}

/// Deterministic ordering key for a node, used only when distances tie.
///
/// # Why not just compare node ids
///
/// Equal distances need *some* deterministic ordering or the search stops
/// being reproducible. Ordering by node id is the obvious choice and it is
/// wrong: ids are handed out in insertion order, so any rule preferring the
/// lower id systematically prefers the *older* node. Hashing removes that
/// correlation while keeping the order fixed. The hash is computed only when
/// distances actually tie, which on real float data is essentially never.
#[inline]
fn tie_break(node: NodeId) -> u64 {
    let mut state = u64::from(node);
    crate::rng::splitmix64(&mut state)
}

/// Ordering key for `node` as seen *from* `base`.
///
/// # Why the tie-break has to be relative
///
/// This is the second half of a bug that a global tie-break only half fixes,
/// and it is worth spelling out because nothing about it is visible from the
/// invariants.
///
/// Take a dataset with many exact duplicates — boilerplate documents,
/// repeated product listings, anything real. Every distance between them is
/// zero, so every comparison is a tie and the selection heuristic's diversity
/// test (`d(e, q) < d(e, r)`, strictly less) is never satisfied. Selection
/// collapses to "keep the first `m` in whatever order ties resolve".
///
/// With ties resolved by node id, that order prefers older nodes, so every
/// pruning step drops the newest neighbour. New nodes lose their incoming
/// edges as fast as they gain them. Nothing looks wrong — no node is
/// stranded, the undirected graph is one component, every invariant in
/// [`Hnsw::validate`] holds — but nothing points *at* the newcomers, so a
/// search can never arrive at them.
///
/// Replacing the id with a hash of the id fixes the direction of the bias and
/// leaves its shape intact, because the order is still *global*: every node
/// pruning its own list keeps the same globally-lowest-hash neighbours. The
/// graph collapses onto a handful of hubs instead of onto the oldest nodes.
///
/// Salting the hash with the pruning node makes each node's ordering its own,
/// so tied neighbours are spread evenly across the graph instead of piling
/// onto a shared few. Measured on 40 duplicate vectors among 140, `m = 4`:
///
/// | tie-break | reachable from entry | duplicates found (of 40) |
/// |---|---|---|
/// | node id | 74 / 140 | 10 |
/// | hash of id | 88 / 140 | 23 |
/// | hash of (base, id) | 140 / 140 | 40 |
#[inline]
fn tie_break_from(base: NodeId, node: NodeId) -> u64 {
    let mut state = (u64::from(base) << 32) ^ u64::from(node) ^ 0x9E37_79B9_7F4A_7C15;
    crate::rng::splitmix64(&mut state)
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// The index
// ---------------------------------------------------------------------------

/// A Hierarchical Navigable Small World index.
///
/// Generic over the distance function; [`Metric`] is the built-in
/// implementation and the only one [`Hnsw::save`] can persist, because the
/// file format stores a metric tag rather than user code.
///
/// # Example
///
/// ```
/// use hnsw::{Hnsw, Metric, Params};
///
/// let mut index = Hnsw::new(3, Metric::L2, Params::default())?;
/// index.insert(10, &[1.0, 0.0, 0.0])?;
/// index.insert(20, &[0.0, 1.0, 0.0])?;
/// index.insert(30, &[0.9, 0.1, 0.0])?;
///
/// let hits = index.search(&[1.0, 0.0, 0.0], 2)?;
/// assert_eq!(hits[0].id, 10);
/// assert_eq!(hits[1].id, 30);
/// # Ok::<(), hnsw::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Hnsw<D: Distance = Metric> {
    pub(crate) dims: usize,
    pub(crate) params: Params,
    pub(crate) dist: D,

    /// All vectors, flattened: node `i` occupies `[i * dims, (i + 1) * dims)`.
    /// One allocation instead of `n`, which keeps the hot loop's memory
    /// access predictable.
    pub(crate) vectors: Vec<f32>,
    pub(crate) ids: Vec<u64>,
    pub(crate) id_map: IdMap,
    pub(crate) levels: Vec<u8>,
    pub(crate) links: Vec<NodeLinks>,
    pub(crate) deleted: Vec<bool>,
    pub(crate) deleted_count: usize,

    pub(crate) entry: Option<NodeId>,
    pub(crate) max_level: usize,
    pub(crate) rng: Rng,
    /// Scratch reused across inserts; only ever touched through `&mut self`.
    pub(crate) scratch: Searcher,
}

impl Hnsw<Metric> {
    /// Create an index over `dims`-dimensional vectors with a built-in metric.
    pub fn new(dims: usize, metric: Metric, params: Params) -> Result<Self> {
        Hnsw::with_distance(dims, metric, params)
    }

    /// The metric this index was built with.
    #[must_use]
    pub fn metric(&self) -> Metric {
        self.dist
    }
}

impl<D: Distance> Hnsw<D> {
    /// Create an index with a user-supplied distance function.
    pub fn with_distance(dims: usize, dist: D, params: Params) -> Result<Self> {
        if dims == 0 {
            return Err(Error::InvalidParams("dims must be at least 1"));
        }
        params.validate()?;
        Ok(Hnsw {
            dims,
            params,
            dist,
            vectors: Vec::new(),
            ids: Vec::new(),
            id_map: IdMap::default(),
            levels: Vec::new(),
            links: Vec::new(),
            deleted: Vec::new(),
            deleted_count: 0,
            entry: None,
            max_level: 0,
            rng: Rng::new(params.seed),
            scratch: Searcher::new(),
        })
    }

    // -- accessors ---------------------------------------------------------

    /// Vector width.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The parameters in force.
    #[must_use]
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Change the default query beam width without rebuilding. This is the
    /// only parameter that can be retuned on a live index.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef.max(1);
    }

    /// The distance function.
    #[must_use]
    pub fn distance(&self) -> &D {
        &self.dist
    }

    /// Number of live (non-tombstoned) vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len() - self.deleted_count
    }

    /// Whether there is nothing left to find.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Nodes in the graph, tombstones included. Always `>= len()`.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.ids.len()
    }

    /// How many nodes are tombstoned.
    #[must_use]
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    /// Index of the topmost layer.
    #[must_use]
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// The node greedy descent starts from, if the index is non-empty.
    #[must_use]
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry
    }

    /// Whether `id` is present and live.
    ///
    /// A tombstoned id reads as absent here but is *not* free for reuse:
    /// [`Hnsw::delete`] explains that it stays reserved until
    /// [`Hnsw::compact`] runs, so re-inserting it fails with
    /// [`Error::DuplicateId`] even though this returns `false`. Use
    /// [`Hnsw::is_deleted`] to tell the two cases apart.
    #[must_use]
    pub fn contains(&self, id: u64) -> bool {
        self.id_map
            .get(&id)
            .is_some_and(|&n| !self.deleted[n as usize])
    }

    /// The stored (post-`preprocess`) vector for an external id.
    #[must_use]
    pub fn vector_of(&self, id: u64) -> Option<&[f32]> {
        self.id_map.get(&id).map(|&n| self.vector(n))
    }

    /// Population of every layer, index 0 first.
    ///
    /// These should decay by roughly a factor of `M` per layer; if they do
    /// not, `ml` is wrong for the chosen `M`.
    #[must_use]
    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.max_level + 1];
        for &lvl in &self.levels {
            for size in sizes.iter_mut().take(lvl as usize + 1) {
                *size += 1;
            }
        }
        if self.levels.is_empty() {
            sizes.clear();
        }
        sizes
    }

    /// Total directed edges across all layers.
    ///
    /// Links start out two-way, so early on this is twice the number of
    /// distinct connections; pruning removes single halves, so it drifts
    /// below that as the index fills.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.links.iter().flatten().map(Vec::len).sum()
    }

    /// The stored vector for an internal node.
    #[must_use]
    pub fn vector(&self, node: NodeId) -> &[f32] {
        let start = node as usize * self.dims;
        &self.vectors[start..start + self.dims]
    }

    /// External id of an internal node.
    #[must_use]
    pub fn id_at(&self, node: NodeId) -> u64 {
        self.ids[node as usize]
    }

    /// Top layer of an internal node.
    #[must_use]
    pub fn level_of(&self, node: NodeId) -> usize {
        self.levels[node as usize] as usize
    }

    /// Neighbours of `node` on `level`, or an empty slice if it does not
    /// reach that high.
    #[must_use]
    pub fn neighbors(&self, node: NodeId, level: usize) -> &[NodeId] {
        self.links[node as usize]
            .get(level)
            .map_or(&[][..], Vec::as_slice)
    }

    #[inline]
    fn dist_to(&self, query: &[f32], node: NodeId) -> f32 {
        self.dist.distance_preprocessed(query, self.vector(node))
    }

    // -- insertion ---------------------------------------------------------

    /// Insert a vector under an external id.
    ///
    /// This is Algorithm 1 of the paper:
    ///
    /// 1. Draw a top layer `l` from the exponential distribution.
    /// 2. From the entry point, greedily descend layers `max_level` down to
    ///    `l + 1` with a beam of one — pure hill climbing, no backtracking,
    ///    because up there we only need to get roughly close.
    /// 3. On each layer from `min(l, max_level)` down to 0, run the
    ///    `ef_construction`-wide beam search, pick `M` neighbours from the
    ///    result with [`Hnsw::select_neighbors`], link, and shrink any
    ///    neighbour that went over its degree cap.
    /// 4. If `l` beat the old maximum, this node becomes the new entry point.
    ///
    /// # Bidirectional links, one-sided pruning
    ///
    /// Step 3 adds `a -> b` and `b -> a` together: a link is always created
    /// in both directions, and in an index small enough that no degree cap
    /// ever binds, the graph stays exactly symmetric. Pruning then removes
    /// only the pruning node's own half, so the graph drifts into being
    /// directed as it fills up. `Hnsw::shrink` explains why removing both
    /// halves — the tidier-looking option — strands nodes.
    ///
    /// What survives as a hard invariant is the more useful one: a node's own
    /// adjacency is only ever written by that node, so **no node is ever left
    /// without outgoing edges**. [`Hnsw::validate`] checks it.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if the vector is the wrong width,
    /// [`Error::DuplicateId`] if the id is already present.
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<NodeId> {
        if vector.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: vector.len(),
            });
        }
        if self.id_map.contains_key(&id) {
            return Err(Error::DuplicateId(id));
        }

        let mut query = vector.to_vec();
        self.dist.preprocess(&mut query);

        let node = NodeId::try_from(self.ids.len())
            .map_err(|_| Error::InvalidParams("index is limited to 2^32 nodes"))?;
        let level = assign_level(&mut self.rng, self.params.ml);

        self.vectors.extend_from_slice(&query);
        self.ids.push(id);
        self.id_map.insert(id, node);
        self.levels.push(level as u8);
        self.links.push(vec![Vec::new(); level + 1]);
        self.deleted.push(false);
        self.scratch.ensure(self.ids.len());

        let Some(entry) = self.entry else {
            self.entry = Some(node);
            self.max_level = level;
            return Ok(node);
        };

        // Phase 1: greedy descent with a beam of one, down to the layer where
        // this node actually needs links.
        let mut current = vec![entry];
        let mut scratch = core::mem::take(&mut self.scratch);
        let mut stats = SearchStats::default();
        if self.max_level > level {
            for lc in ((level + 1)..=self.max_level).rev() {
                let found = self.search_layer(&query, &current, 1, lc, &mut scratch, &mut stats);
                if let Some(best) = found.first() {
                    current = vec![best.node];
                }
            }
        }

        // Phase 2: beam search, select, link, shrink — layer by layer down.
        let top = level.min(self.max_level);
        for lc in (0..=top).rev() {
            let candidates = self.search_layer(
                &query,
                &current,
                self.params.ef_construction,
                lc,
                &mut scratch,
                &mut stats,
            );
            let selected = self.select_neighbors(&candidates, self.params.m, node);
            for &e in &selected {
                self.link(node, e, lc);
            }
            let cap = self.params.max_degree(lc);
            for &e in &selected {
                self.shrink(e, lc, cap);
            }
            current = candidates.iter().map(|c| c.node).collect();
        }
        self.scratch = scratch;

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(node);
        }
        Ok(node)
    }

    /// Insert many vectors from a flat buffer of `n * dims` floats.
    ///
    /// Ids are `first_id`, `first_id + 1`, ... Provided because it is what
    /// every benchmark and CLI path wants, and because doing it in one call
    /// keeps the scratch buffer warm.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if the buffer is not a whole number of
    /// vectors, [`Error::DuplicateId`] if one of the generated ids is already
    /// present, and [`Error::InvalidParams`] if the id range runs past
    /// `u64::MAX`.
    ///
    /// **A batch is not a transaction.** Failure part of the way through
    /// leaves every vector before the failing one inserted; nothing is rolled
    /// back. Check [`Hnsw::len`] rather than assuming the index is untouched.
    pub fn insert_batch(&mut self, first_id: u64, vectors: &[f32]) -> Result<usize> {
        if vectors.len() % self.dims != 0 {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: vectors.len() % self.dims,
            });
        }
        let n = vectors.len() / self.dims;
        self.vectors.reserve(vectors.len());
        for i in 0..n {
            let id = first_id
                .checked_add(i as u64)
                .ok_or(Error::InvalidParams("batch ids run past u64::MAX"))?;
            self.insert(id, &vectors[i * self.dims..(i + 1) * self.dims])?;
        }
        Ok(n)
    }

    /// Add an undirected edge, skipping self-loops and duplicates.
    fn link(&mut self, a: NodeId, b: NodeId, level: usize) {
        if a == b {
            return;
        }
        for (x, y) in [(a, b), (b, a)] {
            let list = &mut self.links[x as usize][level];
            if !list.contains(&y) {
                list.push(y);
            }
        }
    }

    /// Bring `node`'s own adjacency at `level` back under `cap`.
    ///
    /// The dropped neighbours keep *their* edge to `node`, so the graph is
    /// directed. That asymmetry is not sloppiness, it is the property that
    /// keeps the index navigable: a node only ever prunes its own list, and
    /// that list starts at `M` and is only ever cut back to `m_max >= M`, so
    /// **every node keeps its outgoing edges forever**. Remove both halves
    /// instead and the guarantee disappears — a node whose every neighbour
    /// prunes it away is stranded with no edges at all, which is precisely
    /// what happens on degenerate data where distances tie and pruning falls
    /// back to a stable ordering. The paper takes the one-sided route for
    /// this reason and every production implementation follows it.
    fn shrink(&mut self, node: NodeId, level: usize, cap: usize) {
        if self.links[node as usize][level].len() <= cap {
            return;
        }
        let current = core::mem::take(&mut self.links[node as usize][level]);

        let mut scored: Vec<Scored> = current
            .iter()
            .map(|&n| Scored {
                dist: self
                    .dist
                    .distance_preprocessed(self.vector(node), self.vector(n)),
                node: n,
            })
            .collect();
        scored.sort_unstable();
        self.links[node as usize][level] = self.select_neighbors(&scored, cap, node);
    }

    // -- neighbour selection ----------------------------------------------

    /// Run the configured neighbour-selection strategy over an explicit
    /// candidate list, as if `base` were being linked into the graph.
    ///
    /// The graph never calls this — it has the distances already — but it
    /// makes the strategy directly observable, which is how the tests
    /// demonstrate that naive top-`M` throws away the links the heuristic
    /// keeps. Returns node ids ordered by distance to `base`.
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] if `base` is the wrong width.
    pub fn select_from_candidates(
        &self,
        base: &[f32],
        candidates: &[NodeId],
        m: usize,
    ) -> Result<Vec<NodeId>> {
        if base.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: base.len(),
            });
        }
        let mut prepared = base.to_vec();
        self.dist.preprocess(&mut prepared);
        let mut scored: Vec<Scored> = candidates
            .iter()
            .filter(|&&n| (n as usize) < self.ids.len())
            .map(|&n| Scored {
                dist: self.dist_to(&prepared, n),
                node: n,
            })
            .collect();
        scored.sort_unstable();
        Ok(self.select_neighbors(&scored, m, NodeId::MAX))
    }

    /// Choose up to `m` neighbours for a point from a candidate set that is
    /// already sorted ascending by distance to that point.
    ///
    /// The base point itself is never needed here, only the candidates'
    /// distances *to* it, which `search_layer` has already computed and
    /// carried along.
    ///
    /// # Why not simply keep the `m` closest?
    ///
    /// Because that builds a k-nearest-neighbour graph, and greedy search on
    /// a k-NN graph gets stuck.
    ///
    /// Picture two dense clusters, `A` and `B`, far apart. Every point in `A`
    /// has hundreds of `A`-neighbours nearer to it than anything in `B`, so
    /// naive top-`M` fills every adjacency list in `A` with other `A` points
    /// — and likewise inside `B`. Almost no edge is ever spent on the gap.
    /// Real embedding spaces are exactly this shape: dense modes separated by
    /// sparse regions.
    ///
    /// What that costs is a *local minimum*. The walk arrives somewhere in
    /// `A`, every neighbour of every node it can see is further from the
    /// query than the node it is standing on, and the search terminates —
    /// correctly, by its own rules, on the wrong answer. It is not that the
    /// beam was too narrow; there is no edge to walk along. Widening `ef`
    /// buys a little and then stops buying anything, because the ceiling is a
    /// property of the graph and not of the search.
    ///
    /// Measured in this repository — recall@10 against an exact scan over
    /// 5000 points in 64 dimensions, 25 Gaussian clusters, `M = 16`,
    /// `ef_construction = 200`, 400 held-out queries:
    ///
    /// | `ef_search` | naive top-M | heuristic |
    /// |---:|---:|---:|
    /// | 10 | 0.8910 | 0.9645 |
    /// | 16 | 0.9163 | 0.9900 |
    /// | 32 | 0.9338 | 0.9995 |
    /// | 64 | 0.9450 | **1.0000** |
    /// | 128 | 0.9500 | 1.0000 |
    ///
    /// ```text
    /// hnswctl recall --n 5000 --dims 64 --clusters 25 --metric l2 \
    ///         --queries 400 --k 10 --ef 10,16,32,64,128 --selection naive
    /// ```
    ///
    /// The heuristic is exact at a beam of 64. Naive selection is still short
    /// of 0.95 at twice that cost and is visibly flattening — the missing
    /// neighbours are not slightly further down the beam, they are behind an
    /// edge that does not exist. On some seeds it also strands vectors
    /// outright; `tests/heuristic.rs` builds a case where 58 of 5000 are
    /// unreachable from the entry point, which no `ef_search` can fix.
    ///
    /// The heuristic (Algorithm 4) fixes it with one condition. Walk the
    /// candidates nearest-first and keep candidate `e` only if
    ///
    /// ```text
    /// d(e, base) < d(e, r)   for every r already kept
    /// ```
    ///
    /// — that is, keep `e` only if `base` is a *better* connection for it
    /// than any neighbour already chosen. A second `A`-point sitting right
    /// behind the first one fails: it is closer to the already-kept
    /// `A`-neighbour than to `base`, so the edge would be redundant, and it
    /// is discarded. But the nearest point in `B` passes trivially — nothing
    /// kept so far is anywhere near it — so the long link across the gap
    /// survives even though hundreds of `A` points are nearer.
    ///
    /// The effect is a relative neighbourhood graph rather than a k-nearest
    /// graph: links spread over *directions* instead of piling into the
    /// densest one. Every node keeps a route out of its own cluster, so the
    /// greedy walk always has a downhill step available until it is genuinely
    /// at the nearest point. That is why HNSW works on clustered data at all.
    ///
    /// Candidates rejected by the condition are not thrown away; if the
    /// diverse set comes up short of `m` they are appended nearest-first
    /// (the paper's `keepPrunedConnections`), so degree never collapses in
    /// uniformly dense regions.
    ///
    /// Note that the condition is *strictly* less than, so exact duplicates
    /// are never diverse with respect to each other and selection degenerates
    /// to nearest-`m`. The order in which those ties resolve then decides the
    /// whole topology; [`tie_break_from`] is what stops that from going
    /// badly wrong.
    fn select_neighbors(&self, candidates: &[Scored], m: usize, base: NodeId) -> Vec<NodeId> {
        if candidates.len() <= m {
            return candidates.iter().map(|c| c.node).collect();
        }

        // Re-order runs of equal distance from `base`'s own point of view.
        // See `tie_break_from`. Skipped entirely when there are no ties,
        // which is the case for every real float dataset.
        let reordered;
        let candidates: &[Scored] = if candidates.windows(2).any(|w| w[0].dist == w[1].dist) {
            let mut owned = candidates.to_vec();
            owned.sort_unstable_by(|a, b| {
                a.dist
                    .total_cmp(&b.dist)
                    .then_with(|| tie_break_from(base, a.node).cmp(&tie_break_from(base, b.node)))
            });
            reordered = owned;
            &reordered
        } else {
            candidates
        };

        if self.params.selection == Selection::NaiveTopM {
            return candidates.iter().take(m).map(|c| c.node).collect();
        }

        let mut kept: Vec<Scored> = Vec::with_capacity(m);
        let mut discarded: Vec<Scored> = Vec::new();

        for &candidate in candidates {
            if kept.len() >= m {
                break;
            }
            let cand_vec = self.vector(candidate.node);
            let diverse = kept.iter().all(|k| {
                candidate.dist
                    < self
                        .dist
                        .distance_preprocessed(cand_vec, self.vector(k.node))
            });
            if diverse {
                kept.push(candidate);
            } else {
                discarded.push(candidate);
            }
        }

        // keepPrunedConnections: top up with the nearest rejects.
        for candidate in discarded {
            if kept.len() >= m {
                break;
            }
            kept.push(candidate);
        }

        kept.sort_unstable();
        kept.iter().map(|c| c.node).collect()
    }

    // -- search ------------------------------------------------------------

    /// Beam search on a single layer. Algorithm 2 of the paper.
    ///
    /// Two heaps do the work. `candidates` is a min-heap of frontier nodes to
    /// expand; `results` is a max-heap holding the best `ef` seen so far, so
    /// its top is the one to evict. The loop stops as soon as the nearest
    /// unexpanded candidate is further away than the *worst* result already
    /// held — at that point no unexplored node can improve the answer,
    /// because getting to one requires passing through a node that is
    /// already too far.
    ///
    /// Returns the result set sorted ascending by distance. Tombstoned nodes
    /// are included: they are still part of the graph and dropping them here
    /// would sever the paths that run through them.
    fn search_layer(
        &self,
        query: &[f32],
        entries: &[NodeId],
        ef: usize,
        level: usize,
        scratch: &mut Searcher,
        stats: &mut SearchStats,
    ) -> Vec<Scored> {
        scratch.ensure(self.ids.len());
        scratch.begin();

        let ef = ef.max(1);
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<Scored> = BinaryHeap::with_capacity(ef + 1);
        let (mut hops, mut visited) = (0usize, 0usize);

        for &e in entries {
            if !scratch.visit(e) {
                continue;
            }
            visited += 1;
            let scored = Scored {
                dist: self.dist_to(query, e),
                node: e,
            };
            if let Some(trace) = stats.trace.as_mut() {
                trace.push(Visit {
                    node: e,
                    level: level as u8,
                });
            }
            candidates.push(Reverse(scored));
            results.push(scored);
        }
        while results.len() > ef {
            results.pop();
        }

        while let Some(Reverse(nearest)) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |s| s.dist);
            if results.len() >= ef && nearest.dist > worst {
                break;
            }
            hops += 1;

            // Snapshot the adjacency length first: `self` stays immutably
            // borrowed for the whole loop, which is fine, but the neighbour
            // slice is re-fetched each step so the borrow does not outlive
            // the distance call.
            let degree = self.links[nearest.node as usize]
                .get(level)
                .map_or(0, Vec::len);
            for i in 0..degree {
                let neighbor = self.links[nearest.node as usize][level][i];
                if !scratch.visit(neighbor) {
                    continue;
                }
                visited += 1;
                let dist = self.dist_to(query, neighbor);
                if let Some(trace) = stats.trace.as_mut() {
                    trace.push(Visit {
                        node: neighbor,
                        level: level as u8,
                    });
                }
                let worst = results.peek().map_or(f32::INFINITY, |s| s.dist);
                if results.len() < ef || dist < worst {
                    let scored = Scored {
                        dist,
                        node: neighbor,
                    };
                    candidates.push(Reverse(scored));
                    results.push(scored);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        stats.hops += hops;
        stats.visited += visited;
        stats.distance_computations += visited;
        if stats.collect_layers {
            stats.layers.push(LayerStats {
                level,
                hops,
                visited,
            });
        }
        results.into_sorted_vec()
    }

    /// Find the `k` nearest live neighbours of `query` using the index's
    /// default `ef_search`.
    ///
    /// Allocates a fresh [`Searcher`] each call. For anything throughput
    /// sensitive, keep one and call [`Hnsw::search_with`].
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        let mut scratch = Searcher::new();
        self.search_with(&mut scratch, query, k, self.params.ef_search)
    }

    /// Find the `k` nearest live neighbours with an explicit beam width and a
    /// caller-owned scratch buffer. This is the hot path.
    pub fn search_with(
        &self,
        scratch: &mut Searcher,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>> {
        let mut stats = SearchStats::default();
        self.run_search(scratch, query, k, ef, &mut stats)
    }

    /// Search and report what the traversal cost.
    pub fn search_stats(
        &self,
        scratch: &mut Searcher,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<(Vec<Neighbor>, SearchStats)> {
        let mut stats = SearchStats {
            collect_layers: true,
            ..SearchStats::default()
        };
        let hits = self.run_search(scratch, query, k, ef, &mut stats)?;
        Ok((hits, stats))
    }

    /// Search and record every node visit, in order, per layer.
    ///
    /// This is what the browser demo draws. It costs one push per visit, so
    /// it is not the path to benchmark.
    pub fn search_traced(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<(Vec<Neighbor>, SearchStats)> {
        let mut scratch = Searcher::new();
        let mut stats = SearchStats {
            collect_layers: true,
            trace: Some(Vec::new()),
            ..SearchStats::default()
        };
        let hits = self.run_search(&mut scratch, query, k, ef, &mut stats)?;
        Ok((hits, stats))
    }

    fn run_search(
        &self,
        scratch: &mut Searcher,
        query: &[f32],
        k: usize,
        ef: usize,
        stats: &mut SearchStats,
    ) -> Result<Vec<Neighbor>> {
        if query.len() != self.dims {
            return Err(Error::DimensionMismatch {
                expected: self.dims,
                found: query.len(),
            });
        }
        let Some(entry) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let mut q = query.to_vec();
        self.dist.preprocess(&mut q);

        // Greedy descent through the sparse layers: beam of one, no
        // backtracking. This is where the logarithmic behaviour comes from.
        let mut current = vec![entry];
        for lc in (1..=self.max_level).rev() {
            let found = self.search_layer(&q, &current, 1, lc, scratch, stats);
            if let Some(best) = found.first() {
                current = vec![best.node];
            }
        }

        let ef = self.effective_ef(ef.max(k));
        let found = self.search_layer(&q, &current, ef, 0, scratch, stats);

        Ok(found
            .into_iter()
            .filter(|s| !self.deleted[s.node as usize])
            .take(k)
            .map(|s| Neighbor {
                id: self.ids[s.node as usize],
                distance: s.dist,
            })
            .collect())
    }

    /// Widen the beam to compensate for tombstones.
    ///
    /// Deleted nodes stay in the graph and stay in the layer-0 result set;
    /// they are only dropped at the very end. So if a third of the index is
    /// tombstoned, roughly a third of the beam is wasted and `k` results may
    /// not survive the filter. Scaling `ef` by `total / live` restores the
    /// expected number of survivors, and the cap keeps it from exceeding a
    /// full scan.
    fn effective_ef(&self, ef: usize) -> usize {
        let total = self.ids.len();
        let live = self.len();
        let cap = total.max(1);
        if live == 0 || live == total {
            return ef.min(cap);
        }
        let scaled = (ef as u128 * total as u128 / live as u128) as usize;
        scaled.min(cap).max(ef.min(cap))
    }

    // -- deletion ----------------------------------------------------------

    /// Tombstone an id. Returns `true` if it was present and live.
    ///
    /// **The external id stays reserved until [`Hnsw::compact`] runs.** The
    /// node is still in the graph and still in the id map, so the id sits in
    /// a third state that is neither present nor free: [`Hnsw::contains`]
    /// reports it absent, [`Hnsw::is_deleted`] reports it deleted, and
    /// [`Hnsw::insert`] refuses it with [`Error::DuplicateId`]. Compaction is
    /// what releases it.
    ///
    /// # Why not remove the node?
    ///
    /// Because HNSW has no cheap way to. Deleting node `x` means every
    /// neighbour that pointed at `x` loses an edge, and those edges were
    /// chosen by the diversity heuristic against the *whole* candidate set
    /// that existed at their insertion time — a set that is not recorded. To
    /// repair the hole properly you would have to re-run
    /// [`Hnsw::select_neighbors`] for each of `x`'s neighbours over freshly
    /// searched candidates, which costs about as much as inserting them all
    /// again. Skipping the repair and just cutting the edges is worse than it
    /// sounds: `x` may have been the only bridge between two regions, and
    /// removing it silently partitions the graph, which shows up much later
    /// as unexplainably bad recall in one corner of the space.
    ///
    /// So the node stays, keeps carrying traffic, and is filtered out of
    /// results. The costs are honest and bounded:
    ///
    /// - memory is not reclaimed until [`Hnsw::compact`];
    /// - searches still spend distance computations on dead nodes, which
    ///   [`Hnsw::effective_ef`] compensates for by widening the beam;
    /// - so a heavily tombstoned index gets steadily slower, not less
    ///   accurate.
    ///
    /// Every production HNSW — hnswlib, Qdrant, pgvector — makes this same
    /// trade and pairs it with a background rebuild. [`Hnsw::compact`] is the
    /// rebuild.
    pub fn delete(&mut self, id: u64) -> bool {
        let Some(&node) = self.id_map.get(&id) else {
            return false;
        };
        if self.deleted[node as usize] {
            return false;
        }
        self.deleted[node as usize] = true;
        self.deleted_count += 1;
        true
    }

    /// Whether an internal node is tombstoned.
    #[must_use]
    pub fn is_node_deleted(&self, node: NodeId) -> bool {
        self.deleted[node as usize]
    }

    /// The PRNG state, so that a save/load round trip does not perturb the
    /// random stream future inserts will draw from.
    #[must_use]
    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.state()
    }

    /// Whether `id` is present but tombstoned.
    #[must_use]
    pub fn is_deleted(&self, id: u64) -> bool {
        self.id_map
            .get(&id)
            .is_some_and(|&n| self.deleted[n as usize])
    }

    /// Rebuild the graph without the tombstoned nodes and return how many
    /// were reclaimed.
    ///
    /// This is a full reinsertion of every live vector in id order, from a
    /// PRNG re-seeded with the same seed — so the result is exactly the index
    /// you would have got by never inserting the deleted vectors at all, and
    /// is reproducible. It costs a full build; run it off the serving path.
    ///
    /// Note that the vectors reinserted are the *stored* ones, which have
    /// already been through [`Distance::preprocess`]. Preprocessing is
    /// idempotent for the built-in metrics (normalising a unit vector is a
    /// no-op), so this is lossless for them; a custom non-idempotent
    /// `preprocess` would not survive a compaction.
    pub fn compact(&mut self) -> usize {
        if self.deleted_count == 0 {
            return 0;
        }
        let reclaimed = self.deleted_count;

        let mut rebuilt = Hnsw {
            dims: self.dims,
            params: self.params,
            dist: self.dist.clone(),
            vectors: Vec::with_capacity(self.len() * self.dims),
            ids: Vec::with_capacity(self.len()),
            id_map: IdMap::default(),
            levels: Vec::with_capacity(self.len()),
            links: Vec::with_capacity(self.len()),
            deleted: Vec::with_capacity(self.len()),
            deleted_count: 0,
            entry: None,
            max_level: 0,
            rng: Rng::new(self.params.seed),
            scratch: Searcher::new(),
        };

        for node in 0..self.ids.len() {
            if self.deleted[node] {
                continue;
            }
            let id = self.ids[node];
            let start = node * self.dims;
            let vector = self.vectors[start..start + self.dims].to_vec();
            // Cannot fail: same dims, and ids were unique in the source.
            let _ = rebuilt.insert(id, &vector);
        }

        *self = rebuilt;
        reclaimed
    }

    // -- diagnostics -------------------------------------------------------

    /// Check every structural invariant and describe the first violation.
    ///
    /// Cheap enough for tests and for a CLI `--verify` flag; `O(n * M)`.
    ///
    /// 1. Bookkeeping vectors all agree on the node count.
    /// 2. The id map is a bijection onto the nodes.
    /// 3. No node's adjacency reaches above its own top layer.
    /// 4. Degree caps hold on every layer.
    /// 5. No self-loops, no duplicate edges, no out-of-range targets.
    /// 6. Every node has at least one outgoing edge at layer 0 — the
    ///    no-stranded-node guarantee that one-sided pruning buys.
    /// 7. Every neighbour on layer `l` actually reaches layer `l`.
    /// 8. The entry point sits on the top layer.
    pub fn validate(&self) -> core::result::Result<(), String> {
        let n = self.ids.len();
        if self.levels.len() != n || self.links.len() != n || self.deleted.len() != n {
            return Err("node bookkeeping vectors disagree on length".into());
        }
        if self.vectors.len() != n * self.dims {
            return Err("vector arena length does not match node count".into());
        }
        if self.id_map.len() != n {
            return Err(format!(
                "id map holds {} entries for {n} nodes",
                self.id_map.len()
            ));
        }
        for (node, &id) in self.ids.iter().enumerate() {
            match self.id_map.get(&id) {
                Some(&mapped) if mapped as usize == node => {}
                _ => return Err(format!("id {id} does not map back to node {node}")),
            }
        }

        for node in 0..n {
            let level = self.levels[node] as usize;
            if level > MAX_LEVEL {
                return Err(format!("node {node} has level {level} above the cap"));
            }
            if level > self.max_level {
                return Err(format!(
                    "node {node} has level {level} above index max_level {}",
                    self.max_level
                ));
            }
            if self.links[node].len() != level + 1 {
                return Err(format!(
                    "node {node} has {} link layers for level {level}",
                    self.links[node].len()
                ));
            }
            for (lc, list) in self.links[node].iter().enumerate() {
                let cap = self.params.max_degree(lc);
                if lc == 0 && list.is_empty() && n > 1 {
                    return Err(format!("node {node} has no outgoing edges at layer 0"));
                }
                if list.len() > cap {
                    return Err(format!(
                        "node {node} has degree {} at layer {lc}, cap is {cap}",
                        list.len()
                    ));
                }
                let mut seen = list.clone();
                seen.sort_unstable();
                seen.dedup();
                if seen.len() != list.len() {
                    return Err(format!("node {node} has duplicate links at layer {lc}"));
                }
                for &other in list {
                    if other as usize == node {
                        return Err(format!("node {node} links to itself at layer {lc}"));
                    }
                    if other as usize >= n {
                        return Err(format!("node {node} links to missing node {other}"));
                    }
                    if (self.levels[other as usize] as usize) < lc {
                        return Err(format!(
                            "node {node} links at layer {lc} to node {other}, which tops out at {}",
                            self.levels[other as usize]
                        ));
                    }
                }
            }
        }

        match self.entry {
            None if n == 0 => {}
            None => return Err("non-empty index has no entry point".into()),
            Some(e) if (e as usize) < n && self.levels[e as usize] as usize == self.max_level => {}
            Some(e) => return Err(format!("entry point {e} is not on the top layer")),
        }
        Ok(())
    }

    /// Sizes of the connected components of layer 0, largest first,
    /// following edges in either direction.
    ///
    /// A healthy index returns exactly one component covering every node.
    /// More than one means a region is cut off from the rest of the graph
    /// entirely — the failure mode the selection heuristic exists to prevent.
    /// For the stricter question of what a *search* can reach, see
    /// [`Hnsw::reachable_from_entry`].
    #[must_use]
    pub fn components(&self) -> Vec<usize> {
        // Union-find rather than a traversal: a depth-first walk would only
        // follow out-edges, which silently answers the directed question
        // instead of the undirected one.
        fn find(parent: &mut [NodeId], mut x: NodeId) -> NodeId {
            while parent[x as usize] != x {
                parent[x as usize] = parent[parent[x as usize] as usize];
                x = parent[x as usize];
            }
            x
        }

        let n = self.ids.len();
        let mut parent: Vec<NodeId> = (0..n as NodeId).collect();
        for node in 0..n as NodeId {
            for &other in self.neighbors(node, 0) {
                let (a, b) = (find(&mut parent, node), find(&mut parent, other));
                if a != b {
                    parent[a as usize] = b;
                }
            }
        }

        let mut sizes = vec![0usize; n];
        for node in 0..n as NodeId {
            let root = find(&mut parent, node);
            sizes[root as usize] += 1;
        }
        let mut sizes: Vec<usize> = sizes.into_iter().filter(|&s| s > 0).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// Whether layer 0 is a single connected component.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.components().len() <= 1
    }

    /// How many nodes a search could reach on `level`, following directed
    /// edges outward from the entry point.
    ///
    /// This is the connectivity that decides recall. [`Hnsw::components`]
    /// treats every edge as two-way, which flatters the graph: an edge
    /// `a -> b` that lost its return half during pruning still puts `a` and
    /// `b` in one component, but a walk arriving at `b` cannot use it. If
    /// this number falls short of [`Hnsw::node_count`], some vectors are in
    /// the index and unfindable no matter how wide the beam.
    #[must_use]
    pub fn reachable_from_entry(&self, level: usize) -> usize {
        let Some(entry) = self.entry else {
            return 0;
        };
        if (self.levels[entry as usize] as usize) < level {
            return 0;
        }
        let mut seen = vec![false; self.ids.len()];
        seen[entry as usize] = true;
        let mut stack = vec![entry];
        let mut count = 0usize;
        while let Some(node) = stack.pop() {
            count += 1;
            for &next in self.neighbors(node, level) {
                if !seen[next as usize] {
                    seen[next as usize] = true;
                    stack.push(next);
                }
            }
        }
        count
    }

    /// Mean layer-0 degree.
    #[must_use]
    pub fn mean_degree(&self) -> f64 {
        if self.links.is_empty() {
            return 0.0;
        }
        let total: usize = self.links.iter().map(|l| l[0].len()).sum();
        total as f64 / self.links.len() as f64
    }
}

// The internal-only flag lives on `SearchStats` so that `search_layer` has a
// single stats parameter instead of three.
impl SearchStats {
    /// Whether per-layer rows are being collected.
    #[must_use]
    pub fn collects_layers(&self) -> bool {
        self.collect_layers
    }
}
