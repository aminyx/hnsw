//! The recall/throughput harness.
//!
//! An ANN index has two numbers and they trade against each other. Reporting
//! either one alone is meaningless: a linear scan has perfect recall, an
//! index that returns the first `k` ids it sees is infinitely fast, and both
//! are useless. What a real implementation has to show is the *curve* — how
//! much recall a given query budget buys — which is what
//! [`recall_sweep`] produces.
//!
//! Timing is wall clock over a fixed query set, single threaded, with a warm
//! scratch buffer, and reported as the **fastest** of [`TIMED_PASSES`] passes
//! after a warm-up. Taking the minimum rather than the mean is deliberate:
//! the quantity of interest is how long the work takes, and every source of
//! noise available to a benchmark — scheduler preemption, another container
//! on the same host, a page fault, a frequency dip — can only ever make a
//! pass *slower*. The mean measures the machine's mood; the minimum measures
//! the code. Without it the sweep can report a wider beam as faster than a
//! narrow one, which is not a plausible property of the index and is a sure
//! sign the numbers are noise.

#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use crate::brute::recall_at_k;
use crate::distance::Distance;
use crate::error::Result;
use crate::index::{Hnsw, Searcher};

/// How many timed passes each row runs before reporting the fastest.
pub const TIMED_PASSES: usize = 9;

/// One row of the recall/throughput sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallRow {
    /// Beam width used for this row.
    pub ef_search: usize,
    /// Mean recall@k against the exact baseline, in `[0, 1]`.
    pub recall: f64,
    /// Queries per second, single threaded.
    pub qps: f64,
    /// Mean microseconds per query.
    pub latency_us: f64,
    /// Mean distinct nodes evaluated per query.
    pub visited: f64,
    /// Mean adjacency expansions per query.
    pub hops: f64,
    /// Fraction of the index touched per query. The headline number: this is
    /// how much of the linear scan the index avoided.
    pub scanned_fraction: f64,
}

/// Run the same query set at several beam widths and report the trade-off.
///
/// `queries` is a flat `q * dims` buffer; `truth[i]` is the exact top-`k` id
/// list for query `i`, from [`crate::brute::ground_truth`].
pub fn recall_sweep<D: Distance>(
    index: &Hnsw<D>,
    queries: &[f32],
    truth: &[Vec<u64>],
    k: usize,
    ef_values: &[usize],
) -> Result<Vec<RecallRow>> {
    let dims = index.dims();
    let count = queries.len() / dims.max(1);
    let mut scratch = Searcher::new();
    let mut rows = Vec::with_capacity(ef_values.len());

    for &ef in ef_values {
        // Warm-up: fault in the scratch array and prime the caches so the
        // first measured row is not systematically slower than the rest.
        for i in 0..count.min(32) {
            let _ = index.search_with(&mut scratch, &queries[i * dims..(i + 1) * dims], k, ef)?;
        }

        let mut recall_total = 0.0f64;
        let mut visited_total = 0usize;
        let mut hops_total = 0usize;

        let mut elapsed = f64::INFINITY;
        for pass in 0..TIMED_PASSES {
            let start = Instant::now();
            let mut pass_recall = 0.0f64;
            for i in 0..count {
                let hits =
                    index.search_with(&mut scratch, &queries[i * dims..(i + 1) * dims], k, ef)?;
                pass_recall += recall_at_k(&hits, &truth[i]);
            }
            elapsed = elapsed.min(start.elapsed().as_secs_f64());
            if pass == 0 {
                // Every pass returns the same results, so recall only needs
                // computing once — but it is computed inside the timed loop
                // so that the measurement includes the cost of handing the
                // results back, not just of finding them.
                recall_total = pass_recall;
            }
        }

        // Counters come from a second, unmeasured pass so that the stats
        // bookkeeping never contaminates the timing.
        for i in 0..count {
            let (_, stats) =
                index.search_stats(&mut scratch, &queries[i * dims..(i + 1) * dims], k, ef)?;
            visited_total += stats.visited;
            hops_total += stats.hops;
        }

        let n = count.max(1) as f64;
        let visited = visited_total as f64 / n;
        rows.push(RecallRow {
            ef_search: ef,
            recall: recall_total / n,
            qps: if elapsed > 0.0 { n / elapsed } else { f64::NAN },
            latency_us: elapsed * 1e6 / n,
            visited,
            hops: hops_total as f64 / n,
            scanned_fraction: visited / index.node_count().max(1) as f64,
        });
    }
    Ok(rows)
}

/// Time a full build of `n` vectors already held in a flat buffer.
///
/// Returns the elapsed seconds; the index is left in `index`.
pub fn time_build<D: Distance>(index: &mut Hnsw<D>, first_id: u64, vectors: &[f32]) -> Result<f64> {
    let start = Instant::now();
    index.insert_batch(first_id, vectors)?;
    Ok(start.elapsed().as_secs_f64())
}

/// Queries per second of an exhaustive scan over the same data, for scale.
pub fn brute_force_qps<D: Distance>(
    base: &crate::brute::BruteForce<D>,
    queries: &[f32],
    k: usize,
) -> Result<f64> {
    let dims = base.dims();
    let count = queries.len() / dims.max(1);
    if count == 0 {
        return Ok(f64::NAN);
    }
    // The whole query set, not a sample: this number is the denominator of
    // every speedup the README quotes, and a denominator timed over fewer
    // queries than the numerator is not a comparison.
    let mut elapsed = f64::INFINITY;
    for _ in 0..TIMED_PASSES {
        let start = Instant::now();
        for i in 0..count {
            let _ = base.search(&queries[i * dims..(i + 1) * dims], k)?;
        }
        elapsed = elapsed.min(start.elapsed().as_secs_f64());
    }
    Ok(if elapsed > 0.0 {
        count as f64 / elapsed
    } else {
        f64::NAN
    })
}

/// Render a sweep as a GitHub-flavoured Markdown table.
#[must_use]
pub fn markdown_table(rows: &[RecallRow], k: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "| `ef_search` | recall@{k} | QPS | mean latency | nodes visited | % of index scanned |\n"
    ));
    out.push_str("|---:|---:|---:|---:|---:|---:|\n");
    for r in rows {
        out.push_str(&format!(
            "| {} | {:.4} | {:.0} | {:.0} µs | {:.0} | {:.2}% |\n",
            r.ef_search,
            r.recall,
            r.qps,
            r.latency_us,
            r.visited,
            r.scanned_fraction * 100.0
        ));
    }
    out
}

/// Render a sweep as a fixed-width table for the terminal.
#[must_use]
pub fn text_table(rows: &[RecallRow], k: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>10}  {:>9}  {:>10}  {:>11}  {:>9}  {:>9}\n",
        "ef_search",
        format!("recall@{k}"),
        "QPS",
        "latency",
        "visited",
        "scanned"
    ));
    out.push_str(&format!("{}\n", "-".repeat(68)));
    for r in rows {
        out.push_str(&format!(
            "{:>10}  {:>9.4}  {:>10.0}  {:>8.0} µs  {:>9.0}  {:>8.2}%\n",
            r.ef_search,
            r.recall,
            r.qps,
            r.latency_us,
            r.visited,
            r.scanned_fraction * 100.0
        ));
    }
    out
}
