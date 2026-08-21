//! The browser-facing ABI.
//!
//! The whole index — construction, graph, beam search, statistics — is
//! compiled to WebAssembly and runs in the page. There is no server, no API
//! call and no precomputed answer file: the demo builds a real HNSW graph
//! over ~25 000 word vectors in the tab, then serves live queries from it.
//!
//! Deliberately plain `extern "C"` with manual memory, exactly like a C ABI,
//! so the build is nothing but `cargo build --target wasm32-unknown-unknown`
//! — no `wasm-bindgen`, no `wasm-pack`, no npm, nothing to install.
//!
//! Every result is a single allocation laid out as:
//!
//! ```text
//! [ tag: u32 LE ][ len: u32 LE ][ len bytes of payload ]
//!   tag 0 = success, payload is UTF-8 JSON
//!   tag 1 = failure, payload is a UTF-8 message
//! ```
//!
//! JavaScript reads the eight-byte header, copies the payload out of the Wasm
//! heap, then calls `hnsw_free`.

#![allow(clippy::missing_panics_doc)]

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::fmt::Write as _;

use crate::distance::{Distance, Metric};
use crate::embed::{demo_words, embed, embed_into, DEMO_DIMS};
use crate::index::{Hnsw, NodeId, Params};

const HEADER: usize = 8;

/// Upper bound on traced visits reported to the page, so a very wide beam
/// cannot produce a multi-megabyte JSON blob.
const MAX_TRACE: usize = 4000;

struct Demo {
    index: Hnsw<Metric>,
    words: Vec<&'static str>,
    /// How many words have been inserted so far. HNSW is a genuinely
    /// incremental structure — there is no batch phase and no finalisation —
    /// so the page can build it a slice at a time and stay responsive.
    inserted: usize,
}

thread_local! {
    static DEMO: RefCell<Option<Demo>> = const { RefCell::new(None) };
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Allocate `len` bytes for the caller to write query text into.
///
/// # Safety
/// The caller must pass the same `len` back to `hnsw_free_input`.
#[no_mangle]
pub unsafe extern "C" fn hnsw_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let Ok(layout) = Layout::from_size_align(len, 1) else {
        return std::ptr::null_mut();
    };
    alloc(layout)
}

/// Free a buffer obtained from `hnsw_alloc`.
///
/// # Safety
/// `ptr` and `len` must come from a matching `hnsw_alloc` call.
#[no_mangle]
pub unsafe extern "C" fn hnsw_free_input(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let Ok(layout) = Layout::from_size_align(len, 1) else {
        return;
    };
    dealloc(ptr, layout);
}

/// Free a result block returned by any of the functions below.
///
/// # Safety
/// `ptr` must be a pointer one of them returned, freed at most once.
#[no_mangle]
pub unsafe extern "C" fn hnsw_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let len = u32::from_le_bytes([*ptr.add(4), *ptr.add(5), *ptr.add(6), *ptr.add(7)]) as usize;
    let Ok(layout) = Layout::from_size_align(HEADER + len, 1) else {
        return;
    };
    dealloc(ptr, layout);
}

/// Build the `[tag][len][payload]` block the JavaScript side expects.
unsafe fn result_block(tag: u32, payload: &[u8]) -> *mut u8 {
    let total = HEADER + payload.len();
    let Ok(layout) = Layout::from_size_align(total, 1) else {
        return std::ptr::null_mut();
    };
    let out = alloc(layout);
    if out.is_null() {
        return out;
    }
    std::ptr::copy_nonoverlapping(tag.to_le_bytes().as_ptr(), out, 4);
    std::ptr::copy_nonoverlapping((payload.len() as u32).to_le_bytes().as_ptr(), out.add(4), 4);
    if !payload.is_empty() {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), out.add(HEADER), payload.len());
    }
    out
}

unsafe fn text_of(ptr: *const u8, len: usize) -> Result<String, String> {
    if ptr.is_null() || len == 0 {
        return Ok(String::new());
    }
    let bytes = std::slice::from_raw_parts(ptr, len);
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|_| "query is not valid UTF-8".to_string())
}

// ---------------------------------------------------------------------------
// JSON, written by hand
// ---------------------------------------------------------------------------

fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_num(out: &mut String, v: f64, places: usize) {
    if v.is_finite() {
        let _ = write!(out, "{v:.places$}");
    } else {
        out.push_str("null");
    }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

/// Start a build: create an empty index and remember the parameters.
///
/// Returns JSON with the number of words waiting to be inserted. Nothing is
/// inserted yet — call `hnsw_build_step` until it reports `total`.
///
/// # Safety
/// Safe to call at any time; any previous index is dropped and replaced.
#[no_mangle]
pub unsafe extern "C" fn hnsw_build_begin(m: u32, ef_construction: u32, seed: u32) -> *mut u8 {
    match begin_inner(m as usize, ef_construction as usize, u64::from(seed)) {
        Ok(json) => result_block(0, json.as_bytes()),
        Err(message) => result_block(1, message.as_bytes()),
    }
}

fn begin_inner(m: usize, ef_construction: usize, seed: u64) -> Result<String, String> {
    let words = demo_words();
    if words.is_empty() {
        return Err("the embedded word list is empty".into());
    }
    let params = Params::default()
        .with_m(m.clamp(4, 48))
        .with_ef_construction(ef_construction.clamp(16, 400))
        .with_seed(seed);
    let index =
        Hnsw::new(DEMO_DIMS, Metric::Cosine, params).map_err(|e| format!("bad parameters: {e}"))?;

    let mut json = String::with_capacity(160);
    let _ = write!(json, "{{\"total\":{}", words.len());
    let _ = write!(json, ",\"dims\":{DEMO_DIMS}");
    let _ = write!(json, ",\"m\":{}", params.m);
    let _ = write!(json, ",\"m_max0\":{}", params.m_max0);
    let _ = write!(json, ",\"ef_construction\":{}", params.ef_construction);
    let _ = write!(json, ",\"seed\":{seed}}}");

    DEMO.with(|slot| {
        *slot.borrow_mut() = Some(Demo {
            index,
            words,
            inserted: 0,
        });
    });
    Ok(json)
}

/// Insert up to `count` more words. Returns the running total inserted.
///
/// The page calls this in slices of a few hundred between animation frames,
/// so the tab never blocks for more than a frame or two and the progress it
/// shows is the real thing rather than an animation.
///
/// # Safety
/// Does nothing unless `hnsw_build_begin` has been called.
#[no_mangle]
pub unsafe extern "C" fn hnsw_build_step(count: u32) -> u32 {
    DEMO.with(|slot| {
        let mut borrowed = slot.borrow_mut();
        let Some(demo) = borrowed.as_mut() else {
            return 0;
        };
        let end = (demo.inserted + count as usize).min(demo.words.len());
        let mut buffer = vec![0.0f32; DEMO_DIMS];
        while demo.inserted < end {
            let i = demo.inserted;
            embed_into(demo.words[i], &mut buffer);
            // Cannot fail: fixed width, and ids are the loop counter.
            let _ = demo.index.insert(i as u64, &buffer);
            demo.inserted += 1;
        }
        demo.inserted as u32
    })
}

/// Describe the graph as it currently stands: layer populations, edges,
/// degree, connectivity.
///
/// Valid at any point during a build, not only at the end, because there is
/// no point during a build at which the index is not a usable index.
///
/// # Safety
/// Fails cleanly if no build has been started.
#[no_mangle]
pub unsafe extern "C" fn hnsw_build_info() -> *mut u8 {
    match info_inner() {
        Ok(json) => result_block(0, json.as_bytes()),
        Err(message) => result_block(1, message.as_bytes()),
    }
}

fn info_inner() -> Result<String, String> {
    DEMO.with(|slot| {
        let borrowed = slot.borrow();
        let demo = borrowed.as_ref().ok_or("no build has been started")?;
        let index = &demo.index;
        let layers = index.layer_sizes();
        let components = index.components();

        let mut json = String::with_capacity(512);
        json.push('{');
        let _ = write!(json, "\"words\":{}", index.node_count());
        let _ = write!(json, ",\"total\":{}", demo.words.len());
        let _ = write!(json, ",\"dims\":{DEMO_DIMS}");
        let _ = write!(json, ",\"m\":{}", index.params().m);
        let _ = write!(json, ",\"m_max0\":{}", index.params().m_max0);
        let _ = write!(
            json,
            ",\"ef_construction\":{}",
            index.params().ef_construction
        );
        let _ = write!(json, ",\"seed\":{}", index.params().seed);
        let _ = write!(json, ",\"max_level\":{}", index.max_level());
        json.push_str(",\"layers\":[");
        for (i, size) in layers.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let _ = write!(json, "{size}");
        }
        json.push(']');
        let _ = write!(json, ",\"edges\":{}", index.edge_count());
        json.push_str(",\"mean_degree\":");
        json_num(&mut json, index.mean_degree(), 2);
        let _ = write!(json, ",\"components\":{}", components.len());
        let _ = write!(
            json,
            ",\"largest_component\":{}",
            components.first().copied().unwrap_or(0)
        );
        let _ = write!(json, ",\"reachable\":{}", index.reachable_from_entry(0));
        let _ = write!(json, ",\"metric\":\"{}\"", index.metric().name());
        json.push('}');
        Ok(json)
    })
}

/// Query the demo index with a UTF-8 string.
///
/// # Safety
/// `ptr`/`len` must describe UTF-8 bytes allocated by `hnsw_alloc`.
#[no_mangle]
pub unsafe extern "C" fn hnsw_search(ptr: *const u8, len: usize, k: u32, ef: u32) -> *mut u8 {
    let text = match text_of(ptr, len) {
        Ok(t) => t,
        Err(message) => return result_block(1, message.as_bytes()),
    };
    match search_inner(&text, k as usize, ef as usize) {
        Ok(json) => result_block(0, json.as_bytes()),
        Err(message) => result_block(1, message.as_bytes()),
    }
}

fn search_inner(text: &str, k: usize, ef: usize) -> Result<String, String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("type something to search for".into());
    }
    let k = k.clamp(1, 50);
    // A beam narrower than `k` cannot return `k` results, so the index widens
    // it. Report the width actually used, not the one that was asked for.
    let ef = ef.clamp(1, 2000).max(k);

    DEMO.with(|slot| {
        let borrowed = slot.borrow();
        let demo = borrowed
            .as_ref()
            .ok_or("the index has not been built yet")?;
        let index = &demo.index;

        let query = embed(trimmed, DEMO_DIMS);
        let (hits, stats) = index
            .search_traced(&query, k, ef)
            .map_err(|e| e.to_string())?;

        // The exact answer, by linear scan over the index's own storage, so
        // the page can show the recall of the query it just ran.
        let mut prepared = query.clone();
        index.metric().preprocess(&mut prepared);
        let mut exact: Vec<(NodeId, f32)> = (0..index.node_count() as NodeId)
            .map(|n| {
                (
                    n,
                    index
                        .metric()
                        .distance_preprocessed(&prepared, index.vector(n)),
                )
            })
            .collect();
        exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        exact.truncate(k);
        let exact_ids: Vec<u64> = exact.iter().map(|(n, _)| u64::from(*n)).collect();

        let found = hits.iter().filter(|h| exact_ids.contains(&h.id)).count();
        let recall = if exact_ids.is_empty() {
            1.0
        } else {
            found as f64 / exact_ids.len() as f64
        };

        let mut json = String::with_capacity(8192);
        json.push_str("{\"query\":");
        json_string(&mut json, trimmed);

        json.push_str(",\"results\":[");
        for (i, hit) in hits.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str("{\"word\":");
            json_string(&mut json, demo.words[hit.id as usize]);
            json.push_str(",\"distance\":");
            json_num(&mut json, f64::from(hit.distance), 4);
            let _ = write!(json, ",\"exact\":{}", exact_ids.contains(&hit.id));
            json.push('}');
        }
        json.push(']');

        json.push_str(",\"exact\":[");
        for (i, (node, dist)) in exact.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str("{\"word\":");
            json_string(&mut json, demo.words[*node as usize]);
            json.push_str(",\"distance\":");
            json_num(&mut json, f64::from(*dist), 4);
            json.push('}');
        }
        json.push(']');

        json.push_str(",\"recall\":");
        json_num(&mut json, recall, 4);

        json.push_str(",\"stats\":{");
        let _ = write!(json, "\"hops\":{}", stats.hops);
        let _ = write!(json, ",\"visited\":{}", stats.visited);
        let _ = write!(json, ",\"distances\":{}", stats.distance_computations);
        let _ = write!(json, ",\"total\":{}", index.node_count());
        let _ = write!(json, ",\"ef\":{ef}");
        json.push_str(",\"scanned\":");
        json_num(
            &mut json,
            stats.visited as f64 / index.node_count().max(1) as f64 * 100.0,
            2,
        );
        json.push_str(",\"layers\":[");
        for (i, layer) in stats.layers.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let _ = write!(
                json,
                "{{\"level\":{},\"hops\":{},\"visited\":{}}}",
                layer.level, layer.hops, layer.visited
            );
        }
        json.push_str("]}");

        // The traversal itself: distance to the query and layer, in visit
        // order. Plotted, this is the beam search converging.
        let trace = stats.trace.unwrap_or_default();
        let step = (trace.len() / MAX_TRACE).max(1);
        json.push_str(",\"trace\":[");
        for (written, visit) in trace.iter().step_by(step).enumerate() {
            if written > 0 {
                json.push(',');
            }
            json.push('[');
            json_num(
                &mut json,
                f64::from(
                    index
                        .metric()
                        .distance_preprocessed(&prepared, index.vector(visit.node)),
                ),
                4,
            );
            let _ = write!(json, ",{}]", visit.level);
        }
        let _ = write!(json, "],\"trace_step\":{step}");
        json.push('}');
        Ok(json)
    })
}
