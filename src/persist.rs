//! A single-file, versioned, checksummed on-disk format.
//!
//! ```text
//! offset  size            field
//! 0       8               magic "HNSWIDX\0"
//! 8       4               format version (u32 LE)
//! 12      1               metric tag
//! 13      1               selection tag
//! 14      2               reserved, must be zero
//! 16      4               dims
//! 20      4               m
//! 24      4               m_max0
//! 28      4               ef_construction
//! 32      4               ef_search
//! 36      8               ml (f64 LE bits)
//! 44      8               seed
//! 52      32              PRNG state (4 x u64)
//! 84      4               node count
//! 88      8               entry point (u64; u64::MAX means "none")
//! 96      4               max level
//! 100     n * 8           external ids
//! ...     n               top level per node (u8)
//! ...     n               tombstone flags (u8)
//! ...     n * dims * 4    vectors (f32 LE)
//! ...     variable        adjacency: per node, per layer, u32 count + u32 each
//! ...     4               CRC-32 (IEEE) of every preceding byte
//! ```
//!
//! Everything is little-endian and fixed-width, so the format is portable and
//! a reader can bounds-check every field before it allocates. The CRC is the
//! last four bytes and covers the whole payload, so truncation, bit rot and
//! partial writes are all caught by the same check.
//!
//! The PRNG state is persisted deliberately: an index that is loaded and then
//! has more vectors inserted continues on exactly the random stream it would
//! have had if it were never saved. Without that, save/load would silently
//! change the graph.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::distance::Metric;
use crate::error::{Error, Result};
use crate::index::{Hnsw, IdMap, NodeId, Params, Searcher, Selection};
use crate::rng::{Rng, MAX_LEVEL};

/// The eight magic bytes every index file starts with.
pub const MAGIC: &[u8; 8] = b"HNSWIDX\0";

/// Format version this build reads and writes.
pub const FORMAT_VERSION: u32 = 1;

/// Sentinel written when the index has no entry point.
const NO_ENTRY: u64 = u64::MAX;

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3), table-driven
// ---------------------------------------------------------------------------

const CRC_POLY: u32 = 0xEDB8_8320;

/// Build the 256-entry CRC table at compile time.
const fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ CRC_POLY
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC_TABLE: [u32; 256] = crc_table();

/// CRC-32 (IEEE) of a byte slice.
#[must_use]
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[idx];
    }
    !crc
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(Error::Truncated)?;
        if end > self.bytes.len() {
            return Err(Error::Truncated);
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn usize32(&mut self) -> Result<usize> {
        Ok(self.u32()? as usize)
    }
}

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

impl Hnsw<Metric> {
    /// Serialise the whole index, checksum included.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.node_count();
        let mut out = Vec::with_capacity(100 + n * (self.dims() * 4 + 48));

        out.extend_from_slice(MAGIC);
        put_u32(&mut out, FORMAT_VERSION);
        out.push(self.metric().tag());
        out.push(self.params().selection.tag());
        out.extend_from_slice(&[0u8, 0u8]); // reserved

        let p = self.params();
        put_u32(&mut out, self.dims() as u32);
        put_u32(&mut out, p.m as u32);
        put_u32(&mut out, p.m_max0 as u32);
        put_u32(&mut out, p.ef_construction as u32);
        put_u32(&mut out, p.ef_search as u32);
        put_u64(&mut out, p.ml.to_bits());
        put_u64(&mut out, p.seed);
        for word in self.rng_state() {
            put_u64(&mut out, word);
        }

        put_u32(&mut out, n as u32);
        put_u64(&mut out, self.entry_point().map_or(NO_ENTRY, u64::from));
        put_u32(&mut out, self.max_level() as u32);

        for node in 0..n as NodeId {
            put_u64(&mut out, self.id_at(node));
        }
        for node in 0..n as NodeId {
            out.push(self.level_of(node) as u8);
        }
        for node in 0..n as NodeId {
            out.push(u8::from(self.is_node_deleted(node)));
        }
        for node in 0..n as NodeId {
            for value in self.vector(node) {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        for node in 0..n as NodeId {
            for level in 0..=self.level_of(node) {
                let list = self.neighbors(node, level);
                put_u32(&mut out, list.len() as u32);
                for &neighbor in list {
                    put_u32(&mut out, neighbor);
                }
            }
        }

        let crc = crc32(&out);
        put_u32(&mut out, crc);
        out
    }

    /// Parse an index from bytes produced by [`Hnsw::to_bytes`].
    ///
    /// # Errors
    ///
    /// Rejects a wrong magic, an unknown version, a bad checksum, a truncated
    /// file, and any structurally impossible content (levels above the cap,
    /// links to nodes that do not exist, a node count that disagrees with the
    /// payload length).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 12 {
            return Err(Error::Truncated);
        }
        if &bytes[..8] != MAGIC {
            return Err(Error::BadMagic);
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        if bytes.len() < 104 {
            return Err(Error::Truncated);
        }

        let split = bytes.len() - 4;
        let (payload, tail) = bytes.split_at(split);
        let stored = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        let actual = crc32(payload);
        if stored != actual {
            return Err(Error::ChecksumMismatch {
                expected: stored,
                found: actual,
            });
        }

        let mut c = Cursor::new(payload);
        c.take(12)?; // magic + version, already checked
        let metric = Metric::from_tag(c.u8()?).ok_or(Error::Corrupt("unknown metric tag"))?;
        let selection =
            Selection::from_tag(c.u8()?).ok_or(Error::Corrupt("unknown selection tag"))?;
        c.take(2)?; // reserved

        let dims = c.usize32()?;
        if dims == 0 {
            return Err(Error::Corrupt("dims is zero"));
        }
        let params = Params {
            m: c.usize32()?,
            m_max0: c.usize32()?,
            ef_construction: c.usize32()?,
            ef_search: c.usize32()?,
            ml: f64::from_bits(c.u64()?),
            seed: c.u64()?,
            selection,
        };
        params.validate()?;
        let rng_state = [c.u64()?, c.u64()?, c.u64()?, c.u64()?];

        let n = c.usize32()?;
        // Guard against a corrupt count demanding a huge allocation before a
        // single byte has been validated: the payload must physically be able
        // to hold this many nodes.
        let min_bytes_per_node = 8 + 1 + 1 + dims * 4 + 4;
        if n.saturating_mul(min_bytes_per_node) > payload.len() {
            return Err(Error::Corrupt("node count exceeds payload size"));
        }
        let entry_raw = c.u64()?;
        let max_level = c.usize32()?;
        if max_level > MAX_LEVEL {
            return Err(Error::Corrupt("max level above the cap"));
        }

        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(c.u64()?);
        }
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n {
            let level = c.u8()?;
            if level as usize > max_level {
                return Err(Error::Corrupt("node level above index max level"));
            }
            levels.push(level);
        }
        let mut deleted = Vec::with_capacity(n);
        let mut deleted_count = 0usize;
        for _ in 0..n {
            let flag = c.u8()? != 0;
            deleted_count += usize::from(flag);
            deleted.push(flag);
        }
        let mut vectors = Vec::with_capacity(n * dims);
        for _ in 0..n * dims {
            vectors.push(c.f32()?);
        }

        let mut links = Vec::with_capacity(n);
        for &top in &levels {
            let top = top as usize;
            let mut per_layer = Vec::with_capacity(top + 1);
            for level in 0..=top {
                let count = c.usize32()?;
                let cap = params.max_degree(level);
                if count > cap {
                    return Err(Error::Corrupt("adjacency list exceeds the degree cap"));
                }
                // The cap alone is not a bound on the allocation: it comes
                // out of the same file as the count. What cannot be forged is
                // how many bytes are actually left, and every link costs four
                // of them, so a list that claims more than the rest of the
                // payload can hold is rejected before anything is reserved.
                if count.saturating_mul(4) > payload.len() - c.pos {
                    return Err(Error::Corrupt("adjacency list exceeds the remaining bytes"));
                }
                let mut list = Vec::with_capacity(count);
                for _ in 0..count {
                    let target = c.u32()?;
                    if target as usize >= n {
                        return Err(Error::Corrupt("link points past the last node"));
                    }
                    list.push(target);
                }
                per_layer.push(list);
            }
            links.push(per_layer);
        }
        if c.pos != payload.len() {
            return Err(Error::Corrupt("trailing bytes after the adjacency table"));
        }

        let entry = if entry_raw == NO_ENTRY {
            None
        } else {
            let e =
                NodeId::try_from(entry_raw).map_err(|_| Error::Corrupt("entry out of range"))?;
            if e as usize >= n {
                return Err(Error::Corrupt("entry point past the last node"));
            }
            Some(e)
        };
        if entry.is_none() && n != 0 {
            return Err(Error::Corrupt("non-empty index without an entry point"));
        }

        let mut id_map = IdMap::default();
        for (node, &id) in ids.iter().enumerate() {
            if id_map.insert(id, node as NodeId).is_some() {
                return Err(Error::Corrupt("duplicate external id"));
            }
        }

        let index = Hnsw {
            dims,
            params,
            dist: metric,
            vectors,
            ids,
            id_map,
            levels,
            links,
            deleted,
            deleted_count,
            entry,
            max_level,
            rng: Rng::from_state(rng_state),
            scratch: Searcher::new(),
        };
        // Symmetry, degree caps and level consistency are cheap to confirm and
        // a corrupt graph that survives to query time is far more confusing
        // than one rejected at load.
        if index.validate().is_err() {
            return Err(Error::Corrupt("graph invariants do not hold"));
        }
        Ok(index)
    }

    /// Write the index to `path`, replacing whatever was there.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&self.to_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Read an index written by [`Hnsw::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        Hnsw::from_bytes(&bytes)
    }
}
