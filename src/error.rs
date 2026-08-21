//! Error type shared by the index, the file format and the CLI.

use core::fmt;

/// Everything that can go wrong in this crate.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A vector was handed to the index with the wrong number of components.
    DimensionMismatch {
        /// Width the index was created with.
        expected: usize,
        /// Width of the offending vector.
        found: usize,
    },
    /// An id already present in the index was inserted again.
    DuplicateId(u64),
    /// An id that is not in the index was referenced.
    UnknownId(u64),
    /// A construction parameter was outside its legal range.
    InvalidParams(&'static str),
    /// Underlying I/O failure while reading or writing an index file.
    Io(std::io::Error),
    /// The file does not start with the `HNSWIDX\0` magic.
    BadMagic,
    /// The file's format version is not one this build understands.
    UnsupportedVersion {
        /// Version read from the file.
        found: u32,
        /// Version this build writes and reads.
        supported: u32,
    },
    /// The trailing CRC-32 did not match the bytes that preceded it.
    ChecksumMismatch {
        /// CRC stored in the file.
        expected: u32,
        /// CRC computed over the payload actually read.
        found: u32,
    },
    /// The file ended in the middle of a field.
    Truncated,
    /// The file was structurally intact but semantically impossible.
    Corrupt(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch { expected, found } => {
                write!(f, "expected a {expected}-dimensional vector, got {found}")
            }
            Error::DuplicateId(id) => write!(f, "id {id} is already in the index"),
            Error::UnknownId(id) => write!(f, "id {id} is not in the index"),
            Error::InvalidParams(why) => write!(f, "invalid parameters: {why}"),
            Error::Io(e) => write!(f, "i/o error: {e}"),
            Error::BadMagic => f.write_str("not an hnsw index file (bad magic)"),
            Error::UnsupportedVersion { found, supported } => {
                write!(
                    f,
                    "index file version {found}, this build reads {supported}"
                )
            }
            Error::ChecksumMismatch { expected, found } => write!(
                f,
                "checksum mismatch: file says {expected:#010x}, payload hashes to {found:#010x}"
            ),
            Error::Truncated => f.write_str("index file ends mid-record"),
            Error::Corrupt(why) => write!(f, "corrupt index file: {why}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// Convenience alias.
pub type Result<T> = core::result::Result<T, Error>;
