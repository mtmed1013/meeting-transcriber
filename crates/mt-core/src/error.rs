//! Error type shared by every module of `mt-core`.

use std::io;

/// Errors produced by core domain operations.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Local filesystem failure.
    #[error("error de archivo: {0}")]
    Io(#[from] io::Error),
}
