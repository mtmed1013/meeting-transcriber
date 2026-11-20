//! Error type shared by every module of `mt-models`.

use std::io;

/// Errors produced while resolving, planning, assessing or downloading a model.
#[derive(Debug, thiserror::Error)]
pub enum ModelHubError {
    /// The user-supplied reference is malformed or attempts traversal.
    #[error("referencia de modelo inválida: {0}")]
    InvalidReference(String),
    /// The Hugging Face hub does not know this repository (or revision).
    #[error("repositorio no encontrado en Hugging Face: {0}")]
    RepoNotFound(String),
    /// The repository requires an account or approval, which this app does not use.
    #[error("el repositorio {0} requiere cuenta o aprobación (gated); no se puede instalar sin cuenta")]
    Gated(String),
    /// The repository has no file layout this app can install.
    #[error("formato no soportado: {0}")]
    UnsupportedFormat(String),
    /// Transport failure or a retryable server condition.
    #[error("error de red: {0}")]
    Network(String),
    /// The hub answered, but not in the shape we expect.
    #[error("respuesta inesperada del hub: {0}")]
    BadResponse(String),
    /// A downloaded file does not match its published SHA-256.
    #[error("checksum SHA-256 no coincide en {file}: esperado {expected}, obtenido {actual}")]
    ChecksumMismatch {
        file: String,
        expected: String,
        actual: String,
    },
    /// The model folder already exists in the models directory.
    #[error("el modelo ya está instalado en {0}")]
    AlreadyInstalled(String),
    /// The user cancelled the download. Partial files are kept for resuming.
    #[error("descarga cancelada")]
    Cancelled,
    /// Local filesystem failure.
    #[error("error de archivo: {0}")]
    Io(#[from] io::Error),
}

impl From<reqwest::Error> for ModelHubError {
    fn from(err: reqwest::Error) -> Self {
        ModelHubError::Network(err.to_string())
    }
}
