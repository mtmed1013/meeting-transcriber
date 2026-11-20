//! Error type shared by every module of `mt-ai`.

use std::io;

/// Errors produced by AI operations.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    /// The prompt does not fit in the model's input budget. Raised before any network call.
    #[error("el contexto excede el presupuesto del modelo: ~{estimated_tokens} tokens estimados, máximo {budget_tokens}")]
    ContextOverflow {
        estimated_tokens: usize,
        budget_tokens: usize,
    },
    /// The configured endpoint is invalid or cannot be reached (connection refused, timeout).
    #[error("endpoint de IA no disponible: {0}")]
    EndpointUnreachable(String),
    /// The endpoint answered with an error status or the request could not be completed.
    #[error("la solicitud al modelo falló: {0}")]
    RequestFailed(String),
    /// The model answered, but the reply has an unexpected shape or is not the requested JSON.
    #[error("respuesta inválida del modelo: {0}")]
    InvalidResponse(String),
    /// Local runtime failure (for example, the async runtime could not run a task).
    #[error("error de ejecución: {0}")]
    Runtime(String),
    /// Local filesystem failure.
    #[error("error de archivo: {0}")]
    Io(#[from] io::Error),
}
