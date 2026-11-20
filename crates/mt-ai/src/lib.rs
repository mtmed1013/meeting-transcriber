//! AI integration for Meeting Transcriber: conservative token budgeting, an OpenAI-compatible chat
//! client, and the light-analyst pipeline that turns a transcript into
//! `mt_core::report::ReportSections`.

pub mod budget;
pub mod error;
pub mod light_analyst;
pub mod provider;
pub mod runtime;
pub mod tool_calls_text;

pub use budget::{chunk_transcript, estimate_messages_tokens, estimate_tokens};
pub use error::AiError;
pub use light_analyst::{MIN_SPOKEN_WORDS, analyze_transcript, meets_word_threshold};
pub use provider::{ChatMessage, ChatRole, ModelConfig, OpenAiCompatibleProvider};
pub use tool_calls_text::{ParsedCall, ParsedCalls, parse_tool_calls};
