//! Core domain logic for Meeting Transcriber: application settings and `.env` migration,
//! transcript correction rules and word counting, and meeting report rendering.

pub mod config;
pub mod corrections;
pub mod error;
pub mod report;

pub use config::{
    AiSource, AppSettings, CorrectionRule, SttEngine, ThemeMode, WhisperSize,
    default_output_dir, import_dotenv,
};
pub use corrections::{
    SPEAKER_CHANGE_MARKER, apply_corrections, compact_transcript_for_analysis, count_spoken_words,
};
pub use error::CoreError;
pub use report::{
    ActionItem, DecisionItem, IssueItem, MarkdownSections, QuestionItem, ReportSections, TopicItem,
    assemble_markdown_report, assemble_report_markdown, default_report_skill_prompt,
    is_legacy_default_report_skill_prompt, short_meeting_markdown, strip_existing_report,
    write_atomic_in_place,
};
