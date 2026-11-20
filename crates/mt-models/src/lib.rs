//! Model management for Meeting Transcriber: reference parsing, Hugging Face hub access,
//! hardware fit assessment, and resumable downloads.

pub mod assess;
pub mod download;
pub mod error;
pub mod hardware;
pub mod hub;
pub mod mlx_support;
pub mod plan;
pub mod profile;
pub mod reference;

pub use assess::{AssessRequest, Assessment, Blocker, Fit, assess};
pub use download::{CancelToken, Downloader, Progress, discard_partial};
pub use error::ModelHubError;
pub use hardware::{MachineInfo, detect_machine};
pub use hub::{HubClient, RepoInfo, Sibling};
pub use mlx_support::{find_site_packages, model_type_from_dir, supported_model_types, unsupported_reason};
pub use plan::{GgufOption, ModelFormat, ModelPlan, PlannedFile, inspect_repo_format, list_gguf_options};
pub use profile::{SuggestedModelConfig, extract_suggested_config, load_local_suggested_config, save_local_suggested_config};
pub use reference::{RepoRef, parse_reference};
