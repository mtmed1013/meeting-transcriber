//! Suggested runtime settings for a local model, read from the files it ships with.

use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::assess::{chat_templates, gguf_chat_template};
use crate::error::ModelHubError;
use crate::hub::GGUF_HEADER_SAMPLE_BYTES;

/// Sidecar profile written inside an MLX model directory.
const MLX_PROFILE_FILE: &str = "mt-profile.json";
/// Sidecar suffix appended to a GGUF file name.
const GGUF_PROFILE_SUFFIX: &str = ".mt-profile.json";
/// Chat-template switch that lets a user toggle reasoning on and off.
const THINKING_TOGGLE_MARKER: &str = "enable_thinking";
/// Summary used when no source supplied any setting.
const SAFE_DEFAULTS_SUMMARY: &str = "valores seguros por defecto";

/// Runtime settings suggested for one model, with the sources that produced them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuggestedModelConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_context_tokens: Option<u32>,
    pub supports_thinking_toggle: bool,
    pub recommended_thinking: Option<bool>,
    pub source_summary: String,
}

/// Builds the suggested settings from whichever model files were available.
///
/// Out-of-range values are ignored rather than rejected, so a partial or odd file never fails the
/// assessment. `source_summary` lists the sources that contributed a value, or says that safe
/// defaults were used when none did.
pub fn extract_suggested_config(
    config: Option<&Value>,
    generation_config: Option<&Value>,
    tokenizer_config: Option<&Value>,
    gguf_header: Option<&[u8]>,
) -> SuggestedModelConfig {
    let mut profile = SuggestedModelConfig::default();
    let mut sources: Vec<&str> = Vec::new();

    if let Some(generation) = generation_config {
        profile.temperature = finite_non_negative(generation, "temperature");
        profile.top_p = finite_non_negative(generation, "top_p").filter(|v| *v > 0.0 && *v <= 1.0);
        profile.top_k = positive_u32(generation, "top_k");
        profile.repetition_penalty = finite_non_negative(generation, "repetition_penalty");
        profile.presence_penalty = finite_non_negative(generation, "presence_penalty");
        let contributed = profile.temperature.is_some()
            || profile.top_p.is_some()
            || profile.top_k.is_some()
            || profile.repetition_penalty.is_some()
            || profile.presence_penalty.is_some();
        if contributed {
            sources.push("generation_config.json");
        }
    }

    if let Some(max) = config.and_then(max_position_embeddings) {
        profile.max_context_tokens = Some(max);
        sources.push("config.json");
    }

    let tokenizer_toggle = tokenizer_config.is_some_and(|config| {
        chat_templates(config)
            .into_iter()
            .any(|template| template.contains(THINKING_TOGGLE_MARKER))
    });
    if tokenizer_toggle {
        sources.push("tokenizer_config.json");
    }
    let header_toggle = gguf_header
        .and_then(gguf_chat_template)
        .is_some_and(|bytes| String::from_utf8_lossy(bytes).contains(THINKING_TOGGLE_MARKER));
    if header_toggle {
        sources.push("gguf header");
    }
    if tokenizer_toggle || header_toggle {
        profile.supports_thinking_toggle = true;
        profile.recommended_thinking = Some(false);
    }

    profile.source_summary = if sources.is_empty() {
        SAFE_DEFAULTS_SUMMARY.to_string()
    } else {
        sources.join(" + ")
    };
    profile
}

/// Reads the settings saved for a model, or derives them from the files it ships with.
///
/// A saved sidecar (`mt-profile.json` in an MLX directory, `<file>.mt-profile.json` next to a GGUF
/// file) takes precedence. A missing or unreadable model yields safe defaults.
pub fn load_local_suggested_config(model_path: &Path) -> SuggestedModelConfig {
    if model_path.is_dir() {
        if let Some(saved) = read_saved_profile(&model_path.join(MLX_PROFILE_FILE)) {
            return saved;
        }
        let config = read_json(&model_path.join("config.json"));
        let generation = read_json(&model_path.join("generation_config.json"));
        let tokenizer = read_json(&model_path.join("tokenizer_config.json"));
        return extract_suggested_config(
            config.as_ref(),
            generation.as_ref(),
            tokenizer.as_ref(),
            None,
        );
    }
    if model_path.is_file() {
        if let Some(saved) = read_saved_profile(&gguf_sidecar_path(model_path)) {
            return saved;
        }
        let header = read_gguf_header_sample(model_path);
        return extract_suggested_config(None, None, None, header.as_deref());
    }
    extract_suggested_config(None, None, None, None)
}

/// Saves settings beside the model: `mt-profile.json` inside a directory, or `<file>.mt-profile.json`
/// next to a file.
pub fn save_local_suggested_config(
    model_path: &Path,
    profile: &SuggestedModelConfig,
) -> Result<(), ModelHubError> {
    let target = if model_path.is_dir() {
        model_path.join(MLX_PROFILE_FILE)
    } else {
        gguf_sidecar_path(model_path)
    };
    let json = serde_json::to_vec_pretty(profile)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    fs::write(target, json)?;
    Ok(())
}

fn finite_non_negative(value: &Value, key: &str) -> Option<f32> {
    let parsed = value.get(key)?.as_f64()? as f32;
    (parsed.is_finite() && parsed >= 0.0).then_some(parsed)
}

fn positive_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)?
        .as_u64()
        .and_then(|raw| u32::try_from(raw).ok())
        .filter(|parsed| *parsed > 0)
}

/// `max_position_embeddings` at the top level, or under `text_config` for multimodal configs.
fn max_position_embeddings(config: &Value) -> Option<u32> {
    positive_u32(config, "max_position_embeddings")
        .or_else(|| positive_u32(config.get("text_config")?, "max_position_embeddings"))
}

fn gguf_sidecar_path(model_path: &Path) -> PathBuf {
    let mut name = model_path.as_os_str().to_owned();
    name.push(GGUF_PROFILE_SUFFIX);
    PathBuf::from(name)
}

fn read_saved_profile(path: &Path) -> Option<SuggestedModelConfig> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

/// Leading bytes of a GGUF file, where its chat template lives. `None` when the file cannot be read.
fn read_gguf_header_sample(path: &Path) -> Option<Vec<u8>> {
    let mut sample = Vec::with_capacity(GGUF_HEADER_SAMPLE_BYTES);
    fs::File::open(path)
        .ok()?
        .take(GGUF_HEADER_SAMPLE_BYTES as u64)
        .read_to_end(&mut sample)
        .ok()?;
    Some(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const THINKING_TEMPLATE: &str = "{% if enable_thinking %}<think>{% endif %}{{ messages }}";

    /// One GGUF string metadata entry: length-prefixed key, type 8, length-prefixed value.
    fn gguf_string_entry(key: &str, value: &str) -> Vec<u8> {
        let mut entry = (key.len() as u64).to_le_bytes().to_vec();
        entry.extend_from_slice(key.as_bytes());
        entry.extend_from_slice(&8u32.to_le_bytes());
        entry.extend_from_slice(&(value.len() as u64).to_le_bytes());
        entry.extend_from_slice(value.as_bytes());
        entry
    }

    fn gguf_header_with_template(template: &str) -> Vec<u8> {
        let mut header = b"GGUF\x03\x00\x00\x00junk".to_vec();
        header.extend(gguf_string_entry("tokenizer.chat_template", template));
        header
    }

    #[test]
    fn extraction_reads_generation_and_model_config_and_detects_thinking_toggle() {
        let generation = json!({
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "presence_penalty": 1.5
        });
        let config = json!({ "max_position_embeddings": 32768 });
        let tokenizer = json!({ "chat_template": THINKING_TEMPLATE });

        let profile =
            extract_suggested_config(Some(&config), Some(&generation), Some(&tokenizer), None);

        assert_eq!(profile.temperature, Some(0.7));
        assert_eq!(profile.top_p, Some(0.8));
        assert_eq!(profile.top_k, Some(20));
        assert_eq!(profile.repetition_penalty, Some(1.05));
        assert_eq!(profile.presence_penalty, Some(1.5));
        assert_eq!(profile.max_context_tokens, Some(32768));
        assert!(profile.supports_thinking_toggle);
        assert_eq!(profile.recommended_thinking, Some(false));
        assert_eq!(
            profile.source_summary,
            "generation_config.json + config.json + tokenizer_config.json"
        );
    }

    #[test]
    fn thinking_toggle_is_detected_in_chat_template_list_and_gguf_header() {
        let list = json!({
            "chat_template": [
                { "name": "default", "template": "plain chat" },
                { "name": "thinking", "template": THINKING_TEMPLATE }
            ]
        });
        let from_list = extract_suggested_config(None, None, Some(&list), None);
        assert!(from_list.supports_thinking_toggle);
        assert_eq!(from_list.recommended_thinking, Some(false));

        let header = gguf_header_with_template(THINKING_TEMPLATE);
        let from_header = extract_suggested_config(None, None, None, Some(&header));
        assert!(from_header.supports_thinking_toggle);
        assert_eq!(from_header.recommended_thinking, Some(false));
        assert_eq!(from_header.source_summary, "gguf header");
    }

    #[test]
    fn plain_chat_template_does_not_enable_thinking_toggle() {
        let tokenizer = json!({ "chat_template": "{{ messages }}" });
        let profile = extract_suggested_config(None, None, Some(&tokenizer), None);
        assert!(!profile.supports_thinking_toggle);
        assert_eq!(profile.recommended_thinking, None);
    }

    #[test]
    fn invalid_generation_values_are_ignored() {
        let generation = json!({
            "temperature": -0.5,
            "top_p": 0.0,
            "top_k": 0,
            "repetition_penalty": -1.0,
            "presence_penalty": -2.0
        });
        let profile = extract_suggested_config(None, Some(&generation), None, None);
        assert_eq!(profile.temperature, None);
        assert_eq!(profile.top_p, None);
        assert_eq!(profile.top_k, None);
        assert_eq!(profile.repetition_penalty, None);
        assert_eq!(profile.presence_penalty, None);
        assert_eq!(profile.source_summary, "valores seguros por defecto");
    }

    #[test]
    fn boundary_generation_values_are_accepted() {
        let generation = json!({ "temperature": 0.0, "top_p": 1.0, "top_k": 1 });
        let profile = extract_suggested_config(None, Some(&generation), None, None);
        assert_eq!(profile.temperature, Some(0.0));
        assert_eq!(profile.top_p, Some(1.0));
        assert_eq!(profile.top_k, Some(1));
    }

    #[test]
    fn fallback_defaults_are_safe_when_no_source_is_present() {
        let profile = extract_suggested_config(None, None, None, None);
        assert_eq!(
            profile,
            SuggestedModelConfig {
                source_summary: "valores seguros por defecto".into(),
                ..Default::default()
            }
        );
    }

    #[test]
    fn max_context_falls_back_to_text_config() {
        let config = json!({ "text_config": { "max_position_embeddings": 8192 } });
        let profile = extract_suggested_config(Some(&config), None, None, None);
        assert_eq!(profile.max_context_tokens, Some(8192));
        assert_eq!(profile.source_summary, "config.json");
    }

    #[test]
    fn mlx_directory_reads_model_files_and_prefers_its_saved_profile() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path();
        fs::write(model.join("config.json"), r#"{"max_position_embeddings": 4096}"#).unwrap();
        fs::write(model.join("generation_config.json"), r#"{"temperature": 0.7}"#).unwrap();
        fs::write(
            model.join("tokenizer_config.json"),
            json!({ "chat_template": THINKING_TEMPLATE }).to_string(),
        )
        .unwrap();

        let loaded = load_local_suggested_config(model);
        assert_eq!(loaded.temperature, Some(0.7));
        assert_eq!(loaded.max_context_tokens, Some(4096));
        assert!(loaded.supports_thinking_toggle);
        assert_eq!(
            loaded.source_summary,
            "generation_config.json + config.json + tokenizer_config.json"
        );

        let saved = SuggestedModelConfig {
            temperature: Some(0.3),
            top_k: Some(40),
            max_context_tokens: Some(2048),
            source_summary: "manual".into(),
            ..Default::default()
        };
        save_local_suggested_config(model, &saved).unwrap();
        assert!(model.join("mt-profile.json").is_file());
        assert_eq!(load_local_suggested_config(model), saved);
    }

    #[test]
    fn gguf_file_round_trips_through_its_sidecar_and_detects_thinking_in_the_header() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        fs::write(&model, gguf_header_with_template(THINKING_TEMPLATE)).unwrap();

        let loaded = load_local_suggested_config(&model);
        assert!(loaded.supports_thinking_toggle);
        assert_eq!(loaded.recommended_thinking, Some(false));

        let saved = SuggestedModelConfig {
            temperature: Some(0.3),
            supports_thinking_toggle: true,
            recommended_thinking: Some(true),
            source_summary: "manual".into(),
            ..Default::default()
        };
        save_local_suggested_config(&model, &saved).unwrap();
        assert!(dir.path().join("model.gguf.mt-profile.json").is_file());
        assert_eq!(load_local_suggested_config(&model), saved);
    }

    #[test]
    fn missing_model_path_yields_safe_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_local_suggested_config(&dir.path().join("absent.gguf"));
        assert_eq!(loaded.source_summary, "valores seguros por defecto");
    }
}
