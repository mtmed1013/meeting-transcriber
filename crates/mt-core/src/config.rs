//! Application settings, their defaults, and the one-time migration from the legacy `.env` file.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Smallest buffer the Windows recorder accepts; matches the legacy `max(0.2, ...)` clamp.
const MIN_WINDOWS_BUFFER_SECONDS: f64 = 0.2;

/// Speech-to-text engine selection. `Auto` picks Whistle where it is supported, else Whisper.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SttEngine {
    #[default]
    Auto,
    Whistle,
    Whisper,
}

/// Whisper model size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WhisperSize {
    Tiny,
    Base,
    #[default]
    Small,
    Medium,
}

/// Where the AI report model runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AiSource {
    #[default]
    Local,
    Api,
}

/// UI theme preference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThemeMode {
    #[default]
    System,
    Dark,
    Light,
}

/// One find-and-replace correction applied to transcript text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionRule {
    pub find: String,
    pub replace: String,
    /// `false` (default) matches regardless of letter case.
    #[serde(default)]
    pub case_sensitive: bool,
    /// `true` only matches when the occurrence is not adjacent to a Unicode alphanumeric character.
    #[serde(default)]
    pub whole_word: bool,
}

/// Persisted application settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub output_dir: PathBuf,
    pub stt_engine: SttEngine,
    pub whisper_model: WhisperSize,
    pub transcription_language: String,
    pub output_language: String,
    pub speaker_markers: bool,
    pub whisper_vad_filter: bool,
    pub windows_microphone: String,
    pub windows_speaker: String,
    pub windows_buffer_seconds: f64,
    pub mac_system_gain: f64,
    pub mac_microphone_gain: f64,
    pub ai_enabled: bool,
    pub ai_source: AiSource,
    pub ai_models_dir: PathBuf,
    pub ai_model_id: String,
    pub ai_endpoint: String,
    pub ai_model_name: String,
    pub ai_api_key: String,
    /// Apply the Hugging Face recommended generation profile when one is available.
    #[serde(default = "default_true")]
    pub use_hf_recommended_config: bool,
    /// Ask the model to skip its thinking mode (sent only to loopback endpoints).
    #[serde(default = "default_true")]
    pub ai_disable_thinking: bool,
    /// User override of the sampling temperature; `None` keeps the profile value.
    #[serde(default)]
    pub ai_temperature: Option<f32>,
    /// User override of nucleus sampling `top_p`; `None` keeps the profile value.
    #[serde(default)]
    pub ai_top_p: Option<f32>,
    /// User override of `top_k`; `None` keeps the profile value.
    #[serde(default)]
    pub ai_top_k: Option<u32>,
    /// User override of `repetition_penalty`; `None` keeps the profile value.
    #[serde(default)]
    pub ai_repetition_penalty: Option<f32>,
    pub context_limit_tokens: u32,
    pub max_output_tokens: u32,
    pub min_report_words: usize,
    pub replace_existing_report: bool,
    pub corrections: Vec<CorrectionRule>,
    pub theme: ThemeMode,
    #[serde(default = "crate::report::default_report_skill_prompt")]
    pub report_skill_prompt: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            output_dir: default_output_dir(),
            stt_engine: SttEngine::Auto,
            whisper_model: WhisperSize::Small,
            transcription_language: "es".to_string(),
            output_language: "Spanish".to_string(),
            speaker_markers: false,
            whisper_vad_filter: true,
            windows_microphone: String::new(),
            windows_speaker: String::new(),
            windows_buffer_seconds: 1.0,
            mac_system_gain: 1.0,
            mac_microphone_gain: 1.0,
            ai_enabled: false,
            ai_source: AiSource::Local,
            ai_models_dir: PathBuf::new(),
            ai_model_id: String::new(),
            ai_endpoint: "http://127.0.0.1:8080/v1".to_string(),
            ai_model_name: String::new(),
            ai_api_key: String::new(),
            use_hf_recommended_config: true,
            ai_disable_thinking: true,
            ai_temperature: None,
            ai_top_p: None,
            ai_top_k: None,
            ai_repetition_penalty: None,
            context_limit_tokens: 16384,
            max_output_tokens: 2560,
            min_report_words: 50,
            replace_existing_report: true,
            corrections: Vec::new(),
            theme: ThemeMode::System,
            report_skill_prompt: crate::report::default_report_skill_prompt(),
        }
    }
}

fn default_true() -> bool {
    true
}

/// Platform default Obsidian vault folder.
///
/// Windows: `~/Documents/Obsidian`. macOS and other Unix: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian`.
pub fn default_output_dir() -> PathBuf {
    let home = home_dir();
    if cfg!(windows) {
        home.join("Documents").join("Obsidian")
    } else {
        home.join("Library")
            .join("Mobile Documents")
            .join("iCloud~md~obsidian")
            .join("Documents")
            .join("Obsidian")
    }
}

fn home_dir() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var).map(PathBuf::from).unwrap_or_default()
}

/// Migrates legacy `.env` keys into `base`.
///
/// For each key the last non-blank occurrence in the file wins. Blank values, unparsable values,
/// and unknown keys leave the corresponding field as it was. Keys absent from the file never touch
/// their field. `WHISPER_MODEL` / `WINDOWS_WHISPER_MODEL` follow the legacy engine: only the key for
/// the current platform is read. `TRANSCRIPTION_LANGUAGE` takes precedence over `WHISPER_LANGUAGE`.
pub fn import_dotenv(contents: &str, base: &mut AppSettings) {
    let values = parse_dotenv(contents);
    let get = |key: &str| values.get(key).map(String::as_str);

    if let Some(dir) = get("OBSIDIAN_DIR") {
        base.output_dir = PathBuf::from(dir);
    }
    if let Some(engine) = get("TRANSCRIPTION_ENGINE").and_then(parse_engine) {
        base.stt_engine = engine;
    }
    if let Some(language) = get("TRANSCRIPTION_LANGUAGE").or_else(|| get("WHISPER_LANGUAGE")) {
        base.transcription_language = language.to_lowercase();
    }
    let model_key = if cfg!(windows) {
        "WINDOWS_WHISPER_MODEL"
    } else {
        "WHISPER_MODEL"
    };
    if let Some(size) = get(model_key).and_then(parse_whisper_size) {
        base.whisper_model = size;
    }
    if let Some(value) = get("SPEAKER_MARKERS") {
        base.speaker_markers = parse_bool(value);
    }
    if let Some(value) = get("WHISPER_VAD_FILTER") {
        base.whisper_vad_filter = parse_bool(value);
    }
    if let Some(device) = get("WINDOWS_MICROPHONE") {
        base.windows_microphone = device.to_string();
    }
    if let Some(device) = get("WINDOWS_SPEAKER") {
        base.windows_speaker = device.to_string();
    }
    if let Some(seconds) = get("WINDOWS_BUFFER_SECONDS").and_then(parse_non_negative) {
        base.windows_buffer_seconds = seconds.max(MIN_WINDOWS_BUFFER_SECONDS);
    }
    if let Some(gain) = get("MAC_SYSTEM_GAIN").and_then(parse_non_negative) {
        base.mac_system_gain = gain;
    }
    if let Some(gain) = get("MAC_MICROPHONE_GAIN").and_then(parse_non_negative) {
        base.mac_microphone_gain = gain;
    }
}

/// Parses `KEY=VALUE` lines into a map holding the last non-blank value per key.
fn parse_dotenv(contents: &str) -> HashMap<String, String> {
    let mut values = HashMap::new();
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line
            .strip_prefix("export ")
            .map(str::trim_start)
            .unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = parse_dotenv_value(value.trim());
        if key.is_empty() || value.is_empty() {
            continue;
        }
        values.insert(key.to_string(), value);
    }
    values
}

/// Returns the value of one `.env` entry: quoted values are unwrapped, and an inline `#` comment
/// (preceded by whitespace) is dropped from unquoted values.
fn parse_dotenv_value(raw: &str) -> String {
    if let Some(quote) = raw.chars().next().filter(|c| *c == '"' || *c == '\'') {
        let inner = &raw[1..];
        if let Some(end) = inner.find(quote) {
            return inner[..end].trim().to_string();
        }
    }
    let mut prev_is_space = false;
    for (index, ch) in raw.char_indices() {
        if ch == '#' && prev_is_space {
            return raw[..index].trim().to_string();
        }
        prev_is_space = ch.is_whitespace();
    }
    raw.trim().to_string()
}

fn parse_engine(value: &str) -> Option<SttEngine> {
    match value.to_lowercase().as_str() {
        "auto" => Some(SttEngine::Auto),
        "whisper" => Some(SttEngine::Whisper),
        "whistle" => Some(SttEngine::Whistle),
        _ => None,
    }
}

fn parse_whisper_size(value: &str) -> Option<WhisperSize> {
    match value.to_lowercase().as_str() {
        "tiny" => Some(WhisperSize::Tiny),
        "base" => Some(WhisperSize::Base),
        "small" => Some(WhisperSize::Small),
        "medium" => Some(WhisperSize::Medium),
        _ => None,
    }
}

/// Legacy truthiness: everything is true except `0`, `false`, `no`, and `off`.
fn parse_bool(value: &str) -> bool {
    !matches!(value.to_lowercase().as_str(), "0" | "false" | "no" | "off")
}

/// Accepts finite, non-negative numbers only.
fn parse_non_negative(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_migrated_app_contract() {
        let s = AppSettings::default();
        assert_eq!(s.output_dir, default_output_dir());
        assert_eq!(s.stt_engine, SttEngine::Auto);
        assert_eq!(s.whisper_model, WhisperSize::Small);
        assert_eq!(s.transcription_language, "es");
        assert_eq!(s.output_language, "Spanish");
        assert!(!s.speaker_markers);
        assert!(s.whisper_vad_filter);
        assert_eq!(s.windows_microphone, "");
        assert_eq!(s.windows_speaker, "");
        assert_eq!(s.windows_buffer_seconds, 1.0);
        assert_eq!(s.mac_system_gain, 1.0);
        assert_eq!(s.mac_microphone_gain, 1.0);
        assert!(!s.ai_enabled);
        assert_eq!(s.ai_source, AiSource::Local);
        assert_eq!(s.ai_models_dir, PathBuf::new());
        assert_eq!(s.ai_model_id, "");
        assert_eq!(s.ai_endpoint, "http://127.0.0.1:8080/v1");
        assert_eq!(s.ai_model_name, "");
        assert_eq!(s.ai_api_key, "");
        assert_eq!(s.context_limit_tokens, 16384);
        assert_eq!(s.max_output_tokens, 2560);
        assert_eq!(s.min_report_words, 50);
        assert!(s.replace_existing_report);
        assert!(s.corrections.is_empty());
        assert_eq!(s.theme, ThemeMode::System);
    }

    #[test]
    fn default_output_dir_is_the_platform_obsidian_vault() {
        let text = default_output_dir().to_string_lossy().replace('\\', "/");
        if cfg!(windows) {
            assert!(text.ends_with("/Documents/Obsidian"), "{text}");
        } else {
            assert!(
                text.ends_with("/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian"),
                "{text}"
            );
        }
    }

    fn base() -> AppSettings {
        AppSettings::default()
    }

    #[test]
    fn dotenv_obsidian_dir_sets_output_dir_and_blank_keeps_default() {
        let mut s = base();
        import_dotenv("OBSIDIAN_DIR=/vault/notes\n", &mut s);
        assert_eq!(s.output_dir, PathBuf::from("/vault/notes"));

        let mut s = base();
        let before = s.output_dir.clone();
        import_dotenv("OBSIDIAN_DIR=   \n", &mut s);
        assert_eq!(s.output_dir, before);
    }

    #[test]
    fn dotenv_engine_is_case_insensitive_and_invalid_values_are_ignored() {
        for (raw, expected) in [
            ("auto", SttEngine::Auto),
            ("WHISPER", SttEngine::Whisper),
            (" Whistle ", SttEngine::Whistle),
        ] {
            let mut s = base();
            import_dotenv(&format!("TRANSCRIPTION_ENGINE={raw}\n"), &mut s);
            assert_eq!(s.stt_engine, expected, "{raw}");
        }

        let mut s = base();
        s.stt_engine = SttEngine::Whisper;
        import_dotenv("TRANSCRIPTION_ENGINE=vosk\n", &mut s);
        assert_eq!(s.stt_engine, SttEngine::Whisper);
    }

    #[test]
    fn dotenv_language_prefers_transcription_language_and_lowercases() {
        let mut s = base();
        import_dotenv("WHISPER_LANGUAGE=EN\n", &mut s);
        assert_eq!(s.transcription_language, "en");

        let mut s = base();
        import_dotenv("WHISPER_LANGUAGE=en\nTRANSCRIPTION_LANGUAGE=PT\n", &mut s);
        assert_eq!(s.transcription_language, "pt");

        let mut s = base();
        import_dotenv("TRANSCRIPTION_LANGUAGE=  \nWHISPER_LANGUAGE=fr\n", &mut s);
        assert_eq!(s.transcription_language, "fr");

        let mut s = base();
        import_dotenv("TRANSCRIPTION_LANGUAGE=auto\n", &mut s);
        assert_eq!(s.transcription_language, "auto");
    }

    #[test]
    fn dotenv_whisper_model_reads_the_platform_key_like_the_legacy_engine() {
        // The legacy engine reads WINDOWS_WHISPER_MODEL on Windows and WHISPER_MODEL elsewhere.
        let (own, other) = if cfg!(windows) {
            ("WINDOWS_WHISPER_MODEL", "WHISPER_MODEL")
        } else {
            ("WHISPER_MODEL", "WINDOWS_WHISPER_MODEL")
        };

        let mut s = base();
        import_dotenv(&format!("{own}=medium\n"), &mut s);
        assert_eq!(s.whisper_model, WhisperSize::Medium);

        let mut s = base();
        import_dotenv(&format!("{other}=tiny\n{own}=base\n"), &mut s);
        assert_eq!(s.whisper_model, WhisperSize::Base);

        let mut s = base();
        import_dotenv(&format!("{other}=tiny\n"), &mut s);
        assert_eq!(s.whisper_model, WhisperSize::Small);

        let mut s = base();
        import_dotenv(&format!("{own}=large-v3\n"), &mut s);
        assert_eq!(s.whisper_model, WhisperSize::Small);
    }

    #[test]
    fn dotenv_whisper_model_is_case_insensitive() {
        let key = if cfg!(windows) { "WINDOWS_WHISPER_MODEL" } else { "WHISPER_MODEL" };
        let mut s = base();
        import_dotenv(&format!("{key}=TINY\n"), &mut s);
        assert_eq!(s.whisper_model, WhisperSize::Tiny);
    }

    #[test]
    fn dotenv_boolean_flags_follow_legacy_truthiness() {
        for falsy in ["0", "false", "NO", "off", " Off "] {
            let mut s = base();
            import_dotenv(
                &format!("SPEAKER_MARKERS={falsy}\nWHISPER_VAD_FILTER={falsy}\n"),
                &mut s,
            );
            assert!(!s.speaker_markers, "{falsy}");
            assert!(!s.whisper_vad_filter, "{falsy}");
        }
        for truthy in ["1", "true", "yes", "on", "anything"] {
            let mut s = base();
            import_dotenv(
                &format!("SPEAKER_MARKERS={truthy}\nWHISPER_VAD_FILTER={truthy}\n"),
                &mut s,
            );
            assert!(s.speaker_markers, "{truthy}");
            assert!(s.whisper_vad_filter, "{truthy}");
        }
    }

    #[test]
    fn dotenv_blank_boolean_keeps_default() {
        let mut s = base();
        import_dotenv("SPEAKER_MARKERS=\nWHISPER_VAD_FILTER=   \n", &mut s);
        assert!(!s.speaker_markers);
        assert!(s.whisper_vad_filter);
    }

    #[test]
    fn dotenv_windows_devices_are_trimmed_and_blank_values_are_ignored() {
        let mut s = base();
        import_dotenv(
            "WINDOWS_MICROPHONE=  Mic Array  \nWINDOWS_SPEAKER=Speakers (Realtek)\n",
            &mut s,
        );
        assert_eq!(s.windows_microphone, "Mic Array");
        assert_eq!(s.windows_speaker, "Speakers (Realtek)");

        let mut s = base();
        s.windows_microphone = "Keep".to_string();
        import_dotenv("WINDOWS_MICROPHONE=\n", &mut s);
        assert_eq!(s.windows_microphone, "Keep");
    }

    #[test]
    fn dotenv_buffer_seconds_is_clamped_and_invalid_values_keep_default() {
        let mut s = base();
        import_dotenv("WINDOWS_BUFFER_SECONDS=2.5\n", &mut s);
        assert_eq!(s.windows_buffer_seconds, 2.5);

        let mut s = base();
        import_dotenv("WINDOWS_BUFFER_SECONDS=0.01\n", &mut s);
        assert_eq!(s.windows_buffer_seconds, 0.2);

        for bad in ["abc", "-3", "inf", "NaN", ""] {
            let mut s = base();
            import_dotenv(&format!("WINDOWS_BUFFER_SECONDS={bad}\n"), &mut s);
            assert_eq!(s.windows_buffer_seconds, 1.0, "{bad}");
        }
    }

    #[test]
    fn dotenv_mac_gains_accept_only_non_negative_finite_numbers() {
        let mut s = base();
        import_dotenv("MAC_SYSTEM_GAIN=0.5\nMAC_MICROPHONE_GAIN=2\n", &mut s);
        assert_eq!(s.mac_system_gain, 0.5);
        assert_eq!(s.mac_microphone_gain, 2.0);

        let mut s = base();
        import_dotenv("MAC_SYSTEM_GAIN=0\n", &mut s);
        assert_eq!(s.mac_system_gain, 0.0);

        let mut s = base();
        import_dotenv("MAC_SYSTEM_GAIN=-1\nMAC_MICROPHONE_GAIN=nan\n", &mut s);
        assert_eq!(s.mac_system_gain, 1.0);
        assert_eq!(s.mac_microphone_gain, 1.0);
    }

    #[test]
    fn dotenv_parser_handles_comments_export_quotes_and_last_value_wins() {
        let contents = "\
# comment line
export OBSIDIAN_DIR=\"/vault/quoted path\"
WINDOWS_MICROPHONE='Mic 1' # trailing comment
WINDOWS_SPEAKER=Spk # trailing comment
WHISPER_MODEL=tiny
not a key value pair
WHISPER_MODEL=medium
UNKNOWN_KEY=whatever
";
        let mut s = base();
        import_dotenv(contents, &mut s);
        assert_eq!(s.output_dir, PathBuf::from("/vault/quoted path"));
        assert_eq!(s.windows_microphone, "Mic 1");
        assert_eq!(s.windows_speaker, "Spk");
        assert_eq!(s.whisper_model, WhisperSize::Medium);
    }

    #[test]
    fn dotenv_migration_leaves_fields_without_legacy_keys_untouched() {
        let mut s = base();
        s.theme = ThemeMode::Dark;
        s.ai_api_key = "keep-me".to_string();
        s.corrections = vec![CorrectionRule {
            find: "a".to_string(),
            replace: "b".to_string(),
            case_sensitive: false,
            whole_word: false,
        }];
        let before = s.clone();
        import_dotenv("THEME=light\nAI_API_KEY=other\n", &mut s);
        assert_eq!(s, before);
    }

    #[test]
    fn settings_round_trip_through_json_with_lowercase_enum_names() {
        let mut s = base();
        s.stt_engine = SttEngine::Whistle;
        s.ai_source = AiSource::Api;
        s.theme = ThemeMode::Light;
        s.corrections = vec![CorrectionRule {
            find: "ok".to_string(),
            replace: "OK".to_string(),
            case_sensitive: true,
            whole_word: true,
        }];
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"stt_engine\":\"whistle\""), "{json}");
        assert!(json.contains("\"ai_source\":\"api\""), "{json}");
        let back: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn partial_settings_json_fills_missing_fields_with_defaults() {
        let s: AppSettings =
            serde_json::from_str(r#"{"theme":"dark","min_report_words":80}"#).unwrap();
        assert_eq!(s.theme, ThemeMode::Dark);
        assert_eq!(s.min_report_words, 80);
        assert_eq!(s.transcription_language, "es");
        assert_eq!(s.ai_endpoint, "http://127.0.0.1:8080/v1");
    }

    #[test]
    fn correction_rule_flags_default_to_false_when_omitted() {
        let r: CorrectionRule = serde_json::from_str(r#"{"find":"a","replace":"b"}"#).unwrap();
        assert!(!r.case_sensitive);
        assert!(!r.whole_word);
    }

    #[test]
    fn report_skill_prompt_defaults_to_the_meeting_report_analyst_prompt() {
        let s = AppSettings::default();
        assert_eq!(s.report_skill_prompt, crate::report::default_report_skill_prompt());
        assert!(!s.report_skill_prompt.trim().is_empty());
    }

    #[test]
    fn report_skill_prompt_falls_back_to_default_when_omitted_from_json() {
        let s: AppSettings = serde_json::from_str(r#"{"theme":"dark"}"#).unwrap();
        assert_eq!(s.report_skill_prompt, crate::report::default_report_skill_prompt());
    }

    #[test]
    fn report_skill_prompt_keeps_a_stored_value_when_present_in_json() {
        let s: AppSettings =
            serde_json::from_str(r#"{"report_skill_prompt":"Mi prompt"}"#).unwrap();
        assert_eq!(s.report_skill_prompt, "Mi prompt");
    }

    #[test]
    fn hf_and_sampling_settings_default_to_hf_profile_without_overrides() {
        let s = AppSettings::default();
        assert!(s.use_hf_recommended_config);
        assert!(s.ai_disable_thinking);
        assert_eq!(s.ai_temperature, None);
        assert_eq!(s.ai_top_p, None);
        assert_eq!(s.ai_top_k, None);
        assert_eq!(s.ai_repetition_penalty, None);
    }

    #[test]
    fn settings_json_without_hf_and_sampling_fields_keeps_defaults() {
        let s: AppSettings = serde_json::from_str(r#"{"theme":"dark"}"#).unwrap();
        assert!(s.use_hf_recommended_config);
        assert!(s.ai_disable_thinking);
        assert_eq!(s.ai_temperature, None);
        assert_eq!(s.ai_top_p, None);
        assert_eq!(s.ai_top_k, None);
        assert_eq!(s.ai_repetition_penalty, None);
    }

    #[test]
    fn settings_json_keeps_stored_hf_and_sampling_fields() {
        let s: AppSettings = serde_json::from_str(
            r#"{"use_hf_recommended_config":false,"ai_disable_thinking":false,"ai_temperature":0.7,"ai_top_p":0.9,"ai_top_k":40,"ai_repetition_penalty":1.1}"#,
        )
        .unwrap();
        assert!(!s.use_hf_recommended_config);
        assert!(!s.ai_disable_thinking);
        assert_eq!(s.ai_temperature, Some(0.7));
        assert_eq!(s.ai_top_p, Some(0.9));
        assert_eq!(s.ai_top_k, Some(40));
        assert_eq!(s.ai_repetition_penalty, Some(1.1));
    }
}
