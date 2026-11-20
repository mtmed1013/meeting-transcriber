//! Persisted application settings: the Tauri-managed store, the redacted view the settings window
//! reads, partial updates, native folder picking, and the one-time legacy `.env` import.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use mt_core::config::{
    AiSource, AppSettings, CorrectionRule, SttEngine, ThemeMode, WhisperSize, import_dotenv,
};
use mt_core::report::{
    default_report_skill_prompt, is_legacy_default_report_skill_prompt, write_atomic_in_place,
};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

/// Settings file under the app config directory.
const SETTINGS_FILE: &str = "settings.json";
/// Legacy dotenv file, looked up in the working directory and then the resource directory.
const DOTENV_FILE: &str = ".env";
/// Folder under the app local data directory that holds local AI models by default.
const AI_MODELS_DIR: &str = "ai-models";
/// Smallest buffer the Windows recorder accepts (same clamp as `import_dotenv`).
const MIN_WINDOWS_BUFFER_SECONDS: f64 = 0.2;

/// OS and CPU architecture names, as reported by `std::env::consts`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    pub os: &'static str,
    pub arch: &'static str,
}

impl Platform {
    pub fn current() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        }
    }

    /// Apple Silicon Macs, the only machines that run the MLX runtime.
    pub fn is_apple_silicon(&self) -> bool {
        self.os == "macos" && self.arch == "aarch64"
    }

    pub fn recommended_runtime(&self) -> &'static str {
        if self.is_apple_silicon() {
            "mlx"
        } else {
            "gguf"
        }
    }

    /// Mirrors `transcription_engines.py`: Whistle runs on Windows x64 and Apple Silicon macOS.
    pub fn whistle_supported(&self) -> bool {
        (self.os == "windows" && self.arch == "x86_64") || self.is_apple_silicon()
    }
}

/// Current settings plus the file they persist to. Writes go through [`SettingsStore::update`].
#[derive(Debug)]
pub struct SettingsStore {
    settings: Mutex<AppSettings>,
    config_path: PathBuf,
}

impl SettingsStore {
    /// Reads `settings.json`. When it is absent, imports the legacy `.env` if one exists and
    /// persists the result, so the migration runs once.
    pub fn load(app: &AppHandle) -> Self {
        let config_path = app_config_dir(app).join(SETTINGS_FILE);
        match fs::read_to_string(&config_path) {
            Ok(text) => Self::new(config_path, parse_settings(&text)),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                let mut settings = AppSettings::default();
                let migrated =
                    match find_legacy_dotenv(app).and_then(|p| fs::read_to_string(p).ok()) {
                        Some(text) => {
                            import_dotenv(&text, &mut settings);
                            true
                        }
                        None => false,
                    };
                let store = Self::new(config_path, settings);
                if migrated && let Err(err) = store.save(store.get()) {
                    eprintln!("settings: could not persist the migrated .env: {err}");
                }
                store
            }
            Err(err) => {
                eprintln!(
                    "settings: could not read {}: {err}; using defaults",
                    config_path.display()
                );
                Self::new(config_path, AppSettings::default())
            }
        }
    }

    pub fn get(&self) -> AppSettings {
        self.lock().clone()
    }

    /// Persists `next` atomically and makes it current. On failure the previous settings stay.
    pub fn save(&self, next: AppSettings) -> Result<(), String> {
        self.update(|_| Ok(next)).map(|_| ())
    }

    /// Computes the next settings from the current ones while holding the lock, so concurrent
    /// updates cannot interleave and overwrite each other.
    pub fn update<F>(&self, change: F) -> Result<AppSettings, String>
    where
        F: FnOnce(&AppSettings) -> Result<AppSettings, String>,
    {
        let mut current = self.lock();
        let next = change(&current)?;
        self.write(&next)?;
        *current = next.clone();
        Ok(next)
    }

    fn new(config_path: PathBuf, settings: AppSettings) -> Self {
        Self {
            settings: Mutex::new(settings),
            config_path,
        }
    }

    fn lock(&self) -> MutexGuard<'_, AppSettings> {
        self.settings.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn write(&self, settings: &AppSettings) -> Result<(), String> {
        let json = serde_json::to_string_pretty(settings).map_err(|err| err.to_string())?;
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| format!("No se pudo crear {}: {err}", parent.display()))?;
        }
        write_atomic_in_place(&self.config_path, &json)
            .map_err(|err| format!("No se pudo guardar la configuración: {err}"))
    }
}

/// Settings as the settings window sees them. The raw API key never leaves the backend; only
/// whether one is configured does.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsView {
    pub output_dir: String,
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
    pub ai_models_dir: String,
    pub ai_model_id: String,
    pub ai_endpoint: String,
    pub ai_model_name: String,
    pub ai_api_key_configured: bool,
    pub use_hf_recommended_config: bool,
    pub ai_disable_thinking: bool,
    pub ai_temperature: Option<f32>,
    pub ai_top_p: Option<f32>,
    pub ai_top_k: Option<u32>,
    pub ai_repetition_penalty: Option<f32>,
    pub context_limit_tokens: u32,
    pub max_output_tokens: u32,
    pub min_report_words: usize,
    pub replace_existing_report: bool,
    pub corrections: Vec<CorrectionRule>,
    pub theme: ThemeMode,
    pub report_skill_prompt: String,
    pub default_report_skill_prompt: String,
    pub recommended_runtime: String,
    pub platform_os: String,
    pub platform_arch: String,
    pub whistle_supported: bool,
    pub default_ai_models_dir: String,
}

impl SettingsView {
    pub fn new(settings: &AppSettings, platform: Platform, default_ai_models_dir: &Path) -> Self {
        Self {
            output_dir: path_text(&settings.output_dir),
            stt_engine: settings.stt_engine,
            whisper_model: settings.whisper_model,
            transcription_language: settings.transcription_language.clone(),
            output_language: settings.output_language.clone(),
            speaker_markers: settings.speaker_markers,
            whisper_vad_filter: settings.whisper_vad_filter,
            windows_microphone: settings.windows_microphone.clone(),
            windows_speaker: settings.windows_speaker.clone(),
            windows_buffer_seconds: settings.windows_buffer_seconds,
            mac_system_gain: settings.mac_system_gain,
            mac_microphone_gain: settings.mac_microphone_gain,
            ai_enabled: settings.ai_enabled,
            ai_source: settings.ai_source,
            ai_models_dir: path_text(&settings.ai_models_dir),
            ai_model_id: settings.ai_model_id.clone(),
            ai_endpoint: settings.ai_endpoint.clone(),
            ai_model_name: settings.ai_model_name.clone(),
            ai_api_key_configured: !settings.ai_api_key.is_empty(),
            use_hf_recommended_config: settings.use_hf_recommended_config,
            ai_disable_thinking: settings.ai_disable_thinking,
            ai_temperature: settings.ai_temperature,
            ai_top_p: settings.ai_top_p,
            ai_top_k: settings.ai_top_k,
            ai_repetition_penalty: settings.ai_repetition_penalty,
            context_limit_tokens: settings.context_limit_tokens,
            max_output_tokens: settings.max_output_tokens,
            min_report_words: settings.min_report_words,
            replace_existing_report: settings.replace_existing_report,
            corrections: settings.corrections.clone(),
            theme: settings.theme,
            report_skill_prompt: non_blank_report_skill_prompt(&settings.report_skill_prompt),
            default_report_skill_prompt: default_report_skill_prompt(),
            recommended_runtime: platform.recommended_runtime().to_string(),
            platform_os: platform.os.to_string(),
            platform_arch: platform.arch.to_string(),
            whistle_supported: platform.whistle_supported(),
            default_ai_models_dir: path_text(default_ai_models_dir),
        }
    }
}

/// Partial settings update from the settings window. Absent fields keep their saved value.
///
/// An empty `ai_api_key` keeps the saved key unless `clear_ai_api_key` is true. A non-empty key
/// always replaces the saved one.
#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SettingsUpdate {
    pub output_dir: Option<String>,
    pub stt_engine: Option<SttEngine>,
    pub whisper_model: Option<WhisperSize>,
    pub transcription_language: Option<String>,
    pub output_language: Option<String>,
    pub speaker_markers: Option<bool>,
    pub whisper_vad_filter: Option<bool>,
    pub windows_microphone: Option<String>,
    pub windows_speaker: Option<String>,
    pub windows_buffer_seconds: Option<f64>,
    pub mac_system_gain: Option<f64>,
    pub mac_microphone_gain: Option<f64>,
    pub ai_enabled: Option<bool>,
    pub ai_source: Option<AiSource>,
    pub ai_models_dir: Option<String>,
    pub ai_model_id: Option<String>,
    pub ai_endpoint: Option<String>,
    pub ai_model_name: Option<String>,
    pub ai_api_key: Option<String>,
    pub clear_ai_api_key: bool,
    pub use_hf_recommended_config: Option<bool>,
    pub ai_disable_thinking: Option<bool>,
    pub ai_temperature: Option<f32>,
    pub ai_top_p: Option<f32>,
    pub ai_top_k: Option<u32>,
    pub ai_repetition_penalty: Option<f32>,
    /// When true, resets the four sampling overrides to `None` before applying any provided ones.
    pub clear_ai_sampling_overrides: bool,
    pub context_limit_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub min_report_words: Option<usize>,
    pub replace_existing_report: Option<bool>,
    pub corrections: Option<Vec<CorrectionRule>>,
    pub theme: Option<ThemeMode>,
    pub report_skill_prompt: Option<String>,
}

impl SettingsUpdate {
    /// Applies the patch to `current`. Nothing changes when a number is rejected.
    pub fn apply(self, current: &AppSettings) -> Result<AppSettings, String> {
        let mut next = current.clone();
        if let Some(value) = self.output_dir {
            next.output_dir = PathBuf::from(value);
        }
        if let Some(value) = self.stt_engine {
            next.stt_engine = value;
        }
        if let Some(value) = self.whisper_model {
            next.whisper_model = value;
        }
        if let Some(value) = self.transcription_language {
            next.transcription_language = value;
        }
        if let Some(value) = self.output_language {
            next.output_language = value;
        }
        if let Some(value) = self.speaker_markers {
            next.speaker_markers = value;
        }
        if let Some(value) = self.whisper_vad_filter {
            next.whisper_vad_filter = value;
        }
        if let Some(value) = self.windows_microphone {
            next.windows_microphone = value;
        }
        if let Some(value) = self.windows_speaker {
            next.windows_speaker = value;
        }
        if let Some(value) = self.windows_buffer_seconds {
            next.windows_buffer_seconds =
                non_negative("windowsBufferSeconds", value)?.max(MIN_WINDOWS_BUFFER_SECONDS);
        }
        if let Some(value) = self.mac_system_gain {
            next.mac_system_gain = non_negative("macSystemGain", value)?;
        }
        if let Some(value) = self.mac_microphone_gain {
            next.mac_microphone_gain = non_negative("macMicrophoneGain", value)?;
        }
        if let Some(value) = self.ai_enabled {
            next.ai_enabled = value;
        }
        if let Some(value) = self.ai_source {
            next.ai_source = value;
        }
        if let Some(value) = self.ai_models_dir {
            next.ai_models_dir = PathBuf::from(value);
        }
        if let Some(value) = self.ai_model_id {
            next.ai_model_id = value;
        }
        if let Some(value) = self.ai_endpoint {
            next.ai_endpoint = value;
        }
        if let Some(value) = self.ai_model_name {
            next.ai_model_name = value;
        }
        if self.clear_ai_sampling_overrides {
            next.ai_temperature = None;
            next.ai_top_p = None;
            next.ai_top_k = None;
            next.ai_repetition_penalty = None;
        }
        if let Some(value) = self.ai_temperature {
            next.ai_temperature = Some(temperature(value)?);
        }
        if let Some(value) = self.ai_top_p {
            next.ai_top_p = Some(top_p(value)?);
        }
        if let Some(value) = self.ai_top_k {
            next.ai_top_k = Some(top_k(value)?);
        }
        if let Some(value) = self.ai_repetition_penalty {
            next.ai_repetition_penalty = Some(repetition_penalty(value)?);
        }
        if let Some(value) = self.use_hf_recommended_config {
            next.use_hf_recommended_config = value;
        }
        if let Some(value) = self.ai_disable_thinking {
            next.ai_disable_thinking = value;
        }
        next.ai_api_key = match self.ai_api_key.filter(|key| !key.is_empty()) {
            Some(key) => key,
            None if self.clear_ai_api_key => String::new(),
            None => current.ai_api_key.clone(),
        };
        if let Some(value) = self.context_limit_tokens {
            next.context_limit_tokens = value;
        }
        if let Some(value) = self.max_output_tokens {
            next.max_output_tokens = value;
        }
        if let Some(value) = self.min_report_words {
            next.min_report_words = value;
        }
        if let Some(value) = self.replace_existing_report {
            next.replace_existing_report = value;
        }
        if let Some(value) = self.corrections {
            next.corrections = value;
        }
        if let Some(value) = self.theme {
            next.theme = value;
        }
        if let Some(value) = self.report_skill_prompt {
            let value = value.trim();
            next.report_skill_prompt = if value.is_empty() {
                default_report_skill_prompt()
            } else {
                value.to_string()
            };
        }
        Ok(next)
    }
}

/// Default folder for local AI models: `<app local data>/ai-models`.
pub fn default_ai_models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map(|dir| dir.join(AI_MODELS_DIR))
        .map_err(|err| err.to_string())
}

/// The configured AI models folder, or the default one when none is configured.
pub fn resolve_ai_models_dir(app: &AppHandle, settings: &AppSettings) -> Result<PathBuf, String> {
    if settings.ai_models_dir.as_os_str().is_empty() {
        default_ai_models_dir(app)
    } else {
        Ok(settings.ai_models_dir.clone())
    }
}

#[tauri::command]
pub fn get_settings(
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<SettingsView, String> {
    view_for(&app, &store.get())
}

#[tauri::command]
pub fn update_settings(
    update: SettingsUpdate,
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<SettingsView, String> {
    let next = store.update(|current| update.apply(current))?;
    view_for(&app, &next)
}

/// Opens a native folder picker. Resolves to `None` when the user cancels.
#[tauri::command]
pub async fn pick_directory(
    title: Option<String>,
    app: AppHandle,
) -> Result<Option<String>, String> {
    let mut dialog = app.dialog().file();
    if let Some(title) = title {
        dialog = dialog.set_title(title);
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    dialog.pick_folder(move |picked| {
        let _ = sender.send(picked);
    });
    let picked = receiver.await.map_err(|err| err.to_string())?;
    picked
        .map(|path| {
            path.into_path()
                .map(|dir| path_text(&dir))
                .map_err(|err| err.to_string())
        })
        .transpose()
}

/// Opens a native file picker for an existing meeting note (`.md` or `.txt`). Resolves to `None`
/// when the user cancels.
#[tauri::command]
pub async fn pick_meeting_file(app: AppHandle) -> Result<Option<String>, String> {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .add_filter("Notas de reunión", &["md", "txt"])
        .pick_file(move |picked| {
            let _ = sender.send(picked);
        });
    let picked = receiver.await.map_err(|err| err.to_string())?;
    picked
        .map(|path| {
            path.into_path()
                .map(|file| path_text(&file))
                .map_err(|err| err.to_string())
        })
        .transpose()
}

/// Imports the first legacy `.env` found in the working or resource directory into the settings.
#[tauri::command]
pub fn import_legacy_dotenv(
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<SettingsView, String> {
    let path = find_legacy_dotenv(&app)
        .ok_or_else(|| "No se encontró un archivo .env heredado.".to_string())?;
    let text = fs::read_to_string(&path)
        .map_err(|err| format!("No se pudo leer {}: {err}", path.display()))?;
    let next = store.update(|current| {
        let mut next = current.clone();
        import_dotenv(&text, &mut next);
        Ok(next)
    })?;
    view_for(&app, &next)
}

fn view_for(app: &AppHandle, settings: &AppSettings) -> Result<SettingsView, String> {
    let default_dir = default_ai_models_dir(app)?;
    Ok(SettingsView::new(
        settings,
        Platform::current(),
        &default_dir,
    ))
}

fn app_config_dir(app: &AppHandle) -> PathBuf {
    app.path().app_config_dir().unwrap_or_else(|err| {
        eprintln!("settings: no app config directory: {err}");
        PathBuf::from(".")
    })
}

/// First existing `.env` in the working directory, then in the resource directory.
fn find_legacy_dotenv(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(DOTENV_FILE));
    }
    if let Ok(resources) = app.path().resource_dir() {
        candidates.push(resources.join(DOTENV_FILE));
    }
    candidates.into_iter().find(|path| path.is_file())
}

fn parse_settings(text: &str) -> AppSettings {
    let mut settings: AppSettings = serde_json::from_str(text).unwrap_or_else(|err| {
        eprintln!("settings: invalid settings.json, using defaults: {err}");
        AppSettings::default()
    });
    settings.report_skill_prompt = non_blank_report_skill_prompt(&settings.report_skill_prompt);
    settings
}

fn non_negative(name: &str, value: f64) -> Result<f64, String> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(format!(
            "{name} debe ser un número finito mayor o igual que 0."
        ))
    }
}

fn temperature(value: f32) -> Result<f32, String> {
    if value.is_finite() && (0.0..=2.0).contains(&value) {
        Ok(value)
    } else {
        Err("aiTemperature debe ser un número finito entre 0 y 2.".to_string())
    }
}

fn top_p(value: f32) -> Result<f32, String> {
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err("aiTopP debe ser un número finito mayor que 0 y menor o igual que 1.".to_string())
    }
}

fn top_k(value: u32) -> Result<u32, String> {
    if value >= 1 {
        Ok(value)
    } else {
        Err("aiTopK debe ser un entero mayor o igual que 1.".to_string())
    }
}

fn repetition_penalty(value: f32) -> Result<f32, String> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("aiRepetitionPenalty debe ser un número finito mayor que 0.".to_string())
    }
}

/// The saved report prompt, or the default one when the saved value is blank or matches an
/// unmodified previous built-in default.
fn non_blank_report_skill_prompt(saved: &str) -> String {
    if saved.trim().is_empty() || is_legacy_default_report_skill_prompt(saved) {
        default_report_skill_prompt()
    } else {
        saved.to_string()
    }
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const MAC_ARM: Platform = Platform {
        os: "macos",
        arch: "aarch64",
    };
    const WINDOWS_X64: Platform = Platform {
        os: "windows",
        arch: "x86_64",
    };
    const MAC_INTEL: Platform = Platform {
        os: "macos",
        arch: "x86_64",
    };

    fn with_key(key: &str) -> AppSettings {
        AppSettings {
            ai_api_key: key.to_string(),
            ..AppSettings::default()
        }
    }

    fn view(settings: &AppSettings, platform: Platform) -> Value {
        let view = SettingsView::new(settings, platform, Path::new("/data/ai-models"));
        serde_json::to_value(view).unwrap()
    }

    fn update(value: Value) -> SettingsUpdate {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn view_redacts_the_api_key_and_reports_only_whether_one_is_configured() {
        let secret = view(&with_key("sk-secret-123"), MAC_ARM);
        assert_eq!(secret["aiApiKeyConfigured"], json!(true));
        assert!(secret.get("aiApiKey").is_none());
        assert!(!secret.to_string().contains("sk-secret-123"));

        let empty = view(&AppSettings::default(), MAC_ARM);
        assert_eq!(empty["aiApiKeyConfigured"], json!(false));
    }

    #[test]
    fn view_reports_runtime_and_whistle_support_per_platform() {
        let mac = view(&AppSettings::default(), MAC_ARM);
        assert_eq!(mac["recommendedRuntime"], json!("mlx"));
        assert_eq!(mac["whistleSupported"], json!(true));
        assert_eq!(mac["platformOs"], json!("macos"));
        assert_eq!(mac["platformArch"], json!("aarch64"));
        assert_eq!(mac["defaultAiModelsDir"], json!("/data/ai-models"));

        let windows = view(&AppSettings::default(), WINDOWS_X64);
        assert_eq!(windows["recommendedRuntime"], json!("gguf"));
        assert_eq!(windows["whistleSupported"], json!(true));

        let intel = view(&AppSettings::default(), MAC_INTEL);
        assert_eq!(intel["recommendedRuntime"], json!("gguf"));
        assert_eq!(intel["whistleSupported"], json!(false));
    }

    #[test]
    fn update_without_a_new_key_keeps_the_saved_key() {
        let current = with_key("saved");
        let blank = update(json!({ "aiApiKey": "", "theme": "dark" }))
            .apply(&current)
            .unwrap();
        assert_eq!(blank.ai_api_key, "saved");
        assert_eq!(blank.theme, ThemeMode::Dark);

        let absent = update(json!({ "theme": "light" })).apply(&current).unwrap();
        assert_eq!(absent.ai_api_key, "saved");
    }

    #[test]
    fn update_clears_the_key_only_when_asked_and_no_new_key_is_sent() {
        let current = with_key("saved");
        let cleared = update(json!({ "aiApiKey": "", "clearAiApiKey": true }))
            .apply(&current)
            .unwrap();
        assert_eq!(cleared.ai_api_key, "");

        let replaced = update(json!({ "aiApiKey": "new-key", "clearAiApiKey": false }))
            .apply(&current)
            .unwrap();
        assert_eq!(replaced.ai_api_key, "new-key");
    }

    #[test]
    fn update_rejects_negative_gains_and_clamps_the_buffer_like_the_legacy_import() {
        let current = AppSettings::default();
        assert!(
            update(json!({ "macSystemGain": -1.0 }))
                .apply(&current)
                .is_err()
        );
        let nan = SettingsUpdate {
            mac_microphone_gain: Some(f64::NAN),
            ..SettingsUpdate::default()
        };
        assert!(nan.apply(&current).is_err());

        let clamped = update(json!({ "windowsBufferSeconds": 0.01 }))
            .apply(&current)
            .unwrap();
        assert_eq!(clamped.windows_buffer_seconds, 0.2);
    }

    #[test]
    fn update_sets_the_report_skill_prompt_and_trims_it() {
        let current = AppSettings::default();
        let next = update(json!({ "reportSkillPrompt": "  Custom prompt  " }))
            .apply(&current)
            .unwrap();
        assert_eq!(next.report_skill_prompt, "Custom prompt");
    }

    #[test]
    fn update_restores_the_default_report_skill_prompt_when_blank() {
        let current = AppSettings {
            report_skill_prompt: "Custom prompt".to_string(),
            ..AppSettings::default()
        };
        let blank = update(json!({ "reportSkillPrompt": "   " }))
            .apply(&current)
            .unwrap();
        assert_eq!(blank.report_skill_prompt, default_report_skill_prompt());
    }

    #[test]
    fn update_without_report_skill_prompt_keeps_the_saved_one() {
        let current = AppSettings {
            report_skill_prompt: "Custom prompt".to_string(),
            ..AppSettings::default()
        };
        let next = update(json!({ "theme": "dark" })).apply(&current).unwrap();
        assert_eq!(next.report_skill_prompt, "Custom prompt");
    }

    #[test]
    fn view_shows_the_default_report_skill_prompt_when_the_saved_one_is_blank() {
        let settings = AppSettings {
            report_skill_prompt: " ".to_string(),
            ..AppSettings::default()
        };
        let view = view(&settings, MAC_ARM);
        assert_eq!(view["reportSkillPrompt"], json!(default_report_skill_prompt()));
    }

    #[test]
    fn view_exposes_the_hf_profile_and_sampling_overrides_in_camel_case() {
        let settings = AppSettings {
            use_hf_recommended_config: false,
            ai_disable_thinking: false,
            ai_temperature: Some(0.7),
            ai_top_k: Some(40),
            ..AppSettings::default()
        };
        let view = view(&settings, MAC_ARM);
        assert_eq!(view["useHfRecommendedConfig"], json!(false));
        assert_eq!(view["aiDisableThinking"], json!(false));
        assert_eq!(view["aiTemperature"], json!(0.7_f32));
        assert_eq!(view["aiTopP"], Value::Null);
        assert_eq!(view["aiTopK"], json!(40));
        assert_eq!(view["aiRepetitionPenalty"], Value::Null);
    }

    #[test]
    fn update_validates_sampling_overrides_and_clears_them_on_request() {
        let current = AppSettings {
            ai_temperature: Some(0.5),
            ai_top_p: Some(0.9),
            ai_top_k: Some(20),
            ai_repetition_penalty: Some(1.2),
            ..AppSettings::default()
        };
        for rejected in [
            json!({ "aiTemperature": 2.5 }),
            json!({ "aiTemperature": -0.1 }),
            json!({ "aiTopP": 0.0 }),
            json!({ "aiTopP": 1.5 }),
            json!({ "aiTopK": 0 }),
            json!({ "aiRepetitionPenalty": 0.0 }),
        ] {
            assert!(update(rejected.clone()).apply(&current).is_err(), "{rejected}");
        }
        let nan = SettingsUpdate {
            ai_top_p: Some(f32::NAN),
            ..SettingsUpdate::default()
        };
        assert!(nan.apply(&current).is_err());

        let cleared = update(json!({ "clearAiSamplingOverrides": true }))
            .apply(&current)
            .unwrap();
        assert_eq!(cleared.ai_temperature, None);
        assert_eq!(cleared.ai_top_p, None);
        assert_eq!(cleared.ai_top_k, None);
        assert_eq!(cleared.ai_repetition_penalty, None);

        let replaced = update(json!({
            "clearAiSamplingOverrides": true,
            "aiTopK": 50,
            "aiTemperature": 1.0
        }))
        .apply(&current)
        .unwrap();
        assert_eq!(replaced.ai_temperature, Some(1.0));
        assert_eq!(replaced.ai_top_k, Some(50));
        assert_eq!(replaced.ai_top_p, None);
    }

    #[test]
    fn parse_settings_upgrades_unmodified_legacy_default_skill_prompt_and_keeps_custom() {
        let legacy = "You are meeting-report-analyst. Turn a meeting transcript into an accurate, structured report so that someone who did not attend understands its purpose, topics, decisions, action items, questions, blockers, and key facts.\n\nLANGUAGE: OUTPUT_LANGUAGE is authoritative for all generated text. Do not detect or switch language from the transcript. Keep technical terms, code, API names, URLs, product names, and proper nouns unchanged.\n\nACCURACY:\n- Read the whole transcript first. Use only what it states.\n- Never invent decisions, tasks, owners, deadlines, names, dates, numbers, requirements, risks, or answers.\n- Never invent sentiment, engagement, or quality metrics.\n- Use null for a missing owner or due date. Never guess an owner. Keep vague deadlines verbatim, such as 'tomorrow'.\n- Keep speaker labels as written.\n- When statements change, prefer the final agreed position.\n- Fix obvious speech-to-text errors from context. Never guess uncertain meaning.\n\nSECTIONS (call save_meeting_report):\n- executive_summary: 3 to 7 bullets, 150 words at most: purpose, main discussion, outcome, decisions, next steps, major blockers. Skip small talk.\n- key_decisions: explicit decisions, commitments, or approvals only. A suggestion ('maybe', 'we could') is not a decision. Add reason only if explicitly discussed.\n- topics_and_discussion: 3 to 8 chronological chapters, merging related turns. timestamp only if stated, else null. points: substantive summaries.\n- action_items: executable commitments only, with owner or null, task, and due or null.\n- key_questions: material questions about requirements, architecture, rules, risks, deadlines, or ownership. Give answer, or status 'Open' or 'Partially resolved' in OUTPUT_LANGUAGE.\n- open_issues: unresolved problems, blockers, or risks explicitly discussed. Add impact and next_step only if stated.\n- important_facts: exact dates, deadlines, limits, quantities, prices, percentages, URLs, systems, versions, constraints, dependencies.\n\nDELIVERY: Deliver all sections only through save_meeting_report. Use empty arrays when nothing applies. Never reply with prose or Markdown.";
        let upgraded = parse_settings(&json!({ "report_skill_prompt": legacy }).to_string());
        assert_eq!(upgraded.report_skill_prompt, default_report_skill_prompt());

        let custom = parse_settings(&json!({ "report_skill_prompt": "My custom skill" }).to_string());
        assert_eq!(custom.report_skill_prompt, "My custom skill");
    }
}
