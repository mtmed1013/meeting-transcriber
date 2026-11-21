//! Post-meeting AI report and the local AI runtime. Turns a finished meeting file into its report
//! (short meetings are rendered deterministically without a model), owns the local `mlx_lm` /
//! `llama-server` runtime install, and exposes the window and note helpers the UI calls.

use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use mt_ai::runtime::{
    LocalRuntimeKind, LocalServerEndpoint, LocalServerManager, LocalServerSpec,
};
use mt_ai::{ModelConfig, OpenAiCompatibleProvider, analyze_transcript};
use mt_core::{
    AiSource, AppSettings, apply_corrections, assemble_markdown_report, count_spoken_words,
    short_meeting_markdown, strip_existing_report, write_atomic_in_place,
};
use mt_models::{
    SuggestedModelConfig, load_local_suggested_config, model_type_from_dir, supported_model_types,
    unsupported_reason,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;

use crate::mlx_runtime::{self, MlxStatus};
use crate::settings_store::{Platform, SettingsStore, resolve_ai_models_dir};
use crate::stt_provisioning::{blocking, hidden_command, io_error, run_checked, unique_suffix};

/// Progress of `install_ai_runtime`, payload `{ phase, message }`.
const PROGRESS_EVENT: &str = "ai-runtime-progress";
/// Folder under the app data directory that holds the managed MLX runtime (see `mlx_runtime`).
const MLX_DIR: &str = "mlx";
const LLAMA_DIR: &str = "llama";
/// Progress phase of MLX install and update events on `PROGRESS_EVENT`.
const MLX_PROGRESS_PHASE: &str = "mlx";
/// Shown when the managed MLX runtime is not installed.
const MLX_MISSING: &str = "Falta el runtime MLX; instálalo desde Configuración.";
/// Server-log lines quoted when a local model fails to start.
const LOG_TAIL_LINES: usize = 5;
/// Separates the transcript from the warning block a failed AI step appends to the note.
const AI_FAILURE_SEPARATOR: &str = "\n\n---\n\n";
const AI_FAILURE_PREFIX: &str = "⚠️ Reporte IA no generado: ";
const LLAMA_RELEASE_URL: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";
/// Newest first. `latest` can point at a nightly-only tag, so the build-tagged releases are read too.
const LLAMA_RELEASES_URL: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30";
const LLAMA_ASSET_PREFIX: &str = "llama-";
const LLAMA_SERVER_EXE: &str = if cfg!(windows) {
    "llama-server.exe"
} else {
    "llama-server"
};
/// How deep the release archive is searched for `llama-server`.
const MAX_ARCHIVE_DEPTH: usize = 4;
/// The app has one window; settings and session are tabs inside it.
pub(crate) const MAIN_WINDOW: &str = "main";
/// Payload is the tab id to show: `"general"` (settings) or `"session"`.
const NAVIGATE_TAB_EVENT: &str = "navigate-tab";

/// Tauri-managed local AI runtime: the single owned server and an install-in-progress flag.
#[derive(Default)]
pub struct AiRuntimeState {
    manager: LocalServerManager,
    installing: AtomicBool,
}

impl AiRuntimeState {
    /// Cheap handle to the shared server manager.
    pub(crate) fn manager(&self) -> LocalServerManager {
        self.manager.clone()
    }

    /// Kills the owned local server and forbids any later start. Called on app exit.
    pub fn shutdown(&self) {
        self.manager.shutdown();
    }
}

/// What the settings window shows about the local AI runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiRuntimeStatus {
    pub kind: LocalRuntimeKind,
    pub installed: bool,
    pub detail: String,
    pub can_install: bool,
    /// Installed `mlx-lm` version; `None` for llama.cpp or when MLX is not installed.
    pub version: Option<String>,
    /// Latest PyPI version, only known after `check_ai_runtime_update`.
    pub latest: Option<String>,
    pub update_available: bool,
}

/// Whether a local MLX model folder can run on the installed `mlx-lm`. `reason` is set exactly when
/// `supported` is false.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompatibility {
    pub supported: bool,
    pub model_type: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeProgress {
    phase: &'static str,
    message: String,
}

/// The fields of a GitHub release this module reads.
#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<ReleaseAsset>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    size: Option<u64>,
    /// `sha256:<hex>` when GitHub publishes one for the asset.
    #[serde(default)]
    digest: Option<String>,
}

/// Transcript before and after deterministic corrections. `original` is what the report keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedTranscript {
    pub original: String,
    pub cleaned: String,
}

// ---------------------------------------------------------------------------
// Post-meeting report
// ---------------------------------------------------------------------------

/// Strips a previous report when the settings ask for it, then applies the corrections to a copy.
pub fn prepare_transcript(raw: &str, settings: &AppSettings) -> PreparedTranscript {
    let without_warning = strip_ai_failure_block(raw);
    let original = if settings.replace_existing_report {
        strip_existing_report(&without_warning)
    } else {
        without_warning
    };
    let cleaned = apply_corrections(&original, &settings.corrections);
    PreparedTranscript { original, cleaned }
}

/// Removes the warning block a failed AI step left at the end of the note, so a retry reads only the
/// transcript. A note without that trailing block, or with text after it, is returned unchanged.
fn strip_ai_failure_block(raw: &str) -> String {
    let marker = format!("{AI_FAILURE_SEPARATOR}{AI_FAILURE_PREFIX}");
    let Some(index) = raw.rfind(&marker) else {
        return raw.to_string();
    };
    let reason = &raw[index + marker.len()..];
    if reason.trim_end().contains('\n') {
        return raw.to_string();
    }
    let body = raw[..index].trim_end_matches('\n');
    if body.is_empty() {
        String::new()
    } else {
        format!("{body}\n")
    }
}

/// Appends the AI failure block after the transcript kept in the note and returns the reason. A note
/// that cannot be written reports that as well, so the caller always has one message.
fn record_ai_failure(path: &Path, original: &str, reason: &str) -> String {
    let reason = reason.split_whitespace().collect::<Vec<_>>().join(" ");
    let body = original.trim_end_matches('\n');
    let markdown = format!("{body}{AI_FAILURE_SEPARATOR}{AI_FAILURE_PREFIX}{reason}\n");
    match write_report(path, &markdown) {
        Ok(()) => reason,
        Err(err) => format!("{reason} ({err})"),
    }
}

/// Whether the corrected transcript has fewer spoken words than the configured minimum.
pub fn is_short_meeting(prepared: &PreparedTranscript, min_words: usize) -> bool {
    count_spoken_words(&prepared.cleaned) < min_words
}

/// The deterministic report for a short meeting. Never calls a model.
pub fn short_meeting_report(settings: &AppSettings, original: &str) -> String {
    short_meeting_markdown(&settings.output_language, original)
}

/// Generates the report for a finished meeting file and writes it in place. When the AI step fails,
/// the transcript stays in the note with a warning block after it, and the reason is returned.
pub async fn run_post_meeting_report(app: &AppHandle, file_path: &str) -> Result<(), String> {
    let settings = app.state::<SettingsStore>().get();
    let path = PathBuf::from(file_path);
    let raw = fs::read_to_string(&path)
        .map_err(|err| format!("No se pudo leer {}: {err}", path.display()))?;
    let prepared = prepare_transcript(&raw, &settings);

    if is_short_meeting(&prepared, settings.min_report_words) {
        write_report(&path, &short_meeting_report(&settings, &prepared.original))?;
        emit_ai_progress(app, "done", "Informe guardado.", Some(100));
        return Ok(());
    }

    match generate_ai_report(app, &settings, &prepared).await {
        Ok(markdown) => {
            write_report(&path, &markdown)?;
            emit_ai_progress(app, "done", "Informe guardado.", Some(100));
            Ok(())
        }
        Err(reason) => Err(record_ai_failure(&path, &prepared.original, &reason)),
    }
}

/// The AI-written report as Markdown: provider creation, the model call, and, for local models,
/// starting the owned server.
async fn generate_ai_report(
    app: &AppHandle,
    settings: &AppSettings,
    prepared: &PreparedTranscript,
) -> Result<String, String> {
    let config = match settings.ai_source {
        AiSource::Api => api_model_config(settings)?,
        AiSource::Local => {
            emit_ai_progress(app, "prepare", "Iniciando servidor de IA local…", None);
            let endpoint = ensure_local_server(app, settings).await?;
            let suggested = load_local_suggested_config(&endpoint.model_path);
            local_model_config(settings, &endpoint, &suggested)
        }
    };
    let provider = OpenAiCompatibleProvider::new(config).map_err(|err| err.to_string())?;
    emit_ai_progress(app, "analyze", "Analizando la reunión con IA…", Some(5));
    let on_progress = |progress: mt_ai::AnalysisProgress| {
        // Map pipeline 0..=100 into 5..=90 so 95..=100 remain for assembling and writing.
        let mapped = 5 + ((u16::from(progress.percent) * 85) / 100) as u8;
        emit_ai_progress(app, "analyze", &progress.message, Some(mapped));
    };
    let sections = match analyze_transcript(
        &provider,
        &prepared.cleaned,
        &settings.output_language,
        &settings.report_skill_prompt,
        Some(&on_progress),
    )
    .await
    {
        Ok(sections) if !sections.is_empty() => sections,
        Ok(_) => {
            return Err(
                "el modelo no devolvió el informe (revisa que soporte tool calling)".to_string(),
            );
        }
        Err(err) => return Err(err.to_string()),
    };
    emit_ai_progress(app, "write", "Guardando el informe en la nota…", Some(95));
    Ok(assemble_markdown_report(
        &settings.output_language,
        &sections,
        &prepared.original,
    ))
}

/// Local loopback server request timeout in seconds (CPU inference on long meetings can take
/// several minutes per request).
const LOCAL_REQUEST_TIMEOUT_SECONDS: u64 = 600;

/// Connection settings for the remote endpoint the user configured.
fn api_model_config(settings: &AppSettings) -> Result<ModelConfig, String> {
    let endpoint = settings.ai_endpoint.trim();
    if endpoint.is_empty() {
        return Err("Configura el endpoint de IA en Configuración.".to_string());
    }
    let model = settings.ai_model_name.trim();
    if model.is_empty() {
        return Err("Configura el nombre del modelo de IA en Configuración.".to_string());
    }
    Ok(ModelConfig {
        endpoint: endpoint.to_string(),
        api_key: settings.ai_api_key.clone(),
        model: model.to_string(),
        ..token_limits(settings)
    })
}

/// Connection and sampling settings for the owned local server. The endpoint already ends in `/v1`.
/// With the Hugging Face profile on, the model's own sampling values and context window apply;
/// otherwise the manual `ai_*` overrides do.
fn local_model_config(
    settings: &AppSettings,
    endpoint: &LocalServerEndpoint,
    suggested: &SuggestedModelConfig,
) -> ModelConfig {
    let context_limit_tokens = effective_context_tokens(settings, suggested);
    let configured_output = settings.max_output_tokens as usize;
    // The reply reservation must leave room for the prompt inside the (possibly clamped) window.
    let max_output_tokens = if settings.use_hf_recommended_config
        && configured_output >= context_limit_tokens
    {
        context_limit_tokens / 2
    } else {
        configured_output
    };
    let (temperature, top_p, top_k, repetition_penalty, presence_penalty) =
        if settings.use_hf_recommended_config {
            (
                suggested.temperature.unwrap_or(0.2),
                suggested.top_p,
                suggested.top_k,
                suggested.repetition_penalty,
                suggested.presence_penalty,
            )
        } else {
            (
                settings.ai_temperature.unwrap_or(0.2),
                settings.ai_top_p,
                settings.ai_top_k,
                settings.ai_repetition_penalty,
                None,
            )
        };
    ModelConfig {
        endpoint: endpoint.endpoint.clone(),
        // The MLX server loads whichever model the request names, so name the local folder.
        model: endpoint.model_path.to_string_lossy().into_owned(),
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        presence_penalty,
        context_limit_tokens,
        max_output_tokens,
        timeout_seconds: LOCAL_REQUEST_TIMEOUT_SECONDS,
        thinking: suggested.supports_thinking_toggle && !settings.ai_disable_thinking,
        ..ModelConfig::default()
    }
}

/// Context window in tokens. With the Hugging Face profile on, a model that declares a maximum
/// context never gets more than that, whatever the user configured.
fn effective_context_tokens(settings: &AppSettings, suggested: &SuggestedModelConfig) -> usize {
    let configured = settings.context_limit_tokens as usize;
    match suggested.max_context_tokens.filter(|max| *max > 0) {
        Some(max) if settings.use_hf_recommended_config => configured.min(max as usize),
        _ => configured,
    }
}

fn token_limits(settings: &AppSettings) -> ModelConfig {
    ModelConfig {
        context_limit_tokens: settings.context_limit_tokens as usize,
        max_output_tokens: settings.max_output_tokens as usize,
        ..ModelConfig::default()
    }
}

async fn ensure_local_server(
    app: &AppHandle,
    settings: &AppSettings,
) -> Result<LocalServerEndpoint, String> {
    let models_dir = resolve_ai_models_dir(app, settings)?;
    let spec = local_server_spec(app, settings, &models_dir)?;
    let manager = app.state::<AiRuntimeState>().manager();
    manager
        .ensure_ready(&spec)
        .await
        .map_err(|err| err.to_string())
}

fn local_server_spec(
    app: &AppHandle,
    settings: &AppSettings,
    models_dir: &Path,
) -> Result<LocalServerSpec, String> {
    let kind = runtime_kind_for(Platform::current());
    let data_dir = app_data_dir(app)?;
    let root = runtime_root(&data_dir, kind);
    let executable = match kind {
        LocalRuntimeKind::Mlx => {
            // Only the interpreter is needed: `mt_ai` builds the `python -m mlx_lm server` arguments.
            let (python, _base_args) = mlx_runtime::server_command(&root)
                .map_err(|err| format!("{err}. Instálalo desde Configuración."))?;
            python
        }
        LocalRuntimeKind::Gguf => resolve_gguf_server(&data_dir).ok_or_else(|| {
            "No se encontró llama-server. Instálalo desde Configuración.".to_string()
        })?,
    };
    let model_path = local_model_path(models_dir, &settings.ai_model_id, kind)?;
    // Refuse with the real cause here; otherwise the server exits with code 1 and hides it.
    if let (LocalRuntimeKind::Mlx, Some(reason)) =
        (kind, mlx_compatibility(&root, &model_path).reason)
    {
        return Err(reason);
    }
    let suggested = load_local_suggested_config(&model_path);
    Ok(LocalServerSpec {
        kind,
        executable,
        context_tokens: effective_context_tokens(settings, &suggested),
        model_path,
        runtime_root: root,
    })
}

/// Resolves a model id from the settings to a path inside the models folder. Ids are relative
/// folder or file names, as `list_local_ai_models` reports them; anything that could leave the
/// folder is rejected.
fn local_model_path(models_dir: &Path, model_id: &str, kind: LocalRuntimeKind) -> Result<PathBuf, String> {
    let id = model_id.trim();
    if id.is_empty() {
        return Err("Selecciona un modelo local de IA en Configuración.".to_string());
    }
    let relative = Path::new(id);
    if !relative
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err("El modelo local debe estar dentro de la carpeta de modelos.".to_string());
    }
    let path = models_dir.join(relative);
    let present = match kind {
        LocalRuntimeKind::Mlx => path.is_dir(),
        LocalRuntimeKind::Gguf => path.is_file(),
    };
    if !present {
        return Err(format!("No se encontró el modelo local {id}."));
    }
    Ok(path)
}

fn write_report(path: &Path, markdown: &str) -> Result<(), String> {
    write_atomic_in_place(path, markdown)
        .map_err(|err| format!("No se pudo guardar el informe: {err}"))
}

// ---------------------------------------------------------------------------
// Runtime layout, status, and asset selection
// ---------------------------------------------------------------------------

/// Apple Silicon runs MLX; every other supported machine runs llama.cpp.
pub fn runtime_kind_for(platform: Platform) -> LocalRuntimeKind {
    if platform.is_apple_silicon() {
        LocalRuntimeKind::Mlx
    } else {
        LocalRuntimeKind::Gguf
    }
}

/// Release-asset name endings for llama.cpp, most preferred first. Empty when no official build
/// exists for the platform.
fn gguf_asset_suffixes(platform: Platform) -> &'static [&'static str] {
    match (platform.os, platform.arch) {
        ("windows", "x86_64") => &["bin-win-cpu-x64.zip"],
        ("windows", "aarch64") => &["bin-win-cpu-arm64.zip"],
        ("macos", "x86_64") => &["bin-macos-x64.tar.gz", "bin-macos-x64.zip"],
        ("macos", "aarch64") => &["bin-macos-arm64.tar.gz"],
        _ => &[],
    }
}

fn can_install(kind: LocalRuntimeKind, platform: Platform) -> bool {
    match kind {
        LocalRuntimeKind::Mlx => platform.is_apple_silicon(),
        LocalRuntimeKind::Gguf => !gguf_asset_suffixes(platform).is_empty(),
    }
}

/// First llama.cpp asset whose name matches a suffix, trying the suffixes in preference order.
fn select_asset<'a>(assets: &'a [ReleaseAsset], suffixes: &[&str]) -> Option<&'a ReleaseAsset> {
    suffixes.iter().find_map(|suffix| {
        assets.iter().find(|asset| {
            asset.name.starts_with(LLAMA_ASSET_PREFIX) && asset.name.ends_with(suffix)
        })
    })
}

/// The first release, newest first, that has a matching asset. Suffixes keep their preference order
/// inside each release.
fn pick_release<'a>(
    releases: &'a [Release],
    suffixes: &[&str],
) -> Option<(&'a Release, &'a ReleaseAsset)> {
    releases
        .iter()
        .find_map(|release| select_asset(&release.assets, suffixes).map(|asset| (release, asset)))
}

/// Tags of the releases that were inspected, in API order, each listed once.
fn inspected_tags(releases: &[Release]) -> Vec<&str> {
    let mut tags: Vec<&str> = Vec::new();
    for release in releases {
        if !tags.contains(&release.tag_name.as_str()) {
            tags.push(&release.tag_name);
        }
    }
    tags
}

fn missing_binary_message(platform: Platform, suffixes: &[&str], tags: &[&str]) -> String {
    let inspected = if tags.is_empty() {
        "ninguno".to_string()
    } else {
        tags.join(", ")
    };
    format!(
        "No hay un binario de llama.cpp para {}/{} (sufijos probados: {}; releases revisados: {inspected})",
        platform.os,
        platform.arch,
        suffixes.join(", "),
    )
}

/// Folder of the local runtime. The managed MLX layout belongs to `mlx_runtime`.
fn runtime_root(data_dir: &Path, kind: LocalRuntimeKind) -> PathBuf {
    match kind {
        LocalRuntimeKind::Mlx => data_dir.join(MLX_DIR),
        LocalRuntimeKind::Gguf => data_dir.join(LLAMA_DIR),
    }
}

fn mlx_runtime_root(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(runtime_root(&app_data_dir(app)?, LocalRuntimeKind::Mlx))
}

/// Installed state of the runtime this platform uses, read from disk and the environment only.
pub fn runtime_status(data_dir: &Path, platform: Platform) -> AiRuntimeStatus {
    let kind = runtime_kind_for(platform);
    match kind {
        LocalRuntimeKind::Mlx => {
            mlx_ai_status(mlx_runtime::status(&runtime_root(data_dir, kind)), platform)
        }
        LocalRuntimeKind::Gguf => {
            let (installed, detail) = gguf_status(&runtime_root(data_dir, kind));
            AiRuntimeStatus {
                kind,
                installed,
                detail,
                can_install: can_install(kind, platform),
                version: None,
                latest: None,
                update_available: false,
            }
        }
    }
}

/// Status line of the managed MLX runtime. `latest` and `update_available` are set only when the
/// status came from `mlx_runtime::check_update`.
fn mlx_ai_status(status: MlxStatus, platform: Platform) -> AiRuntimeStatus {
    let detail = match status.version.as_deref() {
        None => MLX_MISSING.to_string(),
        Some(version) => match (status.update_available, status.latest.as_deref()) {
            (true, Some(latest)) => {
                format!("MLX listo (mlx-lm {version}). Hay una versión nueva: {latest}.")
            }
            _ => format!("MLX listo (mlx-lm {version})."),
        },
    };
    AiRuntimeStatus {
        kind: LocalRuntimeKind::Mlx,
        installed: status.installed,
        detail,
        can_install: can_install(LocalRuntimeKind::Mlx, platform),
        version: status.version,
        latest: status.latest,
        update_available: status.update_available,
    }
}

/// Whether the installed `mlx-lm` can load the model folder, with the reason when it cannot. A folder
/// without a readable `model_type` is not refused; the server decides for it.
fn mlx_compatibility(root: &Path, model_dir: &Path) -> ModelCompatibility {
    let model_type = model_type_from_dir(model_dir);
    let reason = model_type
        .as_deref()
        .and_then(|model_type| mlx_refusal(root, model_type));
    ModelCompatibility {
        supported: reason.is_none(),
        model_type,
        reason,
    }
}

fn mlx_refusal(root: &Path, model_type: &str) -> Option<String> {
    let status = mlx_runtime::status(root);
    // The venv is two levels above its interpreter (`<venv>/bin/python3`).
    let supported = status
        .python
        .as_deref()
        .and_then(|python| python.parent().and_then(Path::parent))
        .map(supported_model_types)
        .unwrap_or_default();
    let version = status.version.unwrap_or_default();
    // An installed runtime that lists no modules is unreadable, not a runtime without this model.
    if !version.is_empty() && supported.is_empty() {
        return None;
    }
    unsupported_reason(model_type, &supported, &version)
}

fn gguf_status(dir: &Path) -> (bool, String) {
    if dir.join(LLAMA_SERVER_EXE).is_file() {
        (true, "llama-server listo en la carpeta de la app.".to_string())
    } else if find_on_path(LLAMA_SERVER_EXE).is_some() {
        (true, "llama-server encontrado en PATH.".to_string())
    } else {
        (false, "Falta llama-server; instálalo desde la app.".to_string())
    }
}

/// The app's own `llama-server` first, then one on `PATH`.
fn resolve_gguf_server(data_dir: &Path) -> Option<PathBuf> {
    let local = data_dir.join(LLAMA_DIR).join(LLAMA_SERVER_EXE);
    if local.is_file() {
        return Some(local);
    }
    find_on_path(LLAMA_SERVER_EXE)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map_err(|err| format!("No se pudo localizar la carpeta de datos: {err}"))
}

// ---------------------------------------------------------------------------
// Runtime install
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn ai_runtime_status(app: AppHandle) -> Result<AiRuntimeStatus, String> {
    let data_dir = app_data_dir(&app)?;
    let platform = Platform::current();
    blocking(move || Ok(runtime_status(&data_dir, platform))).await
}

/// Installs the local runtime for this platform. Refused unless `confirmed` is true; only one
/// install runs at a time.
#[tauri::command]
pub async fn install_ai_runtime(
    confirmed: bool,
    app: AppHandle,
) -> Result<AiRuntimeStatus, String> {
    if !confirmed {
        return Err("La instalación del runtime de IA requiere confirmación.".to_string());
    }
    let state = app.state::<AiRuntimeState>();
    if state.installing.swap(true, Ordering::SeqCst) {
        return Err("Ya hay una instalación de IA en curso.".to_string());
    }
    let outcome = install_runtime(&app, Platform::current()).await;
    state.installing.store(false, Ordering::SeqCst);
    outcome
}

async fn install_runtime(app: &AppHandle, platform: Platform) -> Result<AiRuntimeStatus, String> {
    let kind = runtime_kind_for(platform);
    if !can_install(kind, platform) {
        return Err("Este equipo no tiene un runtime de IA local compatible.".to_string());
    }
    let data_dir = app_data_dir(app)?;
    // Replacing a binary that the owned server is running from fails on Windows, so stop it first.
    app.state::<AiRuntimeState>().manager().stop();

    match kind {
        LocalRuntimeKind::Mlx => {
            let (app, root) = (app.clone(), runtime_root(&data_dir, kind));
            blocking(move || {
                mlx_runtime::install_latest(&root, &|message: &str| {
                    emit_progress(&app, MLX_PROGRESS_PHASE, message)
                })
            })
            .await?;
        }
        LocalRuntimeKind::Gguf => {
            install_gguf(app, &data_dir.join(LLAMA_DIR), platform).await?;
        }
    }

    blocking(move || Ok(runtime_status(&data_dir, platform))).await
}

/// Checks PyPI for a newer `mlx-lm`. Needs the network. Other runtimes report their installed state.
#[tauri::command]
pub async fn check_ai_runtime_update(app: AppHandle) -> Result<AiRuntimeStatus, String> {
    let platform = Platform::current();
    let data_dir = app_data_dir(&app)?;
    if runtime_kind_for(platform) != LocalRuntimeKind::Mlx {
        return blocking(move || Ok(runtime_status(&data_dir, platform))).await;
    }
    let root = runtime_root(&data_dir, LocalRuntimeKind::Mlx);
    blocking(move || Ok(mlx_ai_status(mlx_runtime::check_update(&root)?, platform))).await
}

/// Installs the latest `mlx-lm` side by side with the active one and activates it on success. Stops
/// the owned server so the next model start runs the new version. Shares the install flag with
/// `install_ai_runtime`.
#[tauri::command]
pub async fn update_ai_runtime(app: AppHandle) -> Result<AiRuntimeStatus, String> {
    let platform = Platform::current();
    if runtime_kind_for(platform) != LocalRuntimeKind::Mlx {
        return Err("Solo el runtime MLX se actualiza desde la app.".to_string());
    }
    let state = app.state::<AiRuntimeState>();
    if state.installing.swap(true, Ordering::SeqCst) {
        return Err("Ya hay una instalación de IA en curso.".to_string());
    }
    let outcome = update_mlx(&app, platform).await;
    state.installing.store(false, Ordering::SeqCst);
    outcome
}

async fn update_mlx(app: &AppHandle, platform: Platform) -> Result<AiRuntimeStatus, String> {
    app.state::<AiRuntimeState>().manager().stop();
    let root = mlx_runtime_root(app)?;
    let handle = app.clone();
    let status = blocking(move || {
        mlx_runtime::update(&root, &|message: &str| {
            emit_progress(&handle, MLX_PROGRESS_PHASE, message)
        })
    })
    .await?;
    Ok(mlx_ai_status(status, platform))
}

/// Starts the selected local model with the settings the report uses and reports whether it answers.
#[tauri::command]
pub async fn test_local_ai_model(app: AppHandle) -> Result<String, String> {
    let settings = app.state::<SettingsStore>().get();
    let models_dir = resolve_ai_models_dir(&app, &settings)?;
    let spec = local_server_spec(&app, &settings, &models_dir)?;
    let manager = app.state::<AiRuntimeState>().manager();
    match manager.ensure_ready(&spec).await {
        Ok(endpoint) => Ok(format!("Modelo listo: {}", endpoint.endpoint)),
        Err(err) => Err(with_server_log(err.to_string(), &spec.runtime_root)),
    }
}

/// Appends the last server-log lines that `message` does not already quote. A startup failure that
/// `mt_ai` explains already quotes its log, so it is not repeated.
fn with_server_log(message: String, runtime_root: &Path) -> String {
    let tail = mlx_runtime::log_tail(runtime_root, LOG_TAIL_LINES);
    let missing: Vec<&str> = tail
        .lines()
        .filter(|line| !message.contains(line))
        .collect();
    if missing.is_empty() {
        message
    } else {
        format!("{message}\nRegistro del servidor:\n{}", missing.join("\n"))
    }
}

/// Whether the local model `model_id` can run on the installed MLX runtime, and why not.
#[tauri::command]
pub fn local_model_compatibility(
    model_id: String,
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<ModelCompatibility, String> {
    if runtime_kind_for(Platform::current()) != LocalRuntimeKind::Mlx {
        return Ok(ModelCompatibility {
            supported: true,
            model_type: None,
            reason: None,
        });
    }
    let models_dir = resolve_ai_models_dir(&app, &store.get())?;
    let model_path = local_model_path(&models_dir, &model_id, LocalRuntimeKind::Mlx)?;
    Ok(mlx_compatibility(&mlx_runtime_root(&app)?, &model_path))
}

/// Downloads the latest official llama.cpp build for this platform and unpacks it.
async fn install_gguf(app: &AppHandle, root: &Path, platform: Platform) -> Result<(), String> {
    let suffixes = gguf_asset_suffixes(platform);
    emit_progress(app, "fetch", "Consultando la última versión de llama.cpp…");
    let client = reqwest::Client::builder()
        .user_agent(concat!("meeting-transcriber/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|err| format!("No se pudo preparar la descarga: {err}"))?;
    let latest = fetch_latest_release(&client).await?;
    let asset = match select_asset(&latest.assets, suffixes) {
        Some(asset) => asset.clone(),
        None => {
            let releases = fetch_releases(&client).await?;
            pick_release(&releases, suffixes)
                .map(|(_, asset)| asset.clone())
                .ok_or_else(|| {
                    missing_binary_message(platform, suffixes, &inspected_tags(&releases))
                })?
        }
    };

    fs::create_dir_all(root).map_err(io_error)?;
    let archive = root.join(format!(".download-{}-{}", unique_suffix(), asset.name));
    let outcome = match download_asset(app, &client, &asset, &archive).await {
        Ok(()) => {
            let (app, root, archive) = (app.clone(), root.to_path_buf(), archive.clone());
            blocking(move || extract_llama(&app, &root, &archive)).await
        }
        Err(err) => Err(err),
    };
    let _ = fs::remove_file(&archive);
    outcome
}

async fn fetch_latest_release(client: &reqwest::Client) -> Result<Release, String> {
    fetch_github_json(client, LLAMA_RELEASE_URL).await
}

/// The build-tagged llama.cpp releases, newest first.
async fn fetch_releases(client: &reqwest::Client) -> Result<Vec<Release>, String> {
    fetch_github_json(client, LLAMA_RELEASES_URL).await
}

async fn fetch_github_json<T: serde::de::DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
) -> Result<T, String> {
    client
        .get(url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|err| format!("No se pudo consultar llama.cpp en GitHub: {err}"))?
        .json::<T>()
        .await
        .map_err(|err| format!("Respuesta inesperada de GitHub: {err}"))
}

/// Streams the asset to `dest`, reporting whole-percent progress, and checks the published SHA-256
/// when GitHub provides one.
async fn download_asset(
    app: &AppHandle,
    client: &reqwest::Client,
    asset: &ReleaseAsset,
    dest: &Path,
) -> Result<(), String> {
    emit_progress(app, "download", &format!("Descargando {}…", asset.name));
    let mut response = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|err| format!("No se pudo descargar llama.cpp: {err}"))?;
    let total = asset.size.filter(|total| *total > 0);
    let mut file = File::create(dest).map_err(io_error)?;
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut last_reported: Option<u64> = None;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| format!("Descarga de llama.cpp interrumpida: {err}"))?
    {
        hasher.update(&chunk);
        file.write_all(&chunk).map_err(io_error)?;
        received += chunk.len() as u64;
        if let Some(total) = total {
            let whole = (received.saturating_mul(100) / total).min(100);
            if last_reported != Some(whole) {
                last_reported = Some(whole);
                emit_progress(
                    app,
                    "download",
                    &format!("Descargando llama.cpp… {whole}%"),
                );
            }
        }
    }
    file.flush().map_err(io_error)?;
    drop(file);

    if let Some(expected) = asset.digest.as_deref().and_then(|d| d.strip_prefix("sha256:")) {
        let actual = format!("{:x}", hasher.finalize());
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "La verificación SHA-256 de {} falló (obtenido {actual}).",
                asset.name
            ));
        }
    }
    Ok(())
}

/// Unpacks the archive with the system `tar` (bsdtar reads `.tar.gz` and `.zip`), then copies
/// `llama-server` and its shared libraries into the runtime folder and runs `--version`.
fn extract_llama(app: &AppHandle, root: &Path, archive: &Path) -> Result<(), String> {
    emit_progress(app, "extract", "Extrayendo llama-server…");
    let staging = root.join(format!(".extract-{}", unique_suffix()));
    fs::create_dir_all(&staging).map_err(io_error)?;
    let outcome = install_llama_files(app, root, archive, &staging);
    let _ = fs::remove_dir_all(&staging);
    outcome
}

fn install_llama_files(
    app: &AppHandle,
    root: &Path,
    archive: &Path,
    staging: &Path,
) -> Result<(), String> {
    run_checked(
        command_with("tar", |c| {
            c.arg("-xf").arg(archive).arg("-C").arg(staging);
        }),
        "La extracción de llama.cpp",
    )?;
    let server = find_file(staging, LLAMA_SERVER_EXE, MAX_ARCHIVE_DEPTH).ok_or_else(|| {
        format!("El paquete de llama.cpp no contiene {LLAMA_SERVER_EXE}.")
    })?;
    let bin_dir = server
        .parent()
        .ok_or_else(|| "Ruta inválida dentro del paquete de llama.cpp.".to_string())?;
    copy_llama_runtime(bin_dir, root)?;

    emit_progress(app, "verify", "Verificando llama-server…");
    run_checked(
        command_with(root.join(LLAMA_SERVER_EXE), |c| {
            c.arg("--version");
        }),
        "La verificación de llama-server",
    )
}

/// A hidden child process built in place, so it can be passed to `run_checked` by value.
fn command_with(program: impl AsRef<OsStr>, build: impl FnOnce(&mut Command)) -> Command {
    let mut command = hidden_command(program);
    build(&mut command);
    command
}

/// Depth-limited search for a file by name.
fn find_file(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && entry.file_name().to_str() == Some(name) {
            return Some(path);
        }
        if depth > 0 && path.is_dir() && let Some(found) = find_file(&path, name, depth - 1) {
            return Some(found);
        }
    }
    None
}

/// Copies `llama-server` and the shared libraries next to each other. Other binaries in the
/// archive (`llama-cli`, benchmarks, …) are left behind.
fn copy_llama_runtime(from: &Path, to: &Path) -> Result<(), String> {
    for entry in fs::read_dir(from).map_err(io_error)?.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_file() && (name == LLAMA_SERVER_EXE || is_shared_library(&name)) {
            fs::copy(&path, to.join(&name)).map_err(io_error)?;
        }
    }
    Ok(())
}

fn is_shared_library(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".dll") || lower.ends_with(".dylib") || lower.contains(".so")
}

/// Progress line for the report run, shown live in the session and in the "analyze a file" card.
const REPORT_PROGRESS_EVENT: &str = "report-progress";

fn ai_progress_payload(
    phase: &'static str,
    message: &str,
    percent: Option<u8>,
) -> serde_json::Value {
    match percent {
        Some(pct) => serde_json::json!({
            "phase": phase,
            "message": message,
            "percent": pct.min(100),
        }),
        None => serde_json::json!({
            "phase": phase,
            "message": message,
        }),
    }
}

fn emit_ai_progress(app: &AppHandle, phase: &'static str, message: &str, percent: Option<u8>) {
    let _ = app.emit(
        REPORT_PROGRESS_EVENT,
        ai_progress_payload(phase, message, percent),
    );
}

fn emit_progress(app: &AppHandle, phase: &'static str, message: &str) {
    let _ = app.emit(
        PROGRESS_EVENT,
        RuntimeProgress {
            phase,
            message: message.to_string(),
        },
    );
}

// ---------------------------------------------------------------------------
// Window and note commands
// ---------------------------------------------------------------------------

/// Runs the post-meeting report for one meeting file on demand.
#[tauri::command]
pub async fn run_report_on_file(path: String, app: AppHandle) -> Result<(), String> {
    run_post_meeting_report(&app, &path).await
}

/// Opens a note with the default application.
#[tauri::command]
pub fn open_note(path: String, app: AppHandle) -> Result<(), String> {
    if !Path::new(&path).is_file() {
        return Err(format!("No se encontró la nota {path}"));
    }
    app.opener()
        .open_path(path, None::<String>)
        .map_err(|err| err.to_string())
}

#[tauri::command]
pub fn open_settings_window(app: AppHandle) -> Result<(), String> {
    show_main_tab(&app, "general")
}

#[tauri::command]
pub fn open_session_window(app: AppHandle) -> Result<(), String> {
    show_main_tab(&app, "session")
}

/// Shows the main window and asks the UI to switch to `tab`. Shared by the commands and the tray.
pub(crate) fn show_main_tab(app: &AppHandle, tab: &str) -> Result<(), String> {
    show_window(app, MAIN_WINDOW)?;
    app.emit(NAVIGATE_TAB_EVENT, tab)
        .map_err(|err| err.to_string())
}

/// Shows, un-minimizes and focuses a window by label.
pub(crate) fn show_window(app: &AppHandle, label: &str) -> Result<(), String> {
    let window = app
        .get_webview_window(label)
        .ok_or_else(|| format!("No existe la ventana {label}"))?;
    window.show().map_err(|err| err.to_string())?;
    let _ = window.unminimize();
    window.set_focus().map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_core::CorrectionRule;
    use mt_core::report::short_meeting_markdown as expected_short;

    const MAC_ARM: Platform = Platform {
        os: "macos",
        arch: "aarch64",
    };
    const MAC_INTEL: Platform = Platform {
        os: "macos",
        arch: "x86_64",
    };
    const WINDOWS_X64: Platform = Platform {
        os: "windows",
        arch: "x86_64",
    };
    const WINDOWS_ARM: Platform = Platform {
        os: "windows",
        arch: "aarch64",
    };

    fn rule(find: &str, replace: &str) -> CorrectionRule {
        CorrectionRule {
            find: find.to_string(),
            replace: replace.to_string(),
            case_sensitive: false,
            whole_word: false,
        }
    }

    fn asset(name: &str) -> ReleaseAsset {
        ReleaseAsset {
            name: name.to_string(),
            browser_download_url: format!("https://example.invalid/{name}"),
            size: None,
            digest: None,
        }
    }

    const PREVIOUS_REPORT_MEETING: &str = "# Reporte de la reunión\n\nresumen anterior\n\n---\n\n# Transcripción completa\n\nAna dijo hola\n";

    #[test]
    fn short_meeting_is_rendered_from_the_original_transcript_without_corrections() {
        let settings = AppSettings {
            min_report_words: 50,
            corrections: vec![rule("Ana", "Anna")],
            ..AppSettings::default()
        };
        let prepared = prepare_transcript(PREVIOUS_REPORT_MEETING, &settings);

        assert_eq!(prepared.original, "Ana dijo hola\n", "old report is stripped");
        assert_eq!(prepared.cleaned, "Anna dijo hola\n", "corrections apply to the copy");
        assert!(is_short_meeting(&prepared, settings.min_report_words));

        let markdown = short_meeting_report(&settings, &prepared.original);
        assert_eq!(markdown, expected_short("Spanish", "Ana dijo hola\n"));
        assert!(markdown.contains("Ana dijo hola"));
        assert!(!markdown.contains("Anna"));
    }

    #[test]
    fn meeting_at_the_word_threshold_is_not_short() {
        let settings = AppSettings {
            min_report_words: 3,
            ..AppSettings::default()
        };
        let enough = prepare_transcript("uno dos tres\n", &settings);
        assert!(!is_short_meeting(&enough, settings.min_report_words));

        let short = prepare_transcript("uno dos\n", &settings);
        assert!(is_short_meeting(&short, settings.min_report_words));
    }

    #[test]
    fn keeping_the_existing_report_leaves_the_transcript_whole() {
        let settings = AppSettings {
            replace_existing_report: false,
            ..AppSettings::default()
        };
        let prepared = prepare_transcript(PREVIOUS_REPORT_MEETING, &settings);
        assert_eq!(prepared.original, PREVIOUS_REPORT_MEETING);
    }

    #[test]
    fn write_report_replaces_the_file_content_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nota.md");
        fs::write(&path, "contenido anterior").unwrap();

        write_report(&path, "# Reporte\n").unwrap();

        assert_eq!(fs::read_to_string(&path).unwrap(), "# Reporte\n");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1, "no temp file left behind");
    }

    #[test]
    fn api_config_needs_an_endpoint_and_a_model_name_and_keeps_the_key() {
        let mut settings = AppSettings {
            ai_source: AiSource::Api,
            ai_endpoint: " https://api.example.test/v1 ".to_string(),
            ai_model_name: "model-x".to_string(),
            ai_api_key: "secret".to_string(),
            ..AppSettings::default()
        };
        let config = api_model_config(&settings).unwrap();
        assert_eq!(config.endpoint, "https://api.example.test/v1");
        assert_eq!(config.model, "model-x");
        assert_eq!(config.api_key, "secret");
        assert_eq!(config.context_limit_tokens, settings.context_limit_tokens as usize);
        assert_eq!(config.max_output_tokens, settings.max_output_tokens as usize);

        settings.ai_endpoint = "  ".to_string();
        assert!(api_model_config(&settings).unwrap_err().contains("endpoint"));
        settings.ai_endpoint = "https://api.example.test/v1".to_string();
        settings.ai_model_name.clear();
        assert!(api_model_config(&settings).unwrap_err().contains("modelo"));
    }

    #[test]
    fn local_config_uses_the_managed_endpoint_and_the_model_folder() {
        let settings = AppSettings::default();
        let endpoint = LocalServerEndpoint {
            endpoint: "http://127.0.0.1:9000/v1".to_string(),
            port: 9000,
            kind: LocalRuntimeKind::Mlx,
            model_path: PathBuf::from("/models/whisper-mlx"),
        };
        let config = local_model_config(&settings, &endpoint, &SuggestedModelConfig::default());
        assert_eq!(config.endpoint, "http://127.0.0.1:9000/v1");
        assert_eq!(config.model, "/models/whisper-mlx");
        assert!(config.api_key.is_empty());
        assert_eq!(config.timeout_seconds, 600);

        let api_settings = AppSettings {
            ai_endpoint: "https://api.example.com/v1".to_string(),
            ai_model_name: "gpt-4o-mini".to_string(),
            ..AppSettings::default()
        };
        let api_config = api_model_config(&api_settings).unwrap();
        assert_eq!(api_config.timeout_seconds, 120);
    }

    fn hf_settings() -> AppSettings {
        AppSettings {
            use_hf_recommended_config: true,
            ai_disable_thinking: true,
            ..AppSettings::default()
        }
    }

    fn hf_suggestion() -> SuggestedModelConfig {
        SuggestedModelConfig {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(20),
            repetition_penalty: Some(1.05),
            presence_penalty: Some(1.5),
            max_context_tokens: Some(8192),
            supports_thinking_toggle: true,
            recommended_thinking: Some(false),
            source_summary: "generation_config.json + config.json".into(),
        }
    }

    fn loopback_endpoint() -> LocalServerEndpoint {
        LocalServerEndpoint {
            endpoint: "http://127.0.0.1:9000/v1".to_string(),
            port: 9000,
            kind: LocalRuntimeKind::Mlx,
            model_path: PathBuf::from("/models/whisper-mlx"),
        }
    }

    #[test]
    fn hf_profile_context_clamps_to_the_model_maximum() {
        let suggested = hf_suggestion();
        assert_eq!(effective_context_tokens(&hf_settings(), &suggested), 8192);

        let small = AppSettings {
            context_limit_tokens: 4096,
            ..hf_settings()
        };
        assert_eq!(effective_context_tokens(&small, &suggested), 4096, "a smaller user limit wins");

        let manual = AppSettings {
            use_hf_recommended_config: false,
            ..hf_settings()
        };
        assert_eq!(effective_context_tokens(&manual, &suggested), 16384, "manual mode ignores the profile");

        let unknown = SuggestedModelConfig {
            max_context_tokens: None,
            ..hf_suggestion()
        };
        assert_eq!(effective_context_tokens(&hf_settings(), &unknown), 16384);
    }

    #[test]
    fn hf_profile_drives_sampling_context_and_thinking() {
        let settings = AppSettings {
            ai_temperature: Some(0.9),
            ai_top_k: Some(99),
            ..hf_settings()
        };
        let config = local_model_config(&settings, &loopback_endpoint(), &hf_suggestion());

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, Some(0.8));
        assert_eq!(config.top_k, Some(20));
        assert_eq!(config.repetition_penalty, Some(1.05));
        assert_eq!(config.presence_penalty, Some(1.5));
        assert_eq!(config.context_limit_tokens, 8192);
        assert_eq!(config.max_output_tokens, 2560, "a reply reservation that fits is kept");
        assert!(!config.thinking, "ai_disable_thinking turns thinking off");
    }

    #[test]
    fn thinking_is_requested_only_when_the_model_supports_it_and_it_is_not_disabled() {
        let enabled = AppSettings {
            ai_disable_thinking: false,
            ..hf_settings()
        };
        assert!(local_model_config(&enabled, &loopback_endpoint(), &hf_suggestion()).thinking);

        let unsupported = SuggestedModelConfig {
            supports_thinking_toggle: false,
            ..hf_suggestion()
        };
        assert!(!local_model_config(&enabled, &loopback_endpoint(), &unsupported).thinking);
    }

    #[test]
    fn manual_overrides_apply_when_the_hf_profile_is_off() {
        let settings = AppSettings {
            use_hf_recommended_config: false,
            ai_temperature: Some(0.4),
            ai_top_p: Some(0.9),
            ai_top_k: Some(40),
            ai_repetition_penalty: Some(1.1),
            ai_disable_thinking: true,
            ..hf_settings()
        };
        let config = local_model_config(&settings, &loopback_endpoint(), &hf_suggestion());
        assert_eq!(config.temperature, 0.4);
        assert_eq!(config.top_p, Some(0.9));
        assert_eq!(config.top_k, Some(40));
        assert_eq!(config.repetition_penalty, Some(1.1));
        assert_eq!(config.presence_penalty, None, "there is no manual presence penalty");
        assert_eq!(config.context_limit_tokens, 16384, "the profile does not clamp the context");
        assert!(!config.thinking);

        let unset = AppSettings {
            use_hf_recommended_config: false,
            ..hf_settings()
        };
        let config = local_model_config(&unset, &loopback_endpoint(), &hf_suggestion());
        assert_eq!(config.temperature, 0.2);
        assert_eq!(config.top_p, None);
        assert_eq!(config.top_k, None);
        assert_eq!(config.repetition_penalty, None);
    }

    #[test]
    fn reply_reservation_never_fills_the_clamped_context() {
        let suggested = SuggestedModelConfig {
            max_context_tokens: Some(512),
            ..hf_suggestion()
        };
        let config = local_model_config(&hf_settings(), &loopback_endpoint(), &suggested);
        assert_eq!(config.context_limit_tokens, 512);
        assert!(config.max_output_tokens < config.context_limit_tokens);
        assert!(config.input_budget_tokens() > 0);
    }

    #[test]
    fn local_model_id_must_name_an_existing_model_inside_the_folder() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("mlx-model")).unwrap();
        fs::create_dir_all(root.join("family")).unwrap();
        fs::write(root.join("family").join("q4.gguf"), b"x").unwrap();

        assert_eq!(
            local_model_path(root, "family/q4.gguf", LocalRuntimeKind::Gguf).unwrap(),
            root.join("family").join("q4.gguf")
        );
        assert_eq!(
            local_model_path(root, " mlx-model ", LocalRuntimeKind::Mlx).unwrap(),
            root.join("mlx-model")
        );
        assert!(local_model_path(root, "", LocalRuntimeKind::Gguf).is_err());
        assert!(local_model_path(root, "../outside.gguf", LocalRuntimeKind::Gguf).is_err());
        assert!(local_model_path(root, "/etc/passwd", LocalRuntimeKind::Gguf).is_err());
        assert!(local_model_path(root, "missing.gguf", LocalRuntimeKind::Gguf).is_err());
    }

    #[test]
    fn runtime_kind_and_install_support_follow_the_platform() {
        assert_eq!(runtime_kind_for(MAC_ARM), LocalRuntimeKind::Mlx);
        assert_eq!(runtime_kind_for(MAC_INTEL), LocalRuntimeKind::Gguf);
        assert_eq!(runtime_kind_for(WINDOWS_X64), LocalRuntimeKind::Gguf);

        assert!(can_install(LocalRuntimeKind::Mlx, MAC_ARM));
        assert!(!can_install(LocalRuntimeKind::Mlx, WINDOWS_X64));
        assert!(can_install(LocalRuntimeKind::Gguf, WINDOWS_X64));
        assert!(can_install(LocalRuntimeKind::Gguf, MAC_INTEL));
        assert!(can_install(LocalRuntimeKind::Gguf, WINDOWS_ARM));
    }

    fn release(tag: &str, names: &[&str]) -> Release {
        Release {
            tag_name: tag.to_string(),
            assets: names.iter().map(|name| asset(name)).collect(),
        }
    }

    #[test]
    fn the_windows_binary_is_found_in_the_build_tagged_release_when_latest_is_nightly_only() {
        let releases = vec![
            release("v1.0.0", &["nightly-tag.txt"]),
            release(
                "b12569",
                &[
                    "llama-b12569-bin-win-cpu-x64.zip",
                    "llama-b12569-bin-win-cuda-x64.zip",
                ],
            ),
            release("b12500", &["llama-b12500-bin-win-cpu-x64.zip"]),
        ];

        let (found, picked) = pick_release(&releases, gguf_asset_suffixes(WINDOWS_X64))
            .expect("the b12569 release has a Windows x64 CPU build");
        assert_eq!(found.tag_name, "b12569");
        assert_eq!(picked.name, "llama-b12569-bin-win-cpu-x64.zip");
    }

    #[test]
    fn pick_release_returns_none_when_no_release_has_a_matching_asset() {
        let releases = vec![
            release("v1.0.0", &["nightly-tag.txt"]),
            release("b12569", &["llama-b12569-bin-win-cuda-x64.zip"]),
        ];

        assert!(pick_release(&releases, gguf_asset_suffixes(WINDOWS_X64)).is_none());
    }

    #[test]
    fn pick_release_prefers_the_suffix_order_inside_one_release() {
        // The zip is listed first, but the tarball is the more preferred suffix for macOS Intel.
        let releases = vec![release(
            "b12569",
            &[
                "llama-b12569-bin-macos-x64.zip",
                "llama-b12569-bin-macos-x64.tar.gz",
            ],
        )];

        let (_, picked) = pick_release(&releases, gguf_asset_suffixes(MAC_INTEL)).unwrap();
        assert_eq!(picked.name, "llama-b12569-bin-macos-x64.tar.gz");
    }

    #[test]
    fn gguf_asset_suffixes_cover_windows_and_macos_and_nothing_else() {
        const LINUX_X64: Platform = Platform {
            os: "linux",
            arch: "x86_64",
        };

        assert_eq!(gguf_asset_suffixes(WINDOWS_X64), &["bin-win-cpu-x64.zip"]);
        assert_eq!(gguf_asset_suffixes(WINDOWS_ARM), &["bin-win-cpu-arm64.zip"]);
        assert_eq!(
            gguf_asset_suffixes(MAC_INTEL),
            &["bin-macos-x64.tar.gz", "bin-macos-x64.zip"]
        );
        assert_eq!(gguf_asset_suffixes(MAC_ARM), &["bin-macos-arm64.tar.gz"]);
        assert!(gguf_asset_suffixes(LINUX_X64).is_empty());
    }

    #[test]
    fn the_missing_binary_error_names_the_platform_suffixes_and_inspected_tags() {
        let releases = vec![
            release("v1.0.0", &["nightly-tag.txt"]),
            release("b12569", &["llama-b12569-bin-win-cuda-x64.zip"]),
        ];
        let tags = inspected_tags(&releases);
        assert_eq!(tags, vec!["v1.0.0", "b12569"]);

        assert_eq!(
            missing_binary_message(WINDOWS_X64, gguf_asset_suffixes(WINDOWS_X64), &tags),
            "No hay un binario de llama.cpp para windows/x86_64 (sufijos probados: bin-win-cpu-x64.zip; releases revisados: v1.0.0, b12569)"
        );
    }

    #[test]
    fn inspected_tags_lists_each_release_once_in_api_order() {
        let releases = vec![
            release("b12569", &[]),
            release("b12568", &[]),
            release("b12569", &[]),
        ];

        assert_eq!(inspected_tags(&releases), vec!["b12569", "b12568"]);
    }

    #[test]
    fn release_asset_selection_matches_the_platform_archive() {
        let assets = vec![
            asset("cudart-llama-bin-win-cuda-x64.zip"),
            asset("llama-b9000-bin-win-cpu-x64.zip"),
            asset("llama-b9000-bin-macos-x64.zip"),
            asset("llama-b9000-bin-macos-x64.tar.gz"),
            asset("llama-b9000-bin-macos-arm64.tar.gz"),
        ];

        let windows = select_asset(&assets, gguf_asset_suffixes(WINDOWS_X64)).unwrap();
        assert_eq!(windows.name, "llama-b9000-bin-win-cpu-x64.zip");

        // The tarball is preferred over the zip of the same build.
        let intel = select_asset(&assets, gguf_asset_suffixes(MAC_INTEL)).unwrap();
        assert_eq!(intel.name, "llama-b9000-bin-macos-x64.tar.gz");

        let arm = select_asset(&assets, gguf_asset_suffixes(MAC_ARM)).unwrap();
        assert_eq!(arm.name, "llama-b9000-bin-macos-arm64.tar.gz");

        assert!(select_asset(&assets, gguf_asset_suffixes(WINDOWS_ARM)).is_none());
    }

    /// A complete MLX install under `<data>/mlx`, laid out as `mlx_runtime` writes it. The venv lists
    /// `supported` as its `mlx_lm/models` modules.
    fn fake_mlx_install(data_dir: &Path, version: &str, supported: &[&str]) -> PathBuf {
        let root = runtime_root(data_dir, LocalRuntimeKind::Mlx);
        let venv = root.join("versions").join(version);
        fs::create_dir_all(venv.join("bin")).unwrap();
        fs::write(venv.join("bin").join("python3"), b"#!/bin/sh\n").unwrap();
        fs::write(venv.join("version.txt"), format!("{version}\n")).unwrap();
        fs::write(root.join("active"), format!("{version}\n")).unwrap();
        let models = venv.join("lib/python3.12/site-packages/mlx_lm/models");
        fs::create_dir_all(&models).unwrap();
        for name in supported {
            fs::write(models.join(format!("{name}.py")), "").unwrap();
        }
        venv
    }

    #[test]
    fn runtime_status_reports_the_installed_mlx_version() {
        let dir = tempfile::tempdir().unwrap();
        fake_mlx_install(dir.path(), "0.21.0", &["llama"]);

        let status = runtime_status(dir.path(), MAC_ARM);

        assert_eq!(status.kind, LocalRuntimeKind::Mlx);
        assert!(status.installed);
        assert!(status.can_install);
        assert_eq!(status.version.as_deref(), Some("0.21.0"));
        assert_eq!(status.detail, "MLX listo (mlx-lm 0.21.0).");
        assert!(!status.update_available);
    }

    #[test]
    fn runtime_status_asks_to_install_mlx_when_it_is_missing() {
        let dir = tempfile::tempdir().unwrap();

        let status = runtime_status(dir.path(), MAC_ARM);

        assert!(!status.installed);
        assert_eq!(status.version, None);
        assert_eq!(
            status.detail,
            "Falta el runtime MLX; instálalo desde Configuración."
        );
    }

    #[test]
    fn an_available_mlx_update_is_named_in_the_status_detail() {
        let status = mlx_ai_status(
            MlxStatus {
                installed: true,
                version: Some("0.21.0".to_string()),
                python: None,
                latest: Some("0.22.0".to_string()),
                update_available: true,
            },
            MAC_ARM,
        );

        assert!(status.update_available);
        assert_eq!(status.latest.as_deref(), Some("0.22.0"));
        assert_eq!(
            status.detail,
            "MLX listo (mlx-lm 0.21.0). Hay una versión nueva: 0.22.0."
        );
    }

    #[test]
    fn the_llama_runtime_status_keeps_its_shape_and_has_no_version() {
        let dir = tempfile::tempdir().unwrap();

        let status = runtime_status(dir.path(), WINDOWS_X64);

        assert_eq!(status.kind, LocalRuntimeKind::Gguf);
        assert!(status.can_install);
        assert_eq!(status.version, None);
        assert!(!status.update_available);
        assert_eq!(status.installed, find_on_path(LLAMA_SERVER_EXE).is_some());
    }

    #[test]
    fn a_model_type_the_installed_runtime_lacks_is_refused_with_the_real_cause() {
        let dir = tempfile::tempdir().unwrap();
        fake_mlx_install(dir.path(), "0.21.0", &["llama"]);
        let root = runtime_root(dir.path(), LocalRuntimeKind::Mlx);
        let model = dir.path().join("brandnew-mlx");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), r#"{"model_type": "brandnew"}"#).unwrap();

        let compatibility = mlx_compatibility(&root, &model);

        assert!(!compatibility.supported);
        assert_eq!(compatibility.model_type.as_deref(), Some("brandnew"));
        assert_eq!(
            compatibility.reason.as_deref(),
            Some("mlx-lm 0.21.0 no soporta el modelo «brandnew». Actualiza el runtime MLX o elige otro modelo.")
        );
    }

    #[test]
    fn supported_models_and_models_without_a_model_type_are_not_refused() {
        let dir = tempfile::tempdir().unwrap();
        fake_mlx_install(dir.path(), "0.21.0", &["llama"]);
        let root = runtime_root(dir.path(), LocalRuntimeKind::Mlx);
        let model = dir.path().join("llama-mlx");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), r#"{"model_type": "llama"}"#).unwrap();

        let supported = mlx_compatibility(&root, &model);
        assert!(supported.supported);
        assert_eq!(supported.reason, None);

        let undeclared = dir.path().join("no-config");
        fs::create_dir_all(&undeclared).unwrap();
        let unknown = mlx_compatibility(&root, &undeclared);
        assert!(unknown.supported);
        assert_eq!(unknown.model_type, None);
    }

    #[test]
    fn an_unreadable_runtime_listing_does_not_refuse_a_model() {
        let dir = tempfile::tempdir().unwrap();
        fake_mlx_install(dir.path(), "0.21.0", &[]);
        let root = runtime_root(dir.path(), LocalRuntimeKind::Mlx);
        let model = dir.path().join("brandnew-mlx");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), r#"{"model_type": "brandnew"}"#).unwrap();

        assert!(mlx_compatibility(&root, &model).supported);
    }

    #[test]
    fn a_model_is_refused_with_the_missing_runtime_cause_when_mlx_is_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        let root = runtime_root(dir.path(), LocalRuntimeKind::Mlx);
        let model = dir.path().join("brandnew-mlx");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), r#"{"model_type": "brandnew"}"#).unwrap();

        let compatibility = mlx_compatibility(&root, &model);

        assert!(!compatibility.supported);
        assert_eq!(
            compatibility.reason.as_deref(),
            Some("No se puede usar el modelo «brandnew»: el runtime MLX no está instalado.")
        );
    }

    #[test]
    fn a_failed_ai_step_keeps_the_transcript_and_appends_the_warning() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nota.md");
        let transcript = "Ana dijo hola\n";
        fs::write(&path, transcript).unwrap();
        let reason = "El servidor local se cerró durante el arranque (código 1)";

        let returned = record_ai_failure(&path, transcript, reason);

        assert_eq!(returned, reason);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "Ana dijo hola\n\n---\n\n⚠️ Reporte IA no generado: El servidor local se cerró durante el arranque (código 1)\n"
        );
    }

    #[test]
    fn a_retry_reads_the_transcript_back_without_the_warning_block() {
        let settings = AppSettings::default();
        let note = "Ana dijo hola\n\n---\n\n⚠️ Reporte IA no generado: motivo\n";

        let prepared = prepare_transcript(note, &settings);

        assert_eq!(prepared.original, "Ana dijo hola\n");
        assert_eq!(prepared.cleaned, "Ana dijo hola\n");
    }

    #[test]
    fn the_server_log_is_quoted_only_when_the_error_does_not_already_quote_it() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("server.log"),
            "Traceback (most recent call last):\n  ~^~~~\nValueError: Model type gemma4 not supported.\n",
        )
        .unwrap();
        let quoted = "El servidor local se cerró durante el arranque (código 1): Traceback (most recent call last): | ValueError: Model type gemma4 not supported.";
        assert_eq!(with_server_log(quoted.to_string(), dir.path()), quoted);

        let bare = with_server_log("Tiempo agotado".to_string(), dir.path());
        assert_eq!(
            bare,
            "Tiempo agotado\nRegistro del servidor:\nTraceback (most recent call last):\nValueError: Model type gemma4 not supported."
        );

        let empty = tempfile::tempdir().unwrap();
        assert_eq!(with_server_log("Tiempo agotado".to_string(), empty.path()), "Tiempo agotado");
    }

    #[test]
    fn a_transcript_without_a_warning_block_is_left_unchanged() {
        let settings = AppSettings {
            replace_existing_report: false,
            ..AppSettings::default()
        };
        let transcript = "Ana dijo hola\n\n---\n\nnota\n";
        assert_eq!(prepare_transcript(transcript, &settings).original, transcript);
    }

    #[test]
    fn copies_only_llama_server_and_its_shared_libraries() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("bin");
        let to = dir.path().join("llama");
        fs::create_dir_all(&from).unwrap();
        fs::create_dir_all(&to).unwrap();
        for name in [
            LLAMA_SERVER_EXE,
            "libllama.dylib",
            "ggml-cpu.dll",
            "libggml.so.0",
            "llama-cli",
            "README.md",
        ] {
            fs::write(from.join(name), b"x").unwrap();
        }

        copy_llama_runtime(&from, &to).unwrap();

        let mut copied: Vec<String> = fs::read_dir(&to)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        copied.sort();
        let mut expected = vec![
            LLAMA_SERVER_EXE.to_string(),
            "libllama.dylib".to_string(),
            "ggml-cpu.dll".to_string(),
            "libggml.so.0".to_string(),
        ];
        expected.sort();
        assert_eq!(copied, expected);
    }

    #[test]
    fn single_main_window_and_tab_event_names_are_fixed() {
        assert_eq!(MAIN_WINDOW, "main", "the only window is the one declared in tauri.conf.json");
        assert_eq!(NAVIGATE_TAB_EVENT, "navigate-tab");
    }

    #[test]
    fn find_file_respects_the_depth_limit() {
        let dir = tempfile::tempdir().unwrap();
        let shallow = dir.path().join("build").join("bin");
        fs::create_dir_all(&shallow).unwrap();
        fs::write(shallow.join(LLAMA_SERVER_EXE), b"x").unwrap();
        let deep = dir.path().join("a").join("b").join("c");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("only-deep"), b"x").unwrap();

        assert!(find_file(dir.path(), LLAMA_SERVER_EXE, 2).is_some());
        assert!(find_file(dir.path(), "only-deep", 1).is_none());
        assert!(find_file(dir.path(), "only-deep", 3).is_some());
    }

    #[test]
    fn ai_progress_payload_serializes_percent_when_present_and_omits_when_absent() {
        let with_pct = ai_progress_payload("analyze", "Analizando…", Some(42));
        assert_eq!(with_pct["phase"], "analyze");
        assert_eq!(with_pct["message"], "Analizando…");
        assert_eq!(with_pct["percent"], 42);

        let clamped = ai_progress_payload("done", "Listo", Some(150));
        assert_eq!(clamped["percent"], 100);

        let without_pct = ai_progress_payload("prepare", "Iniciando…", None);
        assert_eq!(without_pct["phase"], "prepare");
        assert_eq!(without_pct["message"], "Iniciando…");
        assert!(without_pct.get("percent").is_none(), "{without_pct}");
    }
}
