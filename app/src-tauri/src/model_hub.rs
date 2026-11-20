//! Local AI model hub: inspection of a Hugging Face repository against this machine, installation
//! with progress and cancellation reported to the UI, and the scan of models already on disk.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use mt_ai::runtime::LocalServerState;
use mt_core::AppSettings;
use mt_models::{
    AssessRequest, Assessment, CancelToken, Downloader, Fit, GgufOption, HubClient, MachineInfo,
    ModelFormat, ModelPlan, PlannedFile, Progress, RepoInfo, RepoRef, SuggestedModelConfig, assess,
    detect_machine, discard_partial, list_gguf_options, load_local_suggested_config,
    parse_reference, save_local_suggested_config,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::report_pipeline::AiRuntimeState;
use crate::settings_store::{Platform, SettingsStore, resolve_ai_models_dir};

/// Event carrying a full [`HubSnapshot`] after every state change.
const STATE_EVENT: &str = "model-hub-state";
/// Folder profile written by `mt-models` inside an installed model folder.
const FOLDER_PROFILE_FILE: &str = "mt-profile.json";
/// Suffix of a GGUF file's profile sidecar (`<file>.mt-profile.json`), as written by `mt-models`.
const GGUF_PROFILE_SUFFIX: &str = ".mt-profile.json";
/// Rejection for model ids that do not resolve to a path strictly inside the models folder.
const OUTSIDE_MODELS_DIR: &str = "El modelo local debe estar dentro de la carpeta de modelos.";

/// Where the model hub is in its inspect → install cycle.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    #[default]
    Idle,
    Inspecting,
    Ready,
    Downloading,
    Done,
    Failed,
}

/// What the UI shows about the last inspected model.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InspectionView {
    pub repo_id: String,
    pub format: Option<ModelFormat>,
    pub folder: String,
    pub files: Vec<PlannedFile>,
    pub total_bytes: u64,
    pub quant: Option<String>,
    pub gguf_options: Vec<GgufOption>,
    pub fit: Fit,
    pub needed_bytes: u64,
    pub ram_bytes: u64,
    pub disk_ok: bool,
    pub disk_free_bytes: u64,
    pub can_install: bool,
    /// True when the repository declares tool calling.
    pub supports_tools: bool,
    pub verdict: String,
    /// Runtime settings suggested for the model, from the repository files it ships with.
    pub suggested_config: SuggestedModelConfig,
}

impl InspectionView {
    /// Combines the assessment with the download plan. Without a plan nothing can be installed.
    pub fn new(
        reference: &RepoRef,
        assessment: &Assessment,
        plan: Option<&ModelPlan>,
        gguf_options: Vec<GgufOption>,
    ) -> Self {
        Self {
            repo_id: reference.repo_id.clone(),
            format: assessment.format,
            folder: plan.map_or_else(|| reference.folder_stem(), |p| p.folder.clone()),
            files: plan.map(|p| p.files.clone()).unwrap_or_default(),
            total_bytes: plan.map_or(0, |p| p.total_bytes),
            quant: plan.and_then(|p| p.quant.clone()),
            gguf_options,
            fit: assessment.fit,
            needed_bytes: assessment.needed_bytes,
            ram_bytes: assessment.ram_bytes,
            disk_ok: assessment.disk_ok,
            disk_free_bytes: assessment.disk_free_bytes,
            can_install: plan.is_some() && assessment.can_install(),
            supports_tools: assessment.supports_tools,
            verdict: assessment.verdict.clone(),
            suggested_config: assessment.suggested_config.clone(),
        }
    }
}

/// A model already installed in the AI models folder.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalAiModel {
    /// Folder name for MLX models; path relative to the models folder for GGUF files.
    pub id: String,
    pub name: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
    /// Saved profile of the model, or one derived from its files.
    pub suggested_config: SuggestedModelConfig,
}

/// Full hub state as sent to the UI, both as command result and as the `model-hub-state` event.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubSnapshot {
    pub phase: Phase,
    pub error: Option<String>,
    pub downloaded: u64,
    pub total: u64,
    pub inspection: Option<InspectionView>,
}

#[derive(Debug, Default)]
struct Inner {
    phase: Phase,
    error: Option<String>,
    /// Plan of the last successful inspection; `None` when that inspection cannot be installed.
    plan: Option<ModelPlan>,
    inspection: Option<InspectionView>,
    /// Cancellation for the running download only.
    cancel: Option<CancelToken>,
    downloaded: u64,
    total: u64,
}

/// Shared model hub state, managed by Tauri.
#[derive(Debug, Default)]
pub struct ModelHubState {
    inner: Mutex<Inner>,
}

impl ModelHubState {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn snapshot(&self) -> HubSnapshot {
        let inner = self.lock();
        HubSnapshot {
            phase: inner.phase,
            error: inner.error.clone(),
            downloaded: inner.downloaded,
            total: inner.total,
            inspection: inner.inspection.clone(),
        }
    }

    fn begin_inspect(&self) -> Result<(), String> {
        let mut inner = self.lock();
        ensure_not_busy(inner.phase)?;
        *inner = Inner {
            phase: Phase::Inspecting,
            ..Inner::default()
        };
        Ok(())
    }

    fn finish_inspect(&self, plan: Option<ModelPlan>, inspection: InspectionView) {
        let mut inner = self.lock();
        inner.phase = Phase::Ready;
        inner.plan = plan;
        inner.inspection = Some(inspection);
    }

    /// Starts the install of the last inspected plan. A failed attempt may be retried directly.
    fn begin_download(&self) -> Result<(ModelPlan, SuggestedModelConfig, CancelToken), String> {
        let mut inner = self.lock();
        if !matches!(inner.phase, Phase::Ready | Phase::Failed) {
            return Err("Primero consulta un modelo instalable.".to_string());
        }
        let installable = inner.inspection.as_ref().filter(|view| view.can_install);
        let (plan, suggested) = match (&inner.plan, installable) {
            (Some(plan), Some(view)) => (plan.clone(), view.suggested_config.clone()),
            _ => return Err("Este modelo no se puede instalar en este equipo.".to_string()),
        };
        let cancel = CancelToken::new();
        inner.phase = Phase::Downloading;
        inner.error = None;
        inner.cancel = Some(cancel.clone());
        inner.downloaded = 0;
        inner.total = plan.total_bytes;
        Ok((plan, suggested, cancel))
    }

    fn record_progress(&self, progress: Progress) {
        let mut inner = self.lock();
        inner.downloaded = progress.downloaded;
        inner.total = progress.total;
    }

    fn finish_download(&self, outcome: Result<(), String>) {
        let mut inner = self.lock();
        inner.cancel = None;
        match outcome {
            Ok(()) => {
                inner.phase = Phase::Done;
                inner.error = None;
            }
            Err(message) => {
                inner.phase = Phase::Failed;
                inner.error = Some(message);
            }
        }
    }

    fn cancel(&self) {
        let inner = self.lock();
        if let (Phase::Downloading, Some(token)) = (inner.phase, &inner.cancel) {
            token.cancel();
        }
    }

    fn fail(&self, message: String) {
        let mut inner = self.lock();
        inner.phase = Phase::Failed;
        inner.error = Some(message);
    }

    /// Returns to idle. With `discard`, also deletes the partial download of the last plan.
    /// Partial files are kept otherwise, so a later install of the same model resumes.
    fn reset(&self, discard: bool, models_dir: &Path) -> Result<(), String> {
        let mut inner = self.lock();
        ensure_not_busy(inner.phase)?;
        if discard && let Some(plan) = &inner.plan {
            discard_partial(models_dir, &plan.folder).map_err(|err| err.to_string())?;
        }
        *inner = Inner::default();
        Ok(())
    }
}

fn ensure_not_busy(phase: Phase) -> Result<(), String> {
    match phase {
        Phase::Inspecting => Err("Ya hay una consulta de modelo en curso.".to_string()),
        Phase::Downloading => Err("Hay una descarga en curso; cancélala primero.".to_string()),
        _ => Ok(()),
    }
}

#[tauri::command]
pub fn model_hub_state(state: State<'_, ModelHubState>) -> HubSnapshot {
    state.snapshot()
}

/// Looks up a Hugging Face repository and assesses it on this machine. `gguf_filename` picks one
/// GGUF quantization when the repository has several.
#[tauri::command]
pub async fn model_hub_inspect(
    url: String,
    gguf_filename: Option<String>,
    app: AppHandle,
    store: State<'_, SettingsStore>,
    state: State<'_, ModelHubState>,
) -> Result<InspectionView, String> {
    state.begin_inspect()?;
    emit_state(&app, &state);

    let settings = store.get();
    let outcome = match resolve_ai_models_dir(&app, &settings) {
        Ok(models_dir) => {
            inspect(
                &url,
                gguf_filename.as_deref(),
                &models_dir,
                settings.context_limit_tokens,
            )
            .await
        }
        Err(err) => Err(err),
    };
    match outcome {
        Ok((plan, view)) => {
            state.finish_inspect(plan, view.clone());
            emit_state(&app, &state);
            Ok(view)
        }
        Err(message) => {
            state.fail(message.clone());
            emit_state(&app, &state);
            Err(message)
        }
    }
}

/// Installs the last inspected model. Resolves to the installed folder.
#[tauri::command]
pub async fn model_hub_download(
    app: AppHandle,
    store: State<'_, SettingsStore>,
    state: State<'_, ModelHubState>,
) -> Result<String, String> {
    let (plan, suggested, cancel) = state.begin_download()?;
    emit_state(&app, &state);

    let outcome = download(&app, &store.get(), &state, &plan, &suggested, &cancel).await;
    state.finish_download(outcome.clone().map(|_| ()));
    emit_state(&app, &state);
    outcome
}

/// Requests cancellation of the running download. Partial files are kept for resuming.
#[tauri::command]
pub fn model_hub_cancel(app: AppHandle, state: State<'_, ModelHubState>) -> HubSnapshot {
    state.cancel();
    emit_state(&app, &state);
    state.snapshot()
}

/// Clears the hub state. Refused while an inspection or download is running.
#[tauri::command]
pub fn model_hub_reset(
    discard: bool,
    app: AppHandle,
    store: State<'_, SettingsStore>,
    state: State<'_, ModelHubState>,
) -> Result<HubSnapshot, String> {
    let models_dir = resolve_ai_models_dir(&app, &store.get())?;
    state.reset(discard, &models_dir)?;
    emit_state(&app, &state);
    Ok(state.snapshot())
}

/// Lists the models in the folder `dir` when one is given (a blank one counts as none), otherwise in
/// the configured AI models folder (or the default one).
#[tauri::command]
pub fn list_local_ai_models(
    dir: Option<String>,
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<Vec<LocalAiModel>, String> {
    let models_dir =
        listing_models_dir(dir.as_deref(), || resolve_ai_models_dir(&app, &store.get()))?;
    scan_local_models(&models_dir, Platform::current())
}

/// The folder `list_local_ai_models` scans: the explicit `dir` (trimmed) when it is not blank, otherwise
/// the saved settings folder. The saved folder is only resolved when no explicit folder is given.
fn listing_models_dir(
    dir: Option<&str>,
    saved: impl FnOnce() -> Result<PathBuf, String>,
) -> Result<PathBuf, String> {
    match dir.map(str::trim).filter(|dir| !dir.is_empty()) {
        Some(explicit) => Ok(PathBuf::from(explicit)),
        None => saved(),
    }
}

/// Reads the suggested settings of an installed model: the saved profile, or one derived from its files.
#[tauri::command]
pub fn get_local_model_profile(
    model_id: String,
    app: AppHandle,
    store: State<'_, SettingsStore>,
) -> Result<SuggestedModelConfig, String> {
    let models_dir = resolve_ai_models_dir(&app, &store.get())?;
    read_model_profile(&models_dir, &model_id)
}

/// Deletes an installed model. Refused while a download writes into its folder. A local server that
/// runs the model is stopped first so Windows releases its file locks. The selected model is cleared
/// when it is the one deleted.
#[tauri::command]
pub fn delete_local_ai_model(
    model_id: String,
    app: AppHandle,
    store: State<'_, SettingsStore>,
    state: State<'_, ModelHubState>,
) -> Result<(), String> {
    let models_dir = resolve_ai_models_dir(&app, &store.get())?;
    let (_, target) = resolve_model_path(&models_dir, &model_id)?;
    let folder = Path::new(model_id.trim())
        .iter()
        .next()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    {
        let inner = state.lock();
        if inner.phase == Phase::Downloading
            && inner.plan.as_ref().is_some_and(|plan| plan.folder == folder)
        {
            return Err("Hay una descarga en curso en esa carpeta; cancélala primero.".to_string());
        }
        stop_local_server_using(&app, &target);
        delete_model_at(&models_dir, &model_id)?;
    }
    if store.get().ai_model_id.trim() == model_id.trim() {
        store.update(|current| {
            let mut next = current.clone();
            next.ai_model_id = String::new();
            Ok(next)
        })?;
    }
    Ok(())
}

fn emit_state(app: &AppHandle, state: &ModelHubState) {
    if let Err(err) = app.emit(STATE_EVENT, state.snapshot()) {
        eprintln!("model hub: could not emit state: {err}");
    }
}

async fn inspect(
    url: &str,
    gguf_filename: Option<&str>,
    models_dir: &Path,
    context_tokens: u32,
) -> Result<(Option<ModelPlan>, InspectionView), String> {
    let reference = parse_reference(url).map_err(|err| err.to_string())?;
    let hub = HubClient::new().map_err(|err| err.to_string())?;
    let info = hub
        .repo_info(&reference)
        .await
        .map_err(|err| err.to_string())?;
    let config = hub
        .fetch_config(&reference)
        .await
        .map_err(|err| err.to_string())?;
    let generation_config = hub
        .fetch_generation_config(&reference)
        .await
        .map_err(|err| err.to_string())?;
    let tokenizer_config = hub
        .fetch_tokenizer_config(&reference)
        .await
        .map_err(|err| err.to_string())?;
    // The chat template of a GGUF model is usually embedded in the file, so its header is the fallback.
    let gguf_header = match (&tokenizer_config, planned_gguf_file(&info, &reference, gguf_filename)) {
        (None, Some(file)) => hub
            .fetch_gguf_header_sample(&reference, &file)
            .await
            .map_err(|err| err.to_string())?,
        _ => None,
    };
    let target = models_dir.to_path_buf();
    // Disk and RAM probes run external tools, so keep them off the async runtime.
    let machine = tokio::task::spawn_blocking(move || detect_machine(&target))
        .await
        .map_err(|err| err.to_string())?;
    let fetched = FetchedFiles {
        config: config.as_ref(),
        generation_config: generation_config.as_ref(),
        tokenizer_config: tokenizer_config.as_ref(),
        gguf_header: gguf_header.as_deref(),
    };
    Ok(build_inspection(
        &info,
        &reference,
        &fetched,
        &machine,
        gguf_filename,
        context_tokens,
    ))
}

/// Optional repository files fetched for assessment. Each one is `None` when the repository lacks it.
#[derive(Debug, Default, Clone, Copy)]
struct FetchedFiles<'a> {
    config: Option<&'a serde_json::Value>,
    generation_config: Option<&'a serde_json::Value>,
    tokenizer_config: Option<&'a serde_json::Value>,
    gguf_header: Option<&'a [u8]>,
}

/// The GGUF file the install would use (first shard for sharded models). `None` for non-GGUF plans.
fn planned_gguf_file(
    info: &RepoInfo,
    reference: &RepoRef,
    gguf_filename: Option<&str>,
) -> Option<String> {
    let plan = ModelPlan::build(info, reference, gguf_filename).ok()?;
    if plan.format != ModelFormat::Gguf {
        return None;
    }
    plan.files.first().map(|file| file.name.clone())
}

/// Assesses a fetched repository. The plan is `None` when the repository has nothing installable.
fn build_inspection(
    info: &RepoInfo,
    reference: &RepoRef,
    fetched: &FetchedFiles<'_>,
    machine: &MachineInfo,
    gguf_filename: Option<&str>,
    context_tokens: u32,
) -> (Option<ModelPlan>, InspectionView) {
    let assessment = assess(&AssessRequest {
        info,
        reference,
        config: fetched.config,
        generation_config: fetched.generation_config,
        tokenizer_config: fetched.tokenizer_config,
        gguf_header: fetched.gguf_header,
        machine,
        gguf_filename,
        context_tokens,
    });
    let plan = ModelPlan::build(info, reference, gguf_filename).ok();
    let gguf_options = if assessment.format == Some(ModelFormat::Gguf) {
        list_gguf_options(info)
    } else {
        Vec::new()
    };
    let view = InspectionView::new(reference, &assessment, plan.as_ref(), gguf_options);
    (plan, view)
}

/// Runs the installer, saves the inspected settings beside the model, and returns the installed folder.
async fn download(
    app: &AppHandle,
    settings: &AppSettings,
    state: &ModelHubState,
    plan: &ModelPlan,
    suggested: &SuggestedModelConfig,
    cancel: &CancelToken,
) -> Result<String, String> {
    let models_dir = resolve_ai_models_dir(app, settings)?;
    let hub = HubClient::new().map_err(|err| err.to_string())?;
    let downloader = Downloader::new(hub, models_dir);
    let folder = downloader
        .install(plan, cancel, |progress| {
            state.record_progress(progress);
            emit_state(app, state);
        })
        .await
        .map_err(|err| err.to_string())?;
    save_installed_profile(&folder, plan.format, suggested);
    Ok(folder.display().to_string())
}

/// Saves the inspected settings where `load_local_suggested_config` finds them: `mt-profile.json` in the
/// installed folder and, for GGUF, `<file>.mt-profile.json` beside each `.gguf` file. Best effort: the
/// model is already installed, so a failed write is logged instead of failing the install.
fn save_installed_profile(folder: &Path, format: ModelFormat, suggested: &SuggestedModelConfig) {
    let mut targets = vec![folder.to_path_buf()];
    if format == ModelFormat::Gguf {
        targets.extend(gguf_files_in(folder));
    }
    for target in targets {
        if let Err(err) = save_local_suggested_config(&target, suggested) {
            eprintln!("model hub: could not save the profile of {}: {err}", target.display());
        }
    }
}

/// Scans `dir` for installed models. On Apple Silicon these are MLX folders with `config.json` and
/// `.safetensors` weights, at the top level (`org--model`) or one folder down (`org/model`); elsewhere
/// they are `.gguf` files at the top level or one folder down. A missing folder holds no models.
pub fn scan_local_models(dir: &Path, platform: Platform) -> Result<Vec<LocalAiModel>, String> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(format!("No se pudo leer {}: {err}", dir.display())),
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if platform.is_apple_silicon() {
            let top_level = mlx_model(&path, name.clone());
            if path.is_dir() && top_level.is_none() {
                for child in fs::read_dir(&path).into_iter().flatten().flatten() {
                    let child_name = child.file_name().to_string_lossy().into_owned();
                    if !child_name.starts_with('.') {
                        models.extend(mlx_model(&child.path(), format!("{name}/{child_name}")));
                    }
                }
            }
            models.extend(top_level);
        } else if path.is_dir() {
            for child in fs::read_dir(&path).into_iter().flatten().flatten() {
                let child_name = child.file_name().to_string_lossy().into_owned();
                if !child_name.starts_with('.') {
                    models.extend(gguf_model(&child.path(), format!("{name}/{child_name}")));
                }
            }
        } else {
            models.extend(gguf_model(&path, name));
        }
    }
    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}

fn mlx_model(path: &Path, id: String) -> Option<LocalAiModel> {
    if !path.is_dir() || !path.join("config.json").is_file() {
        return None;
    }
    let mut has_weights = false;
    let mut size_bytes = 0;
    for entry in fs::read_dir(path).ok()?.flatten() {
        let file = entry.path();
        if !file.is_file() {
            continue;
        }
        has_weights |= file.extension().and_then(|ext| ext.to_str()) == Some("safetensors");
        size_bytes += entry.metadata().map_or(0, |meta| meta.len());
    }
    has_weights.then(|| LocalAiModel {
        name: id.clone(),
        id,
        format: ModelFormat::Mlx,
        size_bytes,
        suggested_config: load_local_suggested_config(path),
    })
}

/// A `.gguf` file as a model. Sharded models are listed once, through their first shard, with the
/// size of the whole shard set.
fn gguf_model(path: &Path, id: String) -> Option<LocalAiModel> {
    if !path.is_file() {
        return None;
    }
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".gguf")?;
    let (name, size_bytes) = match shard_parts(stem) {
        Some((_, index)) if index != 1 => return None,
        Some((base, _)) => (base.to_string(), shard_set_bytes(path, base)),
        None => (stem.to_string(), file_len(path)),
    };
    Some(LocalAiModel {
        id,
        name,
        format: ModelFormat::Gguf,
        size_bytes,
        suggested_config: load_local_suggested_config(path),
    })
}

/// Splits `<base>-NNNNN-of-NNNNN` into `(base, index)`.
fn shard_parts(stem: &str) -> Option<(&str, u32)> {
    let (head, total) = stem.rsplit_once("-of-")?;
    let (base, index) = head.rsplit_once('-')?;
    let all_digits = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(index) || !all_digits(total) {
        return None;
    }
    Some((base, index.parse().ok()?))
}

/// Every `.gguf` shard of the split model `base` that sits in the same folder as `first`.
fn shard_set_files(first: &Path, base: &str) -> Vec<PathBuf> {
    let Some(dir) = first.parent() else {
        return vec![first.to_path_buf()];
    };
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            entry.file_type().is_ok_and(|kind| kind.is_file())
                && name
                    .to_str()
                    .and_then(|name| name.strip_suffix(".gguf"))
                    .and_then(shard_parts)
                    .is_some_and(|(shard_base, _)| shard_base == base)
        })
        .map(|entry| entry.path())
        .collect()
}

/// Total bytes of every `.gguf` shard of `base` that sits in the same folder as `first`.
fn shard_set_bytes(first: &Path, base: &str) -> u64 {
    shard_set_files(first, base)
        .iter()
        .map(|path| file_len(path))
        .sum()
}

fn file_len(path: &Path) -> u64 {
    path.metadata().map_or(0, |meta| meta.len())
}

/// The `.gguf` files directly inside `dir`.
fn gguf_files_in(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
        })
        .collect()
}

/// Reads the suggested settings of the model `model_id` inside `models_dir`.
fn read_model_profile(models_dir: &Path, model_id: &str) -> Result<SuggestedModelConfig, String> {
    let (_, target) = resolve_model_path(models_dir, model_id)?;
    Ok(load_local_suggested_config(&target))
}

/// Deletes the installed model `model_id` inside `models_dir`: an MLX folder (`org--model`, or
/// `org/model` one folder down), or a GGUF file with its profile sidecar and, for a split model, every
/// shard and sidecar. A parent folder left holding nothing but its folder profile is removed too. The
/// models folder itself is never removed.
fn delete_model_at(models_dir: &Path, model_id: &str) -> Result<(), String> {
    let (root, target) = resolve_model_path(models_dir, model_id)?;
    if target.is_dir() {
        // Only the folders `scan_local_models` lists: MLX models at the top level, or one folder down
        // inside a folder that is not itself an MLX model.
        let depth = target
            .strip_prefix(&root)
            .map_or(0, |relative| relative.components().count());
        let nested_in_model = depth == 2 && target.parent().is_some_and(is_mlx_folder);
        let listed = is_mlx_folder(&target) && (depth == 1 || (depth == 2 && !nested_in_model));
        if !listed {
            return Err(
                "Solo se pueden borrar carpetas MLX de primer o segundo nivel.".to_string(),
            );
        }
        remove_entry(&target, true)?;
    } else {
        let stem = target
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_suffix(".gguf"));
        let Some(stem) = stem else {
            return Err("Solo se pueden borrar carpetas MLX o archivos GGUF.".to_string());
        };
        let files = match shard_parts(stem) {
            Some((base, _)) => shard_set_files(&target, base),
            None => vec![target.clone()],
        };
        for file in files {
            let sidecar = gguf_sidecar(&file);
            remove_entry(&file, false)?;
            if sidecar.exists() {
                remove_entry(&sidecar, false)?;
            }
        }
    }
    // Best effort: a folder that is not emptied by the delete stays as it is.
    if let Some(parent) = target.parent()
        && parent != root.as_path()
        && parent.starts_with(&root)
        && holds_only_profile(parent)
    {
        let _ = fs::remove_dir_all(parent);
    }
    Ok(())
}

/// True when `dir` is an MLX model folder, by the same test `scan_local_models` applies.
fn is_mlx_folder(dir: &Path) -> bool {
    mlx_model(dir, String::new()).is_some()
}

/// Resolves `model_id` (a folder name, or a path relative to the models folder) to the canonical
/// model path, which must lie strictly inside the canonical models folder. Returns `(root, target)`.
fn resolve_model_path(models_dir: &Path, model_id: &str) -> Result<(PathBuf, PathBuf), String> {
    let model_id = model_id.trim();
    if model_id.is_empty() {
        return Err("Selecciona un modelo local de IA.".to_string());
    }
    let relative = Path::new(model_id);
    let normal_only = relative.components().next().is_some()
        && relative.components().all(|component| {
            matches!(component, Component::Normal(name) if !name.to_string_lossy().starts_with('.'))
        });
    if !normal_only {
        return Err(OUTSIDE_MODELS_DIR.to_string());
    }
    let root = fs::canonicalize(models_dir).map_err(|err| {
        format!("No se encontró la carpeta de modelos {}: {err}", models_dir.display())
    })?;
    let target = fs::canonicalize(models_dir.join(relative))
        .map_err(|_| format!("No se encontró el modelo local {model_id}."))?;
    if target == root || !target.starts_with(&root) {
        return Err(OUTSIDE_MODELS_DIR.to_string());
    }
    Ok((root, target))
}

/// Removes one file or folder, reporting which path failed.
fn remove_entry(path: &Path, dir: bool) -> Result<(), String> {
    let result = if dir {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    result.map_err(|err| format!("No se pudo borrar {}: {err}", path.display()))
}

/// True when `dir` is empty or holds only the folder profile.
fn holds_only_profile(dir: &Path) -> bool {
    match fs::read_dir(dir) {
        Ok(mut entries) => entries.all(|entry| {
            entry.is_ok_and(|entry| entry.file_name().to_str() == Some(FOLDER_PROFILE_FILE))
        }),
        Err(_) => false,
    }
}

/// Profile sidecar of a GGUF file: `<file>.mt-profile.json`.
fn gguf_sidecar(file: &Path) -> PathBuf {
    let mut name = file.as_os_str().to_owned();
    name.push(GGUF_PROFILE_SUFFIX);
    PathBuf::from(name)
}

/// Stops the local server when it runs a model inside `target`, so Windows releases the file locks.
fn stop_local_server_using(app: &AppHandle, target: &Path) {
    let manager = app.state::<AiRuntimeState>().manager();
    let status = manager.status();
    let serves_target = status.state != LocalServerState::Stopped
        && status.model_path.as_deref().is_some_and(|path| {
            let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
            canonical.starts_with(target)
        });
    if serves_target {
        manager.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mt_models::Sibling;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const MAC_ARM: Platform = Platform {
        os: "macos",
        arch: "aarch64",
    };
    const WINDOWS_X64: Platform = Platform {
        os: "windows",
        arch: "x86_64",
    };

    fn write(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn list_local_ai_models_returns_only_complete_mlx_folders_on_apple_silicon() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("whisper-mlx/config.json"), 10);
        write(&root.join("whisper-mlx/model.safetensors"), 100);
        write(&root.join("whisper-mlx/tokenizer.json"), 5);
        write(&root.join("no-weights/config.json"), 10);
        write(&root.join("gguf-only/model.gguf"), 50);
        write(
            &root.join(".mt-downloading/whisper-mlx/model.safetensors"),
            1,
        );
        write(&root.join("loose.safetensors"), 1);

        let models = scan_local_models(root, MAC_ARM).unwrap();
        assert_eq!(
            models,
            vec![LocalAiModel {
                id: "whisper-mlx".into(),
                name: "whisper-mlx".into(),
                format: ModelFormat::Mlx,
                size_bytes: 115,
                suggested_config: safe_defaults(),
            }]
        );
    }

    #[test]
    fn list_local_ai_models_finds_gguf_files_at_top_level_and_one_folder_down() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("top-Q4_K_M.gguf"), 7);
        write(&root.join("family/nested-Q8_0.gguf"), 9);
        write(&root.join("family/notes.txt"), 1);
        write(&root.join("family/deep/too-deep.gguf"), 1);
        write(&root.join("split-00001-of-00002.gguf"), 4);
        write(&root.join("split-00002-of-00002.gguf"), 4);
        write(&root.join("partial.gguf.part"), 3);
        write(&root.join("mlx-folder/config.json"), 1);
        write(&root.join("mlx-folder/model.safetensors"), 1);

        let models = scan_local_models(root, WINDOWS_X64).unwrap();
        let gguf = |id: &str, name: &str, size_bytes: u64| LocalAiModel {
            id: id.into(),
            name: name.into(),
            format: ModelFormat::Gguf,
            size_bytes,
            suggested_config: safe_defaults(),
        };
        assert_eq!(
            models,
            vec![
                gguf("family/nested-Q8_0.gguf", "nested-Q8_0", 9),
                gguf("split-00001-of-00002.gguf", "split", 8),
                gguf("top-Q4_K_M.gguf", "top-Q4_K_M", 7),
            ]
        );
    }

    #[test]
    fn list_local_ai_models_is_empty_when_the_folder_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let models = scan_local_models(&dir.path().join("absent"), MAC_ARM).unwrap();
        assert!(models.is_empty());
    }

    fn repo(files: &[(&str, u64)]) -> RepoInfo {
        RepoInfo {
            repo_id: "org/model".into(),
            sha: None,
            gated: serde_json::Value::Bool(false),
            tags: Vec::new(),
            siblings: files
                .iter()
                .map(|(name, size)| Sibling {
                    rfilename: (*name).into(),
                    size: Some(*size),
                    lfs: None,
                })
                .collect(),
        }
    }

    fn machine(os: &str, arch: &str) -> MachineInfo {
        MachineInfo {
            os: os.into(),
            arch: arch.into(),
            ram_bytes: 16 * GIB,
            disk_free_bytes: 100 * GIB,
            unified_memory: os == "macos" && arch == "aarch64",
            gpu_vram_bytes: None,
        }
    }

    fn inspect_repo(
        info: &RepoInfo,
        os: &str,
        arch: &str,
        gguf_filename: Option<&str>,
    ) -> (InspectionView, serde_json::Value) {
        let reference = RepoRef {
            repo_id: "org/model".into(),
            revision: "main".into(),
        };
        let (_, view) = build_inspection(
            info,
            &reference,
            &FetchedFiles::default(),
            &machine(os, arch),
            gguf_filename,
            4096,
        );
        let json = serde_json::to_value(&view).unwrap();
        (view, json)
    }

    #[test]
    fn inspection_view_asks_for_a_quantization_when_several_gguf_options_exist() {
        let info = repo(&[("model-Q4_K_M.gguf", 2 * MIB), ("model-Q8_0.gguf", 3 * MIB)]);
        let (view, json) = inspect_repo(&info, "macos", "aarch64", None);

        assert_eq!(view.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(view.gguf_options.len(), 2);
        assert!(!view.can_install, "{}", view.verdict);
        assert_eq!(view.fit, Fit::Fits);

        let keys = [
            "repoId",
            "format",
            "folder",
            "files",
            "totalBytes",
            "quant",
            "ggufOptions",
            "fit",
            "neededBytes",
            "ramBytes",
            "diskOk",
            "diskFreeBytes",
            "canInstall",
            "supportsTools",
            "verdict",
            "suggestedConfig",
        ];
        let object = json.as_object().unwrap();
        assert_eq!(object.len(), keys.len());
        for key in keys {
            assert!(object.contains_key(key), "missing {key}");
        }
        assert_eq!(json["repoId"], "org/model");
        assert_eq!(json["format"], "gguf");
        assert_eq!(json["fit"], "fits");
        assert_eq!(json["canInstall"], false);
        assert_eq!(json["supportsTools"], false);
    }

    #[test]
    fn inspection_view_can_install_once_a_quantization_is_chosen() {
        let info = with_tool_tag(repo(&[("model-Q4_K_M.gguf", 2 * MIB), ("model-Q8_0.gguf", 3 * MIB)]));
        let (view, json) = inspect_repo(&info, "macos", "aarch64", Some("model-Q8_0.gguf"));

        assert!(view.can_install, "{}", view.verdict);
        assert_eq!(view.quant.as_deref(), Some("Q8_0"));
        assert_eq!(view.folder, "org--model");
        assert_eq!(view.total_bytes, 3 * MIB);
        assert_eq!(json["totalBytes"], 3 * MIB);
        assert_eq!(json["files"][0]["name"], "model-Q8_0.gguf");
    }

    #[test]
    fn inspection_view_blocks_mlx_repositories_on_windows_even_with_a_plan() {
        let info = repo(&[("model.safetensors", GIB), ("config.json", 1)]);
        let (view, json) = inspect_repo(&info, "windows", "x86_64", None);

        assert_eq!(view.format, Some(ModelFormat::Mlx));
        assert!(view.total_bytes > 0, "the plan itself is still shown");
        assert!(!view.can_install);
        assert_eq!(json["format"], "mlx");
        assert!(view.verdict.contains("MLX"), "{}", view.verdict);
    }

    fn with_tool_tag(mut info: RepoInfo) -> RepoInfo {
        info.tags.push("tool-use".into());
        info
    }

    fn tool_tokenizer() -> serde_json::Value {
        serde_json::json!({ "chat_template": "{% if tools %}{{ tools }}{% endif %}" })
    }

    #[test]
    fn inspection_view_blocks_a_model_that_declares_no_tool_support() {
        let info = repo(&[("model-Q4_K_M.gguf", 2 * MIB)]);
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        let (plan, view) = build_inspection(
            &info,
            &reference,
            &FetchedFiles::default(),
            &machine("macos", "aarch64"),
            Some("model-Q4_K_M.gguf"),
            4096,
        );
        assert!(plan.is_some(), "the plan is still shown");
        assert!(!view.supports_tools);
        assert!(!view.can_install, "{}", view.verdict);
    }

    #[test]
    fn inspection_view_accepts_a_chat_template_that_renders_tools() {
        let info = repo(&[("model-Q4_K_M.gguf", 2 * MIB)]);
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        let tokenizer = tool_tokenizer();
        let fetched = FetchedFiles { tokenizer_config: Some(&tokenizer), ..FetchedFiles::default() };
        let (_, view) = build_inspection(
            &info,
            &reference,
            &fetched,
            &machine("macos", "aarch64"),
            Some("model-Q4_K_M.gguf"),
            4096,
        );
        assert!(view.supports_tools);
        assert!(view.can_install, "{}", view.verdict);
    }

    #[test]
    fn inspection_view_reads_tool_support_from_the_gguf_header_without_tokenizer_config() {
        let info = repo(&[("model-Q4_K_M.gguf", 2 * MIB)]);
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        let template = "{% if tools %}{{ tools }}{% endif %}";
        let key = "tokenizer.chat_template";
        let mut header = b"GGUF\x03\x00\x00\x00".to_vec();
        header.extend_from_slice(&(key.len() as u64).to_le_bytes());
        header.extend_from_slice(key.as_bytes());
        header.extend_from_slice(&8u32.to_le_bytes());
        header.extend_from_slice(&(template.len() as u64).to_le_bytes());
        header.extend_from_slice(template.as_bytes());
        let fetched = FetchedFiles { gguf_header: Some(&header), ..FetchedFiles::default() };
        let (_, view) = build_inspection(
            &info,
            &reference,
            &fetched,
            &machine("macos", "aarch64"),
            Some("model-Q4_K_M.gguf"),
            4096,
        );
        assert!(view.supports_tools);
        assert!(view.can_install, "{}", view.verdict);
    }

    #[test]
    fn planned_gguf_file_names_the_file_the_install_would_use() {
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        let info = repo(&[("model-Q4_K_M.gguf", MIB), ("model-Q8_0.gguf", MIB)]);
        assert_eq!(
            planned_gguf_file(&info, &reference, None).as_deref(),
            Some("model-Q4_K_M.gguf")
        );
        assert_eq!(
            planned_gguf_file(&info, &reference, Some("model-Q8_0.gguf")).as_deref(),
            Some("model-Q8_0.gguf")
        );
        let mlx = repo(&[("model.safetensors", MIB), ("config.json", 1)]);
        assert_eq!(planned_gguf_file(&mlx, &reference, None), None);
    }

    fn safe_defaults() -> SuggestedModelConfig {
        SuggestedModelConfig {
            source_summary: "valores seguros por defecto".into(),
            ..Default::default()
        }
    }

    #[test]
    fn list_local_ai_models_reports_the_saved_profile_of_each_model() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("whisper-mlx/config.json"), 10);
        write(&root.join("whisper-mlx/model.safetensors"), 100);
        let saved = SuggestedModelConfig {
            top_k: Some(40),
            source_summary: "manual".into(),
            ..Default::default()
        };
        save_local_suggested_config(&root.join("whisper-mlx"), &saved).unwrap();

        let models = scan_local_models(root, MAC_ARM).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].suggested_config, saved);
    }

    #[test]
    fn inspection_view_suggests_settings_from_the_generation_config() {
        let info = with_tool_tag(repo(&[("model-Q4_K_M.gguf", 2 * MIB)]));
        let reference = RepoRef { repo_id: "org/model".into(), revision: "main".into() };
        let generation = serde_json::json!({ "temperature": 0.6, "top_k": 20 });
        let fetched = FetchedFiles {
            generation_config: Some(&generation),
            ..FetchedFiles::default()
        };
        let (_, view) = build_inspection(
            &info,
            &reference,
            &fetched,
            &machine("macos", "aarch64"),
            Some("model-Q4_K_M.gguf"),
            4096,
        );
        assert_eq!(view.suggested_config.temperature, Some(0.6));
        assert_eq!(view.suggested_config.top_k, Some(20));
        assert_eq!(view.suggested_config.source_summary, "generation_config.json");
        assert!(view.can_install, "{}", view.verdict);
        let json = serde_json::to_value(&view).unwrap();
        assert_eq!(json["suggestedConfig"]["sourceSummary"], "generation_config.json");
    }

    #[test]
    fn delete_model_at_removes_an_mlx_folder_and_keeps_its_siblings() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("whisper-mlx/config.json"), 10);
        write(&models.join("whisper-mlx/model.safetensors"), 100);
        write(&models.join("other-model/config.json"), 10);
        write(&models.join("other-model/model.safetensors"), 10);

        delete_model_at(models, "whisper-mlx").unwrap();

        assert!(!models.join("whisper-mlx").exists());
        assert!(models.join("other-model/model.safetensors").is_file());
    }

    #[test]
    fn delete_model_at_removes_a_gguf_file_with_its_sidecar_and_the_folder_it_empties() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/model-Q4.gguf"), 10);
        write(&models.join("family/model-Q4.gguf.mt-profile.json"), 2);
        write(&models.join("family/mt-profile.json"), 2);
        write(&models.join("top.gguf"), 5);
        write(&models.join("top.gguf.mt-profile.json"), 2);
        write(&models.join("keep.gguf"), 5);

        delete_model_at(models, "family/model-Q4.gguf").unwrap();
        assert!(
            !models.join("family").exists(),
            "a folder left with only its profile is removed"
        );

        delete_model_at(models, "top.gguf").unwrap();
        assert!(!models.join("top.gguf").exists());
        assert!(!models.join("top.gguf.mt-profile.json").exists());
        assert!(models.join("keep.gguf").is_file(), "the models folder itself is kept");
    }

    #[test]
    fn delete_model_at_keeps_a_folder_that_still_holds_another_model() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/model-Q4.gguf"), 10);
        write(&models.join("family/model-Q8.gguf"), 10);
        write(&models.join("family/mt-profile.json"), 2);

        delete_model_at(models, "family/model-Q4.gguf").unwrap();

        assert!(!models.join("family/model-Q4.gguf").exists());
        assert!(models.join("family/model-Q8.gguf").is_file());
        assert!(models.join("family/mt-profile.json").is_file());
    }

    #[test]
    fn delete_model_at_removes_every_shard_of_a_split_model() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/split-00001-of-00002.gguf"), 4);
        write(&models.join("family/split-00002-of-00002.gguf"), 4);
        write(&models.join("family/split-00001-of-00002.gguf.mt-profile.json"), 2);

        delete_model_at(models, "family/split-00001-of-00002.gguf").unwrap();

        assert!(
            !models.join("family").exists(),
            "the whole shard set and its profile are removed"
        );
    }

    #[test]
    fn delete_model_at_rejects_ids_that_leave_the_models_folder() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("ai-models");
        write(&models.join("keep.gguf"), 3);
        write(&dir.path().join("outside/victim.gguf"), 3);

        let rejected = [
            "",
            "   ",
            "../outside",
            "../outside/victim.gguf",
            "keep.gguf/../../outside",
            "/abs/victim.gguf",
            "./keep.gguf",
            ".mt-downloading",
            "missing.gguf",
        ];
        for id in rejected {
            assert!(delete_model_at(&models, id).is_err(), "{id:?} must be rejected");
        }
        assert!(models.join("keep.gguf").is_file());
        assert!(dir.path().join("outside/victim.gguf").is_file());
    }

    #[test]
    fn delete_model_at_rejects_files_that_are_not_gguf_models() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("notes.txt"), 3);

        let error = delete_model_at(models, "notes.txt").unwrap_err();

        assert_eq!(error, "Solo se pueden borrar carpetas MLX o archivos GGUF.");
        assert!(models.join("notes.txt").is_file());
    }

    #[cfg(unix)]
    #[test]
    fn delete_model_at_rejects_a_symlink_that_points_outside_the_models_folder() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("ai-models");
        write(&models.join("keep.gguf"), 3);
        write(&dir.path().join("outside/victim.gguf"), 3);
        std::os::unix::fs::symlink(dir.path().join("outside"), models.join("link")).unwrap();

        assert!(delete_model_at(&models, "link/victim.gguf").is_err());
        assert!(dir.path().join("outside/victim.gguf").is_file());
    }

    #[test]
    fn read_model_profile_returns_the_saved_settings_and_rejects_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/model.gguf"), 4);
        let saved = SuggestedModelConfig {
            temperature: Some(0.3),
            source_summary: "manual".into(),
            ..Default::default()
        };
        save_local_suggested_config(&models.join("family/model.gguf"), &saved).unwrap();

        assert_eq!(read_model_profile(models, "family/model.gguf").unwrap(), saved);
        assert!(read_model_profile(models, "../model.gguf").is_err());
    }

    #[test]
    fn delete_model_at_refuses_a_folder_of_gguf_files_as_if_it_were_an_mlx_model() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/model-Q4.gguf"), 10);
        write(&models.join("family/model-Q8.gguf"), 10);
        write(&models.join("mlx/config.json"), 2);
        write(&models.join("mlx/model.safetensors"), 2);

        assert!(delete_model_at(models, "family").is_err());
        assert!(models.join("family/model-Q8.gguf").is_file());
        assert!(delete_model_at(models, "mlx/model.safetensors").is_err());
        assert!(models.join("mlx/model.safetensors").is_file());
    }

    #[test]
    fn listing_models_dir_prefers_an_explicit_folder_over_the_saved_settings() {
        let saved = || Ok(PathBuf::from("/saved/models"));

        let listed = listing_models_dir(Some("  /picked/models  "), saved).unwrap();

        assert_eq!(listed, PathBuf::from("/picked/models"));
    }

    #[test]
    fn listing_models_dir_uses_the_saved_settings_when_the_folder_is_absent_or_blank() {
        for dir in [None, Some(""), Some("   ")] {
            let listed =
                listing_models_dir(dir, || Ok(PathBuf::from("/saved/models"))).unwrap();
            assert_eq!(listed, PathBuf::from("/saved/models"), "{dir:?}");
        }
        let error = listing_models_dir(None, || Err("no saved folder".to_string())).unwrap_err();
        assert_eq!(error, "no saved folder");
    }

    #[test]
    fn scan_local_models_finds_one_level_and_nested_mlx_folders_on_apple_silicon() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        write(&root.join("whisper-mlx/config.json"), 10);
        write(&root.join("whisper-mlx/model.safetensors"), 100);
        write(&root.join("org/llama/config.json"), 10);
        write(&root.join("org/llama/model.safetensors"), 40);
        write(&root.join("org/notes/readme.txt"), 1);
        write(&root.join("org/notes/config.json"), 1);
        write(&root.join("family/gguf/model.gguf"), 9);
        write(&root.join(".hidden/llama/config.json"), 1);
        write(&root.join(".hidden/llama/model.safetensors"), 1);
        write(&root.join("org/.cache/config.json"), 1);
        write(&root.join("org/.cache/model.safetensors"), 1);

        let models = scan_local_models(root, MAC_ARM).unwrap();

        assert_eq!(
            models,
            vec![
                LocalAiModel {
                    id: "org/llama".into(),
                    name: "org/llama".into(),
                    format: ModelFormat::Mlx,
                    size_bytes: 50,
                    suggested_config: safe_defaults(),
                },
                LocalAiModel {
                    id: "whisper-mlx".into(),
                    name: "whisper-mlx".into(),
                    format: ModelFormat::Mlx,
                    size_bytes: 110,
                    suggested_config: safe_defaults(),
                },
            ]
        );
    }

    #[test]
    fn delete_model_at_removes_a_nested_mlx_folder_and_the_org_folder_it_empties() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("org/llama/config.json"), 10);
        write(&models.join("org/llama/model.safetensors"), 100);
        write(&models.join("other/keep/config.json"), 10);
        write(&models.join("other/keep/model.safetensors"), 10);

        delete_model_at(models, "org/llama").unwrap();

        assert!(!models.join("org").exists(), "the emptied org folder is removed");
        assert!(models.join("other/keep/model.safetensors").is_file());
    }

    #[test]
    fn delete_model_at_keeps_the_org_folder_while_it_still_holds_another_model() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("org/llama/config.json"), 10);
        write(&models.join("org/llama/model.safetensors"), 100);
        write(&models.join("org/qwen/config.json"), 10);
        write(&models.join("org/qwen/model.safetensors"), 100);

        delete_model_at(models, "org/llama").unwrap();

        assert!(!models.join("org/llama").exists());
        assert!(models.join("org/qwen/model.safetensors").is_file());
    }

    #[test]
    fn delete_model_at_refuses_nested_folders_that_are_not_mlx_models() {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path();
        write(&models.join("family/gguf/model-Q4.gguf"), 10);
        write(&models.join("org/config-only/config.json"), 2);
        write(&models.join("top-config-only/config.json"), 2);
        write(&models.join("whisper-mlx/config.json"), 2);
        write(&models.join("whisper-mlx/model.safetensors"), 2);
        write(&models.join("whisper-mlx/sub/config.json"), 2);
        write(&models.join("whisper-mlx/sub/model.safetensors"), 2);

        for id in [
            "family/gguf",
            "org/config-only",
            "top-config-only",
            "whisper-mlx/sub",
        ] {
            assert_eq!(
                delete_model_at(models, id).unwrap_err(),
                "Solo se pueden borrar carpetas MLX de primer o segundo nivel.",
                "{id}"
            );
        }
        assert!(models.join("family/gguf/model-Q4.gguf").is_file());
        assert!(models.join("org/config-only/config.json").is_file());
        assert!(models.join("top-config-only/config.json").is_file());
        assert!(models.join("whisper-mlx/sub/model.safetensors").is_file());
    }
}
