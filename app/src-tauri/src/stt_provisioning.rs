//! Speech-to-text provisioning: engine resolution, the Python runtime and model locations, the
//! status of the currently selected engine, and an install flow that prepares only that engine.

use std::collections::VecDeque;
use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mt_core::{AppSettings, SttEngine, WhisperSize};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};

/// SHA-256 of the pinned `whistle.cact`; must match `install_windows.ps1`.
pub const WHISTLE_SHA256: &str = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb";
/// Pinned Hugging Face URL for `whistle.cact`.
pub const WHISTLE_URL: &str = "https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact";
/// Model file name expected by `transcription_engines.py`.
pub const WHISTLE_FILE: &str = "whistle.cact";

const PROGRESS_EVENT: &str = "stt-provision-progress";
const DEV_MARKER: &str = "meeting_transcriber.py";
/// Shown when the app-managed venv is older than 3.10 or has the wrong architecture.
const INCOMPATIBLE_RUNTIME_DETAIL: &str = "El entorno Python actual es incompatible (< 3.10 o arquitectura incorrecta); se recreará al instalar.";
/// Longest streamed installer line sent to the UI, in characters.
const PROGRESS_MAX_CHARS: usize = 160;
/// Minimum gap between streamed progress events, so the UI stays responsive.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);
/// Trailing output lines quoted when a streamed command fails.
const FAILURE_TAIL_LINES: usize = 5;
const RUNTIME_DIR: &str = "python-runtime";
const MODELS_DIR: &str = "stt-models";
/// Exit 0 only when the selected faster-whisper model is already in the local cache.
const WHISPER_CACHE_CHECK: &str = "import os, sys; os.environ['HF_HUB_OFFLINE'] = '1'; from faster_whisper.utils import download_model; sys.exit(0 if download_model(sys.argv[1], local_files_only=True) else 1)";
/// Downloads and loads the selected faster-whisper model, as the installers do.
const WHISPER_PREFETCH: &str = "import sys; from faster_whisper import WhisperModel; WhisperModel(sys.argv[1], device='cpu', compute_type='int8')";
/// Whistle runs on the Needle 3 engine. This resolves its `libneedle` library without the
/// `needle fetch` CLI, which fails when an unrelated generation wheel is unpublished upstream.
const NEEDLE_ENGINE_PREPARE: &str = r#"
import os, shutil, sys
from needle.agent import fetch
generation = 3
version = fetch.engine_version(generation)
cache = fetch.cache_dir(generation)
lib = os.path.join(cache, fetch.lib_name(generation))

def usable(path):
    return bool(path) and os.path.isfile(path)

if not usable(lib):
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("NEEDLE3_LIB_PATH"),
        os.environ.get("NEEDLE_LIB_PATH"),
        os.path.join(home, ".cache", "cactus-needle", "v3", version, fetch.lib_name(generation)),
    ]
    for candidate in candidates:
        if usable(candidate):
            os.makedirs(cache, exist_ok=True)
            shutil.copy2(candidate, lib)
            print("Motor reutilizado desde " + candidate)
            break

if not usable(lib):
    try:
        fetch.fetch_library(version, cache, generation=generation)
    except Exception as error:
        print("No se pudo obtener el motor de Whistle: " + type(error).__name__ + ": " + str(error))
        sys.exit(3)

print("Motor listo: " + lib)
"#;
/// Succeeds only when the Needle 3 engine library is already available locally.
const NEEDLE_ENGINE_READY: &str = r#"
import os, sys
from needle.agent import fetch
generation = 3
lib = os.path.join(fetch.cache_dir(generation), fetch.lib_name(generation))
override = os.environ.get("NEEDLE3_LIB_PATH") or os.environ.get("NEEDLE_LIB_PATH")
sys.exit(0 if (override and os.path.isfile(override)) or os.path.isfile(lib) else 1)
"#;
/// Loads Whistle for real, which is exactly what the transcriber does at session start.
const NEEDLE_WHISTLE_LOAD: &str = r#"
import sys
from needle import Whistle
model = Whistle(weights=sys.argv[1])
close = getattr(model, "close", None)
if callable(close):
    close()
print("Whistle cargado")
"#;

/// Locations the STT engine needs, resolved once per call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SttPaths {
    /// Folder holding `meeting_transcriber.py` and `requirements.txt`.
    pub script_dir: PathBuf,
    /// Python interpreter that runs the worker and the engine checks.
    pub python_exe: PathBuf,
    /// Folder holding `whistle.cact`; `MT_MODELS_DIR` for the worker.
    pub models_dir: PathBuf,
    /// Isolated virtualenv folder used when no development venv exists.
    pub runtime_dir: PathBuf,
}

/// Status of the currently selected engine and model only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SttModelStatus {
    pub installed: bool,
    pub engine: &'static str,
    pub model_label: String,
    pub detail: String,
    pub python_ready: bool,
    pub models_dir: String,
}

/// Payload of the `stt-provision-progress` event.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProvisionProgress {
    phase: &'static str,
    message: String,
    percent: Option<f64>,
}

/// Maps the requested engine to the one that will run. `auto` picks Whistle on macOS arm64 and
/// Windows x64, and Whisper elsewhere, as `transcription_engines.resolve_engine` does.
pub fn resolve_effective_engine(engine: SttEngine) -> &'static str {
    match engine {
        SttEngine::Whistle => "whistle",
        SttEngine::Whisper => "whisper",
        SttEngine::Auto => auto_engine(std::env::consts::OS, std::env::consts::ARCH),
    }
}

fn auto_engine(os: &str, arch: &str) -> &'static str {
    if matches!((os, arch), ("macos", "aarch64") | ("windows", "x86_64")) {
        "whistle"
    } else {
        "whisper"
    }
}

/// Finds the script folder, the Python interpreter, and the models folder. A development checkout
/// is used when `meeting_transcriber.py` sits in the current directory or its parent.
pub fn resolve_python_and_paths(app: &AppHandle) -> Result<SttPaths, String> {
    let cwd = std::env::current_dir().map_err(io_error)?;
    let data_dir = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("No se pudo localizar la carpeta de datos: {e}"))?;
    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("No se pudo localizar los recursos de la app: {e}"))?;
    Ok(layout(find_dev_root(&cwd), resource_dir, data_dir))
}

fn find_dev_root(cwd: &Path) -> Option<PathBuf> {
    [Some(cwd), cwd.parent()]
        .into_iter()
        .flatten()
        .find(|dir| dir.join(DEV_MARKER).is_file())
        .map(Path::to_path_buf)
}

fn layout(dev_root: Option<PathBuf>, resource_dir: PathBuf, data_dir: PathBuf) -> SttPaths {
    let runtime_dir = data_dir.join(RUNTIME_DIR);
    let dev_python = dev_root
        .as_deref()
        .map(dev_venv_python)
        .filter(|path| path.is_file());
    let python_exe = dev_python.unwrap_or_else(|| runtime_dir.join(python_relative()));
    let dev_models = dev_root
        .as_deref()
        .map(|root| root.join("models"))
        .filter(|models| models.join(WHISTLE_FILE).is_file());
    let models_dir = dev_models.unwrap_or_else(|| data_dir.join(MODELS_DIR));
    let script_dir = dev_root.unwrap_or(resource_dir);
    SttPaths {
        script_dir,
        python_exe,
        models_dir,
        runtime_dir,
    }
}

/// Interpreter path relative to a virtualenv root.
fn python_relative() -> &'static str {
    if cfg!(windows) {
        "Scripts/python.exe"
    } else {
        "bin/python"
    }
}

fn dev_venv_python(repo_root: &Path) -> PathBuf {
    repo_root.join("venv").join(python_relative())
}

/// Builds a child process that does not open a console window on Windows.
pub(crate) fn hidden_command(program: impl AsRef<OsStr>) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut command = Command::new(program);
        command.creation_flags(CREATE_NO_WINDOW);
        command
    }
    #[cfg(not(windows))]
    {
        Command::new(program)
    }
}

/// Stable label used in settings, the `WHISPER_MODEL` env var, and faster-whisper model names.
pub(crate) fn whisper_size_name(size: WhisperSize) -> &'static str {
    match size {
        WhisperSize::Tiny => "tiny",
        WhisperSize::Base => "base",
        WhisperSize::Small => "small",
        WhisperSize::Medium => "medium",
    }
}

/// Reports the status of the selected engine only: Whistle needs its model and the `needle`
/// package; Whisper needs the selected size cached. Never checks the other engine.
#[tauri::command]
pub async fn stt_model_status(
    app: AppHandle,
    settings: AppSettings,
) -> Result<SttModelStatus, String> {
    let paths = resolve_python_and_paths(&app)?;
    blocking(move || Ok(probe_status(&paths, &settings))).await
}

/// Creates the isolated venv when no interpreter exists, installs `requirements.txt`, and then
/// downloads only the selected engine and model.
#[tauri::command]
pub async fn install_selected_stt_model(
    app: AppHandle,
    settings: AppSettings,
) -> Result<SttModelStatus, String> {
    let paths = resolve_python_and_paths(&app)?;
    let engine = resolve_effective_engine(settings.stt_engine);

    // A managed venv that is too old or has the wrong architecture is replaced, never reused.
    let runtime_check = paths.clone();
    let venv_created = blocking(move || {
        if incompatible_managed_runtime(&runtime_check) {
            fs::remove_dir_all(&runtime_check.runtime_dir).map_err(io_error)?;
        }
        Ok(!runtime_check.python_exe.is_file())
    })
    .await?;
    if venv_created {
        emit_progress(&app, "venv", "Creando el entorno Python aislado…", Some(10.0));
        let runtime_dir = paths.runtime_dir.clone();
        blocking(move || create_venv(&runtime_dir)).await?;
    }

    let python = paths.python_exe.clone();
    let speaker_markers = settings.speaker_markers;
    let deps_ready = blocking(move || {
        Ok(engine_import_ready(&python, engine)
            && (!speaker_markers || python_check(&python, "import librosa", &[], &[])))
    })
    .await?;
    if venv_created || !deps_ready {
        emit_progress(&app, "requirements", "Instalando dependencias de Python…", Some(25.0));
        let packages = packages_for_engine(engine, speaker_markers, cfg!(windows));
        let install_app = app.clone();
        let install_paths = paths.clone();
        blocking(move || install_packages(&install_app, &install_paths, &packages)).await?;
    }

    match engine {
        "whistle" => install_whistle(&app, &paths).await?,
        _ => install_whisper(&app, &paths, settings.whisper_model).await?,
    }

    emit_progress(&app, "verify", "Verificando el motor seleccionado…", Some(95.0));
    let check_paths = paths.clone();
    let check_settings = settings.clone();
    let status = blocking(move || Ok(probe_status(&check_paths, &check_settings))).await?;
    if !status.installed {
        return Err(format!("El motor no quedó listo: {}", status.detail));
    }
    emit_progress(&app, "done", "Motor STT listo.", Some(100.0));
    Ok(status)
}

fn probe_status(paths: &SttPaths, settings: &AppSettings) -> SttModelStatus {
    let engine = resolve_effective_engine(settings.stt_engine);
    let incompatible = incompatible_managed_runtime(paths);
    let python_ready = paths.python_exe.is_file() && !incompatible;
    let models_dir = paths.models_dir.display().to_string();

    let mut status = if engine == "whistle" {
        let needle_ready =
            python_ready && engine_import_ready(&paths.python_exe, engine) && needle_engine_ready(&paths.python_exe);
        whistle_status(
            models_dir,
            python_ready,
            whistle_model_sha(&paths.models_dir).as_deref(),
            needle_ready,
        )
    } else {
        let size = whisper_size_name(settings.whisper_model);
        let cached = python_ready
            && python_check(&paths.python_exe, WHISPER_CACHE_CHECK, &[size], &[]);
        whisper_status(models_dir, python_ready, settings.whisper_model, cached)
    };
    if incompatible {
        status.installed = false;
        status.python_ready = false;
        status.detail = INCOMPATIBLE_RUNTIME_DETAIL.to_string();
    }
    status
}

fn whistle_status(
    models_dir: String,
    python_ready: bool,
    model_sha256: Option<&str>,
    needle_ready: bool,
) -> SttModelStatus {
    let model_ready = model_sha256.is_some_and(|hash| hash.eq_ignore_ascii_case(WHISTLE_SHA256));
    let detail = if !python_ready {
        "Falta el entorno Python del motor STT."
    } else if !model_ready {
        "Falta whistle.cact o su SHA-256 no coincide."
    } else if !needle_ready {
        "Falta el motor local de Whistle (libneedle) en el entorno Python."
    } else {
        "Cactus Whistle listo."
    };
    SttModelStatus {
        installed: python_ready && model_ready && needle_ready,
        engine: "whistle",
        model_label: "Cactus Whistle".to_string(),
        detail: detail.to_string(),
        python_ready,
        models_dir,
    }
}

fn whisper_status(
    models_dir: String,
    python_ready: bool,
    size: WhisperSize,
    cached: bool,
) -> SttModelStatus {
    let label = whisper_size_name(size);
    let detail = if !python_ready {
        "Falta el entorno Python del motor STT.".to_string()
    } else if !cached {
        format!("El modelo Whisper {label} no está descargado.")
    } else {
        format!("Whisper {label} listo.")
    };
    SttModelStatus {
        installed: python_ready && cached,
        engine: "whisper",
        model_label: format!("faster-whisper {label}"),
        detail,
        python_ready,
        models_dir,
    }
}

/// SHA-256 of the installed `whistle.cact`, or `None` when the file is missing or unreadable.
fn whistle_model_sha(models_dir: &Path) -> Option<String> {
    let model = models_dir.join(WHISTLE_FILE);
    if !model.is_file() {
        return None;
    }
    sha256_of(&model).ok()
}

fn sha256_of(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(to_hex(&hasher.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Whether the selected engine's Python package imports in `python`.
fn engine_import_ready(python: &Path, engine: &str) -> bool {
    if !python.is_file() {
        return false;
    }
    if engine == "whistle" {
        python_check(
            python,
            "import needle",
            &[],
            &[("NEEDLE_TELEMETRY", "0"), ("DO_NOT_TRACK", "1")],
        )
    } else {
        python_check(python, "import faster_whisper", &[], &[])
    }
}

/// Runs a check command silently and reports whether it exited successfully.
pub(crate) fn python_check(python: &Path, code: &str, args: &[&str], env: &[(&str, &str)]) -> bool {
    hidden_command(python)
        .arg("-c")
        .arg(code)
        .args(args)
        .envs(env.iter().copied())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Whether the app-managed venv (the one under `runtime_dir`) exists but its interpreter is not a
/// Python >= 3.10 with the host architecture. Development venvs are never flagged.
fn incompatible_managed_runtime(paths: &SttPaths) -> bool {
    paths.python_exe.is_file()
        && is_managed_runtime(paths)
        && !crate::python_discovery::is_python_compatible(&paths.python_exe)
}

/// Whether the interpreter lives inside the app-managed runtime folder.
fn is_managed_runtime(paths: &SttPaths) -> bool {
    paths.python_exe.starts_with(&paths.runtime_dir)
}

/// Creates the isolated venv from the best native Python >= 3.10, then ensures it has `pip`.
fn create_venv(runtime_dir: &Path) -> Result<(), String> {
    if let Some(parent) = runtime_dir.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    let interpreter = crate::python_discovery::discover_python()?;
    let mut command = hidden_command(&interpreter.executable);
    command.args(["-m", "venv"]).arg(runtime_dir);
    run_checked(command, "La creación del entorno Python")?;

    let venv_python = runtime_dir.join(python_relative());
    if !python_check(&venv_python, "import pip", &[], &[]) {
        let mut command = hidden_command(&venv_python);
        command.args(["-m", "ensurepip", "--upgrade"]);
        run_checked(command, "La instalación de pip en el entorno Python")?;
    }
    Ok(())
}

/// Packages the selected engine needs, plus `librosa` for speaker markers. The other engine's
/// package is left out, and `soundcard` is added only on Windows.
pub fn packages_for_engine(engine: &str, speaker_markers: bool, windows: bool) -> Vec<&'static str> {
    let mut packages = vec!["numpy>=2.0,<3", "python-dotenv>=1.0,<2", "sounddevice==0.5.3"];
    if windows {
        packages.push("soundcard==0.4.6");
    }
    match engine {
        "whistle" => packages.push("cactus-needle==3.1.0"),
        "whisper" => packages.push("faster-whisper==1.2.1"),
        _ => {}
    }
    if speaker_markers {
        packages.push("librosa==0.11.0");
    }
    packages
}

/// Installs `packages` with `uv` when it is found, otherwise with the venv's pip.
fn install_packages(app: &AppHandle, paths: &SttPaths, packages: &[&str]) -> Result<(), String> {
    let command = match find_uv() {
        Some(uv) => {
            let mut command = hidden_command(uv);
            command
                .args(["pip", "install", "--python"])
                .arg(&paths.python_exe)
                .args(packages);
            command
        }
        None => {
            let mut command = hidden_command(&paths.python_exe);
            command
                .args([
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--progress-bar",
                    "off",
                ])
                .args(packages);
            command
        }
    };
    run_streaming(
        app,
        command,
        "requirements",
        Some(25.0),
        "La instalación de dependencias",
    )
}

/// Locates `uv` on PATH or in the usual install folders.
fn find_uv() -> Option<PathBuf> {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut candidates: Vec<PathBuf> = env::var_os("PATH")
        .map(|path| env::split_paths(&path).map(|dir| dir.join(name)).collect())
        .unwrap_or_default();
    if let Some(home) = env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".local/bin").join(name));
    }
    candidates.push(Path::new("/opt/homebrew/bin").join(name));
    candidates.push(Path::new("/usr/local/bin").join(name));
    candidates.into_iter().find(|candidate| candidate.is_file())
}

async fn install_whistle(app: &AppHandle, paths: &SttPaths) -> Result<(), String> {
    let models_dir = paths.models_dir.clone();
    let present = blocking(move || Ok(whistle_model_sha(&models_dir).is_some_and(|sha| sha.eq_ignore_ascii_case(WHISTLE_SHA256)))).await?;
    if !present {
        download_whistle(app, &paths.models_dir).await?;
    }

    emit_progress(
        app,
        "needle_engine",
        "Preparando el motor local de Whistle…",
        Some(88.0),
    );
    let python = paths.python_exe.clone();
    let weights = paths.models_dir.join(WHISTLE_FILE);
    let app = app.clone();
    blocking(move || prepare_needle_engine(&app, &python, &weights)).await
}

/// Makes the Needle 3 engine available and proves Whistle loads with the downloaded weights.
///
/// The `needle fetch` CLI is deliberately avoided: it downloads every pinned generation wheel and
/// fails with a 404 when one of them is unpublished, even though Whistle itself is installable.
fn prepare_needle_engine(app: &AppHandle, python: &Path, weights: &Path) -> Result<(), String> {
    let prepared = run_streaming(
        app,
        needle_python(python, NEEDLE_ENGINE_PREPARE),
        "needle_engine",
        Some(88.0),
        "Motor de Whistle",
    );

    let mut load = needle_python(python, NEEDLE_WHISTLE_LOAD);
    load.arg(weights);
    match run_streaming(app, load, "needle_verify", Some(94.0), "Verificación de Whistle") {
        Ok(()) => Ok(()),
        Err(load_error) => Err(match prepared {
            Ok(()) => load_error,
            Err(prepare_error) => format!("{prepare_error}. {load_error}"),
        }),
    }
}

/// Hidden, unbuffered Python invocation carrying the telemetry opt-outs the engine expects.
fn needle_python(python: &Path, code: &str) -> Command {
    let mut command = hidden_command(python);
    command
        .arg("-c")
        .arg(code)
        .env("NEEDLE_TELEMETRY", "0")
        .env("DO_NOT_TRACK", "1")
        .env("PYTHONUNBUFFERED", "1");
    command
}

/// Whether the Needle 3 engine library is already present for the selected interpreter.
fn needle_engine_ready(python: &Path) -> bool {
    python_check(
        python,
        NEEDLE_ENGINE_READY,
        &[],
        &[("NEEDLE_TELEMETRY", "0"), ("DO_NOT_TRACK", "1")],
    )
}

async fn install_whisper(app: &AppHandle, paths: &SttPaths, size: WhisperSize) -> Result<(), String> {
    let label = whisper_size_name(size);
    emit_progress(
        app,
        "whisper_prefetch",
        &format!("Descargando Whisper {label}…"),
        Some(55.0),
    );
    let python = paths.python_exe.clone();
    let app = app.clone();
    blocking(move || {
        let mut command = hidden_command(&python);
        command
            .arg("-c")
            .arg(WHISPER_PREFETCH)
            .arg(label)
            .env_remove("HF_HUB_OFFLINE")
            .env("PYTHONUNBUFFERED", "1");
        run_streaming(&app, command, "whisper_prefetch", Some(55.0), "La descarga de Whisper")
    })
    .await
}

/// Downloads `whistle.cact` to a temporary file, verifies its SHA-256, then renames it into place.
async fn download_whistle(app: &AppHandle, models_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(models_dir).map_err(io_error)?;
    let temp = models_dir.join(format!("{WHISTLE_FILE}.{}.download", unique_suffix()));
    let result = fetch_whistle_to(app, &temp)
        .await
        .and_then(|()| fs::rename(&temp, models_dir.join(WHISTLE_FILE)).map_err(io_error));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

async fn fetch_whistle_to(app: &AppHandle, dest: &Path) -> Result<(), String> {
    let mut response = reqwest::get(WHISTLE_URL)
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| format!("No se pudo descargar Cactus Whistle: {e}"))?;
    let total = response.content_length().filter(|total| *total > 0);
    let mut file = File::create(dest).map_err(io_error)?;
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut last_reported: Option<u64> = None;

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| format!("Descarga de Cactus Whistle interrumpida: {e}"))?
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
                    "whistle_download",
                    "Descargando Cactus Whistle…",
                    Some(map_download_percent(whole)),
                );
            }
        }
    }
    file.flush().map_err(io_error)?;
    drop(file);

    let actual = to_hex(&hasher.finalize());
    if !actual.eq_ignore_ascii_case(WHISTLE_SHA256) {
        return Err(format!(
            "La verificación SHA-256 de {WHISTLE_FILE} falló (obtenido {actual})."
        ));
    }
    Ok(())
}

pub(crate) fn run_checked(mut command: Command, label: &str) -> Result<(), String> {
    let output = command
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("{label} no pudo ejecutarse: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{label} falló ({}): {}",
        output.status,
        stderr_tail(&output.stderr)
    ))
}

/// Maps whistle download progress (0..=100 percent of bytes) into the 50..=85 band of the overall
/// install progress.
fn map_download_percent(percent: u64) -> f64 {
    50.0 + (percent.min(100) as f64) * 35.0 / 100.0
}

/// Runs `command` and forwards each output line to the `phase` progress event as
/// `"<prefix>: <line>"`. Fails with the last few output lines.
fn run_streaming(
    app: &AppHandle,
    command: Command,
    phase: &'static str,
    percent: Option<f64>,
    prefix: &str,
) -> Result<(), String> {
    let mut throttle = ProgressThrottle::new(PROGRESS_INTERVAL);
    let result = stream_command(command, prefix, |message| {
        if let Some(message) = throttle.offer(message) {
            emit_progress(app, phase, &message, percent);
        }
    });
    if let Some(message) = throttle.take_pending() {
        emit_progress(app, phase, &message, percent);
    }
    result
}

/// Spawns `command` with piped output and passes each non-blank line to `on_line` as it arrives.
/// Reader threads drain both pipes so the child never blocks on a full buffer.
fn stream_command(
    mut command: Command,
    prefix: &str,
    mut on_line: impl FnMut(String),
) -> Result<(), String> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{prefix} no pudo ejecutarse: {e}"))?;

    let (sender, lines) = mpsc::channel::<String>();
    let mut readers = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        readers.push(spawn_line_reader(stdout, sender.clone()));
    }
    if let Some(stderr) = child.stderr.take() {
        readers.push(spawn_line_reader(stderr, sender.clone()));
    }
    drop(sender);

    let mut tail: VecDeque<String> = VecDeque::with_capacity(FAILURE_TAIL_LINES);
    for raw in lines {
        let Some(line) = progress_line(&raw) else {
            continue;
        };
        if tail.len() == FAILURE_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line.clone());
        on_line(format!("{prefix}: {line}"));
    }
    for reader in readers {
        let _ = reader.join();
    }

    let status = child
        .wait()
        .map_err(|e| format!("{prefix} no pudo completarse: {e}"))?;
    if status.success() {
        return Ok(());
    }
    let tail = Vec::from(tail);
    Err(if tail.is_empty() {
        format!("{prefix} falló ({status})")
    } else {
        format!("{prefix} falló ({status}): {}", tail.join(" | "))
    })
}

/// Reads one pipe line by line (lossy UTF-8) on a background thread until it closes.
fn spawn_line_reader(pipe: impl Read + Send + 'static, sender: mpsc::Sender<String>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut raw = Vec::new();
        loop {
            raw.clear();
            match reader.read_until(b'\n', &mut raw) {
                Ok(0) | Err(_) => break,
                Ok(_) => {
                    if sender.send(String::from_utf8_lossy(&raw).into_owned()).is_err() {
                        break;
                    }
                }
            }
        }
    })
}

/// Trims one output line for the UI, dropping blank lines and capping its length.
fn progress_line(raw: &str) -> Option<String> {
    let line = raw.trim();
    if line.is_empty() {
        return None;
    }
    if line.chars().count() <= PROGRESS_MAX_CHARS {
        return Some(line.to_string());
    }
    let mut shortened: String = line.chars().take(PROGRESS_MAX_CHARS - 1).collect();
    shortened.push('…');
    Some(shortened)
}

/// Lets one progress event through per interval. The newest suppressed line is kept, so the last
/// line of a burst still reaches the UI.
struct ProgressThrottle {
    interval: Duration,
    last_sent: Option<Instant>,
    pending: Option<String>,
}

impl ProgressThrottle {
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_sent: None,
            pending: None,
        }
    }

    fn offer(&mut self, message: String) -> Option<String> {
        let now = Instant::now();
        if self
            .last_sent
            .is_none_or(|sent| now.duration_since(sent) >= self.interval)
        {
            self.last_sent = Some(now);
            self.pending = None;
            Some(message)
        } else {
            self.pending = Some(message);
            None
        }
    }

    fn take_pending(&mut self) -> Option<String> {
        self.pending.take()
    }
}

/// Last few non-empty stderr lines, enough to explain a failed install step.
fn stderr_tail(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !is_decoration(line))
        .map(|line| line.chars().take(300).collect::<String>())
        .collect();
    lines[lines.len().saturating_sub(3)..].join(" | ")
}

/// Python 3.11+ underlines error spans with runs of `~` and `^`; those lines carry no information.
fn is_decoration(line: &str) -> bool {
    line.chars().all(|c| matches!(c, '~' | '^' | ' ' | '|'))
}

fn emit_progress(app: &AppHandle, phase: &'static str, message: &str, percent: Option<f64>) {
    let _ = app.emit(
        PROGRESS_EVENT,
        ProvisionProgress {
            phase,
            message: message.to_string(),
            percent,
        },
    );
}

/// Runs blocking work off the async runtime and flattens the join error into a message.
pub(crate) async fn blocking<T, F>(work: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|e| e.to_string())?
}

pub(crate) fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!("{}-{nanos}", std::process::id())
}

pub(crate) fn io_error(error: io::Error) -> String {
    format!("error de archivo: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_whistle_only_on_macos_arm64_and_windows_x64() {
        assert_eq!(auto_engine("macos", "aarch64"), "whistle");
        assert_eq!(auto_engine("windows", "x86_64"), "whistle");
        assert_eq!(auto_engine("macos", "x86_64"), "whisper");
        assert_eq!(auto_engine("windows", "aarch64"), "whisper");
        assert_eq!(auto_engine("linux", "x86_64"), "whisper");
    }

    #[test]
    fn explicit_engine_choice_is_kept() {
        assert_eq!(resolve_effective_engine(SttEngine::Whisper), "whisper");
        assert_eq!(resolve_effective_engine(SttEngine::Whistle), "whistle");
    }

    #[test]
    fn whistle_status_needs_only_matching_model_and_needle() {
        let ready = whistle_status("m".into(), true, Some(WHISTLE_SHA256), true);
        assert!(ready.installed);
        assert_eq!(ready.engine, "whistle");
        assert_eq!(ready.model_label, "Cactus Whistle");

        let upper = WHISTLE_SHA256.to_uppercase();
        assert!(whistle_status("m".into(), true, Some(upper.as_str()), true).installed);

        assert!(!whistle_status("m".into(), true, Some("deadbeef"), true).installed);
        assert!(!whistle_status("m".into(), true, None, true).installed);
        assert!(!whistle_status("m".into(), true, Some(WHISTLE_SHA256), false).installed);
        let no_python = whistle_status("m".into(), false, Some(WHISTLE_SHA256), false);
        assert!(!no_python.installed);
        assert!(!no_python.python_ready);
    }

    #[test]
    fn whisper_status_checks_only_the_selected_size_cache() {
        let cached = whisper_status("m".into(), true, WhisperSize::Medium, true);
        assert!(cached.installed);
        assert_eq!(cached.engine, "whisper");
        assert_eq!(cached.model_label, "faster-whisper medium");
        assert_eq!(cached.detail, "Whisper medium listo.");

        let missing = whisper_status("m".into(), true, WhisperSize::Tiny, false);
        assert!(!missing.installed);
        assert!(missing.detail.contains("tiny"));
        assert!(!whisper_status("m".into(), false, WhisperSize::Tiny, true).installed);
    }

    fn paths_with(runtime_dir: &Path, python_exe: PathBuf) -> SttPaths {
        SttPaths {
            script_dir: runtime_dir.to_path_buf(),
            python_exe,
            models_dir: runtime_dir.join(MODELS_DIR),
            runtime_dir: runtime_dir.to_path_buf(),
        }
    }

    #[test]
    fn probe_status_reports_a_missing_python_without_running_one() {
        let scratch = tempfile::tempdir().unwrap();
        let runtime_dir = scratch.path().join(RUNTIME_DIR);
        let paths = paths_with(&runtime_dir, runtime_dir.join(python_relative()));
        let settings = AppSettings {
            stt_engine: SttEngine::Whisper,
            whisper_model: WhisperSize::Tiny,
            ..AppSettings::default()
        };

        let status = probe_status(&paths, &settings);

        assert!(!status.installed);
        assert!(!status.python_ready);
        assert_eq!(status.detail, "Falta el entorno Python del motor STT.");
    }

    #[test]
    fn managed_runtime_is_identified_by_its_folder() {
        let runtime_dir = PathBuf::from("/data/python-runtime");
        let managed = paths_with(&runtime_dir, runtime_dir.join("bin/python"));
        assert!(is_managed_runtime(&managed));

        let dev = paths_with(&runtime_dir, PathBuf::from("/dev/venv/bin/python"));
        assert!(!is_managed_runtime(&dev));
        assert!(!incompatible_managed_runtime(&dev));
    }

    #[test]
    fn incompatible_managed_runtime_is_flagged_and_not_reported_ready() {
        let scratch = tempfile::tempdir().unwrap();
        let runtime_dir = scratch.path().join(RUNTIME_DIR);
        let python_exe = runtime_dir.join(python_relative());
        fs::create_dir_all(python_exe.parent().unwrap()).unwrap();
        fs::write(&python_exe, "not an interpreter").unwrap();
        let paths = paths_with(&runtime_dir, python_exe);
        let settings = AppSettings {
            stt_engine: SttEngine::Whisper,
            ..AppSettings::default()
        };

        assert!(incompatible_managed_runtime(&paths));
        let status = probe_status(&paths, &settings);

        assert!(!status.installed);
        assert!(!status.python_ready);
        assert_eq!(status.detail, INCOMPATIBLE_RUNTIME_DETAIL);
    }

    #[test]
    fn packages_for_whistle_install_needle_but_not_faster_whisper() {
        assert_eq!(
            packages_for_engine("whistle", false, false),
            vec![
                "numpy>=2.0,<3",
                "python-dotenv>=1.0,<2",
                "sounddevice==0.5.3",
                "cactus-needle==3.1.0",
            ]
        );
    }

    #[test]
    fn packages_for_whisper_install_faster_whisper_but_not_needle() {
        assert_eq!(
            packages_for_engine("whisper", false, false),
            vec![
                "numpy>=2.0,<3",
                "python-dotenv>=1.0,<2",
                "sounddevice==0.5.3",
                "faster-whisper==1.2.1",
            ]
        );
    }

    #[test]
    fn needle_engine_setup_never_calls_the_broken_cli() {
        for snippet in [NEEDLE_ENGINE_PREPARE, NEEDLE_ENGINE_READY, NEEDLE_WHISTLE_LOAD] {
            assert!(!snippet.contains("subprocess"), "no subprocess in the engine snippets");
            assert!(!snippet.contains("needle fetch"), "the needle fetch CLI must not be used");
        }
        let prepare = NEEDLE_ENGINE_PREPARE;
        let copy_at = prepare.find("shutil.copy2").expect("local reuse is part of the setup");
        let fetch_at = prepare.find("fetch.fetch_library").expect("hub fallback stays available");
        assert!(copy_at < fetch_at, "a local engine library is reused before downloading");
    }

    #[test]
    fn error_tail_drops_python_decoration_lines() {
        let bytes = b"Traceback (most recent call last):\n  File \"x.py\", line 1\n    raise X\n~~~~~~~^^^^\nValueError: boom\n";
        let tail = stderr_tail(bytes);
        assert!(tail.contains("ValueError: boom"), "{tail}");
        assert!(!tail.contains("~~~"), "{tail}");
    }

    #[test]
    fn packages_add_librosa_only_for_speaker_markers() {
        let markers = packages_for_engine("whistle", true, false);
        assert_eq!(markers.last(), Some(&"librosa==0.11.0"));
        assert!(!packages_for_engine("whistle", false, false).contains(&"librosa==0.11.0"));
    }

    #[test]
    fn packages_add_soundcard_only_on_windows() {
        let windows = packages_for_engine("whisper", false, true);
        assert_eq!(
            windows,
            vec![
                "numpy>=2.0,<3",
                "python-dotenv>=1.0,<2",
                "sounddevice==0.5.3",
                "soundcard==0.4.6",
                "faster-whisper==1.2.1",
            ]
        );
        assert!(!packages_for_engine("whisper", false, false).contains(&"soundcard==0.4.6"));
    }

    #[test]
    fn download_percent_maps_into_the_whistle_download_band() {
        assert_eq!(map_download_percent(0), 50.0);
        assert_eq!(map_download_percent(50), 67.5);
        assert_eq!(map_download_percent(100), 85.0);
        assert_eq!(map_download_percent(250), 85.0);
    }

    #[test]
    fn progress_lines_are_trimmed_capped_and_blank_lines_dropped() {
        assert_eq!(
            progress_line("  Collecting numpy \r\n"),
            Some("Collecting numpy".to_string())
        );
        assert_eq!(progress_line(" \t \n"), None);

        let capped = progress_line(&"x".repeat(400)).unwrap();
        assert_eq!(capped.chars().count(), PROGRESS_MAX_CHARS);
        assert!(capped.ends_with('…'));
    }

    #[test]
    fn progress_throttle_sends_first_line_then_only_the_newest_pending_one() {
        let mut throttle = ProgressThrottle::new(Duration::from_secs(60));
        assert_eq!(throttle.offer("a".into()), Some("a".into()));
        assert_eq!(throttle.offer("b".into()), None);
        assert_eq!(throttle.offer("c".into()), None);
        assert_eq!(throttle.take_pending(), Some("c".into()));
        assert_eq!(throttle.take_pending(), None);
    }

    #[cfg(unix)]
    #[test]
    fn run_streaming_forwards_stdout_and_stderr_with_the_prefix() {
        let mut command = Command::new("sh");
        command.args(["-c", "echo first; echo second >&2"]);
        let mut messages = Vec::new();

        let result = stream_command(command, "Prueba", |message| messages.push(message));

        assert!(result.is_ok());
        assert!(messages.contains(&"Prueba: first".to_string()));
        assert!(messages.contains(&"Prueba: second".to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn run_streaming_always_flushes_the_final_line() {
        let mut command = Command::new("sh");
        command.args(["-c", r"printf 'a\nb'"]);
        let mut messages = Vec::new();

        stream_command(command, "Prueba", |message| messages.push(message)).unwrap();

        assert_eq!(messages.last().map(String::as_str), Some("Prueba: b"));
    }

    #[cfg(unix)]
    #[test]
    fn run_streaming_failure_quotes_only_the_last_five_lines() {
        let mut command = Command::new("sh");
        command.args(["-c", "for i in 1 2 3 4 5 6 7; do echo line$i; done; exit 3"]);

        let error = stream_command(command, "Instalando", |_| {}).unwrap_err();

        assert!(error.starts_with("Instalando falló ("), "{error}");
        assert!(error.contains("exit status: 3"), "{error}");
        assert!(error.ends_with("line3 | line4 | line5 | line6 | line7"), "{error}");
        assert!(!error.contains("line2"), "{error}");
    }

    #[test]
    fn run_streaming_reports_a_program_that_cannot_start() {
        let command = Command::new("meeting-transcriber-missing-program-for-tests");

        let error = stream_command(command, "Prueba", |_| {}).unwrap_err();

        assert!(error.starts_with("Prueba no pudo ejecutarse"), "{error}");
    }
}
