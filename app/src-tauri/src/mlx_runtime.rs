//! Managed `mlx-lm` runtime for the local MLX report model.
//!
//! Layout under the runtime root:
//!
//! ```text
//! <root>/bin/uv                 managed uv (SHA-256 verified)
//! <root>/downloads/             transient downloads
//! <root>/versions/<ver>/        venv: bin/python3 + version.txt
//! <root>/active                 marker file holding the active version
//! <root>/server.log             stderr of the last `mlx_lm` server run
//! ```
//!
//! Pure core: every function takes the runtime root. Long operations are blocking and callers run
//! them inside `spawn_blocking`. Progress is reported through `&dyn Fn(&str)`. This module has no
//! Tauri dependency.

use std::cmp::Ordering;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::python_discovery::{MLX_PYTHON_MINOR, discover_python_at_least};

/// File name of the server log inside the runtime root.
pub const SERVER_LOG: &str = "server.log";

const PACKAGE: &str = "mlx-lm";
const VERSIONS_DIR: &str = "versions";
const DOWNLOADS_DIR: &str = "downloads";
const STAGING_DIR: &str = ".staging";
const ACTIVE_MARKER: &str = "active";
const VERSION_FILE: &str = "version.txt";
/// Single venv directory of the pre-versioned layout (`mlx/versions/current`).
const LEGACY_VERSION_DIR: &str = "current";
const VERSION_PROBE: &str = "import mlx_lm; print(mlx_lm.__version__)";
/// `mlx_lm.server` is deprecated; `python -m mlx_lm server` is the supported entry point.
const SERVER_ARGS: [&str; 3] = ["-m", "mlx_lm", "server"];
const UV_ASSET_NAME: &str = "uv-aarch64-apple-darwin.tar.gz";
const UV_DOWNLOAD_URL: &str =
    "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz";
const PYPI_URL: &str = "https://pypi.org/pypi/mlx-lm/json";
const PYPI_TIMEOUT: Duration = Duration::from_secs(10);
const LOG_MAX_BYTES: usize = 64 * 1024;
const LOG_LINE_MAX_CHARS: usize = 300;

const UNSUPPORTED: &str = "El runtime MLX solo está disponible en macOS con Apple Silicon";
const NOT_INSTALLED: &str = "El runtime MLX no está instalado";
const CHECKSUM_MISMATCH: &str = "El SHA-256 de uv no coincide; instalación cancelada";

/// Installed state of the runtime, read from disk (offline).
///
/// `latest` and `update_available` are only filled by [`check_update`], which needs the network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MlxStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub python: Option<PathBuf>,
    pub latest: Option<String>,
    pub update_available: bool,
}

/// The `uv`/`python` steps, injectable so tests never run real tools.
trait Installer {
    /// Creates a venv at `dir` and installs `mlx-lm`, pinned to `version` when given.
    fn build_venv(
        &self,
        dir: &Path,
        version: Option<&str>,
        progress: &dyn Fn(&str),
    ) -> Result<(), String>;

    /// Version reported by the `mlx-lm` package visible to `python`.
    fn package_version(&self, python: &Path) -> Result<String, String>;
}

/// Real installer backed by the managed `uv`, using a discovered Python >= 3.11 as the base.
struct UvInstaller {
    uv: PathBuf,
    base_python: PathBuf,
}

impl Installer for UvInstaller {
    fn build_venv(
        &self,
        dir: &Path,
        version: Option<&str>,
        progress: &dyn Fn(&str),
    ) -> Result<(), String> {
        progress("Creando el entorno virtual de MLX…");
        // `--python` keeps uv from picking an older system interpreter (the legacy 3.9 venv).
        run_command(
            Command::new(&self.uv)
                .arg("venv")
                .arg("--python")
                .arg(&self.base_python)
                .arg(dir),
            "uv venv",
        )?;
        progress(&match version {
            Some(version) => format!("Instalando {PACKAGE}=={version}…"),
            None => format!("Instalando la última versión de {PACKAGE}…"),
        });
        let spec = match version {
            Some(version) => format!("{PACKAGE}=={version}"),
            None => PACKAGE.to_string(),
        };
        run_command(
            Command::new(&self.uv)
                .args(["pip", "install", "--python"])
                .arg(venv_python(dir))
                .arg(spec),
            "uv pip install",
        )?;
        Ok(())
    }

    fn package_version(&self, python: &Path) -> Result<String, String> {
        let output = run_command(
            Command::new(python).args(["-c", VERSION_PROBE]),
            "la consulta de versión de mlx-lm",
        )?;
        let version = output.trim().to_string();
        if version.is_empty() {
            return Err(format!(
                "{PACKAGE} no reportó su versión; instalación cancelada"
            ));
        }
        validate_version(&version)?;
        Ok(version)
    }
}

// ---------------------------------------------------------------------------
// Layout and validation
// ---------------------------------------------------------------------------

fn ensure_supported() -> Result<(), String> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(())
    } else {
        Err(UNSUPPORTED.to_string())
    }
}

/// Rejects empty versions and anything that could escape `versions/`.
pub fn validate_version(version: &str) -> Result<(), String> {
    if version.is_empty()
        || version.starts_with('.')
        || version.contains('/')
        || version.contains('\\')
        || version.contains("..")
        || version.chars().any(char::is_whitespace)
    {
        return Err(format!("La versión solicitada no es válida: {version:?}"));
    }
    Ok(())
}

fn io_error(action: &str, error: impl std::fmt::Display) -> String {
    format!("{action}: {error}")
}

fn uv_binary_path(root: &Path) -> PathBuf {
    root.join("bin").join("uv")
}

fn downloads_dir(root: &Path) -> PathBuf {
    root.join(DOWNLOADS_DIR)
}

fn version_dir(root: &Path, version: &str) -> PathBuf {
    root.join(VERSIONS_DIR).join(version)
}

fn venv_python(venv: &Path) -> PathBuf {
    venv.join("bin").join("python3")
}

/// A complete install is a real directory (never a symlink) inside `versions/` with a venv python
/// and a `version.txt`. The legacy `versions/current` has no `version.txt`, so it never qualifies.
fn is_complete_install(root: &Path, version: &str) -> bool {
    if validate_version(version).is_err() {
        return false;
    }
    let dir = version_dir(root, version);
    let is_real_dir = fs::symlink_metadata(&dir)
        .map(|metadata| metadata.file_type().is_dir())
        .unwrap_or(false);
    is_real_dir && venv_python(&dir).is_file() && dir.join(VERSION_FILE).is_file()
}

/// Version named by the `active` marker, only when it names a complete install. A hand-edited
/// marker (traversal, absolute path, legacy `versions/current`) therefore reads as not installed.
fn active_version(root: &Path) -> Option<String> {
    let marker = fs::read_to_string(root.join(ACTIVE_MARKER)).ok()?;
    let version = marker.trim().to_string();
    is_complete_install(root, &version).then_some(version)
}

/// Makes `version` the active one. The marker is replaced atomically, so a failure leaves the
/// previous marker untouched.
fn activate(root: &Path, version: &str) -> Result<(), String> {
    validate_version(version)?;
    if !is_complete_install(root, version) {
        return Err(format!("La versión {version} de MLX no está instalada"));
    }
    let next = root.join(format!("{ACTIVE_MARKER}.next"));
    fs::write(&next, format!("{version}\n"))
        .map_err(|error| io_error("No se pudo preparar la activación de MLX", error))?;
    fs::rename(&next, root.join(ACTIVE_MARKER)).map_err(|error| {
        let _ = fs::remove_file(&next);
        io_error(
            &format!("No se pudo activar la versión {version} de MLX"),
            error,
        )
    })
}

// ---------------------------------------------------------------------------
// Status and server command
// ---------------------------------------------------------------------------

/// Installed state from disk only. Never touches the network.
pub fn status(root: &Path) -> MlxStatus {
    let version = active_version(root);
    MlxStatus {
        installed: version.is_some(),
        python: version
            .as_deref()
            .map(|version| venv_python(&version_dir(root, version))),
        version,
        latest: None,
        update_available: false,
    }
}

/// Status plus the latest PyPI version. Needs the network (10 s timeout).
pub fn check_update(root: &Path) -> Result<MlxStatus, String> {
    let latest = latest_version()?;
    Ok(with_latest(status(root), latest))
}

fn with_latest(mut status: MlxStatus, latest: String) -> MlxStatus {
    status.update_available = status
        .version
        .as_deref()
        .is_some_and(|current| is_newer(current, &latest));
    status.latest = Some(latest);
    status
}

/// Interpreter and arguments that start `mlx_lm` server; the caller appends host/port/model flags.
pub fn server_command(root: &Path) -> Result<(PathBuf, Vec<String>), String> {
    let python = status(root)
        .python
        .ok_or_else(|| NOT_INSTALLED.to_string())?;
    Ok((
        python,
        SERVER_ARGS.iter().map(|arg| arg.to_string()).collect(),
    ))
}

// ---------------------------------------------------------------------------
// Server log
// ---------------------------------------------------------------------------

pub fn server_log_path(root: &Path) -> PathBuf {
    root.join(SERVER_LOG)
}

/// Last `lines` non-empty, non-decoration lines of the server log, each capped at 300 chars.
/// Decoration is a traceback caret line such as `~~~~^^^^`. A missing log yields an empty string.
pub fn log_tail(root: &Path, lines: usize) -> String {
    let Ok(text) = read_log_tail(&server_log_path(root), LOG_MAX_BYTES) else {
        return String::new();
    };
    let kept: Vec<String> = text
        .lines()
        .filter(|line| !is_blank_or_decoration(line))
        .map(|line| {
            line.trim_end()
                .chars()
                .take(LOG_LINE_MAX_CHARS)
                .collect::<String>()
        })
        .collect();
    let start = kept.len().saturating_sub(lines);
    kept[start..].join("\n")
}

fn is_blank_or_decoration(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.chars().all(|c| matches!(c, '~' | '^'))
}

/// Replaces the server log with the last ~64 KiB of `stderr`, cut at a line boundary.
/// Currently the pipeline reads the log instead of writing it; the writer stays for callers that
/// capture `stderr` themselves (for example, a future Windows helper).
#[cfg_attr(not(test), allow(dead_code))]
pub fn write_server_log(root: &Path, stderr: &[u8]) -> std::io::Result<()> {
    fs::create_dir_all(root)?;
    let tail = if stderr.len() > LOG_MAX_BYTES {
        align_to_line(&stderr[stderr.len() - LOG_MAX_BYTES..])
    } else {
        stderr
    };
    fs::write(server_log_path(root), tail)
}

/// Reads at most the last `max` bytes of `path`, cut at a line boundary.
fn read_log_tail(path: &Path, max: usize) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let start = len.saturating_sub(max as u64);
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let kept = if start > 0 {
        align_to_line(&bytes)
    } else {
        &bytes[..]
    };
    Ok(String::from_utf8_lossy(kept).into_owned())
}

/// Drops a partial first line of a tail. Without a newline the whole slice is one line.
fn align_to_line(bytes: &[u8]) -> &[u8] {
    match bytes.iter().position(|&byte| byte == b'\n') {
        Some(position) => &bytes[position + 1..],
        None => bytes,
    }
}

// ---------------------------------------------------------------------------
// uv bootstrap
// ---------------------------------------------------------------------------

type Fetch<'a> = &'a dyn Fn(&str, &Path) -> Result<(), String>;

/// Returns the managed `uv`, downloading it from the official astral release when missing.
/// Fails closed: without a valid published SHA-256 nothing is installed.
pub fn ensure_uv(root: &Path, progress: &dyn Fn(&str)) -> Result<PathBuf, String> {
    ensure_supported()?;
    ensure_uv_with(root, progress, &curl_to_file)
}

fn ensure_uv_with(
    root: &Path,
    progress: &dyn Fn(&str),
    fetch: Fetch<'_>,
) -> Result<PathBuf, String> {
    let uv = uv_binary_path(root);
    if uv.is_file() {
        return Ok(uv);
    }
    fs::create_dir_all(root.join("bin"))
        .map_err(|error| io_error("No se pudo crear bin", error))?;
    fs::create_dir_all(downloads_dir(root))
        .map_err(|error| io_error("No se pudo crear downloads", error))?;

    let result = download_and_install_uv(root, progress, fetch);
    // Always leave downloads clean; a failed install must not leave a binary behind.
    let _ = fs::remove_file(downloads_dir(root).join(UV_ASSET_NAME));
    let _ = fs::remove_dir_all(downloads_dir(root).join("uv-extract"));
    let _ = fs::remove_file(root.join("bin").join("uv.partial"));
    result.map(|()| uv)
}

fn download_and_install_uv(
    root: &Path,
    progress: &dyn Fn(&str),
    fetch: Fetch<'_>,
) -> Result<(), String> {
    let archive = downloads_dir(root).join(UV_ASSET_NAME);
    progress("Descargando uv (asset oficial macOS arm64)…");
    let _ = fs::remove_file(&archive);
    fetch(UV_DOWNLOAD_URL, &archive)?;

    progress("Verificando SHA-256 de uv…");
    verify_archive(root, &archive, fetch)?;

    let extraction = downloads_dir(root).join("uv-extract");
    let _ = fs::remove_dir_all(&extraction);
    fs::create_dir_all(&extraction)
        .map_err(|error| io_error("No se pudo crear la extracción de uv", error))?;
    progress("Extrayendo uv…");
    run_command(
        Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("-C")
            .arg(&extraction),
        "tar",
    )?;

    let extracted = find_uv_binary(&extraction)?;
    let partial = root.join("bin").join("uv.partial");
    let _ = fs::remove_file(&partial);
    fs::copy(&extracted, &partial).map_err(|error| io_error("No se pudo copiar uv", error))?;
    make_executable(&partial)?;

    progress("Validando que uv se pueda ejecutar…");
    run_command(Command::new(&partial).arg("--version"), "uv --version")?;
    fs::rename(&partial, uv_binary_path(root))
        .map_err(|error| io_error("No se pudo instalar uv", error))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .map_err(|error| io_error("No se pudo marcar uv como ejecutable", error))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn verify_archive(root: &Path, archive: &Path, fetch: Fetch<'_>) -> Result<(), String> {
    let digest_file = downloads_dir(root).join(format!("{UV_ASSET_NAME}.sha256"));
    let digest_url = format!("{UV_DOWNLOAD_URL}.sha256");
    let _ = fs::remove_file(&digest_file);
    let content = fetch(&digest_url, &digest_file).and_then(|()| {
        fs::read_to_string(&digest_file)
            .map_err(|error| io_error("No se pudo leer el SHA-256", error))
    });
    let _ = fs::remove_file(&digest_file);
    let content = content.map_err(|detail| checksum_unavailable(&detail))?;
    let expected = parse_digest(&content)
        .ok_or_else(|| checksum_unavailable("el contenido no es un digest SHA-256"))?;
    if sha256_file(archive)? != expected {
        return Err(CHECKSUM_MISMATCH.to_string());
    }
    Ok(())
}

fn checksum_unavailable(detail: &str) -> String {
    format!(
        "No se pudo obtener un SHA-256 publicado válido para uv ({detail}); la descarga se descartó para no ejecutar un binario sin verificar"
    )
}

/// First token of a `shasum`-style file, validated as 64 hex characters.
fn parse_digest(content: &str) -> Option<String> {
    let digest = content.split_whitespace().next()?.to_ascii_lowercase();
    (digest.len() == 64 && digest.chars().all(|c| c.is_ascii_hexdigit())).then_some(digest)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|error| io_error("No se pudo verificar SHA-256", error))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|error| io_error("No se pudo verificar SHA-256", error))?;
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn curl_to_file(url: &str, destination: &Path) -> Result<(), String> {
    let output = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--output",
        ])
        .arg(destination)
        .arg(url)
        .output()
        .map_err(|error| format!("La descarga falló: no se pudo ejecutar curl: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "La descarga falló: {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

fn find_uv_binary(root: &Path) -> Result<PathBuf, String> {
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| io_error("No se pudo leer el archivo de uv extraído", error))?;
        for entry in entries {
            let path = entry
                .map_err(|error| io_error("No se pudo leer un archivo de uv", error))?
                .path();
            if path.is_dir() {
                pending.push(path);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("uv") {
                return Ok(path);
            }
        }
    }
    Err("El archivo descargado no contiene el binario uv".to_string())
}

fn run_command(command: &mut Command, action: &str) -> Result<String, String> {
    let output = command
        .output()
        .map_err(|error| format!("{action} falló: no se pudo ejecutar: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{action} falló: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Installer for the real machine: discovers a Python >= 3.11 and ensures the managed `uv`.
fn uv_installer(root: &Path, progress: &dyn Fn(&str)) -> Result<UvInstaller, String> {
    ensure_supported()?;
    let base_python = discover_python_at_least(MLX_PYTHON_MINOR)?.executable;
    let uv = ensure_uv(root, progress)?;
    Ok(UvInstaller { uv, base_python })
}

// ---------------------------------------------------------------------------
// Install and update
// ---------------------------------------------------------------------------

/// Installs the latest unpinned `mlx-lm` into a versioned venv and activates it. Blocking.
pub fn install_latest(root: &Path, progress: &dyn Fn(&str)) -> Result<MlxStatus, String> {
    let installer = uv_installer(root, progress)?;
    install_latest_with(root, None, progress, &installer)
}

/// Installs the current PyPI latest side by side and activates it only on success, so the previous
/// version stays installed for rollback. Does nothing when the active version is already current.
pub fn update(root: &Path, progress: &dyn Fn(&str)) -> Result<MlxStatus, String> {
    ensure_supported()?;
    let latest = latest_version()?;
    update_with(root, &latest, progress, || uv_installer(root, progress))
}

fn update_with<I: Installer>(
    root: &Path,
    latest: &str,
    progress: &dyn Fn(&str),
    make_installer: impl FnOnce() -> Result<I, String>,
) -> Result<MlxStatus, String> {
    if !update_needed(root, latest) {
        return Ok(status(root));
    }
    let installer = make_installer()?;
    install_latest_with(root, Some(latest), progress, &installer)
}

fn update_needed(root: &Path, latest: &str) -> bool {
    match active_version(root) {
        None => true,
        Some(current) => is_newer(&current, latest),
    }
}

fn install_latest_with(
    root: &Path,
    requested: Option<&str>,
    progress: &dyn Fn(&str),
    installer: &dyn Installer,
) -> Result<MlxStatus, String> {
    if let Some(requested) = requested {
        validate_version(requested)?;
    }
    let versions = root.join(VERSIONS_DIR);
    fs::create_dir_all(&versions).map_err(|error| io_error("No se pudo crear versions", error))?;

    // The final directory is named after the installed version, which is only known once the
    // package exists: probe in staging, then rebuild for the final path. Venvs are not relocatable,
    // so the staging venv is never moved.
    let staging = versions.join(STAGING_DIR);
    let _ = fs::remove_dir_all(&staging);
    let probed = installer
        .build_venv(&staging, requested, progress)
        .and_then(|()| installer.package_version(&venv_python(&staging)));
    let _ = fs::remove_dir_all(&staging);
    let installed = probed?;
    validate_version(&installed)?;
    if let Some(requested) = requested
        && requested != installed
    {
        return Err(format!(
            "Se esperaba {PACKAGE} {requested} pero se instaló {installed}"
        ));
    }

    if !is_complete_install(root, &installed) {
        build_final_venv(root, &installed, progress, installer)?;
    }
    progress("Activando la versión instalada…");
    activate(root, &installed)?;
    remove_legacy_layout(root, &installed);
    Ok(status(root))
}

fn build_final_venv(
    root: &Path,
    version: &str,
    progress: &dyn Fn(&str),
    installer: &dyn Installer,
) -> Result<(), String> {
    let target = version_dir(root, version);
    // Only interrupted leftovers can be here: a complete install is never rebuilt.
    let _ = fs::remove_dir_all(&target);
    let built = installer
        .build_venv(&target, Some(version), progress)
        .and_then(|()| {
            let reported = installer.package_version(&venv_python(&target))?;
            if reported != version {
                return Err(format!(
                    "Se esperaba {PACKAGE} {version} pero se instaló {reported}"
                ));
            }
            fs::write(target.join(VERSION_FILE), format!("{version}\n"))
                .map_err(|error| io_error("No se pudo guardar version.txt", error))
        });
    if built.is_err() {
        let _ = fs::remove_dir_all(&target);
    }
    built
}

/// Removes the pre-versioned `versions/current` venv once `active` names a new install. Called only
/// after activation, so a failed install keeps the legacy runtime usable. Removal failures are not
/// fatal: the leftover is never reported as installed.
fn remove_legacy_layout(root: &Path, active: &str) {
    if active == LEGACY_VERSION_DIR {
        return;
    }
    let legacy = version_dir(root, LEGACY_VERSION_DIR);
    let is_real_dir = fs::symlink_metadata(&legacy)
        .map(|metadata| metadata.file_type().is_dir())
        .unwrap_or(false);
    if is_real_dir {
        let _ = fs::remove_dir_all(&legacy);
    }
}

// ---------------------------------------------------------------------------
// Latest version (PyPI)
// ---------------------------------------------------------------------------

/// Latest `mlx-lm` version published on PyPI. Blocking (10 s timeout).
pub fn latest_version() -> Result<String, String> {
    fetch_latest_version(PYPI_URL)
}

fn fetch_latest_version(url: &str) -> Result<String, String> {
    let url = url.to_string();
    // A dedicated thread with its own current-thread runtime works from any caller context,
    // including inside a tokio runtime.
    let handle = std::thread::spawn(move || -> Result<String, String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| network_error(&error.to_string()))?;
        runtime.block_on(async move {
            let client = reqwest::Client::builder()
                .timeout(PYPI_TIMEOUT)
                .build()
                .map_err(|error| network_error(&error.to_string()))?;
            let response = client
                .get(&url)
                .send()
                .await
                .and_then(|response| response.error_for_status())
                .map_err(|error| network_error(&error.to_string()))?;
            let body = response
                .text()
                .await
                .map_err(|error| network_error(&error.to_string()))?;
            parse_latest_version(&body)
        })
    });
    handle
        .join()
        .unwrap_or_else(|_| Err(network_error("la consulta se interrumpió")))
}

fn network_error(detail: &str) -> String {
    format!("No se pudo consultar la última versión de {PACKAGE}: {detail}")
}

/// Extracts `info.version` from the PyPI JSON document.
fn parse_latest_version(json: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|error| format!("Respuesta no válida de PyPI para {PACKAGE}: {error}"))?;
    let version = value
        .pointer("/info/version")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("Respuesta no válida de PyPI para {PACKAGE}: falta info.version"))?;
    validate_version(version)
        .map_err(|_| format!("Respuesta no válida de PyPI para {PACKAGE}: versión no válida"))?;
    Ok(version.to_string())
}

// ---------------------------------------------------------------------------
// Version comparison
// ---------------------------------------------------------------------------

/// Splits `0.21.0rc1` into numeric components `[0, 21, 0]` and a pre-release suffix `Some("rc1")`.
/// Never panics on malformed input.
fn parse_version(version: &str) -> (Vec<u64>, Option<String>) {
    let split = version
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(version.len());
    let (numeric, suffix) = version.split_at(split);
    let numbers = numeric
        .split('.')
        .filter(|part| !part.is_empty())
        .map(|part| part.parse::<u64>().unwrap_or(u64::MAX))
        .collect();
    let suffix = suffix.trim_matches(|c: char| c == '.' || c == '-' || c == '_');
    (
        numbers,
        (!suffix.is_empty()).then(|| suffix.to_ascii_lowercase()),
    )
}

/// Numeric-aware ordering: `0.9 < 0.10`, `0.21 == 0.21.0`, and a final release is newer than any of
/// its pre-releases. Pre-releases compare lexically among themselves.
fn compare_versions(a: &str, b: &str) -> Ordering {
    let (a_numbers, a_pre) = parse_version(a);
    let (b_numbers, b_pre) = parse_version(b);
    for index in 0..a_numbers.len().max(b_numbers.len()) {
        let left = a_numbers.get(index).copied().unwrap_or(0);
        let right = b_numbers.get(index).copied().unwrap_or(0);
        match left.cmp(&right) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    match (a_pre, b_pre) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => left.cmp(&right),
    }
}

/// True when `latest` is strictly newer than `current`.
pub fn is_newer(current: &str, latest: &str) -> bool {
    compare_versions(current, latest) == Ordering::Less
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn silent(_: &str) {}

    /// Fake installer: records each build and materialises a venv python, no uv involved.
    struct FakeInstaller {
        reported: String,
        /// 1-based index of the `build_venv` call that fails, if any.
        fail_build_at: Option<usize>,
        builds: RefCell<Vec<Option<String>>>,
    }

    impl FakeInstaller {
        fn new(reported: &str) -> Self {
            Self {
                reported: reported.to_string(),
                fail_build_at: None,
                builds: RefCell::new(Vec::new()),
            }
        }

        fn failing_at(reported: &str, call: usize) -> Self {
            Self {
                fail_build_at: Some(call),
                ..Self::new(reported)
            }
        }

        fn builds(&self) -> Vec<Option<String>> {
            self.builds.borrow().clone()
        }
    }

    impl Installer for FakeInstaller {
        fn build_venv(
            &self,
            dir: &Path,
            version: Option<&str>,
            _progress: &dyn Fn(&str),
        ) -> Result<(), String> {
            self.builds.borrow_mut().push(version.map(str::to_string));
            let call = self.builds.borrow().len();
            fs::create_dir_all(dir.join("bin")).map_err(|error| error.to_string())?;
            fs::write(venv_python(dir), b"#!/bin/sh\n").map_err(|error| error.to_string())?;
            if self.fail_build_at == Some(call) {
                return Err("uv pip install falló: boom".to_string());
            }
            Ok(())
        }

        fn package_version(&self, _python: &Path) -> Result<String, String> {
            Ok(self.reported.clone())
        }
    }

    /// A complete install as the real installer leaves it.
    fn fake_install(root: &Path, version: &str) {
        let dir = version_dir(root, version);
        fs::create_dir_all(dir.join("bin")).expect("bin dir");
        fs::write(venv_python(&dir), b"#!/bin/sh\n").expect("python");
        fs::write(dir.join(VERSION_FILE), format!("{version}\n")).expect("version.txt");
    }

    /// The pre-versioned layout: `versions/current` with no `version.txt`, active by relative name.
    fn legacy_layout(root: &Path) {
        let legacy = version_dir(root, LEGACY_VERSION_DIR);
        fs::create_dir_all(legacy.join("bin")).expect("legacy bin");
        fs::write(venv_python(&legacy), b"#!/bin/sh\n").expect("legacy python");
        fs::write(legacy.join("pyvenv.cfg"), "version = 3.9.6\n").expect("pyvenv.cfg");
        fs::write(root.join(ACTIVE_MARKER), "versions/current\n").expect("legacy marker");
    }

    fn versions_on_disk(root: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(root.join(VERSIONS_DIR))
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    // --- version validation and comparison ----------------------------

    #[test]
    fn rejects_unsafe_version_strings() {
        for bad in [
            "", "../x", "a/b", "a\\b", " 1.0", "1 0", ".staging", "1.0\n", "1..2",
        ] {
            assert!(validate_version(bad).is_err(), "{bad:?} should be rejected");
        }
        assert_eq!(validate_version("0.21.5"), Ok(()));
        assert_eq!(validate_version("0.21.0rc1"), Ok(()));
    }

    #[test]
    fn is_newer_compares_numerically() {
        assert!(is_newer("0.9", "0.10"));
        assert!(!is_newer("0.10", "0.9"));
        assert!(is_newer("0.21.5", "0.21.10"));
        assert!(is_newer("0.21", "0.21.1"));
        assert!(!is_newer("0.21.0", "0.21"));
        assert!(!is_newer("0.21.5", "0.21.5"));
        assert!(is_newer("1.0", "2.0"));
    }

    #[test]
    fn is_newer_tolerates_pre_releases_without_panicking() {
        assert!(is_newer("0.21.0rc1", "0.21.0"));
        assert!(!is_newer("0.21.0", "0.21.0rc1"));
        assert!(is_newer("0.21.0a1", "0.21.0b1"));
        assert!(!is_newer("0.21.0", "0.21.0.dev1"));
        for weird in [
            "",
            "abc",
            "v",
            "...",
            "1..2",
            "99999999999999999999999.1",
            "1.0-beta",
        ] {
            let _ = is_newer(weird, "1.0");
            let _ = is_newer("1.0", weird);
        }
        assert!(!is_newer("1.0", "abc"));
    }

    #[test]
    fn parses_the_latest_version_from_pypi_json() {
        assert_eq!(
            parse_latest_version(r#"{"info":{"version":"0.29.1","name":"mlx-lm"},"releases":{}}"#),
            Ok("0.29.1".to_string())
        );
        for bad in [
            "",
            "not json",
            "{}",
            r#"{"info":{}}"#,
            r#"{"info":{"version":42}}"#,
        ] {
            assert!(parse_latest_version(bad).is_err(), "{bad:?}");
        }
        assert!(parse_latest_version(r#"{"info":{"version":"../evil"}}"#).is_err());
    }

    #[test]
    fn parses_shasum_style_digest_files() {
        let hex = "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert_eq!(
            parse_digest(&format!("{hex}  uv-aarch64-apple-darwin.tar.gz\n")),
            Some(hex.to_ascii_lowercase())
        );
        assert_eq!(parse_digest(""), None);
        assert_eq!(parse_digest("not-a-digest file"), None);
    }

    // --- activation and status ----------------------------------------

    #[test]
    fn activation_replaces_the_marker_and_status_follows_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fake_install(root, "0.20.0");
        fake_install(root, "0.21.0");

        activate(root, "0.20.0").expect("activate 0.20.0");
        activate(root, "0.21.0").expect("activate 0.21.0");

        assert_eq!(
            fs::read_to_string(root.join(ACTIVE_MARKER)).expect("marker"),
            "0.21.0\n"
        );
        assert!(!root.join("active.next").exists());
        let current = status(root);
        assert!(current.installed);
        assert_eq!(current.version.as_deref(), Some("0.21.0"));
        assert_eq!(
            current.python,
            Some(root.join("versions/0.21.0/bin/python3"))
        );
    }

    #[test]
    fn activation_rejects_unknown_incomplete_and_unsafe_versions_without_touching_the_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fake_install(root, "0.20.0");
        activate(root, "0.20.0").expect("activate");
        // A venv without version.txt is not a complete install.
        fs::create_dir_all(version_dir(root, "0.22.0").join("bin")).expect("incomplete");
        fs::write(venv_python(&version_dir(root, "0.22.0")), b"x").expect("python");

        assert!(activate(root, "9.9.9").is_err());
        assert!(activate(root, "0.22.0").is_err());
        assert!(activate(root, "../x").is_err());
        assert_eq!(status(root).version.as_deref(), Some("0.20.0"));
    }

    #[test]
    fn status_on_an_empty_root_is_not_installed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let current = status(dir.path());
        assert_eq!(
            current,
            MlxStatus {
                installed: false,
                version: None,
                python: None,
                latest: None,
                update_available: false,
            }
        );
    }

    #[cfg(unix)]
    #[test]
    fn status_never_reports_an_escaping_or_symlinked_marker_as_installed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outside = tempfile::tempdir().expect("outside");
        let root = dir.path();
        fs::create_dir_all(outside.path().join("bin")).expect("bin");
        fs::write(outside.path().join("bin/python3"), b"x").expect("python");
        fs::write(outside.path().join(VERSION_FILE), "1.0\n").expect("version");
        fs::create_dir_all(root.join(VERSIONS_DIR)).expect("versions");
        std::os::unix::fs::symlink(outside.path(), version_dir(root, "9.9.9")).expect("symlink");

        for marker in ["../outside", "9.9.9", &outside.path().display().to_string()] {
            fs::write(root.join(ACTIVE_MARKER), marker).expect("marker");
            assert!(
                !status(root).installed,
                "{marker} must not count as installed"
            );
        }
    }

    // --- legacy layout ------------------------------------------------

    #[test]
    fn legacy_versions_current_is_reported_as_not_installed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        legacy_layout(root);

        let current = status(root);

        assert!(!current.installed);
        assert_eq!(current.version, None);
        assert_eq!(current.python, None);
        assert!(server_command(root).is_err());
    }

    #[test]
    fn install_migrates_the_legacy_layout_and_removes_it_after_activation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        legacy_layout(root);
        let installer = FakeInstaller::new("0.21.0");

        let current = install_latest_with(root, None, &silent, &installer).expect("install");

        assert!(current.installed);
        assert_eq!(current.version.as_deref(), Some("0.21.0"));
        assert_eq!(
            fs::read_to_string(root.join(ACTIVE_MARKER)).expect("marker"),
            "0.21.0\n"
        );
        assert!(!version_dir(root, LEGACY_VERSION_DIR).exists());
        assert_eq!(versions_on_disk(root), vec!["0.21.0".to_string()]);
    }

    #[test]
    fn failed_migration_keeps_the_legacy_runtime_active_and_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        legacy_layout(root);
        // Fail the final build so nothing is activated.
        let installer = FakeInstaller::failing_at("0.21.0", 2);

        let result = install_latest_with(root, None, &silent, &installer);

        assert!(result.is_err());
        assert_eq!(
            fs::read_to_string(root.join(ACTIVE_MARKER)).expect("marker"),
            "versions/current\n"
        );
        assert!(
            version_dir(root, LEGACY_VERSION_DIR)
                .join("bin/python3")
                .is_file()
        );
        assert!(!version_dir(root, "0.21.0").exists());
        assert!(!status(root).installed);
    }

    // --- server command and log ---------------------------------------

    #[test]
    fn server_command_uses_the_active_venv_python_and_module_form() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        assert!(server_command(root).is_err());

        fake_install(root, "0.21.0");
        activate(root, "0.21.0").expect("activate");

        assert_eq!(
            server_command(root),
            Ok((
                root.join("versions/0.21.0/bin/python3"),
                vec!["-m".to_string(), "mlx_lm".to_string(), "server".to_string()]
            ))
        );
    }

    #[test]
    fn log_tail_keeps_the_last_meaningful_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let log =
            "first line\n\n   \n~~~~~^^^^\n    ^^^^^\nValueError: boom\n   ~~^~~  \nlast line\n";
        fs::write(server_log_path(root), log).expect("log");

        assert_eq!(
            log_tail(root, 10),
            "first line\nValueError: boom\nlast line"
        );
        assert_eq!(log_tail(root, 2), "ValueError: boom\nlast line");
        assert_eq!(log_tail(root, 0), "");
    }

    #[test]
    fn log_tail_caps_each_line_at_300_characters_and_is_empty_without_a_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        assert_eq!(log_tail(root, 5), "");

        fs::write(server_log_path(root), format!("{}\n", "x".repeat(500))).expect("log");
        assert_eq!(log_tail(root, 1).chars().count(), LOG_LINE_MAX_CHARS);

        fs::write(server_log_path(root), format!("{}\n", "ñ".repeat(400))).expect("log");
        assert_eq!(log_tail(root, 1).chars().count(), LOG_LINE_MAX_CHARS);
    }

    #[test]
    fn write_server_log_keeps_short_output_and_truncates_long_output_to_its_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("runtime");

        write_server_log(&root, b"short\nlines\n").expect("short");
        assert_eq!(
            fs::read_to_string(server_log_path(&root)).expect("log"),
            "short\nlines\n"
        );

        let big: String = (0..20_000).map(|index| format!("line {index}\n")).collect();
        write_server_log(&root, big.as_bytes()).expect("long");
        let written = fs::read(server_log_path(&root)).expect("log");
        assert!(written.len() <= LOG_MAX_BYTES, "{} bytes", written.len());
        assert!(written.len() > LOG_MAX_BYTES / 2, "{} bytes", written.len());
        let text = String::from_utf8(written).expect("utf8");
        assert!(text.ends_with("line 19999\n"));
        assert!(
            text.starts_with("line "),
            "cut at a line boundary: {text:.20}"
        );
    }

    // --- update and install -------------------------------------------

    #[test]
    fn update_is_a_noop_when_the_active_version_is_already_latest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fake_install(root, "0.21.0");
        activate(root, "0.21.0").expect("activate");

        let result = update_with(root, "0.21.0", &silent, || {
            Err::<FakeInstaller, String>("the installer must not be built".to_string())
        });

        let current = result.expect("no-op update");
        assert_eq!(current.version.as_deref(), Some("0.21.0"));
        assert_eq!(versions_on_disk(root), vec!["0.21.0".to_string()]);
    }

    #[test]
    fn update_installs_latest_side_by_side_and_keeps_the_previous_version_for_rollback() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fake_install(root, "0.20.0");
        activate(root, "0.20.0").expect("activate");

        let current = update_with(root, "0.21.0", &silent, || {
            Ok::<FakeInstaller, String>(FakeInstaller::new("0.21.0"))
        })
        .expect("update");

        assert_eq!(current.version.as_deref(), Some("0.21.0"));
        assert_eq!(
            versions_on_disk(root),
            vec!["0.20.0".to_string(), "0.21.0".to_string()]
        );
        activate(root, "0.20.0").expect("rollback");
        assert_eq!(status(root).version.as_deref(), Some("0.20.0"));
    }

    #[test]
    fn failed_update_keeps_the_previous_version_active() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fake_install(root, "0.20.0");
        activate(root, "0.20.0").expect("activate");

        let result = update_with(root, "0.21.0", &silent, || {
            Ok::<FakeInstaller, String>(FakeInstaller::failing_at("0.21.0", 2))
        });

        assert!(result.is_err());
        assert_eq!(status(root).version.as_deref(), Some("0.20.0"));
        assert_eq!(versions_on_disk(root), vec!["0.20.0".to_string()]);
    }

    #[test]
    fn install_latest_probes_in_staging_then_builds_the_final_pinned_venv_and_activates() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let installer = FakeInstaller::new("0.21.0");

        let current = install_latest_with(root, None, &silent, &installer).expect("install");

        assert_eq!(installer.builds(), vec![None, Some("0.21.0".to_string())]);
        assert!(current.installed);
        assert_eq!(
            current.python,
            Some(root.join("versions/0.21.0/bin/python3"))
        );
        assert_eq!(
            fs::read_to_string(root.join("versions/0.21.0/version.txt")).expect("version.txt"),
            "0.21.0\n"
        );
        assert_eq!(
            fs::read_to_string(root.join(ACTIVE_MARKER)).expect("marker"),
            "0.21.0\n"
        );
        assert!(!root.join("versions").join(STAGING_DIR).exists());
    }

    #[test]
    fn install_fails_when_the_installed_version_differs_from_the_requested_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let installer = FakeInstaller::new("0.19.9");

        let result = install_latest_with(root, Some("0.21.0"), &silent, &installer);

        assert_eq!(
            result,
            Err("Se esperaba mlx-lm 0.21.0 pero se instaló 0.19.9".to_string())
        );
        assert!(!status(root).installed);
        assert!(versions_on_disk(root).is_empty());
    }

    #[test]
    fn check_update_fills_latest_and_availability_from_the_installed_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        assert_eq!(
            with_latest(status(root), "0.10.0".to_string()),
            MlxStatus {
                installed: false,
                version: None,
                python: None,
                latest: Some("0.10.0".to_string()),
                update_available: false,
            }
        );

        fake_install(root, "0.9.0");
        activate(root, "0.9.0").expect("activate");
        let behind = with_latest(status(root), "0.10.0".to_string());
        assert!(behind.update_available);
        assert_eq!(behind.latest.as_deref(), Some("0.10.0"));
        assert!(!with_latest(status(root), "0.9.0".to_string()).update_available);
    }

    // --- uv bootstrap -------------------------------------------------

    #[cfg(unix)]
    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn an_existing_uv_is_returned_without_downloading() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("bin")).expect("bin");
        fs::write(root.join("bin/uv"), b"fake").expect("uv");
        let fetches = RefCell::new(0);

        let uv = ensure_uv_with(root, &silent, &|_, _| {
            *fetches.borrow_mut() += 1;
            Ok(())
        })
        .expect("uv");

        assert_eq!(uv, root.join("bin/uv"));
        assert_eq!(*fetches.borrow(), 0);
    }

    #[test]
    fn uv_urls_point_to_the_astral_release_and_its_sibling_digest() {
        assert_eq!(
            UV_DOWNLOAD_URL,
            "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz"
        );
        assert_eq!(uv_binary_path(Path::new("/r")), PathBuf::from("/r/bin/uv"));
    }

    #[test]
    fn a_digest_mismatch_fails_closed_and_installs_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let wrong = "0".repeat(64);

        let result = ensure_uv_with(root, &silent, &|url, destination| {
            if url.ends_with(".sha256") {
                fs::write(destination, format!("{wrong}  uv.tar.gz\n")).expect("digest");
            } else {
                fs::write(destination, b"tampered archive").expect("archive");
            }
            Ok(())
        });

        assert_eq!(result, Err(CHECKSUM_MISMATCH.to_string()));
        assert!(!root.join("bin/uv").exists());
        assert!(!downloads_dir(root).join(UV_ASSET_NAME).exists());
    }

    #[test]
    fn a_missing_or_invalid_digest_fails_closed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        let unavailable = ensure_uv_with(root, &silent, &|url, destination| {
            if url.ends_with(".sha256") {
                Err("404".to_string())
            } else {
                fs::write(destination, b"archive").expect("archive");
                Ok(())
            }
        });
        assert!(unavailable.expect_err("no digest").contains("SHA-256"));

        let invalid = ensure_uv_with(root, &silent, &|url, destination| {
            if url.ends_with(".sha256") {
                fs::write(destination, "garbage").expect("digest");
            } else {
                fs::write(destination, b"archive").expect("archive");
            }
            Ok(())
        });
        assert!(invalid.expect_err("invalid digest").contains("SHA-256"));
        assert!(!root.join("bin/uv").exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_verified_archive_installs_an_executable_uv() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let source = tempfile::tempdir().expect("source");
        let package = source.path().join("uv-aarch64-apple-darwin");
        fs::create_dir_all(&package).expect("package");
        fs::write(package.join("uv"), "#!/bin/sh\necho uv 0.0.0\n").expect("uv script");
        let tarball = source.path().join("uv.tar.gz");
        let status = Command::new("tar")
            .arg("-czf")
            .arg(&tarball)
            .arg("-C")
            .arg(source.path())
            .arg("uv-aarch64-apple-darwin")
            .status()
            .expect("tar");
        assert!(status.success());
        let bytes = fs::read(&tarball).expect("tarball");
        let digest = sha256_hex(&bytes);

        let uv = ensure_uv_with(root, &silent, &|url, destination| {
            if url.ends_with(".sha256") {
                fs::write(
                    destination,
                    format!("{digest}  uv-aarch64-apple-darwin.tar.gz\n"),
                )
                .expect("digest");
            } else {
                fs::write(destination, &bytes).expect("archive");
            }
            Ok(())
        })
        .expect("uv");

        assert_eq!(uv, root.join("bin/uv"));
        assert_eq!(
            fs::metadata(&uv).expect("uv").permissions().mode() & 0o777,
            0o755
        );
        assert!(!root.join("bin/uv.partial").exists());
        assert!(!downloads_dir(root).join(UV_ASSET_NAME).exists());
        assert!(!downloads_dir(root).join("uv-extract").exists());
    }

    #[test]
    fn user_facing_errors_are_in_spanish() {
        assert_eq!(
            activate(Path::new("/nonexistent-root"), "9.9.9").expect_err("not installed"),
            "La versión 9.9.9 de MLX no está instalada"
        );
        assert_eq!(
            server_command(Path::new("/nonexistent-root")).expect_err("not installed"),
            NOT_INSTALLED
        );
        assert_eq!(
            CHECKSUM_MISMATCH,
            "El SHA-256 de uv no coincide; instalación cancelada"
        );
    }
}
