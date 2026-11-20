//! Local model server lifecycle: one `mlx_lm.server` (MLX) or `llama-server` (GGUF) child process
//! that serves a single model on loopback.
//!
//! [`LocalServerManager`] owns at most one child. It reuses the child while the runtime kind and
//! model match and the child is alive, replaces it when either changes, kills it when a start
//! fails, and never restarts anything after [`LocalServerManager::shutdown`]. A crash noticed by
//! the watchdog is reported as `Failed`; it is never restarted silently.
//!
//! The server only ever binds `127.0.0.1`. Readiness is `GET /v1/models` answering 200 with a
//! non-empty body. A pidfile in the runtime root lets the next run reap a server orphaned by a
//! crash. Process inspection uses `ps`/`kill` on Unix and PowerShell/`tasklist`/`taskkill` on
//! Windows, and only ever signals a process whose command line proves it is one of our servers.

use std::ffi::OsString;
use std::fs;
use std::path::{MAIN_SEPARATOR, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::AiError;

/// The server is only ever bound to the loopback interface.
const LOOPBACK_HOST: &str = "127.0.0.1";
/// Readiness route polled on the owned child.
const READINESS_PATH: &str = "/v1/models";
const PIDFILE_NAME: &str = "mt-ai-server.pid";
/// File in the runtime root that receives the server's `stderr`, truncated on every launch.
pub const SERVER_LOG_NAME: &str = "server.log";
/// Command-line markers of the server binaries this module launches.
const SERVER_MARKERS: [&str; 2] = ["mlx_lm server", "llama-server"];
const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DEFAULT_WATCHDOG_INTERVAL: Duration = Duration::from_secs(2);
const HEALTH_REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which local inference server to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LocalRuntimeKind {
    /// `mlx_lm.server`: `executable` is a Python interpreter and `model_path` a model folder.
    Mlx,
    /// `llama-server`: `executable` is the binary and `model_path` a `.gguf` file.
    Gguf,
}

/// The runtime this platform should prefer: MLX on macOS Apple Silicon, GGUF elsewhere.
pub fn recommended_local_runtime() -> LocalRuntimeKind {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        LocalRuntimeKind::Mlx
    } else {
        LocalRuntimeKind::Gguf
    }
}

/// Everything needed to start one local server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalServerSpec {
    pub kind: LocalRuntimeKind,
    /// Python interpreter for MLX, or the `llama-server` binary for GGUF.
    pub executable: PathBuf,
    /// Model folder for MLX, or the `.gguf` file for GGUF.
    pub model_path: PathBuf,
    pub context_tokens: usize,
    /// Runtime directory that holds the pidfile; orphans are reaped from here.
    pub runtime_root: PathBuf,
}

/// Where a ready server listens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalServerEndpoint {
    /// OpenAI-compatible base URL, `http://127.0.0.1:<port>/v1`, usable as `ModelConfig::endpoint`.
    pub endpoint: String,
    pub port: u16,
    pub kind: LocalRuntimeKind,
    pub model_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum LocalServerState {
    Stopped,
    Starting,
    Ready,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalServerStatus {
    pub state: LocalServerState,
    pub kind: Option<LocalRuntimeKind>,
    pub model_path: Option<PathBuf>,
    pub endpoint: Option<String>,
    pub message: String,
}

impl LocalServerStatus {
    fn new(
        state: LocalServerState,
        kind: Option<LocalRuntimeKind>,
        model_path: Option<PathBuf>,
        endpoint: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            state,
            kind,
            model_path,
            endpoint,
            message: message.into(),
        }
    }

    fn stopped() -> Self {
        Self::new(
            LocalServerState::Stopped,
            None,
            None,
            None,
            "El servidor local está detenido",
        )
    }
}

/// Startup and watchdog timings, injectable for tests.
#[derive(Debug, Clone, Copy)]
pub struct LocalServerTimings {
    pub startup_timeout: Duration,
    pub poll_interval: Duration,
    pub watchdog_interval: Duration,
}

impl Default for LocalServerTimings {
    fn default() -> Self {
        Self {
            startup_timeout: DEFAULT_STARTUP_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
            watchdog_interval: DEFAULT_WATCHDOG_INTERVAL,
        }
    }
}

/// Spawns the server process. Injectable so tests never need a real runtime.
pub trait ServerLauncher: Send + Sync {
    fn launch(&self, spec: &LocalServerSpec, port: u16) -> Result<Child, AiError>;
}

/// Real launcher: runs the server binary for `spec.kind` bound to `127.0.0.1:<port>`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProcessServerLauncher;

impl ServerLauncher for ProcessServerLauncher {
    fn launch(&self, spec: &LocalServerSpec, port: u16) -> Result<Child, AiError> {
        // A previous run that died without running `Drop` leaves its server orphaned and holding
        // the model in memory. Reap it before starting a new one.
        reap_orphaned_server(&spec.runtime_root);
        let mut command = server_command(spec, port);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(open_server_log(&spec.runtime_root));
        let child = command.spawn().map_err(|error| {
            AiError::Runtime(format!("no se pudo iniciar el servidor local: {error}"))
        })?;
        // Best effort: without the pidfile the only loss is orphan cleanup on the next run.
        let _ = fs::write(
            pidfile_path(&spec.runtime_root),
            format!("{} {}", child.id(), std::process::id()),
        );
        Ok(child)
    }
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

/// Full argument list for the server of `spec.kind`; the host is always loopback.
fn server_args(spec: &LocalServerSpec, port: u16) -> Vec<OsString> {
    let port = port.to_string();
    let model = spec.model_path.clone().into_os_string();
    match spec.kind {
        LocalRuntimeKind::Mlx => vec![
            "-m".into(),
            "mlx_lm".into(),
            "server".into(),
            "--model".into(),
            model,
            "--host".into(),
            LOOPBACK_HOST.into(),
            "--port".into(),
            port.into(),
        ],
        LocalRuntimeKind::Gguf => vec![
            "--model".into(),
            model,
            "--host".into(),
            LOOPBACK_HOST.into(),
            "--port".into(),
            port.into(),
            "--ctx-size".into(),
            spec.context_tokens.to_string().into(),
            // Enables llama-server's native chat template so tool calls are parsed.
            "--jinja".into(),
        ],
    }
}

fn server_command(spec: &LocalServerSpec, port: u16) -> Command {
    let mut command = Command::new(&spec.executable);
    command.args(server_args(spec, port));
    hide_console(&mut command);
    command
}

// ---------------------------------------------------------------------------
// Server log
// ---------------------------------------------------------------------------

/// Lines of `server.log` quoted in a startup failure.
const LOG_TAIL_LINES: usize = 5;
/// Longest line quoted from `server.log`, in characters.
const LOG_LINE_MAX_CHARS: usize = 300;

/// The last `lines` meaningful lines of `<runtime_root>/server.log`, joined with `" | "`.
///
/// Blank lines and Python traceback caret decoration (lines made only of `~^ |`) are skipped,
/// and each kept line is capped at [`LOG_LINE_MAX_CHARS`]. A missing or unreadable log is empty.
pub fn log_tail(runtime_root: &Path, lines: usize) -> String {
    let Ok(bytes) = fs::read(runtime_root.join(SERVER_LOG_NAME)) else {
        return String::new();
    };
    let text = String::from_utf8_lossy(&bytes);
    let meaningful: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !is_traceback_decoration(line))
        .map(|line| line.chars().take(LOG_LINE_MAX_CHARS).collect())
        .collect();
    let start = meaningful.len().saturating_sub(lines);
    meaningful[start..].join(" | ")
}

/// `true` for the caret lines Python prints under a traceback, such as `    ~^^^^`.
fn is_traceback_decoration(line: &str) -> bool {
    line.chars().all(|c| matches!(c, '~' | '^' | ' ' | '|'))
}

/// Opens `<runtime_root>/server.log` (create + truncate) as the child's `stderr`. When the log
/// cannot be opened, `stderr` is discarded so an unwritable runtime root never blocks the start.
fn open_server_log(runtime_root: &Path) -> Stdio {
    let _ = fs::create_dir_all(runtime_root);
    fs::File::create(runtime_root.join(SERVER_LOG_NAME)).map_or_else(|_| Stdio::null(), Stdio::from)
}

// ---------------------------------------------------------------------------
// Pidfile and orphan reaping
// ---------------------------------------------------------------------------

fn pidfile_path(runtime_root: &Path) -> PathBuf {
    runtime_root.join(PIDFILE_NAME)
}

/// Kills the server recorded in the pidfile when its owner is gone, the process is still alive,
/// and its command line proves it is one of our servers launched from `runtime_root`. A server
/// whose owner still runs, and any process that reused the pid, are left alone.
fn reap_orphaned_server(runtime_root: &Path) {
    let pidfile = pidfile_path(runtime_root);
    let text = fs::read_to_string(&pidfile).unwrap_or_default();
    let mut fields = text.split_whitespace().map(str::parse::<u32>);
    let pid = fields.next().and_then(Result::ok);
    let owner = fields.next().and_then(Result::ok);
    // pid 0 is not a process, and our own pid as owner belongs to a previous run whose pid was reused.
    let owner_alive = owner.is_some_and(|owner| {
        owner != 0 && owner != std::process::id() && process_is_alive(owner)
    });
    if owner_alive {
        return;
    }
    let _ = fs::remove_file(&pidfile);
    let Some(pid) = pid else { return };
    if pid == 0 || pid == std::process::id() {
        return;
    }
    let Some(command_line) = process_command_line(pid) else {
        return;
    };
    // Trailing separator so a sibling root such as `<root>2` never matches.
    let root_prefix = format!("{}{MAIN_SEPARATOR}", normalize_for_match(&runtime_root.to_string_lossy()));
    let is_our_server = SERVER_MARKERS
        .iter()
        .any(|marker| command_line.contains(marker));
    if is_our_server && normalize_for_match(&command_line).contains(&root_prefix) {
        kill_process(pid);
    }
}

/// Removes the pidfile only when it still records `pid` as our child.
fn remove_pidfile_if_ours(runtime_root: &Path, pid: u32) {
    let pidfile = pidfile_path(runtime_root);
    let recorded = fs::read_to_string(&pidfile).ok().and_then(|text| {
        text.split_whitespace()
            .next()
            .and_then(|field| field.parse::<u32>().ok())
    });
    if recorded == Some(pid) {
        let _ = fs::remove_file(pidfile);
    }
}

fn normalize_for_match(text: &str) -> String {
    if cfg!(windows) {
        text.replace('/', "\\").to_ascii_lowercase()
    } else {
        text.to_string()
    }
}

/// A helper process with no stdin and, on Windows, no console window.
fn helper_command(program: &str) -> Command {
    let mut command = Command::new(program);
    command.stdin(Stdio::null());
    hide_console(&mut command);
    command
}

#[cfg(windows)]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_command: &mut Command) {}

fn command_output_line(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    let line = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (output.status.success() && !line.is_empty()).then_some(line)
}

/// Full command line of `pid`, or `None` when the process does not exist.
#[cfg(unix)]
fn process_command_line(pid: u32) -> Option<String> {
    command_output_line(helper_command("ps").args(["-o", "command=", "-p", &pid.to_string()]))
}

#[cfg(windows)]
fn process_command_line(pid: u32) -> Option<String> {
    let script = format!("(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine");
    command_output_line(helper_command("powershell").args([
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ]))
}

#[cfg(unix)]
fn process_is_alive(pid: u32) -> bool {
    helper_command("kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    let needle = format!("\"{pid}\"");
    helper_command("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains(&needle))
}

#[cfg(unix)]
fn kill_process(pid: u32) {
    let _ = helper_command("kill")
        .args(["-9", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(windows)]
fn kill_process(pid: u32) {
    let _ = helper_command("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

// ---------------------------------------------------------------------------
// Port reservation
// ---------------------------------------------------------------------------

/// Picks a free loopback port by binding `:0`; the listener is dropped so the server can bind it.
fn reserve_loopback_port() -> Result<u16, AiError> {
    let listener = std::net::TcpListener::bind((LOOPBACK_HOST, 0)).map_err(|error| {
        AiError::Runtime(format!("no hay un puerto local disponible: {error}"))
    })?;
    let port = listener
        .local_addr()
        .map_err(|error| AiError::Runtime(format!("no hay un puerto local disponible: {error}")))?
        .port();
    Ok(port)
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// The only process this manager signals. Dropping it kills and reaps it.
struct OwnedChild {
    process: Child,
    pid: u32,
    port: u16,
    kind: LocalRuntimeKind,
    model_path: PathBuf,
    runtime_root: PathBuf,
}

impl OwnedChild {
    fn endpoint(&self) -> LocalServerEndpoint {
        LocalServerEndpoint {
            endpoint: format!("http://{LOOPBACK_HOST}:{}/v1", self.port),
            port: self.port,
            kind: self.kind,
            model_path: self.model_path.clone(),
        }
    }

    /// `Some(code)` once the child has exited (`code` is `None` when killed by a signal).
    fn exit_code_if_exited(&mut self) -> Option<Option<i32>> {
        match self.process.try_wait() {
            Ok(None) => None,
            Ok(Some(status)) => Some(status.code()),
            Err(_) => Some(None),
        }
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
        remove_pidfile_if_ours(&self.runtime_root, self.pid);
    }
}

struct Inner {
    status: RwLock<LocalServerStatus>,
    child: Mutex<Option<OwnedChild>>,
    /// Bumped on every stop, replacement or failure. A start or watchdog whose generation is
    /// stale must not touch the state anymore.
    generation: AtomicU64,
    /// Serializes `ensure_ready` so two starts never race for the single slot.
    operation: tokio::sync::Mutex<()>,
    shutting_down: AtomicBool,
    launcher: Arc<dyn ServerLauncher>,
    timings: LocalServerTimings,
}

/// Owns at most one local server process and keeps its status in sync with it.
#[derive(Clone)]
pub struct LocalServerManager {
    inner: Arc<Inner>,
}

impl Default for LocalServerManager {
    fn default() -> Self {
        Self::new()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Why a start did not produce a ready server.
enum StartError {
    /// `stop` or `shutdown` interrupted the start; the state is already set by that caller.
    Cancelled,
    Failed(AiError),
}

impl From<AiError> for StartError {
    fn from(error: AiError) -> Self {
        Self::Failed(error)
    }
}

enum ChildPoll {
    Running,
    Cancelled,
    Exited(Option<i32>),
}

impl LocalServerManager {
    pub fn new() -> Self {
        Self::with_launcher(Arc::new(ProcessServerLauncher), LocalServerTimings::default())
    }

    pub fn with_launcher(launcher: Arc<dyn ServerLauncher>, timings: LocalServerTimings) -> Self {
        Self {
            inner: Arc::new(Inner {
                status: RwLock::new(LocalServerStatus::stopped()),
                child: Mutex::new(None),
                generation: AtomicU64::new(0),
                operation: tokio::sync::Mutex::new(()),
                shutting_down: AtomicBool::new(false),
                launcher,
                timings,
            }),
        }
    }

    pub fn status(&self) -> LocalServerStatus {
        self.inner
            .status
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Kills and reaps the owned child (never any other process) and reports `Stopped`.
    pub fn stop(&self) {
        let (_, previous) = self.detach(LocalServerStatus::stopped());
        drop(previous);
    }

    /// Like `stop`, and forbids any later start.
    pub fn shutdown(&self) {
        self.inner.shutting_down.store(true, Ordering::SeqCst);
        self.stop();
    }

    /// Returns a ready endpoint for `spec`.
    ///
    /// - A ready child of the same kind and model is reused while it is alive.
    /// - Otherwise the previous child is stopped, a loopback port is reserved, the server is
    ///   spawned, and this returns once `GET /v1/models` answers 200 with a non-empty body.
    /// - An early exit or the startup deadline fails the call and kills the child.
    pub async fn ensure_ready(
        &self,
        spec: &LocalServerSpec,
    ) -> Result<LocalServerEndpoint, AiError> {
        let _operation = self.inner.operation.lock().await;
        if self.is_shutting_down() {
            return Err(shutting_down_error());
        }
        if let Some(endpoint) = self.reusable(spec) {
            return Ok(endpoint);
        }

        let (generation, previous) = self.detach(starting_status(spec));
        drop(previous);
        match self.start(spec, generation).await {
            Ok(endpoint) => Ok(endpoint),
            Err(StartError::Cancelled) => Err(AiError::Runtime(
                "el arranque del servidor local se canceló".to_string(),
            )),
            Err(StartError::Failed(error)) => {
                let killed = self.detach_if_current(
                    generation,
                    failed_status(spec, error.to_string()),
                );
                drop(killed);
                Err(error)
            }
        }
    }

    fn is_shutting_down(&self) -> bool {
        self.inner.shutting_down.load(Ordering::SeqCst)
    }

    fn set_status(&self, status: LocalServerStatus) {
        *self
            .inner
            .status
            .write()
            .unwrap_or_else(PoisonError::into_inner) = status;
    }

    /// Returns the live ready child when it matches `spec`.
    fn reusable(&self, spec: &LocalServerSpec) -> Option<LocalServerEndpoint> {
        let mut slot = lock(&self.inner.child);
        let owned = slot.as_mut()?;
        if self.status().state != LocalServerState::Ready
            || owned.kind != spec.kind
            || owned.model_path != spec.model_path
            || owned.exit_code_if_exited().is_some()
        {
            return None;
        }
        Some(owned.endpoint())
    }

    /// Stops the owned child, bumps the generation and sets `status`, atomically. The killed
    /// child is returned so the caller drops it outside the lock.
    fn detach(&self, status: LocalServerStatus) -> (u64, Option<OwnedChild>) {
        let mut slot = lock(&self.inner.child);
        let generation = self.inner.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let owned = slot.take();
        self.set_status(status);
        (generation, owned)
    }

    /// Like `detach`, but does nothing when a newer generation has already taken over.
    fn detach_if_current(
        &self,
        generation: u64,
        status: LocalServerStatus,
    ) -> Option<OwnedChild> {
        let mut slot = lock(&self.inner.child);
        if self.inner.generation.load(Ordering::SeqCst) != generation {
            return None;
        }
        self.inner.generation.fetch_add(1, Ordering::SeqCst);
        let owned = slot.take();
        self.set_status(status);
        owned
    }

    fn take_child(&self) -> Option<OwnedChild> {
        lock(&self.inner.child).take()
    }

    fn poll_child(&self, generation: u64) -> ChildPoll {
        let mut slot = lock(&self.inner.child);
        if self.inner.generation.load(Ordering::SeqCst) != generation {
            return ChildPoll::Cancelled;
        }
        match slot.as_mut() {
            None => ChildPoll::Cancelled,
            Some(owned) => match owned.exit_code_if_exited() {
                None => ChildPoll::Running,
                Some(code) => ChildPoll::Exited(code),
            },
        }
    }

    async fn start(
        &self,
        spec: &LocalServerSpec,
        generation: u64,
    ) -> Result<LocalServerEndpoint, StartError> {
        let port = reserve_loopback_port()?;
        let process = self.inner.launcher.launch(spec, port)?;
        let owned = OwnedChild {
            pid: process.id(),
            process,
            port,
            kind: spec.kind,
            model_path: spec.model_path.clone(),
            runtime_root: spec.runtime_root.clone(),
        };
        // Stored before anything else can fail, so the child is always killed on drop.
        *lock(&self.inner.child) = Some(owned);
        if self.is_shutting_down() || self.inner.generation.load(Ordering::SeqCst) != generation {
            drop(self.take_child());
            return Err(StartError::Cancelled);
        }

        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(HEALTH_REQUEST_TIMEOUT)
            .build()
            .map_err(|error| {
                AiError::Runtime(format!("no se pudo comprobar el servidor local: {error}"))
            })?;
        let deadline = tokio::time::Instant::now() + self.inner.timings.startup_timeout;
        loop {
            match self.poll_child(generation) {
                ChildPoll::Running => {}
                ChildPoll::Cancelled => return Err(StartError::Cancelled),
                ChildPoll::Exited(code) => {
                    let tail = log_tail(&spec.runtime_root, LOG_TAIL_LINES);
                    return Err(StartError::Failed(AiError::Runtime(startup_exit_message(
                        code, &tail,
                    ))));
                }
            }
            if models_route_ready(&client, port).await {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(StartError::Failed(AiError::Runtime(
                    "El servidor local no quedó listo a tiempo; se detuvo el proceso".to_string(),
                )));
            }
            tokio::time::sleep(self.inner.timings.poll_interval).await;
        }

        let endpoint = {
            let slot = lock(&self.inner.child);
            match slot.as_ref() {
                Some(owned) if self.inner.generation.load(Ordering::SeqCst) == generation => {
                    let endpoint = owned.endpoint();
                    self.set_status(LocalServerStatus::new(
                        LocalServerState::Ready,
                        Some(endpoint.kind),
                        Some(endpoint.model_path.clone()),
                        Some(endpoint.endpoint.clone()),
                        "El servidor local está listo",
                    ));
                    endpoint
                }
                _ => return Err(StartError::Cancelled),
            }
        };
        self.spawn_watchdog(generation);
        Ok(endpoint)
    }

    /// Reports `Failed` when the owned child exits unexpectedly. It never restarts the server:
    /// the next `ensure_ready` call is the only way back.
    fn spawn_watchdog(&self, generation: u64) {
        let manager = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(manager.inner.timings.watchdog_interval).await;
                match manager.poll_child(generation) {
                    ChildPoll::Running => {}
                    ChildPoll::Cancelled => return,
                    ChildPoll::Exited(code) => {
                        let previous = manager.status();
                        let failed = LocalServerStatus::new(
                            LocalServerState::Failed,
                            previous.kind,
                            previous.model_path,
                            None,
                            format!(
                                "El servidor local se cerró inesperadamente{}",
                                exit_detail(code)
                            ),
                        );
                        drop(manager.detach_if_current(generation, failed));
                        return;
                    }
                }
            }
        });
    }
}

fn shutting_down_error() -> AiError {
    AiError::Runtime(
        "la aplicación se está cerrando; no se inicia el servidor local".to_string(),
    )
}

fn exit_detail(code: Option<i32>) -> String {
    code.map_or_else(String::new, |code| format!(" (código {code})"))
}

/// Startup-exit message; the log tail is appended after the exit code when there is one.
fn startup_exit_message(code: Option<i32>, tail: &str) -> String {
    let base = format!("El servidor local se cerró durante el arranque{}", exit_detail(code));
    if tail.is_empty() {
        base
    } else {
        format!("{base}: {tail}")
    }
}

fn starting_status(spec: &LocalServerSpec) -> LocalServerStatus {
    LocalServerStatus::new(
        LocalServerState::Starting,
        Some(spec.kind),
        Some(spec.model_path.clone()),
        None,
        "Iniciando el servidor local…",
    )
}

fn failed_status(spec: &LocalServerSpec, message: String) -> LocalServerStatus {
    LocalServerStatus::new(
        LocalServerState::Failed,
        Some(spec.kind),
        Some(spec.model_path.clone()),
        None,
        message,
    )
}

/// Ready only on HTTP 200 with a non-empty body.
async fn models_route_ready(client: &reqwest::Client, port: u16) -> bool {
    let url = format!("http://{LOOPBACK_HOST}:{port}{READINESS_PATH}");
    let Ok(response) = client.get(url).send().await else {
        return false;
    };
    if response.status() != reqwest::StatusCode::OK {
        return false;
    }
    response
        .text()
        .await
        .is_ok_and(|body| !body.trim().is_empty())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: LocalRuntimeKind) -> LocalServerSpec {
        LocalServerSpec {
            kind,
            executable: PathBuf::from("/opt/runtime/bin/server"),
            model_path: PathBuf::from("/models/a"),
            context_tokens: 4096,
            runtime_root: PathBuf::from("/nonexistent-runtime-root"),
        }
    }

    fn strings(args: &[OsString]) -> Vec<&str> {
        args.iter().map(|arg| arg.to_str().expect("utf-8 arg")).collect()
    }

    #[test]
    fn server_args_bind_loopback_for_each_kind() {
        let mlx = server_args(&spec(LocalRuntimeKind::Mlx), 8765);
        assert_eq!(
            strings(&mlx),
            [
                "-m",
                "mlx_lm",
                "server",
                "--model",
                "/models/a",
                "--host",
                "127.0.0.1",
                "--port",
                "8765"
            ]
        );

        let gguf = server_args(&spec(LocalRuntimeKind::Gguf), 8766);
        assert_eq!(
            strings(&gguf),
            [
                "--model",
                "/models/a",
                "--host",
                "127.0.0.1",
                "--port",
                "8766",
                "--ctx-size",
                "4096",
                "--jinja"
            ]
        );
    }

    #[test]
    fn runtime_kind_serializes_as_camel_case_and_recommendation_matches_platform() {
        assert_eq!(serde_json::to_string(&LocalRuntimeKind::Mlx).unwrap(), "\"mlx\"");
        assert_eq!(serde_json::to_string(&LocalRuntimeKind::Gguf).unwrap(), "\"gguf\"");
        let expected = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            LocalRuntimeKind::Mlx
        } else {
            LocalRuntimeKind::Gguf
        };
        assert_eq!(recommended_local_runtime(), expected);
    }

    #[test]
    fn log_tail_keeps_the_last_meaningful_lines_capped_and_undecorated() {
        let root = tempfile::tempdir().expect("tempdir");
        let long = "x".repeat(500);
        let log = format!(
            "Loading model\n\n   \n    ~^ |\n  ^^^^\n{long}\nValueError: Model type gemma4 not supported.\n"
        );
        fs::write(root.path().join(SERVER_LOG_NAME), log).expect("server log");
        let capped = "x".repeat(300);

        assert_eq!(
            log_tail(root.path(), 2),
            format!("{capped} | ValueError: Model type gemma4 not supported.")
        );
        assert_eq!(
            log_tail(root.path(), 10),
            format!("Loading model | {capped} | ValueError: Model type gemma4 not supported.")
        );
        assert_eq!(log_tail(root.path(), 0), "");
    }

    #[test]
    fn log_tail_of_a_missing_log_is_empty() {
        let root = tempfile::tempdir().expect("tempdir");
        assert_eq!(log_tail(root.path(), 5), "");
    }
}

#[cfg(all(test, unix))]
mod manager_tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::AtomicUsize;

    const BODY: &str = r#"{"object":"list","data":[]}"#;

    #[derive(Clone, Copy)]
    enum Behavior {
        /// Long-lived `sleep` child plus a loopback server answering 200 with a body.
        Serve,
        /// `sh -c "exit 3"`: exits before it can ever be ready.
        ExitEarly,
    }

    struct FakeLauncher {
        behavior: Behavior,
        spawns: AtomicUsize,
        pids: Mutex<Vec<u32>>,
        stop_servers: Arc<AtomicBool>,
    }

    impl FakeLauncher {
        fn new(behavior: Behavior) -> Arc<Self> {
            Arc::new(Self {
                behavior,
                spawns: AtomicUsize::new(0),
                pids: Mutex::new(Vec::new()),
                stop_servers: Arc::new(AtomicBool::new(false)),
            })
        }

        fn spawns(&self) -> usize {
            self.spawns.load(Ordering::SeqCst)
        }

        fn pids(&self) -> Vec<u32> {
            lock(&self.pids).clone()
        }
    }

    impl Drop for FakeLauncher {
        fn drop(&mut self) {
            self.stop_servers.store(true, Ordering::SeqCst);
        }
    }

    /// Answers every request on `127.0.0.1:port` with 200 and a non-empty body until `stop`.
    fn serve(port: u16, stop: Arc<AtomicBool>) {
        let listener = TcpListener::bind((LOOPBACK_HOST, port)).expect("fake server bind");
        listener.set_nonblocking(true).expect("nonblocking listener");
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let mut request = [0u8; 1024];
                        let _ = stream.read(&mut request);
                        let response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{BODY}",
                            BODY.len()
                        );
                        let _ = stream.write_all(response.as_bytes());
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            }
        });
    }

    impl ServerLauncher for FakeLauncher {
        fn launch(&self, _spec: &LocalServerSpec, port: u16) -> Result<Child, AiError> {
            self.spawns.fetch_add(1, Ordering::SeqCst);
            let mut command = match self.behavior {
                Behavior::Serve => {
                    let mut command = Command::new("sleep");
                    command.arg("300");
                    command
                }
                Behavior::ExitEarly => {
                    let mut command = Command::new("sh");
                    command.args(["-c", "exit 3"]);
                    command
                }
            };
            let child = command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|error| AiError::Runtime(error.to_string()))?;
            lock(&self.pids).push(child.id());
            if matches!(self.behavior, Behavior::Serve) {
                serve(port, self.stop_servers.clone());
            }
            Ok(child)
        }
    }

    fn fast_timings() -> LocalServerTimings {
        LocalServerTimings {
            startup_timeout: Duration::from_secs(3),
            poll_interval: Duration::from_millis(20),
            watchdog_interval: Duration::from_millis(20),
        }
    }

    fn manager(behavior: Behavior) -> (LocalServerManager, Arc<FakeLauncher>) {
        let launcher = FakeLauncher::new(behavior);
        let manager = LocalServerManager::with_launcher(launcher.clone(), fast_timings());
        (manager, launcher)
    }

    fn gguf_spec(model: &str) -> LocalServerSpec {
        LocalServerSpec {
            kind: LocalRuntimeKind::Gguf,
            executable: PathBuf::from("llama-server"),
            model_path: PathBuf::from(model),
            context_tokens: 4096,
            runtime_root: PathBuf::from("/nonexistent-runtime-root"),
        }
    }

    #[tokio::test]
    async fn ready_server_is_reused_for_the_same_model() {
        let (manager, launcher) = manager(Behavior::Serve);
        let spec = gguf_spec("/models/a.gguf");

        let first = manager.ensure_ready(&spec).await.expect("ready");
        let second = manager.ensure_ready(&spec).await.expect("reused");

        assert_eq!(first, second);
        assert_eq!(first.endpoint, format!("http://127.0.0.1:{}/v1", first.port));
        assert_eq!(launcher.spawns(), 1, "the ready server is reused, not respawned");
        let status = manager.status();
        assert_eq!(status.state, LocalServerState::Ready);
        assert_eq!(status.endpoint.as_deref(), Some(first.endpoint.as_str()));
        assert_eq!(status.model_path, Some(PathBuf::from("/models/a.gguf")));
        manager.shutdown();
    }

    #[tokio::test]
    async fn switching_model_stops_the_previous_child() {
        let (manager, launcher) = manager(Behavior::Serve);
        manager
            .ensure_ready(&gguf_spec("/models/a.gguf"))
            .await
            .expect("first model");
        let old_pid = launcher.pids()[0];

        let second = manager
            .ensure_ready(&gguf_spec("/models/b.gguf"))
            .await
            .expect("second model");

        assert_eq!(launcher.spawns(), 2);
        assert!(!process_is_alive(old_pid), "previous child must be gone");
        assert_eq!(second.model_path, PathBuf::from("/models/b.gguf"));
        assert_eq!(
            manager.status().model_path,
            Some(PathBuf::from("/models/b.gguf"))
        );
        manager.shutdown();
    }

    #[tokio::test]
    async fn early_exit_fails_cleanly_with_exit_code() {
        let (manager, _launcher) = manager(Behavior::ExitEarly);

        let error = manager
            .ensure_ready(&gguf_spec("/models/a.gguf"))
            .await
            .expect_err("early exit must fail");

        assert!(
            matches!(&error, AiError::Runtime(msg) if msg.contains("código 3")),
            "{error:?}"
        );
        let status = manager.status();
        assert_eq!(status.state, LocalServerState::Failed);
        assert_eq!(status.endpoint, None);
        assert!(status.message.contains("código 3"), "{}", status.message);
    }

    #[tokio::test]
    async fn shutdown_kills_the_child_and_prevents_restart() {
        let (manager, launcher) = manager(Behavior::Serve);
        let spec = gguf_spec("/models/a.gguf");
        manager.ensure_ready(&spec).await.expect("ready");
        let pid = launcher.pids()[0];

        manager.shutdown();

        assert!(!process_is_alive(pid), "owned child must be killed");
        assert_eq!(manager.status().state, LocalServerState::Stopped);
        let error = manager
            .ensure_ready(&spec)
            .await
            .expect_err("no restart after shutdown");
        assert!(matches!(error, AiError::Runtime(_)), "{error:?}");
        assert_eq!(launcher.spawns(), 1);
    }

    #[tokio::test]
    async fn external_crash_is_reported_as_failed_without_restart() {
        let (manager, launcher) = manager(Behavior::Serve);
        manager
            .ensure_ready(&gguf_spec("/models/a.gguf"))
            .await
            .expect("ready");
        let pid = launcher.pids()[0];
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        for _ in 0..200 {
            if manager.status().state == LocalServerState::Failed {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let status = manager.status();
        assert_eq!(status.state, LocalServerState::Failed);
        assert!(status.message.contains("inesperadamente"), "{}", status.message);
        assert_eq!(launcher.spawns(), 1, "a crash is never restarted silently");
    }

    #[test]
    fn stale_server_is_reaped_and_a_foreign_process_survives() {
        let root = tempfile::tempdir().expect("tempdir");
        let mut dead_owner = Command::new("true").spawn().expect("owner");
        let dead_pid = dead_owner.id();
        let _ = dead_owner.wait();
        // argv carries the marker and the runtime root, like the real server command line.
        let mut stale = Command::new("sh")
            .args(["-c", "read _line", "llama-server"])
            .arg(root.path().join("bin").join("llama-server"))
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("stale server");
        let mut foreign = Command::new("sleep")
            .arg("300")
            .stdin(Stdio::null())
            .spawn()
            .expect("foreign process");
        let pidfile = pidfile_path(root.path());
        fs::write(&pidfile, format!("{} {dead_pid}", stale.id())).expect("pidfile");

        reap_orphaned_server(root.path());
        let _ = stale.wait();

        assert!(!process_is_alive(stale.id()), "stale server must be killed");
        assert!(!pidfile.exists(), "pidfile must be removed");
        assert!(process_is_alive(foreign.id()), "unrelated process must survive");
        let _ = foreign.kill();
        let _ = foreign.wait();
    }

    /// Writes an executable `/bin/sh` script that stands in for the server binary.
    #[cfg(unix)]
    fn write_fake_server(dir: &Path, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("fake-server");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("fake server script");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("executable script");
        path
    }

    fn fake_server_spec(executable: PathBuf, runtime_root: &Path) -> LocalServerSpec {
        LocalServerSpec {
            kind: LocalRuntimeKind::Gguf,
            executable,
            model_path: PathBuf::from("/models/gemma4.gguf"),
            context_tokens: 4096,
            runtime_root: runtime_root.to_path_buf(),
        }
    }

    fn process_manager() -> LocalServerManager {
        LocalServerManager::with_launcher(Arc::new(ProcessServerLauncher), fast_timings())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn startup_exit_quotes_the_server_stderr_tail() {
        let root = tempfile::tempdir().expect("tempdir");
        let executable = write_fake_server(
            root.path(),
            "echo 'Traceback (most recent call last):' >&2\necho '  ~^~~~' >&2\necho 'ValueError: Model type gemma4 not supported.' >&2\nexit 1",
        );
        let runtime_root = root.path().join("runtime");
        let manager = process_manager();

        let error = manager
            .ensure_ready(&fake_server_spec(executable, &runtime_root))
            .await
            .expect_err("early exit must fail");

        let expected = "El servidor local se cerró durante el arranque (código 1): Traceback (most recent call last): | ValueError: Model type gemma4 not supported.";
        assert!(
            matches!(&error, AiError::Runtime(msg) if msg == expected),
            "{error:?}"
        );
        assert!(manager.status().message.contains(expected));
        let log = fs::read_to_string(runtime_root.join(SERVER_LOG_NAME)).expect("server log");
        assert_eq!(
            log,
            "Traceback (most recent call last):\n  ~^~~~\nValueError: Model type gemma4 not supported.\n"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn startup_exit_keeps_the_message_when_the_new_log_is_empty() {
        let root = tempfile::tempdir().expect("tempdir");
        let executable = write_fake_server(root.path(), "exit 1");
        fs::write(root.path().join(SERVER_LOG_NAME), "stale line from a previous run\n")
            .expect("stale log");
        let manager = process_manager();

        let error = manager
            .ensure_ready(&fake_server_spec(executable, root.path()))
            .await
            .expect_err("early exit must fail");

        assert!(
            matches!(&error, AiError::Runtime(msg) if msg == "El servidor local se cerró durante el arranque (código 1)"),
            "{error:?}"
        );
        let log = fs::read_to_string(root.path().join(SERVER_LOG_NAME)).expect("server log");
        assert_eq!(log, "", "each launch truncates the previous log");
    }
}
