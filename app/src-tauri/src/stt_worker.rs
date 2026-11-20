//! Speech-to-text session worker. Runs `meeting_transcriber.py --gui-events`, mirrors its JSON
//! stdout events into a `SessionSnapshot`, emits `session-state`, and runs the post-meeting AI
//! step when the meeting file is finished.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use mt_core::{AppSettings, SttEngine};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::settings_store::SettingsStore;
use crate::stt_provisioning::{SttPaths, hidden_command, resolve_python_and_paths, whisper_size_name};

const SESSION_EVENT: &str = "session-state";
const SCRIPT_NAME: &str = "meeting_transcriber.py";
const MAC_AUDIO_HELPER_RELATIVE: &str =
    "macos/.build/MeetingTranscriberAudio.app/Contents/MacOS/MeetingTranscriberAudio";
const STOP_LINE: &[u8] = b"STOP\n";
const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);
const STDERR_TAIL_LINES: usize = 20;

/// Lifecycle phase of the current recording session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionPhase {
    #[default]
    Idle,
    LoadingEngine,
    Recording,
    Stopping,
    AnalyzingAi,
    Done,
    Error,
}

/// What the UI shows for the current session.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSnapshot {
    pub phase: SessionPhase,
    pub elapsed_seconds: f64,
    pub blocks_saved: u32,
    pub last_timestamp: Option<String>,
    pub note_path: Option<String>,
    pub message: String,
    pub ai_enabled: bool,
}

/// One JSON line printed by `meeting_transcriber.py --gui-events`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub(crate) enum GuiEvent {
    LoadingEngine { engine: String },
    Ready { file_path: Option<String>, engine: Option<String> },
    BlockSaved { timestamp: Option<String> },
    Stopping,
    Finished { file_path: Option<String> },
    Error { message: String },
}

/// The running worker process and the handles that talk to it.
struct ActiveWorker {
    child: Child,
    stdin: ChildStdin,
    started: Instant,
}

/// Tauri-managed state: the session snapshot, the single active worker, and the shutdown flag.
#[derive(Default)]
pub struct SttWorkerState {
    snapshot: Mutex<SessionSnapshot>,
    active: Mutex<Option<ActiveWorker>>,
    shutdown: AtomicBool,
}

impl SttWorkerState {
    /// Applies `change` to the snapshot, refreshes elapsed time while a worker runs, and returns
    /// the resulting snapshot. Never holds the active and snapshot locks at the same time.
    fn update(&self, change: impl FnOnce(&mut SessionSnapshot)) -> SessionSnapshot {
        let elapsed = self.elapsed_seconds();
        let mut snapshot = lock(&self.snapshot);
        change(&mut snapshot);
        if let Some(seconds) = elapsed {
            snapshot.elapsed_seconds = seconds;
        }
        snapshot.clone()
    }

    fn elapsed_seconds(&self) -> Option<f64> {
        lock(&self.active)
            .as_ref()
            .map(|worker| worker.started.elapsed().as_secs_f64())
    }

    fn current(&self) -> SessionSnapshot {
        self.update(|_| {})
    }

    /// Stops the worker gracefully on app exit: sends `STOP`, waits briefly, then kills it.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let Some(mut worker) = lock(&self.active).take() else {
            return;
        };
        let _ = worker.stdin.write_all(STOP_LINE);
        let _ = worker.stdin.flush();
        drop(worker.stdin);

        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match worker.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
                _ => break,
            }
        }
        let _ = worker.child.kill();
        let _ = worker.child.wait();
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Updates the snapshot and emits it to every window.
fn publish(app: &AppHandle, change: impl FnOnce(&mut SessionSnapshot)) {
    let snapshot = app.state::<SttWorkerState>().update(change);
    let _ = app.emit(SESSION_EVENT, snapshot);
}

/// Parses one stdout line. Non-JSON and unknown events yield `None`, so they are ignored.
pub(crate) fn parse_event_line(line: &str) -> Option<GuiEvent> {
    serde_json::from_str(line.trim()).ok()
}

/// Folds one worker event into the snapshot. `finished` picks the AI step when it is enabled.
pub(crate) fn apply_event(snapshot: &mut SessionSnapshot, event: &GuiEvent) {
    match event {
        GuiEvent::LoadingEngine { engine } => {
            snapshot.phase = SessionPhase::LoadingEngine;
            snapshot.message = format!("Cargando motor {engine}…");
        }
        GuiEvent::Ready { file_path, engine } => {
            snapshot.phase = SessionPhase::Recording;
            if let Some(path) = file_path {
                snapshot.note_path = Some(path.clone());
            }
            snapshot.message = match engine.as_deref().filter(|name| !name.is_empty()) {
                Some(name) => format!("Grabando con {name}."),
                None => "Grabando.".to_string(),
            };
        }
        GuiEvent::BlockSaved { timestamp } => {
            snapshot.blocks_saved = snapshot.blocks_saved.saturating_add(1);
            if let Some(stamp) = timestamp {
                snapshot.last_timestamp = Some(stamp.clone());
            }
            snapshot.message = "Bloque guardado.".to_string();
        }
        GuiEvent::Stopping => {
            snapshot.phase = SessionPhase::Stopping;
            snapshot.message = "Deteniendo la grabación…".to_string();
        }
        GuiEvent::Finished { file_path } => {
            // The worker can report an error and still emit `finished`; keep the error visible.
            if snapshot.phase == SessionPhase::Error {
                return;
            }
            if let Some(path) = file_path {
                snapshot.note_path = Some(path.clone());
            }
            if snapshot.ai_enabled && snapshot.note_path.is_some() {
                snapshot.phase = SessionPhase::AnalyzingAi;
                snapshot.message = "Generando informe IA…".to_string();
            } else {
                snapshot.phase = SessionPhase::Done;
                snapshot.message = "Reunión guardada.".to_string();
            }
        }
        GuiEvent::Error { message } => {
            snapshot.phase = SessionPhase::Error;
            snapshot.message = message.clone();
        }
    }
}

/// Environment for the worker. Explicit values take precedence over the repo `.env`, because
/// `python-dotenv` does not override variables that are already set.
pub(crate) fn worker_env(settings: &AppSettings, paths: &SttPaths) -> Vec<(&'static str, String)> {
    let flag = |on: bool| (if on { "1" } else { "0" }).to_string();
    let whisper_size = whisper_size_name(settings.whisper_model).to_string();
    let mut env = vec![
        ("OBSIDIAN_DIR", settings.output_dir.to_string_lossy().into_owned()),
        ("TRANSCRIPTION_ENGINE", engine_setting(settings.stt_engine).to_string()),
        ("WHISPER_MODEL", whisper_size.clone()),
        ("WINDOWS_WHISPER_MODEL", whisper_size),
        ("TRANSCRIPTION_LANGUAGE", settings.transcription_language.clone()),
        ("SPEAKER_MARKERS", flag(settings.speaker_markers)),
        ("WHISPER_VAD_FILTER", flag(settings.whisper_vad_filter)),
        ("WINDOWS_MICROPHONE", settings.windows_microphone.clone()),
        ("WINDOWS_SPEAKER", settings.windows_speaker.clone()),
        ("WINDOWS_BUFFER_SECONDS", settings.windows_buffer_seconds.to_string()),
        ("MAC_SYSTEM_GAIN", settings.mac_system_gain.to_string()),
        ("MAC_MICROPHONE_GAIN", settings.mac_microphone_gain.to_string()),
        ("MT_MODELS_DIR", paths.models_dir.to_string_lossy().into_owned()),
        // Keeps the JSON stream unbuffered and makes emoji logs safe on Windows pipes.
        ("PYTHONUNBUFFERED", "1".to_string()),
        ("PYTHONIOENCODING", "utf-8".to_string()),
    ];
    let helper = paths.script_dir.join(MAC_AUDIO_HELPER_RELATIVE);
    if helper.is_file() {
        env.push(("MAC_AUDIO_HELPER", helper.to_string_lossy().into_owned()));
    }
    env
}

/// The raw setting, not the resolved engine: `transcription_engines.py` resolves `auto` itself.
fn engine_setting(engine: SttEngine) -> &'static str {
    match engine {
        SttEngine::Auto => "auto",
        SttEngine::Whisper => "whisper",
        SttEngine::Whistle => "whistle",
    }
}

#[tauri::command]
pub fn start_session(
    app: AppHandle,
    state: State<'_, SttWorkerState>,
    store: State<'_, SettingsStore>,
) -> Result<SessionSnapshot, String> {
    if state.shutdown.load(Ordering::SeqCst) {
        return Err("La aplicación se está cerrando.".to_string());
    }
    let settings = store.get();
    let paths = resolve_python_and_paths(&app)?;
    let script = paths.script_dir.join(SCRIPT_NAME);
    if !script.is_file() {
        return Err(format!("No se encontró {}", script.display()));
    }

    let mut active = lock(&state.active);
    if active.is_some() {
        return Err("Ya hay una sesión activa.".to_string());
    }

    let mut child = hidden_command(&paths.python_exe)
        .arg(&script)
        .arg("--gui-events")
        .current_dir(&paths.script_dir)
        .envs(worker_env(&settings, &paths))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("No se pudo iniciar el motor de transcripción: {e}"))?;

    let (Some(stdin), Some(stdout), Some(stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        let _ = child.kill();
        return Err("No se pudieron conectar las tuberías del motor de transcripción.".to_string());
    };

    let pid = child.id();
    let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
    *active = Some(ActiveWorker {
        child,
        stdin,
        started: Instant::now(),
    });
    let snapshot = SessionSnapshot {
        phase: SessionPhase::LoadingEngine,
        message: "Preparando el motor…".to_string(),
        ai_enabled: settings.ai_enabled,
        ..SessionSnapshot::default()
    };
    *lock(&state.snapshot) = snapshot.clone();
    drop(active);
    let _ = app.emit(SESSION_EVENT, snapshot.clone());

    let tail = Arc::clone(&stderr_tail);
    thread::spawn(move || drain_stderr(stderr, &tail));
    let reader_app = app.clone();
    thread::spawn(move || read_events(reader_app, stdout, stderr_tail, pid));
    Ok(snapshot)
}

#[tauri::command]
pub fn stop_session(app: AppHandle, state: State<'_, SttWorkerState>) -> Result<SessionSnapshot, String> {
    let mut active = lock(&state.active);
    let phase = lock(&state.snapshot).phase;
    if phase == SessionPhase::Stopping {
        drop(active);
        return Ok(state.current());
    }
    let worker = active
        .as_mut()
        .ok_or_else(|| "No hay una sesión activa.".to_string())?;
    if !matches!(phase, SessionPhase::LoadingEngine | SessionPhase::Recording) {
        return Err("La sesión no está grabando.".to_string());
    }
    worker
        .stdin
        .write_all(STOP_LINE)
        .and_then(|()| worker.stdin.flush())
        .map_err(|e| format!("No se pudo detener el motor de transcripción: {e}"))?;
    drop(active);

    let snapshot = state.update(|s| {
        s.phase = SessionPhase::Stopping;
        s.message = "Deteniendo la grabación…".to_string();
    });
    let _ = app.emit(SESSION_EVENT, snapshot.clone());
    Ok(snapshot)
}

#[tauri::command]
pub fn session_state(state: State<'_, SttWorkerState>) -> SessionSnapshot {
    state.current()
}

fn drain_stderr(stderr: ChildStderr, tail: &Mutex<VecDeque<String>>) {
    for line in BufReader::new(stderr).lines().map_while(Result::ok) {
        let line = line.trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        let mut lines = lock(tail);
        if lines.len() == STDERR_TAIL_LINES {
            lines.pop_front();
        }
        lines.push_back(line);
    }
}

/// Reader thread: applies each stdout event, runs the AI step on `finished`, and finalizes the
/// worker when stdout closes.
fn read_events(app: AppHandle, stdout: ChildStdout, stderr_tail: Arc<Mutex<VecDeque<String>>>, pid: u32) {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let Some(event) = parse_event_line(&line) else {
            continue;
        };
        publish(&app, |s| apply_event(s, &event));
        if let GuiEvent::Finished { file_path: Some(path) } = event {
            run_ai_step(&app, &path);
        }
    }
    finish_worker(&app, pid, &stderr_tail);
}

/// Runs the post-meeting report for the finished meeting file when the session is in
/// `AnalyzingAi`, then settles on `Done`.
fn run_ai_step(app: &AppHandle, file_path: &str) {
    let state = app.state::<SttWorkerState>();
    if lock(&state.snapshot).phase != SessionPhase::AnalyzingAi {
        return;
    }
    if state.shutdown.load(Ordering::SeqCst) {
        publish(app, |s| {
            s.phase = SessionPhase::Done;
            s.message = "Reunión guardada; el informe IA se omitió al cerrar.".to_string();
        });
        return;
    }

    // Runs on the reader thread, not on the async runtime, so blocking here is fine.
    let result = tauri::async_runtime::block_on(crate::report_pipeline::run_post_meeting_report(
        app, file_path,
    ));
    publish(app, |s| {
        s.phase = SessionPhase::Done;
        s.message = match &result {
            Ok(_) => "Informe IA listo.".to_string(),
            Err(error) => format!("Reunión guardada; el informe IA falló: {error}"),
        };
    });
}

/// Called once stdout closes. Marks an unexpected exit as an error, then reaps the process.
fn finish_worker(app: &AppHandle, pid: u32, stderr_tail: &Mutex<VecDeque<String>>) {
    let tail = lock(stderr_tail).iter().cloned().collect::<Vec<_>>().join(" | ");
    publish(app, |s| {
        if !matches!(s.phase, SessionPhase::Done | SessionPhase::Error) {
            s.phase = SessionPhase::Error;
            s.message = if tail.is_empty() {
                "El motor de transcripción terminó inesperadamente.".to_string()
            } else {
                format!("El motor de transcripción terminó inesperadamente: {tail}")
            };
        }
    });

    let state = app.state::<SttWorkerState>();
    let worker = {
        let mut active = lock(&state.active);
        if active.as_ref().is_some_and(|worker| worker.child.id() == pid) {
            active.take()
        } else {
            None
        }
    };
    if let Some(mut worker) = worker {
        drop(worker.stdin);
        let _ = worker.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn event(line: &str) -> GuiEvent {
        parse_event_line(line).unwrap_or_else(|| panic!("no se pudo leer: {line}"))
    }

    fn with_ai(enabled: bool) -> SessionSnapshot {
        SessionSnapshot {
            ai_enabled: enabled,
            ..SessionSnapshot::default()
        }
    }

    #[test]
    fn parses_each_gui_event_line() {
        assert_eq!(
            event(r#"{"event":"loading_engine","engine":"Whisper small"}"#),
            GuiEvent::LoadingEngine {
                engine: "Whisper small".to_string()
            }
        );
        assert_eq!(
            event(r#"{"event":"ready","file_path":"/vault/a.md","engine":"Cactus Whistle"}"#),
            GuiEvent::Ready {
                file_path: Some("/vault/a.md".to_string()),
                engine: Some("Cactus Whistle".to_string()),
            }
        );
        assert_eq!(
            event(r#"{"event":"block_saved","timestamp":"2025-01-02T10:00:00","elapsed":1.2,"pending":0.0}"#),
            GuiEvent::BlockSaved {
                timestamp: Some("2025-01-02T10:00:00".to_string())
            }
        );
        assert_eq!(event(r#"{"event":"stopping"}"#), GuiEvent::Stopping);
        assert_eq!(
            event(r#"{"event":"finished","file_path":"/vault/a.md"}"#),
            GuiEvent::Finished {
                file_path: Some("/vault/a.md".to_string())
            }
        );
        assert_eq!(
            event(r#"{"event":"error","message":"sin micrófono"}"#),
            GuiEvent::Error {
                message: "sin micrófono".to_string()
            }
        );
    }

    #[test]
    fn ignores_unknown_events_and_non_json_lines() {
        assert_eq!(parse_event_line(r#"{"event":"future_event"}"#), None);
        assert_eq!(parse_event_line("🧠 Cargando motor local"), None);
        assert_eq!(parse_event_line(""), None);
    }

    #[test]
    fn block_events_count_and_keep_the_latest_timestamp() {
        let mut s = SessionSnapshot::default();
        apply_event(&mut s, &event(r#"{"event":"block_saved","timestamp":"10:00:00"}"#));
        apply_event(&mut s, &event(r#"{"event":"block_saved","timestamp":"10:00:30"}"#));
        assert_eq!(s.blocks_saved, 2);
        assert_eq!(s.last_timestamp.as_deref(), Some("10:00:30"));
        assert_eq!(s.phase, SessionPhase::Idle);
    }

    #[test]
    fn startup_then_recording_then_stop_follows_the_worker_order() {
        let mut s = with_ai(false);
        apply_event(&mut s, &event(r#"{"event":"loading_engine","engine":"Whisper"}"#));
        assert_eq!(s.phase, SessionPhase::LoadingEngine);
        assert_eq!(s.message, "Cargando motor Whisper…");

        apply_event(&mut s, &event(r#"{"event":"ready","file_path":"/vault/n.md","engine":"Whisper"}"#));
        assert_eq!(s.phase, SessionPhase::Recording);
        assert_eq!(s.note_path.as_deref(), Some("/vault/n.md"));

        apply_event(&mut s, &event(r#"{"event":"stopping"}"#));
        assert_eq!(s.phase, SessionPhase::Stopping);

        apply_event(&mut s, &event(r#"{"event":"finished","file_path":"/vault/n.md"}"#));
        assert_eq!(s.phase, SessionPhase::Done);
    }

    #[test]
    fn finished_with_ai_enabled_waits_for_the_report_step() {
        let mut s = with_ai(true);
        apply_event(&mut s, &event(r#"{"event":"ready","file_path":"/vault/n.md","engine":"X"}"#));
        apply_event(&mut s, &event(r#"{"event":"finished","file_path":"/vault/n.md"}"#));
        assert_eq!(s.phase, SessionPhase::AnalyzingAi);
        assert_eq!(s.note_path.as_deref(), Some("/vault/n.md"));
    }

    #[test]
    fn finished_after_an_error_keeps_the_error_and_skips_the_ai_step() {
        let mut s = with_ai(true);
        apply_event(&mut s, &event(r#"{"event":"error","message":"sin micrófono"}"#));
        apply_event(&mut s, &event(r#"{"event":"finished","file_path":"/vault/n.md"}"#));
        assert_eq!(s.phase, SessionPhase::Error);
        assert_eq!(s.message, "sin micrófono");
    }

    #[test]
    fn finished_without_a_file_path_does_not_start_the_ai_step() {
        let mut s = with_ai(true);
        apply_event(&mut s, &event(r#"{"event":"finished"}"#));
        assert_eq!(s.phase, SessionPhase::Done);
    }

    #[test]
    fn error_event_moves_to_error_with_the_worker_message() {
        let mut s = with_ai(false);
        apply_event(&mut s, &event(r#"{"event":"error","message":"No se encontró el modelo"}"#));
        assert_eq!(s.phase, SessionPhase::Error);
        assert_eq!(s.message, "No se encontró el modelo");
    }

    #[test]
    fn snapshot_serializes_with_camel_case_keys_and_snake_phase_names() {
        let mut s = with_ai(true);
        s.phase = SessionPhase::AnalyzingAi;
        s.blocks_saved = 3;
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["phase"], "analyzing_ai");
        assert_eq!(json["blocksSaved"], 3);
        assert_eq!(json["aiEnabled"], true);
        assert!(json.get("notePath").is_some());
        assert!(json.get("elapsedSeconds").is_some());
    }

    #[test]
    fn worker_env_passes_the_selected_settings_and_the_raw_engine() {
        let settings = AppSettings {
            stt_engine: SttEngine::Whistle,
            whisper_model: mt_core::WhisperSize::Medium,
            speaker_markers: true,
            whisper_vad_filter: false,
            transcription_language: "en".to_string(),
            output_dir: PathBuf::from("/vault"),
            windows_buffer_seconds: 2.5,
            ..AppSettings::default()
        };
        let paths = SttPaths {
            script_dir: PathBuf::from("/nonexistent-app-root"),
            python_exe: PathBuf::from("/py"),
            models_dir: PathBuf::from("/models"),
            runtime_dir: PathBuf::from("/runtime"),
        };

        let env: std::collections::HashMap<_, _> = worker_env(&settings, &paths).into_iter().collect();
        assert_eq!(env["OBSIDIAN_DIR"], "/vault");
        assert_eq!(env["TRANSCRIPTION_ENGINE"], "whistle");
        assert_eq!(env["WHISPER_MODEL"], "medium");
        assert_eq!(env["WINDOWS_WHISPER_MODEL"], "medium");
        assert_eq!(env["TRANSCRIPTION_LANGUAGE"], "en");
        assert_eq!(env["SPEAKER_MARKERS"], "1");
        assert_eq!(env["WHISPER_VAD_FILTER"], "0");
        assert_eq!(env["WINDOWS_BUFFER_SECONDS"], "2.5");
        assert_eq!(env["MT_MODELS_DIR"], "/models");
        assert!(!env.contains_key("MAC_AUDIO_HELPER"));
    }
}
