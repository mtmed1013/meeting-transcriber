import json
import os
import time
import queue
import platform
import struct
import subprocess
import sys
import threading
import signal
import select
from uuid import uuid4
from collections import deque
from contextlib import ExitStack, nullcontext
import numpy as np
import sounddevice as sd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from transcription_engines import create_transcriber, resolve_engine

# ================= CONFIG =================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


def _env_bool(name, default=True):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name, default):
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) and value >= 0 else default


def _macos_version_at_least(required_major, required_minor=0):
    if sys.platform != "darwin":
        return False
    try:
        version_parts = tuple(
            int(part) for part in platform.mac_ver()[0].split(".") if part
        )
    except ValueError:
        return False
    return version_parts >= (required_major, required_minor)

DEFAULT_OBSIDIAN_DIR = (
    os.path.expanduser("~/Documents/Obsidian")
    if os.name == "nt"
    else os.path.expanduser(
        "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian"
    )
)
OBSIDIAN_DIR = os.environ.get("OBSIDIAN_DIR", "").strip() or DEFAULT_OBSIDIAN_DIR
NOTE_TITLE = f"Reunión - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}-{uuid4().hex[:8]}.md"

SAMPLE_RATE = 16000
BLOCK_SECONDS = 30  # cada cuántos segundos se transcribe
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "").strip() or "small"
WINDOWS_WHISPER_MODEL = (
    os.environ.get("WINDOWS_WHISPER_MODEL", "").strip() or "small"
)
TRANSCRIPTION_ENGINE = os.environ.get("TRANSCRIPTION_ENGINE", "auto").strip() or "auto"
TRANSCRIPTION_LANGUAGE = (
    os.environ.get("TRANSCRIPTION_LANGUAGE", "").strip().lower()
    or os.environ.get("WHISPER_LANGUAGE", "es").strip().lower()
    or "es"
)
TRANSCRIPTION_LANGUAGE = (
    None if TRANSCRIPTION_LANGUAGE == "auto" else TRANSCRIPTION_LANGUAGE
)
SPEAKER_MARKERS = _env_bool("SPEAKER_MARKERS", default=False)
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = _env_bool("WHISPER_VAD_FILTER", default=True)
WINDOWS_MICROPHONE = os.environ.get("WINDOWS_MICROPHONE", "").strip()
WINDOWS_SPEAKER = os.environ.get("WINDOWS_SPEAKER", "").strip()
WINDOWS_BUFFER_SECONDS = max(0.2, _env_float("WINDOWS_BUFFER_SECONDS", 1.0))
MAC_AUDIO_HELPER = os.environ.get("MAC_AUDIO_HELPER", "").strip() or os.path.join(
    PROJECT_DIR,
    "macos",
    ".build",
    "MeetingTranscriberAudio.app",
    "Contents",
    "MacOS",
    "MeetingTranscriberAudio",
)
MAC_AUDIO_MIX_FRAMES = 320  # 20 ms a 16 kHz
MAC_AUDIO_JITTER_SECONDS = 0.35
MAC_AUDIO_DIAGNOSTIC_SECONDS = 15
MAC_SHUTDOWN_PROGRESS_SECONDS = 3
MAC_AUDIO_GRACEFUL_STOP_SECONDS = 10
MAC_AUDIO_PROCESS_STOP_TIMEOUT_SECONDS = 5
MAC_AUDIO_FORCE_DRAIN_SECONDS = 2
MAC_SYSTEM_GAIN = _env_float("MAC_SYSTEM_GAIN", 1.0)
MAC_MICROPHONE_GAIN = _env_float("MAC_MICROPHONE_GAIN", 1.0)
MAC_NATIVE_MICROPHONE = _macos_version_at_least(15)
last_voice_signature = None

# ==========================================


audio_queue = queue.Queue()
running = True

# ================= --gui-events (JSON lines for a GUI wrapper) =================
GUI_EVENTS = False
_GUI_EVENT_STREAM = None
_GUI_EVENT_LOCK = threading.Lock()


def format_gui_event(event, **fields):
    return json.dumps({"event": event, **fields})


def emit_gui_event(event, **fields):
    if not GUI_EVENTS or _GUI_EVENT_STREAM is None:
        return
    line = format_gui_event(event, **fields)
    with _GUI_EVENT_LOCK:
        _GUI_EVENT_STREAM.write(line + "\n")
        _GUI_EVENT_STREAM.flush()


def _configure_gui_events(args):
    """Stdout carries only JSON events; human logs move to stderr."""
    global GUI_EVENTS, _GUI_EVENT_STREAM
    if "--gui-events" not in args:
        return
    GUI_EVENTS = True
    _GUI_EVENT_STREAM = sys.stdout
    sys.stdout = sys.stderr


def _announce_stop_request(stop_event):
    if not stop_event.is_set():
        emit_gui_event("stopping")


def start_gui_stdin_listener(stop_event, on_stop=None, stream=None):
    """Daemon thread: a STOP line or EOF requests the same graceful stop as Ctrl+C."""
    source = sys.stdin if stream is None else stream

    def request_stop():
        if on_stop is not None:
            on_stop()
        else:
            stop_event.set()

    def listen():
        while True:
            try:
                line = source.readline()
            except (OSError, ValueError):
                line = ""
            if not line or line.strip() == "STOP":
                request_stop()
                return

    thread = threading.Thread(target=listen, name="gui-events-stdin", daemon=True)
    thread.start()
    return thread


def _output_file_path():
    override = os.environ.get("MT_OUTPUT_FILE", "").strip()
    return override or os.path.join(OBSIDIAN_DIR, NOTE_TITLE)


_configure_gui_events(sys.argv[1:])
try:
    RESOLVED_TRANSCRIPTION_ENGINE = resolve_engine(TRANSCRIPTION_ENGINE)
    emit_gui_event("loading_engine", engine=RESOLVED_TRANSCRIPTION_ENGINE)
    print(f"🧠 Cargando motor local: {RESOLVED_TRANSCRIPTION_ENGINE}...")
    transcriber = create_transcriber(
        TRANSCRIPTION_ENGINE,
        language=TRANSCRIPTION_LANGUAGE,
        whisper_model=WHISPER_MODEL,
        windows_whisper_model=WINDOWS_WHISPER_MODEL,
        vad_filter=WHISPER_VAD_FILTER,
        beam_size=WHISPER_BEAM_SIZE,
    )
except Exception as error:
    emit_gui_event("error", message=str(error))
    raise
print(f"✅ {transcriber.name} listo; iniciando captura de audio")

_librosa_missing_warned = False


def voice_signature(audio, sr):
    global _librosa_missing_warned
    try:
        import librosa
    except ImportError:
        if not _librosa_missing_warned:
            _librosa_missing_warned = True
            print("⚠️ librosa no está instalado; se desactiva la detección de cambio de hablante")
        return None
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
    return np.mean(mfcc, axis=1)

def speaker_changed(current_sig, threshold=25.0):
    global last_voice_signature

    if current_sig is None:
        return False
    if last_voice_signature is None:
        last_voice_signature = current_sig
        return False

    distance = np.linalg.norm(current_sig - last_voice_signature)
    last_voice_signature = current_sig

    return distance > threshold

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def write_header(file_path):
    with open(file_path, "x", encoding="utf-8") as f:
        f.write(f"# 🧠 Reunión\n\n")
        f.write(f"🕒 Inicio: {datetime.now().strftime('%H:%M')}\n\n")
        f.write("---\n\n")

def append_text(file_path, text, audio_time=None):
    timestamp = (audio_time or datetime.now()).strftime("%H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"**[{timestamp}]**\n{text.strip()}\n\n")


def flush_buffer(buffer, file_path, audio_time=None):
    if not buffer:
        return

    audio = np.concatenate(buffer, axis=0).flatten()
    if len(audio) < SAMPLE_RATE * 1.5 and transcriber.name.startswith("Whisper"):
        if os.name == "nt":
            audio = np.pad(audio, (0, int(SAMPLE_RATE * 1.5) - len(audio)))
        else:
            print("⚠️ Audio muy corto, se omite")
            return

    started = time.monotonic()
    print(
        f"🧠 Transcribiendo {len(audio) / SAMPLE_RATE:.1f}s "
        f"con {transcriber.name}..."
    )
    text = transcriber.transcribe(audio)
    if text and SPEAKER_MARKERS and speaker_changed(voice_signature(audio, SAMPLE_RATE)):
        append_text(file_path, "— 🔄 Cambio de hablante estimado —", audio_time)

    if text:
        append_text(file_path, text, audio_time)
        print("📝 Texto guardado")
    else:
        print("⚠️ Sin voz o texto detectado en este bloque")
    with audio_queue.mutex:
        pending = sum(
            len(chunk[1]) if isinstance(chunk, tuple) else len(chunk)
            for chunk in audio_queue.queue
        ) / SAMPLE_RATE
    elapsed = time.monotonic() - started
    print(
        f"📊 {transcriber.name}: {elapsed:.1f}s de procesamiento; "
        f"audio pendiente={pending:.1f}s"
    )
    if text:
        emit_gui_event(
            "block_saved",
            timestamp=(audio_time or datetime.now()).isoformat(),
            elapsed=round(elapsed, 3),
            pending=round(pending, 3),
        )
    if elapsed > len(audio) / SAMPLE_RATE:
        print("⚠️ El motor procesa más lento que la captura; se está acumulando retraso")


def _queued_audio_seconds():
    with audio_queue.mutex:
        packets = tuple(audio_queue.queue)
    samples = sum(
        len(packet[-1] if isinstance(packet, tuple) else packet)
        for packet in packets
    )
    return samples / SAMPLE_RATE


def _macos_shutdown_progress_loop(
    stop_event,
    force_stop_event,
    capture_done,
    shutdown_state,
    finish_event,
):
    while not finish_event.wait(0.5):
        if not stop_event.is_set() or shutdown_state.get("phase") == "complete":
            continue

        now = time.monotonic()
        if now < shutdown_state.get("next_progress_at", 0):
            continue

        queued_seconds = _queued_audio_seconds()
        shutdown_elapsed = now - shutdown_state.get("requested_at", now)
        if (
            capture_done is not None
            and not capture_done.is_set()
            and shutdown_elapsed >= MAC_AUDIO_GRACEFUL_STOP_SECONDS
            and not force_stop_event.is_set()
        ):
            force_stop_event.set()
            print(
                "⚠️ La captura macOS no cerró en "
                f"{MAC_AUDIO_GRACEFUL_STOP_SECONDS}s; se fuerza el cierre del "
                "capturador y se transcribe el audio que ya llegó."
            )

        inference_started = shutdown_state.get("inference_started_at")
        if inference_started is not None:
            elapsed = now - inference_started
            block_seconds = shutdown_state.get("current_block_seconds", 0.0)
            print(
                f"⏳ {transcriber.name} sigue transcribiendo un bloque de "
                f"{block_seconds:.1f}s; lleva {elapsed:.0f}s; "
                f"audio adicional pendiente≈{queued_seconds:.1f}s"
            )
        elif capture_done is not None and not capture_done.is_set():
            elapsed = now - shutdown_state.get("requested_at", now)
            buffered_seconds = shutdown_state.get("buffered_audio_seconds", 0.0)
            print(
                f"⏳ Cierre macOS: deteniendo el capturador; "
                f"{elapsed:.0f}s transcurridos; "
                f"audio pendiente≈{queued_seconds + buffered_seconds:.1f}s"
            )
        else:
            elapsed = now - shutdown_state.get("requested_at", now)
            buffered_seconds = shutdown_state.get("buffered_audio_seconds", 0.0)
            print(
                f"⏳ Cierre macOS: vaciando audio; {elapsed:.0f}s transcurridos; "
                f"audio pendiente≈{queued_seconds + buffered_seconds:.1f}s"
            )

        shutdown_state["next_progress_at"] = now + MAC_SHUTDOWN_PROGRESS_SECONDS


def transcriber_loop(
    file_path,
    stop_event=None,
    capture_done=None,
    shutdown_state=None,
):
    if capture_done is not None and os.name == "nt":
        return windows_transcriber_loop(file_path, capture_done)

    buffer = []
    buffered = 0
    target = SAMPLE_RATE * BLOCK_SECONDS
    interrupted = False
    pending_data = None
    print(f"⏱️ Se transcriben bloques de {BLOCK_SECONDS}s de audio recibido")

    def flush_pending_audio():
        if shutdown_state is not None:
            now = time.monotonic()
            shutdown_state["inference_started_at"] = now
            shutdown_state["current_block_seconds"] = buffered / SAMPLE_RATE
            if stop_event is not None and stop_event.is_set():
                shutdown_state["phase"] = "transcribing"
                shutdown_state["phase_started_at"] = now
        try:
            flush_buffer(buffer, file_path)
        finally:
            if shutdown_state is not None:
                shutdown_state["inference_started_at"] = None
                shutdown_state["current_block_seconds"] = 0.0
                shutdown_state["buffered_audio_seconds"] = 0.0
                if stop_event is not None and stop_event.is_set():
                    shutdown_state["phase"] = "draining_audio"

    while True:
        try:
            if buffered == target:
                flush_pending_audio()
                buffer.clear()
                buffered = 0
                continue

            capture_finished = (
                capture_done.is_set()
                if capture_done is not None
                else stop_event is not None and stop_event.is_set()
            )
            if (
                capture_finished
                and shutdown_state is not None
                and stop_event is not None
                and stop_event.is_set()
                and not shutdown_state.get("capture_done_reported")
            ):
                shutdown_state["phase"] = "draining_audio"
                shutdown_state["capture_done_reported"] = True
                pending_seconds = (
                    _queued_audio_seconds()
                    + buffered / SAMPLE_RATE
                    + (len(pending_data) if pending_data is not None else 0) / SAMPLE_RATE
                )
                print(
                    "✅ Captura macOS detenida; "
                    f"quedan {pending_seconds:.1f}s de audio por transcribir"
                )
            if capture_finished and audio_queue.empty() and pending_data is None:
                break

            if (
                shutdown_state is not None
                and stop_event is not None
                and stop_event.is_set()
                and capture_done is not None
                and capture_done.is_set()
            ):
                shutdown_state["phase"] = "draining_audio"

            if pending_data is None:
                data = audio_queue.get(timeout=0.2)
                if isinstance(data, tuple):
                    data = data[-1]
                pending_data = np.asarray(data).reshape(-1)

            count = min(len(pending_data), target - buffered)
            buffer.append(pending_data[:count])
            buffered += count
            if shutdown_state is not None:
                shutdown_state["buffered_audio_seconds"] = buffered / SAMPLE_RATE
            pending_data = pending_data[count:] if count < len(pending_data) else None
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            if not interrupted:
                print("\n🧠 Finalizando reunión, guardando todo el audio pendiente...")
                interrupted = True
                if stop_event is not None:
                    stop_event.set()
            continue

    if buffer:
        if shutdown_state is not None and stop_event is not None and stop_event.is_set():
            shutdown_state["phase"] = "transcribing"
            shutdown_state["current_block_seconds"] = buffered / SAMPLE_RATE
            shutdown_state["inference_started_at"] = time.monotonic()
            print(
                "🧠 Transcribiendo el último bloque de "
                f"{buffered / SAMPLE_RATE:.1f}s..."
            )
        flush_pending_audio()
    if shutdown_state is not None and stop_event is not None and stop_event.is_set():
        shutdown_state["phase"] = "complete"


def windows_transcriber_loop(file_path, capture_done):
    """Fixed sample-count batches; consume the complete capture tail at shutdown."""
    buffer = []
    buffered = 0
    target = SAMPLE_RATE * BLOCK_SECONDS
    audio_time = None
    wall_origin = datetime.now()
    mono_origin = time.monotonic()
    processed = 0
    processing_seconds = 0.0
    print(
        f"⏱️ Primer bloque: {BLOCK_SECONDS}s de audio recibido, "
        f"más el tiempo de {transcriber.name}"
    )
    while not (capture_done.is_set() and audio_queue.empty()):
        try:
            data = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if isinstance(data, tuple):
            packet_start, data = data
        else:
            packet_start = time.monotonic() - len(data) / SAMPLE_RATE
        while len(data):
            if audio_time is None:
                audio_time = wall_origin + timedelta(seconds=packet_start - mono_origin)
            count = min(len(data), target - buffered)
            buffer.append(data[:count])
            buffered += count
            data = data[count:]
            packet_start += count / SAMPLE_RATE
            if buffered == target:
                started = time.monotonic()
                flush_buffer(buffer, file_path, audio_time)
                processing_seconds += time.monotonic() - started
                processed += buffered
                buffer.clear()
                buffered = 0
                audio_time = None
                if capture_done.is_set():
                    with audio_queue.mutex:
                        pending = sum(len(x[1]) if isinstance(x, tuple) else len(x)
                                      for x in audio_queue.queue) / SAMPLE_RATE
                    estimate = pending * processing_seconds / (processed / SAMPLE_RATE)
                    print(f"💾 Cierre: quedan {pending:.1f}s de audio; espera aproximada {estimate:.0f}s")
    if buffer:
        flush_buffer(buffer, file_path, audio_time)


def _mono_audio(data):
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _empty_audio_level_stats():
    return {"sum_squares": 0.0, "sample_count": 0, "peak": 0.0}


def _accumulate_audio_level(stats, samples):
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if not values.size:
        return

    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    stats["sum_squares"] += float(np.dot(values, values))
    stats["sample_count"] += int(values.size)
    stats["peak"] = max(stats["peak"], float(np.max(np.abs(values))))


_NANOSECONDS_PER_SAMPLE = 1_000_000_000 / SAMPLE_RATE


class _TimestampedAudioBuffer:
    """Small timestamped buffer used to align independent audio callbacks."""

    def __init__(self):
        self._chunks = deque()

    @property
    def first_start_ns(self):
        return self._chunks[0][0] if self._chunks else None

    @property
    def end_ns(self):
        if not self._chunks:
            return None
        start_ns, samples = self._chunks[-1]
        return start_ns + int(round(len(samples) * _NANOSECONDS_PER_SAMPLE))

    def append(self, start_ns, samples):
        samples = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if samples.size:
            self._chunks.append((int(start_ns), samples))

    def read_window(self, start_ns, frames):
        end_ns = start_ns + int(round(frames * _NANOSECONDS_PER_SAMPLE))
        output = np.zeros(frames, dtype=np.float32)
        covered = np.zeros(frames, dtype=bool)

        for chunk_start_ns, samples in self._chunks:
            chunk_end_ns = chunk_start_ns + int(
                round(len(samples) * _NANOSECONDS_PER_SAMPLE)
            )
            if chunk_end_ns <= start_ns:
                continue
            if chunk_start_ns >= end_ns:
                break

            overlap_start_ns = max(start_ns, chunk_start_ns)
            overlap_end_ns = min(end_ns, chunk_end_ns)
            output_start = int(
                round((overlap_start_ns - start_ns) / _NANOSECONDS_PER_SAMPLE)
            )
            sample_start = int(
                round((overlap_start_ns - chunk_start_ns) / _NANOSECONDS_PER_SAMPLE)
            )
            count = int(
                round((overlap_end_ns - overlap_start_ns) / _NANOSECONDS_PER_SAMPLE)
            )
            count = min(count, frames - output_start, len(samples) - sample_start)

            if count > 0:
                output[output_start:output_start + count] = samples[
                    sample_start:sample_start + count
                ]
                covered[output_start:output_start + count] = True

        self.discard_before(end_ns)
        return output, int(covered.sum())

    def discard_before(self, cutoff_ns):
        while self._chunks:
            start_ns, samples = self._chunks[0]
            end_ns = start_ns + int(round(len(samples) * _NANOSECONDS_PER_SAMPLE))
            if end_ns <= cutoff_ns:
                self._chunks.popleft()
                continue
            if start_ns < cutoff_ns:
                samples_to_drop = int(
                    round((cutoff_ns - start_ns) / _NANOSECONDS_PER_SAMPLE)
                )
                samples_to_drop = min(samples_to_drop, len(samples))
                if samples_to_drop:
                    self._chunks[0] = (
                        start_ns + int(round(samples_to_drop * _NANOSECONDS_PER_SAMPLE)),
                        samples[samples_to_drop:],
                    )
            break


def _read_exact(stream, size, force_stop_event=None):
    """Read one framed-pipe value, but bound waits after forced shutdown."""
    chunks = []
    remaining = size
    force_started_at = None

    while remaining:
        if force_stop_event is not None:
            if force_stop_event.is_set():
                if force_started_at is None:
                    force_started_at = time.monotonic()
                elif (
                    time.monotonic() - force_started_at
                    >= MAC_AUDIO_FORCE_DRAIN_SECONDS
                ):
                    return None

            try:
                readable, _, _ = select.select([stream], [], [], 0.1)
            except (OSError, ValueError):
                return None
            if not readable:
                continue

        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def _mac_audio_stderr(process, stop_event, error_queue):
    try:
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "MEETING_AUDIO_READY":
                print("🔊 Captura de ScreenCaptureKit iniciada")
            elif line == "MEETING_AUDIO_MIC_ENABLED":
                print("🎤 Captura nativa del micrófono macOS habilitada")
            elif line == "MEETING_AUDIO_MIC_READY":
                print("🎤 Llegaron paquetes iniciales del micrófono macOS")
            elif line == "MEETING_AUDIO_SAMPLE_READY":
                print("🔊 Llegaron paquetes iniciales del sistema macOS")
            elif line.startswith("ERROR:"):
                error_queue.put(RuntimeError(line))
                stop_event.set()
            else:
                print(f"macOS audio: {line}")
    except Exception as error:
        if not stop_event.is_set():
            error_queue.put(error)
            stop_event.set()


def _mac_audio_reader(
    process,
    source_queue,
    stop_event,
    error_queue,
    native_microphone_ready,
    reader_finished,
    force_stop_event=None,
):
    try:
        # Drain complete packets to EOF; the forced-stop event also bounds waits
        # when the helper pipe stays open after shutdown.
        while True:
            header = _read_exact(
                process.stdout,
                13,
                force_stop_event=force_stop_event,
            )
            if header is None:
                break

            source, timestamp_ns, sample_count = struct.unpack("<cQI", header)
            source_names = {b"S": "system", b"M": "microphone"}
            source_name = source_names.get(source)
            if source_name is None:
                raise RuntimeError("El capturador macOS envió una fuente desconocida")
            if sample_count == 0 or sample_count > SAMPLE_RATE * 4:
                raise RuntimeError("El capturador macOS envió un bloque inválido")

            payload = _read_exact(
                process.stdout,
                sample_count * 4,
                force_stop_event=force_stop_event,
            )
            if payload is None:
                break

            samples = np.frombuffer(payload, dtype="<f4").astype(
                np.float32,
                copy=True,
            )
            # The native helper timestamps the packet at its estimated start
            # (callback time minus packet duration); keep that start unchanged.
            start_ns = timestamp_ns
            if source_name == "microphone":
                native_microphone_ready.set()
            source_queue.put((source_name, start_ns, samples))
    except Exception as error:
        if not stop_event.is_set():
            error_queue.put(error)
            stop_event.set()
    finally:
        reader_finished.set()
        if not stop_event.is_set():
            process_error = process.poll()
            if process_error is not None:
                error_queue.put(
                    RuntimeError(
                        f"El capturador de audio macOS terminó con código {process_error}"
                    )
                )
                stop_event.set()


def _stop_mac_audio_process(process, *, force=False, force_stop_event=None):
    if process is None or process.poll() is not None:
        return

    if force:
        process.kill()
    else:
        process.terminate()
        deadline = time.monotonic() + MAC_AUDIO_PROCESS_STOP_TIMEOUT_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            if force_stop_event is not None and force_stop_event.is_set():
                print("⏩ Se fuerza el cierre inmediato del capturador macOS")
                process.kill()
                break
            time.sleep(0.05)
        if process.poll() is None:
            print("⚠️ El capturador macOS no respondió; se fuerza su cierre")
            if force_stop_event is not None:
                force_stop_event.set()
            process.kill()
    process.wait(timeout=MAC_AUDIO_PROCESS_STOP_TIMEOUT_SECONDS)


def _should_stop_mac_audio_process(stop_event, force_stop_event, source_queue):
    # Stop the producer before draining its queue; queued packets remain intact.
    return stop_event.is_set()


def macos_audio_capture(
    stop_event,
    error_queue,
    capture_done=None,
    force_stop_event=None,
):
    """Capture and mix macOS audio into the in-memory transcription queue."""
    if force_stop_event is None:
        force_stop_event = threading.Event()

    source_queue = queue.Queue()
    source_buffers = {
        "system": _TimestampedAudioBuffer(),
        "microphone": _TimestampedAudioBuffer(),
    }
    audio_level_stats = {
        source: _empty_audio_level_stats()
        for source in ("system", "microphone", "mezcla")
    }
    coverage_stats = {
        source: {"covered": 0, "missing": 0}
        for source in ("system", "microphone")
    }
    process = None
    reader_thread = None
    stderr_thread = None
    native_microphone_ready = threading.Event()
    reader_finished = threading.Event()
    next_audio_report_ns = (
        time.monotonic_ns() + MAC_AUDIO_DIAGNOSTIC_SECONDS * 1_000_000_000
    )

    def report_audio_levels():
        labels = {
            "system": "audio del sistema",
            "microphone": "micrófono",
            "mezcla": "mezcla enviada al transcriptor",
        }
        for source, label in labels.items():
            stats = audio_level_stats[source]
            sample_count = stats["sample_count"]
            expected_frames = (
                coverage_stats[source]["covered"] + coverage_stats[source]["missing"]
                if source in coverage_stats
                else 0
            )
            if not sample_count and not expected_frames:
                continue

            rms = (
                float(np.sqrt(stats["sum_squares"] / sample_count))
                if sample_count
                else 0.0
            )
            coverage = ""
            if source in coverage_stats and expected_frames:
                coverage_percent = (
                    100.0 * coverage_stats[source]["covered"] / expected_frames
                )
                coverage = f"; cobertura temporal={coverage_percent:.1f}%"
            print(
                f"📊 Nivel macOS ({label}): RMS={rms:.6f}; "
                f"pico={stats['peak']:.6f}; "
                f"muestras={sample_count / SAMPLE_RATE:.1f}s{coverage}"
            )
            if source == "system" and rms < 0.001:
                print(
                    "⚠️ El audio del sistema está casi silencioso; si hay voces en "
                    "la reunión, revisa la salida/captura de audio de macOS."
                )
            elif source == "microphone" and rms < 0.005:
                print(
                    "⚠️ El nivel del micrófono está bajo; si estabas hablando, "
                    "revisa el micrófono seleccionado y su nivel de entrada."
                )
            audio_level_stats[source] = _empty_audio_level_stats()
            if source in coverage_stats:
                coverage_stats[source] = {"covered": 0, "missing": 0}

    def microphone_callback(indata, frames, time_info, status):
        if stop_event.is_set():
            return
        if status:
            print(f"Micrófono macOS: {status}")
        samples = _mono_audio(indata)
        start_ns = time.monotonic_ns() - int(
            round(len(samples) * _NANOSECONDS_PER_SAMPLE)
        )
        source_queue.put(("microphone", start_ns, samples))

    def mix_available(mix_next_ns, flush_all=False, flush_tail=False):
        missing_system_frames = 0
        missing_microphone_frames = 0
        now_ns = time.monotonic_ns()
        jitter_ns = int(MAC_AUDIO_JITTER_SECONDS * 1_000_000_000)
        window_ns = int(round(MAC_AUDIO_MIX_FRAMES * _NANOSECONDS_PER_SAMPLE))

        while mix_next_ns is not None:
            frames = MAC_AUDIO_MIX_FRAMES
            window_end_ns = mix_next_ns + window_ns
            if flush_all:
                buffer_end_ns = max(
                    buffer.end_ns or mix_next_ns for buffer in source_buffers.values()
                )
                if window_end_ns > buffer_end_ns:
                    if not flush_tail:
                        break
                    frames = min(
                        frames,
                        int(round((buffer_end_ns - mix_next_ns) / _NANOSECONDS_PER_SAMPLE)),
                    )
                    if frames <= 0:
                        break
                    window_end_ns = mix_next_ns + int(
                        round(frames * _NANOSECONDS_PER_SAMPLE)
                    )
            elif window_end_ns > now_ns - jitter_ns:
                break

            system_audio, system_covered = source_buffers["system"].read_window(
                mix_next_ns,
                frames,
            )
            microphone_audio, microphone_covered = source_buffers[
                "microphone"
            ].read_window(mix_next_ns, frames)
            missing_system_frames += frames - system_covered
            missing_microphone_frames += frames - microphone_covered
            coverage_stats["system"]["covered"] += system_covered
            coverage_stats["system"]["missing"] += frames - system_covered
            coverage_stats["microphone"]["covered"] += microphone_covered
            coverage_stats["microphone"]["missing"] += (
                frames - microphone_covered
            )

            if system_covered or microphone_covered:
                mixed = np.nan_to_num(
                    system_audio * MAC_SYSTEM_GAIN
                    + microphone_audio * MAC_MICROPHONE_GAIN,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                peak = np.max(np.abs(mixed)) if mixed.size else 0.0
                if peak > 1.0:
                    mixed = mixed / peak
                _accumulate_audio_level(audio_level_stats["mezcla"], mixed)
                audio_queue.put(mixed.reshape(-1, 1))

            mix_next_ns = window_end_ns

        return mix_next_ns, missing_system_frames, missing_microphone_frames

    try:
        if not os.path.isfile(MAC_AUDIO_HELPER):
            raise FileNotFoundError(
                "No existe el capturador macOS. Ejecuta start_transcribe.sh para compilarlo."
            )

        process = subprocess.Popen(
            [MAC_AUDIO_HELPER],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # Python handles Ctrl+C, explicitly stops the helper, then flushes
            # queued audio and the final transcription block.
            start_new_session=True,
        )
        reader_thread = threading.Thread(
            target=_mac_audio_reader,
            args=(
                process,
                source_queue,
                stop_event,
                error_queue,
                native_microphone_ready,
                reader_finished,
                force_stop_event,
            ),
            name="macos-system-audio-reader",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_mac_audio_stderr,
            args=(process, stop_event, error_queue),
            name="macos-system-audio-stderr",
            daemon=True,
        )
        reader_thread.start()
        stderr_thread.start()

        if MAC_NATIVE_MICROPHONE:
            print("🎤 Micrófono macOS: ScreenCaptureKit")
            microphone_stream = nullcontext()
        else:
            print("🎤 Micrófono macOS: dispositivo predeterminado")
            microphone_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=microphone_callback,
            )

        mix_next_ns = None
        missing_system_frames = 0
        missing_microphone_frames = 0
        native_microphone_deadline_ns = time.monotonic_ns() + 10_000_000_000
        stop_process_requested = False

        with ExitStack() as microphone_context:
            microphone_context.enter_context(microphone_stream)
            while True:
                if (
                    _should_stop_mac_audio_process(
                        stop_event, force_stop_event, source_queue
                    )
                    and not stop_process_requested
                ):
                    # Exit the input stream context before testing the queue
                    # barrier so no in-flight microphone callback can publish later.
                    microphone_context.close()
                    _stop_mac_audio_process(
                        process,
                        force=force_stop_event.is_set(),
                        force_stop_event=force_stop_event,
                    )
                    stop_process_requested = True

                try:
                    source, start_ns, samples = source_queue.get(timeout=0.05)
                    source_buffers[source].append(start_ns, samples)
                    _accumulate_audio_level(audio_level_stats[source], samples)
                    if mix_next_ns is None:
                        first_starts = [
                            buffer.first_start_ns
                            for buffer in source_buffers.values()
                            if buffer.first_start_ns is not None
                        ]
                        if first_starts:
                            mix_next_ns = min(first_starts)
                except queue.Empty:
                    pass

                if (
                    MAC_NATIVE_MICROPHONE
                    and not native_microphone_ready.is_set()
                    and not stop_event.is_set()
                    and time.monotonic_ns() > native_microphone_deadline_ns
                ):
                    raise RuntimeError(
                        "ScreenCaptureKit no entregó muestras del micrófono macOS"
                    )

                final_input_drained = (
                    stop_process_requested
                    and reader_finished.is_set()
                    and source_queue.empty()
                )
                stopping = stop_event.is_set()
                # On shutdown, buffer every delivered source packet before mixing;
                # otherwise one source can advance the cursor past the other.
                if not stopping or final_input_drained:
                    mix_next_ns, missing_system, missing_microphone = mix_available(
                        mix_next_ns,
                        flush_all=stopping,
                        flush_tail=final_input_drained,
                    )
                    missing_system_frames += missing_system
                    missing_microphone_frames += missing_microphone

                now_ns = time.monotonic_ns()
                if now_ns >= next_audio_report_ns:
                    report_audio_levels()
                    next_audio_report_ns = (
                        now_ns + MAC_AUDIO_DIAGNOSTIC_SECONDS * 1_000_000_000
                    )

                if final_input_drained:
                    break

        if missing_system_frames or missing_microphone_frames:
            print(
                "⚠️ Audio macOS: se rellenaron huecos de captura "
                f"(sistema={missing_system_frames / SAMPLE_RATE:.2f}s, "
                f"micrófono={missing_microphone_frames / SAMPLE_RATE:.2f}s)"
            )
    except Exception as error:
        error_queue.put(error)
        stop_event.set()
    finally:
        stop_event.set()
        try:
            _stop_mac_audio_process(
                process,
                force=force_stop_event.is_set(),
                force_stop_event=force_stop_event,
            )
        except Exception as error:
            error_queue.put(
                RuntimeError(f"No se pudo detener el capturador macOS: {error}")
            )
        try:
            if reader_thread is not None:
                reader_thread.join(timeout=2)
            if stderr_thread is not None:
                stderr_thread.join(timeout=2)
            report_audio_levels()
        finally:
            if capture_done is not None:
                capture_done.set()


def windows_audio_capture(stop_event, error_queue, capture_done):
    try:
        from windows_audio import capture
        capture(audio_queue, stop_event, error_queue, rate=SAMPLE_RATE,
                microphone_name=WINDOWS_MICROPHONE, speaker_name=WINDOWS_SPEAKER,
                buffer_seconds=WINDOWS_BUFFER_SECONDS)
    except Exception as error:
        error_queue.put(error)
        stop_event.set()
    finally:
        capture_done.set()


def main_windows(file_path):
    stop_event = threading.Event()
    capture_done = threading.Event()
    error_queue = queue.Queue()
    capture_thread = threading.Thread(
        target=windows_audio_capture,
        args=(stop_event, error_queue, capture_done),
        name="windows-audio-capture",
        daemon=True,
    )
    def request_stop(signum, frame):
        _announce_stop_request(stop_event)
        if not stop_event.is_set():
            print("\n🧠 Finalizando reunión; guardando todo el audio pendiente...")
            stop_event.set()

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    if GUI_EVENTS:
        start_gui_stdin_listener(stop_event, on_stop=lambda: request_stop(None, None))
    try:
        capture_thread.start()
        transcriber_loop(file_path, stop_event=stop_event, capture_done=capture_done)
    finally:
        stop_event.set()
        capture_thread.join(timeout=5)
        signal.signal(signal.SIGINT, previous_handler)

    while not error_queue.empty():
        error = error_queue.get()
        message = f"No se pudo capturar micrófono + loopback en Windows: {error}"
        emit_gui_event("error", message=message)
        print(f"❌ {message}")


def _handle_macos_interrupt(
    stop_event,
    force_stop_event,
    capture_done,
    shutdown_state,
):
    if not stop_event.is_set():
        now = time.monotonic()
        shutdown_state.update(
            phase="stopping_capture",
            requested_at=now,
            phase_started_at=now,
            next_progress_at=now + MAC_SHUTDOWN_PROGRESS_SECONDS,
        )
        print("\n🧠 Ctrl+C recibido; deteniendo captura y guardando el audio pendiente...")
        stop_event.set()
        return

    phase = shutdown_state.get("phase", "stopping_capture")
    if phase == "complete":
        print("✅ La reunión ya terminó de guardarse")
        return

    inference_started = shutdown_state.get("inference_started_at")
    if inference_started is not None:
        elapsed = time.monotonic() - inference_started
        capture_still_running = capture_done is not None and not capture_done.is_set()
        if capture_still_running and not force_stop_event.is_set():
            force_stop_event.set()
            print(
                f"⏩ Segunda interrupción: se acelerará la captura; "
                f"{transcriber.name} terminará el bloque actual "
                f"(lleva {elapsed:.0f}s) para no perderlo."
            )
        else:
            print(
                f"🧠 {transcriber.name} sigue transcribiendo "
                f"(lleva {elapsed:.0f}s); no se interrumpe para conservar "
                "ese texto."
            )
        return

    if capture_done is not None and capture_done.is_set():
        print("⏳ Captura detenida; se está vaciando la cola y guardando el texto final")
    elif not force_stop_event.is_set():
        force_stop_event.set()
        print(
            "⏩ Segunda interrupción: se detendrá el capturador macOS ahora; "
            "el audio que ya llegó se conservará y transcribirá."
        )
    else:
        print("⏳ Cierre acelerado; esperando que se transcriba el audio ya recibido")


def main_macos(file_path):
    stop_event = threading.Event()
    force_stop_event = threading.Event()
    capture_done = threading.Event()
    progress_done = threading.Event()
    error_queue = queue.Queue()
    shutdown_state = {"phase": "capturing"}
    capture_thread = threading.Thread(
        target=macos_audio_capture,
        args=(stop_event, error_queue, capture_done, force_stop_event),
        name="macos-audio-capture",
        daemon=True,
    )
    def request_stop(signum, frame):
        _announce_stop_request(stop_event)
        _handle_macos_interrupt(
            stop_event,
            force_stop_event,
            capture_done,
            shutdown_state,
        )

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    if GUI_EVENTS:
        start_gui_stdin_listener(stop_event, on_stop=lambda: request_stop(None, None))
    try:
        progress_thread = threading.Thread(
            target=_macos_shutdown_progress_loop,
            args=(
                stop_event,
                force_stop_event,
                capture_done,
                shutdown_state,
                progress_done,
            ),
            name="macos-shutdown-progress",
            daemon=True,
        )
        progress_thread.start()
        capture_thread.start()
        transcriber_loop(
            file_path,
            stop_event=stop_event,
            capture_done=capture_done,
            shutdown_state=shutdown_state,
        )
    finally:
        stop_event.set()
        if capture_thread.ident is not None:
            capture_thread.join(timeout=10)
            if capture_thread.is_alive():
                print(
                    "⚠️ Captura macOS sigue cerrando; se solicita detener "
                    "el proceso auxiliar"
                )
                force_stop_event.set()
                capture_thread.join(timeout=5)
            if capture_thread.is_alive():
                error_queue.put(
                    RuntimeError("El hilo de captura macOS no terminó tras el cierre forzado")
                )
        progress_done.set()
        if "progress_thread" in locals() and progress_thread.ident is not None:
            progress_thread.join(timeout=2)
        signal.signal(signal.SIGINT, previous_handler)

    if not error_queue.empty():
        error = error_queue.get()
        message = f"No se pudo capturar audio del sistema macOS + micrófono: {error}"
        emit_gui_event("error", message=message)
        print(f"❌ {message}")

def main():
    file_path = _output_file_path()
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    try:
        write_header(file_path)
        emit_gui_event("ready", file_path=file_path, engine=transcriber.name)
        if os.name == "nt":
            print(f"📂 Guardando en: {file_path}")
            print("🎙️ Capturando micrófono y audio del sistema... Ctrl+C para terminar\n")
            main_windows(file_path)
        else:
            print(f"📂 Guardando en: {file_path}")
            print("🎙️ Capturando micrófono + audio del sistema... Ctrl+C para terminar\n")
            main_macos(file_path)
        print("\n🛑 Reunión finalizada")
        emit_gui_event("finished", file_path=file_path)
    except Exception as error:
        emit_gui_event("error", message=str(error))
        raise
    finally:
        transcriber.close()

if __name__ == "__main__":
    main()
