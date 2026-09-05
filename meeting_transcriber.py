import os
import time
import queue
import platform
import struct
import subprocess
import sys
import threading
from collections import deque
from contextlib import nullcontext
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import librosa

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
NOTE_TITLE = f"Reunión - {datetime.now().strftime('%Y-%m-%d %H-%M')}.md"

SAMPLE_RATE = 16000
BLOCK_SECONDS = 30  # cada cuántos segundos se transcribe
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "").strip() or "medium"
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = _env_bool("WHISPER_VAD_FILTER", default=True)
WINDOWS_MICROPHONE = os.environ.get("WINDOWS_MICROPHONE", "").strip()
WINDOWS_SPEAKER = os.environ.get("WINDOWS_SPEAKER", "").strip()
WINDOWS_CAPTURE_FRAMES = 3200  # 200 ms a 16 kHz
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
MAC_SYSTEM_GAIN = _env_float("MAC_SYSTEM_GAIN", 1.0)
MAC_MICROPHONE_GAIN = _env_float("MAC_MICROPHONE_GAIN", 1.0)
MAC_NATIVE_MICROPHONE = _macos_version_at_least(15)
last_voice_signature = None

# ==========================================


audio_queue = queue.Queue()
running = True

# 🔥 ESTE BLOQUE FALTABA
model = WhisperModel(
    WHISPER_MODEL,
    device="auto",        # usa Apple Silicon si está disponible
    compute_type="int8"   # rápido y suficiente para reuniones
)

def voice_signature(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=5)
    return np.mean(mfcc, axis=1)

def speaker_changed(current_sig, threshold=25.0):
    global last_voice_signature

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
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# 🧠 Reunión\n\n")
        f.write(f"🕒 Inicio: {datetime.now().strftime('%H:%M')}\n\n")
        f.write("---\n\n")

def append_text(file_path, text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"**[{timestamp}]**\n{text.strip()}\n\n")

def flush_buffer(buffer, file_path):
    if not buffer:
        return

    audio = np.concatenate(buffer, axis=0).flatten()
    if len(audio) < SAMPLE_RATE * 1.5:
        print("⚠️ Audio muy corto, se omite")
        return
        
    transcribe_options = dict(
        language=None,
        vad_filter=WHISPER_VAD_FILTER,
        beam_size=WHISPER_BEAM_SIZE,
    )
    if WHISPER_VAD_FILTER:
        transcribe_options["vad_parameters"] = dict(
            min_silence_duration_ms=300,
        )

    segments, info = model.transcribe(audio, **transcribe_options)

    text = " ".join(seg.text for seg in segments).strip()
    sig = voice_signature(audio, SAMPLE_RATE)

    if speaker_changed(sig):
        append_text(file_path, "— 🔄 Cambio de hablante —")

    if text:
        append_text(file_path, text)
        print("📝 Texto guardado")
    else:
        print("⚠️ Audio muy corto, sin texto detectado")

def transcriber_loop(file_path, stop_event=None):
    buffer = []
    last_flush = time.time()

    try:
        while True:
            if stop_event is not None and stop_event.is_set() and audio_queue.empty():
                break
            try:
                data = audio_queue.get(timeout=1)
                buffer.append(data)

                if time.time() - last_flush >= BLOCK_SECONDS:
                    flush_buffer(buffer, file_path)
                    buffer.clear()
                    last_flush = time.time()

            except queue.Empty:
                if stop_event is not None and stop_event.is_set():
                    break
                continue

    except KeyboardInterrupt:
        print("\n🧠 Finalizando reunión, guardando último audio...")

    # 🔥 FLUSH FINAL GARANTIZADO
    if buffer:
        flush_buffer(buffer, file_path)


def _mono_audio(data):
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


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


def _read_exact(stream, size):
    chunks = []
    remaining = size

    while remaining:
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
                print("🔊 Audio del sistema macOS capturado")
            elif line == "MEETING_AUDIO_MIC_ENABLED":
                print("🎤 Captura nativa del micrófono macOS habilitada")
            elif line == "MEETING_AUDIO_MIC_READY":
                print("🎤 Recibiendo muestras del micrófono macOS")
            elif line == "MEETING_AUDIO_SAMPLE_READY":
                print("🔊 Recibiendo muestras de audio del sistema")
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
):
    try:
        # Continue draining until the helper closes stdout so Ctrl+C does not
        # discard packets that were already captured but still in the pipe.
        while True:
            header = _read_exact(process.stdout, 13)
            if header is None:
                break

            source, timestamp_ns, sample_count = struct.unpack("<cQI", header)
            source_names = {b"S": "system", b"M": "microphone"}
            source_name = source_names.get(source)
            if source_name is None:
                raise RuntimeError("El capturador macOS envió una fuente desconocida")
            if sample_count == 0 or sample_count > SAMPLE_RATE * 4:
                raise RuntimeError("El capturador macOS envió un bloque inválido")

            payload = _read_exact(process.stdout, sample_count * 4)
            if payload is None:
                break

            samples = np.frombuffer(payload, dtype="<f4").astype(
                np.float32,
                copy=True,
            )
            start_ns = timestamp_ns - int(
                round(len(samples) * _NANOSECONDS_PER_SAMPLE)
            )
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


def _stop_mac_audio_process(process):
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def macos_audio_capture(stop_event, error_queue):
    """Capture macOS system audio and microphone without temporary WAV files."""
    source_queue = queue.Queue()
    source_buffers = {
        "system": _TimestampedAudioBuffer(),
        "microphone": _TimestampedAudioBuffer(),
    }
    process = None
    reader_thread = None
    stderr_thread = None
    native_microphone_ready = threading.Event()
    reader_finished = threading.Event()

    def microphone_callback(indata, frames, time_info, status):
        if status:
            print(f"Micrófono macOS: {status}")
        samples = _mono_audio(indata)
        start_ns = time.monotonic_ns() - int(
            round(len(samples) * _NANOSECONDS_PER_SAMPLE)
        )
        source_queue.put(("microphone", start_ns, samples))

    def mix_available(mix_next_ns, flush_all=False):
        missing_system_frames = 0
        missing_microphone_frames = 0
        now_ns = time.monotonic_ns()
        jitter_ns = int(MAC_AUDIO_JITTER_SECONDS * 1_000_000_000)
        window_ns = int(round(MAC_AUDIO_MIX_FRAMES * _NANOSECONDS_PER_SAMPLE))

        while mix_next_ns is not None:
            window_end_ns = mix_next_ns + window_ns
            if flush_all:
                buffer_end_ns = max(
                    buffer.end_ns or mix_next_ns for buffer in source_buffers.values()
                )
                if window_end_ns > buffer_end_ns:
                    break
            elif window_end_ns > now_ns - jitter_ns:
                break

            system_audio, system_covered = source_buffers["system"].read_window(
                mix_next_ns,
                MAC_AUDIO_MIX_FRAMES,
            )
            microphone_audio, microphone_covered = source_buffers[
                "microphone"
            ].read_window(mix_next_ns, MAC_AUDIO_MIX_FRAMES)
            missing_system_frames += MAC_AUDIO_MIX_FRAMES - system_covered
            missing_microphone_frames += MAC_AUDIO_MIX_FRAMES - microphone_covered

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
            # Ctrl+C must be handled by Python first so the final transcript
            # can be flushed before the native helper is stopped explicitly.
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

        with microphone_stream:
            while True:
                try:
                    source, start_ns, samples = source_queue.get(timeout=0.05)
                    source_buffers[source].append(start_ns, samples)
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

                mix_next_ns, missing_system, missing_microphone = mix_available(
                    mix_next_ns,
                    flush_all=stop_event.is_set(),
                )
                missing_system_frames += missing_system
                missing_microphone_frames += missing_microphone

                if stop_event.is_set() and source_queue.empty():
                    if not stop_process_requested:
                        _stop_mac_audio_process(process)
                        stop_process_requested = True
                    if not reader_finished.is_set():
                        continue
                    remaining_data = any(
                        buffer.end_ns is not None
                        and mix_next_ns is not None
                        and buffer.end_ns > mix_next_ns
                        for buffer in source_buffers.values()
                    )
                    if not remaining_data:
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
        _stop_mac_audio_process(process)
        if reader_thread is not None:
            reader_thread.join(timeout=2)
        if stderr_thread is not None:
            stderr_thread.join(timeout=2)


def windows_audio_capture(stop_event, error_queue):
    """Capture the physical microphone and Windows speaker loopback."""
    try:
        import soundcard as sc

        microphone = (
            sc.get_microphone(WINDOWS_MICROPHONE)
            if WINDOWS_MICROPHONE
            else sc.default_microphone()
        )
        speaker = (
            sc.get_speaker(WINDOWS_SPEAKER)
            if WINDOWS_SPEAKER
            else sc.default_speaker()
        )
        loopback = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True,
        )
        print(f"🎤 Micrófono Windows: {microphone.name}")
        print(f"🔊 Loopback Windows: {speaker.name}")

        # SoundCard documents a Windows/WASAPI single-channel issue, so both
        # sources are opened in stereo and folded to mono ourselves.
        with microphone.recorder(
            samplerate=SAMPLE_RATE,
            channels=2,
            blocksize=WINDOWS_CAPTURE_FRAMES,
        ) as mic_recorder, loopback.recorder(
            samplerate=SAMPLE_RATE,
            channels=2,
            blocksize=WINDOWS_CAPTURE_FRAMES,
        ) as loopback_recorder:
            while not stop_event.is_set():
                mic_audio = _mono_audio(
                    mic_recorder.record(numframes=WINDOWS_CAPTURE_FRAMES)
                )
                loopback_audio = _mono_audio(
                    loopback_recorder.record(numframes=WINDOWS_CAPTURE_FRAMES)
                )
                frames = min(len(mic_audio), len(loopback_audio))
                if frames == 0:
                    continue
                # Averaging prevents clipping when both sources are loud.
                mixed = np.nan_to_num(
                    (mic_audio[:frames] + loopback_audio[:frames]) * 0.5,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                audio_queue.put(mixed.reshape(-1, 1))
    except Exception as error:
        error_queue.put(error)
        stop_event.set()


def main_windows(file_path):
    stop_event = threading.Event()
    error_queue = queue.Queue()
    capture_thread = threading.Thread(
        target=windows_audio_capture,
        args=(stop_event, error_queue),
        name="windows-audio-capture",
        daemon=True,
    )
    capture_thread.start()
    try:
        transcriber_loop(file_path, stop_event=stop_event)
    finally:
        stop_event.set()
        capture_thread.join(timeout=5)

    if not error_queue.empty():
        error = error_queue.get()
        print(f"❌ No se pudo capturar micrófono + loopback en Windows: {error}")


def main_macos(file_path):
    stop_event = threading.Event()
    error_queue = queue.Queue()
    capture_thread = threading.Thread(
        target=macos_audio_capture,
        args=(stop_event, error_queue),
        name="macos-audio-capture",
        daemon=True,
    )
    capture_thread.start()

    try:
        transcriber_loop(file_path, stop_event=stop_event)
    finally:
        stop_event.set()
        capture_thread.join(timeout=10)

    if not error_queue.empty():
        error = error_queue.get()
        print(f"❌ No se pudo capturar audio del sistema macOS + micrófono: {error}")

def main():
    os.makedirs(OBSIDIAN_DIR, exist_ok=True)
    file_path = os.path.join(OBSIDIAN_DIR, NOTE_TITLE)
    write_header(file_path)

    if os.name == "nt":
        print(f"📂 Guardando en: {file_path}")
        print("🎙️ Capturando micrófono y audio de Teams... Ctrl+C para terminar\n")
        main_windows(file_path)
        print("\n🛑 Reunión finalizada")
        return

    print(f"📂 Guardando en: {file_path}")
    print("🎙️ Capturando micrófono + audio del sistema... Ctrl+C para terminar\n")
    main_macos(file_path)

    print("\n🛑 Reunión finalizada")

if __name__ == "__main__":
    main()
