import os
import time
import queue
import threading
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
WINDOWS_MICROPHONE = os.environ.get("WINDOWS_MICROPHONE", "").strip()
WINDOWS_SPEAKER = os.environ.get("WINDOWS_SPEAKER", "").strip()
WINDOWS_CAPTURE_FRAMES = 3200  # 200 ms a 16 kHz
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
        
    segments, info = model.transcribe(
        audio,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        beam_size=1
    )

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
    print("🎙️ Escuchando reunión... Ctrl+C para terminar\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback
    )

    with stream:
        try:
            transcriber_loop(file_path)
        except KeyboardInterrupt:
            pass

    print("\n🛑 Reunión finalizada")

if __name__ == "__main__":
    main()
