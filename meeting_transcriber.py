import os
import time
import queue
import threading
import numpy as np
import sounddevice as sd
from datetime import datetime
from faster_whisper import WhisperModel
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import librosa

# ================= CONFIG =================

OBSIDIAN_DIR = "/Users/mateoarteaga/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian"
NOTE_TITLE = f"Reunión - {datetime.now().strftime('%Y-%m-%d %H-%M')}.md"

SAMPLE_RATE = 16000
BLOCK_SECONDS = 30  # cada cuántos segundos se transcribe
WHISPER_MODEL = "medium"  # perfecto para reuniones
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

def transcriber_loop(file_path):
    buffer = []
    last_flush = time.time()

    try:
        while True:
            try:
                data = audio_queue.get(timeout=1)
                buffer.append(data)

                if time.time() - last_flush >= BLOCK_SECONDS:
                    flush_buffer(buffer, file_path)
                    buffer.clear()
                    last_flush = time.time()

            except queue.Empty:
                continue

    except KeyboardInterrupt:
        print("\n🧠 Finalizando reunión, guardando último audio...")

        # 🔥 FLUSH FINAL GARANTIZADO
        if buffer:
            flush_buffer(buffer, file_path)

def main():
    file_path = os.path.join(OBSIDIAN_DIR, NOTE_TITLE)
    write_header(file_path)

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