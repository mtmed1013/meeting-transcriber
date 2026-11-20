#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este instalador solo se puede ejecutar en macOS."
  exit 1
fi

source "$PROJECT_DIR/macos/ensure_command_line_tools.sh"
ensure_command_line_tools

if ! command -v python3 >/dev/null 2>&1; then
  echo "No se encontró Python 3. Instala Python 3.11 o superior y vuelve a ejecutar este instalador."
  exit 1
fi

VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Creando entorno virtual de Python..."
  python3 -m venv "$PROJECT_DIR/venv"
fi

MACOS_ARCH="$(uname -m)"
if [[ "$(sysctl -in hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]]; then
  MACOS_ARCH="arm64"
fi
PYTHON_ARCH="$("$VENV_PYTHON" -c 'import platform; print(platform.machine())')"
if [[ "$MACOS_ARCH" == "arm64" && "$PYTHON_ARCH" != "arm64" && "$PYTHON_ARCH" != "aarch64" ]]; then
  echo "Este Mac tiene Apple Silicon, pero el Python del entorno es $PYTHON_ARCH."
  echo "Instala/usa Python nativo ARM64 y vuelve a ejecutar el instalador."
  exit 1
fi

echo "Instalando dependencias de Python..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$VENV_PYTHON" -m pip uninstall -y transcribe-cpp transcribe-cpp-native

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo "Creado .env con la configuración inicial."
fi

MODEL_DIR="$PROJECT_DIR/models"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  "$VENV_PYTHON" - "$PROJECT_DIR/.env" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

env_path = Path(sys.argv[1])
with env_path.open("r", encoding="utf-8", newline="") as env_file:
    contents = env_file.read()
pattern = re.compile(
    r"(?im)^([ \t]*TRANSCRIPTION_ENGINE[ \t]*=[ \t]*)(?:canary|nemotron)"
    r"(?=[ \t]*(?:#[^\r\n]*)?(?:\r?$))"
)
updated, engine_count = pattern.subn(r"\1auto", contents)
model_pattern = re.compile(
    r"(?im)^([ \t]*WHISPER_MODEL[ \t]*=[ \t]*)medium"
    r"(?=[ \t]*(?:#[^\r\n]*)?(?:\r?$))"
)
updated, model_count = model_pattern.subn(r"\1small", updated)
if engine_count or model_count:
    fd, temporary_path = tempfile.mkstemp(prefix=".env.", dir=env_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            output.write(updated)
        os.chmod(temporary_path, stat.S_IMODE(env_path.stat().st_mode))
        os.replace(temporary_path, env_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    if engine_count:
        print("Motor antiguo migrado a auto en .env.")
    if model_count:
        print("WHISPER_MODEL medium migrado a small en .env.")
PY
fi

for obsolete_model in \
  "$MODEL_DIR/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf" \
  "$MODEL_DIR"/.nemotron-3.5-asr-streaming-0.6b.* \
  "$MODEL_DIR/canary-180m-flash-Q8_0.gguf" \
  "$MODEL_DIR"/.canary-180m-flash.*; do
  if [[ -f "$obsolete_model" ]]; then
    rm -- "$obsolete_model"
    echo "Artefacto local obsoleto de Canary/Nemotron eliminado: $(basename "$obsolete_model")"
  fi
done

if [[ "$MACOS_ARCH" == "arm64" ]]; then
  MODEL_DIR="$PROJECT_DIR/models"
  WHISTLE_MODEL="$MODEL_DIR/whistle.cact"
  WHISTLE_SHA256="b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"
  WHISTLE_URL="https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact"
  mkdir -p "$MODEL_DIR"

  CURRENT_WHISTLE_HASH=""
  if [[ -f "$WHISTLE_MODEL" ]]; then
    CURRENT_WHISTLE_HASH="$(shasum -a 256 "$WHISTLE_MODEL" | awk '{print $1}')"
  fi
  if [[ "$CURRENT_WHISTLE_HASH" != "$WHISTLE_SHA256" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "No se encontró curl, necesario para descargar whistle.cact."
      exit 1
    fi
    TEMP_WHISTLE="$(mktemp "$MODEL_DIR/.whistle.XXXXXX")"
    if ! curl --fail --location --retry 3 --output "$TEMP_WHISTLE" "$WHISTLE_URL"; then
      rm -f "$TEMP_WHISTLE"
      echo "No se pudo descargar Cactus Whistle."
      exit 1
    fi
    DOWNLOADED_WHISTLE_HASH="$(shasum -a 256 "$TEMP_WHISTLE" | awk '{print $1}')"
    if [[ "$DOWNLOADED_WHISTLE_HASH" != "$WHISTLE_SHA256" ]]; then
      rm -f "$TEMP_WHISTLE"
      echo "La verificación SHA-256 de whistle.cact falló."
      exit 1
    fi
    mv -f "$TEMP_WHISTLE" "$WHISTLE_MODEL"
  fi

  NEEDLE_CLI="$PROJECT_DIR/venv/bin/needle"
  if [[ ! -x "$NEEDLE_CLI" ]]; then
    echo "No se encontró Needle en el entorno virtual ARM64."
    echo "Verifica la versión de Python y vuelve a ejecutar el instalador."
    exit 1
  fi
  echo "Preparando el runtime local Needle para Apple Silicon..."
  NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 "$NEEDLE_CLI" fetch
  echo "Verificando la carga local de Cactus Whistle..."
  NEEDLE_TELEMETRY=0 DO_NOT_TRACK=1 "$VENV_PYTHON" -c "from transcription_engines import WhistleTranscriber; engine = WhistleTranscriber(language='es'); engine.close(); print('Cactus Whistle listo.')"
else
  echo "Mac Intel detectado: se usará Whisper Small; se omite Needle y Whistle."
fi

echo "Descargando y verificando Whisper de respaldo (Small por defecto)..."
"$VENV_PYTHON" -c "from dotenv import load_dotenv; import os; from faster_whisper import WhisperModel; load_dotenv('.env', override=True); name = os.getenv('WHISPER_MODEL', '').strip() or 'small'; print(f'Respaldo Whisper: {name}'); WhisperModel(name, device='cpu', compute_type='int8'); print('Respaldo listo.')"

echo "Preparando el capturador nativo de audio del sistema..."
"$PROJECT_DIR/macos/build_audio_capture.sh"

CAPTURE_HELPER="$PROJECT_DIR/macos/.build/MeetingTranscriberAudio.app/Contents/MacOS/MeetingTranscriberAudio"
echo
echo "Comprobando el permiso de captura de audio del sistema..."
if "$CAPTURE_HELPER" --check-permissions; then
  echo "✅ Permiso de captura de audio del sistema confirmado."
else
  echo
  echo "⚠️ No se pudo confirmar el permiso de captura de audio del sistema."
  echo "La instalación quedó preparada; antes de transcribir, abre:"
  echo "Ajustes del Sistema > Privacidad y seguridad >"
  echo "Grabación de pantalla y del audio del sistema."
  echo "Activa la aplicación que macOS identifica en el aviso (normalmente Terminal)."
  echo "Si ofrece la opción, permite solo audio. Luego cierra y vuelve a abrir esa aplicación."
fi

echo
echo "Instalación de macOS terminada."
echo "Configura OBSIDIAN_DIR en .env si hace falta y ejecuta start_transcribe.sh."
