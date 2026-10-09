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
updated, count = pattern.subn(r"\1whisper", contents)
if count:
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
    print("TRANSCRIPTION_ENGINE antiguo migrado a whisper en .env.")
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

echo "Descargando y verificando el modelo Whisper configurado..."
"$VENV_PYTHON" -c "from dotenv import load_dotenv; import os; from faster_whisper import WhisperModel; load_dotenv('.env'); name = os.getenv('WHISPER_MODEL', '').strip() or 'medium'; print(f'Modelo: {name}'); WhisperModel(name, device='cpu', compute_type='int8'); print('Modelo listo.')"

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
