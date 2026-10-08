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

echo "Instalando dependencias de Python..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"

if [[ ! -f "$PROJECT_DIR/.env" ]]; then
  cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
  echo "Creado .env con la configuración inicial."
fi

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
