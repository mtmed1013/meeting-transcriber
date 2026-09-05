#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ "$(uname -s)" = "Darwin" ]; then
  source "$PROJECT_DIR/macos/ensure_command_line_tools.sh"
  ensure_command_line_tools
fi

if [ ! -x "$PROJECT_DIR/venv/bin/python" ]; then
  echo "No existe el entorno virtual. Instala las dependencias primero."
  exit 1
fi

if [ "$(uname -s)" = "Darwin" ]; then
  MAC_AUDIO_HELPER="$PROJECT_DIR/macos/.build/MeetingTranscriberAudio.app/Contents/MacOS/MeetingTranscriberAudio"
  if [ ! -x "$MAC_AUDIO_HELPER" ] || \
     [ "$PROJECT_DIR/macos/MeetingTranscriberAudio/main.swift" -nt "$MAC_AUDIO_HELPER" ] || \
     [ "$PROJECT_DIR/macos/MeetingTranscriberAudio/Info.plist" -nt "$MAC_AUDIO_HELPER" ]; then
    echo "Preparando el capturador nativo de audio de macOS..."
    "$PROJECT_DIR/macos/build_audio_capture.sh"
  fi
fi

exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/meeting_transcriber.py"
