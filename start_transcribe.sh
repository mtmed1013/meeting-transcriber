#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -x "$PROJECT_DIR/venv/bin/python" ]; then
  echo "No existe el entorno virtual. Instala las dependencias primero."
  exit 1
fi

exec "$PROJECT_DIR/venv/bin/python" "$PROJECT_DIR/meeting_transcriber.py"
