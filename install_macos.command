#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

set +e
"$PROJECT_DIR/macos/install_macos.sh"
install_status=$?
set -e

echo
if [[ "$install_status" -eq 0 ]]; then
  echo "Instalación finalizada. Revisa los mensajes anteriores sobre permisos."
  echo "Cuando el permiso esté activo, usa start_transcribe.sh para transcribir."
else
  echo "La instalación no terminó correctamente. Revisa el mensaje anterior."
fi
read -r -p "Presiona Enter para cerrar esta ventana..." _
exit "$install_status"
