#!/usr/bin/env bash
set -euo pipefail

ensure_command_line_tools() {
  if xcrun --find swiftc >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v xcode-select >/dev/null 2>&1; then
    echo "No se encontró xcode-select en macOS."
    return 1
  fi

  echo "Se necesitan las Xcode Command Line Tools para capturar el audio del sistema."
  echo "macOS abrirá el instalador de Apple. Acepta la instalación y, si se solicita,"
  echo "autoriza con una cuenta administradora."
  xcode-select --install >/dev/null 2>&1 || true

  local macos_tools_waited_seconds=0
  local macos_tools_max_wait_seconds=1800
  local macos_tools_poll_seconds=5

  while (( macos_tools_waited_seconds < macos_tools_max_wait_seconds )); do
    if xcrun --find swiftc >/dev/null 2>&1; then
      echo "Xcode Command Line Tools listas."
      return 0
    fi

    sleep "$macos_tools_poll_seconds"
    macos_tools_waited_seconds=$((macos_tools_waited_seconds + macos_tools_poll_seconds))

    if (( macos_tools_waited_seconds % 30 == 0 )); then
      echo "Esperando a que termine la instalación de Apple..."
    fi
  done

  echo "No se completó la instalación de Xcode Command Line Tools."
  echo "Instálalas desde la ventana de macOS y vuelve a ejecutar este script."
  return 1
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  ensure_command_line_tools
fi
