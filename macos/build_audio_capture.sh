#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/MeetingTranscriberAudio"
BUILD_DIR="$SCRIPT_DIR/.build"
APP_DIR="$BUILD_DIR/MeetingTranscriberAudio.app"
BINARY="$APP_DIR/Contents/MacOS/MeetingTranscriberAudio"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este capturador solo se puede compilar en macOS."
  exit 1
fi

source "$SCRIPT_DIR/ensure_command_line_tools.sh"
ensure_command_line_tools

mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
SWIFT_FLAGS=()
if [[ "$MACOS_MAJOR" -ge 15 ]]; then
  SWIFT_FLAGS+=(-D NATIVE_MICROPHONE)
fi

xcrun swiftc \
  -swift-version 5 \
  -parse-as-library \
  -O \
  "${SWIFT_FLAGS[@]}" \
  -framework AVFoundation \
  -framework CoreMedia \
  -framework ScreenCaptureKit \
  "$SOURCE_DIR/main.swift" \
  -o "$BINARY"

cp "$SOURCE_DIR/Info.plist" "$APP_DIR/Contents/Info.plist"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$APP_DIR" >/dev/null
fi

echo "Capturador macOS compilado: $BINARY"
