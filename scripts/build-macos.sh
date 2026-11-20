#!/usr/bin/env bash
# Builds the macOS Meeting Transcriber bundle (.app and .dmg).
#
# Steps:
#   1. Ensures the native audio helper (macos/.build/MeetingTranscriberAudio.app) exists.
#   2. Runs `cargo clippy --workspace -- -D warnings`.
#   3. Runs `cargo tauri build --bundles app,dmg` inside app/src-tauri.
#
# Output: target/release/bundle/macos/*.app and target/release/bundle/dmg/*.dmg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TAURI_DIR="$REPO_ROOT/app/src-tauri"
HELPER_APP="$REPO_ROOT/macos/.build/MeetingTranscriberAudio.app"
HELPER_BINARY="$HELPER_APP/Contents/MacOS/MeetingTranscriberAudio"
BUNDLE_DIR="$REPO_ROOT/target/release/bundle"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "build-macos.sh solo se ejecuta en macOS." >&2
  exit 1
fi

for tool in cargo; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Falta '$tool' en el PATH." >&2
    exit 1
  fi
done

if ! cargo tauri --version >/dev/null 2>&1; then
  echo "Falta 'cargo tauri' (instala con: cargo install tauri-cli --version '^2')." >&2
  exit 1
fi

if [[ ! -x "$HELPER_BINARY" ]]; then
  echo "==> Compilando el capturador de audio nativo (macos/build_audio_capture.sh)"
  "$REPO_ROOT/macos/build_audio_capture.sh"
fi

if [[ ! -x "$HELPER_BINARY" ]]; then
  echo "No se encontró el capturador tras compilarlo: $HELPER_BINARY" >&2
  exit 1
fi

echo "==> cargo clippy --workspace -- -D warnings"
(cd "$REPO_ROOT" && cargo clippy --workspace -- -D warnings)

echo "==> cargo tauri build --bundles app,dmg"
rm -f "$BUNDLE_DIR/macos"/rw.*.dmg 2>/dev/null || true
if ! (cd "$TAURI_DIR" && cargo tauri build --bundles app,dmg); then
  APP_BUNDLE="$BUNDLE_DIR/macos/Meeting Transcriber.app"
  if [[ -d "$APP_BUNDLE" ]] && ! compgen -G "$BUNDLE_DIR/dmg/*.dmg" >/dev/null; then
    echo "==> Finder Automation no disponible; generando .dmg con hdiutil..."
    rm -f "$BUNDLE_DIR/macos"/rw.*.dmg 2>/dev/null || true
    mkdir -p "$BUNDLE_DIR/dmg"
    DMG_STAGE="$(mktemp -d)"
    trap 'rm -rf "$DMG_STAGE"' EXIT
    cp -R "$APP_BUNDLE" "$DMG_STAGE/"
    ln -s /Applications "$DMG_STAGE/Applications"
    VERSION="$(sed -n 's/^version = "\([^"]*\)".*/\1/p' "$REPO_ROOT/Cargo.toml" | head -n1)"
    ARCH="$(uname -m)"
    DMG_OUT="$BUNDLE_DIR/dmg/Meeting Transcriber_${VERSION:-0.1.0}_${ARCH}.dmg"
    hdiutil create -volname "Meeting Transcriber" -srcfolder "$DMG_STAGE" -ov -format UDZO "$DMG_OUT"
  else
    exit 1
  fi
fi

echo "==> Artefactos generados"
find "$BUNDLE_DIR/macos" "$BUNDLE_DIR/dmg" -maxdepth 1 \( -name '*.app' -o -name '*.dmg' \) 2>/dev/null || true
