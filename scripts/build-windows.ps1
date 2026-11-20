# Builds the Windows Meeting Transcriber installers (MSI and NSIS).
#
# Output: target/release/bundle/msi/*.msi and target/release/bundle/nsis/*.exe
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$tauriDir = Join-Path $repoRoot 'app\src-tauri'
$bundleDir = Join-Path $repoRoot 'target\release\bundle'

function Require-Command([string]$name, [string]$hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Falta '$name' en el PATH. $hint"
    }
}

Require-Command 'cargo' 'Instala Rust desde https://rustup.rs'

$tauriOk = $false
try {
    $prevPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & cargo tauri --version *> $null
    $tauriOk = ($LASTEXITCODE -eq 0)
} catch {
    $tauriOk = $false
} finally {
    $ErrorActionPreference = $prevPref
}
if (-not $tauriOk) {
    throw "Falta 'cargo tauri'. Instalalo con: cargo install tauri-cli --version '^2'"
}

Write-Host '==> cargo tauri build --bundles msi,nsis' -ForegroundColor Cyan
Push-Location $tauriDir
try {
    & cargo tauri build --bundles msi,nsis
    if ($LASTEXITCODE -ne 0) {
        throw "cargo tauri build fallo con codigo $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host '==> Instaladores generados' -ForegroundColor Green
$installers = @()
$msiDir = Join-Path $bundleDir 'msi'
$nsisDir = Join-Path $bundleDir 'nsis'
if (Test-Path $msiDir) {
    $installers += Get-ChildItem -Path $msiDir -Filter '*.msi' -File
}
if (Test-Path $nsisDir) {
    $installers += Get-ChildItem -Path $nsisDir -Filter '*.exe' -File
}
if ($installers.Count -eq 0) {
    throw "No se encontro ningun instalador en $bundleDir"
}
$installers | ForEach-Object { Write-Host $_.FullName }
