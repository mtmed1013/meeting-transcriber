$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "No existe la instalación. Ejecuta install_windows.bat primero."
}

Set-Location $ProjectDir
& $VenvPython (Join-Path $ProjectDir "meeting_transcriber.py")
