@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_transcribe.ps1"
if errorlevel 1 (
    echo.
    echo La transcripcion termino con un error.
    pause
)
