$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

$OsArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($OsArchitecture -ne "X64") {
    throw "Este instalador requiere Windows x64; arquitectura detectada: $OsArchitecture."
}

function Get-PythonCommand {
    $commands = @()
    foreach ($name in @("py", "python", "python3")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            $commands += $command.Source
        }
    }

    $knownPath = Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"
    if (Test-Path $knownPath) {
        $commands += $knownPath
    }

    foreach ($command in $commands) {
        if (-not [string]::IsNullOrWhiteSpace($command)) {
            return $command
        }
    }

    return $null
}

$PythonCommand = Get-PythonCommand
if (-not $PythonCommand) {
    $Winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $Winget) {
        throw "No se encontró Python ni winget. Instala Python 3.12 desde https://www.python.org/downloads/windows/ y vuelve a ejecutar este instalador."
    }

    Write-Host "Python no está instalado. Instalándolo con winget..."
    & $Winget.Source install --id Python.Python.3.12 --exact --scope user --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget no pudo instalar Python."
    }

    $PythonCommand = Get-PythonCommand
    if (-not $PythonCommand) {
        throw "Python fue instalado, pero Windows todavía no lo encuentra. Cierra esta ventana, abre una nueva y ejecuta install_windows.bat otra vez."
    }
}

$PythonArgs = @()
if ([System.IO.Path]::GetFileName($PythonCommand) -ieq "py.exe") {
    $PythonArgs = @("-3")
}

$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creando entorno virtual..."
    & $PythonCommand @PythonArgs -m venv venv
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo crear el entorno virtual."
    }
}

Write-Host "Instalando dependencias..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo actualizar pip."
}
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "No se pudieron instalar las dependencias."
}

if (-not (Test-Path (Join-Path $ProjectDir ".env"))) {
    Copy-Item (Join-Path $ProjectDir ".env.example") (Join-Path $ProjectDir ".env")
    Write-Host "Creado .env con la configuración predeterminada de Windows."
}

Write-Host "Descargando el modelo local Cactus Whistle..."
$ModelDir = Join-Path $ProjectDir "models"
$WhistleModel = Join-Path $ModelDir "whistle.cact"
$WhistleSha256 = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb"
$WhistleUrl = "https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$CurrentWhistleHash = if (Test-Path $WhistleModel) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $WhistleModel).Hash.ToLowerInvariant()
} else {
    ""
}
if ($CurrentWhistleHash -ne $WhistleSha256) {
    $TemporaryWhistle = "$WhistleModel.$([guid]::NewGuid().ToString('N')).download"
    $Curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $Curl) {
        throw "No se encontró curl.exe, necesario para descargar whistle.cact."
    }
    & $Curl.Source --fail --location --retry 3 --output $TemporaryWhistle $WhistleUrl
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo descargar Cactus Whistle."
    }
    $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $TemporaryWhistle).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $WhistleSha256) {
        throw "La verificación SHA-256 de whistle.cact falló. Archivo temporal: $TemporaryWhistle"
    }
    Move-Item -LiteralPath $TemporaryWhistle -Destination $WhistleModel -Force
}
Write-Host "Verificando el runtime Needle local..."
$env:NEEDLE_TELEMETRY = "0"
$env:DO_NOT_TRACK = "1"
$NeedleCli = Join-Path $ProjectDir "venv\Scripts\needle.exe"
if (-not (Test-Path $NeedleCli)) {
    throw "No se encontró el comando needle del entorno virtual."
}
& $NeedleCli fetch
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo preparar el runtime local Needle."
}
& $VenvPython -c "from transcription_engines import WhistleTranscriber; engine = WhistleTranscriber(language='es'); engine.close(); print('Cactus Whistle listo.')"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo cargar el modelo Cactus Whistle local."
}

Write-Host "Descargando y verificando Whisper de respaldo..."
& $VenvPython -c "from dotenv import load_dotenv; import os; from faster_whisper import WhisperModel; load_dotenv('.env'); name = os.getenv('WINDOWS_WHISPER_MODEL', '').strip() or 'small'; print(f'Modelo Windows: {name}'); WhisperModel(name, device='cpu', compute_type='int8'); print('Modelo listo.')"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo descargar o cargar el modelo Whisper."
}

Write-Host ""
Write-Host "Instalación terminada. Ejecuta start_transcribe.bat para comenzar."
