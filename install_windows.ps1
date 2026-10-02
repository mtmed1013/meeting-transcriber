$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

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

Write-Host "Descargando y verificando el modelo Whisper..."
& $VenvPython -c "from dotenv import load_dotenv; import os; from faster_whisper import WhisperModel; load_dotenv('.env'); name = os.getenv('WINDOWS_WHISPER_MODEL', '').strip() or 'small'; print(f'Modelo Windows: {name}'); WhisperModel(name, device='cpu', compute_type='int8'); print('Modelo listo.')"
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo descargar o cargar el modelo Whisper."
}

Write-Host ""
Write-Host "Instalación terminada. Ejecuta start_transcribe.bat para comenzar."
