# Meeting Transcriber

Transcripción local de reuniones con Faster-Whisper y guardado de notas en Obsidian.

## Ejecución en macOS

La ruta de Obsidian y el modelo de Whisper se configuran mediante variables de
entorno. Copia `.env.example` a `.env` y ajusta sus valores:

```bash
cp .env.example .env
```

Ejemplo:

```env
OBSIDIAN_DIR=/Users/TU_USUARIO/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian
WHISPER_MODEL=medium
```

`WHISPER_MODEL` puede ser `medium`, `small` o `tiny`. `medium` es el valor
predeterminado y prioriza la precisión.

Para ejecutar manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python meeting_transcriber.py
```

Si no defines `OBSIDIAN_DIR`, macOS usa la ruta de iCloud de Obsidian y Windows
usa `~/Documents/Obsidian`.

## Instalación en Windows

Descarga el proyecto en el equipo Windows y haz doble clic en
`install_windows.bat`. El instalador:

- instala Python 3.12 con `winget` si Python no está disponible;
- crea el entorno virtual;
- instala Faster-Whisper y las dependencias de audio;
- crea `.env` con la configuración inicial;
- descarga y verifica el modelo Whisper configurado.

Después de instalar, inicia una reunión haciendo doble clic en
`start_transcribe.bat`. Puedes cambiar `OBSIDIAN_DIR` y `WHISPER_MODEL` en
`.env` antes de iniciar. En Windows, la aplicación combina el micrófono físico
con el loopback WASAPI del altavoz predeterminado para recibir tu voz y el
audio de Teams. Si necesitas seleccionar dispositivos concretos, usa
`WINDOWS_MICROPHONE` y `WINDOWS_SPEAKER` con parte de sus nombres.
