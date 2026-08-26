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
