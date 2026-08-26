# Meeting Transcriber

Transcripción local de reuniones con Faster-Whisper y guardado de notas en Obsidian.

## Ejecución en macOS

El lanzador `start_transcribe.sh` abre Terminal, activa el entorno virtual y ejecuta el transcriptor.

Para ejecutar manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python meeting_transcriber.py
```

La ruta de Obsidian se configura actualmente en `meeting_transcriber.py` y debe adaptarse por equipo.
