# Diseño: Nemotron en streaming para macOS

## Objetivo

Reemplazar Canary exclusivamente en macOS por Nemotron 3.5 ASR Streaming 0.6B,
ejecutado localmente con `transcribe.cpp` y Metal en Apple Silicon. El texto
estable se debe guardar en la nota de Obsidian aproximadamente cada cinco
segundos, sin esperar bloques de 30 segundos. Windows y su motor no cambian.

Se conserva Whisper Medium como fallback configurable en macOS. No se usa API,
servidor ni servicio remoto de transcripción; el GGUF se procesa en el equipo.

## Alternativas consideradas

1. **Streaming persistente (elegida):** alimentar una sesión Nemotron durante
   toda la reunión y persistir solo el texto comprometido por el modelo. Da la
   latencia esperada y evita volver a transcribir bloques solapados.
2. **Bloques independientes:** llamar al modo offline cada 30 segundos. Cambia
   menos código, pero no aprovecha la arquitectura streaming ni la rapidez de
   Handy; también pierde contexto entre bloques.

## Arquitectura y flujo

- `TRANSCRIPTION_ENGINE=auto` selecciona Whistle en Windows x64, Nemotron en
  macOS Apple Silicon y Whisper en Mac Intel u otras plataformas. El valor
  `whisper` continúa disponible para elegir el fallback. Canary deja de ser un
  motor válido.
- El adaptador de Nemotron carga el GGUF Q8_0 local con el backend Metal y abre
  una sesión de streaming por reunión. Consume audio mono `float32` a 16 kHz,
  igual al contrato actual.
- Para priorizar precisión, se usa `ParakeetStreamOptions` con contexto derecho
  13 (1,12 s). El idioma español configurado como `es` se normaliza a `es-ES`;
  se admite `auto` si el usuario lo selecciona.
- El consumidor de audio alimenta el stream de forma incremental. Lee el texto
  comprometido y compara con el último prefijo persistido; solo agrega el sufijo
  nuevo, evitando duplicados causados por hipótesis tentativas que cambian.
- Al existir texto estable nuevo, se añade a la nota existente con el formato y
  timestamp actuales, como máximo cada aproximadamente cinco segundos. Al
  detener con Ctrl+C, se detiene la captura, se drena la cola y se finaliza el
  stream; cualquier texto final estable se agrega exactamente una vez.
- La captura ScreenCaptureKit, mezcla, permisos, diagnóstico de niveles, ruta y
  encabezado de Obsidian permanecen intactos. No se guardan WAV temporales.

## Instalador y limpieza de Canary

- El instalador de macOS descarga `nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf`
  desde el repositorio GGUF de Handy, con revisión fijada y verificación de
  tamaño y SHA-256 antes de instalarlo.
- Primero valida que el modelo nuevo carga y puede iniciar una sesión. Después
  borra únicamente `models/canary-180m-flash-Q8_0.gguf` y los temporales de
  descarga que coincidan con `.canary-180m-flash.*`. No elimina otros modelos,
  la carpeta de modelos ni archivos de Windows.
- Conserva la dependencia nativa `transcribe-cpp` (también es el runtime de
  Nemotron), mantiene la descarga/verificación de Whisper Medium y no reemplaza
  el `.env` del usuario. Si encuentra el valor exacto
  `TRANSCRIPTION_ENGINE=canary`, lo migra a `auto` preservando las demás
  variables y contenido.
- macOS Intel conserva Whisper Medium. Los instaladores y requisitos de Windows
  no incorporan Nemotron ni Canary.

## Errores y persistencia

- Un error al cargar o iniciar el stream se reporta claramente y no cambia de
  motor silenciosamente durante la reunión. Las líneas ya comprometidas quedan
  guardadas; el usuario puede iniciar otra sesión seleccionando Whisper.
- Un cierre normal o Ctrl+C finaliza el stream y guarda el último texto estable.
  Como cualquier checkpoint, hasta unos cinco segundos de texto aún no
  comprometido podrían perderse ante una caída abrupta del proceso.
- No se promete diarización. Se conserva el formato existente de nota; no se
  añade una línea por cada hipótesis tentativa.

## Validación y criterios de aceptación

- Pruebas unitarias con stream simulado: selección por plataforma, locale,
  configuración Parakeet, deduplicación de prefijos, checkpoints, errores y
  finalización con texto pendiente.
- Probar en Apple Silicon con el GGUF real: cargar en Metal, transcribir audio
  español, validar `feed`/`finalize` y comprobar que la nota recibe solo texto
  estable sin duplicaciones.
- Ejecutar una prueba prolongada de al menos 30 minutos para detectar omisiones
  o bloqueo del decodificador; comparar una muestra contra Whisper Medium.
- Probar Ctrl+C y confirmar el guardado final. Repetir pruebas existentes de
  captura/macOS y Windows para verificar que el cambio de motor no altera sus
  flujos.
- En una reinstalación macOS satisfactoria, confirmar que se instala Nemotron,
  Whisper sigue disponible, Canary exacto desaparece y ningún otro GGUF se
  elimina. Si Nemotron no pasa la validación, conservar Canary.

## Riesgos y límites

- Nemotron Q8_0 ocupa aproximadamente 751 MB; es mayor que el Canary Q8_0
  instalado actualmente. La memoria en ejecución será mayor que el tamaño del
  archivo y se medirá en la prueba local.
- Handy publicó un reporte, posteriormente cerrado, sobre texto incompleto en
  streams largos de este modelo. No se presupone que la causa siga presente ni
  que ya esté corregida; la prueba prolongada es requisito antes de confiarle
  reuniones largas.
- Los WER publicados son benchmarks generales, no garantizan la calidad con
  reuniones, audio de sistema mezclado, acentos colombianos o voces simultáneas.

## Fuentes

- Modelo, locale español, tamaño y ajuste de contexto:
  <https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/nemotron-3.5-asr-streaming-0.6b.md>
- API y contrato de streaming de `transcribe.cpp`:
  <https://github.com/handy-computer/transcribe.cpp/blob/main/docs/bindings.md>
- Handy 0.9 y selección de modelos streaming:
  <https://github.com/cjpais/Handy/blob/main/src/content/release-notes/0.9.0.md>
- Reporte de transcripción incompleta en streams largos:
  <https://github.com/cjpais/Handy/issues/2125>
