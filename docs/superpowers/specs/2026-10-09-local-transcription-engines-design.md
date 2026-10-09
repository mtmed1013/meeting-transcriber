# Diseño: motores locales de transcripción por plataforma

## Objetivo

Usar el motor local más apropiado por plataforma sin cambiar la captura de audio, el guardado incremental ni el formato/ruta de las notas de Obsidian. Mantener Whisper como fallback seleccionable durante la validación.

## Diseño elegido

Un adaptador pequeño de transcripción separará el backend del flujo de captura. La selección será perezosa: el proceso carga únicamente el backend elegido, no inicializa Faster-Whisper al importar el programa. `TRANSCRIPTION_ENGINE=auto` elegirá por defecto Whistle en Windows x64 y Canary en macOS Apple Silicon. `TRANSCRIPTION_ENGINE=whisper` seleccionará Faster-Whisper explícitamente; la variable ausente o `auto` permite esos defaults. En macOS Intel y plataformas no compatibles, `auto` conservará Whisper en vez de fallar al inicializar Canary.

El contrato interno recibirá audio mono PCM `float32` a 16 kHz y devolverá texto. Los adaptadores aislarán parámetros específicos del motor, como VAD y beam size de Whisper. Whistle recibirá muestras locales y el archivo local `whistle.cact`; no se inicializará el agente LLM Needle, no se enviará audio a un servicio remoto y se desactivará la telemetría. Canary usará el binding Python de `transcribe.cpp`, modelo GGUF Q8_0 local y el idioma `es` explícito. No se crearán WAV temporales.

## Flujo e instaladores

- Windows x64: Whistle local predeterminado y Whisper Small disponible como fallback.
- macOS Apple Silicon: Canary 180M Flash Q8_0 predeterminado y Whisper Medium disponible como fallback. La ruta Metal la proporciona el runtime nativo compatible.
- macOS Intel: fallback automático a Whisper Medium; no instalar un backend Canary no soportado.
- Cada instalador prepara solo el motor y el modelo principal de su plataforma más el fallback Whisper de esa misma plataforma. Los pesos se guardan localmente; la red se usa para instalar dependencias y descargar esos artefactos, nunca para procesar audio.
- No reemplazar ni borrar un `.env` existente. Si no hay selección explícita, el valor por defecto por plataforma debe seguir funcionando. Documentar el selector y las variables Whisper existentes.

## Captura, segmentación y notas

La captura WASAPI/ScreenCaptureKit, la mezcla de micrófono/audio del sistema, las colas, el encabezado y el guardado en Obsidian no cambian. Se mantiene el checkpoint por 30 segundos de audio recibido y el flush final al detener.

Windows ya segmenta por cantidad de muestras. macOS debe adoptar segmentación por muestras para que un retraso de inferencia no convierta la siguiente solicitud en un bloque de tamaño indefinido; los datos capturados mientras se infiere permanecen en cola y se transcriben después. El backend Canary no es streaming; por ello el comportamiento sigue siendo por bloques. Se conserva la marca temporal actual por bloque, sin prometer timestamps palabra por palabra ni diarización del motor.

## Errores y compatibilidad

La selección de un backend no disponible debe fallar con un mensaje que indique el motor seleccionado y la acción de instalación/configuración. No cambiar silenciosamente a otro backend durante una reunión, porque eso ocultaría problemas y haría inconsistente la calidad. El usuario puede cambiar a Whisper con la variable de configuración y reiniciar la aplicación. Evitar cargar simultáneamente los dos modelos.

Se deben fijar versiones compatibles de Needle/Whistle y `transcribe-cpp`/runtime nativo después de validar las APIs oficiales y las ruedas por plataforma. El instalador debe comprobar que el modelo esperado existe y que el backend seleccionado puede inicializarse antes de reportar éxito.

## Validación

- Pruebas unitarias con backends simulados: selección por sistema/arquitectura, parámetros e idioma, conversión de muestras, salida vacía/error y persistencia por bloques.
- Verificar que Whisper conserva sus opciones actuales y que los backends nuevos no reciben argumentos exclusivos de Whisper.
- En instalación limpia, verificar que Windows baja solo Whistle + Whisper Small y macOS Apple Silicon solo Canary + Whisper Medium; confirmar que las transcripciones se procesan localmente y no se crean WAV temporales.
- Comparar Canary y Whisper Medium con los mismos segmentos reales en español, español con términos en inglés, nombres propios, voces superpuestas y ruido. Medir errores de palabras, omisiones, latencia y RAM; no inferir calidad de los benchmarks públicos solamente.
- Comprobar finalización con Ctrl+C, guardado del bloque final y conservación exacta del esquema de nota.

## Fuera de alcance

No rediseñar la captura, selección de dispositivos, permisos del sistema, layout de Obsidian, frecuencia de checkpoint (30 s), interfaz de control ni diarización. No quitar Whisper en esta etapa. No habilitar servicios remotos de transcripción.
