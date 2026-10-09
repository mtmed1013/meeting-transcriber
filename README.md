# Meeting Transcriber

Transcripción local de reuniones y guardado de notas en Obsidian.

## Motor local por plataforma

`TRANSCRIPTION_ENGINE=auto` selecciona el motor principal según el equipo:

| Plataforma | Motor predeterminado |
| --- | --- |
| Windows x64 | Cactus Whistle mediante Needle CPU (Whisper Small como respaldo) |
| macOS (Apple Silicon e Intel) | Whisper Medium |

Define `TRANSCRIPTION_ENGINE=whisper` para usar Faster-Whisper en cualquiera
de las dos plataformas, o `whistle` en Windows x64. Un motor
incompatible falla con un mensaje; no hay cambio de motor durante una reunión.

`TRANSCRIPTION_LANGUAGE=es` fija español. Whistle también admite `auto`.
Whisper conserva sus opciones actuales de VAD y beam size. Por compatibilidad,
si falta `TRANSCRIPTION_LANGUAGE`, se sigue leyendo `WHISPER_LANGUAGE` de
instalaciones anteriores.

Whistle usa únicamente `whistle.cact` mediante Needle CPU; no descarga
`needle3.cact`. Whisper se ejecuta localmente y el instalador prepara el modelo
seleccionado antes de iniciar la aplicación. El audio de las reuniones nunca se
envía a un servicio externo.

En macOS, Whisper transcribe bloques de 30 segundos y guarda cada resultado en
la nota de Obsidian. El audio se mantiene en memoria; no se crean WAV temporales.
Al detener con `Ctrl+C`, se procesa también el bloque parcial pendiente. Como
con cualquier captura sin persistencia de audio, un cierre inesperado puede
perder el bloque que todavía no alcanzó a transcribirse.

## Instalación en macOS

La forma recomendada es hacer doble clic en `install_macos.command`. El
instalador:

- comprueba las Xcode Command Line Tools y, si faltan, abre el instalador de
  Apple y espera a que termine;
- crea el entorno virtual e instala las dependencias;
- crea `.env` solo si todavía no existe;
- migra `TRANSCRIPTION_ENGINE=canary` o `nemotron` a `whisper` en un `.env`
  existente;
- descarga y verifica Whisper Medium (o el modelo configurado en `WHISPER_MODEL`);
- elimina del directorio local `models/` los pesos/temporales antiguos de
  Canary y Nemotron;
- compila el capturador nativo de audio del sistema;
- comprueba el permiso de macOS para capturar el audio del sistema.

Durante esa última comprobación, macOS puede solicitar permiso. El instalador
solo consulta el acceso; no inicia una grabación ni guarda audio. Acepta el
permiso para la aplicación que macOS identifica en el aviso (normalmente
Terminal si abriste `install_macos.command`). Si ofrece la opción, puedes
permitir solo audio.

La instalación de las Command Line Tools no puede hacerse de forma silenciosa:
macOS mostrará una ventana donde debes aceptar y posiblemente autorizar con una
cuenta administradora. Si cancelas esa ventana, vuelve a ejecutar el instalador
cuando quieras continuar. También necesitas tener Python 3 instalado; el
instalador no lo descarga automáticamente.

Después de la instalación, puedes ejecutar `start_transcribe.sh` manualmente o
usarlo desde tu Shortcut de Apple.

## Ejecución en macOS

La ruta de Obsidian, el motor y el idioma se configuran mediante variables de
entorno. Copia `.env.example` a `.env` y ajusta sus valores:

```bash
cp .env.example .env
```

Ejemplo:

```env
OBSIDIAN_DIR=/Users/TU_USUARIO/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian
TRANSCRIPTION_ENGINE=auto
TRANSCRIPTION_LANGUAGE=es
WHISPER_MODEL=medium
```

`WHISPER_MODEL` configura Faster-Whisper y puede ser `medium`, `small` o `tiny`;
`medium` es el predeterminado en macOS. El instalador conserva un `.env`
existente y descarga el modelo que allí esté configurado.

Para preparar también el motor y sus pesos locales, ejecuta una vez
`install_macos.command`. Instalar solo las dependencias con `pip` no descarga
los modelos. Después puedes iniciar la transcripción con:

```bash
./start_transcribe.sh
```

`start_transcribe.sh` comprueba las Command Line Tools y compila
automáticamente, una sola vez, el componente nativo que captura el audio del
sistema mediante ScreenCaptureKit. Si el permiso se rechazó previamente, macOS
no volverá a mostrar la solicitud automáticamente: habilita la aplicación que
macOS asocia con la captura en `Ajustes del Sistema > Privacidad y seguridad >
Grabación de pantalla y del audio del sistema`, y luego reinicia esa aplicación
antes de volver a ejecutar el transcriptor. El micrófono continúa siendo un
permiso independiente y puede solicitarse al iniciar una transcripción.

En macOS 15 o superior, el capturador nativo recibe en el mismo flujo el audio
de todo el sistema y el micrófono. Esto evita desincronizaciones entre dos
relojes de audio y funciona tanto con los altavoces como con audífonos. En
macOS 13–14 se conserva el fallback que captura el micrófono con el dispositivo
predeterminado. En ambos casos no se requiere BlackHole.

También significa que notificaciones, música u otros sonidos reproducidos
durante la reunión pueden entrar en la transcripción. El mezclador conserva
audio de una fuente aunque la otra tenga un retraso breve y registra los huecos
que deba rellenar.

La captura nativa necesita Xcode Command Line Tools. Si no usas el instalador,
puedes iniciar su instalación manualmente con:

```bash
xcode-select --install
```

En macOS, el audio combinado se entrega a la cola en memoria y Whisper procesa
bloques de ~30 segundos. Cada bloque reconocido se escribe en la nota antes de
continuar; no se generan archivos WAV en disco. `Ctrl+C` detiene la captura y
hace que macOS transcriba el último bloque parcial antes de cerrar la nota.

El capturador nativo se ejecuta en una sesión separada para que `Ctrl+C` llegue
primero al proceso principal. Así Python puede guardar la última transcripción y
cerrar el capturador de forma ordenada, sin reportar como error el código `-2`
que corresponde a una interrupción voluntaria.

La primera ejecución también puede solicitar permisos separados para `Grabación
de pantalla y audio del sistema` y `Micrófono`, porque el capturador nativo es
un helper independiente. Si el VAD omite voces muy bajas, puedes probar
`WHISPER_VAD_FILTER=false` en `.env` cuando uses Whisper. Las ganancias
`MAC_SYSTEM_GAIN` y `MAC_MICROPHONE_GAIN` permiten compensar niveles de captura
sin modificar el volumen audible del equipo.

Si `.env` de una instalación previa declara Canary o Nemotron como motor, el
instalador lo migra a `whisper` y conserva las demás variables. También limpia
los artefactos locales conocidos de esos dos modelos. El modelo Whistle se
publica con licencia Apache-2.0; revisa las licencias de modelos y runtimes
antes de redistribuirlos.

Si no defines `OBSIDIAN_DIR`, macOS usa la ruta de iCloud de Obsidian y Windows
usa `~/Documents/Obsidian`.

## Instalación en Windows

Descarga el proyecto en el equipo Windows x64 y haz doble clic en
`install_windows.bat`. El instalador:

- instala Python 3.12 con `winget` si Python no está disponible;
- crea el entorno virtual;
- instala Needle, Faster-Whisper y las dependencias de audio;
- crea `.env` con la configuración inicial;
- descarga `whistle.cact`, verifica su SHA-256 y prepara el runtime Needle;
- descarga y verifica Whisper Small (o el modelo de fallback configurado).

Después de instalar, inicia una reunión haciendo doble clic en
`start_transcribe.bat`. Puedes cambiar `OBSIDIAN_DIR`, `TRANSCRIPTION_ENGINE` y
`WINDOWS_WHISPER_MODEL` en `.env` antes de iniciar. En Windows, la aplicación
combina el micrófono físico con el loopback WASAPI del altavoz predeterminado
para recibir tu voz y el audio de la reunión o de cualquier otra aplicación. Si
necesitas seleccionar dispositivos concretos, usa
`WINDOWS_MICROPHONE` y `WINDOWS_SPEAKER` con parte de sus nombres.

### Windows 11, diademas y audio remoto

Windows usa Whistle como motor predeterminado y `WINDOWS_WHISPER_MODEL=small`
como respaldo. Define `TRANSCRIPTION_ENGINE=whisper` para usar ese respaldo;
puedes seleccionar `medium` o `tiny` con `WINDOWS_WHISPER_MODEL`. La variable
Windows tiene prioridad sobre `WHISPER_MODEL`. El instalador conserva un `.env`
existente y descarga el Whisper Windows seleccionado. El archivo
`needle3.cact` no se usa ni se descarga.

`TRANSCRIPTION_LANGUAGE=es` fija español en ambos sistemas. Whistle admite
`auto` si necesitas detección de idioma. Las instalaciones anteriores pueden
seguir usando `WHISPER_LANGUAGE` como variable de idioma. Los marcadores
experimentales de hablante están desactivados; `SPEAKER_MARKERS=true` los activa.
La captura nativa de macOS no cambia.

Para comprobar las fuentes en Windows sin cargar Whisper:

```powershell
.\venv\Scripts\python.exe windows_audio.py --check
```

Habla durante los primeros 15 segundos y reproduce una frase desde Windows
durante los siguientes 15. Comprueba los picos por fuente; muestras recibidas no
significan necesariamente sonido audible. La prueba no genera archivos de audio.

La cola conserva el tiempo estimado del audio para las horas de la nota. El reloj
de muestras no puede adelantarse indefinidamente al reloj real: se regula la
lectura de dispositivos virtuales y se limita el tramo final. Durante el cierre
se muestra una estimación basada en la velocidad medida, no un tiempo garantizado.

La captura usa un hilo por fuente, con COM inicializado en cada hilo, WASAPI en
modo compartido, lecturas de 50 ms y un búfer solicitado de un segundo. Una fuente
retrasada no bloquea a la otra. El loopback se selecciona por el ID del altavoz,
evitando confundir entrada y salida cuando ambas se llaman `Audio remoto`.

En equipos locales, el transcriptor prueba primero el micrófono predeterminado.
Si no puede abrirlo (por ejemplo, una diadema Jabra con un formato que el
controlador de captura no admite), busca otras entradas y prioriza nombres que
parezcan micrófonos integrados como `Microphone Array`, Realtek o Intel Smart
Sound. Indica el dispositivo alternativo en la consola. Si el micrófono
predeterminado funciona, lo conserva aunque esté silencioso; el silencio por sí
solo no permite distinguir una reunión en pausa de un fallo del micrófono.

El loopback continúa capturando desde la salida predeterminada (por ejemplo, la
Jabra), por lo que la transcripción puede combinar el audio de la reunión que
sale por la diadema con tu voz captada por el micrófono integrado. La voz puede
sonar más baja o recoger más ruido según la distancia al micrófono. Puedes
seleccionar manualmente una entrada con `WINDOWS_MICROPHONE`.

En una conexión desde macOS, activa en Windows App la redirección del micrófono
y del sonido. La reunión y el transcriptor deben ejecutarse en la misma sesión
de Windows. Las fuentes se reintentan independientemente; la consola muestra el
tipo y detalle de errores al abrirlas.

La consola distingue carga del modelo, recepción de muestras, silencio o nivel
bajo y procesamiento del motor seleccionado. Cada 15 segundos muestra
discontinuidades por fuente y huecos estimados. El aviso
`data discontinuity in recording` se cuenta
en ese resumen: indica cortes de captura, no la eliminación de texto ya escrito.
Si persisten, puedes probar `WINDOWS_BUFFER_SECONDS=2.0`; WASAPI puede ignorar
el tamaño solicitado. No se utiliza modo exclusivo.

En Windows se transcriben bloques exactos de 30 segundos de audio recibido,
compatibles con el límite de Whistle. Después de cada bloque se muestra el
tiempo de procesamiento y los segundos pendientes. Si el retraso crece
continuamente, el motor no alcanza el ritmo de la reunión: evalúa la carga del
equipo o selecciona Whisper Small como respaldo.
`Ctrl+C` detiene la captura y guarda los bloques pendientes antes de finalizar;
puede tardar si hay una cola acumulada. Las notas tienen nombres únicos y nunca
sobrescriben una nota existente. No se crean archivos WAV temporales.

La alineación de las fuentes usa tiempos monotónicos estimados: SoundCard no
expone los timestamps de WASAPI. No recupera audio que RDP no haya entregado.
Si un controlador queda bloqueado dentro de una lectura, se informa y el cierre
guarda el audio disponible sin esperar indefinidamente por ese hilo; para esa
fuente puede ser necesario reiniciar el programa. El texto guardado previamente
permanece en la nota.

Para validar en tu Windows 11: prueba voces conocidas tanto localmente como por
RDP, alterna conversación y silencio, desconecta/reconecta la sesión y termina
con `Ctrl+C`. Comprueba ambas voces, los contadores de cortes y el último texto.
Las pruebas automatizadas simulan fuentes; no sustituyen esta prueba WASAPI real.
