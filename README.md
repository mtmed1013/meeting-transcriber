# Meeting Transcriber

Transcripción local de reuniones con Faster-Whisper y guardado de notas en Obsidian.

## Instalación en macOS

La forma recomendada es hacer doble clic en `install_macos.command`. El
instalador:

- comprueba las Xcode Command Line Tools y, si faltan, abre el instalador de
  Apple y espera a que termine;
- crea el entorno virtual e instala las dependencias;
- crea `.env` solo si todavía no existe;
- descarga y verifica el modelo Whisper configurado;
- compila el capturador nativo de audio del sistema.

La instalación de las Command Line Tools no puede hacerse de forma silenciosa:
macOS mostrará una ventana donde debes aceptar y posiblemente autorizar con una
cuenta administradora. Si cancelas esa ventana, vuelve a ejecutar el instalador
cuando quieras continuar. También necesitas tener Python 3 instalado; el
instalador no lo descarga automáticamente.

Después de la instalación, puedes ejecutar `start_transcribe.sh` manualmente o
usarlo desde tu Shortcut de Apple.

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

Para ejecutar manualmente, después de instalar Python 3 y las Command Line
Tools:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
./start_transcribe.sh
```

`start_transcribe.sh` comprueba las Command Line Tools y compila
automáticamente, una sola vez, el componente nativo que captura el audio del
sistema mediante ScreenCaptureKit. La primera vez, macOS solicitará
autorización para grabar el audio del sistema. Puedes
revisarla en `Configuración del Sistema > Privacidad y seguridad > Grabación de
pantalla y audio del sistema`. El micrófono continúa siendo un permiso
independiente.

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

El audio no se guarda en archivos WAV temporales. La transcripción mantiene el
flujo existente de Whisper, Obsidian y finalización con `Ctrl+C`.

El capturador nativo se ejecuta en una sesión separada para que `Ctrl+C` llegue
primero al proceso principal. Así Python puede guardar la última transcripción y
cerrar el capturador de forma ordenada, sin reportar como error el código `-2`
que corresponde a una interrupción voluntaria.

La primera ejecución también puede solicitar permisos separados para `Grabación
de pantalla y audio del sistema` y `Micrófono`, porque el capturador nativo es
un helper independiente. Si el VAD omite voces muy bajas, puedes probar
`WHISPER_VAD_FILTER=false` en `.env`. Las ganancias `MAC_SYSTEM_GAIN` y
`MAC_MICROPHONE_GAIN` permiten compensar niveles de captura sin modificar el
volumen audible del equipo.

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
audio de la reunión o de cualquier otra aplicación. Si necesitas seleccionar
dispositivos concretos, usa
`WINDOWS_MICROPHONE` y `WINDOWS_SPEAKER` con parte de sus nombres.

### Windows 11 y sesiones RDP / Windows App

La captura usa un hilo por fuente, con COM inicializado en cada hilo, WASAPI en
modo compartido, lecturas de 50 ms y un búfer solicitado de un segundo. Una fuente
retrasada no bloquea a la otra. El loopback se selecciona por el ID del altavoz,
evitando confundir entrada y salida cuando ambas se llaman `Audio remoto`.

Al conectar desde macOS, activa en Windows App la redirección del micrófono y
del sonido. La reunión y el transcriptor deben ejecutarse en la misma sesión de
Windows. Deja `WINDOWS_MICROPHONE` y `WINDOWS_SPEAKER` vacíos para seguir los
dispositivos predeterminados, incluidos los remotos. Se comprueban los cambios
de dispositivo cada dos segundos cuando el controlador entrega datos. Ante un
error, cada fuente reintenta independientemente con esperas de hasta 15 segundos.

La consola distingue carga del modelo, recepción de muestras, silencio o nivel
bajo y procesamiento de Whisper. Cada 15 segundos muestra discontinuidades por
fuente y huecos estimados. El aviso `data discontinuity in recording` se cuenta
en ese resumen: indica cortes de captura, no la eliminación de texto ya escrito.
Si persisten, puedes probar `WINDOWS_BUFFER_SECONDS=2.0`; WASAPI puede ignorar
el tamaño solicitado. No se utiliza modo exclusivo.

En Windows se transcriben bloques de 30 segundos de audio recibido. Después de
cada bloque se muestra el tiempo de procesamiento y los segundos pendientes.
Si el retraso crece continuamente, Whisper no alcanza el ritmo de la reunión:
evalúa la carga del equipo y, si aceptas menor precisión, prueba `small`.
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
