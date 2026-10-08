"""Independent shared-mode WASAPI readers and a bounded-latency mixer."""

import ctypes
import queue
import threading
import time
import warnings
from collections import deque
from contextlib import contextmanager

import numpy as np


@contextmanager
def com_thread():
    """COM apartments belong to threads, not to the SoundCard import."""
    ole32 = ctypes.OleDLL("ole32")
    result = ole32.CoInitializeEx(None, 0)
    if result not in (0, 1):
        raise RuntimeError(f"No se pudo inicializar COM: {result:#x}")
    try:
        yield
    finally:
        ole32.CoUninitialize()


def select_device(sc, source, microphone_name, speaker_name):
    if source == "microphone":
        return sc.get_microphone(microphone_name) if microphone_name else sc.default_microphone()
    speaker = sc.get_speaker(speaker_name) if speaker_name else sc.default_speaker()
    # IDs disambiguate a microphone and a loopback both named "Audio remoto".
    device = sc.get_microphone(id=speaker.id, include_loopback=True)
    if not device.isloopback:
        raise RuntimeError("El dispositivo seleccionado no es un loopback WASAPI")
    return device


def microphone_candidates(sc, microphone_name=""):
    """Try the configured/default mic first, then prefer built-in inputs."""
    result = []
    if microphone_name:
        try:
            result.append(sc.get_microphone(microphone_name))
        except Exception as error:
            print(f"⚠️ No se encontró WINDOWS_MICROPHONE='{microphone_name}': "
                  f"{type(error).__name__}: {error!r}; buscando entradas disponibles")
    try:
        preferred = sc.default_microphone()
        if all(device.id != preferred.id for device in result):
            result.append(preferred)
    except Exception as error:
        print(f"⚠️ No se pudo consultar el micrófono predeterminado: "
              f"{type(error).__name__}: {error!r}")
    try:
        available = sc.all_microphones(include_loopback=False)
    except TypeError:
        available = sc.all_microphones()

    integrated_terms = (
        "array", "integrated", "internal", "built-in", "builtin",
        "realtek", "conexant", "intel smart sound", "microphone array",
    )
    available.sort(key=lambda device: (
        0 if any(term in device.name.casefold() for term in integrated_terms) else 1,
        device.name.casefold(),
    ))
    seen = {device.id for device in result}
    for device in available:
        if not getattr(device, "isloopback", False) and device.id not in seen:
            result.append(device)
            seen.add(device.id)
    return result


class Timeline:
    """Align sample spans using an estimated monotonic capture timeline."""

    def __init__(self, rate):
        self.rate = rate
        self.chunks = deque()

    def append(self, start, samples):
        self.chunks.append((start, samples))

    def read(self, start, frames):
        end = start + frames / self.rate
        output = np.zeros(frames, dtype=np.float32)
        covered = np.zeros(frames, dtype=bool)
        for chunk_start, samples in self.chunks:
            left = max(0, int(round((chunk_start - start) * self.rate)))
            offset = max(0, int(round((start - chunk_start) * self.rate)))
            count = min(frames - left, len(samples) - offset)
            if count > 0:
                output[left:left + count] = samples[offset:offset + count]
                covered[left:left + count] = True
        while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) / self.rate <= end:
            self.chunks.popleft()
        if self.chunks and self.chunks[0][0] < end:
            chunk_start, samples = self.chunks[0]
            drop = min(len(samples), max(0, int(round((end - chunk_start) * self.rate))))
            self.chunks[0] = (chunk_start + drop / self.rate, samples[drop:])
        return output, int(covered.sum())


def capture(output_queue, stop_event, error_queue, rate=16000,
            microphone_name="", speaker_name="", buffer_seconds=1.0):
    import soundcard as sc

    packets = queue.Queue()
    local = threading.local()
    labels = {"microphone": "micrófono", "system": "loopback"}
    state_lock = threading.Lock()
    states = {source: {"last": time.monotonic(), "level": None, "gaps": 0,
                       "active": False, "frames": 0, "peak": 0.0,
                       "device": ""} for source in labels}
    original_warning = warnings.showwarning

    def report_warning(message, category, filename, lineno, file=None, line=None):
        source = getattr(local, "source", None)
        if source and "data discontinuity in recording" in str(message):
            with state_lock:
                states[source]["gaps"] += 1
            return
        original_warning(message, category, filename, lineno, file=file, line=line)

    def reader(source):
        local.source = source
        try:
            with com_thread():
                failures = 0
                failed_devices = {}
                while not stop_event.is_set():
                    try:
                        if source == "microphone":
                            devices = microphone_candidates(sc, microphone_name)
                        else:
                            devices = [select_device(sc, source, microphone_name, speaker_name)]
                    except Exception as error:
                        failures += 1
                        delay = min(2 ** min(failures, 4), 15)
                        print(f"⚠️ No se pudieron consultar dispositivos de {labels[source]}: "
                              f"{type(error).__name__}: {error!r}; reintento en {delay}s")
                        stop_event.wait(delay)
                        continue
                    any_opened = False
                    for index, device in enumerate(devices):
                        if stop_event.is_set():
                            break
                        if failed_devices.get(device.id, 0) > time.monotonic():
                            continue
                        try:
                            print(f"🎙️ Probando {labels[source]}: {device.name} (id={device.id})")
                            with device.recorder(samplerate=rate, channels=2,
                                                 blocksize=int(rate * buffer_seconds),
                                                 exclusive_mode=False) as recorder:
                                if source == "microphone":
                                    if index:
                                        print(f"🔁 Micrófono predeterminado no disponible; usando respaldo: {device.name}")
                                    else:
                                        print(f"✅ Micrófono seleccionado: {device.name}")
                                with state_lock:
                                    states[source]["device"] = device.name
                                    states[source]["active"] = True
                                any_opened = True
                                cursor = time.monotonic()
                                check_at = cursor + 2
                                first = True
                                while not stop_event.is_set():
                                    data = recorder.record(numframes=rate // 20)
                                    now = time.monotonic()
                                    samples = np.nan_to_num(np.mean(data, axis=1),
                                                            nan=0.0, posinf=0.0, neginf=0.0)
                                    if len(samples):
                                        if now - cursor > buffer_seconds + 0.5:
                                            cursor = now - len(samples) / rate
                                        packets.put((source, cursor, samples.copy()))
                                        cursor += len(samples) / rate
                                        stop_event.wait(max(0, cursor - time.monotonic()))
                                        with state_lock:
                                            states[source].update(last=now, active=True,
                                                level=float(np.sqrt(np.mean(samples ** 2))))
                                            states[source]["frames"] += len(samples)
                                            states[source]["peak"] = max(states[source]["peak"], float(np.max(np.abs(samples))))
                                        if first:
                                            print(f"✅ Recibiendo audio de {labels[source]} (puede ser silencio)")
                                            first = False
                                        failures = 0
                                    if source == "system" and now >= check_at:
                                        current = select_device(sc, source, microphone_name, speaker_name)
                                        if current.id != device.id:
                                            print("🔄 Cambió el altavoz; reabriendo el loopback")
                                            break
                                        check_at = now + 2
                            if source == "microphone" and not stop_event.is_set():
                                # A healthy fallback stays selected; don't churn back to
                                # a headset endpoint that already failed to open.
                                continue
                        except Exception as error:
                            detail = f"{type(error).__name__}: {error!r}"
                            print(f"⚠️ No se pudo abrir {labels[source]} '{device.name}': {detail}")
                            failed_devices[device.id] = time.monotonic() + 60
                            with state_lock:
                                states[source]["active"] = False
                            if source != "microphone":
                                failures += 1
                                stop_event.wait(min(2 ** min(failures, 4), 15))
                                break
                    if source == "microphone" and not any_opened:
                        failures += 1
                        delay = min(2 ** min(failures, 4), 15)
                        print(f"⚠️ No se pudo abrir ningún micrófono; nueva búsqueda en {delay}s")
                        stop_event.wait(delay)
        except Exception as error:
            error_queue.put(RuntimeError(f"Captura {labels[source]}: {error}"))

    warnings.showwarning = report_warning
    workers = [threading.Thread(target=reader, args=(source,),
                                name=f"wasapi-{source}", daemon=True) for source in labels]
    timelines = {source: Timeline(rate) for source in labels}
    frames = rate // 20
    step = frames / rate
    jitter = buffer_seconds + 0.2
    next_start = time.monotonic()
    next_report = next_start + 15
    missing = {source: 0 for source in labels}

    def emit_until(deadline):
        nonlocal next_start
        while next_start + step <= deadline:
            mic, mic_count = timelines["microphone"].read(next_start, frames)
            system, system_count = timelines["system"].read(next_start, frames)
            missing["microphone"] += frames - mic_count
            missing["system"] += frames - system_count
            if mic_count or system_count:
                output_queue.put((next_start, ((mic + system) * 0.5).reshape(-1, 1)))
            next_start += step

    def drain():
        while True:
            try:
                source, start, samples = packets.get_nowait()
                timelines[source].append(start, samples)
            except queue.Empty:
                break

    try:
        for worker in workers:
            worker.start()
        print("🛡️ WASAPI: fuentes independientes, modo compartido, bloques de 50 ms")
        while not stop_event.wait(0.02):
            if not any(worker.is_alive() for worker in workers):
                raise RuntimeError("Ninguna fuente WASAPI pudo iniciar; revisa los errores de captura")
            drain()
            now = time.monotonic()
            emit_until(now - jitter)
            if now >= next_report:
                with state_lock:
                    snapshot = {source: dict(state) for source, state in states.items()}
                    for state in states.values():
                        state["gaps"] = 0
                        state["peak"] = 0.0
                for source, state in snapshot.items():
                    age = now - state["last"]
                    level = state["level"]
                    status = "sin muestras" if age > 3 else (
                        "silencio/nivel bajo" if level is None or level < 0.001 else "audio presente")
                    print(f"📊 {labels[source]} ({state['device'] or 'sin dispositivo'}): {status}; discontinuidades={state['gaps']}; "
                          f"RMS={level or 0:.6f}; pico={state['peak']:.6f}; "
                          f"recibido={state['frames'] / rate:.1f}s; huecos={missing[source] / rate:.1f}s/15s")
                    if age > 3 and state["active"]:
                        print(f"⚠️ {labels[source]} sin entregar datos; comprueba la conexión RDP")
                    missing[source] = 0
                next_report = now + 15
    finally:
        stop_event.set()
        deadline = time.monotonic() + 3
        for worker in workers:
            worker.join(timeout=max(0, deadline - time.monotonic()))
        drain()
        ends = [start + len(samples) / rate for timeline in timelines.values()
                for start, samples in timeline.chunks]
        # Include a partial last window, padded with silence.
        if ends:
            emit_until(min(max(ends), time.monotonic()) + step)
        for worker in workers:
            if worker.is_alive():
                print(f"⚠️ {worker.name} sigue bloqueado en el controlador; guardando audio disponible")
        if warnings.showwarning is report_warning:
            warnings.showwarning = original_warning
        with output_queue.mutex:
            pending = sum(len(packet[1]) for packet in output_queue.queue) / rate
        print(f"📊 Captura cerrada: audio pendiente en cola={pending:.1f}s")


if __name__ == "__main__":
    import os
    import argparse
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Prueba de fuentes WASAPI sin cargar Whisper")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    print("Primeros 15s: habla por el micrófono. Siguientes 15s: reproduce una frase en Windows.")
    print("Comprueba el pico de cada fuente. Esta prueba no guarda audio ni genera una nota.")
    stop = threading.Event()
    timer = threading.Timer(31, stop.set)
    timer.start()
    errors = queue.Queue()
    try:
        capture(queue.Queue(), stop, errors,
                microphone_name=os.getenv("WINDOWS_MICROPHONE", ""),
                speaker_name=os.getenv("WINDOWS_SPEAKER", ""))
    except KeyboardInterrupt:
        stop.set()
    finally:
        timer.cancel()
    while not errors.empty():
        print(f"❌ {errors.get()}")
