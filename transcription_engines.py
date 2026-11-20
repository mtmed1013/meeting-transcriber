"""Local transcription backends selected for the current operating system."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Protocol

import numpy as np


SAMPLE_RATE = 16_000
BLOCK_SECONDS = 30
WHISTLE_MODEL_FILE = "whistle.cact"


def model_dir() -> Path:
    """Local model directory; MT_MODELS_DIR overrides it when set and non-empty."""
    override = os.environ.get("MT_MODELS_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "models"


def whistle_model_path() -> Path:
    """Resolved at call time so a MT_MODELS_DIR loaded later (e.g. .env) applies."""
    return model_dir() / WHISTLE_MODEL_FILE


# Import-time snapshot kept for existing callers; prefer model_dir()/whistle_model_path().
MODEL_DIR = model_dir()
WHISTLE_MODEL_PATH = MODEL_DIR / WHISTLE_MODEL_FILE


class TranscriptionEngine(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...

    def close(self) -> None: ...


def _normalized_machine(machine: str) -> str:
    return machine.strip().lower().replace("-", "_")


def _is_macos_arm64(system: str, machine: str) -> bool:
    return system in {"darwin", "macos"} and machine in {"arm64", "aarch64"}


def resolve_engine(
    requested: str = "auto",
    *,
    system: str | None = None,
    machine: str | None = None,
) -> str:
    """Resolve auto by platform, while rejecting explicitly unsupported engines."""
    system = (system or platform.system()).strip().lower()
    machine = _normalized_machine(machine or platform.machine())
    requested = (requested or "auto").strip().lower()

    windows_x64 = system in {"windows", "win32"} and machine in {"amd64", "x86_64"}
    macos_arm64 = _is_macos_arm64(system, machine)
    if requested == "auto":
        if windows_x64 or macos_arm64:
            return "whistle"
        return "whisper"

    if requested not in {"whisper", "whistle"}:
        raise ValueError(
            "TRANSCRIPTION_ENGINE debe ser auto, whisper o whistle."
        )
    if requested == "whistle" and not (windows_x64 or macos_arm64):
        raise RuntimeError(
            "Whistle local está disponible en Windows x64 y macOS Apple Silicon."
        )
    return requested


def _audio_float32(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ValueError("El motor recibió un bloque de audio vacío.")
    return np.ascontiguousarray(samples)


def _require_model(path: Path, engine_name: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"No se encontró el modelo local de {engine_name}: {path}. "
            "Vuelve a ejecutar el instalador de esta plataforma."
        )


class WhisperTranscriber:
    name = "Whisper"

    def __init__(
        self,
        model_name: str,
        *,
        language: str | None,
        vad_filter: bool,
        beam_size: int,
        windows: bool,
    ):
        # Whisper is imported only when selected, keeping its weights/runtime out
        # of the primary engine's startup path. The installers prefetch the
        # selected Whisper model, so the running app never downloads weights.
        os.environ["HF_HUB_OFFLINE"] = "1"
        from faster_whisper import WhisperModel

        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.model = WhisperModel(
            model_name,
            device="auto",
            compute_type="int8",
            **(
                {"cpu_threads": max(1, min(4, (os.cpu_count() or 2) // 2))}
                if windows
                else {}
            ),
        )
        self.name = f"Whisper {model_name}"

    def transcribe(self, audio: np.ndarray) -> str:
        options = {
            "language": self.language,
            "vad_filter": self.vad_filter,
            "beam_size": self.beam_size,
        }
        if self.vad_filter:
            options["vad_parameters"] = {"min_silence_duration_ms": 300}
        segments, _ = self.model.transcribe(_audio_float32(audio), **options)
        return " ".join(segment.text for segment in segments).strip()

    def close(self) -> None:
        # Faster-Whisper/CTranslate2 does not expose a public explicit close API.
        pass


class WhistleTranscriber:
    name = "Cactus Whistle"

    def __init__(self, model_path: Path | None = None, *, language: str | None):
        if model_path is None:
            model_path = whistle_model_path()
        _require_model(model_path, "Cactus Whistle")
        # Keep inference local and opt out of Cactus's anonymous usage telemetry.
        os.environ["NEEDLE_TELEMETRY"] = "0"
        os.environ["DO_NOT_TRACK"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["NEEDLE_WHISTLE_WEIGHTS"] = str(model_path)

        try:
            import needle
        except ImportError as error:
            raise RuntimeError(
                "No está instalado cactus-needle en este entorno. "
                "Vuelve a ejecutar el instalador de esta plataforma."
            ) from error

        self.language = language
        self.model = needle.Whistle(weights=str(model_path))

    def transcribe(self, audio: np.ndarray) -> str:
        samples = _audio_float32(audio)
        if samples.size > SAMPLE_RATE * BLOCK_SECONDS:
            raise ValueError("Whistle acepta bloques de hasta 30 segundos.")
        result = self.model.transcribe(samples, language=self.language)
        return str(result.get("text", "")).strip()

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


def create_transcriber(
    requested: str = "auto",
    *,
    language: str | None = "es",
    whisper_model: str = "small",
    windows_whisper_model: str = "small",
    vad_filter: bool = True,
    beam_size: int = 5,
    system: str | None = None,
    machine: str | None = None,
) -> TranscriptionEngine:
    requested_engine = (requested or "auto").strip().lower()
    system_name = (system or platform.system()).strip().lower()
    machine_name = _normalized_machine(machine or platform.machine())
    engine = resolve_engine(requested_engine, system=system_name, machine=machine_name)
    windows = system_name in {"windows", "win32"}
    macos_arm64 = _is_macos_arm64(system_name, machine_name)

    if engine == "whisper":
        model_name = windows_whisper_model if windows else whisper_model
        return WhisperTranscriber(
            model_name,
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            windows=windows,
        )
    if engine == "whistle":
        try:
            return WhistleTranscriber(language=language)
        except Exception as whistle_error:
            if requested_engine != "auto" or not macos_arm64:
                raise

            print(
                "⚠️ Cactus Whistle no pudo cargarse antes de iniciar "
                f"({type(whistle_error).__name__}: {whistle_error}). "
                f"Intentando respaldo Whisper {whisper_model}."
            )
            try:
                return WhisperTranscriber(
                    whisper_model,
                    language=language,
                    vad_filter=vad_filter,
                    beam_size=beam_size,
                    windows=False,
                )
            except Exception as whisper_error:
                raise RuntimeError(
                    "No se pudo cargar Cactus Whistle "
                    f"({type(whistle_error).__name__}: {whistle_error}) ni el respaldo "
                    f"Whisper {whisper_model} "
                    f"({type(whisper_error).__name__}: {whisper_error}). "
                    "Revisa la instalación de ambos modelos."
                ) from whisper_error
    raise AssertionError(f"Motor de transcripción sin fábrica: {engine}")
