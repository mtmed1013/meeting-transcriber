import ast
import contextlib
import io
import os
import queue
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import transcription_engines as engines


class TranscriptionEngineTests(unittest.TestCase):
    def test_auto_selects_platform_defaults(self):
        self.assertEqual(
            engines.resolve_engine("auto", system="Windows", machine="AMD64"),
            "whistle",
        )
        self.assertEqual(
            engines.resolve_engine("auto", system="Darwin", machine="arm64"),
            "whisper",
        )
        self.assertEqual(
            engines.resolve_engine("auto", system="Darwin", machine="x86_64"),
            "whisper",
        )

    def test_removed_engines_are_rejected_and_whistle_is_windows_only(self):
        with self.assertRaisesRegex(RuntimeError, "Windows x64"):
            engines.resolve_engine("whistle", system="Darwin", machine="arm64")
        for removed_engine in ("canary", "nemotron"):
            with self.subTest(engine=removed_engine), self.assertRaises(ValueError):
                engines.resolve_engine(removed_engine, system="Darwin", machine="arm64")

    def test_factory_uses_whistle_on_windows_and_medium_whisper_on_macos(self):
        with patch.object(engines, "WhistleTranscriber", return_value="whistle") as whistle:
            self.assertEqual(
                engines.create_transcriber(
                    "auto", system="Windows", machine="AMD64"
                ),
                "whistle",
            )
            whistle.assert_called_once_with(language="es")

        with patch.object(engines, "WhisperTranscriber", return_value="whisper") as whisper:
            self.assertEqual(
                engines.create_transcriber(
                    "auto", system="Darwin", machine="arm64"
                ),
                "whisper",
            )
            whisper.assert_called_once_with(
                "medium", language="es", vad_filter=True, beam_size=5, windows=False,
            )

        with patch.object(engines, "WhisperTranscriber", return_value="whisper") as whisper:
            self.assertEqual(
                engines.create_transcriber(
                    "whisper",
                    system="Windows",
                    machine="AMD64",
                    windows_whisper_model="small",
                ),
                "whisper",
            )
            whisper.assert_called_once_with(
                "small", language="es", vad_filter=True, beam_size=5, windows=True,
            )

    def test_whistle_uses_local_weights_and_disables_network_telemetry(self):
        calls = {}

        class FakeWhistle:
            def __init__(self, *, weights):
                calls["weights"] = weights

            def transcribe(self, samples, *, language):
                calls["audio"] = samples
                calls["language"] = language
                return {"text": "texto local"}

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "whistle.cact"
            model_path.write_bytes(b"local-model-fixture")
            fake_needle = types.SimpleNamespace(Whistle=FakeWhistle)
            with patch.dict(sys.modules, {"needle": fake_needle}), patch.dict(
                os.environ,
                {"NEEDLE_TELEMETRY": "1", "DO_NOT_TRACK": "0", "HF_HUB_OFFLINE": "0"},
            ):
                engine = engines.WhistleTranscriber(model_path, language="es")
                text = engine.transcribe(np.array([[0.1], [0.2]], dtype=np.float64))
                telemetry = os.environ["NEEDLE_TELEMETRY"]
                offline = os.environ["HF_HUB_OFFLINE"]

        self.assertEqual(text, "texto local")
        self.assertEqual(calls["weights"], str(model_path))
        self.assertEqual(calls["language"], "es")
        self.assertEqual(calls["audio"].dtype, np.float32)
        self.assertEqual(calls["audio"].ndim, 1)
        self.assertEqual(telemetry, "0")
        self.assertEqual(offline, "1")

    def test_whisper_retains_vad_beam_and_language_options(self):
        calls = {}

        class FakeWhisperModel:
            def __init__(self, model_name, **options):
                calls["model_name"] = model_name
                calls["model_options"] = options

            def transcribe(self, samples, **options):
                calls["audio"] = samples
                calls["transcribe_options"] = options
                return iter([types.SimpleNamespace(text=" texto ")]), object()

        fake_faster_whisper = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_faster_whisper}):
            engine = engines.WhisperTranscriber(
                "medium",
                language="es",
                vad_filter=True,
                beam_size=5,
                windows=False,
            )
            text = engine.transcribe(np.ones(5, dtype=np.float32))

        self.assertEqual(text, "texto")
        self.assertEqual(calls["model_name"], "medium")
        self.assertEqual(calls["model_options"]["compute_type"], "int8")
        self.assertNotIn("cpu_threads", calls["model_options"])
        self.assertEqual(calls["transcribe_options"]["language"], "es")
        self.assertEqual(calls["transcribe_options"]["beam_size"], 5)
        self.assertEqual(calls["transcribe_options"]["vad_parameters"], {
            "min_silence_duration_ms": 300,
        })

    def test_macos_batches_audio_in_memory_and_flushes_partial_tail(self):
        source = Path("meeting_transcriber.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "transcriber_loop"
        )
        audio_queue = queue.Queue()
        audio_queue.put(np.ones((25, 1), dtype=np.float32))
        capture_done = threading.Event()
        capture_done.set()
        flushed = []
        namespace = {
            "queue": queue,
            "np": np,
            "os": types.SimpleNamespace(name="posix"),
            "audio_queue": audio_queue,
            "SAMPLE_RATE": 10,
            "BLOCK_SECONDS": 1,
            "transcriber": types.SimpleNamespace(name="Whisper medium"),
            "flush_buffer": lambda chunks, path: flushed.append(
                sum(len(chunk) for chunk in chunks)
            ),
            "windows_transcriber_loop": lambda *args: self.fail(
                "macOS no debe usar el loop de Windows"
            ),
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "macos-loop", "exec"),
            namespace,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["transcriber_loop"](
                "note.md",
                stop_event=threading.Event(),
                capture_done=capture_done,
            )

        self.assertEqual(flushed, [10, 10, 5])
        self.assertTrue(audio_queue.empty())


if __name__ == "__main__":
    unittest.main()
