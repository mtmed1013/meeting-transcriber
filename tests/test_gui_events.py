import ast
import contextlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

import transcription_engines as engines


ROOT = Path(__file__).resolve().parents[1]
MEETING_SOURCE = (ROOT / "meeting_transcriber.py").read_text(encoding="utf-8")


def _load(names, **overrides):
    """Exec selected top-level functions of meeting_transcriber.py (repo convention)."""
    tree = ast.parse(MEETING_SOURCE)
    body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = set(names) - {node.name for node in body}
    if missing:
        raise AssertionError(f"meeting_transcriber.py no define: {sorted(missing)}")
    namespace = {
        "__name__": "meeting_transcriber_under_test",
        "json": json,
        "os": os,
        "sys": sys,
        "time": time,
        "threading": threading,
        "queue": queue,
        "np": np,
        "datetime": datetime,
        "GUI_EVENTS": False,
        "_GUI_EVENT_STREAM": None,
        "_GUI_EVENT_LOCK": threading.Lock(),
        "SAMPLE_RATE": 16000,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=body, type_ignores=[]), "meeting_transcriber.py", "exec"), namespace)
    return namespace


class _FlushRecorder(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1
        super().flush()


class _ScriptedStdin:
    def __init__(self, lines):
        self._lines = list(lines)
        self.reads = 0

    def readline(self):
        self.reads += 1
        return self._lines.pop(0) if self._lines else ""


class _FakeEngine:
    def __init__(self, name, text):
        self.name = name
        self._text = text

    def transcribe(self, audio):
        return self._text


def _probe_engine_paths(env_overrides):
    env = {key: value for key, value in os.environ.items() if key != "MT_MODELS_DIR"}
    env.update(env_overrides)
    code = (
        "import json, transcription_engines as e; "
        "print(json.dumps([str(e.MODEL_DIR), str(e.WHISTLE_MODEL_PATH), "
        "str(e.whistle_model_path())]))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(output)


class MtModelsDirTests(unittest.TestCase):
    def test_model_dir_falls_back_to_project_models_when_unset_or_blank(self):
        expected = [
            str(ROOT / "models"),
            str(ROOT / "models" / "whistle.cact"),
            str(ROOT / "models" / "whistle.cact"),
        ]
        self.assertEqual(_probe_engine_paths({}), expected)
        self.assertEqual(_probe_engine_paths({"MT_MODELS_DIR": ""}), expected)
        self.assertEqual(_probe_engine_paths({"MT_MODELS_DIR": "   "}), expected)

    def test_mt_models_dir_overrides_model_dir_and_whistle_paths_at_import(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = str(Path(directory) / "custom-models")
            self.assertEqual(
                _probe_engine_paths({"MT_MODELS_DIR": model_dir}),
                [
                    model_dir,
                    str(Path(model_dir) / "whistle.cact"),
                    str(Path(model_dir) / "whistle.cact"),
                ],
            )

    def test_whistle_path_is_resolved_dynamically_and_module_constant_is_kept(self):
        original_constant = engines.WHISTLE_MODEL_PATH
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MT_MODELS_DIR": directory}):
                self.assertEqual(engines.model_dir(), Path(directory))
                self.assertEqual(
                    engines.whistle_model_path(),
                    Path(directory) / "whistle.cact",
                )
                self.assertEqual(engines.WHISTLE_MODEL_PATH, original_constant)
        self.assertEqual(engines.whistle_model_path().parent, engines.model_dir())

    def test_whistle_transcriber_default_path_honours_mt_models_dir(self):
        calls = {}

        class FakeWhistle:
            def __init__(self, *, weights):
                calls["weights"] = weights

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "whistle.cact"
            model_path.write_bytes(b"local-model-fixture")
            fake_needle = types.SimpleNamespace(Whistle=FakeWhistle)
            with patch.dict(sys.modules, {"needle": fake_needle}), patch.dict(
                os.environ, {"MT_MODELS_DIR": directory}
            ):
                engines.WhistleTranscriber(language="es")

        self.assertEqual(calls["weights"], str(model_path))

    def test_create_transcriber_uses_mt_models_dir_for_whistle(self):
        calls = {}

        class FakeWhistle:
            def __init__(self, *, weights):
                calls["weights"] = weights

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "whistle.cact"
            model_path.write_bytes(b"local-model-fixture")
            fake_needle = types.SimpleNamespace(Whistle=FakeWhistle)
            with patch.dict(sys.modules, {"needle": fake_needle}), patch.dict(
                os.environ, {"MT_MODELS_DIR": directory}
            ):
                engines.create_transcriber("whistle", system="Darwin", machine="arm64")

        self.assertEqual(calls["weights"], str(model_path))

    def test_missing_model_error_names_the_overridden_path(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = re.escape(str(Path(directory) / "whistle.cact"))
            with patch.dict(os.environ, {"MT_MODELS_DIR": directory}):
                with self.assertRaisesRegex(FileNotFoundError, expected):
                    engines.WhistleTranscriber(language="es")


class GuiEventFormattingTests(unittest.TestCase):
    def test_format_gui_event_is_single_line_json_with_event_first(self):
        ns = _load(["format_gui_event"])
        line = ns["format_gui_event"]("ready", file_path="/vault/Acta.md", engine="Cactus Whistle")

        self.assertNotIn("\n", line)
        self.assertEqual(
            json.loads(line),
            {"event": "ready", "file_path": "/vault/Acta.md", "engine": "Cactus Whistle"},
        )
        self.assertTrue(line.startswith('{"event": "ready"'))

    def test_format_gui_event_stopping_is_exact_and_non_ascii_is_escaped(self):
        ns = _load(["format_gui_event"])
        self.assertEqual(ns["format_gui_event"]("stopping"), '{"event": "stopping"}')

        line = ns["format_gui_event"]("error", message="Reunión sin modelo")
        self.assertTrue(line.isascii())
        self.assertEqual(json.loads(line)["message"], "Reunión sin modelo")

    def test_emit_writes_and_flushes_each_event_when_gui_events_enabled(self):
        stream = _FlushRecorder()
        ns = _load(["format_gui_event", "emit_gui_event"], GUI_EVENTS=True, _GUI_EVENT_STREAM=stream)
        ns["emit_gui_event"]("loading_engine", engine="whistle")
        ns["emit_gui_event"]("finished", file_path="/vault/a.md")

        self.assertEqual(
            stream.getvalue(),
            '{"event": "loading_engine", "engine": "whistle"}\n'
            '{"event": "finished", "file_path": "/vault/a.md"}\n',
        )
        self.assertEqual(stream.flush_count, 2)

    def test_emit_is_silent_without_gui_events(self):
        stream = _FlushRecorder()
        ns = _load(["format_gui_event", "emit_gui_event"], GUI_EVENTS=False, _GUI_EVENT_STREAM=stream)
        ns["emit_gui_event"]("stopping")

        self.assertEqual(stream.getvalue(), "")
        self.assertEqual(stream.flush_count, 0)

    def test_stop_request_announces_stopping_only_on_first_request(self):
        stream = _FlushRecorder()
        ns = _load(
            ["format_gui_event", "emit_gui_event", "_announce_stop_request"],
            GUI_EVENTS=True,
            _GUI_EVENT_STREAM=stream,
        )
        stop_event = threading.Event()
        ns["_announce_stop_request"](stop_event)
        stop_event.set()
        ns["_announce_stop_request"](stop_event)

        self.assertEqual(stream.getvalue(), '{"event": "stopping"}\n')

    def test_configure_routes_human_logs_to_stderr_and_keeps_stdout_for_json(self):
        original_stdout = io.StringIO()
        stderr = io.StringIO()
        fake_sys = types.SimpleNamespace(stdout=original_stdout, stderr=stderr)
        ns = _load(["_configure_gui_events"], sys=fake_sys)

        ns["_configure_gui_events"](["--gui-events"])

        self.assertTrue(ns["GUI_EVENTS"])
        self.assertIs(ns["_GUI_EVENT_STREAM"], original_stdout)
        self.assertIs(fake_sys.stdout, stderr)

    def test_without_flag_cli_streams_are_untouched(self):
        original_stdout = io.StringIO()
        fake_sys = types.SimpleNamespace(stdout=original_stdout, stderr=io.StringIO())
        ns = _load(["_configure_gui_events"], sys=fake_sys)

        ns["_configure_gui_events"](["--other-flag", "value"])

        self.assertFalse(ns["GUI_EVENTS"])
        self.assertIsNone(ns["_GUI_EVENT_STREAM"])
        self.assertIs(fake_sys.stdout, original_stdout)


class GuiStdinStopListenerTests(unittest.TestCase):
    def test_stop_line_triggers_on_stop_once_and_ends_the_reader(self):
        ns = _load(["start_gui_stdin_listener"])
        stdin = _ScriptedStdin(["status\n", "  STOP  \n", "never read\n"])
        calls = []

        thread = ns["start_gui_stdin_listener"](
            threading.Event(), on_stop=lambda: calls.append("stop"), stream=stdin
        )
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(thread.daemon)
        self.assertEqual(calls, ["stop"])
        self.assertEqual(stdin.reads, 2)

    def test_eof_triggers_default_stop_event_set(self):
        ns = _load(["start_gui_stdin_listener"])
        stop_event = threading.Event()

        thread = ns["start_gui_stdin_listener"](stop_event, stream=_ScriptedStdin(["STOPPING\n"]))
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(stop_event.is_set())

    def test_eof_calls_on_stop_instead_of_bare_event_set(self):
        ns = _load(["start_gui_stdin_listener"])
        stop_event = threading.Event()
        calls = []

        thread = ns["start_gui_stdin_listener"](
            stop_event, on_stop=lambda: calls.append("stop"), stream=_ScriptedStdin([])
        )
        thread.join(timeout=2)

        self.assertEqual(calls, ["stop"])
        self.assertFalse(stop_event.is_set())

    def test_stdin_stop_runs_the_same_handler_as_ctrl_c(self):
        ns = _load(
            ["_handle_macos_interrupt", "start_gui_stdin_listener"],
            MAC_SHUTDOWN_PROGRESS_SECONDS=3,
            transcriber=_FakeEngine("Cactus Whistle", ""),
        )
        stop_event = threading.Event()
        force_stop_event = threading.Event()
        capture_done = threading.Event()
        shutdown_state = {"phase": "capturing"}

        with contextlib.redirect_stdout(io.StringIO()) as output:
            thread = ns["start_gui_stdin_listener"](
                stop_event,
                on_stop=lambda: ns["_handle_macos_interrupt"](
                    stop_event, force_stop_event, capture_done, shutdown_state
                ),
                stream=_ScriptedStdin(["STOP\n"]),
            )
            thread.join(timeout=2)

        self.assertTrue(stop_event.is_set())
        self.assertFalse(force_stop_event.is_set())
        self.assertEqual(shutdown_state["phase"], "stopping_capture")
        self.assertIn("Ctrl+C recibido", output.getvalue())


class GuiBlockEventTests(unittest.TestCase):
    def _flush(self, text, audio_time=None):
        stream = _FlushRecorder()
        ns = _load(
            ["append_text", "flush_buffer", "format_gui_event", "emit_gui_event"],
            GUI_EVENTS=True,
            _GUI_EVENT_STREAM=stream,
            SAMPLE_RATE=10,
            transcriber=_FakeEngine("Whisper small", text),
            audio_queue=queue.Queue(),
            SPEAKER_MARKERS=False,
            os=types.SimpleNamespace(name="posix"),
        )
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.md"
            with contextlib.redirect_stdout(io.StringIO()):
                ns["flush_buffer"]([np.ones(20, dtype=np.float32)], str(note), audio_time)
            saved = note.read_text(encoding="utf-8") if note.exists() else ""
        return stream, saved

    def test_saved_block_emits_block_saved_with_timestamp_elapsed_and_pending(self):
        stream, saved = self._flush("hola mundo", audio_time=datetime(2025, 1, 2, 3, 4, 5))

        self.assertIn("hola mundo", saved)
        event = json.loads(stream.getvalue().strip())
        self.assertEqual(set(event), {"event", "timestamp", "elapsed", "pending"})
        self.assertEqual(event["event"], "block_saved")
        self.assertEqual(event["timestamp"], "2025-01-02T03:04:05")
        self.assertGreaterEqual(event["elapsed"], 0.0)
        self.assertEqual(event["pending"], 0.0)
        self.assertEqual(stream.flush_count, 1)

    def test_silent_block_emits_no_block_saved_event(self):
        stream, saved = self._flush("")

        self.assertEqual(saved, "")
        self.assertEqual(stream.getvalue(), "")

    def test_flush_buffer_without_gui_events_keeps_note_and_console_output(self):
        ns = _load(
            ["append_text", "flush_buffer", "format_gui_event", "emit_gui_event"],
            GUI_EVENTS=False,
            SAMPLE_RATE=10,
            transcriber=_FakeEngine("Whisper small", " texto "),
            audio_queue=queue.Queue(),
            SPEAKER_MARKERS=False,
            os=types.SimpleNamespace(name="posix"),
        )
        with tempfile.TemporaryDirectory() as directory:
            note = Path(directory) / "note.md"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                ns["flush_buffer"]([np.ones(20, dtype=np.float32)], str(note), datetime(2025, 1, 2, 3, 4, 5))
            saved = note.read_text(encoding="utf-8")

        self.assertEqual(saved, "**[03:04:05]**\ntexto\n\n")
        self.assertIn("📝 Texto guardado", output.getvalue())
        self.assertNotIn("block_saved", output.getvalue())


class OptionalLibrosaTests(unittest.TestCase):
    def test_module_imports_without_librosa_and_speaker_detection_degrades_to_false(self):
        code = "\n".join([
            "import json, sys, types",
            "sys.modules['librosa'] = None",
            "engine = types.SimpleNamespace(name='Fake', close=lambda: None)",
            "sys.modules['transcription_engines'] = types.SimpleNamespace(",
            "    resolve_engine=lambda requested: 'whisper',",
            "    create_transcriber=lambda *args, **kwargs: engine,",
            ")",
            "import numpy as np",
            "import meeting_transcriber as mt",
            "audio = np.zeros(16000, dtype=np.float32)",
            "first = mt.voice_signature(audio, 16000)",
            "second = mt.voice_signature(audio, 16000)",
            "print(json.dumps({",
            "    'first_is_none': first is None,",
            "    'second_is_none': second is None,",
            "    'changed': mt.speaker_changed(first),",
            "    'changed_after_none': mt.speaker_changed(None),",
            "}))",
        ])
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])

        self.assertEqual(
            payload,
            {
                "first_is_none": True,
                "second_is_none": True,
                "changed": False,
                "changed_after_none": False,
            },
        )
        self.assertEqual(result.stdout.count("librosa"), 1)


class MtOutputFileTests(unittest.TestCase):
    def test_output_file_defaults_to_obsidian_note_when_env_unset_or_blank(self):
        ns = _load(["_output_file_path"], OBSIDIAN_DIR="/vault", NOTE_TITLE="Reunión.md")
        expected = os.path.join("/vault", "Reunión.md")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MT_OUTPUT_FILE", None)
            self.assertEqual(ns["_output_file_path"](), expected)
        for value in ("", "   "):
            with self.subTest(value=value), patch.dict(os.environ, {"MT_OUTPUT_FILE": value}):
                self.assertEqual(ns["_output_file_path"](), expected)

    def test_mt_output_file_is_used_as_the_exact_note_path(self):
        ns = _load(["_output_file_path"], OBSIDIAN_DIR="/vault", NOTE_TITLE="Reunión.md")
        with patch.dict(os.environ, {"MT_OUTPUT_FILE": "/tmp/exacto/Acta final.md"}):
            self.assertEqual(ns["_output_file_path"](), "/tmp/exacto/Acta final.md")


if __name__ == "__main__":
    unittest.main()
