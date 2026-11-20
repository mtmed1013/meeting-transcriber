import ast
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import types
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

import numpy as np

import windows_audio


class WindowsAudioTests(unittest.TestCase):
    def test_timeline_preserves_other_source_and_partial_tail(self):
        timeline = windows_audio.Timeline(100)
        timeline.append(1.02, np.ones(3, dtype=np.float32))
        data, covered = timeline.read(1.0, 4)
        np.testing.assert_array_equal(data, [0, 0, 1, 1])
        self.assertEqual(covered, 2)
        data, covered = timeline.read(1.04, 4)
        np.testing.assert_array_equal(data, [1, 0, 0, 0])
        self.assertEqual(covered, 1)

    def test_loopback_uses_id_not_duplicate_name(self):
        calls = []
        sc = types.SimpleNamespace(
            default_speaker=lambda: types.SimpleNamespace(id="render-id", name="Audio remoto"),
            get_microphone=lambda **kw: (calls.append(kw) or types.SimpleNamespace(isloopback=True)))
        windows_audio.select_device(sc, "system", "", "")
        self.assertEqual(calls, [{"id": "render-id", "include_loopback": True}])

    def test_missing_microphone_does_not_block_loopback(self):
        stop = threading.Event()
        output, errors = queue.Queue(), queue.Queue()

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def record(self, numframes):
                # Simulate a remote endpoint returning synthetic blocks instantly.
                return np.ones((numframes, 2), dtype=np.float32)

        device = types.SimpleNamespace(id="speaker", name="Audio remoto", isloopback=True,
                                       recorder=lambda **kw: Recorder())

        def no_microphone():
            raise RuntimeError("RDP microphone disconnected")

        sc = types.SimpleNamespace(default_microphone=no_microphone,
                                   default_speaker=lambda: device,
                                   get_microphone=lambda **kw: device)
        with patch.dict(sys.modules, soundcard=sc), \
                patch.object(windows_audio, "com_thread", contextlib.nullcontext), \
                contextlib.redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=windows_audio.capture,
                args=(output, stop, errors), kwargs={"rate": 1000, "buffer_seconds": 0.1})
            worker.start()
            time.sleep(0.7)
            stop.set()
            worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors.empty())
        self.assertGreater(output.qsize(), 0)
        self.assertTrue(any(np.max(chunk[1]) > 0 for chunk in list(output.queue)))
        self.assertLessEqual(sum(len(chunk[1]) for chunk in list(output.queue)), 1000)

    def test_fixed_batches_and_final_queue_are_consumed(self):
        # Extract only the loop: avoid loading/downloading Whisper during tests.
        tree = ast.parse(Path("meeting_transcriber.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "windows_transcriber_loop")
        chunks = queue.Queue()
        for count in (8, 17, 9):
            chunks.put(np.ones((count, 1)))
        done = threading.Event()
        done.set()
        saved = []
        namespace = dict(queue=queue, time=time, datetime=datetime, timedelta=timedelta,
                         audio_queue=chunks, SAMPLE_RATE=1, BLOCK_SECONDS=10,
                         transcriber=types.SimpleNamespace(name="Test engine"),
                         flush_buffer=lambda buffer, path, stamp=None: saved.append(sum(len(x) for x in buffer)))
        exec(compile(ast.Module(body=[function], type_ignores=[]), "loop", "exec"), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["windows_transcriber_loop"]("note", done)
        self.assertEqual(saved, [10, 10, 10, 4])
        self.assertTrue(chunks.empty())

    def test_source_recovers_after_transient_rdp_error(self):
        stop = threading.Event()
        output, errors = queue.Queue(), queue.Queue()
        attempts = []

        class Recorder:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def record(self, numframes):
                time.sleep(0.05)
                return np.ones((numframes, 2), dtype=np.float32)

        device = types.SimpleNamespace(id="render", name="Audio remoto", isloopback=True,
                                       recorder=lambda **kw: Recorder())

        def microphone():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("RDP reconnecting")
            return device

        sc = types.SimpleNamespace(default_microphone=microphone,
                                   default_speaker=lambda: device,
                                   get_microphone=lambda **kw: device)
        with patch.dict(sys.modules, soundcard=sc), \
                patch.object(windows_audio, "com_thread", contextlib.nullcontext), \
                contextlib.redirect_stdout(io.StringIO()):
            worker = threading.Thread(target=windows_audio.capture,
                args=(output, stop, errors), kwargs={"rate": 1000, "buffer_seconds": 0.1})
            worker.start()
            time.sleep(2.7)
            stop.set()
            worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors.empty())
        self.assertGreaterEqual(len(attempts), 2)
        # The two sources are averaged, so a unity signal from one source is
        # expected to peak at 0.5 when the other source is silent.
        self.assertTrue(any(np.max(chunk[1]) > 0.4 for chunk in list(output.queue)))


class ListDevicesTests(unittest.TestCase):
    @staticmethod
    def device(name, kind):
        def refuse_to_open(**kwargs):
            raise AssertionError(f"--list-devices must not open audio ({kind} {name!r})")
        return types.SimpleNamespace(id=f"{kind}:{name}", name=name, isloopback=False,
                                     recorder=refuse_to_open)

    def run_main(self, argv, sc):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, soundcard=sc), \
                patch.object(windows_audio, "com_thread", contextlib.nullcontext), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = windows_audio.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_list_devices_prints_one_json_line_deduplicated_and_stripped(self):
        calls = []

        def all_microphones(include_loopback=True):
            calls.append(include_loopback)
            print("ruido de SoundCard")  # stray stdout must be diverted to stderr
            return [self.device("  Micrófono A  ", "mic"), self.device("Micrófono A", "mic"),
                    self.device("Micrófono B", "mic")]

        sc = types.SimpleNamespace(
            all_microphones=all_microphones,
            all_speakers=lambda: [self.device("Altavoz 1", "spk"), self.device("Altavoz 1 ", "spk"),
                                  self.device("Altavoz 2", "spk")])
        code, stdout, _ = self.run_main(["--list-devices"], sc)
        self.assertEqual(code, 0)
        self.assertEqual(len(stdout.splitlines()), 1)
        self.assertEqual(json.loads(stdout), {"microphones": ["Micrófono A", "Micrófono B"],
                                             "speakers": ["Altavoz 1", "Altavoz 2"]})
        self.assertEqual(calls, [False])

    def test_list_devices_reports_soundcard_failure_as_empty_lists_and_exit_zero(self):
        def all_microphones(include_loopback=True):
            raise RuntimeError("RDP sin dispositivos de audio")

        sc = types.SimpleNamespace(all_microphones=all_microphones,
                                   all_speakers=lambda: [self.device("Altavoz 1", "spk")])
        code, stdout, stderr = self.run_main(["--list-devices"], sc)
        self.assertEqual(code, 0)
        self.assertEqual(len(stdout.splitlines()), 1)
        self.assertEqual(json.loads(stdout), {"microphones": [], "speakers": [],
                                             "error": "RDP sin dispositivos de audio"})
        self.assertIn("RDP sin dispositivos de audio", stderr)

    def test_list_devices_reports_missing_soundcard_as_error_json_and_exit_zero(self):
        code, stdout, stderr = self.run_main(["--list-devices"], None)
        self.assertEqual(code, 0)
        report = json.loads(stdout)
        self.assertEqual(report["microphones"], [])
        self.assertEqual(report["speakers"], [])
        self.assertTrue(report["error"])
        self.assertIn("soundcard", stderr.lower() + report["error"].lower())

    def test_check_still_captures_with_configured_device_names(self):
        sc = types.SimpleNamespace(all_microphones=lambda include_loopback=True: [],
                                   all_speakers=lambda: [])
        with patch.object(windows_audio, "capture") as capture, \
                patch.dict(os.environ, {"WINDOWS_MICROPHONE": "Mic X", "WINDOWS_SPEAKER": "Alt Y"}), \
                patch.dict(sys.modules, soundcard=sc), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = windows_audio.main(["--check"])
        self.assertEqual(code, 0)
        capture.assert_called_once()
        self.assertEqual(capture.call_args.kwargs, {"microphone_name": "Mic X", "speaker_name": "Alt Y"})

    def test_check_and_list_devices_require_exactly_one_mode(self):
        sc = types.SimpleNamespace(all_microphones=lambda include_loopback=True: [],
                                   all_speakers=lambda: [])
        for argv in ([], ["--check", "--list-devices"]):
            with self.subTest(argv=argv), patch.object(windows_audio, "capture") as capture, \
                    patch.dict(sys.modules, soundcard=sc), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    windows_audio.main(argv)
                self.assertEqual(raised.exception.code, 2)
                capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
