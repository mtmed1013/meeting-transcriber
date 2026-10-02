import ast
import contextlib
import io
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
        self.assertTrue(any(np.max(chunk[1]) > 0.9 for chunk in list(output.queue)))


if __name__ == "__main__":
    unittest.main()
