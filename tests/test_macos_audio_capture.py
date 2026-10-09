import ast
import io
import queue
import struct
import threading
import unittest
from pathlib import Path
import types

import numpy as np


class MacOSAudioCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path("meeting_transcriber.py").read_text(encoding="utf-8"))
        names = {
            "_read_exact",
            "_mac_audio_reader",
            "_empty_audio_level_stats",
            "_accumulate_audio_level",
        }
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = {
            "np": np,
            "queue": queue,
            "struct": struct,
            "threading": threading,
            "SAMPLE_RATE": 16_000,
        }
        exec(
            compile(ast.Module(body=functions, type_ignores=[]), "mac-audio", "exec"),
            namespace,
        )
        cls.namespace = namespace

    def test_native_packet_timestamp_is_already_its_start_time(self):
        first_start_ns = 987_654_321_000
        samples = np.array([0.25, -0.5, 0.75], dtype="<f4")
        packet = (
            struct.pack("<cQI", b"S", first_start_ns, len(samples))
            + samples.tobytes()
        )
        process = types.SimpleNamespace(
            stdout=io.BytesIO(packet),
            poll=lambda: None,
        )
        source_queue = queue.Queue()
        errors = queue.Queue()
        stop_event = threading.Event()
        microphone_ready = threading.Event()
        reader_finished = threading.Event()

        self.namespace["_mac_audio_reader"](
            process,
            source_queue,
            stop_event,
            errors,
            microphone_ready,
            reader_finished,
        )

        source, timestamp_ns, received = source_queue.get_nowait()
        self.assertEqual(source, "system")
        self.assertEqual(timestamp_ns, first_start_ns)
        np.testing.assert_array_equal(received, samples)
        self.assertTrue(reader_finished.is_set())
        self.assertFalse(microphone_ready.is_set())
        self.assertTrue(errors.empty())

    def test_audio_level_stats_report_rms_peak_and_ignore_non_finite_values(self):
        stats = self.namespace["_empty_audio_level_stats"]()
        self.namespace["_accumulate_audio_level"](
            stats,
            np.array([0.25, -0.5, np.nan, np.inf], dtype=np.float32),
        )

        self.assertEqual(stats["sample_count"], 4)
        self.assertAlmostEqual(stats["sum_squares"], 0.3125)
        self.assertEqual(stats["peak"], 0.5)
        self.assertAlmostEqual(
            np.sqrt(stats["sum_squares"] / stats["sample_count"]),
            np.sqrt(0.3125 / 4),
        )


if __name__ == "__main__":
    unittest.main()
