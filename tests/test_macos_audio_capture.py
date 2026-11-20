import ast
import contextlib
import io
import os
import queue
import select
import subprocess
import struct
import threading
import time
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
            "_queued_audio_seconds",
            "_macos_shutdown_progress_loop",
            "_handle_macos_interrupt",
            "_should_stop_mac_audio_process",
            "_stop_mac_audio_process",
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
            "subprocess": subprocess,
            "select": select,
            "time": time,
            "SAMPLE_RATE": 16_000,
            "MAC_SHUTDOWN_PROGRESS_SECONDS": 3,
            "MAC_AUDIO_GRACEFUL_STOP_SECONDS": 10,
            "MAC_AUDIO_PROCESS_STOP_TIMEOUT_SECONDS": 5,
            "MAC_AUDIO_FORCE_DRAIN_SECONDS": 0.05,
            "audio_queue": queue.Queue(),
            "transcriber": types.SimpleNamespace(name="Cactus Whistle"),
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

    def test_forced_shutdown_unblocks_reader_when_helper_pipe_stays_open(self):
        read_fd, write_fd = os.pipe()
        stdout = os.fdopen(read_fd, "rb", buffering=0)
        samples = np.array([0.25, -0.5, 0.75], dtype="<f4")
        packet = (
            struct.pack("<cQI", b"S", 987_654_321_000, len(samples))
            + samples.tobytes()
        )
        os.write(write_fd, packet)
        process = types.SimpleNamespace(stdout=stdout, poll=lambda: None)
        source_queue = queue.Queue()
        errors = queue.Queue()
        stop_event = threading.Event()
        stop_event.set()
        force_stop_event = threading.Event()
        microphone_ready = threading.Event()
        reader_finished = threading.Event()
        reader = threading.Thread(
            target=self.namespace["_mac_audio_reader"],
            args=(
                process,
                source_queue,
                stop_event,
                errors,
                microphone_ready,
                reader_finished,
                force_stop_event,
            ),
        )

        try:
            reader.start()
            time.sleep(0.02)
            force_stop_event.set()
            reader.join(timeout=1)

            self.assertFalse(reader.is_alive())
            self.assertTrue(reader_finished.is_set())
            self.assertTrue(errors.empty())
            source, timestamp_ns, received = source_queue.get_nowait()
            self.assertEqual(source, "system")
            self.assertEqual(timestamp_ns, 987_654_321_000)
            np.testing.assert_array_equal(received, samples)
        finally:
            os.close(write_fd)
            stdout.close()
            if reader.is_alive():
                reader.join(timeout=1)

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

    def test_repeated_ctrl_c_escalates_capture_shutdown_but_preserves_transcription(self):
        stop_event = threading.Event()
        force_stop_event = threading.Event()
        capture_done = threading.Event()
        state = {"phase": "capturing"}
        request_stop = self.namespace["_handle_macos_interrupt"]

        with io.StringIO() as output, contextlib.redirect_stdout(output):
            request_stop(stop_event, force_stop_event, capture_done, state)
            self.assertTrue(stop_event.is_set())
            self.assertFalse(force_stop_event.is_set())
            self.assertEqual(state["phase"], "stopping_capture")

            request_stop(stop_event, force_stop_event, capture_done, state)
            self.assertTrue(force_stop_event.is_set())
            self.assertIn("Segunda interrupción", output.getvalue())

        state.update(
            phase="transcribing",
            inference_started_at=time.monotonic() - 2,
        )
        capture_done.set()
        force_stop_event.clear()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            request_stop(stop_event, force_stop_event, capture_done, state)

        self.assertFalse(force_stop_event.is_set())
        self.assertIn("no se interrumpe", output.getvalue())
        self.assertIn("2s", output.getvalue())

    def test_shutdown_progress_reports_capture_wait_and_final_inference(self):
        class FinishAfterOneReport:
            def __init__(self):
                self.waits = 0

            def wait(self, timeout):
                self.waits += 1
                return self.waits > 1

        audio_queue = self.namespace["audio_queue"]
        audio_queue.put(np.ones(32_000, dtype=np.float32))
        stop_event = threading.Event()
        stop_event.set()
        force_stop_event = threading.Event()
        capture_done = threading.Event()
        state = {
            "phase": "stopping_capture",
            "requested_at": time.monotonic() - 4,
            "next_progress_at": 0,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.namespace["_macos_shutdown_progress_loop"](
                stop_event,
                force_stop_event,
                capture_done,
                state,
                FinishAfterOneReport(),
            )
        self.assertIn("deteniendo el capturador", output.getvalue())
        self.assertIn("audio pendiente≈2.0s", output.getvalue())

        state.update(
            phase="transcribing",
            inference_started_at=time.monotonic() - 3,
            current_block_seconds=5.6,
            next_progress_at=0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.namespace["_macos_shutdown_progress_loop"](
                stop_event,
                force_stop_event,
                capture_done,
                state,
                FinishAfterOneReport(),
            )
        self.assertIn("Cactus Whistle sigue transcribiendo un bloque de 5.6s", output.getvalue())
        self.assertIn("lleva 3s", output.getvalue())

        state.update(
            phase="stopping_capture",
            requested_at=time.monotonic() - 11,
            inference_started_at=None,
            next_progress_at=0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.namespace["_macos_shutdown_progress_loop"](
                stop_event,
                force_stop_event,
                capture_done,
                state,
                FinishAfterOneReport(),
            )
        self.assertTrue(force_stop_event.is_set())
        self.assertIn("se fuerza el cierre del capturador", output.getvalue())

    def test_stop_request_stops_helper_without_discarding_queued_packets(self):
        stop_event = threading.Event()
        force_stop_event = threading.Event()
        source_queue = queue.Queue()
        source_queue.put("captured-packet")
        should_stop = self.namespace["_should_stop_mac_audio_process"]

        self.assertFalse(should_stop(stop_event, force_stop_event, source_queue))

        stop_event.set()
        self.assertTrue(should_stop(stop_event, force_stop_event, source_queue))
        self.assertEqual(source_queue.get_nowait(), "captured-packet")

    def test_mac_audio_process_force_stop_does_not_wait_for_graceful_termination(self):
        class FakeProcess:
            def __init__(self):
                self.running = True
                self.terminated = False
                self.killed = False

            def poll(self):
                return None if self.running else 0

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True
                self.running = False

            def wait(self, timeout):
                self.asserted_timeout = timeout

        process = FakeProcess()
        self.namespace["_stop_mac_audio_process"](process, force=True)

        self.assertTrue(process.killed)
        self.assertFalse(process.terminated)
        self.assertEqual(process.asserted_timeout, 5)


if __name__ == "__main__":
    unittest.main()
