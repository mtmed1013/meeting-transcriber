import ast
import contextlib
import io
import os
import queue
import select
import struct
import subprocess
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeFramedHelper:
    """Expose one native-protocol packet without launching a recorder."""

    def __init__(self, packet):
        read_fd, write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.stderr = io.BytesIO()
        self.returncode = None
        os.write(write_fd, packet)
        os.close(write_fd)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-macos-audio-helper", timeout)
        return self.returncode


class MacOSShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path("meeting_transcriber.py").read_text(encoding="utf-8"))
        names = {
            "_empty_audio_level_stats",
            "_mono_audio",
            "_accumulate_audio_level",
            "_TimestampedAudioBuffer",
            "_read_exact",
            "_mac_audio_reader",
            "_mac_audio_stderr",
            "_should_stop_mac_audio_process",
            "_stop_mac_audio_process",
            "macos_audio_capture",
        }
        nodes = [
            node
            for node in tree.body
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "_NANOSECONDS_PER_SAMPLE"
                    for target in node.targets
                )
            )
        ]
        cls.buffers = []

        class TrackedTimestampedAudioBuffer:
            def __new__(buffer_cls, *args, **kwargs):
                instance = cls.namespace["timestamped_buffer_class"](*args, **kwargs)
                cls.buffers.append(instance)
                return instance

        namespace = {
            "np": np,
            "os": types.SimpleNamespace(
                path=types.SimpleNamespace(isfile=lambda _path: True)
            ),
            "queue": queue,
            "select": select,
            "struct": struct,
            "subprocess": subprocess,
            "threading": threading,
            "time": time,
            "deque": deque,
            "nullcontext": contextlib.nullcontext,
            "ExitStack": contextlib.ExitStack,
            "SAMPLE_RATE": 16_000,
            "MAC_AUDIO_HELPER": "fake-framed-helper",
            "MAC_AUDIO_MIX_FRAMES": 320,
            "MAC_AUDIO_JITTER_SECONDS": 0.35,
            "MAC_AUDIO_DIAGNOSTIC_SECONDS": 3600,
            "MAC_AUDIO_PROCESS_STOP_TIMEOUT_SECONDS": 0.1,
            "MAC_AUDIO_FORCE_DRAIN_SECONDS": 0.05,
            "MAC_NATIVE_MICROPHONE": True,
            "MAC_SYSTEM_GAIN": 1.0,
            "MAC_MICROPHONE_GAIN": 1.0,
            "_NANOSECONDS_PER_SAMPLE": 1_000_000_000 / 16_000,
            "audio_queue": queue.Queue(),
        }
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "macos-shutdown", "exec"),
            namespace,
        )
        namespace["timestamped_buffer_class"] = namespace["_TimestampedAudioBuffer"]
        namespace["_TimestampedAudioBuffer"] = TrackedTimestampedAudioBuffer
        cls.namespace = namespace

    def capture_samples(self, frame_count):
        samples = np.linspace(0.05, 0.45, frame_count, dtype=np.float32)
        timestamp_ns = time.monotonic_ns() - int(
            round(frame_count * 1_000_000_000 / 16_000)
        )
        packet = (
            struct.pack("<cQI", b"S", timestamp_ns, frame_count)
            + samples.astype("<f4", copy=False).tobytes()
        )
        self.capture_packet(packet, samples)

    def capture_packet(self, packet, expected, *, stop=None, stream=None, on_stop=None):
        helper = FakeFramedHelper(packet)
        if on_stop is not None:
            original_terminate = helper.terminate

            def terminate():
                on_stop()
                original_terminate()

            helper.terminate = terminate
        done = threading.Event()
        stop = stop if stop is not None else threading.Event()
        errors = queue.Queue()
        namespace = self.namespace
        namespace["audio_queue"] = queue.Queue()
        self.buffers.clear()

        with patch.object(subprocess, "Popen", return_value=helper), patch.dict(
            namespace,
            MAC_NATIVE_MICROPHONE=stream is None,
            sd=types.SimpleNamespace(InputStream=stream),
        ):
            worker = threading.Thread(
                target=namespace["macos_audio_capture"],
                args=(stop, errors, done),
                daemon=True,
            )
            # Start in shutdown mode: the framed reader must still deliver the
            # packet already written to the helper pipe before capture completes.
            if stream is None:
                stop.set()
            worker.start()
            worker.join(timeout=2)

        try:
            exited = not worker.is_alive()
            completed = done.is_set()
            self.assertTrue(
                exited and completed,
                "macOS shutdown did not finish cleanly: "
                f"capture_exited={exited}, capture_done={completed}",
            )
            self.assertTrue(errors.empty(), list(errors.queue))
            emitted = list(namespace["audio_queue"].queue)
            received = (
                np.concatenate([chunk.reshape(-1) for chunk in emitted])
                if emitted
                else np.array([], dtype=np.float32)
            )
            np.testing.assert_array_equal(received, expected)
        finally:
            if worker.is_alive():
                # Let the failed assertion unwind a production regression that
                # spins forever with an incomplete final mixing window.
                for buffer in self.buffers:
                    buffer._chunks.clear()
                worker.join(timeout=1)
            helper.stdout.close()

    def capture_paired_sources(self, frame_count, order):
        timestamp_ns = time.monotonic_ns()
        sources = {
            b"S": np.full(frame_count, 0.25, dtype="<f4"),
            b"M": np.full(frame_count, 0.5, dtype="<f4"),
        }
        packets = b"".join(
            struct.pack("<cQI", source, timestamp_ns, frame_count)
            + sources[source].tobytes()
            for source in order
        )
        self.capture_packet(packets, np.full(frame_count, 0.75, dtype=np.float32))

    def test_shutdown_sums_320_frames_system_then_microphone(self):
        self.capture_paired_sources(320, (b"S", b"M"))

    def test_shutdown_sums_320_frames_microphone_then_system(self):
        self.capture_paired_sources(320, (b"M", b"S"))

    def test_shutdown_sums_1000_frames_system_then_microphone(self):
        self.capture_paired_sources(1000, (b"S", b"M"))

    def test_shutdown_sums_1000_frames_microphone_then_system(self):
        self.capture_paired_sources(1000, (b"M", b"S"))

    def test_fallback_stream_exit_drains_inflight_callback_before_helper_stop(self):
        frame_count = 1000
        timestamp_ns = time.monotonic_ns()
        stop = threading.Event()
        callback_entered = threading.Event()
        release_callback = threading.Event()
        callback_finished = threading.Event()
        stream_exited = threading.Event()
        callback_errors = []
        mono_audio = self.namespace["_mono_audio"]
        microphone_samples = np.full((frame_count, 1), 0.5, dtype=np.float32)

        def blocked_mono_audio(data):
            callback_entered.set()
            if not release_callback.wait(1):
                raise RuntimeError("stream exit did not quiesce the in-flight callback")
            return mono_audio(data)

        class InputStream:
            def __init__(stream_self, **kwargs):
                stream_self.callback = kwargs["callback"]

            def __enter__(stream_self):
                def deliver():
                    try:
                        stream_self.callback(microphone_samples, frame_count, None, None)
                    except Exception as error:
                        callback_errors.append(error)
                    finally:
                        callback_finished.set()

                stream_self.worker = threading.Thread(target=deliver, daemon=True)
                stream_self.worker.start()
                if not callback_entered.wait(1):
                    raise RuntimeError("microphone callback did not start")
                # The callback has passed the stop guard, but has not queued its
                # samples. Context exit must wait for that publication.
                stop.set()
                return stream_self

            def __exit__(stream_self, *args):
                release_callback.set()
                stream_self.worker.join(1)
                # A callback beginning after stop must contribute no new audio.
                stream_self.callback(np.ones_like(microphone_samples), frame_count, None, None)
                stream_exited.set()

        def assert_stream_quiesced():
            self.assertTrue(stream_exited.is_set(), "helper stopped before microphone context exit")
            self.assertTrue(callback_finished.is_set(), "helper stopped before pending callback finished")
            self.assertFalse(callback_errors, callback_errors)

        packet = (
            struct.pack("<cQI", b"S", timestamp_ns, frame_count)
            + np.full(frame_count, 0.25, dtype="<f4").tobytes()
        )
        fixed_time = types.SimpleNamespace(
            monotonic_ns=lambda: timestamp_ns + frame_count * 62_500,
            monotonic=time.monotonic,
            sleep=time.sleep,
        )
        with patch.dict(self.namespace, _mono_audio=blocked_mono_audio, time=fixed_time):
            self.capture_packet(
                packet,
                np.full(frame_count, 0.75, dtype=np.float32),
                stop=stop,
                stream=InputStream,
                on_stop=assert_stream_quiesced,
            )
        self.assertTrue(stream_exited.is_set())
        self.assertTrue(callback_finished.is_set())
        self.assertFalse(callback_errors, callback_errors)

    def test_shutdown_preserves_exact_1000_frame_tail(self):
        self.capture_samples(1000)

    def test_shutdown_preserves_exact_321_frame_tail(self):
        self.capture_samples(321)

    def test_shutdown_preserves_complete_320_frame_window(self):
        self.capture_samples(320)

    def test_shutdown_preserves_audio_shorter_than_one_window(self):
        self.capture_samples(37)


if __name__ == "__main__":
    unittest.main()
