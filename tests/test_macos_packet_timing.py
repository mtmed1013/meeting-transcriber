"""Exercise the actual Swift writer without recording private audio."""
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "macOS/Xcode required")
class NativePacketTimingTests(unittest.TestCase):
    def test_burst_delivery_preserves_sample_presentation_times(self):
        source = Path("macos/MeetingTranscriberAudio/main.swift").read_text()
        source = source.split("@main\nstruct MeetingTranscriberAudio", 1)[0]
        source += '''
@main struct TimingTest {
    static func main() {
        let writer = FramedAudioWriter()
        let base = CMTimeSubtract(CMClockGetTime(CMClockGetHostTimeClock()),
                                  CMTime(value: 200, timescale: 1000))
        for i in 0..<3 {
            writer.write(samples: [Float32](repeating: 0.25, count: 320), source: 83,
                         presentationTime: CMTimeAdd(base, CMTime(value: Int64(i * 20), timescale: 1000)))
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            swift = Path(directory) / "timing.swift"
            binary = Path(directory) / "timing"
            swift.write_text(source)
            subprocess.run([
                "xcrun", "swiftc", "-parse-as-library", "-swift-version", "5",
                "-framework", "AVFoundation", "-framework", "CoreMedia",
                "-framework", "ScreenCaptureKit", str(swift), "-o", str(binary),
            ], check=True, capture_output=True, timeout=60)
            output = subprocess.check_output([str(binary)], timeout=5)
        timestamps = []
        offset = 0
        while offset < len(output):
            source_id, timestamp, count = struct.unpack_from("<cQI", output, offset)
            self.assertEqual(source_id, b"S")
            self.assertEqual(count, 320)
            timestamps.append(timestamp)
            offset += 13 + count * 4
        self.assertEqual(len(timestamps), 3)
        for first, second in zip(timestamps, timestamps[1:]):
            self.assertAlmostEqual((second - first) / 1e6, 20, delta=0.1)
