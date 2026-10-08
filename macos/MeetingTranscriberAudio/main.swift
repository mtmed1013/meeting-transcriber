import AVFoundation
import CoreMedia
import Darwin
import Foundation
import ScreenCaptureKit

private let targetSampleRate = 16_000
private let outputChannelCount = 1
private let systemAudioSource: UInt8 = 83 // "S"
#if NATIVE_MICROPHONE
private let microphoneAudioSource: UInt8 = 77 // "M"
#endif

final class FramedAudioWriter {
    private let queue = DispatchQueue(label: "meeting-transcriber.audio-writer")
    private let output = FileHandle.standardOutput

    func write(samples: [Float32], source: UInt8) {
        guard !samples.isEmpty else { return }

        let sampleDuration = UInt64(
            (Double(samples.count) / Double(targetSampleRate)) * 1_000_000_000
        )
        var timestamp = DispatchTime.now().uptimeNanoseconds &- sampleDuration

        queue.sync {
            var count = UInt32(samples.count).littleEndian
            var packet = Data([source])
            timestamp = timestamp.littleEndian
            withUnsafeBytes(of: &timestamp) { packet.append(contentsOf: $0) }
            withUnsafeBytes(of: &count) { packet.append(contentsOf: $0) }
            samples.withUnsafeBytes { packet.append(contentsOf: $0) }
            output.write(packet)
        }
    }
}

final class CaptureOutput: NSObject, SCStreamOutput, SCStreamDelegate {
    private let writer: FramedAudioWriter
    private var reportedAudioSample = false
#if NATIVE_MICROPHONE
    private var reportedMicrophoneSample = false
#endif

    init(writer: FramedAudioWriter) {
        self.writer = writer
    }

    func stream(
        _ stream: SCStream,
        didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
        of outputType: SCStreamOutputType
    ) {
#if NATIVE_MICROPHONE
        let isSupportedAudioOutput = outputType == .audio || outputType == .microphone
#else
        let isSupportedAudioOutput = outputType == .audio
#endif
        guard isSupportedAudioOutput else {
            return
        }

        guard sampleBuffer.isValid else {
            if !reportedAudioSample {
                fputs("ERROR: ScreenCaptureKit entregó un sample de audio inválido\n", stderr)
                fflush(stderr)
                reportedAudioSample = true
            }
            return
        }

        guard let samples = monoFloatSamples(from: sampleBuffer) else {
            if !reportedAudioSample {
                fputs("ERROR: No se pudo convertir el primer sample de audio macOS\n", stderr)
                fflush(stderr)
                reportedAudioSample = true
            }
            return
        }

#if NATIVE_MICROPHONE
        if outputType == .microphone && !reportedMicrophoneSample {
            fputs("MEETING_AUDIO_MIC_READY\n", stderr)
            fflush(stderr)
            reportedMicrophoneSample = true
        }
#endif

        if outputType == .audio && !reportedAudioSample {
            fputs("MEETING_AUDIO_SAMPLE_READY\n", stderr)
            fflush(stderr)
            reportedAudioSample = true
        }

#if NATIVE_MICROPHONE
        let source = outputType == .microphone
            ? microphoneAudioSource
            : systemAudioSource
#else
        let source = systemAudioSource
#endif
        writer.write(samples: samples, source: source)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fputs("ERROR: ScreenCaptureKit se detuvo: \(error)\n", stderr)
        fflush(stderr)
        exit(2)
    }
}

private func monoFloatSamples(from sampleBuffer: CMSampleBuffer) -> [Float32]? {
    guard let formatDescription = sampleBuffer.formatDescription,
          let basicDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription)?.pointee
    else {
        return nil
    }

    let channels = max(Int(basicDescription.mChannelsPerFrame), 1)
    let bitsPerChannel = Int(basicDescription.mBitsPerChannel)
    let formatFlags = basicDescription.mFormatFlags
    let isFloat = (formatFlags & kAudioFormatFlagIsFloat) != 0
    let isSignedInteger = (formatFlags & kAudioFormatFlagIsSignedInteger) != 0
    let isLittleEndian = (formatFlags & kAudioFormatFlagIsBigEndian) == 0

    var requiredBufferListSize = 0
    var retainedBlockBuffer: CMBlockBuffer?

    let sizeStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: &requiredBufferListSize,
        bufferListOut: nil,
        bufferListSize: 0,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: &retainedBlockBuffer
    )

    guard requiredBufferListSize > 0,
          sizeStatus == noErr || sizeStatus == kCMSampleBufferError_ArrayTooSmall
    else {
        return nil
    }

    let rawBufferList = UnsafeMutableRawPointer.allocate(
        byteCount: requiredBufferListSize,
        alignment: MemoryLayout<AudioBufferList>.alignment
    )
    defer { rawBufferList.deallocate() }
    let bufferList = rawBufferList.assumingMemoryBound(to: AudioBufferList.self)

    let status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer,
        bufferListSizeNeededOut: nil,
        bufferListOut: bufferList,
        bufferListSize: requiredBufferListSize,
        blockBufferAllocator: nil,
        blockBufferMemoryAllocator: nil,
        flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
        blockBufferOut: &retainedBlockBuffer
    )

    guard status == noErr else { return nil }

    let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
    let bytesPerSample = max(bitsPerChannel / 8, 1)
    let bufferFrameCounts = buffers.map {
        Int($0.mDataByteSize) / bytesPerSample / max(Int($0.mNumberChannels), 1)
    }
    guard let firstFrameCount = bufferFrameCounts.min(), firstFrameCount > 0 else {
        return nil
    }

    func sampleValue(at pointer: UnsafePointer<UInt8>, index: Int) -> Float32? {
        let offset = index * bytesPerSample

        if isFloat && bitsPerChannel == 32 {
            var raw: UInt32 = 0
            memcpy(&raw, pointer.advanced(by: offset), 4)
            if !isLittleEndian {
                raw = UInt32(bigEndian: raw)
            }
            return Float32(bitPattern: raw)
        }

        if isSignedInteger && bitsPerChannel == 16 {
            var raw: UInt16 = 0
            memcpy(&raw, pointer.advanced(by: offset), 2)
            if !isLittleEndian {
                raw = UInt16(bigEndian: raw)
            }
            return Float32(Int16(bitPattern: raw)) / 32_768.0
        }

        if isSignedInteger && bitsPerChannel == 32 {
            var raw: UInt32 = 0
            memcpy(&raw, pointer.advanced(by: offset), 4)
            if !isLittleEndian {
                raw = UInt32(bigEndian: raw)
            }
            return Float32(Int32(bitPattern: raw)) / 2_147_483_648.0
        }

        return nil
    }

    var mono = [Float32](repeating: 0, count: firstFrameCount)
    for frame in 0..<firstFrameCount {
        var sum: Float32 = 0
        var values = 0

        for buffer in buffers {
            guard let data = buffer.mData else { continue }
            let bufferChannels = max(Int(buffer.mNumberChannels), 1)
            let pointer = data.assumingMemoryBound(to: UInt8.self)

            if buffers.count == 1 {
                for channel in 0..<channels {
                    guard let value = sampleValue(
                        at: pointer,
                        index: frame * channels + channel
                    ) else {
                        return nil
                    }
                    sum += value
                    values += 1
                }
            } else {
                for channel in 0..<bufferChannels {
                    guard let value = sampleValue(
                        at: pointer,
                        index: frame * bufferChannels + channel
                    ) else {
                        return nil
                    }
                    sum += value
                    values += 1
                }
            }
        }

        guard values > 0 else { return nil }
        mono[frame] = sum / Float32(values)
    }

    guard isFloat || isSignedInteger,
          (bitsPerChannel == 16 || bitsPerChannel == 32)
    else {
        fputs("ERROR: Formato de audio no compatible: \(bitsPerChannel) bits\n", stderr)
        fflush(stderr)
        return nil
    }

    let inputRate = basicDescription.mSampleRate
    guard inputRate > 0, abs(inputRate - Double(targetSampleRate)) > 1 else {
        return mono
    }

    let outputCount = max(Int(Double(mono.count) * Double(targetSampleRate) / inputRate), 1)
    var resampled = [Float32](repeating: 0, count: outputCount)
    for index in 0..<outputCount {
        let position = Double(index) * inputRate / Double(targetSampleRate)
        let lower = min(Int(position), mono.count - 1)
        let upper = min(lower + 1, mono.count - 1)
        let fraction = Float32(position - Double(lower))
        resampled[index] = mono[lower] + (mono[upper] - mono[lower]) * fraction
    }
    return resampled
}

final class CaptureController {
    private let writer = FramedAudioWriter()
    private let output: CaptureOutput
    private var stream: SCStream?

    init() {
        output = CaptureOutput(writer: writer)
    }

    func checkPermissions() {
        SCShareableContent.getExcludingDesktopWindows(
            false,
            onScreenWindowsOnly: false
        ) { [weak self] availableContent, error in
            guard let self else { return }

            if let error {
                self.fail(
                    "macOS no autorizó la captura del audio del sistema. " +
                    "En Ajustes del Sistema > Privacidad y seguridad > " +
                    "Grabación de pantalla y del audio del sistema, activa " +
                    "la aplicación que aparece en el aviso de macOS " +
                    "(normalmente Terminal). " +
                    "Detalle: \(error)"
                )
                return
            }

            guard let availableContent, !availableContent.displays.isEmpty else {
                self.fail(
                    "macOS no encontró una pantalla disponible para comprobar " +
                    "el permiso de captura."
                )
                return
            }

            fputs("MEETING_AUDIO_PERMISSION_READY\n", stderr)
            fflush(stderr)
            exit(0)
        }
    }

    func start() {
        SCShareableContent.getExcludingDesktopWindows(
            false,
            onScreenWindowsOnly: false
        ) { [weak self] availableContent, error in
            guard let self else { return }

            if let error {
                self.fail("No se pudo consultar el contenido compartible: \(error)")
                return
            }

            guard let display = availableContent?.displays.first else {
                self.fail("No se encontró una pantalla disponible")
                return
            }

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let configuration = SCStreamConfiguration()
            configuration.capturesAudio = true
            configuration.excludesCurrentProcessAudio = true
            configuration.sampleRate = targetSampleRate
            configuration.channelCount = outputChannelCount

#if NATIVE_MICROPHONE
                configuration.captureMicrophone = true
                guard let microphone = AVCaptureDevice.default(for: .audio) else {
                    self.fail("No se encontró un micrófono disponible")
                    return
                }
                configuration.microphoneCaptureDeviceID = microphone.uniqueID
#endif

            do {
                let stream = SCStream(
                    filter: filter,
                    configuration: configuration,
                    delegate: self.output
                )
                let audioQueue = DispatchQueue(label: "meeting-transcriber.system-audio")
                try stream.addStreamOutput(
                    self.output,
                    type: .audio,
                    sampleHandlerQueue: audioQueue
                )
#if NATIVE_MICROPHONE
                    let microphoneQueue = DispatchQueue(
                        label: "meeting-transcriber.microphone-audio"
                    )
                    try stream.addStreamOutput(
                        self.output,
                        type: .microphone,
                        sampleHandlerQueue: microphoneQueue
                    )
                    fputs("MEETING_AUDIO_MIC_ENABLED\n", stderr)
                    fflush(stderr)
#endif
                self.stream = stream
                stream.startCapture { [weak self] error in
                    guard let self else { return }
                    if let error {
                        self.fail("No se pudo iniciar la captura de audio del sistema: \(error)")
                        return
                    }
                    fputs("MEETING_AUDIO_READY\n", stderr)
                    fflush(stderr)
                }
            } catch {
                self.fail("No se pudo configurar ScreenCaptureKit: \(error)")
            }
        }
    }

    private func fail(_ message: String) {
        fputs("ERROR: \(message)\n", stderr)
        fflush(stderr)
        exit(1)
    }
}

@main
struct MeetingTranscriberAudio {
    static func main() {
        let controller = CaptureController()
        if CommandLine.arguments.dropFirst().contains("--check-permissions") {
            controller.checkPermissions()
        } else {
            controller.start()
        }
        dispatchMain()
    }
}
