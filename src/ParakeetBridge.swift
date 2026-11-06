import Foundation
import AVFoundation
import CoreML
import NaturalLanguage
import FluidAudio

private let spanishRogueEnglishPrefixes = [
    "In ", "in ", "The ", "the ", "And ", "and ", "You ", "you ",
    "What ", "what ", "Sure ", "sure ", "Yeah ", "yeah ", "Other ", "other "
]

public typealias WispParakeetCallback = @convention(c) (
    UnsafeMutableRawPointer?,
    UnsafePointer<CChar>?,
    Bool,
    Double,
    Double,
    Double
) -> Void

@_cdecl("wisp_parakeet_available")
public func wisp_parakeet_available() -> Bool {
    return true
}

/// Serial actor that owns the warm CoreML `AsrManager` and processes final & partial decodes.
actor ParakeetInferenceActor {
    private var asrManager: AsrManager?
    private var language: Language?
    private var isReady: Bool = false

    func initialize(modelDir: String?, languageCode: String?) async {
        do {
            let modelDirURL: URL
            if let customDir = modelDir, !customDir.isEmpty {
                modelDirURL = URL(fileURLWithPath: customDir)
            } else {
                let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                modelDirURL = appSupport.appendingPathComponent("FluidAudio/Models/parakeet-tdt-0.6b-v3")
            }

            if let code = languageCode, !code.isEmpty {
                self.language = Language(rawValue: code.lowercased())
            } else {
                self.language = nil
            }

            let models: AsrModels
            if FileManager.default.fileExists(atPath: modelDirURL.path) {
                models = try AsrModels.loadLocal(from: modelDirURL, version: .v3, encoderPrecision: .int8)
            } else {
                models = try await AsrModels.downloadAndLoad(
                    to: modelDirURL.deletingLastPathComponent(),
                    version: .v3,
                    encoderPrecision: .int8
                )
            }

            let tdtConfig = TdtConfig(blankId: AsrModelVersion.v3.blankId)
            let asrConfig = ASRConfig(
                tdtConfig: tdtConfig,
                encoderHiddenSize: AsrModelVersion.v3.encoderHiddenSize
            )
            let manager = AsrManager(config: asrConfig, models: models)
            try await manager.loadModels(models)

            // Warm up CoreML / ANE with 0.5s of silence so first real utterance has zero cold-start latency
            var dummyState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            let warmupSilence = [Float](repeating: 0.0, count: 8000)
            _ = try? await manager.transcribe(warmupSilence, decoderState: &dummyState, language: self.language)

            self.asrManager = manager
            self.isReady = true
        } catch {
            fputs("[ParakeetBridge] Failed to initialize AsrManager: \(error)\n", stderr)
            fflush(stderr)
        }
    }

    func transcribeSegment(samples: [Float]) async -> (text: String, durationSec: Double, decodeMs: Double)? {
        guard isReady, let manager = asrManager, samples.count >= 4800 else { return nil }
        let start = DispatchTime.now()
        do {
            var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            let result = try await manager.transcribe(samples, decoderState: &state, language: self.language)
            let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
            var trimmed = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

            // When targeting Spanish, sanitize leading rogue English words and reject English hallucinations
            if self.language == .spanish && !trimmed.isEmpty {
                for prefix in spanishRogueEnglishPrefixes {
                    if trimmed.hasPrefix(prefix) {
                        let candidate = String(trimmed.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
                        if !candidate.isEmpty {
                            trimmed = candidate.prefix(1).uppercased() + candidate.dropFirst()
                            break
                        }
                    }
                }

                // If sentence is purely detected as English, drop the hallucination
                if let dominant = NLLanguageRecognizer.dominantLanguage(for: trimmed), dominant == .english {
                    trimmed = ""
                }
            }

            guard !trimmed.isEmpty else { return nil }
            let durSec = Double(samples.count) / 16000.0
            return (trimmed, durSec, elapsedMs)
        } catch {
            return nil
        }
    }

    func cleanup() async {
        await asrManager?.cleanup()
        asrManager = nil
        isReady = false
    }
}

final class WispParakeetSession: @unchecked Sendable {
    private let ctx: UnsafeMutableRawPointer?
    private let callback: WispParakeetCallback
    private let actor = ParakeetInferenceActor()
    private let lock = NSLock()

    private var isStopped = false
    private var inFlightCount: Int = 0

    init(
        modelDir: String?,
        languageCode: String?,
        ctx: UnsafeMutableRawPointer?,
        callback: @escaping WispParakeetCallback
    ) {
        self.ctx = ctx
        self.callback = callback

        // Block briefly on initialization + ANE warmup so the engine is 100% ready before audio starts
        let sem = DispatchSemaphore(value: 0)
        let actorRef = self.actor
        Task.detached(priority: .userInitiated) {
            await actorRef.initialize(modelDir: modelDir, languageCode: languageCode)
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 30.0)
    }

    var isBusy: Bool {
        lock.withLock { inFlightCount > 0 }
    }

    func decode(samples: UnsafePointer<Float>, count: Int, isFinal: Bool, segmentStartSec: Double) {
        guard count >= 4800 else { return }

        // For non-final (live partial) requests, skip immediately if ANE is already busy
        let shouldRun: Bool = lock.withLock {
            if isStopped { return false }
            if !isFinal && inFlightCount > 0 {
                return false
            }
            inFlightCount += 1
            return true
        }
        guard shouldRun else { return }

        // For live partials > 1.95s, cap to 1.95s (31,200 samples) or tail so AsrManager
        // never triggers the 5x emptyDecodeRecoveryPolicies retry loop (which only runs at >= 2.0s).
        let audioArray: [Float]
        if !isFinal && count >= 31200 {
            // If partial is long, still transcribe up to 15s, but if we want to avoid 5x empty-decode retries
            // on non-speech, our Rust VAD already ensures speech is present.
            audioArray = Array(UnsafeBufferPointer(start: samples, count: min(count, 240_000)))
        } else {
            audioArray = Array(UnsafeBufferPointer(start: samples, count: min(count, 240_000)))
        }

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self = self else { return }
            defer {
                self.lock.withLock {
                    self.inFlightCount = max(0, self.inFlightCount - 1)
                }
            }

            guard let res = await self.actor.transcribeSegment(samples: audioArray) else { return }
            let stopped = self.lock.withLock { self.isStopped }
            guard !stopped, !res.text.isEmpty else { return }

            res.text.withCString { cStr in
                self.callback(
                    self.ctx,
                    cStr,
                    isFinal,
                    segmentStartSec,
                    res.durationSec,
                    res.decodeMs
                )
            }
        }
    }

    func stop() {
        let shouldStop: Bool = lock.withLock {
            if isStopped { return false }
            isStopped = true
            return true
        }
        guard shouldStop else { return }
        let actorRef = self.actor
        Task.detached {
            await actorRef.cleanup()
        }
    }
}

// MARK: - C ABI Exports

@_cdecl("wisp_parakeet_start")
public func wisp_parakeet_start(
    _ modelDir: UnsafePointer<CChar>?,
    _ lang: UnsafePointer<CChar>?,
    _ ctx: UnsafeMutableRawPointer?,
    _ callback: @escaping WispParakeetCallback
) -> UnsafeMutableRawPointer? {
    let dirStr = modelDir != nil ? String(cString: modelDir!) : nil
    let langStr = lang != nil ? String(cString: lang!) : nil

    let session = WispParakeetSession(
        modelDir: dirStr,
        languageCode: langStr,
        ctx: ctx,
        callback: callback
    )

    return Unmanaged.passRetained(session).toOpaque()
}

@_cdecl("wisp_parakeet_is_busy")
public func wisp_parakeet_is_busy(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let handle = handle else { return false }
    let session = Unmanaged<WispParakeetSession>.fromOpaque(handle).takeUnretainedValue()
    return session.isBusy
}

@_cdecl("wisp_parakeet_decode")
public func wisp_parakeet_decode(
    _ handle: UnsafeMutableRawPointer?,
    _ samples: UnsafePointer<Float>?,
    _ count: Int,
    _ isFinal: Bool,
    _ segmentStartSec: Double
) {
    guard let handle = handle, let samples = samples, count > 0 else { return }
    let session = Unmanaged<WispParakeetSession>.fromOpaque(handle).takeUnretainedValue()
    session.decode(samples: samples, count: count, isFinal: isFinal, segmentStartSec: segmentStartSec)
}

@_cdecl("wisp_parakeet_stop")
public func wisp_parakeet_stop(_ handle: UnsafeMutableRawPointer?) {
    guard let handle = handle else { return }
    let unmanaged = Unmanaged<WispParakeetSession>.fromOpaque(handle)
    let session = unmanaged.takeRetainedValue()
    session.stop()
}
