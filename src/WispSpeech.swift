// On-device streaming speech recognition via macOS 26's SpeechAnalyzer / SpeechTranscriber, exposed to
// Rust over a small C ABI. The transcriber runs Apple's built-in models (no download of *our* models;
// the OS fetches the per-language asset on first use), streams volatile (partial) + final results, and
// we forward each as (text, isFinal) to a Rust callback.
//
// Bridging notes: @_cdecl entry points are synchronous, but the Speech API is async. `start` returns a
// handle immediately and kicks the (async) setup — model install, analyzer start, result consumption —
// onto a Task. `feed` yields audio into an AsyncStream the analyzer consumes; audio that arrives before
// setup finishes is buffered briefly and flushed once the analyzer's audio format is known.

import AVFoundation
import CoreMedia
import Foundation
import Speech

/// Whether on-device transcription is usable on this OS (macOS 26+ with the SpeechTranscriber API).
@_cdecl("wisp_applespeech_available")
public func wisp_applespeech_available() -> Bool {
    if #available(macOS 26.0, *) { return true }
    return false
}

@available(macOS 26.0, *)
final class WispSpeechSession {
    private let callback: @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool, Double, Double) -> Void
    private let ctx: UnsafeMutableRawPointer?

    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let continuation: AsyncStream<AnalyzerInput>.Continuation
    private let inputStream: AsyncStream<AnalyzerInput>

    /// The audio format the analyzer wants; nil until setup resolves it. Audio fed before then is held.
    private var analyzerFormat: AVAudioFormat?
    private var converter: AVAudioConverter?
    private var pending: [(samples: [Float], rate: Double)] = []
    private let lock = NSLock()

    /// Tracks how many completed sentences within the current continuous SpeechTranscriber utterance
    /// have already been finalized and emitted to Rust, so back-to-back speakers / sentences are
    /// split into individual timed subtitle cues in real time.
    private var committedSentenceCount: Int = 0
    private var committedWordCount: Int = 0
    private var committedAnchorWord: String = ""
    private var segmentTimings: [(startSec: Double, endSec: Double)] = []
    private var wordTimeline: [Double] = []
    private var totalFedAudioSec: Double = 0.0

    struct SentenceSegment {
        var text: String
        var startSec: Double
        var endSec: Double
        var wordCount: Int
        var nonSpaceChars: Int
        var startWordIdx: Int
        var endWordIdx: Int

        var durationSec: Double {
            (startSec >= 0 && endSec >= startSec) ? (endSec - startSec) : -1.0
        }
    }

    init(locale: String,
         ctx: UnsafeMutableRawPointer?,
         callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool, Double, Double) -> Void) {
        self.ctx = ctx
        self.callback = callback
        // `.volatileResults` streams partial hypotheses as you speak; `.fastResults` biases the
        // transcriber toward low-latency emission over maximal accuracy — the right trade for a live
        // caption. `.audioTimeRange` attaches exact per-word CMTimeRanges in the fed audio stream.
        self.transcriber = SpeechTranscriber(
            locale: Locale(identifier: locale),
            transcriptionOptions: [],
            reportingOptions: [.volatileResults, .fastResults],
            attributeOptions: [.audioTimeRange])
        // User-initiated priority keeps the analyzer responsive under load; `.lingering` keeps the model
        // warm between sessions so a restart doesn't pay the cold-load latency again.
        self.analyzer = SpeechAnalyzer(
            modules: [transcriber],
            options: SpeechAnalyzer.Options(priority: .userInitiated, modelRetention: .lingering))
        (self.inputStream, self.continuation) = AsyncStream<AnalyzerInput>.makeStream()
    }

    private static let commonAbbreviations: Set<String> = [
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "sra.", "srta.", "st.", "vs.", "etc."
    ]

    private static func endsWithSentencePunctuation(_ str: String) -> Bool {
        let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }

        // Strip trailing closing quotes/brackets if present
        let quoteSet = CharacterSet(charactersIn: "\"”'’»)」』】")
        let core = trimmed.trimmingCharacters(in: quoteSet)
        guard let last = core.last else { return false }

        if last == "!" || last == "?" || last == "。" || last == "！" || last == "？" || last == "…" {
            return true
        }
        if last == "." {
            let lower = core.lowercased()
            let lastWord = lower.split(whereSeparator: { $0.isWhitespace }).last.map(String.init) ?? lower
            if commonAbbreviations.contains(lastWord) {
                return false
            }
            // Avoid splitting on single-letter initials like "J."
            if lastWord.count <= 2 && lastWord.first?.isLetter == true {
                return false
            }
            return true
        }
        return false
    }

    private static func endsWithClauseSeparator(_ str: String) -> Bool {
        let trimmed = str.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.last else { return false }
        return last == "," || last == ";" || last == ":" || last == "、" || last == "，" || last == "—"
    }

    private static func isAttachedPunctuationOnly(_ str: String) -> Bool {
        guard !str.isEmpty else { return false }
        if str.first?.isWhitespace == true { return false }
        let punct = CharacterSet(charactersIn: ".!?,;:。！？，、…\"”'’»)」』】—")
        return str.unicodeScalars.allSatisfy { punct.contains($0) }
    }

    private static func stripLeadingClosingPunctuation(_ str: String) -> String {
        let strayLeading = CharacterSet(charactersIn: "?!.,;:。！？，、…")
        var scalars = str.unicodeScalars
        while let first = scalars.first, strayLeading.contains(first) || first.properties.isWhitespace {
            scalars.removeFirst()
        }
        var out = String(String.UnicodeScalarView(scalars))
        // Clean stray comma/period right after Spanish opening inverted punctuation (`¡, no` -> `¡no`)
        if out.hasPrefix("¡") || out.hasPrefix("¿") {
            let prefixChar = String(out.prefix(1))
            var rest = out.dropFirst().unicodeScalars
            while let first = rest.first, strayLeading.contains(first) || first.properties.isWhitespace {
                rest.removeFirst()
            }
            out = prefixChar + String(String.UnicodeScalarView(rest))
        }
        return out
    }

    private static func bareWord(_ str: String) -> String {
        let punct = CharacterSet.punctuationCharacters.union(.whitespacesAndNewlines)
        return str.trimmingCharacters(in: punct).lowercased()
    }

    private static func extractLastBareWord(_ str: String) -> String {
        let parts = str.split(whereSeparator: { $0.isWhitespace })
        for p in parts.reversed() {
            let b = bareWord(String(p))
            if !b.isEmpty { return b }
        }
        return ""
    }

    /// Segments the *uncommitted* portion of a `SpeechTranscriber.Result` into individual sentences.
    /// Any words before `committedWordCount` (already finalized and emitted in an earlier sentence
    /// of this recognition turn) are skipped so Apple's retroactive punctuation changes (e.g. changing
    /// `.` to `,` on an earlier word) can never re-emit already finalized words.
    private static func extractSentenceSegments(
        _ result: SpeechTranscriber.Result,
        wordTimeline: inout [Double],
        committedWordCount: inout Int,
        committedAnchorWord: String,
        nowFedSec: Double
    ) -> [SentenceSegment] {
        struct WordToken {
            let text: String
            let leadingSpace: Bool
            let startSec: Double?
            let endSec: Double?
        }

        let fallbackStart = result.range.start.isNumeric ? result.range.start.seconds : -1.0
        let fallbackEnd = (result.range.start.isNumeric && result.range.duration.isNumeric)
            ? (result.range.start.seconds + result.range.duration.seconds)
            : -1.0

        var tokens: [WordToken] = []
        for run in result.text.runs {
            let runStr = String(result.text[run.range].characters)
            if runStr.isEmpty { continue }

            var runStart: Double? = nil
            var runEnd: Double? = nil
            if let tr = run.audioTimeRange, tr.start.isNumeric && tr.duration.isNumeric && tr.duration.seconds > 0 {
                let trimmed = runStr.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    runStart = tr.start.seconds
                    runEnd = tr.start.seconds + tr.duration.seconds
                }
            }

            // Tokenize runStr by whitespace and CJK sentence-ending punctuation (`。`, `！`, `？`, `…`)
            var subTokens: [(text: String, leadingSpace: Bool)] = []
            var currentTok = ""
            var currentLeadingSpace = false
            var sawWhitespace = false

            for ch in runStr {
                if ch.isWhitespace {
                    if !currentTok.isEmpty {
                        subTokens.append((currentTok, currentLeadingSpace))
                        currentTok = ""
                    }
                    sawWhitespace = true
                } else {
                    if currentTok.isEmpty {
                        currentLeadingSpace = sawWhitespace || !tokens.isEmpty || !subTokens.isEmpty
                    }
                    sawWhitespace = false
                    currentTok.append(ch)
                    if ch == "。" || ch == "！" || ch == "？" || ch == "…" {
                        subTokens.append((currentTok, currentLeadingSpace))
                        currentTok = ""
                        currentLeadingSpace = false
                    }
                }
            }
            if !currentTok.isEmpty {
                subTokens.append((currentTok, currentLeadingSpace))
            }
            if subTokens.isEmpty { continue }

            let totalChars = max(1, subTokens.reduce(0) { $0 + $1.text.count })
            var charOffset = 0

            for sub in subTokens {
                var wStart: Double? = nil
                var wEnd: Double? = nil
                if let rs = runStart, let re = runEnd, re >= rs {
                    let span = re - rs
                    let f0 = Double(charOffset) / Double(totalChars)
                    let f1 = Double(charOffset + sub.text.count) / Double(totalChars)
                    wStart = rs + span * f0
                    wEnd = rs + span * f1
                }
                charOffset += sub.text.count
                tokens.append(WordToken(
                    text: sub.text,
                    leadingSpace: sub.leadingSpace,
                    startSec: wStart,
                    endSec: wEnd
                ))
            }
        }

        if tokens.isEmpty {
            return []
        }

        // Count real words and record arrival time in `wordTimeline` for any newly recognized words
        var realWordTexts: [String] = []
        for tok in tokens {
            if !isAttachedPunctuationOnly(tok.text) {
                realWordTexts.append(tok.text)
            }
        }
        let totalWords = realWordTexts.count
        if totalWords > wordTimeline.count {
            let added = totalWords - wordTimeline.count
            let prevSec = wordTimeline.last ?? (fallbackStart >= 0 ? fallbackStart : max(0.0, nowFedSec - Double(added) * 0.18))
            let span = max(0.0, nowFedSec - prevSec)
            for k in 1...added {
                let interp = added == 1 ? nowFedSec : (prevSec + span * (Double(k) / Double(added)))
                wordTimeline.append(interp)
            }
        }

        // If we have already committed words in this turn, align `committedWordCount` with `committedAnchorWord`
        // in case Apple's volatile recognizer inserted or merged 1 word earlier in the turn.
        if committedWordCount > 0 && !committedAnchorWord.isEmpty && totalWords > 0 {
            let expectedIdx = min(committedWordCount - 1, totalWords - 1)
            if bareWord(realWordTexts[expectedIdx]) != committedAnchorWord {
                let searchLo = max(0, committedWordCount - 3)
                let searchHi = min(totalWords - 1, committedWordCount + 1)
                if searchLo <= searchHi {
                    for candidate in searchLo...searchHi {
                        if bareWord(realWordTexts[candidate]) == committedAnchorWord {
                            committedWordCount = candidate + 1
                            break
                        }
                    }
                }
            }
        }
        if committedWordCount > totalWords {
            committedWordCount = max(0, totalWords - 1)
        }

        var segments: [SentenceSegment] = []
        var curText = ""
        var curWordCount: Int = 0
        var curStart: Double = -1.0
        var curEnd: Double = -1.0
        var curNonSpaceChars: Int = 0
        var segStartWordIdx: Int = committedWordCount
        var globalWordIdx: Int = 0

        for i in 0..<tokens.count {
            let tok = tokens[i]
            if isAttachedPunctuationOnly(tok.text) {
                // Skip punctuation attached to already-committed words
                if globalWordIdx <= committedWordCount && curText.isEmpty {
                    continue
                }
                if !curText.isEmpty {
                    curText += tok.text
                    curNonSpaceChars += tok.text.count
                }
            } else {
                if globalWordIdx < committedWordCount {
                    globalWordIdx += 1
                    continue
                }
                let cleanedTok = curText.isEmpty ? stripLeadingClosingPunctuation(tok.text) : tok.text
                if !cleanedTok.isEmpty {
                    if curText.isEmpty {
                        segStartWordIdx = globalWordIdx
                    } else if tok.leadingSpace {
                        curText += " "
                    }
                    curText += cleanedTok
                    curWordCount += 1
                    curNonSpaceChars += cleanedTok.count
                }
                globalWordIdx += 1
            }

            if curText.isEmpty {
                continue
            }

            if let s = tok.startSec {
                if curStart < 0 || s < curStart { curStart = s }
            }
            if let e = tok.endSec {
                if curEnd < 0 || e > curEnd { curEnd = e }
            }

            // If the next token is purely attached closing punctuation, fold it first
            if i + 1 < tokens.count && isAttachedPunctuationOnly(tokens[i + 1].text) {
                continue
            }

            let hasSentenceEnd = curNonSpaceChars >= 3 && endsWithSentencePunctuation(curText)
            let hasLongClauseBreak = (curWordCount >= 18 || curNonSpaceChars >= 95)
                && endsWithClauseSeparator(curText)

            if (hasSentenceEnd || hasLongClauseBreak) && i + 1 < tokens.count && !curText.isEmpty {
                segments.append(SentenceSegment(
                    text: curText,
                    startSec: curStart,
                    endSec: curEnd,
                    wordCount: max(1, curWordCount),
                    nonSpaceChars: curNonSpaceChars,
                    startWordIdx: segStartWordIdx,
                    endWordIdx: max(segStartWordIdx + 1, globalWordIdx)
                ))
                curText = ""
                curWordCount = 0
                curStart = -1.0
                curEnd = -1.0
                curNonSpaceChars = 0
                segStartWordIdx = globalWordIdx
            }
        }

        if !curText.isEmpty {
            segments.append(SentenceSegment(
                text: curText,
                startSec: curStart >= 0 ? curStart : (segments.isEmpty && committedWordCount == 0 ? fallbackStart : -1.0),
                endSec: curEnd >= 0 ? curEnd : (segments.isEmpty && committedWordCount == 0 ? fallbackEnd : -1.0),
                wordCount: max(1, curWordCount),
                nonSpaceChars: curNonSpaceChars,
                startWordIdx: segStartWordIdx,
                endWordIdx: max(segStartWordIdx + 1, globalWordIdx)
            ))
        }

        return segments
    }

    private func resolveAndEmitSegment(_ seg: SentenceSegment, index: Int, isFinal: Bool, fallbackTurnStart: Double) {
        let trimmed = Self.stripLeadingClosingPunctuation(seg.text).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        lock.lock()
        let nowFedSec = totalFedAudioSec
        lock.unlock()

        // Compute word-arrival start & end from `wordTimeline` when per-word `audioTimeRange` isn't attached yet
        var wordStartSec: Double = -1.0
        var wordEndSec: Double = -1.0
        if !wordTimeline.isEmpty {
            let sIdx = min(max(0, seg.startWordIdx), wordTimeline.count - 1)
            let eIdx = min(max(sIdx, seg.endWordIdx - 1), wordTimeline.count - 1)
            if sIdx == 0 && fallbackTurnStart >= 0 {
                wordStartSec = fallbackTurnStart
            } else {
                wordStartSec = max(0.0, wordTimeline[sIdx] - 0.14)
            }
            wordEndSec = max(wordStartSec + 0.12, wordTimeline[eIdx])
        }

        // Ensure segmentTimings has an entry up to `index`
        while segmentTimings.count <= index {
            let newIdx = segmentTimings.count
            let prevEnd = newIdx > 0 ? segmentTimings[newIdx - 1].endSec : -1.0
            let defaultStart = wordStartSec >= 0
                ? wordStartSec
                : (prevEnd >= 0 ? prevEnd : max(0.0, nowFedSec - 0.25))
            segmentTimings.append((startSec: defaultStart, endSec: max(defaultStart + 0.12, wordEndSec >= 0 ? wordEndSec : nowFedSec)))
        }

        var resolvedStart = segmentTimings[index].startSec
        var resolvedEnd = segmentTimings[index].endSec

        if seg.startSec >= 0 {
            resolvedStart = seg.startSec
        } else if wordStartSec >= 0 {
            resolvedStart = wordStartSec
        }

        if seg.endSec > resolvedStart {
            resolvedEnd = seg.endSec
        } else if isFinal && wordEndSec > resolvedStart {
            resolvedEnd = wordEndSec
        } else if !isFinal {
            resolvedEnd = max(resolvedStart + 0.12, wordEndSec > resolvedStart ? wordEndSec : nowFedSec)
        } else {
            resolvedEnd = max(resolvedStart + 0.15, resolvedEnd)
        }

        segmentTimings[index] = (startSec: resolvedStart, endSec: resolvedEnd)
        let durationSec = max(0.12, resolvedEnd - resolvedStart)

        trimmed.withCString { cstr in
            self.callback(self.ctx, cstr, isFinal, resolvedStart, durationSec)
        }
    }

    func begin() {
        Task { [weak self] in
            guard let self else { return }
            do {
                if let request = try await AssetInventory.assetInstallationRequest(supporting: [self.transcriber]) {
                    try await request.downloadAndInstall()
                }
                let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [self.transcriber])
                self.setFormat(format)
                try await self.analyzer.start(inputSequence: self.inputStream)
                for try await result in self.transcriber.results {
                    self.lock.lock()
                    let nowFedSec = self.totalFedAudioSec
                    self.lock.unlock()

                    let fallbackTurnStart = result.range.start.isNumeric ? result.range.start.seconds : -1.0
                    let segments = Self.extractSentenceSegments(
                        result,
                        wordTimeline: &self.wordTimeline,
                        committedWordCount: &self.committedWordCount,
                        committedAnchorWord: self.committedAnchorWord,
                        nowFedSec: nowFedSec
                    )
                    if segments.isEmpty {
                        if result.isFinal {
                            self.committedSentenceCount = 0
                            self.committedWordCount = 0
                            self.committedAnchorWord = ""
                            self.segmentTimings.removeAll()
                            self.wordTimeline.removeAll()
                        }
                        continue
                    }

                    if result.isFinal {
                        for seg in segments {
                            self.resolveAndEmitSegment(
                                seg,
                                index: self.committedSentenceCount,
                                isFinal: true,
                                fallbackTurnStart: fallbackTurnStart
                            )
                            self.committedSentenceCount += 1
                            self.committedWordCount = seg.endWordIdx
                            self.committedAnchorWord = Self.extractLastBareWord(seg.text)
                        }
                        self.committedSentenceCount = 0
                        self.committedWordCount = 0
                        self.committedAnchorWord = ""
                        self.segmentTimings.removeAll()
                        self.wordTimeline.removeAll()
                    } else {
                        // Cut and finalize as soon as the next sentence has >= 2 words (>= 5 chars)
                        // so tentative 1-word punctuation hypotheses never cause false splits!
                        var localIdx = 0
                        while localIdx + 1 < segments.count {
                            let nextSeg = segments[localIdx + 1]
                            if nextSeg.wordCount >= 2 && nextSeg.nonSpaceChars >= 5 {
                                let completedSeg = segments[localIdx]
                                self.resolveAndEmitSegment(
                                    completedSeg,
                                    index: self.committedSentenceCount,
                                    isFinal: true,
                                    fallbackTurnStart: fallbackTurnStart
                                )
                                self.committedSentenceCount += 1
                                self.committedWordCount = completedSeg.endWordIdx
                                self.committedAnchorWord = Self.extractLastBareWord(completedSeg.text)
                                localIdx += 1
                            } else {
                                break
                            }
                        }

                        // Pipe the remaining active sentence (or combined unconfirmed tail) into a live partial!
                        if localIdx < segments.count {
                            var activeSeg = segments[localIdx]
                            // If there is a 1-word unconfirmed tail in `localIdx + 1`, hold off showing
                            // the next sentence's first syllable on the old sentence so the cut stays clean.
                            self.resolveAndEmitSegment(
                                activeSeg,
                                index: self.committedSentenceCount,
                                isFinal: false,
                                fallbackTurnStart: fallbackTurnStart
                            )
                        }
                    }
                }
            } catch {
                "".withCString { cstr in self.callback(self.ctx, cstr, true, -1.0, -1.0) }
            }
        }
    }

    private func setFormat(_ format: AVAudioFormat?) {
        lock.lock()
        analyzerFormat = format
        let held = pending
        pending.removeAll()
        lock.unlock()
        for chunk in held { yield(chunk.samples, rate: chunk.rate) }
    }

    func feed(_ samples: [Float], rate: Double) {
        lock.lock()
        let ready = analyzerFormat != nil
        if !ready {
            // Cap the pre-setup hold to ~2 s so a slow first-run model install can't grow it unbounded.
            if pending.reduce(0, { $0 + $1.samples.count }) < Int(rate * 2) {
                pending.append((samples, rate))
            }
            lock.unlock()
            return
        }
        lock.unlock()
        yield(samples, rate: rate)
    }

    private func yield(_ samples: [Float], rate: Double) {
        guard let target = analyzerFormat else { return }
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false) else { return }
        guard let inBuf = makeBuffer(samples, format: source) else { return }

        let out: AVAudioPCMBuffer
        if source.sampleRate == target.sampleRate && source.commonFormat == target.commonFormat {
            out = inBuf
        } else {
            if converter == nil || converter?.outputFormat != target { converter = AVAudioConverter(from: source, to: target) }
            guard let converter else { return }
            let ratio = target.sampleRate / source.sampleRate
            let cap = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 16
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return }
            var supplied = false
            var err: NSError?
            converter.convert(to: outBuf, error: &err) { _, status in
                if supplied { status.pointee = .noDataNow; return nil }
                supplied = true
                status.pointee = .haveData
                return inBuf
            }
            if err != nil { return }
            out = outBuf
        }
        lock.lock()
        totalFedAudioSec += Double(samples.count) / max(1.0, rate)
        lock.unlock()
        continuation.yield(AnalyzerInput(buffer: out))
    }

    private func makeBuffer(_ samples: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return nil }
        buf.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buf.floatChannelData { samples.withUnsafeBufferPointer { ch[0].update(from: $0.baseAddress!, count: samples.count) } }
        return buf
    }

    func stop() {
        continuation.finish()
        Task { [analyzer] in try? await analyzer.finalizeAndFinishThroughEndOfInput() }
    }
}

/// Starts a session for `locale` (BCP-47, e.g. "en-US"/"zh-CN"/"yue-Hant"); returns an opaque handle or
/// null on an unsupported OS. `callback(ctx, text, isFinal, startSec, durationSec)` fires for each partial/final result.
@_cdecl("wisp_applespeech_start")
public func wisp_applespeech_start(
    _ locale: UnsafePointer<CChar>?,
    _ ctx: UnsafeMutableRawPointer?,
    _ callback: @escaping @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?, Bool, Double, Double) -> Void
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 26.0, *) else { return nil }
    let requested = locale.map { String(cString: $0) } ?? "en-US"
    // "auto" means "follow the system language" — resolve it to the host's current locale.
    let id = requested == "auto" ? Locale.current.identifier : requested
    let session = WispSpeechSession(locale: id, ctx: ctx, callback: callback)
    session.begin()
    return Unmanaged.passRetained(session).toOpaque()
}

/// Feeds `count` mono Float32 samples at `rate` Hz into the session.
@_cdecl("wisp_applespeech_feed")
public func wisp_applespeech_feed(_ handle: UnsafeMutableRawPointer?, _ samples: UnsafePointer<Float>?, _ count: Int, _ rate: Double) {
    guard #available(macOS 26.0, *), let handle, let samples else { return }
    let session = Unmanaged<WispSpeechSession>.fromOpaque(handle).takeUnretainedValue()
    session.feed(Array(UnsafeBufferPointer(start: samples, count: count)), rate: rate)
}

/// Stops the session and releases the handle.
@_cdecl("wisp_applespeech_stop")
public func wisp_applespeech_stop(_ handle: UnsafeMutableRawPointer?) {
    guard #available(macOS 26.0, *), let handle else { return }
    let session = Unmanaged<WispSpeechSession>.fromOpaque(handle).takeRetainedValue()
    // The Rust caller invokes this while joining the capture thread that owns the engine. Finishing
    // the analyzer and releasing it (the ARC deinit on return) can block, which would stall that join
    // and make "Stop" hang. Hand the session to a background queue: stop() and the deinit run there,
    // and this @_cdecl call returns immediately so the capture thread joins at once.
    DispatchQueue.global(qos: .utility).async {
        session.stop()
    }
}
