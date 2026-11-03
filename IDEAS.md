# Future Real-Time Subtitle & Streaming Ideas

Notes and architectural ideas from researching state-of-the-art real-time Whisper systems (`ufal/whisper_streaming`, `LocalAgreement-n`, and `whisper.cpp` streaming).

---

## 1. Overlapping / Growing Window with In-Place Refinement (`LocalAgreement`)
Whisper's Transformer encoder-decoder is stateless per HTTP request and becomes more accurate as it hears more surrounding sentence context (5–15s vs. 1–2s).

- **How it works**:
  - Instead of waiting until the end of a long sentence or sending isolated short slices, maintain a growing audio buffer from the start of the current utterance:
    1. At `2.0s` of speech → transcribe `[0.0s .. 2.0s]` and render a live *draft* subtitle line in dim text.
    2. At `4.0s` of speech → transcribe `[0.0s .. 4.0s]` (the entire sentence so far) and rewrite the terminal line in-place as Whisper refines its earlier guess with more context.
    3. When the speaker pauses → run the final pass on the completed utterance and lock/commit the subtitle line.
- **LocalAgreement-2 Policy**:
  - Compare token prefixes across two consecutive window passes (`t-1` and `t`). Any prefix that agrees across two consecutive passes is locked in as confirmed text.
- **Trade-offs**:
  - Multiplies GPU inference calls per sentence (2–3x more HTTP requests to `whisper-server`), and requires careful tuning of agreement thresholds so draft rewrites don't flicker.

---

## 2. Client-Managed Context Prompt Carryover (`prompt` Field)
- **Why server-side context (`-mc`) is disabled**:
  - When `whisper-server` manages context automatically (`-mc -1`), it feeds its previous output tokens into the next window even when the next window is pure anime SFX, music, or silence—causing the 3x repeating line loop bug.
- **Client-side alternative**:
  - Keep `-mc 0` on `whisper-server`, and have `whisper-cli` explicitly send the last confirmed subtitle in the multipart `prompt` field **only** when the new chunk has high human speech confidence (`Voice > 60%`) and duration `>= 1.5s`.
  - When a chunk has low voice confidence or follows a long silence gap (`> 4s`), omit `prompt` so Whisper cannot hallucinate or repeat prior lines.

---

## 3. Source Separation / Vocal Isolation for Heavy Anime SFX (`Demucs` / `Apple VoiceIsolation`)
- Raw dialogue is already >95% accurate, and errors primarily happen when loud explosions, battle SFX, or rock/orchestral music overlap directly on top of a character's voice.
- **Potential approaches**:
  - **macOS `AVCaptureDevice` / `AudioUnit` Voice Isolation (`kAUVoiceIOProperty_VoiceProcessingQuality`)**: Apple's built-in neural voice isolation audio unit strips non-vocal background noise and SFX in real time on the Apple Neural Engine before feeding PCM samples to Whisper.
  - **Lightweight RNNoise / DeepFilterNet**: Fast real-time vocal denoising pass (`16kHz` mono) prior to WAV encoding when `--speech` detects low SNR (loud background floor during speech).
