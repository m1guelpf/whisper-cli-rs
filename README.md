# whisper-cli

A high-performance command line tool in Rust for transcribing audio and video files using modern Speech-to-Text APIs (Groq, OpenAI, Deepgram) and self-hosted local OpenAI-compatible endpoints (`whisper-server`, Faster-Whisper, Ollama, LocalAI, vLLM).

Generates transcriptions in `.txt`, `.vtt`, and `.srt` formats automatically.

--- 

### Recommended companion Chrome & Firefox extension: 
**https://github.com/aramrw/whisper-cli-sub-injector**

---

> **--live** flag cli example

```
~/Programming/whisper-cli> whisper --live --speech --preset fast -l es -m large-v3-turbo-q5_0
🚀 Starting background whisper-server [fast] on port 8080...
   Command: whisper-server -m /Users/aramsamifanni/.cache/whisper-models/ggml-large-v3-turbo-q5_0.bin --port 8080 -t 8 -fa -mc 0 -sns -l es -nth 0.45 -ac 768 -nf -bo 1 -bs 1
✅ whisper-server ready in 0.5s!

╭──────────────────────────────────────────────────────────────╮
│ 🎧 macOS Live System Audio Subtitles (ScreenCaptureKit)      │
╰──────────────────────────────────────────────────────────────╯
  • Provider : Custom (http://localhost:8080 • fast preset)
  • Model    : ggml-large-v3-turbo-q5_0.bin
  • Subtitles: http://127.0.0.1:8989/events (Browser Injector)
  • Language : es
  • Mode     : Smart Speech VAD (waits for sentence end up to 15s, filters SFX/music) (Press Ctrl+C to stop)

[00:44]  Gracias. ¿Qué tal si mejor yo se lo llevo a Kafka?
[00:48]  No, no hay problema. Yo mismo se lo llevaré.
[01:06]  Parece que ya lo aprobaron. Pero este será su último año, ¿no? Ah.
[01:11]  Así es su última vez, aunque ojalá le dieran una oportunidad.
[01:15]  Se ha esforzado con empeño durante mucho tiempo.
[01:29]  ambos pasamos el primer examen
[01:36]  ¿De verdad? ¡Oye!
[01:38]  Oye, ¿pero por qué te ves así? Jejeje
[01:41]  ¿Te despertaste de malas, Ichikawa? ¡Pues solo mírate la cara! ¿En serio eres consciente de tu situación actual? Ay, lo siento mucho, es que... A veces se me sale sin que me dé cuenta. 
```

---

## Features

- **Native macOS Live Subtitles (`--live`)**: Tap directly into macOS system audio via Apple's `ScreenCaptureKit` (no BlackHole or virtual audio drivers required) for real-time terminal subtitles from Crunchyroll, YouTube, Netflix, or any app.
- **Ultra-Fast Cloud & Local Transcription**: Uses Groq (`whisper-large-v3-turbo`), OpenAI (`whisper-1`), Deepgram (`nova-3`), or local GPU-accelerated endpoints.
- **Universal Audio & Video Support**: Automatic `ffmpeg` pre-decoding for `.mp3`, `.mp4`, `.m4a`, `.wav`, `.mkv`, `.mov`, `.flac`, `.ogg`, and `.webm`.
- **Multi-Provider Architecture**: Switch easily between Groq, OpenAI, Deepgram, or local/self-hosted servers.
- **Multiple Output Formats**: Automatically writes `.txt`, `.vtt`, and `.srt` subtitle files in the same directory as the input file.
- **Terminal Output Mode**: Use `-s` / `--stdout` to output transcript text directly to standard output without saving files.
- **Word-level Timestamps**: Generate karaoke-style word timestamps with `--karaoke`.
- **Language Selection & Translation**: Explicit language specification (`-l es`, `-l en`, `-l ja`) and direct translation to English (`-t / --translate`).

---

## Installation & Build

Ensure you have Rust installed ([rustup.rs](https://rustup.rs/)).

```bash
git clone https://github.com/aramrw/whisper-cli && cd whisper-cli
cargo build --release
```

Install binary system-wide:
```bash
cargo install --path .
```

---

## Setting Up a Local Server (`whisper-cpp` on macOS)

To run transcriptions 100% locally with Apple Silicon Metal GPU acceleration without sending data to cloud APIs:

### 1. Install `whisper-cpp`
```bash
brew install whisper-cpp
```

### 2. Download a model file
* **Medium** (Recommended for speed & high accuracy):
  ```bash
  mkdir -p ~/.cache/whisper-models
  curl -L -o ~/.cache/whisper-models/ggml-medium.bin "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
  ```
* **Large v3 Turbo**:
  ```bash
  curl -L -o ~/.cache/whisper-models/ggml-large-v3-turbo.bin "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
  ```

### 3. Start the Local Server (Using Built-in Presets or Manually)

You can launch `whisper-server` directly through `whisper --serve <preset>` (or auto-start it in the background with `whisper --live --preset <preset>`):

```bash
# Balanced preset (default): low hallucinations, great for Spanish & Japanese
whisper --serve balanced -l es

# Live preset: ~3x faster encoder window (-ac 512) & no temperature fallback stalls (-nf)
whisper --serve live -l ja

# Fast preset: ultra-low latency (-ac 384 -nf -bo 1 -bs 1)
whisper --serve fast -l es

# Accurate preset: 5-way beam search (-bo 5 -bs 5) for maximum quality
whisper --serve accurate -l ja
```

| Preset | Flags Applied | Best For |
| :--- | :--- | :--- |
| **`balanced`** | `-t 8 -fa -nth 0.5 -mc 0 -sns` | Proven baseline for Spanish & Japanese with very low hallucinations |
| **`live`** | `balanced` + `-ac 512 -nf -bo 1` | Real-time video/anime subtitles (~3x faster Metal encoder on short chunks) |
| **`fast`** | `-t 8 -fa -nth 0.45 -mc 0 -sns -ac 384 -nf -bo 1 -bs 1` | Rapid dialogue with minimal subtitle delay (`2.8s` chunks) |
| **`accurate`** | `-t 8 -fa -nth 0.55 -mc 0 -sns -bo 5 -bs 5` | Full file transcription or difficult audio using 5-beam search |

> **What Do These `whisper-server` Flags Hook Into? (Applies to ALL `ggml` Whisper Models)**
> - `-fa` (`--flash-attn`): Enables Flash Attention on Apple Metal GPU.
> - `-t 8` (`--threads`): Uses 8 CPU performance threads.
> - `-nth 0.5` (`--no-speech-thold`): Discards segments where `<|nospeech|>` probability exceeds `0.5` (prevents silence/SFX hallucinations).
> - `-mc 0` (`--max-context`): Resets text context memory between windows so an error never loops indefinitely.
> - `-sns` (`--suppress-nst`): Masks out non-speech tokens (`[Music]`, `[Laughter]`, `♪`) at the logits level.
> - `-ac 512` (`--audio-ctx`): Shrinks Whisper's encoder window from the default 30s (`1500` frames) to ~10s (`512` frames) so 3–4s live chunks aren't padded with 26s of silence (**3x–4x speedup**).
> - `-nf` (`--no-fallback`): Disables multi-pass temperature retry loops (`0.0 → 1.0`) when audio is noisy, preventing 2–3s stalls and high-temperature hallucinations.
> - `-bo N` / `-bs N`: Controls candidate sampling (`--best-of`) and beam search width (`--beam-size`).

You can also override the model (`-m medium` or `-m /path/to/model.bin`), port (`--port 8081`), or append custom flags (`--server-args "-nth 0.6"`).

### 4. Transcribe a File with `whisper-cli`
```bash
whisper -p custom -u http://localhost:8080 audio.mp3
```
#### `-s` `(--stdout)` displays subs directly in the terminal.

```bash
whisper -p custom -u http://localhost:8080 -l es -s audio.mp3
```

---

## Live Real-Time macOS System Audio Subtitles (`--live`)

Stream live subtitles directly from any app playing audio on your Mac (Crunchyroll, YouTube, podcasts, video calls) using native macOS `ScreenCaptureKit`—**no BlackHole or virtual audio routing required**.

> **First-Time Permission**: On first run, macOS will ask to allow your terminal app under **System Settings → Privacy & Security → Screen & System Audio Recording**.

### One-Command Live Mode (Auto-Starts `whisper-server` in Background)
```bash
# Spanish live subtitles (auto-starts whisper-server with the 'live' preset)
whisper --live --preset live -l es

# Japanese live subtitles (e.g. Crunchyroll / Anime)
whisper --live --preset live -l ja

# Using your 'balanced' preset
whisper --live --preset balanced -l es
```

### Smart Sentence Mode (`--speech` — Ignores Anime SFX/Music & Waits for Sentence End)
Uses Apple's Neural Engine `SoundAnalysis` framework + a `300Hz–3400Hz` vocal-band tracker to distinguish human voices from explosions/music and hold long sentences open until the character finishes speaking:
```bash
whisper --live --speech --preset live -l es
whisper --live --speech --preset live -l ja
```

### Connect to an Already-Running `whisper-server`
```bash
whisper --live --speech -p custom -l es
whisper --live --speech -p custom -l ja
```

### Live Subtitles Translated to English in Real Time
```bash
whisper --live --speech --preset live -l ja --translate
```

### Live Subtitles with Groq Cloud (Ultra-Low Latency)
```bash
whisper --live --speech -p groq -l es
```

### Test macOS System Audio Capture & Waveform Monitor
```bash
whisper --test-audio
```

### Save Live Session to `.srt`, `.vtt`, and `.txt` on Exit (`Ctrl+C`)
```bash
whisper --live --preset live -l ja episode_01
```

---

## API Key Setup (Cloud Providers)

Set the environment variable for your preferred provider:

* **Groq** (Default - recommended for speed & free tier):
  ```bash
  export GROQ_API_KEY="gsk_..."
  ```
* **OpenAI**:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
* **Deepgram**:
  ```bash
  export DEEPGRAM_API_KEY="..."
  ```

Alternatively, pass `-k / --api-key` directly on the command line.

---

## Usage Examples

### 1. Transcribe with Groq (Default)
```bash
whisper video.mp4
```

### 2. Specify Language (e.g. Spanish, Japanese, English)
```bash
whisper -l es spanish_audio.mp3
whisper -l ja anime_clip.mp4
whisper -l en podcast.m4a
```

### 3. Generate `.srt`, `.vtt`, and `.txt` Files locally via `whisper-server`
```bash
whisper -p custom -u http://localhost:8080 -l es ~/Downloads/video.mp4
```

### 4. Print Directly to Terminal Stdout (No Files Created)
```bash
whisper -s -p custom -u http://localhost:8080 audio.mp3
```

### 5. Generate Word-Level Subtitles (Karaoke mode)
```bash
whisper --karaoke video.mp4
```

### 6. Translate Audio to English
```bash
whisper --translate video.mp4
```

---

## CLI Options (`whisper --help`)

```text
Generate a transcript of an audio file, launch a local whisper-server with tuned presets, or stream real-time subtitles from macOS system audio using Groq, OpenAI, Deepgram, or a custom OpenAI-compatible API endpoint (like whisper-server).

Usage: whisper [OPTIONS] [AUDIO]

Arguments:
  [AUDIO]  Path to the audio file to transcribe (optional when using --live, --serve, or --test-audio)

Options:
  -p, --provider <PROVIDER>          Speech-to-Text provider to use (groq, openai, deepgram, custom) [default: groq]
  -m, --model <MODEL>                Name or path of the model to use (e.g. large-v3-turbo, medium, whisper-1, nova-3)
  -l, --lang <LANG>                  Language spoken in the audio (ISO 639-1 code e.g. 'en', 'es', 'ja')
      --live                         Capture macOS system audio in real time and stream subtitles to the terminal
      --serve [<SERVE>]              Launch local whisper-server in the foreground using a preset (balanced, live, fast, accurate)
      --preset <PRESET>              Server preset to use (auto-starts background whisper-server when used with --live)
      --port <PORT>                  Port for local whisper-server (used with --serve and --preset) [default: 8080]
      --server-args <SERVER_ARGS>    Extra raw flags to append to whisper-server (e.g. --server-args "-nth 0.6 -ac 256")
      --test-audio                   Test macOS system audio capture with a live waveform & byte monitor
      --chunk-secs <CHUNK_SECS>      Maximum audio chunk duration in seconds for --live mode
  -k, --api-key <API_KEY>            API Key (defaults to GROQ_API_KEY, OPENAI_API_KEY, or DEEPGRAM_API_KEY env var)
  -u, --api-url <API_URL>            Custom API Base URL for local/self-hosted OpenAI-compatible speech-to-text endpoints
  -t, --translate                    Toggle translation to English
      --karaoke                      Generate timestamps for each word
  -s, --stdout                       Print transcript directly to terminal stdout
      --no-files                     Skip writing .txt, .vtt, and .srt files to disk
  -h, --help                         Print help
  -V, --version                      Print version
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
