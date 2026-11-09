# whisper-cli

A command-line tool in Rust for audio file transcription and real-time macOS system audio subtitles.

Works with on-device engines (Apple Speech, NVIDIA Parakeet CoreML, whisper.cpp, Moonshine) as well as cloud Speech-to-Text APIs (Groq, OpenAI, Deepgram) and self-hosted OpenAI-compatible endpoints.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Models and Downloads](#models-and-downloads)
4. [Transcribing Audio Files](#transcribing-audio-files)
5. [Live System Audio Subtitles (macOS)](#live-system-audio-subtitles-macos)
6. [Browser Extension Subtitles](#browser-extension-subtitles)
7. [Local Server Presets (whisper-cpp)](#local-server-presets-whisper-cpp)
8. [API Key Setup](#api-key-setup)
9. [CLI Reference](#cli-reference)

---

## Installation

Ensure you have Rust installed ([rustup.rs](https://rustup.rs/)).

```bash
git clone https://github.com/aramrw/whisper-cli.git
cd whisper-cli

# Install binary to ~/.cargo/bin:
cargo install --path .
```

To include the optional Moonshine ONNX streaming engine:
```bash
cargo install --path . --features moonshine
```

---

## Quick Start

### 1. Transcribe an Audio File
```bash
# Uses Groq cloud by default (requires GROQ_API_KEY environment variable)
whisper audio.mp3

# Or transcribe locally using Parakeet (macOS Apple Silicon)
whisper -p parakeet audio.mp3
```
By default, this writes `.txt`, `.srt`, and `.vtt` subtitle files in the same directory as the audio file.

### 2. Live macOS System Audio Subtitles
Capture system audio in real time (from YouTube, Crunchyroll, meetings, or any app) without virtual audio cables:

```bash
# Using Apple Speech on Apple Neural Engine (no model download needed):
whisper --apple -l en

# Or using local Whisper with automatic model download:
whisper --live --preset live -l en
```

> **macOS Permission**: On first launch of any live audio mode, macOS will request permission under **System Settings → Privacy & Security → Screen & System Audio Recording**.

---

## Models and Downloads

Depending on the engine you use, models are handled differently:

| Engine | Flag | Where it runs | How models are handled |
| :--- | :--- | :--- | :--- |
| **Apple Speech** | `--apple` | Apple Neural Engine (macOS) | **No download needed.** Uses built-in macOS speech assets. (If a language isn't installed, enable it in macOS Dictation settings). |
| **Local Whisper** | `--live --preset <name>` or `--serve` | Apple Metal GPU / CPU | **Auto-downloads on prompt.** If the model file is not found in `~/.cache/whisper-models/`, the CLI prompts to download it from Hugging Face. |
| **Parakeet** | `--parakeet` or `-p parakeet` | Apple Neural Engine CoreML | **Auto-prompts to install.** Uses `macparakeet-cli`. If missing, prompts to install via Homebrew and downloads the model automatically. |
| **Moonshine** | `--moonshine` | ONNX Runtime | **Auto-downloads on first run.** Caches models in `~/.cache/whisper-models/moonshine/`. |
| **Cloud APIs** | Default, `-p openai`, `-p deepgram` | Cloud servers | **No local models.** Requires setting the corresponding API key. |

### Manual Whisper Model Download (Optional)

If you prefer to download `whisper.cpp` models manually instead of letting the CLI do it:

```bash
mkdir -p ~/.cache/whisper-models

# Large v3 Turbo (~1.6 GB)
curl -L -o ~/.cache/whisper-models/ggml-large-v3-turbo.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"

# Medium (~1.5 GB)
curl -L -o ~/.cache/whisper-models/ggml-medium.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"
```

---

## Transcribing Audio Files

### Basic Transcription
```bash
whisper podcast.mp3
```

### Specify Spoken Language
Specify an ISO 639-1 language code (`en`, `es`, `ja`, etc.):
```bash
whisper -l es spanish_lesson.mp3
whisper -l ja anime_raw.mp4
```

### Choose Provider
```bash
# Cloud providers:
whisper -p groq audio.mp3
whisper -p openai audio.mp3
whisper -p deepgram audio.mp3

# Local providers:
whisper -p parakeet audio.mp3
whisper -p custom -u http://localhost:8080 audio.mp3
```

### Output Options
```bash
# Print transcript directly to terminal stdout without writing files:
whisper -s --no-files audio.mp3

# Translate spoken audio to English:
whisper --translate japanese_audio.mp3

# Generate word-level timestamps (karaoke):
whisper --karaoke song.mp3
```

---

## Live System Audio Subtitles (macOS)

Live mode captures audio directly from your system output via `ScreenCaptureKit` and streams text as speech occurs.

### Choosing a Live Engine

#### 1. Apple Speech (`--apple`)
Uses macOS native `SpeechTranscriber` running on the Apple Neural Engine. Fast word-by-word streaming with zero setup.
```bash
whisper --apple -l en
whisper --apple -l ja
whisper --apple -l es
```
*Note: To add languages, go to **System Settings → Keyboard → Dictation** and enable the language. If offline recognition is needed, enable Siri temporarily to download offline voice assets.*

#### 2. NVIDIA Parakeet TDT (`--parakeet`)
Uses the Parakeet CoreML model on Apple Silicon via MacParakeet.
```bash
whisper --parakeet -l en
whisper --parakeet -l es
```

#### 3. Local Whisper Server (`--live --preset <preset>`)
Spawns and manages a local `whisper-server` process with optimized presets:
```bash
# Real-time subtitle preset
whisper --live --preset live -l en

# Balanced preset (low hallucinations)
whisper --live --preset balanced -l es
```

#### 4. Moonshine Streaming (`--moonshine`)
Runs the Moonshine v2 ONNX engine (requires building with `--features moonshine`):
```bash
whisper --moonshine -l en
whisper --moonshine -l es
```

#### 5. Cloud Live Streaming (`--live -p groq`)
Sends rolling audio chunks to cloud APIs:
```bash
whisper --live -p groq -l en
```

### Live Audio Features

- **Smart Speech VAD (`--speech`)**: Uses Apple Neural Engine `SoundAnalysis` and vocal band filtering to ignore background sound effects and music, holding segments open until a sentence finishes.
  ```bash
  whisper --live --speech --preset live -l ja
  ```
- **Quiet / Headless Mode (`-q`, `--silent`)**: Disables terminal animations, HUD, and text output to minimize CPU usage when you only want to stream subtitles to the browser extension.
  ```bash
  whisper --apple -l en -q
  ```
- **Save Live Session to Files**: Pass an output name at the end of the command to save `.srt`, `.vtt`, and `.txt` files when exiting with `Ctrl+C`:
  ```bash
  whisper --live --preset live -l ja session_01
  ```
- **Audio Capture Diagnostic (`--test-audio`)**: Inspect macOS system audio capture with an in-terminal volume waveform:
  ```bash
  whisper --test-audio
  ```

---

## Browser Extension Subtitles

When running any live subtitle mode (`--apple`, `--parakeet`, `--moonshine`, or `--live`), the CLI broadcasts a live Server-Sent Events (SSE) subtitle stream to:

```text
http://127.0.0.1:8989/events
```

You can use the companion Chrome and Firefox extension to overlay these subtitles directly onto web video players (YouTube, Crunchyroll, Netflix, etc.):

- Extension repository: **[whisper-cli-sub-injector](https://github.com/aramrw/whisper-cli-sub-injector)**

<img width="377" height="172" alt="Subtitle Overlay Preview" src="https://github.com/user-attachments/assets/5be1d420-0bcf-4af0-91bd-61351834a639" />

You can change the broadcast port using `--sub-port <PORT>`.

---

## Local Server Presets (whisper-cpp)

To run a standalone local `whisper-server` on macOS using Apple Silicon Metal acceleration:

### 1. Install `whisper-cpp`
```bash
brew install whisper-cpp
```

### 2. Start Server with a Preset
The CLI can manage and launch `whisper-server` using pre-configured presets:
```bash
whisper --serve live -l en
whisper --serve balanced -l es
```

### Preset Details

| Preset | Applied Flags | Use Case |
| :--- | :--- | :--- |
| **`balanced`** | `-t 8 -fa -nth 0.5 -mc 0 -sns` | General-purpose baseline with low hallucination rate. |
| **`live`** | `balanced` + `-ac 512 -nf -bo 1` | Real-time video subtitles; smaller encoder window. |
| **`fast`** | `-t 8 -fa -nth 0.45 -mc 0 -sns -ac 384 -nf -bo 1 -bs 1` | Low latency for rapid dialogue. |
| **`realtime`** | `-ac 256 -nf -bo 1 -bs 1` | Minimal delay streaming (~1.8s chunk window). |
| **`accurate`** | `-t 8 -fa -nth 0.55 -mc 0 -sns -bo 5 -bs 5` | Full context with 5-way beam search for difficult audio. |

### Explanation of Server Flags
- `-fa` (`--flash-attn`): Enables Flash Attention on Apple Silicon Metal GPU.
- `-t 8` (`--threads`): Allocates 8 CPU threads.
- `-nth 0.5` (`--no-speech-thold`): Drops segments where silence/non-speech probability is high.
- `-mc 0` (`--max-context`): Clears context memory between segments to avoid repeating loops.
- `-sns` (`--suppress-nst`): Suppresses non-speech tokens (`[Music]`, `[Applause]`).
- `-ac <frames>` (`--audio-ctx`): Reduces audio context size (e.g. `512` frames instead of 30s `1500` frames) to speed up short-chunk encoding.
- `-nf` (`--no-fallback`): Disables temperature fallback retries, preventing stalls on noisy audio.
- `-bo` / `-bs`: Sets candidate sampling (`--best-of`) and beam search width (`--beam-size`).

You can pass extra server flags using `--server-args`:
```bash
whisper --serve live --server-args "-nth 0.6"
```

---

## API Key Setup

For cloud transcription providers, set the appropriate environment variable:

```bash
# Groq (Default)
export GROQ_API_KEY="gsk_..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Deepgram
export DEEPGRAM_API_KEY="..."
```

Alternatively, pass `-k / --api-key` on the command line:
```bash
whisper -k "gsk_..." audio.mp3
```

---

## CLI Reference

```text
Usage: whisper [OPTIONS] [AUDIO]

Arguments:
  [AUDIO]  Path to the audio file to transcribe (or output base name in live mode)

Options:
  -p, --provider <PROVIDER>        Speech-to-Text provider (groq, openai, deepgram, parakeet, custom) [default: groq]
  -m, --model <MODEL>              Name or path of model to use
  -l, --lang <LANG>                Language code (ISO 639-1, e.g. 'en', 'es', 'ja')

Live & Streaming:
      --apple                      Use native Apple Speech on Apple Neural Engine (macOS only)
      --parakeet                   Use NVIDIA Parakeet TDT CoreML via MacParakeet (macOS only)
      --moonshine                  Use Moonshine v2 streaming ONNX engine
      --live                       Capture macOS system audio in real time
      --speech                     Wait for speaker sentence pauses and filter background SFX/music
      --test-audio                 Test system audio capture with terminal waveform monitor
      --chunk-secs <CHUNK_SECS>    Maximum audio chunk duration in seconds for live mode
      --sub-port <SUB_PORT>        Port for HTTP/SSE subtitle broadcasting [default: 8989]
  -q, --silent                     Headless mode (disables terminal HUD to reduce CPU usage)

Server & Presets:
      --serve [<PRESET>]           Launch whisper-server in foreground (balanced, live, fast, realtime, accurate)
      --preset <PRESET>            Auto-start background whisper-server when used with --live
      --port <PORT>                Port for local whisper-server [default: 8080]
      --server-args <ARGS>         Extra flags to pass to whisper-server

Output & Formatting:
  -t, --translate                  Translate speech to English
      --karaoke                    Generate word-level timestamps
  -s, --stdout                     Print transcript directly to terminal stdout
      --no-files                   Skip writing .txt, .vtt, and .srt files

Connection:
  -k, --api-key <API_KEY>          API key for cloud provider
  -u, --api-url <API_URL>          Custom API base URL (defaults to http://localhost:8080 for custom)

General:
  -h, --help                       Print help
  -V, --version                    Print version
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
