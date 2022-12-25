# A Whisper CLI, built with Rust

This project attempts to build a simple Whisper CLI with Rust, to replace the base Python one. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood, making it significantly faster on M1 systems.

## Installation

You can download the binary corresponding to your OS from the [latest release](https://github.com/m1guelpf/whisper-cli-rs/releases/latest), or build it from scratch by cloning the repo and running `cargo build --release`.

## Usage

Run the CLI tool with `whisper_cli <audio_file>`. It'll transcribe the audio with the selected model (downloading it first if it doesn't exist), and generate `<audio_file>.txt` (a full transcription) and subtitle files in various formats (`<audio_file>.vtt` and `<audio_file>.srt`).

## Develop

Make sure you have the latest version of rust installed (use [rustup](https://rustup.rs/)). Then, you can build the project by running `cargo build`, and run it with `cargo run`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
