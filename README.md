<div align="center">
  <h1>A Whisper CLI, built with Rust</h1>

  <a href="https://crates.io/crates/whisper_cli">
    <img src="https://img.shields.io/crates/v/whisper_cli.svg" alt="crates.io" />
  </a>
  <a href="https://crates.io/crates/whisper_cli">
    <img src="https://img.shields.io/crates/d/whisper_cli.svg" alt="download count badge" />
  </a>
  <a href="https://docs.rs/whisper_cli">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="docs.rs" />
  </a>
  <br /></br />
</div>

This project attempts to build a simple Whisper CLI with Rust, to replace the base Python one. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood, making it significantly faster on M1 systems.

## Installation

You can download the binary corresponding to your OS from the [latest release](https://github.com/m1guelpf/whisper-cli-rs/releases/latest), or build it from scratch with `cargo install whisper_cli`.

## Run from anywhere

Put the `whisper` binary in `/usr/local/bin` on Unix systems (Mac/Linux) & make sure it has permissions to execute (use `chmod +x whisper` in terminal.)

Close & Re-open the terminal to test it by typing `whisper --help`. It should output the following.

## Usage

```bash
$ whisper --help
Generate a transcript of an audio file using the Whisper speech-to-text engine. The transcript will be saved as a .txt, .vtt, and .srt file in the same directory as the audio file.

Usage: whisper [OPTIONS] <AUDIO>

Arguments:
  <AUDIO> Path to the audio file to transcribe

Options:
  -m, --model <MODEL>
          Name of the Whisper model to use

          [default: medium]
          [possible values: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large, large-v1]

  -l, --lang <LANG>
          Language spoken in the audio. Attempts to auto-detect by default

          [possible values: auto, en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, haw, ln, ha, ba, jw, su]

  -t, --translate
          Toggle translation

  -k, --karaoke
          Generate timestamps for each word

  -h, --help
          Print help information (use `-h` for a summary)

  -V, --version
          Print version information
```

## Develop

### Requirements

- **rust**: Make sure you have the latest version of rust installed (use [rustup](https://rustup.rs/))
- **cmake**: If you are on Mac, you can install it with `brew install cmake`

### Build

```sh
cargo build
```

### Run

```sh
cargo run
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
