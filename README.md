# A Whisper CLI, built with Rust

This project attempts to build a simple Whisper CLI with Rust, to replace the base Python one. It uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) under the hood, making it significantly faster on M1 systems.

## Installation

You can download the binary corresponding to your OS from the [latest release](https://github.com/m1guelpf/whisper-cli-rs/releases/latest), or build it from scratch by cloning the repo and running `cargo build --release`.

## Run from anywhere

Put the `whisper_cli` binary in `/usr/local/bin` on Unix systems (Mac/Linux) & make sure it has permissions to execute (use `chmod +x whisper_cli` in terminal.)

Close & Re-open the terminal to test it by typing `whisper_cli --help`. It should output the following.

## Usage

```bash
$ whisper_cli --help
Generate a transcript of an audio file using the Whisper speech-to-text engine. The transcript will be saved as a .txt, .vtt, and .srt file in the same directory as the audio file.

Usage: whisper_cli [OPTIONS] <AUDIO>

Arguments:
  <AUDIO> Path to the audio file to transcribe

Options:
  -m, --model <MODEL>
          Name of the Whisper model to use

          [default: medium]
          [possible values: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large, large-v1]

  -l, --lang <LANG>
          Language spoken in the audio. Attempts to auto-detect by default

          [default: auto]
          [possible values: auto, en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, haw, ln, ha, ba, jw, su]

  -h, --help
          Print help information (use `-h` for a summary)

  -V, --version
          Print version information
```

## Develop

Make sure you have the latest version of rust installed (use [rustup](https://rustup.rs/)). Then, you can build the project by running `cargo build`, and run it with `cargo run`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
