#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use clap::Parser;
use cpal::traits::HostTrait;
use futures_util::{pin_mut, StreamExt};
use std::path::Path;
use utils::write_to;

mod utils;

use whisper_cli::{Language, Model, Size, Whisper};

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Locally transcribe audio files, using Whisper.",
    long_about = "Generate a transcript of an audio file using the Whisper speech-to-text engine. The transcript will be saved as a .txt, .vtt, and .srt file in the same directory as the audio file."
)]
struct Args {
    /// Name of the Whisper model to use
    #[clap(short, long, default_value = "medium")]
    model: Size,

    /// Language spoken in the audio. Attempts to auto-detect by default.
    #[clap(short, long)]
    lang: Option<Language>,

    /// Path to the audio file to transcribe
    audio: Option<String>,

    /// Toggle translation
    #[clap(short, long, default_value = "false")]
    translate: bool,

    /// Generate timestamps for each word
    #[clap(short, long, default_value = "false")]
    karaoke: bool,

    /// Stream audio from a microphone
    #[clap(short, long, default_value = "false")]
    stream: bool,
}

#[tokio::main]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
async fn main() {
    let mut args = Args::parse();

    if args.model.is_english_only() && (args.lang == Some(Language::Auto) || args.lang.is_none()) {
        args.lang = Some(Language::English);
    }

    assert!(
        !args.model.is_english_only() || args.lang == Some(Language::English),
        "The selected model only supports English."
    );

    let mut whisper = Whisper::new(Model::new(args.model), args.lang).await;

    if args.stream {
        assert!(
            args.audio.is_none(),
            "Cannot stream and transcribe an audio file at the same time."
        );

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("Failed to get default input device.");

        let stream = whisper.listen(device).unwrap();
        pin_mut!(stream);

        while let Some(Ok(chunk)) = stream.next().await {
            // if chunk.offset > 0 {
            //     print!(
            //         "\x1B[{}D{}\x1B[{}D",
            //         chunk.offset,
            //         " ".repeat(chunk.offset),
            //         chunk.offset
            //     );
            // }

            print!("{}", chunk.text);
            // std::io::stdout().flush().unwrap();
        }
    }

    let Some(audio) = args.audio else { panic!("Please provide a path to an audio file.") };

    let audio = Path::new(&audio);
    assert!(audio.exists(), "The provided audio file does not exist.");

    let transcript = whisper
        .transcribe(audio, args.translate, args.karaoke)
        .unwrap();

    let file_name = audio.file_name().unwrap().to_str().unwrap();
    write_to(
        audio.with_file_name(format!("{file_name}.txt")),
        &transcript.as_text(),
    );
    write_to(
        audio.with_file_name(format!("{file_name}.vtt")),
        &transcript.as_vtt(),
    );
    write_to(
        audio.with_file_name(format!("{file_name}.srt")),
        &transcript.as_srt(),
    );

    println!("time: {:?}", transcript.processing_time);
}
