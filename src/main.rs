#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use crate::{
    model::{Model, Size},
    whisper::Whisper,
};
use clap::Parser;
use std::path::Path;
use utils::write_to;
use whisper::Language;

mod ffmpeg_decoder;
mod model;
mod transcript;
mod utils;
mod whisper;

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
    #[clap(short, long, default_value = "auto")]
    lang: Language,

    /// Path to the audio file to transcribe
    audio: String,
}

#[tokio::main]
async fn main() {
    let mut args = Args::parse();
    let audio = Path::new(&args.audio);
    let file_name = audio.file_name().unwrap().to_str().unwrap();

    if !audio.exists() {
        panic!("The provided audio file does not exist.");
    }

    if args.model.is_english_only() && args.lang == Language::Auto {
        args.lang = Language::English;
    }

    if args.model.is_english_only() && args.lang != Language::English {
        panic!("The selected model only supports English.");
    }

    let mut whisper = Whisper::new(Model::new(args.model), args.lang).await;
    let transcript = whisper.transcribe(audio).unwrap();

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
