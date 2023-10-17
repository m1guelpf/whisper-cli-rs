#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use clap::Parser;
use serde::Serialize;
use std::path::Path;
use warp::{self, Filter};
use whisper_cli::{Language, Model, Size, Whisper};

use crate::utils::write_to;

mod utils;

#[derive(Serialize)]
struct TranscriptionResponse {
    text: String,
}

/* 
#[derive(Deserialize)]
struct TranscriptionRequest {
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<f32>,
} 
*/

#[derive(Parser)]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Parser)]
enum SubCommand {
    #[command(about = "Start the transcription server.")]
    Serve {
        /// Port to listen on
        #[clap(short, long, default_value = "3030")]
        port: u16,
    },
    #[command(about = "Transcribe a given audio file.")]
    Transcribe(TranscribeArgs),
}

#[derive(Parser)]
struct TranscribeArgs {
    /// Name of the Whisper model to use
    #[clap(short, long, default_value = "medium")]
    model: Size,

    /// Language spoken in the audio. Attempts to auto-detect by default.
    #[clap(short, long)]
    lang: Option<Language>,

    /// Path to the audio file to transcribe
    #[clap(name = "AUDIO")]
    audio: String,

    /// Toggle translation
    #[clap(short, long, default_value = "false")]
    translate: bool,

    /// Generate timestamps for each word
    #[clap(short, long, default_value = "false")]
    karaoke: bool,

    /// Write transcription results to .txt, .vtt, and .srt files.
    #[clap(short, long, default_value = "false")]
    write: bool,
}

#[tokio::main]
async fn main() {
    for arg in std::env::args() {
        println!("{}", arg);
    }
    let opts = Opts::parse();
    match opts.subcmd {
        SubCommand::Serve { port } => start_server(port).await,
        SubCommand::Transcribe(args) => transcribe_audio(args).await,
    }
}

async fn start_server(port: u16) {
    // Define the API endpoint
    let transcription = warp::post()
        .and(warp::path("v1"))
        .and(warp::path("audio"))
        .and(warp::path("transcriptions"))
        .and(warp::multipart::form())
        .and_then(handle_transcription);

    let routes = transcription;

    warp::serve(routes).run(([0, 0, 0, 0], port)).await;
}

async fn handle_transcription(
    form: warp::multipart::FormData,
) -> Result<impl warp::Reply, warp::Rejection> {
    // Handle the uploaded file and parameters from the multipart form.
    // For simplicity, I'm providing a placeholder response here.
    // You'd replace this with the actual transcription logic.
    Ok(warp::reply::json(&format!(
        "{{
          \"text\": \"Transcription placeholder for uploaded audio.\"
      }}"
    )))
}

async fn transcribe_audio(mut args: TranscribeArgs) {
    // Your previous CLI functionality here
    let audio = Path::new(&args.audio);
    let file_name = audio.file_name().unwrap().to_str().unwrap();

    assert!(audio.exists(), "The provided audio file does not exist.");

    if args.model.is_english_only() && (args.lang == Some(Language::Auto) || args.lang.is_none()) {
        args.lang = Some(Language::English);
    }

    assert!(
        !args.model.is_english_only() || args.lang == Some(Language::English),
        "The selected model only supports English."
    );

    let mut whisper = Whisper::new(Model::new(args.model), args.lang).await;
    let transcript = whisper
        .transcribe(audio, args.translate, args.karaoke)
        .unwrap();
    println!("time: {:?}", transcript.processing_time);

    if args.write {
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
    } else {
        println!("");
        println!("🔊 {}", transcript.as_text());
    }
}
