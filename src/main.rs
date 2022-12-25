#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use crate::{
    model::{Model, Size},
    whisper::Whisper,
};
use std::path::Path;
use utils::write_to;

mod ffmpeg_decoder;
mod model;
mod transcript;
mod utils;
mod whisper;

#[tokio::main]
async fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    assert!(args.len() == 2, "Usage: rs-whisper <audio_file>");

    let audio = Path::new(&args[1]);
    let file_name = audio.file_name().unwrap().to_str().unwrap();

    let mut whisper = Whisper::new(Model::new(Size::Medium)).await;
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
