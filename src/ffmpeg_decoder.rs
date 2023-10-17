use anyhow::{anyhow, Result};
use audrey::Reader;
use std::env::temp_dir;
use std::path::Path;
use std::process::Stdio;
use std::{fs::File, process::Command};

// ffmpeg -i input.mp3 -ar 16000 output.wav
fn use_ffmpeg<P: AsRef<Path>>(input_path: P) -> Result<Vec<i16>> {
    let temp_file = temp_dir().join(format!("{}.wav", uuid::Uuid::new_v4()));
    let mut pid = Command::new("ffmpeg")
        .args([
            "-i",
            input_path
                .as_ref()
                .to_str()
                .ok_or_else(|| anyhow!("invalid path"))?,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            (temp_file.to_str().unwrap()),
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
        ])
        .stdin(Stdio::null())
        .spawn()?;

    if pid.wait()?.success() {
        let output = File::open(&temp_file)?;
        let mut reader = Reader::new(output)?;
        let samples: Result<Vec<i16>, _> = reader.samples().collect();
        std::fs::remove_file(temp_file)?;
        samples.map_err(std::convert::Into::into)
    } else {
        Err(anyhow!("unable to convert file"))
    }
}

pub fn read_file<P: AsRef<Path>>(audio_file_path: P) -> Result<Vec<f32>> {
    if is_ffmpeg_available() {
        println!("ffmpeg is available");
        let audio_buf = use_ffmpeg(&audio_file_path)?;
        Ok(whisper_rs::convert_integer_to_float_audio(&audio_buf))
    } else {
        println!("ffmpeg not found. Using hound decoder as fallback");
        let mut reader = hound::WavReader::open(audio_file_path)?;
    
        // Convert i16 samples to f32 and normalize to [-1.0, 1.0]
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / std::i16::MAX as f32)
            .collect();
        
        Ok(samples)
    }

}

fn is_ffmpeg_available() -> bool {
    match std::process::Command::new("ffmpeg").arg("-version").status() {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}