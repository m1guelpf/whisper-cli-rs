#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use std::{env, path::Path, sync::Arc};
use utils::write_to;

mod utils;

use whisper_cli::{
    live::{run_audio_test, run_live_subtitles, LiveConfig},
    server::{resolve_model_path, run_server_foreground, ManagedServer, ServerConfig, ServerPreset},
    CustomApiProvider, DeepgramProvider, GroqProvider, OpenAiProvider, Transcriber,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ProviderKind {
    #[clap(name = "groq")]
    Groq,
    #[clap(name = "openai")]
    OpenAi,
    #[clap(name = "deepgram")]
    Deepgram,
    #[clap(name = "custom")]
    Custom,
}

#[derive(Parser)]
#[command(
    author,
    version,
    about = "Transcribe audio files or live macOS system audio using modern Speech-to-Text APIs.",
    long_about = "Generate a transcript of an audio file, launch a local whisper-server with tuned presets, or stream real-time subtitles from macOS system audio using Groq, OpenAI, Deepgram, or a custom OpenAI-compatible API endpoint (like whisper-server)."
)]
struct Args {
    /// Speech-to-Text provider to use (groq, openai, deepgram, custom)
    #[clap(short, long, value_enum, default_value = "groq")]
    provider: ProviderKind,

    /// Name or path of the model to use (e.g. large-v3-turbo, medium, whisper-1, nova-3)
    #[clap(short, long)]
    model: Option<String>,

    /// Language spoken in the audio (ISO 639-1 code e.g. 'en', 'es', 'ja'). Auto-detected by default.
    #[clap(short = 'l', long)]
    lang: Option<String>,

    /// Path to the audio file to transcribe (optional when using --live, --serve, or --test-audio)
    #[clap(required_unless_present_any = ["live", "test_audio", "serve"])]
    audio: Option<String>,

    /// Capture macOS system audio in real time and stream subtitles to the terminal
    #[clap(long, default_value = "false")]
    live: bool,

    /// Wait for the speaker to finish their sentence before sending (uses Apple Neural Engine SoundAnalysis + vocal-band detection to ignore SFX/music)
    #[clap(long, default_value = "false")]
    speech: bool,

    /// Launch local whisper-server in the foreground using a preset (balanced, live, fast, accurate)
    #[clap(long, value_enum, num_args = 0..=1, default_missing_value = "balanced")]
    serve: Option<ServerPreset>,

    /// Server preset to use (auto-starts background whisper-server when used with --live)
    #[clap(long, value_enum)]
    preset: Option<ServerPreset>,

    /// Port for local whisper-server (used with --serve and --preset)
    #[clap(long, default_value = "8080")]
    port: u16,

    /// Extra raw flags to append to whisper-server (e.g. --server-args "-nth 0.6 -ac 256")
    #[clap(long)]
    server_args: Option<String>,

    /// Test macOS system audio capture with a live waveform & byte monitor (no server or API key needed)
    #[clap(long, default_value = "false")]
    test_audio: bool,

    /// Maximum audio chunk duration in seconds for --live mode
    #[clap(long)]
    chunk_secs: Option<f32>,

    /// Port for broadcasting live subtitles over HTTP/SSE to browser extensions (used with --live)
    #[clap(long, default_value = "8989")]
    sub_port: u16,

    /// API Key for the chosen provider (defaults to GROQ_API_KEY, OPENAI_API_KEY, or DEEPGRAM_API_KEY env var)
    #[clap(short = 'k', long)]
    api_key: Option<String>,

    /// Custom API Base URL for local/self-hosted OpenAI-compatible speech-to-text endpoints (defaults to http://localhost:8080 for custom provider)
    #[clap(short = 'u', long)]
    api_url: Option<String>,

    /// Toggle translation to English
    #[clap(short, long, default_value = "false")]
    translate: bool,

    /// Generate timestamps for each word
    #[clap(long, default_value = "false")]
    karaoke: bool,

    /// Print transcript directly to terminal stdout
    #[clap(short = 's', long, default_value = "false")]
    stdout: bool,

    /// Skip writing .txt, .vtt, and .srt files to disk
    #[clap(long, default_value = "false")]
    no_files: bool,
}

fn effective_provider(args: &Args) -> ProviderKind {
    if args.preset.is_some() {
        ProviderKind::Custom
    } else {
        args.provider
    }
}

fn effective_api_url(args: &Args) -> String {
    args.api_url
        .clone()
        .unwrap_or_else(|| format!("http://localhost:{}", args.port))
}

fn provider_label(args: &Args) -> String {
    match effective_provider(args) {
        ProviderKind::Groq => format!(
            "Groq ({})",
            args.model.as_deref().unwrap_or("whisper-large-v3-turbo")
        ),
        ProviderKind::OpenAi => {
            format!("OpenAI ({})", args.model.as_deref().unwrap_or("whisper-1"))
        }
        ProviderKind::Deepgram => {
            format!("Deepgram ({})", args.model.as_deref().unwrap_or("nova-3"))
        }
        ProviderKind::Custom => {
            let url = effective_api_url(args);
            if let Some(preset) = args.preset {
                format!("Custom ({url} • {} preset)", preset.as_str())
            } else {
                format!("Custom ({url})")
            }
        }
    }
}

fn create_transcriber(args: &Args) -> Result<Box<dyn Transcriber>> {
    let provider = effective_provider(args);
    let key_from_env = match provider {
        ProviderKind::Groq => env::var("GROQ_API_KEY").ok(),
        ProviderKind::OpenAi => env::var("OPENAI_API_KEY").ok(),
        ProviderKind::Deepgram => env::var("DEEPGRAM_API_KEY").ok(),
        ProviderKind::Custom => env::var("CUSTOM_API_KEY").ok(),
    };

    let api_key = args.api_key.clone().or(key_from_env);

    match provider {
        ProviderKind::Groq => {
            let key = api_key.ok_or_else(|| {
                anyhow!("Groq API key required. Provide --api-key or set GROQ_API_KEY env var.")
            })?;
            Ok(Box::new(GroqProvider::new(
                key,
                args.model.clone(),
                args.lang.clone(),
            )))
        }
        ProviderKind::OpenAi => {
            let key = api_key.ok_or_else(|| {
                anyhow!("OpenAI API key required. Provide --api-key or set OPENAI_API_KEY env var.")
            })?;
            Ok(Box::new(OpenAiProvider::new(
                key,
                args.model.clone(),
                args.lang.clone(),
            )))
        }
        ProviderKind::Deepgram => {
            let key = api_key.ok_or_else(|| {
                anyhow!("Deepgram API key required. Provide --api-key or set DEEPGRAM_API_KEY env var.")
            })?;
            Ok(Box::new(DeepgramProvider::new(
                key,
                args.model.clone(),
                args.lang.clone(),
            )))
        }
        ProviderKind::Custom => {
            let url = effective_api_url(args);
            Ok(Box::new(CustomApiProvider::new(
                url,
                api_key,
                args.model.clone(),
                args.lang.clone(),
            )))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.test_audio {
        return run_audio_test().await;
    }

    if let Some(serve_preset) = args.serve {
        let preset = args.preset.unwrap_or(serve_preset);
        let cfg = ServerConfig {
            preset,
            model: args.model.clone(),
            lang: args.lang.clone(),
            port: args.port,
            extra_args: args.server_args.clone(),
            speech_mode: args.speech,
        };
        return run_server_foreground(&cfg);
    }

    // If --preset was passed (e.g. with --live or file transcription), ensure whisper-server is running
    let _managed_server = if let Some(preset) = args.preset {
        let cfg = ServerConfig {
            preset,
            model: args.model.clone(),
            lang: args.lang.clone(),
            port: args.port,
            extra_args: args.server_args.clone(),
            speech_mode: args.speech,
        };
        Some(ManagedServer::start(&cfg).await?)
    } else {
        None
    };

    let transcriber = create_transcriber(&args)?;

    if args.live {
        let chunk_secs = args.chunk_secs.unwrap_or_else(|| {
            args.preset
                .map_or(4.0, ServerPreset::recommended_chunk_secs)
        });
        let model_label = resolve_model_path(args.model.as_deref())
            .ok()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().to_string()))
            .or_else(|| args.model.clone());
        let config = LiveConfig {
            provider_label: provider_label(&args),
            model: model_label,
            lang: args.lang.clone(),
            translate: args.translate,
            karaoke: args.karaoke,
            stdout_only: args.stdout,
            chunk_secs,
            speech_mode: args.speech,
            sub_port: args.sub_port,
        };
        let transcript = run_live_subtitles(Arc::new(transcriber), config).await?;

        if let Some(output_arg) = &args.audio {
            if !args.no_files && !args.stdout && !transcript.utterances.is_empty() {
                let out_path = Path::new(output_arg);
                let file_name = out_path
                    .file_name()
                    .ok_or_else(|| anyhow!("Invalid output path"))?
                    .to_str()
                    .ok_or_else(|| anyhow!("Non-UTF8 filename"))?;

                write_to(
                    out_path.with_file_name(format!("{file_name}.txt")),
                    &transcript.as_text(),
                );
                write_to(
                    out_path.with_file_name(format!("{file_name}.vtt")),
                    &transcript.as_vtt(),
                );
                write_to(
                    out_path.with_file_name(format!("{file_name}.srt")),
                    &transcript.as_srt(),
                );
                eprintln!("Saved live session transcript to {file_name}.{{txt,vtt,srt}}");
            }
        }
        return Ok(());
    }

    let audio_str = args
        .audio
        .as_deref()
        .ok_or_else(|| anyhow!("Audio file path is required unless --live or --serve is specified."))?;
    let audio = Path::new(audio_str);
    let file_name = audio
        .file_name()
        .ok_or_else(|| anyhow!("Invalid audio path"))?
        .to_str()
        .ok_or_else(|| anyhow!("Non-UTF8 filename"))?;

    if !audio.exists() {
        return Err(anyhow!(
            "The provided audio file '{}' does not exist.",
            audio_str
        ));
    }

    let prepared_audio = whisper_cli::ffmpeg_decoder::prepare_audio(audio)?;
    let transcript = transcriber
        .transcribe(prepared_audio.path(), args.translate, args.karaoke)
        .await?;

    if args.stdout {
        println!("{}", transcript.as_text());
    }

    if !args.no_files && !args.stdout {
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
        println!(
            "Success! Transcription completed in {:?}",
            transcript.processing_time
        );
    }

    Ok(())
}
