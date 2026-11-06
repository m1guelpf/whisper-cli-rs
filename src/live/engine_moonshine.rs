use crate::{
    live::config::LiveConfig,
    transcript::Transcript,
};
use anyhow::{anyhow, Result};

pub const MOONSHINE_POLL_INTERVAL_MS: u64 = 50;
pub const POLL_ON_AUDIO_ARRIVAL: bool = true;

#[cfg(feature = "moonshine")]
use crate::{
    live::{
        audio::{compute_peak, compute_rms, whisper_macos_speech_permille, MacOsAudioStream},
        hud::{format_bytes, SPINNER_FRAMES},
        sub_server::start_sub_server,
        text_filter::is_hallucination,
    },
    transcript::Utternace,
    utils::format_timestamp,
};
#[cfg(feature = "moonshine")]
use std::{
    collections::VecDeque,
    fs::File,
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

#[cfg(feature = "moonshine")]
fn resolve_moonshine_model_info(lang_str: &str) -> (&'static str, moonshine_rs::ModelArch, &'static str) {
    let lang_lower = lang_str.trim().to_ascii_lowercase();
    match lang_lower.as_str() {
        "es" | "spanish" => ("es", moonshine_rs::ModelArch::SmallStreaming, "Small Streaming (4.9% WER)"),
        "en" | "english" => ("en", moonshine_rs::ModelArch::SmallStreaming, "Small Streaming"),
        "ja" | "japanese" => ("ja", moonshine_rs::ModelArch::SmallStreaming, "Small Streaming"),
        "de" | "german" => ("de", moonshine_rs::ModelArch::SmallStreaming, "Small Streaming"),
        "ar" | "arabic" => ("ar", moonshine_rs::ModelArch::TinyStreaming, "Tiny Streaming"),
        "vi" | "vietnamese" => ("vi", moonshine_rs::ModelArch::TinyStreaming, "Tiny Streaming"),
        "zh" | "chinese" => ("zh", moonshine_rs::ModelArch::TinyStreaming, "Tiny Streaming"),
        "ko" | "korean" => ("ko", moonshine_rs::ModelArch::Tiny, "Tiny"),
        "uk" | "ukrainian" => ("uk", moonshine_rs::ModelArch::Base, "Base"),
        _ => ("es", moonshine_rs::ModelArch::SmallStreaming, "Small Streaming"),
    }
}

#[cfg(feature = "moonshine")]
fn ensure_moonshine_assets(
    cache_dir: &Path,
    lang_code: &str,
    arch: moonshine_rs::ModelArch,
    stdout_only: bool,
) -> Result<()> {
    let deps_json = moonshine_rs::get_stt_dependencies(lang_code, Some(arch), false)
        .map_err(|e| anyhow!("Failed to resolve Moonshine model dependencies: {e:?}"))?;
    let v: serde_json::Value = serde_json::from_str(&deps_json)?;
    let files = v["groups"][0]["files"]
        .as_array()
        .ok_or_else(|| anyhow!("Invalid Moonshine manifest structure"))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;

    for f in files {
        let name = f["name"].as_str().unwrap_or_default();
        let url = f["url"].as_str().unwrap_or_default();
        let target = cache_dir.join(name);
        if !target.exists() {
            if !stdout_only {
                eprintln!("  📥 Downloading Moonshine asset: {name} ...");
            }
            let mut resp = client.get(url).send()?.error_for_status()?;
            let mut out = File::create(&target)?;
            resp.copy_to(&mut out)?;
        }
    }
    Ok(())
}

#[cfg(feature = "moonshine")]
fn render_moonshine_hud_line(
    tick: usize,
    current_rms: f32,
    recent_peak: f32,
    audio_bytes: u64,
    current_preview: Option<&str>,
) {
    let spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len()];
    let norm_rms = (current_rms / 0.12).clamp(0.0, 1.0).sqrt();
    let filled = (norm_rms * 8.0).round() as usize;
    let empty = 8_usize.saturating_sub(filled);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

    let preview_hud = if let Some(p) = current_preview {
        let max_chars = 40;
        let char_count = p.chars().count();
        if char_count > max_chars {
            let tail: String = p.chars().skip(char_count - max_chars).collect();
            format!(" │ 🗣️ ...{tail}")
        } else {
            format!(" │ 🗣️ {p}")
        }
    } else {
        String::new()
    };

    let hud_line = format!(
        "{spinner} [{bar}] │ 📥 {} │ Peak: {:.2}{preview_hud}",
        format_bytes(audio_bytes),
        recent_peak,
    );

    eprint!("\r\x1b[2K{hud_line}");
    let _ = std::io::stderr().flush();
}

#[cfg(feature = "moonshine")]
pub async fn run_live_moonshine(config: LiveConfig) -> Result<Transcript> {
    let lang_str = config.lang.as_deref().unwrap_or("es");
    let (lang_code, arch, model_label) = resolve_moonshine_model_info(lang_str);

    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("whisper")
        .join("moonshine")
        .join(lang_code);
    std::fs::create_dir_all(&cache_dir)?;

    if !config.stdout_only {
        eprintln!("╭──────────────────────────────────────────────────────────────╮");
        eprintln!("│ ⚡ Moonshine v2 Speech Recognition (Ultra-Fast Streaming)    │");
        eprintln!("╰──────────────────────────────────────────────────────────────╯");
        eprintln!("  • Engine   : Moonshine v2 (Ergodic Linear Attention STT)");
        eprintln!("  • Model    : {model_label} [{lang_code}]");
        eprintln!("  • Latency  : ~35-50ms (Zero-Lag Edge Streaming)");
        eprintln!("  • Audio    : ScreenCaptureKit (System Audio, No Mic/BlackHole)");
        if config.sub_port > 0 {
            eprintln!(
                "  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)",
                config.sub_port
            );
        }
        eprintln!("  • Language : {lang_code}");
        eprintln!("  • Status   : Checking model assets...\n");
    }

    ensure_moonshine_assets(&cache_dir, lang_code, arch, config.stdout_only)?;

    if !config.stdout_only {
        eprintln!("  ✅ Model assets verified. Starting session...\n");
    }

    let transcriber = Arc::new(
        moonshine_rs::Transcriber::from_files(&cache_dir, arch, None).map_err(|e| {
            anyhow!(
                "Failed to load Moonshine model from {}: {e:?}",
                cache_dir.display()
            )
        })?,
    );
    let mut stream = transcriber
        .create_owned_stream()
        .map_err(|e| anyhow!("Failed to initialize Moonshine streaming session: {e:?}"))?;

    let session_start = Instant::now();
    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(
        VecDeque::<serde_json::Value>::with_capacity(32),
    ));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    #[cfg(target_os = "macos")]
    let (_audio_stream, mut audio_rx) = MacOsAudioStream::start()?;

    #[cfg(not(target_os = "macos"))]
    return Err(anyhow!("ScreenCaptureKit audio streaming is only available on macOS"));

    let mut hud_interval = tokio::time::interval(Duration::from_millis(50));
    let mut tick: usize = 0;
    let mut recent_peak: f32 = 0.0;
    let mut current_rms: f32 = 0.0;
    let mut audio_bytes: u64 = 0;
    let mut current_preview: Option<String> = None;
    let mut last_audio_instant = Instant::now();
    let mut last_poll_instant = Instant::now();
    let mut committed_ids: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    let mut all_utterances: Vec<Utternace> = Vec::new();
    let mut last_preview_text: String = String::new();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        let mut should_poll = false;
        let mut force_poll = false;

        tokio::select! {
            _ = &mut ctrl_c => {
                break;
            }
            maybe_samples = audio_rx.recv() => {
                let Some(samples) = maybe_samples else { break; };
                let rms = compute_rms(&samples);
                let pkt_peak = compute_peak(&samples);

                #[cfg(target_os = "macos")]
                let ane_conf = unsafe { whisper_macos_speech_permille() } as f32 / 1000.0;
                #[cfg(not(target_os = "macos"))]
                let ane_conf = 0.0;

                let has_sound = if config.speech_mode {
                    ane_conf >= 0.20 || pkt_peak >= 0.02
                } else {
                    pkt_peak >= 0.005 || rms >= 0.005
                };

                if has_sound {
                    last_audio_instant = Instant::now();
                }
                current_rms = rms;
                recent_peak = (recent_peak * 0.95).max(pkt_peak);
                audio_bytes += (samples.len() * 4) as u64;

                let _ = stream.add_audio(&samples, 16000);
                if POLL_ON_AUDIO_ARRIVAL && last_poll_instant.elapsed() >= Duration::from_millis(12) {
                    should_poll = true;
                }
            }
            _ = hud_interval.tick() => {
                tick = tick.wrapping_add(1);
                recent_peak *= 0.92;

                let is_quiet = last_audio_instant.elapsed() > Duration::from_millis(500);
                force_poll = is_quiet && current_preview.is_some();
                should_poll = true;
            }
        }

        if should_poll {
            last_poll_instant = Instant::now();
            if let Ok(transcript) = stream.poll(force_poll) {
                for line in &transcript.lines {
                    let text = line.text.trim();
                    if text.is_empty() || is_hallucination(text) {
                        continue;
                    }

                    let start_sec = line.start_time;
                    let duration_sec = line.duration.max(0.25);
                    let start_ms = (start_sec * 1000.0) as i64;
                    let duration_ms = (duration_sec * 1000.0) as u64;
                    let end_ms = start_ms + duration_ms as i64;
                    let now_ms = session_start.elapsed().as_millis() as i64;
                    let lag_ms = (now_ms - end_ms).clamp(20, 2000) as u64;
                    let full_ts = format_timestamp(start_ms / 10, false, ".");
                    let short_ts = if full_ts.len() >= 5 { full_ts[..5].to_string() } else { full_ts };

                    let cue_val = serde_json::json!({
                        "id": format!("ms-{}", line.id),
                        "text": text,
                        "timestamp": short_ts,
                        "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
                        "clipStartMs": start_ms,
                        "clipEndMs": end_ms,
                        "durationMs": duration_ms,
                        "lagMs": lag_ms,
                        "isFinal": line.is_complete,
                    });

                    if line.is_complete {
                        if committed_ids.insert(line.id) {
                            current_preview = None;
                            last_preview_text.clear();

                            {
                                let mut hist = sub_history.lock().unwrap();
                                if hist.len() >= 25 {
                                    hist.pop_front();
                                }
                                hist.push_back(cue_val.clone());
                            }

                            let _ = sub_broadcast_tx.send(cue_val.to_string());

                            all_utterances.push(Utternace {
                                start: start_ms / 10,
                                stop: (start_ms + duration_ms as i64) / 10,
                                text: text.to_string(),
                            });

                            if config.stdout_only {
                                println!("{text}");
                                let _ = std::io::stdout().flush();
                            } else {
                                eprintln!("\r\x1b[2K\x1b[35m⚡ [{short_ts}]\x1b[0m \x1b[1m{text}\x1b[0m");
                                let _ = std::io::stderr().flush();
                            }
                        }
                    } else if !committed_ids.contains(&line.id) {
                        if text != last_preview_text {
                            last_preview_text = text.to_string();
                            current_preview = Some(format!("[{short_ts}] {text}"));
                            let _ = sub_broadcast_tx.send(cue_val.to_string());
                        }
                    }
                }
            }

            if !config.stdout_only {
                render_moonshine_hud_line(
                    tick,
                    current_rms,
                    recent_peak,
                    audio_bytes,
                    current_preview.as_deref(),
                );
            }
        }
    }

    let _ = stream.finalize();
    if !config.stdout_only {
        eprintln!("\n👋 Stopping Moonshine streaming...");
    }

    Ok(Transcript {
        processing_time: session_start.elapsed(),
        utterances: all_utterances,
        word_utterances: None,
    })
}

#[cfg(not(feature = "moonshine"))]
pub async fn run_live_moonshine(_config: LiveConfig) -> Result<Transcript> {
    Err(anyhow!(
        "The Moonshine streaming engine is not compiled into this binary.\n\
         To enable it, build with: cargo build --release --features moonshine"
    ))
}
