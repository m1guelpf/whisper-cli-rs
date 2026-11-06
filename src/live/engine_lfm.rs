use crate::{
    live::{
        audio::{
            compute_peak, compute_rms, encode_pcm16_wav_bytes, find_best_split_index,
            whisper_macos_audio_bytes_received, whisper_macos_audio_packets_received,
            whisper_macos_speech_permille, MacOsAudioStream, SAMPLE_RATE,
        },
        config::LiveConfig,
        hud::render_live_hud,
        ramp::{AdaptiveRampState, AudioSegment},
        sub_server::start_sub_server,
        text_filter::{is_hallucination, update_prompt_context},
        vad::SpeechDetector,
    },
    transcript::{Transcript, Utternace},
    utils::{base64_encode, format_timestamp, prompt_auto_download},
};
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use std::{
    collections::VecDeque,
    io::Write,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::{sync::mpsc, task::JoinHandle};

pub const DEFAULT_LFM_MODEL: &str = "lfm3-audio";
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

pub struct ManagedOllama {
    child: Option<Child>,
}

impl Drop for ManagedOllama {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn resolve_lfm_model_name(model_arg: Option<&str>) -> String {
    match model_arg {
        Some(m) if !m.trim().is_empty() => m.trim().to_string(),
        _ => DEFAULT_LFM_MODEL.to_string(),
    }
}

pub fn resolve_ollama_url(api_url_arg: Option<&str>) -> String {
    if let Some(u) = api_url_arg {
        let trimmed = u.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Ok(env_host) = std::env::var("OLLAMA_HOST") {
        let trimmed = env_host.trim().trim_end_matches('/');
        if !trimmed.is_empty() {
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                return trimmed.to_string();
            }
            return format!("http://{trimmed}");
        }
    }
    DEFAULT_OLLAMA_URL.to_string()
}

fn model_matches_tag(installed_name: &str, requested: &str) -> bool {
    let inst = installed_name.trim().to_ascii_lowercase();
    let req = requested.trim().to_ascii_lowercase();
    inst == req
        || inst == format!("{req}:latest")
        || inst.starts_with(&format!("{req}:"))
        || req == format!("{inst}:latest")
}

/// Ensures Ollama is running and the requested LFM model (`lfm3-audio`) is pulled locally.
/// If missing, prompts the user interactively (`[Y/n]`) to auto-download it via `ollama pull`.
pub async fn ensure_ollama_and_lfm_model(
    base_url: &str,
    model_name: &str,
    stdout_only: bool,
) -> Result<ManagedOllama> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;
    let tags_url = format!("{base_url}/api/tags");

    let mut spawned_child: Option<Child> = None;

    // 1. Check if Ollama server is already reachable
    if client.get(&tags_url).send().await.is_err() {
        // Check if `ollama` CLI is installed
        let has_ollama = Command::new("ollama")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if !has_ollama {
            if !prompt_auto_download("Ollama CLI (brew install ollama)", "Homebrew")? {
                return Err(anyhow!(
                    "Ollama is required for --lfm ({model_name}). Install it with `brew install ollama`."
                ));
            }
            let status = Command::new("brew")
                .args(["install", "ollama"])
                .status()
                .map_err(|e| anyhow!("Failed to run `brew install ollama`: {e}"))?;
            if !status.success() {
                return Err(anyhow!("`brew install ollama` exited with status: {status}"));
            }
        }

        if !stdout_only {
            eprintln!("🚀 Starting background `ollama serve` on {base_url}...");
        }
        let child = Command::new("ollama")
            .arg("serve")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("Failed to start `ollama serve`: {e}"))?;
        spawned_child = Some(child);

        let start = Instant::now();
        let mut ready = false;
        while start.elapsed() < Duration::from_secs(8) {
            if client.get(&tags_url).send().await.is_ok() {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
        if !ready {
            return Err(anyhow!(
                "Timed out waiting for `ollama serve` to listen at {base_url}"
            ));
        }
    }

    // 2. Check if `model_name` exists in `/api/tags`
    let tags_resp: serde_json::Value = client
        .get(&tags_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let is_installed = tags_resp["models"]
        .as_array()
        .map(|arr| {
            arr.iter().any(|m| {
                let name = m["name"].as_str().unwrap_or_default();
                let model = m["model"].as_str().unwrap_or_default();
                model_matches_tag(name, model_name) || model_matches_tag(model, model_name)
            })
        })
        .unwrap_or(false);

    if !is_installed {
        let display = if model_name.starts_with("lfm3-audio") {
            format!("{model_name} (LiquidAI LFM3-Audio-2B, ~2.0 GB nvfp4)")
        } else {
            model_name.to_string()
        };
        if !prompt_auto_download(&display, "Ollama local registry")? {
            return Err(anyhow!(
                "Model '{model_name}' is not installed in Ollama. Run `ollama pull {model_name}` to download it."
            ));
        }

        eprintln!("📥 Running `ollama pull {model_name}`...");
        let status = Command::new("ollama")
            .args(["pull", model_name])
            .status()
            .map_err(|e| anyhow!("Failed to execute `ollama pull {model_name}`: {e}"))?;
        if !status.success() {
            return Err(anyhow!(
                "`ollama pull {model_name}` failed with exit status: {status}"
            ));
        }
        eprintln!("✅ Model '{model_name}' downloaded successfully!\n");
    }

    // 3. Pre-warm model in Metal GPU memory if not already loaded in `/api/ps`
    let ps_url = format!("{base_url}/api/ps");
    let is_loaded = if let Ok(resp) = client.get(&ps_url).send().await {
        if let Ok(v) = resp.json::<serde_json::Value>().await {
            v["models"]
                .as_array()
                .map(|arr| {
                    arr.iter().any(|m| {
                        let name = m["name"].as_str().unwrap_or_default();
                        model_matches_tag(name, model_name)
                    })
                })
                .unwrap_or(false)
        } else {
            false
        }
    } else {
        false
    };

    if !is_loaded {
        if !stdout_only {
            eprintln!("🔥 Warming up '{model_name}' in Metal GPU memory...");
        }
        let silent_wav = encode_pcm16_wav_bytes(&vec![0.0_f32; 1600], SAMPLE_RATE as u32);
        let b64 = base64_encode(&silent_wav);
        let warmup_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(45))
            .build()?;
        let _ = warmup_client
            .post(format!("{base_url}/api/chat"))
            .json(&serde_json::json!({
                "model": model_name,
                "messages": [{
                    "role": "user",
                    "content": "ok",
                    "images": [b64]
                }],
                "stream": false,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1
                }
            }))
            .send()
            .await;
    }

    Ok(ManagedOllama {
        child: spawned_child,
    })
}

pub fn language_display_name(lang_code: Option<&str>) -> Option<&'static str> {
    let code = lang_code?.trim().to_ascii_lowercase();
    match code.as_str() {
        "es" | "spa" | "spanish" => Some("Spanish"),
        "ja" | "jpn" | "japanese" => Some("Japanese"),
        "en" | "eng" | "english" => Some("English"),
        "fr" | "fra" | "french" => Some("French"),
        "de" | "deu" | "german" => Some("German"),
        "pt" | "por" | "portuguese" => Some("Portuguese"),
        "it" | "ita" | "italian" => Some("Italian"),
        "zh" | "zho" | "chinese" => Some("Chinese"),
        "ko" | "kor" | "korean" => Some("Korean"),
        "ru" | "rus" | "russian" => Some("Russian"),
        "ar" | "ara" | "arabic" => Some("Arabic"),
        _ => None,
    }
}

pub fn build_lfm_messages(
    lang: Option<&str>,
    translate: bool,
    audio_b64: String,
) -> Vec<serde_json::Value> {
    let lang_name = language_display_name(lang);

    let prompt = if translate {
        if let Some(l) = lang_name {
            format!("Translate the spoken {l} audio directly to English. Output only the English translation.")
        } else {
            "Translate the spoken audio directly to English. Output only the English translation.".to_string()
        }
    } else if let Some(l) = lang_name {
        format!("Transcribe the spoken {l} audio verbatim. Output only the exact {l} words spoken.")
    } else {
        "Transcribe the spoken audio verbatim. Output only the words spoken.".to_string()
    };

    vec![
        serde_json::json!({
            "role": "user",
            "content": prompt,
            "images": [audio_b64],
        }),
    ]
}

pub fn clean_lfm_text(raw: &str) -> String {
    let mut s = raw.trim();
    if s.is_empty() {
        return String::new();
    }

    // Strip common LLM prefixes if present
    for prefix in [
        "Transcription:",
        "Transcript:",
        "Transcripción:",
        "Translation:",
        "Traducción:",
        "Subtitle:",
        "Subtítulo:",
        "Audio:",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.trim();
        }
    }

    // Strip wrapping quotes if the entire line is quoted
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
        || (s.starts_with('«') && s.ends_with('»') && s.len() >= 4)
    {
        let mut chars = s.chars();
        chars.next();
        chars.next_back();
        s = chars.as_str().trim();
    }

    let lower = s.to_ascii_lowercase();

    // Check if the LLM echoed its own system prompt or instructions
    let prompt_echoes = [
        "transcribe the",
        "transcribe audio",
        "transcribir",
        "output only",
        "verbatim",
        "spoken words",
        "translate the",
        "english translation",
    ];
    if prompt_echoes.iter().any(|p| lower.contains(p)) {
        return String::new();
    }

    let refusal_phrases = [
        "no speech",
        "no intelligible",
        "no hay voz",
        "no hay habla",
        "no se detecta voz",
        "no se escucha",
        "sin voz",
        "sin audio",
        "cannot hear any speech",
        "i cannot transcribe",
        "there is no speech",
        "only music",
        "background music",
        "sound effects",
    ];
    if refusal_phrases.iter().any(|p| lower.contains(p)) {
        return String::new();
    }

    if is_hallucination(s) {
        return String::new();
    }

    s.replace('\n', " ").trim().to_string()
}

#[cfg(target_os = "macos")]
fn print_lfm_banner(config: &LiveConfig, model_name: &str, base_url: &str) {
    if config.stdout_only {
        return;
    }
    let lang_str = config.lang.as_deref().unwrap_or("auto-detect");
    let mode_str = if config.translate {
        format!("{lang_str} -> English (translate)")
    } else {
        lang_str.to_string()
    };
    eprintln!("╭──────────────────────────────────────────────────────────────╮");
    eprintln!("│ 🟣 Liquid AI LFM3-Audio-2B Live Streaming (Ollama Metal)     │");
    eprintln!("╰──────────────────────────────────────────────────────────────╯");
    eprintln!("  • Engine   : LiquidAI/LFM3-Audio-2B (Causal FastConformer + LFM3)");
    eprintln!("  • Model    : {model_name} via Ollama ({base_url})");
    eprintln!("  • Latency  : ~80ms Audio Encode • ~115ms Token Streaming TTFT");
    eprintln!("  • Audio    : ScreenCaptureKit (System Audio, No Mic/BlackHole)");
    if config.sub_port > 0 {
        eprintln!(
            "  • Subtitles: http://127.0.0.1:{}/events (Token-by-Token SSE)",
            config.sub_port
        );
    }
    eprintln!("  • Language : {mode_str} (Press Ctrl+C to stop)\n");
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn spawn_lfm_streaming_worker(
    base_url: String,
    model_name: String,
    lang: Option<String>,
    translate: bool,
    stdout_only: bool,
    sub_broadcast: tokio::sync::broadcast::Sender<String>,
    sub_history: Arc<Mutex<VecDeque<serde_json::Value>>>,
    live_preview: Arc<Mutex<Option<String>>>,
    session_start: Instant,
) -> (
    mpsc::Sender<AudioSegment>,
    Arc<AtomicBool>,
    JoinHandle<Vec<Utternace>>,
) {
    let (segment_tx, mut segment_rx) = mpsc::channel::<AudioSegment>(8);
    let is_transcribing = Arc::new(AtomicBool::new(false));
    let is_transcribing_worker = Arc::clone(&is_transcribing);

    let handle = tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let chat_url = format!("{base_url}/api/chat");

        let mut all_utterances: Vec<Utternace> = Vec::new();
        let mut last_printed = String::new();
        let mut seg_counter: u64 = 0;

        while let Some(seg) = segment_rx.recv().await {
            // Require at least 0.9s of audio for LLM audio processing to prevent micro-chunk hallucinations
            if seg.samples.len() < (0.90 * SAMPLE_RATE as f32) as usize {
                continue;
            }

            is_transcribing_worker.store(true, Ordering::Relaxed);
            seg_counter += 1;
            let cue_id = format!("lfm-{seg_counter}");

            let wav_bytes = encode_pcm16_wav_bytes(&seg.samples, SAMPLE_RATE as u32);
            let audio_b64 = base64_encode(&wav_bytes);
            let messages = build_lfm_messages(lang.as_deref(), translate, audio_b64);

            let payload = serde_json::json!({
                "model": &model_name,
                "messages": messages,
                "stream": true,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 120,
                }
            });

            let ts = format_timestamp(seg.start_offset_ms / 10, false, ".");
            let short_ts = ts.split('.').next().unwrap_or(&ts).to_string();
            let audio_duration_ms = (seg.samples.len() * 1000 / SAMPLE_RATE) as u64;
            let duration_ms = ((seg.end_offset_ms - seg.start_offset_ms).max(120) as u64)
                .min(audio_duration_ms.max(120));

            let req_start = Instant::now();
            let mut ttft_ms: Option<u64> = None;
            let mut accumulated = String::new();
            let mut line_buf = String::new();

            match client.post(&chat_url).json(&payload).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let mut stream = resp.bytes_stream();
                    while let Some(Ok(chunk)) = stream.next().await {
                        if let Ok(text_chunk) = std::str::from_utf8(&chunk) {
                            line_buf.push_str(text_chunk);
                            while let Some(nl_idx) = line_buf.find('\n') {
                                let line: String = line_buf.drain(..=nl_idx).collect();
                                let line_trimmed = line.trim();
                                if line_trimmed.is_empty() {
                                    continue;
                                }
                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(line_trimmed) {
                                    if let Some(tok) = v["message"]["content"].as_str() {
                                        if !tok.is_empty() {
                                            if ttft_ms.is_none() {
                                                ttft_ms = Some(req_start.elapsed().as_millis() as u64);
                                            }
                                            accumulated.push_str(tok);
                                            let partial = clean_lfm_text(&accumulated);
                                            if !partial.is_empty() {
                                                if let Ok(mut prev) = live_preview.lock() {
                                                    *prev = Some(format!("[{short_ts}] {partial}"));
                                                }
                                                let now_ms = session_start.elapsed().as_millis() as i64;
                                                let lag_ms = (now_ms - seg.end_offset_ms).max(0) as u64;
                                                let partial_cue = serde_json::json!({
                                                    "id": &cue_id,
                                                    "text": &partial,
                                                    "timestamp": &short_ts,
                                                    "receivedAt": std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap_or_default()
                                                        .as_millis() as u64,
                                                    "clipStartMs": seg.start_offset_ms,
                                                    "clipEndMs": seg.end_offset_ms,
                                                    "durationMs": duration_ms,
                                                    "lagMs": lag_ms,
                                                    "isFinal": false,
                                                });
                                                let _ = sub_broadcast.send(partial_cue.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if let Ok(mut prev) = live_preview.lock() {
                        *prev = None;
                    }

                    let final_text = clean_lfm_text(&accumulated);
                    if !final_text.is_empty() && final_text != last_printed {
                        let total_ms = req_start.elapsed().as_millis() as u64;
                        let first_tok = ttft_ms.unwrap_or(total_ms);

                        if stdout_only {
                            println!("{final_text}");
                            let _ = std::io::stdout().flush();
                        } else {
                            eprint!("\r\x1b[2K");
                            println!(
                                "\x1b[35m🟣 [{short_ts}]\x1b[0m  \x1b[1m{final_text}\x1b[0m \x1b[2m({first_tok}ms TTFT)\x1b[0m"
                            );
                            let _ = std::io::stdout().flush();
                        }

                        let now_ms = session_start.elapsed().as_millis() as i64;
                        let lag_ms = (now_ms - seg.end_offset_ms).max(0) as u64;
                        let final_cue = serde_json::json!({
                            "id": &cue_id,
                            "text": &final_text,
                            "timestamp": &short_ts,
                            "receivedAt": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64,
                            "clipStartMs": seg.start_offset_ms,
                            "clipEndMs": seg.end_offset_ms,
                            "durationMs": duration_ms,
                            "lagMs": lag_ms,
                            "isFinal": true,
                        });
                        {
                            let mut hist = sub_history.lock().unwrap();
                            if hist.len() >= 25 {
                                hist.pop_front();
                            }
                            hist.push_back(final_cue.clone());
                        }
                        let _ = sub_broadcast.send(final_cue.to_string());
                        last_printed = final_text.clone();
                        all_utterances.push(Utternace {
                            start: seg.start_offset_ms / 10,
                            stop: seg.end_offset_ms / 10,
                            text: final_text,
                        });
                    }
                }
                Ok(resp) => {
                    if !stdout_only {
                        eprint!("\r\x1b[2K");
                        eprintln!(
                            "\x1b[33m⚠️  Ollama returned HTTP {}: check `ollama list`\x1b[0m",
                            resp.status()
                        );
                    }
                }
                Err(err) => {
                    if !stdout_only {
                        eprint!("\r\x1b[2K");
                        eprintln!("\x1b[33m⚠️  Ollama LFM error: {err}\x1b[0m");
                    }
                }
            }

            is_transcribing_worker.store(false, Ordering::Relaxed);
        }

        all_utterances
    });

    (segment_tx, is_transcribing, handle)
}

#[cfg(not(target_os = "macos"))]
pub async fn run_live_lfm(_config: LiveConfig) -> Result<Transcript> {
    Err(anyhow!(
        "--lfm live system audio capture is only supported on macOS (via native ScreenCaptureKit)."
    ))
}

#[cfg(target_os = "macos")]
pub async fn run_live_lfm(config: LiveConfig) -> Result<Transcript> {
    let model_name = resolve_lfm_model_name(config.model.as_deref());
    let base_url = resolve_ollama_url(config.api_url.as_deref());

    let _managed_ollama =
        ensure_ollama_and_lfm_model(&base_url, &model_name, config.stdout_only).await?;

    let (_stream, mut audio_rx) = MacOsAudioStream::start()?;
    let session_start = Instant::now();

    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(Mutex::new(VecDeque::<serde_json::Value>::with_capacity(32)));
    if config.sub_port > 0 {
        start_sub_server(
            config.sub_port,
            sub_broadcast_tx.clone(),
            Arc::clone(&sub_history),
        )
        .await;
    }

    print_lfm_banner(&config, &model_name, &base_url);

    let live_preview = Arc::new(Mutex::new(None::<String>));
    let (segment_tx, is_transcribing_ui, worker_handle) = spawn_lfm_streaming_worker(
        base_url,
        model_name,
        config.lang.clone(),
        config.translate,
        config.stdout_only,
        sub_broadcast_tx.clone(),
        Arc::clone(&sub_history),
        Arc::clone(&live_preview),
        session_start,
    );

    // Use natural dialogue chunking for LFM3-Audio (1.5s - 3.0s sentences with natural pause cuts)
    let is_realtime = config.preset == Some(crate::server::ServerPreset::Realtime);
    let mut ramp = AdaptiveRampState::new(is_realtime, config.speech_mode, config.chunk_secs);
    let pre_roll_capacity = (0.25 * SAMPLE_RATE as f32) as usize;
    let overlap_samples = (0.14 * SAMPLE_RATE as f32) as usize;
    let vad_threshold: f32 = 0.0045;

    let stdout_only = config.stdout_only;
    let speech_mode = config.speech_mode;

    let mut pre_roll: VecDeque<f32> = VecDeque::with_capacity(pre_roll_capacity + 2048);
    let max_buffer_capacity = (15.0 * SAMPLE_RATE as f32) as usize;
    let mut speech_buffer: Vec<f32> = Vec::with_capacity(max_buffer_capacity + 4096);
    let mut spark_history: VecDeque<f32> = VecDeque::from(vec![0.0; 16]);
    let mut speech_detector = SpeechDetector::new();
    let mut speaking = false;
    let mut silence_samples: usize = 0;
    let mut active_speech_samples: usize = 0;
    let mut segment_start_ms: i64 = 0;
    let mut current_rms: f32 = 0.0;
    let mut current_speech_prob: f32 = 0.0;
    let mut recent_peak: f32 = 0.0;
    let mut tick: usize = 0;
    let mut last_packet_instant = Instant::now();
    let mut last_speech_instant = Instant::now();
    let mut ui_interval = tokio::time::interval(Duration::from_millis(60));

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                if !stdout_only {
                    eprint!("\r\x1b[2K");
                    eprintln!("\n👋 Stopping LFM3-Audio-2B live stream...");
                }
                break;
            }
            _ = ui_interval.tick() => {
                let quiet_elapsed = last_packet_instant.elapsed();
                if quiet_elapsed >= Duration::from_millis(120) {
                    current_rms *= 0.65;
                    recent_peak *= 0.82;
                    current_speech_prob *= 0.75;
                    if current_rms < 0.001 {
                        current_rms = 0.0;
                    }
                    if recent_peak < 0.005 {
                        recent_peak = 0.0;
                    }
                    spark_history.push_back(current_rms);
                    if spark_history.len() > 16 {
                        spark_history.pop_front();
                    }
                    if !speaking && quiet_elapsed >= Duration::from_millis(450) {
                        pre_roll.clear();
                    }
                }

                if !speaking {
                    ramp.check_silence_reset();
                }

                if speaking {
                    let (
                        _max_chunk_samples,
                        min_pause_split_samples,
                        pause_silence_samples,
                        soft_long_sentence_samples,
                        breath_pause_samples,
                        short_phrase_min_samples,
                        long_silence_samples,
                        _is_cold,
                    ) = ramp.current_thresholds();

                    let elapsed_silence_samples =
                        (last_speech_instant.elapsed().as_secs_f32() * SAMPLE_RATE as f32) as usize;
                    let effective_silence = silence_samples.max(elapsed_silence_samples);

                    let natural_pause_ready = speech_buffer.len() >= min_pause_split_samples
                        && effective_silence >= pause_silence_samples;
                    let long_sentence_breath_ready = speech_mode
                        && speech_buffer.len() >= soft_long_sentence_samples
                        && effective_silence >= breath_pause_samples;
                    let short_phrase_ready = active_speech_samples >= short_phrase_min_samples
                        && effective_silence >= long_silence_samples;
                    let dropped_noise = active_speech_samples < short_phrase_min_samples
                        && effective_silence >= long_silence_samples;

                    if dropped_noise {
                        speech_buffer.clear();
                        speaking = false;
                        silence_samples = 0;
                        active_speech_samples = 0;
                    } else if natural_pause_ready || long_sentence_breath_ready || short_phrase_ready {
                        let chunk = std::mem::take(&mut speech_buffer);
                        speaking = false;
                        silence_samples = 0;
                        active_speech_samples = 0;

                        if compute_rms(&chunk) >= 0.005 || compute_peak(&chunk) >= 0.015 {
                            let now_ms = session_start.elapsed().as_millis() as i64;
                            let silence_lag_ms = last_speech_instant.elapsed().as_millis() as i64;
                            let speech_end_ms = (now_ms - silence_lag_ms).max(segment_start_ms + 120);
                            let was_cold = ramp.record_emission();
                            let _ = segment_tx.try_send(AudioSegment {
                                samples: chunk,
                                start_offset_ms: segment_start_ms,
                                end_offset_ms: speech_end_ms,
                                is_cold_start: was_cold,
                            });
                        }
                    }
                }

                if !stdout_only {
                    tick = tick.wrapping_add(1);
                    let bytes = unsafe { whisper_macos_audio_bytes_received() };
                    let pkts = unsafe { whisper_macos_audio_packets_received() };
                    let in_flight = is_transcribing_ui.load(Ordering::Relaxed);
                    let speech_secs = speech_buffer.len() as f32 / SAMPLE_RATE as f32;
                    let (curr_max_samples, _, _, _, _, _, _, _) = ramp.current_thresholds();
                    let curr_max_secs = curr_max_samples as f32 / SAMPLE_RATE as f32;

                    let preview_str = live_preview
                        .lock()
                        .ok()
                        .and_then(|g| g.clone());

                    let stage_or_preview = if let Some(p) = preview_str {
                        let max_chars = 34;
                        let count = p.chars().count();
                        if count > max_chars {
                            let tail: String = p.chars().skip(count - max_chars).collect();
                            format!("🟣 ...{tail}")
                        } else {
                            format!("🟣 {p}")
                        }
                    } else {
                        ramp.stage_label().to_string()
                    };

                    let hud = render_live_hud(
                        tick,
                        current_rms,
                        recent_peak,
                        current_speech_prob,
                        &spark_history,
                        speaking,
                        speech_secs,
                        curr_max_secs,
                        in_flight,
                        bytes,
                        pkts,
                        speech_mode,
                        &stage_or_preview,
                    );
                    eprint!("\r\x1b[2K{hud}");
                    let _ = std::io::stderr().flush();
                }
            }
            maybe_samples = audio_rx.recv() => {
                let Some(samples) = maybe_samples else {
                    break;
                };

                let rms = compute_rms(&samples);
                let pkt_peak = compute_peak(&samples);
                if pkt_peak < 0.005 && rms < 0.005 {
                    continue;
                }

                last_packet_instant = Instant::now();
                let ane_conf = unsafe { whisper_macos_speech_permille() } as f32 / 1000.0;
                let (detected_voice, speech_prob, _vocal_rms) =
                    speech_detector.analyze(&samples, ane_conf);

                current_rms = rms;
                current_speech_prob = speech_prob;
                recent_peak = (recent_peak * 0.95).max(pkt_peak);
                spark_history.push_back(rms);
                if spark_history.len() > 16 {
                    spark_history.pop_front();
                }

                let frame_has_speech = if speech_mode {
                    detected_voice
                } else {
                    detected_voice || rms >= vad_threshold || pkt_peak >= 0.025
                };

                let now_ms = session_start.elapsed().as_millis() as i64;

                if !speaking {
                    ramp.check_silence_reset();
                    if frame_has_speech {
                        speaking = true;
                        last_speech_instant = Instant::now();
                        silence_samples = 0;
                        active_speech_samples = samples.len();
                        let pre_roll_ms = (pre_roll.len() * 1000 / SAMPLE_RATE) as i64;
                        segment_start_ms = (now_ms - pre_roll_ms).max(0);
                        speech_buffer.clear();
                        speech_buffer.extend(pre_roll.iter().copied());
                        speech_buffer.extend_from_slice(&samples);
                        pre_roll.clear();
                    } else {
                        pre_roll.extend(samples.iter().copied());
                        while pre_roll.len() > pre_roll_capacity {
                            pre_roll.pop_front();
                        }
                    }
                    continue;
                }

                speech_buffer.extend_from_slice(&samples);
                if frame_has_speech {
                    last_speech_instant = Instant::now();
                    silence_samples = 0;
                    active_speech_samples += samples.len();
                } else {
                    silence_samples += samples.len();
                }

                let (
                    max_chunk_samples,
                    min_pause_split_samples,
                    pause_silence_samples,
                    soft_long_sentence_samples,
                    breath_pause_samples,
                    short_phrase_min_samples,
                    long_silence_samples,
                    _is_cold,
                ) = ramp.current_thresholds();

                let natural_pause_ready = speech_buffer.len() >= min_pause_split_samples
                    && silence_samples >= pause_silence_samples;
                let long_sentence_breath_ready = speech_mode
                    && speech_buffer.len() >= soft_long_sentence_samples
                    && silence_samples >= breath_pause_samples;
                let short_phrase_ready = active_speech_samples >= short_phrase_min_samples
                    && silence_samples >= long_silence_samples;
                let dropped_noise = active_speech_samples < short_phrase_min_samples
                    && silence_samples >= long_silence_samples;

                if dropped_noise {
                    speech_buffer.clear();
                    speaking = false;
                    silence_samples = 0;
                    active_speech_samples = 0;
                } else if natural_pause_ready || long_sentence_breath_ready || short_phrase_ready {
                    let trailing_silence_ms = (silence_samples * 1000 / SAMPLE_RATE) as i64;
                    let speech_end_ms = (now_ms - trailing_silence_ms).max(segment_start_ms + 120);
                    let chunk = std::mem::take(&mut speech_buffer);
                    speaking = false;
                    silence_samples = 0;
                    active_speech_samples = 0;

                    if compute_rms(&chunk) >= 0.005 || compute_peak(&chunk) >= 0.015 {
                        let was_cold = ramp.record_emission();
                        let _ = segment_tx.try_send(AudioSegment {
                            samples: chunk,
                            start_offset_ms: segment_start_ms,
                            end_offset_ms: speech_end_ms,
                            is_cold_start: was_cold,
                        });
                    }
                } else if speech_buffer.len() >= max_chunk_samples {
                    let search_win = (0.50 * SAMPLE_RATE as f32) as usize;
                    let split_at = find_best_split_index(&speech_buffer, search_win);

                    let chunk = speech_buffer[..split_at].to_vec();
                    let keep_from = split_at.saturating_sub(overlap_samples);
                    let remainder = speech_buffer[keep_from..].to_vec();

                    if compute_rms(&chunk) >= 0.005 || compute_peak(&chunk) >= 0.015 {
                        let was_cold = ramp.record_emission();
                        let _ = segment_tx.try_send(AudioSegment {
                            samples: chunk,
                            start_offset_ms: segment_start_ms,
                            end_offset_ms: now_ms,
                            is_cold_start: was_cold,
                        });
                    }

                    speech_buffer = remainder;
                    let rem_ms = (speech_buffer.len() * 1000 / SAMPLE_RATE) as i64;
                    segment_start_ms = (now_ms - rem_ms).max(0);
                    silence_samples = 0;
                    active_speech_samples = speech_buffer.len();
                }
            }
        }
    }

    let exit_min_phrase = (0.20 * SAMPLE_RATE as f32) as usize;
    if speaking
        && active_speech_samples >= exit_min_phrase
        && (compute_rms(&speech_buffer) >= 0.005 || compute_peak(&speech_buffer) >= 0.015)
    {
        let now_ms = session_start.elapsed().as_millis() as i64;
        let _ = segment_tx
            .send(AudioSegment {
                samples: speech_buffer,
                start_offset_ms: segment_start_ms,
                end_offset_ms: now_ms,
                is_cold_start: false,
            })
            .await;
    }
    drop(segment_tx);

    let utterances = worker_handle.await.unwrap_or_default();

    Ok(Transcript {
        processing_time: session_start.elapsed(),
        utterances,
        word_utterances: None,
    })
}
