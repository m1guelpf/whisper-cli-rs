use crate::{
    live::{
        audio::{
            compute_peak, compute_rms, wisp_applespeech_available, wisp_applespeech_feed,
            wisp_applespeech_start, wisp_applespeech_stop, MacOsAudioStream,
        },
        clock::SampleClockTracker,
        config::LiveConfig,
        hud::{format_bytes, SPINNER_FRAMES},
        sub_server::start_sub_server,
    },
    transcript::{Transcript, Utternace},
    utils::format_timestamp,
};
use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    ffi::{c_char, c_void, CStr},
    io::Write,
    sync::Arc,
    time::{Duration, Instant},
};

pub enum SpeechEvent {
    Partial {
        preview: String,
        cue_val: serde_json::Value,
    },
    Final {
        text: String,
        short_ts: String,
        start_ms: i64,
        duration_ms: u64,
        cue_val: serde_json::Value,
    },
}

pub struct AppleCbState {
    pub event_tx: tokio::sync::mpsc::UnboundedSender<SpeechEvent>,
    pub sub_broadcast_tx: Option<tokio::sync::broadcast::Sender<String>>,
    pub sub_history: Option<Arc<std::sync::Mutex<VecDeque<serde_json::Value>>>>,
    pub last_id: Arc<std::sync::Mutex<String>>,
    pub last_finalized_text: Arc<std::sync::Mutex<String>>,
    pub start_instant: Instant,
    pub clock_tracker: Arc<std::sync::Mutex<SampleClockTracker>>,
    pub all_utterances: Arc<std::sync::Mutex<Vec<Utternace>>>,
}

impl AppleCbState {
    pub fn broadcasts_directly(&self) -> bool {
        self.sub_broadcast_tx.is_some()
    }
}

pub extern "C" fn apple_speech_callback(
    ctx: *mut c_void,
    text_ptr: *const c_char,
    is_final: bool,
    range_start_sec: f64,
    range_dur_sec: f64,
) {
    if ctx.is_null() || text_ptr.is_null() {
        return;
    }
    let c_str = unsafe { CStr::from_ptr(text_ptr) };
    let Ok(text) = c_str.to_str() else { return };
    let text = text.trim();
    if text.is_empty() {
        return;
    }

    let state = unsafe { &*(ctx as *const AppleCbState) };
    {
        let mut last_fin = state.last_finalized_text.lock().unwrap();
        if text == last_fin.as_str() {
            return;
        }
        if is_final {
            *last_fin = text.to_string();
        }
    }

    let now_ms = state.start_instant.elapsed().as_millis() as i64;
    let (start_ms, end_ms) = {
        let mut tracker = state.clock_tracker.lock().unwrap();
        tracker.resolve_clip_bounds(range_start_sec, range_dur_sec, now_ms, is_final)
    };
    let duration_ms = (end_ms - start_ms).max(120) as u64;
    let lag_ms = (now_ms - end_ms).max(0) as u64;
    let ts = format_timestamp(start_ms / 10, false, ".");
    let short_ts = ts.split('.').next().unwrap_or(&ts).to_string();

    let cue_id = {
        let mut id_lock = state.last_id.lock().unwrap();
        if is_final {
            let id = id_lock.clone();
            *id_lock = uuid::Uuid::new_v4().to_string();
            id
        } else {
            id_lock.clone()
        }
    };

    let cue = serde_json::json!({
        "id": cue_id,
        "text": text,
        "timestamp": short_ts,
        "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
        "clipStartMs": start_ms,
        "clipEndMs": end_ms,
        "durationMs": duration_ms,
        "lagMs": lag_ms,
        "isFinal": is_final,
    });

    // Broadcast directly from the FFI callback to avoid waiting for the Tokio HUD/audio select loop
    if is_final {
        if let Some(ref hist_arc) = state.sub_history {
            let mut hist = hist_arc.lock().unwrap();
            if hist.len() >= 25 {
                hist.pop_front();
            }
            hist.push_back(cue.clone());
        }
    }
    if let Some(ref bcast_tx) = state.sub_broadcast_tx {
        let _ = bcast_tx.send(cue.to_string());
    }

    if is_final {
        let _ = state.event_tx.send(SpeechEvent::Final {
            text: text.to_string(),
            short_ts,
            start_ms,
            duration_ms,
            cue_val: cue,
        });
    } else {
        let _ = state.event_tx.send(SpeechEvent::Partial {
            preview: format!("[{short_ts}] {text}"),
            cue_val: cue,
        });
    }
}

pub fn map_locale_identifier(lang_str: &str) -> &'static str {
    let lang_lower = lang_str.trim().to_ascii_lowercase();
    match lang_lower.as_str() {
        "es" => "es-ES",
        "ja" => "ja-JP",
        "zh" | "cmn" => "zh-CN",
        "fr" => "fr-FR",
        "de" => "de-DE",
        "it" => "it-IT",
        "ko" => "ko-KR",
        "pt" => "pt-BR",
        "ru" => "ru-RU",
        other if other.contains('-') => "en-US", // Fallback or standard
        _ => "en-US",
    }
}

fn print_apple_banner(config: &LiveConfig, lang_str: &str) {
    if config.silent {
        if config.sub_port > 0 {
            eprintln!("⚡ [whisper] Headless stream on http://127.0.0.1:{}/events (Ctrl+C to stop)", config.sub_port);
        } else {
            eprintln!("⚡ [whisper] Headless stream active (Ctrl+C to stop)");
        }
        return;
    }
    if !config.stdout_only {
        eprintln!("╭──────────────────────────────────────────────────────────────╮");
        eprintln!("│ ⚡ macOS Apple Speech Recognition (Neural Engine Streaming)  │");
        eprintln!("╰──────────────────────────────────────────────────────────────╯");
        eprintln!("  • Engine   : Apple Speech.framework (Zero-Lag Streaming)");
        eprintln!("  • Audio    : ScreenCaptureKit (System Audio, No Mic/BlackHole)");
        if config.sub_port > 0 {
            eprintln!(
                "  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)",
                config.sub_port
            );
        }
        eprintln!("  • Language : {lang_str}");
        eprintln!("  • Status   : Streaming live... (Press Ctrl+C to stop)\n");
    }
}

fn render_apple_hud_line(
    tick: usize,
    current_rms: f32,
    recent_peak: f32,
    audio_bytes: u64,
    current_preview: Option<&str>,
) {
    let spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len()];
    let normalized = (current_rms / 0.12).clamp(0.0, 1.0).sqrt();
    let filled = (normalized * 6.0).round() as usize;
    let empty = 6_usize.saturating_sub(filled);
    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    let bytes_str = format_bytes(audio_bytes);

    let status = if let Some(text) = current_preview {
        let max_len = 35;
        let count = text.chars().count();
        let truncated = if count > max_len {
            let tail: String = text.chars().skip(count - max_len).collect();
            format!("...{tail}")
        } else {
            text.to_string()
        };
        format!("\x1b[32m🗣️ {truncated}\x1b[0m")
    } else if audio_bytes == 0 {
        "\x1b[2m⚪ Waiting for audio...\x1b[0m".to_string()
    } else if recent_peak >= 0.015 {
        "\x1b[36m⚡ Listening\x1b[0m".to_string()
    } else {
        "\x1b[2m🎙️ Silence\x1b[0m".to_string()
    };

    let hud = format!("{spinner} [{bar}] │ 📥 {bytes_str} │ Peak: {recent_peak:.2} │ {status}");
    eprint!("\r\x1b[2K{hud}");
    let _ = std::io::stderr().flush();
}

#[cfg(target_os = "macos")]
pub async fn run_live_apple_speech(config: LiveConfig) -> Result<Transcript> {
    let session_start = Instant::now();
    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(
        VecDeque::<serde_json::Value>::with_capacity(32),
    ));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    let lang_str = config.lang.as_deref().unwrap_or("en");
    print_apple_banner(&config, lang_str);

    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<SpeechEvent>();
    let all_utterances = Arc::new(std::sync::Mutex::new(Vec::new()));
    let last_id = Arc::new(std::sync::Mutex::new(uuid::Uuid::new_v4().to_string()));
    let last_finalized_text = Arc::new(std::sync::Mutex::new(String::new()));
    let clock_tracker = Arc::new(std::sync::Mutex::new(SampleClockTracker::new()));

    let cb_state = Arc::new(AppleCbState {
        event_tx,
        sub_broadcast_tx: Some(sub_broadcast_tx.clone()),
        sub_history: Some(Arc::clone(&sub_history)),
        last_id: Arc::clone(&last_id),
        last_finalized_text: Arc::clone(&last_finalized_text),
        start_instant: session_start,
        clock_tracker: Arc::clone(&clock_tracker),
        all_utterances: Arc::clone(&all_utterances),
    });

    if !unsafe { wisp_applespeech_available() } {
        return Err(anyhow!(
            "macOS 26+ SpeechAnalyzer is not supported on this macOS version."
        ));
    }

    let locale_str = if lang_str.contains('-') {
        lang_str
    } else {
        map_locale_identifier(lang_str)
    };
    let c_locale = std::ffi::CString::new(locale_str)?;

    let state_ptr = Arc::into_raw(cb_state) as *mut c_void;
    let engine_ptr = unsafe {
        wisp_applespeech_start(c_locale.as_ptr(), state_ptr, apple_speech_callback)
    };

    if engine_ptr.is_null() {
        let _ = unsafe { Arc::from_raw(state_ptr as *const AppleCbState) };
        return Err(anyhow!("Failed to initialize macOS 26 SpeechAnalyzer session"));
    }

    // Start ScreenCaptureKit system audio feed
    let (_audio_stream, mut audio_rx) = MacOsAudioStream::start()?;

    let mut hud_interval = if config.silent {
        tokio::time::interval(std::time::Duration::from_secs(3600))
    } else {
        tokio::time::interval(std::time::Duration::from_millis(60))
    };
    let mut tick: usize = 0;
    let mut recent_peak: f32 = 0.0;
    let mut current_rms: f32 = 0.0;
    let mut audio_bytes: u64 = 0;
    let mut current_preview: Option<String> = None;
    let mut pending_final: Option<serde_json::Value> = None;
    let mut last_audio_instant = Instant::now();
    let silence_chunk = [0.0f32; 960]; // 60ms silence at 16kHz
    let mut trailing_silence_fed = 0usize;

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                break;
            }
            Some(event) = event_rx.recv() => {
                match event {
                    SpeechEvent::Partial { preview, cue_val } => {
                        if !config.silent {
                            current_preview = Some(preview);
                        }
                        pending_final = Some(cue_val);
                    }
                    SpeechEvent::Final { text, short_ts, start_ms, duration_ms, cue_val: _ } => {
                        current_preview = None;
                        pending_final = None;

                        all_utterances.lock().unwrap().push(Utternace {
                            start: start_ms / 10,
                            stop: (start_ms + duration_ms as i64) / 10,
                            text: text.clone(),
                        });

                        if config.silent {
                            // Headless mode: no terminal emission
                        } else if config.stdout_only {
                            println!("{text}");
                            let _ = std::io::stdout().flush();
                        } else {
                            eprintln!("\r\x1b[2K\x1b[36m⚡ [{short_ts}]\x1b[0m \x1b[1m{text}\x1b[0m");
                            let _ = std::io::stderr().flush();
                        }
                    }
                }
            }
            _ = hud_interval.tick() => {
                tick = tick.wrapping_add(1);
                recent_peak *= 0.92;

                if last_audio_instant.elapsed() > Duration::from_millis(80) && trailing_silence_fed < 22 {
                    trailing_silence_fed += 1;
                    let now_ms = session_start.elapsed().as_millis() as i64;
                    clock_tracker.lock().unwrap().record_feed(silence_chunk.len(), now_ms, false);
                    unsafe {
                        wisp_applespeech_feed(engine_ptr, silence_chunk.as_ptr(), silence_chunk.len(), 16000.0);
                    }
                }

                if last_audio_instant.elapsed() > Duration::from_millis(1400) {
                    if let Some(mut cue_val) = pending_final.take() {
                        let now_ms = session_start.elapsed().as_millis() as i64;
                        let cue_text = cue_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let should_finalize = {
                            let mut last_fin = last_finalized_text.lock().unwrap();
                            if !cue_text.is_empty() && cue_text != *last_fin {
                                *last_fin = cue_text.clone();
                                true
                            } else {
                                false
                            }
                        };
                        if should_finalize {
                            let short_ts = cue_val.get("timestamp").and_then(|v| v.as_str()).unwrap_or("00:00").to_string();
                            let start_ms = cue_val.get("clipStartMs").and_then(|v| v.as_i64()).unwrap_or(0);
                            let end_ms = cue_val.get("clipEndMs").and_then(|v| v.as_i64()).unwrap_or(now_ms);
                            let dur = cue_val.get("durationMs").and_then(|v| v.as_u64()).unwrap_or(1000);
                            let lag_ms = (now_ms - end_ms).max(0) as u64;

                            if let Some(obj) = cue_val.as_object_mut() {
                                obj.insert("isFinal".to_string(), serde_json::Value::Bool(true));
                                obj.insert("lagMs".to_string(), serde_json::json!(lag_ms));
                            }

                            clock_tracker.lock().unwrap().mark_utterance_finalized();

                            {
                                let mut hist = sub_history.lock().unwrap();
                                if hist.len() >= 25 {
                                    hist.pop_front();
                                }
                                hist.push_back(cue_val.clone());
                            }
                            let _ = sub_broadcast_tx.send(cue_val.to_string());

                            all_utterances.lock().unwrap().push(Utternace {
                                start: start_ms / 10,
                                stop: (start_ms + dur as i64) / 10,
                                text: cue_text.clone(),
                            });

                            if config.stdout_only {
                                println!("{cue_text}");
                                let _ = std::io::stdout().flush();
                            } else {
                                eprintln!("\r\x1b[2K\x1b[36m⚡ [{short_ts}]\x1b[0m \x1b[1m{cue_text}\x1b[0m");
                                let _ = std::io::stderr().flush();
                            }

                            *last_id.lock().unwrap() = uuid::Uuid::new_v4().to_string();
                        }
                        current_preview = None;
                    }
                }

                if !config.stdout_only && !config.silent {
                    render_apple_hud_line(
                        tick,
                        current_rms,
                        recent_peak,
                        audio_bytes,
                        current_preview.as_deref(),
                    );
                }
            }
            maybe_samples = audio_rx.recv() => {
                let Some(samples) = maybe_samples else {
                    break;
                };

                let rms = compute_rms(&samples);
                let peak = compute_peak(&samples);
                current_rms = rms;
                recent_peak = recent_peak.max(peak);
                audio_bytes += (samples.len() * 4) as u64;
                last_audio_instant = Instant::now();
                trailing_silence_fed = 0;

                let now_ms = session_start.elapsed().as_millis() as i64;
                clock_tracker.lock().unwrap().record_feed(samples.len(), now_ms, true);

                unsafe {
                    wisp_applespeech_feed(engine_ptr, samples.as_ptr(), samples.len(), 16000.0);
                }
            }
        }
    }

    if !config.stdout_only && !config.silent {
        eprintln!("\n👋 Stopping Apple Speech live capture...");
    }

    unsafe {
        wisp_applespeech_stop(engine_ptr);
    }

    let final_cb_state = unsafe { Arc::from_raw(state_ptr as *const AppleCbState) };
    let utts = final_cb_state.all_utterances.lock().unwrap().clone();

    Ok(Transcript {
        processing_time: session_start.elapsed(),
        utterances: utts,
        word_utterances: None,
    })
}
