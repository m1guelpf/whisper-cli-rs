use crate::{
    live::{
        audio::{
            compute_peak, compute_rms, find_best_split_index, whisper_macos_audio_bytes_received,
            whisper_macos_audio_packets_received, whisper_macos_speech_permille, MacOsAudioStream,
            TempWavFile, SAMPLE_RATE,
        },
        config::LiveConfig,
        hud::render_live_hud,
        ramp::{AdaptiveRampState, AudioSegment},
        sub_server::start_sub_server,
        text_filter::{clean_subtitle_text, update_prompt_context},
        vad::SpeechDetector,
    },
    provider::Transcriber,
    transcript::{Transcript, Utternace},
    utils::format_timestamp,
};
use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{sync::mpsc, task::JoinHandle};

#[cfg(not(target_os = "macos"))]
pub async fn run_live_subtitles(
    _transcriber: Arc<Box<dyn Transcriber>>,
    _config: LiveConfig,
) -> Result<Transcript> {
    Err(anyhow!(
        "--live system audio capture is only supported on macOS (via native ScreenCaptureKit)."
    ))
}

#[cfg(target_os = "macos")]
fn print_vad_banner(config: &LiveConfig, is_realtime: bool) {
    if config.stdout_only {
        return;
    }
    let lang_str = config.lang.as_deref().unwrap_or("auto-detect");
    let mode_str = if config.translate {
        format!("{lang_str} -> English (translate)")
    } else {
        lang_str.to_string()
    };
    let window_desc = if is_realtime {
        "Adaptive Near-Real-Time (3.2s Anchor ⚓ -> 1.4s Realtime ⚡, Prompt Chained)".to_string()
    } else if config.speech_mode {
        let eff_max = config.chunk_secs.max(15.0);
        format!("Smart Speech VAD (waits for sentence end up to {eff_max:.0}s, filters SFX/music)")
    } else {
        let eff_max = config.chunk_secs.clamp(1.5, 15.0);
        format!("{:.1}s adaptive VAD chunks", eff_max)
    };
    eprintln!("╭──────────────────────────────────────────────────────────────╮");
    eprintln!("│ 🎧 macOS Live System Audio Subtitles (ScreenCaptureKit)      │");
    eprintln!("╰──────────────────────────────────────────────────────────────╯");
    eprintln!("  • Provider : {}", config.provider_label);
    if let Some(m) = &config.model {
        eprintln!("  • Model    : {m}");
    }
    if config.sub_port > 0 {
        eprintln!(
            "  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)",
            config.sub_port
        );
    }
    eprintln!("  • Language : {mode_str}");
    eprintln!("  • Mode     : {window_desc} (Press Ctrl+C to stop)\n");
}

#[cfg(target_os = "macos")]
fn spawn_transcription_worker(
    transcriber: Arc<Box<dyn Transcriber>>,
    sub_broadcast: tokio::sync::broadcast::Sender<String>,
    sub_history: Arc<std::sync::Mutex<VecDeque<serde_json::Value>>>,
    session_start: Instant,
    stdout_only: bool,
    translate: bool,
    karaoke: bool,
) -> (
    mpsc::Sender<AudioSegment>,
    Arc<AtomicBool>,
    JoinHandle<Vec<Utternace>>,
) {
    let (segment_tx, mut segment_rx) = mpsc::channel::<AudioSegment>(8);
    let is_transcribing = Arc::new(AtomicBool::new(false));
    let is_transcribing_worker = Arc::clone(&is_transcribing);

    let handle = tokio::spawn(async move {
        let mut all_utterances: Vec<Utternace> = Vec::new();
        let mut last_printed = String::new();
        let mut prompt_context = String::new();

        while let Some(seg) = segment_rx.recv().await {
            is_transcribing_worker.store(true, Ordering::Relaxed);

            if seg.is_cold_start {
                prompt_context.clear();
            }

            let prompt_opt = if prompt_context.trim().is_empty() {
                None
            } else {
                Some(prompt_context.as_str())
            };

            if let Ok(wav) = TempWavFile::new(&seg.samples) {
                match transcriber
                    .transcribe_with_prompt(wav.path(), translate, karaoke, prompt_opt)
                    .await
                {
                    Ok(transcript) => {
                        let subtitle = clean_subtitle_text(&transcript);
                        if !subtitle.is_empty() && subtitle != last_printed {
                            update_prompt_context(&mut prompt_context, &subtitle);
                            let ts = format_timestamp(seg.start_offset_ms / 10, false, ".");
                            let short_ts = ts.split('.').next().unwrap_or(&ts);

                            if stdout_only {
                                println!("{subtitle}");
                                let _ = std::io::stdout().flush();
                            } else {
                                eprint!("\r\x1b[2K");
                                println!("\x1b[2m[{short_ts}]\x1b[0m  \x1b[1m{subtitle}\x1b[0m");
                                let _ = std::io::stdout().flush();
                            }

                            let now_ms = session_start.elapsed().as_millis() as i64;
                            let audio_duration_ms = (seg.samples.len() * 1000 / SAMPLE_RATE) as u64;
                            let duration_ms = ((seg.end_offset_ms - seg.start_offset_ms).max(120) as u64)
                                .min(audio_duration_ms.max(120));
                            let lag_ms = (now_ms - seg.end_offset_ms).max(0) as u64;
                            let cue = serde_json::json!({
                                "id": uuid::Uuid::new_v4().to_string(),
                                "text": &subtitle,
                                "timestamp": short_ts,
                                "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
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
                                hist.push_back(cue.clone());
                            }
                            let _ = sub_broadcast.send(cue.to_string());
                            last_printed = subtitle.clone();
                            all_utterances.push(Utternace {
                                start: seg.start_offset_ms / 10,
                                stop: seg.end_offset_ms / 10,
                                text: subtitle,
                            });
                        }
                    }
                    Err(err) => {
                        if !stdout_only {
                            eprint!("\r\x1b[2K");
                            eprintln!("\x1b[33m⚠️  Transcription error: {err}\x1b[0m");
                        }
                    }
                }
            }

            is_transcribing_worker.store(false, Ordering::Relaxed);
        }

        all_utterances
    });

    (segment_tx, is_transcribing, handle)
}

#[cfg(target_os = "macos")]
pub async fn run_live_subtitles(
    transcriber: Arc<Box<dyn Transcriber>>,
    config: LiveConfig,
) -> Result<Transcript> {
    if config.apple {
        return crate::live::engine_apple::run_live_apple_speech(config).await;
    }
    if config.moonshine {
        return crate::live::engine_moonshine::run_live_moonshine(config).await;
    }

    let (_stream, mut audio_rx) = MacOsAudioStream::start()?;
    let session_start = Instant::now();

    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(
        VecDeque::<serde_json::Value>::with_capacity(32),
    ));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    let is_realtime = config.preset == Some(crate::server::ServerPreset::Realtime);
    let mut ramp = AdaptiveRampState::new(is_realtime, config.speech_mode, config.chunk_secs);
    print_vad_banner(&config, is_realtime);

    let (segment_tx, is_transcribing_ui, worker_handle) = spawn_transcription_worker(
        Arc::clone(&transcriber),
        sub_broadcast_tx.clone(),
        Arc::clone(&sub_history),
        session_start,
        config.stdout_only,
        config.translate,
        config.karaoke,
    );

    let pre_roll_capacity = if is_realtime {
        (0.20 * SAMPLE_RATE as f32) as usize
    } else {
        (0.35 * SAMPLE_RATE as f32) as usize
    };
    let overlap_samples = if is_realtime {
        (0.12 * SAMPLE_RATE as f32) as usize
    } else {
        (0.18 * SAMPLE_RATE as f32) as usize
    };
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
    let mut ui_interval = tokio::time::interval(Duration::from_millis(80));

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                if !stdout_only {
                    eprint!("\r\x1b[2K");
                    eprintln!("\n👋 Stopping live audio capture...");
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
                        ramp.stage_label(),
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
                    rms >= vad_threshold || pkt_peak >= 0.025
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

    // Flush any remaining speech on exit
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
