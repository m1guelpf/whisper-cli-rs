use crate::{
    provider::Transcriber,
    transcript::{Transcript, Utternace},
    utils::format_timestamp,
};
use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    env::temp_dir,
    ffi::{c_char, c_void, CStr},
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

const SAMPLE_RATE: usize = 16_000;
const SPARK_CHARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[cfg(target_os = "macos")]
type WhisperAudioCallback = extern "C" fn(samples: *const f32, count: usize, user_data: *mut c_void);

#[cfg(target_os = "macos")]
extern "C" {
    fn whisper_macos_audio_start(
        callback: WhisperAudioCallback,
        user_data: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    fn whisper_macos_audio_stop(handle: *mut c_void);
    fn whisper_macos_free_error(err: *mut c_char);
    fn whisper_macos_audio_packets_received() -> u64;
    fn whisper_macos_video_packets_received() -> u64;
    fn whisper_macos_audio_bytes_received() -> u64;
    fn whisper_macos_audio_last_status() -> i32;
    fn whisper_macos_audio_sample_rate() -> u32;
    fn whisper_macos_speech_permille() -> u32;
    fn wisp_applespeech_available() -> bool;
    fn wisp_applespeech_start(
        locale: *const c_char,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, bool),
    ) -> *mut c_void;
    fn wisp_applespeech_feed(handle: *mut c_void, samples: *const f32, count: usize, rate: f64);
    fn wisp_applespeech_stop(handle: *mut c_void);
}

#[cfg(target_os = "macos")]
extern "C" fn audio_callback_trampoline(
    samples: *const f32,
    count: usize,
    user_data: *mut c_void,
) {
    if samples.is_null() || count == 0 || user_data.is_null() {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts(samples, count) };
    let sender = unsafe { &*(user_data as *const mpsc::UnboundedSender<Vec<f32>>) };
    let _ = sender.send(slice.to_vec());
}

#[cfg(target_os = "macos")]
struct MacOsAudioStream {
    handle: *mut c_void,
    sender_ptr: *mut mpsc::UnboundedSender<Vec<f32>>,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MacOsAudioStream {}

#[cfg(target_os = "macos")]
impl MacOsAudioStream {
    fn start() -> Result<(Self, mpsc::UnboundedReceiver<Vec<f32>>)> {
        let (tx, rx) = mpsc::unbounded_channel();
        let sender_box = Box::new(tx);
        let sender_ptr = Box::into_raw(sender_box);
        let mut error_out: *mut c_char = std::ptr::null_mut();

        let handle = unsafe {
            whisper_macos_audio_start(
                audio_callback_trampoline,
                sender_ptr.cast::<c_void>(),
                &mut error_out,
            )
        };

        if handle.is_null() {
            unsafe {
                let _ = Box::from_raw(sender_ptr);
            }
            let err_msg = if error_out.is_null() {
                "Unknown error starting macOS ScreenCaptureKit audio stream".to_string()
            } else {
                let msg = unsafe { CStr::from_ptr(error_out) }
                    .to_string_lossy()
                    .into_owned();
                unsafe {
                    whisper_macos_free_error(error_out);
                }
                msg
            };
            return Err(anyhow!(err_msg));
        }

        Ok((
            Self {
                handle,
                sender_ptr,
            },
            rx,
        ))
    }
}

#[cfg(target_os = "macos")]
impl Drop for MacOsAudioStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                whisper_macos_audio_stop(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
        if !self.sender_ptr.is_null() {
            unsafe {
                let _ = Box::from_raw(self.sender_ptr);
            }
            self.sender_ptr = std::ptr::null_mut();
        }
    }
}

/// Hybrid acoustic vocal-band (300Hz–3400Hz) + Apple Neural Engine (`SoundAnalysis`) speech detector.
/// Distinguishes human speech from loud anime SFX (explosions, sword clashes) and background music.
struct SpeechDetector {
    hp_prev_in: f32,
    hp_prev_out: f32,
    lp_prev_out: f32,
    noise_floor_rms: f32,
    smoothed_speech_prob: f32,
}

impl SpeechDetector {
    fn new() -> Self {
        Self {
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
            lp_prev_out: 0.0,
            noise_floor_rms: 0.003,
            smoothed_speech_prob: 0.0,
        }
    }

    /// Analyzes a frame of 16kHz mono audio and combines fast 20ms vocal-band acoustics
    /// with macOS `SoundAnalysis` neural speech confidence (`ane_speech_conf` in `0.0..=1.0`).
    /// Returns `(is_speech, speech_probability, vocal_band_rms)`.
    fn analyze(&mut self, samples: &[f32], ane_speech_conf: f32) -> (bool, f32, f32) {
        if samples.is_empty() {
            return (false, 0.0, 0.0);
        }

        // 1st-order High-Pass (~280 Hz at 16kHz) + 1st-order Low-Pass (~3400 Hz at 16kHz)
        // Isolates the human vowel/formant frequency band while rejecting sub-bass rumbles & high-freq SFX
        let hp_alpha: f32 = 0.90;
        let lp_alpha: f32 = 0.57;

        let mut total_sq: f32 = 0.0;
        let mut vocal_sq: f32 = 0.0;
        let mut zero_crossings: usize = 0;
        let mut prev_vocal: f32 = self.lp_prev_out;

        for &x in samples {
            total_sq += x * x;

            let hp = hp_alpha * (self.hp_prev_out + x - self.hp_prev_in);
            self.hp_prev_in = x;
            self.hp_prev_out = hp;

            let vocal = self.lp_prev_out + lp_alpha * (hp - self.lp_prev_out);
            self.lp_prev_out = vocal;

            vocal_sq += vocal * vocal;
            if (vocal >= 0.0 && prev_vocal < 0.0) || (vocal < 0.0 && prev_vocal >= 0.0) {
                zero_crossings += 1;
            }
            prev_vocal = vocal;
        }

        let n = samples.len() as f32;
        let total_rms = (total_sq / n).sqrt();
        let vocal_rms = (vocal_sq / n).sqrt();
        let zcr = zero_crossings as f32 / n;

        // Instant silence check: if vocal-band energy is very low, no one is speaking right now
        // (even if the 0.6s neural window from a previous word hasn't finished decaying yet).
        if vocal_rms < 0.004 && total_rms < 0.005 {
            self.noise_floor_rms = (self.noise_floor_rms * 0.95 + vocal_rms * 0.05).max(0.002);
            self.smoothed_speech_prob *= 0.60;
            return (false, self.smoothed_speech_prob, vocal_rms);
        }

        // Ratio of energy in the 300-3400Hz vocal band vs broadband energy
        let band_ratio = (vocal_rms / (total_rms + 1e-6)).clamp(0.0, 1.2);

        // Human speech ZCR in the vocal band typically sits between 0.02 (160Hz) and 0.24 (1900Hz)
        let zcr_voiced = if (0.015..=0.25).contains(&zcr) {
            1.0_f32
        } else {
            0.35_f32
        };

        // Update background music / ambient noise floor slowly when ANE says non-speech
        if ane_speech_conf < 0.20 {
            self.noise_floor_rms =
                (self.noise_floor_rms * 0.96 + vocal_rms * 0.04).clamp(0.002, 0.08);
        } else if vocal_rms < self.noise_floor_rms {
            self.noise_floor_rms =
                (self.noise_floor_rms * 0.85 + vocal_rms * 0.15).clamp(0.002, 0.08);
        }

        // How far above the continuous background music floor is the current vocal-band energy?
        let snr_above_floor = (vocal_rms / (self.noise_floor_rms * 1.45 + 0.002)).clamp(0.0, 2.5);
        let acoustic_score =
            ((band_ratio * 0.85) * zcr_voiced * (snr_above_floor * 0.65)).clamp(0.0, 1.0);

        // Combine Apple Neural Engine SoundAnalysis confidence with 20ms vocal-band score:
        // - If ANE is confident it's speech (> 0.30) AND vocal_rms is active -> high probability
        // - If ANE says it's pure SFX/music (< 0.12) even though it's loud -> suppress score
        let raw_prob = if ane_speech_conf >= 0.25 {
            (ane_speech_conf * 0.70 + acoustic_score * 0.30).clamp(0.0, 1.0)
        } else if ane_speech_conf > 0.01 {
            // ANE is active and sees mostly non-speech (music/SFX), require strong vocal-band modulation
            (ane_speech_conf * 0.55 + acoustic_score * 0.35).clamp(0.0, 1.0)
        } else {
            // Fallback during initial 300ms before first ANE window fires
            acoustic_score
        };

        self.smoothed_speech_prob = self.smoothed_speech_prob * 0.45 + raw_prob * 0.55;
        let is_speech =
            vocal_rms >= 0.0045 && (self.smoothed_speech_prob >= 0.36 || ane_speech_conf >= 0.42);

        (is_speech, self.smoothed_speech_prob, vocal_rms)
    }
}

struct TempWavFile {
    path: PathBuf,
}

impl TempWavFile {
    fn new(samples: &[f32]) -> Result<Self> {
        let path = temp_dir().join(format!("whisper_live_{}.wav", uuid::Uuid::new_v4()));
        write_pcm16_wav(&path, samples, SAMPLE_RATE as u32)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWavFile {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn write_pcm16_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let max_abs = samples
        .iter()
        .fold(0.0_f32, |acc, &s| acc.max(s.abs()));

    // Gently normalize low-volume streams so Whisper still receives a clear signal
    let gain = if max_abs > 0.006 && max_abs < 0.35 {
        (0.70 / max_abs).min(8.0)
    } else {
        1.0
    };

    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36 + data_bytes;
    let byte_rate = sample_rate * 2;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"RIFF")?;
    writer.write_all(&riff_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16_u32.to_le_bytes())?; // PCM fmt chunk size
    writer.write_all(&1_u16.to_le_bytes())?; // AudioFormat = 1 (PCM)
    writer.write_all(&1_u16.to_le_bytes())?; // NumChannels = 1 (Mono)
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&2_u16.to_le_bytes())?; // BlockAlign = 2
    writer.write_all(&16_u16.to_le_bytes())?; // BitsPerSample = 16
    writer.write_all(b"data")?;
    writer.write_all(&data_bytes.to_le_bytes())?;

    for &sample in samples {
        let scaled = (sample * gain).clamp(-1.0, 1.0);
        let pcm16 = (scaled * 32767.0) as i16;
        writer.write_all(&pcm16.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn compute_peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .fold(0.0_f32, |acc, &s| acc.max(s.abs()))
}

/// Find the quietest 25ms window within the last `search_window` samples to split continuous speech cleanly.
fn find_best_split_index(samples: &[f32], search_window: usize) -> usize {
    let len = samples.len();
    let window = 400; // 25ms at 16kHz
    if len <= search_window || search_window <= window * 2 {
        return len;
    }

    let start_search = len - search_window;
    let mut best_idx = len;
    let mut min_energy = f32::MAX;

    let step = 80; // 5ms step
    let mut idx = start_search;
    while idx + window <= len {
        let energy: f32 = samples[idx..idx + window].iter().map(|&s| s * s).sum();
        if energy < min_energy {
            min_energy = energy;
            best_idx = idx + window / 2;
        }
        idx += step;
    }

    best_idx
}

fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Filter out pure bracketed/parenthesized non-speech tags like [Music], (Laughter), ♪
    if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('♪') && trimmed.ends_with('♪'))
    {
        return true;
    }

    let cleaned: String = trimmed
        .chars()
        .filter(|c| {
            !c.is_ascii_punctuation()
                && !matches!(*c, '¡' | '¿' | '♪' | '…' | '。' | '、' | '！' | '？')
        })
        .collect::<String>()
        .to_lowercase();
    let cleaned = cleaned.trim();

    if cleaned.is_empty() {
        return true;
    }

    const HALLUCINATIONS: &[&str] = &[
        "thank you for watching",
        "thanks for watching",
        "thank you very much for watching",
        "subtitles by the amaraorg community",
        "subtitles by",
        "transcribed by",
        "please subscribe",
        "like and subscribe",
        "suscríbete al canal",
        "suscribete al canal",
        "subtítulos realizados por la comunidad de amaraorg",
        "subtitulos realizados por la comunidad de amaraorg",
        "gracias por ver el video",
        "ご視聴ありがとうございました",
        "チャンネル登録よろしくお願いします",
        "チャンネル登録お願いします",
        "字幕",
        "音楽",
        "拍手",
        "music",
        "applause",
        "laughter",
        "you",
    ];

    HALLUCINATIONS.contains(&cleaned)
}

fn clean_subtitle_text(transcript: &Transcript) -> String {
    let mut parts = Vec::new();
    for u in &transcript.utterances {
        let t = u.text.trim();
        if !t.is_empty() && !is_hallucination(t) {
            parts.push(t);
        }
    }
    let joined = parts.join(" ");
    if is_hallucination(&joined) {
        String::new()
    } else {
        joined
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn render_sparkline(history: &VecDeque<f32>) -> String {
    history
        .iter()
        .map(|&rms| {
            let norm = (rms / 0.10).clamp(0.0, 1.0).sqrt();
            let idx = ((norm * (SPARK_CHARS.len() - 1) as f32).round() as usize)
                .min(SPARK_CHARS.len() - 1);
            SPARK_CHARS[idx]
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn render_live_hud(
    tick: usize,
    rms: f32,
    peak: f32,
    speech_prob: f32,
    spark_history: &VecDeque<f32>,
    is_speaking: bool,
    speech_secs: f32,
    max_secs: f32,
    in_flight: bool,
    audio_bytes: u64,
    audio_pkts: u64,
    speech_mode: bool,
    stage_label: &str,
) -> String {
    let spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len()];
    let normalized = (rms / 0.12).clamp(0.0, 1.0).sqrt();
    let filled = (normalized * 10.0).round() as usize;
    let empty = 10_usize.saturating_sub(filled);

    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
    let spark = render_sparkline(spark_history);
    let bytes_str = format_bytes(audio_bytes);
    let speech_pct = (speech_prob * 100.0).round() as u32;

    let state = if in_flight {
        "\x1b[33m⏳ Sending\x1b[0m".to_string()
    } else if is_speaking {
        if speech_mode {
            format!("\x1b[32m🗣️ {speech_secs:.1}s (Voice {speech_pct}%){stage_label}\x1b[0m")
        } else {
            format!("\x1b[32m🗣️ {speech_secs:.1}s/{max_secs:.1}s{stage_label}\x1b[0m")
        }
    } else if audio_bytes == 0 {
        "\x1b[2m⚪ Waiting for macOS audio...\x1b[0m".to_string()
    } else if speech_mode && rms >= 0.008 && speech_pct < 25 {
        format!("\x1b[35m🎵 SFX/Music ({speech_pct}%)\x1b[0m")
    } else {
        format!("\x1b[36m🎙️ Listening ({speech_pct}%)\x1b[0m")
    };

    format!(
        "{spinner} [{bar}] \x1b[36m{spark}\x1b[0m │ 📥 {bytes_str} ({audio_pkts} pkts) │ Peak: {peak:.2} │ {state}"
    )
}

pub struct LiveConfig {
    pub provider_label: String,
    pub model: Option<String>,
    pub preset: Option<crate::server::ServerPreset>,
    pub lang: Option<String>,
    pub translate: bool,
    pub karaoke: bool,
    pub stdout_only: bool,
    pub chunk_secs: f32,
    pub speech_mode: bool,
    pub sub_port: u16,
    pub apple: bool,
    pub moonshine: bool,
}

pub async fn start_sub_server(
    port: u16,
    sub_tx: tokio::sync::broadcast::Sender<String>,
    history: Arc<std::sync::Mutex<VecDeque<serde_json::Value>>>,
) {
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("\x1b[33m⚠️  Could not bind subtitle broadcast server to {addr}: {e}\x1b[0m");
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };

            let mut sub_rx = sub_tx.subscribe();
            let history_client = Arc::clone(&history);
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};

                let mut buf = [0u8; 1024];
                let n = match socket.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => return,
                };
                let req = String::from_utf8_lossy(&buf[..n]);
                let first_line = req.lines().next().unwrap_or("");

                if first_line.starts_with("OPTIONS") {
                    let resp = "HTTP/1.1 204 No Content\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n\
Content-Length: 0\r\n\
\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/status") {
                    let body = r#"{"status":"online"}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\n\
Access-Control-Allow-Origin: *\r\n\
Content-Type: application/json\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\
\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/history") {
                    let body = {
                        let hist = history_client.lock().unwrap();
                        serde_json::to_string(&*hist).unwrap_or_else(|_| "[]".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\n\
Access-Control-Allow-Origin: *\r\n\
Content-Type: application/json\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\
\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/events") {
                    let header = "HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Cache-Control: no-cache\r\n\
Connection: keep-alive\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n\
\r\n\
retry: 1000\r\n\
\r\n";
                    if socket.write_all(header.as_bytes()).await.is_err() {
                        return;
                    }

                    // Send recent history buffer on initial connect so new tabs have previous context
                    let initial_cues: Vec<serde_json::Value> = {
                        let hist = history_client.lock().unwrap();
                        hist.iter().cloned().collect()
                    };
                    for cue in initial_cues {
                        let sse_chunk = format!("data: {cue}\n\n");
                        if socket.write_all(sse_chunk.as_bytes()).await.is_err() {
                            return;
                        }
                    }

                    let mut ping_interval = tokio::time::interval(Duration::from_secs(15));
                    loop {
                        tokio::select! {
                            msg = sub_rx.recv() => {
                                match msg {
                                    Ok(data) => {
                                        let sse_chunk = format!("data: {data}\n\n");
                                        if socket.write_all(sse_chunk.as_bytes()).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                                    Err(_) => break,
                                }
                            }
                            _ = ping_interval.tick() => {
                                if socket.write_all(b":keep-alive\n\n").await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    return;
                }

                let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes()).await;
            });
        }
    });
}

struct AudioSegment {
    samples: Vec<f32>,
    start_offset_ms: i64,
    end_offset_ms: i64,
    is_cold_start: bool,
}

fn update_prompt_context(prompt: &mut String, new_text: &str) {
    let clean = new_text.trim();
    if clean.is_empty() {
        return;
    }
    if prompt.is_empty() {
        prompt.push_str(clean);
    } else {
        prompt.push(' ');
        prompt.push_str(clean);
    }

    // Keep only the most recent ~30 words (~50 tokens) to stay well within decoder prompt budget
    let words: Vec<&str> = prompt.split_whitespace().collect();
    if words.len() > 30 {
        *prompt = words[words.len() - 25..].join(" ");
    }
}

struct AdaptiveRampState {
    is_realtime: bool,
    speech_mode: bool,
    base_chunk_secs: f32,
    consecutive_utterances: usize,
    last_utterance_end: Instant,
}

impl AdaptiveRampState {
    fn new(is_realtime: bool, speech_mode: bool, base_chunk_secs: f32) -> Self {
        Self {
            is_realtime,
            speech_mode,
            base_chunk_secs,
            consecutive_utterances: 0,
            last_utterance_end: Instant::now() - Duration::from_secs(10),
        }
    }

    fn check_silence_reset(&mut self) {
        if self.consecutive_utterances > 0
            && self.last_utterance_end.elapsed() > Duration::from_millis(1800)
        {
            self.consecutive_utterances = 0;
        }
    }

    fn current_thresholds(&self) -> (usize, usize, usize, usize, usize, usize, usize, bool) {
        let is_cold = self.consecutive_utterances == 0;

        if self.is_realtime {
            let (max_s, min_split_s, pause_s) = match self.consecutive_utterances {
                0 => (3.2_f32, 1.10_f32, 0.38_f32), // Turn 0: Cold anchor (solid sentence)
                1 => (2.2_f32, 0.70_f32, 0.26_f32), // Turn 1: Warm intermediate
                _ => (1.4_f32, 0.45_f32, 0.18_f32), // Turn 2+: Full streaming cadence
            };
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (min_split_s * SAMPLE_RATE as f32) as usize,
                (pause_s * SAMPLE_RATE as f32) as usize,
                (1.5 * SAMPLE_RATE as f32) as usize,
                (0.14 * SAMPLE_RATE as f32) as usize,
                (0.20 * SAMPLE_RATE as f32) as usize,
                (0.35 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        } else if self.speech_mode {
            let max_s = self.base_chunk_secs.max(15.0);
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (1.10 * SAMPLE_RATE as f32) as usize,
                (0.46 * SAMPLE_RATE as f32) as usize,
                (7.5 * SAMPLE_RATE as f32) as usize,
                (0.24 * SAMPLE_RATE as f32) as usize,
                (0.32 * SAMPLE_RATE as f32) as usize,
                (0.68 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        } else {
            let max_s = self.base_chunk_secs.clamp(1.5, 15.0);
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (0.80 * SAMPLE_RATE as f32) as usize,
                (0.36 * SAMPLE_RATE as f32) as usize,
                (6.0 * SAMPLE_RATE as f32) as usize,
                (0.20 * SAMPLE_RATE as f32) as usize,
                (0.30 * SAMPLE_RATE as f32) as usize,
                (0.60 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        }
    }

    fn record_emission(&mut self) -> bool {
        let was_cold = self.consecutive_utterances == 0;
        self.consecutive_utterances = self.consecutive_utterances.saturating_add(1);
        self.last_utterance_end = Instant::now();
        was_cold
    }

    fn stage_label(&self) -> &'static str {
        if !self.is_realtime {
            ""
        } else {
            match self.consecutive_utterances {
                0 => " [Anchor ⚓]",
                1 => " [Warming 🚀]",
                _ => " [Realtime ⚡]",
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub async fn run_audio_test() -> Result<()> {
    Err(anyhow!(
        "--test-audio is only supported on macOS (via native ScreenCaptureKit)."
    ))
}

#[cfg(target_os = "macos")]
pub async fn run_audio_test() -> Result<()> {
    let (_stream, mut audio_rx) = MacOsAudioStream::start()?;
    let start = Instant::now();

    eprintln!("╭──────────────────────────────────────────────────────────────╮");
    eprintln!("│ 🎛️  macOS Native System Audio Diagnostic Monitor             │");
    eprintln!("╰──────────────────────────────────────────────────────────────╯");
    eprintln!("  Play any video/music (Crunchyroll, YouTube, Spotify) to see live bytes, waveform & voice %.");
    eprintln!("  Press Ctrl+C to stop.\n");

    let mut spark_history: VecDeque<f32> = VecDeque::from(vec![0.0; 16]);
    let mut detector = SpeechDetector::new();
    let mut current_rms: f32 = 0.0;
    let mut current_speech_prob: f32 = 0.0;
    let mut is_voice: bool = false;
    let mut max_session_peak: f32 = 0.0;
    let mut tick: usize = 0;
    let mut interval = tokio::time::interval(Duration::from_millis(80));

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                eprint!("\r\x1b[2K");
                let bytes = unsafe { whisper_macos_audio_bytes_received() };
                let pkts = unsafe { whisper_macos_audio_packets_received() };
                eprintln!(
                    "\n✅ Audio test finished in {:.1}s — Received {} across {} packets (Max Peak: {:.3}).",
                    start.elapsed().as_secs_f32(),
                    format_bytes(bytes),
                    pkts,
                    max_session_peak
                );
                break;
            }
            maybe_samples = audio_rx.recv() => {
                if let Some(samples) = maybe_samples {
                    current_rms = compute_rms(&samples);
                    let pkt_peak = compute_peak(&samples);
                    if pkt_peak > max_session_peak {
                        max_session_peak = pkt_peak;
                    }
                    let ane_conf = unsafe { whisper_macos_speech_permille() } as f32 / 1000.0;
                    let (voice, prob, _) = detector.analyze(&samples, ane_conf);
                    is_voice = voice;
                    current_speech_prob = prob;

                    spark_history.push_back(current_rms);
                    if spark_history.len() > 16 {
                        spark_history.pop_front();
                    }
                }
            }
            _ = interval.tick() => {
                tick = tick.wrapping_add(1);
                let bytes = unsafe { whisper_macos_audio_bytes_received() };
                let pkts = unsafe { whisper_macos_audio_packets_received() };
                let sr = unsafe { whisper_macos_audio_sample_rate() };

                let hud = render_live_hud(
                    tick,
                    current_rms,
                    max_session_peak,
                    current_speech_prob,
                    &spark_history,
                    is_voice,
                    0.0,
                    4.0,
                    false,
                    bytes,
                    pkts,
                    true,
                    "",
                );
                eprint!("\r\x1b[2K{hud} ({sr}Hz)");
                let _ = std::io::stderr().flush();

                current_rms *= 0.80;
                current_speech_prob *= 0.85;
                spark_history.push_back(current_rms);
                if spark_history.len() > 16 {
                    spark_history.pop_front();
                }
            }
        }
    }

    Ok(())
}

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
pub async fn run_live_apple_speech(config: LiveConfig) -> Result<Transcript> {
    let session_start = Instant::now();
    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(VecDeque::<serde_json::Value>::with_capacity(32)));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    let lang_str = config.lang.as_deref().unwrap_or("en");

    if !config.stdout_only {
        eprintln!("╭──────────────────────────────────────────────────────────────╮");
        eprintln!("│ ⚡ macOS Apple Speech Recognition (Neural Engine Streaming)  │");
        eprintln!("╰──────────────────────────────────────────────────────────────╯");
        eprintln!("  • Engine   : Apple Speech.framework (Zero-Lag Streaming)");
        eprintln!("  • Audio    : ScreenCaptureKit (System Audio, No Mic/BlackHole)");
        if config.sub_port > 0 {
            eprintln!("  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)", config.sub_port);
        }
        eprintln!("  • Language : {lang_str}");
        eprintln!("  • Status   : Streaming live... (Press Ctrl+C to stop)\n");
    }

    enum SpeechEvent {
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

    struct AppleCbState {
        event_tx: tokio::sync::mpsc::UnboundedSender<SpeechEvent>,
        last_id: Arc<std::sync::Mutex<String>>,
        start_instant: Instant,
        last_phrase_instant: Arc<std::sync::Mutex<Instant>>,
        all_utterances: Arc<std::sync::Mutex<Vec<Utternace>>>,
    }

    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<SpeechEvent>();
    let all_utterances = Arc::new(std::sync::Mutex::new(Vec::new()));
    let last_id = Arc::new(std::sync::Mutex::new(uuid::Uuid::new_v4().to_string()));
    let last_phrase_instant = Arc::new(std::sync::Mutex::new(Instant::now()));

    let cb_state = Arc::new(AppleCbState {
        event_tx,
        last_id: Arc::clone(&last_id),
        start_instant: session_start,
        last_phrase_instant: Arc::clone(&last_phrase_instant),
        all_utterances: Arc::clone(&all_utterances),
    });

    extern "C" fn apple_speech_callback(
        ctx: *mut c_void,
        text_ptr: *const c_char,
        is_final: bool,
    ) {
        if ctx.is_null() || text_ptr.is_null() { return; }
        let c_str = unsafe { CStr::from_ptr(text_ptr) };
        let Ok(text) = c_str.to_str() else { return };
        let text = text.trim();
        if text.is_empty() { return; }

        let state = unsafe { &*(ctx as *const AppleCbState) };
        let now_ms = state.start_instant.elapsed().as_millis() as i64;
        let phrase_elapsed_ms = {
            let mut last = state.last_phrase_instant.lock().unwrap();
            let elapsed = last.elapsed().as_millis() as u64;
            if is_final {
                *last = Instant::now();
            }
            elapsed
        };
        let duration_ms = phrase_elapsed_ms.max(800);
        let start_ms = now_ms.saturating_sub(duration_ms as i64).max(0);
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
            "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
            "durationMs": duration_ms,
            "lagMs": 30,
        });

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

    if !unsafe { wisp_applespeech_available() } {
        return Err(anyhow!("macOS 26+ SpeechAnalyzer is not supported on this macOS version."));
    }

    let lang_lower = lang_str.trim().to_ascii_lowercase();
    let locale_str: &str = match lang_lower.as_str() {
        "es" => "es-ES",
        "ja" => "ja-JP",
        "zh" | "cmn" => "zh-CN",
        "fr" => "fr-FR",
        "de" => "de-DE",
        "it" => "it-IT",
        "ko" => "ko-KR",
        "pt" => "pt-BR",
        "ru" => "ru-RU",
        other if other.contains('-') => other,
        _ => "en-US",
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

    let mut hud_interval = tokio::time::interval(std::time::Duration::from_millis(60));
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
                        current_preview = Some(preview);
                        pending_final = Some(cue_val.clone());
                        // Immediately broadcast real-time streaming words to browser extension overlay!
                        let _ = sub_broadcast_tx.send(cue_val.to_string());
                    }
                    SpeechEvent::Final { text, short_ts, start_ms, duration_ms, cue_val } => {
                        current_preview = None;
                        pending_final = None;

                        // 1. Update subtitle history
                        {
                            let mut hist = sub_history.lock().unwrap();
                            if hist.len() >= 25 {
                                hist.pop_front();
                            }
                            hist.push_back(cue_val.clone());
                        }

                        // 2. Broadcast to browser overlay
                        let _ = sub_broadcast_tx.send(cue_val.to_string());

                        // 3. Store utterance
                        all_utterances.lock().unwrap().push(Utternace {
                            start: start_ms / 10,
                            stop: (start_ms + duration_ms as i64) / 10,
                            text: text.clone(),
                        });

                        // 4. Print clean final subtitle on its own line
                        if config.stdout_only {
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

                // When audio pauses, feed brief trailing silence to help Apple's model finalize
                if last_audio_instant.elapsed() > Duration::from_millis(80) && trailing_silence_fed < 6 {
                    trailing_silence_fed += 1;
                    unsafe {
                        wisp_applespeech_feed(engine_ptr, silence_chunk.as_ptr(), silence_chunk.len(), 16000.0);
                    }
                }

                // If playback paused/stopped and we have unfinalized streaming speech:
                if last_audio_instant.elapsed() > Duration::from_millis(600) {
                    if let Some(cue_val) = pending_final.take() {
                        let cue_text = cue_val.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let short_ts = cue_val.get("timestamp").and_then(|v| v.as_str()).unwrap_or("00:00").to_string();
                        let dur = cue_val.get("durationMs").and_then(|v| v.as_u64()).unwrap_or(1000);

                        {
                            let mut hist = sub_history.lock().unwrap();
                            if hist.len() >= 25 {
                                hist.pop_front();
                            }
                            hist.push_back(cue_val.clone());
                        }
                        all_utterances.lock().unwrap().push(Utternace {
                            start: 0,
                            stop: dur as i64 / 10,
                            text: cue_text.clone(),
                        });

                        if config.stdout_only {
                            println!("{cue_text}");
                            let _ = std::io::stdout().flush();
                        } else {
                            eprintln!("\r\x1b[2K\x1b[36m⚡ [{short_ts}]\x1b[0m \x1b[1m{cue_text}\x1b[0m");
                            let _ = std::io::stderr().flush();
                        }

                        // Next speech will get a new ID
                        *last_id.lock().unwrap() = uuid::Uuid::new_v4().to_string();
                        current_preview = None;
                    }
                }

                if !config.stdout_only {
                    let spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len()];
                    let normalized = (current_rms / 0.12).clamp(0.0, 1.0).sqrt();
                    let filled = (normalized * 6.0).round() as usize;
                    let empty = 6_usize.saturating_sub(filled);
                    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));
                    let bytes_str = format_bytes(audio_bytes);

                    let status = if let Some(ref text) = current_preview {
                        let max_len = 35;
                        let count = text.chars().count();
                        let truncated = if count > max_len {
                            let tail: String = text.chars().skip(count - max_len).collect();
                            format!("...{tail}")
                        } else {
                            text.clone()
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

                unsafe {
                    wisp_applespeech_feed(engine_ptr, samples.as_ptr(), samples.len(), 16000.0);
                }
            }
        }
    }

    if !config.stdout_only {
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

#[cfg(feature = "moonshine")]
pub async fn run_live_moonshine(config: LiveConfig) -> Result<Transcript> {
    let lang_str = config.lang.as_deref().unwrap_or("es");
    let lang_lower = lang_str.trim().to_ascii_lowercase();

    let (lang_code, arch, model_label) = match lang_lower.as_str() {
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
    };

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
            eprintln!("  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)", config.sub_port);
        }
        eprintln!("  • Language : {lang_code}");
        eprintln!("  • Status   : Checking model assets...\n");
    }

    // Resolve dependencies and auto-download if missing
    let deps_json = moonshine_rs::get_stt_dependencies(lang_code, Some(arch), false)
        .map_err(|e| anyhow!("Failed to resolve Moonshine model dependencies: {e:?}"))?;
    let v: serde_json::Value = serde_json::from_str(&deps_json)?;
    let files = v["groups"][0]["files"].as_array()
        .ok_or_else(|| anyhow!("Invalid Moonshine manifest structure"))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()?;

    for f in files {
        let name = f["name"].as_str().unwrap_or_default();
        let url = f["url"].as_str().unwrap_or_default();
        let target = cache_dir.join(name);
        if !target.exists() {
            if !config.stdout_only {
                eprintln!("  📥 Downloading Moonshine asset: {name} ...");
            }
            let mut resp = client.get(url).send()?.error_for_status()?;
            let mut out = File::create(&target)?;
            resp.copy_to(&mut out)?;
        }
    }

    if !config.stdout_only {
        eprintln!("  ✅ Model assets verified. Starting session...\n");
    }

    let transcriber = Arc::new(moonshine_rs::Transcriber::from_files(&cache_dir, arch, None)
        .map_err(|e| anyhow!("Failed to load Moonshine model from {}: {e:?}", cache_dir.display()))?);
    let mut stream = transcriber.create_owned_stream()
        .map_err(|e| anyhow!("Failed to initialize Moonshine streaming session: {e:?}"))?;

    let session_start = Instant::now();
    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(VecDeque::<serde_json::Value>::with_capacity(32)));
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
    let mut committed_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut all_utterances: Vec<Utternace> = Vec::new();
    let mut last_preview_text: String = String::new();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
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
            }
            _ = hud_interval.tick() => {
                tick = tick.wrapping_add(1);
                recent_peak *= 0.92;

                let is_quiet = last_audio_instant.elapsed() > Duration::from_millis(500);
                let force_poll = is_quiet && current_preview.is_some();

                if let Ok(transcript) = stream.poll(force_poll) {
                    for line in &transcript.lines {
                        let text = line.text.trim();
                        if text.is_empty() || is_hallucination(text) {
                            continue;
                        }

                        let start_sec = line.start_time;
                        let duration_sec = line.duration.max(0.5);
                        let start_ms = (start_sec * 1000.0) as i64;
                        let duration_ms = (duration_sec * 1000.0) as u64;
                        let full_ts = format_timestamp(start_ms / 10, false, ".");
                        let short_ts = if full_ts.len() >= 5 { full_ts[..5].to_string() } else { full_ts };

                        let cue_val = serde_json::json!({
                            "id": format!("ms-{}", line.id),
                            "text": text,
                            "timestamp": short_ts,
                            "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
                            "durationMs": duration_ms,
                            "lagMs": 35,
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
                    let spinner = SPINNER_FRAMES[tick % SPINNER_FRAMES.len()];
                    let norm_rms = (current_rms / 0.12).clamp(0.0, 1.0).sqrt();
                    let filled = (norm_rms * 8.0).round() as usize;
                    let empty = 8_usize.saturating_sub(filled);
                    let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

                    let preview_hud = if let Some(ref p) = current_preview {
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

#[cfg(target_os = "macos")]
pub async fn run_live_subtitles(
    transcriber: Arc<Box<dyn Transcriber>>,
    config: LiveConfig,
) -> Result<Transcript> {
    if config.apple {
        return run_live_apple_speech(config).await;
    }
    if config.moonshine {
        return run_live_moonshine(config).await;
    }

    let (_stream, mut audio_rx) = MacOsAudioStream::start()?;
    let session_start = Instant::now();

    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(std::sync::Mutex::new(VecDeque::<serde_json::Value>::with_capacity(32)));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    let is_realtime = config.preset == Some(crate::server::ServerPreset::Realtime);
    let mut ramp = AdaptiveRampState::new(is_realtime, config.speech_mode, config.chunk_secs);

    if !config.stdout_only {
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
            eprintln!("  • Subtitles: http://127.0.0.1:{}/events (Browser Injector)", config.sub_port);
        }
        eprintln!("  • Language : {mode_str}");
        eprintln!("  • Mode     : {window_desc} (Press Ctrl+C to stop)\n");
    }

    let (segment_tx, mut segment_rx) = mpsc::channel::<AudioSegment>(8);
    let is_transcribing = Arc::new(AtomicBool::new(false));
    let is_transcribing_ui = Arc::clone(&is_transcribing);

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

    // Background worker that transcribes queued speech segments sequentially in order
    let translate = config.translate;
    let karaoke = config.karaoke;
    let transcriber_worker = Arc::clone(&transcriber);
    let sub_broadcast_worker = sub_broadcast_tx.clone();
    let sub_history_worker = Arc::clone(&sub_history);
    let session_start_worker = session_start;

    let worker_handle = tokio::spawn(async move {
        let mut all_utterances: Vec<Utternace> = Vec::new();
        let mut last_printed = String::new();
        let mut prompt_context = String::new();

        while let Some(seg) = segment_rx.recv().await {
            is_transcribing.store(true, Ordering::Relaxed);

            if seg.is_cold_start {
                prompt_context.clear();
            }

            let prompt_opt = if prompt_context.trim().is_empty() {
                None
            } else {
                Some(prompt_context.as_str())
            };

            if let Ok(wav) = TempWavFile::new(&seg.samples) {
                match transcriber_worker
                    .transcribe_with_prompt(wav.path(), translate, karaoke, prompt_opt)
                    .await
                {
                    Ok(transcript) => {
                        let subtitle = clean_subtitle_text(&transcript);
                        if !subtitle.is_empty() && subtitle != last_printed {
                            update_prompt_context(&mut prompt_context, &subtitle);
                            if stdout_only {
                                println!("{subtitle}");
                                let _ = std::io::stdout().flush();
                            } else {
                                let ts = format_timestamp(seg.start_offset_ms / 10, false, ".");
                                let short_ts = ts.split('.').next().unwrap_or(&ts);
                                eprint!("\r\x1b[2K");
                                println!("\x1b[2m[{short_ts}]\x1b[0m  \x1b[1m{subtitle}\x1b[0m");
                                let _ = std::io::stdout().flush();
                            }
                            let ts = format_timestamp(seg.start_offset_ms / 10, false, ".");
                            let short_ts = ts.split('.').next().unwrap_or(&ts);
                            let now_ms = session_start_worker.elapsed().as_millis() as i64;
                            let audio_duration_ms = (seg.samples.len() * 1000 / SAMPLE_RATE) as u64;
                            let duration_ms = audio_duration_ms.max((seg.end_offset_ms - seg.start_offset_ms).max(0) as u64);
                            let lag_ms = (now_ms - seg.end_offset_ms).max(0) as u64;
                            let cue = serde_json::json!({
                                "id": uuid::Uuid::new_v4().to_string(),
                                "text": &subtitle,
                                "timestamp": short_ts,
                                "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
                                "durationMs": duration_ms,
                                "lagMs": lag_ms,
                            });
                            {
                                let mut hist = sub_history_worker.lock().unwrap();
                                if hist.len() >= 25 {
                                    hist.pop_front();
                                }
                                hist.push_back(cue.clone());
                            }
                            let _ = sub_broadcast_worker.send(cue.to_string());
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

            is_transcribing.store(false, Ordering::Relaxed);
        }

        all_utterances
    });

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
                    // No packets >= 0.005 arrived recently; decay meters & clear stale pre-roll
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

                // When < 0.005 packets are gated in macos_audio.m, flush active speech once silence duration is reached
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
                            let was_cold = ramp.record_emission();
                            let _ = segment_tx.try_send(AudioSegment {
                                samples: chunk,
                                start_offset_ms: segment_start_ms,
                                end_offset_ms: now_ms,
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
                // Guard against silent packets (< 0.005)
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
                    // In --speech mode, require vocal-band + neural speech detection (ignores pure SFX/music)
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

                // Currently in an active speech segment
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
                // In --speech mode, if a sentence is already > 7.5s long, split on any 0.24s breath pause
                let long_sentence_breath_ready = speech_mode
                    && speech_buffer.len() >= soft_long_sentence_samples
                    && silence_samples >= breath_pause_samples;
                let short_phrase_ready = active_speech_samples >= short_phrase_min_samples
                    && silence_samples >= long_silence_samples;
                let dropped_noise = active_speech_samples < short_phrase_min_samples
                    && silence_samples >= long_silence_samples;

                if dropped_noise {
                    // Brief click/SFX burst that wasn't sustained speech; reset
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
                        let was_cold = ramp.record_emission();
                        let _ = segment_tx.try_send(AudioSegment {
                            samples: chunk,
                            start_offset_ms: segment_start_ms,
                            end_offset_ms: now_ms,
                            is_cold_start: was_cold,
                        });
                    }
                } else if speech_buffer.len() >= max_chunk_samples {
                    // Reached safety max window: split at quietest micro-pause in last 0.50s
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hallucination_filter() {
        assert!(is_hallucination("[Music]"));
        assert!(is_hallucination("(Laughter)"));
        assert!(is_hallucination("Thank you for watching!"));
        assert!(is_hallucination("¡Suscríbete al canal!"));
        assert!(is_hallucination("ご視聴ありがとうございました。"));
        assert!(is_hallucination(
            "Subtítulos realizados por la comunidad de Amara.org"
        ));
        assert!(!is_hallucination("Hola, ¿cómo estás?"));
        assert!(!is_hallucination("これはテストです。"));
    }

    #[test]
    fn test_speech_detector_voice_vs_sfx() {
        let mut detector = SpeechDetector::new();
        // Silence -> not speech
        let silence = vec![0.0_f32; 800];
        let (is_speech, _, _) = detector.analyze(&silence, 0.0);
        assert!(!is_speech);

        // Simulated 400Hz vowel harmonic with high ANE speech confidence -> speech
        let voice: Vec<f32> = (0..800)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 400.0 / 16000.0).sin() * 0.15)
            .collect();
        let _ = detector.analyze(&voice, 0.85);
        let (is_speech_voice, prob_voice, _) = detector.analyze(&voice, 0.85);
        assert!(is_speech_voice, "expected speech, got prob={prob_voice}");
    }

    #[test]
    fn test_temp_wav_encoding() {
        let samples: Vec<f32> = (0..1600)
            .map(|i| (i as f32 * 0.1).sin() * 0.2)
            .collect();
        let wav = TempWavFile::new(&samples).expect("should create temp wav");
        let bytes = std::fs::read(wav.path()).expect("should read temp wav");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
    }

    #[test]
    fn test_find_best_split_index() {
        let mut samples = vec![0.2_f32; 8000];
        for s in &mut samples[6400..6900] {
            *s = 0.0001;
        }
        let split = find_best_split_index(&samples, 2000);
        assert!(split >= 6400 && split <= 6900, "split was {split}");
    }
}
