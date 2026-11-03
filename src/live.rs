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
            format!("\x1b[32m🗣️ {speech_secs:.1}s (Voice {speech_pct}%)\x1b[0m")
        } else {
            format!("\x1b[32m🗣️ {speech_secs:.1}s/{max_secs:.0}s\x1b[0m")
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
}

pub async fn start_sub_server(
    port: u16,
    sub_tx: tokio::sync::broadcast::Sender<String>,
    history: Arc<tokio::sync::Mutex<VecDeque<serde_json::Value>>>,
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
                        let hist = history_client.lock().await;
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
                    {
                        let hist = history_client.lock().await;
                        for cue in hist.iter() {
                            let sse_chunk = format!("data: {cue}\n\n");
                            if socket.write_all(sse_chunk.as_bytes()).await.is_err() {
                                return;
                            }
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
pub async fn run_live_subtitles(
    transcriber: Arc<Box<dyn Transcriber>>,
    config: LiveConfig,
) -> Result<Transcript> {
    let (_stream, mut audio_rx) = MacOsAudioStream::start()?;
    let session_start = Instant::now();

    let (sub_broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(64);
    let sub_history = Arc::new(tokio::sync::Mutex::new(VecDeque::<serde_json::Value>::with_capacity(32)));
    if config.sub_port > 0 {
        start_sub_server(config.sub_port, sub_broadcast_tx.clone(), Arc::clone(&sub_history)).await;
    }

    let is_realtime = config.preset == Some(crate::server::ServerPreset::Realtime);

    // In realtime preset, enforce tight 1.2s-2.5s chunks; in --speech mode allow up to 15s sentences
    let effective_max_secs = if is_realtime {
        config.chunk_secs.clamp(1.2, 2.5)
    } else if config.speech_mode {
        config.chunk_secs.max(15.0)
    } else {
        config.chunk_secs.clamp(1.5, 15.0)
    };

    if !config.stdout_only {
        let lang_str = config.lang.as_deref().unwrap_or("auto-detect");
        let mode_str = if config.translate {
            format!("{lang_str} -> English (translate)")
        } else {
            lang_str.to_string()
        };
        let window_desc = if is_realtime {
            format!("Near-Real-Time Streaming ({effective_max_secs:.1}s rolling chunks, 0.18s micro-pauses)")
        } else if config.speech_mode {
            format!("Smart Speech VAD (waits for sentence end up to {effective_max_secs:.0}s, filters SFX/music)")
        } else {
            format!("{:.1}s adaptive VAD chunks", effective_max_secs)
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

    let max_chunk_samples = (effective_max_secs * SAMPLE_RATE as f32) as usize;
    let soft_long_sentence_samples = if is_realtime {
        (1.5 * SAMPLE_RATE as f32) as usize
    } else {
        (7.5 * SAMPLE_RATE as f32) as usize
    };
    let min_pause_split_samples = if is_realtime {
        (0.55 * SAMPLE_RATE as f32) as usize
    } else {
        (1.10 * SAMPLE_RATE as f32) as usize
    };
    let short_phrase_min_samples = if is_realtime {
        (0.20 * SAMPLE_RATE as f32) as usize
    } else {
        (0.32 * SAMPLE_RATE as f32) as usize
    };
    let pause_silence_samples = if is_realtime {
        // Fast 0.18s micro-pause split for rapid dialogue
        (0.18 * SAMPLE_RATE as f32) as usize
    } else if config.speech_mode {
        // Wait ~0.46s of non-speech so natural mid-sentence word gaps aren't cut prematurely
        (0.46 * SAMPLE_RATE as f32) as usize
    } else {
        (0.36 * SAMPLE_RATE as f32) as usize
    };
    let breath_pause_samples = if is_realtime {
        (0.14 * SAMPLE_RATE as f32) as usize
    } else {
        (0.24 * SAMPLE_RATE as f32) as usize
    };
    let long_silence_samples = if is_realtime {
        (0.35 * SAMPLE_RATE as f32) as usize
    } else {
        (0.68 * SAMPLE_RATE as f32) as usize
    };
    let pre_roll_capacity = if is_realtime {
        (0.20 * SAMPLE_RATE as f32) as usize
    } else {
        (0.35 * SAMPLE_RATE as f32) as usize
    };
    let overlap_samples = if is_realtime {
        (0.10 * SAMPLE_RATE as f32) as usize
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

    let worker_handle = tokio::spawn(async move {
        let mut all_utterances: Vec<Utternace> = Vec::new();
        let mut last_printed = String::new();

        while let Some(seg) = segment_rx.recv().await {
            is_transcribing.store(true, Ordering::Relaxed);

            if let Ok(wav) = TempWavFile::new(&seg.samples) {
                match transcriber_worker
                    .transcribe(wav.path(), translate, karaoke)
                    .await
                {
                    Ok(transcript) => {
                        let subtitle = clean_subtitle_text(&transcript);
                        if !subtitle.is_empty() && subtitle != last_printed {
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
                            let cue = serde_json::json!({
                                "id": uuid::Uuid::new_v4().to_string(),
                                "text": &subtitle,
                                "timestamp": short_ts,
                                "receivedAt": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
                            });
                            {
                                let mut hist = sub_history_worker.lock().await;
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
    let mut speech_buffer: Vec<f32> = Vec::with_capacity(max_chunk_samples + 4096);
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

                // When < 0.005 packets are gated in macos_audio.m, flush active speech once silence duration is reached
                if speaking {
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
                            let _ = segment_tx.try_send(AudioSegment {
                                samples: chunk,
                                start_offset_ms: segment_start_ms,
                                end_offset_ms: now_ms,
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

                    let hud = render_live_hud(
                        tick,
                        current_rms,
                        recent_peak,
                        current_speech_prob,
                        &spark_history,
                        speaking,
                        speech_secs,
                        effective_max_secs,
                        in_flight,
                        bytes,
                        pkts,
                        speech_mode,
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
                        let _ = segment_tx.try_send(AudioSegment {
                            samples: chunk,
                            start_offset_ms: segment_start_ms,
                            end_offset_ms: now_ms,
                        });
                    }
                } else if speech_buffer.len() >= max_chunk_samples {
                    // Reached safety max window: split at quietest micro-pause in last 0.65s
                    let search_win = (0.65 * SAMPLE_RATE as f32) as usize;
                    let split_at = find_best_split_index(&speech_buffer, search_win);

                    let chunk = speech_buffer[..split_at].to_vec();
                    let keep_from = split_at.saturating_sub(overlap_samples);
                    let remainder = speech_buffer[keep_from..].to_vec();

                    if compute_rms(&chunk) >= 0.005 || compute_peak(&chunk) >= 0.015 {
                        let _ = segment_tx.try_send(AudioSegment {
                            samples: chunk,
                            start_offset_ms: segment_start_ms,
                            end_offset_ms: now_ms,
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
    if speaking
        && active_speech_samples >= short_phrase_min_samples
        && (compute_rms(&speech_buffer) >= 0.005 || compute_peak(&speech_buffer) >= 0.015)
    {
        let now_ms = session_start.elapsed().as_millis() as i64;
        let _ = segment_tx
            .send(AudioSegment {
                samples: speech_buffer,
                start_offset_ms: segment_start_ms,
                end_offset_ms: now_ms,
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
