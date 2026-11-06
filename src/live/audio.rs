use crate::live::{
    hud::{format_bytes, render_live_hud},
    vad::SpeechDetector,
};
use anyhow::{anyhow, Result};
use std::{
    collections::VecDeque,
    env::temp_dir,
    ffi::{c_char, c_void},
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

#[cfg(target_os = "macos")]
use std::ffi::CStr;

pub const SAMPLE_RATE: usize = 16_000;

#[cfg(target_os = "macos")]
pub type WhisperAudioCallback =
    extern "C" fn(samples: *const f32, count: usize, user_data: *mut c_void);

#[cfg(target_os = "macos")]
extern "C" {
    pub fn whisper_macos_audio_start(
        callback: WhisperAudioCallback,
        user_data: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub fn whisper_macos_audio_stop(handle: *mut c_void);
    pub fn whisper_macos_free_error(err: *mut c_char);
    pub fn whisper_macos_audio_packets_received() -> u64;
    pub fn whisper_macos_video_packets_received() -> u64;
    pub fn whisper_macos_audio_bytes_received() -> u64;
    pub fn whisper_macos_audio_last_status() -> i32;
    pub fn whisper_macos_audio_sample_rate() -> u32;
    pub fn whisper_macos_speech_permille() -> u32;
    pub fn wisp_applespeech_available() -> bool;
    pub fn wisp_applespeech_start(
        locale: *const c_char,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, bool, f64, f64),
    ) -> *mut c_void;
    pub fn wisp_applespeech_feed(handle: *mut c_void, samples: *const f32, count: usize, rate: f64);
    pub fn wisp_applespeech_stop(handle: *mut c_void);
    pub fn wisp_parakeet_available() -> bool;
    pub fn wisp_parakeet_start(
        model_dir: *const c_char,
        lang: *const c_char,
        ctx: *mut c_void,
        callback: extern "C" fn(*mut c_void, *const c_char, bool, f64, f64, f64),
    ) -> *mut c_void;
    pub fn wisp_parakeet_is_busy(handle: *mut c_void) -> bool;
    pub fn wisp_parakeet_decode(
        handle: *mut c_void,
        samples: *const f32,
        count: usize,
        is_final: bool,
        segment_start_sec: f64,
    );
    pub fn wisp_parakeet_stop(handle: *mut c_void);
}

#[cfg(target_os = "macos")]
pub extern "C" fn audio_callback_trampoline(
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
pub struct MacOsAudioStream {
    handle: *mut c_void,
    sender_ptr: *mut mpsc::UnboundedSender<Vec<f32>>,
}

#[cfg(target_os = "macos")]
unsafe impl Send for MacOsAudioStream {}

#[cfg(target_os = "macos")]
impl MacOsAudioStream {
    pub fn start() -> Result<(Self, mpsc::UnboundedReceiver<Vec<f32>>)> {
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

pub struct TempWavFile {
    path: PathBuf,
}

impl TempWavFile {
    pub fn new(samples: &[f32]) -> Result<Self> {
        let path = temp_dir().join(format!("whisper_live_{}.wav", uuid::Uuid::new_v4()));
        write_pcm16_wav(&path, samples, SAMPLE_RATE as u32)?;
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
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

pub fn encode_pcm16_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let max_abs = samples
        .iter()
        .fold(0.0_f32, |acc, &s| acc.max(s.abs()));

    // Gently normalize low-volume streams so the model receives a clear signal
    let gain = if max_abs > 0.006 && max_abs < 0.35 {
        (0.70 / max_abs).min(8.0)
    } else {
        1.0
    };

    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36 + data_bytes;
    let byte_rate = sample_rate * 2;

    let mut buf = Vec::with_capacity(44 + samples.len() * 2);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16_u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1_u16.to_le_bytes()); // AudioFormat = 1 (PCM)
    buf.extend_from_slice(&1_u16.to_le_bytes()); // NumChannels = 1 (Mono)
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2_u16.to_le_bytes()); // BlockAlign = 2
    buf.extend_from_slice(&16_u16.to_le_bytes()); // BitsPerSample = 16
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_bytes.to_le_bytes());

    for &sample in samples {
        let scaled = (sample * gain).clamp(-1.0, 1.0);
        let pcm16 = (scaled * 32767.0) as i16;
        buf.extend_from_slice(&pcm16.to_le_bytes());
    }

    buf
}

pub fn write_pcm16_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let bytes = encode_pcm16_wav_bytes(samples, sample_rate);
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&bytes)?;
    writer.flush()?;
    Ok(())
}

pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

pub fn compute_peak(samples: &[f32]) -> f32 {
    samples
        .iter()
        .fold(0.0_f32, |acc, &s| acc.max(s.abs()))
}

/// Find the quietest 25ms window within the last `search_window` samples to split continuous speech cleanly.
pub fn find_best_split_index(samples: &[f32], search_window: usize) -> usize {
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
