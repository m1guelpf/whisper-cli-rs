use std::collections::VecDeque;

pub const SPARK_CHARS: &[char] = &[' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
pub const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn render_sparkline(history: &VecDeque<f32>) -> String {
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
pub fn render_live_hud(
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
