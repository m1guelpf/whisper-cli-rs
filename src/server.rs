use anyhow::{anyhow, Result};
use clap::ValueEnum;
use std::{
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ServerPreset {
    /// Proven baseline: low hallucinations, full 30s context (-t 8 -fa -nth 0.5 -mc 0 -sns)
    #[clap(name = "balanced")]
    Balanced,
    /// Tuned for real-time subtitles: ~3x faster encoder window (-ac 512) + no fallback stalls (-nf -bo 1)
    #[clap(name = "live")]
    Live,
    /// Ultra-low latency for rapid dialogue: tight 7.5s encoder window (-ac 384 -nf -bo 1 -bs 1)
    #[clap(name = "fast")]
    Fast,
    /// Near-real-time streaming: ultra-low latency (~1.8s rolling window, 0.18s pause cuts, -ac 256)
    #[clap(name = "realtime")]
    Realtime,
    /// Maximum transcription quality: 5-way beam search (-bo 5 -bs 5) + full context
    #[clap(name = "accurate")]
    Accurate,
}

impl ServerPreset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Live => "live",
            Self::Fast => "fast",
            Self::Realtime => "realtime",
            Self::Accurate => "accurate",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Balanced => {
                "Balanced (Low hallucinations, full 30s context: -t 8 -fa -nth 0.5 -mc 0 -sns)"
            }
            Self::Live => {
                "Live Subtitles (~3x faster 10s audio context, zero fallback stalls: -ac 512 -nf -bo 1)"
            }
            Self::Fast => {
                "Fast / Low-Latency (7.5s audio context, greedy single-pass: -ac 384 -nf -bo 1 -bs 1)"
            }
            Self::Realtime => {
                "Near-Real-Time Streaming (~1.8s chunks, 0.18s micro-pauses: -ac 256 -nf -bo 1 -bs 1)"
            }
            Self::Accurate => {
                "High Accuracy (Full 30s context + 5-beam search: -bo 5 -bs 5 -nth 0.55)"
            }
        }
    }

    pub fn recommended_chunk_secs(self) -> f32 {
        match self {
            Self::Realtime => 1.8,
            Self::Fast => 2.8,
            Self::Live => 3.5,
            Self::Balanced | Self::Accurate => 4.0,
        }
    }
}

pub struct ServerConfig {
    pub preset: ServerPreset,
    pub model: Option<String>,
    pub lang: Option<String>,
    pub port: u16,
    pub extra_args: Option<String>,
    pub speech_mode: bool,
}

/// Resolves a model argument (like `large-v3-turbo`, `medium`, or a file path) to a `.bin` path in `~/.cache/whisper-models`.
pub fn resolve_model_path(model_arg: Option<&str>) -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    let cache_dir = home.join(".cache").join("whisper-models");

    if let Some(m) = model_arg {
        let direct = PathBuf::from(m);
        if direct.exists() {
            return Ok(direct);
        }

        // Check shorthand names in ~/.cache/whisper-models/
        let candidates = [
            cache_dir.join(m),
            cache_dir.join(format!("{m}.bin")),
            cache_dir.join(format!("ggml-{m}.bin")),
        ];
        for cand in candidates {
            if cand.exists() {
                return Ok(cand);
            }
        }

        return Err(anyhow!(
            "Model '{}' not found directly or in {}",
            m,
            cache_dir.display()
        ));
    }

    // Default model preference order: ggml-large-v3-turbo.bin -> ggml-medium.bin -> ggml-base.bin
    let defaults = [
        "ggml-large-v3-turbo.bin",
        "ggml-medium.bin",
        "ggml-large-v3.bin",
        "ggml-small.bin",
        "ggml-base.bin",
    ];
    for name in defaults {
        let p = cache_dir.join(name);
        if p.exists() {
            return Ok(p);
        }
    }

    Ok(cache_dir.join("ggml-large-v3-turbo.bin"))
}

pub fn build_server_args(config: &ServerConfig) -> Result<(PathBuf, Vec<String>)> {
    let model_path = resolve_model_path(config.model.as_deref())?;
    let model_str = model_path
        .to_str()
        .ok_or_else(|| anyhow!("Invalid UTF-8 in model path"))?
        .to_string();

    let lang = config.lang.as_deref().unwrap_or("auto");

    let mut args = vec![
        "-m".to_string(),
        model_str,
        "--port".to_string(),
        config.port.to_string(),
        "-t".to_string(),
        "8".to_string(),
        "-fa".to_string(),
        "-mc".to_string(),
        "0".to_string(),
        "-sns".to_string(),
        "-l".to_string(),
        lang.to_string(),
    ];

    match config.preset {
        ServerPreset::Balanced => {
            args.extend(["-nth".to_string(), "0.5".to_string()]);
        }
        ServerPreset::Live => {
            let ac = if config.speech_mode { "768" } else { "512" };
            args.extend([
                "-nth".to_string(),
                "0.5".to_string(),
                "-ac".to_string(),
                ac.to_string(),
                "-nf".to_string(),
                "-bo".to_string(),
                "1".to_string(),
            ]);
        }
        ServerPreset::Fast => {
            let ac = if config.speech_mode { "768" } else { "384" };
            args.extend([
                "-nth".to_string(),
                "0.45".to_string(),
                "-ac".to_string(),
                ac.to_string(),
                "-nf".to_string(),
                "-bo".to_string(),
                "1".to_string(),
                "-bs".to_string(),
                "1".to_string(),
            ]);
        }
        ServerPreset::Realtime => {
            args.extend([
                "-nth".to_string(),
                "0.4".to_string(),
                "-ac".to_string(),
                "256".to_string(),
                "-nf".to_string(),
                "-bo".to_string(),
                "1".to_string(),
                "-bs".to_string(),
                "1".to_string(),
            ]);
        }
        ServerPreset::Accurate => {
            args.extend([
                "-nth".to_string(),
                "0.55".to_string(),
                "-bo".to_string(),
                "5".to_string(),
                "-bs".to_string(),
                "5".to_string(),
            ]);
        }
    }

    if let Some(extra) = &config.extra_args {
        for token in extra.split_whitespace() {
            args.push(token.to_string());
        }
    }

    Ok((model_path, args))
}

pub fn is_port_listening(port: u16) -> bool {
    TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(150),
    )
    .is_ok()
}

/// Run `whisper-server` in the foreground (`whisper --serve <preset>`).
pub fn run_server_foreground(config: &ServerConfig) -> Result<()> {
    let (model_path, args) = build_server_args(config)?;

    if !Path::new(&model_path).exists() {
        return Err(anyhow!(
            "Whisper model file not found at '{}'. Download it or specify a path with -m.",
            model_path.display()
        ));
    }

    eprintln!("╭──────────────────────────────────────────────────────────────╮");
    eprintln!("│ 🚀 Launching whisper-server ({:<8} preset)                │", config.preset.as_str());
    eprintln!("╰──────────────────────────────────────────────────────────────╯");
    eprintln!("  • Preset : {}", config.preset.description());
    eprintln!("  • Model  : {}", model_path.display());
    eprintln!("  • Command: whisper-server {}\n", args.join(" "));

    let status = Command::new("whisper-server")
        .args(&args)
        .status()
        .map_err(|e| anyhow!("Failed to execute 'whisper-server': {e}. Is whisper-cpp installed (`brew install whisper-cpp`)?"))?;

    if !status.success() {
        return Err(anyhow!("whisper-server exited with status: {status}"));
    }

    Ok(())
}

pub struct ManagedServer {
    child: Option<Child>,
}

impl ManagedServer {
    /// Spawns `whisper-server` in the background with the given preset and waits until it is listening.
    pub async fn start(config: &ServerConfig) -> Result<Self> {
        if is_port_listening(config.port) {
            eprintln!(
                "ℹ️  whisper-server is already listening on 127.0.0.1:{} — reusing existing server.",
                config.port
            );
            return Ok(Self { child: None });
        }

        let (model_path, args) = build_server_args(config)?;
        if !model_path.exists() {
            return Err(anyhow!(
                "Whisper model file not found at '{}'. Download it or pass -m <PATH>.",
                model_path.display()
            ));
        }

        eprintln!(
            "🚀 Starting background whisper-server [{}] on port {}...",
            config.preset.as_str(),
            config.port
        );
        eprintln!("   Command: whisper-server {}", args.join(" "));

        let mut child = Command::new("whisper-server")
            .args(&args)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| {
                anyhow!(
                    "Failed to spawn 'whisper-server': {e}. Ensure whisper-cpp is installed (`brew install whisper-cpp`)."
                )
            })?;

        let start = Instant::now();
        let timeout = Duration::from_secs(30);
        while start.elapsed() < timeout {
            if is_port_listening(config.port) {
                eprintln!(
                    "✅ whisper-server ready in {:.1}s!\n",
                    start.elapsed().as_secs_f32()
                );
                return Ok(Self { child: Some(child) });
            }
            if let Ok(Some(status)) = child.try_wait() {
                return Err(anyhow!(
                    "whisper-server exited unexpectedly during startup with status: {status}"
                ));
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        let _ = child.kill();
        Err(anyhow!(
            "Timed out waiting for whisper-server to listen on port {}",
            config.port
        ))
    }
}

impl Drop for ManagedServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_server_args_presets() {
        let cfg = ServerConfig {
            preset: ServerPreset::Balanced,
            model: Some("/tmp/test-model.bin".to_string()),
            lang: Some("es".to_string()),
            port: 8080,
            extra_args: None,
            speech_mode: false,
        };
        // Note: resolve_model_path checks existence if Some, so test with None (falls back to default path)
        let cfg_default = ServerConfig {
            preset: ServerPreset::Live,
            model: None,
            lang: Some("ja".to_string()),
            port: 8080,
            extra_args: Some("-et 2.2".to_string()),
            speech_mode: false,
        };
        let (_path, args) = build_server_args(&cfg_default).expect("should build args");
        assert!(args.contains(&"-ac".to_string()));
        assert!(args.contains(&"512".to_string()));
        assert!(args.contains(&"-nf".to_string()));
        assert!(args.contains(&"ja".to_string()));
        assert!(args.contains(&"-et".to_string()));
        assert!(args.contains(&"2.2".to_string()));
        let _ = cfg;
    }
}
