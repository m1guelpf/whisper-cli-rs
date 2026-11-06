use crate::{
    live::engine_parakeet::{
        ensure_parakeet_model, resolve_macparakeet_cli, run_parakeet_transcribe_file,
    },
    provider::Transcriber,
    transcript::{Transcript, Utternace},
};
use anyhow::Result;
use async_trait::async_trait;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

pub struct ParakeetProvider {
    cli_path: PathBuf,
    model_variant: String,
}

impl ParakeetProvider {
    pub fn new(model_arg: Option<String>) -> Result<Self> {
        let cli_path = resolve_macparakeet_cli()?;
        let model_variant = ensure_parakeet_model(&cli_path, model_arg.as_deref())?;
        Ok(Self {
            cli_path,
            model_variant,
        })
    }
}

#[async_trait]
impl Transcriber for ParakeetProvider {
    async fn transcribe(
        &self,
        audio_path: &Path,
        _translate: bool,
        _word_timestamps: bool,
    ) -> Result<Transcript> {
        let start = Instant::now();
        let cli = self.cli_path.clone();
        let m_var = self.model_variant.clone();
        let path_buf = audio_path.to_path_buf();

        let json_resp = tokio::task::spawn_blocking(move || {
            run_parakeet_transcribe_file(&cli, &path_buf, &m_var)
        })
        .await??;

        let mut utterances = Vec::new();
        if let Some(segments) = json_resp.segments {
            for seg in segments {
                let text = seg.text.trim();
                if !text.is_empty() {
                    utterances.push(Utternace {
                        start: (seg.start_ms / 10) as i64,
                        stop: (seg.end_ms / 10) as i64,
                        text: text.to_string(),
                    });
                }
            }
        } else if let Some(raw) = json_resp.raw_transcript {
            let text = raw.trim();
            if !text.is_empty() {
                let dur = json_resp.duration_ms.unwrap_or(1000);
                utterances.push(Utternace {
                    start: 0,
                    stop: (dur / 10) as i64,
                    text: text.to_string(),
                });
            }
        }

        Ok(Transcript {
            processing_time: start.elapsed(),
            utterances,
            word_utterances: None,
        })
    }
}
