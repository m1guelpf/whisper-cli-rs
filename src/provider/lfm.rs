use crate::{
    live::engine_lfm::{
        build_lfm_messages, clean_lfm_text, ensure_ollama_and_lfm_model, resolve_lfm_model_name,
        resolve_ollama_url,
    },
    provider::Transcriber,
    transcript::{Transcript, Utternace},
    utils::base64_encode,
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

pub struct LfmProvider {
    base_url: String,
    model: String,
    lang: Option<String>,
}

impl LfmProvider {
    pub fn new(base_url: Option<String>, model: Option<String>, lang: Option<String>) -> Self {
        Self {
            base_url: resolve_ollama_url(base_url.as_deref()),
            model: resolve_lfm_model_name(model.as_deref()),
            lang,
        }
    }
}

#[async_trait]
impl Transcriber for LfmProvider {
    async fn transcribe(
        &self,
        audio_path: &Path,
        translate: bool,
        _word_timestamps: bool,
    ) -> Result<Transcript> {
        self.transcribe_with_prompt(audio_path, translate, false, None)
            .await
    }

    async fn transcribe_with_prompt(
        &self,
        audio_path: &Path,
        translate: bool,
        _word_timestamps: bool,
        prompt: Option<&str>,
    ) -> Result<Transcript> {
        let _managed = ensure_ollama_and_lfm_model(&self.base_url, &self.model, false).await?;
        let start = Instant::now();

        let wav_bytes = fs::read(audio_path)?;
        let audio_b64 = base64_encode(&wav_bytes);
        let messages = build_lfm_messages(self.lang.as_deref(), translate, audio_b64);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;

        let resp = client
            .post(format!("{}/api/chat", self.base_url))
            .json(&serde_json::json!({
                "model": &self.model,
                "messages": messages,
                "stream": false,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.0,
                }
            }))
            .send()
            .await?
            .error_for_status()?;

        let body: serde_json::Value = resp.json().await?;
        let raw_text = body["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("Unexpected Ollama response format: {body}"))?;
        let text = clean_lfm_text(raw_text);

        let utterances = if text.is_empty() {
            Vec::new()
        } else {
            vec![Utternace {
                start: 0,
                stop: (start.elapsed().as_millis() as i64 / 10).max(100),
                text,
            }]
        };

        Ok(Transcript {
            processing_time: start.elapsed(),
            utterances,
            word_utterances: None,
        })
    }
}
