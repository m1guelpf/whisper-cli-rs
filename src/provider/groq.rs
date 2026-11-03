use crate::{
    provider::Transcriber,
    transcript::{Transcript, Utternace},
};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::{
    multipart::{Form, Part},
    Client,
};
use serde::Deserialize;
use std::{path::Path, time::Instant};
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

#[derive(Debug, Deserialize)]
struct GroqVerboseSegment {
    start: f64,
    end: f64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct GroqVerboseWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Deserialize)]
struct GroqVerboseResponse {
    #[serde(default)]
    segments: Vec<GroqVerboseSegment>,
    #[serde(default)]
    words: Vec<GroqVerboseWord>,
    #[serde(default)]
    text: String,
}

pub struct GroqProvider {
    client: Client,
    api_key: String,
    model: String,
    lang: Option<String>,
}

impl GroqProvider {
    pub fn new(api_key: String, model: Option<String>, lang: Option<String>) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model: model.unwrap_or_else(|| "whisper-large-v3-turbo".to_string()),
            lang,
        }
    }
}

#[async_trait]
impl Transcriber for GroqProvider {
    async fn transcribe(
        &self,
        audio_path: &Path,
        translate: bool,
        word_timestamps: bool,
    ) -> Result<Transcript> {
        self.transcribe_with_prompt(audio_path, translate, word_timestamps, None).await
    }

    async fn transcribe_with_prompt(
        &self,
        audio_path: &Path,
        translate: bool,
        word_timestamps: bool,
        prompt: Option<&str>,
    ) -> Result<Transcript> {
        let file = File::open(audio_path).await?;
        let file_name = audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.wav")
            .to_string();

        let stream = FramedRead::new(file, BytesCodec::new());
        let file_part = Part::stream(reqwest::Body::wrap_stream(stream)).file_name(file_name);

        let endpoint = if translate {
            "https://api.groq.com/openai/v1/audio/translations"
        } else {
            "https://api.groq.com/openai/v1/audio/transcriptions"
        };

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("response_format", "verbose_json");

        if let Some(lang) = &self.lang {
            form = form.text("language", lang.clone());
        }

        if let Some(p) = prompt {
            let trimmed = p.trim();
            if !trimmed.is_empty() {
                form = form.text("prompt", trimmed.to_string());
            }
        }

        if word_timestamps {
            form = form.text("timestamp_granularities[]", "word");
        }

        let st = Instant::now();
        let res = self
            .client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        if !res.status().is_success() {
            let err_text = res.text().await.unwrap_or_default();
            return Err(anyhow!("Groq API error: {}", err_text));
        }

        let response: GroqVerboseResponse = res.json().await?;

        let utterances = response
            .segments
            .into_iter()
            .map(|s| Utternace {
                start: (s.start * 100.0) as i64,
                stop: (s.end * 100.0) as i64,
                text: s.text,
            })
            .collect();

        let word_utterances = if word_timestamps && !response.words.is_empty() {
            Some(
                response
                    .words
                    .into_iter()
                    .map(|w| Utternace {
                        start: (w.start * 100.0) as i64,
                        stop: (w.end * 100.0) as i64,
                        text: w.word,
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(Transcript {
            processing_time: Instant::now().duration_since(st),
            utterances,
            word_utterances,
        })
    }
}
