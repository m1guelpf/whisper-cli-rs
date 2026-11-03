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
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

#[derive(Debug, Deserialize)]
struct CustomSegment {
    #[serde(default)]
    start: f64,
    #[serde(default)]
    end: f64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct CustomWord {
    word: String,
    start: f64,
    end: f64,
}

#[derive(Debug, Deserialize)]
struct CustomVerboseResponse {
    #[serde(default)]
    segments: Vec<CustomSegment>,
    #[serde(default)]
    words: Vec<CustomWord>,
    #[serde(default)]
    text: String,
}

pub struct CustomApiProvider {
    client: Client,
    api_url: String,
    api_key: Option<String>,
    model: String,
    lang: Option<String>,
    use_inference_endpoint: AtomicBool,
}

impl CustomApiProvider {
    pub fn new(
        api_url: String,
        api_key: Option<String>,
        model: Option<String>,
        lang: Option<String>,
    ) -> Self {
        let is_inference = api_url.trim_end_matches('/').ends_with("/inference");
        Self {
            client: Client::new(),
            api_url,
            api_key,
            model: model.unwrap_or_else(|| "whisper-1".to_string()),
            lang,
            use_inference_endpoint: AtomicBool::new(is_inference),
        }
    }

    async fn build_form(
        &self,
        audio_path: &Path,
        file_name: &str,
        translate: bool,
        word_timestamps: bool,
        is_inference: bool,
        prompt: Option<&str>,
    ) -> Result<Form> {
        let file = File::open(audio_path).await?;
        let stream = FramedRead::new(file, BytesCodec::new());
        let file_part =
            Part::stream(reqwest::Body::wrap_stream(stream)).file_name(file_name.to_string());

        let mut form = Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("response_format", "verbose_json");

        if let Some(lang) = &self.lang {
            form = form.text("language", lang.clone());
        }

        if is_inference && translate {
            form = form.text("translate", "true");
        }

        if let Some(p) = prompt {
            let trimmed = p.trim();
            if !trimmed.is_empty() {
                form = form.text("prompt", trimmed.to_string());
            }
        }

        if word_timestamps && !is_inference {
            form = form.text("timestamp_granularities[]", "word");
        }

        Ok(form)
    }
}

#[async_trait]
impl Transcriber for CustomApiProvider {
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
        let file_name = audio_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("audio.wav")
            .to_string();

        let base_url = self.api_url.trim_end_matches('/').trim_end_matches("/v1");
        let inference_endpoint = format!("{base_url}/inference");

        let use_inference = self.use_inference_endpoint.load(Ordering::Relaxed);
        let endpoint = if use_inference {
            inference_endpoint.clone()
        } else {
            let endpoint_suffix = if translate {
                "/audio/translations"
            } else {
                "/audio/transcriptions"
            };
            let mut ep = self.api_url.trim_end_matches('/').to_string();
            if !ep.ends_with("/v1")
                && !ep.ends_with("/audio/transcriptions")
                && !ep.ends_with("/audio/translations")
            {
                ep.push_str("/v1");
            }
            if !ep.ends_with("/audio/transcriptions") && !ep.ends_with("/audio/translations") {
                ep.push_str(endpoint_suffix);
            }
            ep
        };

        let form = self
            .build_form(audio_path, &file_name, translate, word_timestamps, use_inference, prompt)
            .await?;

        let mut req = self.client.post(&endpoint).multipart(form);
        if let Some(key) = &self.api_key {
            req = req.header("Authorization", format!("Bearer {key}"));
        }

        let st = Instant::now();
        let mut res = req.send().await?;

        // Fallback to /inference if /v1/audio/transcriptions is not found (whisper-server default)
        if !use_inference
            && res.status() == reqwest::StatusCode::NOT_FOUND
            && endpoint.contains("/v1/audio/")
        {
            let form_alt = self
                .build_form(audio_path, &file_name, translate, word_timestamps, true, prompt)
                .await?;

            let mut req_alt = self.client.post(&inference_endpoint).multipart(form_alt);
            if let Some(key) = &self.api_key {
                req_alt = req_alt.header("Authorization", format!("Bearer {key}"));
            }
            if let Ok(alt_res) = req_alt.send().await {
                if alt_res.status().is_success() {
                    self.use_inference_endpoint.store(true, Ordering::Relaxed);
                    res = alt_res;
                }
            }
        }

        if !res.status().is_success() {
            let err_text = res.text().await.unwrap_or_default();
            return Err(anyhow!("Custom API error: {}", err_text));
        }

        let response: CustomVerboseResponse = res.json().await?;

        let mut utterances: Vec<Utternace> = response
            .segments
            .into_iter()
            .map(|s| Utternace {
                start: (s.start * 100.0) as i64,
                stop: (s.end * 100.0) as i64,
                text: s.text,
            })
            .collect();

        if utterances.is_empty() && !response.text.trim().is_empty() {
            utterances.push(Utternace {
                start: 0,
                stop: 0,
                text: response.text,
            });
        }

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
