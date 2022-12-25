use crate::{
    ffmpeg_decoder,
    model::Model,
    transcript::{Transcript, Utternace},
};
use anyhow::{anyhow, Result};
use std::{path::Path, time::Instant};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

pub struct Whisper {
    ctx: WhisperContext,
    lang: String,
}

impl Whisper {
    pub async fn new(model: Model) -> Self {
        model.download().await;

        Self {
            ctx: WhisperContext::new(model.get_path().to_str().unwrap())
                .expect("Failed to load model."),
            lang: "auto".to_string(),
        }
    }

    #[allow(dead_code)]
    pub fn set_language(&mut self, lang: String) {
        self.lang = lang;
    }

    pub fn transcribe<P: AsRef<Path>>(&mut self, audio: P) -> Result<Transcript> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { n_past: 0 });
        params.set_translate(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(self.lang.as_str());

        let audio = ffmpeg_decoder::read_file(audio)?;

        let st = Instant::now();
        self.ctx.full(params, &audio).expect("Failed to transcribe");

        let num_segments = self.ctx.full_n_segments();
        if num_segments == 0 {
            return Err(anyhow!("No segments found"));
        };

        let mut utterances = Vec::new();
        for i in 0..num_segments {
            let segment = self
                .ctx
                .full_get_segment_text(i)
                .map_err(|e| anyhow!("failed to get segment due to {:?}", e))?;
            let start_timestamp = self.ctx.full_get_segment_t0(i);
            let end_timestamp = self.ctx.full_get_segment_t1(i);

            utterances.push(Utternace {
                start: start_timestamp,
                stop: end_timestamp,
                text: segment,
            });
        }

        Ok(Transcript {
            utterances,
            processing_time: Instant::now().duration_since(st),
        })
    }
}
