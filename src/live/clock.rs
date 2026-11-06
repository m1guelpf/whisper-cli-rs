use std::collections::VecDeque;

pub const SAMPLE_RATE: usize = 16_000;

#[derive(Debug, Clone)]
pub struct FedChunk {
    pub start_sample: u64,
    pub end_sample: u64,
    pub start_session_ms: i64,
    pub end_session_ms: i64,
    pub is_real_audio: bool,
}

/// Maps cumulative sample offsets fed into a streaming recognizer (16 kHz) back to exact
/// session wall-clock milliseconds (`start_session_ms` .. `end_session_ms`), while stripping
/// silence gaps between utterances and synthetic trailing silence chunks.
#[derive(Debug, Default)]
pub struct SampleClockTracker {
    chunks: VecDeque<FedChunk>,
    total_fed_samples: u64,
    phrase_floor_sample: u64,
    active_burst_start_ms: Option<i64>,
    locked_utterance_start_ms: Option<i64>,
    last_real_audio_ms: i64,
}

impl SampleClockTracker {
    pub fn new() -> Self {
        Self {
            chunks: VecDeque::with_capacity(4096),
            total_fed_samples: 0,
            phrase_floor_sample: 0,
            active_burst_start_ms: None,
            locked_utterance_start_ms: None,
            last_real_audio_ms: 0,
        }
    }

    pub fn record_feed(&mut self, sample_count: usize, now_ms: i64, is_real_audio: bool) {
        if sample_count == 0 {
            return;
        }
        let chunk_dur_ms = ((sample_count as i64) * 1000 / SAMPLE_RATE as i64).max(1);
        let start_session_ms = (now_ms - chunk_dur_ms).max(0);
        let start_sample = self.total_fed_samples;
        let end_sample = start_sample + sample_count as u64;
        self.total_fed_samples = end_sample;

        if is_real_audio {
            // If we haven't locked an active utterance yet and there was a silence gap (>350ms)
            // since the last real audio packet, reset the burst floor to this exact first byte.
            let had_silence_gap = self.last_real_audio_ms > 0
                && (start_session_ms - self.last_real_audio_ms) > 350;
            if self.active_burst_start_ms.is_none()
                || (self.locked_utterance_start_ms.is_none() && had_silence_gap)
            {
                self.active_burst_start_ms = Some(start_session_ms);
                self.phrase_floor_sample = start_sample;
            }
            self.last_real_audio_ms = now_ms;
        }

        if self.chunks.len() >= 4096 {
            self.chunks.pop_front();
        }
        self.chunks.push_back(FedChunk {
            start_sample,
            end_sample,
            start_session_ms,
            end_session_ms: now_ms,
            is_real_audio,
        });
    }

    pub fn sample_to_real_start_ms(&self, sample_idx: u64) -> Option<i64> {
        let target = sample_idx.max(self.phrase_floor_sample);
        for c in &self.chunks {
            if !c.is_real_audio || c.end_sample <= target {
                continue;
            }
            if target <= c.start_sample {
                return Some(c.start_session_ms);
            }
            let span_samples = (c.end_sample - c.start_sample).max(1) as f64;
            let offset_samples = (target - c.start_sample) as f64;
            let span_ms = (c.end_session_ms - c.start_session_ms).max(0) as f64;
            return Some(c.start_session_ms + (span_ms * (offset_samples / span_samples)).round() as i64);
        }
        None
    }

    pub fn sample_to_real_end_ms(&self, sample_idx: u64) -> Option<i64> {
        let target = sample_idx.max(self.phrase_floor_sample);
        for c in self.chunks.iter().rev() {
            if !c.is_real_audio || c.start_sample > target {
                continue;
            }
            if target >= c.end_sample {
                return Some(c.end_session_ms);
            }
            let span_samples = (c.end_sample - c.start_sample).max(1) as f64;
            let offset_samples = (target - c.start_sample) as f64;
            let span_ms = (c.end_session_ms - c.start_session_ms).max(0) as f64;
            return Some(c.start_session_ms + (span_ms * (offset_samples / span_samples)).round() as i64);
        }
        None
    }

    /// Resolves `(clip_start_ms, clip_end_ms)` in session milliseconds for a partial or final result.
    pub fn resolve_clip_bounds(
        &mut self,
        range_start_sec: f64,
        range_dur_sec: f64,
        now_ms: i64,
        is_final: bool,
    ) -> (i64, i64) {
        let mapped_start = if range_start_sec >= 0.0 {
            let start_sample = (range_start_sec * SAMPLE_RATE as f64).round() as u64;
            self.sample_to_real_start_ms(start_sample)
        } else {
            None
        };

        let clip_start_ms = match (mapped_start, self.locked_utterance_start_ms) {
            (Some(ms), _) => {
                // Lock the utterance start on first valid measurement; allow slight earlier refinement
                let refined = match self.locked_utterance_start_ms {
                    Some(prev) => prev.min(ms).max(self.active_burst_start_ms.unwrap_or(0)),
                    None => ms,
                };
                self.locked_utterance_start_ms = Some(refined);
                refined
            }
            (None, Some(locked)) => locked,
            (None, None) => {
                let fallback = self
                    .active_burst_start_ms
                    .unwrap_or_else(|| (now_ms - 250).max(0));
                self.locked_utterance_start_ms = Some(fallback);
                fallback
            }
        };

        let mapped_end = if range_start_sec >= 0.0 && range_dur_sec > 0.0 {
            let end_sample = ((range_start_sec + range_dur_sec) * SAMPLE_RATE as f64).round() as u64;
            self.sample_to_real_end_ms(end_sample)
        } else {
            None
        };

        let clip_end_ms = match mapped_end {
            Some(ms) => ms.max(clip_start_ms + 120).min(now_ms),
            None => now_ms.max(clip_start_ms + 120),
        };

        if is_final {
            if range_start_sec >= 0.0 && range_dur_sec > 0.0 {
                let end_sample = ((range_start_sec + range_dur_sec) * SAMPLE_RATE as f64).round() as u64;
                self.phrase_floor_sample = end_sample.min(self.total_fed_samples);
                self.active_burst_start_ms = Some(clip_end_ms);
                self.locked_utterance_start_ms = None;
            } else {
                self.mark_utterance_finalized();
            }
        }

        (clip_start_ms, clip_end_ms)
    }

    pub fn mark_utterance_finalized(&mut self) {
        self.phrase_floor_sample = self.total_fed_samples;
        self.active_burst_start_ms = None;
        self.locked_utterance_start_ms = None;
    }
}
