use std::time::{Duration, Instant};

const SAMPLE_RATE: usize = 16_000;

#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub samples: Vec<f32>,
    pub start_offset_ms: i64,
    pub end_offset_ms: i64,
    pub is_cold_start: bool,
}

pub struct AdaptiveRampState {
    pub is_realtime: bool,
    pub speech_mode: bool,
    pub base_chunk_secs: f32,
    pub consecutive_utterances: usize,
    pub last_utterance_end: Instant,
}

impl AdaptiveRampState {
    pub fn new(is_realtime: bool, speech_mode: bool, base_chunk_secs: f32) -> Self {
        Self {
            is_realtime,
            speech_mode,
            base_chunk_secs,
            consecutive_utterances: 0,
            last_utterance_end: Instant::now() - Duration::from_secs(10),
        }
    }

    pub fn check_silence_reset(&mut self) {
        if self.consecutive_utterances > 0
            && self.last_utterance_end.elapsed() > Duration::from_millis(1800)
        {
            self.consecutive_utterances = 0;
        }
    }

    pub fn current_thresholds(&self) -> (usize, usize, usize, usize, usize, usize, usize, bool) {
        let is_cold = self.consecutive_utterances == 0;

        if self.is_realtime {
            let (max_s, min_split_s, pause_s) = match self.consecutive_utterances {
                0 => (3.2_f32, 1.10_f32, 0.38_f32), // Turn 0: Cold anchor (solid sentence)
                1 => (2.2_f32, 0.70_f32, 0.26_f32), // Turn 1: Warm intermediate
                _ => (1.4_f32, 0.45_f32, 0.18_f32), // Turn 2+: Full streaming cadence
            };
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (min_split_s * SAMPLE_RATE as f32) as usize,
                (pause_s * SAMPLE_RATE as f32) as usize,
                (1.5 * SAMPLE_RATE as f32) as usize,
                (0.14 * SAMPLE_RATE as f32) as usize,
                (0.20 * SAMPLE_RATE as f32) as usize,
                (0.35 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        } else if self.speech_mode {
            let max_s = self.base_chunk_secs.max(15.0);
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (1.10 * SAMPLE_RATE as f32) as usize,
                (0.46 * SAMPLE_RATE as f32) as usize,
                (7.5 * SAMPLE_RATE as f32) as usize,
                (0.24 * SAMPLE_RATE as f32) as usize,
                (0.32 * SAMPLE_RATE as f32) as usize,
                (0.68 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        } else {
            let max_s = self.base_chunk_secs.clamp(1.5, 15.0);
            (
                (max_s * SAMPLE_RATE as f32) as usize,
                (0.80 * SAMPLE_RATE as f32) as usize,
                (0.36 * SAMPLE_RATE as f32) as usize,
                (6.0 * SAMPLE_RATE as f32) as usize,
                (0.20 * SAMPLE_RATE as f32) as usize,
                (0.30 * SAMPLE_RATE as f32) as usize,
                (0.60 * SAMPLE_RATE as f32) as usize,
                is_cold,
            )
        }
    }

    pub fn record_emission(&mut self) -> bool {
        let was_cold = self.consecutive_utterances == 0;
        self.consecutive_utterances = self.consecutive_utterances.saturating_add(1);
        self.last_utterance_end = Instant::now();
        was_cold
    }

    pub fn stage_label(&self) -> &'static str {
        if !self.is_realtime {
            ""
        } else {
            match self.consecutive_utterances {
                0 => " [Anchor ⚓]",
                1 => " [Warming 🚀]",
                _ => " [Realtime ⚡]",
            }
        }
    }
}
