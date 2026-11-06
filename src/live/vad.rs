/// Hybrid acoustic vocal-band (300Hz–3400Hz) + Apple Neural Engine (`SoundAnalysis`) speech detector.
/// Distinguishes human speech from loud anime SFX (explosions, sword clashes) and background music.
pub struct SpeechDetector {
    hp_prev_in: f32,
    hp_prev_out: f32,
    lp_prev_out: f32,
    noise_floor_rms: f32,
    smoothed_speech_prob: f32,
}

impl Default for SpeechDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechDetector {
    pub fn new() -> Self {
        Self {
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
            lp_prev_out: 0.0,
            noise_floor_rms: 0.003,
            smoothed_speech_prob: 0.0,
        }
    }

    /// Analyzes a frame of 16kHz mono audio and combines fast 20ms vocal-band acoustics
    /// with macOS `SoundAnalysis` neural speech confidence (`ane_speech_conf` in `0.0..=1.0`).
    /// Returns `(is_speech, speech_probability, vocal_band_rms)`.
    pub fn analyze(&mut self, samples: &[f32], ane_speech_conf: f32) -> (bool, f32, f32) {
        if samples.is_empty() {
            return (false, 0.0, 0.0);
        }

        // 1st-order High-Pass (~280 Hz at 16kHz) + 1st-order Low-Pass (~3400 Hz at 16kHz)
        // Isolates the human vowel/formant frequency band while rejecting sub-bass rumbles & high-freq SFX
        let hp_alpha: f32 = 0.90;
        let lp_alpha: f32 = 0.57;

        let mut total_sq: f32 = 0.0;
        let mut vocal_sq: f32 = 0.0;
        let mut zero_crossings: usize = 0;
        let mut prev_vocal: f32 = self.lp_prev_out;

        for &x in samples {
            total_sq += x * x;

            let hp = hp_alpha * (self.hp_prev_out + x - self.hp_prev_in);
            self.hp_prev_in = x;
            self.hp_prev_out = hp;

            let vocal = self.lp_prev_out + lp_alpha * (hp - self.lp_prev_out);
            self.lp_prev_out = vocal;

            vocal_sq += vocal * vocal;
            if (vocal >= 0.0 && prev_vocal < 0.0) || (vocal < 0.0 && prev_vocal >= 0.0) {
                zero_crossings += 1;
            }
            prev_vocal = vocal;
        }

        let n = samples.len() as f32;
        let total_rms = (total_sq / n).sqrt();
        let vocal_rms = (vocal_sq / n).sqrt();
        let zcr = zero_crossings as f32 / n;

        // Instant silence check: if vocal-band energy is very low, no one is speaking right now
        // (even if the 0.6s neural window from a previous word hasn't finished decaying yet).
        if vocal_rms < 0.004 && total_rms < 0.005 {
            self.noise_floor_rms = (self.noise_floor_rms * 0.95 + vocal_rms * 0.05).max(0.002);
            self.smoothed_speech_prob *= 0.60;
            return (false, self.smoothed_speech_prob, vocal_rms);
        }

        // Ratio of energy in the 300-3400Hz vocal band vs broadband energy
        let band_ratio = (vocal_rms / (total_rms + 1e-6)).clamp(0.0, 1.2);

        // Human speech ZCR in the vocal band typically sits between 0.02 (160Hz) and 0.24 (1900Hz)
        let zcr_voiced = if (0.015..=0.25).contains(&zcr) {
            1.0_f32
        } else {
            0.35_f32
        };

        // Update background music / ambient noise floor slowly when ANE says non-speech
        if ane_speech_conf < 0.20 {
            self.noise_floor_rms =
                (self.noise_floor_rms * 0.96 + vocal_rms * 0.04).clamp(0.002, 0.08);
        } else if vocal_rms < self.noise_floor_rms {
            self.noise_floor_rms =
                (self.noise_floor_rms * 0.85 + vocal_rms * 0.15).clamp(0.002, 0.08);
        }

        // How far above the continuous background music floor is the current vocal-band energy?
        let snr_above_floor = (vocal_rms / (self.noise_floor_rms * 1.45 + 0.002)).clamp(0.0, 2.5);
        let acoustic_score =
            ((band_ratio * 0.85) * zcr_voiced * (snr_above_floor * 0.65)).clamp(0.0, 1.0);

        // Combine Apple Neural Engine SoundAnalysis confidence with 20ms vocal-band score:
        // - If ANE is confident it's speech (> 0.30) AND vocal_rms is active -> high probability
        // - If ANE says it's pure SFX/music (< 0.12) even though it's loud -> suppress score
        let raw_prob = if ane_speech_conf >= 0.25 {
            (ane_speech_conf * 0.70 + acoustic_score * 0.30).clamp(0.0, 1.0)
        } else if ane_speech_conf > 0.01 {
            // ANE is active and sees mostly non-speech (music/SFX), require strong vocal-band modulation
            (ane_speech_conf * 0.55 + acoustic_score * 0.35).clamp(0.0, 1.0)
        } else {
            // Fallback during initial 300ms before first ANE window fires
            acoustic_score
        };

        self.smoothed_speech_prob = self.smoothed_speech_prob * 0.45 + raw_prob * 0.55;
        let is_speech =
            vocal_rms >= 0.0045 && (self.smoothed_speech_prob >= 0.36 || ane_speech_conf >= 0.42);

        (is_speech, self.smoothed_speech_prob, vocal_rms)
    }
}
