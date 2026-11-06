pub mod audio;
pub mod clock;
pub mod config;
pub mod engine_apple;
// pub mod engine_lfm;
pub mod engine_moonshine;
pub mod engine_parakeet;
pub mod engine_vad;
pub mod hud;
pub mod ramp;
pub mod sub_server;
pub mod text_filter;
pub mod vad;

pub use audio::*;
pub use clock::*;
pub use config::*;
pub use engine_apple::*;
// pub use engine_lfm::*;
pub use engine_moonshine::*;
pub use engine_parakeet::*;
pub use engine_vad::*;
pub use hud::*;
pub use ramp::*;
pub use sub_server::*;
pub use text_filter::*;
pub use vad::*;

#[cfg(test)]
mod bench_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hallucination_filter() {
        assert!(is_hallucination("[Music]"));
        assert!(is_hallucination("(Laughter)"));
        assert!(is_hallucination("Thank you for watching!"));
        assert!(is_hallucination("¡Suscríbete al canal!"));
        assert!(is_hallucination("ご視聴ありがとうございました。"));
        assert!(is_hallucination(
            "Subtítulos realizados por la comunidad de Amara.org"
        ));
        assert!(!is_hallucination("Hola, ¿cómo estás?"));
        assert!(!is_hallucination("これはテストです。"));
    }

    #[test]
    fn test_speech_detector_voice_vs_sfx() {
        let mut detector = SpeechDetector::new();
        // Silence -> not speech
        let silence = vec![0.0_f32; 800];
        let (is_speech, _, _) = detector.analyze(&silence, 0.0);
        assert!(!is_speech);

        // Simulated 400Hz vowel harmonic with high ANE speech confidence -> speech
        let voice: Vec<f32> = (0..800)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 400.0 / 16000.0).sin() * 0.15)
            .collect();
        let _ = detector.analyze(&voice, 0.85);
        let (is_speech_voice, prob_voice, _) = detector.analyze(&voice, 0.85);
        assert!(is_speech_voice, "expected speech, got prob={prob_voice}");
    }

    #[test]
    fn test_temp_wav_encoding() {
        let samples: Vec<f32> = (0..1600)
            .map(|i| (i as f32 * 0.1).sin() * 0.2)
            .collect();
        let wav = TempWavFile::new(&samples).expect("should create temp wav");
        let bytes = std::fs::read(wav.path()).expect("should read temp wav");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 44 + samples.len() * 2);
    }

    #[test]
    fn test_find_best_split_index() {
        let mut samples = vec![0.2_f32; 8000];
        for s in &mut samples[6400..6900] {
            *s = 0.0001;
        }
        let split = find_best_split_index(&samples, 2000);
        assert!(split >= 6400 && split <= 6900, "split was {split}");
    }

    #[test]
    fn test_sample_clock_tracker_exact_clip_bounds() {
        let mut tracker = SampleClockTracker::new();

        // Line 1: spoken from 0ms to 2000ms (4 chunks of 8000 samples = 500ms each)
        for step in 1..=4 {
            tracker.record_feed(8000, step * 500, true);
        }
        // Trailing synthetic silence fed from 2000ms to 2360ms (6 chunks of 960 samples = 60ms each)
        for s in 1..=6 {
            tracker.record_feed(960, 2000 + s * 60, false);
        }
        // Finalize Line 1 at 2360ms (audioTimeRange: 0.0s .. 2.0s)
        let (s1, e1) = tracker.resolve_clip_bounds(0.0, 2.0, 2360, true);
        assert_eq!(s1, 0);
        assert_eq!(e1, 2000);

        // Now 17.64 seconds of silence pass in the video (no packets arrive).
        // Line 2 starts at 20,000ms and ends at 21,500ms (3 chunks of 8000 samples = 500ms each).
        tracker.record_feed(8000, 20_500, true);
        // Partial 1 arrives at 20,520ms
        let (p2_start, p2_end) = tracker.resolve_clip_bounds(2.36, 0.5, 20_520, false);
        assert_eq!(p2_start, 20_000, "Line 2 must start at 20,000ms, not 2,000ms");
        assert_eq!(p2_end, 20_500);

        tracker.record_feed(8000, 21_000, true);
        tracker.record_feed(8000, 21_500, true);

        // Trailing synthetic silence fed from 21,500ms to 21,860ms
        for s in 1..=6 {
            tracker.record_feed(960, 21_500 + s * 60, false);
        }
        // Finalize Line 2 at 22,000ms (500ms after last real audio byte at 21,500ms)
        let (s2, e2) = tracker.resolve_clip_bounds(2.36, 1.5, 22_000, true);
        assert_eq!(s2, 20_000, "Line 2 final start should be 20,000ms");
        assert_eq!(e2, 21_500, "Line 2 final end should exclude synthetic trailing silence");
    }

    #[test]
    fn test_sample_clock_tracker_back_to_back_sentences() {
        let mut tracker = SampleClockTracker::new();

        // Continuous audio from 0ms to 4000ms (8 chunks of 8000 samples = 500ms each)
        // Character A speaks Sentence 1 from 0.0s .. 1.4s
        // Character B speaks Sentence 2 from 1.5s .. 3.2s with no silence gap
        for step in 1..=8 {
            tracker.record_feed(8000, step * 500, true);
        }

        // At t = 2000ms, Sentence 1 (0.0s .. 1.4s) is finalized mid-stream while Sentence 2 (1.5s .. 2.0s) is already streaming
        let (s1, e1) = tracker.resolve_clip_bounds(0.0, 1.4, 2000, true);
        assert_eq!(s1, 0);
        assert_eq!(e1, 1400);

        // Immediately after at t = 2000ms, Sentence 2 partial (1.5s .. 2.0s) resolves its bounds:
        let (s2_part, e2_part) = tracker.resolve_clip_bounds(1.5, 0.5, 2000, false);
        assert_eq!(s2_part, 1500, "Sentence 2 must start at 1500ms even though 2000ms of audio was already fed");
        assert_eq!(e2_part, 2000);

        // At t = 4000ms, Sentence 2 finalizes (1.5s .. 3.2s)
        let (s2_final, e2_final) = tracker.resolve_clip_bounds(1.5, 1.7, 4000, true);
        assert_eq!(s2_final, 1500);
        assert_eq!(s2_final + 1700, e2_final);
    }
}
