use crate::{
    ffmpeg_decoder,
    model::Model,
    transcript::{PartialTranscript, Transcript, Utternace},
};
use anyhow::{anyhow, Context, Error, Result};
use async_fn_stream::try_fn_stream;
use cpal::traits::{DeviceTrait, StreamTrait};
use futures_util::Stream;
use ringbuf::{Consumer, LocalRb, Rb, SharedRb};
use std::{cmp, mem::MaybeUninit, path::Path, sync::Arc, thread, time::Duration, time::Instant};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperToken};

const NUM_ITERS: usize = 2;
const LATENCY_MS: f32 = 5000.0;
const NUM_ITERS_SAVED: usize = 2;

type AudioStream = Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum Language {
    #[clap(name = "auto")]
    Auto,
    #[clap(name = "en")]
    English,
    #[clap(name = "zh")]
    Chinese,
    #[clap(name = "de")]
    German,
    #[clap(name = "es")]
    Spanish,
    #[clap(name = "ru")]
    Russian,
    #[clap(name = "ko")]
    Korean,
    #[clap(name = "fr")]
    French,
    #[clap(name = "ja")]
    Japanese,
    #[clap(name = "pt")]
    Portuguese,
    #[clap(name = "tr")]
    Turkish,
    #[clap(name = "pl")]
    Polish,
    #[clap(name = "ca")]
    Catalan,
    #[clap(name = "nl")]
    Dutch,
    #[clap(name = "ar")]
    Arabic,
    #[clap(name = "sv")]
    Swedish,
    #[clap(name = "it")]
    Italian,
    #[clap(name = "id")]
    Indonesian,
    #[clap(name = "hi")]
    Hindi,
    #[clap(name = "fi")]
    Finnish,
    #[clap(name = "vi")]
    Vietnamese,
    #[clap(name = "he")]
    Hebrew,
    #[clap(name = "uk")]
    Ukrainian,
    #[clap(name = "el")]
    Greek,
    #[clap(name = "ms")]
    Malay,
    #[clap(name = "cs")]
    Czech,
    #[clap(name = "ro")]
    Romanian,
    #[clap(name = "da")]
    Danish,
    #[clap(name = "hu")]
    Hungarian,
    #[clap(name = "ta")]
    Tamil,
    #[clap(name = "no")]
    Norwegian,
    #[clap(name = "th")]
    Thai,
    #[clap(name = "ur")]
    Urdu,
    #[clap(name = "hr")]
    Croatian,
    #[clap(name = "bg")]
    Bulgarian,
    #[clap(name = "lt")]
    Lithuanian,
    #[clap(name = "la")]
    Latin,
    #[clap(name = "mi")]
    Maori,
    #[clap(name = "ml")]
    Malayalam,
    #[clap(name = "cy")]
    Welsh,
    #[clap(name = "sk")]
    Slovak,
    #[clap(name = "te")]
    Telugu,
    #[clap(name = "fa")]
    Persian,
    #[clap(name = "lv")]
    Latvian,
    #[clap(name = "bn")]
    Bengali,
    #[clap(name = "sr")]
    Serbian,
    #[clap(name = "az")]
    Azerbaijani,
    #[clap(name = "sl")]
    Slovenian,
    #[clap(name = "kn")]
    Kannada,
    #[clap(name = "et")]
    Estonian,
    #[clap(name = "mk")]
    Macedonian,
    #[clap(name = "br")]
    Breton,
    #[clap(name = "eu")]
    Basque,
    #[clap(name = "is")]
    Icelandic,
    #[clap(name = "hy")]
    Armenian,
    #[clap(name = "ne")]
    Nepali,
    #[clap(name = "mn")]
    Mongolian,
    #[clap(name = "bs")]
    Bosnian,
    #[clap(name = "kk")]
    Kazakh,
    #[clap(name = "sq")]
    Albanian,
    #[clap(name = "sw")]
    Swahili,
    #[clap(name = "gl")]
    Galician,
    #[clap(name = "mr")]
    Marathi,
    #[clap(name = "pa")]
    Punjabi,
    #[clap(name = "si")]
    Sinhala,
    #[clap(name = "km")]
    Khmer,
    #[clap(name = "sn")]
    Shona,
    #[clap(name = "yo")]
    Yoruba,
    #[clap(name = "so")]
    Somali,
    #[clap(name = "af")]
    Afrikaans,
    #[clap(name = "oc")]
    Occitan,
    #[clap(name = "ka")]
    Georgian,
    #[clap(name = "be")]
    Belarusian,
    #[clap(name = "tg")]
    Tajik,
    #[clap(name = "sd")]
    Sindhi,
    #[clap(name = "gu")]
    Gujarati,
    #[clap(name = "am")]
    Amharic,
    #[clap(name = "yi")]
    Yiddish,
    #[clap(name = "lo")]
    Lao,
    #[clap(name = "uz")]
    Uzbek,
    #[clap(name = "fo")]
    Faroese,
    #[clap(name = "ht")]
    HaitianCreole,
    #[clap(name = "ps")]
    Pashto,
    #[clap(name = "tk")]
    Turkmen,
    #[clap(name = "nn")]
    Nynorsk,
    #[clap(name = "mt")]
    Maltese,
    #[clap(name = "sa")]
    Sanskrit,
    #[clap(name = "lb")]
    Luxembourgish,
    #[clap(name = "my")]
    Myanmar,
    #[clap(name = "bo")]
    Tibetan,
    #[clap(name = "tl")]
    Tagalog,
    #[clap(name = "mg")]
    Malagasy,
    #[clap(name = "as")]
    Assamese,
    #[clap(name = "tt")]
    Tatar,
    #[clap(name = "haw")]
    Hawaiian,
    #[clap(name = "ln")]
    Lingala,
    #[clap(name = "ha")]
    Hausa,
    #[clap(name = "ba")]
    Bashkir,
    #[clap(name = "jw")]
    Javanese,
    #[clap(name = "su")]
    Sundanese,
}

impl From<Language> for &str {
    #[allow(clippy::too_many_lines)]
    fn from(val: Language) -> Self {
        match val {
            Language::Auto => "auto",
            Language::English => "en",
            Language::Chinese => "zh",
            Language::German => "de",
            Language::Spanish => "es",
            Language::Russian => "ru",
            Language::Korean => "ko",
            Language::French => "fr",
            Language::Japanese => "ja",
            Language::Portuguese => "pt",
            Language::Turkish => "tr",
            Language::Polish => "pl",
            Language::Catalan => "ca",
            Language::Dutch => "nl",
            Language::Arabic => "ar",
            Language::Swedish => "sv",
            Language::Italian => "it",
            Language::Indonesian => "id",
            Language::Hindi => "hi",
            Language::Finnish => "fi",
            Language::Vietnamese => "vi",
            Language::Hebrew => "he",
            Language::Ukrainian => "uk",
            Language::Greek => "el",
            Language::Malay => "ms",
            Language::Czech => "cs",
            Language::Romanian => "ro",
            Language::Danish => "da",
            Language::Hungarian => "hu",
            Language::Tamil => "ta",
            Language::Norwegian => "no",
            Language::Thai => "th",
            Language::Urdu => "ur",
            Language::Croatian => "hr",
            Language::Bulgarian => "bg",
            Language::Lithuanian => "lt",
            Language::Latin => "la",
            Language::Maori => "mi",
            Language::Malayalam => "ml",
            Language::Welsh => "cy",
            Language::Slovak => "sk",
            Language::Telugu => "te",
            Language::Persian => "fa",
            Language::Latvian => "lv",
            Language::Bengali => "bn",
            Language::Serbian => "sr",
            Language::Azerbaijani => "az",
            Language::Slovenian => "sl",
            Language::Kannada => "kn",
            Language::Estonian => "et",
            Language::Macedonian => "mk",
            Language::Breton => "br",
            Language::Basque => "eu",
            Language::Icelandic => "is",
            Language::Armenian => "hy",
            Language::Nepali => "ne",
            Language::Mongolian => "mn",
            Language::Bosnian => "bs",
            Language::Kazakh => "kk",
            Language::Albanian => "sq",
            Language::Swahili => "sw",
            Language::Galician => "gl",
            Language::Marathi => "mr",
            Language::Punjabi => "pa",
            Language::Sinhala => "si",
            Language::Khmer => "km",
            Language::Shona => "sn",
            Language::Yoruba => "yo",
            Language::Somali => "so",
            Language::Afrikaans => "af",
            Language::Occitan => "oc",
            Language::Georgian => "ka",
            Language::Belarusian => "be",
            Language::Tajik => "tg",
            Language::Sindhi => "sd",
            Language::Gujarati => "gu",
            Language::Amharic => "am",
            Language::Yiddish => "yi",
            Language::Lao => "lo",
            Language::Uzbek => "uz",
            Language::Faroese => "fo",
            Language::HaitianCreole => "ht",
            Language::Pashto => "ps",
            Language::Turkmen => "tk",
            Language::Nynorsk => "nn",
            Language::Maltese => "mt",
            Language::Sanskrit => "sa",
            Language::Luxembourgish => "lb",
            Language::Myanmar => "my",
            Language::Tibetan => "bo",
            Language::Tagalog => "tl",
            Language::Malagasy => "mg",
            Language::Assamese => "as",
            Language::Tatar => "tt",
            Language::Hawaiian => "haw",
            Language::Lingala => "ln",
            Language::Hausa => "ha",
            Language::Bashkir => "ba",
            Language::Javanese => "jw",
            Language::Sundanese => "su",
        }
    }
}

pub struct Whisper {
    ctx: WhisperContext,
    lang: Option<Language>,
}

impl Whisper {
    pub async fn new(model: Model, lang: Option<Language>) -> Self {
        model.download().await;

        Self {
            lang,
            ctx: WhisperContext::new(model.get_path().to_str().unwrap())
                .expect("Failed to load model."),
        }
    }

    pub fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio: P,
        translate: bool,
        word_timestamps: bool,
    ) -> Result<Transcript> {
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        params.set_translate(translate);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(word_timestamps);
        params.set_language(self.lang.map(Into::into));

        let audio = ffmpeg_decoder::read_file(audio)?;

        let st = Instant::now();
        let mut state = self.ctx.create_state().expect("failed to create state");
        state.full(params, &audio).expect("failed to transcribe");

        let num_segments = state.full_n_segments().expect("failed to get segments");
        if num_segments == 0 {
            return Err(anyhow!("No segments found"));
        };

        let mut words = Vec::new();
        let mut utterances = Vec::new();
        for s in 0..num_segments {
            let text = state
                .full_get_segment_text(s)
                .map_err(|e| anyhow!("failed to get segment due to {:?}", e))?;
            let start = state
                .full_get_segment_t0(s)
                .map_err(|e| anyhow!("failed to get segment due to {:?}", e))?;
            let stop = state
                .full_get_segment_t1(s)
                .map_err(|e| anyhow!("failed to get segment due to {:?}", e))?;

            utterances.push(Utternace { text, start, stop });

            if !word_timestamps {
                continue;
            }

            let num_tokens = state
                .full_n_tokens(s)
                .map_err(|e| anyhow!("failed to get segment due to {:?}", e))?;

            for t in 0..num_tokens {
                let text = state
                    .full_get_token_text(s, t)
                    .map_err(|e| anyhow!("failed to get token due to {:?}", e))?;
                let token_data = state
                    .full_get_token_data(s, t)
                    .map_err(|e| anyhow!("failed to get token due to {:?}", e))?;

                if text.starts_with("[_") {
                    continue;
                }

                words.push(Utternace {
                    text,
                    start: token_data.t0,
                    stop: token_data.t1,
                });
            }
        }

        Ok(Transcript {
            utterances,
            processing_time: Instant::now().duration_since(st),
            word_utterances: if word_timestamps { Some(words) } else { None },
        })
    }

    pub fn listen(
        &self,
        device: cpal::Device,
    ) -> Result<impl Stream<Item = Result<PartialTranscript, anyhow::Error>> + '_> {
        let config: cpal::StreamConfig = device.default_input_config()?.into();

        let sampling_freq = config.sample_rate.0 as f32 / 2.0;
        let latency_frames = (LATENCY_MS / 1_000.0) * config.sample_rate.0 as f32;
        let latency_samples = latency_frames as usize * config.channels as usize;

        let ring = SharedRb::new(latency_samples * 2);
        let (mut producer, mut consumer) = ring.split();

        let input_stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mut output_fell_behind = false;

                for &sample in data {
                    if producer.push(sample).is_err() {
                        output_fell_behind = true;
                    }
                }

                if output_fell_behind {
                    eprintln!("output stream fell behind: try increasing latency");
                }
            },
            |err| eprintln!("an error occurred on stream: {err}"),
            None,
        )?;

        input_stream.play()?;
        consumer.pop_iter().count();

        Ok(self.transcribe_stream(sampling_freq, latency_samples, consumer))
    }

    fn transcribe_stream(
        &self,
        sampling_freq: f32,
        latency_samples: usize,
        mut consumer: AudioStream,
    ) -> impl Stream<Item = Result<PartialTranscript, anyhow::Error>> + '_ {
        try_fn_stream(|emitter| async move {
            let mut state = self.ctx.create_state()?;

            let mut iter_num_samples = LocalRb::new(NUM_ITERS);
            let mut iter_tokens = LocalRb::new(NUM_ITERS_SAVED);
            let mut iter_samples = LocalRb::new(latency_samples * NUM_ITERS * 2);
            for _ in 0..NUM_ITERS {
                iter_num_samples.push(0).map_err(Error::msg)?;
            }

            let mut start_time = Instant::now();

            let mut loop_num = 0;
            let mut words = String::new();
            let mut num_chars_to_delete = 0;

            loop {
                loop_num += 1;

                // Only run every LATENCY_MS
                let duration = start_time.elapsed();
                let latency = Duration::from_millis(LATENCY_MS as u64);
                if duration < latency {
                    let sleep_time = latency - duration;
                    thread::sleep(sleep_time);
                } else {
                    panic!("Classification got behind. It took to long. Try using a smaller model and/or more threads");
                }
                start_time = Instant::now();

                let samples: Vec<_> = consumer.pop_iter().collect();
                dbg!(&samples);
                let samples =
                    whisper_rs::convert_stereo_to_mono_audio(&samples).map_err(Error::msg)?;

                let num_samples_to_delete = iter_num_samples
                    .push_overwrite(samples.len())
                    .ok_or_else(|| anyhow!("num samples to delete is off"))?;

                for _ in 0..num_samples_to_delete {
                    iter_samples.pop();
                }

                iter_samples.push_iter(&mut samples.into_iter());

                let (head, tail) = iter_samples.as_slices();
                let current_samples = [head, tail].concat();

                // Get tokens to be deleted
                if loop_num > 1 {
                    let num_tokens = state.full_n_tokens(0)?;
                    let token_time_end = state.full_get_segment_t1(0)?;
                    let token_time_per_ms =
                        token_time_end as f32 / (LATENCY_MS * cmp::min(loop_num, NUM_ITERS) as f32); // token times are not a value in ms, they're 150 per second
                    let ms_per_token_time = 1.0 / token_time_per_ms;

                    let mut tokens_saved = vec![];
                    // Skip beginning and end token
                    for i in 1..num_tokens - 1 {
                        let token = state.full_get_token_data(0, i)?;
                        let token_t0_ms = token.t0 as f32 * ms_per_token_time;
                        let ms_to_delete = num_samples_to_delete as f32 / (sampling_freq / 1000.0);

                        // Save tokens for whisper context
                        if (loop_num > NUM_ITERS) && token_t0_ms < ms_to_delete {
                            tokens_saved.push(token.id);
                        }
                    }

                    num_chars_to_delete = words.chars().count();
                    if loop_num > NUM_ITERS {
                        num_chars_to_delete -= tokens_saved
                            .iter()
                            .map(|x| self.ctx.token_to_str(*x).unwrap())
                            .collect::<String>()
                            .chars()
                            .count();
                    }
                    iter_tokens.push_overwrite(tokens_saved);
                }

                // Make the model params
                let (head, tail) = iter_tokens.as_slices();
                let tokens = [head, tail]
                    .concat()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<WhisperToken>>();

                let mut params = FullParams::new(SamplingStrategy::default());
                params.set_tokens(&tokens);
                params.set_no_context(true);
                params.set_print_special(false);
                params.set_suppress_blank(true);
                params.set_print_realtime(false);
                params.set_print_progress(false);
                params.set_token_timestamps(true);
                params.set_print_timestamps(false);
                params.set_duration_ms(LATENCY_MS as i32);
                params.set_language(self.lang.map(Into::into));

                // Run the model
                state.full(params, &current_samples)?;

                let num_tokens = state.full_n_tokens(0)?;

                let utterances = (1..num_tokens - 1)
                    .map(|i| -> Result<Utternace> {
                        let text = state
                            .full_get_token_text(0, i)
                            .context("failed to get token text")?;
                        let token_data = state
                            .full_get_token_data(0, i)
                            .context("failed to get token data")?;

                        Ok(Utternace {
                            text,
                            stop: token_data.t1,
                            start: token_data.t0,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;

                words = (1..num_tokens - 1)
                    .map(|i| {
                        state
                            .full_get_token_text(0, i)
                            .context("failed to get token text")
                    })
                    .collect::<Result<String>>()?;

                emitter
                    .emit(PartialTranscript {
                        utterances,
                        text: words.clone(),
                        offset: num_chars_to_delete,
                    })
                    .await;
            }
        })
    }
}
