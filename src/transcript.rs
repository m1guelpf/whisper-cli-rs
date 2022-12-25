use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::utils::format_timestamp;

#[derive(Debug, Serialize, Deserialize)]
pub struct Transcript {
    pub utterances: Vec<Utternace>,
    pub processing_time: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Utternace {
    pub start: i64,
    pub stop: i64,
    pub text: String,
}

impl Transcript {
    pub fn as_text(&self) -> String {
        self.utterances
            .iter()
            .fold(String::new(), |transcript, fragment| {
                transcript + format!("{}\n", fragment.text.trim()).as_str()
            })
    }

    pub fn as_vtt(&self) -> String {
        self.utterances
            .iter()
            .fold(String::new(), |transcript, fragment| {
                transcript
                    + format!(
                        "{} --> {}\n{}\n",
                        format_timestamp(fragment.start, false, "."),
                        format_timestamp(fragment.stop, false, "."),
                        fragment.text.trim().replace("-->", "->")
                    )
                    .as_str()
            })
    }

    pub fn as_srt(&self) -> String {
        self.utterances
            .iter()
            .fold((1, String::new()), |(i, transcript), fragment| {
                (
                    i + 1,
                    transcript
                        + format!(
                            "{i}\n{} --> {}\n{}\n",
                            format_timestamp(fragment.start, true, ","),
                            format_timestamp(fragment.stop, true, ","),
                            fragment.text.trim().replace("-->", "->")
                        )
                        .as_str(),
                )
            })
            .1
    }
}
