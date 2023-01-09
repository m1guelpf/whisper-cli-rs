use crate::utils::download_file;
use dirs::cache_dir;
use std::{fmt::Display, fs, path::PathBuf};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum Size {
    #[clap(name = "tiny.en")]
    TinyEnglish,
    #[clap(name = "tiny")]
    Tiny,
    #[clap(name = "base.en")]
    BaseEnglish,
    #[clap(name = "base")]
    Base,
    #[clap(name = "small.en")]
    SmallEnglish,
    #[clap(name = "small")]
    Small,
    #[clap(name = "medium.en")]
    MediumEnglish,
    #[clap(name = "medium")]
    Medium,
    #[clap(name = "large")]
    Large,
    #[clap(name = "large-v1")]
    LargeV1,
}

impl Size {
    pub fn get_path(self) -> PathBuf {
        let mut path = cache_dir().expect("Could not find cache directory");
        path.push("whisper");
        path.push("models");
        path.push(format!("{self}.bin"));

        path
    }

    pub const fn is_english_only(self) -> bool {
        matches!(
            self,
            Self::TinyEnglish | Self::BaseEnglish | Self::SmallEnglish | Self::MediumEnglish
        )
    }
}

impl Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::TinyEnglish => "tiny.en",
            Self::Tiny => "tiny",
            Self::BaseEnglish => "base.en",
            Self::Base => "base",
            Self::SmallEnglish => "small.en",
            Self::Small => "small",
            Self::MediumEnglish => "medium.en",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::LargeV1 => "large-v1",
        };

        write!(f, "{key}")
    }
}

pub struct Model {
    size: Size,
}

impl Model {
    pub const fn new(size: Size) -> Self {
        Self { size }
    }

    pub fn get_path(&self) -> PathBuf {
        self.size.get_path()
    }

    pub async fn download(&self) {
        let path = self.get_path();
        if path.exists() {
            return;
        }

        let cache_dir = path.parent().expect("Failed to get cache dir");
        if !cache_dir.exists() {
            fs::create_dir_all(cache_dir).expect("Failed to create cache dir.");
        }

        download_file(
            &format!(
                "https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-{}.bin",
                self.size
            ),
            path.to_str().unwrap(),
        )
        .await;
    }
}
