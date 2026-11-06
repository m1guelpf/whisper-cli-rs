use crate::server::ServerPreset;

pub struct LiveConfig {
    pub provider_label: String,
    pub model: Option<String>,
    pub api_url: Option<String>,
    pub preset: Option<ServerPreset>,
    pub lang: Option<String>,
    pub translate: bool,
    pub karaoke: bool,
    pub stdout_only: bool,
    pub chunk_secs: f32,
    pub speech_mode: bool,
    pub sub_port: u16,
    pub apple: bool,
    pub moonshine: bool,
    pub parakeet: bool,
    // pub lfm: bool,
}
