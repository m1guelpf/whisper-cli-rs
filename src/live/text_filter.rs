use crate::transcript::Transcript;

pub fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return true;
    }

    // Filter out pure bracketed/parenthesized non-speech tags like [Music], (Laughter), ♪
    if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
        || (trimmed.starts_with('♪') && trimmed.ends_with('♪'))
    {
        return true;
    }

    let cleaned: String = trimmed
        .chars()
        .filter(|c| {
            !c.is_ascii_punctuation()
                && !matches!(*c, '¡' | '¿' | '♪' | '…' | '。' | '、' | '！' | '？')
        })
        .collect::<String>()
        .to_lowercase();
    let cleaned = cleaned.trim();

    if cleaned.is_empty() {
        return true;
    }

    const HALLUCINATIONS: &[&str] = &[
        "thank you for watching",
        "thanks for watching",
        "thank you very much for watching",
        "subtitles by the amaraorg community",
        "subtitles by",
        "transcribed by",
        "please subscribe",
        "like and subscribe",
        "suscríbete al canal",
        "suscribete al canal",
        "subtítulos realizados por la comunidad de amaraorg",
        "subtitulos realizados por la comunidad de amaraorg",
        "gracias por ver el video",
        "ご視聴ありがとうございました",
        "チャンネル登録よろしくお願いします",
        "チャンネル登録お願いします",
        "字幕",
        "音楽",
        "拍手",
        "music",
        "applause",
        "laughter",
        "you",
    ];

    HALLUCINATIONS.contains(&cleaned)
}

pub fn clean_subtitle_text(transcript: &Transcript) -> String {
    let mut parts = Vec::new();
    for u in &transcript.utterances {
        let t = u.text.trim();
        if !t.is_empty() && !is_hallucination(t) {
            parts.push(t);
        }
    }
    let joined = parts.join(" ");
    if is_hallucination(&joined) {
        String::new()
    } else {
        joined
    }
}

pub fn update_prompt_context(prompt: &mut String, new_text: &str) {
    let clean = new_text.trim();
    if clean.is_empty() {
        return;
    }
    if prompt.is_empty() {
        prompt.push_str(clean);
    } else {
        prompt.push(' ');
        prompt.push_str(clean);
    }

    // Keep only the most recent ~30 words (~50 tokens) to stay well within decoder prompt budget
    let words: Vec<&str> = prompt.split_whitespace().collect();
    if words.len() > 30 {
        *prompt = words[words.len() - 25..].join(" ");
    }
}
