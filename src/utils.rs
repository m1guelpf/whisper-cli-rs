#![allow(dead_code)]
use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use num::integer::div_floor;
use reqwest::Client;
use std::{
    cmp::min,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

/// Prompts the user interactively on the terminal (`[Y/n]`) whether to auto-download a missing model.
/// Defaults to `true` if the user presses Enter or types `y`/`yes`.
pub fn prompt_auto_download(model_name: &str, source_hint: &str) -> Result<bool> {
    eprint!(
        "\n\x1b[33m⚠️  Model '{model_name}' was not detected locally ({source_hint}).\x1b[0m\n\
         \x1b[1m👉 Would you like to auto-download it now? [Y/n]: \x1b[0m"
    );
    let _ = std::io::stderr().flush();

    let mut answer = String::new();
    let read_ok = if let Ok(tty) = File::open("/dev/tty") {
        let mut reader = BufReader::new(tty);
        reader.read_line(&mut answer).is_ok()
    } else {
        std::io::stdin().read_line(&mut answer).is_ok()
    };

    if !read_ok {
        return Ok(false);
    }

    let trimmed = answer.trim().to_ascii_lowercase();
    Ok(trimmed.is_empty() || trimmed == "y" || trimmed == "yes")
}

/// Downloads a file synchronously with an `indicatif` progress bar.
pub fn download_file_blocking(url: &str, dest: &Path, label: &str) -> Result<()> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| anyhow!("Failed to connect to {url}: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow!("HTTP error downloading {label} from {url}: {e}"))?;

    let total_size = resp.content_length().unwrap_or(0);
    let pb = if total_size > 0 {
        let bar = ProgressBar::new(total_size);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("  📥 {msg}\n  {spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );
        bar
    } else {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("  📥 {msg} ({bytes} downloaded, {bytes_per_sec})")
                .unwrap(),
        );
        bar
    };
    pb.set_message(format!("Downloading {label}..."));

    let tmp_dest = dest.with_extension("part");
    let mut out = File::create(&tmp_dest)?;
    let mut buffer = [0_u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        let n = resp.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        out.write_all(&buffer[..n])?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }
    out.flush()?;
    drop(out);

    fs::rename(&tmp_dest, dest)?;
    pb.finish_with_message(format!("✅ Downloaded {label} to {}", dest.display()));
    eprintln!();
    Ok(())
}

/// Standard RFC 4648 Base64 encoder for in-memory WAV payloads.
pub fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity((input.len() + 2) / 3 * 4);

    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(TABLE[((triple >> 18) & 0x3F) as usize]);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize]);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3F) as usize]);
        } else {
            out.push(b'=');
        }
    }

    // SAFETY: TABLE and '=' are strictly ASCII bytes.
    unsafe { String::from_utf8_unchecked(out) }
}

pub async fn download_file(url: &str, path: &str) {
    let res = Client::new()
        .get(url)
        .send()
        .await
        .or(Err(format!("Failed to GET from '{}'", &url)))
        .unwrap();

    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from '{}'", &url))
        .unwrap();

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading {url}",));

    let mut file = File::create(path)
        .or(Err(format!("Failed to create file '{path}'")))
        .unwrap();

    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.or(Err("Error while downloading file")).unwrap();

        file.write_all(&chunk)
            .or(Err("Error while writing to file"))
            .unwrap();

        let new = min(downloaded + (chunk.len() as u64), total_size);
        pb.set_position(new);
        downloaded = new;
    }

    pb.finish_with_message(format!("Downloaded {url} to {path}"));
}

pub fn format_timestamp(seconds: i64, always_include_hours: bool, decimal_marker: &str) -> String {
    assert!(seconds >= 0, "non-negative timestamp expected");
    let mut milliseconds = seconds * 10;

    let hours = div_floor(milliseconds, 3_600_000);
    milliseconds -= hours * 3_600_000;

    let minutes = div_floor(milliseconds, 60_000);
    milliseconds -= minutes * 60_000;

    let seconds = div_floor(milliseconds, 1_000);
    milliseconds -= seconds * 1_000;

    let hours_marker = if always_include_hours || hours != 0 {
        format!("{hours}:")
    } else {
        String::new()
    };

    format!("{hours_marker}{minutes:02}:{seconds:02}{decimal_marker}{milliseconds:03}")
}

pub fn write_to(path: PathBuf, content: &String) {
    File::create(path)
        .unwrap()
        .write_all(content.as_bytes())
        .unwrap();
}
