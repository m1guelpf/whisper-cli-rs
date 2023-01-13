#![allow(dead_code)]
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use num::integer::div_floor;
use reqwest::Client;
use std::{cmp::min, fs::File, io::Write, path::PathBuf};

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
    let mut milliseconds = seconds * 1000;

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
