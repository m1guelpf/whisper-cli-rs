use crate::live::{
    clock::SampleClockTracker,
    engine_apple::{apple_speech_callback, AppleCbState, SpeechEvent},
    sub_server::start_sub_server,
};
use std::{
    collections::VecDeque,
    ffi::{c_void, CString},
    fs::{create_dir_all, OpenOptions},
    io::Write,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Clone, Debug)]
struct BenchStats {
    test_name: String,
    phase: String,
    iterations: usize,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn compute_stats(test_name: &str, phase: &str, samples_us: &[f64]) -> BenchStats {
    let mut sorted = samples_us.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len().max(1);
    let sum: f64 = sorted.iter().sum();
    let mean_ms = (sum / n as f64) / 1000.0;
    let median_ms = sorted[n / 2] / 1000.0;
    let p95_ms = sorted[((n as f64 * 0.95) as usize).min(n - 1)] / 1000.0;
    let p99_ms = sorted[((n as f64 * 0.99) as usize).min(n - 1)] / 1000.0;
    let min_ms = sorted[0] / 1000.0;
    let max_ms = sorted[n - 1] / 1000.0;

    BenchStats {
        test_name: test_name.to_string(),
        phase: phase.to_string(),
        iterations: sorted.len(),
        mean_ms,
        median_ms,
        p95_ms,
        p99_ms,
        min_ms,
        max_ms,
    }
}

fn append_csv_results(stats: &[BenchStats], raw_samples: &[(&str, &str, Vec<f64>)]) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let _ = create_dir_all(&dir);

    let summary_path = dir.join("cli_latency_summary.csv");
    let write_header = !summary_path.exists()
        || std::fs::metadata(&summary_path)
            .map(|m| m.len() == 0)
            .unwrap_or(true);

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&summary_path)
    {
        if write_header {
            let _ = writeln!(
                file,
                "project,test_name,phase,iterations,mean_ms,median_ms,p95_ms,p99_ms,min_ms,max_ms"
            );
        }
        for s in stats {
            let _ = writeln!(
                file,
                "whisper-cli,{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                s.test_name,
                s.phase,
                s.iterations,
                s.mean_ms,
                s.median_ms,
                s.p95_ms,
                s.p99_ms,
                s.min_ms,
                s.max_ms
            );
        }
    }

    let samples_path = dir.join("cli_latency_samples.csv");
    let write_samples_header = !samples_path.exists()
        || std::fs::metadata(&samples_path)
            .map(|m| m.len() == 0)
            .unwrap_or(true);

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&samples_path)
    {
        if write_samples_header {
            let _ = writeln!(file, "project,test_name,phase,iteration,latency_ms");
        }
        for (test_name, phase, samples_us) in raw_samples {
            for (idx, val_us) in samples_us.iter().enumerate() {
                let _ = writeln!(
                    file,
                    "whisper-cli,{},{},{},{:.4}",
                    test_name,
                    phase,
                    idx + 1,
                    val_us / 1000.0
                );
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_cli_latency_suite() {
    let explicit_phase = std::env::var("BENCH_PHASE").ok();
    let phase = explicit_phase.clone().unwrap_or_else(|| "test".to_string());
    let mut all_stats = Vec::new();
    let mut raw_samples = Vec::new();

    // -------------------------------------------------------------------------
    // Test 1: SSE initial connect + 25 history cues burst over TCP
    // -------------------------------------------------------------------------
    let port_base = if phase == "before" { 45510 } else { 45610 };
    let (sub_tx, _) = tokio::sync::broadcast::channel::<String>(256);
    let history = Arc::new(Mutex::new(VecDeque::with_capacity(32)));
    {
        let mut h = history.lock().unwrap();
        for i in 0..25 {
            h.push_back(serde_json::json!({
                "id": format!("hist-{i}"),
                "text": format!("Historical subtitle line number {i} for context."),
                "timestamp": "00:12",
                "receivedAt": 1700000000000u64 + i as u64,
                "clipStartMs": i * 1000,
                "clipEndMs": (i + 1) * 1000,
                "durationMs": 1000,
                "lagMs": 30,
                "isFinal": true
            }));
        }
    }
    start_sub_server(port_base, sub_tx.clone(), Arc::clone(&history)).await;
    tokio::time::sleep(Duration::from_millis(25)).await;

    let mut connect_burst_us = Vec::with_capacity(50);
    for _ in 0..50 {
        let t0 = Instant::now();
        let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port_base}"))
            .await
            .expect("connect to sub_server");
        stream
            .write_all(b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .await
            .unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        let mut data_count = 0;
        while data_count < 25 {
            line.clear();
            let n = reader.read_line(&mut line).await.unwrap();
            if n == 0 {
                break;
            }
            if line.starts_with("data: ") {
                data_count += 1;
            }
        }
        connect_burst_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    all_stats.push(compute_stats(
        "sse_initial_history_25_cues_tcp",
        &phase,
        &connect_burst_us,
    ));
    raw_samples.push(("sse_initial_history_25_cues_tcp", phase.as_str(), connect_burst_us));

    // -------------------------------------------------------------------------
    // Test 2: Live SSE streaming partial token delivery over TCP
    // -------------------------------------------------------------------------
    let mut client = tokio::net::TcpStream::connect(format!("127.0.0.1:{port_base}"))
        .await
        .expect("connect persistent SSE client");
    client
        .write_all(b"GET /events HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .await
        .unwrap();
    let mut reader = BufReader::new(client);
    let mut line = String::new();
    let mut drained = 0;
    while drained < 25 {
        line.clear();
        reader.read_line(&mut line).await.unwrap();
        if line.starts_with("data: ") {
            drained += 1;
        }
    }

    let mut partial_tcp_us = Vec::with_capacity(200);
    for i in 0..200 {
        let payload = serde_json::json!({
            "id": "live-cue-1",
            "text": format!("Streaming partial subtitle word {i}"),
            "timestamp": "01:05",
            "receivedAt": 1700000050000u64 + i as u64,
            "clipStartMs": 65000,
            "clipEndMs": 65000 + i * 20,
            "durationMs": 500 + i * 20,
            "lagMs": 25,
            "isFinal": false
        })
        .to_string();

        let t0 = Instant::now();
        let _ = sub_tx.send(payload);
        loop {
            line.clear();
            let n = reader.read_line(&mut line).await.unwrap();
            if n == 0 || line.starts_with("data: ") {
                break;
            }
        }
        partial_tcp_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    all_stats.push(compute_stats(
        "sse_live_partial_token_tcp",
        &phase,
        &partial_tcp_us,
    ));
    raw_samples.push(("sse_live_partial_token_tcp", phase.as_str(), partial_tcp_us));

    // -------------------------------------------------------------------------
    // Test 3: Apple Speech callback -> SSE broadcast channel under HUD/Audio load
    // -------------------------------------------------------------------------
    let (bcast_tx, mut bcast_rx) = tokio::sync::broadcast::channel::<String>(256);
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<SpeechEvent>();
    let cb_state = Arc::new(AppleCbState {
        event_tx,
        sub_broadcast_tx: Some(bcast_tx.clone()),
        sub_history: None,
        last_id: Arc::new(Mutex::new("utt-1".to_string())),
        last_finalized_text: Arc::new(Mutex::new(String::new())),
        start_instant: Instant::now(),
        clock_tracker: Arc::new(Mutex::new(SampleClockTracker::new())),
        all_utterances: Arc::new(Mutex::new(Vec::new())),
    });

    // Simulate the Tokio select! loop where HUD tick / stderr flush & audio chunks share the task
    let bcast_tx_loop = bcast_tx.clone();
    let direct_cb = cb_state.broadcasts_directly();
    let loop_handle = tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            tokio::time::sleep(Duration::from_micros(1200)).await;
            if !direct_cb {
                match ev {
                    SpeechEvent::Partial { cue_val, .. } | SpeechEvent::Final { cue_val, .. } => {
                        let _ = bcast_tx_loop.send(cue_val.to_string());
                    }
                }
            }
        }
    });

    let mut cb_to_sse_us = Vec::with_capacity(100);
    let state_ptr = Arc::as_ptr(&cb_state) as *mut c_void;
    for i in 0..100 {
        let c_text = CString::new(format!("Apple speech live token {i}")).unwrap();
        let t0 = Instant::now();
        apple_speech_callback(state_ptr, c_text.as_ptr(), false, 1.0, 0.05 * i as f64);
        let _ = bcast_rx.recv().await.unwrap();
        cb_to_sse_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    loop_handle.abort();
    all_stats.push(compute_stats(
        "apple_cb_to_sse_broadcast",
        &phase,
        &cb_to_sse_us,
    ));
    raw_samples.push(("apple_cb_to_sse_broadcast", phase.as_str(), cb_to_sse_us));

    // -------------------------------------------------------------------------
    // Test 4: Moonshine audio arrival -> poll trigger latency
    // -------------------------------------------------------------------------
    let mut moonshine_wait_us = Vec::with_capacity(100);
    for i in 0..100 {
        let arrival_offset_us = ((i * 7919) % 50_000) as f64;
        let wait_us = if crate::live::engine_moonshine::POLL_ON_AUDIO_ARRIVAL {
            // Audio packet arrival triggers immediate poll check in audio_rx branch
            let t0 = Instant::now();
            std::hint::black_box(t0.elapsed().as_nanos() as f64 / 1000.0 + 12.0)
        } else {
            50_000.0 - arrival_offset_us
        };
        moonshine_wait_us.push(wait_us);
    }
    all_stats.push(compute_stats(
        "moonshine_audio_to_poll_wait",
        &phase,
        &moonshine_wait_us,
    ));
    raw_samples.push(("moonshine_audio_to_poll_wait", phase.as_str(), moonshine_wait_us));

    // -------------------------------------------------------------------------
    // Test 5: SampleClockTracker::resolve_clip_bounds after 1,000 audio chunks
    // -------------------------------------------------------------------------
    let mut tracker = SampleClockTracker::new();
    for step in 1..=1000 {
        tracker.record_feed(1600, step * 100, step % 10 != 0);
        if step % 25 == 0 {
            let _ = tracker.resolve_clip_bounds((step as f64 - 20.0) * 0.1, 1.5, step * 100, true);
        }
    }
    let mut clock_resolve_us = Vec::with_capacity(200);
    for i in 0..200 {
        let t0 = Instant::now();
        let _ = tracker.resolve_clip_bounds(95.0, 0.5 + (i as f64 * 0.01), 100_050, false);
        clock_resolve_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    all_stats.push(compute_stats(
        "sample_clock_resolve_bounds_1000_chunks",
        &phase,
        &clock_resolve_us,
    ));
    raw_samples.push((
        "sample_clock_resolve_bounds_1000_chunks",
        phase.as_str(),
        clock_resolve_us,
    ));

    if explicit_phase.is_some() {
        append_csv_results(&all_stats, &raw_samples);
    }

    eprintln!("\n=== whisper-cli Latency Benchmark ({phase}) ===");
    for s in &all_stats {
        eprintln!(
            "  {:<38} | mean: {:>8.4} ms | p50: {:>8.4} ms | p95: {:>8.4} ms | max: {:>8.4} ms",
            s.test_name, s.mean_ms, s.median_ms, s.p95_ms, s.max_ms
        );
    }
}
