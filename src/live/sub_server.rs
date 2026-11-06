use std::{collections::VecDeque, sync::Arc, time::Duration};

pub async fn start_sub_server(
    port: u16,
    sub_tx: tokio::sync::broadcast::Sender<String>,
    history: Arc<std::sync::Mutex<VecDeque<serde_json::Value>>>,
) {
    let addr = format!("127.0.0.1:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("\x1b[33m⚠️  Could not bind subtitle broadcast server to {addr}: {e}\x1b[0m");
            return;
        }
    };

    tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(conn) => conn,
                Err(_) => break,
            };
            let _ = socket.set_nodelay(true);

            let mut sub_rx = sub_tx.subscribe();
            let history_client = Arc::clone(&history);
            tokio::spawn(async move {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};

                let mut buf = [0u8; 1024];
                let n = match socket.read(&mut buf).await {
                    Ok(n) if n > 0 => n,
                    _ => return,
                };
                let req = String::from_utf8_lossy(&buf[..n]);
                let first_line = req.lines().next().unwrap_or("");

                if first_line.starts_with("OPTIONS") {
                    let resp = "HTTP/1.1 204 No Content\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n\
Content-Length: 0\r\n\
\r\n";
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/status") {
                    let body = r#"{"status":"online"}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\n\
Access-Control-Allow-Origin: *\r\n\
Content-Type: application/json\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\
\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/history") {
                    let body = {
                        let hist = history_client.lock().unwrap();
                        serde_json::to_string(&*hist).unwrap_or_else(|_| "[]".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\n\
Access-Control-Allow-Origin: *\r\n\
Content-Type: application/json\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\
\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = socket.write_all(resp.as_bytes()).await;
                    return;
                }

                if first_line.contains("/events") {
                    let header = "HTTP/1.1 200 OK\r\n\
Content-Type: text/event-stream\r\n\
Cache-Control: no-cache\r\n\
Connection: keep-alive\r\n\
Access-Control-Allow-Origin: *\r\n\
Access-Control-Allow-Methods: GET, OPTIONS\r\n\
Access-Control-Allow-Headers: *\r\n\
\r\n\
retry: 1000\r\n\
\r\n";
                    // Coalesce HTTP header and initial history buffer into a single TCP write
                    let initial_payload = {
                        let hist = history_client.lock().unwrap();
                        let mut buf = String::with_capacity(header.len() + hist.len() * 160);
                        buf.push_str(header);
                        for cue in hist.iter() {
                            use std::fmt::Write as _;
                            let _ = write!(buf, "data: {cue}\n\n");
                        }
                        buf
                    };
                    if socket.write_all(initial_payload.as_bytes()).await.is_err() {
                        return;
                    }

                    let mut ping_interval = tokio::time::interval(Duration::from_secs(15));
                    loop {
                        tokio::select! {
                            msg = sub_rx.recv() => {
                                match msg {
                                    Ok(data) => {
                                        let sse_chunk = format!("data: {data}\n\n");
                                        if socket.write_all(sse_chunk.as_bytes()).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                                    Err(_) => break,
                                }
                            }
                            _ = ping_interval.tick() => {
                                if socket.write_all(b":keep-alive\n\n").await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    return;
                }

                let resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = socket.write_all(resp.as_bytes()).await;
            });
        }
    });
}
