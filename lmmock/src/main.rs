use axum::{Router, routing::post};
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

mod args;
mod chat;

use args::Args;
use chat::MockResponse;

#[derive(Clone)]
pub struct AppState {
    pub pretty: bool,
    pub output: Option<String>,
    pub responses_map: Option<HashMap<String, MockResponse>>,
    pub request_log: Option<Arc<Mutex<File>>>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let mut responses_map = None;
    if let Some(responses_file) = &args.responses {
        if let Ok(file) = File::open(responses_file) {
            let mut map: HashMap<String, MockResponse> = HashMap::new();
            let reader = BufReader::new(file);
            for line in reader.lines().flatten() {
                if let Ok(mock_resp) = serde_json::from_str::<MockResponse>(&line) {
                    let entry = map.entry(mock_resp.id.clone()).or_insert(MockResponse {
                        id: mock_resp.id,
                        content: None,
                        stream_content: None,
                    });
                    if mock_resp.content.is_some() {
                        entry.content = mock_resp.content;
                    }
                    if mock_resp.stream_content.is_some() {
                        entry.stream_content = mock_resp.stream_content;
                    }
                }
            }
            responses_map = Some(map);
        }
    }

    let mut request_log = None;
    if let Some(output_dir) = &args.output {
        let _ = std::fs::create_dir_all(output_dir);
        let path = std::path::Path::new(output_dir).join("requests.jsonlines");
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            request_log = Some(Arc::new(Mutex::new(file)));
        }
    }

    let state = AppState {
        pretty: args.pretty,
        output: args.output,
        responses_map,
        request_log,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
