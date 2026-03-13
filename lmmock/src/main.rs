use axum::{Router, routing::post};
use clap::Parser;
use std::net::SocketAddr;
use tokio::net::TcpListener;

mod chat;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = false)]
    pretty: bool,
}

#[derive(Clone)]
pub struct AppState {
    pub pretty: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let state = AppState {
        pretty: args.pretty,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions_handler))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
