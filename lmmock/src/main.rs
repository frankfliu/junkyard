use axum::{Router, routing::post};
use std::net::SocketAddr;
use tokio::net::TcpListener;

mod chat;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/v1/chat/completions", post(chat::chat_completions_handler));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    println!("Listening on {}", addr);
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
