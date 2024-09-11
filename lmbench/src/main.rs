use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use reqwest::{Client, Method};

use args::Args;
use request::Request;

mod args;
mod request;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let request = Request::new(&args);

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(args.connect_timeout))
        .tcp_nodelay(true)
        .build()?;
    let mut builder = client
        .request(request.method.clone(), &request.url)
        .headers(request.headers.clone());
    if request.method == Method::POST || request.method == Method::PUT {
        builder = builder.body(request.get_request_body());
    }
    let response = builder.send().await?;

    // let headers = response.headers();
    let body = response.text().await?;

    println!("{body}");
    Ok(())
}
