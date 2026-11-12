use crate::{args::Args, record::Record};
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::header::HeaderMap;
use std::time::{Duration, Instant};

#[cfg(feature = "bedrock")]
use aws_config::{BehaviorVersion, meta::region::RegionProviderChain};
#[cfg(feature = "bedrock")]
use aws_sdk_bedrock::primitives::Blob;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::Client as BedrockClient;

pub struct Response {
    pub body: String,
    pub headers: HeaderMap,
    pub duration: Duration,
    /// time to first byte
    pub ttfb: Duration,
    /// (cumulative body length, arrival time) of each received chunk
    pub chunks: Vec<(usize, Duration)>,
}

#[derive(Clone)]
pub enum ClientWrapper {
    Default(Client),
    #[cfg(feature = "bedrock")]
    Bedrock(BedrockClient),
}

impl ClientWrapper {
    #[cfg(feature = "bedrock")]
    pub async fn new(cli: &Args) -> Result<Self, anyhow::Error> {
        if cli.url.starts_with("bedrock-runtime") {
            let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");
            let config = aws_config::defaults(BehaviorVersion::latest())
                .region(region_provider)
                .load()
                .await;
            Ok(ClientWrapper::Bedrock(BedrockClient::new(&config)))
        } else {
            let client = Client::builder()
                .connect_timeout(Duration::from_secs(cli.connect_timeout))
                .timeout(Duration::from_secs(cli.timeout))
                .build()?;
            Ok(ClientWrapper::Default(client))
        }
    }

    #[cfg(not(feature = "bedrock"))]
    pub async fn new(cli: &Args) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(cli.connect_timeout))
            .timeout(Duration::from_secs(cli.timeout))
            .build()?;
        Ok(ClientWrapper::Default(client))
    }

    #[cfg(feature = "bedrock")]
    pub async fn send_request(
        &self,
        cli: &Args,
        record: &Record,
        print_to_console: bool,
    ) -> Result<Response, anyhow::Error> {
        match self {
            ClientWrapper::Default(client) => {
                Self::send_request_default(client, cli, record, print_to_console).await
            }
            ClientWrapper::Bedrock(client) => {
                let body = Blob::new(record.body.as_ref().unwrap().as_bytes());
                let model_id = cli.model_id.as_ref().unwrap();

                let mut stream = client
                    .invoke_model_with_response_stream()
                    .body(body)
                    .model_id(model_id)
                    .content_type("application/json")
                    .send()
                    .await?;

                let mut body_bytes = Vec::new();
                let start = Instant::now();
                let mut ttft = Duration::from_secs(0);
                let mut first_chunk = true;
                while let Some(event) = stream.body.recv().await? {
                    if let aws_sdk_bedrockruntime::types::ResponseStream::Chunk(payload) = event
                        && let Some(bytes) = payload.bytes
                    {
                        if first_chunk {
                            ttft = start.elapsed();
                            first_chunk = false;
                        }
                        body_bytes.extend_from_slice(bytes.as_ref());
                    }
                }
                let mut headers = HeaderMap::new();
                headers.insert(
                    reqwest::header::CONTENT_TYPE,
                    "application/octet-json".parse()?,
                );
                Ok(Response {
                    body: String::from_utf8(body_bytes)?,
                    headers,
                    duration: start.elapsed(),
                    ttfb: ttft,
                    chunks: Vec::new(),
                })
            }
        }
    }

    pub async fn send_request_default(
        client: &Client,
        cli: &Args,
        record: &Record,
        print_to_console: bool,
    ) -> Result<Response, anyhow::Error> {
        let request_builder = record.clone().into_request_builder(client);
        tracing::trace!({
           request = record.body,
        });

        let start = Instant::now();
        let mut ttft = Duration::from_secs(0);
        let res = request_builder.send().await.map_err(|e| {
            if print_to_console || cli.output.is_none() {
                eprintln!("request send error: {:?}", e);
            }
            e
        })?;
        let headers = res.headers().clone();
        let status = res.status();

        if cli.verbose && print_to_console {
            println!("Status: {}", status);
            println!("Headers:\n{:#?}", headers);
            println!("Body:");
        }

        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            if print_to_console {
                eprintln!("{}", text);
            }
            return Err(anyhow::anyhow!(
                "request failed: {}, body: {}",
                status,
                text
            ));
        }

        let mut stream = res.bytes_stream();
        let mut body_bytes = Vec::new();

        let mut chunks = Vec::new();
        let mut first_chunk = true;
        while let Some(item) = stream.next().await {
            let chunk = item?;
            let elapsed = start.elapsed();
            if print_to_console {
                print!("{}", String::from_utf8_lossy(&chunk));
            }
            if first_chunk {
                ttft = elapsed;
                first_chunk = false;
            }
            body_bytes.extend_from_slice(&chunk);
            chunks.push((body_bytes.len(), elapsed));
        }
        if print_to_console {
            println!();
        }
        Ok(Response {
            body: String::from_utf8(body_bytes)?,
            headers,
            duration: start.elapsed(),
            ttfb: ttft,
            chunks,
        })
    }

    #[cfg(not(feature = "bedrock"))]
    pub async fn send_request(
        &self,
        cli: &Args,
        record: &Record,
        print_to_console: bool,
    ) -> Result<Response, anyhow::Error> {
        match self {
            ClientWrapper::Default(client) => {
                Self::send_request_default(client, cli, record, print_to_console).await
            }
        }
    }
}
