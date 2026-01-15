use crate::{args::Args, record::Record};
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::header::HeaderMap;
use std::time::Duration;

#[cfg(feature = "bedrock")]
use aws_config::{BehaviorVersion, meta::region::RegionProviderChain};
#[cfg(feature = "bedrock")]
use aws_sdk_bedrock::primitives::Blob;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::Client as BedrockClient;

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
                .timeout(Duration::from_secs(cli.connect_timeout))
                .build()?;
            Ok(ClientWrapper::Default(client))
        }
    }

    #[cfg(not(feature = "bedrock"))]
    pub async fn new(cli: &Args) -> Result<Self, anyhow::Error> {
        let client = Client::builder()
            .timeout(Duration::from_secs(cli.connect_timeout))
            .build()?;
        Ok(ClientWrapper::Default(client))
    }

    #[cfg(feature = "bedrock")]
    pub async fn send_request(
        &self,
        cli: &Args,
        record: &Record,
        total_requests: u64,
    ) -> Result<(String, HeaderMap), anyhow::Error> {
        match self {
            ClientWrapper::Default(client) => {
                Self::send_request_default(client, cli, record, total_requests).await
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
                while let Some(event) = stream.body.recv().await? {
                    if let aws_sdk_bedrockruntime::types::ResponseStream::Chunk(payload) = event
                        && let Some(bytes) = payload.bytes
                    {
                        body_bytes.extend_from_slice(bytes.as_ref());
                    }
                }
                let mut headers = HeaderMap::new();
                headers.insert(
                    reqwest::header::CONTENT_TYPE,
                    "application/octet-json".parse()?,
                );
                Ok((String::from_utf8(body_bytes)?, headers))
            }
        }
    }

    pub async fn send_request_default(
        client: &Client,
        cli: &Args,
        record: &Record,
        total_requests: u64,
    ) -> Result<(String, HeaderMap), anyhow::Error> {
        let request_builder = record.clone().into_request_builder(client);
        tracing::trace!({
                   request = record.body,
                });

        let res = request_builder.send().await.map_err(|e| {
            eprintln!("request send error: {:?}", e);
            e
        })?;
        let headers = res.headers().clone();
        let status = res.status();

        if cli.verbose && total_requests == 1 {
            println!("Status: {}", status);
            println!("Headers:\n{:#?}", headers);
            println!("Body:");
        }

        if !status.is_success() {
            if total_requests == 1 {
                let text = res.text().await?;
                eprintln!("{}", text);
            }
            return Err(anyhow::anyhow!("request failed: {}", status));
        }

        let mut stream = res.bytes_stream();
        let mut body_bytes = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk = item?;
            if total_requests == 1 {
                print!("{}", String::from_utf8_lossy(&chunk));
            }
            body_bytes.extend_from_slice(&chunk);
        }
        if total_requests == 1 {
            println!();
        }
        Ok((String::from_utf8(body_bytes)?, headers))
    }

    #[cfg(not(feature = "bedrock"))]
    pub async fn send_request(
        &self,
        cli: &Args,
        record: &Record,
        total_requests: u64,
    ) -> Result<(String, HeaderMap), anyhow::Error> {
        match self {
            ClientWrapper::Default(client) => {
                Self::send_request_default(client, cli, record, total_requests).await
            }
        }
    }
}
