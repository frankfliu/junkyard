use crate::args::Args;
use anyhow::Ok;
use reqwest::{Client, Method, RequestBuilder, multipart};

#[derive(Debug, Clone)]
pub struct Record {
    pub method: Method,
    pub headers: Vec<(String, String)>,
    pub body: Option<String>,
    pub form: Vec<String>,
    pub form_string: Vec<String>,
}

impl Record {
    pub fn new(cli: &Args, body: Option<String>) -> Self {
        let method = cli.get_method();
        let headers = cli
            .header
            .iter()
            .map(|h| {
                let parts: Vec<&str> = h.splitn(2, ':').collect();
                (parts[0].trim().to_string(), parts[1].trim().to_string())
            })
            .collect();
        Self {
            method,
            headers,
            body,
            form: cli.form.clone(),
            form_string: cli.form_string.clone(),
        }
    }

    pub async fn into_request_builder(
        self,
        client: &Client,
        url: &str,
    ) -> Result<RequestBuilder, anyhow::Error> {
        let mut request_builder = client.request(self.method, url);
        for (key, value) in self.headers {
            request_builder = request_builder.header(key, value);
        }
        if let Some(body) = self.body {
            request_builder = request_builder.body(body);
        }

        if !self.form.is_empty() || !self.form_string.is_empty() {
            let mut form = multipart::Form::new();
            for item in &self.form {
                let (key, value) = item.split_once('=').unwrap_or(("", ""));
                if value.starts_with('@') {
                    let path = &value[1..];
                    let file_content = tokio::fs::read(path).await?;
                    let part = multipart::Part::bytes(file_content);
                    form = form.part(key.to_string(), part);
                } else {
                    form = form.text(key.to_string(), value.to_string());
                }
            }
            for item in &self.form_string {
                let (key, value) = item.split_once('=').unwrap_or(("", ""));
                form = form.text(key.to_string(), value.to_string());
            }
            request_builder = request_builder.multipart(form);
        }

        Ok(request_builder)
    }
}
