use crate::args::Args;
use anyhow::Ok;
use lazy_static::lazy_static;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use reqwest::{
    Client, Method, RequestBuilder,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Value, from_str};
use std::collections::HashMap;
use tokenizers::Tokenizer;

lazy_static! {
    static ref TOKENIZER_CACHE: Mutex<HashMap<String, Tokenizer>> = Mutex::new(HashMap::new());
}

pub(crate) static TOKENIZER_NAME: Lazy<Option<String>> =
    Lazy::new(|| std::env::var("TOKENIZER_NAME").ok());

#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub form: HashMap<String, String>,
    pub body: Option<String>,
    pub input_tokens: usize,
}

impl Record {
    pub async fn new(cli: &Args) -> Result<Self, anyhow::Error> {
        let method = cli.get_method();

        let mut headers = HeaderMap::new();
        for h in &cli.header {
            let parts: Vec<&str> = h.splitn(2, ':').collect();
            if parts.len() == 2 {
                let name = HeaderName::from_bytes(parts[0].trim().as_bytes())?;
                let value = HeaderValue::from_str(parts[1].trim())?;
                headers.append(name, value);
            }
        }

        let body = if let Some(data) = &cli.data {
            if let Some(path) = data.strip_prefix('@') {
                Some(tokio::fs::read_to_string(path).await?)
            } else {
                Some(data.clone())
            }
        } else if let Some(data) = &cli.data_raw {
            Some(data.clone())
        } else if let Some(data) = &cli.data_urlencode {
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded"),
            );
            serde_urlencoded::to_string(data).ok()
        } else {
            None
        };

        let mut form: HashMap<String, String> = HashMap::new();
        for item in &cli.form {
            let (key, value) = item.split_once('=').unwrap_or(("", ""));
            let value_str = if let Some(path) = value.strip_prefix('@') {
                tokio::fs::read_to_string(path).await?
            } else {
                value.to_string()
            };
            form.insert(key.to_string(), value_str);
        }

        for item in &cli.form_string {
            let (key, value) = item.split_once('=').unwrap_or(("", ""));
            form.insert(key.to_string(), value.to_string());
        }

        Ok(Self {
            id: "data".to_string(),
            method,
            url: cli.url.clone(),
            headers,
            form,
            body,
            input_tokens: 0,
        })
    }

    pub fn into_request_builder(self, client: &Client) -> RequestBuilder {
        let mut request_builder = client.request(self.method, self.url);
        request_builder = request_builder.headers(self.headers);

        if !self.form.is_empty() {
            request_builder = request_builder.form(&self.form);
        } else if let Some(body) = self.body {
            request_builder = request_builder.body(body);
        }

        request_builder
    }

    pub fn count_input_tokens(&self) -> usize {
        if let Some(body) = &self.body {
            count_text_tokens(body)
        } else {
            0
        }
    }

    pub fn set_extra_parameters(
        &mut self,
        extra_parameters: &Option<String>,
    ) -> Result<(), anyhow::Error> {
        if let (Some(extra_params_str), Some(content_type)) = (
            extra_parameters,
            self.headers.get(reqwest::header::CONTENT_TYPE),
        ) {
            if content_type.to_str()?.contains("application/json") {
                if let Some(body) = self.body.as_mut() {
                    let mut json_body: Value = from_str(body)?;
                    let extra_params: Value = from_str(extra_params_str)?;

                    if let (Some(body_obj), Some(extra_obj)) =
                        (json_body.as_object_mut(), extra_params.as_object())
                    {
                        for (k, v) in extra_obj {
                            body_obj.insert(k.clone(), v.clone());
                        }
                    }
                    *body = serde_json::to_string(&json_body)?;
                }
            }
        }
        Ok(())
    }
}

pub fn count_text_tokens(text: &str) -> usize {
    if let Some(tokenizer_name) = TOKENIZER_NAME.as_ref() {
        let mut cache = TOKENIZER_CACHE.lock();
        if !cache.contains_key(tokenizer_name) {
            let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None).unwrap();
            cache.insert(tokenizer_name.clone(), tokenizer);
        }
        let tokenizer = cache.get(tokenizer_name).unwrap();
        let encoding = tokenizer.encode(text, false).unwrap();
        encoding.get_ids().len()
    } else {
        // Default to a simple word count if no tokenizer is specified
        text.split_whitespace().count()
    }
}
