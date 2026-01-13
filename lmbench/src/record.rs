use crate::args::Args;
use anyhow::Ok;
use lazy_static::lazy_static;
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
        } else if let Some(data) = &cli.data_urlencoded {
            headers.insert(
                reqwest::header::CONTENT_TYPE,
                HeaderValue::from_static("application/x-www-form-urlencoded"),
            );
            Some(data.clone())
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

                    fn merge(a: &mut Value, b: Value) {
                        match (a, b) {
                            (Value::Object(a_map), Value::Object(b_map)) => {
                                for (k, v) in b_map {
                                    if let Some(a_val) = a_map.get_mut(&k) {
                                        merge(a_val, v);
                                    } else {
                                        a_map.insert(k, v);
                                    }
                                }
                            }
                            (Value::Array(a_arr), Value::Array(b_arr)) => {
                                a_arr.extend(b_arr);
                            }
                            (a, b) => {
                                *a = b;
                            }
                        }
                    }

                    merge(&mut json_body, extra_params);

                    *body = serde_json::to_string(&json_body)?;
                }
            }
        }
        Ok(())
    }
}

pub fn count_text_tokens(text: &str) -> usize {
    let tokenizer_name = std::env::var("TOKENIZER_NAME").ok();
    if let Some(tokenizer_name) = tokenizer_name {
        let mut cache = TOKENIZER_CACHE.lock();
        if !cache.contains_key(&tokenizer_name) {
            let tokenizer = Tokenizer::from_pretrained(&tokenizer_name, None).unwrap();
            cache.insert(tokenizer_name.clone(), tokenizer);
        }
        let tokenizer = cache.get(&tokenizer_name).unwrap();
        let encoding = tokenizer.encode(text, false).unwrap();
        encoding.get_ids().len()
    } else {
        // Default to a simple word count if no tokenizer is specified
        text.split_whitespace().count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Method;
    use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderValue};
    use serde_json::json;
    use std::collections::HashMap;

    use clap::Parser;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_record_new_with_data_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "hello from file";
        temp_file.write_all(content.as_bytes()).unwrap();
        let args = Args::parse_from([
            "lmbench",
            "-X",
            "POST",
            "-H",
            "Content-Type: text/plain",
            "-H",
            "Invalid-Header",
            "-d",
            format!("@{}", temp_file.path().to_str().unwrap()).as_str(),
            "https://localhost/generate",
        ]);

        let record = Record::new(&args).await.unwrap();
        assert_eq!(record.body, Some(content.to_string()));
    }

    #[tokio::test]
    async fn test_record_new_with_data_raw() {
        let args = Args::parse_from([
            "lmbench",
            "--data-raw",
            "hello",
            "https://localhost/generate",
        ]);

        let record = Record::new(&args).await.unwrap();
        assert_eq!(record.body, Some("hello".to_string()));
        assert_eq!(record.method, Method::POST);
    }

    #[tokio::test]
    async fn test_record_new_with_data_urlencoded() {
        let args = Args::parse_from([
            "lmbench",
            "-H",
            "Content-Type: text/plain",
            "--data-urlencoded",
            "hello",
            "https://localhost/generate",
        ]);

        let record = Record::new(&args).await.unwrap();
        assert_eq!(record.body, Some("hello".to_string()));
        assert_eq!(record.method, Method::POST);
        assert_eq!(
            record.headers.get(CONTENT_TYPE).unwrap(),
            HeaderValue::from_static("application/x-www-form-urlencoded")
        );
    }

    #[tokio::test]
    async fn test_record_new_with_form() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = "hello from file";
        temp_file.write_all(content.as_bytes()).unwrap();

        let args = Args::parse_from([
            "lmbench",
            "-F",
            format!("key1=@{}", temp_file.path().to_str().unwrap()).as_str(),
            "-F",
            "key2=value",
            "--form-string",
            "key3=value",
            "https://localhost/generate",
        ]);

        let record = Record::new(&args).await.unwrap();
        assert_eq!(record.form.get("key1"), Some(&content.to_string()));
        assert_eq!(record.form.get("key2"), Some(&"value".to_string()));
        assert_eq!(record.form.get("key3"), Some(&"value".to_string()));
    }

    #[test]
    fn test_count_text_tokens() {
        // Test with default word count
        assert_eq!(count_text_tokens("hello world"), 2);
        assert_eq!(count_text_tokens("hello   world"), 2);
        assert_eq!(count_text_tokens(""), 0);

        unsafe {
            std::env::set_var("TOKENIZER_NAME", "gpt2");
            assert_eq!(count_text_tokens("hello world"), 2);
            std::env::remove_var("TOKENIZER_NAME");
        }
    }

    #[tokio::test]
    async fn test_into_request_builder_with_body() {
        let record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost/test".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some("test body".to_string()),
            input_tokens: 0,
        };

        let client = Client::new();
        let request = record.into_request_builder(&client).build().unwrap();

        assert_eq!(*request.method(), Method::POST);
        assert_eq!(request.url().as_str(), "http://localhost/test");

        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        assert_eq!(body_bytes, "test body".as_bytes());

        let record = Record {
            id: "test".to_string(),
            method: Method::GET,
            url: "http://localhost/test".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: None,
            input_tokens: 0,
        };
        let request = record.into_request_builder(&client).build().unwrap();
        assert!(request.body().is_none());
    }

    #[tokio::test]
    async fn test_into_request_builder_with_form() {
        let mut form = HashMap::new();
        form.insert("key".to_string(), "value".to_string());
        let record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost/test".to_string(),
            headers: HeaderMap::new(),
            form,
            body: None,
            input_tokens: 0,
        };

        let client = Client::new();
        let request = record.into_request_builder(&client).build().unwrap();

        assert_eq!(*request.method(), Method::POST);
        assert_eq!(request.url().as_str(), "http://localhost/test");

        let body_bytes = request.body().unwrap().as_bytes().unwrap();
        assert_eq!(body_bytes, "key=value".as_bytes());
    }

    #[test]
    fn test_set_extra_parameters() {
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!({"prompt": "hello"}).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let extra_parameters = Some(json!({ "temperature": 0.5 }).to_string());
        record.set_extra_parameters(&extra_parameters).unwrap();

        let expected_body = json!({
            "prompt": "hello",
            "temperature": 0.5
        });
        assert_eq!(record.body, Some(expected_body.to_string()));

        // Test recursive merge
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!({ "a": { "b": 1 } }).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let extra_parameters = Some(json!({ "a": { "c": 2 } }).to_string());
        record.set_extra_parameters(&extra_parameters).unwrap();

        let expected_body = json!({
            "a": {
                "b": 1,
                "c": 2
            }
        });
        assert_eq!(record.body, Some(expected_body.to_string()));

        // Test with no extra parameters
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!({ "prompt": "hello" }).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let extra_parameters = None;
        record.set_extra_parameters(&extra_parameters).unwrap();
        assert_eq!(record.body, Some(json!({ "prompt": "hello" }).to_string()));

        // Test with non-json body
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some("hello".to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let extra_parameters = Some(json!({ "temperature": 0.5 }).to_string());
        assert!(record.set_extra_parameters(&extra_parameters).is_err());

        // Test with json array body
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!([1, 2, 3]).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        let extra_parameters = Some(json!({ "temperature": 0.5 }).to_string());
        record.set_extra_parameters(&extra_parameters).unwrap();
        assert_eq!(record.body, Some(r#"{"temperature":0.5}"#.to_string()));

        // Test with empty body
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: None,
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        record.set_extra_parameters(&extra_parameters).unwrap();
        assert_eq!(record.body, None);

        // Test with text/plain content type
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!({"prompt": "hello"}).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain"),
        );
        record.set_extra_parameters(&extra_parameters).unwrap();
        assert_eq!(record.body, Some(json!({"prompt": "hello"}).to_string()));

        // Test array merge
        let mut record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: Some(json!({ "a": [1, 2] }).to_string()),
            input_tokens: 0,
        };
        record.headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let extra_parameters = Some(json!({ "a": [3, 4] }).to_string());
        record.set_extra_parameters(&extra_parameters).unwrap();

        let expected_body = json!({
            "a": [1, 2, 3, 4]
        });
        assert_eq!(record.body, Some(expected_body.to_string()));
    }

    #[test]
    fn test_count_input_tokens() {
        let record = Record {
            id: "test".to_string(),
            method: Method::POST,
            url: "http://localhost".to_string(),
            headers: HeaderMap::new(),
            form: HashMap::new(),
            body: None,
            input_tokens: 0,
        };
        assert_eq!(record.count_input_tokens(), 0);
    }
}
