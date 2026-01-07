use clap::{ArgGroup, Parser};
use reqwest::Method;
use std::path::PathBuf;

/// A curl-like tool for interacting with LLMs
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None, term_width = 160)]
#[command(group(
    ArgGroup::new("body_group")
        .args(["data", "data_raw", "data_urlencoded"]),
))]
#[command(group(
    ArgGroup::new("form_group")
        .args(["form", "form_string"])
        .conflicts_with("body_group")
        .multiple(true),
))]
pub struct Args {
    /// Concurrent clients
    #[arg(short, long, default_value_t = 1)]
    pub(crate) clients: u32,

    /// Maximum time allowed for connection
    #[arg(long, value_name = "SECONDS", default_value_t = 60)]
    pub(crate) connect_timeout: u64,

    /// HTTP POST data, '@' allowed
    #[arg(short, long)]
    pub(crate) data: Option<String>,

    /// HTTP POST data (no file)
    #[arg(long, value_name = "DATA")]
    pub(crate) data_raw: Option<String>,

    /// HTTP POST data url encoded
    #[arg(long, value_name = "DATA")]
    pub(crate) data_urlencoded: Option<String>,

    /// dataset directory
    #[arg(long, value_name = "DIRECTORY", conflicts_with_all = ["body_group", "form_group"])]
    pub(crate) dataset: Option<PathBuf>,

    /// Delay in millis for initial requests (e.g. 10 or rand(100, 200))
    #[arg(long)]
    pub(crate) delay: Option<String>,

    /// Duration of the test in seconds
    #[arg(long)]
    pub(crate) duration: Option<u32>,

    /// extra parameters for json dataset
    #[arg(long)]
    pub(crate) extra_parameters: Option<String>,

    /// Specify HTTP multipart POST data, '@' allowed
    #[arg(short = 'F', long, value_name = "CONTENT")]
    pub(crate) form: Vec<String>,

    /// Specify HTTP multipart POST data (no file)
    #[arg(long, value_name = "STRING")]
    pub(crate) form_string: Vec<String>,

    /// Send the -d data with a HTTP GET
    #[arg(short = 'G', long)]
    pub(crate) get: bool,

    /// Pass custom header LINE to server
    #[arg(short = 'H', long, value_name = "LINE")]
    pub(crate) header: Vec<String>,

    /// Include protocol headers in the output
    #[arg(short, long)]
    pub(crate) include: bool,

    /// Predefine schema (gemini/openai/anthropic/TGI) or custom jq expression for token output
    #[arg(short, long, value_name = "EXPRESSION")]
    pub(crate) jq: Option<String>,

    /// Write to response to output directory
    #[arg(short = 'o', long, value_name = "DIRECTORY")]
    pub(crate) output: Option<String>,

    /// Number of requests to perform
    #[arg(short = 'N', long, default_value_t = 1)]
    pub(crate) repeat: u32,

    /// Random seed
    #[arg(long, value_name = "RANDOM_SEED")]
    pub(crate) seed: Option<i32>,

    /// Silent mode
    #[arg(short, long)]
    pub(crate) silent: bool,

    /// Output token per seconds
    #[arg(short, long)]
    pub(crate) tokens: bool,

    /// Specify request HTTP method to use
    #[arg(short = 'X', long, value_name = "METHOD")]
    pub(crate) request: Option<String>,

    /// http/https URL
    #[arg()]
    pub(crate) url: String,
}

impl Args {
    pub(crate) fn get_method(&self) -> Method {
        if let Some(method) = &self.request {
            return Method::from_bytes(method.as_bytes()).unwrap_or(Method::GET);
        }
        if self.get {
            return Method::GET;
        }
        if self.data.is_some()
            || self.data_raw.is_some()
            || self.data_urlencoded.is_some()
            || !self.form.is_empty()
            || !self.form_string.is_empty()
            || self.dataset.is_some()
        {
            return Method::POST;
        }
        Method::GET
    }

    pub(crate) fn get_jq(&self, stream: bool) -> String {
        match self.jq.as_deref() {
            Some("gemini") => {
                if self.url.contains("streamGenerateContent") {
                    "$[*].candidates[*].content.parts[*].text".to_string()
                } else {
                    "$.candidates[*].content.parts[*].text".to_string()
                }
            }
            Some("openai") => {
                if stream {
                    "$.choices[*].delta.content".to_string()
                } else {
                    "$.choices[*].message.content".to_string()
                }
            }
            Some("anthropic") => {
                if stream {
                    "$.delta.text".to_string()
                } else {
                    "$.content[*].text".to_string()
                }
            }
            Some("TGI") | None => {
                if stream {
                    "$.token.text".to_string()
                } else {
                    "$.generated_text".to_string()
                }
            }
            Some(expr) => expr.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::Method;

    fn default_args() -> Args {
        Args::parse_from(["lmbench", "https://localhost/generate"])
    }

    #[test]
    fn test_get_method() {
        let mut args = default_args();
        assert_eq!(args.get_method(), Method::GET);

        args.request = Some("POST".to_string());
        assert_eq!(args.get_method(), Method::POST);

        args.request = Some("PUT".to_string());
        assert_eq!(args.get_method(), Method::PUT);

        args.request = None;
        args.get = true;
        assert_eq!(args.get_method(), Method::GET);

        args.get = false;
        args.data = Some("data".to_string());
        assert_eq!(args.get_method(), Method::POST);

        args.data = None;
        args.data_raw = Some("data_raw".to_string());
        assert_eq!(args.get_method(), Method::POST);

        args.data_raw = None;
        args.data_urlencoded = Some("data_urlencoded".to_string());
        assert_eq!(args.get_method(), Method::POST);

        args.data_urlencoded = None;
        args.form = vec!["form".to_string()];
        assert_eq!(args.get_method(), Method::POST);

        args.form = vec![];
        args.form_string = vec!["form_string".to_string()];
        assert_eq!(args.get_method(), Method::POST);

        args.form_string = vec![];
        args.dataset = Some("dataset".into());
        assert_eq!(args.get_method(), Method::POST);
    }

    #[test]
    fn test_get_jq() {
        let mut args = default_args();

        // Test None case
        assert_eq!(args.get_jq(true), "$.token.text");
        assert_eq!(args.get_jq(false), "$.generated_text");

        // Test gemini
        args.jq = Some("gemini".to_string());
        args.url = "https://host/streamGenerateContent".to_string();
        assert_eq!(
            args.get_jq(true),
            "$[*].candidates[*].content.parts[*].text"
        );
        args.url = "https://host/generateContent".to_string();
        assert_eq!(args.get_jq(false), "$.candidates[*].content.parts[*].text");

        // Test openai
        args.jq = Some("openai".to_string());
        assert_eq!(args.get_jq(true), "$.choices[*].delta.content");
        assert_eq!(args.get_jq(false), "$.choices[*].message.content");

        // Test anthropic
        args.jq = Some("anthropic".to_string());
        assert_eq!(args.get_jq(true), "$.delta.text");
        assert_eq!(args.get_jq(false), "$.content[*].text");

        // Test TGI
        args.jq = Some("TGI".to_string());
        assert_eq!(args.get_jq(true), "$.token.text");
        assert_eq!(args.get_jq(false), "$.generated_text");

        // Test custom expression
        args.jq = Some(".foo.bar".to_string());
        assert_eq!(args.get_jq(true), ".foo.bar");
        assert_eq!(args.get_jq(false), ".foo.bar");
    }
}
