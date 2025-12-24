use clap::Parser;
use reqwest::Method;
use std::path::PathBuf;

/// A curl-like tool for interacting with LLMs
#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None, term_width = 160)]
pub(crate) struct Args {
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
    pub(crate) data_urlencode: Option<String>,

    /// dataset directory
    #[arg(long, value_name = "DIRECTORY")]
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

    /// Json query expression for token output
    #[arg(short, long, value_name = "EXPRESSION")]
    pub(crate) jq: Option<String>,

    /// Save the output json to a file
    #[arg(long, value_name = "JSON_PATH")]
    pub(crate) json_path: Option<String>,

    /// Write to FILE instead of stdout
    #[arg(short = 'o', long, value_name = "FILE")]
    pub(crate) output: Option<String>,

    /// print out json format
    #[arg(short = 'P', long)]
    pub(crate) json_output: bool,

    /// Number of requests to perform
    #[arg(short = 'N', long, default_value_t = 1)]
    pub(crate) repeat: u32,

    /// Random seed
    #[arg(long, value_name = "RANDOM_SEED")]
    pub(crate) seed: Option<i32>,

    /// Silent mode
    #[arg(long)]
    pub(crate) silent: bool,

    /// Output token per seconds
    #[arg(short, long)]
    pub(crate) tokens: bool,

    /// Tokenizer name for token counting
    #[arg(long, value_name = "TOKENIZER_NAME")]
    pub(crate) tokenizer: Option<String>,

    /// Make the operation more talkative
    #[arg(short, long)]
    pub(crate) verbose: bool,

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
            || self.data_urlencode.is_some()
            || !self.form.is_empty()
            || !self.form_string.is_empty()
            || self.dataset.is_some()
        {
            return Method::POST;
        }
        Method::GET
    }
}
