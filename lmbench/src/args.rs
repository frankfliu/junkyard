use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
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

    /// Pass custom header LINE to server
    #[arg(short = 'H', long, value_name = "LINE")]
    pub(crate) header: Vec<String>,

    /// Include protocol headers in the output
    #[arg(short, long)]
    pub(crate) include: bool,

    /// Json query expression for token output
    #[arg(short, long, value_name = "EXPRESSION")]
    pub(crate) jq: Option<String>,

    /// Number of requests to perform
    #[arg(short = 'N', long, default_value_t = 1)]
    pub(crate) repeat: u32,

    /// Write to FILE instead of stdout
    #[arg(short, long, value_name = "FILE")]
    pub(crate) output: Option<String>,

    /// print out json format
    #[arg(short = 'P', long)]
    pub(crate) json_output: bool,

    /// Random seed
    #[arg(long, value_name = "RANDOM_SEED")]
    pub(crate) seed: Option<i32>,

    /// Transfer FILE to destination
    #[arg(short = 'T', long, value_name = "FILE")]
    pub(crate) upload_file: Option<PathBuf>,

    /// Output token per seconds
    #[arg(short, long)]
    pub(crate) tokens: bool,

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
