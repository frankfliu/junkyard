use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long, default_value_t = false)]
    pub pretty: bool,

    /// Folder to store captured requests
    #[arg(short, long)]
    pub output: Option<String>,

    /// JSONL file containing mock generated content
    #[arg(short, long)]
    pub responses: Option<String>,
}
