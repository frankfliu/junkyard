use clap::Parser;
use lmbench::args::Args;
use lmbench::run;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let cli = Args::parse();
    run(cli).await
}
