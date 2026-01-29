use clap::Parser;
use lmbench::args::Args;
use lmbench::run;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let cli = Args::parse();
    let _guard = if let Some(output_dir) = cli.output.clone() {
        tokio::fs::create_dir_all(&output_dir).await?;
        let file = std::fs::File::create(Path::new(&output_dir).join("output.log"))?;
        let (non_blocking, guard) = tracing_appender::non_blocking(file);
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_level(false)
            .with_target(false)
            .json()
            .flatten_event(true)
            .try_init()
            .ok();
        Some(guard)
    } else {
        None
    };
    let _ = run(cli).await?;
    Ok(())
}
