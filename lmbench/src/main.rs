use clap::Parser;
use lmbench::args::Args;
use lmbench::run;
use std::path::Path;
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let cli = Args::parse();

    let console_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(EnvFilter::from_default_env());

    let (file_layer, _guard) = if let Some(output_dir) = &cli.output {
        tokio::fs::create_dir_all(output_dir).await?;
        let file = std::fs::File::create(Path::new(output_dir).join("output.log"))?;
        let (non_blocking, guard) = tracing_appender::non_blocking(file);
        let layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_level(false)
            .with_target(false)
            .json()
            .flatten_event(true)
            .with_filter(EnvFilter::new("lmbench=info"));
        (Some(layer), Some(guard))
    } else {
        (None, None)
    };

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    let _ = run(cli).await?;
    Ok(())
}
