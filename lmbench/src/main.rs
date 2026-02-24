use clap::Parser;
use lmbench::args::Args;
use lmbench::run;
use std::path::Path;
use std::process::ExitCode;
use tracing_subscriber::{EnvFilter, Layer, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Args::parse();

    let console_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(EnvFilter::from_default_env());

    let (file_layer, _guard) = if let Some(output_dir) = &cli.output {
        tokio::fs::create_dir_all(output_dir).await.unwrap();
        let log_path = Path::new(output_dir).join("output.log");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(cli.retry_failed)
            .truncate(!cli.retry_failed)
            .open(log_path)
            .unwrap();
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

    let stat = run(cli.clone()).await.unwrap();
    tracing::info!(task_id = "-1", total_time_ms = stat.total_time_ms);

    if stat.get_error_rate() > cli.error_rate {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
