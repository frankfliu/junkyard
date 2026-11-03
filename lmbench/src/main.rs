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
        .with_filter(EnvFilter::from_default_env().add_directive("error_log=off".parse().unwrap()));

    let (file_layer, error_layer, _guards) = if let Some(output_dir) = &cli.output {
        tokio::fs::create_dir_all(output_dir).await.unwrap();
        let open_log = |name: &str| {
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(cli.retry_failed)
                .truncate(!cli.retry_failed)
                .open(Path::new(output_dir).join(name))
                .unwrap()
        };
        let (non_blocking, guard) = tracing_appender::non_blocking(open_log("output.log"));
        let layer = fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_level(false)
            .with_target(false)
            .json()
            .flatten_event(true)
            .with_filter(EnvFilter::new("lmbench=info"));
        let (error_writer, error_guard) = tracing_appender::non_blocking(open_log("error.log"));
        let error_layer = fmt::layer()
            .with_writer(error_writer)
            .with_ansi(false)
            .with_level(false)
            .with_target(false)
            .json()
            .flatten_event(true)
            .with_filter(EnvFilter::new("error_log=error"));
        (Some(layer), Some(error_layer), Some((guard, error_guard)))
    } else {
        (None, None, None)
    };

    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .with(error_layer)
        .init();

    let stat = run(cli.clone()).await.unwrap();
    tracing::info!(task_id = "-1", total_time_ms = stat.total_time_ms);

    if stat.get_error_rate() > cli.error_rate {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
