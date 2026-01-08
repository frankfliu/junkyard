pub mod args;
pub(crate) mod record;
pub mod stats;

use args::Args;

use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use rand::Rng;
use record::{Record, count_text_tokens};
use reqwest::Client;

use stats::{Stats, generate_stats};

use serde_json::{Value, from_str};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

type JobResult = Result<(Vec<Duration>, usize, usize, usize), anyhow::Error>;

pub async fn run(cli: Args) -> Result<Stats, anyhow::Error> {
    let (tx, rx) = mpsc::channel::<(String, String)>(1024);

    let writer_handle = if let Some(output_dir) = cli.output.clone() {
        tokio::fs::create_dir_all(&output_dir).await?;
        Some(tokio::spawn(writer_thread(rx, output_dir)))
    } else {
        None
    };

    let client = Client::builder()
        .timeout(Duration::from_secs(cli.connect_timeout))
        .build()?;

    let dataset = load_dataset(&cli).await?;

    let mut handles: Vec<JoinHandle<JobResult>> = Vec::new();
    let start_time = Instant::now();
    let test_duration = cli.duration.map(|d| Duration::from_secs(d as u64));
    let total_requests = (cli.clients * cli.repeat) as u64;

    let bar = if !cli.silent && total_requests > 1 {
        let bar = ProgressBar::new(total_requests);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );
        Some(bar)
    } else {
        None
    };

    for _ in 0..cli.clients {
        let client = client.clone();
        let cli = cli.clone();
        let dataset = dataset.clone();
        let bar = bar.clone();
        let tx = tx.clone();
        handles.push(tokio::spawn(async move {
            if let Some(delay) = get_delay(&cli.delay) {
                tokio::time::sleep(delay).await;
            }

            let mut latencies = Vec::new();
            let mut total_output_tokens = 0;
            let mut total_input_tokens = 0;
            let mut error_requests = 0;
            for i in 0..cli.repeat {
                if let Some(test_duration) = test_duration {
                    if start_time.elapsed() > test_duration {
                        break;
                    }
                }

                let record = dataset[i as usize % dataset.len()].clone();
                match process_single_record(&client, &cli, &record, total_requests, &tx).await {
                    Ok((duration, output_tokens)) => {
                        latencies.push(duration);
                        total_output_tokens += output_tokens;
                    }
                    Err(_) => {
                        error_requests += 1;
                    }
                }
                total_input_tokens += record.input_tokens;
                if let Some(bar) = &bar {
                    bar.inc(1);
                }
            }
            Ok((
                latencies,
                total_output_tokens,
                total_input_tokens,
                error_requests,
            ))
        }));
    }

    drop(tx);

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await??);
    }

    if let Some(writer_handle) = writer_handle {
        writer_handle.await?;
    }

    if let Some(bar) = bar {
        bar.finish();
    }

    let mut all_latencies = Vec::new();
    let mut all_output_tokens = 0;
    let mut all_input_tokens = 0;
    let mut all_errors = 0;
    for (latencies, total_output_tokens, total_input_tokens, error_requests) in results {
        all_latencies.extend(latencies);
        all_output_tokens += total_output_tokens;
        all_input_tokens += total_input_tokens;
        all_errors += error_requests;
    }

    let stats = generate_stats(
        &all_latencies,
        all_output_tokens,
        all_input_tokens,
        all_errors,
    );
    if total_requests > 1 {
        println!("{}", stats);
    }

    Ok(stats)
}

async fn process_single_record(
    client: &Client,
    cli: &Args,
    record: &Record,
    total_requests: u64,
    tx: &mpsc::Sender<(String, String)>,
) -> Result<(Duration, usize), anyhow::Error> {
    let request_builder = record.clone().into_request_builder(client);

    let start = Instant::now();
    let res = request_builder.send().await?;

    if !res.status().is_success() {
        return Err(anyhow::anyhow!("request failed: {}", res.status()));
    }

    let headers = res.headers().clone();
    let status = res.status();

    if cli.include && total_requests == 1 {
        println!("Status: {}", status);
        println!("Headers:\n{:#?}", headers);
        println!("Body:");
    }

    let mut stream = res.bytes_stream();
    let mut body_bytes = Vec::new();

    while let Some(item) = stream.next().await {
        let chunk = item?;
        if total_requests == 1 {
            print!("{}", String::from_utf8_lossy(&chunk));
        }
        body_bytes.extend_from_slice(&chunk);
    }
    if total_requests == 1 {
        println!();
    }
    let body_text = String::from_utf8(body_bytes)?;

    let duration = start.elapsed();

    let mut output_tokens = 0;
    if cli.tokens {
        let text_response = get_text_response(&cli, &headers, &body_text);
        output_tokens += count_text_tokens(&text_response);
        if cli.output.is_some() {
            let json = if cli.include {
                let mut serializable_headers = std::collections::HashMap::new();
                for (key, value) in headers.iter() {
                    serializable_headers
                        .insert(key.to_string(), value.to_str().unwrap_or("").to_string());
                }
                serde_json::json!({
                    "duration": duration.as_millis(),
                    "response": body_text,
                    "headers": serializable_headers,
                })
            } else {
                serde_json::json!({
                    "duration": duration.as_millis(),
                    "response": text_response,
                })
            };
            tx.send((record.id.clone(), json.to_string())).await?;
        }
    } else {
        if cli.output.is_some() {
            tx.send((record.id.clone(), body_text.clone())).await?;
        }
    }

    Ok((duration, output_tokens))
}

async fn load_dataset(cli: &Args) -> Result<Vec<Record>, anyhow::Error> {
    let mut record = Record::new(cli).await?;
    let Some(path) = &cli.dataset else {
        record.input_tokens = record.count_input_tokens();
        return Ok(vec![record]);
    };

    let path = Path::new(path);
    let mut data: Vec<(String, String)> = Vec::new();
    if path.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(path)?
            .map(|res| res.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort();
        for path in entries {
            if path.is_file() {
                let content = fs::read_to_string(&path)?;
                data.push((
                    path.file_stem().unwrap().to_string_lossy().into_owned(),
                    content,
                ));
            }
        }
    } else {
        let content = fs::read_to_string(path)?;
        for (i, line) in content.lines().enumerate() {
            data.push(((i + 1).to_string(), line.to_string()));
        }
    }

    data.into_iter()
        .map(|(k, v)| {
            let mut record = record.clone();
            record.id = k;
            record.body = Some(v);
            record.set_extra_parameters(&cli.extra_parameters)?;
            record.input_tokens = record.count_input_tokens();
            Ok(record)
        })
        .collect()
}

fn get_delay(delay: &Option<String>) -> Option<Duration> {
    delay.as_ref().map(|d| {
        if d.starts_with("rand(") {
            let parts: Vec<&str> = d[5..d.len() - 1].split(',').map(|s| s.trim()).collect();
            if parts.len() == 1 {
                let max = parts[0].parse().unwrap_or(0);
                Duration::from_millis(rand::rng().random_range(0..max))
            } else if parts.len() == 2 {
                let min = parts[0].parse().unwrap_or(0);
                let max = parts[1].parse().unwrap_or(0);
                Duration::from_millis(rand::rng().random_range(min..max))
            } else {
                Duration::from_millis(0)
            }
        } else {
            Duration::from_millis(d.parse().unwrap_or(0))
        }
    })
}

fn get_text_response(cli: &Args, headers: &reqwest::header::HeaderMap, body: &str) -> String {
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if content_type.contains("text/event-stream") {
        let aggregated_body = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != "[DONE]")
            .filter_map(|line| from_str::<Value>(line).ok())
            .map(|json| apply_jq(&json, &cli.get_jq(true), false))
            .collect::<Vec<String>>()
            .join("");
        if aggregated_body.is_empty() {
            eprintln!("warning: no output token found in response");
        }
        return aggregated_body;
    } else if content_type.contains("application/json") {
        let json: Value =
            from_str(body).unwrap_or_else(|_| panic!("Failed to parse json body: {}", body));
        return apply_jq(&json, &cli.get_jq(false), true);
    }
    body.to_string()
}

fn apply_jq(json: &Value, jq_expr: &str, warn: bool) -> String {
    let tokens_val = jsonpath_lib::select(json, jq_expr)
        .unwrap_or_else(|_| panic!("Failed to apply jq: {}", jq_expr));
    if warn && tokens_val.is_empty() {
        eprintln!("warning: jq expression returned no results: {}", jq_expr);
    }
    tokens_val
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<&str>>()
        .join("")
}

async fn writer_thread(mut rx: mpsc::Receiver<(String, String)>, output_dir: String) {
    let mut seen_ids = HashSet::new();
    while let Some((id, body_text)) = rx.recv().await {
        let path = Path::new(&output_dir).join(format!("{}.jsonlines", id));
        let mut options = tokio::fs::OpenOptions::new();
        options.create(true);
        if seen_ids.contains(&id) {
            options.append(true);
        } else {
            options.write(true).truncate(true);
            seen_ids.insert(id);
        }
        let mut file = match options.open(&path).await {
            Ok(file) => file,
            Err(e) => {
                eprintln!("failed to open file: {}: {}", path.display(), e);
                continue;
            }
        };
        if let Err(e) = file.write_all(format!("{}\n", body_text).as_bytes()).await {
            eprintln!("failed to write to file: {}: {}", path.display(), e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Args;
    use reqwest::header::{HeaderMap, HeaderValue};
    use serde_json::json;
    use tempfile;

    fn default_args() -> Args {
        Args {
            clients: 1,
            connect_timeout: 60,
            data: None,
            data_raw: None,
            data_urlencoded: None,
            dataset: None,
            delay: None,
            duration: None,
            extra_parameters: None,
            form: vec![],
            form_string: vec![],
            get: false,
            header: vec![],
            include: false,
            jq: None,
            output: None,
            repeat: 1,
            seed: None,
            silent: false,
            tokens: false,
            request: None,
            url: "https://localhost".to_string(),
        }
    }

    #[test]
    fn test_get_delay() {
        assert_eq!(
            get_delay(&Some("100".to_string())),
            Some(Duration::from_millis(100))
        );

        let delay = get_delay(&Some("rand(100)".to_string()));
        assert!(delay.is_some());
        let millis = delay.unwrap().as_millis();
        assert!(millis < 100);

        let delay = get_delay(&Some("rand(100, 200)".to_string()));
        assert!(delay.is_some());
        let millis = delay.unwrap().as_millis();
        assert!(millis >= 100 && millis < 200);

        assert_eq!(
            get_delay(&Some("invalid".to_string())),
            Some(Duration::from_millis(0))
        );
        assert_eq!(get_delay(&None), None);
    }

    #[test]
    fn test_apply_jq() {
        let json = json!([{"foo": "bar"}, {"foo": "baz"}]);
        assert_eq!(apply_jq(&json, "$[*].foo", false), "barbaz");

        let json = json!({"bar": "baz"});
        assert_eq!(apply_jq(&json, "$.foo", false), "");
    }

    #[test]
    fn test_get_text_response() {
        let args = default_args();
        let mut headers = HeaderMap::new();

        // Test with plain text
        let body = "hello world";
        assert_eq!(get_text_response(&args, &headers, body), "hello world");

        // Test with application/json
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        let body = "{\"foo\": \"bar\"}";
        assert_eq!(get_text_response(&args, &headers, body), "");

        // Test with text/event-stream
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let body = "data: {\"foo\": \"bar\"}\n\ndata: {\"foo\": \"baz\"}\n\ndata: [DONE]\n";
        assert_eq!(get_text_response(&args, &headers, body), "");
    }

    #[tokio::test]
    async fn test_load_dataset_from_file() {
        let mut args = default_args();
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();
        std::fs::write(&path, "line 1\nline 2").unwrap();
        args.dataset = Some(path.into());
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].body, Some("line 1".to_string()));
        assert_eq!(records[1].body, Some("line 2".to_string()));
    }

    #[tokio::test]
    async fn test_load_dataset_from_directory() {
        let mut args = default_args();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        std::fs::write(dir.path().join("b.txt"), "content 2").unwrap();
        std::fs::write(dir.path().join("a.txt"), "content 1").unwrap();
        args.dataset = Some(path.into());
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "a");
        assert_eq!(records[0].body, Some("content 1".to_string()));
        assert_eq!(records[1].id, "b");
        assert_eq!(records[1].body, Some("content 2".to_string()));
    }
}
