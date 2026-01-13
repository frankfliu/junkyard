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

use serde_json::{Value, from_str, json};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

type JobResult = Result<(Vec<Duration>, usize, usize, usize, usize, usize), anyhow::Error>;

pub async fn run(cli: Args) -> Result<Stats, anyhow::Error> {
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
        handles.push(tokio::spawn(async move {
            if let Some(delay) = get_delay(&cli.delay) {
                tokio::time::sleep(delay).await;
            }

            let mut latencies = Vec::new();
            let mut total_input_tokens = 0;
            let mut total_output_tokens = 0;
            let mut total_server_input_tokens = 0;
            let mut total_server_output_tokens = 0;
            let mut error_requests = 0;
            for i in 0..cli.repeat {
                if let Some(test_duration) = test_duration {
                    if start_time.elapsed() > test_duration {
                        break;
                    }
                }

                let record = dataset[i as usize % dataset.len()].clone();
                match process_single_record(&client, &cli, &record, total_requests).await {
                    Ok((
                        duration,
                        benchmark_output_tokens,
                        server_input_tokens,
                        server_output_tokens,
                    )) => {
                        latencies.push(duration);
                        total_output_tokens += benchmark_output_tokens;
                        total_server_input_tokens += server_input_tokens;
                        total_server_output_tokens += server_output_tokens;
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
                total_input_tokens,
                total_output_tokens,
                total_server_input_tokens,
                total_server_output_tokens,
                error_requests,
            ))
        }));
    }

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.await??);
    }

    if let Some(bar) = bar {
        bar.finish();
    }

    let mut all_latencies = Vec::new();
    let mut all_output_tokens = 0;
    let mut all_input_tokens = 0;
    let mut all_server_input_tokens = 0;
    let mut all_server_output_tokens = 0;
    let mut all_errors = 0;
    for (
        latencies,
        total_input_tokens,
        total_output_tokens,
        total_server_it,
        total_server_ot,
        error_requests,
    ) in results
    {
        all_latencies.extend(latencies);
        all_input_tokens += total_input_tokens;
        all_output_tokens += total_output_tokens;
        all_server_input_tokens += total_server_it;
        all_server_output_tokens += total_server_ot;
        all_errors += error_requests;
    }

    let stats = generate_stats(
        &all_latencies,
        all_input_tokens,
        all_output_tokens,
        all_server_input_tokens,
        all_server_output_tokens,
        all_errors,
    );
    if cli.verbose || total_requests > 1 {
        println!("{}", stats);
    }

    Ok(stats)
}

async fn process_single_record(
    client: &Client,
    cli: &Args,
    record: &Record,
    total_requests: u64,
) -> Result<(Duration, usize, usize, usize), anyhow::Error> {
    let request_builder = record.clone().into_request_builder(client);
    tracing::trace!({
       request = record.body,
    });

    let start = Instant::now();
    let res = request_builder.send().await.map_err(|e| {
        eprintln!("request send error: {:?}", e);
        e
    })?;
    let headers = res.headers().clone();
    let status = res.status();

    if cli.verbose && total_requests == 1 {
        println!("Status: {}", status);
        println!("Headers:\n{:#?}", headers);
        println!("Body:");
    }

    if !status.is_success() {
        if total_requests == 1 {
            let text = res.text().await?;
            eprintln!("{}", text);
        }
        return Err(anyhow::anyhow!("request failed: {}", status));
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

    let mut benchmark_output_tokens = 0;
    let mut server_input_tokens = 0;
    let mut server_output_tokens = 0;
    if cli.tokens {
        let (text_response, it, ot) = get_text_response(&cli, &headers, &body_text);
        let token_count = count_text_tokens(&text_response);
        server_input_tokens += it;
        server_output_tokens += ot;
        benchmark_output_tokens += token_count;
        if cli.output.is_some() {
            if cli.verbose {
                let mut serializable_headers = std::collections::HashMap::new();
                for (key, value) in headers.iter() {
                    serializable_headers
                        .insert(key.to_string(), value.to_str().unwrap_or("").to_string());
                }
                tracing::info!(
                    task_id = record.id,
                    duration = duration.as_millis(),
                    benchmark_output_tokens = token_count,
                    server_input_tokens = it,
                    server_output_tokens = ot,
                    generated_text = text_response,
                    response = body_text,
                    headers = serde_json::to_string(&serializable_headers).unwrap(),
                );
            } else {
                tracing::info!(
                    task_id = record.id,
                    duration = duration.as_millis(),
                    benchmark_output_tokens = token_count,
                    server_input_tokens = it,
                    server_output_tokens = ot,
                    generated_text = text_response,
                );
            }
        }
    } else {
        if cli.output.is_some() {
            tracing::info!(
                task_id = record.id,
                duration = duration.as_millis(),
                response = body_text,
            );
        }
    }

    Ok((
        duration,
        benchmark_output_tokens,
        server_input_tokens,
        server_output_tokens,
    ))
}

async fn load_dataset(cli: &Args) -> Result<Vec<Record>, anyhow::Error> {
    let mut record = Record::new(cli).await?;
    let Some(path) = &cli.dataset else {
        if let Some(body) = record.body {
            record.body = Some(convert_to_llm_format(cli, body));
        }
        record.set_extra_parameters(&cli.extra_parameters)?;
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
            data.push((i.to_string(), line.to_string()));
        }
    }

    let bar = if !cli.silent {
        let bar = ProgressBar::new(data.len() as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        Some(bar)
    } else {
        None
    };

    let records = data
        .into_iter()
        .map(|(k, v)| {
            let mut record = record.clone();
            record.id = k;
            record.body = Some(convert_to_llm_format(cli, v));
            record.set_extra_parameters(&cli.extra_parameters)?;
            if cli.tokens {
                record.input_tokens = record.count_input_tokens();
            }
            if let Some(bar) = &bar {
                bar.inc(1);
            }
            Ok(record)
        })
        .collect::<Result<Vec<Record>, anyhow::Error>>()?;

    if let Some(bar) = bar {
        bar.finish();
    }

    Ok(records)
}

fn convert_to_llm_format(cli: &Args, v: String) -> String {
    let Ok(json) = from_str::<Value>(&v) else {
        return v;
    };

    match cli.jq.as_deref() {
        Some("TGI") => {
            if let Some(inputs) = json.get("prompts") {
                json!({"inputs": inputs}).to_string()
            } else {
                v
            }
        }
        Some("gemini") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                json!({
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": inputs
                                }
                            ]
                        }
                    ]
                })
                .to_string()
            } else {
                v
            }
        }
        Some("anthropic") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                json!({
                    "anthropic_version": "vertex-2023-10-16",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": inputs
                        }
                    ]
                })
                .to_string()
            } else {
                v
            }
        }
        Some("openai") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                json!({
                    "messages": [
                        {
                            "role": "user",
                            "content": inputs
                        }
                    ]
                })
                .to_string()
            } else {
                v
            }
        }
        _ => v,
    }
}

fn get_delay(delay: &Option<String>) -> Option<Duration> {
    delay.as_ref().map(|d| {
        if d.starts_with("rand(") && d.ends_with(")") {
            let parts: Vec<&str> = d[5..d.len() - 1].split(',').map(|s| s.trim()).collect();
            if parts.len() == 1 {
                let max = parts[0].parse::<u64>().unwrap_or(0);
                Duration::from_millis(rand::rng().random_range(0..=max))
            } else if parts.len() == 2 {
                let min = parts[0].parse::<u64>().unwrap_or(0);
                let max = parts[1].parse::<u64>().unwrap_or(0);
                if min > max {
                    return Duration::from_millis(0);
                }
                Duration::from_millis(rand::rng().random_range(min..=max))
            } else {
                Duration::from_millis(0)
            }
        } else {
            Duration::from_millis(d.parse::<u64>().unwrap_or(0))
        }
    })
}

fn get_text_response(
    cli: &Args,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> (String, usize, usize) {
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if content_type.contains("text/event-stream") {
        let aggregated_resp = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != "[DONE]")
            .filter_map(|line| from_str::<Value>(line).ok())
            .map(|json| parse_json_response(&cli, &json, true, false))
            .collect::<Vec<(String, usize, usize)>>();
        let aggregated_text = aggregated_resp
            .iter()
            .map(|s| s.0.as_str())
            .collect::<Vec<_>>()
            .join("");
        if aggregated_text.is_empty() {
            eprintln!("warning: no output token found in response");
        }
        let input_tokens: usize = aggregated_resp.iter().map(|s| s.1).sum::<usize>();
        let output_tokens: usize = aggregated_resp.iter().map(|s| s.2).sum::<usize>();
        (aggregated_text, input_tokens, output_tokens)
    } else if content_type.contains("application/json") {
        let json: Value =
            from_str(body).unwrap_or_else(|_| panic!("Failed to parse json body: {}", body));
        parse_json_response(&cli, &json, false, true)
    } else {
        (body.to_string(), 0, 0)
    }
}

fn parse_json_response(
    cli: &Args,
    json: &Value,
    stream: bool,
    warn: bool,
) -> (String, usize, usize) {
    let jq_expr = &cli.get_jq_for_text(stream);
    let selected = jsonpath_lib::select(json, jq_expr)
        .unwrap_or_else(|_| panic!("Failed to apply jq: {}", jq_expr));
    if warn && selected.is_empty() {
        eprintln!("warning: jq expression returned no results: {}", jq_expr);
    }
    let text = selected
        .iter()
        .filter_map(|v| v.as_str())
        .collect::<Vec<&str>>()
        .join("");
    let input_tokens = if let Some(jq) = cli.get_jq_for_input_tokens() {
        let selected = jsonpath_lib::select(json, &jq)
            .unwrap_or_else(|_| panic!("Failed to apply jq: {}", jq));
        selected.iter().filter_map(|v| v.as_u64()).sum::<u64>() as usize
    } else {
        0
    };
    let output_tokens = if let Some(jq) = cli.get_jq_for_output_tokens() {
        let selected = jsonpath_lib::select(json, &jq)
            .unwrap_or_else(|_| panic!("Failed to apply jq: {}", jq));
        selected.iter().filter_map(|v| v.as_u64()).sum::<u64>() as usize
    } else {
        0
    };
    (text, input_tokens, output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Args;
    use clap::Parser;
    use reqwest::header::{HeaderMap, HeaderValue};
    use serde_json::json;
    use tempfile;

    fn default_args() -> Args {
        Args::parse_from(["lmbench", "https://localhost"])
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
        assert!(millis <= 100);

        let delay = get_delay(&Some("rand(100, 200)".to_string()));
        assert!(delay.is_some());
        let millis = delay.unwrap().as_millis();
        assert!(millis >= 100 && millis <= 200);

        assert_eq!(
            get_delay(&Some("rand(1,2,3)".to_string())),
            Some(Duration::from_millis(0))
        );

        assert_eq!(
            get_delay(&Some("invalid".to_string())),
            Some(Duration::from_millis(0))
        );
        assert_eq!(get_delay(&None), None);

        assert_eq!(
            get_delay(&Some("-100".to_string())),
            Some(Duration::from_millis(0))
        );

        assert_eq!(
            get_delay(&Some("rand(200, 100)".to_string())),
            Some(Duration::from_millis(0))
        );
    }

    #[test]
    fn test_get_text_response() {
        let args = default_args();
        let mut headers = HeaderMap::new();

        // Test with plain text
        let body = "hello world";
        assert_eq!(
            get_text_response(&args, &headers, body),
            ("hello world".to_string(), 0, 0)
        );

        // Test with application/json
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        let body = r#"{"foo": "bar"}"#;
        assert_eq!(
            get_text_response(&args, &headers, body),
            ("".to_string(), 0, 0)
        );

        // Test with text/event-stream
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let body = "data: {\"foo\": \"bar\"}\n\ndata: {\"foo\": \"baz\"}\n\ndata: [DONE]\n";
        assert_eq!(
            get_text_response(&args, &headers, body),
            ("".to_string(), 0, 0)
        );
    }

    #[tokio::test]
    async fn test_load_dataset_from_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();
        fs::write(&path, "{\"inputs\":\"line 1\"}\n{\"prompts\":\"line 2\"}").unwrap();

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "-j",
            "gemini",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].body,
            Some(r#"{"contents":[{"role":"user","parts":[{"text":"line 1"}]}]}"#.to_string())
        );
        assert_eq!(
            records[1].body,
            Some(r#"{"contents":[{"role":"user","parts":[{"text":"line 2"}]}]}"#.to_string())
        );

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "-j",
            "anthropic",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].body,
            Some(r#"{"anthropic_version":"vertex-2023-10-16","max_tokens":1024,"messages":[{"role":"user","content":"line 1"}]}"#.to_string())
        );
        assert_eq!(
            records[1].body,
            Some(r#"{"anthropic_version":"vertex-2023-10-16","max_tokens":1024,"messages":[{"role":"user","content":"line 2"}]}"#.to_string())
        );

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "-j",
            "openai",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0].body,
            Some(r#"{"messages":[{"role":"user","content":"line 1"}]}"#.to_string())
        );
        assert_eq!(
            records[1].body,
            Some(r#"{"messages":[{"role":"user","content":"line 2"}]}"#.to_string())
        );

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "-j",
            "TGI",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].body, Some(r#"{"inputs":"line 1"}"#.to_string()));
        assert_eq!(records[1].body, Some(r#"{"inputs":"line 2"}"#.to_string()));
    }

    #[tokio::test]
    async fn test_load_dataset_from_directory() {
        let mut args = default_args();
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("dummy")).unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        fs::write(dir.path().join("b.txt"), "content 2").unwrap();
        fs::write(dir.path().join("a.txt"), "content 1").unwrap();
        args.dataset = Some(path.into());
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "a");
        assert_eq!(records[0].body, Some("content 1".to_string()));
        assert_eq!(records[1].id, "b");
        assert_eq!(records[1].body, Some("content 2".to_string()));
    }

    #[test]
    #[should_panic(expected = "Failed to parse json body: invalid json")]
    fn test_get_text_response_panic_with_invalid_json() {
        let args = default_args();
        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        let body = "invalid json";
        get_text_response(&args, &headers, body);
    }

    #[test]
    fn test_parse_json_response() {
        let args = default_args();
        let json = json!([{"foo": "bar"}, {"foo": "baz"}]);
        assert_eq!(parse_json_response(&args, &json, false, false).0, "");

        let json = json!({"bar": "baz"});
        assert_eq!(parse_json_response(&args, &json, false, true).0, "");
    }

    #[test]
    #[should_panic(expected = "Failed to apply jq: invalid jq expression")]
    fn test_parse_json_response_panic_with_invalid_expression() {
        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "-j",
            "invalid jq expression",
        ]);
        let json = json!({"foo": "bar"});
        parse_json_response(&args, &json, false, false);
    }
}
