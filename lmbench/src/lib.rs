extern crate core;

pub mod args;
pub mod client_wrapper;
pub mod jq;
pub(crate) mod record;
pub mod stats;

use args::Args;
use client_wrapper::ClientWrapper;
use std::cmp::max;
use std::collections::HashSet;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

use rand::RngExt;
use rand_distr::{Distribution, Exp};
use record::{Record, count_text_tokens};
use stats::{RequestMetrics, ServingCollector, Stats, generate_stats};

#[cfg(feature = "harmony")]
use harmony_protocol::chat::{Content, Role};
#[cfg(feature = "harmony")]
use harmony_protocol::{HarmonyEncodingName, load_harmony_encoding};
use serde_json::{Value, from_str, json};
use std::fs;
use std::path::Path;
use std::sync::Arc;
#[cfg(feature = "harmony")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;

#[cfg(feature = "harmony")]
static HARMONY_PARSING_ENABLED: OnceLock<bool> = OnceLock::new();

#[derive(Default)]
struct JobStats {
    latencies: Vec<Duration>,
    ttfts: Vec<Duration>,
    input_tokens: usize,
    output_tokens: usize,
    server_input_tokens: usize,
    server_output_tokens: usize,
    error_requests: usize,
    thread_time: u128,
    serving: ServingCollector,
}

type JobResult = Result<JobStats, anyhow::Error>;

struct RequestResult {
    duration: Duration,
    ttft: Option<Duration>,
    benchmark_output_tokens: usize,
    server_input_tokens: usize,
    server_output_tokens: usize,
    metrics: Option<RequestMetrics>,
}

#[derive(Debug, Default, PartialEq)]
struct ParsedResponse {
    text: Vec<String>,
    server_input_tokens: usize,
    server_output_tokens: usize,
    server_cached_tokens: usize,
    /// (arrival time, text) of streamed frames carrying output text
    frames: Vec<(Duration, String)>,
}

pub async fn run(mut cli: Args) -> Result<Stats, anyhow::Error> {
    let client = ClientWrapper::new(&cli).await?;

    let dataset = Arc::new(load_dataset(&cli).await?);
    if cli.repeat == 0 {
        cli.repeat = dataset.len() as u32;
    }

    let mut handles: Vec<JoinHandle<JobResult>> = Vec::new();
    let start_time = Instant::now();
    let test_duration = cli.duration.map(|d| Duration::from_secs(d as u64));
    let finished_requests = Arc::new(AtomicU32::new(0));
    let index = Arc::new(AtomicUsize::new(0));

    let bar = if !cli.silent && cli.repeat > 1 {
        let bar = ProgressBar::new(cli.repeat as u64);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "Benchmarking: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );
        bar.set_position(0);
        Some(bar)
    } else {
        None
    };

    for _ in 0..cli.clients {
        let client = client.clone();
        let cli = cli.clone();
        let dataset = dataset.clone();
        let bar = bar.clone();
        let index = index.clone();
        let finished_requests = finished_requests.clone();
        handles.push(tokio::spawn(async move {
            let mut job = JobStats::default();
            while finished_requests.fetch_add(1, Ordering::SeqCst) < cli.repeat {
                if let Some(test_duration) = test_duration
                    && start_time.elapsed() > test_duration
                {
                    break;
                }

                if let Some(delay) = get_delay(&cli) {
                    tokio::time::sleep(delay).await;
                }

                let i = index.fetch_add(1, Ordering::SeqCst);
                let record = &dataset[i % dataset.len()];
                match process_single_record(&client, &cli, record, cli.repeat == 1).await {
                    Ok(result) => {
                        job.latencies.push(result.duration);
                        if let Some(t) = result.ttft {
                            job.ttfts.push(t);
                        }
                        job.output_tokens += result.benchmark_output_tokens;
                        job.server_input_tokens += result.server_input_tokens;
                        job.server_output_tokens += result.server_output_tokens;
                        job.thread_time += result.duration.as_millis();
                        if let Some(metrics) = result.metrics {
                            job.serving.add(metrics);
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            target: "error_log",
                            task_id = record.id,
                            error = format!("{:#}", e),
                        );
                        job.error_requests += 1;
                    }
                }
                job.input_tokens += record.input_tokens;
                if let Some(bar) = &bar {
                    bar.inc(1);
                }
            }
            Ok(job)
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
    let mut all_ttfts = Vec::new();
    let mut all_input_tokens = 0;
    let mut all_output_tokens = 0;
    let mut all_server_input_tokens = 0;
    let mut all_server_output_tokens = 0;
    let mut all_errors = 0;
    let mut max_thread_time = 0;
    let mut serving = ServingCollector::default();
    for job in results {
        all_latencies.extend(job.latencies);
        all_ttfts.extend(job.ttfts);
        all_input_tokens += job.input_tokens;
        all_output_tokens += job.output_tokens;
        all_server_input_tokens += job.server_input_tokens;
        all_server_output_tokens += job.server_output_tokens;
        all_errors += job.error_requests;
        max_thread_time = max(job.thread_time, max_thread_time);
        serving.merge(job.serving);
    }

    let mut stats = generate_stats(
        &all_latencies,
        &all_ttfts,
        all_input_tokens,
        all_output_tokens,
        all_server_input_tokens,
        all_server_output_tokens,
        all_errors,
        max_thread_time,
    );
    if !serving.is_empty() {
        stats.serving = Some(serving.generate_stats(cli.otps_floor));
    }
    if cli.verbose || cli.repeat > 1 {
        println!("{}", stats);
    }

    Ok(stats)
}

async fn process_single_record(
    client: &ClientWrapper,
    cli: &Args,
    record: &Record,
    print_to_console: bool,
) -> Result<RequestResult, anyhow::Error> {
    let response = client.send_request(cli, record, print_to_console).await?;
    let (body_text, headers, duration) = (response.body, response.headers, response.duration);

    let mut benchmark_output_tokens = 0;
    let parsed = get_text_response(cli, &headers, &body_text, &response.chunks);
    let (text_response, server_input_tokens, server_output_tokens) = (
        parsed.text,
        parsed.server_input_tokens,
        parsed.server_output_tokens,
    );
    // TTFT ends at the first frame carrying output tokens when streaming
    let ttft = parsed.frames.first().map_or(response.ttfb, |f| f.0);
    let mut opt_ttft = Some(ttft);
    let mut metrics = None;
    if cli.tokens {
        let token_count = text_response
            .iter()
            .map(|v| count_text_tokens(&v))
            .sum::<usize>();
        benchmark_output_tokens += token_count;
        if !parsed.frames.is_empty() {
            let frames: Vec<(Duration, usize)> = parsed
                .frames
                .iter()
                .map(|(t, text)| (*t, count_text_tokens(text).max(1)))
                .collect();
            let output_tokens = if server_output_tokens > 0 {
                server_output_tokens
            } else {
                token_count
            };
            let uncached_input_tokens = if server_input_tokens > 0 {
                server_input_tokens.saturating_sub(parsed.server_cached_tokens)
            } else {
                record.input_tokens
            };
            metrics = RequestMetrics::new(
                &frames,
                output_tokens,
                server_input_tokens,
                parsed.server_cached_tokens,
                uncached_input_tokens,
                cli.min_output_tokens,
            );
        }
        let otps = metrics.as_ref().and_then(|m| m.otps);
        let uncached_input_tokens = metrics.as_ref().map(|m| m.uncached_input_tokens);
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
                    ttft_us = ttft.as_micros(),
                    benchmark_output_tokens = token_count,
                    server_input_tokens = server_input_tokens,
                    server_output_tokens = server_output_tokens,
                    server_cached_tokens = parsed.server_cached_tokens,
                    uncached_input_tokens = uncached_input_tokens,
                    otps = otps,
                    generated_text = tracing::field::valuable(&text_response),
                    response = body_text,
                    headers = tracing::field::valuable(&serializable_headers),
                );
            } else {
                tracing::info!(
                    task_id = record.id,
                    duration = duration.as_millis(),
                    ttft_us = ttft.as_micros(),
                    benchmark_output_tokens = token_count,
                    server_input_tokens = server_input_tokens,
                    server_output_tokens = server_output_tokens,
                    server_cached_tokens = parsed.server_cached_tokens,
                    uncached_input_tokens = uncached_input_tokens,
                    otps = otps,
                    generated_text = tracing::field::valuable(&text_response),
                );
            }
        }
    } else if cli.output.is_some() {
        if text_response.is_empty() {
            tracing::info!(
                task_id = record.id,
                duration = duration.as_millis(),
                ttft_us = ttft.as_micros(),
                response = body_text,
            );
        } else {
            tracing::info!(
                task_id = record.id,
                duration = duration.as_millis(),
                ttft_us = ttft.as_micros(),
                server_input_tokens = server_input_tokens,
                server_output_tokens = server_output_tokens,
                generated_text = tracing::field::valuable(&text_response),
            );
        }
        opt_ttft = None;
    }

    Ok(RequestResult {
        duration,
        ttft: opt_ttft,
        benchmark_output_tokens,
        server_input_tokens,
        server_output_tokens,
        metrics,
    })
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

    let bar = if !cli.silent {
        let bar = ProgressBar::new(100);
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "Loading dataset: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )?
                .progress_chars("#>-"),
        );
        bar.set_position(0);
        Some(bar)
    } else {
        None
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
        let extension = path.extension().and_then(|e| e.to_str());
        let name = path.file_stem().unwrap().to_string_lossy().into_owned();
        if extension == Some("jsonl") {
            for (i, line) in content.lines().enumerate() {
                data.push((i.to_string(), line.to_string()));
            }
        } else if extension == Some("json") {
            let trimmed = content.trim();
            if trimmed.starts_with("[") && trimmed.ends_with("]") {
                if let Ok(Value::Array(arr)) = from_str::<Value>(trimmed) {
                    for (i, item) in arr.iter().enumerate() {
                        let s = if item.is_string() {
                            item.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(item)?
                        };
                        data.push((i.to_string(), s));
                    }
                }
            } else {
                data.push((name, content));
            }
        } else {
            data.push((name, content));
        }
    }

    if cli.retry_failed {
        if let Some(output_dir) = &cli.output {
            let output_file = Path::new(output_dir).join("output.log");
            if output_file.exists() {
                let content = fs::read_to_string(output_file)?;
                let mut finished_tasks = HashSet::new();
                for line in content.lines() {
                    if let Ok(v) = from_str::<Value>(line) {
                        let task_id = v["task_id"].as_str().unwrap_or("").to_string();
                        finished_tasks.insert(task_id);
                    }
                }
                data.retain(|(k, _)| !finished_tasks.contains(k));
            }
        }
    }

    if let Some(bar) = &bar {
        bar.set_length(data.len() as u64);
    }

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
    if let Some(expr) = cli.input_jq.as_deref() {
        let inputs = jq::jq(expr, &json);
        if inputs.is_empty() {
            panic!("Failed to convert to target LLM request format.");
        }
        return serde_json::to_string(&inputs[0]).unwrap();
    }

    match cli.server.as_deref() {
        Some("TGI") => {
            if let Some(inputs) = json.get("prompts") {
                return json!({"inputs": inputs}).to_string();
            }
        }
        Some("gemini") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                return json!({
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
                .to_string();
            }
        }
        Some("anthropic") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                return json!({
                    "anthropic_version": "vertex-2023-10-16",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": inputs
                        }
                    ]
                })
                .to_string();
            }
        }
        Some("openai") => {
            if let Some(inputs) = json.get("inputs").or(json.get("prompts")) {
                return json!({
                    "messages": [
                        {
                            "role": "user",
                            "content": inputs
                        }
                    ]
                })
                .to_string();
            }
        }
        _ => (),
    }
    v
}

fn get_delay(cli: &Args) -> Option<Duration> {
    if let Some(rate) = cli.request_rate {
        if rate <= 0.0 {
            return None;
        }
        let lambda = rate / cli.clients as f64;
        let exp = Exp::new(lambda).unwrap();
        return Some(Duration::from_secs_f64(exp.sample(&mut rand::rng())));
    }

    if let Some(d) = cli.delay.as_ref() {
        if d.starts_with("rand(") && d.ends_with(")") {
            let parts: Vec<&str> = d[5..d.len() - 1].split(',').map(|s| s.trim()).collect();
            if parts.len() == 1 {
                let max = parts[0].parse::<u64>().unwrap_or(0);
                return Some(Duration::from_millis(rand::rng().random_range(0..=max)));
            } else if parts.len() == 2 {
                let min = parts[0].parse::<u64>().unwrap_or(0);
                let max = parts[1].parse::<u64>().unwrap_or(0);
                if min > max {
                    return None;
                }
                return Some(Duration::from_millis(rand::rng().random_range(min..=max)));
            }
        } else if let Ok(ms) = d.parse::<u64>() {
            return Some(Duration::from_millis(ms));
        }
    }
    None
}

/// `chunks` are (cumulative body length, arrival time) of the received body chunks
fn get_text_response(
    cli: &Args,
    headers: &reqwest::header::HeaderMap,
    body: &str,
    chunks: &[(usize, Duration)],
) -> ParsedResponse {
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if content_type.contains("text/event-stream") {
        let mut parsed = ParsedResponse::default();
        let mut aggregated_text = String::new();
        let mut offset = 0;
        for line in body.split_inclusive('\n') {
            offset += line.len();
            let Some(data) = line.trim_end_matches(['\r', '\n']).strip_prefix("data: ") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(json) = from_str::<Value>(data) else {
                continue;
            };
            let resp = parse_json_response(cli, &json, true, false);
            if let Some(text) = resp.text.first()
                && !text.is_empty()
            {
                // a frame arrives with the chunk completing its line
                let i = chunks.partition_point(|(end, _)| *end < offset);
                let time = chunks.get(i).map_or(Duration::ZERO, |c| c.1);
                parsed.frames.push((time, text.clone()));
                aggregated_text.push_str(text);
            }
            // usage is a per request total, possibly repeated across frames
            parsed.server_input_tokens = max(parsed.server_input_tokens, resp.server_input_tokens);
            parsed.server_output_tokens =
                max(parsed.server_output_tokens, resp.server_output_tokens);
            parsed.server_cached_tokens =
                max(parsed.server_cached_tokens, resp.server_cached_tokens);
        }
        if aggregated_text.is_empty() {
            eprintln!("warning: no output token found in response");
        }
        parsed.text = vec![aggregated_text];
        parsed
    } else if content_type.contains("application/json") {
        let json: Value =
            from_str(body).unwrap_or_else(|_| panic!("Failed to parse json body: {}", body));
        parse_json_response(cli, &json, false, true)
    } else {
        ParsedResponse {
            text: vec![body.to_string()],
            ..Default::default()
        }
    }
}

fn sum_jq(expr: Option<String>, json: &Value) -> usize {
    let Some(expr) = expr else {
        return 0;
    };
    jq::jq(&expr, json)
        .iter()
        .filter_map(|v| v.as_u64())
        .sum::<u64>() as usize
}

fn parse_json_response(cli: &Args, json: &Value, stream: bool, warn: bool) -> ParsedResponse {
    let jq_expr = &cli.get_jq_for_text(stream);
    let selected = jq::jq(jq_expr, json);
    let mut text = Vec::new();
    for group in &selected {
        if let Some(s) = group.as_str() {
            text.push(extract_harmony_message(s));
        } else if let Some(arr) = group.as_array() {
            let joined = arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<&str>>()
                .join("");
            text.push(extract_harmony_message(&joined));
        }
    }
    if text.is_empty() && warn {
        eprintln!("warning: jq expression returned no results: {}", jq_expr);
    }
    // usage may arrive in a frame without text, e.g. openai stream_options.include_usage
    ParsedResponse {
        text,
        server_input_tokens: sum_jq(cli.get_jq_for_input_tokens(), json),
        server_output_tokens: sum_jq(cli.get_jq_for_output_tokens(), json),
        server_cached_tokens: sum_jq(cli.get_jq_for_cached_tokens(), json),
        frames: Vec::new(),
    }
}

#[cfg(feature = "harmony")]
fn extract_harmony_message(s: &str) -> String {
    // Check if harmony response parsing is enabled
    // In test mode, check the environment variable directly to allow tests to modify it
    // In production, use cached value for performance
    let harmony_parsing_enabled = if cfg!(test) {
        std::env::var("HARMONY_RESPONSE_PARSING")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false)
    } else {
        *HARMONY_PARSING_ENABLED.get_or_init(|| {
            std::env::var("HARMONY_RESPONSE_PARSING")
                .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                .unwrap_or(false)
        })
    };

    if !harmony_parsing_enabled || !s.contains("<|") || !s.contains("|>") {
        return s.to_string();
    }

    // Load the specific Harmony encoding used by the gpt-oss models
    let encoding = match load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss) {
        Ok(enc) => enc,
        Err(_) => return s.to_string(),
    };

    // Tokenize the response string, allowing all special tokens
    let allowed_special = encoding.tokenizer().special_tokens().clone();
    let (response_tokens, _) = encoding.tokenizer().encode(s, &allowed_special);

    // Parse the model's response tokens into structured `Message` objects
    let messages = match encoding
        .parse_messages_from_completion_tokens(response_tokens, Some(Role::Assistant))
    {
        Ok(msgs) => msgs,
        Err(_) => return s.to_string(),
    };

    // Extract text content from all parsed messages
    let mut result = String::new();
    for message in messages {
        for content in &message.content {
            match content {
                Content::Text(text_content) => {
                    // Extract text from content
                    result.push_str(&text_content.text);
                }
                _ => {
                    // Skip other content types
                }
            }
        }
    }

    // If we found message content, return it; otherwise return original string
    if !result.is_empty() {
        result
    } else {
        s.to_string()
    }
}

#[cfg(not(feature = "harmony"))]
fn extract_harmony_message(s: &str) -> String {
    s.to_string()
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
        let mut args = default_args();

        args.delay = Some("100".to_string());
        assert_eq!(get_delay(&args), Some(Duration::from_millis(100)));

        args.delay = Some("rand(100)".to_string());
        let delay = get_delay(&args);
        assert!(delay.is_some());
        let millis = delay.unwrap().as_millis();
        assert!(millis <= 100);

        args.delay = Some("rand(100, 200)".to_string());
        let delay = get_delay(&args);
        assert!(delay.is_some());
        let millis = delay.unwrap().as_millis();
        assert!(millis >= 100 && millis <= 200);

        args.delay = Some("rand(1,2,3)".to_string());
        assert_eq!(get_delay(&args), None);

        args.delay = Some("invalid".to_string());
        assert_eq!(get_delay(&args), None);

        args.delay = None;
        assert_eq!(get_delay(&args), None);

        args.delay = Some("-100".to_string());
        assert_eq!(get_delay(&args), None);

        args.delay = Some("rand_exp(100)".to_string());
        let delay = get_delay(&args);
        assert!(delay.is_none());

        args.delay = Some("rand(200, 100)".to_string());
        assert_eq!(get_delay(&args), None);

        args.delay = None;
        args.request_rate = Some(10.0);
        let delay = get_delay(&args);
        assert!(delay.is_some());
    }

    #[test]
    fn test_get_text_response() {
        let args = default_args();
        let mut headers = HeaderMap::new();

        // Test with plain text
        let body = "hello world";
        assert_eq!(
            get_text_response(&args, &headers, body, &[]).text,
            vec!["hello world".to_string()]
        );

        // Test with application/json
        headers.insert("content-type", HeaderValue::from_static("application/json"));
        let body = r#"{"foo": "bar"}"#;
        assert_eq!(
            get_text_response(&args, &headers, body, &[]),
            ParsedResponse::default()
        );

        // Test with text/event-stream
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let body = "data: {\"foo\": \"bar\"}\n\ndata: {\"foo\": \"baz\"}\n\ndata: [DONE]\n";
        assert_eq!(
            get_text_response(&args, &headers, body, &[]).text,
            vec!["".to_string()]
        );
    }

    #[test]
    fn test_get_text_response_stream_frames() {
        let args = Args::parse_from(["lmbench", "--server", "openai", "https://localhost"]);
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        let frames = [
            r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
            r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#,
            r#"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}"#,
            "data: [DONE]",
        ];
        let mut body = String::new();
        let mut chunks = Vec::new();
        for (i, frame) in frames.iter().enumerate() {
            body.push_str(frame);
            body.push_str("\n\n");
            chunks.push((body.len(), Duration::from_millis(i as u64 * 10)));
        }
        let parsed = get_text_response(&args, &headers, &body, &chunks);
        assert_eq!(parsed.text, vec!["hmmhello".to_string()]);
        assert_eq!(
            parsed.frames,
            vec![
                (Duration::from_millis(10), "hmm".to_string()),
                (Duration::from_millis(20), "hello".to_string())
            ]
        );
        assert_eq!(parsed.server_input_tokens, 10);
        assert_eq!(parsed.server_output_tokens, 2);
        assert_eq!(parsed.server_cached_tokens, 4);
    }

    #[tokio::test]
    async fn test_load_dataset_from_file() {
        let file = tempfile::NamedTempFile::with_suffix(".json").unwrap();
        let path = file.path().to_str().unwrap().to_string();
        fs::write(
            &path,
            "[{\"inputs\":\"line 1\"},\n{\"prompts\":\"line 2\"}]",
        )
        .unwrap();

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--server",
            "gemini",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            from_str::<Value>(records[0].body.as_ref().unwrap()).unwrap(),
            json!({"contents":[{"role":"user","parts":[{"text":"line 1"}]}]})
        );
        assert_eq!(
            from_str::<Value>(records[1].body.as_ref().unwrap()).unwrap(),
            json!({"contents":[{"role":"user","parts":[{"text":"line 2"}]}]})
        );

        let file = tempfile::NamedTempFile::with_suffix(".jsonl").unwrap();
        let path = file.path().to_str().unwrap().to_string();
        fs::write(&path, "{\"inputs\":\"line 1\"}\n{\"prompts\":\"line 2\"}").unwrap();
        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--server",
            "anthropic",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            from_str::<Value>(records[0].body.as_ref().unwrap()).unwrap(),
            json!({"anthropic_version":"vertex-2023-10-16","max_tokens":1024,"messages":[{"role":"user","content":"line 1"}]})
        );
        assert_eq!(
            from_str::<Value>(records[1].body.as_ref().unwrap()).unwrap(),
            json!({"anthropic_version":"vertex-2023-10-16","max_tokens":1024,"messages":[{"role":"user","content":"line 2"}]})
        );

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--server",
            "openai",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            from_str::<Value>(records[0].body.as_ref().unwrap()).unwrap(),
            json!({"messages":[{"role":"user","content":"line 1"}]})
        );
        assert_eq!(
            from_str::<Value>(records[1].body.as_ref().unwrap()).unwrap(),
            json!({"messages":[{"role":"user","content":"line 2"}]})
        );

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--server",
            "TGI",
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].body, Some(r#"{"inputs":"line 1"}"#.to_string()));
        assert_eq!(records[1].body, Some(r#"{"inputs":"line 2"}"#.to_string()));

        let file = tempfile::NamedTempFile::with_suffix(".txt").unwrap();
        let path = file.path().to_str().unwrap().to_string();
        fs::write(&path, r#"{"inputs":"line 1"}"#).unwrap();

        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--input-jq",
            r#"{"prompt": .inputs}"#,
            "--dataset",
            path.as_str(),
        ]);
        let records = load_dataset(&args).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].body, Some(r#"{"prompt":"line 1"}"#.to_string()));
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
        get_text_response(&args, &headers, body, &[]);
    }

    #[test]
    fn test_parse_json_response() {
        let args = default_args();

        let json = json!({"generated_text": "test"});
        assert_eq!(
            parse_json_response(&args, &json, false, false).text[0],
            "test"
        );

        let json = json!({"bar": "baz"});
        assert!(
            parse_json_response(&args, &json, false, true)
                .text
                .is_empty()
        );

        // Test with string primitive
        let mut args = default_args();
        args.server = None;
        args.output_jq = Some(".foo".to_string());
        let json = json!({"foo": "bar"});
        assert_eq!(
            parse_json_response(&args, &json, false, false).text[0],
            "bar"
        );

        // Test with array of strings
        args.output_jq = Some(".foo".to_string());
        let json = json!({"foo": ["bar", "baz"]});
        assert_eq!(
            parse_json_response(&args, &json, false, false).text[0],
            "barbaz"
        );
    }

    #[test]
    #[should_panic(expected = "Failed to parse jq expression")]
    fn test_parse_json_response_panic_with_invalid_expression() {
        let args = Args::parse_from([
            "lmbench",
            "https://localhost",
            "--output-jq",
            "invalid jq expression",
        ]);
        let json = json!({"foo": "bar"});
        parse_json_response(&args, &json, false, false);
    }

    #[test]
    #[cfg(feature = "harmony")]
    fn test_extract_harmony_message() {
        // Test with harmony parsing disabled (default)
        let harmony_text = "<|start|>assistant<|channel|>analysis<|message|>Hello world!<|end|>";
        // Without harmony parsing enabled, should return original string
        assert_eq!(extract_harmony_message(harmony_text), harmony_text);

        // Enable harmony parsing for this test
        unsafe {
            std::env::set_var("HARMONY_RESPONSE_PARSING", "true");
        }

        // Test basic harmony response format that successfully parses
        let harmony_text = "<|start|>assistant<|channel|>analysis<|message|>Hello world!<|end|>";
        assert_eq!(extract_harmony_message(harmony_text), "Hello world!");

        // Test with message tag
        let harmony_text = "<|start|><|channel|>main<|message|>Test message<|end|>";
        let result = extract_harmony_message(harmony_text);
        // This format may or may not parse successfully depending on the harmony library
        // If it extracts text, verify it's correct; otherwise it returns the original
        assert!(result == "Test message" || result == harmony_text);

        // Test non-harmony text (should return as-is)
        let normal_text = "Just a normal response";
        assert_eq!(
            extract_harmony_message(normal_text),
            "Just a normal response"
        );

        // Test text without harmony markers (should return as-is even with parsing enabled)
        let simple_text = "Regular text without special tokens";
        assert_eq!(
            extract_harmony_message(simple_text),
            "Regular text without special tokens"
        );

        // Clean up
        unsafe {
            std::env::remove_var("HARMONY_RESPONSE_PARSING");
        }
    }
}
