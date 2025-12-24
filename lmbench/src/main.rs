pub(crate) mod args;
pub(crate) mod record;
pub(crate) mod stats;

use args::Args;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use parking_lot::Mutex;
use rand::Rng;
use record::Record;
use reqwest::Client;

use serde_json::{Value, from_str};
use stats::generate_stats;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;
use tokio::task::JoinHandle;

lazy_static! {
    static ref TOKENIZER_CACHE: Mutex<HashMap<String, Tokenizer>> = Mutex::new(HashMap::new());
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let cli = Args::parse();

    if cli.verbose {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    } else {
        unsafe { std::env::set_var("RUST_LOG", "warn") };
    }
    env_logger::init();

    let client = Client::builder()
        .timeout(Duration::from_secs(cli.connect_timeout))
        .build()?;

    let dataset = if let Some(path) = &cli.dataset {
        load_dataset(path.to_str().unwrap(), &cli.extra_parameters)?
    } else {
        Vec::new()
    };

    let mut handles: Vec<JoinHandle<Result<(Vec<Duration>, usize, usize), anyhow::Error>>> =
        Vec::new();
    let start_time = Instant::now();
    let test_duration = cli.duration.map(|d| Duration::from_secs(d as u64));

    let bar = if !cli.silent {
        let bar = ProgressBar::new((cli.clients * cli.repeat) as u64);
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
            let mut total_output_tokens = 0;
            let mut total_input_tokens = 0;
            for i in 0..cli.repeat {
                if let Some(bar) = &bar {
                    bar.inc(1);
                }
                if let Some(test_duration) = test_duration {
                    if start_time.elapsed() > test_duration {
                        break;
                    }
                }

                let body = if !dataset.is_empty() {
                    Some(dataset[i as usize % dataset.len()].clone())
                } else if let Some(data) = &cli.data {
                    Some(data.clone())
                } else if let Some(data) = &cli.data_raw {
                    Some(data.clone())
                } else if let Some(data) = &cli.data_urlencode {
                    let encoded: String = serde_urlencoded::to_string(data)?;
                    Some(encoded)
                } else {
                    None
                };

                if cli.tokens {
                    if let Some(body_str) = &body {
                        total_input_tokens += count_tokens(body_str, &cli.tokenizer)?;
                    }
                }

                let record = Record::new(&cli, body);
                let request_builder = record.into_request_builder(&client, &cli.url).await?;

                let start = Instant::now();
                let res = request_builder.send().await?;
                let duration = start.elapsed();

                latencies.push(duration);

                let status = res.status();
                let headers = res.headers().clone();
                let body_text = res.text().await?;

                if cli.tokens {
                    if let Some(jq_expr) = &cli.jq {
                        let json: Value = from_str(&body_text)?;
                        let tokens_val = jsonpath_lib::select(&json, jq_expr)?;
                        if let Some(tokens) = tokens_val.get(0) {
                            if let Some(s) = tokens.as_str() {
                                total_output_tokens += count_tokens(s, &cli.tokenizer)?;
                            }
                        }
                    } else {
                        total_output_tokens += count_tokens(&body_text, &cli.tokenizer)?;
                    }
                }

                if cli.include {
                    println!("Status: {}", status);
                    println!("Headers:\n{:#?}", headers);
                    println!("Body:\n{}", body_text);
                }
            }
            Ok((latencies, total_output_tokens, total_input_tokens))
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
    for (latencies, total_output_tokens, total_input_tokens) in results {
        all_latencies.extend(latencies);
        all_output_tokens += total_output_tokens;
        all_input_tokens += total_input_tokens;
    }

    let stats = generate_stats(&all_latencies, all_output_tokens, all_input_tokens);
    if cli.json_output || cli.json_path.is_some() {
        let json = serde_json::to_string_pretty(&stats)?;
        if let Some(path) = cli.json_path {
            fs::write(path, json)?;
        } else {
            println!("{}", json);
        }
    } else {
        println!("{}", stats);
    }

    Ok(())
}

fn load_dataset(
    path: &str,
    extra_parameters: &Option<String>,
) -> Result<Vec<String>, anyhow::Error> {
    let path = Path::new(path);
    let mut data = Vec::new();
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                let content = fs::read_to_string(path)?;
                data.push(content);
            }
        }
    } else {
        let content = fs::read_to_string(path)?;
        data.push(content);
    }

    if let Some(extra_params_str) = extra_parameters {
        let extra_params: Value = from_str(extra_params_str)?;
        for item in &mut data {
            let mut json: Value = from_str(item)?;
            if let (Some(obj), Some(extra_obj)) = (json.as_object_mut(), extra_params.as_object()) {
                for (k, v) in extra_obj {
                    obj.insert(k.clone(), v.clone());
                }
                *item = serde_json::to_string(&json)?;
            }
        }
    }

    Ok(data)
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

fn count_tokens(text: &str, tokenizer_name: &Option<String>) -> Result<usize, anyhow::Error> {
    if let Some(tokenizer_name) = tokenizer_name {
        let mut cache = TOKENIZER_CACHE.lock();
        if !cache.contains_key(tokenizer_name) {
            let tokenizer = Tokenizer::from_pretrained(tokenizer_name, None)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
            cache.insert(tokenizer_name.clone(), tokenizer);
        }
        let tokenizer = cache.get(tokenizer_name).unwrap();
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().len())
    } else {
        // Default to a simple word count if no tokenizer is specified
        Ok(text.split_whitespace().count())
    }
}
