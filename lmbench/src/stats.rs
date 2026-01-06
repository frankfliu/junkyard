use serde::Serialize;
use std::fmt;
use std::time::Duration;

#[derive(Serialize, Debug)]
pub struct Stats {
    pub success_requests: usize,
    pub error_requests: usize,
    pub total_time_ms: u128,
    pub avg_latency_ms: u128,
    pub min_latency_ms: u128,
    pub max_latency_ms: u128,
    pub p50_latency_ms: u128,
    pub p90_latency_ms: u128,
    pub p99_latency_ms: u128,
    pub tps: f64,
    pub total_output_tokens: usize,
    pub output_tokens_per_second: f64,
    pub total_input_tokens: usize,
    pub input_tokens_per_second: f64,
}

pub fn generate_stats(
    latencies: &[Duration],
    total_output_tokens: usize,
    total_input_tokens: usize,
    error_requests: usize,
) -> Stats {
    if latencies.is_empty() {
        return Stats {
            success_requests: 0,
            error_requests,
            total_time_ms: 0,
            avg_latency_ms: 0,
            min_latency_ms: 0,
            max_latency_ms: 0,
            p50_latency_ms: 0,
            p90_latency_ms: 0,
            p99_latency_ms: 0,
            tps: 0.0,
            total_output_tokens: 0,
            output_tokens_per_second: 0.0,
            total_input_tokens: 0,
            input_tokens_per_second: 0.0,
        };
    }

    let mut sorted_latencies = latencies.to_vec();
    sorted_latencies.sort();

    let success_requests = latencies.len();
    let total_time: Duration = latencies.iter().sum();
    let total_time_ms = total_time.as_millis();
    let avg_latency_ms = total_time_ms / success_requests as u128;
    let min_latency_ms = sorted_latencies.first().unwrap().as_millis();
    let max_latency_ms = sorted_latencies.last().unwrap().as_millis();
    let p50_latency_ms = sorted_latencies[success_requests / 2].as_millis();
    let p90_latency_ms = sorted_latencies[(success_requests as f64 * 0.9) as usize].as_millis();
    let p99_latency_ms = sorted_latencies[(success_requests as f64 * 0.99) as usize].as_millis();
    let tps = success_requests as f64 / total_time.as_secs_f64();
    let output_tokens_per_second = total_output_tokens as f64 / total_time.as_secs_f64();
    let input_tokens_per_second = total_input_tokens as f64 / total_time.as_secs_f64();

    Stats {
        success_requests,
        error_requests,
        total_time_ms,
        avg_latency_ms,
        min_latency_ms,
        max_latency_ms,
        p50_latency_ms,
        p90_latency_ms,
        p99_latency_ms,
        tps,
        total_output_tokens,
        output_tokens_per_second,
        total_input_tokens,
        input_tokens_per_second,
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string_pretty(&self).unwrap();
        writeln!(f, "{}", json)
    }
}
