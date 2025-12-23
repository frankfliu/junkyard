use serde::Serialize;
use std::fmt;
use std::time::Duration;

#[derive(Serialize, Debug)]
pub struct Stats {
    pub total_requests: usize,
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
) -> Stats {
    if latencies.is_empty() {
        return Stats {
            total_requests: 0,
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

    let total_requests = latencies.len();
    let total_time: Duration = latencies.iter().sum();
    let total_time_ms = total_time.as_millis();
    let avg_latency_ms = total_time_ms / total_requests as u128;
    let min_latency_ms = sorted_latencies.first().unwrap().as_millis();
    let max_latency_ms = sorted_latencies.last().unwrap().as_millis();
    let p50_latency_ms = sorted_latencies[total_requests / 2].as_millis();
    let p90_latency_ms = sorted_latencies[(total_requests as f64 * 0.9) as usize].as_millis();
    let p99_latency_ms = sorted_latencies[(total_requests as f64 * 0.99) as usize].as_millis();
    let tps = total_requests as f64 / total_time.as_secs_f64();
    let output_tokens_per_second = total_output_tokens as f64 / total_time.as_secs_f64();
    let input_tokens_per_second = total_input_tokens as f64 / total_time.as_secs_f64();

    Stats {
        total_requests,
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
        writeln!(f, "--- Summary ---")?;
        writeln!(f, "Total Requests: {}", self.total_requests)?;
        writeln!(f, "Total Time: {}ms", self.total_time_ms)?;
        writeln!(f, "Average Latency: {}ms", self.avg_latency_ms)?;
        writeln!(f, "Min Latency: {}ms", self.min_latency_ms)?;
        writeln!(f, "Max Latency: {}ms", self.max_latency_ms)?;
        writeln!(f, "p50 Latency: {}ms", self.p50_latency_ms)?;
        writeln!(f, "p90 Latency: {}ms", self.p90_latency_ms)?;
        writeln!(f, "p99 Latency: {}ms", self.p99_latency_ms)?;
        writeln!(f, "TPS: {:.2}", self.tps)?;
        if self.total_output_tokens > 0 {
            writeln!(f, "Total Output Tokens: {}", self.total_output_tokens)?;
            writeln!(
                f,
                "Output Tokens per second: {:.2}",
                self.output_tokens_per_second
            )?;
        }
        if self.total_input_tokens > 0 {
            writeln!(f, "Total Input Tokens: {}", self.total_input_tokens)?;
            writeln!(
                f,
                "Input Tokens per second: {:.2}",
                self.input_tokens_per_second
            )?;
        }
        Ok(())
    }
}
