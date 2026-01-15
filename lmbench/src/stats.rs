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
    pub avg_ttft_ms: Option<u128>,
    pub min_ttft_ms: Option<u128>,
    pub max_ttft_ms: Option<u128>,
    pub p50_ttft_ms: Option<u128>,
    pub p90_ttft_ms: Option<u128>,
    pub p99_ttft_ms: Option<u128>,
    pub qps: f64,
    pub total_output_tokens: usize,
    pub output_tokens_per_second: f64,
    pub total_input_tokens: usize,
    pub input_tokens_per_second: f64,
    pub server_input_tokens: Option<usize>,
    pub server_output_tokens: Option<usize>,
}

pub fn generate_stats(
    latencies: &[Duration],
    ttfts: &[Duration],
    total_input_tokens: usize,
    total_output_tokens: usize,
    server_input_tokens: usize,
    server_output_tokens: usize,
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
            avg_ttft_ms: None,
            min_ttft_ms: None,
            max_ttft_ms: None,
            p50_ttft_ms: None,
            p90_ttft_ms: None,
            p99_ttft_ms: None,
            qps: 0.0,
            total_output_tokens: 0,
            output_tokens_per_second: 0.0,
            total_input_tokens: 0,
            input_tokens_per_second: 0.0,
            server_input_tokens: None,
            server_output_tokens: None,
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
    let (avg_ttft_ms, min_ttft_ms, max_ttft_ms, p50_ttft_ms, p90_ttft_ms, p99_ttft_ms) =
        if !ttfts.is_empty() {
            let mut sorted_ttfts = ttfts.to_vec();
            sorted_ttfts.sort();
            (
                Some(total_time_ms / success_requests as u128),
                Some(sorted_ttfts.first().unwrap().as_millis()),
                Some(sorted_ttfts.last().unwrap().as_millis()),
                Some(sorted_ttfts[success_requests / 2].as_millis()),
                Some(sorted_ttfts[(success_requests as f64 * 0.9) as usize].as_millis()),
                Some(sorted_ttfts[(success_requests as f64 * 0.99) as usize].as_millis()),
            )
        } else {
            (None, None, None, None, None, None)
        };
    let qps = success_requests as f64 / total_time.as_secs_f64();
    let output_tokens_per_second = total_output_tokens as f64 / total_time.as_secs_f64();
    let input_tokens_per_second = total_input_tokens as f64 / total_time.as_secs_f64();

    let server_it = if server_input_tokens == 0 {
        None
    } else {
        Some(server_input_tokens)
    };
    let server_ot = if server_output_tokens == 0 {
        None
    } else {
        Some(server_output_tokens)
    };

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
        avg_ttft_ms,
        min_ttft_ms,
        max_ttft_ms,
        p50_ttft_ms,
        p90_ttft_ms,
        p99_ttft_ms,
        qps,
        total_output_tokens,
        output_tokens_per_second,
        total_input_tokens,
        input_tokens_per_second,
        server_input_tokens: server_it,
        server_output_tokens: server_ot,
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string_pretty(&self).unwrap();
        writeln!(f, "{}", json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_generate_stats() {
        let latencies = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
            Duration::from_millis(600),
            Duration::from_millis(700),
            Duration::from_millis(800),
            Duration::from_millis(900),
            Duration::from_millis(1000),
        ];
        let ttfts = vec![];
        let total_input_tokens = 500;
        let total_output_tokens = 1000;
        let error_requests = 2;

        let stats = generate_stats(
            &latencies,
            &ttfts,
            total_input_tokens,
            total_output_tokens,
            0,
            0,
            error_requests,
        );

        assert_eq!(stats.success_requests, 10);
        assert_eq!(stats.error_requests, 2);
        assert_eq!(stats.total_time_ms, 5500);
        assert_eq!(stats.avg_latency_ms, 550);
        assert_eq!(stats.min_latency_ms, 100);
        assert_eq!(stats.max_latency_ms, 1000);
        assert_eq!(stats.p50_latency_ms, 600);
        assert_eq!(stats.p90_latency_ms, 1000);
        assert_eq!(stats.p99_latency_ms, 1000);
        assert!((stats.qps - 1.8181).abs() < 0.0001);
        assert!((stats.output_tokens_per_second - 181.8181).abs() < 0.0001);
        assert!((stats.input_tokens_per_second - 90.9090).abs() < 0.0001);

        let expected_json = serde_json::to_string_pretty(&stats).unwrap();
        assert_eq!(stats.to_string(), format!("{}\n", expected_json));
    }

    #[test]
    fn test_generate_stats_empty() {
        let latencies = vec![];
        let ttfts = vec![];
        let total_input_tokens = 0;
        let total_output_tokens = 0;
        let error_requests = 0;

        let stats = generate_stats(
            &latencies,
            &ttfts,
            total_input_tokens,
            total_output_tokens,
            0,
            0,
            error_requests,
        );

        assert_eq!(stats.success_requests, 0);
        assert_eq!(stats.error_requests, 0);
        assert_eq!(stats.total_time_ms, 0);
        assert_eq!(stats.avg_latency_ms, 0);
        assert_eq!(stats.min_latency_ms, 0);
        assert_eq!(stats.max_latency_ms, 0);
        assert_eq!(stats.p50_latency_ms, 0);
        assert_eq!(stats.p90_latency_ms, 0);
        assert_eq!(stats.p99_latency_ms, 0);
        assert_eq!(stats.qps, 0.0);
        assert_eq!(stats.total_output_tokens, 0);
        assert_eq!(stats.output_tokens_per_second, 0.0);
        assert_eq!(stats.total_input_tokens, 0);
        assert_eq!(stats.input_tokens_per_second, 0.0);
    }
}
