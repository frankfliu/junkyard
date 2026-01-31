use serde::Serialize;
use serde_with::skip_serializing_none;
use std::fmt;
use std::time::Duration;

#[skip_serializing_none]
#[derive(Serialize, Debug, Default)]
pub struct Distribution {
    pub min: u128,
    pub max: u128,
    pub mean: u128,
    pub p50: u128,
    pub p90: u128,
    pub p99: u128,
}

#[skip_serializing_none]
#[derive(Serialize, Debug)]
pub struct Stats {
    pub success_requests: usize,
    pub error_requests: usize,
    pub total_time_ms: u128,
    pub latency_ms: Distribution,
    pub ttft_ms: Option<Distribution>,

    #[serde(with = "round_float")]
    pub qps: f64,
    pub input_tokens: Option<usize>,
    #[serde(skip)]
    pub input_tokens_per_min: Option<f64>,
    pub output_tokens: Option<usize>,
    #[serde(with = "round_option_float")]
    pub output_tokens_per_min: Option<f64>,
    pub server_input_tokens: Option<usize>,
    pub server_output_tokens: Option<usize>,
}

mod round_float {
    use serde::Serializer;

    pub fn serialize<S>(decimal: &f64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64((decimal * 10000.0).round() / 10000.0)
    }
}

mod round_option_float {
    use serde::Serializer;

    pub fn serialize<S>(decimal: &Option<f64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if let Some(decimal) = decimal {
            serializer.serialize_f64((decimal * 10000.0).round() / 10000.0)
        } else {
            serializer.serialize_none()
        }
    }
}

fn get_distribution(durations: &[Duration]) -> Distribution {
    let count = durations.len();
    let mut sorted_durations = durations.to_vec();
    sorted_durations.sort();

    let total: Duration = sorted_durations.iter().sum();
    let mean = total.as_millis() / count as u128;
    let min = sorted_durations.first().unwrap().as_millis();
    let max = sorted_durations.last().unwrap().as_millis();
    let p50 = sorted_durations[count / 2].as_millis();
    let p90 = sorted_durations[(count as f64 * 0.9) as usize].as_millis();
    let p99 = sorted_durations[(count as f64 * 0.99) as usize].as_millis();

    Distribution {
        min,
        max,
        mean,
        p50,
        p90,
        p99,
    }
}

pub fn generate_stats(
    latencies: &[Duration],
    ttfts: &[Duration],
    input_tokens: usize,
    output_tokens: usize,
    server_input_tokens: usize,
    server_output_tokens: usize,
    error_requests: usize,
) -> Stats {
    if latencies.is_empty() {
        return Stats {
            success_requests: 0,
            error_requests,
            total_time_ms: 0,
            latency_ms: Distribution::default(),
            ttft_ms: None,
            qps: 0.0,
            output_tokens: None,
            output_tokens_per_min: None,
            input_tokens: None,
            input_tokens_per_min: None,
            server_input_tokens: None,
            server_output_tokens: None,
        };
    }

    let success_requests = latencies.len();
    let total_time: Duration = latencies.iter().sum();
    let total_time_ms = total_time.as_millis();

    let latency_ms = get_distribution(latencies);
    let ttft_ms = if !ttfts.is_empty() {
        Some(get_distribution(ttfts))
    } else {
        None
    };

    let qps = success_requests as f64 / total_time.as_secs_f64();
    let (input_tokens, input_tokens_per_min, output_tokens, output_tokens_per_min) =
        if output_tokens > 0 {
            (
                Some(input_tokens),
                Some(input_tokens as f64 / total_time.as_secs_f64() * 60.),
                Some(output_tokens),
                Some(output_tokens as f64 / total_time.as_secs_f64() * 60.),
            )
        } else {
            (None, None, None, None)
        };

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
        latency_ms,
        ttft_ms,
        qps,
        input_tokens,
        input_tokens_per_min,
        output_tokens,
        output_tokens_per_min,
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
        let input_tokens = 500;
        let output_tokens = 1000;
        let error_requests = 2;

        let stats = generate_stats(
            &latencies,
            &ttfts,
            input_tokens,
            output_tokens,
            0,
            0,
            error_requests,
        );

        assert_eq!(stats.success_requests, 10);
        assert_eq!(stats.error_requests, 2);
        assert_eq!(stats.total_time_ms, 5500);
        assert_eq!(stats.latency_ms.mean, 550);
        assert_eq!(stats.latency_ms.min, 100);
        assert_eq!(stats.latency_ms.max, 1000);
        assert_eq!(stats.latency_ms.p50, 600);
        assert_eq!(stats.latency_ms.p90, 1000);
        assert_eq!(stats.latency_ms.p99, 1000);
        assert!((stats.qps - 1.8181).abs() < 0.0001);
        assert!((stats.input_tokens_per_min.unwrap() - 5454.5454).abs() < 0.0001);
        assert!((stats.output_tokens_per_min.unwrap() - 10909.0909).abs() < 0.0001);

        let expected_json = serde_json::to_string_pretty(&stats).unwrap();
        assert_eq!(stats.to_string(), format!("{}\n", expected_json));
    }

    #[test]
    fn test_generate_stats_empty() {
        let latencies = vec![];
        let ttfts = vec![];
        let input_tokens = 0;
        let output_tokens = 0;
        let error_requests = 0;

        let stats = generate_stats(
            &latencies,
            &ttfts,
            input_tokens,
            output_tokens,
            0,
            0,
            error_requests,
        );

        assert_eq!(stats.success_requests, 0);
        assert_eq!(stats.error_requests, 0);
        assert_eq!(stats.total_time_ms, 0);
        assert_eq!(stats.latency_ms.mean, 0);
        assert_eq!(stats.latency_ms.min, 0);
        assert_eq!(stats.latency_ms.max, 0);
        assert_eq!(stats.latency_ms.p50, 0);
        assert_eq!(stats.latency_ms.p90, 0);
        assert_eq!(stats.latency_ms.p99, 0);
        assert_eq!(stats.qps, 0.0);
        assert_eq!(stats.input_tokens, None);
        assert_eq!(stats.input_tokens_per_min, None);
        assert_eq!(stats.output_tokens, None);
        assert_eq!(stats.output_tokens_per_min, None);
    }
}
