use serde::Serialize;
use serde_with::skip_serializing_none;
use std::collections::BTreeMap;
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
    pub serving: Option<ServingStats>,
}

/// Input buckets by uncached (incremental) input tokens: [lower, upper)
const BUCKETS: [(usize, usize, &str); 5] = [
    (0, 8 * 1024, "0-8K"),
    (8 * 1024, 32 * 1024, "8K-32K"),
    (32 * 1024, 128 * 1024, "32K-128K"),
    (128 * 1024, 256 * 1024, "128K-256K"),
    (256 * 1024, usize::MAX, "256K-1M"),
];

fn bucket_index(uncached_input_tokens: usize) -> usize {
    BUCKETS
        .iter()
        .position(|(lo, hi, _)| (*lo..*hi).contains(&uncached_input_tokens))
        .unwrap()
}

/// Log-scale histogram (1% resolution) over microseconds, so per-token samples
/// of long running benchmarks stay bounded in memory.
#[derive(Debug, Default, Clone)]
pub struct Histogram {
    counts: BTreeMap<i32, u64>,
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
}

impl Histogram {
    const RATIO: f64 = 1.01;

    pub fn record(&mut self, value_us: f64, n: u64) {
        if n == 0 {
            return;
        }
        let idx = (value_us.max(1e-3).ln() / Self::RATIO.ln()).floor() as i32;
        *self.counts.entry(idx).or_default() += n;
        if self.count == 0 || value_us < self.min {
            self.min = value_us;
        }
        if self.count == 0 || value_us > self.max {
            self.max = value_us;
        }
        self.count += n;
        self.sum += value_us * n as f64;
    }

    pub fn merge(&mut self, other: &Histogram) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 || other.min < self.min {
            self.min = other.min;
        }
        if self.count == 0 || other.max > self.max {
            self.max = other.max;
        }
        for (idx, n) in &other.counts {
            *self.counts.entry(*idx).or_default() += n;
        }
        self.count += other.count;
        self.sum += other.sum;
    }

    fn percentile(&self, p: f64) -> f64 {
        let rank = (self.count as f64 * p) as u64;
        let mut cumulative = 0;
        for (idx, n) in &self.counts {
            cumulative += n;
            if cumulative > rank {
                let value = Self::RATIO.powf(*idx as f64 + 0.5);
                return value.clamp(self.min, self.max);
            }
        }
        self.max
    }

    fn distribution_ms(&self) -> Option<FloatDistribution> {
        if self.count == 0 {
            return None;
        }
        Some(FloatDistribution {
            min: self.min / 1000.,
            max: self.max / 1000.,
            mean: self.sum / self.count as f64 / 1000.,
            p50: self.percentile(0.5) / 1000.,
            p90: self.percentile(0.9) / 1000.,
            p99: self.percentile(0.99) / 1000.,
        })
    }
}

/// Per request serving metrics, following the Token In / Token Out definitions:
/// TTFT ends at the first frame carrying output tokens, OTPS excludes the first
/// frame's tokens, and TPOT splits each frame's gap evenly over its tokens.
pub struct RequestMetrics {
    pub ttft: Duration,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub uncached_input_tokens: usize,
    /// None if the output is too short to measure
    pub otps: Option<f64>,
    pub tpot: Histogram,
}

impl RequestMetrics {
    /// `frames` are (arrival time, token count) of response frames carrying output tokens
    pub fn new(
        frames: &[(Duration, usize)],
        output_tokens: usize,
        prompt_tokens: usize,
        cached_tokens: usize,
        uncached_input_tokens: usize,
        min_output_tokens: usize,
    ) -> Option<Self> {
        let (first_time, first_tokens) = *frames.first()?;
        let (last_time, _) = *frames.last().unwrap();
        let mut otps = None;
        let mut tpot = Histogram::default();
        let generation_time = last_time.saturating_sub(first_time).as_secs_f64();
        if frames.len() >= 2 && output_tokens >= min_output_tokens && generation_time > 0. {
            otps = Some(output_tokens.saturating_sub(first_tokens) as f64 / generation_time);
            for pair in frames.windows(2) {
                let (prev, (time, tokens)) = (pair[0].0, pair[1]);
                let gap_us = time.saturating_sub(prev).as_secs_f64() * 1e6;
                tpot.record(gap_us / tokens as f64, tokens as u64);
            }
        }
        Some(Self {
            ttft: first_time,
            prompt_tokens,
            cached_tokens,
            uncached_input_tokens,
            otps,
            tpot,
        })
    }
}

#[derive(Default, Clone)]
struct BucketCollector {
    ttfts: Vec<Duration>,
    otps: Vec<f64>,
    tpot: Histogram,
    short_output_requests: usize,
}

impl BucketCollector {
    fn merge(&mut self, other: BucketCollector) {
        self.ttfts.extend(other.ttfts);
        self.otps.extend(other.otps);
        self.tpot.merge(&other.tpot);
        self.short_output_requests += other.short_output_requests;
    }
}

#[derive(Default, Clone)]
pub struct ServingCollector {
    prompt_tokens: usize,
    cached_tokens: usize,
    buckets: [BucketCollector; BUCKETS.len()],
}

impl ServingCollector {
    pub fn add(&mut self, metrics: RequestMetrics) {
        self.prompt_tokens += metrics.prompt_tokens;
        self.cached_tokens += metrics.cached_tokens;
        let bucket = &mut self.buckets[bucket_index(metrics.uncached_input_tokens)];
        bucket.ttfts.push(metrics.ttft);
        if let Some(otps) = metrics.otps {
            bucket.otps.push(otps);
            bucket.tpot.merge(&metrics.tpot);
        } else {
            bucket.short_output_requests += 1;
        }
    }

    pub fn merge(&mut self, other: ServingCollector) {
        self.prompt_tokens += other.prompt_tokens;
        self.cached_tokens += other.cached_tokens;
        for (bucket, other) in self.buckets.iter_mut().zip(other.buckets) {
            bucket.merge(other);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.iter().all(|b| b.ttfts.is_empty())
    }

    pub fn generate_stats(&self, otps_floor: f64) -> ServingStats {
        let mut all = BucketCollector::default();
        let mut buckets = Vec::new();
        for (bucket, (_, _, name)) in self.buckets.iter().zip(BUCKETS) {
            if bucket.ttfts.is_empty() {
                continue;
            }
            buckets.push(bucket.generate_stats(name, otps_floor));
            all.merge(bucket.clone());
        }
        let cache_hit_rate = if self.prompt_tokens > 0 {
            Some(self.cached_tokens as f64 / self.prompt_tokens as f64)
        } else {
            None
        };
        ServingStats {
            cache_hit_rate,
            server_prompt_tokens: self.prompt_tokens,
            server_cached_tokens: self.cached_tokens,
            all: all.generate_stats("all", otps_floor),
            buckets,
        }
    }
}

impl BucketCollector {
    fn generate_stats(&self, name: &str, otps_floor: f64) -> BucketStats {
        let (otps, otps_below_floor_ratio) = if self.otps.is_empty() {
            (None, None)
        } else {
            let below = self.otps.iter().filter(|o| **o <= otps_floor).count();
            (
                Some(get_float_distribution(&self.otps)),
                Some(below as f64 / self.otps.len() as f64),
            )
        };
        BucketStats {
            input_tokens: name.to_string(),
            requests: self.ttfts.len(),
            short_output_requests: self.short_output_requests,
            ttft_ms: get_distribution(&self.ttfts),
            otps,
            otps_below_floor_ratio,
            tpot_ms: self.tpot.distribution_ms(),
        }
    }
}

#[skip_serializing_none]
#[derive(Serialize, Debug)]
pub struct ServingStats {
    #[serde(with = "round_option_float")]
    pub cache_hit_rate: Option<f64>,
    pub server_prompt_tokens: usize,
    pub server_cached_tokens: usize,
    pub all: BucketStats,
    pub buckets: Vec<BucketStats>,
}

#[skip_serializing_none]
#[derive(Serialize, Debug)]
pub struct BucketStats {
    /// uncached input tokens range
    pub input_tokens: String,
    pub requests: usize,
    /// requests excluded from OTPS/TPOT for too few output tokens or frames
    pub short_output_requests: usize,
    pub ttft_ms: Distribution,
    pub otps: Option<FloatDistribution>,
    /// ratio of measured requests with OTPS not above the floor
    #[serde(with = "round_option_float")]
    pub otps_below_floor_ratio: Option<f64>,
    pub tpot_ms: Option<FloatDistribution>,
}

#[derive(Serialize, Debug, Default)]
pub struct FloatDistribution {
    #[serde(with = "round_float")]
    pub min: f64,
    #[serde(with = "round_float")]
    pub max: f64,
    #[serde(with = "round_float")]
    pub mean: f64,
    #[serde(with = "round_float")]
    pub p50: f64,
    #[serde(with = "round_float")]
    pub p90: f64,
    #[serde(with = "round_float")]
    pub p99: f64,
}

fn get_float_distribution(values: &[f64]) -> FloatDistribution {
    let count = values.len();
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    FloatDistribution {
        min: sorted[0],
        max: sorted[count - 1],
        mean: sorted.iter().sum::<f64>() / count as f64,
        p50: sorted[count / 2],
        p90: sorted[(count as f64 * 0.9) as usize],
        p99: sorted[(count as f64 * 0.99) as usize],
    }
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
    total_time_ms: u128,
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
            serving: None,
        };
    }

    let success_requests = latencies.len();

    let latency_ms = get_distribution(latencies);
    let ttft_ms = if !ttfts.is_empty() {
        Some(get_distribution(ttfts))
    } else {
        None
    };
    let total_time = total_time_ms as f64 / 1000.;
    let qps = success_requests as f64 / total_time;
    let (input_tokens, input_tokens_per_min, output_tokens, output_tokens_per_min) =
        if output_tokens > 0 {
            (
                Some(input_tokens),
                Some(input_tokens as f64 / total_time * 60.),
                Some(output_tokens),
                Some(output_tokens as f64 / total_time * 60.),
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
        serving: None,
    }
}

impl Stats {
    pub fn get_error_rate(&self) -> f64 {
        let total_requests = self.success_requests + self.error_requests;
        if total_requests == 0 {
            0.0
        } else {
            self.error_requests as f64 / total_requests as f64
        }
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
            100,
        );

        assert_eq!(stats.success_requests, 10);
        assert_eq!(stats.error_requests, 2);
        assert_eq!(stats.total_time_ms, 100);
        assert_eq!(stats.latency_ms.mean, 550);
        assert_eq!(stats.latency_ms.min, 100);
        assert_eq!(stats.latency_ms.max, 1000);
        assert_eq!(stats.latency_ms.p50, 600);
        assert_eq!(stats.latency_ms.p90, 1000);
        assert_eq!(stats.latency_ms.p99, 1000);
        assert!((stats.qps - 100.).abs() < 0.0001);
        assert!((stats.input_tokens_per_min.unwrap() - 300000.0).abs() < 0.0001);
        assert!((stats.output_tokens_per_min.unwrap() - 600000.0).abs() < 0.0001);

        let expected_json = serde_json::to_string_pretty(&stats).unwrap();
        assert_eq!(stats.to_string(), format!("{}\n", expected_json));
    }

    #[test]
    fn test_request_metrics() {
        let ms = Duration::from_millis;
        // first frame 2 tokens at 500ms, then 1 token / 20ms, last frame 2 tokens after 40ms
        let mut frames = vec![(ms(500), 2)];
        for i in 1..=30 {
            frames.push((ms(500 + i * 20), 1));
        }
        frames.push((ms(1140), 2));
        let output_tokens = frames.iter().map(|f| f.1).sum();
        assert_eq!(output_tokens, 34);

        let m = RequestMetrics::new(&frames, output_tokens, 10_000, 2_000, 8_000, 32).unwrap();
        assert_eq!(m.ttft, ms(500));
        // (34 - 2) tokens over 640ms
        assert!((m.otps.unwrap() - 50.).abs() < 1e-9);
        // 30 tokens at 20ms + 2 tokens at 40ms / 2
        let tpot = m.tpot.distribution_ms().unwrap();
        assert!((tpot.mean - 20.).abs() < 1e-9);
        assert!((tpot.p50 - 20.).abs() < 0.2);

        // too short output
        let m = RequestMetrics::new(&frames, output_tokens, 0, 0, 0, 64).unwrap();
        assert_eq!(m.otps, None);
        // single frame
        let m = RequestMetrics::new(&frames[..1], 40, 0, 0, 0, 32).unwrap();
        assert_eq!(m.otps, None);
        assert!(RequestMetrics::new(&[], 0, 0, 0, 0, 32).is_none());
    }

    #[test]
    fn test_serving_collector() {
        let ms = Duration::from_millis;
        let frames =
            |ttft: u64, gap: u64| (0..40).map(|i| (ms(ttft + i * gap), 1)).collect::<Vec<_>>();
        let mut a = ServingCollector::default();
        a.add(RequestMetrics::new(&frames(100, 10), 40, 1000, 500, 500, 32).unwrap());
        a.add(RequestMetrics::new(&frames(200, 40), 40, 1000, 0, 1000, 32).unwrap());
        let mut b = ServingCollector::default();
        b.add(RequestMetrics::new(&frames(3000, 20), 40, 40_000, 0, 40_000, 32).unwrap());
        b.add(RequestMetrics::new(&frames(300, 20)[..2], 2, 1000, 0, 1000, 32).unwrap());
        a.merge(b);

        let stats = a.generate_stats(35.);
        assert!((stats.cache_hit_rate.unwrap() - 500. / 43_000.).abs() < 1e-9);
        assert_eq!(stats.buckets.len(), 2);
        assert_eq!(stats.buckets[0].input_tokens, "0-8K");
        assert_eq!(stats.buckets[0].requests, 3);
        assert_eq!(stats.buckets[0].short_output_requests, 1);
        assert_eq!(stats.buckets[0].ttft_ms.p50, 200);
        // 100 OTPS and 25 OTPS
        assert_eq!(stats.buckets[0].otps_below_floor_ratio, Some(0.5));
        assert_eq!(stats.buckets[1].input_tokens, "32K-128K");
        assert_eq!(stats.buckets[1].otps.as_ref().unwrap().p50, 50.);
        assert_eq!(stats.all.requests, 4);
        assert_eq!(stats.all.otps.as_ref().unwrap().max, 100.);
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
            100,
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
